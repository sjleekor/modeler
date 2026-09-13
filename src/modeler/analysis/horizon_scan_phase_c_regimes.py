"""Phase C regime series, built on the KRX session grid (Stage 1b §2).

Design: ``docs/dev/20260829_macro_features/01_design/03_stage1b_conditional_ic_phase_c.md``
§2.1 (grid), §2.3 (definitions), §5 (G1/G2), §6.2 (alternative cut), §6.5
(persistence).

A regime is a date-level binary ``s_t`` that partitions the daily IC series
Phase C conditions on. Two things about it are easy to get wrong, and both are
what this module exists to pin down.

**The grid is KRX sessions, not the fact's own dates.** ``common_feature_daily_fact``
is indexed by every weekday for 2014-2023 —
``src/collector/kr/infra/calendar/data/holidays_krx.csv`` only covers 2024-2026,
so a KRX holiday sits in the fact carrying the previous
session's value. Counting ``t-20`` or ``t-252`` on that axis makes the window a
different length before and after 2024. So the fact is joined onto the scan's
own session list and every LAG/rolling window is taken on that.

**The window must be full.** A 252-session median over 40 sessions is not the
same statistic, and a regime that flips because its window was short is an
artefact. Every transform emits NULL until its window is complete, which is
also what makes the occupancy counts below honest.

Nothing here reads a label, a return, or an IC. The whole point of computing it
before the overlay is committed (``04`` §3 step 3a) is that it cannot be
influenced by an outcome — and so the G1/G2 feasibility numbers can be written
into the preregistration record rather than discovered afterwards.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

REGIME_SERIES_TABLE = "regime_series"

# Trailing windows, in KRX sessions.
SHORT_WINDOW = 20
LONG_WINDOW = 252


@dataclass(frozen=True)
class RegimeSpec:
    """One preregistered regime: its continuous ``z_t`` and where it comes from.

    ``window`` is how many sessions of history ``z_t`` needs; ``z_t`` is NULL
    before that. ``role`` mirrors the overlay's own ``phase_c.regimes[].role``
    — only ``primary`` regimes carry preregistered pairs into BH.
    """

    regime_id: str
    role: str
    transform: str
    source_codes: tuple[str, ...]
    window: int
    z_sql: str

    @property
    def min_sessions(self) -> int:
        """Sessions of history ``z_t`` needs before it can exist.

        A ``k``-session *difference* reaches back to row ``t-k``, so it needs
        ``k+1`` rows; a ``k``-session rolling mean or median needs ``k``. Two
        different numbers, and conflating them puts a regime's first value one
        session in the wrong place.
        """
        is_difference = self.transform.startswith(("diff_", "log_ratio_"))
        return self.window + 1 if is_difference else self.window


# §2.3. `z_sql` is written against the pivoted session frame below; `w` is the
# session-ordered window, `w20`/`w252` its trailing forms.
REGIME_SPECS: tuple[RegimeSpec, ...] = (
    RegimeSpec(
        regime_id="vix_up",
        role="primary",
        transform="diff_20_sessions",
        source_codes=("global_vix_level",),
        window=SHORT_WINDOW,
        # Kim, Park & Ok (2019) condition on the change in VIX, not its level.
        z_sql="global_vix_level - LAG(global_vix_level, 20) OVER w",
    ),
    RegimeSpec(
        regime_id="vix_high",
        role="primary",
        transform="minus_median_252_sessions",
        source_codes=("global_vix_level",),
        window=LONG_WINDOW,
        # Nagel (2012) conditions on the VIX *level*, which is a different
        # variable from vix_up — hence both are registered (review M6).
        z_sql="global_vix_level - MEDIAN(global_vix_level) OVER w252",
    ),
    RegimeSpec(
        regime_id="market_up",
        role="primary",
        transform="log_ratio_252_sessions",
        source_codes=("market_kospi_close",),
        window=LONG_WINDOW,
        z_sql="LN(market_kospi_close / NULLIF(LAG(market_kospi_close, 252) OVER w, 0))",
    ),
    RegimeSpec(
        regime_id="liq_high",
        role="primary",
        transform="log_mean20_over_median252_sum",
        source_codes=("market_kospi_turnover_value", "market_kosdaq_turnover_value"),
        window=LONG_WINDOW,
        # Both markets summed: the daily IC is an n-weighted blend of the two,
        # so the liquidity regime has to be market-wide too (§2.3).
        z_sql=("LN(AVG(turnover_total) OVER w20 " "/ NULLIF(MEDIAN(turnover_total) OVER w252, 0))"),
    ),
    RegimeSpec(
        regime_id="term_steep",
        role="exploratory",
        transform="minus_median_252_sessions",
        source_codes=("rate_kr_term_spread_10y_3y",),
        window=LONG_WINDOW,
        z_sql=("rate_kr_term_spread_10y_3y - MEDIAN(rate_kr_term_spread_10y_3y) OVER w252"),
    ),
    RegimeSpec(
        regime_id="kosdaq_rel_up",
        role="exploratory",
        transform="sum20_diff",
        source_codes=("market_kosdaq_ret_1d", "market_kospi_ret_1d"),
        window=SHORT_WINDOW,
        z_sql=("SUM(market_kosdaq_ret_1d) OVER w20 - SUM(market_kospi_ret_1d) OVER w20"),
    ),
    RegimeSpec(
        regime_id="krw_weak_20d",
        role="exploratory",
        transform="log_ratio_20_sessions",
        source_codes=("fx_usdkrw_level",),
        window=SHORT_WINDOW,
        z_sql="LN(fx_usdkrw_level / NULLIF(LAG(fx_usdkrw_level, 20) OVER w, 0))",
    ),
)

REGIME_IDS: tuple[str, ...] = tuple(spec.regime_id for spec in REGIME_SPECS)

SOURCE_FEATURE_CODES: tuple[str, ...] = tuple(
    dict.fromkeys(code for spec in REGIME_SPECS for code in spec.source_codes)
)


def build_regime_series_sql(
    *,
    session_view: str = "label_scan",
    fact_view: str = "common_feature_daily_fact",
) -> str:
    """One row per KRX session with every regime's ``z_t``, ``s_t`` and alt cut.

    ``session_view`` is the scan's own date axis (``label_scan``), so the
    output joins to ``daily_ic`` on ``trade_date`` with nothing left over.

    The alternative cut (§6.2, gate G6) is built here beside the registered
    one: ``z_t`` above its own trailing 252-session median, rather than above
    zero. It exists to show whether the ``> 0`` threshold is doing the work,
    and is a non-fatal diagnostic — never the judged partition.
    """
    codes = ", ".join(f"'{code}'" for code in SOURCE_FEATURE_CODES)
    pivot = ",\n                   ".join(
        f"MAX(CASE WHEN feature_code = '{code}' THEN CAST(value_numeric AS DOUBLE) END) AS {code}"
        for code in SOURCE_FEATURE_CODES
    )
    # `session_idx` is the running session count, so this is exactly "enough
    # sessions of history" — and it distinguishes a difference (needs k+1 rows)
    # from a rolling window (needs k), which a fixed COUNT frame cannot.
    z_columns = ",\n                ".join(
        f"CASE WHEN session_idx >= {spec.min_sessions} THEN {spec.z_sql} END "
        f"AS z_{spec.regime_id}"
        for spec in REGIME_SPECS
    )
    # z_t rides along: §6.2's alternative cut and §6.4's continuous regression
    # both read it, and it is what makes a regime's binary auditable.
    z_outputs = ", ".join(f"z_{rid}" for rid in REGIME_IDS)
    cut_columns = ",\n            ".join(
        f"CAST(z_{rid} > 0 AS BOOLEAN) AS s_{rid},\n"
        f"            CASE WHEN COUNT(z_{rid}) OVER w252 = {LONG_WINDOW} "
        f"THEN CAST(z_{rid} > MEDIAN(z_{rid}) OVER w252 AS BOOLEAN) END AS alt_s_{rid}"
        for rid in REGIME_IDS
    )
    return f"""
        WITH sessions AS (
            SELECT DISTINCT trade_date FROM {session_view}
        ),
        fact_wide AS (
            SELECT feature_date,
                   {pivot}
            FROM {fact_view}
            WHERE feature_code IN ({codes})
            GROUP BY feature_date
        ),
        joined AS (
            SELECT s.trade_date,
                   ROW_NUMBER() OVER (ORDER BY s.trade_date) AS session_idx,
                   f.* EXCLUDE (feature_date),
                   COALESCE(f.market_kospi_turnover_value, 0)
                       + COALESCE(f.market_kosdaq_turnover_value, 0) AS turnover_total
            FROM sessions s
            LEFT JOIN fact_wide f ON f.feature_date = s.trade_date
        ),
        continuous AS (
            SELECT trade_date, session_idx,
                {z_columns}
            FROM joined
            WINDOW
                w AS (ORDER BY trade_date),
                w20 AS (ORDER BY trade_date
                        ROWS BETWEEN {SHORT_WINDOW - 1} PRECEDING AND CURRENT ROW),
                w252 AS (ORDER BY trade_date
                         ROWS BETWEEN {LONG_WINDOW - 1} PRECEDING AND CURRENT ROW)
        )
        SELECT trade_date, session_idx,
            {z_outputs},
            {cut_columns}
        FROM continuous
        WINDOW w252 AS (ORDER BY trade_date
                        ROWS BETWEEN {LONG_WINDOW - 1} PRECEDING AND CURRENT ROW)
    """


def build_regime_series(
    con: duckdb.DuckDBPyConnection,
    *,
    session_view: str = "label_scan",
    fact_view: str = "common_feature_daily_fact",
    sample_start: date | str | None = None,
    sample_end: date | str | None = None,
) -> pl.DataFrame:
    """Materialize the regime series, optionally trimmed to a date window.

    Trimming happens *after* the windows are taken, so a regime at the first
    in-sample session still sees its full 252 sessions of history.

    ``sample_end`` matters as much as the start: every preregistered pair sits
    at the ``common_survivor`` coordinate, whose daily IC stops at
    ``common_formation_end``. Measuring occupancy past that would count
    sessions Phase C never conditions on.
    """
    frame = con.execute(
        build_regime_series_sql(session_view=session_view, fact_view=fact_view)
    ).pl()
    if sample_start is not None:
        start = _as_date(sample_start)
        frame = frame.filter(pl.col("trade_date") >= start)
    if sample_end is not None:
        frame = frame.filter(pl.col("trade_date") <= _as_date(sample_end))
    return frame.sort("trade_date")


def _as_date(value: date | str) -> date:
    return date.fromisoformat(value) if isinstance(value, str) else value


# --- G1 occupancy, G2 subperiod feasibility, §6.5 persistence ---


def regime_occupancy(
    frame: pl.DataFrame,
    *,
    min_dates: int = 250,
    min_share: float = 0.20,
) -> list[dict[str, Any]]:
    """G1: each side of the partition needs enough dates, in count and in share.

    A regime that fails this is not evidence of anything — it is a variable
    that barely varies over the sample. The pair stays in the BH population at
    ``p=1.0`` rather than being dropped (§5), so failing here shrinks power, it
    does not shrink ``m``.
    """
    rows: list[dict[str, Any]] = []
    for spec in REGIME_SPECS:
        column = f"s_{spec.regime_id}"
        valid = frame.filter(pl.col(column).is_not_null())
        n = valid.height
        n_s1 = int(valid[column].sum()) if n else 0
        n_s0 = n - n_s1
        share_s1 = (n_s1 / n) if n else float("nan")
        passes = n_s1 >= min_dates and n_s0 >= min_dates and min_share <= share_s1 <= 1 - min_share
        rows.append(
            {
                "regime_id": spec.regime_id,
                "role": spec.role,
                "n_dates": n,
                "n_dates_s1": n_s1,
                "n_dates_s0": n_s0,
                "share_s1": share_s1,
                "first_date": valid["trade_date"].min() if n else None,
                "last_date": valid["trade_date"].max() if n else None,
                "g1_pass": passes,
            }
        )
    return rows


def regime_persistence(frame: pl.DataFrame) -> list[dict[str, Any]]:
    """§6.5: how many transitions, and how long a run lasts on each side.

    This is the number that says how small the effective sample really is. A
    regime averaging 63 sessions per run has far fewer independent switches
    than its date count suggests, which is exactly why G4's circular-shift
    placebo is mandatory rather than optional (§5.1).
    """
    rows: list[dict[str, Any]] = []
    for spec in REGIME_SPECS:
        column = f"s_{spec.regime_id}"
        values = [v for v in frame[column].to_list() if v is not None]
        transitions = sum(1 for a, b in zip(values, values[1:], strict=False) if a != b)
        runs: dict[bool, list[int]] = {True: [], False: []}
        previous: bool | None = None
        for value in values:
            if value == previous:
                runs[value][-1] += 1
            else:
                runs[value].append(1)
            previous = value
        rows.append(
            {
                "regime_id": spec.regime_id,
                "role": spec.role,
                "n_regime_transitions": transitions,
                "n_runs_s1": len(runs[True]),
                "n_runs_s0": len(runs[False]),
                "mean_run_length_s1": (
                    sum(runs[True]) / len(runs[True]) if runs[True] else float("nan")
                ),
                "mean_run_length_s0": (
                    sum(runs[False]) / len(runs[False]) if runs[False] else float("nan")
                ),
            }
        )
    return rows


def regime_subperiod_counts(
    frame: pl.DataFrame,
    period_sets: list[dict[str, Any]],
    *,
    placeholders: dict[str, date] | None = None,
    min_dates_per_regime: int = 40,
) -> list[dict[str, Any]]:
    """G2 feasibility: per preregistered subperiod, days on each side.

    §5's period-consistency gate only counts a subperiod as *valid* when both
    sides clear ``min_dates_per_regime``. Which subperiods those are is decided
    by the regime alone, so it is settled here — before the overlay hash — and
    written into the record. ``market_up`` is the one to watch: it is close to
    a constant within some calendar years.
    """
    placeholders = placeholders or {}
    rows: list[dict[str, Any]] = []
    for period in period_sets:
        start = _resolve(period["start"], placeholders)
        end = _resolve(period["end"], placeholders)
        window = frame.filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
        for spec in REGIME_SPECS:
            column = f"s_{spec.regime_id}"
            valid = window.filter(pl.col(column).is_not_null())
            n_s1 = int(valid[column].sum()) if valid.height else 0
            n_s0 = valid.height - n_s1
            rows.append(
                {
                    "period_id": period["id"],
                    "period_start": start,
                    "period_end": end,
                    "regime_id": spec.regime_id,
                    "role": spec.role,
                    "n_dates": valid.height,
                    "n_dates_s1": n_s1,
                    "n_dates_s0": n_s0,
                    "g2_valid": n_s1 >= min_dates_per_regime and n_s0 >= min_dates_per_regime,
                }
            )
    return rows


def _resolve(value: Any, placeholders: dict[str, date]) -> date:
    if isinstance(value, str):
        if value in placeholders:
            return placeholders[value]
        return date.fromisoformat(value)
    return value


def summarize_regimes(
    frame: pl.DataFrame,
    period_sets: list[dict[str, Any]],
    *,
    placeholders: dict[str, date] | None = None,
    min_dates: int = 250,
    min_share: float = 0.20,
    min_dates_per_regime: int = 40,
) -> dict[str, list[dict[str, Any]]]:
    """The three tables ``04`` §3 step 3a asks to record before the hash."""
    return {
        "occupancy": regime_occupancy(frame, min_dates=min_dates, min_share=min_share),
        "persistence": regime_persistence(frame),
        "subperiods": regime_subperiod_counts(
            frame,
            period_sets,
            placeholders=placeholders,
            min_dates_per_regime=min_dates_per_regime,
        ),
    }


def render_regime_summary_md(summary: dict[str, list[dict[str, Any]]]) -> str:
    """Markdown for ``05_preregistration_record.md``'s regime section."""
    lines: list[str] = ["### G1 국면 점유율", ""]
    lines.append("| regime | role | 세션 | s=1 | s=0 | 점유율 | 시작 | 끝 | G1 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    persistence = {row["regime_id"]: row for row in summary["persistence"]}
    for row in summary["occupancy"]:
        lines.append(
            f"| `{row['regime_id']}` | {row['role']} | {row['n_dates']:,} | "
            f"{row['n_dates_s1']:,} | "
            f"{row['n_dates_s0']:,} | {row['share_s1']:.3f} | {row['first_date']} | "
            f"{row['last_date']} | {'통과' if row['g1_pass'] else '**미달**'} |"
        )
    lines += ["", "### 국면 지속 (§6.5)", ""]
    lines.append("| regime | 전환 횟수 | 평균 지속 s=1 | 평균 지속 s=0 |")
    lines.append("|---|---|---|---|")
    for rid in REGIME_IDS:
        row = persistence[rid]
        lines.append(
            f"| `{rid}` | {row['n_regime_transitions']} | "
            f"{row['mean_run_length_s1']:.1f} | {row['mean_run_length_s0']:.1f} |"
        )
    lines += ["", "### G2 구간별 유효 여부 (양쪽 ≥ 40일)", ""]
    periods = list(dict.fromkeys(row["period_id"] for row in summary["subperiods"]))
    lines.append("| regime | " + " | ".join(periods) + " | 유효 구간 |")
    lines.append("|---" * (len(periods) + 2) + "|")
    by_key = {(row["regime_id"], row["period_id"]): row for row in summary["subperiods"]}
    for rid in REGIME_IDS:
        cells = []
        valid = 0
        for period in periods:
            row = by_key[(rid, period)]
            mark = "✅" if row["g2_valid"] else "✗"
            valid += int(row["g2_valid"])
            cells.append(f"{row['n_dates_s1']}/{row['n_dates_s0']} {mark}")
        lines.append(f"| `{rid}` | " + " | ".join(cells) + f" | **{valid}/{len(periods)}** |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sample-start", default="2015-06-16")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    from modeler.analysis.horizon_scan_config import CONFIG_PATH, load_config
    from modeler.etl.config import DEFAULT_SOURCE, DataRoot, LakeConfig
    from modeler.etl.lake import connect, register_persisted_derived_mart
    from modeler.etl.mart import mart_root, register_mart_view

    if args.snapshot_date is None:
        parser.error("--snapshot-date is required")
    root = DataRoot.resolve(market="kr")
    output_dir = args.output_dir or (root.output / "scan" / "horizon_scan" / "phase_c_regimes")
    config = load_config(args.config or CONFIG_PATH)
    base = LakeConfig(
        root=root,
        snapshot_date=args.snapshot_date,
        source=args.source or DEFAULT_SOURCE,
    )
    manifest = json.loads(
        (mart_root(base) / "_manifests" / "_SUCCESS.json").read_text(encoding="utf-8")
    )
    from dataclasses import replace as _replace

    lake = _replace(base, analysis_config_hash=manifest["config_hash"])
    con = connect(lake)
    register_mart_view(con, lake, "label_scan")
    register_persisted_derived_mart(con, lake, "common_feature_daily_fact")

    period_sets = config.raw["sample"]["period_sets"]["common"]
    (common_end,) = con.execute(
        "SELECT max(trade_date) FROM label_scan WHERE common_formation_120d"
    ).fetchone()
    # The judged window: every preregistered pair is a common_survivor cell, so
    # its daily IC ends at common_formation_end.
    frame = build_regime_series(con, sample_start=args.sample_start, sample_end=common_end)
    full = build_regime_series(con, sample_start=args.sample_start)
    summary = summarize_regimes(
        frame, period_sets, placeholders={"common_formation_end": common_end}
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    # The stored series keeps every session past common_formation_end too: the
    # `available` sample_kind and any future cell need them, and trimming is a
    # read-time decision, not a property of the series.
    full.write_parquet(output_dir / f"{REGIME_SERIES_TABLE}.parquet")
    for name, rows in summary.items():
        pl.DataFrame(rows, infer_schema_length=None).write_parquet(
            output_dir / f"regime_{name}.parquet"
        )
    (output_dir / "regime_summary.md").write_text(
        render_regime_summary_md(summary), encoding="utf-8"
    )
    print(render_regime_summary_md(summary))
    print(
        f"judged window: {args.sample_start} .. {common_end} ({frame.height} sessions); "
        f"series stored through {full['trade_date'].max()} ({full.height} sessions)"
    )
    print(f"written: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
