"""Phase A analysis panel: collision-safe join projection (A-1, §3.1).

The panel is built from seven A0 marts that were designed independently and
therefore share several column names by accident (``in_universe`` in both
universe marts, ``is_halted``/``short_regime``/``short_balance_is_available``
in more than one, ``label_ok_*`` meaning different things in
``dim_universe_*_daily`` vs. ``label_scan``). §3.1 requires those collisions be
blocked by an explicit select list rather than resolved implicitly by
whichever join partner happens to win a ``SELECT *``.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any

import duckdb
import numpy as np
import polars as pl

from modeler.analysis.horizon_scan_daily_ic import DailyIcSink, normalize_cell_identity
from modeler.etl.metrics import (
    benjamini_hochberg,
    choose_nw_lag,
    daily_market_weighted_ic,
    daily_market_weighted_spread,
    exact_binomial_sign_test_p,
    market_weight_means,
    n_hac_pairs,
    newey_west_tstat,
    per_date_market_quantile_spread,
    per_date_market_rank_ic,
    two_sided_normal_p,
)

PANEL_KEY = ("trade_date", "ticker", "market")

LABEL_VIEW = "label_scan"
PRICE_VIEW = "feat_price"
FLOW_VIEW = "feat_flow"
BROAD_VIEW = "dim_universe_broad_daily"
TRADABLE_VIEW = "dim_universe_tradable_daily"
PIT_VIEW = "dim_stock_pit_daily"
QUALITY_VIEW = "dim_price_quality_daily"

# Only the universe-formation fields the panel actually needs downstream
# (§A-4's tradable/management-filter disclosure); each universe mart's own
# ``label_ok*`` columns are formation *inputs*, superseded by label_scan's own
# CA/quality-masked ``label_ok_*`` and intentionally left out of the panel.
_UNIVERSE_COLUMNS = [
    "in_universe",
    "membership_reconstruction_available",
    "management_filter_available",
]


def build_analysis_panel_sql(
    *,
    label_view: str = LABEL_VIEW,
    price_view: str = PRICE_VIEW,
    flow_view: str = FLOW_VIEW,
    broad_view: str = BROAD_VIEW,
    tradable_view: str = TRADABLE_VIEW,
    pit_view: str = PIT_VIEW,
    quality_view: str = QUALITY_VIEW,
) -> str:
    """Return the §3.1 join as one SQL SELECT.

    Join order is fixed: ``label_scan`` is the base (LEFT JOIN target), then
    price/flow features, then broad/tradable universe flags, then PIT
    share/mktcap, then quality/CA/short-regime flags. All seven marts are
    verified duplicate-free on ``(trade_date, ticker, market)`` by A0
    (``tests/integration/test_horizon_scan_inputs_smoke.py``), so this LEFT
    JOIN chain cannot fan out — see ``assert_panel_join_preserves_keys``.
    """
    broad_cols = ", ".join(
        f'b."{c}" AS in_broad' if c == "in_universe" else f'b."{c}"' for c in _UNIVERSE_COLUMNS
    )
    tradable_cols = ", ".join(
        f't."{c}" AS in_tradable' if c == "in_universe" else f't."{c}" AS tradable_{c}'
        for c in _UNIVERSE_COLUMNS
    )
    return f"""
        WITH calendar AS (
            SELECT trade_date,
                   ROW_NUMBER() OVER (ORDER BY trade_date) AS formation_session_idx
            FROM (SELECT DISTINCT trade_date FROM {label_view})
        )
        SELECT
            l.* ,
            cal.formation_session_idx,
            p.* EXCLUDE (trade_date, ticker, market),
            f.* EXCLUDE (trade_date, ticker, market, short_regime, short_balance_is_available),
            {broad_cols},
            {tradable_cols},
            pit.issued_shares_pit, pit.treasury_shares_pit, pit.float_shares_pit,
            pit.market_cap_pit, pit.shares_is_available AS pit_shares_is_available,
            pit.shares_invalid_flag AS pit_shares_invalid_flag,
            q.valid_session_idx, q.is_halted, q.volume_zero, q.simple_ret, q.log_ret,
            q.ca_mask, q.ca_event, q.ca_event_cumulative,
            q.ca_rule_applicability_unknown, q.short_regime, q.short_balance_is_available
        FROM {label_view} l
        LEFT JOIN calendar cal USING (trade_date)
        LEFT JOIN {price_view} p USING (trade_date, ticker, market)
        LEFT JOIN {flow_view} f USING (trade_date, ticker, market)
        LEFT JOIN {broad_view} b USING (trade_date, ticker, market)
        LEFT JOIN {tradable_view} t USING (trade_date, ticker, market)
        LEFT JOIN {pit_view} pit USING (trade_date, ticker, market)
        LEFT JOIN {quality_view} q USING (trade_date, ticker, market)
    """


def register_analysis_panel(
    con: duckdb.DuckDBPyConnection, view_name: str = "analysis_panel", **kwargs
) -> str:
    """Create ``view_name`` from :func:`build_analysis_panel_sql` and verify it.

    Raises ``RuntimeError`` (never silently dedups) if any join step would
    have fanned out — see ``assert_panel_join_preserves_keys``.
    """
    label_view = kwargs.get("label_view", LABEL_VIEW)
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {build_analysis_panel_sql(**kwargs)}")
    assert_panel_join_preserves_keys(con, view_name, label_view)
    return view_name


def assert_panel_join_preserves_keys(
    con: duckdb.DuckDBPyConnection, panel_view: str, label_view: str = LABEL_VIEW
) -> None:
    """§3.1: "각 join 전후 row count와 key distinct count가 같아야 한다."

    Since ``label_scan`` is the LEFT JOIN base, a correct join can only ever
    match its row/key count exactly — anything higher means a join partner
    had duplicate keys and silently fanned out rows.
    """
    base_rows, base_keys = con.execute(
        f"SELECT count(*), count(DISTINCT (trade_date, ticker, market)) FROM {label_view}"
    ).fetchone()
    panel_rows, panel_keys = con.execute(
        f"SELECT count(*), count(DISTINCT (trade_date, ticker, market)) FROM {panel_view}"
    ).fetchone()
    if panel_rows != base_rows or panel_keys != base_keys or panel_rows != panel_keys:
        raise RuntimeError(
            f"analysis panel join fanned out: base(rows={base_rows}, keys={base_keys}) "
            f"!= panel(rows={panel_rows}, keys={panel_keys}); a join partner has "
            "duplicate (trade_date, ticker, market) keys"
        )


# --- A-1 segment dimensions: broad-derived tertiles, reused verbatim on tradable ---


def build_broad_quantile_segment_sql(
    *,
    panel_view: str = "analysis_panel",
    value_col: str,
    segment_col: str,
    labels: tuple[str, ...] = ("small", "mid", "large"),
    min_names: int,
) -> str:
    """§A-1: cutpoint *values* computed once from ``in_broad`` rows, then applied
    by plain comparison to every row (broad and tradable alike) — never
    recomputed within the tradable subset (§A-1 test list).

    Ties use average-rank percentile (matches ``_rankdata``/n in
    ``research/etl/metrics.py``), not min-rank ``PERCENT_RANK``. A
    ``(trade_date, market)`` cross-section with fewer than ``min_names`` valid
    values, or degenerate enough that a cutpoint can't be placed (e.g. a
    constant value), gets a NULL segment for every row that date/market.
    """
    n_labels = len(labels)
    if n_labels < 2:
        raise ValueError("at least two labels are required to form a segment")
    boundaries = [i / n_labels for i in range(1, n_labels)]
    cutpoint_cols = ", ".join(
        f"MAX({value_col}) FILTER (WHERE pct <= {b}) AS cut_{i}" for i, b in enumerate(boundaries)
    )
    null_guard = " OR ".join(
        ["c.n_names IS NULL", f"c.n_names < {min_names}"]
        + [f"c.cut_{i} IS NULL" for i in range(n_labels - 1)]
    )
    case_lines = [
        f"WHEN p.{value_col} <= c.cut_{i} THEN '{label}'" for i, label in enumerate(labels[:-1])
    ]
    case_sql = "\n                ".join(case_lines)
    return f"""
        WITH broad_ranked AS (
            SELECT trade_date, market, {value_col},
                   RANK() OVER (
                       PARTITION BY trade_date, market ORDER BY {value_col}
                   ) AS min_rank,
                   COUNT(*) OVER (PARTITION BY trade_date, market, {value_col}) AS tie_count,
                   COUNT(*) OVER (PARTITION BY trade_date, market) AS n_names
            FROM {panel_view}
            WHERE in_broad AND {value_col} IS NOT NULL AND isfinite({value_col})
        ),
        broad_pct AS (
            SELECT trade_date, market, {value_col}, n_names,
                   (min_rank + (tie_count - 1) / 2.0) / n_names AS pct
            FROM broad_ranked
        ),
        cutpoints AS (
            SELECT trade_date, market, MAX(n_names) AS n_names, {cutpoint_cols}
            FROM broad_pct
            GROUP BY trade_date, market
        )
        SELECT p.trade_date, p.ticker, p.market,
            CASE
                WHEN {null_guard} THEN NULL
                WHEN p.{value_col} IS NULL OR NOT isfinite(p.{value_col}) THEN NULL
                {case_sql}
                ELSE '{labels[-1]}'
            END AS {segment_col}
        FROM {panel_view} p
        LEFT JOIN cutpoints c USING (trade_date, market)
    """


# --- A-1 period segments: preregistered calendar buckets, common vs. available ---


def _resolve_period_bound(value: Any, placeholders: dict[str, date]) -> date:
    if isinstance(value, str):
        if value not in placeholders:
            raise KeyError(f"unresolved period placeholder: {value!r}")
        return placeholders[value]
    return value


def build_period_segment_sql(
    period_sets: list[dict[str, Any]],
    placeholders: dict[str, date],
    *,
    date_col: str = "trade_date",
) -> str:
    """A SQL ``CASE`` expression assigning one preregistered period id per row.

    ``period_sets`` entries carry literal dates except for the placeholder
    bounds (``common_formation_end``, ``horizon_eligible_end``) resolved via
    ``placeholders`` — §A-1: the common-formation tail and each horizon's
    available tail are cut at their *actual* computed end, never a fixed
    literal guess.
    """
    when_clauses = []
    for period in period_sets:
        start = _resolve_period_bound(period["start"], placeholders)
        end = _resolve_period_bound(period["end"], placeholders)
        when_clauses.append(
            f"WHEN {date_col} BETWEEN DATE '{start}' AND DATE '{end}' THEN '{period['id']}'"
        )
    return "CASE " + " ".join(when_clauses) + " ELSE NULL END"


def resolve_common_formation_end(
    con: duckdb.DuckDBPyConnection, panel_view: str = "analysis_panel"
) -> date:
    """MAX(trade_date) where the 120d common-formation flag is achievable.

    §A-1: "공통 120d formation에는 2025-04 이후 구간이 결정론적으로 존재하지
    않으므로... 실제 common_formation_end에서 자른다" — this is a computed
    value, not a config literal.
    """
    (result,) = con.execute(
        f"SELECT max(trade_date) FROM {panel_view} WHERE common_formation_120d"
    ).fetchone()
    if result is None:
        raise RuntimeError("no row has common_formation_120d=true; cannot resolve period end")
    return result


def resolve_horizon_eligible_end(
    con: duckdb.DuckDBPyConnection,
    panel_view: str,
    end_date_col: str,
    holdout_start: date | str,
) -> date:
    """MAX(trade_date) for which ``end_date_col`` clears the holdout boundary.

    ``end_date_col`` is ``label_end_date_{h}d`` for a cumulative horizon or
    ``bucket_end_date_{h1}_{h2}d`` for a bucket — the available sample's tail
    is horizon-specific (§A-1's ``2025_04_holdout``/``2025_04_...`` period).
    """
    (result,) = con.execute(f"""
        SELECT max(trade_date) FROM {panel_view}
        WHERE {end_date_col} IS NOT NULL AND {end_date_col} < DATE '{holdout_start}'
        """).fetchone()
    if result is None:
        raise RuntimeError(f"no row has a pre-holdout {end_date_col}; cannot resolve period end")
    return result


# --- A-1 per-family coverage (§3.4) ---


def family_coverage_stats(
    con: duckdb.DuckDBPyConnection,
    *,
    formation_view: str,
    feature_col: str,
    label_ok_col: str,
) -> dict[str, Any]:
    """§3.4 coverage denominators for one family/feature/horizon cell.

    ``formation_view`` must already be filtered to the date/universe/holdout
    formation conditions (§3.2/§3.3) the caller wants a denominator over —
    this function only adds the feature/label numerators.
    """
    row = con.execute(f"""
        SELECT
            count(*) AS n_formation,
            count(*) FILTER (
                WHERE {feature_col} IS NOT NULL AND isfinite({feature_col})
            ) AS n_feature_finite,
            count(*) FILTER (WHERE {label_ok_col}) AS n_label_valid,
            min(trade_date) AS effective_sample_start,
            max(trade_date) AS effective_sample_end
        FROM {formation_view}
        """).fetchone()
    n_formation, n_feature_finite, n_label_valid, effective_start, effective_end = row
    return {
        "n_formation": n_formation,
        "feature_coverage": (n_feature_finite / n_formation) if n_formation else float("nan"),
        "label_coverage": (n_label_valid / n_formation) if n_formation else float("nan"),
        "effective_sample_start": effective_start,
        "effective_sample_end": effective_end,
    }


# --- A-1 native/lag1 shift-invariant check ---


def assert_lag1_matches_prior_valid_session(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str,
    native_col: str,
    lag1_col: str,
    session_col: str = "valid_session_idx",
    tolerance: float = 1e-9,
) -> None:
    """A0 §A-1 test: ``lag1[t]`` must equal ``native``'s prior *valid* session,
    per ticker — never a raw calendar-day shift (halts must be skipped).
    """
    (mismatches,) = con.execute(f"""
        WITH ordered AS (
            SELECT ticker, market, {session_col},
                   {lag1_col} AS lag1_value,
                   LAG({native_col}) OVER (
                       PARTITION BY ticker, market ORDER BY {session_col}
                   ) AS prior_native
            FROM {panel_view}
            WHERE {session_col} IS NOT NULL
        )
        SELECT count(*) FROM ordered
        WHERE lag1_value IS NOT NULL AND prior_native IS NOT NULL
          AND abs(lag1_value - prior_native) > {tolerance}
        """).fetchone()
    if mismatches:
        raise RuntimeError(
            f"{lag1_col} does not equal {native_col}'s prior valid session for "
            f"{mismatches} row(s)"
        )


# --- A-2 core cumulative/bucket IC scan (§4) ---

_UNIVERSE_FLAG_COLUMN = {"broad": "in_broad", "tradable": "in_tradable"}


def _target_columns(*, scan_type: str, h_start: int, h_end: int) -> tuple[str, str, str]:
    """Return (rank_col, raw_col, ok_col) for a cumulative horizon or bucket."""
    if scan_type == "cum":
        return f"y_rank_{h_end}d", f"raw_label_{h_end}d", f"label_ok_{h_end}d"
    if scan_type == "bucket":
        return (
            f"y_rank_bucket_{h_start}_{h_end}d",
            f"raw_bucket_label_{h_start}_{h_end}d",
            f"bucket_ok_{h_start}_{h_end}d",
        )
    raise ValueError(f"unknown scan_type {scan_type!r}")


def build_formation_sql(
    *,
    panel_view: str,
    feature_col: str,
    scan_type: str,
    h_start: int,
    h_end: int,
    universe: str,
    sample_kind: str,
    sample_start: str,
    extra_where: str | None = None,
) -> str:
    """SQL selecting the formation-eligible rows for one scan cell (§3.2/§3.3).

    ``sample_kind="common_survivor"`` adds the 120d common-formation/survivor
    gate on top; ``"available"`` uses only this horizon/bucket's own
    pre-holdout label validity — ``label_ok``/``bucket_ok`` already encode the
    holdout boundary and CA-masked label finiteness (research/etl/labels.py),
    so no separate holdout-date filter is needed here.

    ``extra_where`` is an additional SQL predicate (e.g. ``market = 'KOSPI'``,
    ``size_segment = 'small'``, a period-id equality) ANDed onto the same
    formation conditions — this is how §A-4's independent market/size/
    liquidity/period/short-regime axes reuse this one formation definition
    instead of each re-deriving eligibility.
    """
    if universe not in _UNIVERSE_FLAG_COLUMN:
        raise ValueError(f"unknown universe {universe!r}")
    rank_col, raw_col, ok_col = _target_columns(scan_type=scan_type, h_start=h_start, h_end=h_end)
    conditions = [
        f"trade_date >= DATE '{sample_start}'",
        _UNIVERSE_FLAG_COLUMN[universe],
        f"{feature_col} IS NOT NULL",
        f"isfinite({feature_col})",
        "NOT ca_mask",
        ok_col,
    ]
    if sample_kind == "common_survivor":
        conditions += ["common_formation_120d", "common_survivor_120d"]
    elif sample_kind != "available":
        raise ValueError(f"unknown sample_kind {sample_kind!r}")
    if extra_where:
        conditions.append(f"({extra_where})")
    where = " AND ".join(conditions)
    return f"""
        SELECT trade_date, ticker, market, formation_session_idx,
               {feature_col} AS feature_value,
               {rank_col} AS target_rank,
               {raw_col} AS target_raw
        FROM {panel_view}
        WHERE {where}
    """


def _filter_formation_frame(
    frame: pl.DataFrame,
    *,
    feature_col: str,
    scan_type: str,
    h_start: int,
    h_end: int,
    universe: str,
    sample_kind: str,
    sample_start: str,
) -> pl.DataFrame:
    """Apply the formation SQL contract to a feature-wide fetched frame."""
    rank_col, raw_col, ok_col = _target_columns(scan_type=scan_type, h_start=h_start, h_end=h_end)
    if universe not in _UNIVERSE_FLAG_COLUMN:
        raise ValueError(f"unknown universe {universe!r}")
    required = {
        "trade_date",
        "ticker",
        "market",
        "formation_session_idx",
        "ca_mask",
        feature_col,
        rank_col,
        raw_col,
        ok_col,
        _UNIVERSE_FLAG_COLUMN[universe],
    }
    if sample_kind == "common_survivor":
        required.update({"common_formation_120d", "common_survivor_120d"})
    elif sample_kind != "available":
        raise ValueError(f"unknown sample_kind {sample_kind!r}")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"formation frame is missing columns: {missing}")
    filtered = frame.filter(
        (pl.col("trade_date") >= pl.lit(sample_start).str.to_date())
        & pl.col(_UNIVERSE_FLAG_COLUMN[universe]).fill_null(False)
        & pl.col(feature_col).is_not_null()
        & pl.col(feature_col).is_finite()
        & ~pl.col("ca_mask").fill_null(True)
        & pl.col(ok_col).fill_null(False)
    )
    if sample_kind == "common_survivor":
        filtered = filtered.filter(
            pl.col("common_formation_120d").fill_null(False)
            & pl.col("common_survivor_120d").fill_null(False)
        )
    return filtered.select(
        [
            "trade_date",
            "ticker",
            "market",
            "formation_session_idx",
            pl.col(feature_col).alias("feature_value"),
            pl.col(rank_col).alias("target_rank"),
            pl.col(raw_col).alias("target_raw"),
        ]
    ).sort(["trade_date", "market", "ticker"])


def scan_cell(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str = "analysis_panel",
    feature_col: str,
    scan_type: str,
    h_start: int,
    h_end: int,
    universe: str,
    sample_kind: str,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    expected_sign: str | None = None,
    extra_where: str | None = None,
    compute_spread: bool = True,
    scan_engine: str = "legacy",
    formation_frame: pl.DataFrame | None = None,
    cell_identity: dict[str, Any] | None = None,
    daily_sink: DailyIcSink | None = None,
) -> dict[str, Any]:
    """One (feature, horizon/bucket, universe, sample_kind) cell's IC/NW/spread.

    Returns a dict matching the subset of the ``horizon_ic.parquet`` schema
    (§6.2) that A-2 owns; BH/segment/robustness fields are left for later
    stages. ``status`` is one of ``"valid"``, ``"insufficient"`` (with a
    ``status_reason``) — never silently dropped from the registry.

    ``compute_spread=False`` skips the quantile-spread computation (q5_spread_*
    stay ``None``) — the A-6 permutation/placebo replicate loops call this
    tens of thousands of times and never read the spread fields, so skipping
    it is a pure performance win with no effect on IC/NW/BH.

    ``daily_sink`` (Stage 0) is a pure side channel: the per-date frames this
    function already built are handed to it once the cell is ``valid``, and
    nothing about the returned dict or the path that produced it changes.
    Omitting it — as every replicate, period-split and lag1 caller does —
    leaves the old behaviour byte for byte. ``cell_identity`` is the
    normalized registry identity written onto those rows and is required
    whenever a sink is given.
    """
    if daily_sink is not None and cell_identity is None:
        raise ValueError("daily_sink requires cell_identity")
    lag = choose_nw_lag(
        scan_type=scan_type,
        horizon=h_end if scan_type == "cum" else None,
        bucket_width=(h_end - h_start) if scan_type == "bucket" else None,
    )
    min_dates_required = max(min_dates_per_cell, lag + 2)
    result: dict[str, Any] = {
        "scan_type": scan_type,
        "h_start": h_start,
        "h_end": h_end,
        "width": h_end - h_start,
        "universe": universe,
        "sample_kind": sample_kind,
        "effective_sample_start": None,
        "effective_sample_end": None,
        "n_dates": 0,
        "n_obs": 0,
        "n_obs_mean": None,
        "n_obs_min": None,
        "n_obs_median": None,
        "ic_mean": None,
        "ic_std": None,
        "icir": None,
        "t_naive": None,
        "t_nw": None,
        "p_nw": None,
        "n_hac_pairs_min": None,
        "kospi_weight_mean": None,
        "kosdaq_weight_mean": None,
        "q5_spread_raw": None,
        "q5_spread_aligned": None,
        "status": "insufficient",
        "status_reason": None,
    }
    if formation_frame is not None:
        if extra_where:
            raise ValueError("extra_where cannot be combined with formation_frame")
        frame = _filter_formation_frame(
            formation_frame,
            feature_col=feature_col,
            scan_type=scan_type,
            h_start=h_start,
            h_end=h_end,
            universe=universe,
            sample_kind=sample_kind,
            sample_start=sample_start,
        )
    else:
        formation_sql = build_formation_sql(
            panel_view=panel_view,
            feature_col=feature_col,
            scan_type=scan_type,
            h_start=h_start,
            h_end=h_end,
            universe=universe,
            sample_kind=sample_kind,
            sample_start=sample_start,
            extra_where=extra_where,
        )
        frame = con.execute(formation_sql).pl().sort(["trade_date", "market", "ticker"])
    if frame.is_empty():
        result["status_reason"] = "no_formation_rows"
        return result
    result["effective_sample_start"] = frame["trade_date"].min()
    result["effective_sample_end"] = frame["trade_date"].max()

    market_ic = per_date_market_rank_ic(
        frame,
        pred_col="feature_value",
        realized_col="target_rank",
        min_names=min_names,
        engine=scan_engine,
    )
    market_ic = market_ic.filter(pl.col("rank_ic").is_finite())
    if market_ic.is_empty():
        result["status_reason"] = "no_valid_cross_section"
        return result
    n_obs_series = market_ic["n"].to_numpy()
    result["n_obs"] = int(n_obs_series.sum())
    result["n_obs_mean"] = float(n_obs_series.mean())
    result["n_obs_min"] = int(n_obs_series.min())
    result["n_obs_median"] = float(np.median(n_obs_series))
    result.update(market_weight_means(market_ic))

    daily = daily_market_weighted_ic(market_ic)
    session_by_date = frame.select(["trade_date", "formation_session_idx"]).unique("trade_date")
    daily = daily.join(session_by_date, on="trade_date", how="left").sort("trade_date")
    daily = daily.filter(pl.col("rank_ic").is_finite())
    n_dates = daily.height
    result["n_dates"] = n_dates
    if n_dates < min_dates_required:
        result["status_reason"] = f"insufficient_dates:{n_dates}<{min_dates_required}"
        return result

    values = daily["rank_ic"].to_numpy()
    sessions = daily["formation_session_idx"].to_numpy()
    ic_mean = float(values.mean())
    ic_std = float(values.std(ddof=1)) if n_dates > 1 else float("nan")
    finite_std = bool(ic_std) and math.isfinite(ic_std) and ic_std != 0
    icir = ic_mean / ic_std if finite_std else float("nan")
    t_naive = ic_mean / (ic_std / math.sqrt(n_dates)) if finite_std else float("nan")
    t_nw = newey_west_tstat(values, sessions, lag)
    result.update(
        {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "icir": icir,
            "t_naive": t_naive,
            "t_nw": t_nw,
            "p_nw": two_sided_normal_p(t_nw),
            "n_hac_pairs_min": n_hac_pairs(sessions, lag) if lag > 0 else 0,
        }
    )

    market_spread: pl.DataFrame | None = None
    daily_spread: pl.DataFrame | None = None
    if compute_spread:
        market_spread = per_date_market_quantile_spread(
            frame,
            feature_col="feature_value",
            raw_label_col="target_raw",
            n_quantiles=quantile_count,
            min_names=min_names_for_spread,
        )
        daily_spread = daily_market_weighted_spread(market_spread)
        if not daily_spread.is_empty():
            q5_raw = float(daily_spread["spread"].mean())
            sign = {"+": 1.0, "-": -1.0}.get(expected_sign or "", 1.0)
            result["q5_spread_raw"] = q5_raw
            result["q5_spread_aligned"] = sign * q5_raw

    result["status"] = "valid"
    if daily_sink is not None:
        daily_sink.emit(
            cell_identity,
            daily=daily,
            market_ic=market_ic,
            daily_spread=daily_spread,
            market_spread=market_spread,
        )
    return result


# --- A-3 global BH-FDR and core discovery flags (§2.3, §4.4) ---


def assert_rows_match_registry(rows: list[dict[str, Any]], registry: list[dict[str, Any]]) -> None:
    """§2.3: the scanned rows must be exactly the registered hypothesis set —
    no extra cells inflating ``m``, none silently dropped."""
    row_ids = {r["hypothesis_id"] for r in rows}
    registry_ids = {r["hypothesis_id"] for r in registry}
    if row_ids != registry_ids:
        missing = registry_ids - row_ids
        extra = row_ids - registry_ids
        raise ValueError(f"scanned rows do not match the registry: missing={missing} extra={extra}")
    if len(rows) != len(row_ids):
        raise ValueError("scanned rows contain duplicate hypothesis_id values")


def assert_unique_hypothesis_ids(rows: list[dict[str, Any]]) -> None:
    ids = [row.get("hypothesis_id") for row in rows]
    if any(hid is None for hid in ids):
        raise ValueError("combined BH rows must all have a hypothesis_id")
    duplicates = sorted({hid for hid in ids if ids.count(hid) > 1})
    if duplicates:
        raise ValueError(f"combined BH population contains duplicate hypothesis_id: {duplicates}")


def _aligned_ic(row: dict[str, Any]) -> float | None:
    ic = row.get("ic_mean")
    if ic is None:
        return None
    sign = row.get("expected_sign")
    if sign == "-":
        return -ic
    return ic  # "+" and two-sided (None) both read the observed IC directly


def compute_isolated_spikes(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """§4.4: a candidate cell (positive aligned IC) is an isolated spike if
    every registered neighbor on the *same* scan axis (cumulative or bucket,
    within the same family) is opposite-signed or not computable. A grid-end
    cell is judged against its one available neighbor.
    """
    by_family_scan: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_family_scan.setdefault((row["family"], row["scan_type"]), []).append(row)

    result: dict[str, bool] = {}
    for cells in by_family_scan.values():
        ordered = sorted(cells, key=lambda r: r["h_end"])
        for i, row in enumerate(ordered):
            neighbors = ([ordered[i - 1]] if i > 0 else []) + (
                [ordered[i + 1]] if i < len(ordered) - 1 else []
            )
            aligned = _aligned_ic(row)
            if aligned is None or aligned <= 0 or not neighbors:
                result[row["hypothesis_id"]] = False
                continue
            all_bad = all((_aligned_ic(n) or 0) <= 0 for n in neighbors)
            result[row["hypothesis_id"]] = bool(all_bad)
    return result


def apply_global_bh(
    rows: list[dict[str, Any]],
    *,
    q_threshold: float = 0.10,
) -> list[dict[str, Any]]:
    """One global BH-FDR pass across the fixed primary hypothesis registry (§2.3).

    ``rows`` must already carry ``hypothesis_id``, ``family``, ``scan_type``,
    ``h_end``, ``status``, ``p_nw``, ``expected_sign``, ``ic_mean`` (i.e. the
    merged output of :func:`scan_cell` plus the registry's own fields).
    Insufficient/constant cells get ``p_for_bh=1.0`` (rule 4) so ``m`` never
    shrinks; ties break on ``hypothesis_id`` (rule 5); ``q_fdr_global < 0.10``
    is strict, not ``<=`` (rule 6).
    """
    ordered = sorted(rows, key=lambda r: r["hypothesis_id"])
    p_for_bh = np.array(
        [
            (
                r["p_nw"]
                if r.get("status") == "valid"
                and r.get("p_nw") is not None
                and math.isfinite(r["p_nw"])
                else 1.0
            )
            for r in ordered
        ]
    )
    q_values = benjamini_hochberg(p_for_bh)
    isolated_spike_by_id = compute_isolated_spikes(ordered)

    out: list[dict[str, Any]] = []
    for row, p_bh, q in zip(ordered, p_for_bh, q_values):
        row = dict(row)
        row["p_for_bh"] = float(p_bh)
        row["q_fdr_global"] = float(q) if math.isfinite(q) else None
        bh_pass = row["q_fdr_global"] is not None and row["q_fdr_global"] < q_threshold
        row["bh_pass"] = bh_pass

        expected_sign = row.get("expected_sign")
        ic_mean = row.get("ic_mean")
        if expected_sign in ("+", "-") and ic_mean is not None:
            sign = 1.0 if expected_sign == "+" else -1.0
            row["expected_sign_pass"] = bool(sign * ic_mean > 0)
        else:
            row["expected_sign_pass"] = None  # two-sided family: no directional gate (rule 3)

        row["isolated_spike"] = isolated_spike_by_id.get(row["hypothesis_id"], False)
        row["primary_discovery"] = bool(
            row.get("status") == "valid"
            and bh_pass
            and row["expected_sign_pass"] is not False
            and not row["isolated_spike"]
        )
        out.append(row)
    return out


# --- A-2 registry-level orchestration ---

UNIVERSE_SAMPLE_COMBOS: tuple[tuple[str, str], ...] = (
    ("broad", "common_survivor"),
    ("broad", "available"),
    ("tradable", "common_survivor"),
    ("tradable", "available"),
)


def run_registry_scan(
    con: duckdb.DuckDBPyConnection,
    registry: list[dict[str, Any]],
    *,
    panel_view: str = "analysis_panel",
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    universe_sample_combos: tuple[tuple[str, str], ...] = UNIVERSE_SAMPLE_COMBOS,
    scan_engine: str = "legacy",
    reuse_formation_frames: bool = True,
    daily_sink: DailyIcSink | None = None,
) -> list[dict[str, Any]]:
    """Run :func:`scan_cell` for every registered hypothesis across every
    (universe, sample_kind) combo (A-2 steps 4-5) — one row per combination,
    each carrying the hypothesis's own registry fields alongside the cell's
    computed stats so downstream stages never need to re-join back to config.

    ``daily_sink`` (Stage 0) persists each valid cell's per-date IC/spread
    series. This is the *only* ``scan_cell`` caller that passes one, which is
    what keeps the replicate loops, the period-split scans and the direct lag1
    calls out of ``daily_ic.parquet`` (and their cost unchanged). Flushing
    happens on the same per-feature boundary the formation-frame reuse already
    works in, so the sink's peak memory is one feature's daily rows.
    """
    rows: list[dict[str, Any]] = []

    def _identity(hyp: dict[str, Any], universe: str, sample_kind: str) -> dict[str, Any] | None:
        if daily_sink is None:
            return None
        return normalize_cell_identity(hyp, universe=universe, sample_kind=sample_kind)

    if reuse_formation_frames and registry:
        # A feature frame is only live while that feature's hypotheses are
        # being scanned. Holding the whole registry's frames here multiplies
        # the panel-sized allocation by the number of features.
        hypotheses_by_feature: dict[str, list[dict[str, Any]]] = {}
        for hyp in registry:
            hypotheses_by_feature.setdefault(hyp["feature"], []).append(hyp)
        for feature_registry in hypotheses_by_feature.values():
            feature = feature_registry[0]["feature"]
            target_cols = sorted(
                {
                    col
                    for hyp in feature_registry
                    for col in _target_columns(
                        scan_type=hyp["scan_type"], h_start=hyp["h_start"], h_end=hyp["h_end"]
                    )
                }
            )
            cols = [
                "trade_date",
                "ticker",
                "market",
                "formation_session_idx",
                "ca_mask",
                "in_broad",
                "in_tradable",
                "common_formation_120d",
                "common_survivor_120d",
                feature,
                *target_cols,
            ]
            selected = ", ".join(dict.fromkeys(cols))
            frame = (
                con.execute(
                    f"SELECT {selected} FROM {panel_view} "
                    f"WHERE trade_date >= DATE '{sample_start}'"
                )
                .pl()
                .sort(["trade_date", "market", "ticker"])
            )

            for hyp in feature_registry:
                for universe, sample_kind in universe_sample_combos:
                    cell = scan_cell(
                        con,
                        panel_view=panel_view,
                        feature_col=hyp["feature"],
                        scan_type=hyp["scan_type"],
                        h_start=hyp["h_start"],
                        h_end=hyp["h_end"],
                        universe=universe,
                        sample_kind=sample_kind,
                        sample_start=sample_start,
                        min_names=min_names,
                        min_names_for_spread=min_names_for_spread,
                        quantile_count=quantile_count,
                        min_dates_per_cell=min_dates_per_cell,
                        expected_sign=hyp.get("expected_sign"),
                        scan_engine=scan_engine,
                        formation_frame=frame,
                        cell_identity=_identity(hyp, universe, sample_kind),
                        daily_sink=daily_sink,
                    )
                    rows.append({**hyp, **cell})
            if daily_sink is not None:
                daily_sink.flush_feature(feature)
        return rows

    for hyp in registry:
        for universe, sample_kind in universe_sample_combos:
            cell = scan_cell(
                con,
                panel_view=panel_view,
                feature_col=hyp["feature"],
                scan_type=hyp["scan_type"],
                h_start=hyp["h_start"],
                h_end=hyp["h_end"],
                universe=universe,
                sample_kind=sample_kind,
                sample_start=sample_start,
                min_names=min_names,
                min_names_for_spread=min_names_for_spread,
                quantile_count=quantile_count,
                min_dates_per_cell=min_dates_per_cell,
                expected_sign=hyp.get("expected_sign"),
                scan_engine=scan_engine,
                cell_identity=_identity(hyp, universe, sample_kind),
                daily_sink=daily_sink,
            )
            rows.append({**hyp, **cell})
    # Without frame reuse the registry is walked in its own order, so a
    # feature's hypotheses need not be contiguous; flushing per feature here
    # would risk a second write for the same file. Flush all features once the
    # walk is over instead — the memory argument above only applies to the
    # reuse path, which is what every real run uses.
    if daily_sink is not None:
        for feature in dict.fromkeys(entry["feature"] for entry in registry):
            daily_sink.flush_feature(feature)
    return rows


# --- A-4 robustness gates (§A-4) ---
#
# These take already-computed cell ICs/p-values as plain arguments — they do
# not query the panel themselves. A gate's own segment/variant cell (e.g. the
# tradable-universe IC, or the lag1-variant IC) is produced by the same
# `scan_cell` used for the primary discovery cell; only the pass/fail rule
# lives here.


def compute_tradable_pass(
    *, ic_broad: float | None, ic_tradable: float | None, min_retention: float = 0.50
) -> dict[str, Any]:
    """§A-4: tradable retention = |IC_tradable| / |IC_broad|; pass needs the
    same sign AND retention >= threshold. A zero/non-finite broad IC can't
    anchor a ratio, so retention is NULL and the gate fails (not "N/A")."""
    if (
        ic_broad is None
        or ic_tradable is None
        or not math.isfinite(ic_broad)
        or not math.isfinite(ic_tradable)
        or ic_broad == 0
    ):
        return {"tradable_retention": None, "tradable_pass": False}
    retention = abs(ic_tradable) / abs(ic_broad)
    same_direction = (ic_broad > 0) == (ic_tradable > 0)
    return {
        "tradable_retention": retention,
        "tradable_pass": bool(same_direction and retention >= min_retention),
    }


def compute_period_sign_pass(
    period_ics: list[float | None], *, expected_sign: str | None
) -> dict[str, Any]:
    """§A-4: majority-of-valid-subperiods sign check. ``None`` entries are
    ``insufficient`` periods and are excluded from both the denominator and
    numerator (§3.4) — they neither help nor hurt ``period_sign_pass``.
    Two-sided families (``expected_sign is None``) use the *observed* sign of
    the aligned IC directly (matches ``_aligned_ic``'s convention).
    """
    valid = [ic for ic in period_ics if ic is not None and math.isfinite(ic)]
    if not valid:
        return {
            "valid_subperiods": 0,
            "sign_consistent_subperiods": 0,
            "period_sign_pass": False,
        }
    sign = -1.0 if expected_sign == "-" else 1.0
    consistent = sum(1 for ic in valid if sign * ic > 0)
    return {
        "valid_subperiods": len(valid),
        "sign_consistent_subperiods": consistent,
        "period_sign_pass": bool(consistent > len(valid) / 2),
    }


def compute_available_direction_pass(
    *, ic_common_survivor: float | None, ic_available: float | None
) -> dict[str, Any]:
    """§A-4: the available (max-coverage) sample must agree in sign with the
    common-survivor discovery — a reversal is attrition, not confirmation."""
    if (
        ic_common_survivor is None
        or ic_available is None
        or not math.isfinite(ic_common_survivor)
        or not math.isfinite(ic_available)
    ):
        return {"available_direction_pass": None}
    return {"available_direction_pass": bool((ic_common_survivor > 0) == (ic_available > 0))}


def compute_delay_pass(
    *,
    ic_native: float | None,
    ic_lag1: float | None,
    p_nw_lag1: float | None,
    min_retention: float = 0.50,
    p_max: float = 0.05,
) -> dict[str, Any]:
    """§A-4 delay gate (only evaluated for h<=5 / bucket (0,5] evidence):
    same direction, >=50% IC retention at a one-session delay, and the
    delayed variant's own NW p-value must still clear 0.05 — a family whose
    official variant is already lag1 evaluates this against itself."""
    if (
        ic_native is None
        or ic_lag1 is None
        or not math.isfinite(ic_native)
        or not math.isfinite(ic_lag1)
        or ic_native == 0
    ):
        return {"delay_retention": None, "delay_pass": False}
    same_direction = (ic_native > 0) == (ic_lag1 > 0)
    retention = abs(ic_lag1) / abs(ic_native)
    p_ok = p_nw_lag1 is not None and math.isfinite(p_nw_lag1) and p_nw_lag1 < p_max
    return {
        "delay_retention": retention,
        "delay_pass": bool(same_direction and retention >= min_retention and p_ok),
    }


# --- A-5 all-offset non-overlap subsampling (§A-5) ---


def build_offset_formation_sql(
    *,
    panel_view: str,
    feature_col: str,
    scan_type: str,
    h_start: int,
    h_end: int,
    universe: str,
    sample_kind: str,
    sample_start: str,
    stride: int,
    offset: int,
) -> str:
    """The §A-2 formation SQL further restricted to one non-overlapping offset:
    ``formation_ordinal % stride = offset`` (§A-5) — cumulative h uses
    ``stride=h``, bucket width w uses ``stride=w``, over the shared KRX
    calendar ordinal (``formation_session_idx``), never re-derived per family.
    """
    base = build_formation_sql(
        panel_view=panel_view,
        feature_col=feature_col,
        scan_type=scan_type,
        h_start=h_start,
        h_end=h_end,
        universe=universe,
        sample_kind=sample_kind,
        sample_start=sample_start,
    )
    return f"SELECT * FROM ({base}) offset_base WHERE formation_session_idx % {stride} = {offset}"


def scan_offset(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str = "analysis_panel",
    feature_col: str,
    scan_type: str,
    h_start: int,
    h_end: int,
    universe: str,
    sample_kind: str,
    sample_start: str,
    stride: int,
    offset: int,
    min_names: int,
    nonoverlap_min_dates: int,
    alignment_sign: float,
    scan_engine: str = "legacy",
) -> dict[str, Any]:
    """One non-overlapping offset's IC mean and exact sign-test p (§A-5).

    ``alignment_sign`` (+1/-1) is fixed once for the whole cell from the
    family's ``expected_sign``, or from the *observed* discovery sign for a
    two-sided family — never re-derived per offset (that would let a
    favorable offset pick its own direction).
    """
    result: dict[str, Any] = {
        "offset": offset,
        "n_dates": 0,
        "ic_mean": None,
        "n_success": None,
        "n_trials": None,
        "p_sign_test": None,
        "status": "insufficient",
    }
    sql = build_offset_formation_sql(
        panel_view=panel_view,
        feature_col=feature_col,
        scan_type=scan_type,
        h_start=h_start,
        h_end=h_end,
        universe=universe,
        sample_kind=sample_kind,
        sample_start=sample_start,
        stride=stride,
        offset=offset,
    )
    frame = con.execute(sql).pl()
    if frame.is_empty():
        return result
    market_ic = per_date_market_rank_ic(
        frame,
        pred_col="feature_value",
        realized_col="target_rank",
        min_names=min_names,
        engine=scan_engine,
    )
    market_ic = market_ic.filter(pl.col("rank_ic").is_finite())
    if market_ic.is_empty():
        return result
    daily = daily_market_weighted_ic(market_ic)
    daily = daily.filter(pl.col("rank_ic").is_finite())
    n_dates = daily.height
    result["n_dates"] = n_dates
    if n_dates < nonoverlap_min_dates:
        return result

    values = daily["rank_ic"].to_numpy()
    result["ic_mean"] = float(values.mean())
    aligned = alignment_sign * values
    nonzero = aligned[aligned != 0]
    n_trials = int(nonzero.size)
    if n_trials < nonoverlap_min_dates:
        return result
    n_success = int((nonzero > 0).sum())
    result["n_success"] = n_success
    result["n_trials"] = n_trials
    result["p_sign_test"] = exact_binomial_sign_test_p(n_success, n_trials)
    result["status"] = "valid"
    return result


def run_nonoverlap_offsets(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str = "analysis_panel",
    feature_col: str,
    scan_type: str,
    h_start: int,
    h_end: int,
    universe: str,
    sample_kind: str,
    sample_start: str,
    min_names: int,
    nonoverlap_min_dates: int,
    alignment_sign: float,
    scan_engine: str = "legacy",
) -> dict[str, Any]:
    """All-offset non-overlap summary for one cell (§A-5 completion criterion:
    every valid primary cell gets an offset summary or an explicit
    insufficient reason — never a single "best" offset picked as if
    representative).
    """
    stride = h_end if scan_type == "cum" else (h_end - h_start)
    offsets = [
        scan_offset(
            con,
            panel_view=panel_view,
            feature_col=feature_col,
            scan_type=scan_type,
            h_start=h_start,
            h_end=h_end,
            universe=universe,
            sample_kind=sample_kind,
            sample_start=sample_start,
            stride=stride,
            offset=o,
            min_names=min_names,
            nonoverlap_min_dates=nonoverlap_min_dates,
            alignment_sign=alignment_sign,
            scan_engine=scan_engine,
        )
        for o in range(stride)
    ]
    valid = [o for o in offsets if o["status"] == "valid"]
    n_offsets_total = stride
    n_offsets_valid = len(valid)
    summary: dict[str, Any] = {
        "n_offsets_total": n_offsets_total,
        "n_offsets_valid": n_offsets_valid,
        "offset_ic_mean_median": None,
        "offset_ic_mean_min": None,
        "offset_ic_mean_max": None,
        "offset_sign_agreement_ratio": None,
        "offset_sign_test_p_median": None,
        "offset_sign_test_p_min": None,
        "offset_sign_test_p_max": None,
        "offset_status": "insufficient",
        "offsets": offsets,
    }
    if not valid:
        return summary

    ic_means = np.array([o["ic_mean"] for o in valid])
    p_values = np.array([o["p_sign_test"] for o in valid])
    summary.update(
        {
            "offset_ic_mean_median": float(np.median(ic_means)),
            "offset_ic_mean_min": float(ic_means.min()),
            "offset_ic_mean_max": float(ic_means.max()),
            "offset_sign_agreement_ratio": float(np.mean(alignment_sign * ic_means > 0)),
            "offset_sign_test_p_median": float(np.median(p_values)),
            "offset_sign_test_p_min": float(p_values.min()),
            "offset_sign_test_p_max": float(p_values.max()),
            "offset_status": (
                "complete" if n_offsets_valid == n_offsets_total else "some_insufficient"
            ),
        }
    )
    return summary
