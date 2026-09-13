"""Phase B B-7 core scan (04_specific_plan_B.md §5, §6 B-7) plus the combined
A+B BH pass B-9 needs (§6 B-9 steps 1-2) — "서로 다른 grain을 공통 결과
schema로 투영한다."

Continuous families (7 families, 32 candidate cells) reuse Phase A's
``scan_cell``/``run_registry_scan`` (``horizon_scan_runner.py``) verbatim
against a Phase-B-extended panel that LEFT JOINs ``feat_fin_scan_daily``/
``feat_event_scan_daily`` onto the existing ``analysis_panel`` view — the
grain ``(trade_date, ticker, market)`` and every eligibility/quality/universe
column the formation SQL needs are already there; only the two Phase B
feature marts need adding.

The SUE event family (1 family, 6 event-bucket candidate cells) has a
genuinely different grain — ``(ticker, original_rcept_no,
event_formation_date, market)``, not ``(trade_date, ticker, market)`` — so
``scan_cell``'s daily formation SQL does not apply (§5.2). ``scan_event_cohort_cell``
implements §5.4's cohort-rank-and-pool algorithm instead:

    1. within each ``(event_formation_date, market)`` cohort, SUE and bucket
       excess return are each converted to an average-rank percentile
       (tie-aware, matching every other rank-IC computation in this repo);
       a market only contributes its ranks for a date once it has at least
       ``min_events_per_market_contribution`` events that day.
    2. per formation date, the (already market-neutral) rank pairs from
       every contributing market are pooled — not re-ranked — and one
       Pearson correlation of the pooled percentile ranks is computed once
       the pooled count reaches ``min_events_per_cohort_total``.
    3. the resulting one-value-per-formation-date series is gap-aware
       Newey-West corrected using the *global* (ticker-independent) trading
       session index of each formation date — distinct from
       ``fin_sue_event``'s own per-ticker ``d_idx``, which only orders one
       ticker's own bucket-return walk and cannot compare distances across
       different tickers' events.

Every generic math primitive ``scan_cell`` uses (``newey_west_tstat``,
``n_hac_pairs``, ``two_sided_normal_p``, ``choose_nw_lag``,
``market_weight_means``) is reused unchanged here, and the output dict uses
scan_cell's own field names 1:1 wherever the concepts line up (``n_dates`` is
the cohort/formation-date count, ``n_obs*`` are the per
``(event_formation_date, market)`` qualifying group sizes) — this is what
lets continuous and event rows share one downstream schema, plus one
event-only diagnostic field, ``n_independent_filing_windows`` (§5.4: "분기
마감 주변 소수 window에 몰리는 의존성을 숨기지 않는다").

Scope for this PR (my/milestones/kr/modeling/dev/20260731_raw_features/01_feature_candidate/
04_specific_plan_B.md §10 splits B-PR8 from B-PR9): only the *primary* SUE
sample (``is_primary_constant_sample``) is scanned here. The secondary/
available sample, segment diagnostics, and the bootstrap/permutation
robustness gates (§6 B-8) are B-PR9's scope — deliberately not built here.
"""

from __future__ import annotations

import math
from typing import Any

import duckdb
import numpy as np
import polars as pl

from modeler.analysis.horizon_scan_daily_ic import DailyIcSink
from modeler.analysis.horizon_scan_phase_b_source_quality import source_quality_allows_grade_a
from modeler.analysis.horizon_scan_runner import (
    apply_global_bh,
    assert_panel_join_preserves_keys,
    assert_rows_match_registry,
    assert_unique_hypothesis_ids,
    run_registry_scan,
)
from modeler.etl.features.sue_event import _bucket_col
from modeler.etl.metrics import (
    choose_nw_lag,
    market_weight_means,
    n_hac_pairs,
    newey_west_tstat,
    two_sided_normal_p,
)

PHASE_B_PANEL_VIEW = "analysis_panel_phase_b"

# §2.3's fixed (universe, sample_kind) discovery coordinate — the one row per
# continuous cell that becomes this cell's "primary" statistic.
_DISCOVERY_UNIVERSE = "broad"
_DISCOVERY_SAMPLE_KIND = "common_survivor"


def build_phase_b_panel_sql(
    *,
    analysis_panel_view: str = "analysis_panel",
    fin_scan_view: str | None = "feat_fin_scan_daily",
    event_scan_view: str | None = "feat_event_scan_daily",
    additional_daily_views: tuple[str, ...] = (),
) -> str:
    """LEFT JOIN Phase B daily feature marts onto the Phase A panel.

    Both marts are one row per raw ``daily_ohlcv`` row (B-4/B-5 docstrings) —
    the same grain ``analysis_panel_view`` is already keyed on — so this
    cannot fan out; ``register_phase_b_panel`` still verifies it via
    ``assert_panel_join_preserves_keys``.

    Either mart view may be passed as ``None`` when the real-lake orchestrator
    (``horizon_scan_phase_b_run.py``) could not materialize it in this run
    (a genuinely absent upstream raw source, e.g. ``dart_capital_change_raw``
    not collected yet) — the corresponding columns are simply omitted from the
    panel. ``readiness_dependencies`` is what keeps a family that needs the
    missing mart out of ``ready_primary`` in the first place (§6 B-0); this
    only prevents the *other* mart's families from being blocked too by a
    join that would otherwise fail to compile against a nonexistent view.
    """
    select_cols = ["ap.*"]
    joins = []
    if fin_scan_view is not None:
        select_cols.append("fs.* EXCLUDE (trade_date, ticker, market)")
        joins.append(f"LEFT JOIN {fin_scan_view} fs USING (trade_date, ticker, market)")
    if event_scan_view is not None:
        select_cols.append("es.* EXCLUDE (trade_date, ticker, market)")
        joins.append(f"LEFT JOIN {event_scan_view} es USING (trade_date, ticker, market)")
    for index, daily_view in enumerate(additional_daily_views):
        alias = f"x{index}"
        select_cols.append(f"{alias}.* EXCLUDE (trade_date, ticker, market)")
        joins.append(f"LEFT JOIN {daily_view} {alias} USING (trade_date, ticker, market)")
    return f"""
        SELECT
            {", ".join(select_cols)}
        FROM {analysis_panel_view} ap
        {" ".join(joins)}
    """


def register_phase_b_panel(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = PHASE_B_PANEL_VIEW,
    analysis_panel_view: str = "analysis_panel",
    fin_scan_view: str | None = "feat_fin_scan_daily",
    event_scan_view: str | None = "feat_event_scan_daily",
    additional_daily_views: tuple[str, ...] = (),
) -> str:
    """Register the Phase B panel view; raises ``RuntimeError`` if it fanned out."""
    con.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS "
        + build_phase_b_panel_sql(
            analysis_panel_view=analysis_panel_view,
            fin_scan_view=fin_scan_view,
            event_scan_view=event_scan_view,
            additional_daily_views=additional_daily_views,
        )
    )
    assert_panel_join_preserves_keys(con, view_name, analysis_panel_view)
    return view_name


# --- continuous families: pure reuse of Phase A's runner ---


def run_phase_b_continuous_scan(
    con: duckdb.DuckDBPyConnection,
    ready_continuous_cells: list[dict[str, Any]],
    *,
    panel_view: str = PHASE_B_PANEL_VIEW,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    scan_engine: str = "legacy",
    daily_sink: DailyIcSink | None = None,
) -> list[dict[str, Any]]:
    """Scan every ready continuous candidate cell via Phase A's own runner.

    ``ready_continuous_cells`` must be ``role == "ready_primary"`` rows from
    ``horizon_scan_phase_b.build_phase_b_readiness_rows`` with
    ``cell_type in ("cumulative", "bucket")`` — blocked cells are never
    queried here (their dependency mart view may not even be registered).
    Returns ``len(ready_continuous_cells) * 4`` rows (one per universe ×
    sample_kind combo), exactly like ``run_registry_scan`` for Phase A.

    ``daily_sink`` (Stage 0) is handed straight through. The SUE event family
    has its own grain and its own scan, so it contributes nothing here —
    ``cohort_ic.parquet`` is deliberately not written (00_overview §1.4).
    """
    registry = [
        {**cell, "scan_type": "cum" if cell["cell_type"] == "cumulative" else "bucket"}
        for cell in ready_continuous_cells
    ]
    return run_registry_scan(
        con,
        registry,
        panel_view=panel_view,
        sample_start=sample_start,
        min_names=min_names,
        min_names_for_spread=min_names_for_spread,
        quantile_count=quantile_count,
        min_dates_per_cell=min_dates_per_cell,
        scan_engine=scan_engine,
        daily_sink=daily_sink,
    )


# --- SUE event family: cohort rank-IC/NW (§5.4) ---


def build_event_cohort_frame_sql(
    *,
    event_view: str = "fin_sue_event",
    calendar_view: str = "daily_ohlcv",
    sue_col: str = "fin_sue",
    h_start: int,
    h_end: int,
    sample_start: str,
) -> str:
    """Primary-sample SUE events for one bucket, plus the *global* (not
    per-ticker) trading-session index of their formation date.

    ``ticker`` rides along (unused by the core rank-IC computation) so
    ``horizon_scan_phase_b_robustness.py``'s issuer-cluster bootstrap can
    resample this same frame by issuer without a second query.
    """
    excess_col = _bucket_col(h_start, h_end, "excess")
    return f"""
        WITH calendar AS (
            SELECT trade_date, ROW_NUMBER() OVER (ORDER BY trade_date) AS session_idx
            FROM (SELECT DISTINCT trade_date FROM {calendar_view})
        )
        SELECT
            e.ticker, e.original_rcept_no, e.event_formation_date, e.market,
            e.bsns_year, e.reprt_code,
            c.session_idx AS formation_session_idx,
            e.{sue_col} AS sue_value,
            e.{excess_col} AS excess_value
        FROM {event_view} e
        JOIN calendar c ON c.trade_date = e.event_formation_date
        WHERE e.is_primary_constant_sample
          AND e.event_formation_date >= DATE '{sample_start}'
          AND e.{sue_col} IS NOT NULL
          AND e.{excess_col} IS NOT NULL
    """


def execute_event_cohort_frame(con: duckdb.DuckDBPyConnection, **kwargs: Any) -> pl.DataFrame:
    """Fetch a SUE frame, with a deterministic fallback for old fixtures."""
    sql = build_event_cohort_frame_sql(**kwargs)
    try:
        frame = con.execute(sql).pl()
    except duckdb.BinderException as exc:
        if "original_rcept_no" not in str(exc):
            raise
        frame = con.execute(sql.replace("e.original_rcept_no, ", "")).pl()
    if "original_rcept_no" not in frame.columns:
        frame = frame.with_columns(
            pl.concat_str(
                [
                    pl.col("ticker"),
                    pl.col("event_formation_date").cast(pl.Utf8),
                    pl.col("market"),
                ],
                separator="|",
            ).alias("original_rcept_no")
        )
    return frame


def _empty_event_cohort_result(h_start: int, h_end: int) -> dict[str, Any]:
    return {
        "h_start": h_start,
        "h_end": h_end,
        "width": h_end - h_start,
        "effective_sample_start": None,
        "effective_sample_end": None,
        "n_dates": 0,
        "n_obs": 0,
        "n_obs_mean": None,
        "n_obs_min": None,
        "n_obs_median": None,
        "n_independent_filing_windows": 0,
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


def _pool_cohort_ranks(
    frame: pl.DataFrame,
    *,
    min_events_per_market_contribution: int,
    min_events_per_cohort_total: int,
) -> dict[str, Any]:
    """Common core of §5.4 steps 1-2: rank within ``(event_formation_date,
    market)``, drop groups below the per-market contribution floor, then pool
    by date and keep only dates whose pooled total clears the cohort-total
    floor.

    Deliberately has NO minimum-cohort-*count* gate — that decision belongs
    to the caller. A permutation/bootstrap replicate legitimately wants
    whatever cohort dates a resampled draw happens to qualify, even when that
    count is tiny; only ``scan_event_cohort_cell``'s official statistic
    enforces ``min_event_cohorts``.

    Returns ``{"cohort_rows": [(date, session_idx, rank_ic, n_pooled), ...],
    "market_n_stats": df|None, "qualifying": df|None}`` — ``market_n_stats``/
    ``qualifying`` are ``None`` when no group meets the contribution floor at
    all (as opposed to meeting it but never reaching the cohort-total floor).
    """
    if frame.is_empty():
        return {"cohort_rows": [], "market_n_stats": None, "qualifying": None}
    with_group_n = frame.with_columns(
        pl.len().over(["event_formation_date", "market"]).alias("market_n")
    )
    qualifying = with_group_n.filter(pl.col("market_n") >= min_events_per_market_contribution)
    if qualifying.is_empty():
        return {"cohort_rows": [], "market_n_stats": None, "qualifying": None}
    qualifying = qualifying.with_columns(
        (
            pl.col("sue_value").rank(method="average").over(["event_formation_date", "market"])
            / pl.col("market_n")
        ).alias("sue_pctrank"),
        (
            pl.col("excess_value").rank(method="average").over(["event_formation_date", "market"])
            / pl.col("market_n")
        ).alias("excess_pctrank"),
    )
    market_n_stats = qualifying.select(["event_formation_date", "market", "market_n"]).unique()
    cohort_rows = _pool_qualifying_by_date(
        qualifying, min_events_per_cohort_total=min_events_per_cohort_total
    )
    return {"cohort_rows": cohort_rows, "market_n_stats": market_n_stats, "qualifying": qualifying}


def _pool_qualifying_by_date(
    qualifying: pl.DataFrame, *, min_events_per_cohort_total: int
) -> list[tuple[Any, int, float, int]]:
    """The date-pooling tail of §5.4 steps 1-2, split out of
    ``_pool_cohort_ranks`` so a caller that already has a ``qualifying``
    frame (real or rank-permuted — see
    ``horizon_scan_phase_b_joint_permutation.py``'s §6 B-8 "결합 단면
    permutation") can re-pool it without recomputing the per-market
    contribution floor or the rank columns themselves.
    """
    cohort_rows: list[tuple[Any, int, float, int]] = []
    order_cols = ["formation_session_idx", "event_formation_date", "market"]
    order_cols.extend(col for col in ("ticker", "original_rcept_no") if col in qualifying.columns)
    qualifying = qualifying.sort(order_cols)
    by_date = qualifying.group_by(["event_formation_date"], maintain_order=True)
    for (formation_date,), grp in by_date:
        n_pooled = grp.height
        if n_pooled < min_events_per_cohort_total:
            continue
        sue_pct = grp["sue_pctrank"].to_numpy()
        excess_pct = grp["excess_pctrank"].to_numpy()
        if sue_pct.std() == 0 or excess_pct.std() == 0:
            continue
        rank_ic = float(np.corrcoef(sue_pct, excess_pct)[0, 1])
        if not math.isfinite(rank_ic):
            continue
        session_values = grp["formation_session_idx"].unique().to_list()
        if len(session_values) != 1:
            raise ValueError("event_formation_date maps to multiple formation_session_idx values")
        session_idx = int(session_values[0])
        cohort_rows.append((formation_date, session_idx, rank_ic, n_pooled))
    return cohort_rows


def _aggregate_cohort_rows(
    cohort_rows: list[tuple[Any, int, float, int]], *, lag: int
) -> dict[str, Any]:
    """IC/NW aggregate of an already-pooled cohort series — the tail half of
    §5.4 (steps 3-4), split out so bootstrap/permutation replicates can reuse
    it on a resampled ``cohort_rows`` without re-deriving the NW math."""
    ordered = sorted(cohort_rows, key=lambda r: (int(r[1]), r[0]))
    ic_arr = np.array([r[2] for r in ordered], dtype=float)
    session_arr = np.array([r[1] for r in ordered], dtype=int)
    if session_arr.size > 1 and np.any(np.diff(session_arr) <= 0):
        raise ValueError("cohort formation sessions must be strictly increasing")
    n_cohorts = len(cohort_rows)
    ic_mean = float(ic_arr.mean())
    ic_std = float(ic_arr.std(ddof=1)) if n_cohorts > 1 else float("nan")
    finite_std = bool(ic_std) and math.isfinite(ic_std) and ic_std != 0
    t_nw = newey_west_tstat(ic_arr, session_arr, lag)
    return {
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "icir": ic_mean / ic_std if finite_std else float("nan"),
        "t_naive": ic_mean / (ic_std / math.sqrt(n_cohorts)) if finite_std else float("nan"),
        "t_nw": t_nw,
        "p_nw": two_sided_normal_p(t_nw),
        "n_hac_pairs_min": n_hac_pairs(session_arr, lag) if lag > 0 else 0,
    }


def scan_event_cohort_cell(
    con: duckdb.DuckDBPyConnection,
    *,
    event_view: str = "fin_sue_event",
    calendar_view: str = "daily_ohlcv",
    sue_col: str = "fin_sue",
    h_start: int,
    h_end: int,
    sample_start: str,
    min_events_per_market_contribution: int,
    min_events_per_cohort_total: int,
    min_event_cohorts: int,
    expected_sign: str | None = None,
) -> dict[str, Any]:
    """One SUE event-bucket cell's cohort rank-IC/NW (§5.4).

    ``expected_sign`` is accepted for interface parity with ``scan_cell`` but
    unused here — §5.3's spread-sign-alignment step never applies to event
    cells (no spread is computed for them, ``q5_spread_*`` stay ``None``).
    """
    del expected_sign
    lag = choose_nw_lag(scan_type="bucket", bucket_width=h_end - h_start)
    min_cohorts_required = max(min_event_cohorts, lag + 2)
    result = _empty_event_cohort_result(h_start, h_end)

    frame = execute_event_cohort_frame(
        con,
        event_view=event_view,
        calendar_view=calendar_view,
        sue_col=sue_col,
        h_start=h_start,
        h_end=h_end,
        sample_start=sample_start,
    )
    if not frame.is_empty():
        frame = frame.filter(pl.col("sue_value").is_finite() & pl.col("excess_value").is_finite())
    if frame.is_empty():
        result["status_reason"] = "no_formation_rows"
        return result
    result["effective_sample_start"] = frame["event_formation_date"].min()
    result["effective_sample_end"] = frame["event_formation_date"].max()

    pooled = _pool_cohort_ranks(
        frame,
        min_events_per_market_contribution=min_events_per_market_contribution,
        min_events_per_cohort_total=min_events_per_cohort_total,
    )
    market_n_stats = pooled["market_n_stats"]
    if market_n_stats is None:
        result["status_reason"] = "no_market_meets_min_contribution"
        return result
    n_obs_series = market_n_stats["market_n"].to_numpy()
    result["n_obs"] = int(n_obs_series.sum())
    result["n_obs_mean"] = float(n_obs_series.mean())
    result["n_obs_min"] = int(n_obs_series.min())
    result["n_obs_median"] = float(np.median(n_obs_series))
    result.update(
        market_weight_means(
            market_n_stats.rename({"market_n": "n"}), date_col="event_formation_date"
        )
    )

    cohort_rows = pooled["cohort_rows"]
    if not cohort_rows:
        result["status_reason"] = "no_valid_cohort"
        return result

    qualifying_dates = [r[0] for r in cohort_rows]
    windows = (
        pooled["qualifying"]
        .filter(pl.col("event_formation_date").is_in(qualifying_dates))
        .select(["bsns_year", "reprt_code"])
        .unique()
    )
    result["n_independent_filing_windows"] = windows.height

    cohort_rows.sort(key=lambda r: r[0])
    n_cohorts = len(cohort_rows)
    result["n_dates"] = n_cohorts
    if n_cohorts < min_cohorts_required:
        result["status_reason"] = f"insufficient_cohorts:{n_cohorts}<{min_cohorts_required}"
        return result

    result.update(_aggregate_cohort_rows(cohort_rows, lag=lag))
    result["status"] = "valid"
    return result


def run_phase_b_event_scan(
    con: duckdb.DuckDBPyConnection,
    ready_event_cells: list[dict[str, Any]],
    *,
    event_view: str = "fin_sue_event",
    calendar_view: str = "daily_ohlcv",
    sample_start: str,
    min_events_per_market_contribution: int,
    min_events_per_cohort_total: int,
    min_event_cohorts: int,
) -> list[dict[str, Any]]:
    """Scan every ready SUE event-bucket cell (§5.4).

    Mirrors ``run_phase_b_continuous_scan``'s "one row per registry cell"
    shape (each input cell's fields ride along into its output row) so both
    can feed ``assemble_phase_b_primary_table``.
    """
    rows = []
    for cell in ready_event_cells:
        stats = scan_event_cohort_cell(
            con,
            event_view=event_view,
            calendar_view=calendar_view,
            sue_col=cell["feature"],
            h_start=cell["h_start"],
            h_end=cell["h_end"],
            sample_start=sample_start,
            min_events_per_market_contribution=min_events_per_market_contribution,
            min_events_per_cohort_total=min_events_per_cohort_total,
            min_event_cohorts=min_event_cohorts,
            expected_sign=cell.get("expected_sign"),
        )
        rows.append({**cell, "scan_type": "event_bucket", **stats})
    return rows


# --- assembly: one common schema across continuous + event cells (§6 B-7) ---


def assemble_phase_b_primary_table(
    readiness_rows: list[dict[str, Any]],
    scanned_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Project the (up to 38) readiness-frozen candidate cells onto one
    common statistics schema.

    ``blocked_exploratory`` cells get a ``not_evaluated`` placeholder with no
    stats computed; ``ready_primary`` cells get their scanned row — the sole
    cohort-scan row for the event family, or the ``(broad, common_survivor)``
    discovery-coordinate row for continuous families (the other 3 universe/
    sample_kind combos scanned per continuous cell are robustness inputs for
    B-8/B-9, not part of this primary table).

    Raises ``ValueError`` if a ``ready_primary`` cell in ``readiness_rows``
    has no matching scanned row (a scan/readiness mismatch bug), mirroring
    ``horizon_scan_runner.assert_rows_match_registry``'s "never silently
    drop a registered hypothesis" rule.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for row in scanned_rows:
        is_discovery_combo = (
            row.get("universe") == _DISCOVERY_UNIVERSE
            and row.get("sample_kind") == _DISCOVERY_SAMPLE_KIND
        )
        if row.get("scan_type") == "event_bucket" or is_discovery_combo:
            by_id[row["hypothesis_id"]] = row

    table: list[dict[str, Any]] = []
    for cell in readiness_rows:
        if cell["role"] == "ready_primary":
            scanned = by_id.get(cell["hypothesis_id"])
            if scanned is None:
                raise ValueError(f"ready_primary cell {cell['hypothesis_id']} was never scanned")
            table.append(scanned)
        else:
            table.append(
                {**cell, "status": "not_evaluated", "status_reason": "blocked_exploratory"}
            )
    return table


def phase_b_primary_stats_rows(assembled_table: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The subset of ``assemble_phase_b_primary_table``'s output that actually
    carries computed statistics — exactly ``M_B_ready`` rows (§6 B-7's
    completion criterion)."""
    return [row for row in assembled_table if row.get("role") == "ready_primary"]


def assert_scan_matches_ready_population(
    scanned_ready_rows: list[dict[str, Any]], readiness_rows: list[dict[str, Any]]
) -> None:
    """Every ``ready_primary`` readiness cell was scanned exactly once, no
    extras — reuses ``horizon_scan_runner.assert_rows_match_registry``,
    which is generic over any ``hypothesis_id``-keyed population."""
    ready_cells = [cell for cell in readiness_rows if cell["role"] == "ready_primary"]
    assert_rows_match_registry(scanned_ready_rows, ready_cells)


# --- B-9 steps 1-2: Phase-B-only and combined A+B BH ---


def apply_phase_b_only_bh(
    phase_b_ready_rows: list[dict[str, Any]], *, q_threshold: float = 0.10
) -> list[dict[str, Any]]:
    """One BH pass across only Phase B's own readiness-frozen ready_primary
    population (``m = M_B_ready``), independent of Phase A — the
    ``q_fdr_phase_b`` card field. Reuses ``apply_global_bh`` verbatim.
    """
    augmented = apply_global_bh(phase_b_ready_rows, q_threshold=q_threshold)
    renamed = []
    for row in augmented:
        row = dict(row)
        row["q_fdr_phase_b"] = row.pop("q_fdr_global")
        row["bh_pass_phase_b"] = row.pop("bh_pass")
        row["primary_discovery_phase_b"] = row.pop("primary_discovery")
        renamed.append(row)
    return renamed


def apply_combined_ab_bh(
    phase_a_ready_rows: list[dict[str, Any]],
    phase_b_ready_rows: list[dict[str, Any]],
    *,
    q_threshold: float = 0.10,
) -> list[dict[str, Any]]:
    """§6 B-9 steps 1-2: one BH pass across Phase A's fixed 75-hypothesis
    population plus Phase B's readiness-frozen ready_primary population
    (``m = 75 + M_B_ready``) — the ``q_fdr_global_ab`` card field.

    Reuses ``apply_global_bh`` unchanged: it is generic over any fixed
    hypothesis population keyed by ``hypothesis_id``, and
    ``compute_isolated_spikes`` groups by ``(family, scan_type)`` internally,
    so Phase A and Phase B families never cross-contaminate each other's
    neighbor checks even though both flow through one call. Output fields are
    renamed so this combined pass never overwrites either phase's own
    single-population BH result under the same key.
    """
    combined_rows = [*phase_a_ready_rows, *phase_b_ready_rows]
    assert_unique_hypothesis_ids(combined_rows)
    combined = apply_global_bh(combined_rows, q_threshold=q_threshold)
    renamed = []
    for row in combined:
        row = dict(row)
        row["q_fdr_global_ab"] = row.pop("q_fdr_global")
        row["bh_pass_ab"] = row.pop("bh_pass")
        row["primary_discovery_ab"] = row.pop("primary_discovery")
        renamed.append(row)
    return renamed


# --- screen_pass (§9 B-9) ---


def compute_phase_b_period_sign_pass(
    period_ics: list[float | None], *, expected_sign: str | None
) -> dict[str, Any]:
    """§9 B-9 screen_pass rule 4: family별 유효 common period 개수에 따라 다른
    문턱을 쓴다 — 3개 이상이면 strict majority, 정확히 2개면 둘 다 같은 방향,
    1개 이하는 무조건 실패.

    Reuses ``horizon_scan_runner.compute_period_sign_pass``'s majority
    arithmetic (``consistent > len(valid) / 2`` already means "strict
    majority" for n>=3 and "both agree" for n==2) but overrides it for n<=1,
    where that same formula would let a single sign-consistent period pass —
    Phase A's own gate tolerates that (§A-4), Phase B's §9 rule 4 explicitly
    does not ("1개 이하는 실패").
    """
    valid = [ic for ic in period_ics if ic is not None and math.isfinite(ic)]
    if len(valid) <= 1:
        return {
            "valid_subperiods": len(valid),
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


def compute_phase_b_screen_pass(
    *,
    role: str,
    primary_discovery: bool,
    isolated_spike: bool,
    tradable_pass: bool,
    period_sign_pass: bool,
    available_direction_pass: bool | None,
    robustness_required: bool,
    robustness_pass: bool | None,
    ca_holdout_policy_pass: bool = True,
) -> dict[str, Any]:
    """§9 B-9's 9-rule Phase B screen_pass, one row per ready_primary cell.

    Only ``role == "ready_primary"`` cells are evaluated at all (rule 6 —
    every other cell in this pipeline is already ``blocked_exploratory`` by
    construction, so this is a structural gate, not a computed one).
    ``primary_discovery`` is expected to be ``apply_combined_ab_bh``'s
    ``primary_discovery_ab`` field, which already folds rule 1
    (``q_fdr_global_ab < threshold``) and the expected-sign check together
    (mirrors ``horizon_scan_report.compute_screen_pass``'s own
    ``primary_discovery`` parameter) — ``isolated_spike`` is passed
    separately anyway, purely so a spike-caused failure shows up as its own
    named gate in ``failed_gates`` rather than being folded into an opaque
    "primary_discovery" failure.

    ``robustness_required``/``robustness_pass`` generalize rules 7 (long
    continuous: temporal placebo + non-overlap offset) and 8 (SUE: issuer +
    filing-cycle cluster confirmation) into one slot — a cell neither rule
    applies to (a short/mid continuous cell) passes this gate vacuously
    (``robustness_required=False``). ``ca_holdout_policy_pass`` (rule 9)
    defaults to ``True``: neither ``run_phase_b_core`` nor ``run_combined_ab``
    expose a debug/holdout override today, so the policy trivially matches
    the official contract until such an override exists.
    """
    if role != "ready_primary":
        return {"screen_pass": False, "not_applicable_role": True, "failed_gates": []}

    checks: dict[str, bool] = {
        "primary_discovery": primary_discovery,
        "isolated_spike_clear": not isolated_spike,
        "tradable_pass": tradable_pass,
        "period_sign_pass": period_sign_pass,
        "ca_holdout_policy_pass": ca_holdout_policy_pass,
    }
    if available_direction_pass is not None:
        checks["available_direction_pass"] = available_direction_pass
    if robustness_required:
        checks["robustness_pass"] = bool(robustness_pass)

    failed_gates = [name for name, ok in checks.items() if not ok]
    return {
        "screen_pass": not failed_gates,
        "not_applicable_role": False,
        "failed_gates": failed_gates,
    }


# --- evidence grade (§9 B-9) ---

# value/profitability/accrual families read PIT financials without any PIT
# industry classification (not yet available in this codebase's raw layer) —
# §9 caps only these 3 families at grade B regardless of how clean their
# statistics look, since the missing industry context is a genuine
# interpretive limitation, not something a strong screen_pass can outweigh.
# Size/asset-growth/net-issuance/payout/SUE have no such structural cap.
PIT_INDUSTRY_CAPPED_FAMILIES = frozenset(
    {"fin_value_z", "fin_gross_profitability", "fin_accruals_to_assets"}
)


def compute_phase_b_evidence_grade(
    *,
    role: str,
    family: str,
    screen_pass: bool,
    failed_gates: list[str],
    valid_subperiods: int,
    all_offsets_evaluable: bool,
    n_independent_filing_windows: int | None = None,
    grade_a_min_independent_filing_windows: int = 20,
    source_quality_status: str | None = None,
) -> str:
    """§9 B-9's A/B/C/D/NE evidence grade, mirroring
    ``horizon_scan_report.assign_evidence_grade``'s structure (role gate
    first, then the specific C-routing conditions, then the screen_pass
    split) with Phase B's own grade-cap rules.

    ``failed_gates`` is expected to be ``compute_phase_b_screen_pass``'s own
    output for this cell — no new failure-reason computation is needed since
    that function already names which check failed. An available-sample sign
    flip or a failed robustness confirmation (rules 5/7-8) route straight to
    ``"C"`` even though both are also ``screen_pass`` failures — Phase A's
    ``assign_evidence_grade`` checks its own ``available_sign_flip`` before
    the pass/fail split for the same reason (a diagnosed reversal or a failed
    null-experiment confirmation is a different kind of failure than "no
    signal at all", and downstream reporting treats it differently).

    ``n_independent_filing_windows`` (§6 B-8 SUE point 5: "코호트가 부족하면
    grade A 금지") is ``None`` for continuous cells — the check is skipped,
    not treated as insufficient — and B-7's core scan
    (``scan_event_cohort_cell``) already computes this count for every SUE
    cell, so no new statistic is needed here, only the comparison against
    the preregistered ``grade_a_min_independent_filing_windows`` threshold
    (config ``phase_b.grade_a_min_independent_filing_windows``, default 20).

    ``source_quality_status`` is §9's "source 비치명 경고", computed by
    ``horizon_scan_phase_b_source_quality`` from B-10 Stage 2's vintage-side
    diagnostics. Only ``"ok"``/``"not_applicable"`` allow grade A; a warning,
    an unmeasurable ratio, and ``None`` (no diagnostic available at all) each
    cap at B. It is deliberately non-fatal — it can never push a cell below
    the grade its statistics earned.

    Segment diagnostics, the plan's third "B" trigger, still have no computed
    values anywhere (§5.5 is out of scope). Wire that one in here once it
    exists rather than fabricating a warning signal that doesn't exist yet.
    """
    if role != "ready_primary":
        return "NE"
    if "available_direction_pass" in failed_gates:
        return "C"
    if "robustness_pass" in failed_gates:
        return "C"
    if (
        screen_pass
        and all_offsets_evaluable
        and family not in PIT_INDUSTRY_CAPPED_FAMILIES
        and valid_subperiods != 2
        and source_quality_allows_grade_a(source_quality_status)
        and (
            n_independent_filing_windows is None
            or n_independent_filing_windows >= grade_a_min_independent_filing_windows
        )
    ):
        return "A"
    if screen_pass:
        return "B"
    return "D"
