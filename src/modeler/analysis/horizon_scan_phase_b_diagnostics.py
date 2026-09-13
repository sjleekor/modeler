"""Phase B non-blocking diagnostics: cross-phase primary-feature rank
correlation (§5.5, §7.1 ``primary_feature_rank_correlation.parquet``) and the
SUE event-formation-ordinal non-overlap check (§6 B-8 SUE point 5).

Neither function here gates ``screen_pass`` — §9 B-9 lists both as reporting/
grade-cap inputs, not hard screen conditions. Both are built entirely from
already-tested primitives elsewhere in this package; see each function's
docstring for what it reuses.
"""

from __future__ import annotations

import math
from typing import Any

import duckdb
import numpy as np
import polars as pl

from modeler.analysis.horizon_scan_phase_b_robustness import compute_nonoverlap_robustness_pass
from modeler.analysis.horizon_scan_phase_b_scan import (
    _DISCOVERY_SAMPLE_KIND,
    _DISCOVERY_UNIVERSE,
    _aggregate_cohort_rows,
    _pool_cohort_ranks,
    execute_event_cohort_frame,
)
from modeler.analysis.horizon_scan_runner import scan_cell
from modeler.etl.metrics import choose_nw_lag, daily_market_weighted_ic, per_date_market_rank_ic

# --- rank correlation (§5.5) ---


def build_rank_correlation_sql(
    *, panel_view: str, feature_a: str, feature_b: str, sample_start: str
) -> str:
    """Rows where both features are simultaneously formation-eligible, at the
    same (broad, common_survivor) discovery coordinate every other Phase B
    primary statistic uses — this is a contemporaneous feature-vs-feature
    correlation, not a feature-vs-future-label scan, so
    ``horizon_scan_runner.build_formation_sql`` (which always joins in a
    horizon-shifted label) does not apply; the eligibility conditions here
    are copied from it verbatim (``in_broad``, ``NOT ca_mask``,
    ``common_formation_120d``/``common_survivor_120d``, both features
    non-null/finite).
    """
    return f"""
        SELECT trade_date, market,
               {feature_a} AS feature_a_value,
               {feature_b} AS feature_b_value
        FROM {panel_view}
        WHERE trade_date >= DATE '{sample_start}'
          AND in_broad
          AND {feature_a} IS NOT NULL AND isfinite({feature_a})
          AND {feature_b} IS NOT NULL AND isfinite({feature_b})
          AND NOT ca_mask
          AND common_formation_120d
          AND common_survivor_120d
    """


def compute_phase_b_rank_correlation(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str,
    feature_pairs: list[tuple[str, str, str, str]],
    sample_start: str,
    min_names: int,
    scan_engine: str = "legacy",
) -> list[dict[str, Any]]:
    """One row per ``(family_a, feature_a, family_b, feature_b)`` pair: a
    market-weighted daily Spearman rank-correlation series summarized into a
    mean/std/min/max — "행렬과 분포" (§5.5). Reuses
    ``per_date_market_rank_ic``/``daily_market_weighted_ic``
    (``research/etl/metrics.py``) unchanged — both are already generic over
    any two numeric columns, not specific to feature-vs-label.

    ``feature_pairs`` is caller-supplied (typically every ready Phase A
    continuous family's primary feature crossed with every *ready* Phase B
    continuous family's primary feature — SUE is excluded, its grain isn't
    daily) so this function stays agnostic to readiness/family bookkeeping.
    """
    rows: list[dict[str, Any]] = []
    for family_a, feature_a, family_b, feature_b in feature_pairs:
        sql = build_rank_correlation_sql(
            panel_view=panel_view,
            feature_a=feature_a,
            feature_b=feature_b,
            sample_start=sample_start,
        )
        frame = con.execute(sql).pl()
        market_ic = per_date_market_rank_ic(
            frame,
            pred_col="feature_a_value",
            realized_col="feature_b_value",
            min_names=min_names,
            engine=scan_engine,
        )
        market_ic = market_ic.filter(pl.col("rank_ic").is_finite())
        daily = daily_market_weighted_ic(market_ic)
        base = {
            "family_a": family_a,
            "feature_a": feature_a,
            "family_b": family_b,
            "feature_b": feature_b,
        }
        if daily.is_empty():
            rows.append(
                {
                    **base,
                    "n_dates": 0,
                    "mean_rank_corr": None,
                    "std_rank_corr": None,
                    "min_rank_corr": None,
                    "max_rank_corr": None,
                }
            )
            continue
        corr = daily["rank_ic"].to_numpy()
        rows.append(
            {
                **base,
                "n_dates": daily.height,
                "mean_rank_corr": float(np.mean(corr)),
                "std_rank_corr": float(np.std(corr, ddof=1)) if daily.height > 1 else float("nan"),
                "min_rank_corr": float(np.min(corr)),
                "max_rank_corr": float(np.max(corr)),
            }
        )
    return rows


# --- SUE event-formation-ordinal non-overlap (§6 B-8 SUE point 5) ---


def run_sue_event_ordinal_nonoverlap(
    con: duckdb.DuckDBPyConnection,
    ready_event_cells: list[dict[str, Any]],
    *,
    sample_start: str,
    min_events_per_market_contribution: int,
    min_events_per_cohort_total: int,
    min_event_cohorts: int,
    ordinal_stride: int,
    valid_offset_ratio_min: float = 0.80,
    expected_sign_ratio_min: float = 0.60,
    event_view: str = "fin_sue_event",
    calendar_view: str = "daily_ohlcv",
) -> list[dict[str, Any]]:
    """§6 B-8 SUE point 5: split the real pooled cohort dates into
    ``ordinal_stride`` interleaved subsamples by formation-date rank (cohort
    ordinal position mod ``ordinal_stride``) — the event analog of
    continuous's ``run_nonoverlap_offsets`` stride trick, which instead
    splits *trading days* by the feature's own horizon width (no such
    natural width exists for an event cohort).

    Reuses the real statistic's own building blocks unchanged:
    ``_pool_cohort_ranks``/``_aggregate_cohort_rows``
    (``horizon_scan_phase_b_scan.py``, already shared with the issuer/
    filing-cycle bootstraps) for the rank-and-pool + IC/NW math, and
    ``compute_nonoverlap_robustness_pass`` (``horizon_scan_phase_b_robustness
    .py``) for the same valid-offset-ratio/expected-sign-ratio gate
    continuous long cells use. Non-blocking: this never fails
    ``screen_pass`` — a cell with too few surviving subsamples only caps
    ``evidence_grade`` (via ``n_independent_filing_windows``, computed
    separately in B-7's core scan), it does not disqualify the cell outright.
    """
    rows: list[dict[str, Any]] = []
    for cell in ready_event_cells:
        h_start, h_end = cell["h_start"], cell["h_end"]
        lag = choose_nw_lag(scan_type="bucket", bucket_width=h_end - h_start)
        frame = execute_event_cohort_frame(
            con,
            event_view=event_view,
            calendar_view=calendar_view,
            sue_col=cell["feature"],
            h_start=h_start,
            h_end=h_end,
            sample_start=sample_start,
        )
        if not frame.is_empty():
            frame = frame.filter(
                pl.col("sue_value").is_finite() & pl.col("excess_value").is_finite()
            )
        pooled = _pool_cohort_ranks(
            frame,
            min_events_per_market_contribution=min_events_per_market_contribution,
            min_events_per_cohort_total=min_events_per_cohort_total,
        )
        cohort_rows = sorted(pooled["cohort_rows"], key=lambda r: r[0])

        subsample_ic_means: list[float] = []
        n_valid = 0
        for offset in range(ordinal_stride):
            subsample = [r for i, r in enumerate(cohort_rows) if i % ordinal_stride == offset]
            if len(subsample) < min_event_cohorts:
                continue
            agg = _aggregate_cohort_rows(subsample, lag=lag)
            ic_mean = agg["ic_mean"]
            if ic_mean is not None and math.isfinite(ic_mean):
                n_valid += 1
                subsample_ic_means.append(ic_mean)

        sign = -1.0 if cell.get("expected_sign") == "-" else 1.0
        sign_ratio = (
            float(np.mean([sign * v > 0 for v in subsample_ic_means]))
            if subsample_ic_means
            else None
        )
        summary = {
            "n_offsets_total": ordinal_stride,
            "n_offsets_valid": n_valid,
            "offset_sign_agreement_ratio": sign_ratio,
            "offset_status": "complete" if n_valid == ordinal_stride else "some_insufficient",
        }
        gate = compute_nonoverlap_robustness_pass(
            summary,
            valid_offset_ratio_min=valid_offset_ratio_min,
            expected_sign_ratio_min=expected_sign_ratio_min,
        )
        rows.append({"hypothesis_id": cell["hypothesis_id"], **summary, **gate})
    return rows


# --- secondary-feature diagnostic scan (02_stage1a §4) ---


def build_secondary_diagnostic_cells(
    families: list[dict[str, Any]],
    *,
    available_families: set[str] | None = None,
) -> list[dict[str, Any]]:
    """One diagnostic cell per (family, secondary column, primary horizon).

    The registry the real scan runs only ever carries a family's *primary*
    feature (``horizon_scan_readiness._hypothesis_rows`` picks
    ``official_feature_variant`` of the primary column), and the exploratory
    horizon set is not scanned by the CLI at all. So the question
    ``02_stage1a`` §4 asks of the macro families — how much of the beta's IC
    survives once the market factor is taken out, i.e. ``macro_beta_*`` versus
    ``macro_rawbeta_*`` — has no cell anywhere in the run to read. This builds
    those cells.

    Cumulative cells only, at the family's own preregistered primary horizons.
    A bucket cell would double the count without answering a different
    question, and the comparison is against the primary cumulative cell.
    """
    cells: list[dict[str, Any]] = []
    for family in families:
        name = family["family"]
        if available_families is not None and name not in available_families:
            continue
        secondary = [f["column"] for f in family["features"] if f["role"] == "secondary"]
        if not secondary:
            continue
        primary = next(f["column"] for f in family["features"] if f["role"] == "primary")
        for column in secondary:
            for horizon in family.get("primary_horizon_set", []):
                cells.append(
                    {
                        "family": name,
                        "fdr_family": family.get("fdr_family"),
                        "primary_feature": primary,
                        "feature": column,
                        "feature_role": "secondary",
                        "scan_type": "cum",
                        "h_start": 0,
                        "h_end": int(horizon),
                        "expected_sign": family.get("expected_sign"),
                    }
                )
    return cells


def run_secondary_feature_diagnostics(
    con: duckdb.DuckDBPyConnection,
    cells: list[dict[str, Any]],
    *,
    panel_view: str,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    scan_engine: str = "legacy",
    primary_ic_by_cell: dict[tuple[str, int], float | None] | None = None,
) -> list[dict[str, Any]]:
    """Scan each diagnostic cell at the discovery coordinate. Never a discovery.

    Deliberately outside every BH population and every gate: these rows are
    card material (§9's reporting inputs), and adding them to ``m`` would
    inflate the correction with hypotheses nobody preregistered. They also
    take no ``daily_sink`` — ``daily_ic.parquet`` holds the registered scan's
    series, not diagnostics (01_stage0 §3.1).

    ``primary_ic_by_cell`` maps ``(family, h_end)`` to the primary cell's own
    IC so each row carries the difference the diagnostic exists to show. A cell
    whose feature column is absent from the panel is reported as
    ``column_absent`` rather than skipped — a missing secondary column is a
    fact about the mart worth seeing on the card.
    """
    columns = {row[0] for row in con.execute(f"DESCRIBE {panel_view}").fetchall()}
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if cell["feature"] not in columns:
            rows.append({**cell, "status": "not_evaluated", "status_reason": "column_absent"})
            continue
        stats = scan_cell(
            con,
            panel_view=panel_view,
            feature_col=cell["feature"],
            scan_type=cell["scan_type"],
            h_start=cell["h_start"],
            h_end=cell["h_end"],
            universe=_DISCOVERY_UNIVERSE,
            sample_kind=_DISCOVERY_SAMPLE_KIND,
            sample_start=sample_start,
            min_names=min_names,
            min_names_for_spread=min_names_for_spread,
            quantile_count=quantile_count,
            min_dates_per_cell=min_dates_per_cell,
            expected_sign=cell.get("expected_sign"),
            scan_engine=scan_engine,
        )
        primary_ic = (primary_ic_by_cell or {}).get((cell["family"], cell["h_end"]))
        secondary_ic = stats["ic_mean"] if stats["status"] == "valid" else None
        rows.append(
            {
                **cell,
                **stats,
                "universe": _DISCOVERY_UNIVERSE,
                "sample_kind": _DISCOVERY_SAMPLE_KIND,
                "primary_ic_mean": primary_ic,
                "ic_mean_minus_primary": (
                    None
                    if primary_ic is None or secondary_ic is None
                    else secondary_ic - primary_ic
                ),
            }
        )
    return rows
