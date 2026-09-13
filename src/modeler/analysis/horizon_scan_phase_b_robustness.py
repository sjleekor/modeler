"""Phase B B-8 robustness/null experiments (04_specific_plan_B.md §6 B-8).

Continuous families reuse Phase A's own A-5/A-6b machinery verbatim against
the Phase B panel — non-overlap offset subsampling (``run_nonoverlap_offsets``)
and the circular date-shift temporal placebo (``run_temporal_placebo``) are
both already generic over ``panel_view``/a caller-supplied registry, so no
Phase-B-specific reimplementation is needed for either.

The SUE event family gets two new cluster bootstraps that have no Phase A
analog (continuous families have no repeated-issuer/repeated-filing-window
structure to resample): an issuer-cluster bootstrap (resample tickers with
replacement, keeping each drawn ticker's *entire* primary-sample event
history together) and a ``(bsns_year, reprt_code)`` filing-cycle block
bootstrap (resample whole reporting-quarter cohorts together) — §6 B-8 SUE
points 2-3. Both reuse ``horizon_scan_phase_b_scan._pool_cohort_ranks`` so the
resampled statistic is computed by the exact same rank-and-pool logic as the
real one, and both derive replicate seeds via
``horizon_scan_permutation.derive_replicate_seed`` for the same
kind/index/config_hash reproducibility guarantee every other null experiment
in this repo already has.

Scope for this PR (see docs/dev/.../04_specific_plan_B.md §10's B-PR8/B-PR9
split): the joint A+B continuous + SUE cross-sectional permutation (§6 B-8's
"결합 단면 permutation") and the event-formation-ordinal non-overlap
diagnostic are deliberately deferred — both are non-blocking diagnostics in
§6 B-9's screen_pass rules (only the two cluster bootstraps and the temporal
placebo/non-overlap-offset gates are hard screen_pass inputs, per rules 7-8),
so this PR ships the gates that actually decide screen_pass first.
"""

from __future__ import annotations

import math
from typing import Any

import duckdb
import numpy as np
import polars as pl

from modeler.analysis.horizon_scan_permutation import (
    _append_checkpoint,
    _load_checkpoint,
    derive_replicate_seed,
    run_temporal_placebo,
    select_long_horizon_hypotheses,
)
from modeler.analysis.horizon_scan_phase_b_scan import (
    _pool_cohort_ranks,
    execute_event_cohort_frame,
)
from modeler.analysis.horizon_scan_runner import run_nonoverlap_offsets

# --- continuous: non-overlap offsets (§6 B-8 continuous point 2) ---

_DEFAULT_NONOVERLAP_MIN_DATES = {"default": 20, "cumulative_120": 12, "bucket_60_120": 12}


def _nonoverlap_min_dates_for_cell(
    cell: dict[str, Any], scan_type: str, overrides: dict[str, int]
) -> int:
    """§2.4's ``phase_b.nonoverlap_min_dates`` override: the two h=120 cells
    (cumulative 120d, bucket (60,120]) get a lower floor than the Phase A
    default — a stride-120/stride-60 offset split leaves too few dates per
    offset for the default-20 floor to ever pass."""
    if scan_type == "cum" and cell["h_end"] == 120:
        return overrides.get("cumulative_120", 12)
    if scan_type == "bucket" and cell["h_start"] == 60 and cell["h_end"] == 120:
        return overrides.get("bucket_60_120", 12)
    return overrides.get("default", 20)


def compute_nonoverlap_robustness_pass(
    summary: dict[str, Any],
    *,
    valid_offset_ratio_min: float = 0.80,
    expected_sign_ratio_min: float = 0.60,
) -> dict[str, Any]:
    """§6 B-8 continuous point 2's robustness rule: ``valid_offset_ratio``
    (not raw offset *count*) must clear 0.80, and — among only the valid
    offsets — the expected-sign agreement ratio must clear 0.60. Neither rule
    requires every offset to be evaluable (``run_nonoverlap_offsets`` already
    tolerates per-offset insufficiency)."""
    n_total = summary.get("n_offsets_total") or 0
    valid_offset_ratio = (summary.get("n_offsets_valid") or 0) / n_total if n_total else 0.0
    sign_ratio = summary.get("offset_sign_agreement_ratio")
    passed = (
        valid_offset_ratio >= valid_offset_ratio_min
        and sign_ratio is not None
        and sign_ratio >= expected_sign_ratio_min
    )
    return {"valid_offset_ratio": valid_offset_ratio, "nonoverlap_robustness_pass": bool(passed)}


def run_phase_b_continuous_nonoverlap(
    con: duckdb.DuckDBPyConnection,
    ready_continuous_cells: list[dict[str, Any]],
    *,
    panel_view: str,
    sample_start: str,
    min_names: int,
    scan_engine: str = "legacy",
    nonoverlap_min_dates_overrides: dict[str, int] | None = None,
    valid_offset_ratio_min: float = 0.80,
    expected_sign_ratio_min: float = 0.60,
) -> list[dict[str, Any]]:
    """Non-overlap offset diagnostic for every ready continuous cell,
    evaluated at the (broad, common_survivor) discovery coordinate — the same
    coordinate ``assemble_phase_b_primary_table`` picks as each cell's
    primary statistic.
    """
    overrides = nonoverlap_min_dates_overrides or _DEFAULT_NONOVERLAP_MIN_DATES
    rows: list[dict[str, Any]] = []
    for cell in ready_continuous_cells:
        scan_type = "cum" if cell["cell_type"] == "cumulative" else "bucket"
        min_dates = _nonoverlap_min_dates_for_cell(cell, scan_type, overrides)
        alignment_sign = -1.0 if cell.get("expected_sign") == "-" else 1.0
        summary = run_nonoverlap_offsets(
            con,
            panel_view=panel_view,
            feature_col=cell["feature"],
            scan_type=scan_type,
            h_start=cell["h_start"],
            h_end=cell["h_end"],
            universe="broad",
            sample_kind="common_survivor",
            sample_start=sample_start,
            min_names=min_names,
            nonoverlap_min_dates=min_dates,
            alignment_sign=alignment_sign,
            scan_engine=scan_engine,
        )
        gate = compute_nonoverlap_robustness_pass(
            summary,
            valid_offset_ratio_min=valid_offset_ratio_min,
            expected_sign_ratio_min=expected_sign_ratio_min,
        )
        extra = {"scan_type": scan_type, "nonoverlap_min_dates": min_dates}
        rows.append({**cell, **extra, **summary, **gate})
    return rows


# --- continuous: temporal placebo reuse (§6 B-8 continuous point 4) ---


def select_phase_b_long_horizon_cells(
    ready_continuous_cells: list[dict[str, Any]], *, min_nw_lag: int = 59
) -> list[dict[str, Any]]:
    """The ``nw_lag>=59`` ready continuous cells — thin ``scan_type``-adding
    wrapper around ``horizon_scan_permutation.select_long_horizon_hypotheses``,
    which is otherwise reused unmodified."""
    with_scan_type = [
        {**cell, "scan_type": "cum" if cell["cell_type"] == "cumulative" else "bucket"}
        for cell in ready_continuous_cells
    ]
    return select_long_horizon_hypotheses(with_scan_type, min_nw_lag=min_nw_lag)


def run_phase_b_temporal_placebo(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str,
    long_horizon_cells: list[dict[str, Any]],
    real_t_nw_by_id: dict[str, float | None],
    config_hash: str,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    scan_engine: str = "legacy",
    n_replicates: int = 100,
    min_shift_sessions: int = 120,
    p_max: float = 0.10,
    checkpoint_path=None,
    checkpoint_fingerprint: dict[str, Any] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """§6 B-8 continuous point 4 — pure pass-through to
    ``horizon_scan_permutation.run_temporal_placebo`` against the Phase B
    panel; that function is already panel/registry-agnostic."""
    return run_temporal_placebo(
        con,
        panel_view=panel_view,
        long_horizon_registry=long_horizon_cells,
        real_t_nw_by_id=real_t_nw_by_id,
        config_hash=config_hash,
        sample_start=sample_start,
        min_names=min_names,
        min_names_for_spread=min_names_for_spread,
        quantile_count=quantile_count,
        min_dates_per_cell=min_dates_per_cell,
        scan_engine=scan_engine,
        n_replicates=n_replicates,
        min_shift_sessions=min_shift_sessions,
        p_max=p_max,
        checkpoint_path=checkpoint_path,
        checkpoint_fingerprint=checkpoint_fingerprint,
        workers=workers,
    )


# --- SUE: issuer-cluster / filing-cycle block bootstrap (§6 B-8 SUE 2-3) ---


def _bootstrap_two_sided_p(values: list[float]) -> float:
    """Percentile-method two-sided bootstrap p-value against zero: twice the
    smaller tail fraction, clipped to 1.0. Standard nonparametric convention
    for "does this bootstrap distribution of the estimator cross zero" —
    distinct from ``horizon_scan_permutation``'s null-distribution p-values
    (there is no null distribution here; the bootstrap is centered on the
    real estimate, not on zero).
    """
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    p_le = float((arr <= 0).mean())
    p_ge = float((arr >= 0).mean())
    return float(min(1.0, 2 * min(p_le, p_ge)))


def run_cluster_bootstrap(
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
    cluster_cols: list[str],
    cluster_kind: str,
    hypothesis_id: str,
    config_hash: str,
    n_replicates: int,
    expected_sign: str | None = None,
    p_max: float = 0.10,
    checkpoint_path=None,
    checkpoint_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One cluster-resample-with-replacement bootstrap of a SUE event-bucket
    cell's pooled cohort ``ic_mean`` (§5.4's statistic, §6 B-8 SUE points
    2-3). ``cluster_cols`` is ``["ticker"]`` for the issuer bootstrap or
    ``["bsns_year", "reprt_code"]`` for the filing-cycle block bootstrap —
    every row belonging to a drawn cluster travels together (a repeated draw
    duplicates that cluster's full event history, the defining property of a
    cluster, as opposed to plain-event, bootstrap).

    Reuses ``horizon_scan_phase_b_scan._pool_cohort_ranks`` per replicate —
    same rank-and-pool logic the real (non-bootstrapped) statistic uses, just
    with no minimum-cohort-count gate (a resampled draw legitimately has
    fewer qualifying cohort dates sometimes; that's the point of resampling).
    """
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
        return {
            "n_clusters": 0,
            "n_valid_replicates": 0,
            "replicate_ic_means": [],
            "bootstrap_mean": None,
            "bootstrap_p": None,
            "cluster_confirm_pass": False,
            "status_reason": "no_formation_rows",
        }

    keys_df = frame.select(cluster_cols).unique().sort(cluster_cols)
    key_tuples = [tuple(row) for row in keys_df.rows()]
    n_clusters = len(key_tuples)
    groups: dict[tuple, pl.DataFrame] = {}
    for key in key_tuples:
        mask = pl.lit(True)
        for col, value in zip(cluster_cols, key):
            mask = mask & (pl.col(col) == value)
        groups[key] = frame.filter(mask)

    checkpoint = _load_checkpoint(checkpoint_path, fingerprint=checkpoint_fingerprint)
    replicate_rows: list[dict[str, Any]] = [checkpoint[i] for i in sorted(checkpoint)]
    for i in range(n_replicates):
        if i in checkpoint:
            continue
        seed = derive_replicate_seed(
            placebo_kind=f"{cluster_kind}:{hypothesis_id}",
            replicate_index=i,
            config_hash=config_hash,
        )
        rng = np.random.default_rng(seed)
        draw_idx = rng.integers(0, n_clusters, size=n_clusters)
        resampled = pl.concat([groups[key_tuples[j]] for j in draw_idx], how="vertical")
        pooled = _pool_cohort_ranks(
            resampled,
            min_events_per_market_contribution=min_events_per_market_contribution,
            min_events_per_cohort_total=min_events_per_cohort_total,
        )
        cohort_rows = pooled["cohort_rows"]
        ic_mean = float(np.mean([r[2] for r in cohort_rows])) if cohort_rows else None
        row = {"replicate": i, "seed": seed, "ic_mean": ic_mean}
        _append_checkpoint(checkpoint_path, row, fingerprint=checkpoint_fingerprint)
        replicate_rows.append(row)

    replicate_rows.sort(key=lambda r: r["replicate"])
    replicate_ids = {r["replicate"] for r in replicate_rows}
    if len(replicate_rows) != n_replicates or len(replicate_ids) != n_replicates:
        raise RuntimeError(
            f"{cluster_kind} bootstrap replicate set is incomplete or duplicated: "
            f"expected {n_replicates} unique replicates, got {len(replicate_ids)}"
        )

    replicate_ic_means = [r["ic_mean"] for r in replicate_rows]
    finite = [v for v in replicate_ic_means if v is not None and math.isfinite(v)]
    if not finite:
        return {
            "n_clusters": n_clusters,
            "n_valid_replicates": 0,
            "replicate_ic_means": replicate_ic_means,
            "bootstrap_mean": None,
            "bootstrap_p": None,
            "cluster_confirm_pass": False,
            "status_reason": "no_valid_replicate",
        }

    bootstrap_mean = float(np.mean(finite))
    p = _bootstrap_two_sided_p(finite)
    sign = -1.0 if expected_sign == "-" else 1.0
    sign_ok = (sign * bootstrap_mean) > 0
    return {
        "n_clusters": n_clusters,
        "n_valid_replicates": len(finite),
        "replicate_ic_means": replicate_ic_means,
        "bootstrap_mean": bootstrap_mean,
        "bootstrap_p": p,
        "cluster_confirm_pass": bool(sign_ok and p < p_max),
        "status_reason": None,
    }


def run_issuer_cluster_bootstrap(
    con: duckdb.DuckDBPyConnection,
    *,
    hypothesis_id: str,
    config_hash: str,
    h_start: int,
    h_end: int,
    sample_start: str,
    min_events_per_market_contribution: int,
    min_events_per_cohort_total: int,
    n_replicates: int = 999,
    expected_sign: str | None = None,
    p_max: float = 0.10,
    event_view: str = "fin_sue_event",
    calendar_view: str = "daily_ohlcv",
    sue_col: str = "fin_sue",
    checkpoint_path=None,
    checkpoint_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """§6 B-8 SUE point 2: resample issuers (tickers) — each drawn ticker's
    entire primary-sample event history travels together."""
    return run_cluster_bootstrap(
        con,
        event_view=event_view,
        calendar_view=calendar_view,
        sue_col=sue_col,
        h_start=h_start,
        h_end=h_end,
        sample_start=sample_start,
        min_events_per_market_contribution=min_events_per_market_contribution,
        min_events_per_cohort_total=min_events_per_cohort_total,
        cluster_cols=["ticker"],
        cluster_kind="issuer_bootstrap",
        hypothesis_id=hypothesis_id,
        config_hash=config_hash,
        n_replicates=n_replicates,
        expected_sign=expected_sign,
        p_max=p_max,
        checkpoint_path=checkpoint_path,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )


def run_filing_cycle_block_bootstrap(
    con: duckdb.DuckDBPyConnection,
    *,
    hypothesis_id: str,
    config_hash: str,
    h_start: int,
    h_end: int,
    sample_start: str,
    min_events_per_market_contribution: int,
    min_events_per_cohort_total: int,
    n_replicates: int = 999,
    expected_sign: str | None = None,
    p_max: float = 0.10,
    event_view: str = "fin_sue_event",
    calendar_view: str = "daily_ohlcv",
    sue_col: str = "fin_sue",
    checkpoint_path=None,
    checkpoint_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """§6 B-8 SUE point 3: resample ``(bsns_year, reprt_code)`` filing-cycle
    blocks — every event from every issuer sharing that reporting window
    travels together, testing dependence across the same reporting cycle
    rather than across one issuer's history."""
    return run_cluster_bootstrap(
        con,
        event_view=event_view,
        calendar_view=calendar_view,
        sue_col=sue_col,
        h_start=h_start,
        h_end=h_end,
        sample_start=sample_start,
        min_events_per_market_contribution=min_events_per_market_contribution,
        min_events_per_cohort_total=min_events_per_cohort_total,
        cluster_cols=["bsns_year", "reprt_code"],
        cluster_kind="filing_cycle_bootstrap",
        hypothesis_id=hypothesis_id,
        config_hash=config_hash,
        n_replicates=n_replicates,
        expected_sign=expected_sign,
        p_max=p_max,
        checkpoint_path=checkpoint_path,
        checkpoint_fingerprint=checkpoint_fingerprint,
    )


def evaluate_sue_cluster_confirmation(
    issuer_result: dict[str, Any], filing_cycle_result: dict[str, Any]
) -> dict[str, Any]:
    """§6 B-9 screen_pass rule 8: SUE passes only if *both* cluster
    bootstraps confirm — expected sign and ``bootstrap_p < p_max``."""
    return {
        "sue_cluster_confirm_pass": bool(
            issuer_result["cluster_confirm_pass"] and filing_cycle_result["cluster_confirm_pass"]
        ),
        "issuer_bootstrap_p": issuer_result["bootstrap_p"],
        "issuer_bootstrap_mean": issuer_result["bootstrap_mean"],
        "filing_cycle_bootstrap_p": filing_cycle_result["bootstrap_p"],
        "filing_cycle_bootstrap_mean": filing_cycle_result["bootstrap_mean"],
    }
