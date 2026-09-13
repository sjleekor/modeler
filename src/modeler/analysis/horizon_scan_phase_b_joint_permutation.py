"""§6 B-8 "결합 단면 permutation" — jointly permutes Phase A's 75 continuous
primary cells, Phase B's ready continuous primary cells, and Phase B's ready
SUE event cells under one replicate index, applies one combined BH pass per
replicate, and returns the null distribution of "how many discoveries would
this population produce by chance".

Deliberately does **not** compute the final empirical p-value here —
comparing this null distribution against the *real* combined discovery count
needs ``run_combined_ab``'s own result, and ``run_combined_ab`` is a pure
disk-reader by design (B-PR10: it never reconnects to the live lake so any
two published run directories can be combined without re-touching raw data).
This module's entry point runs inside ``run_phase_b_core`` instead (which
already holds the live connection/panel), and only the true count →
empirical p-value conversion happens later, in ``run_combined_ab``, as a
cheap pure-math step over this module's own persisted output.

Limitation (§6 B-8's own text: "이 permutation은 NW의 시계열 보정을 검증하지
않는다는 제한을 report에 명시한다"): like Phase A's own cross-sectional
permutation (``horizon_scan_permutation.run_cross_sectional_permutation``),
this breaks the feature-label link within each date×market block
(continuous) or within each event cohort (SUE) — it diagnoses join/leakage
and multiple-testing scale, not whether the Newey-West overlap correction
itself under-corrects long-horizon serial dependence.

``combined_continuous_registry`` must already carry ``scan_type``/``h_start``/
``h_end``/``feature``/``hypothesis_id``/``expected_sign`` per cell (Phase A's
own registry already does; Phase B's ``ready_continuous`` readiness rows use
``cell_type`` instead of ``scan_type`` and must be normalized by the caller
first — the same one-line mapping ``run_phase_b_continuous_scan`` already
does).
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from modeler.analysis.horizon_scan_mapping import (
    JOINT_CS_MAPPING_CONTRACT,
    apply_group_permutation,
    build_and_apply_group_permutation,
    build_group_permutation_mapping,
)
from modeler.analysis.horizon_scan_permutation import (
    _append_checkpoint,
    _load_checkpoint,
    _registry_target_columns,
    _scan_registry_once,
    derive_replicate_seed,
    fetch_broad_common_survivor_frame,
    permute_within_groups,
)
from modeler.analysis.horizon_scan_phase_b_scan import (
    _aggregate_cohort_rows,
    _pool_cohort_ranks,
    _pool_qualifying_by_date,
    execute_event_cohort_frame,
)
from modeler.analysis.horizon_scan_runner import apply_global_bh, assert_unique_hypothesis_ids
from modeler.etl.metrics import choose_nw_lag


def _permute_qualifying_sue_ranks(qualifying: pl.DataFrame, *, seed: int) -> pl.DataFrame:
    """Shuffle ``sue_pctrank`` within each ``(event_formation_date, market)``
    group, leaving ``excess_pctrank`` (and every other column) fixed to its
    original row — the "frozen rank vector" permutation §6 B-8 describes,
    applied directly to the already-computed ranks rather than re-deriving
    them from a resampled/permuted raw SUE value. Equivalent to permuting the
    raw values and re-ranking (rank is invariant to a shared within-group
    permutation of both sides), just without the redundant recompute — the
    real ``qualifying`` frame (from ``_pool_cohort_ranks``) is fetched once
    and reused across every replicate.
    """
    required = {"event_formation_date", "market", "ticker", "original_rcept_no"}
    missing = sorted(required - set(qualifying.columns))
    if missing:
        raise ValueError(f"SUE permutation frame is missing event grain columns: {missing}")
    canonical = qualifying
    grain = ["event_formation_date", "market", "ticker", "original_rcept_no"]
    canonical = canonical.sort(grain)
    if canonical.select(grain).is_duplicated().any():
        raise ValueError("SUE event grain key is duplicated")

    rng = np.random.default_rng(seed)
    parts = []
    for _, grp in canonical.group_by(["event_formation_date", "market"], maintain_order=True):
        shuffled = grp["sue_pctrank"].to_numpy()[rng.permutation(grp.height)]
        parts.append(grp.with_columns(pl.Series("sue_pctrank", shuffled)))
    return pl.concat(parts, how="vertical")


def _scan_sue_null_row(
    cell: dict[str, Any],
    qualifying: pl.DataFrame | None,
    *,
    seed: int,
    min_events_per_cohort_total: int,
) -> dict[str, Any]:
    """One SUE cell's null row for one replicate — no minimum-cohort-count
    gate (matches the issuer/filing-cycle bootstraps and the event-ordinal
    non-overlap diagnostic: a permutation replicate legitimately wants
    whatever cohort dates happen to qualify, however few)."""
    row = {
        **cell,
        "scan_type": "event_bucket",
        "status": "insufficient",
        "ic_mean": None,
        "p_nw": None,
    }
    if qualifying is None or qualifying.is_empty():
        return row
    permuted = _permute_qualifying_sue_ranks(qualifying, seed=seed)
    cohort_rows = _pool_qualifying_by_date(
        permuted, min_events_per_cohort_total=min_events_per_cohort_total
    )
    if not cohort_rows:
        return row
    lag = choose_nw_lag(scan_type="bucket", bucket_width=cell["h_end"] - cell["h_start"])
    agg = _aggregate_cohort_rows(cohort_rows, lag=lag)
    row.update(agg)
    ic_mean = agg.get("ic_mean")
    row["status"] = "valid" if ic_mean is not None and math.isfinite(ic_mean) else "insufficient"
    return row


def validate_reused_phase_a_mapping_hashes(
    frame: pl.DataFrame,
    phase_a_cell_stats: list[dict[str, Any]],
    *,
    config_hash: str,
    mapping_contract_version: str,
) -> dict[int, tuple[dict[tuple[Any, Any], np.ndarray], str]]:
    """Check A/B key-set identity before any Phase B null scan starts."""
    mappings_by_replicate: dict[int, tuple[dict[tuple[Any, Any], np.ndarray], str]] = {}
    expected = {
        int(row["replicate"]): row["mapping_hash"]
        for row in phase_a_cell_stats
        if row.get("mapping_hash") is not None
    }
    for replicate, expected_hash in sorted(expected.items()):
        mapping, actual_hash = build_group_permutation_mapping(
            frame,
            replicate_index=replicate,
            config_hash=config_hash,
            mapping_contract_version=mapping_contract_version,
        )
        if actual_hash != expected_hash:
            raise ValueError(
                f"Phase A/B mapping_hash mismatch for replicate {replicate}: "
                f"{expected_hash!r} != {actual_hash!r}"
            )
        mappings_by_replicate[replicate] = (mapping, actual_hash)
    return mappings_by_replicate


def _compute_combined_replicate(
    con: duckdb.DuckDBPyConnection,
    *,
    replicate: int,
    base_frame: pl.DataFrame | None,
    combined_continuous_registry: list[dict[str, Any]],
    ready_sue_cells: list[dict[str, Any]],
    sue_qualifying_by_hid: dict[str, pl.DataFrame | None],
    reused_rows: list[dict[str, Any]],
    config_hash: str,
    panel_view: str,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    min_events_per_cohort_total: int,
    q_threshold: float,
    scan_engine: str,
    mapping_contract_version: str,
    reused_mapping: tuple[dict[tuple[Any, Any], np.ndarray], str] | None = None,
) -> dict[str, Any]:
    continuous_rows: list[dict[str, Any]] = list(reused_rows)
    seed = derive_replicate_seed(
        placebo_kind="combined_cross_sectional", replicate_index=replicate, config_hash=config_hash
    )
    if base_frame is not None:
        feature_cols = sorted({hyp["feature"] for hyp in combined_continuous_registry})
        if mapping_contract_version == JOINT_CS_MAPPING_CONTRACT:
            if reused_mapping is None:
                permuted, mapping_hash = build_and_apply_group_permutation(
                    base_frame,
                    permute_cols=feature_cols,
                    replicate_index=replicate,
                    config_hash=config_hash,
                    mapping_contract_version=mapping_contract_version,
                )
            else:
                mapping, mapping_hash = reused_mapping
                permuted = apply_group_permutation(
                    base_frame, permute_cols=feature_cols, mappings=mapping
                )
        elif mapping_contract_version == "v1":
            permuted = permute_within_groups(
                base_frame,
                group_cols=["trade_date", "market"],
                permute_cols=feature_cols,
                seed=seed,
            )
            mapping_hash = None
        else:
            raise ValueError(f"unknown mapping contract: {mapping_contract_version!r}")
        continuous_rows = _scan_registry_once(
            con,
            combined_continuous_registry,
            panel_view=panel_view,
            sample_start=sample_start,
            min_names=min_names,
            min_names_for_spread=min_names_for_spread,
            quantile_count=quantile_count,
            min_dates_per_cell=min_dates_per_cell,
            scan_engine=scan_engine,
            frame=permuted,
        )
    else:
        mapping_hash = None

    sue_rows = [
        _scan_sue_null_row(
            cell,
            sue_qualifying_by_hid.get(cell["hypothesis_id"]),
            seed=derive_replicate_seed(
                placebo_kind=f"combined_sue_rank_permutation:{cell['hypothesis_id']}",
                replicate_index=replicate,
                config_hash=config_hash,
            ),
            min_events_per_cohort_total=min_events_per_cohort_total,
        )
        for cell in ready_sue_cells
    ]
    combined_rows = continuous_rows + sue_rows
    assert_unique_hypothesis_ids(combined_rows)
    bh_rows = apply_global_bh(combined_rows, q_threshold=q_threshold)
    return {
        "replicate": replicate,
        "seed": seed,
        "mapping_hash": mapping_hash,
        "n_discoveries": sum(1 for r in bh_rows if r["primary_discovery"]),
    }


def run_combined_cross_sectional_permutation(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str,
    combined_continuous_registry: list[dict[str, Any]],
    ready_sue_cells: list[dict[str, Any]],
    config_hash: str,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    min_events_per_market_contribution: int,
    min_events_per_cohort_total: int,
    n_replicates: int = 100,
    q_threshold: float = 0.10,
    event_view: str = "fin_sue_event",
    calendar_view: str = "daily_ohlcv",
    checkpoint_path: Path | None = None,
    scan_engine: str = "legacy",
    mapping_contract_version: str = "v1",
    checkpoint_fingerprint: dict[str, Any] | None = None,
    reused_phase_a_cell_stats: list[dict[str, Any]] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """§6 B-8: ``n_replicates`` joint permutations across
    ``combined_continuous_registry`` (Phase A's 75 + Phase B's ready
    continuous cells) and ``ready_sue_cells``, one combined BH pass per
    replicate. Returns ``{"replicate_summaries": [...],
    "null_discovery_counts": [...], "n_replicates": ...}`` — no
    ``p_empirical_count`` yet (see module docstring for why).

    Resume-safe via ``checkpoint_path`` (JSONL, one row per replicate),
    same convention as every other null experiment in this repo.
    """
    if reused_phase_a_cell_stats is not None:
        phase_a_ids = {row["hypothesis_id"] for row in reused_phase_a_cell_stats}
        combined_ids = {row["hypothesis_id"] for row in combined_continuous_registry}
        if not phase_a_ids.issubset(combined_ids):
            raise ValueError("reused Phase A permutation rows are outside the combined registry")
        combined_continuous_registry = [
            row for row in combined_continuous_registry if row["hypothesis_id"] not in phase_a_ids
        ]
    feature_cols = sorted({hyp["feature"] for hyp in combined_continuous_registry})
    target_cols = _registry_target_columns(combined_continuous_registry)
    base_frame = (
        fetch_broad_common_survivor_frame(
            con,
            panel_view=panel_view,
            extra_cols=feature_cols + target_cols,
            sample_start=sample_start,
        )
        if combined_continuous_registry
        else None
    )
    reused_mappings_by_replicate: dict[int, tuple[dict[tuple[Any, Any], np.ndarray], str]] = {}
    if reused_phase_a_cell_stats is not None and base_frame is not None:
        reused_mappings_by_replicate = validate_reused_phase_a_mapping_hashes(
            base_frame,
            reused_phase_a_cell_stats,
            config_hash=config_hash,
            mapping_contract_version=mapping_contract_version,
        )

    sue_qualifying_by_hid: dict[str, pl.DataFrame | None] = {}
    for cell in ready_sue_cells:
        frame = execute_event_cohort_frame(
            con,
            event_view=event_view,
            calendar_view=calendar_view,
            sue_col=cell["feature"],
            h_start=cell["h_start"],
            h_end=cell["h_end"],
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
        sue_qualifying_by_hid[cell["hypothesis_id"]] = pooled["qualifying"]

    checkpoint = _load_checkpoint(checkpoint_path, fingerprint=checkpoint_fingerprint)
    summaries: list[dict[str, Any]] = [checkpoint[i] for i in sorted(checkpoint)]
    reused_by_replicate: dict[int, list[dict[str, Any]]] = {}
    if reused_phase_a_cell_stats is not None:
        for row in reused_phase_a_cell_stats:
            reused_by_replicate.setdefault(int(row["replicate"]), []).append(row)
    if workers < 1:
        raise ValueError("workers must be >= 1")
    pending = [i for i in range(n_replicates) if i not in checkpoint]
    kwargs = {
        "base_frame": base_frame,
        "combined_continuous_registry": combined_continuous_registry,
        "ready_sue_cells": ready_sue_cells,
        "sue_qualifying_by_hid": sue_qualifying_by_hid,
        "config_hash": config_hash,
        "panel_view": panel_view,
        "sample_start": sample_start,
        "min_names": min_names,
        "min_names_for_spread": min_names_for_spread,
        "quantile_count": quantile_count,
        "min_dates_per_cell": min_dates_per_cell,
        "min_events_per_cohort_total": min_events_per_cohort_total,
        "q_threshold": q_threshold,
        "scan_engine": scan_engine,
        "mapping_contract_version": mapping_contract_version,
    }

    def _one(i: int) -> dict[str, Any]:
        return _compute_combined_replicate(
            con,
            replicate=i,
            reused_rows=reused_by_replicate.get(i, []),
            reused_mapping=reused_mappings_by_replicate.get(i),
            **kwargs,
        )

    if workers == 1:
        computed = (_one(i) for i in pending)
        computed_rows = zip(pending, computed)
    else:
        # Keep the large frame in-process; Polars releases the GIL in the hot
        # kernels and this avoids copying the frame into every process.
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = [executor.submit(_one, i) for i in pending]
        computed_rows = zip(pending, (future.result() for future in futures))
    try:
        for _replicate, summary in computed_rows:
            _append_checkpoint(checkpoint_path, summary, fingerprint=checkpoint_fingerprint)
            summaries.append(summary)
    finally:
        if workers > 1:
            executor.shutdown(wait=True)

    summaries.sort(key=lambda s: s["replicate"])
    replicate_ids = {s["replicate"] for s in summaries}
    if len(summaries) != n_replicates or len(replicate_ids) != n_replicates:
        raise RuntimeError(
            "combined cross-sectional permutation replicate set is incomplete or "
            f"duplicated: expected {n_replicates} unique replicates, got {len(replicate_ids)}"
        )

    return {
        "replicate_summaries": summaries,
        "null_discovery_counts": [s["n_discoveries"] for s in summaries],
        "n_replicates": n_replicates,
    }
