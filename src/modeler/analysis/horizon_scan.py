"""Phase A CLI: run-contract preflight, full pipeline stage orchestration, official/debug gate.

Wires together every stage built across A-PR1..A-PR5 — panel join (A-1), core
scan + global BH (A-2/A-3), robustness gates (A-4), non-overlap offsets (A-5),
permutation/placebo/canary (A-6), decay/pattern/screen/grade/card (A-7/A-8),
and plots/markdown/atomic-publish (A-9) — against the real A0 lake for the 17
Phase A families (12 ready, 1 reference, 4 short-exploratory).

``--smoke-family <family>`` restricts every stage to that one family's
hypotheses (§8.3 "빠른 smoke") and forces ``official=false`` via
``determine_official_mode`` — it is not a valid substitute for an official
run's statistical BH population. ``--permutations N`` overrides both replicate
counts (cross-sectional and temporal) for the same reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from modeler.analysis.horizon_scan_checkpoint import (
    build_checkpoint_fingerprint,
    checkpoint_namespace,
    coordinator_lock,
)
from modeler.analysis.horizon_scan_config import (
    CONFIG_PATH,
    DEFAULT_SCAN_ENGINE,
    SCAN_ENGINES,
    HorizonScanConfig,
    load_config,
    validate_scan_engine,
)
from modeler.analysis.horizon_scan_daily_ic import (
    DAILY_IC_DIR_NAME,
    DAILY_SPREAD_DIR_NAME,
    ParquetDailyIcSink,
    assert_daily_ic_reconciled,
    daily_ic_success_fields,
    reconcile_daily_ic,
)
from modeler.analysis.horizon_scan_permutation import (
    run_cross_sectional_permutation,
    run_lookahead_canary,
    run_temporal_placebo,
    select_long_horizon_hypotheses,
)
from modeler.analysis.horizon_scan_phase_b_run import run_combined_ab, run_phase_b_core
from modeler.analysis.horizon_scan_readiness import (
    build_primary_hypothesis_registry,
    build_short_exploratory_registry,
)
from modeler.analysis.horizon_scan_report import (
    assign_evidence_grade,
    build_family_card,
    classify_pattern_auto,
    compute_decay_summary,
    compute_screen_pass,
    render_family_plots,
    write_markdown_report,
)
from modeler.analysis.horizon_scan_run_spec import (
    REQUIRED_A0_MARTS,
    analysis_kernel_paths,
    build_run_spec,
    kst_now_iso,
    publish_run,
    write_run_spec,
)
from modeler.analysis.horizon_scan_runner import (
    apply_global_bh,
    build_period_segment_sql,
    compute_available_direction_pass,
    compute_delay_pass,
    compute_period_sign_pass,
    compute_tradable_pass,
    register_analysis_panel,
    resolve_common_formation_end,
    run_nonoverlap_offsets,
    run_registry_scan,
    scan_cell,
)
from modeler.analysis.horizon_scan_timing import StageTimingRecorder
from modeler.etl.config import REMOTE_SOURCE, DataRoot, LakeConfig
from modeler.etl.horizon_scan_inputs import REQUIRED_RAW_INPUTS
from modeler.etl.lake import connect
from modeler.etl.mart import mart_root, register_mart_view
from modeler.etl.snapshot import resolve_config

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- registry / config plumbing ---


def filter_registry_to_family(
    registry: list[dict[str, Any]], family: str | None
) -> list[dict[str, Any]]:
    if family is None:
        return registry
    return [r for r in registry if r["family"] == family]


def scan_kwargs_from_config(
    config: HorizonScanConfig, *, scan_engine: str = DEFAULT_SCAN_ENGINE
) -> dict[str, Any]:
    stats = config.raw["stats"]
    validate_scan_engine(scan_engine)
    return {
        "sample_start": str(config.raw["sample"]["start"]),
        "min_names": int(stats["min_names_per_date_market"]),
        "min_names_for_spread": int(stats["min_names_for_spread"]),
        "quantile_count": int(stats["quantile_count"]),
        "min_dates_per_cell": int(stats["min_dates_per_cell"]),
        "scan_engine": scan_engine,
    }


def register_a0_marts(con: duckdb.DuckDBPyConnection, lake: LakeConfig) -> None:
    for name in REQUIRED_A0_MARTS:
        register_mart_view(con, lake, name)


def register_period_segment_view(
    con: duckdb.DuckDBPyConnection,
    config: HorizonScanConfig,
    *,
    panel_view: str = "analysis_panel",
    view_name: str = "analysis_panel_periods",
) -> tuple[str, Any]:
    """A-4's period gate needs a period id per row for the common-survivor
    population only: the 'common' period set's last bound is the single
    global ``common_formation_end`` — the 'available' set's last bound is
    per-horizon (``horizon_eligible_end``) and is intentionally out of scope
    for this gate (it would need a fresh column per cell, not once per run).

    Returns ``(view_name, common_formation_end)`` — the resolved calendar
    date, not the config's ``common_formation_horizon`` window-length
    parameter, so callers (e.g. the run report) never confuse the two.
    """
    common_formation_end = resolve_common_formation_end(con, panel_view)
    period_sql = build_period_segment_sql(
        config.raw["sample"]["period_sets"]["common"],
        {"common_formation_end": common_formation_end},
    )
    con.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS "
        f"SELECT *, ({period_sql}) AS period_id_common FROM {panel_view}"
    )
    return view_name, common_formation_end


def compute_period_ics(
    con: duckdb.DuckDBPyConnection,
    *,
    panel_view: str,
    hyp: dict[str, Any],
    period_ids: list[str],
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    scan_engine: str = "legacy",
) -> list[float | None]:
    ics: list[float | None] = []
    for period_id in period_ids:
        cell = scan_cell(
            con,
            panel_view=panel_view,
            feature_col=hyp["feature"],
            scan_type=hyp["scan_type"],
            h_start=hyp["h_start"],
            h_end=hyp["h_end"],
            universe="broad",
            sample_kind="common_survivor",
            sample_start=sample_start,
            min_names=min_names,
            min_names_for_spread=min_names_for_spread,
            quantile_count=quantile_count,
            min_dates_per_cell=min_dates_per_cell,
            expected_sign=hyp.get("expected_sign"),
            extra_where=f"period_id_common = '{period_id}'",
            compute_spread=False,
            scan_engine=scan_engine,
        )
        ics.append(cell["ic_mean"] if cell["status"] == "valid" else None)
    return ics


def delay_gate_required(hyp: dict[str, Any]) -> bool:
    """§A-4: the delay gate only applies to h<=5 cumulative or bucket (0,5]."""
    if hyp["scan_type"] == "cum":
        return hyp["h_end"] <= 5
    return hyp["h_start"] == 0 and hyp["h_end"] == 5


def compute_family_delay_gate(
    con: duckdb.DuckDBPyConnection,
    *,
    hyp: dict[str, Any],
    lag1_feature: str,
    native_ic: float | None,
    min_retention: float,
    p_max: float,
    sample_start: str,
    min_names: int,
    min_names_for_spread: int,
    quantile_count: int,
    min_dates_per_cell: int,
    scan_engine: str = "legacy",
) -> dict[str, Any]:
    """A family whose official variant is already lag1 evaluates this against
    itself (native_ic IS the lag1 IC) — trivially passes, matching §A-4's own
    documented convention rather than special-cased."""
    lag1_cell = scan_cell(
        con,
        panel_view="analysis_panel",
        feature_col=lag1_feature,
        scan_type=hyp["scan_type"],
        h_start=hyp["h_start"],
        h_end=hyp["h_end"],
        universe="broad",
        sample_kind="common_survivor",
        sample_start=sample_start,
        min_names=min_names,
        min_names_for_spread=min_names_for_spread,
        quantile_count=quantile_count,
        min_dates_per_cell=min_dates_per_cell,
        expected_sign=hyp.get("expected_sign"),
        compute_spread=False,
        scan_engine=scan_engine,
    )
    ic_lag1 = lag1_cell["ic_mean"] if lag1_cell["status"] == "valid" else None
    p_nw_lag1 = lag1_cell["p_nw"] if lag1_cell["status"] == "valid" else None
    return compute_delay_pass(
        ic_native=native_ic,
        ic_lag1=ic_lag1,
        p_nw_lag1=p_nw_lag1,
        min_retention=min_retention,
        p_max=p_max,
    )


# --- per-family A-4/A-7/A-8 orchestration ---


def build_family_result(
    con: duckdb.DuckDBPyConnection,
    config: HorizonScanConfig,
    family: dict[str, Any],
    *,
    all_rows: list[dict[str, Any]],
    bh_rows_by_family: dict[str, list[dict[str, Any]]],
    period_view: str,
    period_ids: list[str],
    temporal_per_cell: dict[str, dict[str, Any]],
    scan_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """One family's full A-4..A-8 result: card + the per-cell scan rows the
    caller needs for plots (cumulative curves, bucket bars, native/lag1)."""
    fam_name = family["family"]
    role = family["role"]
    expected_sign = family.get("expected_sign")
    domain = family.get("fdr_family")
    primary_feature = next(f["column"] for f in family["features"] if f["role"] == "primary")
    secondary_features = [f["column"] for f in family["features"] if f["role"] == "secondary"]

    family_bh_rows = bh_rows_by_family.get(fam_name, [])
    cum_rows = [r for r in family_bh_rows if r["scan_type"] == "cum"]
    bucket_rows = [r for r in family_bh_rows if r["scan_type"] == "bucket"]
    decay = compute_decay_summary(
        cum_rows,
        bucket_rows,
        expected_sign=expected_sign,
        half_life_fraction=float(config.raw["decision"]["half_life_fraction"]),
    )

    by_key = {
        (r["hypothesis_id"], r["universe"], r["sample_kind"]): r
        for r in all_rows
        if r["family"] == fam_name
    }

    cell_gates: dict[str, dict[str, Any]] = {}
    for row in family_bh_rows:
        hid = row["hypothesis_id"]
        broad = by_key.get((hid, "broad", "common_survivor"))
        tradable = by_key.get((hid, "tradable", "common_survivor"))
        available = by_key.get((hid, "broad", "available"))
        tradable_gate = compute_tradable_pass(
            ic_broad=broad["ic_mean"] if broad else None,
            ic_tradable=tradable["ic_mean"] if tradable else None,
            min_retention=float(config.raw["decision"]["tradable_min_abs_ic_retention"]),
        )
        available_gate = compute_available_direction_pass(
            ic_common_survivor=broad["ic_mean"] if broad else None,
            ic_available=available["ic_mean"] if available else None,
        )
        required = delay_gate_required(row)
        if required:
            delay_gate = compute_family_delay_gate(
                con,
                hyp=row,
                lag1_feature=family["variant_columns"]["lag1"],
                native_ic=broad["ic_mean"] if broad else None,
                min_retention=float(config.raw["decision"]["delay_min_abs_ic_retention"]),
                p_max=float(config.raw["decision"]["delay_confirm_p_nw"]),
                **scan_kwargs,
            )
        else:
            delay_gate = {"delay_retention": None, "delay_pass": None}
        cell_gates[hid] = {
            "tradable_pass": tradable_gate["tradable_pass"],
            "available_direction_pass": available_gate["available_direction_pass"],
            "delay_required": required,
            "delay_pass": delay_gate["delay_pass"],
            "isolated_spike": bool(row.get("isolated_spike", False)),
            "primary_discovery": bool(row.get("primary_discovery", False)),
            "temporal_null_required": hid in temporal_per_cell,
            "temporal_null_pass": temporal_per_cell.get(hid, {}).get("temporal_null_pass"),
        }

    representative = None
    if decay.get("onset_h") is not None:
        representative = next((r for r in cum_rows if r["h_end"] == decay["onset_h"]), None)
    if representative is None and decay.get("peak_h_cum") is not None:
        representative = next((r for r in cum_rows if r["h_end"] == decay["peak_h_cum"]), None)
    if representative is None:
        representative = (cum_rows or bucket_rows or [None])[0]

    period_gate = {
        "valid_subperiods": 0,
        "sign_consistent_subperiods": 0,
        "period_sign_pass": False,
    }
    offset_summary: dict[str, Any] | None = None
    if representative is not None:
        period_ics = compute_period_ics(
            con, panel_view=period_view, hyp=representative, period_ids=period_ids, **scan_kwargs
        )
        period_gate = compute_period_sign_pass(period_ics, expected_sign=expected_sign)
        alignment_sign = -1.0 if expected_sign == "-" else 1.0
        offset_summary = run_nonoverlap_offsets(
            con,
            feature_col=representative["feature"],
            scan_type=representative["scan_type"],
            h_start=representative["h_start"],
            h_end=representative["h_end"],
            universe="broad",
            sample_kind="common_survivor",
            sample_start=scan_kwargs["sample_start"],
            min_names=scan_kwargs["min_names"],
            nonoverlap_min_dates=int(config.raw["stats"]["nonoverlap_min_dates"]),
            alignment_sign=alignment_sign,
            scan_engine=scan_kwargs["scan_engine"],
        )

    cell_screen: dict[str, dict[str, Any]] = {}
    for hid, gates in cell_gates.items():
        cell_screen[hid] = compute_screen_pass(
            role=role,
            primary_discovery=gates["primary_discovery"],
            tradable_pass=gates["tradable_pass"],
            period_sign_pass=period_gate["period_sign_pass"],
            isolated_spike=gates["isolated_spike"],
            available_direction_pass=gates["available_direction_pass"],
            delay_required=gates["delay_required"],
            delay_pass=gates["delay_pass"],
            temporal_null_required=gates["temporal_null_required"],
            temporal_null_pass=gates["temporal_null_pass"],
        )
    family_screen_pass = any(r["screen_pass"] for r in cell_screen.values())
    has_primary_discovery = any(g["primary_discovery"] for g in cell_gates.values())
    available_sign_flip = any(
        g["available_direction_pass"] is False
        for g in cell_gates.values()
        if g["primary_discovery"]
    )
    all_offsets_evaluable = bool(offset_summary and offset_summary["offset_status"] == "complete")
    has_nonfatal_warning = (period_gate["valid_subperiods"] < 3) or not all_offsets_evaluable
    grade = assign_evidence_grade(
        role="reference" if role == "reference_only" else ("ready" if role == "ready" else role),
        screen_pass=family_screen_pass,
        has_nonfatal_warning=has_nonfatal_warning,
        all_offsets_evaluable=all_offsets_evaluable,
        available_sign_flip=available_sign_flip,
    )
    # exploratory_short_regime cells are, by construction, always "outside"
    # the primary/BH grid (hypothesis_role != "primary") — so any of them
    # clearing the config's |t_nw|>3 diagnostic threshold is exactly the
    # "가설 밖 ... |t_nw|>3" case classify_pattern_auto's exploratory_only
    # branch means. Ready/reference families would need their own
    # exploratory_horizon_set/secondary-feature cells scanned to evaluate
    # this the same way — not done in this CLI (scope note, §8.5) — so they
    # always read has_exploratory_significant=False.
    exploratory_abs_t_nw = float(config.raw["stats"]["exploratory_abs_t_nw"])
    has_exploratory_significant = role == "exploratory_short_regime" and any(
        r.get("t_nw") is not None
        and math.isfinite(r["t_nw"])
        and abs(r["t_nw"]) > exploratory_abs_t_nw
        for r in family_bh_rows
    )
    pattern = classify_pattern_auto(
        has_primary_discovery=has_primary_discovery,
        has_exploratory_significant=has_exploratory_significant,
        peak_bucket=decay.get("peak_bucket"),
        sign_flip_bucket=decay.get("sign_flip_bucket"),
        segment_gates_all_pass=family_screen_pass if has_primary_discovery else None,
    )
    discoveries = [hid for hid, g in cell_gates.items() if g["primary_discovery"]]
    discovery_h_ends = [r["h_end"] for r in family_bh_rows if r["hypothesis_id"] in discoveries]
    band = (min(discovery_h_ends), max(discovery_h_ends)) if discovery_h_ends else None
    long_cell_p = next(
        (v["p_temporal_nw"] for hid, v in temporal_per_cell.items() if hid in cell_gates), None
    )
    long_cell_pass = next(
        (v["temporal_null_pass"] for hid, v in temporal_per_cell.items() if hid in cell_gates), None
    )

    lag1_ic = None
    if representative is not None:
        lag1_cell = scan_cell(
            con,
            panel_view="analysis_panel",
            feature_col=family["variant_columns"]["lag1"],
            scan_type=representative["scan_type"],
            h_start=representative["h_start"],
            h_end=representative["h_end"],
            universe="broad",
            sample_kind="common_survivor",
            expected_sign=expected_sign,
            compute_spread=False,
            **scan_kwargs,
        )
        lag1_ic = lag1_cell["ic_mean"] if lag1_cell["status"] == "valid" else None

    def _combo_ic(universe: str, sample_kind: str) -> float | None:
        if representative is None:
            return None
        return by_key.get((representative["hypothesis_id"], universe, sample_kind), {}).get(
            "ic_mean"
        )

    broad_ic = _combo_ic("broad", "common_survivor")
    tradable_ic = _combo_ic("tradable", "common_survivor")
    available_ic = _combo_ic("broad", "available")

    card = build_family_card(
        family=fam_name,
        domain=domain,
        primary_feature=primary_feature,
        secondary_features=secondary_features,
        expected_sign=expected_sign,
        observed_sign=(
            ("+" if (representative and (representative.get("ic_mean") or 0) >= 0) else "-")
            if representative
            else None
        ),
        decay_summary=decay,
        pattern_auto=pattern,
        primary_discoveries=discoveries,
        candidate_horizon_band=band,
        broad_ic=broad_ic,
        tradable_ic=tradable_ic,
        tradable_retention=(abs(tradable_ic / broad_ic) if broad_ic and tradable_ic else None),
        valid_subperiods=period_gate["valid_subperiods"],
        sign_consistent_subperiods=period_gate["sign_consistent_subperiods"],
        native_ic=broad_ic,
        lag1_ic=lag1_ic,
        delay_pass=(
            next((g["delay_pass"] for g in cell_gates.values() if g["delay_required"]), None)
        ),
        common_survivor_ic=broad_ic,
        available_ic=available_ic,
        attrition_warning=available_sign_flip,
        nonoverlap_offset_summary=offset_summary,
        kospi_weight_mean=representative.get("kospi_weight_mean") if representative else None,
        kosdaq_weight_mean=representative.get("kosdaq_weight_mean") if representative else None,
        p_temporal_nw=long_cell_p,
        temporal_null_pass=long_cell_pass,
        q_fdr_global=representative.get("q_fdr_global") if representative else None,
        evidence_grade=grade,
        screen_pass=family_screen_pass,
        sparse_primary_grid=fam_name in config.raw["stats"].get("sparse_primary_grid_families", []),
        exploratory_short_regime=role == "exploratory_short_regime",
        warnings=(["insufficient_offset_coverage"] if not all_offsets_evaluable else []),
        limitations=(
            ["survival_bias_unresolved"] if any(r["h_end"] >= 60 for r in family_bh_rows) else []
        ),
    )
    return {
        "card": card,
        "cum_rows": cum_rows,
        "bucket_rows": bucket_rows,
        "offset_summary": offset_summary,
    }


# --- top-level Phase A run ---


def run_phase_a(
    *,
    snapshot_date: str | None,
    source: str,
    data_lake_root: Path | None,
    smoke_family: str | None,
    permutations: int | None,
    include_holdout: bool,
    holdout_start: str | None,
    output_root: Path,
    command_line: list[str],
    resume: bool = False,
    checkpoint_root: Path | None = None,
    workers: int = 1,
    scan_engine: str = DEFAULT_SCAN_ENGINE,
    config_path: Path | str = CONFIG_PATH,
) -> Path:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    validate_scan_engine(scan_engine)
    config = load_config(config_path)
    root = DataRoot(data_lake_root) if data_lake_root else DataRoot.resolve(market="kr")
    lake, resolution = resolve_config(
        root, source, required_inputs=REQUIRED_RAW_INPUTS, snapshot_date=snapshot_date
    )
    # Mart cache validity is keyed by analysis_config_hash (research/etl/mart.py);
    # resolve_config only copies whatever the input LakeConfig already carried
    # there (None by default), so it must be set explicitly here — matching
    # build_a0_inputs's own `replace(pinned, analysis_config_hash=...)` — or
    # every register_mart_view call below fails as a false "hash mismatch".
    lake = replace(lake, analysis_config_hash=config.config_hash)
    manifest_path = mart_root(lake) / "_manifests" / "_SUCCESS.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"A0 manifest is required before a Phase A scan: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    a0_manifest_content_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    started_at = kst_now_iso()
    timings = StageTimingRecorder()
    run_spec = build_run_spec(
        config,
        manifest,
        snapshot_date=resolution.snapshot_date,
        source=resolution.source,
        resolution_auto_selected=resolution.auto_selected,
        smoke_family=smoke_family,
        permutation_repeats_override=permutations,
        include_holdout=include_holdout,
        holdout_start_override=holdout_start,
        repo_root=REPO_ROOT,
        code_paths=analysis_kernel_paths(REPO_ROOT),
        command_line=command_line,
        started_at=started_at,
    )
    run_spec["a0_manifest_content_hash"] = a0_manifest_content_hash
    run_spec["scan_engine"] = scan_engine

    # The run directory is named from the run_spec alone, so it can be
    # resolved here — before the scan — which is what lets the Stage 0 sink
    # write straight into `core/` instead of buffering a run's worth of daily
    # rows in memory. `write_run_spec` still creates and guards the directory
    # later; this only computes the paths.
    run_dir_root = (
        output_root
        / "phase=A"
        / f"snapshot_date={resolution.snapshot_date}"
        / f"source={resolution.source}"
        / f"config_hash={config.config_hash}"
    )
    tmp_run_dir = run_dir_root / f"run_id={run_spec['run_id']}.tmp"
    core_dir = tmp_run_dir / "core"

    timings.start("setup_and_register")
    con = connect(lake)
    register_a0_marts(con, lake)
    register_analysis_panel(con)
    period_view, common_formation_end = register_period_segment_view(con, config)
    period_ids = [p["id"] for p in config.raw["sample"]["period_sets"]["common"]]
    timings.stop("setup_and_register")

    primary_registry = filter_registry_to_family(
        build_primary_hypothesis_registry(config), smoke_family
    )
    short_registry = filter_registry_to_family(
        build_short_exploratory_registry(config), smoke_family
    )
    scan_kwargs = scan_kwargs_from_config(config, scan_engine=scan_engine)
    q_threshold = float(config.raw["stats"]["global_bh_q"])

    timings.start("real_scan")
    daily_sink = ParquetDailyIcSink(core_dir)
    all_rows = (
        run_registry_scan(con, primary_registry, **scan_kwargs, daily_sink=daily_sink)
        if primary_registry
        else []
    )
    short_rows = (
        run_registry_scan(con, short_registry, **scan_kwargs, daily_sink=daily_sink)
        if short_registry
        else []
    )
    daily_ic_summary = daily_sink.finalize()
    daily_ic_reconcile = assert_daily_ic_reconciled(
        reconcile_daily_ic(
            all_rows + short_rows,
            daily_ic_dir=core_dir / DAILY_IC_DIR_NAME,
            daily_spread_dir=core_dir / DAILY_SPREAD_DIR_NAME,
        )
    )
    timings.stop("real_scan")

    broad_common_survivor = [
        r for r in all_rows if r["universe"] == "broad" and r["sample_kind"] == "common_survivor"
    ]
    bh_rows = (
        apply_global_bh(broad_common_survivor, q_threshold=q_threshold)
        if broad_common_survivor
        else []
    )
    real_discovery_count = sum(1 for r in bh_rows if r["primary_discovery"])

    placebo_cfg = config.raw["placebo"]
    permutation_repeats = (
        permutations if permutations is not None else int(placebo_cfg["cross_sectional_repeats"])
    )
    temporal_repeats = (
        permutations if permutations is not None else int(placebo_cfg["temporal_long_cell_repeats"])
    )
    long_registry = select_long_horizon_hypotheses(primary_registry) if primary_registry else []
    checkpoint_base = checkpoint_root or output_root.parent / "horizon_scan_checkpoints"
    a0_manifest_hash = a0_manifest_content_hash
    execution = config.raw.get("execution", {})
    mapping_contract = str(execution.get("mapping_contract_version", "v1"))
    cross_namespace = checkpoint_namespace(
        checkpoint_base,
        phase="A",
        snapshot_date=resolution.snapshot_date,
        source=resolution.source,
        config_hash=config.config_hash,
        experiment="cross_sectional",
        contract=mapping_contract,
    )
    temporal_namespace = checkpoint_namespace(
        checkpoint_base,
        phase="A",
        snapshot_date=resolution.snapshot_date,
        source=resolution.source,
        config_hash=config.config_hash,
        experiment="temporal_placebo",
        contract="temporal_placebo_v1",
    )

    def _fingerprint(registry: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
        return build_checkpoint_fingerprint(
            registry=registry,
            a0_manifest_hash=a0_manifest_hash,
            readiness_population_hash=None,
            smoke_family=smoke_family,
            requested_replicates=repeats,
            include_holdout=include_holdout,
            holdout_start=holdout_start,
            scan_engine=scan_engine,
            row_order_contract=str(execution.get("row_order_contract", "legacy_input_order")),
            sue_nw_order_contract=str(execution.get("sue_nw_order_contract", "legacy")),
            sue_permutation_order_contract=str(
                execution.get("sue_permutation_order_contract", "legacy")
            ),
            mapping_contract_version=mapping_contract,
            analysis_kernel_hash=run_spec["analysis_kernel_hash"],
            duckdb_version=run_spec["duckdb_version"],
            polars_version=run_spec["polars_version"],
            numpy_version=run_spec["numpy_version"],
        )

    cross_fingerprint = _fingerprint(primary_registry, permutation_repeats)
    temporal_fingerprint = _fingerprint(long_registry, temporal_repeats) if long_registry else None
    run_spec["checkpoint"] = {
        "root": str(checkpoint_base),
        "resume": resume,
        "workers": workers,
        "namespaces": [str(cross_namespace), str(temporal_namespace)],
    }
    if not resume:
        for namespace in (cross_namespace, temporal_namespace):
            if any(namespace.glob("replicate=*.json")):
                raise RuntimeError(
                    f"checkpoint namespace already contains results; use --resume or choose "
                    f"another --checkpoint-root: {namespace}"
                )

    permutation_result: dict[str, Any] = {
        "replicate_summaries": [],
        "real_discovery_count": real_discovery_count,
        "p_empirical_count": None,
        "n_replicates": 0,
        "cell_stats": [],
    }
    if primary_registry and permutation_repeats > 0:
        timings.start("cross_sectional_permutation")
        try:
            with coordinator_lock(cross_namespace):
                permutation_result = run_cross_sectional_permutation(
                    con,
                    panel_view="analysis_panel",
                    primary_registry=primary_registry,
                    real_discovery_count=real_discovery_count,
                    config_hash=config.config_hash,
                    n_replicates=permutation_repeats,
                    q_threshold=q_threshold,
                    checkpoint_path=cross_namespace,
                    checkpoint_fingerprint=cross_fingerprint,
                    workers=workers,
                    worker_lake=lake,
                    **scan_kwargs,
                    mapping_contract_version=mapping_contract,
                    collect_cell_stats=True,
                )
        finally:
            timings.stop("cross_sectional_permutation")
            timings.write(cross_namespace / "timings.json")

    temporal_result: dict[str, Any] = {"replicate_meta": [], "per_cell": {}, "n_replicates": 0}
    if long_registry and temporal_repeats > 0:
        timings.start("temporal_placebo")
        real_t_nw_by_id = {r["hypothesis_id"]: r["t_nw"] for r in broad_common_survivor}
        try:
            with coordinator_lock(temporal_namespace):
                temporal_result = run_temporal_placebo(
                    con,
                    panel_view="analysis_panel",
                    long_horizon_registry=long_registry,
                    real_t_nw_by_id=real_t_nw_by_id,
                    config_hash=config.config_hash,
                    n_replicates=temporal_repeats,
                    min_shift_sessions=int(config.raw["placebo"]["temporal_min_shift_sessions"]),
                    p_max=float(config.raw["placebo"]["temporal_p_max"]),
                    checkpoint_path=temporal_namespace,
                    checkpoint_fingerprint=temporal_fingerprint,
                    workers=workers,
                    **scan_kwargs,
                )
        finally:
            timings.stop("temporal_placebo")
            timings.write(temporal_namespace / "timings.json")

    canary = run_lookahead_canary(
        con,
        sample_start=scan_kwargs["sample_start"],
        min_names=scan_kwargs["min_names"],
        min_dates_per_cell=scan_kwargs["min_dates_per_cell"],
    )

    bh_rows_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in bh_rows:
        bh_rows_by_family.setdefault(row["family"], []).append(row)
    for row in short_rows:
        if row["universe"] == "broad" and row["sample_kind"] == "common_survivor":
            bh_rows_by_family.setdefault(row["family"], []).append(row)

    phase_a_families = [f for f in config.families if f.get("phase") == "A"]
    if smoke_family is not None:
        phase_a_families = [f for f in phase_a_families if f["family"] == smoke_family]

    family_results = {
        f["family"]: build_family_result(
            con,
            config,
            f,
            all_rows=all_rows + short_rows,
            bh_rows_by_family=bh_rows_by_family,
            period_view=period_view,
            period_ids=period_ids,
            temporal_per_cell=temporal_result["per_cell"],
            scan_kwargs=scan_kwargs,
        )
        for f in phase_a_families
    }

    plots_dir = tmp_run_dir / "plots"
    timings.start("artifact_render")
    for fam_name, result in family_results.items():
        card = result["card"]
        cumulative_curves: dict[str, list[dict[str, Any]]] = {}
        for combo_label, universe, sample_kind in (
            ("broad_common_survivor", "broad", "common_survivor"),
            ("broad_available", "broad", "available"),
            ("tradable_common_survivor", "tradable", "common_survivor"),
            ("tradable_available", "tradable", "available"),
        ):
            cumulative_curves[combo_label] = [
                {"h_end": r["h_end"], "ic_mean": r["ic_mean"]}
                for r in (all_rows + short_rows)
                if r["family"] == fam_name
                and r["scan_type"] == "cum"
                and r["universe"] == universe
                and r["sample_kind"] == sample_kind
            ]
        render_family_plots(
            family=fam_name,
            output_dir=plots_dir,
            cumulative_curves=cumulative_curves,
            bucket_rows=result["bucket_rows"],
            expected_sign=card["expected_sign"],
            native_rows=cumulative_curves["broad_common_survivor"],
            lag1_rows=[],
            period_rows=[],
            segment_rows=[],
            coverage_rows=[],
            offset_summary=result["offset_summary"] or {"offsets": []},
        )

    write_run_spec(tmp_run_dir, run_spec)
    run_manifest = {
        **manifest,
        "artifacts": {**manifest.get("artifacts", {}), **daily_ic_summary.as_manifest_artifacts()},
    }
    (tmp_run_dir / "manifest.json").write_text(
        json.dumps(run_manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    core_dir.mkdir(parents=True, exist_ok=True)
    timings.write(tmp_run_dir / "timings.json")
    # §6.2: the broad/common-survivor primary rows must carry their BH fields
    # (q_fdr_global, bh_pass, primary_discovery, isolated_spike) — all_rows is
    # the pre-BH scan output, so swap in the BH-augmented row wherever one
    # exists; every other combo (tradable/available, short-exploratory) is
    # written as-is, with those BH-only fields absent (never a NULL 0 stand-in).
    bh_by_id = {r["hypothesis_id"]: r for r in bh_rows}
    output_rows = [
        (
            bh_by_id.get(r["hypothesis_id"], r)
            if r["universe"] == "broad" and r["sample_kind"] == "common_survivor"
            else r
        )
        for r in all_rows
    ] + short_rows
    if output_rows:
        pl.DataFrame(output_rows, infer_schema_length=None).write_parquet(
            core_dir / "horizon_ic.parquet"
        )
    if permutation_result.get("cell_stats"):
        pl.DataFrame(permutation_result["cell_stats"], infer_schema_length=None).write_parquet(
            core_dir / "permutation_cell_stats.parquet"
        )
    cards_dir = tmp_run_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    cards = [result["card"] for result in family_results.values()]
    (cards_dir / "family_cards.json").write_text(
        json.dumps(cards, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    price_cards = [c for c in cards if c["domain"] in ("price", "reference")]
    flow_cards = [c for c in cards if c["domain"] == "flow"]
    timings.stop("artifact_render")
    report_context = {
        "run_identity": {
            "run_id": run_spec["run_id"],
            "snapshot_date": run_spec["snapshot_date"],
            "source": run_spec["source"],
            "config_hash": run_spec["config_hash"],
            "official": run_spec["official"],
            "started_at": run_spec["started_at"],
            "finished_at": kst_now_iso(),
        },
        "preflight": {"status": "ok"},
        "sample_coverage": {
            "holdout_start": str(config.raw["sample"]["holdout_start"]),
            "effective_sample_start": scan_kwargs["sample_start"],
            "effective_sample_end": str(
                max(
                    (r["effective_sample_end"] for r in bh_rows if r["effective_sample_end"]),
                    default="",
                )
            )
            or None,
            "common_formation_end": str(common_formation_end),
        },
        "bh_summary": {
            "n_hypotheses": len(primary_registry),
            "n_valid": len(bh_rows),
            "n_bh_pass": sum(1 for r in bh_rows if r["bh_pass"]),
            "n_primary_discovery": real_discovery_count,
            "q_threshold": q_threshold,
        },
        "short_exploratory_summary": {
            "n_cells": len(short_registry),
            "n_valid": sum(1 for r in short_rows if r["status"] == "valid"),
        },
        "permutation_summary": {
            "real_discovery_count": permutation_result["real_discovery_count"],
            "p_empirical_count": permutation_result["p_empirical_count"],
            "n_replicates": permutation_result["n_replicates"],
        },
        "temporal_summary": {
            "n_replicates": temporal_result["n_replicates"],
            "per_cell": temporal_result["per_cell"],
        },
        "price_cards": price_cards,
        "flow_cards": flow_cards,
        "warnings": [f"{c['family']}: {w}" for c in cards for w in c["warnings"]],
        "acceptance_gate": [
            f"{c['family']}: band={c['candidate_horizon_band']} grade={c['evidence_grade']}"
            for c in cards
            if c["screen_pass"]
        ],
        "deferred_candidates": [
            f"{c['family']}: pattern={c['pattern_auto']}" for c in cards if not c["screen_pass"]
        ],
        "limitations": sorted({lim for c in cards for lim in c["limitations"]})
        + [
            f"look-ahead canary: canary_pass={canary['canary_pass']} "
            "(technical check, not an official artifact)"
        ],
        "timings": timings.as_dict(),
    }
    write_markdown_report(tmp_run_dir / "03a_horizon_scan_results.md", report_context)
    timings.write(tmp_run_dir / "timings.json")

    final_run_dir = run_dir_root / f"run_id={run_spec['run_id']}"
    return publish_run(
        tmp_run_dir,
        final_run_dir,
        run_spec=run_spec,
        success_extra=daily_ic_success_fields(daily_ic_reconcile),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="A", choices=["A", "B", "AB"])
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--source", default=REMOTE_SOURCE)
    parser.add_argument("--data-lake-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--smoke-family", default=None)
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--include-holdout", action="store_true")
    parser.add_argument("--holdout-start", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--resume", action="store_true", help="resume compatible replicate checkpoints"
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=None,
        help="checkpoint root outside the published phase directories",
    )
    parser.add_argument("--workers", type=int, default=1, help="replicate workers (default: 1)")
    parser.add_argument(
        "--scan-engine",
        choices=SCAN_ENGINES,
        default=DEFAULT_SCAN_ENGINE,
        help="rank IC implementation recorded in run_spec (default: polars_native_v1)",
    )
    parser.add_argument(
        "--phase-a-run-dir",
        type=Path,
        default=None,
        help="--phase AB only: a published phase=A run directory to combine",
    )
    parser.add_argument(
        "--phase-b-run-dir",
        type=Path,
        default=None,
        help="--phase AB only: a published phase=B run directory to combine",
    )
    parser.add_argument(
        "--phase-a-reuse-run-dir",
        type=Path,
        default=None,
        help="--phase B only: reuse Phase A permutation cell statistics",
    )
    # F-9.10. `argv or []` recorded only ["horizon_scan"] for every real run:
    # __main__ calls main() with argv=None, parse_args(None) then reads sys.argv
    # itself, and the empty list is what reached run_spec.json. All four
    # 2026-08-30 runs are missing their arguments because of this. config_hash
    # still pins the contract, so lineage was never lost -- what was lost is the
    # ability to reproduce a run from its own spec. Resolve the effective argv
    # once and use the same list for parsing and for the record.
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(effective_argv)
    command_line = ["horizon_scan", *effective_argv]
    output_root = args.output_root or (DataRoot.resolve(market="kr").output / "scan" / "horizon_scan")

    if args.phase == "A":
        published = run_phase_a(
            snapshot_date=args.snapshot_date,
            source=args.source,
            data_lake_root=args.data_lake_root,
            smoke_family=args.smoke_family,
            permutations=args.permutations,
            include_holdout=args.include_holdout,
            holdout_start=args.holdout_start,
            output_root=output_root,
            command_line=command_line,
            resume=args.resume,
            checkpoint_root=args.checkpoint_root,
            workers=args.workers,
            scan_engine=args.scan_engine,
            config_path=args.config,
        )
    elif args.phase == "B":
        published = run_phase_b_core(
            snapshot_date=args.snapshot_date,
            source=args.source,
            data_lake_root=args.data_lake_root,
            phase_a_run_dir=args.phase_a_reuse_run_dir,
            output_root=output_root,
            command_line=command_line,
            resume=args.resume,
            checkpoint_root=args.checkpoint_root,
            workers=args.workers,
            scan_engine=args.scan_engine,
            config_path=args.config,
        )
    else:
        if args.phase_a_run_dir is None or args.phase_b_run_dir is None:
            parser.error("--phase AB requires --phase-a-run-dir and --phase-b-run-dir")
        published = run_combined_ab(
            phase_a_run_dir=args.phase_a_run_dir,
            phase_b_run_dir=args.phase_b_run_dir,
            output_root=output_root,
            command_line=command_line,
            config_path=args.config,
        )
    print(published)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
