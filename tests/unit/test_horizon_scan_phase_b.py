from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import duckdb
import pytest
from modeler.analysis.horizon_scan_config import HorizonScanConfig, load_config
from modeler.analysis.horizon_scan_phase_b import (
    PHASE_B_CONTENT_HASH_EXCLUDE_NAMES,
    PHASE_B_MAX_CANDIDATES,
    ReceiptValuePairingReport,
    assert_receipt_value_pairing_verified,
    assert_required_raw_tables_present,
    build_phase_b_candidate_registry,
    build_phase_b_readiness_rows,
    build_phase_b_run_spec,
    check_receipt_value_pairing,
    phase_b_candidate_cells,
    summarize_phase_b_readiness,
    write_phase_b_readiness_freeze,
    write_phase_b_run_spec,
)
from modeler.analysis.horizon_scan_run_spec import PreflightError, publish_run

REPO_ROOT = Path(__file__).resolve().parents[2]

_HARD_REQUIRED_RAW_TABLES = {
    "daily_ohlcv",
    "dart_financial_statement_raw",
    "dart_share_count_raw",
    "dart_shareholder_return_raw",
    "dart_xbrl_fact_raw",
    "dart_xbrl_document",
    "dart_corp_master",
    "stock_master",
    "stock_master_snapshot",
    "stock_master_snapshot_items",
}


def _fake_a0_run_spec(config: HorizonScanConfig) -> dict:
    return {
        "snapshot_date": "2026-08-01",
        "source": "sj2_remote",
        "raw_manifest_hash": "raw-hash-abc",
        "a0_manifest_hash": "a0-hash-def",
        "config_schema_version": config.raw["schema_version"],
        "label_policy_version": {
            "holdout_start": "2025-08-01",
            "holdout_boundary": "label_end_date",
        },
        "quality_policy_version": {"placeholder": True},
        "universe_policy_version": {"placeholder": True},
    }


# --- candidate registry ------------------------------------------------------


def test_phase_b_candidate_registry_has_38_cells() -> None:
    config = load_config()
    cells = build_phase_b_candidate_registry(config)
    assert len(cells) == PHASE_B_MAX_CANDIDATES


def test_phase_b_candidate_cell_ids_are_unique() -> None:
    config = load_config()
    cells = phase_b_candidate_cells(config)
    ids = [cell["hypothesis_id"] for cell in cells]
    assert len(set(ids)) == len(ids)


def test_sue_family_contributes_six_event_bucket_cells() -> None:
    config = load_config()
    cells = phase_b_candidate_cells(config)
    sue_cells = [c for c in cells if c["family"] == "fin_sue"]
    assert len(sue_cells) == 6
    assert all(c["cell_type"] == "event_bucket" for c in sue_cells)
    assert all("|event|" in c["hypothesis_id"] for c in sue_cells)
    assert sue_cells[0]["expected_sign"] == "+"


def test_continuous_families_have_no_event_bucket_cells() -> None:
    config = load_config()
    cells = phase_b_candidate_cells(config)
    continuous = [c for c in cells if c["family"] != "fin_sue"]
    assert len(continuous) == 32
    assert all(c["cell_type"] in {"cumulative", "bucket"} for c in continuous)


def test_candidate_registry_rejects_drifted_count() -> None:
    config = load_config()
    raw = deepcopy(config.raw)
    family = next(f for f in raw["families"] if f["family"] == "fin_log_mcap")
    family["primary_horizon_set"] = [60, 120, 20]  # drift: now contributes 5, not 4
    mutated = HorizonScanConfig(raw=raw, config_hash="test", path=config.path)
    with pytest.raises(ValueError, match="must have 38 cells"):
        build_phase_b_candidate_registry(mutated)


# --- required raw table hard gate -------------------------------------------


def test_assert_required_raw_tables_present_accepts_complete_manifest() -> None:
    config = load_config()
    assert_required_raw_tables_present(config, _HARD_REQUIRED_RAW_TABLES)


def test_assert_required_raw_tables_present_rejects_incomplete_manifest() -> None:
    config = load_config()
    incomplete = _HARD_REQUIRED_RAW_TABLES - {"dart_corp_master"}
    with pytest.raises(PreflightError, match="dart_corp_master"):
        assert_required_raw_tables_present(config, incomplete)


def test_new_phase_b_raw_tables_are_not_hard_required() -> None:
    # dart_filing_receipt_raw / dart_capital_change_raw absence must not block
    # the whole run — only the candidates that depend on them (§1.1 condition 3).
    config = load_config()
    assert_required_raw_tables_present(config, _HARD_REQUIRED_RAW_TABLES)
    required = set(config.raw["phase_b"]["required_raw_tables"])
    assert "dart_filing_receipt_raw" not in required
    assert "dart_capital_change_raw" not in required


# --- outcome-blind readiness freeze -----------------------------------------


def test_readiness_freeze_blocks_everything_when_no_marts_available() -> None:
    config = load_config()
    rows = build_phase_b_readiness_rows(config, available_assets=set())
    assert len(rows) == PHASE_B_MAX_CANDIDATES
    assert all(row["role"] == "blocked_exploratory" for row in rows)
    summary = summarize_phase_b_readiness(config, rows)
    assert summary["m_b_ready"] == 0
    assert summary["blocked_exploratory_count"] == 38
    assert summary["combined_ab_hypothesis_count"] == 75


def test_readiness_freeze_frees_financial_families_once_their_mart_exists() -> None:
    config = load_config()
    rows = build_phase_b_readiness_rows(
        config, available_assets={"feat_fin_scan_daily", "label_scan"}
    )
    by_family = {}
    for row in rows:
        by_family.setdefault(row["family"], []).append(row)
    financial_families = {
        "fin_log_mcap",
        "fin_value_z",
        "fin_gross_profitability",
        "fin_asset_growth_yoy",
        "fin_accruals_to_assets",
    }
    for family in financial_families:
        assert all(r["role"] == "ready_primary" for r in by_family[family])
    assert all(r["role"] == "blocked_exploratory" for r in by_family["fin_sue"])
    assert all(r["role"] == "blocked_exploratory" for r in by_family["ev_net_share_issuance_yoy"])
    assert all(r["role"] == "blocked_exploratory" for r in by_family["ev_payout_yield"])

    summary = summarize_phase_b_readiness(config, rows)
    assert summary["m_b_ready"] == 24  # 4+4+8+4+4
    assert summary["combined_ab_hypothesis_count"] == 99


def test_readiness_rows_never_take_a_label_or_p_value_argument() -> None:
    import inspect

    signature = inspect.signature(build_phase_b_readiness_rows)
    assert "label" not in signature.parameters
    assert "p_value" not in signature.parameters
    assert "return_value" not in signature.parameters


def test_summarize_rejects_a_row_count_that_is_not_38() -> None:
    config = load_config()
    with pytest.raises(ValueError, match="equal the frozen candidate count"):
        summarize_phase_b_readiness(config, [{"role": "ready_primary"}])


def test_write_phase_b_readiness_freeze_is_immutable(tmp_path: Path) -> None:
    config = load_config()
    rows = build_phase_b_readiness_rows(config, available_assets=set())
    output_path = tmp_path / "phase_b_readiness_freeze.json"
    write_phase_b_readiness_freeze(config, rows, output_path, generated_at="2026-08-10T12:00:00")
    with pytest.raises(FileExistsError):
        write_phase_b_readiness_freeze(
            config, rows, output_path, generated_at="2026-08-10T12:00:01"
        )


def test_write_phase_b_readiness_freeze_content_round_trips(tmp_path: Path) -> None:
    config = load_config()
    rows = build_phase_b_readiness_rows(config, available_assets=set())
    output_path = tmp_path / "phase_b_readiness_freeze.json"
    write_phase_b_readiness_freeze(config, rows, output_path, generated_at="2026-08-10T12:00:00")
    payload = json.loads(output_path.read_text())
    assert payload["max_candidates"] == 38
    assert payload["m_b_ready"] == 0
    assert len(payload["cells"]) == 38
    assert payload["config_hash"] == config.config_hash


# --- phase_b_run_spec.json ---------------------------------------------------


def test_build_phase_b_run_spec_requires_hard_raw_tables() -> None:
    config = load_config()
    with pytest.raises(PreflightError, match="dart_corp_master"):
        build_phase_b_run_spec(
            config,
            a0_run_spec=_fake_a0_run_spec(config),
            raw_manifest_tables=_HARD_REQUIRED_RAW_TABLES - {"dart_corp_master"},
            repo_root=REPO_ROOT,
            code_paths=[REPO_ROOT / "src/modeler/analysis/horizon_scan_phase_b.py"],
            command_line=["test"],
            started_at="2026-08-10T12:00:00+09:00",
        )


def test_build_phase_b_run_spec_inherits_a0_fingerprint() -> None:
    config = load_config()
    a0_run_spec = _fake_a0_run_spec(config)
    spec = build_phase_b_run_spec(
        config,
        a0_run_spec=a0_run_spec,
        raw_manifest_tables=_HARD_REQUIRED_RAW_TABLES,
        repo_root=REPO_ROOT,
        code_paths=[REPO_ROOT / "src/modeler/analysis/horizon_scan_phase_b.py"],
        command_line=["test"],
        started_at="2026-08-10T12:00:00+09:00",
    )
    assert spec["phase"] == "B"
    assert spec["snapshot_date"] == a0_run_spec["snapshot_date"]
    assert spec["source"] == a0_run_spec["source"]
    assert spec["raw_manifest_hash"] == a0_run_spec["raw_manifest_hash"]
    assert spec["a0_manifest_hash"] == a0_run_spec["a0_manifest_hash"]
    assert spec["holdout_start"] == "2025-08-01"
    assert spec["config_hash"] == config.config_hash
    assert "run_id" in spec and spec["phase_b_code_hash"][:8] in spec["run_id"]


def test_write_phase_b_run_spec_refuses_overwrite(tmp_path: Path) -> None:
    config = load_config()
    spec = build_phase_b_run_spec(
        config,
        a0_run_spec=_fake_a0_run_spec(config),
        raw_manifest_tables=_HARD_REQUIRED_RAW_TABLES,
        repo_root=REPO_ROOT,
        code_paths=[REPO_ROOT / "src/modeler/analysis/horizon_scan_phase_b.py"],
        command_line=["test"],
        started_at="2026-08-10T12:00:00+09:00",
    )
    write_phase_b_run_spec(tmp_path, spec)
    with pytest.raises(FileExistsError):
        write_phase_b_run_spec(tmp_path, spec)


def test_publish_phase_b_run_is_atomic_with_success_marker_last(tmp_path: Path) -> None:
    config = load_config()
    spec = build_phase_b_run_spec(
        config,
        a0_run_spec=_fake_a0_run_spec(config),
        raw_manifest_tables=_HARD_REQUIRED_RAW_TABLES,
        repo_root=REPO_ROOT,
        code_paths=[REPO_ROOT / "src/modeler/analysis/horizon_scan_phase_b.py"],
        command_line=["test"],
        started_at="2026-08-10T12:00:00+09:00",
    )
    tmp_run_dir = tmp_path / "run.tmp"
    final_run_dir = tmp_path / f"run_id={spec['run_id']}"
    write_phase_b_run_spec(tmp_run_dir, spec)
    rows = build_phase_b_readiness_rows(config, available_assets=set())
    write_phase_b_readiness_freeze(
        config,
        rows,
        tmp_run_dir / "phase_b_readiness_freeze.json",
        generated_at="2026-08-10T12:00:00",
    )

    published = publish_run(
        tmp_run_dir,
        final_run_dir,
        run_spec=spec,
        required_artifacts=("phase_b_run_spec.json", "phase_b_readiness_freeze.json"),
        content_hash_exclude_names=PHASE_B_CONTENT_HASH_EXCLUDE_NAMES,
    )
    assert published == final_run_dir
    assert (final_run_dir / "_SUCCESS.json").is_file()
    assert not tmp_run_dir.exists()

    # Rebuild an intact tmp dir so the second call fails on "final dir already
    # exists", not on "tmp dir missing" (the first call renamed it away).
    write_phase_b_run_spec(tmp_run_dir, spec)
    write_phase_b_readiness_freeze(
        config,
        rows,
        tmp_run_dir / "phase_b_readiness_freeze.json",
        generated_at="2026-08-10T12:00:00",
    )
    with pytest.raises(FileExistsError):
        publish_run(
            tmp_run_dir,
            final_run_dir,
            run_spec=spec,
            required_artifacts=("phase_b_run_spec.json", "phase_b_readiness_freeze.json"),
            content_hash_exclude_names=PHASE_B_CONTENT_HASH_EXCLUDE_NAMES,
        )


def test_content_hash_is_identical_across_reruns_despite_different_timestamps(
    tmp_path: Path,
) -> None:
    config = load_config()

    def _make_run(run_label: str, started_at: str) -> Path:
        spec = build_phase_b_run_spec(
            config,
            a0_run_spec=_fake_a0_run_spec(config),
            raw_manifest_tables=_HARD_REQUIRED_RAW_TABLES,
            repo_root=REPO_ROOT,
            code_paths=[REPO_ROOT / "src/modeler/analysis/horizon_scan_phase_b.py"],
            command_line=["test"],
            started_at=started_at,
            finished_at=started_at,
        )
        run_dir = tmp_path / run_label
        write_phase_b_run_spec(run_dir, spec)
        rows = build_phase_b_readiness_rows(config, available_assets=set())
        write_phase_b_readiness_freeze(
            config, rows, run_dir / "phase_b_readiness_freeze.json", generated_at=started_at
        )
        return run_dir

    from modeler.analysis.horizon_scan_run_spec import compute_run_content_hash

    run_a = _make_run("run_a", "2026-08-10T12:00:00+09:00")
    run_b = _make_run("run_b", "2026-08-10T15:30:00+09:00")

    hash_a = compute_run_content_hash(run_a, exclude_names=PHASE_B_CONTENT_HASH_EXCLUDE_NAMES)
    hash_b = compute_run_content_hash(run_b, exclude_names=PHASE_B_CONTENT_HASH_EXCLUDE_NAMES)
    assert hash_a == hash_b


# --- receipt-value pairing preflight ----------------------------------------


def _connect_with_financial_and_xbrl(financial_rows: list[tuple], xbrl_rows: list[tuple]):
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE dart_financial_statement_raw ("
        "corp_code VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, rcept_no VARCHAR)"
    )
    con.execute(
        "CREATE TABLE dart_xbrl_fact_raw ("
        "corp_code VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, rcept_no VARCHAR)"
    )
    if financial_rows:
        con.executemany(
            "INSERT INTO dart_financial_statement_raw VALUES (?, ?, ?, ?)", financial_rows
        )
    if xbrl_rows:
        con.executemany("INSERT INTO dart_xbrl_fact_raw VALUES (?, ?, ?, ?)", xbrl_rows)
    return con


def test_check_receipt_value_pairing_reports_coverage_when_clean() -> None:
    con = _connect_with_financial_and_xbrl(
        financial_rows=[
            ("00126380", 2025, "11011", "r1"),
            ("00164779", 2025, "11011", "r2"),
        ],
        xbrl_rows=[("00126380", 2025, "11011", "r1")],
    )
    report = check_receipt_value_pairing(con)
    assert report.financial_key_count == 2
    assert report.xbrl_covered_key_count == 1
    assert report.coverage_ratio == pytest.approx(0.5)
    assert report.verified_same_receipt is True
    assert report.multi_receipt_filing_keys == ()


def test_check_receipt_value_pairing_detects_multiple_receipts_per_filing_key() -> None:
    con = _connect_with_financial_and_xbrl(
        financial_rows=[
            ("00126380", 2025, "11011", "r1"),
            ("00126380", 2025, "11011", "r1-revision"),
        ],
        xbrl_rows=[],
    )
    report = check_receipt_value_pairing(con)
    assert report.verified_same_receipt is False
    assert report.multi_receipt_filing_keys == (("00126380", 2025, "11011"),)


def test_assert_receipt_value_pairing_verified_passes_when_clean() -> None:
    report = ReceiptValuePairingReport(
        financial_key_count=1,
        xbrl_covered_key_count=1,
        coverage_ratio=1.0,
        multi_receipt_filing_keys=(),
        verified_same_receipt=True,
    )
    assert_receipt_value_pairing_verified(report)  # does not raise


def test_assert_receipt_value_pairing_verified_raises_on_mismatch() -> None:
    report = ReceiptValuePairingReport(
        financial_key_count=2,
        xbrl_covered_key_count=0,
        coverage_ratio=0.0,
        multi_receipt_filing_keys=(("00126380", 2025, "11011"),),
        verified_same_receipt=False,
    )
    with pytest.raises(PreflightError, match="receipt-value pairing failed"):
        assert_receipt_value_pairing_verified(report)


# --- macro overlay readiness (02_stage1a §3, 05_preregistration_record) ---


def _macro_config():
    from modeler.analysis.horizon_scan_config import CONFIG_PATH

    return load_config(CONFIG_PATH.with_name("horizon_scan_macro_20260829.yaml"))


_MACRO_FAMILIES = {
    "macro_beta_usdkrw",
    "macro_beta_wti",
    "macro_beta_kr10y",
    "macro_beta_sp500_lag",
    "macro_beta_vix",
    "px_market_beta",
}


def test_macro_families_are_blocked_without_the_exposure_mart() -> None:
    """A snapshot whose ``common_feature_daily_fact`` was never built has no
    ``feat_macro_exposure`` either — both names are absent, and the six
    families take the ordinary blocked path rather than failing the run."""
    config = _macro_config()
    rows = build_phase_b_readiness_rows(
        config, available_assets={"feat_fin_scan_daily", "label_scan"}
    )
    macro = [r for r in rows if r["family"] in _MACRO_FAMILIES]
    assert len(macro) == 24  # 6 families x 4 cells
    assert all(r["role"] == "blocked_exploratory" for r in macro)
    assert all("feat_macro_exposure" in r["missing_dependencies"] for r in macro)


def test_macro_families_need_the_fact_as_well_as_the_mart() -> None:
    """``common_feature_daily_fact`` is a declared dependency in its own right:
    the readiness gate and the mart have to be looking at the same fact, so a
    run that has the mart but never bound the fact is not ready."""
    config = _macro_config()
    rows = build_phase_b_readiness_rows(
        config, available_assets={"feat_macro_exposure", "label_scan"}
    )
    macro = [r for r in rows if r["family"] in _MACRO_FAMILIES]
    assert all(r["role"] == "blocked_exploratory" for r in macro)
    assert all("common_feature_daily_fact" in r["missing_dependencies"] for r in macro)


def test_macro_families_are_ready_once_both_dependencies_exist() -> None:
    config = _macro_config()
    rows = build_phase_b_readiness_rows(
        config,
        available_assets={"feat_macro_exposure", "common_feature_daily_fact", "label_scan"},
    )
    macro = [r for r in rows if r["family"] in _MACRO_FAMILIES]
    assert len(macro) == 24
    assert all(r["role"] == "ready_primary" for r in macro)
    summary = summarize_phase_b_readiness(config, rows)
    assert summary["m_b_ready"] == 24
    # Phase A's 75 are untouched by the append; the combined population grows
    # only by what actually became ready.
    assert summary["combined_ab_hypothesis_count"] == 75 + 24


def test_macro_overlay_freezes_102_phase_b_candidates() -> None:
    config = _macro_config()
    rows = build_phase_b_readiness_rows(config, available_assets=set())
    assert len(rows) == 102
    summary = summarize_phase_b_readiness(config, rows)
    assert summary["blocked_exploratory_count"] == 102
