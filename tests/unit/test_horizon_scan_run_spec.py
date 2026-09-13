from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from modeler.analysis.horizon_scan_config import HorizonScanConfig, load_config
from modeler.analysis.horizon_scan_run_spec import (
    REQUIRED_A0_MARTS,
    REQUIRED_RUN_ARTIFACTS,
    PreflightError,
    assert_a0_manifest_matches,
    assert_family_registry_complete,
    assert_holdout_untouched,
    build_run_spec,
    compute_run_content_hash,
    determine_official_mode,
    phase_a_code_hash,
    publish_run,
    run_preflight_checks,
    write_run_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _valid_manifest(config: HorizonScanConfig) -> dict:
    return {
        "status": "success",
        "config_hash": config.config_hash,
        "smoke_only": False,
        "raw_marker": "/tmp/raw_marker.json",
        "marts": [
            {"view": name, "row_count": 100, "schema_hash": f"hash-{name}"}
            for name in REQUIRED_A0_MARTS
        ],
    }


def test_manifest_check_accepts_matching_successful_non_smoke_manifest() -> None:
    config = load_config()
    assert_a0_manifest_matches(config, _valid_manifest(config))


def test_manifest_check_rejects_config_hash_mismatch() -> None:
    config = load_config()
    manifest = _valid_manifest(config)
    manifest["config_hash"] = "different-hash"
    with pytest.raises(PreflightError, match="config_hash does not match"):
        assert_a0_manifest_matches(config, manifest)


def test_manifest_check_rejects_smoke_only_manifest() -> None:
    config = load_config()
    manifest = _valid_manifest(config)
    manifest["smoke_only"] = True
    with pytest.raises(PreflightError, match="smoke_only=true"):
        assert_a0_manifest_matches(config, manifest)


def test_manifest_check_rejects_missing_or_empty_marts() -> None:
    config = load_config()
    manifest = _valid_manifest(config)
    manifest["marts"] = [m for m in manifest["marts"] if m["view"] != "feat_flow"]
    with pytest.raises(PreflightError, match="absent"):
        assert_a0_manifest_matches(config, manifest)

    manifest = _valid_manifest(config)
    manifest["marts"][0]["row_count"] = 0
    with pytest.raises(PreflightError, match="empty"):
        assert_a0_manifest_matches(config, manifest)


def test_family_registry_completeness_rejects_wrong_role_counts() -> None:
    config = load_config()
    assert_family_registry_complete(config)  # real config passes

    raw = deepcopy(config.raw)
    reference = next(f for f in raw["families"] if f["role"] == "reference_only")
    reference["role"] = "ready"  # now 13 ready / 0 reference
    mutated = HorizonScanConfig(raw=raw, config_hash="test", path=config.path)
    with pytest.raises(PreflightError, match="role counts must be"):
        assert_family_registry_complete(mutated)


def test_holdout_override_is_always_rejected() -> None:
    assert_holdout_untouched(include_holdout=False, holdout_start_override=None)
    with pytest.raises(PreflightError, match="not permitted"):
        assert_holdout_untouched(include_holdout=True, holdout_start_override=None)
    with pytest.raises(PreflightError, match="not permitted"):
        assert_holdout_untouched(include_holdout=False, holdout_start_override="2020-01-01")


def test_manual_snapshot_selection_is_never_official() -> None:
    official, smoke_only, reasons = determine_official_mode(
        resolution_auto_selected=False,
        smoke_family=None,
        permutation_repeats_override=None,
        config_permutation_repeats=100,
    )
    assert official is False
    assert smoke_only is True
    assert reasons == ["snapshot_manually_overridden"]


def test_smoke_family_and_permutation_override_are_never_official() -> None:
    official, smoke_only, reasons = determine_official_mode(
        resolution_auto_selected=True,
        smoke_family="px_reversal_5d",
        permutation_repeats_override=1,
        config_permutation_repeats=100,
    )
    assert official is False
    assert smoke_only is True
    assert reasons == ["smoke_family_filter", "permutation_repeats_override"]


def test_full_auto_selected_run_with_no_overrides_is_official() -> None:
    official, smoke_only, reasons = determine_official_mode(
        resolution_auto_selected=True,
        smoke_family=None,
        permutation_repeats_override=None,
        config_permutation_repeats=100,
    )
    assert official is True
    assert smoke_only is False
    assert reasons == []


def test_run_preflight_checks_rejects_hypothesis_registry_drift() -> None:
    config = load_config()
    manifest = _valid_manifest(config)
    raw = deepcopy(config.raw)
    reversal = next(f for f in raw["families"] if f["family"] == "px_reversal_5d")
    reversal["primary_horizon_set"].remove(1)
    drifted = HorizonScanConfig(raw=raw, config_hash=config.config_hash, path=config.path)
    with pytest.raises(ValueError, match="must have 75 cells, got 74"):
        run_preflight_checks(drifted, manifest)


def test_phase_a_code_hash_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    first = phase_a_code_hash([a, b])
    second = phase_a_code_hash([b, a])  # order-independent (sorted internally)
    assert first == second

    b.write_text("y = 3\n")
    assert phase_a_code_hash([a, b]) != first


def test_build_run_spec_rejects_holdout_override_before_touching_manifest() -> None:
    config = load_config()
    manifest = _valid_manifest(config)
    manifest["status"] = "failed"  # would also fail preflight, but holdout check runs first
    with pytest.raises(PreflightError, match="not permitted"):
        build_run_spec(
            config,
            manifest,
            snapshot_date="2026-08-01",
            source="sj2_remote",
            resolution_auto_selected=True,
            smoke_family=None,
            permutation_repeats_override=None,
            include_holdout=True,
            holdout_start_override=None,
            repo_root=REPO_ROOT,
            code_paths=[Path(__file__)],
            command_line=["horizon_scan", "--phase", "A"],
            started_at="2026-08-02T00:00:00+09:00",
        )


def test_build_run_spec_produces_expected_fields_for_official_run() -> None:
    config = load_config()
    manifest = _valid_manifest(config)
    spec = build_run_spec(
        config,
        manifest,
        snapshot_date="2026-08-01",
        source="sj2_remote",
        resolution_auto_selected=True,
        smoke_family=None,
        permutation_repeats_override=None,
        include_holdout=False,
        holdout_start_override=None,
        repo_root=REPO_ROOT,
        code_paths=[Path(__file__)],
        command_line=["horizon_scan", "--phase", "A"],
        started_at="2026-08-02T00:00:00+09:00",
    )
    assert spec["official"] is True
    assert spec["smoke_only"] is False
    assert spec["config_hash"] == config.config_hash
    assert spec["phase"] == "A"
    assert set(spec["marts"]) == set(REQUIRED_A0_MARTS)
    assert spec["python_version"]
    assert spec["run_id"].endswith(spec["phase_a_code_hash"][:8])


def test_build_run_spec_marks_smoke_family_run_as_non_official() -> None:
    config = load_config()
    manifest = _valid_manifest(config)
    spec = build_run_spec(
        config,
        manifest,
        snapshot_date="2026-08-01",
        source="sj2_remote",
        resolution_auto_selected=True,
        smoke_family="px_reversal_5d",
        permutation_repeats_override=None,
        include_holdout=False,
        holdout_start_override=None,
        repo_root=REPO_ROOT,
        code_paths=[Path(__file__)],
        command_line=["horizon_scan", "--phase", "A", "--smoke-family", "px_reversal_5d"],
        started_at="2026-08-02T00:00:00+09:00",
    )
    assert spec["official"] is False
    assert spec["smoke_only"] is True
    assert spec["smoke_only_reasons"] == ["smoke_family_filter"]


def test_write_run_spec_refuses_to_overwrite(tmp_path: Path) -> None:
    target = write_run_spec(tmp_path, {"a": 1})
    assert target.is_file()
    with pytest.raises(FileExistsError, match="immutable"):
        write_run_spec(tmp_path, {"a": 2})


def test_write_run_spec_serializes_a_real_config_derived_spec(tmp_path: Path) -> None:
    # quality_policy_version/label_policy_version embed raw YAML sections
    # verbatim, and PyYAML parses bare YYYY-MM-DD scalars (e.g.
    # sample.holdout_start, quality.price_limit_regimes[].start) as
    # datetime.date — this must not crash json.dumps.
    config = load_config()
    spec = build_run_spec(
        config,
        _valid_manifest(config),
        snapshot_date="2026-08-01",
        source="sj2_remote",
        resolution_auto_selected=True,
        smoke_family=None,
        permutation_repeats_override=None,
        include_holdout=False,
        holdout_start_override=None,
        repo_root=REPO_ROOT,
        code_paths=[Path(__file__)],
        command_line=["horizon_scan", "--phase", "A"],
        started_at="2026-08-02T00:00:00+09:00",
    )
    target = write_run_spec(tmp_path, spec)
    persisted = json.loads(target.read_text(encoding="utf-8"))
    assert persisted["label_policy_version"]["holdout_start"] == "2025-08-01"


def _seed_tmp_run_dir(tmp_path: Path, *, core_content: str = "ic,1.0\n") -> Path:
    tmp_run_dir = tmp_path / "run_id.tmp"
    (tmp_run_dir / "core").mkdir(parents=True)
    (tmp_run_dir / "plots").mkdir(parents=True)
    (tmp_run_dir / "run_spec.json").write_text('{"started_at": "t0"}', encoding="utf-8")
    (tmp_run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (tmp_run_dir / "core" / "horizon_ic.parquet").write_text(core_content, encoding="utf-8")
    (tmp_run_dir / "plots" / "family.png").write_text("binary-ish", encoding="utf-8")
    return tmp_run_dir


def test_compute_run_content_hash_ignores_provenance_but_not_core_output(tmp_path: Path) -> None:
    run_a = _seed_tmp_run_dir(tmp_path)
    hash_a = compute_run_content_hash(run_a)

    # a different run_spec.json (different started_at) must not change the hash
    (run_a / "run_spec.json").write_text('{"started_at": "t1-different"}', encoding="utf-8")
    assert compute_run_content_hash(run_a) == hash_a

    # a different plots/ PNG must not change the hash either
    (run_a / "plots" / "family.png").write_text("different-binary-ish", encoding="utf-8")
    assert compute_run_content_hash(run_a) == hash_a

    # but a change to the actual scan output must change it
    (run_a / "core" / "horizon_ic.parquet").write_text("ic,0.5\n", encoding="utf-8")
    assert compute_run_content_hash(run_a) != hash_a


def test_publish_run_renames_atomically_and_writes_success_last(tmp_path: Path) -> None:
    tmp_run_dir = _seed_tmp_run_dir(tmp_path)
    final_dir = tmp_path / "run_id"
    run_spec = {"run_id": "20260802-abcdef01", "config_hash": "cfg-hash"}

    published = publish_run(tmp_run_dir, final_dir, run_spec=run_spec)

    assert published == final_dir
    assert not tmp_run_dir.exists()  # rename, not copy
    assert (final_dir / "core" / "horizon_ic.parquet").is_file()
    success = json.loads((final_dir / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert success["status"] == "success"
    assert success["run_id"] == "20260802-abcdef01"
    assert success["config_hash"] == "cfg-hash"
    assert success["content_hash"] == compute_run_content_hash(final_dir)


def test_publish_run_rejects_incomplete_tmp_dir(tmp_path: Path) -> None:
    tmp_run_dir = tmp_path / "run_id.tmp"
    tmp_run_dir.mkdir()
    (tmp_run_dir / "run_spec.json").write_text("{}", encoding="utf-8")  # manifest.json missing
    with pytest.raises(RuntimeError, match="missing"):
        publish_run(tmp_run_dir, tmp_path / "run_id", run_spec={})


def test_publish_run_refuses_to_overwrite_an_existing_final_dir(tmp_path: Path) -> None:
    tmp_run_dir = _seed_tmp_run_dir(tmp_path)
    final_dir = tmp_path / "run_id"
    final_dir.mkdir()
    with pytest.raises(FileExistsError, match="immutable"):
        publish_run(tmp_run_dir, final_dir, run_spec={})


# --- Stage 0: _SUCCESS.json extras, REQUIRED_RUN_ARTIFACTS unchanged (§2.5) ---


def test_required_run_artifacts_still_names_only_the_pre_stage_0_pair() -> None:
    """A run published before Stage 0 has no ``daily_ic.parquet``; adding it to
    the required list would retroactively invalidate every one of them."""
    assert REQUIRED_RUN_ARTIFACTS == ("run_spec.json", "manifest.json")


def test_publish_run_records_the_daily_ic_reconciliation_in_success(tmp_path: Path) -> None:
    tmp_run_dir = _seed_tmp_run_dir(tmp_path)
    (tmp_run_dir / "core" / "daily_ic.parquet" / "family=fam").mkdir(parents=True)
    (tmp_run_dir / "core" / "daily_ic.parquet" / "family=fam" / "f.parquet").write_text(
        "daily", encoding="utf-8"
    )
    published = publish_run(
        tmp_run_dir,
        tmp_path / "run_id",
        run_spec={"run_id": "r", "config_hash": "c"},
        success_extra={"daily_ic_reconciled": True, "daily_ic_reconcile_max_abs_diff": 0.0},
    )
    success = json.loads((published / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert success["daily_ic_reconciled"] is True
    assert success["daily_ic_reconcile_max_abs_diff"] == 0.0
    assert success["content_hash"] == compute_run_content_hash(published)


def test_publish_run_omits_the_daily_ic_fields_when_no_sink_ran(tmp_path: Path) -> None:
    """A run directory with no ``daily_ic`` (an older published run, or a
    caller that never built a sink) still publishes — the manifest and
    _SUCCESS keys are additive, not required."""
    published = publish_run(
        _seed_tmp_run_dir(tmp_path), tmp_path / "run_id", run_spec={"run_id": "r"}
    )
    success = json.loads((published / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert "daily_ic_reconciled" not in success
