from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from modeler.analysis.engine_parity_report import ARTIFACTS, build_report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_run_tree(root: Path, *, engine: str, delta: float = 0.0) -> dict[str, Path]:
    runs = {phase: root / f"run_id={engine}-{phase.lower()}" for phase in ("A", "B", "AB")}
    contract = {
        "config_hash": "config",
        "snapshot_date": "2026-08-23",
        "source": "sj2_remote",
        "a0_manifest_content_hash": "a0",
        "row_order_contract": "canonical",
        "mapping_contract_version": "joint_cs_v2",
        "sue_nw_order_contract": "sue_nw_sorted_v2",
        "sue_permutation_order_contract": "sue_rank_canonical_v2",
        "scan_engine": engine,
    }
    _write_json(runs["A"] / "run_spec.json", contract)
    _write_json(runs["B"] / "phase_b_run_spec.json", contract)
    _write_json(
        runs["AB"] / "manifest.json",
        {
            "config_hash": "config",
            "m_ab": 2,
            "phase_b_screen_pass_count": 1,
            "phase_b_evidence_grade_counts": {"A": 1},
            "combined_cross_sectional_permutation": {
                "real_discovery_count": 1,
                "n_replicates": 100,
                "p_empirical_count": 1 / 101,
            },
        },
    )

    for name, phase, relative in ARTIFACTS:
        path = runs[phase] / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "phase_a_permutation_cells":
            frame = pd.DataFrame(
                {
                    "replicate": list(range(100)),
                    "hypothesis_id": ["a"] * 100,
                    "ic_mean": [0.1 + delta] * 100,
                    "passed": [True] * 100,
                }
            )
        elif name == "phase_b_permutation_summary":
            frame = pd.DataFrame(
                {
                    "replicate": list(range(100)),
                    "seed": list(range(100)),
                    "mapping_hash": ["hash"] * 100,
                    "n_discoveries": [1] * 100,
                }
            )
        elif name == "phase_b_event_ic":
            frame = pd.DataFrame(
                {
                    "family": ["fin_sue"] * 6,
                    "hypothesis_id": [f"sue-{index}" for index in range(6)],
                    "status": ["insufficient"] * 6,
                    "ic_mean": [float("nan")] * 6,
                }
            )
        elif name == "phase_ab_primary":
            frame = pd.DataFrame(
                {
                    "hypothesis_id": ["a-primary", "b-primary"],
                    "role": [None, "ready_primary"],
                    "status": ["valid", "valid"],
                    "ic_mean": [0.1 + delta, 0.2 + delta],
                    "decision": [True, True],
                    "primary_discovery_phase_b": [None, True],
                }
            )
        elif name == "phase_ab_overlay":
            frame = pd.DataFrame(
                {
                    "hypothesis_id": ["a-primary"],
                    "status": ["valid"],
                    "ic_mean": [0.1 + delta],
                    "decision": [True],
                    "primary_discovery_phase_a": [True],
                    "discovery_changed_vs_phase_a_only": [False],
                }
            )
        else:
            frame = pd.DataFrame(
                {
                    "hypothesis_id": [name],
                    "status": ["valid"],
                    "ic_mean": [0.1 + delta],
                    "decision": [True],
                }
            )
        frame.to_parquet(path, index=False)
    return runs


def test_build_report_accepts_tolerant_float_delta_and_exact_decisions(tmp_path: Path) -> None:
    legacy = _seed_run_tree(tmp_path / "legacy", engine="legacy")
    native = _seed_run_tree(
        tmp_path / "native", engine="polars_native_v1", delta=5e-13
    )

    report = build_report(
        legacy_a=legacy["A"],
        legacy_b=legacy["B"],
        legacy_ab=legacy["AB"],
        native_a=native["A"],
        native_b=native["B"],
        native_ab=native["AB"],
    )

    assert report["passed"] is True
    assert report["replicates"] == {"phase_a": 100, "phase_b_joint": 100}
    assert report["sue"] == {
        "cells": 6,
        "all_insufficient": True,
        "joint_null_contribution": "none",
    }
    ab_manifest = report["ab_manifest"]
    assert ab_manifest["matched"] is True
    assert ab_manifest["mismatches"] == []
    assert ab_manifest["legacy"]["m_ab"] == 2
    assert ab_manifest["legacy"]["phase_b_primary_discovery_count"] == 1
    assert ab_manifest["legacy"]["phase_a_primary_discovery_count"] == 1
    assert ab_manifest["legacy"]["phase_a_discovery_change_count"] == 0


def test_build_report_flags_ab_manifest_mismatch(tmp_path: Path) -> None:
    legacy = _seed_run_tree(tmp_path / "legacy", engine="legacy")
    native = _seed_run_tree(tmp_path / "native", engine="polars_native_v1")

    native_manifest_path = native["AB"] / "manifest.json"
    manifest = json.loads(native_manifest_path.read_text(encoding="utf-8"))
    manifest["phase_b_screen_pass_count"] = manifest["phase_b_screen_pass_count"] + 1
    _write_json(native_manifest_path, manifest)

    report = build_report(
        legacy_a=legacy["A"],
        legacy_b=legacy["B"],
        legacy_ab=legacy["AB"],
        native_a=native["A"],
        native_b=native["B"],
        native_ab=native["AB"],
    )

    assert report["passed"] is False
    assert report["ab_manifest"]["matched"] is False
    assert any(
        "phase_b_screen_pass_count" in item for item in report["ab_manifest"]["mismatches"]
    )


# --- Stage 0 daily_ic parity (01_stage0 §4.3) ---


def _seed_daily_ic(run_dir: Path, relative: str, *, delta: float = 0.0) -> None:
    """A hive-partitioned ``daily_ic.parquet`` tree, as the sink writes it."""
    frame = pd.DataFrame(
        {
            "hypothesis_id": ["fam|f|cum|0|20"] * 4,
            "family": ["fam"] * 4,
            "feature": ["f"] * 4,
            "scan_type": ["cum"] * 4,
            "h_start": [0] * 4,
            "h_end": [20] * 4,
            "universe": ["broad"] * 4,
            "sample_kind": ["common_survivor"] * 4,
            "hypothesis_role": ["primary"] * 4,
            "trade_date": pd.to_datetime(
                ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
            ).date,
            "formation_session_idx": [1, 2, 3, 4],
            "rank_ic": [0.01 + delta, 0.02 + delta, 0.03 + delta, 0.04 + delta],
            "n_obs": [500] * 4,
        }
    )
    target = run_dir / relative / "family=fam"
    target.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target / "f.parquet", index=False)


def test_daily_ic_is_skipped_when_neither_run_has_it(tmp_path: Path) -> None:
    """Every run published before Stage 0 has none. The harness still has to be
    able to compare those, so their absence is a skip, not a failure."""
    legacy = _seed_run_tree(tmp_path / "legacy", engine="legacy")
    native = _seed_run_tree(tmp_path / "native", engine="polars_native_v1")

    report = build_report(
        legacy_a=legacy["A"],
        legacy_b=legacy["B"],
        legacy_ab=legacy["AB"],
        native_a=native["A"],
        native_b=native["B"],
        native_ab=native["AB"],
    )
    assert report["passed"] is True
    assert set(report["skipped_artifacts"]) == {
        "phase_a_daily_ic",
        "phase_a_daily_spread",
        "phase_b_daily_ic",
        "phase_b_daily_spread",
    }
    assert "phase_a_daily_ic" not in {a["name"] for a in report["artifacts"]}


def test_matching_daily_ic_trees_pass_within_tolerance(tmp_path: Path) -> None:
    legacy = _seed_run_tree(tmp_path / "legacy", engine="legacy")
    native = _seed_run_tree(tmp_path / "native", engine="polars_native_v1", delta=5e-13)
    _seed_daily_ic(legacy["A"], "core/daily_ic.parquet")
    _seed_daily_ic(native["A"], "core/daily_ic.parquet", delta=5e-13)

    report = build_report(
        legacy_a=legacy["A"],
        legacy_b=legacy["B"],
        legacy_ab=legacy["AB"],
        native_a=native["A"],
        native_b=native["B"],
        native_ab=native["AB"],
    )
    by_name = {a["name"]: a for a in report["artifacts"]}
    assert by_name["phase_a_daily_ic"]["passed"] is True
    assert by_name["phase_a_daily_ic"]["rows"] == 4
    assert by_name["phase_a_daily_ic"]["max_scaled_delta"] < 1e-12
    assert "phase_a_daily_ic" not in report["skipped_artifacts"]
    assert report["passed"] is True


def test_a_daily_ic_difference_beyond_tolerance_fails_the_parity(tmp_path: Path) -> None:
    legacy = _seed_run_tree(tmp_path / "legacy", engine="legacy")
    native = _seed_run_tree(tmp_path / "native", engine="polars_native_v1")
    _seed_daily_ic(legacy["A"], "core/daily_ic.parquet")
    _seed_daily_ic(native["A"], "core/daily_ic.parquet", delta=1e-6)

    report = build_report(
        legacy_a=legacy["A"],
        legacy_b=legacy["B"],
        legacy_ab=legacy["AB"],
        native_a=native["A"],
        native_b=native["B"],
        native_ab=native["AB"],
    )
    by_name = {a["name"]: a for a in report["artifacts"]}
    assert by_name["phase_a_daily_ic"]["passed"] is False
    assert report["passed"] is False


def test_daily_ic_present_in_only_one_run_is_an_asymmetry_not_a_skip(tmp_path: Path) -> None:
    legacy = _seed_run_tree(tmp_path / "legacy", engine="legacy")
    native = _seed_run_tree(tmp_path / "native", engine="polars_native_v1")
    _seed_daily_ic(legacy["A"], "core/daily_ic.parquet")

    report = build_report(
        legacy_a=legacy["A"],
        legacy_b=legacy["B"],
        legacy_ab=legacy["AB"],
        native_a=native["A"],
        native_b=native["B"],
        native_ab=native["AB"],
    )
    by_name = {a["name"]: a for a in report["artifacts"]}
    assert by_name["phase_a_daily_ic"]["passed"] is False
    assert "only one run" in by_name["phase_a_daily_ic"]["detail"]
    assert report["passed"] is False


def test_the_markdown_names_what_it_skipped(tmp_path: Path) -> None:
    from modeler.analysis.engine_parity_report import render_markdown

    legacy = _seed_run_tree(tmp_path / "legacy", engine="legacy")
    native = _seed_run_tree(tmp_path / "native", engine="polars_native_v1")
    report = build_report(
        legacy_a=legacy["A"],
        legacy_b=legacy["B"],
        legacy_ab=legacy["AB"],
        native_a=native["A"],
        native_b=native["B"],
        native_ab=native["AB"],
    )
    markdown = render_markdown(report)
    assert "Stage 0 이전 run" in markdown
    assert "`phase_a_daily_ic`" in markdown
