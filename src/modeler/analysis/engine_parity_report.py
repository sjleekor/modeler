"""Compare legacy/native Horizon Scan runs and publish a compact parity report."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype

REL_TOL = 1e-12


@dataclass(frozen=True)
class ArtifactResult:
    name: str
    rows: int
    columns: int
    exact_columns: int
    tolerant_columns: int
    max_scaled_delta: float
    passed: bool
    detail: str = ""


ARTIFACTS = (
    ("phase_a_horizon_ic", "A", "core/horizon_ic.parquet"),
    ("phase_a_permutation_cells", "A", "core/permutation_cell_stats.parquet"),
    ("phase_b_horizon_ic", "B", "core/horizon_ic.parquet"),
    ("phase_b_event_ic", "B", "core/event_ic.parquet"),
    ("phase_b_permutation_summary", "B", "core/permutation_summary.parquet"),
    ("phase_ab_primary", "AB", "combined_ab_primary_hypotheses.parquet"),
    ("phase_ab_overlay", "AB", "phase_a_card_overlay.parquet"),
)

# Stage 0's per-date series. Optional because every run published before it
# existed has none, and the harness still has to be able to compare those.
# Only the main scan writes it — the replicate loops take no sink — so this
# compares exactly the cells `phase_*_horizon_ic` summarizes, one row per date
# instead of one per cell.
OPTIONAL_ARTIFACTS = (
    ("phase_a_daily_ic", "A", "core/daily_ic.parquet"),
    ("phase_a_daily_spread", "A", "core/daily_spread.parquet"),
    ("phase_b_daily_ic", "B", "daily_ic.parquet"),
    ("phase_b_daily_spread", "B", "daily_spread.parquet"),
)


def _normalise_scalar(value: Any) -> str:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if value is None or pd.isna(value):
        return "<NULL>"
    return str(value)


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "replicate",
        "hypothesis_id",
        "family",
        "feature",
        "cell_type",
        "h_start",
        "h_end",
        "universe",
        "sample_kind",
        "offset",
    ]
    preferred = [*preferred, "trade_date"]  # daily_ic has many rows per cell
    keys = [column for column in preferred if column in frame.columns]
    if not keys:
        keys = list(frame.columns)
    sortable = frame.copy()
    for column in keys:
        if not is_numeric_dtype(sortable[column]):
            sortable[column] = sortable[column].map(_normalise_scalar)
    return sortable.sort_values(keys, kind="mergesort", na_position="first").reset_index(drop=True)


def _read_artifact(path: Path) -> pd.DataFrame:
    """Read one parquet file, or a whole hive-partitioned tree as one frame.

    ``daily_ic.parquet`` is a directory of ``family=<f>/<feature>.parquet``.
    The files are concatenated in sorted path order rather than handed to
    pandas as a dataset, so the ``family=`` directory never turns into a
    second, partition-derived ``family`` column beside the real one.
    """
    if path.is_dir():
        files = sorted(p for p in path.rglob("*.parquet") if p.is_file())
        if not files:
            raise FileNotFoundError(f"no parquet files under {path}")
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return pd.read_parquet(path)


def compare_parquet(name: str, legacy_path: Path, native_path: Path) -> ArtifactResult:
    legacy = _read_artifact(legacy_path)
    native = _read_artifact(native_path)
    if legacy.shape != native.shape:
        return ArtifactResult(
            name, len(legacy), len(legacy.columns), 0, 0, math.inf, False,
            f"shape mismatch: legacy={legacy.shape}, native={native.shape}",
        )
    if list(legacy.columns) != list(native.columns):
        return ArtifactResult(
            name, len(legacy), len(legacy.columns), 0, 0, math.inf, False,
            "column order/schema mismatch",
        )

    legacy = _sort_frame(legacy)
    native = _sort_frame(native)
    exact_columns = 0
    tolerant_columns = 0
    max_scaled_delta = 0.0
    failures: list[str] = []

    for column in legacy.columns:
        left = legacy[column]
        right = native[column]
        if is_float_dtype(left.dtype) and is_float_dtype(right.dtype):
            tolerant_columns += 1
            left_values = left.to_numpy(dtype=float)
            right_values = right.to_numpy(dtype=float)
            same_null = pd.isna(left_values) == pd.isna(right_values)
            if not bool(same_null.all()):
                failures.append(f"{column}: null mask mismatch")
                continue
            finite = ~(pd.isna(left_values) | pd.isna(right_values))
            if finite.any():
                scaled = abs(left_values[finite] - right_values[finite]) / pd.Series(
                    [max(1.0, abs(v)) for v in left_values[finite]]
                ).to_numpy()
                column_max = float(scaled.max())
                max_scaled_delta = max(max_scaled_delta, column_max)
                if column_max > REL_TOL:
                    failures.append(f"{column}: max scaled delta={column_max:.3e}")
            continue

        exact_columns += 1
        if is_integer_dtype(left.dtype) and is_integer_dtype(right.dtype):
            equal = left.equals(right)
        else:
            equal = left.map(_normalise_scalar).equals(right.map(_normalise_scalar))
        if not equal:
            failures.append(f"{column}: exact mismatch")

    return ArtifactResult(
        name=name,
        rows=len(legacy),
        columns=len(legacy.columns),
        exact_columns=exact_columns,
        tolerant_columns=tolerant_columns,
        max_scaled_delta=max_scaled_delta,
        passed=not failures,
        detail="; ".join(failures[:8]),
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_id(path: Path) -> str:
    return path.name.removeprefix("run_id=")


def _ab_manifest_summary(ab_run_dir: Path) -> dict[str, Any]:
    """AB 판정 요약. manifest.json에 없는 세 값은 published parquet에서 다시 센다.

    ``phase_b_primary_discovery_count``/``phase_a_primary_discovery_count``/
    ``phase_a_discovery_change_count``는 manifest 필드가 아니라
    ``combined_ab_primary_hypotheses.parquet``·``phase_a_card_overlay.parquet``의
    기존 boolean 컬럼(``primary_discovery_phase_b``/``primary_discovery_phase_a``/
    ``discovery_changed_vs_phase_a_only``)을 그대로 합산한 값이다.
    """
    manifest = _read_json(ab_run_dir / "manifest.json")
    primary = pd.read_parquet(ab_run_dir / "combined_ab_primary_hypotheses.parquet")
    overlay = pd.read_parquet(ab_run_dir / "phase_a_card_overlay.parquet")
    phase_b_rows = primary.loc[primary["role"] == "ready_primary"]
    combined_perm = manifest.get("combined_cross_sectional_permutation", {}) or {}
    return {
        "config_hash": manifest.get("config_hash"),
        "m_ab": manifest.get("m_ab"),
        "phase_b_screen_pass_count": manifest.get("phase_b_screen_pass_count"),
        "phase_b_evidence_grade_counts": manifest.get("phase_b_evidence_grade_counts"),
        "phase_b_primary_discovery_count": int(phase_b_rows["primary_discovery_phase_b"].sum()),
        "phase_a_primary_discovery_count": int(overlay["primary_discovery_phase_a"].sum()),
        "phase_a_discovery_change_count": int(overlay["discovery_changed_vs_phase_a_only"].sum()),
        "combined_cross_sectional_permutation_empirical_p": combined_perm.get("p_empirical_count"),
    }


def compare_ab_manifest(legacy_ab: Path, native_ab: Path) -> dict[str, Any]:
    legacy_summary = _ab_manifest_summary(legacy_ab)
    native_summary = _ab_manifest_summary(native_ab)
    mismatches: list[str] = []
    for key, legacy_value in legacy_summary.items():
        native_value = native_summary.get(key)
        if key == "combined_cross_sectional_permutation_empirical_p":
            if legacy_value is None or native_value is None:
                if legacy_value != native_value:
                    mismatches.append(f"{key}: legacy={legacy_value!r}, native={native_value!r}")
                continue
            scaled = abs(legacy_value - native_value) / max(1.0, abs(legacy_value))
            if scaled > REL_TOL:
                mismatches.append(
                    f"{key}: legacy={legacy_value!r}, native={native_value!r} "
                    f"(scaled delta={scaled:.3e})"
                )
            continue
        if legacy_value != native_value:
            mismatches.append(f"{key}: legacy={legacy_value!r}, native={native_value!r}")
    return {
        "legacy": legacy_summary,
        "native": native_summary,
        "matched": not mismatches,
        "mismatches": mismatches,
    }


def build_report(
    *,
    legacy_a: Path,
    legacy_b: Path,
    legacy_ab: Path,
    native_a: Path,
    native_b: Path,
    native_ab: Path,
) -> dict[str, Any]:
    legacy = {"A": legacy_a, "B": legacy_b, "AB": legacy_ab}
    native = {"A": native_a, "B": native_b, "AB": native_ab}
    results = [
        compare_parquet(name, legacy[phase] / relative, native[phase] / relative)
        for name, phase, relative in ARTIFACTS
    ]
    skipped: list[str] = []
    for name, phase, relative in OPTIONAL_ARTIFACTS:
        legacy_path = legacy[phase] / relative
        native_path = native[phase] / relative
        if not legacy_path.exists() and not native_path.exists():
            skipped.append(name)
            continue
        if legacy_path.exists() != native_path.exists():
            # One engine wrote it and the other did not — that is a real
            # asymmetry between the two runs, not a missing-input condition.
            detail = (
                f"present in only one run: legacy={legacy_path.exists()}, "
                f"native={native_path.exists()}"
            )
            results.append(ArtifactResult(name, 0, 0, 0, 0, math.inf, False, detail))
            continue
        results.append(compare_parquet(name, legacy_path, native_path))

    legacy_a_spec = _read_json(legacy_a / "run_spec.json")
    native_a_spec = _read_json(native_a / "run_spec.json")
    legacy_b_spec = _read_json(legacy_b / "phase_b_run_spec.json")
    native_b_spec = _read_json(native_b / "phase_b_run_spec.json")
    contract_keys = (
        "config_hash",
        "snapshot_date",
        "source",
        "a0_manifest_content_hash",
        "row_order_contract",
        "mapping_contract_version",
        "sue_nw_order_contract",
        "sue_permutation_order_contract",
    )
    contract_mismatches = []
    for key in contract_keys:
        for left, right, phase in (
            (legacy_a_spec, native_a_spec, "A"),
            (legacy_b_spec, native_b_spec, "B"),
        ):
            if left.get(key) != right.get(key):
                contract_mismatches.append(
                    f"{phase}.{key}: legacy={left.get(key)!r}, native={right.get(key)!r}"
                )

    a_cells = pd.read_parquet(legacy_a / "core/permutation_cell_stats.parquet")
    a_replicates = len(a_cells["replicate"].unique())
    b_replicates = len(pd.read_parquet(legacy_b / "core/permutation_summary.parquet"))
    sue = pd.read_parquet(legacy_b / "core/event_ic.parquet")
    sue = sue.loc[sue["family"] == "fin_sue"]
    sue_insufficient = bool(len(sue) == 6 and (sue["status"] == "insufficient").all())

    ab_manifest = compare_ab_manifest(legacy_ab, native_ab)

    passed = (
        not contract_mismatches
        and legacy_a_spec.get("scan_engine") == "legacy"
        and native_a_spec.get("scan_engine") == "polars_native_v1"
        and legacy_b_spec.get("scan_engine") == "legacy"
        and native_b_spec.get("scan_engine") == "polars_native_v1"
        and a_replicates == 100
        and b_replicates == 100
        and all(result.passed for result in results)
        and sue_insufficient
        and ab_manifest["matched"]
    )
    return {
        "schema_version": "horizon_scan_engine_parity_v1",
        "tolerance": {"scaled_absolute": REL_TOL},
        "runs": {
            "legacy": {phase: _run_id(path) for phase, path in legacy.items()},
            "native": {phase: _run_id(path) for phase, path in native.items()},
        },
        "replicates": {"phase_a": a_replicates, "phase_b_joint": b_replicates},
        "contracts_match": not contract_mismatches,
        "contract_mismatches": contract_mismatches,
        "sue": {
            "cells": int(len(sue)),
            "all_insufficient": sue_insufficient,
            "joint_null_contribution": "none" if sue_insufficient else "present",
        },
        "artifacts": [asdict(result) for result in results],
        "skipped_artifacts": skipped,
        "ab_manifest": ab_manifest,
        "passed": passed,
    }


def render_markdown(report: dict[str, Any]) -> str:
    status = "통과" if report["passed"] else "실패"
    lines = [
        "# Horizon Scan legacy/native full parity",
        "",
        f"- 판정: **{status}**",
        f"- tolerance: `{report['tolerance']['scaled_absolute']}`",
        f"- Phase A replicate: `{report['replicates']['phase_a']}`",
        f"- Phase B joint replicate: `{report['replicates']['phase_b_joint']}`",
        f"- legacy run: `{report['runs']['legacy']}`",
        f"- native run: `{report['runs']['native']}`",
        "",
        "| artifact | rows | max scaled delta | 판정 |",
        "|---|---:|---:|---|",
    ]
    for artifact in report["artifacts"]:
        lines.append(
            f"| `{artifact['name']}` | {artifact['rows']} | "
            f"{artifact['max_scaled_delta']:.3e} | "
            f"{'통과' if artifact['passed'] else '실패'} |"
        )
    if report.get("skipped_artifacts"):
        lines.extend(
            [
                "",
                "두 run 모두 다음 artifact가 없어 비교하지 않았습니다 (Stage 0 이전 run): "
                + ", ".join(f"`{name}`" for name in report["skipped_artifacts"]),
            ]
        )
    lines.extend(
        [
            "",
            "## SUE sorted-v2",
            "",
            f"real SUE cell은 `{report['sue']['cells']}`개이며 모두 "
            f"`{'insufficient' if report['sue']['all_insufficient'] else 'evaluable'}`입니다. "
            "따라서 이번 snapshot에서 SUE가 joint null에 더한 유효 row는 없습니다.",
            "",
        ]
    )
    if report["contract_mismatches"]:
        lines.extend(["## Contract mismatch", ""])
        lines.extend(f"- {item}" for item in report["contract_mismatches"])
        lines.append("")

    ab_manifest = report["ab_manifest"]
    ab_status = "통과" if ab_manifest["matched"] else "실패"
    lines.extend(
        [
            "## AB manifest 판정 요약",
            "",
            f"- 판정: **{ab_status}**",
            "",
            "| 항목 | legacy | native |",
            "|---|---:|---:|",
        ]
    )
    for key, legacy_value in ab_manifest["legacy"].items():
        native_value = ab_manifest["native"].get(key)
        lines.append(f"| `{key}` | {legacy_value} | {native_value} |")
    lines.append("")
    if ab_manifest["mismatches"]:
        lines.extend(["### AB manifest mismatch", ""])
        lines.extend(f"- {item}" for item in ab_manifest["mismatches"])
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for engine in ("legacy", "native"):
        for phase in ("a", "b", "ab"):
            parser.add_argument(f"--{engine}-{phase}", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        legacy_a=args.legacy_a,
        legacy_b=args.legacy_b,
        legacy_ab=args.legacy_ab,
        native_a=args.native_a,
        native_b=args.native_b,
        native_ab=args.native_ab,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render_markdown(report), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
