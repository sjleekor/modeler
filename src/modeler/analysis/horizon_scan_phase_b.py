"""Phase B (financial/event) contract — B-0 preflight, candidate registry,
outcome-blind readiness freeze, and the receipt-value pairing hard gate.

See ``my/milestones/kr/modeling/dev/20260731_raw_features/01_feature_candidate/
04_specific_plan_B.md`` §1 (entry/exit conditions), §2 (pre-registration contract), §6 B-0.

This module never reads a label, return, IC, or p-value column (§1.1 rule 6,
§6 B-0 test list "readiness 단계가 label/return/IC/p-value column을 읽으면
실패") — every function signature here only takes config, raw table/mart
*names*, and a DuckDB connection used solely to enumerate keys, never to join
against a label mart. The B-2..B-9 marts referenced by each family's
``readiness_dependencies`` (``feat_fin_scan_daily``, ``feat_event_scan_daily``,
``fin_sue_event``, ...) are not implemented in this codebase yet, so every
candidate legitimately freezes as ``blocked_exploratory`` until those land —
that is the correct outcome-blind state, not a bug in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

from modeler.analysis.horizon_scan_config import (
    DEFAULT_SCAN_ENGINE,
    HorizonScanConfig,
    bucket_primary_cells,
)
from modeler.analysis.horizon_scan_run_spec import (
    PreflightError,
    analysis_kernel_hash,
    git_metadata,
    package_versions,
    phase_a_code_hash,
)
from modeler.etl.features.event_scan import (
    EVENT_FEATURE_FORMULA_VERSION,
    PAYOUT_FEATURE_FORMULA_VERSION,
)
from modeler.etl.features.fin_scan import FIN_FEATURE_FORMULA_VERSION

PHASE_B_MAX_CANDIDATES = 38


def phase_b_families(config: HorizonScanConfig) -> list[dict[str, Any]]:
    """Return the 8 registered Phase B families, in config order."""
    return [f for f in config.families if f.get("phase") == "B"]


def phase_b_candidate_cells(config: HorizonScanConfig) -> list[dict[str, Any]]:
    """One row per §2.1 candidate cell — continuous cumulative/bucket cells for
    the 7 financial/issuance/payout families, event-bucket cells for SUE.
    """
    cells: list[dict[str, Any]] = []
    for family in phase_b_families(config):
        primary_feature = next(
            feature["column"] for feature in family["features"] if feature["role"] == "primary"
        )
        common = {
            "family": family["family"],
            "fdr_family": family.get("fdr_family"),
            "feature": primary_feature,
            "expected_sign": family.get("expected_sign"),
            "readiness_dependencies": tuple(family["readiness_dependencies"]),
        }
        if "event_buckets" in family:
            for b_start, b_end in family["event_buckets"]:
                cells.append(
                    {
                        **common,
                        "hypothesis_id": (
                            f"{family['family']}|{primary_feature}|event|{b_start}|{b_end}"
                        ),
                        "cell_type": "event_bucket",
                        "h_start": b_start,
                        "h_end": b_end,
                    }
                )
            continue
        for h in family["primary_horizon_set"]:
            cells.append(
                {
                    **common,
                    "hypothesis_id": f"{family['family']}|{primary_feature}|cum|0|{h}",
                    "cell_type": "cumulative",
                    "h_start": 0,
                    "h_end": h,
                }
            )
        for b_start, b_end in bucket_primary_cells(family, config.buckets):
            hypothesis_id = f"{family['family']}|{primary_feature}|bucket|{b_start}|{b_end}"
            cells.append(
                {
                    **common,
                    "hypothesis_id": hypothesis_id,
                    "cell_type": "bucket",
                    "h_start": b_start,
                    "h_end": b_end,
                }
            )
    ids = [cell["hypothesis_id"] for cell in cells]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"Phase B candidate registry has duplicate ids: {dupes}")
    return cells


def build_phase_b_candidate_registry(config: HorizonScanConfig) -> list[dict[str, Any]]:
    """Build exactly the config-frozen number of Phase B candidate cells.

    Mirrors ``build_primary_hypothesis_registry``'s defense-in-depth pattern —
    ``validate_config`` already checks this count at load time; this re-checks
    it against the *cells actually enumerated*, so a registry-building bug
    (not a config bug) still fails loudly.
    """
    cells = phase_b_candidate_cells(config)
    expected = int(config.raw["phase_b"]["primary_candidate_count_max"])
    if len(cells) != expected:
        raise ValueError(
            f"Phase B candidate registry must have {expected} cells, "
            f"got {len(cells)}"
        )
    return cells


def assert_required_raw_tables_present(config: HorizonScanConfig, raw_tables: set[str]) -> None:
    """§1.1 condition 3 — the pre-existing A0 raw dependency must be complete.

    The two *new* Phase B raw tables (``dart_filing_receipt_raw``,
    ``dart_capital_change_raw``) are deliberately excluded from this hard
    gate: their absence blocks only the specific candidates that depend on
    them via ``readiness_dependencies``, not the whole Phase B run.
    """
    required = set(config.raw["phase_b"]["required_raw_tables"])
    missing = sorted(required - raw_tables)
    if missing:
        raise PreflightError(f"Phase B raw manifest missing required tables: {missing}")


def build_phase_b_readiness_rows(
    config: HorizonScanConfig,
    *,
    available_assets: set[str],
) -> list[dict[str, Any]]:
    """One row per §2.1 candidate cell, role/status decided purely from
    dependency *presence* — never from a label, return, IC, or p-value.

    ``available_assets`` is the union of raw-table names with data and
    mart/view names that exist (e.g. from ``DESCRIBE`` on a live DuckDB
    connection). A cell is ``ready_primary`` only if every name in its
    family's ``readiness_dependencies`` is in this set.
    """
    rows: list[dict[str, Any]] = []
    for cell in build_phase_b_candidate_registry(config):
        missing = [dep for dep in cell["readiness_dependencies"] if dep not in available_assets]
        ready = not missing
        rows.append(
            {
                **cell,
                "role": "ready_primary" if ready else "blocked_exploratory",
                "status": "ready" if ready else "blocked_missing_dependency",
                "missing_dependencies": ",".join(missing),
            }
        )
    return rows


def summarize_phase_b_readiness(
    config: HorizonScanConfig, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """§2.3: ``M_B_ready`` and the combined ``M_AB = 75 + M_B_ready`` count."""
    ready_primary = [r for r in rows if r["role"] == "ready_primary"]
    blocked = [r for r in rows if r["role"] == "blocked_exploratory"]
    expected = int(config.raw["phase_b"]["primary_candidate_count_max"])
    if len(ready_primary) + len(blocked) != expected:
        raise ValueError(
            "ready_primary + blocked_exploratory must equal the frozen candidate count "
            f"({expected}), got {len(ready_primary) + len(blocked)}"
        )
    phase_a_count = config.raw["phase_b"]["phase_a_primary_count"]
    return {
        "m_b_ready": len(ready_primary),
        "blocked_exploratory_count": len(blocked),
        "phase_a_primary_count": phase_a_count,
        "combined_ab_hypothesis_count": phase_a_count + len(ready_primary),
    }


def write_phase_b_readiness_freeze(
    config: HorizonScanConfig,
    rows: list[dict[str, Any]],
    output_path: Path,
    *,
    generated_at: str,
) -> Path:
    """Atomic, immutable write of ``phase_b_readiness_freeze.json`` (§6 B-0.6).

    Once written for a given ``config_hash``, the freeze does not change —
    reflecting source/formula coverage improving requires a new config
    version and a fresh freeze, never an in-place edit (§6 B-0, §2.3).
    """
    summary = summarize_phase_b_readiness(config, rows)
    payload = {
        "config_hash": config.config_hash,
        "generated_at": generated_at,
        "max_candidates": int(config.raw["phase_b"]["primary_candidate_count_max"]),
        **summary,
        "cells": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"phase_b_readiness_freeze.json already exists and is immutable: {output_path}"
        )
    temp = output_path.with_suffix(output_path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(output_path)
    return output_path


def build_phase_b_run_spec(
    config: HorizonScanConfig,
    *,
    a0_run_spec: dict[str, Any],
    raw_manifest_tables: set[str],
    repo_root: Path,
    code_paths: list[Path],
    command_line: list[str],
    started_at: str,
    finished_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the immutable ``phase_b_run_spec.json`` payload (§1.3, §6 B-0.6).

    Phase B inherits A0's fingerprint fields from ``a0_run_spec`` rather than
    re-resolving them (§1.1: "A0에서 선택한 snapshot_date, source ... 를 그대로
    사용한다"). Does not require a Phase A run to exist yet (§1.1: "Phase B
    core scan 자체는 Phase A 실행 완료 전에도 계산할 수 있다") — combining with
    Phase A's raw p-values happens later, at B-9.
    """
    assert_required_raw_tables_present(config, raw_manifest_tables)
    build_phase_b_candidate_registry(config)
    code_hash = phase_a_code_hash(code_paths)  # generic file-list hasher, reused across phases
    git = git_metadata(repo_root)
    return {
        "phase": "B",
        "snapshot_date": a0_run_spec["snapshot_date"],
        "source": a0_run_spec["source"],
        "raw_manifest_hash": a0_run_spec["raw_manifest_hash"],
        "a0_manifest_hash": a0_run_spec["a0_manifest_hash"],
        "config_schema_version": config.raw["schema_version"],
        "config_hash": config.config_hash,
        "holdout_start": a0_run_spec["label_policy_version"]["holdout_start"],
        "quality_policy_version": a0_run_spec["quality_policy_version"],
        "universe_policy_version": a0_run_spec["universe_policy_version"],
        # §1.3 fingerprint: "mapping rule ... 이 다르면 동일 snapshot의 기존
        # Phase B artifact를 재사용하지 않는다". The isu_dcrs_stle catalog is
        # that mapping rule for the event features; the ratio/winsorize/fs_basis
        # rules are the equivalent for the financial ones. Neither is covered by
        # config_hash (scan YAML) or phase_b_code_hash (analysis modules only).
        "event_feature_formula_version": EVENT_FEATURE_FORMULA_VERSION,
        "payout_feature_formula_version": PAYOUT_FEATURE_FORMULA_VERSION,
        "fin_feature_formula_version": FIN_FEATURE_FORMULA_VERSION,
        "phase_b_code_hash": code_hash,
        "analysis_kernel_hash": analysis_kernel_hash(repo_root),
        "scan_engine": DEFAULT_SCAN_ENGINE,
        "row_order_contract": config.raw.get("execution", {}).get(
            "row_order_contract", "legacy_input_order"
        ),
        "sue_nw_order_contract": config.raw.get("execution", {}).get(
            "sue_nw_order_contract", "legacy"
        ),
        "sue_permutation_order_contract": config.raw.get("execution", {}).get(
            "sue_permutation_order_contract", "legacy"
        ),
        "mapping_contract_version": config.raw.get("execution", {}).get(
            "mapping_contract_version", "v1"
        ),
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
        "run_id": f"{started_at[:19].replace(':', '').replace('-', '')}-{code_hash[:8]}",
        **package_versions(),
        "command_line": command_line,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def write_phase_b_run_spec(run_dir: Path, run_spec: dict[str, Any]) -> Path:
    """Write ``phase_b_run_spec.json`` once; refuses to overwrite an existing one."""
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "phase_b_run_spec.json"
    if target.exists():
        raise FileExistsError(f"phase_b_run_spec.json already exists and is immutable: {target}")
    temp = run_dir / "phase_b_run_spec.json.tmp"
    temp.write_text(
        json.dumps(run_spec, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(target)
    return target


# The rendered report carries this run's timestamps, so hashing it would make
# two byte-identical scans look different. Excluding a name that an older run
# never wrote is a no-op, so previously published runs still verify.
PHASE_B_CONTENT_HASH_EXCLUDE_NAMES = frozenset(
    {
        "phase_b_run_spec.json",
        "_SUCCESS.json",
        "phase_b_readiness_freeze.json",
        "03b_horizon_scan_results.md",
        "timings.json",
    }
)


@dataclass(frozen=True, slots=True)
class ReceiptValuePairingReport:
    """§1.2/§3.5 receipt-value pairing preflight result.

    This only proves the *structural* precondition the whole "single
    captured receipt, next-session availability" design depends on: no
    filing key in the currently collected raw has more than one distinct
    ``rcept_no`` (i.e. no vintage history has silently been collapsed
    already). It does not perform the numeric period/basis/unit/value
    tolerance comparison against XBRL — that join needs the mapping-rule
    logic B-2 (``stock_metric_vintage_fact``) builds, not raw tables alone.
    """

    financial_key_count: int
    xbrl_covered_key_count: int
    coverage_ratio: float
    multi_receipt_filing_keys: tuple[tuple[str, int, str], ...]
    verified_same_receipt: bool


def check_receipt_value_pairing(con: duckdb.DuckDBPyConnection) -> ReceiptValuePairingReport:
    """Query ``dart_financial_statement_raw``/``dart_xbrl_fact_raw`` for the
    single-captured-receipt invariant and XBRL coverage ratio.

    ``con`` must have both tables registered/available; no label, return, or
    price data is touched.
    """
    financial_rows = con.execute(
        "SELECT DISTINCT corp_code, bsns_year, reprt_code, rcept_no "
        "FROM dart_financial_statement_raw"
    ).fetchall()
    xbrl_rows = con.execute(
        "SELECT DISTINCT corp_code, bsns_year, reprt_code, rcept_no FROM dart_xbrl_fact_raw"
    ).fetchall()
    multi_rows = con.execute(
        "SELECT corp_code, bsns_year, reprt_code "
        "FROM dart_financial_statement_raw "
        "GROUP BY corp_code, bsns_year, reprt_code "
        "HAVING COUNT(DISTINCT rcept_no) > 1 "
        "ORDER BY corp_code, bsns_year, reprt_code"
    ).fetchall()

    financial_keys = {tuple(row) for row in financial_rows}
    xbrl_keys = {tuple(row) for row in xbrl_rows}
    covered = financial_keys & xbrl_keys
    coverage_ratio = (len(covered) / len(financial_keys)) if financial_keys else 0.0

    return ReceiptValuePairingReport(
        financial_key_count=len(financial_keys),
        xbrl_covered_key_count=len(covered),
        coverage_ratio=coverage_ratio,
        multi_receipt_filing_keys=tuple((row[0], row[1], row[2]) for row in multi_rows),
        verified_same_receipt=len(multi_rows) == 0,
    )


def assert_receipt_value_pairing_verified(report: ReceiptValuePairingReport) -> None:
    """§1.2 hard gate: any filing key backed by more than one distinct
    ``rcept_no`` in current raw contradicts the single-captured-receipt
    assumption the continuous-feature availability policy depends on, and
    blocks the whole official Phase B run — not just the affected candidates.
    """
    if not report.verified_same_receipt:
        raise PreflightError(
            f"receipt-value pairing failed: {len(report.multi_receipt_filing_keys)} filing "
            "key(s) have more than one distinct rcept_no in dart_financial_statement_raw "
            f"(sample: {report.multi_receipt_filing_keys[:10]})"
        )
