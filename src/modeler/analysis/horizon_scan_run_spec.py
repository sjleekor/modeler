"""Phase A run contract: preflight gate and the immutable ``run_spec.json`` (A-0).

See ``my/milestones/kr/modeling/dev/20260731_raw_features/01_feature_candidate/
04_specific_plan_A.md`` §1, §1.2, §5 (A-0). ``run_spec.json`` is the first artifact written for an
official run and no later stage may change it; a debug/smoke invocation still
produces one (for reproducibility of the debug run itself) but is marked
``official=false`` and never accompanied by a final ``_SUCCESS.json``.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import polars as pl

from modeler.analysis.horizon_scan_config import DEFAULT_SCAN_ENGINE, HorizonScanConfig
from modeler.analysis.horizon_scan_readiness import (
    build_primary_hypothesis_registry,
    build_short_exploratory_registry,
)

REQUIRED_A0_MARTS = (
    "dim_stock_pit_daily",
    "dim_price_quality_daily",
    "dim_universe_broad_daily",
    "dim_universe_tradable_daily",
    "feat_price",
    "feat_flow",
    "label_scan",
)
KST = ZoneInfo("Asia/Seoul")


class PreflightError(RuntimeError):
    """An official Phase A entry condition (§1.1) is not met."""


def assert_a0_manifest_matches(config: HorizonScanConfig, manifest: dict[str, Any]) -> None:
    """§1.1 conditions 1-3: a successful, non-smoke A0 run at this config_hash."""
    if manifest.get("status") != "success":
        raise PreflightError(f"A0 manifest status is not success: {manifest.get('status')!r}")
    if manifest.get("config_hash") != config.config_hash:
        raise PreflightError(
            "A0 manifest config_hash does not match the Phase A config; rerun "
            "`uv run python -m modeler.etl.horizon_scan_inputs --force` first"
        )
    if manifest.get("smoke_only"):
        raise PreflightError("A0 manifest is smoke_only=true; a full A0 run is required")
    marts_by_view = {m["view"]: m for m in manifest.get("marts", [])}
    absent = [name for name in REQUIRED_A0_MARTS if name not in marts_by_view]
    empty = [
        name for name in REQUIRED_A0_MARTS if marts_by_view.get(name, {}).get("row_count", 0) <= 0
    ]
    if absent or empty:
        raise PreflightError(f"A0 marts missing or empty: absent={absent} empty={empty}")


def assert_family_registry_complete(config: HorizonScanConfig) -> None:
    """§1.1 condition 4: 17 Phase A families split 12 ready / 1 reference / 4 short."""
    phase_a = [f for f in config.families if f.get("phase") == "A"]
    if len(phase_a) != 17:
        raise PreflightError(f"Phase A must register 17 families, found {len(phase_a)}")
    counts = {
        "ready": sum(1 for f in phase_a if f.get("role") == "ready"),
        "reference_only": sum(1 for f in phase_a if f.get("role") == "reference_only"),
        "exploratory_short_regime": sum(
            1 for f in phase_a if f.get("role") == "exploratory_short_regime"
        ),
    }
    expected = {"ready": 12, "reference_only": 1, "exploratory_short_regime": 4}
    if counts != expected:
        raise PreflightError(f"Phase A family role counts must be {expected}, found {counts}")


def assert_holdout_untouched(*, include_holdout: bool, holdout_start_override: str | None) -> None:
    """§1.1: `--include-holdout`/`--holdout-start` are debug-only, never official."""
    if include_holdout or holdout_start_override is not None:
        raise PreflightError(
            "--include-holdout/--holdout-start override the holdout boundary and are "
            "not permitted in an official Phase A run"
        )


def run_preflight_checks(config: HorizonScanConfig, manifest: dict[str, Any]) -> None:
    """Run every §1.1 entry-condition check; raises on the first violation.

    Also validates (via the registry builders) that the primary/short-exploratory
    hypothesis counts are exactly 75/28 before any scan stage starts.
    """
    assert_a0_manifest_matches(config, manifest)
    assert_family_registry_complete(config)
    build_primary_hypothesis_registry(config)
    build_short_exploratory_registry(config)


def determine_official_mode(
    *,
    resolution_auto_selected: bool,
    smoke_family: str | None,
    permutation_repeats_override: int | None,
    config_permutation_repeats: int,
) -> tuple[bool, bool, list[str]]:
    """Return ``(official, smoke_only, reasons)`` per §1.1's debug-override list.

    A manually-pinned snapshot, a ``--smoke-family`` filter, or a permutation
    count that differs from the preregistered config all force ``official=false``
    — availability is never chosen after peeking at which override looks best.
    """
    reasons: list[str] = []
    if not resolution_auto_selected:
        reasons.append("snapshot_manually_overridden")
    if smoke_family is not None:
        reasons.append("smoke_family_filter")
    if (
        permutation_repeats_override is not None
        and permutation_repeats_override != config_permutation_repeats
    ):
        reasons.append("permutation_repeats_override")
    smoke_only = bool(reasons)
    return not smoke_only, smoke_only, reasons


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        return subprocess.run(
            args, cwd=repo_root, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def package_versions() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "duckdb_version": duckdb.__version__,
        "polars_version": pl.__version__,
        "numpy_version": np.__version__,
    }


def phase_a_code_hash(code_paths: list[Path]) -> str:
    """Stable hash of the scan driver/stats/report source and config content.

    Two runs of identical snapshot/config but different code must not share a
    result directory (§1.2): the run_id embeds this hash's first 8 hex chars.
    """
    hasher = hashlib.sha256()
    for path in sorted(code_paths):
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def analysis_kernel_paths(repo_root: Path) -> list[Path]:
    """Shared source list for Phase A/B kernel lineage.

    The old phase-specific globs omitted ``research/etl/metrics.py`` and made
    the Phase B scope narrower than the shared runner.  This explicit list is
    used by both phases and includes newly added ``horizon_scan_*.py`` files.
    """
    analysis_dir = repo_root / "src" / "modeler" / "analysis"
    return [
        repo_root / "src" / "modeler" / "etl" / "metrics.py",
        *sorted(analysis_dir.glob("horizon_scan*.py")),
    ]


def analysis_kernel_hash(repo_root: Path) -> str:
    return phase_a_code_hash(analysis_kernel_paths(repo_root))


def kst_now_iso() -> str:
    return datetime.now(tz=KST).isoformat()


def build_run_spec(
    config: HorizonScanConfig,
    manifest: dict[str, Any],
    *,
    snapshot_date: str,
    source: str,
    resolution_auto_selected: bool,
    smoke_family: str | None,
    permutation_repeats_override: int | None,
    include_holdout: bool,
    holdout_start_override: str | None,
    repo_root: Path,
    code_paths: list[Path],
    command_line: list[str],
    started_at: str,
    finished_at: str | None = None,
) -> dict[str, Any]:
    """Assemble the immutable run_spec payload (§1.2 fingerprint + provenance).

    Raises ``PreflightError`` before assembling anything if any §1.1 entry
    condition fails, so an official run never gets as far as writing a spec
    for an invalid input state.
    """
    assert_holdout_untouched(
        include_holdout=include_holdout, holdout_start_override=holdout_start_override
    )
    run_preflight_checks(config, manifest)
    official, smoke_only, debug_reasons = determine_official_mode(
        resolution_auto_selected=resolution_auto_selected,
        smoke_family=smoke_family,
        permutation_repeats_override=permutation_repeats_override,
        config_permutation_repeats=int(config.raw["placebo"]["cross_sectional_repeats"]),
    )
    code_hash = phase_a_code_hash(code_paths)
    marts_by_view = {m["view"]: m for m in manifest.get("marts", [])}
    return {
        "phase": "A",
        "snapshot_date": snapshot_date,
        "source": source,
        "raw_manifest_hash": manifest.get("raw_marker"),
        "a0_manifest_hash": manifest.get("config_hash"),
        "config_schema_version": config.raw["schema_version"],
        "config_hash": config.config_hash,
        "marts": {
            name: {
                "schema_hash": marts_by_view[name]["schema_hash"],
                "row_count": marts_by_view[name]["row_count"],
            }
            for name in REQUIRED_A0_MARTS
        },
        "quality_policy_version": config.raw["quality"],
        "universe_policy_version": config.raw["universe"],
        "label_policy_version": {
            "holdout_start": config.raw["sample"]["holdout_start"],
            "holdout_boundary": config.raw["sample"]["holdout_boundary"],
        },
        "git_commit": git_metadata(repo_root)["git_commit"],
        "git_dirty": git_metadata(repo_root)["git_dirty"],
        "phase_a_code_hash": code_hash,
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
        "run_id": f"{started_at[:19].replace(':', '').replace('-', '')}-{code_hash[:8]}",
        **package_versions(),
        "command_line": command_line,
        "started_at": started_at,
        "finished_at": finished_at,
        "official": official,
        "smoke_only": smoke_only,
        "smoke_only_reasons": debug_reasons,
    }


def write_run_spec(run_dir: Path, run_spec: dict[str, Any]) -> Path:
    """Write ``run_spec.json`` once; refuses to silently overwrite an existing one.

    ``quality_policy_version``/``label_policy_version`` embed raw YAML config
    sections verbatim (§1.2), and PyYAML parses bare ``YYYY-MM-DD`` scalars
    (e.g. ``sample.holdout_start``, ``quality.price_limit_regimes[].start``)
    as ``datetime.date`` — not JSON-serializable by default, hence
    ``default=str`` (equivalent to ``date.isoformat()``).
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "run_spec.json"
    if target.exists():
        raise FileExistsError(f"run_spec.json already exists and is immutable: {target}")
    temp = run_dir / "run_spec.json.tmp"
    temp.write_text(
        json.dumps(run_spec, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temp.replace(target)
    return target


# --- A-9 atomic run-directory publish (§A-9 completion, §6.1 directory contract) ---

REQUIRED_RUN_ARTIFACTS: tuple[str, ...] = ("run_spec.json", "manifest.json")

# Provenance/presentation files carry run-specific timestamps (started_at,
# published_at, run identity prose) that legitimately differ between two
# reruns of the same snapshot/config/code — hashing them would make the
# §8.3 "재실행 hash 동일" reproducibility check fail even when every
# statistical result is bit-for-bit identical. Only the derived-output
# subtrees (core/, robustness/, permutation/, cards/) are hashed; plots/ is
# excluded too since PNG bytes are not the reproducibility target.
_CONTENT_HASH_EXCLUDE_NAMES = frozenset(
    {"run_spec.json", "_SUCCESS.json", "03a_horizon_scan_results.md", "timings.json"}
)
_CONTENT_HASH_EXCLUDE_DIRS = frozenset({"plots"})


def compute_run_content_hash(
    run_dir: Path,
    *,
    exclude_names: frozenset[str] = _CONTENT_HASH_EXCLUDE_NAMES,
    exclude_dirs: frozenset[str] = _CONTENT_HASH_EXCLUDE_DIRS,
) -> str:
    """SHA256 over every non-excluded file's (relative path, bytes), sorted by
    path — stable across two runs of the same snapshot/config/code, and
    sensitive to any change in the actual scan output.
    """
    hasher = hashlib.sha256()
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(run_dir)
        if rel.name in exclude_names or exclude_dirs & set(rel.parts[:-1]):
            continue
        hasher.update(rel.as_posix().encode())
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def publish_run(
    tmp_run_dir: Path,
    final_run_dir: Path,
    *,
    run_spec: dict[str, Any],
    required_artifacts: tuple[str, ...] = REQUIRED_RUN_ARTIFACTS,
    content_hash_exclude_names: frozenset[str] = _CONTENT_HASH_EXCLUDE_NAMES,
    success_extra: dict[str, Any] | None = None,
) -> Path:
    """Atomic tmp-dir-then-rename publish (§A-9: "임시 run directory에 쓴 뒤
    ... 최종 directory로 rename한다. `_SUCCESS.json`은 마지막에 기록한다").

    Validates the required artifacts exist in ``tmp_run_dir``, renames the
    whole tree in one filesystem operation (so a concurrent reader never sees
    a final directory that exists but is incomplete), then writes
    ``_SUCCESS.json`` into the now-final directory last.

    ``content_hash_exclude_names`` defaults to Phase A's provenance-file set;
    a caller whose run-spec filename differs (e.g. Phase B's
    ``phase_b_run_spec.json``) must pass its own set or the timestamp fields
    inside that file would leak into the reproducibility hash.

    ``success_extra`` adds run-level facts to ``_SUCCESS.json`` — Stage 0's
    ``daily_ic_reconciled``/``daily_ic_reconcile_max_abs_diff``. It records a
    check the caller has *already* enforced: a run whose stored daily IC does
    not rebuild its own summary must raise before reaching this function, so
    a published ``_SUCCESS.json`` never carries a false there.
    ``REQUIRED_RUN_ARTIFACTS`` is deliberately not extended for daily_ic —
    runs published before Stage 0 stay valid.
    """
    missing = [name for name in required_artifacts if not (tmp_run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f"cannot publish an incomplete run: missing {missing}")
    if final_run_dir.exists():
        raise FileExistsError(f"run directory already exists and is immutable: {final_run_dir}")
    content_hash = compute_run_content_hash(tmp_run_dir, exclude_names=content_hash_exclude_names)
    final_run_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp_run_dir.rename(final_run_dir)
    success = {
        "status": "success",
        "run_id": run_spec.get("run_id"),
        "config_hash": run_spec.get("config_hash"),
        "content_hash": content_hash,
        "published_at": kst_now_iso(),
        **(success_extra or {}),
    }
    (final_run_dir / "_SUCCESS.json").write_text(
        json.dumps(success, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return final_run_dir
