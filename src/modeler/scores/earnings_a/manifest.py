"""Run manifests for F3 (label-free) and F4 (hash-only): code state, inputs, spec, digests."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from modeler.scores.earnings_a import seal, spec
from modeler.scores.earnings_a.paths import RAW_TABLES, DataPaths, sha256_file

SCHEMA_VERSION = 1
PACKAGE_DIR = Path(__file__).resolve().parent
DIRTY_PATHSPECS = ("src/modeler/scores/earnings_a", "tests/scores/earnings_a")
MART_CODE_FILES = {
    "v4p": "src/modeler/etl/marts/financial_quarters_v4p.py",
    "v4": "src/modeler/etl/marts/financial_quarters_v4.py",
}


def _git(repo_dir: Path, *args: str) -> tuple[int, str]:
    p = subprocess.run(
        ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=False
    )
    return p.returncode, p.stdout.strip()


def git_state(repo_dir: str | Path) -> dict[str, Any]:
    """HEAD, branch, describe, and whether the earnings_a code or tests are dirty."""
    repo = Path(repo_dir)
    _, head = _git(repo, "rev-parse", "HEAD")
    _, branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    _, describe = _git(repo, "describe", "--tags", "--always")
    rc, status = _git(repo, "status", "--porcelain", "--", *DIRTY_PATHSPECS)
    return {"head": head, "branch": branch, "dirty": bool(rc != 0 or status), "describe": describe}


def repo_root(start: str | Path = PACKAGE_DIR) -> Path:
    _, top = _git(Path(start), "rev-parse", "--show-toplevel")
    return Path(top)


def main_repo_root(start: str | Path = PACKAGE_DIR) -> Path:
    """The main working tree (parent of the common git dir), also when ``start`` is a worktree."""
    _, common = _git(Path(start), "rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(common).parent


def package_file_sha256(pkg_dir: str | Path = PACKAGE_DIR) -> dict[str, str]:
    root = Path(pkg_dir)
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
        if "__pycache__" not in p.parts
    }


def env_versions() -> dict[str, str]:
    import numpy
    import pandas
    import polars
    import sklearn

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "polars": polars.__version__,
        "duckdb": duckdb.__version__,
        "scikit-learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def _mart_manifest_sha(paths: DataPaths, parquet: Path) -> str | None:
    for p in parquet.parents:
        if (p / "manifest.json").is_file():
            return sha256_file(p / "manifest.json")
        if p == paths.stock_data_root:
            break
    return None


def _count_rows(parquet: Path) -> int:
    con = duckdb.connect(":memory:")
    try:
        q = "'" + str(parquet).replace("'", "''") + "'"
        return int(con.execute(f"SELECT count(*) FROM read_parquet({q})").fetchone()[0])
    finally:
        con.close()


def _entry(paths: DataPaths, parquet: Path) -> dict[str, Any]:
    return {
        "path": str(parquet),
        "sha256": sha256_file(parquet),
        "rows": _count_rows(parquet),
        "manifest_json_sha256": _mart_manifest_sha(paths, parquet),
    }


def _r2_entry_row_count(paths: DataPaths) -> dict[str, Any]:
    """Hash, row count and mart manifest hash of the label mart. No values are read."""
    return _entry(paths, paths.r2_parquet)


def describe_inputs(paths: DataPaths, *, include_r2: bool = True) -> dict[str, Any]:
    """sha256, row count and mart manifest hash. Row counts only, never values."""
    out = {"v4p": _entry(paths, paths.v4p_parquet)}
    if include_r2:
        out["r2"] = _r2_entry_row_count(paths)
    return out


def mart_code_sha256() -> dict[str, str | None]:
    root = main_repo_root()
    out: dict[str, str | None] = {}
    for k, rel in MART_CODE_FILES.items():
        f = root / rel
        out[k] = sha256_file(f) if f.is_file() else None
    return out


def build_manifest(
    kind: str,
    paths: DataPaths,
    *,
    code: dict[str, Any],
    inputs: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a manifest. ``code`` is ``git_state(...)``; hashes of the package are added."""
    root = repo_root()
    m: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "label_spec_version": spec.LABEL_SPEC_VERSION,
        "code": {
            "repo": root.name,
            **code,
            "seal_py_sha256": sha256_file(PACKAGE_DIR / "seal.py"),
            "package_file_sha256": package_file_sha256(PACKAGE_DIR),
            "mart_code_sha256": mart_code_sha256(),
        },
        "inputs": inputs if inputs is not None else describe_inputs(paths),
        "raw_snapshot": spec.SNAPSHOT,
        "raw_tables_used": list(RAW_TABLES),
        "decision_dates": {
            "DEV": [d.isoformat() for d in seal.DEV_DATES],
            "GAP": [d.isoformat() for d in seal.GAP_DATES],
            "SEALED": [d.isoformat() for d in seal.SEALED_DATES],
            "CONFIRM": [d.isoformat() for d in seal.CONFIRM_DATES],
        },
        "seeds": list(spec.SEEDS),
        "bootstrap_seed": spec.BOOTSTRAP_SEED,
        "bootstrap_draws": spec.BOOTSTRAP_DRAWS,
        "hgb_grid": [dict(g) for g in spec.HGB_GRID],
        "ridge_alphas": list(spec.RIDGE_ALPHAS),
        "clip_q": list(spec.CLIP_Q),
        "env": env_versions(),
    }
    m.update(extra or {})
    return m


def write_once(path: str | Path, obj: Any) -> Path:
    """Write JSON to a new file; raise ``FileExistsError`` if it exists."""
    p = Path(path)
    with open(p, "x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    return p
