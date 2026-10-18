"""Write-once model bundle of the next KR model (spec G-15).

Layout of a bundle directory::

    model.joblib            HistGradientBoostingRegressor (the chosen point, fixed max_iter)
    ridge.joblib            Ridge baseline (optional, record only; G-17)
    golden_inputs.parquet   up to 200 model-ready rows (keys + ranked feature columns, NULL kept)
    golden_preds.parquet    the scores of those rows (``score``, ``ridge_score``)
    manifest.json           everything else, including the sha256 of the four files above

``load_bundle(path, verify_golden=True)`` re-checks every hash and recomputes the golden scores.
The golden rows are *model-ready* (already rank-transformed): they verify the model files and the
library stack, not the preprocessing (that is pinned by its tests and the ``preprocessing`` spec
in the manifest).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl

from modeler.models._03_kr_next import preprocess as pp

SCHEMA_VERSION = "kr-next-bundle.v1"
MODEL_FILE = "model.joblib"
RIDGE_FILE = "ridge.joblib"
GOLDEN_INPUTS_FILE = "golden_inputs.parquet"
GOLDEN_PREDS_FILE = "golden_preds.parquet"
MANIFEST_FILE = "manifest.json"
GOLDEN_ROWS = 200
KEY_COLS = ("trade_date", "ticker", "market")
#: tolerance of the golden comparison (same library stack gives an exact match)
GOLDEN_RTOL = 1e-9
GOLDEN_ATOL = 1e-12
LIBRARIES = ("scikit-learn", "polars", "duckdb", "numpy", "joblib")


class BundleError(ValueError):
    """The bundle is missing, altered, or does not reproduce its golden scores."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def feature_list_hash(feature_cols: list[str] | tuple[str, ...]) -> str:
    """sha256 of the feature names joined by newlines, in model order."""
    return hashlib.sha256("\n".join(feature_cols).encode("utf-8")).hexdigest()


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in LIBRARIES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "missing"
    return out


def git_state(path: Path) -> dict[str, Any]:
    """``{"commit": sha | None, "dirty": bool | None}`` of the repository containing ``path``."""

    def run(*args: str) -> str | None:
        try:
            r = subprocess.run(
                ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
        return r.stdout.strip()

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def model_matrix(frame: pl.DataFrame, cols: list[str] | tuple[str, ...]) -> np.ndarray:
    """Float64 matrix of ``cols`` (NULL -> NaN)."""
    return frame.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()


def ridge_matrix(frame: pl.DataFrame, cols: list[str] | tuple[str, ...]) -> np.ndarray:
    """Ridge input: the ranked columns with NULL -> 0.5 (G-17)."""
    return model_matrix(pp.ridge_fill(frame, list(cols)), cols)


def pick_golden(frame: pl.DataFrame, n: int = GOLDEN_ROWS, seed: int = 0) -> pl.DataFrame:
    """Deterministic sample of ``n`` rows (all rows when fewer), sorted by the keys."""
    if frame.height > n:
        frame = frame.sample(n=n, seed=seed)
    return frame.sort(list(KEY_COLS))


def write_bundle(
    out_dir: str | Path,
    *,
    model: Any,
    feature_cols: list[str] | tuple[str, ...],
    golden_frame: pl.DataFrame,
    params: dict[str, Any],
    max_iter: int,
    score_matrix: dict[str, Any],
    determinism: dict[str, Any],
    snapshot_id: str,
    prereg_tag: str,
    prereg_commit: str,
    modeler_commit: str | None,
    modeler_dirty: bool | None = None,
    ridge: Any | None = None,
    ridge_alpha: float | None = None,
    prereg_doc_sha256: str | None = None,
    preprocessing: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a new bundle directory. Refuses to touch an existing path (write-once)."""
    out = Path(out_dir)
    if out.exists():
        raise BundleError(f"bundle path already exists, refusing to overwrite: {out}")
    cols = list(feature_cols)
    miss = [c for c in (*KEY_COLS, *cols) if c not in golden_frame.columns]
    if miss:
        raise BundleError(f"golden frame lacks columns: {miss}")
    golden = pick_golden(golden_frame.select([*KEY_COLS, *cols]))
    if golden.is_empty():
        raise BundleError("golden frame is empty")

    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}.tmp-", dir=out.parent))
    try:
        joblib.dump(model, tmp / MODEL_FILE)
        files = {MODEL_FILE: sha256_file(tmp / MODEL_FILE)}
        preds = {"score": model.predict(model_matrix(golden, cols))}
        if ridge is not None:
            joblib.dump(ridge, tmp / RIDGE_FILE)
            files[RIDGE_FILE] = sha256_file(tmp / RIDGE_FILE)
            preds["ridge_score"] = ridge.predict(ridge_matrix(golden, cols))
        golden.with_columns([pl.col(c).cast(pl.Float64) for c in cols]).write_parquet(
            tmp / GOLDEN_INPUTS_FILE
        )
        pl.DataFrame({**{k: golden[k] for k in KEY_COLS}, **preds}).write_parquet(
            tmp / GOLDEN_PREDS_FILE
        )
        files[GOLDEN_INPUTS_FILE] = sha256_file(tmp / GOLDEN_INPUTS_FILE)
        files[GOLDEN_PREDS_FILE] = sha256_file(tmp / GOLDEN_PREDS_FILE)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "feature_cols": cols,
            "feature_list_sha256": feature_list_hash(cols),
            "preprocessing": preprocessing or {},
            "params": params,
            "max_iter": int(max_iter),
            "ridge_alpha": ridge_alpha,
            "score_matrix": score_matrix,
            "determinism": determinism,
            "modeler_commit": modeler_commit,
            "modeler_dirty": modeler_dirty,
            "libraries": library_versions(),
            "snapshot_id": snapshot_id,
            "prereg_tag": prereg_tag,
            "prereg_commit": prereg_commit,
            "prereg_doc_sha256": prereg_doc_sha256,
            "golden_rows": golden.height,
            "files": files,
            **(extra or {}),
        }
        (tmp / MANIFEST_FILE).write_text(
            json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return out


@dataclass
class LoadedBundle:
    directory: Path
    manifest: dict[str, Any]
    model: Any
    ridge: Any | None

    @property
    def feature_cols(self) -> list[str]:
        return list(self.manifest["feature_cols"])

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        """Score model-ready rows (rank-transformed feature columns)."""
        return self.model.predict(model_matrix(frame, self.feature_cols))

    def predict_ridge(self, frame: pl.DataFrame) -> np.ndarray:
        if self.ridge is None:
            raise BundleError("bundle has no ridge model")
        return self.ridge.predict(ridge_matrix(frame, self.feature_cols))


def load_bundle(path: str | Path, *, verify_golden: bool = True) -> LoadedBundle:
    """Load a bundle; hash mismatch always raises, golden mismatch raises when verifying."""
    d = Path(path)
    mpath = d / MANIFEST_FILE
    if not mpath.is_file():
        raise BundleError(f"no manifest in bundle: {d}")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise BundleError(f"unknown bundle schema: {manifest.get('schema_version')!r}")
    files: dict[str, str] = manifest["files"]
    for name, want in files.items():
        f = d / name
        if not f.is_file():
            raise BundleError(f"bundle file missing: {name}")
        if sha256_file(f) != want:
            raise BundleError(f"sha256 mismatch for {name}")
    if feature_list_hash(manifest["feature_cols"]) != manifest["feature_list_sha256"]:
        raise BundleError("feature list hash mismatch")

    model = joblib.load(d / MODEL_FILE)
    ridge = joblib.load(d / RIDGE_FILE) if RIDGE_FILE in files else None
    bundle = LoadedBundle(directory=d, manifest=manifest, model=model, ridge=ridge)
    if verify_golden:
        verify_golden_preds(bundle)
    return bundle


def verify_golden_preds(bundle: LoadedBundle) -> None:
    gi = pl.read_parquet(bundle.directory / GOLDEN_INPUTS_FILE)
    gp = pl.read_parquet(bundle.directory / GOLDEN_PREDS_FILE)
    if gi.select(list(KEY_COLS)).to_dicts() != gp.select(list(KEY_COLS)).to_dicts():
        raise BundleError("golden inputs and preds are not on the same rows")
    got: dict[str, np.ndarray] = {"score": bundle.predict(gi)}
    if bundle.ridge is not None:
        got["ridge_score"] = bundle.predict_ridge(gi)
    for name, values in got.items():
        want = gp[name].to_numpy()
        if not np.allclose(values, want, rtol=GOLDEN_RTOL, atol=GOLDEN_ATOL, equal_nan=True):
            worst = float(np.nanmax(np.abs(values - want)))
            raise BundleError(f"golden {name} mismatch (max abs diff {worst:.3g})")
