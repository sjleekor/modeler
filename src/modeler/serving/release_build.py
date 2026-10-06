"""Build the frozen daily-briefing serving release from a reviewed source list.

The coordinator trusts ``release.json`` and the ``src/`` tree beside it.  This
CLI copies only files whose SHA-256 appears in a reviewed ``sha256sum`` list,
pins the three model bundles, and self-checks with the coordinator's own
``_release_jobs`` validator.  It never touches the network, a database or a lake.

``--ms-bundle`` (optional) adds the market-sector section: the frozen MS1 run bundle
(``score_daily build-bundle``) is copied to ``bundles/market_sector/`` file by file and pinned by
the SHA-256 of its ``bundle.json``; the scoring code (``src/modeler/scores/**``) is part of the same
reviewed ``src`` inventory, so the reviewed source list must cover those files too.

``code_sha256`` is computed over the *absolute* release paths (that is how
``_release_jobs`` recomputes it), so a release is bound to its final location.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from modeler.scores.market_sector.bundle import BUNDLE_FILE, verify_bundle_dir

from .daily_inputs import (
    MS_BUNDLE_DIR,
    MS_BUNDLE_PATH,
    MS_CODE_PATH,
    MS_ENTRYPOINT,
    _release_jobs,
    _release_market_sector,
)
from .orchestration import code_inventory_sha256

HEX = re.compile(r"[0-9a-f]{64}\Z")
SCHEMA = "daily-briefing-release.v1"
ADAPTER = "src/modeler/serving/adapters.py"

# Explicit package-data allowlist: (collector-relative path). KR prepare reads
# the KRX holiday table through importlib.resources.
DATA_FILES = ("src/collector/kr/infra/calendar/data/holidays_krx.csv",)

# (market, model_id, model_version, entrypoint, bundle dir, kind)
JOBS = (
    ("KR", "kr_daily_h20_v1", "1.0.0", "modeler.serving.adapters:infer_kr_daily", "bundles/kr", "kr"),
    ("US", "us_exploratory_20260929_r1_lightgbm", "1", "modeler.serving.adapters:infer_us_model",
     "bundles/lightgbm", "lightgbm"),
    ("US", "us_exploratory_20260929_r1_ridge", "1", "modeler.serving.adapters:infer_us_model",
     "bundles/ridge", "ridge"),
)
US_PINS = (("model_sha256", "model.joblib"),
           ("golden_features_sha256", "golden_features.parquet"),
           ("golden_predictions_sha256", "golden_predictions.parquet"))


class BuildError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_source_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](\S.*)", line)
        if not match:
            raise BuildError(f"source manifest line {number} is malformed")
        digest, name = match.groups()
        if name in entries:
            raise BuildError(f"source manifest lists a path twice: {name}")
        entries[name] = digest
    if not entries:
        raise BuildError("source manifest is empty")
    return entries


def _skipped(name: str) -> bool:
    return name == "__pycache__" or name.startswith("._")


def python_files(package_dir: Path) -> list[Path]:
    """Return every ``*.py`` under package_dir; stop on any symlink."""
    if package_dir.is_symlink() or not package_dir.is_dir():
        raise BuildError(f"source package is missing or a symlink: {package_dir}")
    found: list[Path] = []
    for current, dirs, files in os.walk(package_dir, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not _skipped(d))
        for name in dirs:
            if (Path(current) / name).is_symlink():
                raise BuildError(f"symlink in source tree: {Path(current) / name}")
        for name in sorted(files):
            if _skipped(name):
                continue
            path = Path(current) / name
            if path.is_symlink():
                raise BuildError(f"symlink in source tree: {path}")
            if name.endswith(".py"):
                found.append(path)
    return found


def _copy_verified(source: Path, target: Path, expected: str, label: str) -> str:
    if source.is_symlink() or not source.is_file():
        raise BuildError(f"{label} is not a regular file")
    if sha256_file(source) != expected:
        raise BuildError(f"{label} SHA-256 differs before copy")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256_file(target) != expected:
        raise BuildError(f"{label} SHA-256 differs after copy")
    return expected


def _require_listed(listed: dict[str, str], key: str, digest: str) -> None:
    if key not in listed:
        raise BuildError(f"file is not in the reviewed source manifest: {key}")
    if listed[key] != digest:
        raise BuildError(f"file SHA-256 differs from the reviewed source manifest: {key}")


def _listed_key(source: Path, modeler_root: Path, label: str) -> str:
    try:
        rel = source.resolve(strict=True).relative_to(modeler_root.resolve(strict=True))
    except (ValueError, OSError):
        raise BuildError(f"{label} source is not under modeler_root") from None
    return f"modeler/{rel.as_posix()}"


def copy_sources(modeler_root: Path, collector_root: Path, listed: dict[str, str],
                 release: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    inventory: list[dict[str, str]] = []
    plan = (("modeler", modeler_root, "modeler"), ("collector", collector_root, "collector"))
    for repo, root, package in plan:
        package_dir = root / "src" / package
        for path in python_files(package_dir):
            rel = path.relative_to(root)
            digest = sha256_file(path)
            _require_listed(listed, f"{repo}/{rel.as_posix()}", digest)
            _copy_verified(path, release / rel, digest, rel.as_posix())
            inventory.append({"path": rel.as_posix(), "sha256": digest})
    data_files: list[dict[str, str]] = []
    for rel_name in DATA_FILES:
        source = collector_root / rel_name
        if source.is_symlink() or not source.is_file():
            raise BuildError(f"allowlisted data file is missing: {rel_name}")
        digest = sha256_file(source)
        _require_listed(listed, f"collector/{rel_name}", digest)
        _copy_verified(source, release / rel_name, digest, rel_name)
        data_files.append({"path": rel_name, "sha256": digest})
    if ADAPTER not in {item["path"] for item in inventory}:
        raise BuildError("adapters.py is absent from the copied source")
    inventory.sort(key=lambda item: item["path"])
    return inventory, data_files


def _pinned_names(kind: str, manifest: dict[str, Any]) -> dict[str, str]:
    """Map bundle file name -> SHA-256 that the bundle manifest promises."""
    if kind == "kr":
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            raise BuildError("KR bundle manifest has no files map")
        pins = dict(files)
    else:
        pins = {name: manifest.get(key) for key, name in US_PINS}
        if manifest.get("variant") != kind:
            raise BuildError(f"US bundle variant is not {kind}")
    for name, digest in pins.items():
        if "/" in name or name.startswith(".") or not isinstance(digest, str) or not HEX.fullmatch(digest):
            raise BuildError(f"bundle manifest pin is invalid: {name}")
    return pins


def copy_bundle(source_dir: Path, target_dir: Path, kind: str, model_version: str,
                market: str) -> str:
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise BuildError(f"bundle directory is missing or a symlink: {source_dir}")
    manifest_path = source_dir / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise BuildError("bundle manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise BuildError("bundle manifest must be a JSON object")
    if manifest.get("model_version") != model_version:
        raise BuildError(f"bundle model_version {manifest.get('model_version')!r} "
                         f"differs from expected {model_version!r}")
    if manifest.get("market", market) != market:
        raise BuildError("bundle market differs from expected")
    pins = _pinned_names(kind, manifest)
    manifest_sha = sha256_file(manifest_path)
    for name, digest in sorted(pins.items()):
        _copy_verified(source_dir / name, target_dir / name, digest, f"{kind} bundle {name}")
    _copy_verified(manifest_path, target_dir / "manifest.json", manifest_sha, f"{kind} bundle manifest")
    return manifest_sha


def copy_ms_bundle(source_dir: Path, target_dir: Path) -> str:
    """Copy the market-sector bundle after checking every file against its ``bundle.json``."""
    source_dir = Path(source_dir)
    manifest = verify_bundle_dir(source_dir)
    for rel, digest in sorted(manifest["files"].items()):
        _copy_verified(source_dir / rel, target_dir / rel, digest, f"market sector bundle {rel}")
    _copy_verified(source_dir / BUNDLE_FILE, target_dir / BUNDLE_FILE, manifest["_sha256"],
                   "market sector bundle.json")
    return manifest["_sha256"]


def build_release(*, modeler_root: Path, collector_root: Path, kr_bundle: Path,
                  us_lightgbm_bundle: Path, us_ridge_bundle: Path, model_cards: Path,
                  runtime_manifest: Path, uv_lock: Path, source_manifest: Path,
                  output: Path, synthetic_fixture: bool = False,
                  ms_bundle: Path | None = None) -> dict[str, Any]:
    output = Path(output)
    parent = output.parent.resolve(strict=True)
    final = parent / output.name
    if os.path.lexists(final):
        raise BuildError(f"output path already exists: {final}")
    listed = read_source_manifest(source_manifest)
    stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.build-", dir=parent))
    try:
        inventory, data_files = copy_sources(Path(modeler_root), Path(collector_root), listed, stage)
        bundle_sources = {"kr": Path(kr_bundle), "lightgbm": Path(us_lightgbm_bundle),
                          "ridge": Path(us_ridge_bundle)}
        files = [(final / item["path"], item["sha256"]) for item in inventory]
        code_sha = code_inventory_sha256(files)
        adapter_sha = next(item["sha256"] for item in inventory if item["path"] == ADAPTER)
        jobs = []
        for market, model_id, version, entrypoint, bundle_rel, kind in JOBS:
            manifest_sha = copy_bundle(bundle_sources[kind], stage / bundle_rel, kind, version, market)
            jobs.append({"market": market, "model_id": model_id, "model_version": version,
                         "entrypoint": entrypoint, "bundle_path": f"{bundle_rel}/manifest.json",
                         "bundle_sha256": manifest_sha, "code_path": ADAPTER,
                         "code_path_sha256": adapter_sha, "code_files": inventory,
                         "code_sha256": code_sha})
        ms_block = None
        if ms_bundle is not None:
            code = next((item for item in inventory if item["path"] == MS_CODE_PATH), None)
            if code is None:
                raise BuildError("score_daily.py is absent from the copied source")
            ms_block = {"entrypoint": MS_ENTRYPOINT, "bundle_path": MS_BUNDLE_PATH,
                        "bundle_sha256": copy_ms_bundle(Path(ms_bundle), stage / MS_BUNDLE_DIR),
                        "code_path": MS_CODE_PATH, "code_path_sha256": code["sha256"],
                        "code_sha256": code_sha}
        for source, name in ((model_cards, "model-cards.json"), (runtime_manifest, "runtime.json"),
                             (uv_lock, "uv.lock")):
            source = Path(source)
            digest = sha256_file(source)
            _require_listed(listed, _listed_key(source, Path(modeler_root), name), digest)
            _copy_verified(source, stage / name, digest, name)
        release = {"schema_version": SCHEMA, "frozen": True,
                   "synthetic_fixture": bool(synthetic_fixture), "release_root": str(final),
                   "source_manifest_sha256": sha256_file(source_manifest),
                   "data_files": data_files, "jobs": jobs}
        if ms_block is not None:
            release["market_sector"] = ms_block
        (stage / "release.json").write_text(
            json.dumps(release, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.rename(stage, final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    try:
        checked, fixture = _release_jobs(final / "release.json")
        ms_checked = _release_market_sector(final / "release.json", checked)
        if (ms_bundle is None) != (ms_checked is None):
            raise BuildError("market sector block does not match --ms-bundle")
    except BaseException:
        shutil.rmtree(final, ignore_errors=True)
        raise
    weights = {}
    for (_, model_id, _, _, bundle_rel, kind) in JOBS:
        manifest = json.loads((final / bundle_rel / "manifest.json").read_text(encoding="utf-8"))
        weights[model_id] = _pinned_names(kind, manifest)["model.joblib"]
    summary_ms = None if ms_checked is None else {
        "bundle_sha256": ms_checked["bundle_sha256"], "code_path": MS_CODE_PATH}
    return {"release_json": str(final / "release.json"), "market_sector": summary_ms,
            "release_json_sha256": sha256_file(final / "release.json"),
            "python_file_count": len(inventory), "code_sha256": code_sha,
            "source_manifest_sha256": release["source_manifest_sha256"],
            "data_files": data_files, "weight_sha256": weights,
            "jobs_verified": sorted(model for _, model in checked),
            "synthetic_fixture": fixture}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m modeler.serving.release_build",
                                     description=__doc__)
    for name in ("modeler-root", "collector-root", "kr-bundle", "us-lightgbm-bundle",
                 "us-ridge-bundle", "model-cards", "runtime-manifest", "uv-lock",
                 "source-manifest", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--synthetic-fixture", action="store_true")
    parser.add_argument("--ms-bundle", type=Path, default=None,
                        help="market-sector bundle directory (score_daily build-bundle output)")
    args = parser.parse_args(argv)
    try:
        summary = build_release(
            modeler_root=args.modeler_root, collector_root=args.collector_root,
            kr_bundle=args.kr_bundle, us_lightgbm_bundle=args.us_lightgbm_bundle,
            us_ridge_bundle=args.us_ridge_bundle, model_cards=args.model_cards,
            runtime_manifest=args.runtime_manifest, uv_lock=args.uv_lock,
            source_manifest=args.source_manifest, output=args.output,
            synthetic_fixture=args.synthetic_fixture, ms_bundle=args.ms_bundle)
    except (BuildError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"release_build failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
