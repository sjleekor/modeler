"""Build the frozen release directory of the shadow serving (one model, no briefing).

Same pattern as ``serving/release_build.py`` (reviewed list of source files with sha256,
copy-and-verify, ``release.json``, refuse if the output exists) but *separate*: the briefing
release has exactly three models hard-coded (``release_build.JOBS``), so this one does not
touch it. Nothing here imports ``modeler.serving``.

What is copied: the import closure of ``shadow`` and of this module inside the
``modeler`` package (static AST walk of ``import`` / ``from ... import``, every package
``__init__.py`` on the way included) and, optionally, the frozen bundle (file by file, each
checked against the sha256 the bundle's ``manifest.json`` promises). Third-party packages
(polars, duckdb, scikit-learn, scipy, numpy, joblib) come from the runtime venv, not from here.

Release layout::

    release.json
    src/modeler/...            the closure
    bundle/                    the frozen bundle (when ``--bundle`` is given)

``code_sha256`` is computed over *relative* paths, so the release can be moved (unlike the
briefing release, which binds absolute paths). Run from a release with::

    PYTHONPATH=<release>/src python -m modeler.models._03_kr_next.shadow score ...

Reviewed list: ``manifest`` prints the sha256sum lines of the closure; a human reviews them
and the build takes that file as ``--source-manifest``. A file that is not listed, or whose
digest differs, stops the build.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from modeler.models._03_kr_next import bundle as bundle_mod

SCHEMA = "kr-next-shadow-release.v1"
ROOT_MODULE = "modeler.models._03_kr_next.shadow"
#: the release also ships this tool, so a server can ``verify`` a release with its own code
ROOT_MODULES = (ROOT_MODULE, "modeler.models._03_kr_next.shadow_release")
ENTRYPOINT = "modeler.models._03_kr_next.shadow:main"
TOP_PACKAGES = ("modeler",)
HEX = re.compile(r"[0-9a-f]{64}\Z")


class ShadowReleaseError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_inventory_sha256(files: list[tuple[str, str]]) -> str:
    records = sorted(files)
    if not records or len({p for p, _ in records}) != len(records):
        raise ShadowReleaseError("code inventory must contain unique files")
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------- import closure


def _module_file(src_root: Path, module: str) -> Path | None:
    base = src_root.joinpath(*module.split("."))
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _imports(path: Path, module: str, is_package: bool) -> tuple[set[str], set[str]]:
    """``(required, optional)`` absolute dotted names a file imports.

    ``import a.b`` and ``from a.b import c`` require the module ``a.b``; ``a.b.c`` is optional
    (it is a submodule or just an attribute of ``a.b``).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = module if is_package else module.rpartition(".")[0]
    out: set[str] = set()
    opt: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".")
                if node.level > len(parts):
                    raise ShadowReleaseError(f"relative import leaves the package: {path}")
                keep = parts[: len(parts) - (node.level - 1)]
                base = ".".join([*keep, *([node.module] if node.module else [])])
            else:
                base = node.module or ""
            if not base:
                continue
            out.add(base)
            opt.update(f"{base}.{a.name}" for a in node.names if a.name != "*")
    return out, opt


def import_closure(modeler_root: Path, root_modules: tuple[str, ...] = ROOT_MODULES) -> list[str]:
    """Sorted paths (relative to ``modeler_root``, e.g. ``src/modeler/etl/lake.py``)."""
    src = Path(modeler_root) / "src"
    seen: dict[str, Path] = {}
    stack = [(m, True) for m in root_modules]
    while stack:
        mod, required = stack.pop()
        if mod in seen or mod.split(".")[0] not in TOP_PACKAGES:
            continue
        f = _module_file(src, mod)
        if f is None:
            if required:
                raise ShadowReleaseError(f"imported module {mod} is not in {src}")
            continue  # an attribute of a module (``from x import name``), not a module
        seen[mod] = f
        parts = mod.split(".")
        for i in range(1, len(parts)):  # python runs the parents' __init__ first
            stack.append((".".join(parts[:i]), True))
        req, opt = _imports(f, mod, f.name == "__init__.py")
        stack.extend((m, True) for m in sorted(req))
        stack.extend((m, False) for m in sorted(opt))
    return sorted(p.relative_to(modeler_root).as_posix() for p in seen.values())


# --------------------------------------------------------------------------- reviewed list


def source_manifest_text(modeler_root: Path) -> str:
    """sha256sum lines (``<hex>  modeler/<path>``) of the closure, for the reviewer."""
    root = Path(modeler_root)
    lines = [f"{sha256_file(root / rel)}  modeler/{rel}" for rel in import_closure(root)]
    return "\n".join(lines) + "\n"


def read_source_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](\S.*)", line)
        if not match:
            raise ShadowReleaseError(f"source manifest line {number} is malformed")
        digest, name = match.groups()
        if name in entries:
            raise ShadowReleaseError(f"source manifest lists a path twice: {name}")
        entries[name] = digest
    if not entries:
        raise ShadowReleaseError("source manifest is empty")
    return entries


def _copy_verified(source: Path, target: Path, expected: str, label: str) -> None:
    if source.is_symlink() or not source.is_file():
        raise ShadowReleaseError(f"{label} is not a regular file")
    if sha256_file(source) != expected:
        raise ShadowReleaseError(f"{label} SHA-256 differs before copy")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if sha256_file(target) != expected:
        raise ShadowReleaseError(f"{label} SHA-256 differs after copy")


def _copy_bundle(source_dir: Path, target_dir: Path) -> str:
    source_dir = Path(source_dir)
    manifest_path = source_dir / bundle_mod.MANIFEST_FILE
    if source_dir.is_symlink() or not manifest_path.is_file():
        raise ShadowReleaseError(f"bundle directory has no manifest.json: {source_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if manifest.get("schema_version") != bundle_mod.SCHEMA_VERSION or not isinstance(files, dict):
        raise ShadowReleaseError("bundle manifest has the wrong schema or no files map")
    for name, digest in sorted(files.items()):
        if "/" in name or name.startswith(".") or not HEX.fullmatch(str(digest)):
            raise ShadowReleaseError(f"bundle manifest pin is invalid: {name}")
        _copy_verified(source_dir / name, target_dir / name, digest, f"bundle {name}")
    manifest_sha = sha256_file(manifest_path)
    _copy_verified(manifest_path, target_dir / bundle_mod.MANIFEST_FILE, manifest_sha, "manifest")
    return manifest_sha


# --------------------------------------------------------------------------- build and verify


def build_release(
    *,
    modeler_root: Path,
    source_manifest: Path,
    output: Path,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    """Copy the reviewed closure (and the bundle) to ``output``; refuses an existing path."""
    modeler_root, output = Path(modeler_root), Path(output)
    parent = output.parent.resolve(strict=True)
    final = parent / output.name
    if os.path.lexists(final):
        raise ShadowReleaseError(f"output path already exists: {final}")
    listed = read_source_manifest(source_manifest)
    rels = import_closure(modeler_root)
    if not rels:
        raise ShadowReleaseError("import closure is empty")
    stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.build-", dir=parent))
    try:
        inventory: list[dict[str, str]] = []
        for rel in rels:
            source = modeler_root / rel
            digest = sha256_file(source)
            key = f"modeler/{rel}"
            if key not in listed:
                raise ShadowReleaseError(f"file is not in the reviewed source manifest: {key}")
            if listed[key] != digest:
                raise ShadowReleaseError(f"file SHA-256 differs from the reviewed manifest: {key}")
            _copy_verified(source, stage / rel, digest, rel)
            inventory.append({"path": rel, "sha256": digest})
        bundle_block = None
        if bundle_dir is not None:
            bundle_block = {
                "path": f"bundle/{bundle_mod.MANIFEST_FILE}",
                "manifest_sha256": _copy_bundle(Path(bundle_dir), stage / "bundle"),
            }
        release = {
            "schema_version": SCHEMA,
            "frozen": True,
            "entrypoint": ENTRYPOINT,
            "source_manifest_sha256": sha256_file(source_manifest),
            "files": inventory,
            "code_sha256": code_inventory_sha256([(i["path"], i["sha256"]) for i in inventory]),
            "bundle": bundle_block,
        }
        (stage / "release.json").write_text(
            json.dumps(release, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.rename(stage, final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    try:
        report = verify_release(final)
    except BaseException:
        shutil.rmtree(final, ignore_errors=True)
        raise
    return {"release_json": str(final / "release.json"), **report}


def verify_release(release: Path) -> dict[str, Any]:
    """Re-hash every file, check the closure of the *copied* tree is complete, check the bundle."""
    release = Path(release)
    meta = json.loads((release / "release.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA:
        raise ShadowReleaseError("unknown release schema")
    pairs = []
    for item in meta["files"]:
        f = release / item["path"]
        if not f.is_file() or sha256_file(f) != item["sha256"]:
            raise ShadowReleaseError(f"release file missing or changed: {item['path']}")
        pairs.append((item["path"], item["sha256"]))
    if code_inventory_sha256(pairs) != meta["code_sha256"]:
        raise ShadowReleaseError("code_sha256 does not match the file list")
    try:
        copied = import_closure(release)  # same walk, now over the release tree
    except ShadowReleaseError as exc:
        raise ShadowReleaseError(f"release is not closed under its own imports: {exc}") from exc
    if copied != sorted(p for p, _ in pairs):
        raise ShadowReleaseError("release is not closed under its own imports (file list)")
    bundle_ok = None
    if meta.get("bundle"):
        manifest = release / meta["bundle"]["path"]
        if sha256_file(manifest) != meta["bundle"]["manifest_sha256"]:
            raise ShadowReleaseError("bundle manifest changed")
        bundle_mod.load_bundle(manifest.parent, verify_golden=True)
        bundle_ok = True
    return {
        "python_file_count": len(pairs),
        "code_sha256": meta["code_sha256"],
        "bundle_verified": bundle_ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shadow_release", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="print the sha256sum lines of the closure for review")
    m.add_argument("--modeler-root", required=True, type=Path)
    b = sub.add_parser("build")
    b.add_argument("--modeler-root", required=True, type=Path)
    b.add_argument("--source-manifest", required=True, type=Path)
    b.add_argument("--output", required=True, type=Path)
    b.add_argument("--bundle", type=Path, default=None)
    v = sub.add_parser("verify")
    v.add_argument("--release", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "manifest":
            sys.stdout.write(source_manifest_text(args.modeler_root))
            return 0
        if args.cmd == "verify":
            print(json.dumps(verify_release(args.release), sort_keys=True, indent=2))
            return 0
        summary = build_release(
            modeler_root=args.modeler_root,
            source_manifest=args.source_manifest,
            output=args.output,
            bundle_dir=args.bundle,
        )
    except (ShadowReleaseError, bundle_mod.BundleError, OSError, ValueError, KeyError) as exc:
        print(f"shadow_release failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
