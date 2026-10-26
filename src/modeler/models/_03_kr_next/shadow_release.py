"""Build the frozen release directory of the shadow serving (one model, no briefing).

Same pattern as ``serving/release_build.py`` (reviewed list of source files with sha256,
copy-and-verify, ``release.json``, refuse if the output exists) but *separate*: the briefing
release has exactly three models hard-coded (``release_build.JOBS``), so this one does not
touch it. Nothing here imports ``modeler.serving``.

What is copied: the import closure of ``ROOT_MODULES`` (shadow, shadow_prepare, views, build_panel,
forward, this module) inside the ``modeler`` *and* ``collector`` packages (the server venv has no
collector; ``--collector-root`` is the collector repository), plus the allowlisted package data
those modules read via ``importlib.resources`` (``DATA_FILES``). The walk is static AST
(``import`` / ``from ... import``, every package ``__init__.py`` on the way included). The frozen
bundle is optional (file by file, each checked against the sha256 the bundle's ``manifest.json``
promises). Third-party packages
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
#: the release also ships this tool, so a server can ``verify`` a release with its own code.
#: Server jobs: ``shadow`` (score), ``shadow_prepare`` (+ ``views``, ``build_panel``), ``forward``
#: (evaluation). ``train`` is not a root (training runs on the Mac); it ships only because
#: ``build_panel`` imports ``require_frozen_tag`` from it.
ROOT_MODULES = (
    ROOT_MODULE,
    "modeler.models._03_kr_next.shadow_release",
    "modeler.models._03_kr_next.shadow_prepare",
    "modeler.models._03_kr_next.views",
    "modeler.models._03_kr_next.build_panel",
    "modeler.models._03_kr_next.forward",
)
ENTRYPOINT = "modeler.models._03_kr_next.shadow:main"
TOP_PACKAGES = ("modeler", "collector")
#: package data the closure may read through ``importlib.resources.files`` (release paths; same
#: allowlist as ``serving/release_build.DATA_FILES``). A ``files(...) / "x"`` use in the closure
#: that resolves outside this list stops the build.
DATA_FILES = ("src/collector/kr/infra/calendar/data/holidays_krx.csv",)
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


def _resource_uses(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """``files("pkg") / "a" / "b"`` uses in a file: ``[("pkg", ("a", "b"))]``.

    A bare ``files("pkg")`` yields ``("pkg", ())`` so the package is shipped. Arguments that are
    not string literals cannot be resolved and stop the build.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parent = {id(c): n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}

    def literal(node: ast.AST) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        raise ShadowReleaseError(f"files() use that is not a string literal: {path}:{node.lineno}")

    out: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id != "files":
            continue
        if len(node.args) != 1:
            raise ShadowReleaseError(
                f"files() use that is not a string literal: {path}:{node.lineno}"
            )
        parts: list[str] = []
        cur: ast.AST = node
        up = parent.get(id(cur))
        while isinstance(up, ast.BinOp) and isinstance(up.op, ast.Div) and up.left is cur:
            parts.append(literal(up.right))
            cur, up = up, parent.get(id(up))
        out.append((literal(node.args[0]), tuple(parts)))
    return out


def _src_dirs(modeler_root: Path, collector_root: Path | None) -> dict[str, Path]:
    dirs = {"modeler": Path(modeler_root) / "src"}
    if collector_root is not None:
        dirs["collector"] = Path(collector_root) / "src"
    return dirs


def closure(
    modeler_root: Path,
    collector_root: Path | None = None,
    root_modules: tuple[str, ...] = ROOT_MODULES,
) -> tuple[list[str], list[str]]:
    """``(python_files, data_files)``, sorted release paths (``src/<pkg>/...``).

    Follows imports into ``modeler`` and ``collector``; ``collector_root`` is the collector
    repository (when verifying: the release itself). Data files are the allowlisted package data
    that ``importlib.resources.files`` uses in the closure reach; each must exist.
    """
    srcs = _src_dirs(modeler_root, collector_root)
    seen: dict[str, Path] = {}
    data: set[str] = set()
    stack = [(m, True) for m in root_modules]
    while stack:
        mod, required = stack.pop()
        top = mod.split(".")[0]
        if mod in seen or top not in TOP_PACKAGES:
            continue
        if top not in srcs:
            raise ShadowReleaseError(f"imported module {mod} needs a collector_root")
        f = _module_file(srcs[top], mod)
        if f is None:
            if required:
                raise ShadowReleaseError(f"imported module {mod} is not in {srcs[top]}")
            continue  # an attribute of a module (``from x import name``), not a module
        seen[mod] = f
        parts = mod.split(".")
        for i in range(1, len(parts)):  # python runs the parents' __init__ first
            stack.append((".".join(parts[:i]), True))
        req, opt = _imports(f, mod, f.name == "__init__.py")
        stack.extend((m, True) for m in sorted(req))
        stack.extend((m, False) for m in sorted(opt))
        for pkg, rest in _resource_uses(f):
            stack.append((pkg, True))  # the resource package is shipped too
            if not rest:
                continue
            rel = "src/" + "/".join([*pkg.split("."), *rest])
            if rel not in DATA_FILES:
                raise ShadowReleaseError(f"package data is not in the reviewed allowlist: {rel}")
            data.add(rel)
    py = sorted(f"src/{p.relative_to(srcs[m.split('.')[0]]).as_posix()}" for m, p in seen.items())
    for rel in sorted(data):
        source = _source_of(
            rel, srcs["modeler"].parent, srcs.get("collector", srcs["modeler"]).parent
        )
        if source.is_symlink() or not source.is_file():
            raise ShadowReleaseError(f"package data file is missing: {rel}")
    return py, sorted(data)


def import_closure(
    modeler_root: Path,
    root_modules: tuple[str, ...] = ROOT_MODULES,
    collector_root: Path | None = None,
) -> list[str]:
    """Sorted python files of the closure (see ``closure``)."""
    return closure(modeler_root, collector_root, root_modules)[0]


def _source_of(rel: str, modeler_root: Path, collector_root: Path | None) -> Path:
    top = rel.split("/")[1]
    root = modeler_root if top == "modeler" else collector_root
    if root is None:
        raise ShadowReleaseError(f"{rel} needs a collector_root")
    return Path(root) / rel


# --------------------------------------------------------------------------- reviewed list


def source_manifest_text(modeler_root: Path, collector_root: Path) -> str:
    """sha256sum lines (``<hex>  modeler/<path>`` or ``collector/<path>``) of closure and data."""
    mroot, croot = Path(modeler_root), Path(collector_root)
    py, data = closure(mroot, croot)
    lines = [
        f"{sha256_file(_source_of(rel, mroot, croot))}  {rel.split('/')[1]}/{rel}"
        for rel in [*py, *data]
    ]
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
    collector_root: Path,
    source_manifest: Path,
    output: Path,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    """Copy the reviewed closure (and the bundle) to ``output``; refuses an existing path."""
    modeler_root, collector_root = Path(modeler_root), Path(collector_root)
    output = Path(output)
    parent = output.parent.resolve(strict=True)
    final = parent / output.name
    if os.path.lexists(final):
        raise ShadowReleaseError(f"output path already exists: {final}")
    listed = read_source_manifest(source_manifest)
    rels, data_rels = closure(modeler_root, collector_root)
    if not rels:
        raise ShadowReleaseError("import closure is empty")
    stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.build-", dir=parent))
    try:
        inventory: list[dict[str, str]] = []
        data_inventory: list[dict[str, str]] = []
        for rel in [*rels, *data_rels]:
            source = _source_of(rel, modeler_root, collector_root)
            digest = sha256_file(source)
            key = f"{rel.split('/')[1]}/{rel}"
            if key not in listed:
                raise ShadowReleaseError(f"file is not in the reviewed source manifest: {key}")
            if listed[key] != digest:
                raise ShadowReleaseError(f"file SHA-256 differs from the reviewed manifest: {key}")
            _copy_verified(source, stage / rel, digest, rel)
            (data_inventory if rel in data_rels else inventory).append(
                {"path": rel, "sha256": digest}
            )
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
            "data_files": data_inventory,
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
        # same walk, now over the release tree (modeler and collector both resolve inside it)
        copied, copied_data = closure(release, release)
    except ShadowReleaseError as exc:
        raise ShadowReleaseError(f"release is not closed under its own imports: {exc}") from exc
    if copied != sorted(p for p, _ in pairs):
        raise ShadowReleaseError("release is not closed under its own imports (file list)")
    data_paths = []
    for item in meta.get("data_files", []):
        f = release / item["path"]
        if not f.is_file() or sha256_file(f) != item["sha256"]:
            raise ShadowReleaseError(f"release data file missing or changed: {item['path']}")
        data_paths.append(item["path"])
    if copied_data != sorted(data_paths):
        raise ShadowReleaseError("release is not closed under its own imports (data files)")
    bundle_ok = None
    if meta.get("bundle"):
        manifest = release / meta["bundle"]["path"]
        if sha256_file(manifest) != meta["bundle"]["manifest_sha256"]:
            raise ShadowReleaseError("bundle manifest changed")
        bundle_mod.load_bundle(manifest.parent, verify_golden=True)
        bundle_ok = True
    return {
        "python_file_count": len(pairs),
        "data_file_count": len(data_paths),
        "collector_modules": sorted(
            p[len("src/") : -len(".py")].replace("/", ".")
            for p, _ in pairs
            if p.startswith("src/collector/")
        ),
        "code_sha256": meta["code_sha256"],
        "bundle_verified": bundle_ok,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shadow_release", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest", help="print the sha256sum lines of the closure for review")
    m.add_argument("--modeler-root", required=True, type=Path)
    m.add_argument("--collector-root", required=True, type=Path)
    b = sub.add_parser("build")
    b.add_argument("--modeler-root", required=True, type=Path)
    b.add_argument("--collector-root", required=True, type=Path)
    b.add_argument("--source-manifest", required=True, type=Path)
    b.add_argument("--output", required=True, type=Path)
    b.add_argument("--bundle", type=Path, default=None)
    v = sub.add_parser("verify")
    v.add_argument("--release", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cmd == "manifest":
            sys.stdout.write(source_manifest_text(args.modeler_root, args.collector_root))
            return 0
        if args.cmd == "verify":
            print(json.dumps(verify_release(args.release), sort_keys=True, indent=2))
            return 0
        summary = build_release(
            modeler_root=args.modeler_root,
            collector_root=args.collector_root,
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
