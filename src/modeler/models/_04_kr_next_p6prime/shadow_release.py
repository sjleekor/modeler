"""The P6′ shadow release: the P6 release procedure with the P6′ roots, schema and bundle.

Same steps, file layout and checks as ``_03_kr_next.shadow_release`` (``manifest`` -> human
review -> ``build`` -> ``verify``; static import closure over ``modeler`` and ``collector``, the
allowlisted holiday CSV, copy-and-verify, ``code_sha256`` over relative paths, the bundle copied
file by file against its manifest pins). It reuses that module's functions (``closure`` with
explicit roots, ``read_source_manifest``, ``_copy_verified``, ``_copy_bundle``, ``_source_of``,
``code_inventory_sha256``) and differs in four things:

* roots: :data:`ROOT_MODULES` (the P6′ ``shadow``, ``shadow_prepare`` and this module). Their
  imports pull in the P6 modules they wrap (``_03_kr_next.shadow``, ``shadow_prepare``,
  ``views``, ``build_panel``, ``shadow_release``) and the v3 fact / v4p mart code;
* ``schema_version`` ``kr-next-p6prime-shadow-release.v1`` and entrypoint
  ``modeler.models._04_kr_next_p6prime.shadow:main`` -- the P6 ``verify`` refuses this release
  and this ``verify`` refuses a P6 release;
* the bundle must be a P6′ bundle (``prereg_tag`` = ``kr-next-p6prime-frozen``) at build and at
  verify time; a P6 bundle stops the build;
* ``release.json`` also names the model (``"model": "kr-next-p6prime"``).

Run from a release with ``PYTHONPATH=<release>/src python -m
modeler.models._04_kr_next_p6prime.shadow_release verify --release <release>``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import shadow_release as p6
from modeler.models._04_kr_next_p6prime import shadow as shadow_mod
from modeler.models._04_kr_next_p6prime import tags

SCHEMA = "kr-next-p6prime-shadow-release.v1"
ROOT_MODULES = (
    "modeler.models._04_kr_next_p6prime.shadow",
    "modeler.models._04_kr_next_p6prime.shadow_prepare",
    "modeler.models._04_kr_next_p6prime.shadow_release",
)
ENTRYPOINT = "modeler.models._04_kr_next_p6prime.shadow:main"
ShadowReleaseError = p6.ShadowReleaseError
sha256_file = p6.sha256_file


def closure(modeler_root: Path, collector_root: Path | None = None) -> tuple[list[str], list[str]]:
    """``(python_files, data_files)`` of the P6′ roots (P6's walk and allowlist)."""
    return p6.closure(modeler_root, collector_root, ROOT_MODULES)


def source_manifest_text(modeler_root: Path, collector_root: Path) -> str:
    """sha256sum lines (``<hex>  modeler/<path>`` / ``collector/<path>``) for the review."""
    mroot, croot = Path(modeler_root), Path(collector_root)
    py, data = closure(mroot, croot)
    lines = [
        f"{sha256_file(p6._source_of(rel, mroot, croot))}  {rel.split('/')[1]}/{rel}"
        for rel in [*py, *data]
    ]
    return "\n".join(lines) + "\n"


def check_bundle_tag(bundle_dir: Path) -> str:
    try:
        tag, _ = shadow_mod.bundle_manifest_tag(bundle_dir)
    except shadow_mod.ShadowError as exc:
        raise ShadowReleaseError(str(exc)) from exc
    return tag


def build_release(
    *,
    modeler_root: Path,
    collector_root: Path,
    source_manifest: Path,
    output: Path,
    bundle_dir: Path | None = None,
) -> dict[str, Any]:
    """Copy the reviewed closure (and the P6′ bundle) to ``output``; refuses an existing path."""
    modeler_root, collector_root = Path(modeler_root), Path(collector_root)
    output = Path(output)
    parent = output.parent.resolve(strict=True)
    final = parent / output.name
    if os.path.lexists(final):
        raise ShadowReleaseError(f"output path already exists: {final}")
    if bundle_dir is not None:
        check_bundle_tag(Path(bundle_dir))
    listed = p6.read_source_manifest(source_manifest)
    rels, data_rels = closure(modeler_root, collector_root)
    if not rels:
        raise ShadowReleaseError("import closure is empty")
    stage = Path(tempfile.mkdtemp(prefix=f".{final.name}.build-", dir=parent))
    try:
        inventory: list[dict[str, str]] = []
        data_inventory: list[dict[str, str]] = []
        for rel in [*rels, *data_rels]:
            source = p6._source_of(rel, modeler_root, collector_root)
            digest = sha256_file(source)
            key = f"{rel.split('/')[1]}/{rel}"
            if key not in listed:
                raise ShadowReleaseError(f"file is not in the reviewed source manifest: {key}")
            if listed[key] != digest:
                raise ShadowReleaseError(f"file SHA-256 differs from the reviewed manifest: {key}")
            p6._copy_verified(source, stage / rel, digest, rel)
            (data_inventory if rel in data_rels else inventory).append(
                {"path": rel, "sha256": digest}
            )
        bundle_block = None
        if bundle_dir is not None:
            bundle_block = {
                "path": f"bundle/{bundle_mod.MANIFEST_FILE}",
                "manifest_sha256": p6._copy_bundle(Path(bundle_dir), stage / "bundle"),
                "prereg_tag": tags.PREREG_TAG,
                "bundle_tag": tags.BUNDLE_TAG,
            }
        release = {
            "schema_version": SCHEMA,
            "model": shadow_mod.MODEL,
            "frozen": True,
            "entrypoint": ENTRYPOINT,
            "source_manifest_sha256": sha256_file(source_manifest),
            "files": inventory,
            "data_files": data_inventory,
            "code_sha256": p6.code_inventory_sha256([(i["path"], i["sha256"]) for i in inventory]),
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
    """P6's checks (every file re-hashed, closure of the copied tree complete) + P6′ bundle."""
    release = Path(release)
    meta = json.loads((release / "release.json").read_text(encoding="utf-8"))
    if meta.get("schema_version") != SCHEMA or meta.get("model") != shadow_mod.MODEL:
        raise ShadowReleaseError("not a P6′ shadow release (schema_version/model)")
    pairs = []
    for item in meta["files"]:
        f = release / item["path"]
        if not f.is_file() or sha256_file(f) != item["sha256"]:
            raise ShadowReleaseError(f"release file missing or changed: {item['path']}")
        pairs.append((item["path"], item["sha256"]))
    if p6.code_inventory_sha256(pairs) != meta["code_sha256"]:
        raise ShadowReleaseError("code_sha256 does not match the file list")
    try:
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
        check_bundle_tag(manifest.parent)
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
    parser = argparse.ArgumentParser(
        prog="p6prime_shadow_release", description=__doc__.split("\n")[0]
    )
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
        print(f"p6prime_shadow_release failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
