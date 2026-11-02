"""P6′ panel builder: the P6 builder's functions, the P6′ tag and the P6′ view folder.

Same two modes and the same assembly as ``_03_kr_next.build_panel`` (it calls that module's
``check_outputs_free``, ``register_inputs`` and ``build_panel_file``, and ``dataset.KrNextViews``);
only ``main`` is new, because the P6 ``main`` checks the P6 tag ``kr-next-frozen``:

* ``labeled`` refuses (exit 2, before the lake is opened) unless ``--prereg-repo`` has the P6′
  tag ``kr-next-p6prime-frozen``. The P6 tag does not count;
* the views come from the P6′ view folder (default
  ``<lake>/derived/kr_next_p6prime_views/snapshot_date=<S>``). The folder's manifest must be a
  P6′ manifest and every file must match its sha256 (``inputs.load_p6prime_manifest``), so the
  P6 folder cannot be passed by mistake;
* the output may not lie inside the P6 dataset folder ``<lake>/datasets/kr_next``.

Outputs and manifests are those of the P6 builder (``panel_<mode>.parquet``,
``manifest_<mode>.json``, write-once), plus ``prereg_tag``/``prereg_commit`` (labeled),
``views_dir``, ``views_manifest_sha256`` and ``p6prime_fin_content_digest``.

Usage::

    python -m modeler.models._04_kr_next_p6prime.build_panel --mode serving --feature-set noflow \
        --lake <stock_data/kr> --snapshot 2026-10-23 --start 2017-01-01 --end 2026-10-22 \
        --out-dir <dir> --prereg-repo <my repo> --memory-limit 10GB
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb

from modeler.etl.config import DataRoot, LakeConfig
from modeler.models._03_kr_next import build_panel as p6_bp
from modeler.models._03_kr_next import dataset as ds
from modeler.models._03_kr_next import views as p6_views
from modeler.models._04_kr_next_p6prime import inputs, tags

MODES = p6_bp.MODES
BuildRefused = p6_bp.BuildRefused


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m modeler.models._04_kr_next_p6prime.build_panel",
        description="Guarded panel builder of P6′ (P6 assembly, P6′ tag and view folder)",
    )
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--lake", required=True, help="stock_data lake root (kr)")
    ap.add_argument("--snapshot", required=True, help="raw snapshot_date")
    ap.add_argument("--start", default=ds.DEFAULT_START_DATE, help="first trade_date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="last trade_date, inclusive (YYYY-MM-DD)")
    ap.add_argument("--out-dir", required=True, help="output directory (files must not exist)")
    ap.add_argument(
        "--prereg-repo", required=True, help=f"repository that holds the {tags.PREREG_TAG} tag"
    )
    ap.add_argument("--source", default=p6_bp.DEFAULT_SOURCE, help="raw lake source name")
    ap.add_argument("--feature-set", default="main", choices=("main", "noflow"))
    ap.add_argument("--views-dir", default=None, help="P6′ view folder")
    ap.add_argument("--memory-limit", default="10GB")
    ap.add_argument("--threads", type=int, default=8)
    return ap.parse_args(argv)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    for name, value in (("--start", args.start), ("--end", args.end)):
        if not p6_bp._DATE.match(value):
            print(f"REFUSED: {name} must be YYYY-MM-DD, got {value!r}", file=sys.stderr)
            return 2
    labeled = args.mode == "labeled"
    lake = Path(args.lake)
    out_dir = Path(args.out_dir)
    # 1) guards first: the P6′ freeze tag before anything is read, then where to write
    extra: dict[str, Any] = {}
    try:
        if labeled:
            extra["prereg_tag"] = tags.PREREG_TAG
            extra["prereg_commit"] = tags.require_prereg_tag(args.prereg_repo)
        if _inside(out_dir, lake / "datasets" / "kr_next"):
            raise BuildRefused(f"--out-dir is inside the P6 dataset folder: {out_dir}")
        p6_bp.check_outputs_free(out_dir, args.mode)
    except (tags.FrozenTagMissing, BuildRefused) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    views_dir = (
        Path(args.views_dir) if args.views_dir else inputs.default_views_dir(lake, args.snapshot)
    )
    try:
        vman = inputs.load_p6prime_manifest(views_dir)
    except inputs.InputsRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    extra["views_dir"] = str(views_dir)
    extra["views_manifest_sha256"] = p6_views.sha256_file(views_dir / inputs.MANIFEST_NAME)
    extra["p6prime_fin_content_digest"] = vman["views"][inputs.FIN_FILE].get("content_digest")

    tmp_dir = out_dir.parent / f".{out_dir.name}_duckdb_tmp"
    config = LakeConfig(root=DataRoot(base=lake), snapshot_date=args.snapshot, source=args.source)
    con = duckdb.connect()
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET threads={int(args.threads)}")
        con.execute(f"SET memory_limit='{args.memory_limit}'")
        con.execute(f"SET temp_directory='{tmp_dir.as_posix()}'")
        hashes = p6_bp.register_inputs(
            con, config, views_dir, labeled=labeled, feature_set=args.feature_set
        )
        extra["derived_views_sha256"] = hashes
        extra["raw_root"] = str(config.raw_root)
        manifest = p6_bp.build_panel_file(
            con,
            mode=args.mode,
            snapshot=args.snapshot,
            start=args.start,
            end=args.end,
            out_dir=out_dir,
            feature_set=args.feature_set,
            views=ds.KrNextViews(),
            extra_manifest=extra,
        )
    except BuildRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    finally:
        con.close()
        if tmp_dir.exists():
            for p in tmp_dir.glob("*"):
                p.unlink(missing_ok=True)
            tmp_dir.rmdir()
    print(f"panel written: {out_dir} rows={manifest['row_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
