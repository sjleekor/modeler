"""Prepare step of the next KR model's shadow serving: the serving-mode panel of the signal day.

``kr-next-shadow.sh`` scores ``panels_in/D=<D>/panel.parquet`` (+ ``panel_manifest.json``). This
module writes exactly that, through the same code path as ``build_panel`` (``serving`` mode: no
label, no return; ``liq_notice_*`` present):

* ``t`` = the session before ``D`` (``sessions`` = the KR calendar the wrapper reads).
* inputs: one raw export snapshot (``raw_root``) and the derived views of ``views.py`` built from
  the same snapshot (``views_dir``, checked against its manifest).
* output (write-once): ``<out_root>/panels_in/D=<D>/panel.parquet`` and ``panel_manifest.json``.

``panel_manifest.json`` (G-22, G-16): mode, ``t``, ``D``, the sha256 of every view SQL the panel
statement builds inline (``view_sql_sha256``), the whole panel statement, the derived views
(file sha256 and SQL sha256 each), the sha256 of the views manifest file, the raw snapshot, the
panel's own sha256 and the modeler commit. ``shadow score`` records it verbatim in ``record.json``.

Commands::

    python -m modeler.models._03_kr_next.shadow_prepare run --date D --raw-lake <raw_postgres> \
        --out-root <shadow root> --sessions <calendar json>     # views (if missing) + panel
    python -m modeler.models._03_kr_next.shadow_prepare prepare --date D --raw-root <snapshot> \
        --views-dir <views> --out-root <shadow root> --sessions <calendar json>

``run`` builds ``<out_root>/views/D-<D>/`` from the snapshot ``snapshot_date=<D>`` of the lake
when it is not there yet (a finished one is reused after its hashes check out), then prepares the
panel. Exit codes: 0 written, 2 refused (existing panel, missing input, hash mismatch), 31 raw
snapshot missing.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import duckdb

from modeler.etl.config import RAW_LAKE_NAME, DataRoot, LakeConfig
from modeler.etl.universe import UniverseFilter
from modeler.models._03_kr_next import build_panel as bp
from modeler.models._03_kr_next import dataset as ds
from modeler.models._03_kr_next import views as views_mod
from modeler.models._03_kr_next.shadow import ShadowError, _load_sessions, previous_session

SCHEMA = "kr-next-panel-manifest.v1"
PANEL_FILE = "panel.parquet"
MANIFEST_FILE = "panel_manifest.json"
DEFAULT_SOURCE = bp.DEFAULT_SOURCE
#: Measured on the Mac (2026-10-18, 2025-07-30 panel, 4 threads): the process peak footprint is
#: about the limit + 1.2GB (6GB -> 7.9GB, 4GB -> 5.7GB, 2GB -> 3.2GB; 1GB runs out of memory),
#: and the panel is identical at 6GB and 2GB. The Cronicle event has memory_limit 8GB, so 2GB
#: leaves room.
DEFAULT_MEMORY_LIMIT = "2GB"
DEFAULT_THREADS = 4
EXIT_RAW_MISSING = 31

_SNAPSHOT_DIR = re.compile(r"^snapshot_date=(\d{4}-\d{2}-\d{2})$")
_SOURCE_DIR = re.compile(r"^source=(.+)$")


class PrepareRefused(RuntimeError):
    """A guard stopped the prepare step; the CLI turns it into exit code 2."""


def panel_dir(out_root: str | Path, d: date) -> Path:
    return Path(out_root) / "panels_in" / f"D={d.isoformat()}"


def views_dir_for(out_root: str | Path, d: date) -> Path:
    # no "=" in the name: DuckDB reads "D=<date>" in a parquet path as a hive partition column
    return Path(out_root) / "views" / f"D-{d.isoformat()}"


def lake_config_for(raw_root: str | Path) -> LakeConfig:
    """``LakeConfig`` of ``<base>/raw/raw_postgres/snapshot_date=<s>/source=<x>`` (checked)."""
    p = Path(raw_root)
    m_src, m_snap = _SOURCE_DIR.match(p.name), _SNAPSHOT_DIR.match(p.parent.name)
    if not (m_src and m_snap and p.parent.parent.name == RAW_LAKE_NAME):
        raise PrepareRefused(
            f"raw root must look like <lake>/raw/{RAW_LAKE_NAME}/snapshot_date=<d>/"
            f"source=<s>, got {p}"
        )
    base = p.parent.parent.parent.parent  # <lake>/raw/raw_postgres/snapshot/source -> <lake>
    if p.parent.parent.parent.name != "raw":
        raise PrepareRefused(f"raw root is not under a <lake>/raw directory: {p}")
    cfg = LakeConfig(root=DataRoot(base=base), snapshot_date=m_snap.group(1), source=m_src.group(1))
    if Path(cfg.raw_root) != p:
        raise PrepareRefused(f"raw root does not resolve to itself: {p}")
    return cfg


def _same_path(a: str | Path, b: str | Path) -> bool:
    return Path(a).resolve() == Path(b).resolve()


def prepare_day(
    d: date,
    *,
    raw_root: str | Path,
    views_dir: str | Path,
    out_root: str | Path,
    sessions: Sequence[date],
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = DEFAULT_THREADS,
    through: str | date | None = None,
    feature_set: ds.FeatureSet = "noflow",
    universe_filter: UniverseFilter | None = None,
    modeler_dir: Path | None = None,
) -> dict[str, Any]:
    """Write ``<out_root>/panels_in/D=<d>/panel.parquet`` and ``panel_manifest.json``; write-once.

    ``through`` cuts the raw tables the panel reads at that day (reproducibility runs); it has
    to equal the views' cut. ``universe_filter`` is for tests (``None`` = the real universe).
    Returns ``{"dir", "panel", "manifest_path", "manifest"}``. Raises ``PrepareRefused``.
    """
    started = time.time()
    raw = Path(raw_root)
    try:
        t = previous_session(sessions, d)
    except ShadowError as exc:
        raise PrepareRefused(str(exc)) from exc
    if through is not None:
        through = date.fromisoformat(str(through)).isoformat()
    final = panel_dir(out_root, d)
    panel_final, manifest_final = final / PANEL_FILE, final / MANIFEST_FILE
    if panel_final.exists():
        raise PrepareRefused(f"panel already exists, not overwriting: {panel_final}")

    config = lake_config_for(raw)
    try:
        vman = views_mod.verify_views(views_dir)
    except views_mod.ViewsRefused as exc:
        raise PrepareRefused(str(exc)) from exc
    if not _same_path(vman["raw_root"], raw):
        raise PrepareRefused(
            f"the views were built from {vman['raw_root']}, not from this raw snapshot {raw}"
        )
    if vman.get("through") != through:
        raise PrepareRefused(
            f"the views are cut at {vman.get('through')!r}, this panel at {through!r}"
        )
    views_manifest_sha = views_mod.sha256_file(Path(views_dir) / views_mod.MANIFEST_NAME)

    # no "=" in the name: DuckDB would read "<name>=<value>" in the path as a hive partition column
    build = final.parent / f".build-{d.isoformat()}"
    if build.exists():
        shutil.rmtree(build)
    (build / "tmp").mkdir(parents=True)
    con = duckdb.connect()
    try:
        con.execute(f"SET threads={int(threads)}")
        con.execute(f"SET memory_limit='{memory_limit}'")
        con.execute(f"SET temp_directory='{(build / 'tmp').as_posix()}'")
        try:
            hashes = bp.register_inputs(
                con, config, Path(views_dir), labeled=False, feature_set=feature_set
            )
        except bp.BuildRefused as exc:
            raise PrepareRefused(str(exc)) from exc
        for name, sha in hashes.items():  # the files register_inputs read are the manifest's
            if sha != vman["views"][name]["sha256"]:
                raise PrepareRefused(f"derived view {name} differs from the views manifest")
        if through is not None:
            raw_tables = [
                *bp._RAW_COMMON,
                *(["krx_security_flow_raw"] if feature_set == "main" else []),
            ]
            try:
                views_mod.register_raw(con, raw, raw_tables, through=through)
            except views_mod.ViewsRefused as exc:
                raise PrepareRefused(str(exc)) from exc
        try:
            built = bp.build_panel_file(
                con,
                mode="serving",
                snapshot=config.snapshot_date,
                start=t.isoformat(),
                end=t.isoformat(),
                out_dir=build,
                feature_set=feature_set,
                views=ds.KrNextViews(),
                universe_filter=universe_filter,
                modeler_dir=modeler_dir,
            )
        except bp.BuildRefused as exc:
            raise PrepareRefused(str(exc)) from exc
    except BaseException:
        con.close()
        shutil.rmtree(build, ignore_errors=True)
        raise
    con.close()

    try:
        if built["row_count"] == 0:
            raise PrepareRefused(f"the serving panel of t={t} has no rows (raw snapshot {raw})")
        built_panel = build / bp.output_paths(build, "serving")[0].name
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "mode": "serving",
            "t": t.isoformat(),
            "D": d.isoformat(),
            "feature_set": feature_set,
            "row_count": built["row_count"],
            "columns": built["columns"],
            "panel_sha256": built["panel_sha256"],
            "panel_sql_sha256": built["panel_sql_sha256"],
            "view_sql_sha256": built["view_sql_sha256"],
            "derived_views": {
                n: {"sha256": i["sha256"], "sql_sha256": i["sql_sha256"], "rows": i["rows"]}
                for n, i in vman["views"].items()
            },
            "views_manifest_sha256": views_manifest_sha,
            "views_dir": str(Path(views_dir)),
            "raw_snapshot": str(raw),
            "raw_snapshot_id": config.snapshot_date,
            "raw_source": config.source,
            "through": through,
            "modeler_commit": built["modeler_commit"],
            "modeler_dirty": built["modeler_dirty"],
            "memory_limit": memory_limit,
            "threads": int(threads),
            "elapsed_seconds": round(time.time() - started, 1),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        final.mkdir(parents=True, exist_ok=True)
        if panel_final.exists():
            raise PrepareRefused(f"panel already exists, not overwriting: {panel_final}")
        if manifest_final.exists():  # an orphan: a run that died before it moved the panel
            manifest_final.unlink()
        with manifest_final.open("x", encoding="utf-8") as f:
            f.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        shutil.move(str(built_panel), str(panel_final))  # last: the wrapper looks for this file
    finally:
        shutil.rmtree(build, ignore_errors=True)
    return {
        "dir": final,
        "panel": panel_final,
        "manifest_path": manifest_final,
        "manifest": manifest,
    }


def prepare_from_lake(
    d: date,
    *,
    raw_lake: str | Path,
    out_root: str | Path,
    sessions: Sequence[date],
    source: str = DEFAULT_SOURCE,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = DEFAULT_THREADS,
) -> dict[str, Any]:
    """Views (built once per D from the snapshot ``snapshot_date=<d>``) and then the panel."""
    raw = Path(raw_lake) / f"snapshot_date={d.isoformat()}" / f"source={source}"
    if not raw.is_dir():
        raise FileNotFoundError(f"raw snapshot of D is missing: {raw}")
    vdir = views_dir_for(out_root, d)
    if not (vdir / views_mod.MANIFEST_NAME).exists():
        if vdir.exists():  # a build that died before its manifest: not a finished build
            shutil.rmtree(vdir)
        views_mod.build_views(raw, vdir, memory_limit=memory_limit, threads=threads)
    return prepare_day(
        d,
        raw_root=raw,
        views_dir=vdir,
        out_root=out_root,
        sessions=sessions,
        memory_limit=memory_limit,
        threads=threads,
    )


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="kr_next_shadow_prepare", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "prepare"):
        s = sub.add_parser(name)
        s.add_argument("--date", required=True, type=date.fromisoformat, help="report date D")
        s.add_argument("--out-root", required=True, type=Path, help="shadow root")
        s.add_argument("--sessions", required=True, type=Path, help="JSON list or calendar json")
        s.add_argument("--memory-limit", default=DEFAULT_MEMORY_LIMIT)
        s.add_argument("--threads", type=int, default=DEFAULT_THREADS)
        if name == "run":
            s.add_argument("--raw-lake", required=True, type=Path, help="the raw_postgres dir")
            s.add_argument("--source", default=DEFAULT_SOURCE)
        else:
            s.add_argument("--raw-root", required=True, type=Path)
            s.add_argument("--views-dir", required=True, type=Path)
            s.add_argument("--through", default=None)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = _args(argv)
    try:
        sessions = _load_sessions(a.sessions)
        if a.cmd == "run":
            res = prepare_from_lake(
                a.date,
                raw_lake=a.raw_lake,
                out_root=a.out_root,
                sessions=sessions,
                source=a.source,
                memory_limit=a.memory_limit,
                threads=a.threads,
            )
        else:
            res = prepare_day(
                a.date,
                raw_root=a.raw_root,
                views_dir=a.views_dir,
                out_root=a.out_root,
                sessions=sessions,
                memory_limit=a.memory_limit,
                threads=a.threads,
                through=a.through,
            )
    except FileNotFoundError as exc:
        print(f"kr_next_shadow_prepare: {exc}", file=sys.stderr)
        return EXIT_RAW_MISSING
    except (PrepareRefused, views_mod.ViewsRefused, ValueError, OSError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    m = res["manifest"]
    print(
        json.dumps(
            {
                "panel": str(res["panel"]),
                "t": m["t"],
                "D": m["D"],
                "rows": m["row_count"],
                "panel_sha256": m["panel_sha256"],
                "elapsed_seconds": m["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
