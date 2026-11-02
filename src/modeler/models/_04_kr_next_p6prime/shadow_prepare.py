"""P6′ shadow prepare: the serving panel of D from the P6′ view folder of D's raw snapshot.

Same contract as ``_03_kr_next.shadow_prepare`` (``panels_in/D=<D>/panel.parquet`` +
``panel_manifest.json``, write-once, t = the session before D, the frozen t-completeness rule
with exit 33 and ``skips/D=<D>.json``, exit 31 for a missing raw snapshot, 2 for a refusal), and
the panel is built by that module's ``prepare_day`` -- only the view folder differs:

1. ``<out_root>/views_p6/D-<D>/``: the P6 view build (``_03_kr_next.views.build_views``) of the
   day's raw snapshot. P6′ needs it because its price/PIT/quality files are P6's, copied byte for
   byte (draft §1.1 "가격·PIT·품질 뷰: 같은 파일"). It is built inside the P6′ root; the P6
   shadow root is not read.
2. ``<out_root>/views/D-<D>/``: the P6′ view folder (``inputs.build_p6prime_views``): the P6
   PIT and quality files copied, the v3 delisted-keeping vintage fact and the v4p mart under
   the P6 file name ``fin_quarterly_metric_vintage_dlr.parquet``. Its session calendar is the
   frozen research rule (first ``daily_ohlcv`` date .. 10 sessions after the snapshot's last
   price date, KRX holiday CSV), so on the freeze snapshot it reproduces the §2 pins
   (``p6prime.pins`` in its manifest; on other days the content pins differ by nature and the
   SQL/schema/plan pins must still hold).
3. ``panels_in/D=<D>/``: first the sidecar ``p6prime_panel.json`` (which folder, its manifest
   hash, the v4p hashes, the pins, this package's code hashes), then the P6
   ``prepare_day`` with ``views_dir`` = the P6′ folder. ``prepare_day`` re-hashes every view
   file against the folder manifest and records them in ``panel_manifest.json``.

``out_root`` must be a P6′ path (a component contains ``p6prime``, :func:`.shadow.require_p6prime_root`),
so the P6 shadow root (panels, views, skips) is never written. A finished folder of step 1 or 2
is reused after its hashes check out; a folder without a manifest is a dead build and is rebuilt.

Commands (same as P6)::

    python -m modeler.models._04_kr_next_p6prime.shadow_prepare run --date D \
        --raw-lake <raw_postgres> --out-root <P6′ shadow root> --sessions <calendar json>
    python -m modeler.models._04_kr_next_p6prime.shadow_prepare prepare --date D \
        --raw-root <snapshot> --views-dir <P6′ view folder> --out-root <root> --sessions <json>

Nothing here reads a label or a return.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from modeler.etl.universe import UniverseFilter
from modeler.models._03_kr_next import shadow as p6_shadow
from modeler.models._03_kr_next import shadow_prepare as p6
from modeler.models._03_kr_next import views as p6_views
from modeler.models._03_kr_next.shadow import _atomic_write_bytes, _json_dumps, _load_sessions
from modeler.models._04_kr_next_p6prime import inputs, tags
from modeler.models._04_kr_next_p6prime import shadow as shadow_mod

PANEL_FILE = p6.PANEL_FILE
MANIFEST_FILE = p6.MANIFEST_FILE
SIDECAR_FILE = shadow_mod.SIDECAR_FILE
DEFAULT_SOURCE = p6.DEFAULT_SOURCE
DEFAULT_MEMORY_LIMIT = p6.DEFAULT_MEMORY_LIMIT
DEFAULT_THREADS = p6.DEFAULT_THREADS
EXIT_RAW_MISSING = p6.EXIT_RAW_MISSING
EXIT_T_INCOMPLETE = p6.EXIT_T_INCOMPLETE
P6_VIEWS_DIRNAME = "views_p6"

PrepareRefused = p6.PrepareRefused
PriceIncomplete = p6.PriceIncomplete


def panel_dir(out_root: str | Path, d: date) -> Path:
    return p6.panel_dir(out_root, d)


def views_dir_for(out_root: str | Path, d: date) -> Path:
    """The P6′ view folder of D (``<out_root>/views/D-<D>``, the P6 layout)."""
    return p6.views_dir_for(out_root, d)


def p6_views_dir_for(out_root: str | Path, d: date) -> Path:
    """The P6 view build of D's snapshot inside the P6′ root (``views_p6/D-<D>``)."""
    return Path(out_root) / P6_VIEWS_DIRNAME / f"D-{d.isoformat()}"


def _root(out_root: str | Path) -> Path:
    try:
        return shadow_mod.require_p6prime_root(out_root)
    except shadow_mod.P6RootRefused as exc:
        raise PrepareRefused(str(exc)) from exc


def sidecar_for(views_dir: Path, vman: dict[str, Any], d: date) -> dict[str, Any]:
    """The P6′ sidecar of a panel built from the P6′ folder ``views_dir`` (manifest ``vman``)."""
    block = vman["p6prime"]
    views = vman["views"]
    pins = block.get("pins") or {}
    cfiles = shadow_mod.code_files()
    return {
        "schema": shadow_mod.SIDECAR_SCHEMA,
        "model": shadow_mod.MODEL,
        "prereg_tag": tags.PREREG_TAG,
        "bundle_tag": tags.BUNDLE_TAG,
        "D": d.isoformat(),
        "views_dir": str(views_dir),
        "views_manifest_sha256": p6_views.sha256_file(views_dir / inputs.MANIFEST_NAME),
        "fin_file_holds": block.get("fin_file_holds"),
        "fin_sha256": views[inputs.FIN_FILE]["sha256"],
        "fin_rows": views[inputs.FIN_FILE]["rows"],
        "fin_content_digest": views[inputs.FIN_FILE].get("content_digest"),
        "fact_sha256": views[inputs.FACT_TABLE]["sha256"],
        "fact_rows": views[inputs.FACT_TABLE]["rows"],
        "fact_content_digest": views[inputs.FACT_TABLE].get("content_digest"),
        "fin_cache_metadata": block.get("fin_cache_metadata"),
        "fact_sql_hash": (block.get("fact_cache_metadata") or {}).get("sql_hash"),
        "p6_views_dir": block.get("p6_views_dir"),
        "p6_views_manifest_sha256": block.get("p6_manifest_sha256"),
        "copied_from_p6": {n: views[n]["sha256"] for n in inputs.COPIED_FROM_P6},
        "calendar": vman.get("calendar"),
        "pins": {k: v.get("same") for k, v in pins.items()},
        "pins_ok": block.get("pins_ok"),
        "build_seconds": block.get("build_seconds"),
        "code": {"files": cfiles, "inventory_sha256": p6_shadow.code_inventory_sha256(cfiles)},
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def prepare_day(
    d: date,
    *,
    raw_root: str | Path,
    views_dir: str | Path,
    out_root: str | Path,
    sessions: Sequence[date],
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = DEFAULT_THREADS,
    universe_filter: UniverseFilter | None = None,
    modeler_dir: Path | None = None,
) -> dict[str, Any]:
    """Sidecar, then the P6 ``prepare_day`` on the P6′ folder; returns P6's dict + sidecar.

    Refuses (``PrepareRefused``) a non-P6′ ``out_root``, a folder that is not a P6′ folder (the
    P6 folder in particular) and an existing panel; the P6 function adds its own guards.
    """
    root = _root(out_root)
    vdir = Path(views_dir)
    try:
        vman = inputs.load_p6prime_manifest(vdir)
    except inputs.InputsRefused as exc:
        raise PrepareRefused(str(exc)) from exc
    final = panel_dir(root, d)
    if (final / PANEL_FILE).exists():
        raise PrepareRefused(f"panel already exists, not overwriting: {final / PANEL_FILE}")
    sidecar = sidecar_for(vdir, vman, d)
    final.mkdir(parents=True, exist_ok=True)
    # before the panel: a panel never exists without its sidecar (a failed run leaves only the
    # sidecar, which the next attempt replaces)
    _atomic_write_bytes(
        final / SIDECAR_FILE, (_json_dumps(sidecar, indent=2) + "\n").encode("utf-8")
    )
    res = p6.prepare_day(
        d,
        raw_root=raw_root,
        views_dir=vdir,
        out_root=root,
        sessions=sessions,
        memory_limit=memory_limit,
        threads=threads,
        feature_set="noflow",
        universe_filter=universe_filter,
        modeler_dir=modeler_dir,
    )
    return {**res, "sidecar_path": final / SIDECAR_FILE, "sidecar": sidecar}


def _finished(folder: Path) -> bool:
    if (folder / inputs.MANIFEST_NAME).exists():
        return True
    if folder.exists():  # a build that died before its manifest: not a finished build
        shutil.rmtree(folder)
    return False


def ensure_views(
    d: date,
    raw: Path,
    out_root: Path,
    *,
    memory_limit: str = DEFAULT_MEMORY_LIMIT,
    threads: int = DEFAULT_THREADS,
) -> dict[str, Any]:
    """Build (or reuse) the P6 and the P6′ view folders of D; returns paths and seconds."""
    p6_dir, p6p_dir = p6_views_dir_for(out_root, d), views_dir_for(out_root, d)
    seconds: dict[str, float] = {}
    started = time.time()
    if not _finished(p6_dir):
        p6_views.build_views(raw, p6_dir, memory_limit=memory_limit, threads=threads)
    seconds["p6_views"] = round(time.time() - started, 1)
    started = time.time()
    if not _finished(p6p_dir):
        try:
            inputs.build_p6prime_views(raw, p6_dir, p6p_dir, memory_limit=memory_limit,
                                       threads=threads)  # fmt: skip
        except inputs.InputsRefused as exc:
            raise PrepareRefused(str(exc)) from exc
    seconds["p6prime_views"] = round(time.time() - started, 1)
    try:
        vman = inputs.load_p6prime_manifest(p6p_dir)
        _, p6_sha = inputs.check_p6_folder(p6_dir)
    except inputs.InputsRefused as exc:
        raise PrepareRefused(str(exc)) from exc
    block = vman["p6prime"]
    if block.get("p6_manifest_sha256") != p6_sha or Path(block["p6_views_dir"]) != p6_dir:
        raise PrepareRefused(f"P6′ folder {p6p_dir} was not built from {p6_dir}")
    return {"p6_views_dir": p6_dir, "views_dir": p6p_dir, "seconds": seconds}


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
    """P6 views, P6′ views (once per D, from ``snapshot_date=<d>``), then the panel."""
    root = _root(out_root)
    raw = Path(raw_lake) / f"snapshot_date={d.isoformat()}" / f"source={source}"
    if not raw.is_dir():
        raise FileNotFoundError(f"raw snapshot of D is missing: {raw}")
    p6.check_price_complete(d, raw, sessions)  # before any view is built (cheap, exit 33)
    started = time.time()
    built = ensure_views(d, raw, root, memory_limit=memory_limit, threads=threads)
    views_seconds = round(time.time() - started, 1)
    res = prepare_day(
        d,
        raw_root=raw,
        views_dir=built["views_dir"],
        out_root=root,
        sessions=sessions,
        memory_limit=memory_limit,
        threads=threads,
    )
    return {**res, "views_seconds": views_seconds, "views_seconds_by_step": built["seconds"]}


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="kr_next_p6prime_shadow_prepare", description=__doc__.split("\n")[0]
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "prepare"):
        s = sub.add_parser(name)
        s.add_argument("--date", required=True, type=date.fromisoformat, help="report date D")
        s.add_argument("--out-root", required=True, type=Path, help="P6′ shadow root")
        s.add_argument("--sessions", required=True, type=Path, help="JSON list or calendar json")
        s.add_argument("--memory-limit", default=DEFAULT_MEMORY_LIMIT)
        s.add_argument("--threads", type=int, default=DEFAULT_THREADS)
        if name == "run":
            s.add_argument("--raw-lake", required=True, type=Path, help="the raw_postgres dir")
            s.add_argument("--source", default=DEFAULT_SOURCE)
        else:
            s.add_argument("--raw-root", required=True, type=Path)
            s.add_argument("--views-dir", required=True, type=Path, help="P6′ view folder")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    a = _args(argv)
    try:
        _root(a.out_root)  # before anything, including a skip file, is written
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
            )
    except FileNotFoundError as exc:
        print(f"kr_next_p6prime_shadow_prepare: {exc}", file=sys.stderr)
        return EXIT_RAW_MISSING
    except PriceIncomplete as exc:
        path = p6.write_skip(a.out_root, exc.record)
        print(f"kr_next_p6prime_shadow_prepare: {exc}; day left out, see {path}", file=sys.stderr)
        return EXIT_T_INCOMPLETE
    except (PrepareRefused, p6_views.ViewsRefused, inputs.InputsRefused, ValueError,
            OSError) as exc:  # fmt: skip
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    m, sc = res["manifest"], res["sidecar"]
    print(
        json.dumps(
            {
                "panel": str(res["panel"]),
                "t": m["t"],
                "D": m["D"],
                "rows": m["row_count"],
                "panel_sha256": m["panel_sha256"],
                "elapsed_seconds": m["elapsed_seconds"],
                "views_seconds": res.get("views_seconds"),
                "views_seconds_by_step": res.get("views_seconds_by_step"),
                "fin_file_holds": sc["fin_file_holds"],
                "fin_rows": sc["fin_rows"],
                "pins_ok": sc["pins_ok"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
