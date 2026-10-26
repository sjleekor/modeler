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
snapshot missing, 33 the price rows of t are incomplete (see below).

Completeness of t (this model's own input check, frozen here; the briefing's kr-prepare is not
consulted): the day's raw snapshot must hold at least ``MIN_PRICE_TICKER_RATIO`` (0.97) as many
``daily_ohlcv`` rows on t as on the session before t. Flows are not checked (the model does not
use them). When it fails nothing is built, ``<out_root>/skips/D=<D>.json`` is written once
(``reason`` = ``t_price_incomplete``) and the exit code is 33: the day is left out, nothing is
substituted. The definition is a copy of ``modeler.serving.kr_reference.assess_session`` (price
part): ``count(*)`` of ``daily_ohlcv`` rows per ``trade_date``, halted rows included, no
filter; copied so later changes of the briefing code cannot move this frozen rule.
"""

from __future__ import annotations

import argparse
import json
import os
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
EXIT_T_INCOMPLETE = 33
#: Frozen copy of ``kr_reference.DEFAULT_MIN_TICKER_RATIO`` (= the collector gate's value).
MIN_PRICE_TICKER_RATIO = 0.97
SKIP_REASON = "t_price_incomplete"
SKIPS_DIR = "skips"

_SNAPSHOT_DIR = re.compile(r"^snapshot_date=(\d{4}-\d{2}-\d{2})$")
_SOURCE_DIR = re.compile(r"^source=(.+)$")


class PrepareRefused(RuntimeError):
    """A guard stopped the prepare step; the CLI turns it into exit code 2."""


class PriceIncomplete(PrepareRefused):
    """t has too few price rows against the session before it; the day is left out (exit 33)."""

    def __init__(self, record: dict[str, Any]) -> None:
        self.record = record
        super().__init__(
            f"price rows of t={record['t']} incomplete: {record['n_t']} vs "
            f"{record['n_prev']} on {record['prev']} (ratio {record['ratio']}, "
            f"threshold {record['threshold']})"
        )


def price_counts(raw_root: str | Path, days: Sequence[date]) -> dict[date, int]:
    """``count(*)`` of ``daily_ohlcv`` rows per trade_date for ``days`` (same as kr_reference)."""
    pattern = str(Path(raw_root) / "daily_ohlcv" / "**" / "*.parquet").replace("'", "''")
    lits = ", ".join(f"DATE '{x.isoformat()}'" for x in days)
    con = duckdb.connect()
    try:
        con.execute("SET threads = 2")
        rows = con.execute(
            f"SELECT trade_date, count(*) FROM read_parquet('{pattern}', hive_partitioning=false) "
            f"WHERE trade_date IN ({lits}) GROUP BY trade_date"
        ).fetchall()
    except duckdb.IOException as exc:
        raise FileNotFoundError(f"daily_ohlcv cannot be read under {raw_root}") from exc
    finally:
        con.close()
    return {r[0]: int(r[1]) for r in rows}


def check_price_complete(
    d: date,
    raw_root: str | Path,
    sessions: Sequence[date],
    *,
    threshold: float = MIN_PRICE_TICKER_RATIO,
) -> dict[str, Any]:
    """Raise ``PriceIncomplete`` unless n_t / n_prev >= threshold; else return the numbers.

    prev = the KR session before t (calendar). No previous session or no rows on prev means the
    ratio cannot be judged, which kr_reference also rejects.
    """
    try:
        t = previous_session(sessions, d)
    except ShadowError as exc:
        raise PrepareRefused(str(exc)) from exc
    try:
        prev: date | None = previous_session(sessions, t)
    except ShadowError:
        prev = None
    counts = price_counts(raw_root, [t] + ([prev] if prev else []))
    n_t = counts.get(t, 0)
    n_prev = counts.get(prev, 0) if prev else 0
    ratio = n_t / n_prev if n_prev else None
    record = {
        "D": d.isoformat(),
        "t": t.isoformat(),
        "prev": prev.isoformat() if prev else None,
        "n_t": n_t,
        "n_prev": n_prev,
        "ratio": round(ratio, 6) if ratio is not None else None,
        "threshold": threshold,
        "reason": SKIP_REASON,
        "raw": str(raw_root),
    }
    if ratio is None or ratio < threshold:
        raise PriceIncomplete(record)
    return record


def write_skip(out_root: str | Path, record: dict[str, Any]) -> Path:
    """Write ``<out_root>/skips/D=<D>.json`` once; an existing file is never overwritten."""
    d = Path(out_root) / SKIPS_DIR
    d.mkdir(parents=True, exist_ok=True)
    final = d / f"D={record['D']}.json"
    tmp = d / f".D={record['D']}.json.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        os.link(tmp, final)  # a hard link never overwrites
    except FileExistsError:
        pass
    finally:
        tmp.unlink(missing_ok=True)
    return final


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
    check_price_complete(d, raw, sessions)  # raises PriceIncomplete (exit 33)
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
    check_price_complete(d, raw, sessions)  # before the views are built (cheap, exit 33)
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
    except PriceIncomplete as exc:
        path = write_skip(a.out_root, exc.record)
        print(f"kr_next_shadow_prepare: {exc}; day left out, see {path}", file=sys.stderr)
        return EXIT_T_INCOMPLETE
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
