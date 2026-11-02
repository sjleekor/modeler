"""The P6′ view folder: P6's price/PIT views as they are, the v4p financial mart under P6's name.

Decision O-7 (2026-11-02): P6′ reads a new view folder,
``<lake>/derived/kr_next_p6prime_views/snapshot_date=<S>/``, through the P6 panel builder's
``--views-dir``. The folder holds the files the P6 builder reads (``DERIVED_VIEW_FILES``):

* ``dim_stock_pit_daily.parquet``, ``dim_price_quality_daily.parquet`` -- **copied byte for byte**
  from the P6 view folder of the same raw snapshot (sha256 checked against the P6 manifest before
  and after the copy);
* ``fin_quarterly_metric_vintage_dlr.parquet`` -- the P6′ mart
  ``fin_quarterly_metric_vintage_dlr_v4p`` built here from the raw snapshot, under the P6 file
  name so the P6 assembly reads it without a code change;
* ``stock_metric_vintage_fact_dl_v3.parquet`` -- the v3 delisted-keeping vintage fact the v4p mart
  is built from (not read by the panel; kept for audit and rebuilds, like P6's ``_dl`` file).

Build chain (all repository code, no SQL of its own): raw tables
(``_03_kr_next.views.register_raw``) ->
``metric_vintages_delisted.materialize_stock_metric_vintage_fact_dl(semantics="v3",
plan="staged")`` -> ``financial_quarters_v4p.materialize_fin_quarterly_metric_vintage_dlr_v4p``
-> the folder. The fact's session calendar is the one the research build used: sessions from the
first ``daily_ohlcv`` date to :data:`CALENDAR_SESSIONS_AFTER_ANCHOR` sessions after the last one,
KRX holidays from the collector CSV.

The manifest has the P6 view-manifest keys (``schema`` ``kr-next-views.v1``, ``views`` with file,
sha256, bytes, rows and SQL hash per file, so ``_03_kr_next.views.verify_views`` works on it) and
a ``p6prime`` block (P6 folder and manifest hash, mart cache metadata, the pins of the draft §2 and
whether they match). Write-once: built in a work directory next to the target, moved in, manifest
last. Nothing here reads a label or a return.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb

from modeler.etl.config import DataRoot, EngineOptions, LakeConfig
from modeler.etl.lake import connect
from modeler.etl.mart import mart_table_dir
from modeler.etl.marts import financial_quarters_v4p as fq4p
from modeler.etl.marts import metric_vintages as mv
from modeler.etl.marts import metric_vintages_delisted as dl
from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import views as p6_views
from modeler.models._04_kr_next_p6prime import tags

VIEWS_DIRNAME = "kr_next_p6prime_views"
P6_VIEWS_DIRNAME = "kr_next_views"
MANIFEST_NAME = p6_views.MANIFEST_NAME
SCHEMA = p6_views.SCHEMA
FACT_SEMANTICS = "v3"
FACT_PLAN = "staged"
FACT_TABLE = mv.semantics_table_name(dl.SMVF_DL_TABLE, FACT_SEMANTICS)
#: the v4p mart is written under the file name the P6 panel builder reads
FIN_FILE = dl.FQMV_DLR_TABLE
COPIED_FROM_P6: tuple[str, ...] = (p6_views.PIT_VIEW, p6_views.QUALITY_TABLE)
VIEW_FILES: tuple[str, ...] = (*COPIED_FROM_P6, FACT_TABLE, FIN_FILE)
RAW_TABLES = p6_views.RAW_FOR_VIEWS
CALENDAR_SESSIONS_AFTER_ANCHOR = 10
#: what a rebuild of the preregistration draft §2 inputs (raw 2026-10-23) must reproduce:
#: SQL/schema/plan hashes, the session calendar and the order-independent content digests of
#: the research files (:func:`content_digest`, DuckDB 1.5.5, measured 2026-11-02).
#: ``--require-pins`` enforces these.
PINS: dict[str, str] = {
    "fact_sql_hash": "b648b31539164e73d33737d0cb339ddaa70f6bf50b3024cdfef988f5b4bbf112",
    "fact_content_digest": "23365a8e571d564ef06199e721ad80c51e767d62be50592444da5199c998412d",
    "fin_sql_hash": "4c4dc337fc28d94e9682e84ebdecebeb19aeb14726375b2bd52317c7e282b00c",
    "fin_schema_hash": "98a24a727ac4ade845642205cac5f0dfe49383a86a19c91c35c163e8eb7d75d0",
    "fin_plan_hash": "4e4798634081b327b377a3d72abe747d3f652a82500e27dda6732291ddc9cc2e",
    "fin_content_digest": "37adf05378b4761d702bb0cc909b6996fa874d0bee2030423ffe476bab0cb7c3",
    "calendar_sha256": "2ad05c327341f3811c44194d75e38f353d3f433e4e35910b9472bff96bda724e",
}
#: file sha256 of the research builds pinned in the draft §2. Recorded for comparison only: the
#: parallel write orders rows differently from run to run, so a rebuild has other file bytes.
RESEARCH_FILE_SHA256: dict[str, str] = {
    "fact_sha256": "3617426a7f2d3d00a4acb554776128da8a277a47b7170ee91ce1ef7415182ad2",
    "fin_sha256": "e277f5aeed7c48dfaf40b11e6a19c863cf2c53c45b187eac4150f12ce1c6b0c4",
}


class InputsRefused(RuntimeError):
    """A guard stopped the build (existing output, missing input, P6 file not as recorded)."""


def default_views_dir(lake: str | Path, snapshot: str) -> Path:
    return Path(lake) / "derived" / VIEWS_DIRNAME / f"snapshot_date={snapshot}"


def default_p6_views_dir(lake: str | Path, snapshot: str) -> Path:
    return Path(lake) / "derived" / P6_VIEWS_DIRNAME / f"snapshot_date={snapshot}"


def calendar_sha256(days: Sequence[date]) -> str:
    return hashlib.sha256(",".join(d.isoformat() for d in days).encode()).hexdigest()


def session_calendar(
    first: date, anchor: date, holidays: set[date], *, after: int = CALENDAR_SESSIONS_AFTER_ANCHOR
) -> list[date]:
    """KRX sessions from ``first`` through the ``after``-th session after ``anchor``.

    Stops the build when the holiday CSV does not reach that far (later holidays are unknown).
    """
    from collector.kr.shared import get_trading_days

    coverage_end = date(max(holidays).year, 12, 31)
    days = list(
        get_trading_days(first, min(anchor + timedelta(days=after * 3 + 14), coverage_end),
                         holidays=holidays)
    )  # fmt: skip
    head = [d for d in days if d <= anchor]
    tail = [d for d in days if d > anchor][:after]
    if len(tail) < after:
        raise InputsRefused(
            f"fewer than {after} sessions after {anchor} inside the holiday CSV (ends "
            f"{coverage_end})"
        )
    return head + tail


def _snapshot_of(raw_root: Path) -> tuple[str, str]:
    snap, source = raw_root.parent.name, raw_root.name
    if not snap.startswith("snapshot_date=") or not source.startswith("source="):
        raise InputsRefused(f"raw root must end in snapshot_date=<d>/source=<s>: {raw_root}")
    return snap.split("=", 1)[1], source.split("=", 1)[1]


def check_outputs_free(out_dir: str | Path) -> None:
    out = Path(out_dir)
    taken = [str(p) for p in (*(out / f"{n}.parquet" for n in VIEW_FILES), out / MANIFEST_NAME)]
    taken = [p for p in taken if Path(p).exists()]
    if taken:
        raise InputsRefused(f"output already exists, not overwriting: {', '.join(taken)}")


def check_p6_folder(p6_views_dir: Path) -> tuple[dict[str, Any], str]:
    """Verify every file of the P6 view folder against its manifest; returns (manifest, sha)."""
    try:
        man = p6_views.verify_views(p6_views_dir)
    except p6_views.ViewsRefused as e:
        raise InputsRefused(f"P6 view folder is not as recorded: {e}") from e
    missing = [n for n in COPIED_FROM_P6 if n not in man["views"]]
    if missing:
        raise InputsRefused(f"P6 view manifest lacks {missing}")
    return man, p6_views.sha256_file(p6_views_dir / MANIFEST_NAME)


def compare_pins(
    manifest: Mapping[str, Any], pins: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """``{pin: {"expected", "actual", "same"}}`` for a P6′ view manifest.

    Default: :data:`PINS` plus :data:`RESEARCH_FILE_SHA256` (the latter informational).
    """
    v = manifest["views"]
    p = manifest["p6prime"]
    actual = {
        "fact_sha256": v[FACT_TABLE]["sha256"],
        "fact_sql_hash": p["fact_cache_metadata"].get("sql_hash"),
        "fact_content_digest": v[FACT_TABLE].get("content_digest"),
        "fin_sha256": v[FIN_FILE]["sha256"],
        "fin_sql_hash": p["fin_cache_metadata"].get("sql_hash"),
        "fin_schema_hash": p["fin_cache_metadata"].get("schema_hash"),
        "fin_plan_hash": p["fin_cache_metadata"].get("plan_hash"),
        "fin_content_digest": v[FIN_FILE].get("content_digest"),
        "calendar_sha256": manifest["calendar"]["sessions_sha256"],
    }
    want = dict(pins) if pins is not None else {**PINS, **RESEARCH_FILE_SHA256}
    return {
        k: {"expected": want[k], "actual": actual[k], "same": want[k] == actual[k]} for k in want
    }


def _count(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    lit = "'" + str(path).replace("'", "''") + "'"
    return int(con.execute(f"SELECT count(*) FROM read_parquet({lit})").fetchone()[0])


def content_digest(con: duckdb.DuckDBPyConnection, path: str | Path) -> str:
    """Order-independent digest of a parquet file's rows (column names, types and values).

    The marts are written by a parallel ``COPY`` without ``ORDER BY``, so two builds of the same
    rows can differ in row order and therefore in file bytes and sha256. This digest is the same
    for the same multiset of rows: sha256 over the sorted md5 of each row's text, prefixed by the
    schema. Values are rendered by DuckDB's ``CAST(... AS VARCHAR)``; compare digests made with
    the same DuckDB version.
    """
    lit = "'" + str(path).replace("'", "''") + "'"
    rel = f"read_parquet({lit}, hive_partitioning=false)"
    schema = con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()
    cols = ", ".join(
        'COALESCE(CAST("{}" AS VARCHAR), chr(0))'.format(r[0].replace('"', '""')) for r in schema
    )
    row = con.execute(
        f"SELECT sha256(string_agg(h, '' ORDER BY h)) FROM "
        f"(SELECT md5(concat_ws(chr(31), {cols})) AS h FROM {rel})"
    ).fetchone()
    head = json.dumps([[r[0], r[1]] for r in schema])
    return hashlib.sha256((head + (row[0] or "")).encode()).hexdigest()


def build_p6prime_views(
    raw_root: str | Path,
    p6_views_dir: str | Path,
    out_dir: str | Path,
    *,
    memory_limit: str = "10GB",
    threads: int = 8,
    holiday_csv: Path | None = None,
    require_pins: bool = False,
    modeler_dir: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build the P6′ view folder (write-once) and return its manifest.

    ``raw_root`` = ``.../raw_postgres/snapshot_date=<S>/source=<s>``; ``p6_views_dir`` = the P6
    view folder of the same ``S``. ``require_pins`` refuses (before anything is moved into
    ``out_dir``) when a pin of :data:`PINS` differs.
    """
    log = progress or (lambda _m: None)
    raw, p6_dir, out = Path(raw_root), Path(p6_views_dir), Path(out_dir)
    if not raw.is_dir():
        raise InputsRefused(f"raw root is not a directory: {raw}")
    snapshot, source = _snapshot_of(raw)
    check_outputs_free(out)
    p6_man, p6_man_sha = check_p6_folder(p6_dir)
    if p6_man.get("raw_snapshot") != f"snapshot_date={snapshot}" or p6_man.get("through"):
        raise InputsRefused(
            f"P6 view folder is of {p6_man.get('raw_snapshot')} through={p6_man.get('through')},"
            f" not of the whole snapshot_date={snapshot}"
        )
    holidays, holiday_info = p6_views.load_holidays(holiday_csv)
    log(f"P6 folder verified ({p6_dir}); building the v3 fact and v4p from {raw}")

    work = out.with_name(f".{out.name}.work")
    if work.exists():
        shutil.rmtree(work)
    files_dir = work / "files"
    files_dir.mkdir(parents=True)
    cfg = LakeConfig(
        root=DataRoot(base=work / "lake"),
        snapshot_date=snapshot,
        source=source,
        engine=EngineOptions(
            threads=int(threads), memory_limit=memory_limit, temp_directory=str(work / "tmp")
        ),
    )
    try:
        con = connect(cfg)
        try:
            p6_views.register_raw(con, raw, RAW_TABLES)
            first, anchor = con.execute(
                "SELECT min(trade_date), max(trade_date) FROM daily_ohlcv"
            ).fetchone()
            if first is None:
                raise InputsRefused("daily_ohlcv has no rows")
            days = session_calendar(first, anchor, holidays)
            started = datetime.now(UTC)
            dl.materialize_stock_metric_vintage_fact_dl(
                con,
                cfg,
                trading_days=days,
                semantics=FACT_SEMANTICS,
                plan=FACT_PLAN,
                stage_dir=work / "stages",
            )
            log(f"{FACT_TABLE} built")
            fq4p.materialize_fin_quarterly_metric_vintage_dlr_v4p(con, cfg, vintage_view=FACT_TABLE)
            seconds = round((datetime.now(UTC) - started).total_seconds(), 1)
            log(f"{fq4p.FQMV_DLR_V4P_TABLE} built ({seconds}s for both)")
            built: dict[str, Path] = {}
            meta: dict[str, dict[str, Any]] = {}
            for name, table in ((FACT_TABLE, FACT_TABLE), (FIN_FILE, fq4p.FQMV_DLR_V4P_TABLE)):
                tdir = mart_table_dir(cfg, table)
                parts = sorted(tdir.glob("*.parquet"))
                if [p.name for p in parts] != ["part-000000.parquet"]:
                    raise InputsRefused(f"{table}: expected one part file, got {parts}")
                meta[name] = json.loads((tdir / "_cache_metadata.json").read_text("utf-8"))
                built[name] = files_dir / f"{name}.parquet"
                shutil.move(str(parts[0]), str(built[name]))
            rows = {name: _count(con, path) for name, path in built.items()}
            digests = {name: content_digest(con, path) for name, path in built.items()}
            log("content digests computed")
        finally:
            con.close()

        for name in COPIED_FROM_P6:
            src = p6_dir / p6_man["views"][name]["file"]
            dst = files_dir / f"{name}.parquet"
            shutil.copyfile(src, dst)
            if p6_views.sha256_file(dst) != p6_man["views"][name]["sha256"]:
                raise InputsRefused(f"copy of {src} does not match the P6 manifest sha256")

        view_info: dict[str, Any] = {}
        for name in COPIED_FROM_P6:
            info = dict(p6_man["views"][name])
            view_info[name] = {**info, "file": f"{name}.parquet", "copied_from_p6": True}
        for name in (FACT_TABLE, FIN_FILE):
            f = built[name]
            view_info[name] = {
                "file": f.name,
                "sha256": p6_views.sha256_file(f),
                "bytes": f.stat().st_size,
                "rows": rows[name],
                "sql_sha256": meta[name]["sql_hash"],
                "content_digest": digests[name],
            }
        git = bundle_mod.git_state(modeler_dir or Path(__file__).resolve().parent)
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "raw_root": str(raw),
            "raw_snapshot": raw.parent.name,
            "raw_source": raw.name,
            "raw_tables": p6_views.raw_inventory(raw, RAW_TABLES),
            "through": None,
            "cuts": {t: None for t in RAW_TABLES},
            "quality_pit": p6_man.get("quality_pit"),
            "vintage": {
                "semantics": FACT_SEMANTICS,
                "plan": FACT_PLAN,
                "plan_hash": meta[FACT_TABLE].get("plan_hash"),
            },
            "views": view_info,
            "modeler_commit": git["commit"],
            "modeler_dirty": git["dirty"],
            "duckdb_version": duckdb.__version__,
            "memory_limit": memory_limit,
            "threads": int(threads),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "quality_pit_sql_sha256": p6_man.get("quality_pit_sql_sha256"),
            "calendar": {
                "first": days[0].isoformat(),
                "last": days[-1].isoformat(),
                "sessions": len(days),
                "sessions_sha256": calendar_sha256(days),
                "rule": f"first daily_ohlcv date .. {CALENDAR_SESSIONS_AFTER_ANCHOR} sessions "
                f"after the last ({anchor.isoformat()})",
                "holiday_csv": holiday_info,
            },
            "p6prime": {
                "prereg_tag": tags.PREREG_TAG,
                "fin_file_holds": fq4p.FQMV_DLR_V4P_TABLE,
                "p6_views_dir": str(p6_dir),
                "p6_manifest_sha256": p6_man_sha,
                "copied_from_p6": list(COPIED_FROM_P6),
                "fact_cache_metadata": meta[FACT_TABLE],
                "fin_cache_metadata": meta[FIN_FILE],
                "functions": [
                    "metric_vintages_delisted.materialize_stock_metric_vintage_fact_dl("
                    f"semantics={FACT_SEMANTICS!r}, plan={FACT_PLAN!r})",
                    "financial_quarters_v4p.materialize_fin_quarterly_metric_vintage_dlr_v4p("
                    f"vintage_view={FACT_TABLE!r}, filing_receipt_view="
                    f"{fq4p.DEFAULT_RECEIPT_VIEW!r})",
                ],
                "build_seconds": seconds,
            },
        }
        manifest["p6prime"]["pins"] = compare_pins(manifest)
        bad = {k: v for k, v in manifest["p6prime"]["pins"].items() if k in PINS and not v["same"]}
        manifest["p6prime"]["pins_ok"] = not bad
        if require_pins and bad:
            raise InputsRefused(f"pins differ: {json.dumps(bad)}")

        out.mkdir(parents=True, exist_ok=True)
        check_outputs_free(out)
        for name in VIEW_FILES:
            shutil.move(str(files_dir / f"{name}.parquet"), str(out / f"{name}.parquet"))
        with (out / MANIFEST_NAME).open("x", encoding="utf-8") as fh:
            fh.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return manifest


def load_p6prime_manifest(views_dir: str | Path) -> dict[str, Any]:
    """The manifest of a finished P6′ view folder, every file re-hashed.

    Refuses a folder that is not a P6′ folder (no ``p6prime`` block naming the v4p mart) -- in
    particular the P6 folder itself.
    """
    try:
        man = p6_views.verify_views(views_dir)
    except p6_views.ViewsRefused as e:
        raise InputsRefused(str(e)) from e
    block = man.get("p6prime") or {}
    if block.get("fin_file_holds") != fq4p.FQMV_DLR_V4P_TABLE:
        raise InputsRefused(f"not a P6′ view folder (no p6prime block): {views_dir}")
    return man


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m modeler.models._04_kr_next_p6prime.inputs",
        description="Build the P6′ view folder (P6 price/PIT views + v4p), write-once",
    )
    ap.add_argument("--lake", required=True, help="stock_data lake root (kr)")
    ap.add_argument("--snapshot", required=True, help="raw snapshot_date S")
    ap.add_argument("--source", default="sj2_remote")
    ap.add_argument("--p6-views-dir", default=None, help="default <lake>/derived/kr_next_views")
    ap.add_argument("--out-dir", default=None, help="default <lake>/derived/kr_next_p6prime_views")
    ap.add_argument("--memory-limit", default="10GB")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--require-pins", action="store_true", help="refuse unless §2 pins match")
    args = ap.parse_args(argv)
    lake = Path(args.lake)
    raw = lake / "raw" / "raw_postgres" / f"snapshot_date={args.snapshot}" / f"source={args.source}"
    try:
        m = build_p6prime_views(
            raw,
            args.p6_views_dir or default_p6_views_dir(lake, args.snapshot),
            args.out_dir or default_views_dir(lake, args.snapshot),
            memory_limit=args.memory_limit,
            threads=args.threads,
            require_pins=args.require_pins,
            progress=lambda s: print(s, flush=True),
        )
    except (InputsRefused, p6_views.ViewsRefused, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: v["rows"] for k, v in m["views"].items()}, indent=2))
    print(json.dumps({k: v["same"] for k, v in m["p6prime"]["pins"].items()}, indent=2))
    print(f"pins_ok (PINS, research file bytes excluded): {m['p6prime']['pins_ok']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
