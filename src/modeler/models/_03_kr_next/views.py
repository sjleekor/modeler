"""Derived views the next KR model's panel reads but does not build itself (G-13, G-22).

``build_panel`` takes four derived marts as parquet files (``DERIVED_VIEW_FILES``): the
price-quality mart, the non-empty PIT view and the delisted-keeping vintage marts. The lake
holds none of them as raw tables, so a prepare step has to build them from one raw export
snapshot. This module does that with the existing builders; it writes no SQL of its own.

Files (``<out_dir>/<name>.parquet``) and their builders:

* ``dim_stock_pit_daily``: ``stock_pit_nonempty.build_stock_pit_nonempty_sql``
  (``PLAN_ASOF_INTERVALS``, as the briefing serving).
* ``dim_price_quality_daily``: ``quality.build_price_quality_sql`` over the frozen PIT text
  (``build_stock_pit_sql``, the lake's ``dim_stock_pit_daily``). Default
  ``quality_pit="serving"``, as every measurement so far. ``"nonempty"`` builds it over the
  non-empty PIT instead; 1 of 2,060 rows of the 2025-07-30 panel differs.
* ``stock_metric_vintage_fact_dl``: ``build_stock_metric_vintage_fact_dl_plan`` (v1 semantics,
  staged plan), KRX calendar from the collector's holiday CSV.
* ``fin_quarterly_metric_vintage_dlr``: ``build_fin_quarterly_metric_vintage_dlr_sql`` over the
  ``_dl`` file.

``build_panel`` reads three of the four files (``DERIVED_VIEW_FILES``);
``stock_metric_vintage_fact_dl`` is the input of the ``_dlr`` mart; kept for audit and rebuilds.

``through`` cuts every dated raw table at that day (rows a database would have held at its close,
the same predicates as ``serving.kr_live_prepare.cut_predicate`` plus the KRX cap and flow tables)
so a view build can be repeated on a historical date; ``None`` takes the snapshot whole.

Output is write-once: files are built in a work directory next to ``out_dir`` and moved in; the
manifest (``manifest.json``, exclusive create) is written last, so a directory without it is not a
finished build. Nothing here reads a label or a return.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, date, datetime, timedelta
from importlib.resources import files
from pathlib import Path
from typing import Any

import duckdb

from modeler.etl.mart import _write_with_plan
from modeler.etl.marts import metric_vintages as _mv
from modeler.etl.marts.metric_vintages_delisted import (
    FQMV_DLR_TABLE,
    SMVF_DL_TABLE,
    build_fin_quarterly_metric_vintage_dlr_sql,
    build_stock_metric_vintage_fact_dl_plan,
)
from modeler.etl.quality import QUALITY_TABLE, build_price_quality_sql
from modeler.etl.stock_pit import PLAN_ASOF_INTERVALS, build_stock_pit_sql
from modeler.etl.stock_pit_nonempty import build_stock_pit_nonempty_sql
from modeler.models._03_kr_next import bundle as bundle_mod

SCHEMA = "kr-next-views.v1"
MANIFEST_NAME = "manifest.json"
PIT_VIEW = "dim_stock_pit_daily"
#: files written, in build order (``build_panel`` reads all but ``stock_metric_vintage_fact_dl``)
VIEW_FILES: tuple[str, ...] = (
    PIT_VIEW,
    QUALITY_TABLE,
    SMVF_DL_TABLE,
    FQMV_DLR_TABLE,
)
#: raw tables the four builders read
RAW_FOR_VIEWS: tuple[str, ...] = (
    "daily_ohlcv",
    "dart_share_count_raw",
    "dart_financial_statement_raw",
    "dart_shareholder_return_raw",
    "dart_xbrl_fact_raw",
    "dart_filing_receipt_raw",
    "dart_corp_master",
    "stock_master",
)
CALENDAR_LOOKAHEAD_DAYS = 45  # same reach as the research builds (cut + 45 days)
QUALITY_PITS = ("nonempty", "serving")

_RCEPT_DATE_SQL = (
    "TRY_STRPTIME(NULLIF(SUBSTR(CAST(rcept_no AS VARCHAR), 1, 8), ''), '%Y%m%d')::DATE"
)
_TRADE_DATE_TABLES = ("daily_ohlcv", "daily_market_cap", "krx_security_flow_raw")
_RCEPT_TABLES = (
    "dart_financial_statement_raw",
    "dart_share_count_raw",
    "dart_shareholder_return_raw",
    "dart_xbrl_fact_raw",
)
_UNCUT_TABLES = ("dart_corp_master", "stock_master")
_RECEIPT_TABLE = "dart_filing_receipt_raw"
HOLIDAY_CSV = "collector/kr/infra/calendar/data/holidays_krx.csv"


class ViewsRefused(RuntimeError):
    """A guard stopped the build (existing output, missing input, bad argument)."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _lit(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


# --------------------------------------------------------------------------- raw registration


def cut_predicate(table: str, through: str | date) -> str | None:
    """SQL keeping the rows present at the close of ``through``; None = the table is not cut."""
    day = f"DATE '{date.fromisoformat(str(through)).isoformat()}'"
    if table in _TRADE_DATE_TABLES:
        return f"trade_date <= {day}"
    if table in _RCEPT_TABLES:
        return f"COALESCE({_RCEPT_DATE_SQL} <= {day}, TRUE)"
    if table == _RECEIPT_TABLE:
        return f"COALESCE({_RCEPT_DATE_SQL} <= {day}, TRUE) AND COALESCE(rcept_dt <= {day}, TRUE)"
    if table in _UNCUT_TABLES:
        return None
    raise ViewsRefused(f"no cut is defined for raw table {table!r}; refusing to leave it uncut")


def raw_table_glob(raw_root: Path, table: str) -> str:
    return str(raw_root / table / "**" / "*.parquet")


def register_raw(
    con: duckdb.DuckDBPyConnection,
    raw_root: str | Path,
    tables: Iterable[str],
    *,
    through: str | date | None = None,
) -> dict[str, str | None]:
    """One view per raw table over ``<raw_root>/<table>/**/*.parquet`` (hive off),
    cut at ``through``.

    Returns ``{table: predicate or None}``. A table without parquet files raises ``ViewsRefused``.
    """
    root = Path(raw_root)
    cuts: dict[str, str | None] = {}
    for table in tables:
        glob = raw_table_glob(root, table)
        n = con.execute("SELECT count(*) FROM glob(?)", [glob]).fetchone()[0]
        if not n:
            raise ViewsRefused(f"no parquet files for raw table {table!r} at {glob}")
        predicate = None if through is None else cut_predicate(table, through)
        where = f" WHERE {predicate}" if predicate else ""
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS "
            f"SELECT * FROM read_parquet({_lit(glob)}, hive_partitioning=false){where}"
        )
        cuts[table] = predicate
    return cuts


def raw_inventory(raw_root: Path, tables: Iterable[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for table in tables:
        fs = [p for p in (raw_root / table).rglob("*.parquet") if p.is_file()]
        out[table] = {"files": len(fs), "bytes": sum(p.stat().st_size for p in fs)}
    return out


# --------------------------------------------------------------------------- calendar


def load_holidays(path: Path | None = None) -> tuple[set[date], dict[str, Any]]:
    """KRX holidays from the collector CSV; a missing or empty file stops the build.

    (The collector loader falls back to weekends only, which would count holidays as sessions.)
    """
    target = path or Path(str(files("collector.kr.infra.calendar") / "data" / "holidays_krx.csv"))
    if not target.is_file():
        raise ViewsRefused(f"KRX holiday calendar CSV is missing: {target}")
    holidays: set[date] = set()
    with target.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                holidays.add(date.fromisoformat(row["date"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ViewsRefused(f"KRX holiday calendar has an invalid row: {row!r}") from exc
    if not holidays:
        raise ViewsRefused(f"KRX holiday calendar has no rows: {target}")
    return holidays, {
        "name": HOLIDAY_CSV if path is None else target.name,
        "sha256": sha256_file(target),
        "rows": len(holidays),
        "max_date": max(holidays).isoformat(),
    }


def vintage_calendar(
    lower: date, upper: date, holidays: set[date], *, lookahead: int = CALENDAR_LOOKAHEAD_DAYS
) -> list[date]:
    """Sessions from ``lower`` to ``upper + lookahead`` days, stopping at the CSV's last year.

    The vintage mart looks up the first session after each receipt date, so the calendar has to
    reach past the last price date. It never goes beyond the end of the last holiday year (later
    holidays are unknown); if not even the day after ``upper`` fits, the build stops.
    """
    from collector.kr.shared import get_trading_days

    coverage_end = date(max(holidays).year, 12, 31)
    end = min(upper + timedelta(days=lookahead), coverage_end)
    days = list(get_trading_days(lower, end, holidays=holidays))
    if not any(d > upper for d in days):
        raise ViewsRefused(
            f"no session after {upper} inside the holiday CSV coverage (ends {coverage_end})"
        )
    return days


# --------------------------------------------------------------------------- build


def _paths(out_dir: Path) -> tuple[list[Path], Path]:
    return [out_dir / f"{n}.parquet" for n in VIEW_FILES], out_dir / MANIFEST_NAME


def check_outputs_free(out_dir: str | Path) -> None:
    files_, manifest = _paths(Path(out_dir))
    taken = [str(p) for p in (*files_, manifest) if p.exists()]
    if taken:
        raise ViewsRefused(f"output already exists, not overwriting: {', '.join(taken)}")


def _copy(con: duckdb.DuckDBPyConnection, sql: str, target: Path) -> int:
    con.execute(f"COPY ({sql}) TO {_lit(target)} (FORMAT PARQUET, COMPRESSION ZSTD)")
    return int(con.execute(f"SELECT count(*) FROM read_parquet({_lit(target)})").fetchone()[0])


def _bind(con: duckdb.DuckDBPyConnection, name: str, path: Path) -> None:
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({_lit(path)})")


def build_views(
    raw_root: str | Path,
    out_dir: str | Path,
    *,
    through: str | date | None = None,
    memory_limit: str = "2GB",
    threads: int = 4,
    quality_pit: str = "serving",
    holiday_csv: Path | None = None,
    modeler_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the four derived view files from one raw export snapshot; write-once.

    ``raw_root`` = ``.../raw_postgres/snapshot_date=<d>/source=<s>``. Returns the manifest that
    is also written to ``<out_dir>/manifest.json``. Raises ``ViewsRefused`` for an existing
    output, a missing raw table or holiday CSV, an unknown ``quality_pit``.
    """
    raw = Path(raw_root)
    out = Path(out_dir)
    if quality_pit not in QUALITY_PITS:
        raise ViewsRefused(f"quality_pit must be one of {QUALITY_PITS}, got {quality_pit!r}")
    if through is not None:
        through = date.fromisoformat(str(through)).isoformat()
    if not raw.is_dir():
        raise ViewsRefused(f"raw root is not a directory: {raw}")
    check_outputs_free(out)
    holidays, holiday_info = load_holidays(holiday_csv)

    work = out.with_name(f".{out.name}.work")
    if work.exists():
        shutil.rmtree(work)
    (work / "tmp").mkdir(parents=True)
    con = duckdb.connect()
    sql_hashes: dict[str, str] = {}
    rows: dict[str, int] = {}
    extra: dict[str, Any] = {}
    try:
        con.execute(f"SET threads={int(threads)}")
        con.execute(f"SET memory_limit={_lit(memory_limit)}")
        con.execute(f"SET temp_directory={_lit(work / 'tmp')}")
        cuts = register_raw(con, raw, RAW_FOR_VIEWS, through=through)

        # 1. PIT without empty share filings (asof_intervals plan)
        pit_sql = build_stock_pit_nonempty_sql(plan=PLAN_ASOF_INTERVALS)
        pit_file = work / f"{PIT_VIEW}.parquet"
        sql_hashes[PIT_VIEW] = sha256_text(pit_sql)
        rows[PIT_VIEW] = _copy(con, pit_sql, pit_file)

        # 2. price quality over the PIT view named dim_stock_pit_daily
        if quality_pit == "serving":
            serving_sql = build_stock_pit_sql(plan=PLAN_ASOF_INTERVALS)
            serving_file = work / "pit_serving_for_quality.parquet"
            _copy(con, serving_sql, serving_file)
            _bind(con, PIT_VIEW, serving_file)
            extra["quality_pit_sql_sha256"] = sha256_text(serving_sql)
        else:
            _bind(con, PIT_VIEW, pit_file)
        quality_sql = build_price_quality_sql()
        quality_file = work / f"{QUALITY_TABLE}.parquet"
        sql_hashes[QUALITY_TABLE] = sha256_text(quality_sql)
        rows[QUALITY_TABLE] = _copy(con, quality_sql, quality_file)

        # 3. delisted-keeping vintage fact (v1, staged) on the KRX session calendar
        lo, hi = con.execute("SELECT min(trade_date), max(trade_date) FROM daily_ohlcv").fetchone()
        if lo is None:
            raise ViewsRefused("daily_ohlcv has no rows (after the cut)")
        days = vintage_calendar(lo, hi, holidays)
        _mv._register_calendar(con, days)
        plan = build_stock_metric_vintage_fact_dl_plan(semantics="v1", plan="staged")
        smvf_file = work / f"{SMVF_DL_TABLE}.parquet"
        _write_with_plan(
            con,
            plan,
            SMVF_DL_TABLE,
            _lit(smvf_file),
            ["FORMAT PARQUET", "COMPRESSION ZSTD"],
            work / "stages",
        )
        sql_hashes[SMVF_DL_TABLE] = plan.plan_hash
        rows[SMVF_DL_TABLE] = int(
            con.execute(f"SELECT count(*) FROM read_parquet({_lit(smvf_file)})").fetchone()[0]
        )
        _bind(con, SMVF_DL_TABLE, smvf_file)

        # 4. quarterly vintage with rule R1
        dlr_sql = build_fin_quarterly_metric_vintage_dlr_sql(
            vintage_view=SMVF_DL_TABLE, semantics="v1"
        )
        dlr_file = work / f"{FQMV_DLR_TABLE}.parquet"
        sql_hashes[FQMV_DLR_TABLE] = sha256_text(dlr_sql)
        rows[FQMV_DLR_TABLE] = _copy(con, dlr_sql, dlr_file)
        extra["calendar"] = {
            "first": days[0].isoformat(),
            "last": days[-1].isoformat(),
            "sessions": len(days),
            "holiday_csv": holiday_info,
        }
    except BaseException:
        con.close()
        shutil.rmtree(work, ignore_errors=True)
        raise
    con.close()

    # move in (write-once), manifest last
    out.mkdir(parents=True, exist_ok=True)
    try:
        check_outputs_free(out)
        view_info: dict[str, Any] = {}
        for name in VIEW_FILES:
            src = work / f"{name}.parquet"
            dst = out / f"{name}.parquet"
            shutil.move(str(src), str(dst))
            view_info[name] = {
                "file": dst.name,
                "sha256": sha256_file(dst),
                "bytes": dst.stat().st_size,
                "rows": rows[name],
                "sql_sha256": sql_hashes[name],
            }
        git = bundle_mod.git_state(modeler_dir or Path(__file__).resolve().parent)
        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "raw_root": str(raw),
            "raw_snapshot": raw.parent.name,
            "raw_source": raw.name,
            "raw_tables": raw_inventory(raw, RAW_FOR_VIEWS),
            "through": through,
            "cuts": {k: v for k, v in cuts.items()},
            "quality_pit": quality_pit,
            "vintage": {"semantics": "v1", "plan": "staged", "plan_hash": plan.plan_hash},
            "views": view_info,
            "modeler_commit": git["commit"],
            "modeler_dirty": git["dirty"],
            "duckdb_version": duckdb.__version__,
            "memory_limit": memory_limit,
            "threads": int(threads),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            **extra,
        }
        with (out / MANIFEST_NAME).open("x", encoding="utf-8") as f:
            f.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return manifest


def load_manifest(views_dir: str | Path) -> dict[str, Any]:
    path = Path(views_dir) / MANIFEST_NAME
    if not path.is_file():
        raise ViewsRefused(f"views manifest missing (not a finished build): {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_views(
    views_dir: str | Path, manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Recompute each view file's sha256; ``ViewsRefused`` on any difference from the manifest."""
    vd = Path(views_dir)
    man = dict(manifest) if manifest is not None else load_manifest(vd)
    for name, info in man["views"].items():
        f = vd / info["file"]
        if not f.is_file():
            raise ViewsRefused(f"view file missing: {f}")
        if sha256_file(f) != info["sha256"]:
            raise ViewsRefused(f"view file differs from its manifest sha256: {f}")
    return man


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m modeler.models._03_kr_next.views",
        description="Build the derived views the next KR panel reads, write-once",
    )
    ap.add_argument("--raw-root", required=True, help=".../snapshot_date=<d>/source=<s>")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--through", default=None, help="cut the raw data at YYYY-MM-DD")
    ap.add_argument("--memory-limit", default="2GB")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quality-pit", default="serving", choices=QUALITY_PITS)
    args = ap.parse_args(argv)
    try:
        m = build_views(
            args.raw_root,
            args.out_dir,
            through=args.through,
            memory_limit=args.memory_limit,
            threads=args.threads,
            quality_pit=args.quality_pit,
        )
    except (ViewsRefused, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({k: v["rows"] for k, v in m["views"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
