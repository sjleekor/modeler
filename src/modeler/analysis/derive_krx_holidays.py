"""F-9.2 — derive KRX trading holidays 2014-2023 from the raw lake.

``src/collector/kr/infra/calendar/data/holidays_krx.csv`` only covers 2024-2026, so for 2014-2023 the
``common_feature_daily_fact`` grid is every *weekday* — holidays included,
carrying the previous session's value. Counting ``t-20`` or ``t-252`` on that
grid therefore means a different window length either side of 2024, which is
why the regime work had to route around it on the KRX session grid
(``03_stage1b`` §2.1). The workaround works; every other consumer that rolls
the fact directly falls into the same trap.

No new collection is needed. ``daily_market_cap`` (KRX Open API, 2014-06-02-)
has **zero rows on a non-session day** — the opposite of pykrx, which returns a
whole market of price-zero rows. So a weekday absent from it is a KRX holiday.
Before 2014-06-02 the same logic runs on ``daily_ohlcv``, where a holiday is a
day *every* ticker is missing rather than one where a few are halted or
delisted.

Two guards, because "no rows" can also mean "we never collected that day":

* the derived set is cross-checked against the 2024-2026 rows the CSV already
  has, and the script fails if it does not reproduce them;
* a run of more than ``MAX_RUN_DAYS`` consecutive missing weekdays is reported
  as a suspected collection gap rather than emitted as holidays.

Example::

    uv run python -m modeler.analysis.derive_krx_holidays \
        --snapshot-date 2026-08-23 --source sj2_remote --write
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import logging
from pathlib import Path

import duckdb

from modeler.etl.config import DataRoot, DEFAULT_SOURCE, EngineOptions, LakeConfig
from modeler.etl.lake import connect, register_views

logger = logging.getLogger(__name__)

CSV_PATH = Path("src/collector/kr/infra/calendar/data/holidays_krx.csv")
#: daily_market_cap's first session; before this the OHLCV fallback is used.
MARKET_CAP_START = dt.date(2014, 6, 2)
#: A longer run of missing weekdays is a suspected collection gap, not a
#: holiday sequence. The real maximum measured is 6 -- 2017-10-02..10-09, when
#: 10-02 was declared a temporary holiday and joined National Foundation Day,
#: Chuseok and Hangul Day into a ten-day break. 8 is the first value that
#: cannot be a holiday cluster, and the range clamp below is what actually
#: keeps an uncollected tail from reaching this check.
MAX_RUN_DAYS = 8
DERIVED_NAME = "Derived (no KRX session)"


def _session_dates(con: duckdb.DuckDBPyConnection) -> set[dt.date]:
    """Every date that had a KRX session, from both sources."""
    rows = con.execute("""
        SELECT DISTINCT trade_date FROM daily_market_cap
        UNION
        SELECT DISTINCT trade_date FROM daily_ohlcv
          WHERE NOT (open = 0 AND high = 0 AND low = 0)
        """).fetchall()
    return {r[0] for r in rows}


def derive(
    con: duckdb.DuckDBPyConnection, start: dt.date, end: dt.date
) -> tuple[list[dt.date], list[tuple[dt.date, dt.date, int]]]:
    """Return (holiday weekdays, suspected collection gaps) over [start, end]."""
    sessions = _session_dates(con)
    missing: list[dt.date] = []
    day = start
    while day <= end:
        if day.weekday() < 5 and day not in sessions:
            missing.append(day)
        day += dt.timedelta(days=1)

    # Split into consecutive weekday runs so an over-long run can be flagged.
    holidays: list[dt.date] = []
    gaps: list[tuple[dt.date, dt.date, int]] = []
    run: list[dt.date] = []

    def _flush() -> None:
        if not run:
            return
        if len(run) >= MAX_RUN_DAYS:
            gaps.append((run[0], run[-1], len(run)))
        else:
            holidays.extend(run)

    for day in missing:
        if run:
            prev = run[-1]
            step = day - prev
            bridged = step.days <= 3 and prev.weekday() == 4 and day.weekday() == 0
            if step.days != 1 and not bridged:
                _flush()
                run = []
        run.append(day)
    _flush()
    return holidays, gaps


def read_csv() -> dict[dt.date, str]:
    if not CSV_PATH.is_file():
        return {}
    with CSV_PATH.open(encoding="utf-8") as f:
        return {
            dt.date.fromisoformat(row["date"]): row["name"]
            for row in csv.DictReader(f)
            if row.get("date")
        }


def write_csv(entries: dict[dt.date, str]) -> None:
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        # csv.writer defaults to \r\n; this repository's file is LF.
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["date", "name"])
        for day in sorted(entries):
            w.writerow([day.isoformat(), entries[day]])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2026-12-31")
    parser.add_argument("--write", action="store_true", help="append to the CSV")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.snapshot_date is None:
        parser.error("--snapshot-date is required")
    config = LakeConfig(
        root=DataRoot.resolve(market="kr"),
        snapshot_date=args.snapshot_date,
        source=args.source or DEFAULT_SOURCE,
        engine=EngineOptions(threads=4, memory_limit="4GB"),
    )
    con = connect(config)
    register_views(con, config, tables=["daily_market_cap", "daily_ohlcv"])

    existing = read_csv()
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)

    # Clamp to the data. Past the last session in the lake every weekday is
    # "missing" for the trivial reason that nothing was collected yet, and
    # emitting those as holidays would put 94 fictitious closures into the
    # calendar. The run-length guard would catch it, but clamping is the honest
    # fix: the question is only answerable inside the collected range.
    last_session = max(_session_dates(con))
    if end > last_session:
        print(f"clamping end {end} -> {last_session} (last session in the lake)")
        end = last_session

    # Guard 1: reproduce the years the CSV already covers.
    if existing:
        check_from = min(existing)
        check_to = min(max(existing), max(_session_dates(con)))
        # Weekdays only. The CSV also lists holidays that fell on a weekend
        # (2024-02-10/11, 2025-03-01, 2025-10-05), which the calendar never
        # needs and this derivation cannot produce -- a weekend is already a
        # non-trading day. Comparing against them would fail the guard for a
        # reason that has nothing to do with the data.
        known = {d for d in existing if check_from <= d <= check_to and d.weekday() < 5}
        derived_known, _ = derive(con, check_from, check_to)
        derived_known_set = set(derived_known)
        missed = sorted(known - derived_known_set)
        extra = sorted(derived_known_set - known)
        print(
            f"cross-check {check_from} .. {check_to}: "
            f"CSV has {len(known)}, derived {len(derived_known_set)}"
        )
        if missed:
            print(f"  MISSED (in CSV, not derived): {[d.isoformat() for d in missed]}")
        if extra:
            print(f"  EXTRA (derived, not in CSV):  {[d.isoformat() for d in extra]}")
        if missed:
            print("cross-check FAILED; refusing to write")
            return 1
        # `extra` is not a failure: the CSV is hand-maintained and was found to
        # be missing four real weekday closures (2024-10-01 Armed Forces Day
        # substitute, 2025-01-27 Lunar New Year substitute, 2025-06-03 election
        # day, 2025-12-31 KRX year-end close). Those get added like any other.
        print("  cross-check OK (derived covers every CSV weekday)")

    holidays, gaps = derive(con, start, end)
    print(f"\nderived {len(holidays)} holiday weekdays over {start} .. {end}")
    by_year: dict[int, int] = {}
    for d in holidays:
        by_year[d.year] = by_year.get(d.year, 0) + 1
    for y in sorted(by_year):
        print(f"  {y}: {by_year[y]}")
    if gaps:
        print(f"\nsuspected collection gaps (>= {MAX_RUN_DAYS} consecutive weekdays), NOT emitted:")
        for a, b, n in gaps:
            print(f"  {a} .. {b}  ({n} weekdays)")

    if not args.write:
        print("\n(dry run; pass --write to append)")
        return 0

    merged = dict(existing)
    added = 0
    for d in holidays:
        if d not in merged:
            merged[d] = DERIVED_NAME
            added += 1
    write_csv(merged)
    print(f"\nwrote {CSV_PATH}: {len(existing)} -> {len(merged)} rows (+{added})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
