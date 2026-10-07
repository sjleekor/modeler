"""Bounded KR/US session manifests with explicit provenance and exceptions.

The US list is transcribed from NYSE's 2026/2027 equity calendar. The KR
list uses the project's dated KRX holiday file through 2027; 2027 is provisional
(statutory rules cross-checked against exchange_calendars XKRX on 2026-10-07,
before KRX's 2027 notice). Unknown future KR dates are outside coverage rather
than guessed from weekdays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from modeler.serving.calendars import SessionCalendar

NYSE_SOURCE = "https://www.nyse.com/trade/hours-calendars"
KRX_RULES_SOURCE = "https://global.krx.co.kr/contents/GLB/06/0602/0602020204/GLB0602020204T1.jsp"
PINNED_KRX_HOLIDAY_SHA256 = "9b62b5956c9185773cdd999fad3306bbf35b2d34d0eb71ce31bdb8b9a6a10538"
KASA_2026_SOURCE = "https://www.kasa.go.kr/prog/bbsArticle/BBSMSTR_000000000010/view.do?bbsId=BBSMSTR_000000000010&nttId=B000000001860Pe2zT3"
KRX_2026_CALENDAR_SOURCE = "https://kind.krx.co.kr/external/dst/reference/11625/2026%20%EC%BD%94%EC%8A%A4%EB%8B%A5%EC%8B%9C%EC%9E%A5%20%EA%B3%B5%EC%8B%9C%EC%9D%BC%EC%A0%95%20%EC%BA%98%EB%A6%B0%EB%8D%94_vF.pdf"
HOLIDAY_LAW_2026_SOURCE = "https://www.law.go.kr/LSW/lsLinkCommonInfo.do?lsJoLnkSeq=1033028569"
JULY_HOLIDAY_2026_SOURCE = "https://www.mois.go.kr/video/bbs/type019/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000255&nttId=123641&searchCode1="
ELECTION_2026_SOURCE = "https://www.mois.go.kr/frt/bbs/type010/commonSelectBoardArticle.do?bbsId=BBSMSTR_000000000008&nttId=126486"
KR_HOLIDAYS_2026 = frozenset(date.fromisoformat(day) for day in (
    "2026-01-01", "2026-02-16", "2026-02-17", "2026-02-18",
    "2026-03-02", "2026-05-01", "2026-05-05", "2026-05-25",
    "2026-06-03", "2026-07-17", "2026-08-17", "2026-09-24",
    "2026-09-25", "2026-10-05", "2026-10-09", "2026-12-25",
    "2026-12-31",
))
#: 2027 weekday closures, provisional until KRX publishes its 2027 notice (December).
#: Lunar New Year 02-07 (Sun) -> 02-08 and substitute 02-09; Liberation Day 08-15 (Sun) -> 08-16;
#: National Foundation Day 10-03 (Sun) -> 10-04; Hangeul Day 10-09 (Sat) -> 10-11;
#: Christmas 12-25 (Sat) -> 12-27; year-end closure 12-31.
KR_HOLIDAYS_2027 = frozenset(date.fromisoformat(day) for day in (
    "2027-01-01", "2027-02-08", "2027-02-09", "2027-03-01",
    "2027-05-05", "2027-05-13", "2027-08-16", "2027-09-14",
    "2027-09-15", "2027-09-16", "2027-10-04", "2027-10-11",
    "2027-12-27", "2027-12-31",
))
_KR_WEEKDAY_HOLIDAYS = {2026: KR_HOLIDAYS_2026, 2027: KR_HOLIDAYS_2027}
#: First session of the year and CSAT day can open late; no exchange notice is on file.
_KR_UNCONFIRMED_HOURS = frozenset({"2026-01-02", "2026-11-19", "2027-01-04", "2027-11-18"})

_NYSE_HOLIDAYS = frozenset(date.fromisoformat(day) for day in (
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
))
_NYSE_EARLY_CLOSES = frozenset(date.fromisoformat(day) for day in (
    "2026-11-27", "2026-12-24", "2027-11-26",
))


def _weekdays(start: date, end: date, closed: set[date] | frozenset[date]) -> list[str]:
    days: list[str] = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5 and cursor not in closed:
            days.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return days


def nyse_equity_manifest(start: date, end: date) -> dict[str, Any]:
    """Use NYSE's published 2026/2027 holidays and 13:00 early closes."""
    if start < date(2026, 1, 1) or end > date(2027, 12, 31) or start > end:
        raise ValueError("NYSE official calendar coverage is 2026-01-01 through 2027-12-31")
    sessions = _weekdays(start, end, _NYSE_HOLIDAYS)
    overrides = [
        {"date": day.isoformat(), "open_at": "09:30:00", "close_at": "13:00:00", "confirmed": True}
        for day in sorted(_NYSE_EARLY_CLOSES)
        if start <= day <= end
    ]
    manifest = {
        "market": "US", "timezone": "America/New_York",
        "coverage_start": start.isoformat(), "coverage_end": end.isoformat(),
        "default_open_at": "09:30:00", "default_close_at": "16:00:00",
        "sessions": sessions, "overrides": overrides, "unconfirmed_dates": [],
        "source": {"url": NYSE_SOURCE, "checked_on": "2026-09-30", "basis": "NYSE equities published holiday and early-close table"},
    }
    SessionCalendar.from_manifest(manifest)
    return manifest


def publish_calendar_manifest(manifest: dict[str, Any], *, output_dir: Path) -> Path:
    """Publish one immutable calendar revision by content hash."""
    raw = (json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"calendar-{manifest['market']}-{digest[:16]}.json"
    if target.exists():
        if target.read_bytes() != raw:
            raise FileExistsError("calendar revision hash collision")
        return target
    descriptor, name = tempfile.mkstemp(prefix=".calendar-", dir=output_dir)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, target)
    finally:
        Path(name).unlink(missing_ok=True)
    return target


def coverage_status(manifest: dict[str, Any], *, as_of: date,
                    warning_days: int = 60) -> dict[str, Any]:
    """Warn before a bounded published manifest stops covering report dates."""
    if warning_days < 0:
        raise ValueError("warning_days must not be negative")
    end = date.fromisoformat(str(manifest["coverage_end"]))
    remaining = (end - as_of).days
    status = "expired" if remaining < 0 else (
        "renewal_due" if remaining <= warning_days else "ok"
    )
    return {"status": status, "coverage_end": end.isoformat(),
            "as_of": as_of.isoformat(), "days_remaining": remaining,
            "warning_days": warning_days}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market", choices=("KR", "US"), required=True)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--holiday-csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today(),
                        help="coverage 경고를 계산할 현재 D (기본값: 오늘)")
    args = parser.parse_args(argv)
    if args.market == "KR":
        if args.holiday_csv is None:
            parser.error("--holiday-csv is required for KR")
        manifest = krx_project_manifest(args.start, args.end, holiday_csv=args.holiday_csv)
    else:
        manifest = nyse_equity_manifest(args.start, args.end)
    path = publish_calendar_manifest(manifest, output_dir=args.output_dir)
    print(json.dumps({"path": str(path), "market": args.market,
                      "sessions": len(manifest["sessions"]),
                      "unconfirmed_dates": manifest["unconfirmed_dates"],
                      "coverage": coverage_status(manifest, as_of=args.as_of)}, sort_keys=True))
    return 0


def krx_project_manifest(
    start: date, end: date, *, holiday_csv: Path,
) -> dict[str, Any]:
    """Read dated project holidays, requiring evidence for exceptional hours.

    The project file is only an integrity-pinned input. Its 2026 weekdays are
    separately checked against government holidays and KRX market rules; 2027
    weekdays are checked against the provisional statutory list above.
    Dated exceptional hours require their own exchange notice.
    """
    if start < date(2026, 1, 1) or end > date(2027, 12, 31) or start > end:
        raise ValueError("KR project holiday file coverage is bounded to 2026-2027")
    raw = holiday_csv.read_bytes()
    holiday_sha = hashlib.sha256(raw).hexdigest()
    if holiday_sha != PINNED_KRX_HOLIDAY_SHA256:
        raise ValueError("KR 2026-2027 holiday CSV does not match the pinned complete file")
    with holiday_csv.open(newline="") as stream:
        holidays = {date.fromisoformat(row["date"]) for row in csv.DictReader(stream)}
    for year, reviewed in _KR_WEEKDAY_HOLIDAYS.items():
        actual = {day for day in holidays if day.year == year and day.weekday() < 5}
        if actual != reviewed:
            raise ValueError(f"KR {year} weekday closures differ from reviewed calendar")
    sessions = _weekdays(start, end, holidays)
    # Annual closures and regular hours have primary-source support. The
    # first trading day and CSAT can shift the opening hour separately.
    unconfirmed = [day for day in sessions if day in _KR_UNCONFIRMED_HOURS]
    manifest = {
        "market": "KR", "timezone": "Asia/Seoul",
        "coverage_start": start.isoformat(), "coverage_end": end.isoformat(),
        "default_open_at": "09:00:00", "default_close_at": "15:30:00",
        "sessions": sessions, "overrides": [], "unconfirmed_dates": unconfirmed,
        "source": {"url": KRX_RULES_SOURCE, "holiday_csv_sha256": holiday_sha,
                   "holiday_source_status": (
                       "project_derived_integrity_pinned_2026_primary_crosschecked_2027_provisional_pre_notice"
                       if end.year >= 2027 else
                       "project_derived_integrity_pinned_and_2026_weekday_dates_primary_crosschecked"),
                   "checked_on": "2026-09-30",
                   "primary_sources": [KASA_2026_SOURCE, HOLIDAY_LAW_2026_SOURCE,
                                       JULY_HOLIDAY_2026_SOURCE, ELECTION_2026_SOURCE,
                                       KRX_2026_CALENDAR_SOURCE, KRX_RULES_SOURCE],
                   "basis": "2026 government holiday calendar, subsequent public-holiday amendments, election day, KRX year-end closure and regular hours; dated exceptional hours unverified"},
    }
    SessionCalendar.from_manifest(manifest)
    return manifest


if __name__ == "__main__":
    raise SystemExit(main())
