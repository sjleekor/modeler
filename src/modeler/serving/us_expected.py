"""US expected-session (E) table generator for the daily briefing.

Policy draft 1 (provisional, decisions_20260930 item 3): for report date D the
expected US session is the last XNYS session that had already closed when the
``sdc_daily_us`` collection of D-1 started (15:00 KST). It assumes the daily
derive (15:30 KST) publishes that session before D 09:30. The output is the
``us-expected-source.v1`` file read by ``modeler.serving.daily_inputs``.

``daily_inputs`` only accepts ``confirmed`` (or ``synthetic_fixture``) sources,
so this generator writes ``unreviewed`` by default: the US market stays
``unavailable`` until a person deliberately regenerates with ``--reviewed-status
confirmed``. Only Korean trading days appear in the table; KR holidays are left out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import date, datetime, time, timedelta
from importlib.resources import files
from pathlib import Path
from typing import Any

from modeler.serving.calendar_sources import krx_project_manifest, nyse_equity_manifest
from modeler.serving.calendars import SEOUL, SessionCalendar

SCHEMA_VERSION = "us-expected-source.v1"
RULE_NAME = "prev_day_1500_kst_last_closed_xnys"
COLLECTION_AT_KST = time(15, 0)
DEFAULT_MARKET_LAG_LIMIT = 2
NYSE_LOOKBACK_DAYS = 14
REVIEWED_STATUSES = ("unreviewed", "confirmed")
POLICY_REFERENCE = "my/milestones/common/20260929_daily_market_briefing/01_implementation/decisions_20260930.md#3"


def default_holiday_csv() -> Path:
    return Path(str(files("collector.kr.infra.calendar") / "data" / "holidays_krx.csv"))


def _manifest_sha256(manifest: dict[str, Any]) -> str:
    raw = (json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def expected_session(us_calendar: SessionCalendar, report_date: date,
                     collection_at: time = COLLECTION_AT_KST) -> date:
    """Last XNYS session already closed at D-1 `collection_at` KST.

    Raises when the calendar cannot answer, instead of guessing.
    """
    instant = datetime.combine(report_date - timedelta(days=1), collection_at, SEOUL)
    found = us_calendar.latest_completed_before(instant)
    if found is None:
        raise ValueError(f"US calendar coverage cannot resolve E for {report_date.isoformat()}")
    if us_calendar.session(found) is None or not us_calendar.session(found).confirmed:
        raise ValueError(f"US session {found.isoformat()} is not a confirmed session")
    return found


def close_kst(us_calendar: SessionCalendar, session_day: date) -> datetime:
    """Close instant of an XNYS session, computed in New York time, shown in KST."""
    session = us_calendar.session(session_day)
    if session is None:
        raise ValueError(f"{session_day.isoformat()} is not an XNYS session")
    return datetime.combine(session_day, session.close_at, us_calendar.timezone).astimezone(SEOUL)


def kr_report_dates(kr_calendar: SessionCalendar, start: date, end: date) -> list[date]:
    days: list[date] = []
    cursor = start
    while cursor <= end:
        if kr_calendar.is_session(cursor) is None:
            raise ValueError(f"KR calendar does not cover {cursor.isoformat()}")
        if kr_calendar.is_session(cursor):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def build_table(start: date, end: date, *, holiday_csv: Path | None = None,
                market_lag_limit_sessions: int = DEFAULT_MARKET_LAG_LIMIT,
                reviewed_status: str = "unreviewed", review_note: str | None = None,
                generated_at: datetime | None = None,
                collection_at: time = COLLECTION_AT_KST) -> dict[str, Any]:
    if start > end:
        raise ValueError("start must not be after end")
    if reviewed_status not in REVIEWED_STATUSES:
        raise ValueError(f"reviewed_status must be one of {REVIEWED_STATUSES}")
    if reviewed_status == "confirmed" and not review_note:
        raise ValueError("confirmed requires a review note naming who confirmed the policy")
    if (isinstance(market_lag_limit_sessions, bool) or not isinstance(market_lag_limit_sessions, int)
            or market_lag_limit_sessions < 0):
        raise ValueError("market_lag_limit_sessions must be a nonnegative integer")
    generated = generated_at or datetime.now(SEOUL)
    if generated.tzinfo is None:
        raise ValueError("generated_at must include a timezone")
    csv_path = holiday_csv or default_holiday_csv()
    kr_manifest = krx_project_manifest(start, end, holiday_csv=csv_path)
    nyse_start = max(date(2026, 1, 1), start - timedelta(days=NYSE_LOOKBACK_DAYS))
    us_manifest = nyse_equity_manifest(nyse_start, end)
    kr_calendar = SessionCalendar.from_manifest(kr_manifest)
    us_calendar = SessionCalendar.from_manifest(us_manifest)
    table = {day.isoformat(): expected_session(us_calendar, day, collection_at).isoformat()
             for day in kr_report_dates(kr_calendar, start, end)}
    kr_sha = _manifest_sha256(kr_manifest)
    us_sha = _manifest_sha256(us_manifest)
    source_reference = (
        f"rule={RULE_NAME}; provisional policy draft 1 ({POLICY_REFERENCE}); "
        f"collection assumption {collection_at.strftime('%H:%M')} KST on D-1 (sdc_daily_us) "
        f"with daily derive 15:30 KST; us_calendar=nyse_equity_manifest sha256={us_sha}; "
        f"kr_calendar=krx_project_manifest sha256={kr_sha}; generated_at={generated.isoformat()}"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "reviewed_status": reviewed_status,
        "provisional": True,
        "review_note": review_note,
        "rule": RULE_NAME,
        "collection_assumption": {
            "timezone": "Asia/Seoul", "collection_time": collection_at.strftime("%H:%M"),
            "collection_day_offset": -1, "derive_time_after": "15:30",
            "event_ids": ["sdc_daily_us", "sdc_daily_us_derive_daily"]},
        "generated_at": generated.isoformat(),
        "calendar_sources": {
            "us": {"builder": "modeler.serving.calendar_sources.nyse_equity_manifest",
                   "manifest_sha256": us_sha, "coverage_start": us_manifest["coverage_start"],
                   "coverage_end": us_manifest["coverage_end"], "source": us_manifest["source"]},
            "kr": {"builder": "modeler.serving.calendar_sources.krx_project_manifest",
                   "manifest_sha256": kr_sha, "holiday_csv_sha256": kr_manifest["source"]["holiday_csv_sha256"],
                   "coverage_start": kr_manifest["coverage_start"], "coverage_end": kr_manifest["coverage_end"]}},
        "source_reference": source_reference,
        "market_lag_limit_sessions": market_lag_limit_sessions,
        "report_date_range": {"start": start.isoformat(), "end": end.isoformat(), "count": len(table)},
        "expected_session_by_report_date": table,
    }


def write_table(table: dict[str, Any], output: Path) -> Path:
    """Write a new file atomically. An existing different file is never replaced."""
    raw = (json.dumps(table, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    descriptor, name = tempfile.mkstemp(prefix=".us-expected-", dir=output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(name, output)
    finally:
        Path(name).unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modeler-serving-us-expected", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="KR trading days start..end -> us-expected-source.v1 JSON")
    build.add_argument("--start", type=date.fromisoformat, required=True)
    build.add_argument("--end", type=date.fromisoformat, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--holiday-csv", type=Path)
    build.add_argument("--market-lag-limit-sessions", type=int, default=DEFAULT_MARKET_LAG_LIMIT)
    build.add_argument("--reviewed-status", choices=REVIEWED_STATUSES, default="unreviewed")
    build.add_argument("--review-note")
    one = sub.add_parser("session", help="print E for one calendar report date (no KR filter; used to pick native prepare A)")
    one.add_argument("--report-date", type=date.fromisoformat, required=True)
    args = parser.parse_args(argv)
    if args.command == "session":
        try:
            day = args.report_date
            calendar = SessionCalendar.from_manifest(nyse_equity_manifest(
                max(date(2026, 1, 1), day - timedelta(days=NYSE_LOOKBACK_DAYS)), min(day, date(2027, 12, 31))))
            print(expected_session(calendar, day).isoformat())
        except ValueError as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=__import__("sys").stderr)
            return 1
        return 0
    try:
        table = build_table(args.start, args.end, holiday_csv=args.holiday_csv,
                            market_lag_limit_sessions=args.market_lag_limit_sessions,
                            reviewed_status=args.reviewed_status, review_note=args.review_note)
        path = write_table(table, args.output)
    except (ValueError, OSError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False), file=__import__("sys").stderr)
        return 1
    raw = path.read_bytes()
    print(json.dumps({"status": "written", "path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                      "report_dates": table["report_date_range"]["count"],
                      "reviewed_status": table["reviewed_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
