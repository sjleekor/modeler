"""Real calendar coverage, early close, DST and unknown KR exceptions."""

from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from modeler.serving.calendar_sources import coverage_status, krx_project_manifest, nyse_equity_manifest
from modeler.serving.calendars import SessionCalendar


def test_nyse_official_2026_2027_early_close_and_dst():
    cal = SessionCalendar.from_manifest(
        nyse_equity_manifest(date(2026, 9, 1), date(2027, 12, 31))
    )
    assert cal.is_session(date(2026, 11, 26)) is False
    assert cal.session(date(2026, 11, 27)).close_at.isoformat() == "13:00:00"
    assert cal.session(date(2027, 11, 26)).close_at.isoformat() == "13:00:00"
    assert cal.is_session(date(2027, 1, 1)) is False
    ny = ZoneInfo("America/New_York")
    assert datetime(2026, 10, 30, 16, tzinfo=ny).utcoffset().total_seconds() == -4 * 3600
    assert datetime(2026, 11, 2, 16, tzinfo=ny).utcoffset().total_seconds() == -5 * 3600


def test_kr_csv_is_bounded_and_csat_exception_unknown(tmp_path):
    csv = tmp_path / "holidays.csv"
    csv.write_text("date,name\n2026-09-24,Chuseok\n2026-12-31,Closure\n")
    with pytest.raises(ValueError, match="pinned complete file"):
        krx_project_manifest(date(2026, 9, 1), date(2026, 12, 31), holiday_csv=csv)
    csv = (Path(__file__).parents[4] / "collector" / "src" / "collector" / "kr" /
           "infra" / "calendar" / "data" / "holidays_krx.csv")
    cal = SessionCalendar.from_manifest(
        krx_project_manifest(date(2026, 9, 1), date(2026, 12, 31), holiday_csv=csv)
    )
    assert cal.is_session(date(2026, 9, 24)) is False
    assert cal.session(date(2026, 11, 19)).confirmed is False
    assert cal.session(date(2026, 10, 1)).confirmed is True
    assert cal.session(date(2026, 9, 30)).confirmed is True
    assert cal.is_session(date(2027, 1, 4)) is None
    first = SessionCalendar.from_manifest(
        krx_project_manifest(date(2026, 1, 1), date(2026, 1, 31), holiday_csv=csv)
    )
    assert first.session(date(2026, 1, 2)).confirmed is False
    with pytest.raises(ValueError, match="bounded to 2026"):
        krx_project_manifest(date(2026, 12, 1), date(2027, 1, 31), holiday_csv=csv)


def test_calendar_manifest_rejects_conflicting_exception_evidence():
    manifest = nyse_equity_manifest(date(2026, 11, 27), date(2026, 11, 27))
    manifest["overrides"].append(dict(manifest["overrides"][0]))
    with pytest.raises(ValueError, match="unique"):
        SessionCalendar.from_manifest(manifest)
    manifest["overrides"].pop()
    manifest["unconfirmed_dates"] = ["2026-11-27"]
    with pytest.raises(ValueError, match="conflicts"):
        SessionCalendar.from_manifest(manifest)
    manifest["overrides"].clear()
    manifest["unconfirmed_dates"] = ["2026-11-27", "2026-11-27"]
    with pytest.raises(ValueError, match="unique"):
        SessionCalendar.from_manifest(manifest)


def test_calendar_coverage_warns_60_days_before_end():
    manifest = nyse_equity_manifest(date(2027, 1, 1), date(2027, 12, 31))
    assert coverage_status(manifest, as_of=date(2027, 10, 31))["status"] == "ok"
    due = coverage_status(manifest, as_of=date(2027, 11, 1))
    assert (due["status"], due["days_remaining"]) == ("renewal_due", 60)
    assert coverage_status(manifest, as_of=date(2028, 1, 1))["status"] == "expired"
