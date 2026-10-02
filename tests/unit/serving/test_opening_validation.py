from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from modeler.serving.calendars import SessionCalendar
from modeler.serving.opening_validation import validate_opening
from modeler.serving.orchestration import run_daily

SEOUL = ZoneInfo("Asia/Seoul")
D = date(2026, 9, 30)
DECISION = datetime(2026, 9, 30, 10, tzinfo=SEOUL)
CALENDAR = SessionCalendar.from_manifest({
    "market": "KR", "timezone": "Asia/Seoul", "coverage_start": "2026-09-29",
    "coverage_end": "2026-09-30", "sessions": ["2026-09-29", "2026-09-30"],
    "default_open_at": "09:00", "default_close_at": "15:30", "overrides": [],
    "unconfirmed_dates": []})


def _opening() -> dict:
    return {"report_date": D.isoformat(), "decision_at": DECISION.isoformat(),
            "synthetic_fixture": True, "status": "ok", "session_state": "regular",
            "received_at": "2026-09-30T09:55:00+09:00",
            "observed_at": "2026-09-30T09:53:00+09:00", "max_age_seconds": 600,
            "indices": [{"observed_at": "2026-09-30T09:53:00+09:00"}],
            "industries": [{"observed_at": "2026-09-30T09:54:00+09:00"}],
            "publication": {"status": "unresolved", "evidence": []}}


def _check(opening: dict) -> dict:
    return validate_opening(opening, report_date=D, decision_at=DECISION,
                            calendar=CALENDAR, fixture_mode=True)


def test_recent_source_observations_pass_without_public_rights() -> None:
    opening = _opening()
    assert _check(opening) == opening


def test_received_time_cannot_substitute_for_old_source_time() -> None:
    opening = _opening()
    opening["indices"][0]["observed_at"] = "2026-09-30T09:01:00+09:00"
    opening["observed_at"] = "2026-09-30T09:01:00+09:00"
    assert _check(opening)["status"] == "unavailable"


def test_missing_industry_or_future_source_is_unavailable() -> None:
    opening = _opening()
    opening["industries"] = []
    assert _check(opening)["assessment"] == "opening_required_groups_missing"
    opening = _opening()
    opening["indices"][0]["observed_at"] = "2026-09-30T10:01:00+09:00"
    assert _check(opening)["status"] == "unavailable"


def test_wrong_date_and_fixture_flag_are_unavailable() -> None:
    opening = _opening()
    opening["report_date"] = "2026-09-29"
    assert _check(opening)["status"] == "unavailable"
    opening = _opening()
    opening["synthetic_fixture"] = False
    assert _check(opening)["status"] == "unavailable"


def test_opening_uses_pinned_calendar_when_kr_input_is_missing(tmp_path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    envelope = run_daily(report_date=D, decision_at=DECISION, prepared_root=prepared,
        run_root=tmp_path / "runs", jobs=[], opening=_opening(),
        opening_calendar=CALENDAR, fixture_mode=True,
        now=datetime(2026, 9, 30, 10, 1, tzinfo=SEOUL))
    assert envelope["opening"]["status"] == "ok"
    assert envelope["status"] == "failed"
    assert (tmp_path / "runs" / "report-2026-09-30.json").is_file()
