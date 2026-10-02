"""Bound opening observations to the same D/10:00 decision as inference."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from .calendars import SessionCalendar


def validate_opening(opening: dict[str, Any] | None, *, report_date: date,
                     decision_at: datetime, calendar: SessionCalendar | None,
                     fixture_mode: bool) -> dict[str, Any]:
    if opening is None:
        return {"status": "unavailable", "assessment": "opening_artifact_missing",
                "publication": {"status": "unresolved", "evidence": []}}

    def unavailable(reason: str) -> dict[str, Any]:
        return {"status": "unavailable", "assessment": reason,
                "session_state": "unavailable", "indices": [], "industries": [],
                "publication": {"status": "unresolved", "evidence": []}}

    if not isinstance(opening, dict) or opening.get("report_date") != report_date.isoformat():
        return unavailable("opening_report_date_mismatch")
    if bool(opening.get("synthetic_fixture", False)) != fixture_mode:
        return unavailable("opening_fixture_mode_mismatch")
    if calendar is None or calendar.market != "KR" or calendar.is_session(report_date) is not True:
        return unavailable("opening_calendar_unavailable")
    session = calendar.session(report_date)
    if session is None or not session.confirmed:
        return unavailable("opening_session_unconfirmed")
    if opening.get("status") != "ok":
        return {**opening, "status": "unavailable", "indices": [], "industries": [],
                "publication": {"status": "unresolved", "evidence": []}}
    if opening.get("session_state") != "regular":
        return unavailable("opening_not_regular_session")
    try:
        if datetime.fromisoformat(opening["decision_at"]) != decision_at:
            return unavailable("opening_decision_mismatch")
        max_age = opening["max_age_seconds"]
        if isinstance(max_age, bool) or not isinstance(max_age, int) or not 0 < max_age <= 3600:
            return unavailable("opening_source_age_policy_invalid")
        received = datetime.fromisoformat(opening["received_at"])
        if received.tzinfo is None or received.utcoffset() is None or received > decision_at:
            return unavailable("opening_received_after_decision")
        session_open = datetime.combine(report_date, session.open_at, calendar.timezone)
        session_close = datetime.combine(report_date, session.close_at, calendar.timezone)
        if not session_open < decision_at < session_close:
            return unavailable("opening_outside_regular_session")
        groups = [opening.get(field) for field in ("indices", "industries")]
        if any(not isinstance(group, list) or not group for group in groups):
            return unavailable("opening_required_groups_missing")
        rows = [row for group in groups for row in group]
        if any(not isinstance(row, dict) for row in rows):
            return unavailable("opening_source_observations_missing")
        source_times = []
        for row in rows:
            source = datetime.fromisoformat(row["observed_at"])
            if (source.tzinfo is None or source.utcoffset() is None or
                    not session_open <= source <= received <= decision_at or
                    source < decision_at - timedelta(seconds=max_age)):
                return unavailable("opening_source_time_unverified")
            source_times.append(source)
        if datetime.fromisoformat(opening["observed_at"]) != min(source_times):
            return unavailable("opening_source_summary_time_mismatch")
    except (KeyError, TypeError, ValueError):
        return unavailable("opening_source_time_unverified")
    return opening
