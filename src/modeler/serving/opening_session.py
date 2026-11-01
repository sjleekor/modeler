"""개장 관측 스냅샷을 브리핑에 써도 되는지 판정하는 순수 함수.

collector ``collector.kr.adapters.kis_intraday``(``d693773``)의 ``assess_session_status``를
그대로 옮겼다(2026-11-01). modeler는 collector 어댑터를 import하지 않는다(ruff TID251).
운영에서 이 판정을 쓰는 곳은 modeler ``serving.opening_prepare``뿐이고, collector에서는
시험만 부른다. 판정 규칙은 브리핑이 관측을 쓸지 정하는 쪽이 가진다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

SEOUL = ZoneInfo("Asia/Seoul")


def assess_session_status(
    snapshot: dict[str, Any],
    *,
    calendar_open: bool | None,
    session_open_at: datetime | None = None,
    session_close_at: datetime | None = None,
    slot_at: datetime | None = None,
    max_age_seconds: int | None = None,
) -> str:
    """Fail closed unless both the local calendar and source time are usable."""
    if calendar_open is False:
        return "unavailable_closed_session"
    if calendar_open is not True:
        return "unavailable_calendar_unverified"
    observations = snapshot.get("observations", [])
    if (not isinstance(observations, list) or not observations or
            any(not isinstance(row, dict) or not row.get("source_observed_at")
                for row in observations)):
        return "observation_time_unverified"
    if (
        session_open_at is None
        or session_close_at is None
        or slot_at is None
        or max_age_seconds is None
        or max_age_seconds <= 0
    ):
        return "observation_time_unverified"
    bounds = (session_open_at, session_close_at, slot_at)
    if any(value.tzinfo is None or value.utcoffset() is None for value in bounds):
        return "observation_time_unverified"
    session_open = session_open_at.astimezone(SEOUL)
    session_close = session_close_at.astimezone(SEOUL)
    slot = slot_at.astimezone(SEOUL)
    if (not session_open <= slot < session_close or
            len({session_open.date(), session_close.date(), slot.date()}) != 1):
        return "unavailable_outside_session_window"
    for observation in observations:
        try:
            source_at = datetime.fromisoformat(observation["source_observed_at"])
        except (KeyError, TypeError, ValueError):
            return "observation_time_unverified"
        if source_at.tzinfo is None or source_at.utcoffset() is None:
            return "observation_time_unverified"
        source_at = source_at.astimezone(SEOUL)
        if not session_open <= source_at <= slot:
            return "observation_time_unverified"
        if (slot - source_at).total_seconds() > max_age_seconds:
            return "stale_source_observation"
    return "open_session_observed"
