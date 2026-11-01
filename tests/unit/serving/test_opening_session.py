"""개장 관측 판정(``opening_session``) 시험.

collector ``tests/unit/test_kis_intraday.py``의 판정 사례를 그대로 옮기고, 경계 사례를 더한다.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from modeler.serving.opening_session import SEOUL, assess_session_status

KST = ZoneInfo("Asia/Seoul")
OPEN = datetime(2026, 9, 30, 9, 0, tzinfo=KST)
CLOSE = datetime(2026, 9, 30, 15, 30, tzinfo=KST)
SLOT = datetime(2026, 9, 30, 9, 5, tzinfo=KST)


def _assess(observed_at, **overrides):
    kwargs = dict(
        calendar_open=True,
        session_open_at=OPEN,
        session_close_at=CLOSE,
        slot_at=SLOT,
        max_age_seconds=300,
    )
    kwargs.update(overrides)
    return assess_session_status({"observations": [{"source_observed_at": observed_at}]}, **kwargs)


def test_seoul_is_asia_seoul():
    assert SEOUL == KST


def test_collector_cases_unchanged():
    snapshot = {"observations": [{"source_observed_at": None}]}
    assert assess_session_status(snapshot, calendar_open=False) == "unavailable_closed_session"
    assert assess_session_status(snapshot, calendar_open=None) == "unavailable_calendar_unverified"
    assert assess_session_status(snapshot, calendar_open=True) == "observation_time_unverified"
    assert _assess("2026-09-30T09:05:00+09:00") == "open_session_observed"
    assert _assess("2026-09-29T15:30:00+09:00") == "observation_time_unverified"


def test_boundaries_fail_closed():
    assert _assess("2026-09-30T09:00:00+09:00") == "open_session_observed"
    assert _assess("2026-09-30T08:59:59+09:00") == "observation_time_unverified"
    assert _assess("2026-09-30T09:05:01+09:00") == "observation_time_unverified"
    assert _assess("2026-09-30T09:00:00", slot_at=SLOT) == "observation_time_unverified"
    assert _assess("not-a-time") == "observation_time_unverified"
    assert _assess("2026-09-30T09:00:00+09:00", max_age_seconds=299) == "stale_source_observation"
    assert _assess("2026-09-30T09:00:00+09:00", max_age_seconds=0) == "observation_time_unverified"
    assert (
        _assess("2026-09-30T09:05:00+09:00", slot_at=CLOSE) == "unavailable_outside_session_window"
    )
    assert _assess("2026-09-30T09:05:00+09:00", slot_at=datetime(2026, 9, 30, 9, 5)) == (
        "observation_time_unverified"
    )
    assert assess_session_status({"observations": []}, calendar_open=True) == (
        "observation_time_unverified"
    )


def test_utc_inputs_are_compared_in_seoul_time():
    utc = ZoneInfo("UTC")
    assert (
        _assess(
            "2026-09-30T00:04:00+00:00",
            session_open_at=datetime(2026, 9, 30, 0, 0, tzinfo=utc),
            session_close_at=datetime(2026, 9, 30, 6, 30, tzinfo=utc),
            slot_at=datetime(2026, 9, 30, 0, 5, tzinfo=utc),
        )
        == "open_session_observed"
    )
