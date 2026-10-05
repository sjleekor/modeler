"""Fail-closed freshness assessment for D/K/U/E/A dates and the input cutoff."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .calendars import SessionCalendar

SEOUL = ZoneInfo("Asia/Seoul")

#: 오래된 입력으로 만든 순위를 표시하는 상한(세션 수, 2026-10-05 Q5). 이 값을 넘으면 순위를
#: 쓰지 않고 사유만 남깁니다. 신선도는 막지 않고 표시하지만, 상한 없이 표시하면 KR이 2주
#: 깨졌을 때 2주 전 순위가 매일 오늘 날짜로 나갑니다. `reporting/markdown.py`의 LAG_SUPPRESS와
#: 같은 값입니다.
MAX_STALE_SESSIONS = 5


@dataclass(frozen=True)
class Freshness:
    status: str
    report_date: str
    feature_asof_date: str | None
    latest_us_session: str | None
    expected_us_session: str | None
    actual_us_session: str | None
    input_cutoff: str
    verified_available_by: str | None
    delivery_lag: int | None
    market_lag: int | None
    reason: str | None
    availability_evidence_type: str | None = None
    source_first_available_at: str | None = None
    availability_evidence: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def assess_freshness(*, report_date: date, market: str, feature_asof_date: date | None,
                     decision_at: datetime, calendar: SessionCalendar,
                     input_cutoff: datetime | None = None,
                     latest_us_session: date | None = None,
                     expected_us_session: date | None = None,
                     actual_us_session: date | None = None,
                     verified_available_by: datetime | None = None,
                     availability_evidence_type: str | None = None,
                     availability_evidence: dict[str, Any] | None = None,
                     source_first_available_at: datetime | None = None,
                     max_us_delivery_lag: int = 2,
                     max_us_market_lag: int | None = None) -> Freshness:
    cutoff_at = input_cutoff or (decision_at - timedelta(minutes=30))
    cutoff = cutoff_at.isoformat()
    asof = feature_asof_date.isoformat() if feature_asof_date else None
    latest = latest_us_session.isoformat() if latest_us_session else None
    expected = expected_us_session.isoformat() if expected_us_session else None
    actual = actual_us_session.isoformat() if actual_us_session else None
    first_available = source_first_available_at.isoformat() if source_first_available_at else None
    availability_at = verified_available_by
    available = availability_at.isoformat() if availability_at else None

    def _result(status: str, d: str, f: str | None, u: str | None, e: str | None,
                a: str | None, cutoff_value: str, available_value: str | None,
                delivery: int | None, market_lag: int | None, reason: str | None) -> Freshness:
        return Freshness(status, d, f, u, e, a, cutoff_value, available_value,
                         delivery, market_lag, reason, availability_evidence_type,
                         first_available, availability_evidence)
    if (decision_at.tzinfo is None or decision_at.utcoffset() is None or
            cutoff_at.tzinfo is None or cutoff_at.utcoffset() is None):
        return _result("failed", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, None, None, "decision/cutoff lacks timezone")
    seoul_decision = decision_at.astimezone(SEOUL)
    seoul_cutoff = cutoff_at.astimezone(SEOUL)
    if (seoul_decision.date() != report_date or
            seoul_decision.time().replace(tzinfo=None) != time(10, 0)):
        return _result("failed", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, None, None, "decision_at is not D 10:00 Asia/Seoul")
    expected_cutoff = seoul_decision.replace(hour=9, minute=30, second=0, microsecond=0)
    if seoul_cutoff != expected_cutoff:
        return _result("failed", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, None, None, "input_cutoff must be D 09:30 Asia/Seoul")
    if input_cutoff and input_cutoff > decision_at:
        return _result("failed", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, None, None, "input cutoff is after decision_at")
    if availability_at is None:
        return _result("unavailable", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         None, None, None, "verified input availability time is missing")
    if (availability_at.tzinfo is None or availability_at.utcoffset() is None or
            availability_at > cutoff_at):
        return _result("unavailable", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                       available, None, None, "verified input availability is after cutoff; do not infer")
    if verified_available_by is not None and (
            availability_evidence_type != "prepared_features_completion" or
            not isinstance(availability_evidence, dict) or
            any(not isinstance(availability_evidence.get(key), str) or
                not re.fullmatch(r"[0-9a-f]{64}", availability_evidence[key])
                for key in ("features_sha256", "native_prepare_manifest_sha256"))):
        return _result("unavailable", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                       available, None, None, "prepared-feature completion evidence is missing or unsupported")
    if market == "KR":
        if calendar.market != "KR":
            return _result("failed", report_date.isoformat(), asof, None, None, None, cutoff,
                             available, None, None, "KR freshness received a non-KR calendar")
        if calendar.is_session(report_date) is None:
            reason = "Korean calendar coverage is unknown"
        elif calendar.is_session(report_date) is False:
            reason = "report date is not a Korean session"
        else:
            k_session = calendar.previous_session(report_date)
            if k_session is None:
                reason = "previous Korean session is outside calendar coverage"
            else:
                k_details = calendar.session(k_session)
                k_close = (datetime.combine(k_session, k_details.close_at, calendar.timezone)
                           if k_details else None)
                if (k_details is None or not k_details.confirmed or k_close is None or
                        k_close > decision_at or calendar.latest_completed_before(decision_at) != k_session):
                    return _result("unavailable", report_date.isoformat(), asof, None, None, None,
                                     cutoff, available, None, None,
                                     "K is not a confirmed, completed previous KR session")
                if feature_asof_date == k_session:
                    return _result("ok", report_date.isoformat(), asof, None, None, None, cutoff,
                                     available, 0, 0, None)
                # K보다 이른 기준일 K'(2026-10-05 변경 3): 막지 않고 stale로 표시합니다.
                # 지연 세션 수를 셀 수 없으면(달력이 K'를 모름) 표시할 근거가 없어 fail-closed입니다.
                lag = (calendar.session_distance(feature_asof_date, k_session)
                       if feature_asof_date else None)
                if lag is None or calendar.is_session(feature_asof_date) is not True:
                    return _result("unavailable", report_date.isoformat(), asof, None, None, None,
                                     cutoff, available, None, None,
                                     "KR feature session is not a session of the KR calendar")
                if lag < 0:
                    return _result("failed", report_date.isoformat(), asof, None, None, None, cutoff,
                                     available, lag, None, "KR feature session is newer than K")
                return _result("stale", report_date.isoformat(), asof, None, None, None, cutoff,
                                 available, lag, None, "KR feature session does not match K")
        return _result("unavailable", report_date.isoformat(), asof, None, None, None, cutoff,
                         available, None, None, reason)
    if market != "US":
        return _result("failed", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, None, None, "unknown market")
    if calendar.market != "US":
        return _result("failed", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, None, None, "US freshness received a non-US calendar")
    if latest_us_session is None or expected_us_session is None or actual_us_session is None:
        return _result("unavailable", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, None, None, "US U, E, or A session is missing")
    raw_delivery_lag = calendar.session_distance(actual_us_session, expected_us_session)
    market_lag = calendar.session_distance(actual_us_session, latest_us_session)
    if raw_delivery_lag is None or market_lag is None:
        return _result("unavailable", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, raw_delivery_lag, market_lag, "US calendar does not cover U, E, and A")
    delivery_lag = max(0, raw_delivery_lag)
    session_details = [calendar.session(day) for day in
                       (latest_us_session, expected_us_session, actual_us_session)]
    if any(item is None or not item.confirmed for item in session_details):
        return _result("unavailable", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, delivery_lag, market_lag, "US U, E, and A must be confirmed sessions")
    if expected_us_session > latest_us_session:
        return _result("failed", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, delivery_lag, market_lag, "US expected session E is newer than U")
    if feature_asof_date != actual_us_session or actual_us_session > latest_us_session:
        return _result("failed", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, delivery_lag, market_lag, "US A is inconsistent with U or feature date")
    if market_lag < 0:
        return _result("failed", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, delivery_lag, market_lag, "US actual session A is newer than U")
    if max_us_market_lag is None:
        return _result("unavailable", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, delivery_lag, market_lag, "US market-lag ceiling is not configured")
    if isinstance(max_us_market_lag, bool) or not isinstance(max_us_market_lag, int) or max_us_market_lag < 0:
        return _result("failed", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, delivery_lag, market_lag, "US market-lag ceiling must be a nonnegative integer")
    if calendar.latest_completed_before(decision_at) != latest_us_session:
        return _result("failed", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, delivery_lag, market_lag,
                         "US U is not the latest completed session at decision_at")
    # 2026-10-05 변경 4: 지연 한도와 중단 한도는 막지 않고 stale로 표시합니다. 순위를 낼지는
    # MAX_STALE_SESSIONS(5세션)가 정합니다(`orchestration.run_daily`).
    if market_lag > max_us_market_lag:
        return _result("stale", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, delivery_lag, market_lag, "US actual session exceeds market-lag ceiling")
    if delivery_lag >= max_us_delivery_lag:
        return _result("stale", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, delivery_lag, market_lag, "US delivery lag reached the stop limit")
    if delivery_lag == 1:
        return _result("stale", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                         available, delivery_lag, market_lag, "US data arrived one session after E")
    return _result("ok", report_date.isoformat(), asof, latest, expected, actual, cutoff,
                     available, delivery_lag, market_lag,
                     "A is newer than E but not newer than U" if actual_us_session > expected_us_session else None)
