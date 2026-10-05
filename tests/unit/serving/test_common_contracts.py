from __future__ import annotations

import math
from datetime import date, datetime, time, timezone
from zoneinfo import ZoneInfo

import pytest

from modeler.serving.calendars import Session, SessionCalendar
from modeler.serving.freshness import assess_freshness
from modeler.serving.schema import report_template, validate_report

SEOUL = ZoneInfo("Asia/Seoul")
UTC = timezone.utc


def _report() -> dict:
    report = report_template(market="KR", report_date="2026-09-29",
        decision_at=datetime(2026, 9, 29, 10, 0, tzinfo=SEOUL),
        feature_asof_date="2026-09-28", model_id="kr_daily_h20_v1", model_version="1")
    report["status"] = "ok"
    report["rankings"] = [{"rank": 1, "symbol": "005930", "name": "fixture", "score": 0.1}]
    return report


def test_report_template_defaults_to_unresolved_and_accepts_valid_report():
    report = _report()
    assert report["publication"] == {"status": "unresolved", "evidence": []}
    validate_report(report)


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(decision_at="2026-09-29T10:01:00+09:00"),
    lambda x: x["rankings"].append({"rank": 2, "symbol": "005930", "name": "duplicate", "score": 1.0}),
    lambda x: x["rankings"][0].update(rank=True),
    lambda x: x["rankings"][0].update(score=math.nan),
    lambda x: x["rankings"][0].update(score=True),
])
def test_report_rejects_bad_cutoff_or_ranking(mutate):
    report = _report()
    mutate(report)
    with pytest.raises(ValueError):
        validate_report(report)


def test_calendar_unknown_dates_and_session_overrides_are_explicit():
    day = date(2026, 9, 29)
    shifted = Session(day, time(10), time(16), confirmed=True)
    calendar = SessionCalendar.from_dates("KR", [day], coverage_start=day,
        coverage_end=day, overrides={day: shifted}, unconfirmed_dates=[day])
    assert calendar.is_session(day) is True
    assert calendar.is_session(date(2026, 9, 30)) is None
    assert calendar.session(day) == shifted
    assert calendar.cutoff(day, time(9, 30)).strftime("%H:%M") == "09:30"


def test_us_calendar_uses_per_session_close_for_dst_and_early_close():
    day = date(2026, 11, 27)
    calendar = SessionCalendar.from_manifest({
        "market": "US", "timezone": "America/New_York", "coverage_start": day.isoformat(),
        "coverage_end": day.isoformat(), "sessions": [day.isoformat()],
        "default_open_at": "09:30", "default_close_at": "16:00",
        "overrides": [{"date": day.isoformat(), "open_at": "09:30", "close_at": "13:00",
                       "confirmed": True}], "unconfirmed_dates": [],
    })
    assert calendar.session(day).close_at == time(13)
    assert calendar.latest_completed_before(datetime(2026, 11, 27, 19, 0, tzinfo=UTC)) == day
    assert calendar.latest_completed_before(datetime(2026, 11, 27, 17, 59, tzinfo=UTC)) is None


def test_calendar_unknown_local_date_or_newer_unconfirmed_session_is_not_certified():
    old = SessionCalendar.from_dates("US", [date(2026, 9, 18), date(2026, 9, 19)],
        coverage_start=date(2026, 9, 18), coverage_end=date(2026, 9, 19))
    assert old.latest_completed_before(datetime(2026, 9, 30, 21, tzinfo=UTC)) is None
    uncertain = SessionCalendar.from_dates("US", [date(2026, 9, 28), date(2026, 9, 29)],
        coverage_start=date(2026, 9, 28), coverage_end=date(2026, 9, 29),
        unconfirmed_dates=[date(2026, 9, 29)])
    assert uncertain.latest_completed_before(datetime(2026, 9, 30, 1, tzinfo=UTC)) is None
    assert uncertain.latest_completed_before(datetime(2026, 9, 29, 19, tzinfo=UTC)) is None


@pytest.mark.parametrize("patch", [
    {"timezone": "UTC"},
    {"coverage_start": "2026-11-28", "coverage_end": "2026-11-27"},
    {"default_open_at": "17:00", "default_close_at": "09:00"},
])
def test_calendar_manifest_rejects_invalid_timezone_coverage_and_hours(patch):
    base = {"market": "US", "timezone": "America/New_York", "coverage_start": "2026-11-27",
        "coverage_end": "2026-11-27", "sessions": ["2026-11-27"],
        "default_open_at": "09:30", "default_close_at": "16:00",
        "overrides": [], "unconfirmed_dates": []}
    with pytest.raises(ValueError):
        SessionCalendar.from_manifest({**base, **patch})


def _kr_calendar() -> SessionCalendar:
    return SessionCalendar.from_dates("KR", [date(2026, 9, 28), date(2026, 9, 29)],
        coverage_start=date(2026, 9, 28), coverage_end=date(2026, 9, 29))


def _availability(at: datetime):
    return {"verified_available_by": at,
        "availability_evidence_type": "prepared_features_completion",
        "availability_evidence": {"features_sha256": "a" * 64,
            "native_prepare_manifest_sha256": "b" * 64}}


def test_kr_freshness_uses_k_and_separate_0930_cutoff():
    result = assess_freshness(report_date=date(2026, 9, 29), market="KR",
        feature_asof_date=date(2026, 9, 28),
        decision_at=datetime(2026, 9, 29, 10, tzinfo=SEOUL), calendar=_kr_calendar(),
        **_availability(datetime(2026, 9, 29, 9, 25, tzinfo=SEOUL)))
    assert result.status == "ok"
    assert result.input_cutoff == "2026-09-29T09:30:00+09:00"


def test_freshness_rejects_nonzero_decision_seconds():
    result = assess_freshness(report_date=date(2026, 9, 29), market="KR",
        feature_asof_date=date(2026, 9, 28),
        decision_at=datetime(2026, 9, 29, 10, 0, 59, tzinfo=SEOUL), calendar=_kr_calendar(),
        **_availability(datetime(2026, 9, 29, 9, 25, tzinfo=SEOUL)))
    assert result.status == "failed"


def test_verified_feature_completion_is_distinct_from_unknown_source_arrival():
    result = assess_freshness(report_date=date(2026, 9, 29), market="KR",
        feature_asof_date=date(2026, 9, 28),
        decision_at=datetime(2026, 9, 29, 10, tzinfo=SEOUL), calendar=_kr_calendar(),
        verified_available_by=datetime(2026, 9, 29, 9, 25, tzinfo=SEOUL),
        availability_evidence_type="prepared_features_completion",
        availability_evidence={"features_sha256": "a" * 64,
            "native_prepare_manifest_sha256": "b" * 64},
        source_first_available_at=None)
    assert result.status == "ok"
    assert result.verified_available_by == "2026-09-29T09:25:00+09:00"
    assert result.source_first_available_at is None
    assert result.availability_evidence_type == "prepared_features_completion"
    assert result.availability_evidence["features_sha256"] == "a" * 64


def test_kr_input_after_cutoff_is_not_eligible_for_inference():
    result = assess_freshness(report_date=date(2026, 9, 29), market="KR",
        feature_asof_date=date(2026, 9, 28),
        decision_at=datetime(2026, 9, 29, 10, tzinfo=SEOUL), calendar=_kr_calendar(),
        **_availability(datetime(2026, 9, 29, 9, 31, tzinfo=SEOUL)))
    assert result.status == "unavailable"
    assert "do not infer" in result.reason


def _kr_calendar_wide() -> SessionCalendar:
    days = [date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23), date(2026, 9, 24),
            date(2026, 9, 25), date(2026, 9, 28), date(2026, 9, 29)]
    return SessionCalendar.from_dates("KR", days, coverage_start=days[0], coverage_end=days[-1])


def _kr_freshness(asof: date, *, calendar: SessionCalendar | None = None):
    return assess_freshness(report_date=date(2026, 9, 29), market="KR", feature_asof_date=asof,
        decision_at=datetime(2026, 9, 29, 10, tzinfo=SEOUL), calendar=calendar or _kr_calendar_wide(),
        **_availability(datetime(2026, 9, 29, 9, 25, tzinfo=SEOUL)))


def test_kr_session_before_k_is_stale_with_its_lag_in_sessions():
    """2026-10-05 change 3: K' (earlier than K) is shown as stale; the lag counts KR sessions."""
    assert _kr_freshness(date(2026, 9, 28)).status == "ok"
    one = _kr_freshness(date(2026, 9, 25))
    assert one.status == "stale" and one.delivery_lag == 1
    assert one.reason == "KR feature session does not match K"
    far = _kr_freshness(date(2026, 9, 21))
    assert far.status == "stale" and far.delivery_lag == 5


def test_kr_stale_input_still_fails_closed_on_calendar_and_cutoff_facts():
    assert _kr_freshness(date(2026, 9, 29)).status == "failed"  # newer than K
    outside = _kr_freshness(date(2026, 9, 18))  # before the calendar's coverage: lag unknown
    assert outside.status == "unavailable" and outside.delivery_lag is None
    weekend = _kr_freshness(date(2026, 9, 26))  # a date the calendar knows is not a session
    assert weekend.status == "unavailable"
    unconfirmed = SessionCalendar.from_dates("KR", [date(2026, 9, 25), date(2026, 9, 28), date(2026, 9, 29)],
        coverage_start=date(2026, 9, 25), coverage_end=date(2026, 9, 29),
        unconfirmed_dates=[date(2026, 9, 28)])
    assert _kr_freshness(date(2026, 9, 25), calendar=unconfirmed).status == "unavailable"
    late = assess_freshness(report_date=date(2026, 9, 29), market="KR", feature_asof_date=date(2026, 9, 25),
        decision_at=datetime(2026, 9, 29, 10, tzinfo=SEOUL), calendar=_kr_calendar_wide(),
        **_availability(datetime(2026, 9, 29, 9, 31, tzinfo=SEOUL)))
    assert late.status == "unavailable" and "do not infer" in late.reason


def _us_calendar() -> SessionCalendar:
    dates = [date(2026, 9, 23), date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28)]
    return SessionCalendar.from_dates("US", dates, coverage_start=dates[0], coverage_end=dates[-1])


def _us_freshness(actual: date, expected: date, latest: date, *, source_time=None, cap=2):
    return assess_freshness(report_date=date(2026, 9, 29), market="US",
        feature_asof_date=actual, decision_at=datetime(2026, 9, 29, 10, tzinfo=SEOUL),
        calendar=_us_calendar(), latest_us_session=latest, expected_us_session=expected,
        actual_us_session=actual, **_availability(source_time or datetime(2026, 9, 29, 9, 25, tzinfo=SEOUL)),
        max_us_market_lag=cap)


def test_us_a_newer_than_e_is_allowed_when_it_is_u_and_arrived_before_cutoff():
    result = _us_freshness(date(2026, 9, 28), date(2026, 9, 25), date(2026, 9, 28))
    assert result.status == "ok"
    assert result.delivery_lag == 0
    assert result.market_lag == 0


def test_us_lag_is_shown_not_blocked_one_session_two_sessions_and_beyond_the_old_ceiling():
    """2026-10-05 change 4: the 2-session stop limit and the market-lag ceiling now mark stale."""
    one_late = _us_freshness(date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28))
    two_late = _us_freshness(date(2026, 9, 23), date(2026, 9, 25), date(2026, 9, 28))
    assert one_late.status == "stale" and one_late.delivery_lag == 1
    # Two sessions behind E, three behind U (> the configured ceiling 2): stale with both lags.
    assert two_late.status == "stale" and two_late.delivery_lag == 2 and two_late.market_lag == 3
    assert two_late.reason == "US actual session exceeds market-lag ceiling"
    # A looser ceiling keeps the delivery-lag reason.
    looser = _us_freshness(date(2026, 9, 23), date(2026, 9, 25), date(2026, 9, 28), cap=9)
    assert looser.status == "stale" and looser.reason == "US delivery lag reached the stop limit"


def test_us_accuracy_checks_still_block_even_when_lag_is_only_displayed():
    # Cutoff PIT, unconfirmed sessions and an A newer than U are accuracy, not freshness.
    late = _us_freshness(date(2026, 9, 28), date(2026, 9, 28), date(2026, 9, 28),
                         source_time=datetime(2026, 9, 29, 9, 31, tzinfo=SEOUL))
    assert late.status == "unavailable" and "do not infer" in late.reason
    future_a = _us_freshness(date(2026, 9, 28), date(2026, 9, 25), date(2026, 9, 25))
    assert future_a.status == "failed"


def test_us_a_after_u_fails_and_missing_market_lag_cap_stays_unavailable():
    future_a = _us_freshness(date(2026, 9, 28), date(2026, 9, 25), date(2026, 9, 25))
    no_cap = _us_freshness(date(2026, 9, 24), date(2026, 9, 25), date(2026, 9, 28), cap=None)
    assert future_a.status == "failed"
    assert no_cap.status == "unavailable"


def test_missing_source_available_at_never_counts_as_fresh():
    result = assess_freshness(report_date=date(2026, 9, 29), market="US",
        feature_asof_date=date(2026, 9, 28), decision_at=datetime(2026, 9, 29, 10, tzinfo=SEOUL),
        calendar=_us_calendar(), latest_us_session=date(2026, 9, 28),
        expected_us_session=date(2026, 9, 28), actual_us_session=date(2026, 9, 28),
        source_first_available_at=None, max_us_market_lag=2)
    assert result.status == "unavailable"
