from datetime import UTC, date, datetime

import pytest

from modeler.scores.common.assets import (
    ASSETS,
    assets_for_market,
    get_asset,
    registry_hash,
)
from modeler.scores.common.calendar import CalendarError, SessionCalendar
from tests.scores._helpers import make_cal, weekdays


def test_registry_us_and_kr_entries():
    us = {a.asset_id: a for a in assets_for_market("US")}
    assert set(us) == {"us_spx", "us_ndx", "us_fin", "us_hlth", "us_ind", "us_enrg", "us_tech"}
    assert us["us_ndx"].asset_type == "style" and us["us_ndx"].parent_benchmark == "us_spx"
    assert us["us_spx"].parent_benchmark is None and us["us_spx"].proxy == "SPY"
    assert all(a.calendar == "XNYS" and a.currency == "USD" for a in us.values())
    kr = {a.asset_id: a for a in assets_for_market("KR")}
    assert set(kr) == {"kr_kospi", "kr_kosdaq", "kr_fin", "kr_hlth", "kr_ind", "kr_enrg", "kr_tech"}
    assert kr["kr_kospi"].proxy == "코스피" and kr["kr_kospi"].fallback_proxy == "market_kospi_ecos"
    assert get_asset("kr_kosdaq").source == "kr_krx_index_daily"
    assert len(registry_hash()) == 64 and len(ASSETS) == 14 and len(us) + len(kr) == 14
    with pytest.raises(KeyError):
        get_asset("nope")


def test_decision_entry_exit_us_est_and_edt():
    # 2024-01-05(금) 다음 세션 = 01-08(월) 개장 09:30 EST = 14:30 UTC -> 결정 14:00 UTC
    cal = SessionCalendar.from_sessions("XNYS", weekdays(date(2024, 1, 2), 100), calendar_basis="t")
    assert cal.decision_at(date(2024, 1, 5)) == datetime(2024, 1, 8, 14, 0, tzinfo=UTC)
    assert cal.entry_session(date(2024, 1, 5)) == date(2024, 1, 8)
    entry = date(2024, 1, 8)
    assert cal.exit_session(entry, 60) == cal.sessions[cal.index_of(entry) + 60]
    # EDT(4월): 09:30 EDT = 13:30 UTC -> 결정 13:00 UTC
    apr = SessionCalendar.from_sessions("XNYS", weekdays(date(2024, 4, 1), 10), calendar_basis="t")
    assert apr.decision_at(date(2024, 4, 1)) == datetime(2024, 4, 2, 13, 0, tzinfo=UTC)


def test_kr_open_is_0900_kst():
    cal = SessionCalendar.from_sessions("XKRX", weekdays(date(2024, 1, 2), 5), calendar_basis="t")
    # 09:00 KST = 00:00 UTC -> 결정 = 전날 23:30 UTC
    assert cal.decision_at(date(2024, 1, 2)) == datetime(2024, 1, 2, 23, 30, tzinfo=UTC)


def test_early_close_uses_session_close():
    from datetime import time

    ds = weekdays(date(2024, 11, 26), 3)  # 11-26,27,28(추수감사절 휴장은 여기선 무시)
    cal = SessionCalendar.from_sessions(
        "XNYS", ds, calendar_basis="t", close_local=[None, time(13, 0), None]
    )
    assert cal.close_at(ds[1]) == datetime(2024, 11, 27, 18, 0, tzinfo=UTC)  # 13:00 EST


def test_not_a_session_and_horizon_errors():
    cal = make_cal(5)
    with pytest.raises(CalendarError):
        cal.index_of(date(2024, 1, 6))
    with pytest.raises(CalendarError):
        cal.decision_at(cal.sessions[-1])
    with pytest.raises(CalendarError):
        cal.exit_session(cal.sessions[-1], 3)


def test_session_table_has_utc_timestamps():
    t = make_cal(5).session_table()
    assert str(t.schema["decision_at"]) == "Datetime(time_unit='us', time_zone='UTC')"
    assert t["decision_at"][-1] is None


def test_xkrx_exchange_calendars_if_available():
    cal = SessionCalendar.from_exchange_calendars("XKRX", date(2024, 1, 1), date(2024, 3, 31))
    if cal is None:
        pytest.skip("exchange_calendars 없음")
    assert date(2024, 2, 9) not in cal.sessions  # 설 연휴
    assert cal.opens[0].hour == 1  # 새해 첫 거래일은 10:00 KST 개장 (달력이 반영한다)
    assert cal.open_at(date(2024, 1, 3)).hour == 0  # 평일 09:00 KST
