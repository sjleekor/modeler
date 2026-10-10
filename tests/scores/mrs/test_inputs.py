"""MRS 입력층 시험: 격자(토요일 세션·US 달력 일치), PIT 행, VIX 근사, 현금, 실현 배열."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.scores.common.calendar import UTC_TS
from modeler.scores.common.kr_inputs import (
    KrLake,
    _load_series,
    _with_available_at,
    kr_available_at,
    load_kr_series,
)
from modeler.scores.common.panel import PRICE_AVAILABILITY_BUFFER
from modeler.scores.market_sector.features import fred_available_at
from modeler.scores.mrs import config
from modeler.scores.mrs.inputs import (
    KR_PIT_SERIES,
    US_TABLES_FOR_KR,
    GridMismatchError,
    build_kr_grid,
    build_us_grid,
    load_kr_inputs,
    load_us_inputs,
    pin_us_lake,
    realized_on_grid,
    resolve_kr_lake,
)

from .fake_lakes import (
    ALL_KR_SERIES,
    KR_SNAP,
    US_LAST_PRICE,
    US_SNAP,
    kr_grid_dates,
    write_kr_lake,
    write_us_lake,
)

KST = ZoneInfo("Asia/Seoul")


@pytest.fixture
def kr_lake(tmp_path) -> KrLake:
    write_kr_lake(tmp_path)
    return resolve_kr_lake(DataRoot(tmp_path / "kr"))


def _us_lake(tmp_path, tables=None, **kw):
    write_us_lake(tmp_path, **kw)
    root = DataRoot(tmp_path / "us")
    if tables is None:
        return pin_us_lake(root)
    return pin_us_lake(root, tables, symbols=())


# --------------------------------------------------------------------------- KR 격자
def test_kr_grid_has_saturday_sessions_and_decision_at(kr_lake):
    g = build_kr_grid(kr_lake)
    assert list(g.dates) == kr_grid_dates()
    assert any(d.weekday() == 5 for d in g.dates)  # 토요일 세션 포함
    assert g.calendar.calendar_basis == "observed_price_dates"
    # 금요일 결정 = 토요일 개장 09:00 KST - 30분
    fri = date(1996, 1, 5)
    i = g.dates.index(fri)
    assert g.dates[i + 1] == date(1996, 1, 6)
    assert g.decision_at[i] == datetime(1996, 1, 6, 8, 30, tzinfo=KST).astimezone(UTC)
    # 토요일 결정 = 월요일 08:30 KST
    sat = g.dates.index(date(1996, 1, 6))
    assert g.dates[sat + 1] == date(1996, 1, 8)
    assert g.decision_at[sat] == datetime(1996, 1, 8, 8, 30, tzinfo=KST).astimezone(UTC)
    # 마지막 격자일은 결정 시각이 없다
    assert g.decision_at[-1] is None
    assert all(a is not None for a in g.decision_at[:-1])
    fr = g.decision_frame()
    assert fr.height == len(g) - 1 and fr.schema["decision_at"] == UTC_TS


# --------------------------------------------------------------------------- KR PIT 행
def test_load_kr_series_matches_existing_rules(kr_lake):
    g = build_kr_grid(kr_lake)
    got = load_kr_series(kr_lake, "fx_usdkrw_ecos", g.dates)
    ref = _with_available_at(_load_series(kr_lake, "fx_usdkrw_ecos", g.dates))
    assert got.equals(ref)
    assert got.schema["available_at"] == UTC_TS
    assert load_kr_series(kr_lake, "nope", g.dates) is None


def test_kr_available_at_uses_stored_available_from_date_even_if_later_than_next_session(kr_lake):
    """금요일 값(다음 격자 세션=토요일)은 저장된 월요일 08:30 KST에 가용이다 (MI29)."""
    inp = load_kr_inputs(kr_lake, None, hash_files=False)
    k = inp.series["market_kospi_ecos"]
    fri = k.filter(pl.col("date") == date(1996, 1, 5))["available_at"].item()
    assert fri == kr_available_at(date(1996, 1, 8))
    assert fri > inp.grid.calendar.open_at(date(1996, 1, 6))  # 토요일 개장보다 늦다
    # 목요일 값은 금요일 08:30
    thu = k.filter(pl.col("date") == date(1996, 1, 4))["available_at"].item()
    assert thu == kr_available_at(date(1996, 1, 5))
    # 그래서 금요일 결정 시각(토요일 08:30)에는 금요일 값이 안 보인다
    d_fri = inp.grid.decision_at[inp.grid.dates.index(date(1996, 1, 5))]
    visible = k.filter(pl.col("available_at") <= d_fri)["date"].max()
    assert visible == date(1996, 1, 4)


def test_kr_bundle_series_missing_cash_and_files(tmp_path):
    sd = {sid: kr_grid_dates() for sid in ALL_KR_SERIES if sid != "rate_kr_gov10y"}
    write_kr_lake(tmp_path, sd)
    write_us_lake(tmp_path)
    lake = resolve_kr_lake(DataRoot(tmp_path / "kr"))
    us = pin_us_lake(DataRoot(tmp_path / "us"), US_TABLES_FOR_KR, symbols=())
    inp = load_kr_inputs(lake, us)
    assert "rate_kr_gov10y" in inp.missing_series
    assert "rate_kr_gov10y" not in inp.series
    assert set(inp.series) == (set(KR_PIT_SERIES) - {"rate_kr_gov10y"}) | {"VIXCLS"}
    assert inp.cash is not None and inp.cash.series_id == "rate_kr_cd91"
    assert inp.cash.staleness_days == config.CASH_STALENESS_DAYS
    assert inp.snapshot == {"common_feature_observation_raw": KR_SNAP, "macro_series": US_SNAP}
    by_table = {f.table: f for f in inp.input_files}
    assert set(by_table) == {"common_feature_observation_raw", "macro_series"}
    f = by_table["common_feature_observation_raw"]
    assert f.sha256 == hashlib.sha256(open(f.path, "rb").read()).hexdigest()
    for df in inp.series.values():
        assert df.schema["available_at"] == UTC_TS and df.schema["date"] == pl.Date


# --------------------------------------------------------------------------- VIX
def test_vix_main_rows_follow_ms0_rule_and_keep_vintages(tmp_path, kr_lake):
    us = _us_lake(tmp_path, US_TABLES_FOR_KR)
    inp = load_kr_inputs(kr_lake, us, hash_files=False)
    v = inp.series["VIXCLS"]
    two = v.filter(pl.col("date") == date(1996, 2, 1)).sort("available_at")
    assert two["value"].to_list() == [15.0, 99.0]  # vintage 둘 다 남는다
    ref = fred_available_at(
        pl.DataFrame({"date": [date(1996, 2, 1)], "realtime_start": [date(1996, 3, 1)]})
    )["available_at"].item()
    assert two["available_at"][1] == ref  # 17:00 ET
    backfilled = v.filter(pl.col("date") == date(1996, 2, 5))["available_at"].item()
    expect = datetime(2010, 11, 22, 17, tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)
    assert backfilled == expect


def test_vix_proxy_first_vintage_and_next_kr_session(tmp_path, kr_lake):
    us = _us_lake(tmp_path, US_TABLES_FOR_KR)
    inp = load_kr_inputs(kr_lake, us, hash_files=False)
    p = inp.vix_proxy
    assert p["date"].is_unique().all() if hasattr(p["date"].is_unique(), "all") else True
    assert p["date"].n_unique() == p.height
    row = lambda d: p.filter(pl.col("date") == d)  # noqa: E731
    # 개정 전 처음 vintage 값
    assert row(date(1996, 2, 1))["value"].item() == 15.0
    assert row(date(1996, 2, 1))["available_at"].item() == kr_available_at(date(1996, 2, 2))
    # 금요일 관측 -> 다음 KR 격자 세션은 토요일 08:30
    assert row(date(1996, 2, 2))["available_at"].item() == kr_available_at(date(1996, 2, 3))
    # realtime_start(2010-11-22) 무시: 관측일 다음 KR 세션
    assert row(date(1996, 2, 5))["available_at"].item() == kr_available_at(date(1996, 2, 6))
    # 마지막 격자일(토요일) 직전 금요일은 남고, 격자 뒤 관측은 빠진다
    assert row(date(1996, 3, 29))["available_at"].item() == kr_available_at(date(1996, 3, 30))
    assert row(date(1996, 4, 1)).height == 0
    assert inp.diagnostics["vix_proxy_dropped_no_later_session"] >= 1
    # 근사는 관측일 이후 엄격히 뒤 (같은 날 08:30 불가)
    assert (p["available_at"].dt.convert_time_zone("Asia/Seoul").dt.date() > p["date"]).all()
    assert p.schema["available_at"] == UTC_TS


def test_kr_inputs_without_us_lake_reports_vix_missing(kr_lake):
    inp = load_kr_inputs(kr_lake, None, hash_files=False)
    assert "VIXCLS" in inp.missing_series and inp.vix_proxy.height == 0


# --------------------------------------------------------------------------- 실현 배열
def test_realized_on_grid_is_null_without_carry():
    grid = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)]
    rows = pl.DataFrame({"date": [date(2024, 1, 2), date(2024, 1, 4)], "value": [10.0, 12.0]})
    out = realized_on_grid(grid, rows)
    assert out[0] == 10.0 and out[2] == 12.0
    assert np.isnan(out[1]) and np.isnan(out[3])  # 앞 값을 끌어오지 않는다
    # 격자 밖 관측일(토요일)은 무시
    rows2 = pl.DataFrame({"date": [date(2024, 1, 6)], "value": [5.0]})
    assert np.isnan(realized_on_grid(grid, rows2)).all()


def test_kr_realized_arrays_null_on_missing_same_day(tmp_path):
    g = kr_grid_dates()
    skip = date(1996, 2, 14)
    sd = {sid: list(g) for sid in ALL_KR_SERIES}
    sd["market_kosdaq_ecos"] = [d for d in g if d != skip]
    write_kr_lake(tmp_path, sd)
    lake = resolve_kr_lake(DataRoot(tmp_path / "kr"))
    inp = load_kr_inputs(lake, None, hash_files=False)
    i = list(inp.grid.dates).index(skip)
    kd = inp.realized["kr_kosdaq"]
    assert np.isnan(kd[i]) and not np.isnan(kd[i - 1]) and not np.isnan(kd[i + 1])
    assert int(np.isnan(kd).sum()) == 1
    assert not np.isnan(inp.realized["kr_kospi"]).any()


# --------------------------------------------------------------------------- US
def test_us_grid_range_equals_lake_and_decision_at(tmp_path):
    lake = _us_lake(tmp_path)
    g, lake_cal = build_us_grid(lake)
    assert g.dates[0] == date(1993, 1, 29)
    # SPY 마지막 가격일 다음 세션까지
    assert (
        g.dates[-1]
        == lake_cal.session_at(lake_cal.index_of(US_LAST_PRICE) + 1)
        == date(2024, 6, 28)
    )
    assert g.dates[-2] == US_LAST_PRICE
    # 겹치는 구간 개장·폐장 일치, 결정 = 다음 세션 개장 - 30분
    i = g.dates.index(date(2024, 3, 8))
    assert g.decision_at[i] == g.calendar.open_at(date(2024, 3, 11)) - timedelta(minutes=30)
    assert g.decision_at[-1] is None
    for s in lake_cal.sessions:
        if s <= g.dates[-1]:
            assert g.calendar.open_at(s) == lake_cal.open_at(s)
            assert g.calendar.close_at(s) == lake_cal.close_at(s)


def test_us_grid_mismatch_raises_missing_session(tmp_path):
    lake = _us_lake(tmp_path, drop_cal_session=date(2024, 2, 14))
    with pytest.raises(GridMismatchError, match="2024-02-14"):
        build_us_grid(lake)


def test_us_grid_mismatch_raises_close_time(tmp_path):
    lake = _us_lake(tmp_path, close_override={date(2024, 3, 1): time(13, 0)})
    with pytest.raises(GridMismatchError, match="2024-03-01"):
        build_us_grid(lake)


def test_us_inputs_tr_index_availability_macro_and_cash(tmp_path):
    lake = _us_lake(tmp_path)
    inp = load_us_inputs(lake)
    tr = inp.series["tr_index"]
    assert tr.schema["available_at"] == UTC_TS and tr.height == len(
        [d for d in inp.grid.dates if date(2024, 1, 2) <= d <= US_LAST_PRICE]
    )
    cal = inp.grid.calendar
    for d, at in zip(tr["date"].to_list(), tr["available_at"].to_list(), strict=True):
        assert at == cal.close_at(d) + PRICE_AVAILABILITY_BUFFER
    assert set(inp.series) == {"tr_index", "VIXCLS", "BAA10Y", "T10Y2Y"}
    assert inp.cash is not None and inp.cash.series_id == "DGS3MO"
    assert inp.missing_series == ()
    assert set(inp.snapshot) == {"prices_daily", "corp_actions", "macro_series", "trading_calendar"}
    assert {f.table for f in inp.input_files} == set(inp.snapshot)
    # 실현 배열: 가격이 있는 날만 값, 그 앞(2024-01-02 전)은 null
    r = inp.realized["us_spx"]
    assert len(r) == len(inp.grid)
    assert np.isnan(r[: inp.grid.dates.index(date(2024, 1, 2))]).all()
    assert not np.isnan(r[inp.grid.dates.index(date(2024, 1, 3))])
    assert np.isnan(r[-1])  # 마지막 격자일은 SPY 가격이 없다
