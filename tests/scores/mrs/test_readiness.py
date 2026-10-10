"""MRS readiness 시험: 개수·날짜만 나오는지, 백필 PASS/FAIL, MI29 건수, CLI 산출물."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta

import pytest

from modeler.etl.config import DataRoot
from modeler.scores.mrs import config
from modeler.scores.mrs import readiness as rd
from modeler.scores.mrs.inputs import build_kr_grid, resolve_kr_lake

from .fake_lakes import (
    ALL_KR_SERIES,
    KR_SNAP,
    SENTINEL,
    kr_grid_dates,
    write_kr_lake,
    write_us_lake,
)


def _weekdays(a: date, b: date) -> list[date]:
    out, d = [], a
    while d <= b:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# --------------------------------------------------------------------------- 순수 계산
def test_gap_stats_counts_gaps_over_10_days():
    ds = [date(2020, 1, 1), date(2020, 1, 10), date(2020, 1, 21), date(2020, 3, 1)]
    assert rd.gap_stats(ds) == (2, 40)  # 11일, 40일만 초과 (9일은 아님)
    assert rd.gap_stats([date(2020, 1, 1)]) == (0, 0)


def test_missing_by_period_counts_grid_dates_without_same_day_obs():
    grid = [date(1999, 12, 31), date(2000, 1, 3), date(2000, 1, 4), date(2010, 1, 4)]
    obs = [date(1999, 12, 31), date(2000, 1, 3)]
    out = rd.missing_by_period(obs, grid, rd.PERIODS_KR)
    assert out == {"1995-1999": 0, "2000-2009": 1, "2010-": 1}


# --------------------------------------------------------------------------- 백필
def _good(start: date) -> list[date]:
    return _weekdays(start, date(2014, 7, 15))


def test_backfill_pass_when_all_three_start_in_time_and_no_gap():
    req = config.BACKFILL_REQUIRED_START
    obs = {sid: _good(d) for sid, d in req.items()}
    r = rd.backfill_check(obs)
    assert r["overall"] == "PASS"
    assert all(v["pass"] and v["reason"] is None for v in r["series"].values())


def test_backfill_tolerance_is_three_calendar_days():
    req = config.BACKFILL_REQUIRED_START
    obs = {sid: _good(d) for sid, d in req.items()}
    # 필요 시작일 + 3일에 시작: 통과. + 4일: 실패
    fx0 = req["fx_usdkrw_ecos"]
    ok = dict(obs, fx_usdkrw_ecos=[fx0 + timedelta(days=3), *_good(fx0 + timedelta(days=4))])
    assert rd.backfill_check(ok)["series"]["fx_usdkrw_ecos"]["pass"]
    late = dict(obs, fx_usdkrw_ecos=[fx0 + timedelta(days=4), *_good(fx0 + timedelta(days=5))])
    r = rd.backfill_check(late)
    assert not r["series"]["fx_usdkrw_ecos"]["pass"]
    assert r["overall"] == "FAIL"


def test_backfill_fail_when_one_series_partial_or_missing_or_gap():
    req = config.BACKFILL_REQUIRED_START
    obs = {sid: _good(d) for sid, d in req.items()}
    # 현재 맥 레이크: 2014-06-13에 시작 -> 셋 다 FAIL
    cur = {sid: _weekdays(date(2014, 6, 13), date(2014, 7, 15)) for sid in req}
    r = rd.backfill_check(cur)
    assert r["overall"] == "FAIL" and not any(v["pass"] for v in r["series"].values())
    # 한 계열만 부분 백필: 전체 FAIL (일부만 된 상태로 시작하지 않는다)
    part = dict(obs, rate_kr_gov10y=_weekdays(date(2014, 6, 13), date(2014, 7, 15)))
    assert rd.backfill_check(part)["overall"] == "FAIL"
    # 계열 없음
    miss = dict(obs, rate_kr_gov3y=None)
    r = rd.backfill_check(miss)
    assert r["series"]["rate_kr_gov3y"]["reason"] == "series_missing" and r["overall"] == "FAIL"
    # 중간 공백 > 10일
    hole = [d for d in obs["fx_usdkrw_ecos"] if not (date(2003, 3, 1) <= d <= date(2003, 3, 20))]
    r = rd.backfill_check(dict(obs, fx_usdkrw_ecos=hole))
    row = r["series"]["fx_usdkrw_ecos"]
    assert (
        not row["pass"]
        and row["gaps_gt10_in_window"] == 1
        and row["reason"] == "gap_gt10_in_window"
    )
    # 공백 정확히 10일은 통과
    ok10 = [d for d in obs["fx_usdkrw_ecos"] if d != date(2003, 3, 5)]  # 평일 하나 비움
    assert rd.backfill_check(dict(obs, fx_usdkrw_ecos=ok10))["series"]["fx_usdkrw_ecos"]["pass"]
    # 2014-06-13 앞에서 끝나고 공백이 10일 넘으면 FAIL (끝 날짜까지 덮어야 한다)
    short = [d for d in obs["rate_kr_gov10y"] if d < date(2014, 1, 1)]
    assert not rd.backfill_check(dict(obs, rate_kr_gov10y=short))["series"]["rate_kr_gov10y"][
        "pass"
    ]


# --------------------------------------------------------------------------- 가짜 레이크
@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    g = kr_grid_dates()
    sd = {sid: list(g) for sid in ALL_KR_SERIES}
    sd["market_kosdaq_ecos"] = [d for d in g if d != date(1996, 2, 14)]  # 격자일 하나 빠짐
    sd["fx_usdkrw_ecos"] = [d for d in g if not (date(1996, 2, 1) <= d <= date(1996, 2, 20))]
    write_kr_lake(tmp_path, sd)
    write_us_lake(tmp_path)
    return tmp_path


def test_readiness_kr_counts(world):
    lake = resolve_kr_lake(DataRoot(world / "kr"))
    us = rd.pin_us_lake(DataRoot(world / "us"), rd.US_TABLES_FOR_KR, symbols=())
    rep = rd.readiness_kr(lake, us)
    g = kr_grid_dates()
    assert rep["grid"] == {"first": "1996-01-02", "last": "1996-03-30", "n": len(g)}
    k = rep["series"]["market_kospi_ecos"]
    assert k["rows"] == len(g) and k["min_date"] == "1996-01-02" and k["max_date"] == "1996-03-30"
    assert k["missing_same_date"] == {"1995-1999": 0, "2000-2009": 0, "2010-": 0}
    # MI29: 토요일 세션이 뒤따르는 금요일 값은 저장된 월요일 가용일이 다음 격자 세션보다 늦다
    fri_before_sat = [d for d in g if d.weekday() == 4 and d + timedelta(days=1) in g]
    assert k["late_available_from"] == len(fri_before_sat) > 0
    # 같은 날 관측 빠진 수
    assert rep["series"]["market_kosdaq_ecos"]["missing_same_date"]["1995-1999"] == 1
    fx = rep["series"]["fx_usdkrw_ecos"]
    n_hole = len([d for d in g if date(1996, 2, 1) <= d <= date(1996, 2, 20)])
    assert fx["missing_same_date"]["1995-1999"] == n_hole
    assert fx["gaps_gt10"] == 1 and fx["max_gap_days"] > 10
    # 없는 계열은 exists False (이 가짜 레이크엔 거래대금 등은 있고 VIX는 US 레이크에 있다)
    assert rep["series"]["VIXCLS"]["exists"] and "first_available_date" in rep["series"]["VIXCLS"]
    # 백필: 가짜 레이크는 1996에 시작 -> 셋 다 FAIL
    assert rep["backfill"]["overall"] == "FAIL"
    assert rep["expected_first_dates"]["status"] == "computed_with_fake_values"
    fd = {r["component"]: r for r in rep["expected_first_dates"]["first_dates"]}
    assert fd["kr_trend_252"]["first_pct_date"] is None  # 가짜 레이크는 1,260개에 한참 모자란다


def test_readiness_kr_missing_series_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    g = kr_grid_dates()
    write_kr_lake(tmp_path, {sid: list(g) for sid in ALL_KR_SERIES if sid != "rate_kr_gov3y"})
    lake = resolve_kr_lake(DataRoot(tmp_path / "kr"))
    rep = rd.readiness_kr(lake, None)
    assert rep["series"]["rate_kr_gov3y"] == {"exists": False}
    assert rep["series"]["VIXCLS"] == {"exists": False}
    assert rep["backfill"]["series"]["rate_kr_gov3y"]["reason"] == "series_missing"


def test_readiness_us_counts(world):
    us = rd.pin_us_lake(DataRoot(world / "us"))
    rep = rd.readiness_us(us)
    assert rep["grid"]["first"] == "1993-01-29" and rep["grid"]["last"] == "2024-06-28"
    spy = rep["series"]["SPY"]
    assert spy["min_date"] == "2024-01-02" and spy["max_date"] == "2024-06-27"
    assert "1993-1994" in spy["missing_same_date"]
    assert spy["missing_same_date"]["2010-"] > 0  # 2010~2023 격자일은 SPY 관측이 없다
    assert rep["backfill"] is None
    assert set(rep["series"]) == {"SPY", "VIXCLS", "BAA10Y", "T10Y2Y", "DGS3MO"}


# --------------------------------------------------------------------------- 값 통계 금지
_FORBIDDEN_KEYS = {"value", "values", "mean", "median", "std", "sum", "mean_value", "last_value"}


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def test_readiness_output_has_no_value_statistics(world):
    now = datetime(2026, 10, 10, 21, 5)
    results = rd.run("all", kr_snapshot=KR_SNAP, now=now)
    assert [r[0]["market"] for r in results] == ["kr", "us"]
    for rep, path in results:
        text = path.read_text()
        assert str(int(SENTINEL)) not in text  # 가짜 레이크 값(987654.xxx)이 새지 않는다
        assert not (set(_keys(json.loads(text))) & _FORBIDDEN_KEYS)
        assert str(int(SENTINEL)) not in rd.format_table(rep)
    # 경로: stock_data/<market>/output/regime_score_readiness/<KR 스냅샷>_<시각>/readiness.json
    kr_path = results[0][1]
    assert kr_path == (
        world
        / "kr"
        / "output"
        / "regime_score_readiness"
        / f"{KR_SNAP}_20261010T2105"
        / "readiness.json"
    )
    assert results[1][1].parent.parent == world / "us" / "output" / "regime_score_readiness"


def test_cli_main_prints_table_and_writes_json(world, capsys):
    assert rd.main(["--market", "kr", "--kr-snapshot", KR_SNAP]) == 0
    out = capsys.readouterr().out
    assert "market_kospi_ecos" in out and "백필 확인" in out and "전체: FAIL" in out
    dirs = list((world / "kr" / "output" / "regime_score_readiness").iterdir())
    assert len(dirs) == 1 and re.fullmatch(rf"{KR_SNAP}_\d{{8}}T\d{{4}}", dirs[0].name)
    data = json.loads((dirs[0] / "readiness.json").read_text())
    assert data["backfill"]["overall"] == "FAIL"
    assert not (world / "us" / "output").exists()  # kr만 돌렸다


def test_loaders_drop_value_columns(world, monkeypatch):
    """readiness 로더가 값 열을 들고 있지 않다: 날짜 리스트만 돌려준다."""
    lake = resolve_kr_lake(DataRoot(world / "kr"))
    g = build_kr_grid(lake)
    obs, avail = rd._kr_dates(lake, "market_kospi_ecos", g)
    assert all(isinstance(d, date) for d in obs + avail)
    us = rd.pin_us_lake(DataRoot(world / "us"))
    dates, rows, first = rd._us_macro_dates(us, "BAA10Y")
    assert all(isinstance(d, date) for d in dates) and isinstance(rows, int)


def _synthetic_kr_pit(n: int = 420):
    """합성 KR 격자(평일 n일)와 PIT 날짜 프레임(값 없음, 08:30 KST 다음 평일 가용)."""
    import polars as pl

    from modeler.scores.common.calendar import UTC_TS, SessionCalendar
    from modeler.scores.common.kr_inputs import kr_available_at
    from modeler.scores.mrs.inputs import make_grid

    ds = _weekdays(date(2001, 1, 2), date(2003, 12, 31))[:n]
    grid = make_grid("KR", SessionCalendar.from_observed_dates("XKRX", ds), ds)
    nxt = {d: ds[i + 1] if i + 1 < len(ds) else d + timedelta(days=1) for i, d in enumerate(ds)}
    frame = pl.DataFrame(
        {"date": ds, "available_at": pl.Series([kr_available_at(nxt[d]) for d in ds], dtype=UTC_TS)}
    )
    ids = (
        "market_kospi_ecos",
        "trdval_kospi_ecos",
        "trdval_kosdaq_ecos",
        "foreign_net_kospi_ecos",
        "foreign_net_kosdaq_ecos",
        "fx_usdkrw_ecos",
        "rate_kr_gov3y",
        "rate_kr_gov10y",
    )
    pit = {sid: frame for sid in ids}
    pit["VIXCLS"] = frame
    return grid, pit


def test_expected_first_dates_runs_engine_on_fake_values_only():
    grid, pit = _synthetic_kr_pit()
    a = rd.expected_first_dates("kr", grid, pit, seed=1, warmup=50)
    b = rd.expected_first_dates("kr", grid, pit, seed=999, warmup=50)
    assert a["status"] == "computed_with_fake_values" and "가짜 값" in a["note"]
    assert a["first_dates"] == b["first_dates"]  # 값(시드)이 바뀌어도 날짜는 같다
    by = {r["component"]: r for r in a["first_dates"]}
    assert set(by) == set(config.KR_COMPONENTS)
    # kr_trend_252: 252행 뒤 첫 값, warm-up 50개가 차는 날 첫 백분위 (격자 인덱스로 확인)
    ds = list(grid.dates)
    assert by["kr_trend_252"]["first_value_date"] == ds[252].isoformat()
    assert by["kr_trend_252"]["first_pct_date"] == ds[252 + 49].isoformat()
    # 입력 값 열이 없는 프레임만 받는다 (값은 훅이 만든 난수)
    assert all("value" not in f.columns for f in pit.values())


def test_expected_first_dates_reports_missing_inputs():
    grid, pit = _synthetic_kr_pit()
    pit = {k: v for k, v in pit.items() if k != "rate_kr_gov10y"}
    r = rd.expected_first_dates("kr", grid, pit, warmup=50)
    assert r["status"] == "missing_inputs" and "rate_kr_gov10y" in r["note"]


def test_no_expected_flag_skips_engine(world, capsys):
    assert rd.main(["--market", "us", "--no-expected"]) == 0
    out = capsys.readouterr().out
    assert "skipped" in out
