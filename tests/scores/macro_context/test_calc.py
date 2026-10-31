"""calc: 방향 문턱·급·백분위·사본 이음·가용일 보기·월간 날짜 맞춤(합성 값)."""

from __future__ import annotations

import datetime as dt
import math

import numpy as np
import pytest

from modeler.scores.macro_context import calc
from modeler.scores.macro_context.load import view
from modeler.scores.macro_context.spec import DAILY_BY_ID, MONTHLY_BY_ID

D = dt.date(2026, 10, 28)


def wdays(n: int, end: dt.date = dt.date(2026, 10, 27)) -> list[dt.date]:
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= dt.timedelta(days=1)
    return out[::-1]


def kr_view(sid: str, values, dates=None):
    dates = dates or wdays(len(values))
    return {
        sid: {d: (float(v), d + dt.timedelta(days=1)) for d, v in zip(dates, values, strict=True)}
    }


# --------------------------------------------------------------------------- 문턱·방향·급
def test_classify_boundaries():
    assert calc.classify(0.0, 0.1, 0.5) == ("flat", False, "보합", "→")
    assert calc.classify(0.0999, 0.1, 0.5)[0] == "flat"
    assert calc.classify(-0.0999, 0.1, 0.5)[0] == "flat"
    assert calc.classify(0.1, 0.1, 0.5) == ("up", False, "상승", "▲")  # 문턱과 같으면 보합이 아니다
    assert calc.classify(-0.3, 0.1, 0.5) == ("down", False, "하락", "▼")
    assert calc.classify(0.5, 0.1, 0.5) == ("up", True, "급등", "▲▲")  # 급 문턱과 같으면 급
    assert calc.classify(-0.6, 0.1, 0.5)[2] == "급락"
    assert calc.classify(0.2, 0.1, 0.5, spread=True)[2] == "확대"
    assert calc.classify(0.6, 0.1, 0.5, spread=True)[2] == "급확대"
    assert calc.classify(-0.2, 0.1, 0.5, spread=True)[2] == "축소"
    assert calc.classify(-0.6, 0.1, 0.5, spread=True)[2] == "급축소"
    assert calc.classify(None, 0.1, 0.5) is None


def test_classify_flow_grades():
    assert calc.classify_flow(0.05, 0.1, 0.5) == ("up", False, "순매수", "보합")
    assert calc.classify_flow(-0.2, 0.1, 0.5) == ("down", False, "순매도", "보통")
    assert calc.classify_flow(-0.5, 0.1, 0.5) == ("down", True, "순매도", "급")
    assert calc.classify_flow(0.0, 0.1, 0.5)[0] == "up"  # 0은 부호상 순매수 쪽, 등급은 보합


def test_thresholds_quantiles_and_min_sample():
    hist = np.arange(1, 101, dtype=float)  # |Δ| 1..100
    th = calc.thresholds(np.concatenate([hist, -hist, [np.nan]]))
    assert th is not None and th[2] == 200
    a = np.abs(np.concatenate([hist, -hist]))
    assert th[0] == pytest.approx(np.percentile(a, 30)) and th[1] == pytest.approx(
        np.percentile(a, 90)
    )
    assert calc.thresholds(np.arange(19, dtype=float)) is None
    assert calc.thresholds(np.arange(20, dtype=float)) is not None


def test_pctile_counts_values_at_or_below():
    v = np.array([1.0, 2.0, 3.0, 4.0, np.nan])
    assert calc.pctile(v, 3.0) == (75.0, 4)
    assert calc.pctile(v, 0.5)[0] == 0.0
    assert calc.pctile(v, 4.0)[0] == 100.0


def test_minus_years_leap_day():
    assert calc.minus_years(dt.date(2028, 2, 29), 10) == dt.date(2018, 2, 28)
    assert calc.minus_years(dt.date(2026, 10, 28), 10) == dt.date(2016, 10, 28)


# --------------------------------------------------------------------------- 일간 지표
def test_daily_diff_ramp_is_strong_up_with_exact_threshold():
    n = 3200  # 10년 이상
    spec = DAILY_BY_ID["kr_cd91"]
    out, (dates, x) = calc.build_daily(
        spec, D, {}, kr_view("rate_kr_cd91", np.arange(n) * 0.01), {}
    )
    p = {q["label"]: q for q in out["periods"]}
    assert p["1개월"]["change"] == pytest.approx(0.20)
    assert p["1개월"]["direction"] == "up" and p["1개월"]["strong"] is True  # 문턱 = 변화 = 0.20
    assert p["1개월"]["threshold"]["flat_below"] == pytest.approx(0.20)
    assert p["3개월"]["change"] == pytest.approx(0.60)
    assert out["percentile"]["value"] == pytest.approx(100.0)
    assert out["value"]["level"] == pytest.approx((n - 1) * 0.01)
    calc.annotate_daily(out)
    assert out["asof"] == "2026-10-27" and out["asof_lag_sessions"] == 0 and out["stale"] is False
    assert out["range_1y"]["from_high"] == pytest.approx(0.0) and out["range_1y"]["n"] == 250


def test_daily_zero_change_is_flat_and_short_history_has_no_percentile():
    out, _ = calc.build_daily(
        DAILY_BY_ID["kr_cd91"], D, {}, kr_view("rate_kr_cd91", np.full(3200, 3.0)), {}
    )
    assert all(p["direction"] == "flat" and p["strong"] is False for p in out["periods"])
    short, _ = calc.build_daily(
        DAILY_BY_ID["kr_cd91"], D, {}, kr_view("rate_kr_cd91", np.arange(300) * 0.01), {}
    )
    assert (
        short["percentile"]["value"] is None and "10년 이력 없음" in short["percentile"]["reason"]
    )
    # 이력이 짧아 기간 변화 표본이 20개 미만이면 방향을 판정하지 않는다
    assert "direction" in short["periods"][0]  # 300개 중 10년 창 안 표본은 충분
    tiny, _ = calc.build_daily(
        DAILY_BY_ID["kr_cd91"], D, {}, kr_view("rate_kr_cd91", np.arange(70) * 0.01), {}
    )
    assert "direction" not in [p for p in tiny["periods"] if p["label"] == "3개월"][0]
    assert [p for p in tiny["periods"] if p["label"] == "6개월"][0]["reason"] == "관측 부족"


def test_stale_daily_when_asof_lags_more_than_three_weekdays():
    # 기준일 다음 날부터 D 전날까지 평일 수: 10-21(수) -> 22·23·26·27 = 4(지연), 10-22(목) -> 3(지연 아님)
    for end, lag, stale in ((dt.date(2026, 10, 21), 4, True), (dt.date(2026, 10, 22), 3, False)):
        dates = wdays(300, end=end)
        out, _ = calc.build_daily(
            DAILY_BY_ID["kr_cd91"], D, {}, kr_view("rate_kr_cd91", np.arange(300) * 0.01, dates), {}
        )
        calc.annotate_daily(out)
        assert out["asof_lag_sessions"] == lag and out["stale"] is stale


def test_pct_kind_uses_ratio_and_skips_nonpositive():
    x = 100 * np.exp(np.cumsum(np.full(3200, 0.001)))
    out, _ = calc.build_daily(DAILY_BY_ID["usdkrw"], D, {}, kr_view("fx_usdkrw_ecos", x), {})
    p = {q["label"]: q for q in out["periods"]}
    assert p["1주"]["change"] == pytest.approx((math.exp(0.005) - 1) * 100, abs=1e-5)
    assert out["range_1y"]["unit"] == "%"


def test_ratio_kind_exposes_ratio_only_and_flat_when_constant():
    out, _ = calc.build_daily(
        DAILY_BY_ID["tv_kospi"], D, {}, kr_view("trdval_kospi_ecos", np.full(3200, 5e4)), {}
    )
    assert out["value"] == {"level": 1.0, "display": "1.00배"}  # 평균 조원 값은 body에 없다(R2)
    assert out["periods"][0]["direction"] == "flat" and out["periods"][0]["ratio"] == 1.0
    assert "avg20_tril" not in str(out) and "avg250_tril" not in str(out)
    with pytest.raises(calc.MacroInputError, match="series_too_short"):
        calc.build_daily(
            DAILY_BY_ID["tv_kospi"], D, {}, kr_view("trdval_kospi_ecos", np.full(100, 1.0)), {}
        )


def test_sum_kind_flow_direction_and_reference_period():
    x = np.full(3200, 1000.0)  # 일 1000(억) -> 20세션 합 = 2조원
    out, _ = calc.build_daily(
        DAILY_BY_ID["fn_kospi"], D, {}, kr_view("foreign_net_kospi_ecos", x), {}
    )
    p = {q["label"]: q for q in out["periods"]}
    assert p["1개월"]["change"] == pytest.approx(2.0) and p["1개월"]["direction"] == "up"
    assert p["6개월"]["reference_only"] is True and p["1주"]["reference_only"] is False
    assert out["value"]["label"] == "20세션 순매수 합"


def test_missing_series_raises():
    with pytest.raises(calc.MacroInputError, match="series_missing:rate_kr_cd91"):
        calc.build_daily(DAILY_BY_ID["kr_cd91"], D, {}, {}, {})


# --------------------------------------------------------------------------- 사본 이음
def _us_view(sid, values, end):
    dates = wdays(len(values), end=end)
    return {
        sid: {d: (float(v), d + dt.timedelta(days=1)) for d, v in zip(dates, values, strict=True)}
    }, dates


def test_splice_with_copy_after_last_macro_date_and_rounding():
    us, dates = _us_view("VIXCLS", np.full(3200, 20.0), dt.date(2026, 10, 21))
    last = dates[-1]
    cdates = wdays(10, end=dt.date(2026, 10, 27))  # 10-14 ~ 10-27, 겹치는 날 포함
    kr = {"global_vix": {d: (30.123456, d) for d in cdates}}
    spec = DAILY_BY_ID["vix"]
    out, (dd, x) = calc.build_daily(spec, D, us, kr, {})
    s = out["splice"]
    assert s["copy_series_id"] == "global_vix" and s["copy_rounding"] == 2
    assert (
        s["copy_first_date"] > last.isoformat() and s["macro_series_last_date"] == last.isoformat()
    )
    assert all(d > last for d in dd[-s["copy_rows"] :]) and dd[-1] == dt.date(2026, 10, 27)
    assert x[-1] == 30.12  # VIX 사본만 소수 둘째 자리 반올림
    assert x[len(x) - s["copy_rows"] - 1] == 20.0  # 겹치는 날은 macro_series 값
    assert out["asof"] == "2026-10-27"


def test_splice_without_copy_rows_uses_macro_series_only_and_new_copy_ids():
    us, dates = _us_view("DGS3MO", np.full(3200, 4.0), dt.date(2026, 10, 21))
    out, (dd, _) = calc.build_daily(
        DAILY_BY_ID["us_3m"], D, us, {}, {}
    )  # 한국 표에 rate_us3m 행이 없다
    assert out["splice"]["copy_series_id"] == "rate_us3m" and out["splice"]["copy_rows"] == 0
    assert out["splice"]["splice_point"] == "이을 사본 행 없음 (macro_series만)"
    assert dd[-1] == dates[-1]
    # 다섯 사본의 대응이 spec에 있다
    from modeler.scores.macro_context.spec import SPLICE_MAP

    for cid, usid in {
        "rate_us3m": "DGS3MO",
        "spread_us_baa10y": "BAA10Y",
        "fx_usd_broad": "DTWEXBGS",
        "commodity_brent_fred": "DCOILBRENTEU",
        "rate_us10y_real": "DFII10",
    }.items():
        assert SPLICE_MAP[cid] == usid
    # 사본이 있으면 같은 규칙으로 잇는다(반올림 없음)
    kr = {
        "rate_us3m": {
            dt.date(2026, 10, 22): (4.123456, dt.date(2026, 10, 23)),
            dt.date(2026, 10, 27): (4.5, dt.date(2026, 10, 28)),
        }
    }
    out2, (_, x2) = calc.build_daily(DAILY_BY_ID["us_3m"], D, us, kr, {})
    assert out2["splice"]["copy_rows"] == 2 and x2[-2] == 4.123456 and x2[-1] == 4.5


def test_derived_krus_uses_common_dates_only():
    dates = wdays(300)
    series = {"us_10y": (dates, np.full(300, 5.0)), "kr_gov10y": (dates[5:], np.full(295, 3.5))}
    out, (dd, x) = calc.build_daily(DAILY_BY_ID["krus_10y"], D, {}, {}, series)
    assert len(dd) == 295 and x[-1] == pytest.approx(1.5) and out["splice"]["common_dates"] == 295


# --------------------------------------------------------------------------- 가용일 보기
def test_view_filters_available_rows_and_keeps_latest_and_first():
    d1, d2 = dt.date(2026, 10, 1), dt.date(2026, 10, 2)
    rows = {
        "S": [
            (d1, 1.0, dt.date(2026, 10, 2)),
            (d1, 1.5, dt.date(2026, 10, 20)),
            (d1, 9.9, dt.date(2026, 11, 30)),
            (d2, 2.0, dt.date(2026, 10, 3)),
            (dt.date(2026, 10, 3), 3.0, dt.date(2026, 10, 29)),
        ]
    }
    cur, first = view(rows, D)  # D=10-28: 11-30 개정과 10-29 가용 행은 못 쓴다
    assert cur["S"] == {d1: (1.5, dt.date(2026, 10, 20)), d2: (2.0, dt.date(2026, 10, 3))}
    assert first["S"][d1] == dt.date(2026, 10, 2)  # 처음 값이 생긴 가용일
    cur5, _ = view(rows, dt.date(2026, 10, 10))
    assert cur5["S"][d1] == (1.0, dt.date(2026, 10, 2))  # 그날 가용분만 쓴다
    assert view(rows, dt.date(2026, 9, 1)) == ({}, {})


# --------------------------------------------------------------------------- 월간
def monthly_view(sid, values, last=dt.date(2026, 9, 1), skip=(), lag_days=20):
    ms, y, m = [], last.year, last.month
    for _ in values:
        ms.append(dt.date(y, m, 1))
        y, m = (y - 1, 12) if m == 1 else (y, m - 1)
    ms = ms[::-1]
    cur, first = {}, {}
    for mm, v in zip(ms, values, strict=True):
        if mm in skip:
            continue
        nxt = dt.date(mm.year + (mm.month == 12), mm.month % 12 + 1, 1)
        rel = nxt + dt.timedelta(days=lag_days)
        cur[mm] = (float(v), rel)
        first[mm] = rel
    return {sid: cur}, {sid: first}


def test_monthly_aligns_by_date_not_row_order_and_breaks_streak_at_missing_month():
    n = 160
    ramp = np.arange(n, dtype=float) + 100
    spec = MONTHLY_BY_ID["kr_sent"]  # 대표: 3개월 변화
    kr, _ = monthly_view("macro_consumer_sentiment", ramp)
    out, (ks, vm) = calc.build_monthly(spec, D, {}, kr)
    assert out["asof"] == "2026-09" and out["streak"]["months"] >= 100  # 끊김 없이 이어진 증가
    # 끝에서 6번째 달이 비면 3개월 변화가 NaN인 달이 둘이라 연속이 2개월에서 끊긴다
    gap = dt.date(2026, 4, 1)
    kr2, _ = monthly_view("macro_consumer_sentiment", ramp, skip={gap})
    out2, (ks2, vm2) = calc.build_monthly(spec, D, {}, kr2)
    assert out2["missing_months_inside"] == 1
    assert out2["streak"]["months"] == 2 and out2["streak"]["text"] == "2개월 연속 상승"
    # 행 순번으로 맞췄다면 3행 전이지만 날짜로는 4개월 전이다: 2026-09의 3개월 변화 = 09 - 06
    d3 = [m for m in out2["measures"] if m["key"] == "d3"][0]
    assert d3["change"] == pytest.approx(3.0)
    assert out2["value"]["level"] == pytest.approx(float(ramp[-1]))


def test_monthly_yoy_nan_when_comparison_month_missing_is_tolerated():
    ramp = np.linspace(100, 180, 160)
    spec = MONTHLY_BY_ID["kr_cpi"]  # 값 = 전년비
    kr, _ = monthly_view("macro_cpi", ramp, skip={dt.date(2025, 9, 1)})
    out, _ = calc.build_monthly(spec, D, {}, kr)
    assert out["value"]["level"] is None and out["value"]["display"] is None
    assert out["percentile"]["value"] is None and "비어" in out["percentile"]["reason"]
    assert out["measures"][0].get("direction") is None
    assert calc.conds({**out, "kind": "monthly"}) == {"a": [], "b": None, "c": None, "d": None}


def test_monthly_release_date_and_status_from_three_year_median_delay():
    ramp = np.linspace(100, 180, 160)
    spec = MONTHLY_BY_ID["kr_sent"]
    kr, kr_first = monthly_view("macro_consumer_sentiment", ramp)  # 월말 + 20일 가용
    out, series = calc.build_monthly(spec, D, {}, kr)
    calc.annotate_monthly(out, spec, D, {}, kr_first, series)
    assert out["release_date"] == "2026-10-21" and out["asof"] == "2026-09"
    assert out["status"] == "정상" and out["stale"] is False and out["delay_sample_n"] > 0
    assert out["prev_value"]["asof"] == "2026-08"
    # 마지막 두 달이 안 나오면 지연 2개월
    kr2, kr_first2 = monthly_view("macro_consumer_sentiment", ramp[:-2], last=dt.date(2026, 7, 1))
    out2, series2 = calc.build_monthly(spec, D, {}, kr2)
    calc.annotate_monthly(out2, spec, D, {}, kr_first2, series2)
    assert (
        out2["status"] == "지연 2개월"
        and out2["stale"] is True
        and out2["expected_last_month"] == "2026-09"
    )


def test_conds_a_b_c_from_periods():
    def period(label, h, change, direction, strong, flat, strong_from):
        return dict(
            label=label,
            h=h,
            change=change,
            direction=direction,
            direction_label="x",
            strong=strong,
            threshold=dict(flat_below=flat, strong_from=strong_from, n=100),
        )

    o = dict(
        kind="diff",
        id="us_2y",
        periods=[
            period("1개월", 20, -0.2, "down", False, 0.1, 0.5),
            period("3개월", 60, 0.3, "up", False, 0.1, 0.5),
            period("6개월", 120, 1.0, "up", True, 0.2, 0.9),
        ],
        percentile=dict(value=96.0),
    )
    c = calc.conds(o)
    assert [x["label"] for x in c["a"]] == ["6개월"] and c["a"][0]["mult"] == pytest.approx(
        1.0 / 0.9, abs=1e-6
    )
    assert c["b"]["short"] == "1개월" and c["b"]["long"] == "6개월"
    assert c["b"]["mag"] == pytest.approx(min(0.2 / 0.1, 1.0 / 0.2))
    assert c["c"] == {"p": 96.0}
    o["percentile"] = dict(value=50.0)
    assert calc.conds(o)["c"] is None
    # 보합 기간은 중기·단기 비교에서 뺀다
    o2 = dict(
        kind="diff",
        id="us_2y",
        percentile={},
        periods=[
            period("1개월", 20, 0.0, "flat", False, 0.1, 0.5),
            period("3개월", 60, 0.3, "up", False, 0.1, 0.5),
            period("6개월", 120, 0.5, "up", False, 0.2, 0.9),
        ],
    )
    assert calc.conds(o2)["b"] is None
