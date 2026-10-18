from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from modeler.scores.etf_baseline import lines as L
from modeler.scores.etf_baseline.inputs import build_return_panel
from modeler.scores.etf_baseline.ledger import simulate


def _days(n, start=date(2020, 1, 1)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _frame(days, close, shares=None):
    shares = shares or [1000.0] * len(days)
    return pl.DataFrame({"date": days, "close": close, "shares": shares})


def test_dividend_goes_to_trading_day_before_record_date():
    days = _days(6)  # 월~금, 월
    close = [100.0] * 6
    rec = days[3]  # 기준일 = 4번째 거래일 -> 직전 거래일 = days[2]
    dist = pl.DataFrame({"code": ["X"], "record_date": [rec], "amount": [2.0]})
    p = build_return_panel(days, {"X": _frame(days, close)}, dist)
    r = p.ret["X"]
    assert r[2] == pytest.approx(0.02)
    assert r[3] == 0 and r[1] == 0
    assert p.dist_added["X"] == 1


def test_dividend_on_weekend_record_date_uses_previous_trading_day():
    days = _days(8)
    sat = days[4] + timedelta(days=1)  # 금 다음 토요일
    dist = pl.DataFrame({"code": ["X"], "record_date": [sat], "amount": [1.0]})
    p = build_return_panel(days, {"X": _frame(days, [50.0] * 8)}, dist)
    assert p.ret["X"][4] == pytest.approx(1 / 50)


def test_split_detected_and_adjusted():
    days = _days(4)
    close = [100.0, 100.0, 25.0, 25.0]
    shares = [1000.0, 1000.0, 4000.0, 4000.0]
    p = build_return_panel(days, {"X": _frame(days, close, shares)}, None)
    assert len(p.split_candidates) == 1 and p.split_candidates[0]["date"] == days[2]
    assert p.ret["X"][2] == pytest.approx(0.0)
    assert p.jumps == []


def test_big_move_without_share_change_is_reported_not_fixed():
    days = _days(3)
    p = build_return_panel(days, {"X": _frame(days, [100.0, 50.0, 50.0])}, None)
    assert p.split_candidates == [] and len(p.jumps) == 1


def test_missing_close_after_listing_reported():
    days = _days(4)
    f = _frame([days[0], days[1], days[3]], [10.0, 10.0, 10.0])
    p = build_return_panel(days, {"X": f}, None)
    assert p.missing_days["X"] == [days[2]]


def test_signal_not_ready_before_252_and_no_lookahead():
    rng = np.random.default_rng(0)
    T = 600
    a = rng.normal(0, 0.01, T)
    b = rng.normal(0, 0.01, T)
    a[:5] = np.nan
    b[:5] = np.nan
    full = L.risk_share_signal(a, b)
    first = 5 + 20 - 1  # 첫 σ_20
    assert full.n_obs[first] == 1
    ready = 5 + 251  # 252번째 위험 몫 수익
    assert full.n_obs[ready] < 252
    assert np.isnan(full.w_vm[ready - 1]) and not np.isnan(full.w_vm[ready])
    # 뒤를 바꿔도 앞의 값은 같다(미래를 보지 않는다)
    a2, b2 = a.copy(), b.copy()
    a2[400:] *= 10
    b2[400:] *= 10
    cut = L.risk_share_signal(a2, b2)
    k = 399
    assert cut.w_vm[k] == pytest.approx(full.w_vm[k])
    assert cut.sigma_target[k] == pytest.approx(full.sigma_target[k])
    assert 0.5 <= np.nanmin(full.w_vm) and np.nanmax(full.w_vm) <= 1.0


def test_wvm_clip_extremes():
    T = 400
    a = np.full(T, 0.001)
    a[300:] = np.tile([0.2, -0.2], 50)  # 갑자기 변동성 폭증 -> 하한 0.5
    s = L.risk_share_signal(a, a)
    assert s.w_vm[T - 1] == 0.5
    b = np.tile([0.01, -0.01], T // 2)
    b[300:] = 0.0
    s2 = L.risk_share_signal(b, b)
    assert s2.w_vm[T - 1] == 1.0


def test_line_weights_sum_one_and_values():
    sp = L.line_specs(L.US_MAIN)
    for name in sp:
        w = sp[name].weights(w_vm=0.8, w_fix=0.4)
        assert w.sum() == pytest.approx(1.0)
    assert sp["B1"].weights() == pytest.approx([0.2] * 5 + [0.0])
    assert sp["B2"].weights() == pytest.approx([0.3, 0.3, 0.4, 0.0])
    assert sp["B3"].weights() == pytest.approx([0.15, 0.15, 0.7, 0.0])
    assert sp["B4"].weights(w_vm=0.5) == pytest.approx([0.15, 0.15, 0.4, 0.3])
    assert sp["B5"].weights(w_fix=0.4) == pytest.approx([0.2, 0.2, 0.4, 0.2])
    assert sp["R1i"].weights() == pytest.approx([0.7, 0.3])
    assert L.line_specs(L.US_U2)["B2"].assets[1] == "360750"


def test_decisions_month_end_and_fill_within_end():
    days = _days(120, date(2020, 1, 1))
    end = len(days) - 1
    first = max(d for d in days if d.month == 1)
    dec = L.decision_schedule(days, first, end)
    assert days[dec[0]] == first
    assert all(days[i].month != days[i + 1].month for i in dec)
    assert dec[-1] + 1 <= end
    with pytest.raises(ValueError):
        L.decision_schedule(days, days[3], end)


def test_buyhold_has_single_target_and_no_rebalance():
    sp = L.line_specs(L.US_MAIN)
    tg = L.build_targets(sp["R1"], [3, 10, 20])
    assert list(tg) == [4]
    ret = np.array([0.0] + [0.01] * 29).reshape(-1, 1)
    c = np.full(30, 0.0)
    c[0] = np.nan
    res = simulate(ret, c, 4, 29, tg, 0.003)
    assert res.rebalances == []
    assert res.nav[-1] == pytest.approx(1.01**25)


def test_monthly_rebalance_line_costs_only_drift():
    sp = L.line_specs(L.US_MAIN)
    T = 40
    ret = np.zeros((T, 1))
    ret[6:] = 0.01
    c = np.full(T, 0.0)
    c[0] = np.nan
    tg = L.build_targets(sp["R1i"], [3, 20])
    res = simulate(ret, c, 4, T - 1, tg, 0.003)
    # 시작 매수(첫 목표)는 기록·비용 없음, 이후 월 재조정만 기록
    assert len(res.rebalances) == 1
    assert res.rebalances[0]["idx"] == 21 and res.rebalances[0]["turnover"] > 0.0
