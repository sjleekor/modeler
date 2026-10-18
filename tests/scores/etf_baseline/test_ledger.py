from datetime import date, timedelta

import numpy as np
import pytest

from modeler.scores.etf_baseline.ledger import simulate
from modeler.scores.etf_baseline.metrics import compute_metrics, longest_underwater, max_drawdown


def _cash(T, v=0.0):
    c = np.full(T, v)
    c[0] = np.nan
    return c


def test_weights_sum_one_and_drift():
    # ETF 두 개 + 현금. 둘째 날 A +10%, B 0%, 현금 0%.
    ret = np.array([[0, 0], [0.10, 0.0], [0.0, 0.0]])
    t = {0: np.array([0.5, 0.5, 0.0])}
    res = simulate(ret, _cash(3), 0, 2, t, 0.0)
    assert res.nav[0] == pytest.approx(1.05)
    # 흘러간 비중: A 0.55/1.05, B 0.5/1.05 -> 합 1, 둘째 날 시작 비중으로 기록
    assert res.weights_held[1].sum() == pytest.approx(1.0)
    assert res.weights_held[1][0] == pytest.approx(0.55 / 1.05)


def test_fill_next_day_close_new_weight_starts_after():
    # 결정일 t=1, 체결 t=2 종가. t=2 수익은 옛 비중, t=3부터 새 비중.
    ret = np.array([[0.0], [0.0], [0.10], [0.20]])
    tg = {0: np.array([1.0, 0.0]), 2: np.array([0.5, 0.5])}
    res = simulate(ret, _cash(4), 0, 3, tg, 0.0)
    assert res.nav[1] == pytest.approx(1.10)  # t=2 : 옛 비중 100%
    assert res.nav[2] == pytest.approx(1.10 * 1.10)  # t=3 : 새 비중 50%
    assert res.rebalances[-1]["idx"] == 2


def test_new_weight_not_in_fill_day_return():
    # 체결일 수익이 달라도 체결 전 NAV는 목표 비중에 무관(비용 0)
    ret = np.array([[0.0], [0.05], [0.0]])
    a = simulate(ret, _cash(3), 0, 2, {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}, 0.0)
    b = simulate(ret, _cash(3), 0, 2, {0: np.array([1.0, 0.0]), 1: np.array([0.5, 0.5])}, 0.0)
    assert a.nav[0] == pytest.approx(b.nav[0]) == pytest.approx(1.05)


def test_cost_is_sum_abs_dw_times_one_way_and_cash_free():
    # 1.0 -> 0.7/0.3 : ETF 이동 0.3, 현금 쪽 0.3은 비용 없음
    ret = np.array([[0.0], [0.0], [0.0]])
    res = simulate(ret, _cash(3), 0, 2, {0: np.array([1.0, 0.0]), 1: np.array([0.7, 0.3])}, 0.003)
    assert res.rebalances[0]["turnover"] == pytest.approx(0.3)
    assert res.rebalances[0]["cost"] == pytest.approx(0.0009)
    assert res.nav[0] == pytest.approx(1 - 0.0009)


def test_two_etf_turnover_both_legs():
    ret = np.zeros((3, 2))
    res = simulate(
        ret, _cash(3), 0, 2, {0: np.array([0.6, 0.4, 0.0]), 1: np.array([0.4, 0.4, 0.2])}, 0.005
    )
    # |0.4-0.6| + |0.4-0.4| = 0.2 (현금 0.2는 제외)
    assert res.rebalances[0]["turnover"] == pytest.approx(0.2)


def test_start_has_no_cost_and_no_rebalance_without_targets():
    ret = np.array([[0.0], [0.1], [0.1]])
    res = simulate(ret, _cash(3), 0, 2, {0: np.array([1.0, 0.0])}, 0.01)
    assert res.rebalances == []
    assert res.nav[-1] == pytest.approx(1.21)


def test_cash_accrues_and_null_cash_day_dropped():
    ret = np.zeros((4, 1))
    c = np.array([np.nan, 0.01, np.nan, 0.01])
    res = simulate(ret, c, 0, 3, {0: np.array([0.0, 1.0])}, 0.0)
    assert list(res.idx) == [1, 3]
    assert res.nav[-1] == pytest.approx(1.0201)


def test_bad_targets_raise():
    with pytest.raises(ValueError):
        simulate(np.zeros((2, 1)), _cash(2), 0, 1, {0: np.array([0.5, 0.4])}, 0.0)


def test_mdd_and_underwater():
    d0 = date(2020, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(6)]
    nav = np.array([1.0, 1.2, 0.9, 1.0, 1.2, 1.3])
    assert max_drawdown(nav) == pytest.approx(0.9 / 1.2 - 1)
    uw = longest_underwater(dates, nav)
    assert uw["calendar_days"] == 3 and uw["trading_days"] == 3 and not uw["ongoing"]
    nav2 = np.array([1.0, 1.2, 0.9, 1.0])
    uw2 = longest_underwater(dates[:4], nav2)
    assert uw2["ongoing"] and uw2["calendar_days"] == 2


def test_metrics_turnover_and_sharpe():
    d0 = date(2020, 1, 1)
    dates = [d0 + timedelta(days=i) for i in range(4)]
    nav = np.array([1.0, 1.01, 1.0201, 1.030301])
    cash = np.zeros(3)
    tos = [(date(2020, 1, 2), 0.2), (date(2020, 1, 3), 0.4), (date(2021, 1, 1), 0.2)]
    m = compute_metrics(dates + [], nav, cash, tos)
    # 연도별 합: 2020 -> (0.2+0.4)/2 = 0.3, 2021 -> 0.1, 평균 0.2
    assert m["turnover_ann"] == pytest.approx(0.2)
    assert m["mdd"] == 0.0
    assert m["cagr"] == pytest.approx(1.030301 ** (252 / 3) - 1)
