"""MRS 장부·지표·placebo·게이트 (사전등록 §5, §0-2, §0-3) — 합성 데이터만."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from modeler.scores.common.cash import build_cash_account
from modeler.scores.mrs import backtest as B
from modeler.scores.mrs import config as C
from modeler.scores.mrs import weights as W


def _grid(n: int, start: date = date(2000, 1, 3)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _by_rule(rows):
    return {r["rule"]: r for r in rows}


# --------------------------------------------------------------------------- 장부 시점
def test_ledger_timing_hand_example():
    g = _grid(6)
    price = np.array([100.0, 110.0, 121.0, 60.5, 66.55, 66.55])
    mrs = np.array([50.0, 20.0, 80.0, 20.0, 50.0, 50.0])  # w = .75 .5 1 .5 .75 .75
    r_cash = np.array([0.01] * 5 + [np.nan])
    led = B.build_ledger(
        dates=g,
        price=price,
        mrs=mrs,
        sigma20=np.full(6, np.nan),
        sigma_target=np.full(6, np.nan),
        r_cash=r_cash,
        cost_rt_bp=100.0,
    )
    np.testing.assert_allclose(led.w["mrs"], [0.75, 0.5, 1.0, 0.5, 0.75, 0.75])
    # a_i = w_{i-1}, a_0 = 1.0
    np.testing.assert_allclose(led.a["mrs"], [1.0, 0.75, 0.5, 1.0, 0.5, 0.75])
    np.testing.assert_allclose(led.r_bh[:5], [0.1, 0.1, -0.5, 0.1, 0.0])
    oneway = 0.005
    # i=0: 비용 없음
    assert led.r["mrs"][0] == pytest.approx(1.0 * 0.1)
    # i=1: g_0에 정한 w_0=.75가 g_1->g_2 아니라 g_1 종가 체결 -> g_1->g_2 구간에 적용
    assert led.r["mrs"][1] == pytest.approx(0.75 * 0.1 + 0.25 * 0.01 - 0.25 * oneway)
    # i=2: w_1=.5 (g_1 결정, g_2 체결), 비용은 |.5-.75| 를 i=2에서
    assert led.r["mrs"][2] == pytest.approx(0.5 * -0.5 + 0.5 * 0.01 - 0.25 * oneway)
    assert led.r["mrs"][2] == pytest.approx(-0.24625)
    # i=3: a_3 = w_2 = 1.0, 비용 |1-.5|
    assert led.r["mrs"][3] == pytest.approx(1.0 * 0.1 - 0.5 * oneway)
    # 마지막 구간은 가격이 없어 NaN
    assert np.isnan(led.r["mrs"][5])
    # 보유 수익에는 비용이 없고 mrs 장부가 아니라 r_den이다
    np.testing.assert_allclose(led.r_den[:5], led.r_bh[:5])
    frame = led.to_frame()
    assert frame.height == 6 and "a_vm_nofloor" in frame.columns


# --------------------------------------------------------------------------- MDD·CAGR
def test_mdd_hand_path_includes_v0():
    m = B.path_metrics(np.array([-0.1, 0.05]), date(2000, 1, 1), date(2000, 12, 31))
    assert m["MDD"] == pytest.approx(0.1)  # V0=1이 최댓값
    m = B.path_metrics(np.array([0.1, -0.2, 0.1]), date(2000, 1, 1), date(2001, 1, 1))
    # V: 1, 1.1, 0.88, 0.968 -> DD = 0.2
    assert m["MDD"] == pytest.approx(0.2)
    m = B.path_metrics(np.array([0.1, 0.1]), date(2000, 1, 1), date(2001, 1, 1))
    assert m["MDD"] == 0.0 and m["Calmar"] is None  # MDD=0이면 Calmar null


def test_cagr_calendar_years():
    first, last_next = date(2000, 1, 3), date(2002, 1, 3)  # 731일
    years = 731 / 365.25
    m = B.path_metrics(np.array([0.1, 0.1]), first, last_next)  # V_end=1.21 (세션 수와 무관)
    assert m["CAGR"] == pytest.approx(1.21 ** (1 / years) - 1)
    m2 = B.path_metrics(np.array([0.1] * 2 + [0.0] * 500), first, last_next)
    assert m2["CAGR"] == pytest.approx(m["CAGR"])
    assert m["Calmar"] is None


def _random_world(n=60, seed=1):
    rng = np.random.default_rng(seed)
    g = _grid(n)
    price = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    mrs = rng.uniform(0, 100, n)
    sub = {k: rng.uniform(0, 100, n) for k in "TVL"}
    sig = np.abs(rng.normal(0.01, 0.003, n)) + 0.002
    return g, price, mrs, sub, sig, np.full(n, 0.01)


def test_null_cash_session_excluded_from_all_rules():
    g, price, mrs, sub, sig, tgt = _random_world()
    n = len(g)
    r_cash = np.full(n, 0.0001)
    r_cash[10] = np.nan
    rows = B.evaluate_window(
        dates=g, price=price, mrs=mrs, sub_scores=sub, sigma20=sig, sigma_target=tgt,
        r_cash=r_cash, start=g[0], end=g[-1], cost_rt_bp=60.0,
    )  # fmt: skip
    r = _by_rule(rows)
    assert all(x["n_excluded"] == 1 for x in rows)
    assert all(x["n_sessions"] == n - 2 - 1 for x in rows)  # i=1..n-2 중 하나 뺌
    # 분모 MDD는 같은 세션 집합에서 잰 보유 경로다
    r_bh = B.forward_returns(price)
    s = [i for i in range(1, n - 1) if i != 10]
    exp = B.path_metrics(r_bh[s], g[s[0]], g[s[-1] + 1])
    assert r["mrs"]["MDD_denom"] == pytest.approx(exp["MDD"])
    assert r["vm"]["MDD_denom"] == r["mrs"]["MDD_denom"] == r["vm_nofloor"]["MDD_denom"]
    # availability_ok 거짓 세션도 같이 빠진다
    av = np.ones(n, dtype=bool)
    av[20] = False
    rows2 = B.evaluate_window(
        dates=g, price=price, mrs=mrs, sub_scores=sub, sigma20=sig, sigma_target=tgt,
        r_cash=r_cash, start=g[0], end=g[-1], cost_rt_bp=60.0, availability_ok=av,
    )  # fmt: skip
    assert rows2[0]["n_excluded"] == 2


def test_window_session_set_t0_and_end():
    g = _grid(10)
    mrs = np.array([np.nan, np.nan, 50, 50, 50, 50, 50, 50, 50, 50.0])
    t0, idx = B.window_session_set(g, mrs, g[0], g[7])
    assert t0 == 2
    assert idx.tolist() == [3, 4, 5, 6]  # g_{i+1} <= g[7]
    t0, idx = B.window_session_set(g, mrs, g[4], g[-1])
    assert t0 == 4 and idx.tolist() == [5, 6, 7, 8]


# --------------------------------------------------------------------------- G2 분기
def test_g2_when_denominator_cagr_nonpositive():
    g = _grid(80)
    price = 100 * np.exp(-0.002 * np.arange(80))  # 꾸준한 하락
    mrs = np.full(80, 10.0)  # w = 0.5 -> 손실이 절반
    rows = B.evaluate_window(
        dates=g, price=price, mrs=mrs, sub_scores={}, sigma20=np.full(80, np.nan),
        sigma_target=np.full(80, np.nan), r_cash=np.zeros(80), start=g[0], end=g[-1],
        cost_rt_bp=0.0,
    )  # fmt: skip
    r = _by_rule(rows)["mrs"]
    assert r["CAGR_denom"] < 0
    assert r["RET_ratio"] is None  # 비율 정의 불가
    assert r["CAGR"] > r["CAGR_denom"]
    assert r["g2"] is True
    # 순수 함수: 분모 <= 0이면 CAGR 직접 비교
    assert B.g2({"CAGR": -0.05, "CAGR_denom": -0.04, "RET_ratio": None}) is False
    assert B.g2({"CAGR": -0.04, "CAGR_denom": -0.04, "RET_ratio": None}) is True
    # 분모 > 0이면 0.90 문턱
    assert B.g2({"CAGR": 0.09, "CAGR_denom": 0.1, "RET_ratio": 0.9}) is True
    assert B.g2({"CAGR": 0.08, "CAGR_denom": 0.1, "RET_ratio": 0.8}) is False
    assert B.g2({"CAGR": None, "CAGR_denom": 0.1, "RET_ratio": None}) is False


def test_g5_needs_both_conditions():
    vm = {"MDD_ratio": 0.8, "RET_ratio": 0.95, "CAGR": 0.095, "CAGR_denom": 0.1}
    both = {"MDD_ratio": 0.7, "RET_ratio": 0.97, "CAGR": 0.097, "CAGR_denom": 0.1}
    only_mdd = {"MDD_ratio": 0.7, "RET_ratio": 0.90, "CAGR": 0.09, "CAGR_denom": 0.1}
    only_ret = {"MDD_ratio": 0.85, "RET_ratio": 0.99, "CAGR": 0.099, "CAGR_denom": 0.1}
    tie = {"MDD_ratio": 0.8, "RET_ratio": 0.99, "CAGR": 0.099, "CAGR_denom": 0.1}
    assert B.beats(both, vm) is True
    assert B.beats(only_mdd, vm) is False
    assert B.beats(only_ret, vm) is False
    assert B.beats(tie, vm) is False  # 엄격 부등호
    # 분모 CAGR <= 0이면 CAGR 직접 비교
    vm_n = {"MDD_ratio": 0.8, "RET_ratio": None, "CAGR": -0.05, "CAGR_denom": -0.04}
    m_n = {"MDD_ratio": 0.7, "RET_ratio": None, "CAGR": -0.03, "CAGR_denom": -0.04}
    assert B.beats(m_n, vm_n) is True
    assert B.beats({**m_n, "CAGR": -0.06}, vm_n) is False
    assert B.beats({**m_n, "MDD_ratio": None}, vm_n) is False


# --------------------------------------------------------------------------- 게이트 분류·등급
def test_gate_class_first_failure_ordering():
    T, F = True, False
    assert B.gate_class([F, F, F, F, F]) == "D"
    assert B.gate_class([F, T, T, T, T]) == "D"  # G1 먼저
    assert B.gate_class([T, F, F, F, F]) == "C"
    assert B.gate_class([T, T, F, T, T]) == "R"
    assert B.gate_class([T, T, T, F, F]) == "S"
    assert B.gate_class([T, T, T, T, F]) == "X"
    assert B.gate_class([T, T, T, T, T]) == "PASS"
    assert B.gate_class([T, T, None, T, T]) == "R"  # null은 실패


def test_g1_g3_g4():
    assert B.g1(0.8) is True and B.g1(0.81) is False and B.g1(None) is False
    assert B.g3(0.0392) is True and B.g3(0.05) is False and B.g3(None) is False
    assert B.g4([-0.01, -0.02, 0.01]) is True
    assert B.g4([-0.01, 0.02, 0.01]) is False
    assert B.g4([-0.01, -0.02, None]) is True
    assert B.g4([-0.01, None, None]) is False  # 살아 있는 것만 센다


def test_official_grade_a_vs_b():
    kr = {"gate_class": "PASS"}
    us_ok = {"g1": True, "g2": True}
    assert B.official_grade(kr, us_ok, {"MDD_ratio": 0.9}) == "A"
    assert B.official_grade(kr, us_ok, {"MDD_ratio": 1.0}) == "B"  # < 1 이어야 한다
    assert B.official_grade(kr, {"g1": True, "g2": False}, {"MDD_ratio": 0.9}) == "B"
    assert B.official_grade(kr, us_ok, {"MDD_ratio": None}) == "B"
    assert B.official_grade(kr, None, None) == "B"
    for letter in "DCRSX":
        assert B.official_grade({"gate_class": letter}, us_ok, {"MDD_ratio": 0.5}) == letter
    assert B.official_grade(None, us_ok, None) is None


# --------------------------------------------------------------------------- placebo
def test_placebo_shifts_deterministic():
    s = B.placebo_shifts()
    assert s == B.placebo_shifts()
    assert len(s) == 50 == len(set(s)) and s == sorted(s)
    assert all(20 <= k <= 1000 for k in s)
    rng = np.random.default_rng(20260924)
    assert s == sorted(int(k) for k in rng.choice(np.arange(20, 1001), size=50, replace=False))


def test_placebo_p_formula():
    assert B.placebo_p(0.3, np.array([0.1, 0.3, 0.5])) == pytest.approx(3 / 4)
    assert B.placebo_p(0.05, np.full(50, 0.2)) == pytest.approx(1 / 51)
    assert B.placebo_p(0.5, np.full(50, 0.2)) == pytest.approx(51 / 51)


def test_perfect_timer_on_crash_path_gets_minimum_p():
    n = 1300
    g = _grid(n)
    crash = 600
    r = np.full(n - 1, 0.0005)
    r[crash : crash + 40] = -0.015
    price = 100 * np.concatenate([[1.0], np.cumprod(1 + r)])
    mrs = np.full(n, 90.0)  # w = 1
    mrs[crash - 1 : crash + 42] = 10.0  # 폭락 직전~직후만 w = 0.5 (a_i = w_{i-1}이라 한 칸 앞)
    rows = B.evaluate_window(
        dates=g, price=price, mrs=mrs, sub_scores={}, sigma20=np.full(n, np.nan),
        sigma_target=np.full(n, np.nan), r_cash=np.zeros(n), start=g[0], end=g[-1],
        cost_rt_bp=0.0,
    )  # fmt: skip
    row = _by_rule(rows)["mrs"]
    assert row["n_sessions"] > 1000
    assert row["MDD_ratio"] < 0.8
    assert row["placebo_p"] == pytest.approx(1 / 51)
    assert row["g1"] and row["g3"]
    # 세션 1,000 이하면 placebo 불가 -> null, G3 실패
    short = B.evaluate_window(
        dates=g[:400], price=price[:400], mrs=mrs[:400], sub_scores={},
        sigma20=np.full(400, np.nan), sigma_target=np.full(400, np.nan),
        r_cash=np.zeros(400), start=g[0], end=g[399], cost_rt_bp=0.0,
    )  # fmt: skip
    assert _by_rule(short)["mrs"]["placebo_p"] is None and _by_rule(short)["mrs"]["g3"] is False


def test_placebo_cost_recomputed():
    a = np.array([1.0, 1.0, 0.5, 0.5, 1.0, 1.0])
    z = np.zeros(6)
    mdd = B.placebo_mdds(a, z, z, 0.01, [2])
    # 이동 후 a' = [1,1,1,1,.5,.5]: 수익 0, 비용은 변화에서만(원형: j=0 앞 = 마지막 0.5)
    ap = np.roll(a, 2)
    step = np.abs(ap - np.roll(ap, 1))
    v = np.concatenate([[1.0], np.cumprod(1 - step * 0.01)])
    assert mdd[0] == pytest.approx(np.max(1 - v / np.maximum.accumulate(v)))
    assert step[0] == 0.5  # 원형 경계 비용 포함


# --------------------------------------------------------------------------- HAC t
def _dense_hac_t(y, d, idx, lag):
    """독립 구현: 쌍마다 이중 루프 샌드위치. lag=0이면 White(HC0)."""
    n = len(y)
    X = np.column_stack([np.ones(n), d])
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    e = y - X @ beta
    u = X * e[:, None]
    meat = u.T @ u
    for i in range(n):
        for j in range(n):
            dist = idx[j] - idx[i]
            if 1 <= dist <= lag:
                w = 1 - dist / (lag + 1.0)
                meat = meat + w * (np.outer(u[i], u[j]) + np.outer(u[j], u[i]))
    cov = xtx_inv @ meat @ xtx_inv
    return beta[1] / np.sqrt(cov[1, 1])


@pytest.mark.parametrize("lag", [0, 5, 20])
def test_hac_t_matches_independent_implementation(lag):
    rng = np.random.default_rng(7)
    n = 120
    idx = np.cumsum(rng.integers(1, 4, n))  # 간격 있는 격자 인덱스
    d = (rng.uniform(size=n) < 0.5).astype(float)
    y = 0.01 * d + np.convolve(rng.normal(size=n + 4), np.ones(5) / 5, "valid")[:n]
    got = B._hac_t(y, d, idx, lag)
    assert got == pytest.approx(_dense_hac_t(y, d, idx, lag), rel=1e-9)


# --------------------------------------------------------------------------- Q 통계
def test_quintile_stats_vs_manual_loop():
    rng = np.random.default_rng(3)
    n = 300
    g = _grid(n)
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    score = rng.uniform(0, 100, n)
    score[50] = np.nan
    t0, end = 10, g[-1]
    got = B.quintile_stats(price, score, g, t0, end, with_mdd=True)
    ts = [t for t in range(t0, n - 21) if np.isfinite(score[t])]
    f = np.array([price[t + 21] / price[t + 1] - 1 for t in ts])
    sc = score[ts]
    q20, q80 = np.quantile(sc, [0.2, 0.8])
    q1, q5 = sc <= q20, sc >= q80
    assert got["gap"] == pytest.approx(f[q1].mean() - f[q5].mean())
    assert got["n_q1"] == q1.sum() and got["n_q5"] == q5.sum()

    def mdd(t):
        path = price[t + 1 : t + 22]
        return float(np.max(1 - path / np.maximum.accumulate(path)))

    ta = np.array(ts)
    assert got["Q1_MDD"] == pytest.approx(np.mean([mdd(t) for t in ta[q1]]))
    assert got["Q5_MDD"] == pytest.approx(np.mean([mdd(t) for t in ta[q5]]))
    sel = q1 | q5
    assert got["t"] == pytest.approx(
        _dense_hac_t(f[sel], q1[sel].astype(float), ta[sel], C.HAC_LAG), rel=1e-9
    )
    # 끝 날짜가 앞 수익을 자르면 표본이 줄어든다
    cut = B.quintile_stats(price, score, g, t0, g[200], with_mdd=False)
    assert cut["n_q1"] + cut["n_q5"] < got["n_q1"] + got["n_q5"]


def test_quintile_degenerate_scores_are_null():
    n = 100
    g = _grid(n)
    price = np.linspace(100, 120, n)
    out = B.quintile_stats(price, np.full(n, 50.0), g, 0, g[-1])
    assert out["gap"] is None and out["t"] is None


def test_sub_gap_sign_for_informative_score():
    rng = np.random.default_rng(11)
    n = 700
    g = _grid(n)
    ret = rng.normal(0, 0.01, n)
    price = 100 * np.exp(np.cumsum(ret))
    # 점수가 앞으로의 20일 수익과 같은 방향(점수 높을수록 앞 수익 높음) -> Q1<Q5 -> gap<0
    fwd = np.array([np.log(price[min(t + 21, n - 1)] / price[min(t + 1, n - 1)]) for t in range(n)])
    good = 50 + 1000 * fwd + rng.normal(0, 1, n)
    bad = -good
    rows = B.evaluate_window(
        dates=g, price=price, mrs=good, sub_scores={"T": good, "V": bad, "L": good},
        sigma20=np.full(n, np.nan), sigma_target=np.full(n, np.nan), r_cash=np.zeros(n),
        start=g[0], end=g[-1], cost_rt_bp=0.0,
    )  # fmt: skip
    r = _by_rule(rows)["mrs"]
    assert r["sub_gap_T"] < 0 and r["sub_gap_L"] < 0 and r["sub_gap_V"] > 0
    assert r["Q1_Q5_gap"] < 0
    assert r["gate_class"] in {"D", "C", "R", "S", "X", "PASS"}
    # 하위 점수가 하나도 없으면 G4 실패
    rows2 = B.evaluate_window(
        dates=g, price=price, mrs=good, sub_scores={}, sigma20=np.full(n, np.nan),
        sigma_target=np.full(n, np.nan), r_cash=np.zeros(n), start=g[0], end=g[-1],
        cost_rt_bp=0.0,
    )  # fmt: skip
    assert _by_rule(rows2)["mrs"]["g4"] is False


# --------------------------------------------------------------------------- 고정 비중·IRP
def test_w_fix_and_beats_fix():
    g, price, mrs, sub, sig, tgt = _random_world(n=80, seed=5)
    n = len(g)
    r_cash = np.full(n, 0.0002)
    rows = B.evaluate_window(
        dates=g, price=price, mrs=mrs, sub_scores=sub, sigma20=sig, sigma_target=tgt,
        r_cash=r_cash, start=g[0], end=g[-1], cost_rt_bp=200.0,
    )  # fmt: skip
    r = _by_rule(rows)["mrs"]
    a = B.applied_weights(W.mrs_weight(mrs))
    s = np.arange(1, n - 1)
    assert r["w_fix_mean"] == pytest.approx(a[s].mean())
    r_bh = B.forward_returns(price)
    r_fix = r["w_fix_mean"] * r_bh[s] + (1 - r["w_fix_mean"]) * r_cash[s]  # 비용 0
    fx = B.path_metrics(r_fix, g[1], g[n - 1])
    assert r["MDD_fix"] == pytest.approx(fx["MDD"])
    assert r["CAGR_fix"] == pytest.approx(fx["CAGR"])
    den = B.path_metrics(r_bh[s], g[1], g[n - 1])
    assert r["MDD_ratio_fix"] == pytest.approx(fx["MDD"] / den["MDD"])
    expect = r["MDD_ratio"] < r["MDD_ratio_fix"] and r["RET_ratio"] > r["RET_ratio_fix"]
    assert r["beats_fix"] is expect
    # 비중이 안 변하는 규칙은 고정 비중과 같다 -> 둘 다 만족 못 한다
    flat = B.evaluate_window(
        dates=g, price=price, mrs=np.full(n, 50.0), sub_scores={}, sigma20=sig,
        sigma_target=tgt, r_cash=r_cash, start=g[0], end=g[-1], cost_rt_bp=0.0,
    )  # fmt: skip
    f = _by_rule(flat)["mrs"]
    assert f["beats_fix"] is False
    assert f["MDD_ratio"] == pytest.approx(f["MDD_ratio_fix"], rel=1e-9)
    # vm 행에는 고정 비중 칸이 없다
    assert _by_rule(rows)["vm"]["w_fix_mean"] is None


def test_irp_cap_protocol():
    g, price, mrs, sub, sig, tgt = _random_world(n=80, seed=9)
    n = len(g)
    r_cash = np.full(n, 0.0002)
    kw = dict(
        dates=g, price=price, mrs=mrs, sigma20=sig, sigma_target=tgt, r_cash=r_cash,
        cost_rt_bp=60.0,
    )  # fmt: skip
    led = B.build_ledger(irp_cap=0.7, **kw)
    for rule in B.RULES:
        assert led.w[rule].max() <= 0.7 + 1e-15 and led.a[rule].max() <= 0.7 + 1e-15
    np.testing.assert_allclose(led.w["mrs"], np.minimum(W.mrs_weight(mrs), 0.7))
    np.testing.assert_allclose(led.w["vm"], np.minimum(W.vm_weight(sig, tgt), 0.7))
    r_bh = B.forward_returns(price)
    np.testing.assert_allclose(led.r_den[:-1], 0.7 * r_bh[:-1] + 0.3 * r_cash[:-1])
    rows = B.evaluate_window(sub_scores=sub, start=g[0], end=g[-1], irp_cap=0.7, **kw)
    r = _by_rule(rows)["mrs"]
    s = np.arange(1, n - 1)
    den = B.path_metrics(led.r_den[s], g[1], g[n - 1])
    assert r["MDD_denom"] == pytest.approx(den["MDD"])
    assert r["w_fix_mean"] == pytest.approx(led.a["mrs"][s].mean())
    assert r["w_fix_mean"] <= 0.7
    assert r["g1"] in (True, False) and r["grade"] is None


def test_with_gates_false_leaves_gate_columns_null():
    g, price, mrs, sub, sig, tgt = _random_world()
    rows = B.evaluate_window(
        dates=g, price=price, mrs=mrs, sub_scores=sub, sigma20=sig, sigma_target=tgt,
        r_cash=np.full(len(g), 0.0001), start=g[0], end=g[-1], cost_rt_bp=60.0,
        with_gates=False, meta={"market": "KR", "protocol": "ktb_synth_cash_sens"},
    )  # fmt: skip
    for r in rows:
        assert r["market"] == "KR"
        for k in ("placebo_p", "g1", "g2", "g3", "g4", "g5", "gate_class", "grade"):
            assert r[k] is None
        assert r["MDD_ratio"] is not None and r["turnover"] is not None


def test_row_schema_and_vm_rows():
    g, price, mrs, sub, sig, tgt = _random_world()
    rows = B.evaluate_window(
        dates=g, price=price, mrs=mrs, sub_scores=sub, sigma20=sig, sigma_target=tgt,
        r_cash=np.full(len(g), 0.0001), start=g[0], end=g[-1], cost_rt_bp=60.0,
    )  # fmt: skip
    assert [r["rule"] for r in rows] == ["mrs", "vm", "vm_nofloor"]
    for r in rows:
        assert set(r) == set(B.RESULT_COLUMNS)
    vm = _by_rule(rows)["vm"]
    assert vm["g4"] is None and vm["g5"] is None and vm["gate_class"] is None
    assert vm["g1"] is not None and vm["turnover"] is not None
    m = _by_rule(rows)["mrs"]
    assert m["t0_date"] == g[0] and m["window_end"] == g[-1]
    assert m["turnover"] == pytest.approx(
        np.mean(np.abs(np.diff(B.applied_weights(W.mrs_weight(mrs)))[0 : len(g) - 2])) * 252
    )
    assert m["g5_vs_nofloor"] in (True, False)


def test_cash_returns_from_account_and_zero_cash():
    g = _grid(5)
    rates = pl.DataFrame(
        {"date": [g[0]], "realtime_start": [g[0]], "value": [3.65]},
        schema={"date": pl.Date, "realtime_start": pl.Date, "value": pl.Float64},
    )
    acct = build_cash_account(rates, g, series_id="X")
    rc = B.cash_returns(acct, 5)
    assert np.isnan(rc[-1])
    days = (g[1] - g[0]).days
    assert rc[0] == pytest.approx(0.0365 * days / 365)
    assert np.all(B.zero_cash(5) == 0.0)
