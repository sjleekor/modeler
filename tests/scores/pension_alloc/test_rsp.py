import numpy as np
import pytest
from scipy.optimize import check_grad

from modeler.scores.pension_alloc import rsp as R
from tests.scores.pension_alloc._synth import bdays, month_ends, slot_returns


def _data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, 6, 3)), 0.01 * rng.standard_normal((n, 6)) + 0.003


@pytest.mark.parametrize("structure", ["main", "irp"])
def test_gradient_matches_finite_difference(structure):
    x, y = _data()
    th = np.random.default_rng(1).normal(0, 0.5, R.N_THETA)
    err = check_grad(
        lambda t: R.loss_and_grad(t, x, y, 0.05, structure)[0],
        lambda t: R.loss_and_grad(t, x, y, 0.05, structure)[1],
        th,
    )
    assert err < 1e-6


def test_optimizer_lowers_loss_and_prefers_the_good_slot():
    rng = np.random.default_rng(3)
    n = 80
    x = rng.standard_normal((n, 6, 3))
    # 입력의 첫 열이 높은 슬롯이 실제로 수익이 높다
    y = 0.01 * rng.standard_normal((n, 6)) + 0.02 * x[:, :, 0]
    th0 = np.zeros(R.N_THETA)
    pol = R.fit_policy(x, y, 1e-4, seed=5)
    xs = (x - pol.mean) / pol.std
    assert R.loss_and_grad(pol.theta, xs, y, 1e-4)[0] < R.loss_and_grad(th0, xs, y, 1e-4)[0]
    assert pol.theta[0] > 0  # 첫 입력 계수는 양수
    w = pol.weights(x[0])
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()


def test_fit_is_seed_deterministic():
    x, y = _data()
    a = R.fit_policy(x, y, 0.01, 7).theta
    b = R.fit_policy(x, y, 0.01, 7).theta
    assert np.array_equal(a, b)


def test_inner_split_no_overlap():
    tr, ho = R.inner_split(60, 24)
    assert len(tr) == 36 and len(ho) == 24
    assert set(tr).isdisjoint(ho) and tr.max() + 1 == ho.min() and ho.max() == 59
    with pytest.raises(ValueError):
        R.inner_split(24, 24)


def test_select_gamma_trains_only_on_earlier_part_and_ties_to_larger(monkeypatch):
    x, y = _data(60)
    seen = []
    orig = R.train_ensemble

    def spy(xr, rr, g, seeds, structure="main"):
        seen.append((len(rr), np.array_equal(rr, y[: len(rr)]), g))
        return orig(xr, rr, g, seeds, structure)

    monkeypatch.setattr(R, "train_ensemble", spy)
    g, score = R.select_gamma(x, y, (0.001, 0.01, 0.1), (1, 2), holdout=24)
    assert all(n == 36 and ok for n, ok, _ in seen) and len(seen) == 3
    assert g in score
    # 모든 γ가 같은 점수(동일 정책 모형)를 내는 경우 -> 가장 큰 γ
    monkeypatch.setattr(R, "monthly_sharpe", lambda r: 1.0)
    assert R.select_gamma(x, y, (0.1, 0.001, 0.01), (1,), holdout=24)[0] == 0.1


def test_seed_ensemble_is_mean_and_defaults():
    assert R.DEFAULT_SEEDS == tuple(range(20261018, 20261028)) and len(R.DEFAULT_SEEDS) == 10
    x, y = _data()
    pols = R.train_ensemble(x, y, 0.01, (1, 2, 3, 4))
    sw, w = R.ensemble_weights(pols, x[0])
    assert sw.shape == (4, 6) and np.allclose(w, sw.mean(axis=0)) and w.sum() == pytest.approx(1)
    assert not np.allclose(sw[0], sw[1])


def test_irp_structure_caps_risk_at_70pct():
    rng = np.random.default_rng(0)
    for _ in range(50):
        th = rng.normal(0, 5, R.N_THETA)
        xs = rng.standard_normal((6, 3)) * 3
        w = R.policy_weights(th, xs, "irp")
        risk = w[[0, 1, 4]].sum()
        assert risk <= 0.7 + 1e-12 and w.sum() == pytest.approx(1.0)
        assert (w[list(R.SAFE_SLOTS)] >= 0.1 - 1e-12).all()


def _rule(structure="main", T=700, **kw):
    ret = slot_returns(T, seed=4)
    days = bdays(T)
    me = [i for i in month_ends(days) if i >= 62]
    return (
        R.SoftmaxPolicyRule(
            ret,
            me,
            kw.pop("gammas", (0.001, 0.05)),
            seeds=(1, 2, 3),
            structure=structure,
            min_train=kw.pop("min_train", 12),
            retrain_every=kw.pop("retrain_every", 3),
            holdout=kw.pop("holdout", 5),
        ),
        ret,
    )


def test_schedule_and_no_unrealized_pairs():
    rule, ret = _rule()
    assert rule.first_k == 13
    assert [rule.retrain_k(k) for k in (13, 15, 16, 18, 19)] == [13, 13, 16, 16, 19]
    with pytest.raises(ValueError):
        rule.retrain_k(12)
    r = 19
    x, y = rule.training_set(r)
    assert len(y) == r - 1  # 쌍 0..r-2
    # 마지막 쌍의 실현 수익은 결정일 r까지 닫혀 있다
    assert rule.me[r - 2 + 1] + 1 <= rule.me[r]


@pytest.mark.parametrize("structure", ["main", "irp"])
def test_decide_outputs(structure):
    rule, _ = _rule(structure)
    out = rule.decide(14)
    assert out["weights"].sum() == pytest.approx(1.0) and out["seed_weights"].shape == (3, 6)
    assert np.allclose(out["weights"], out["seed_weights"].mean(axis=0))
    assert out["gamma"] in (0.001, 0.05)
    if structure == "irp":
        assert out["weights"][[0, 1, 4]].sum() <= 0.7 + 1e-12


def test_cutoff_property_rsp():
    rule, ret = _rule()
    for k in (13, 14, 17):
        a = rule.decide(k)
        t = int(rule.me[k])
        # 결정일 t까지만 남긴 자료 vs t 뒤를 쓰레기로 바꾼 자료
        cut = R.SoftmaxPolicyRule(
            ret[: t + 1],
            rule.me[: k + 1],
            rule.gammas,
            rule.seeds,
            rule.structure,
            rule.min_train,
            rule.retrain_every,
            rule.holdout,
        )
        junk_ret = ret.copy()
        junk_ret[t + 1 :] = np.random.default_rng(8).normal(0, 0.05, junk_ret[t + 1 :].shape)
        junk = R.SoftmaxPolicyRule(
            junk_ret,
            rule.me,
            rule.gammas,
            rule.seeds,
            rule.structure,
            rule.min_train,
            rule.retrain_every,
            rule.holdout,
        )
        for other in (cut, junk):
            b = other.decide(k)
            assert np.array_equal(a["weights"], b["weights"])
            assert np.array_equal(a["seed_weights"], b["seed_weights"])
            assert a["gamma"] == b["gamma"]


def test_gammas_required():
    with pytest.raises(ValueError):
        R.SoftmaxPolicyRule(slot_returns(200), [100], ())
    with pytest.raises(TypeError):
        R.SoftmaxPolicyRule(slot_returns(200), [100])  # type: ignore[call-arg]


# ------------------------------------------------------------------ F7·F11
def test_loss_uses_cd91_excess_with_smooth_floor():
    rng = np.random.default_rng(1)
    n = 50
    xs = rng.standard_normal((n, 6, 3))
    rr = 0.01 * rng.standard_normal((n, 6)) + 0.004
    rr[:, 5] = 0.002  # 현금 슬롯 = CD91
    th = np.zeros(R.N_THETA)
    th[R.N_FEAT + 5] = 40.0  # 사실상 전부 현금
    w = R.policy_weights(th, xs)
    assert w[:, 5].min() > 0.999
    loss, g = R.loss_and_grad(th, xs, rr, 0.0)
    assert abs(loss) < 1e-3 and np.isfinite(g).all()  # 0/0이 아니라 부드러운 하한 덕에 유한
    assert R.SHARPE_EPS == 0.005
    # 손으로 계산한 값
    th2 = np.random.default_rng(2).normal(0, 0.3, R.N_THETA)
    w2 = R.policy_weights(th2, xs)
    r_ex = (w2 * rr).sum(axis=1) - rr[:, 5]
    expect = -r_ex.mean() / np.sqrt(r_ex.var() + 0.005**2) + 0.01 * th2 @ th2
    assert R.loss_and_grad(th2, xs, rr, 0.01)[0] == pytest.approx(expect)


@pytest.mark.parametrize("structure", ["main", "irp"])
def test_gradient_with_cash_excess_matches_finite_difference(structure):
    rng = np.random.default_rng(3)
    xs = rng.standard_normal((60, 6, 3))
    rr = 0.02 * rng.standard_normal((60, 6)) + 0.003
    rr[:, 5] = 0.002 + 0.0005 * rng.standard_normal(60)
    th = rng.normal(0, 0.5, R.N_THETA)
    err = check_grad(
        lambda t: R.loss_and_grad(t, xs, rr, 0.02, structure)[0],
        lambda t: R.loss_and_grad(t, xs, rr, 0.02, structure)[1],
        th,
    )
    assert err < 1e-6


def test_all_cash_is_not_a_loss_minimum():
    rng = np.random.default_rng(5)
    n = 90
    x = rng.standard_normal((n, 6, 3))
    rr = 0.01 * rng.standard_normal((n, 6)) + 0.02 * x[:, :, 0] + 0.003
    rr[:, 5] = 0.002
    pol = R.fit_policy(x, rr, 1e-3, seed=1)
    xs = (x - pol.mean) / pol.std
    cash_th = np.zeros(R.N_THETA)
    cash_th[R.N_FEAT + 5] = 30.0
    assert R.loss_and_grad(pol.theta, xs, rr, 1e-3)[0] < R.loss_and_grad(cash_th, xs, rr, 1e-3)[0]
    # 전부 현금에서 위험 슬롯 쪽으로 움직이면 손실이 줄어든다(최소가 아니다)
    g = R.loss_and_grad(cash_th, xs, rr, 0.0)[1]
    step = cash_th - 1e-2 * g / np.linalg.norm(g)
    assert R.loss_and_grad(step, xs, rr, 0.0)[0] < R.loss_and_grad(cash_th, xs, rr, 0.0)[0]
    assert pol.weights(x[0])[5] < 0.9


def test_inner_gamma_selection_uses_excess_sharpe(monkeypatch):
    x, y = _data(60)
    seen = []
    monkeypatch.setattr(R, "monthly_sharpe", lambda r: seen.append(r.copy()) or 0.0)
    R.select_gamma(x, y, (0.01,), (1,), holdout=24)
    expect = (
        R.ensemble_weights(R.train_ensemble(x[:36], y[:36], 0.01, (1,)), x[36])[1] * y[36]
    ).sum() - y[36, 5]
    assert len(seen[0]) == 24 and seen[0][0] == pytest.approx(expect)


def _rsp_with_nan_cash(nan_days):
    rule, ret = _rule(min_train=12, retrain_every=3, holdout=5)
    ret2 = ret.copy()
    ret2[nan_days, 5] = np.nan
    rule2 = R.SoftmaxPolicyRule(ret2, rule.me, rule.gammas, rule.seeds, "main", 12, 3, 5)
    return rule, rule2


def test_months_with_nan_cash_are_dropped_and_counted():
    rule, _ = _rule(min_train=12, retrain_every=3, holdout=5)
    me = rule.me
    # 쌍 3(수익 기간 me[3]+2 .. me[4]+1) 안의 날 하나를 비운다
    bad = int(me[3]) + 5
    _, r2 = _rsp_with_nan_cash([bad])
    assert r2.pair(3) is None and r2.pair(2) is not None
    # 입력 창(63일)에 걸쳐 쌍 4·5의 입력에도 영향이 있어도 현금 입력은 NaN을 건너뛴다
    assert r2.pair(4) is not None
    # 한 달 빠져 학습 쌍이 하나 줄고, 첫 학습 결정이 한 달 늦다
    assert r2.first_k == rule.first_k + 1
    x, y = r2.training_set(15)
    assert len(y) == 14 - 1 and r2.n_dropped(15) == 1
    assert not np.isnan(y).any()
    out = r2.decide(15)
    assert out["n_dropped"] == 1 and out["weights"].sum() == pytest.approx(1.0)
    assert out["cash_weight"] == out["weights"][5]
    assert out["risk_exposure"] == pytest.approx(out["weights"][[0, 1, 4]].sum())


def test_etf_nan_makes_pair_invalid_but_cash_nan_input_does_not_raise():
    ret = slot_returns(300, seed=1)
    ret[100, 5] = np.nan
    assert np.isfinite(R.slot_inputs(ret, 150)).all()  # 현금 열 NaN은 건너뜀
    ret[140, 0] = np.nan
    with pytest.raises(ValueError):
        R.slot_inputs(ret, 150)


def test_rsp_is_bit_deterministic():
    a, _ = _rule()
    b, _ = _rule()
    assert np.array_equal(a.decide(14)["seed_weights"], b.decide(14)["seed_weights"])
