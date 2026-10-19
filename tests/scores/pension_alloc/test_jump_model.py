import numpy as np
import pytest

from modeler.scores.pension_alloc import jump_model as J
from tests.scores.pension_alloc._synth import bdays, month_ends, two_regime

SEEDS = (11, 12, 13)


def test_ewm_mean_matches_recursion_and_is_causal():
    x = np.random.default_rng(0).standard_normal(50)
    y = J.ewm_mean(x, 10)
    a = 1 - 0.5 ** (1 / 10)
    ref = [x[0]]
    for v in x[1:]:
        ref.append(a * v + (1 - a) * ref[-1])
    assert np.allclose(y, ref)
    assert np.array_equal(J.ewm_mean(x[:30], 10), y[:30])


def test_features_downside_and_sortino():
    r = np.array([0.01, -0.02, 0.0, -0.01, 0.02] * 20)
    f = J.jm_features(r)
    assert f.shape == (100, 3)
    assert (f[:, 0] >= 0).all()
    dd = J.downside_dev(r, 20)
    assert np.allclose(f[-1, 1], J.ewm_mean(r, 20)[-1] / dd[-1])
    # 하락이 없으면 하방편차 0 -> Sortino 0
    assert (J.sortino(np.full(10, 0.01), 20) == 0).all()


def test_viterbi_penalty_and_ties():
    x = np.array([[0.0]] * 5 + [[1.0]] * 5)
    th = np.array([[0.0], [1.0]])
    s, obj = J.viterbi_jump(x, th, 0.1)
    assert list(s) == [0] * 5 + [1] * 5 and obj == pytest.approx(0.1)
    # 벌점이 크면 전환하지 않는다
    s2, _ = J.viterbi_jump(x, th, 100.0)
    assert len(set(s2)) == 1
    # 단일 전환 목적함수 직접 계산과 일치
    x2 = np.array([[0.0], [0.2], [1.0], [0.9], [0.0]])
    s3, o3 = J.viterbi_jump(x2, th, 0.3)
    manual = 0.5 * ((x2[:, 0] - th[s3, 0]) ** 2).sum() + 0.3 * (np.diff(s3) != 0).sum()
    assert o3 == pytest.approx(manual)


def test_fit_recovers_two_planted_regimes():
    r, st = two_regime(2400, block=400, seed=1)
    x = J.standardize(J.jm_features(r)[300:])
    truth = st[300:]
    theta, s, obj, n_hit = J.fit_jm(x, 15.0, SEEDS)
    agree = max((s == truth).mean(), (s != truth).mean())
    assert agree > 0.85  # EWM 피쳐 지연 때문에 100%는 아니다
    bear = J.risk_state(s, r[300:])
    assert (s == bear)[truth == 1].mean() > 0.8  # 약세 구간을 위험으로 부른다
    # 목적함수가 낮은 쪽을 남긴다: 시드 하나보다 나쁘지 않다
    one = J._fit_once(x, 15.0, SEEDS[0], 100)
    assert obj <= one[2] + 1e-9


def test_labeling_rules():
    s = np.array([0, 0, 1, 1, 1])
    ex = np.array([0.01, 0.01, -0.05, 0.02, 0.02])
    assert J.risk_state(s, ex, "sum") == 1  # 상태1 합 -0.01 < 0.02
    # 합은 상태1이 낮지만 복리는 둘 다 양/음 비교: 직접 계산과 일치
    c0, c1 = np.prod(1 + ex[:2]) - 1, np.prod(1 + ex[2:]) - 1
    assert J.risk_state(s, ex, "compound") == (0 if c0 < c1 else 1)
    # 상태 이름이 뒤바뀌어도 위험은 낮은 쪽
    assert J.risk_state(1 - s, ex) == 0
    with pytest.raises(ValueError):
        J.risk_state(s, ex, "x")


def _rule(T=600, **kw):
    r, _ = two_regime(T, block=120, seed=3)
    cash = np.full(T, 0.00005)
    days = bdays(T)
    cfg = J.JMConfig(
        lambdas=kw.pop("lambdas", (5.0, 30.0)),
        seeds=SEEDS,
        window_len=kw.pop("window_len", 150),
        val_months=kw.pop("val_months", 6),
        **kw,
    )
    return J.JumpModelRule(r, cash, month_ends(days), cfg), r, cash


def test_config_requires_grid_and_seeds():
    with pytest.raises(TypeError):
        J.JMConfig()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        J.JMConfig(lambdas=(), seeds=(1,))


def test_online_state_uses_only_past_window():
    rule, r, cash = _rule()
    d = rule.me[rule.me >= 400][0]
    base = rule.is_risk(int(d), 5.0)
    r2 = r.copy()
    r2[d + 1 :] = np.random.default_rng(9).standard_normal(len(r) - d - 1)
    other = J.JumpModelRule(r2, cash, rule.me, rule.cfg)
    assert other.is_risk(int(d), 5.0) == base
    # 창 밖 과거(window 이전)를 바꿔도 상태가 안 변한다는 보장은 없지만 창 안은 같다
    lo = d - rule.cfg.window_len + 1
    assert np.array_equal(rule.feat[lo : d + 1][-1], other.feat[lo : d + 1][-1])


class _Fake(J.JumpModelRule):
    """is_risk를 λ별 고정 규칙으로 바꿔 λ 선택만 시험한다."""

    def __init__(self, *a, plan, **k):
        super().__init__(*a, **k)
        self.plan = plan

    def is_risk(self, d, lam):
        return self.plan[float(lam)](d)


def _fake(plan, drift, lambdas):
    T = 500
    rng = np.random.default_rng(5)
    r = drift + 0.005 * rng.standard_normal(T)
    cash = np.full(T, 0.0001)
    me = month_ends(bdays(T))
    cfg = J.JMConfig(lambdas=lambdas, seeds=(1,), window_len=100, val_months=8)
    return _Fake(r, cash, me, cfg, plan=plan), me


def test_lambda_selection_picks_best_and_ties_go_to_larger():
    never, always = (lambda d: False), (lambda d: True)
    # 상승장: 항상 안전(주식 보유)이 항상 위험(현금)보다 샤프가 높다
    rule, me = _fake({10.0: always, 20.0: never, 30.0: never}, 0.002, (10.0, 20.0, 30.0))
    t = me[-1]
    lam, sh = rule.select_lambda(t)
    assert sh[20.0] == sh[30.0] > sh[10.0]
    assert lam == 30.0  # 동점 -> 더 큰 λ
    # 하락장: 항상 위험(현금, 초과수익 0)이 낫다
    rule, me = _fake({10.0: always, 20.0: never}, -0.003, (10.0, 20.0))
    assert rule.select_lambda(me[-1])[0] == 10.0
    # 격자 순서가 뒤섞여도 같다
    rule, me = _fake({10.0: never, 20.0: never, 30.0: never}, 0.002, (30.0, 10.0, 20.0))
    assert rule.select_lambda(me[-1])[0] == 30.0


def test_validation_strategy_delay_and_cost_by_hand():
    pat = {0: False, 1: True, 2: False}
    seq: list[bool] = []

    def plan(d):
        return seq[d]

    T = 500
    rng = np.random.default_rng(2)
    r = 0.001 + 0.01 * rng.standard_normal(T)
    cash = np.full(T, 0.0002)
    me = month_ends(bdays(T))
    cfg = J.JMConfig(lambdas=(1.0,), seeds=(1,), window_len=50, val_months=3)
    rule = _Fake(r, cash, me, cfg, plan={1.0: plan})
    t = me[-1]
    ve = rule.val_month_ends(t)
    risk_at = dict(zip(map(int, ve), [False, True, False], strict=True))
    seq.extend([risk_at.get(i, False) for i in range(T)])
    got = rule.validation_sharpe(t, 1.0)
    # 손계산: 신호 d -> d+1 종가 체결 -> d+2부터 새 포지션, 변화일(d+1)에 편도 비용
    pos = [0.0 if risk_at[int(d)] else 1.0 for d in ve]
    rets = []
    for i in range(int(ve[0]) + 2, t + 1):
        j = max(k for k in range(3) if ve[k] + 1 < i)
        x = pos[j] * r[i] + (1 - pos[j]) * cash[i]
        for k in (1, 2):
            if i == ve[k] + 1:
                x -= abs(pos[k] - pos[k - 1]) * 0.0005
        rets.append(x - cash[i])
    rets = np.array(rets)
    assert got == pytest.approx(rets.mean() / rets.std(ddof=1) * np.sqrt(252))
    del pat


def test_decide_maps_state_to_m():
    rule, me = _fake({5.0: lambda d: True}, 0.0, (5.0,))
    out = rule.decide(me[-1])
    assert out["is_risk"] and out["m"] == 0.5 and out["lambda"] == 5.0
    rule, me = _fake({5.0: lambda d: False}, 0.0, (5.0,))
    assert rule.decide(me[-1])["m"] == 1.0


def test_cutoff_property_jm():
    T = 520
    rule, r, cash = _rule(T=T, lambdas=(5.0, 30.0), window_len=120, val_months=5)
    ts = [int(t) for t in rule.me if t >= 400][:3]
    for t in ts:
        a = rule.decide(t)
        # 뒤를 잘라낸 자료 vs 뒤를 쓰레기로 바꾼 자료 -> t의 결정이 같다
        cut = J.JumpModelRule(r[: t + 1], cash[: t + 1], rule.me[rule.me <= t], rule.cfg)
        junk_r = r.copy()
        junk_r[t + 1 :] = 0.5 * np.random.default_rng(1).standard_normal(T - t - 1)
        junk = J.JumpModelRule(junk_r, cash, rule.me, rule.cfg)
        for other in (cut, junk):
            b = other.decide(t)
            assert (a["m"], a["lambda"], a["is_risk"]) == (b["m"], b["lambda"], b["is_risk"])
            assert a["val_sharpe"] == b["val_sharpe"]


# ------------------------------------------------------------------ F4·F6·F7
def test_selection_uses_no_data_after_previous_month_end():
    rule, r, cash = _rule(T=520, lambdas=(5.0, 30.0), window_len=120, val_months=5)
    t = int([t for t in rule.me if t >= 420][0])
    out = rule.decide(t)
    t_prev = rule.prev_month_end(t)
    assert out["lambda_selected_at"] == t_prev < t
    # t_prev 뒤(t 포함)를 쓰레기로 바꿔도 λ와 검증 샤프는 같다. 상태는 t까지 쓰므로 따로 본다.
    r2, c2 = r.copy(), cash.copy()
    r2[t_prev + 1 :] = np.random.default_rng(4).normal(0.0, 0.05, len(r) - t_prev - 1)
    c2[t_prev + 1 :] = 0.01
    other = J.JumpModelRule(r2, c2, rule.me, rule.cfg)
    assert other.select_lambda(t_prev) == rule.select_lambda(t_prev)
    assert other.decide(t)["lambda"] == out["lambda"]
    # 첫 월말에는 이전 월말이 없다
    with pytest.raises(ValueError):
        rule.prev_month_end(int(rule.me[0]))


def test_nan_cash_days_are_skipped_not_filled_in_validation():
    T = 500
    rng = np.random.default_rng(2)
    r = 0.001 + 0.01 * rng.standard_normal(T)
    cash = np.full(T, 0.0002)
    cash[0] = np.nan
    me = month_ends(bdays(T))
    cfg = J.JMConfig(lambdas=(1.0,), seeds=(1,), window_len=50, val_months=3)
    t = me[-1]
    ve = [int(d) for d in me if d <= t][-3:]
    risk = dict(zip(ve, [False, True, False], strict=True))
    holes = np.array([ve[1] + 1, ve[1] + 2, ve[1] + 3, t - 2])  # 체결일(ve[1]+1)도 구멍
    cash[holes] = np.nan
    rule = _Fake(r, cash, me, cfg, plan={1.0: lambda d: risk[int(d)]})
    got = rule.validation_sharpe(t, 1.0)
    pos = [0.0 if risk[d] else 1.0 for d in ve]
    eff = [d + 1 for d in ve]
    rets, carry = [], 0.0
    for i in range(ve[0] + 2, t + 1):
        for k in (1, 2):
            if i == eff[k]:
                carry += abs(pos[k] - pos[k - 1]) * 0.0005
        if np.isnan(cash[i]):
            continue
        j = max(k for k in range(3) if eff[k] < i)
        x = pos[j] * r[i] + (1 - pos[j]) * cash[i] - carry
        carry = 0.0
        rets.append(x - cash[i])
    rets = np.array(rets)
    assert got == pytest.approx(rets.mean() / rets.std(ddof=1) * np.sqrt(252))
    assert len(rets) == t - (ve[0] + 2) + 1 - 4  # 구멍 4개는 표본에서 빠진다


def test_windows_count_valid_days_only_and_warmup():
    T = 600
    r, _ = two_regime(T, block=100, seed=1)
    cash = np.full(T, 0.0001)
    cash[0] = np.nan
    cash[200:210] = np.nan
    cfg = J.JMConfig(lambdas=(5.0,), seeds=(1,), window_len=100, val_months=3, feature_warmup=30)
    rule = J.JumpModelRule(r, cash, month_ends(bdays(T)), cfg)
    # 0일은 현금 NaN이라 유효하지 않다: 유효한 날 수 = d (0일 제외)
    assert rule.n_valid[150] == 150 and rule.n_valid[300] == 300 - 10
    lo, hi = rule.window_bounds(300)
    assert hi - lo == 100 and hi == 290
    with pytest.raises(ValueError):
        rule.window_bounds(100)  # 창 100 + 예열 30 = 130 유효일 필요
    # 창 안에 구멍이 있어도 창은 항상 100개 유효한 날이다
    assert rule.feat.shape[0] == rule.valid.sum()
    ready = rule.first_ready_index()
    assert rule.n_valid[ready] == 130 and rule.n_valid[ready - 1] == 129
    d0 = rule.first_decision_index()
    ok = [int(d) for d in rule.me if d >= ready]
    assert d0 == ok[3]


def test_diagnostics_recorded_and_deterministic_bits():
    rule, r, cash = _rule(T=520, lambdas=(5.0, 30.0), window_len=120, val_months=5)
    t = int([t for t in rule.me if t >= 420][0])
    a = rule.decide(t)
    d = a["diag"]
    assert set(d) == {"n_hit_max", "empty_state", "label_disagree", "bear_share"}
    assert 0.0 <= d["bear_share"] <= 1.0
    # 같은 입력 -> 같은 비트 (캐시·스레드와 무관)
    again = J.JumpModelRule(r, cash, rule.me, rule.cfg).decide(t)
    assert a["val_sharpe"] == again["val_sharpe"] and a["m"] == again["m"]
    # 반복 상한 1이면 수렴 못 한 적합이 센다
    cfg1 = J.JMConfig(lambdas=(5.0,), seeds=(1, 2), window_len=120, val_months=5, max_iter=1)
    x = J.standardize(J.jm_features(r)[200:320])
    assert J.fit_jm(x, 5.0, (1, 2), 1)[3] >= 0 and cfg1.max_iter == 1
