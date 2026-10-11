"""E3 통계량: 무게(하한 무게 포함)·C·순열 p·부트스트랩·게이트·등급(규칙 1-a·지연 상한)·중단 제안·E3a."""

import numpy as np
import pytest

from modeler.scores.quality import etf_e3_power as pw
from modeler.scores.quality import etf_e3_stat as st


def sd_from(sizes, G, H, years=None, group=None, valid=None, thr=st.SCORE_THR):
    """G·H: 그룹-연도별 ETF 수준 벡터(쌍마다 같은 값 — H[a,b] = H_a)."""
    n_gy = len(sizes)
    years = years or [2015] * n_gy
    group = group or list(range(n_gy))
    sdm = [np.subtract.outer(np.asarray(g), np.asarray(g)) for g in G]
    out = [np.repeat(np.asarray(h)[:, None], len(h), axis=1) for h in H]
    pv = valid or [np.ones((n, n), bool) for n in sizes]
    return st.make_stat_data(
        sizes, years, group, [str(i) for i in range(max(group) + 1)], sdm, out, pv, score_thr=thr
    )


def test_weights_and_half_pair_downweight():
    # gy0: n=3, 전체 쌍 3, 조건 3(≥ 절반) → 무게 3. 모두 맞음(c=1)
    # gy1: n=4, 전체 쌍 6, G 차 ≥ 5bp 인 쌍이 2(< 절반)  → 무게 4 × (2/6), 쌍 둘 다 틀림(c=0)
    G0, H0 = [0.0, 0.1, 0.2], [0.0, 0.1, 0.2]
    G1 = [0.0, 0.1, 0.1001, 0.1002]  # 0↔나머지 3쌍 + 1,2,3 사이는 0.01~0.02 bp 차(문턱 미만)
    G1 = [0.0, 0.0, 0.0, 0.1]  # 조건 쌍: (0,3)(1,3)(2,3) = 3쌍... 아래에서 2쌍으로 만든다
    G1 = [0.0, 0.0001, 0.0002, 0.1]  # (0,3)(1,3)(2,3) 3쌍 조건(0~2 사이는 5bp 미만)
    sizes = [3, 4]
    H1 = [0.5, 0.5, 0.5, 0.0]  # 3 이 G 는 크지만 H 는 작다 → 3쌍 모두 틀림
    sd = sd_from(sizes, [G0, G1], [H0, H1])
    rs = st.real_stat(sd)
    # gy1: k=3, T=6 → k/T = 0.5 (절반 이상) → 무게 4
    assert rs["w"].tolist() == [3.0, 4.0] and rs["C"] == pytest.approx(3 / 7)
    # 조건 쌍을 2개로 줄이면(3 번 ETF 의 G 를 낮춰 0↔3 만 조건) 절반 미만 → 4 × (2/6)
    G1b = [0.0, 0.0001, 0.1, 0.1002]
    # (0,2)(1,2)(0,3)(1,3) 4쌍. 더 줄이기 위해 pair_valid 로 두 쌍을 막는다
    pv = [np.ones((3, 3), bool), np.ones((4, 4), bool)]
    pv[1][0, 3] = pv[1][3, 0] = pv[1][1, 3] = pv[1][3, 1] = False
    sd2 = sd_from(sizes, [G0, G1b], [H0, [0.5, 0.5, 0.0, 0.0]], valid=pv)
    rs2 = st.real_stat(sd2)
    assert rs2["k"].tolist() == [3.0, 2.0]
    assert rs2["w"][1] == pytest.approx(4 * 2 / 6)
    # 그 두 쌍은 (0,2)(1,2): G 가 2 쪽이 큰데 H 도 0 → 틀림
    assert rs2["C"] == pytest.approx(3 / (3 + 4 * 2 / 6))


def test_tie_and_condition_thresholds():
    G = [[0.0, 0.0004, 0.0006]]  # (0,1) 4bp < 5bp 제외, (0,2) 6bp 통과, (1,2) 2bp 제외
    H = [[0.0, 0.00004, 0.01]]  # (0,2) 결과차 1%, 나머지 쌍은 이미 제외
    rs = st.real_stat(sd_from([3], G, H))
    assert rs["k"].tolist() == [1.0]
    H2 = [[0.0, 0.0, 0.00003]]  # (0,2) 결과차 0.3bp < 0.5bp 동률 → 제외
    rs2 = st.real_stat(sd_from([3], G, H2))
    assert rs2["k"].tolist() == [0.0] and np.isnan(rs2["C"])


def test_c_all_correct_and_bootstrap_bounds():
    rng = np.random.default_rng(0)
    G, H, yrs, grp = [], [], [], []
    for g in range(40):
        a = rng.normal(size=4) * 0.05
        G.append(a)
        H.append(a * 2)  # 순서 보존
        yrs.append(2015 + g % 10)
        grp.append(g // 2)
    sd = sd_from([4] * 40, G, H, yrs, grp)
    r = st.judge_c(sd, n_perm=199, n_boot=300)
    assert r["C"] == pytest.approx(1.0)
    assert r["gates"]["boot_lower_95"] == pytest.approx(1.0) and r["p"] == pytest.approx(1 / 200)
    assert r["gates"]["G2"] and r["gates"]["G3"]


def test_cluster_boot_matches_resampling_formula():
    num = np.array([1.0, 0.0, 2.0])
    w = np.array([2.0, 2.0, 2.0])
    b = st.cluster_boot(num, w, n_boot=500, seed=1)
    assert np.nanmin(b) >= 0.0 and np.nanmax(b) <= 1.0
    assert abs(np.nanmean(b) - num.sum() / w.sum()) < 0.03
    loo = st.leave_one_group_out(num, w)
    assert loo.tolist() == pytest.approx([2 / 4, 3 / 4, 1 / 4])


def test_perm_null_is_roughly_uniform():
    sk = pw.skeleton_from_structure([(g // 2, 2015 + g % 10, 3 + g % 3) for g in range(30)])
    sc = dict(persist_share=0.0, noise_bp=10.0, ter_sd_bp=10.0, beta=1.0, operator_share=0.0)
    ps = []
    for i in range(150):
        rng = np.random.default_rng(1000 + i)
        sd, _ = pw.simulate(sk, sc, rng)
        r = st.real_stat(sd)
        p, _ = st.perm_pvalue(sd, r["C"], n_perm=99, seed=i)
        ps.append(p)
    ps = np.array(ps)
    assert 0.40 < ps.mean() < 0.60
    assert (ps < 0.10).mean() < 0.20 and (ps > 0.9).mean() > 0.03


def test_perm_p_small_under_strong_effect():
    sk = pw.skeleton_from_structure([(g // 2, 2015 + g % 10, 4) for g in range(150)])
    sc = dict(persist_share=0.4, noise_bp=10.0, ter_sd_bp=10.0, beta=1.0, operator_share=0.0)
    sd, _ = pw.simulate(sk, sc, np.random.default_rng(5))
    r = st.real_stat(sd)
    p, cs = st.perm_pvalue(sd, r["C"], n_perm=199, seed=3)
    assert r["C"] > 0.55 and p < 0.05
    assert abs(np.nanmean(cs) - 0.5) < 0.03  # 순열 분포의 중심은 0.5


def g(**kw):
    base = {"G1": True, "G2": True, "G3": True, "boot_upper_95": 0.8, "boot_lower_95": 0.6}
    base.update(kw)
    return base


MEASURED = {"cap": False, "status": "실측"}


def test_grade_logic_all_branches():
    assert st.grade(g(), False, MEASURED)["grade"] == "A"
    assert st.grade(g(G3=False), False, MEASURED)["grade"] == "C"
    assert st.grade(g(), True, MEASURED)["grade"] == "C"  # 규칙 1-a 상한
    assert st.grade(g(), False, {"cap": True, "status": "실측"})["grade"] == "C"  # 지연 > 92일
    assert st.grade(g(G2=False), True, MEASURED)["grade"] == "D"  # 1-a 가 D 를 올리지는 않는다
    assert st.grade(g(G1=False), False, MEASURED)["grade"] == "D"


def test_grade_labels_and_provisional():
    a = st.grade(g(), False, MEASURED)
    assert a["labels"] == ["(전진 확인 대기)"]
    a2 = st.grade(g(), False, None)  # 지연 실측 6개 미만
    assert a2["labels"] == ["(전진 확인 대기)", "잠정(지연 실측 전)"]
    d = st.grade(g(G2=False), False, None)
    assert d["labels"] == []


def test_stop_proposal_rule():
    low = st.grade(g(G2=False, boot_upper_95=0.54), False, MEASURED)
    assert low["grade"] == "D" and low["stop_proposal"] == "접기 제안"
    hi = st.grade(g(G2=False, boot_upper_95=0.55), False, MEASURED)
    assert hi["stop_proposal"] == "보류(감쇠 폭 추정)"
    assert st.grade(g(), False, MEASURED)["stop_proposal"] is None


def test_rule_1a_thresholds():
    assert st.rule_1a(0.2, 0.69) and st.rule_1a(1.31, 1.9)
    assert not st.rule_1a(0.2, 0.7) and not st.rule_1a(0.9, 1.1) and not st.rule_1a(1.3, 1.9)


def test_g1_and_g3_gates():
    G, H = [[0.0, 0.1, 0.2, 0.3]] * 4, [[0.0, 0.1, 0.2, 0.3]] * 4
    sd = sd_from([4] * 4, G, H, [2015, 2016, 2020, 2021], [0, 1, 2, 3])
    r = st.judge_c(sd, n_perm=99, n_boot=100)
    assert r["gates"]["G1"] is False  # 쌍 24 < 300
    # 후반만 틀리게 만든다: G3 탈락
    H2 = [[0.0, 0.1, 0.2, 0.3]] * 2 + [[0.3, 0.2, 0.1, 0.0]] * 2
    sd2 = sd_from([4] * 4, G, H2, [2015, 2016, 2020, 2021], [0, 1, 2, 3])
    gt = st.gates(sd2, st.real_stat(sd2), 0.01, np.array([0.9] * 10))
    assert gt["C_early"] == 1.0 and gt["C_late"] == 0.0 and gt["G3"] is False


def test_beta_hat_group_year_fixed_effect_and_ci():
    rng = np.random.default_rng(2)
    xs, ys, grp = [], [], []
    for k in range(60):
        x = rng.normal(size=5) * 0.003
        y = (
            0.5 * x + rng.normal(size=5) * 0.0005 + rng.normal() * 0.1
        )  # 그룹-연도 수준 차이는 고정효과가 지운다
        xs.append(x)
        ys.append(y)
        grp.append(k // 3)
    b = st.beta_hat(xs, ys, np.array(grp), n_boot=300)
    assert b["beta"] == pytest.approx(0.5, abs=0.1) and b["ci_lo"] < 0.5 < b["ci_hi"] + 0.1
    assert b["rule_1a"]  # 참값 0.5 → 구간 상한 < 0.7
    b1 = st.beta_hat(
        [2.0 * x for x in xs], ys, np.array(grp), n_boot=300
    )  # 참값 0.25 가 아니라 비율 확인용
    assert b1["beta"] == pytest.approx(0.25, abs=0.06)
    b_ok = st.beta_hat(xs, [2.0 * y for y in ys], np.array(grp), n_boot=300)  # 참값 1.0
    assert b_ok["beta"] == pytest.approx(1.0, abs=0.2) and not b_ok["rule_1a"]
    # β = 0 이면 구간 상한 < 0.7 → 규칙 1-a
    ys0 = [rng.normal(size=5) * 0.0005 for _ in xs]
    b0 = st.beta_hat(xs, ys0, np.array(grp), n_boot=300)
    assert b0["rule_1a"] and abs(b0["beta"]) < 0.3
    fast = st.beta_hat_fast(
        np.concatenate(xs), np.concatenate(ys), np.arange(0, 305, 5), np.array(grp), n_boot=300
    )
    assert fast["beta"] == pytest.approx(b["beta"])


def test_lowest_fee_gain_and_tie_drop():
    F = [np.array([0.1, 0.3, 0.5]), np.array([0.2, 0.2, 0.4]), np.array([0.1, 0.5])]
    R = [np.array([0.05, 0.04, 0.03]), np.array([0.1, 0.0, 0.0]), np.array([0.02, 0.0])]
    out = st.lowest_fee_gain(F, R, np.array([0, 0, 1]), n_boot=100)
    # gy0: 0.05 − median(0.04) = +100bp, gy1 동률 → 뺌, gy2: 0.02 − median(0.01) = +100bp
    assert out["n_group_years"] == 2 and out["n_dropped_tie"] == 1
    assert out["gain_bp_weighted_mean"] == pytest.approx(100.0)


def test_c_a_uses_lower_fee_as_higher_score():
    # F 가 낮은 ETF 의 결과가 더 좋으면(R 큼) C_a = 1
    n = 3
    F = np.array([0.1, 0.3, 0.5])
    R = -F / 100.0  # 보수가 수익에서 그대로 빠짐
    sd = st.make_stat_data(
        [n],
        [2016],
        [0],
        ["g"],
        [F[None, :] - F[:, None]],
        [np.repeat(R[:, None], n, 1)],
        [np.ones((n, n), bool)],
        score_thr=st.F_THR_PCT,
    )
    assert st.real_stat(sd)["C"] == pytest.approx(1.0)
