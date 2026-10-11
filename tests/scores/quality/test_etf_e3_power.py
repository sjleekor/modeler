"""E3 검정력 모의: 구조 개수, 시나리오, 귀무 오경보, β=0 에서 규칙 1-a."""

import numpy as np

from modeler.scores.quality import etf_e3_power as pw

STRUCT = [(g // 2, 2015 + g % 10, 3 + g % 4) for g in range(120)]


def test_skeleton_counts():
    sk = pw.skeleton_from_structure(STRUCT)
    sizes = np.array([s[2] for s in STRUCT])
    assert sk.E == sizes.sum() and len(sk.pa) == (sizes * (sizes - 1) // 2).sum()
    assert sk.n_groups == 60


def test_scenario_grid_is_complete():
    assert len(pw.scenario_list()) == 4 * 3 * 2 * 3 * 2
    assert {0.0, 0.3} == set(pw.GRID["operator_share"]) and {0.0, 0.5, 1.0} == set(pw.GRID["beta"])


def test_null_false_alarm_low_and_cap_when_beta_zero():
    sk = pw.skeleton_from_structure(STRUCT)
    null = dict(persist_share=0.0, noise_bp=10.0, ter_sd_bp=10.0, beta=1.0, operator_share=0.0)
    b0 = dict(null, beta=0.0)
    rows = pw.run_power(sk, [null, b0], n_rep=60, workers=1)
    assert rows[0]["p_G2_raw"] <= 0.15  # 귀무에서 G2 통과는 드물다
    # β = 0 이면 G = gross + F/100, H = gross + A/100 → 보수가 점수와 결과에 함께 들어가 C 가 올라간다
    assert rows[1]["mean_C"] > rows[0]["mean_C"] + 0.05
    assert rows[1]["p_cap_1a"] > 0.9  # β̂ ≈ 0 → 구간 상한 < 0.7
    assert rows[1]["p_G2_after_1a"] <= rows[1]["p_G2_raw"]


def test_power_increases_with_persistence():
    sk = pw.skeleton_from_structure(STRUCT)
    lo = dict(persist_share=0.0, noise_bp=10.0, ter_sd_bp=10.0, beta=1.0, operator_share=0.0)
    hi = dict(lo, persist_share=0.3)
    rows = pw.run_power(sk, [lo, hi], n_rep=40, workers=1)
    assert (
        rows[1]["mean_C"] > rows[0]["mean_C"] + 0.05 and rows[1]["p_G2_raw"] > rows[0]["p_G2_raw"]
    )
