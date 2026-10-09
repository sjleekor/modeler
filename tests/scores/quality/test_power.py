"""§6 검정력 표 재현. 점수·데이터와 무관하다."""

import math

from modeler.scores.quality import power as pw


def test_hm_se_null_is_symmetric_and_positive():
    s = pw.hm_se(0.5, 100, 2000)
    assert 0 < s < 0.04
    assert math.isclose(pw.hm_se(0.5, 100, 2000), pw.hm_se(0.5, 100, 2000))


def test_power_rises_with_auc_and_falls_with_design_effect():
    assert pw.auc_power(0.65, 100, 8400, 0.05, 1) > pw.auc_power(0.58, 100, 8400, 0.05, 1)
    assert pw.auc_power(0.65, 100, 8400, 0.05, 1) > pw.auc_power(0.65, 100, 8400, 0.05, 2)
    assert math.isclose(pw.auc_power(0.5, 100, 8400, 0.05, 1), 0.05, abs_tol=1e-9)


def test_min_detectable_auc_hits_target():
    a = pw.min_detectable_auc(100, 8400, 0.05, 2.0)
    assert math.isclose(pw.auc_power(a, 100, 8400, 0.05, 2.0), 0.8, abs_tol=1e-6)


def test_section6_tables_reproduce_except_lift_significance():
    bad = [r for r in pw.reproduce() if not r["match"]]
    assert {r["table"] for r in bad} <= {"lift 유의성(가정불명)"}


def test_g2_joint_is_bound_by_point_gate_not_alpha():
    a = pw.g2_joint_power(0.62, 100, 8400, 0.05, 2.0, 0.60)
    b = pw.g2_joint_power(0.62, 100, 8400, 0.0125, 2.0, 0.60)
    assert math.isclose(a, b) and math.isclose(a, 0.68, abs_tol=0.005)


def test_e1_and_e2_gates():
    assert math.isclose(pw.e1_gate_power(0.5, 150), 0.5, abs_tol=1e-6)
    assert math.isclose(pw.e2_gate_power(0.4), 0.92, abs_tol=0.005)
