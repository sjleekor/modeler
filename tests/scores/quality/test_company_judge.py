"""company_judge 시험 — 합성 자료만 쓴다(사전등록 §5.4·§5.5)."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import roc_auc_score

from modeler.scores.quality import company_judge as cj


def _frame(scores, events, fys=None, corps=None) -> pl.DataFrame:
    n = len(scores)
    return pl.DataFrame(
        {
            "corp_code": corps if corps is not None else [f"c{i}" for i in range(n)],
            "fy": fys if fys is not None else [2020] * n,
            "score": [None if s is None else float(s) for s in scores],
            "event": events,
        },
        schema_overrides={"score": pl.Float64},
    )


def _synth(n_corp=300, years=(2019, 2020, 2021, 2022), signal=1.0, seed=0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_corp):
        base = rng.normal()
        for y in years:
            s = 50 + 15 * (base + rng.normal(scale=0.5))
            p = 1 / (1 + np.exp(-(-2.2 - signal * (s - 50) / 15)))
            rows.append((f"c{c}", y, s, int(rng.random() < p)))
    return pl.DataFrame(rows, schema=["corp_code", "fy", "score", "event"], orient="row")


# ---------------------------------------------------------------- AUC
def test_auc_hand_with_ties_matches_sklearn():
    risk = [1, 2, 2, 3, 3, 3, 5]
    ev = [0, 0, 1, 0, 1, 1, 1]
    # 손계산: 사건 위험 {2,3,3,5}, 비사건 {1,2,3} -> (2:1.5)+(3:2.5)*2+(5:3) = 9.5 / 12
    assert cj.auc(risk, ev) == pytest.approx(9.5 / 12)
    assert cj.auc(risk, ev) == pytest.approx(roc_auc_score(ev, risk))


def test_auc_random_matches_sklearn_and_degenerate_nan():
    rng = np.random.default_rng(1)
    r = rng.integers(0, 20, 200).astype(float)
    e = (rng.random(200) < 0.2).astype(int)
    assert cj.auc(r, e) == pytest.approx(roc_auc_score(e, r))
    assert np.isnan(cj.auc([1, 2, 3], [0, 0, 0]))
    assert np.isnan(cj.auc([1, 2, 3], [1, 1, 1]))


def test_risk_is_100_minus_score_and_nulls_dropped():
    df = _frame([10, 20, None, 80, 90], [1, 1, 1, 0, 0])
    st = cj.outcome_stats(df, n_boot=10)
    assert st["n_dropped"] == 1 and st["n_rows"] == 4
    assert st["pooled_auc"] == 1.0  # 점수 낮은 쪽이 사건


def test_yearly_auc_counts():
    df = _frame([1, 2, 3, 4, 1, 2], [1, 0, 0, 0, 1, 0], fys=[2020] * 4 + [2021] * 2)
    d, _ = cj._clean(df)
    y = {r["fy"]: r for r in cj.yearly_auc(d)}
    assert y[2020]["n"] == 4 and y[2020]["n_events"] == 1
    assert y[2021]["n"] == 2 and y[2021]["auc"] == 1.0


# ---------------------------------------------------------------- 부트스트랩
def test_weighted_auc_equals_replicated_rows():
    d, _ = cj._clean(_synth(n_corp=30, years=(2019, 2020)))
    wa = cj._WeightedAuc(d)
    rng = np.random.default_rng(5)
    counts = np.bincount(rng.integers(0, wa.n_corps, wa.n_corps), minlength=wa.n_corps)
    corp_counts = dict(zip(wa.corp_uniq.tolist(), counts.tolist()))
    rep = pl.concat(
        [d.filter(pl.col("corp_code") == c) for c, k in corp_counts.items() for _ in range(k)]
    )
    assert wa.auc(counts) == pytest.approx(cj.auc(rep["risk"].to_numpy(), rep["event"].to_numpy()))


def test_bootstrap_seed_reproducible():
    d, _ = cj._clean(_synth(n_corp=60))
    a = cj.cluster_bootstrap_auc(d, 50, seed=1)
    b = cj.cluster_bootstrap_auc(d, 50, seed=1)
    c = cj.cluster_bootstrap_auc(d, 50, seed=2)
    assert np.array_equal(a, b, equal_nan=True)
    assert not np.array_equal(a, c, equal_nan=True)


def test_p_value_formula_and_nan_excluded():
    boot = np.array([0.4, 0.5, 0.6, 0.7, np.nan, 0.55, np.nan, 0.45])
    s = cj.boot_summary(boot)
    assert s["n_nan"] == 2 and s["n_valid"] == 6
    assert s["n_le_half"] == 3  # 0.4, 0.5, 0.45
    assert s["p_value"] == pytest.approx((3 + 1) / (6 + 1))
    assert cj.boot_summary(np.array([np.nan, np.nan]))["n_nan"] == 2


def test_bootstrap_nan_when_no_events_drawn():
    # 사건이 한 회사에만 있고 회사 2개 -> 그 회사를 못 뽑는 회차는 nan
    df = _frame([10, 90, 50, 60], [1, 0, 0, 0], corps=["a", "a", "b", "b"])
    d, _ = cj._clean(df)
    boot = cj.cluster_bootstrap_auc(d, 200, seed=3)
    assert np.isnan(boot).any() and np.isfinite(boot).any()
    s = cj.boot_summary(boot)
    assert s["n_nan"] == int(np.isnan(boot).sum())
    assert s["p_value"] == pytest.approx((s["n_le_half"] + 1) / (s["n_valid"] + 1))


def test_bootstrap_p_small_with_signal_large_without():
    strong = cj.outcome_stats(_synth(signal=1.5, seed=1), n_boot=200)
    assert strong["bootstrap"]["p_value"] < 0.05
    rng = np.random.default_rng(9)
    noise = _frame(rng.normal(50, 10, 400), (rng.random(400) < 0.1).astype(int))
    assert cj.outcome_stats(noise, n_boot=200)["bootstrap"]["p_value"] > 0.05


def test_leave_one_year_out_range():
    d, _ = cj._clean(_synth(n_corp=120, signal=1.2))
    r = cj.leave_one_year_out(d)
    assert set(r["by_year"]) == {"2019", "2020", "2021", "2022"}
    assert r["min"] <= r["max"] and r["min"] == min(r["by_year"].values())


# ---------------------------------------------------------------- 분위·포착률
def test_capture20_lift_relation_no_ties():
    n = 100
    scores = np.arange(n, dtype=float)
    ev = np.zeros(n, dtype=int)
    ev[[0, 1, 2, 3, 4, 5, 50, 60, 70, 90]] = 1  # 하위 20%(점수 0..19)에 6건
    d, _ = cj._clean(_frame(scores, ev))
    c = cj.capture(d, 0.20)
    assert c["n_selected"] == 20 and c["n_events_selected"] == 6
    assert c["capture"] == pytest.approx(0.6)
    assert c["lift"] == pytest.approx(c["capture"] / 0.20)
    assert cj.capture(d, 0.10)["n_selected"] == 10


def test_capture_ties_use_average_rank():
    # 점수 동률이 경계를 걸치면 묶음 전체가 평균 순위로 들어가거나 빠진다
    scores = [0] * 4 + list(range(1, 7))  # n=10, 20% = 2행. 동률 4개의 평균 순위 2.5 > 2 -> 빠짐
    d, _ = cj._clean(_frame(scores, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
    assert cj.capture(d, 0.20)["n_selected"] == 0
    d2, _ = cj._clean(_frame([0, 0, 1, 2, 3, 4, 5, 6, 7, 8], [1] + [0] * 9))
    assert cj.capture(d2, 0.20)["n_selected"] == 2  # 평균 순위 1.5 ≤ 2


def test_quintiles_within_year_vs_pooled():
    # 2020은 점수 수준이 낮고 2021은 높다 -> 풀링 컷이면 분위가 연도로 갈린다
    s = list(range(0, 10)) + list(range(100, 110))
    ev = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    d, _ = cj._clean(_frame(s, ev, fys=[2020] * 10 + [2021] * 10))
    w = cj.quintile_rates(d, cut="within_year")
    p = cj.quintile_rates(d, cut="pooled")
    assert w["quintiles"][0]["n"] == 4 and w["quintiles"][0]["n_events"] == 2
    assert w["quintiles"][0]["lift"] == pytest.approx(0.5 / 0.1)
    assert p["quintiles"][0]["n_events"] == 1  # 풀링에서는 2020 최저점만
    assert sum(q["n"] for q in w["quintiles"]) == 20
    with pytest.raises(ValueError):
        cj.quintile_rates(d, cut="x")


# ---------------------------------------------------------------- 고정 순서
def test_fixed_sequence_stops_after_failure():
    r = cj.fixed_sequence({"O1": 0.20, "O2": 0.001, "O3": 0.001, "O4": 0.001})
    assert r["O1"]["status"] == "기각 못 함"
    for o in ("O2", "O3", "O4"):
        assert r[o]["tested"] is False and r[o]["status"] == "검정 안 함(앞 단계 탈락)"
    r2 = cj.fixed_sequence({"O1": 0.01, "O2": 0.04, "O3": 0.05, "O4": 0.001})
    assert [r2[o]["status"] for o in cj.OUTCOME_ORDER] == [
        "기각",
        "기각",
        "기각 못 함",
        "검정 안 함(앞 단계 탈락)",
    ]


def test_judge_outcomes_o1_fail_skips_rest():
    rng = np.random.default_rng(4)

    def noise():
        return _frame(rng.normal(50, 10, 300), (rng.random(300) < 0.15).astype(int))

    res = cj.judge_outcomes(
        {"O1": noise(), "O2": _synth(signal=2.0), "O3": _synth(signal=2.0)}, n_boot=100
    )
    assert res["order"] == ["O1", "O2", "O3"]
    assert res["outcomes"]["O1"]["test"]["status"] == "기각 못 함"
    assert res["outcomes"]["O2"]["gates"]["g2"]["status"] == "검정 안 함"
    assert res["outcomes"]["O2"]["grade"] == "검정 안 함(앞 단계 탈락)"
    assert res["summary"]["verdict"] == cj.FOLD_TEXT


# ---------------------------------------------------------------- G4
def _years(spec):  # [(auc, n_events)]
    return [{"fy": 2019 + i, "n": 100, "n_events": e, "auc": a} for i, (a, e) in enumerate(spec)]


def test_g4_examples_from_prereg():
    # 6개 -> 4
    assert cj.g4_gate(_years([(0.6, 20)] * 4 + [(0.4, 20)] * 2))["pass"]
    assert not cj.g4_gate(_years([(0.6, 20)] * 3 + [(0.4, 20)] * 3))["pass"]
    # 4개 -> 3
    assert cj.g4_gate(_years([(0.6, 20)] * 3 + [(0.4, 20)]))["pass"]
    assert not cj.g4_gate(_years([(0.6, 20)] * 2 + [(0.4, 20)] * 2))["pass"]
    # 3개 -> 2
    assert cj.g4_gate(_years([(0.6, 20)] * 2 + [(0.4, 20)]))["pass"]
    assert not cj.g4_gate(_years([(0.6, 20)] + [(0.4, 20)] * 2))["pass"]


def test_g4_skips_years_below_10_events():
    # 사건 9건인 해는 AUC가 0.9여도 세지 않는다 -> 센 해 3개 중 양성 1개 -> 탈락
    g = cj.g4_gate(_years([(0.9, 9), (0.6, 20), (0.4, 20), (0.4, 20)]))
    assert g["n_counted"] == 3 and g["n_positive"] == 1 and not g["pass"]
    assert g["skipped_years"] == [2019]
    # 분모를 전체로 하면 같은 입력에서 양성 1/4로 역시 탈락, 센 해 2/3 구성에서는 갈린다
    y = _years([(0.4, 9), (0.6, 20), (0.4, 20), (0.6, 20)])  # 센 해 3개 중 2개 양성
    assert cj.g4_gate(y, denom="counted")["pass"]
    assert not cj.g4_gate(y, denom="all")["pass"]  # 분모 4 -> 3개 필요
    assert cj.g4_gate(_years([(0.6, 5)] * 3))["pass"] is False  # 센 해 0 -> 탈락


# ---------------------------------------------------------------- 게이트·등급
def test_g1_bands():
    assert cj.g1_gate(29)["status"] == "표본 부족" and not cj.g1_gate(29)["pass"]
    assert cj.g1_gate(30)["status"] == "탐색 판정" and cj.g1_gate(30)["pass"]
    assert cj.g1_gate(49)["exploratory"] and not cj.g1_gate(50)["exploratory"]


@pytest.mark.parametrize(
    "g1,g2,g3,g4,exp",
    [
        (True, False, True, True, "D"),
        (False, True, True, True, "D"),
        (True, True, False, True, "C"),
        (True, True, True, False, "B"),
        (True, True, True, True, "A"),
    ],
)
def test_assign_grade_four_paths(g1, g2, g3, g4, exp):
    assert cj.assign_grade(g1, g2, g3, g4) == exp


def _gates(g1=True, g2=True, g3=True, g4=True):
    return {"g1": {"pass": g1}, "g2": {"pass": g2}, "g3": {"pass": g3}, "g4": {"pass": g4}}


def test_grade_of_g1_fail_text_and_stop():
    g = cj.grade_of(_gates(g1=False))
    assert g["grade"] == "등급 없음(표본 부족)" and g["grade_rule"] == "D" and g["stop_proposal"]
    assert cj.grade_of(_gates())["grade"] == "A" and not cj.grade_of(_gates())["stop_proposal"]
    assert cj.grade_of(_gates(g4=False))["grade"] == "B"
    assert cj.grade_of(_gates(g3=False))["stop_proposal"]


def test_g2_requires_auc_and_p():
    ok = {"tested": True, "rejected": True, "p": 0.01, "status": "기각"}
    assert cj.g2_gate(0.62, ok)["pass"] is True
    assert cj.g2_gate(0.58, ok)["pass"] is False  # p는 통과해도 AUC < 0.60
    no = {"tested": True, "rejected": False, "p": 0.2, "status": "기각 못 함"}
    assert cj.g2_gate(0.7, no)["pass"] is False
    skip = {"tested": False, "rejected": None, "p": 0.0, "status": cj.TEST_SKIPPED}
    assert cj.g2_gate(0.9, skip)["pass"] is None


def test_summary_verdict_texts():
    mk = lambda g: {"grade": g}  # noqa: E731
    a = cj.summary_verdict({"O1": mk("B"), "O2": mk("D")})
    assert (
        a["verdict"] == "재무 건전성 점수로 채택 후보"
        and not a["stop_proposal"]
        and not a["aux_has_ab"]
    )
    b = cj.summary_verdict({"O1": mk("C"), "O2": mk("A"), "O3": mk("B")})
    assert b["verdict"] == "접기 제안 — 중단 조건 충족, 사용자 확인 대기"
    assert (
        b["aux_ab_outcomes"] == ["O2", "O3"]
        and b["stop_text"] == "중단 조건 충족 — 사용자 확인 대기"
    )
    assert cj.summary_verdict({"O1": mk("등급 없음(표본 부족)")})["stop_proposal"]


# ---------------------------------------------------------------- 묶음·기록·manifest
def test_judge_outcomes_end_to_end_adopt(tmp_path):
    frames = {o: _synth(n_corp=400, signal=1.6, seed=i) for i, o in enumerate(cj.OUTCOME_ORDER)}
    res = cj.judge_outcomes(frames, seed=7, n_boot=100)
    o1 = res["outcomes"]["O1"]
    assert o1["gates"]["g1"]["pass"] and o1["gates"]["g2"]["pass"]
    assert o1["grade"] in ("A", "B", "C")
    assert set(o1["stats"]) >= {
        "pooled_auc",
        "by_year",
        "quintiles",
        "capture",
        "bootstrap",
        "leave_one_year_out",
    }
    again = cj.judge_outcomes(frames, seed=7, n_boot=100)
    assert again["outcomes"]["O1"]["stats"]["bootstrap"] == o1["stats"]["bootstrap"]
    with pytest.raises(ValueError):
        cj.judge_outcomes({"O9": frames["O1"]}, n_boot=5)

    man = cj.write_manifest(tmp_path / "sub" / "m.json", res, extra={"snapshot": "x"})
    loaded = json.loads((tmp_path / "sub" / "m.json").read_text(encoding="utf-8"))
    assert loaded["seed"] == 7 and loaded["n_boot"] == 100
    assert loaded["code_sha256"] == cj.code_sha256() == man["code_sha256"]
    assert loaded["extra"]["snapshot"] == "x"


def test_record_cells_sixteen():
    f = _synth(n_corp=80, years=(2019, 2020), signal=1.0)
    cells = {d: {o: f for o in cj.OUTCOME_ORDER} for d in ("C1", "C2", "C3", "F")}
    r = cj.record_cells(cells, n_boot=30)
    assert sum(len(v) for v in r.values()) == 16
    c = r["C1"]["O1"]
    assert c["ci95"][0] <= c["ci95"][1] and 0.5 < c["auc"] <= 1
