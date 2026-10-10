"""부분 E 판정 결합(etf_judge)과 E1 수정 셋의 합성 예시 시험."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from modeler.scores.quality import etf_e1 as e1
from modeler.scores.quality import etf_judge as j
from tests.scores.quality import test_etf_e1 as t1


# ---------------------------------------------------------------- Holm(I18)
def _item(name, p, bounds, ok):
    """bounds: alpha -> 하한 값(dict)."""
    return {"name": name, "p": p, "bound_fn": lambda a, b=bounds: b[a], "bound_ok": ok}


def test_holm_order_and_stage_alpha():
    ok = lambda x: x >= 0.30
    r = j.holm_combine([
        _item("A", 0.04, {0.025: 0.40, 0.05: 0.35}, ok),
        _item("B", 0.01, {0.025: 0.33, 0.05: 0.36}, ok),
    ])
    assert r["B"]["stage"] == 1 and r["B"]["stage_alpha"] == 0.025 and r["B"]["holm_lower_bound"] == 0.33
    assert r["A"]["stage"] == 2 and r["A"]["stage_alpha"] == 0.05 and r["A"]["holm_lower_bound"] == 0.35
    assert r["B"]["rejected"] and r["A"]["rejected"] and r["A"]["reached"]


def test_holm_first_stage_fail_means_both_fail():
    ok = lambda x: x >= 0.30
    r = j.holm_combine([
        _item("A", 0.30, {0.025: 0.10, 0.05: 0.50}, ok),  # 두 번째 단계에선 문턱을 넘지만 닿지 못한다
        _item("B", 0.02, {0.025: 0.20, 0.05: 0.90}, ok),  # 첫 단계 α 0.025 하한이 문턱 미만
    ])
    assert not r["B"]["rejected"] and r["B"]["status"] == "단계 1 기각 못 함"
    a = r["A"]
    assert not a["rejected"] and not a["reached"]
    assert a["status"] == "Holm 앞 단계 탈락"
    assert a["bound_alpha"] == 0.025 and a["holm_lower_bound"] == 0.10  # 하한은 α 0.025 분위수로 적는다


def test_holm_tie_keeps_list_order_and_nan_bound_fails():
    ok = lambda x: x > 0
    r = j.holm_combine([
        _item("X", 0.1, {0.025: float("nan"), 0.05: 1.0}, ok),
        _item("Y", 0.1, {0.025: 1.0, 0.05: 1.0}, ok),
    ])
    assert r["X"]["stage"] == 1 and not r["X"]["rejected"]
    assert not r["Y"]["rejected"] and r["Y"]["status"] == "Holm 앞 단계 탈락"


def test_holm_p_rule_is_reference_only():
    ok = lambda x: x >= 0.30
    r = j.holm_combine([_item("A", 0.01, {0.025: 0.2, 0.05: 0.2}, ok), _item("B", 0.5, {0.025: 0.9, 0.05: 0.9}, ok)])
    assert r["A"]["p_rule_reject"] is True and r["A"]["rejected"] is False


# ---------------------------------------------------------------- 등급 규칙
@pytest.mark.parametrize(
    "g1,g2,g3,cap,want",
    [
        (False, True, True, False, "D"),
        (True, False, True, False, "D"),
        (False, False, False, False, "D"),
        (True, True, False, False, "C"),
        (True, True, True, False, "A"),
        (True, True, True, True, "C"),  # 점수 없는 사건 > 30% → A를 C로
        (True, False, True, True, "D"),
    ],
)
def test_assign_grade(g1, g2, g3, cap, want):
    assert j.assign_grade(g1, g2, g3, cap) == want


def _e1_stats(n_scored=60, hit_rate=0.6, unscored=0.1, h1=0.4, h2=0.4):
    return {
        "n_scored": n_scored, "hit_rate": hit_rate, "unscored_share": unscored,
        "halves": {"first": {"hit_rate": h1}, "second": {"hit_rate": h2}},
    }


def _holm(rej, lb=0.35):
    return {"rejected": rej, "holm_lower_bound": lb, "status": "x"}


def test_e1_gates_grades():
    assert j.e1_gates(_e1_stats(), _holm(True))["grade"] == "A"
    assert j.e1_gates(_e1_stats(h2=0.29), _holm(True))["grade"] == "C"  # G3 탈락
    assert j.e1_gates(_e1_stats(h2=None), _holm(True))["grade"] == "C"  # 반쪽 비어 있으면 탈락
    assert j.e1_gates(_e1_stats(hit_rate=0.49), _holm(True))["grade"] == "D"  # 점추정 < 0.50
    assert j.e1_gates(_e1_stats(), _holm(False))["grade"] == "D"  # Holm 기각 못 함
    g = j.e1_gates(_e1_stats(hit_rate=0.50), _holm(True))
    assert g["g2"]["pass"] and g["grade"] == "A"  # 0.50 경계는 통과


def test_e1_unscored_cap_blocks_a():
    g = j.e1_gates(_e1_stats(unscored=0.31), _holm(True))
    assert g["unscored_cap_exceeded"] and g["grade"] == "C"
    g = j.e1_gates(_e1_stats(unscored=0.30), _holm(True))  # 30%는 초과가 아니다
    assert not g["unscored_cap_exceeded"] and g["grade"] == "A"
    g = j.e1_gates(_e1_stats(unscored=0.5, hit_rate=0.1), _holm(False))
    assert g["grade"] == "D"  # D는 올라가지 않는다


@pytest.mark.parametrize(
    "n,status,passed",
    [(29, "표본 부족", False), (30, "탐색 판정", True), (49, "탐색 판정", True), (50, "판정", True)],
)
def test_e1_g1_boundaries(n, status, passed):
    g = j.e1_gates(_e1_stats(n_scored=n), _holm(True))
    assert g["g1"]["status"] == status and g["g1"]["pass"] is passed


def test_e1_sample_shortage_has_no_grade_and_lowers_purpose():
    g = j.e1_gates(_e1_stats(n_scored=29), _holm(True))
    assert g["grade"] is None and g["purpose"] == "감쇠 폭 추정" and g["grade_rule"] == "D"
    assert j.e1_gates(_e1_stats(n_scored=30), _holm(True))["purpose"] == "판정"


def _e2_stats(pooled=200, rho=0.4, n_years=6, n_pos=6):
    return {"pooled_etf_years": pooled, "weighted_rho": rho, "n_years": n_years, "n_pos_years": n_pos}


def test_e2_gates_grades_and_g1_boundary():
    assert j.e2_gates(_e2_stats(), _holm(True, 0.1))["grade"] == "A"
    assert j.e2_gates(_e2_stats(rho=0.29), _holm(True, 0.1))["grade"] == "D"
    assert j.e2_gates(_e2_stats(rho=0.30), _holm(True, 0.1))["g2"]["pass"]
    assert j.e2_gates(_e2_stats(), _holm(False, -0.1))["grade"] == "D"
    g = j.e2_gates(_e2_stats(pooled=149), _holm(True, 0.1))
    assert g["g1"]["status"] == "표본 부족" and g["grade"] is None and g["grade_rule"] == "D"
    assert j.e2_gates(_e2_stats(pooled=150), _holm(True, 0.1))["g1"]["pass"]


def test_e2_g3_two_thirds_boundary():
    assert j.e2_gates(_e2_stats(n_years=6, n_pos=4), _holm(True))["grade"] == "A"  # 4/6 = 2/3 통과
    assert j.e2_gates(_e2_stats(n_years=6, n_pos=3), _holm(True))["grade"] == "C"
    assert j.e2_gates(_e2_stats(n_years=3, n_pos=2), _holm(True))["grade"] == "A"
    assert j.e2_gates(_e2_stats(n_years=7, n_pos=4), _holm(True))["grade"] == "C"  # 4/7 < 2/3
    assert j.e2_gates(_e2_stats(n_years=0, n_pos=0), _holm(True))["grade"] == "C"  # 해가 없으면 G3 탈락


def test_e2_holm_bound_must_be_strictly_positive():
    ok = lambda x: x > j.E2_G2_LOWER
    r = j.holm_combine([_item("H_E2", 0.001, {0.025: 0.0, 0.05: 0.0}, ok)])
    assert not r["H_E2"]["rejected"]  # 하한 = 0이면 "> 0"이 아니다


# ---------------------------------------------------------------- 중단 조항
def test_stop_clause_text():
    d = {"grade_rule": "D", "grade": "D"}
    a = {"grade_rule": "A", "grade": "A"}
    c = {"grade_rule": "C", "grade": "C"}
    s = j.stop_clause(d, a)
    assert s["fired"] and s["text"] == "중단 조건 충족 — 사용자 확인 대기" and s["which"] == ["E1 국내형"]
    s = j.stop_clause(a, d)
    assert s["fired"] and s["which"] == ["E2 괴리"]
    assert j.stop_clause(d, d)["which"] == ["E1 국내형", "E2 괴리"]
    s = j.stop_clause(c, a)
    assert not s["fired"] and s["text"] is None
    # 표본 부족(등급 없음)이어도 규칙상 D면 해당, 비고를 남긴다
    s = j.stop_clause({"grade_rule": "D", "grade": None}, a)
    assert s["fired"] and s["note"]


# ---------------------------------------------------------------- 정정 E-1 재분류(동결 규칙 판정)
def test_correction_constants_frozen_rules():
    assert j.CORRECTION_ID == "E-1R"
    assert j.JUDGMENT_FAMILY == ("H_E1", "H_E2")
    assert j.HOLM_ALPHAS == (0.025, 0.05)
    assert not hasattr(j, "E2_ALPHA")
    assert j.INTERP_TABLE_SHA256 == "29c89969b8f5d467f2c998110eb8de00afb237de240fe2aad76abc7d07d728b1"
    assert j.DEFAULT_INTERP_TABLE.endswith("provenance/interp_table_v1_approved.md")
    assert "private/tmp" not in j.DEFAULT_INTERP_TABLE
    assert j.DEFAULT_OUT_REL_JUDGMENT == "kr/output/quality_score_etf_judgment_20261010_frozen"
    assert j.JUDGE_VERSION == "quality-score-v0/etf_judge/3-frozen"


def test_correction_note_text():
    n = j.CORRECTION_NOTE
    assert "정정 E-1 재분류(10-10 16:01 사용자)" in n
    assert "판정 변경 부분 철회, 동결 규칙으로 판정" in n
    assert "룩어헤드 기록과 'E1 국내형 순자산 단독 전체 풀'은 결과 전 기록용(P3)" in n
    assert "룩어헤드" in j.LOOKAHEAD_NOTE and "02 문서 §5" in j.LOOKAHEAD_NOTE


def test_two_family_holm_two_stages():
    ok1 = lambda x: x >= j.E1_G2_LOWER
    ok2 = lambda x: x > j.E2_G2_LOWER
    r = j.holm_combine(
        [
            _item("H_E1", 0.001, {0.025: 0.31, 0.05: 0.31}, ok1),
            _item("H_E2", 0.02, {0.025: 0.0, 0.05: 0.01}, ok2),
        ]
    )
    assert list(r) == list(j.JUDGMENT_FAMILY)
    assert r["H_E1"]["stage"] == 1 and r["H_E1"]["stage_alpha"] == 0.025 and r["H_E1"]["rejected"]
    assert r["H_E2"]["stage"] == 2 and r["H_E2"]["stage_alpha"] == 0.05 and r["H_E2"]["rejected"]


def test_render_md_has_correction_and_lookahead_lines():
    import inspect

    src = inspect.getsource(j.render_md)
    assert "lookahead_note" in src and "correction" in src
    assert "e1_domestic_frozen" not in inspect.getsource(j.run)


# ---------------------------------------------------------------- KIND 일치
KIND = pl.DataFrame(
    {
        "notice_date": [date(2013, 3, 4), date(2013, 6, 10), date(2020, 5, 7), date(2020, 5, 20)],
        "kind_name_raw": ["KODEX 가 나", "TIGER Ｘ", "RARE 이름", "RARE 이름"],
        "name_norm": ["KODEX가나", "TIGERX", "RARE이름", "RARE이름"],
        "kind_code5": ["12345", "22222", "33333", "44444"],
        "acptno": ["a", "b", "c", "d"],
        "maturity_name": ["0", "0", "0", "0"],
    }
)


def _ev(rows):
    return pl.DataFrame(
        [{"region": "domestic", "isu_nm": "", **r} for r in rows],
        schema_overrides={"last_date": pl.Date},
    )


def test_norm_name_rule():
    assert j.norm_name("KODEX 가　나 ") == "KODEX가나"
    assert j.norm_name("ＴＩＧＥＲ Ｘ") == "TIGERX"  # NFKC: 전각 → 반각
    assert j.norm_name(None) == ""


def test_kind_match_code_first_then_name():
    ev = _ev([
        dict(isu_cd="123450", isu_nm="전혀 다른 이름", last_date=date(2013, 3, 6)),  # 코드 일치
        dict(isu_cd="999990", isu_nm="TIGER  X", last_date=date(2013, 6, 10)),  # 이름 일치(공백 차이)
        dict(isu_cd="000000", isu_nm="없는 ETF", last_date=date(2014, 1, 3)),  # 불일치
        # 코드 1순위: 코드(22222)가 맞으면 이름(KODEX가나)이 달라도 코드로 센다
        dict(isu_cd="222220", isu_nm="KODEX 가나", last_date=date(2013, 6, 11)),
    ])
    m = j.kind_match(ev, KIND)
    r = m.to_dicts()
    assert [x["kind_method"] for x in r] == ["code", "name", None, "code"]
    assert r[0]["kind_notice_date"] == date(2013, 3, 4) and r[0]["kind_days"] == -2
    assert r[1]["kind_days"] == 0
    assert r[2]["kind_notice_date"] is None and r[2]["kind_days"] is None
    assert r[3]["kind_notice_date"] == date(2013, 6, 10) and r[3]["kind_days"] == -1


def test_kind_name_match_picks_nearest_notice_date():
    ev = _ev([dict(isu_cd="000001", isu_nm="RARE 이름", last_date=date(2020, 5, 21))])
    m = j.kind_match(ev, KIND).row(0, named=True)
    assert m["kind_method"] == "name" and m["kind_notice_date"] == date(2020, 5, 20) and m["kind_days"] == -1


def test_kind_window_gap_with_30_day_margin():
    assert j.kind_window_of(date(2024, 12, 1)) == "in"  # 2025-01-01 − 30일 = 2024-12-02부터 창 밖
    assert j.kind_window_of(date(2024, 12, 2)) == "out"
    assert j.kind_window_of(date(2025, 6, 1)) == "out"
    assert j.kind_window_of(date(2025, 11, 7)) == "out"  # 2025-10-08 + 30일 = 2025-11-07
    assert j.kind_window_of(date(2025, 11, 8)) == "in"


def test_kind_summary_excludes_out_of_window_from_rate():
    ev = _ev([
        dict(isu_cd="123450", last_date=date(2013, 3, 6)),  # 창 안, 코드 일치
        dict(isu_cd="999990", isu_nm="TIGERX", last_date=date(2013, 6, 10)),  # 창 안, 이름 일치
        dict(isu_cd="000000", isu_nm="없음", last_date=date(2014, 1, 3)),  # 창 안, 불일치
        dict(isu_cd="000001", isu_nm="없음", last_date=date(2025, 3, 3), region="foreign"),  # 창 밖
    ])
    s = j.kind_summary(j.kind_match(ev, KIND))["all"]
    assert s["n_events"] == 4 and s["n_out_of_window"] == 1 and s["n_in_window"] == 3
    assert s["n_matched"] == 2 and s["n_matched_code"] == 1 and s["n_matched_name"] == 1 and s["n_unmatched"] == 1
    assert s["match_rate"] == pytest.approx(2 / 3)
    assert s["notice_minus_last_days"]["n"] == 2
    assert j.kind_summary(j.kind_match(ev, KIND))["foreign"]["n_in_window"] == 0
    assert j.kind_summary(j.kind_match(ev, KIND))["foreign"]["match_rate"] is None


def test_load_kind_keeps_only_exchange_delisting(tmp_path):
    p = tmp_path / "k.csv"
    p.write_text(
        "﻿notice_date,kind_name_raw,name_norm,kind_code5,event_type,acptno,maturity_name\n"
        "2010-04-13,KODEX 15,KODEX15,10545,exchange_delisting,x,0\n"
        "2010-05-01,KODEX 16,KODEX16,00001,manager_reason_notice,y,0\n",
        encoding="utf-8",
    )
    k = j.load_kind(p)
    assert k.height == 1 and k["kind_code5"][0] == "10545" and k["notice_date"][0] == date(2010, 4, 13)


# ---------------------------------------------------------------- judgment 보호
def test_run_judgment_requires_env(monkeypatch, tmp_path):
    monkeypatch.delenv(j.JUDGMENT_ENV, raising=False)
    with pytest.raises(PermissionError):
        j.run("judgment", out_dir=str(tmp_path / "o"))
    monkeypatch.setenv(j.JUDGMENT_ENV, "  ")
    with pytest.raises(PermissionError):
        j.run("judgment", out_dir=str(tmp_path / "o"))
    with pytest.raises(ValueError):
        j.run("x", out_dir=str(tmp_path / "o"))
    assert not (tmp_path / "o").exists()  # 보호에 걸리면 아무것도 안 쓴다


# ---------------------------------------------------------------- E1 수정 1: p값 nan (I25)
def test_e1_p_value_drops_nan_rounds_from_numerator_and_b():
    a = np.array([0.1, np.nan, 0.5, 0.2, np.nan])
    assert e1.p_value(a) == pytest.approx((2 + 1) / (3 + 1))  # B = 3(nan 뺌), 분자 = 2 + 1
    assert e1.p_value(np.array([np.nan, np.nan])) == 1.0
    assert e1.p_value(np.array([0.1, 0.3, 0.5, 0.2])) == pytest.approx(4 / 5)  # nan 없으면 그대로


def test_e1_boot_summary_reports_nan_round_count():
    arrays = {k: np.array([0.4, np.nan, 0.1, 0.9]) for k in ("hit", "far", "base_hit", "base_far", "diff")}
    s = e1._boot_summary(e1.BootResult("domestic", 4, 1, arrays))
    assert s["p_n_nan_rounds"] == 1 and s["p_b_used"] == 3
    assert s["p_hit_le_0.30"] == pytest.approx((1 + 1) / (3 + 1))


# ---------------------------------------------------------------- E1 수정 2: 선행 분포 사슬 범위 (I26)
def test_chain_end_default_is_period_end_for_dev_and_september_for_judgment():
    assert e1.CHAIN_END["dev"] == date(2014, 12, 31)
    assert e1.CHAIN_END["judgment"] == date(2026, 9, 30)


def test_chain_end_extends_chain_beyond_period_but_not_formation():
    """판정 사슬은 구간 끝(2026-03) 뒤 월말 점수도 쓴다. 사건의 F·점수 구간은 그대로다."""
    pn = t1.fake_panel(date(2024, 1, 1), date(2026, 12, 31))
    life = t1._life([dict(isu_cd="Z", last_date=date(2026, 8, 20), status="disappeared")])
    months = [date(2025, 12, 31), date(2026, 1, 30), date(2026, 2, 27), date(2026, 3, 31),
              date(2026, 4, 30), date(2026, 5, 29), date(2026, 6, 30), date(2026, 7, 31)]
    sc = t1._scores([dict(isu_cd="Z", month_end=d, alert=(d != date(2026, 7, 31))) for d in months])
    # F = 2026-01-30 (L 2026-08-20의 6개월 전 2026-02-20 이하의 월말)은 판정 구간(끝 2026-03-31) 안
    ev = e1._events_scored(sc, life, pn, "judgment", 6)
    r = ev.row(0, named=True)
    assert r["formation_month_end"] == date(2026, 1, 30) and r["scored"] and r["alert"] is True
    # 사슬은 L 앞 마지막 점수 월말 2026-07-31에서 시작하고 그 월말이 경보가 아니라 0
    assert r["lead_months"] == 0 and r["lead_censored_by_period"] is False
    # 사슬 끝을 구간 끝 2026-03-31로 자르면(옛 동작) 2026-03-31부터 거꾸로 2025-12-31까지 경보가 이어진다
    ev0 = e1._events_scored(sc, life, pn, "judgment", 6, chain_end=date(2026, 3, 31))
    r0 = ev0.row(0, named=True)
    assert r0["lead_months"] == 8 and r0["lead_censored_by_period"] is True
    assert r0["formation_month_end"] == r["formation_month_end"]  # F와 점수 구간은 그대로


def test_dev_chain_still_ends_at_2014_12():
    life, sc = t1._case()
    extra = t1._scores([dict(isu_cd="E1", month_end=date(2015, 2, 27), alert=False)])
    ev = e1._events_scored(pl.concat([sc, extra]), life, t1.PN, "dev", 6)
    assert {x["isu_cd"]: x for x in ev.iter_rows(named=True)}["E1"]["lead_months"] == 9


# ---------------------------------------------------------------- E1 수정 3: 만기형 포함 민감도
@pytest.fixture(scope="module")
def fixture_scored(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e1j")
    days = t1.weekdays(date(2022, 1, 3), date(2023, 12, 29))
    pn = t1.make_panel(tmp, t1.build_rows(days, t1.SPEC))
    return pn, e1.ep.lifecycle(pn)


def test_include_maturity_default_unchanged(fixture_scored):
    pn, life = fixture_scored
    base = e1.monthly_scores(pn, life)
    assert base.equals(e1.monthly_scores(pn, life, include_maturity=False))
    assert "MAT" not in set(base["isu_cd"])


def test_include_maturity_adds_maturity_etf_to_pool(fixture_scored):
    pn, life = fixture_scored
    base = e1.monthly_scores(pn, life)
    inc = e1.monthly_scores(pn, life, include_maturity=True)
    assert "MAT" in set(inc["isu_cd"])
    assert "LEV" not in set(inc["isu_cd"])  # 연금 부적격은 그대로 빠진다
    m = date(2023, 12, 29)
    b = base.filter((pl.col("month_end") == m) & (pl.col("region") == "domestic") & pl.col("in_pool"))
    i = inc.filter((pl.col("month_end") == m) & (pl.col("region") == "domestic") & pl.col("in_pool"))
    assert i.height == b.height + 1 and i["pool_size"].unique().to_list() == [b.height + 1]


def test_event_etfs_include_maturity():
    life = t1._life([
        dict(isu_cd="N", last_date=date(2013, 7, 15)),
        dict(isu_cd="M", last_date=date(2013, 7, 15), exclude_maturity=True),
        dict(isu_cd="L", last_date=date(2013, 7, 15), pension_ineligible_candidate=True),
        dict(isu_cd="U", last_date=date(2013, 7, 15), region="unknown", exclude_maturity=True),
    ])
    assert e1.event_etfs(life)["isu_cd"].to_list() == ["N"]
    assert e1.event_etfs(life, include_maturity=False).equals(e1.event_etfs(life))
    assert e1.event_etfs(life, include_maturity=True)["isu_cd"].to_list() == ["N", "M"]


def test_outcome_stats_include_maturity_counts_maturity_events():
    life = t1._life([
        dict(isu_cd="N", last_date=date(2013, 7, 15)),
        dict(isu_cd="M", last_date=date(2013, 7, 15), exclude_maturity=True),
    ])
    sc = t1._scores([
        dict(isu_cd="N", month_end=date(2012, 12, 31), alert=True),
        dict(isu_cd="M", month_end=date(2012, 12, 31), alert=False),
    ])
    d0 = e1.outcome_stats(sc, life, "dev", 6, panel=t1.PN)["by_region"]["domestic"]
    d1 = e1.outcome_stats(sc, life, "dev", 6, panel=t1.PN, include_maturity=True)["by_region"]["domestic"]
    assert d0["n_events"] == 1 and d1["n_events"] == 2 and d1["hits"] == 1 and d1["hit_rate"] == 0.5
    b1 = e1.bootstrap(sc, life, "dev", 6, panel=t1.PN, b=50, include_maturity=True)["domestic"]
    b0 = e1.bootstrap(sc, life, "dev", 6, panel=t1.PN, b=50)["domestic"]
    assert np.nanmin(b0.arrays["hit"]) == 1.0 and np.nanmin(b1.arrays["hit"]) == 0.0


# ---------------------------------------------------------------- 정정 E-1: 순자산 단독 전체 풀
def test_netasst_full_pool_includes_corr_missing_and_alerts(fixture_scored):
    pn, life = fixture_scored
    full = e1.monthly_scores_netasst_full(pn, life)
    old = e1.monthly_scores(pn, life)
    assert list(full.columns) == list(old.columns)
    assert set(full["region"].unique()) == {"domestic"}  # 해외형은 안 낸다
    m = date(2023, 12, 29)
    d = full.filter(pl.col("month_end") == m)
    pool = d.filter(pl.col("in_pool"))
    # 순자산 있는 국내형 대상 전부(상관 유무 무관). DZ는 순자산 0 → 풀 밖
    assert set(pool["isu_cd"]) == {"D1", "D2", "D3", "D4", "A1", "YNG"}
    assert not d.filter(pl.col("isu_cd") == "DZ")["in_pool"][0]
    assert pool["pool_size"].unique().to_list() == [6]
    p = {r["isu_cd"]: r for r in pool.iter_rows(named=True)}
    assert p["A1"]["pct_netasst"] == 0.0 and p["D4"]["pct_netasst"] == 100.0
    assert p["YNG"]["pct_netasst"] == pytest.approx(20.0)
    for r in p.values():
        assert r["e1"] == r["e1_pct"] == r["pct_netasst"]
        assert r["alert"] == r["baseline_alert"] == (r["pct_netasst"] <= 10)
        assert r["corr"] is None and r["pct_corr"] is None and r["corr_gap"] is None
    assert p["A1"]["alert"] is True and p["YNG"]["alert"] is False


def test_netasst_full_pool_keeps_etf_with_missing_corr(tmp_path):
    """기초지수 종가가 비어 상관이 결측인 ETF도 풀에 있다 — 옛 점수는 풀 밖이다(정정 E-1 사유)."""
    days = t1.weekdays(date(2022, 1, 3), date(2023, 12, 29))
    rows = t1.build_rows(days, t1.SPEC)
    for r in rows:
        if r["ISU_CD"] == "D4":
            r["OBJ_STKPRC_IDX"] = ""
    pn = t1.make_panel(tmp_path, rows)
    life = e1.ep.lifecycle(pn)
    m = date(2023, 12, 29)
    full = e1.monthly_scores_netasst_full(pn, life).filter(pl.col("month_end") == m)
    old = e1.monthly_scores(pn, life).filter(pl.col("month_end") == m)
    f = full.filter(pl.col("isu_cd") == "D4").row(0, named=True)
    o = old.filter(pl.col("isu_cd") == "D4").row(0, named=True)
    assert f["in_pool"] and f["pct_netasst"] == 100.0
    assert not o["in_pool"]
    assert full.filter(pl.col("in_pool")).height == old.filter(pl.col("in_pool") & (pl.col("region") == "domestic")).height + 1


def test_netasst_full_feeds_outcome_stats_and_bootstrap(fixture_scored):
    pn, life = fixture_scored
    full = e1.monthly_scores_netasst_full(pn, life)
    # 사건이 없는 합성 패널이라도 호출은 돼야 한다(구간 보호는 dev)
    st = e1.outcome_stats(full, life, "dev", 6, panel=pn)
    assert "domestic" in st["by_region"]
    bt = e1.bootstrap(full, life, "dev", 6, panel=pn, b=20, seed=1)
    assert "domestic" in bt


def test_netasst_full_default_behavior_of_existing_scores_unchanged(fixture_scored):
    pn, life = fixture_scored
    a = e1.monthly_scores(pn, life)
    e1.monthly_scores_netasst_full(pn, life)
    assert a.equals(e1.monthly_scores(pn, life))
