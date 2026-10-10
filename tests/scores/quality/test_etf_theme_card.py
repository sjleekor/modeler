"""테마 상품 카드 시험. 합성 데이터만 쓴다. 점수·결과와 무관하다."""

from datetime import date, timedelta

import polars as pl

from modeler.scores.quality import etf_panel as ep
from modeler.scores.quality import etf_theme_card as tc

from .test_etf_panel import day_rows, make_panel, weekdays

ME = date(2026, 9, 30)


def reason(**kw):
    base = dict(
        has_score=False, pension_ineligible=False, exclude_maturity=False, region="domestic",
        first_date=date(2020, 1, 2), month_end=ME, netasst=1e10,
    )  # fmt: skip
    return tc.no_score_reason(**{**base, **kw})


def test_no_score_reason_classification():
    assert reason(has_score=True) == ""
    assert reason(pension_ineligible=True, exclude_maturity=True) == tc.R_PENSION
    assert reason(exclude_maturity=True, region="unknown") == tc.R_MATURITY
    assert reason(region="unknown") == tc.R_REGION
    assert reason(first_date=date(2025, 10, 1)) == tc.R_NEW  # 12개월 안 참
    assert reason(first_date=date(2025, 9, 30)) == tc.R_CORR  # 딱 1년이면 대상
    assert reason(netasst=None) == tc.R_NETASST
    assert reason() == tc.R_CORR  # 국내형, 순자산 있음, 점수 없음 = 상관 쌍 부족
    assert reason(region="foreign") == tc.R_OTHER  # 해외형은 순자산만 있으면 점수가 있어야 정상
    assert set(tc.NO_SCORE_REASONS) == {
        "상장 1년 미만", "기초지수 종가 부족(상관 쌍 < 200)", "만기형", "연금 부적격 후보",
        "국내/해외 미분류", "순자산 없음",
    }  # fmt: skip


def test_synthetic_token_and_account_memo():
    assert tc.synthetic_display("TIGER 미국S&P500합성H") is True
    assert tc.synthetic_display("KODEX 200") is False
    assert tc.synthetic_display(None) is False
    assert tc.account_memo(True, False) == "연금저축·IRP 불가"
    assert tc.account_memo(False, True) == "IRP 제한 가능"
    assert tc.account_memo(True, True) == "연금저축·IRP 불가; IRP 제한 가능"
    assert tc.account_memo(False, False) == ""


def test_bond_mix_display_token():
    assert tc.bond_mix_display("KODEX 삼성그룹채권혼합") is True
    assert tc.bond_mix_display("KODEX 200") is False
    assert tc.bond_mix_display(None) is False
    assert "bond_mix_display" in tc.CARD_COLUMNS


def test_recent_252_session_window(tmp_path):
    n = 300
    days = weekdays(date(2024, 1, 1), n)
    # 거래대금 = 일 번호(0..299). 최근 252일 = 번호 48..299 → 중앙값 173.5
    rows = day_rows(
        days,
        {"A": lambda i, d: {"TDD_CLSPRC": "100", "NAV": "100", "ACC_TRDVAL": str(i)}},
    )
    pn = make_panel(tmp_path, rows)
    assert tc.window_start_idx(pn.end_idx) == n - 252
    ws = tc.window_stats(pn).row(0, named=True)
    assert ws["n_days_252"] == 252
    assert ws["trdval_median_252"] == 173.5
    assert ws["gap_mean_252"] == 0.0 and ws["n_gap_days_252"] == 252


def test_gap_mean_uses_window_only(tmp_path):
    days = weekdays(date(2024, 1, 1), 300)
    # 창 밖(앞 48일)은 괴리 10%, 창 안은 괴리 1%
    rows = day_rows(
        days,
        {"A": lambda i, d: {"TDD_CLSPRC": "110" if i < 48 else "101", "NAV": "100", "ACC_TRDVAL": "5"}},
    )
    pn = make_panel(tmp_path, rows)
    ws = tc.window_stats(pn).row(0, named=True)
    assert abs(ws["gap_mean_252"] - 0.01) < 1e-12


def test_card_has_no_return_or_price_columns():
    assert len(tc.CARD_COLUMNS) == len(set(tc.CARD_COLUMNS))
    for c in tc.CARD_COLUMNS:
        low = c.lower()
        assert not any(w in low for w in tc.FORBIDDEN_COLUMN_WORDS), c
    assert "fee" in tc.CARD_COLUMNS and tc.FEE_PLACEHOLDER == "원천 미정"


def _tiny_cards(tmp_path):
    """상장 중 ETF 셋: 반도체 일반, 반도체 레버리지, 신규(1년 미만)."""
    days = weekdays(date(2025, 6, 2), 330)
    end = days[-1]
    names = {"AAA": "KODEX 반도체", "BBB": "KODEX 반도체레버리지", "CCC": "KODEX 반도체합성"}
    first = {"AAA": 0, "BBB": 0, "CCC": 280}

    def fn(isu):
        def f(i, d):
            if i < first[isu]:
                return None
            return {
                "TDD_CLSPRC": "100", "NAV": "100", "ACC_TRDVAL": "1000",
                "INVSTASST_NETASST_TOTAMT": "100000000000",
                "IDX_IND_NM": "KRX 반도체", "OBJ_STKPRC_IDX": str(1000 + i),
            }  # fmt: skip

        return f

    rows = []
    for isu in names:
        for r in day_rows(days, {isu: fn(isu)}):
            r["ISU_NM"] = names[isu]
            rows.append(r)
    pn = make_panel(tmp_path, rows)
    life = ep.lifecycle(pn)
    mem = tc.recompute_membership(pn)
    me = pn.month_ends.filter(pl.col("date") < end)["date"][-1]
    sc = tc.e1.monthly_scores(pn, life, last_month_end=me)
    cards = tc.build_cards(pn, life, mem, sc, month_end=me)
    return cards, me


def test_build_cards_end_to_end(tmp_path):
    cards, _ = _tiny_cards(tmp_path)
    assert set(cards["theme"]) >= {"반도체"}
    semi = cards.filter(pl.col("theme") == "반도체")
    assert semi.height == 3
    by = {r["isu_cd"]: r for r in semi.iter_rows(named=True)}
    assert by["BBB"]["pension_eligible_candidate"] is False
    assert by["BBB"]["account_memo"] == tc.MEMO_NOT_ALLOWED
    assert by["BBB"]["e1_no_score_reason"] == tc.R_PENSION
    assert by["CCC"]["synthetic_display"] is True and by["CCC"]["account_memo"] == tc.MEMO_SYNTHETIC
    assert by["CCC"]["e1_no_score_reason"] == tc.R_NEW
    assert by["AAA"]["fee"] == "원천 미정"
    assert by["AAA"]["n_days_252"] == 252
    # 수익률·종가 칸이 없다
    assert not {"close", "tdd_clsprc", "return", "ret"} & {c.lower() for c in cards.columns}
    # 순자산 내림차순(같은 테마 안): null이 뒤로
    s = tc.build_summary(cards).filter(pl.col("theme") == "반도체").row(0, named=True)
    assert s["n_listed"] == 3 and s["n_pension_eligible"] == 2
    assert s[f"n_no_score__{tc.R_PENSION}"] == 1 and s[f"n_no_score__{tc.R_NEW}"] == 1
    md = tc.render_user_theme_md(cards, date(2026, 9, 30), date(2026, 10, 8))
    assert "## 반도체" in md and "상장 3 · 적격 2" in md
    assert "KODEX 반도체레버리지" not in md  # 적격 후보만 표에 넣는다


def test_e1_judgment_label():
    assert tc.e1_judgment_label(True, "domestic") == "국내형 판정 D(2026-10-10, 사용자 결정: 유지 — 설명값)"
    assert tc.e1_judgment_label(True, "foreign") == "해외형 기록용(순자산 단독)"
    assert tc.e1_judgment_label(False, "domestic") == ""
    assert tc.e1_judgment_label(True, None) == ""


def test_card_e1_judgment_column_and_md_header(tmp_path):
    cards, _ = _tiny_cards(tmp_path)
    assert "e1_judgment" in cards.columns
    for r in cards.iter_rows(named=True):
        if r["e1_pct"] is None:
            assert r["e1_judgment"] == ""
        elif r["e1_type"] == "domestic":
            assert r["e1_judgment"] == tc.E1_JUDGMENT_DOMESTIC
        else:
            assert r["e1_judgment"] == tc.E1_JUDGMENT_FOREIGN
    md = tc.render_user_theme_md(cards, date(2026, 9, 30), date(2026, 10, 8))
    assert "국내형 E1은 동결 규칙으로 판정해 D였습니다" in md and "유지" in md
    assert "E2 괴리 판정 A(같은 비교 그룹 안 이듬해 괴리 순위 상관) — 테마 상품에는 참고로만" in md


def test_bond_mix_column_summary_and_md(tmp_path):
    cards, _ = _tiny_cards(tmp_path)
    assert "bond_mix_display" in cards.columns
    # 임시 카드의 한 행 이름을 채권혼합으로 바꿔 요약·표시를 본다(판정 칸은 그대로)
    cards2 = cards.with_columns(
        pl.when(pl.col("isu_cd") == "AAA")
        .then(pl.lit("KODEX 반도체채권혼합"))
        .otherwise(pl.col("isu_nm"))
        .alias("isu_nm")
    ).with_columns(pl.col("isu_nm").str.contains(tc.BOND_MIX_TOKEN).alias("bond_mix_display"))
    s = tc.build_summary(cards2).filter(pl.col("theme") == "반도체").row(0, named=True)
    assert s["n_bond_mix"] == 1
    md = tc.render_user_theme_md(cards2, date(2026, 9, 30), date(2026, 10, 8))
    assert "채권혼합 상품은 주식 비중이 절반 안팎이라 테마 노출이 묽습니다(표시만, 사전은 그대로)" in md
    assert "| 합성 | 채권혼합 |" in md
