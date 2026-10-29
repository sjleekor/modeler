"""사업보고서(b) 해석 표 단위 시험 — 합성 표 조각으로 어댑터·단위·기간·행 종류 규칙을 고정한다.

끝의 개발 표본 시험은 맥에 셀 표 사본이 있을 때만 돈다(정확도 검정이 아니라 회귀 고정).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from modeler.etl.disclosures import bizinterp as bi
from modeler.etl.disclosures.bizparse import parse_num

META = {
    "rcept_no": "20250310000001",
    "rcept_dt": "2025-03-10",
    "corp_code": "00000001",
    "stock_code": "000001",
    "corp_cls": "Y",
    "report_nm": "사업보고서 (2024.12)",
    "src_zip_sha256": "x",
    "parsed_at": "t",
    "available_from": "2025-03-11",
}


def mk(paths: list[str | None], body: list[list], unit: str | None = "(단위 : 천원)") -> list[dict]:
    """열마다 머리 경로(``|`` 구분)와 본문 행(글자 또는 ``(글자, rowspan, colspan)``)으로 셀 표를 만든다.

    셀 표 규칙처럼 rowspan이 걸친 칸은 아래 행에 셀을 두지 않는다(열 위치는 논리 열).
    """
    cells: list[dict] = []

    def cell(r, c, t, rs=1, cs=1, hdr=False, path=None):
        v, pct, _ = parse_num(t) if t else (None, False, False)
        return {
            "rcept_no": META["rcept_no"],
            "section_key": "x",
            "table_idx": 0,
            "row_idx": r,
            "col_idx": c,
            "rowspan": rs,
            "colspan": cs,
            "is_header": hdr,
            "header_path_raw": path,
            "text_raw": t,
            "num_value": v,
            "num_has_pct": pct,
            "unit_raw": unit,
            "unit_src": "p_before" if unit else None,
            "tbl_header_rule": "th",
        }

    for c, p in enumerate(paths):
        cells.append(cell(0, c, (p or "").split("|")[-1], hdr=True))
    occupied: dict[tuple[int, int], bool] = {}
    for ri, row in enumerate(body, start=1):
        c = 0
        for spec in row:
            while occupied.get((ri, c)):
                c += 1
            t, rs, cs = (spec, 1, 1) if isinstance(spec, str) else spec
            cells.append(cell(ri, c, t, rs, cs, path=paths[c] if c < len(paths) else None))
            for dr in range(rs):
                for dc in range(cs):
                    if dr or dc:
                        occupied[(ri + dr, c + dc)] = True
            c += cs
    return cells


def run(section, cells):
    a = bi.assign_table(section, cells)
    assert a is not None
    return a, bi.interp_table(META, section, 0, cells, a)


# ---------------------------------------------------------------- 단위 (§2.4)


@pytest.mark.parametrize(
    "u,status,factor",
    [
        ("(단위 : 원)", "krw", 1.0),
        ("(단위: 천원)", "krw", 1e3),
        ("(단위 : 백만원)", "krw", 1e6),
        ("(단위 : 억원)", "krw", 1e8),
        ("(단위 : 십억원)", "krw", 1e9),
        ("(단위 : 조원)", "krw", 1e12),
        ("(단위 : 백만원, %)", "krw", 1e6),  # 수량 단위는 무시
        ("(단위 : 톤, 백만원)", "krw", 1e6),
        ("(단위 : K$ )", "foreign", None),
        ("(단위 : USD)", "foreign", None),
        ("(단위 : 천불)", "foreign", None),
        ("(단위 : 백만원, 천원)", "conflict", None),
        ("(단위 : 백만원,천불,Ton)", "conflict", None),
        ("(단위 : 2018년도 까지 백만원)", "conflict", None),
        ("(단위 : 톤)", "nomoney", None),
        ("(단위 : %)", "nomoney", None),
        (None, "missing", None),
        ("  ", "missing", None),
    ],
)
def test_unit_status(u, status, factor):
    assert bi.unit_parse(u) == (status, factor)


# ---------------------------------------------------------------- 행 종류 (§2.2)


@pytest.mark.parametrize(
    "texts,kind",
    [
        (["합 계"], "total"),
        (["총 계"], "total"),
        (["매출액 합계"], "total"),
        (["소 계"], "subtotal"),
        (["계"], "subtotal"),
        (["연결조정"], "adjust"),
        (["내부거래 제거"], "adjust"),
        (["기타"], "item"),  # 기타는 항목
        (["A사"], "item"),
        (["제품", "합계"], "total"),
    ],
)
def test_row_kind(texts, kind):
    assert bi.row_kind_of(texts) == kind


# ---------------------------------------------------------------- 금액(비율) 한 칸 (§2.3)


def test_split_amount_ratio():
    assert bi.split_amount_ratio("1,000 (3.4%)") == (1000.0, 3.4)
    assert bi.split_amount_ratio("182,800,000 (3.4%)") == (182800000.0, 3.4)
    assert bi.split_amount_ratio("(1,234) (5.0)") == (-1234.0, 5.0)
    assert bi.split_amount_ratio("1,000") is None
    assert bi.split_amount_ratio("가나다") is None


# ---------------------------------------------------------------- 이름·익명 (§2.6)


@pytest.mark.parametrize(
    "name,anon",
    [
        ("A사", True),
        ("A", True),
        ("고객A", True),
        ("주요고객1", True),
        ("거래처 B", True),
        ("갑사", True),
        ("Customer A", True),
        ("삼성전자", False),
        ("(주)엘지", False),
    ],
)
def test_anon(name, anon):
    assert bi.anon_info(name)[0] is anon


def test_name_note_split():
    assert bi.split_name_note("삼성전자(*1)") == ("삼성전자", "*1")
    assert bi.split_name_note("A사 (주1)") == ("A사", "주1")
    assert bi.split_name_note("현대차") == ("현대차", None)


# ---------------------------------------------------------------- 기간 (§2.5)


def _per(labels, fy0=2024):
    return bi.resolve_periods(labels, fy0)


def test_period_rel_words():
    r = _per(["당기", "전기", "전전기"])
    assert [(x["period_rel"], x["fiscal_year"]) for x in r] == [(0, 2024), (-1, 2023), (-2, 2022)]
    r = _per(["제 33(당) 기", "제 32(전) 기"])
    assert [x["fiscal_year"] for x in r] == [2024, 2023]


def test_period_ordinal_to_year():
    # 제N기 → 가장 큰 N을 FY0로
    r = _per(["제55기", "제54기", "제53기"], fy0=2019)
    assert [x["fiscal_year"] for x in r] == [2019, 2018, 2017]
    assert [x["period_rel"] for x in r] == [0, -1, -2]


def test_period_year_wins_and_conflict():
    r = _per(["2024년(제31기)", "2023년(제30기)"])
    assert [x["fiscal_year"] for x in r] == [2024, 2023]
    assert not any(x["period_conflict"] for x in r)
    r = _per(["2023년(당기)"])
    assert r[0]["fiscal_year"] == 2023 and r[0]["period_conflict"]  # 연도와 당기가 안 맞음


def test_period_short_year_and_date_range():
    r = _per(["'24년", "'23년"])
    assert [x["fiscal_year"] for x in r] == [2024, 2023]
    r = _per(["제15기(2024.01.01~2024.12.31)"])
    assert r[0]["fiscal_year"] == 2024 and not r[0]["period_irregular"]
    r = _per(["제17기 (20.07.01~20.12.31)"], fy0=2020)
    assert r[0]["fiscal_year"] == 2020 and r[0]["period_irregular"]  # 6개월짜리 사업연도


def test_period_month_is_irregular():
    r = _per(["2023년 10월"], fy0=2023)
    assert r[0]["fiscal_year"] == 2023 and r[0]["period_irregular"]
    r = _per(["제55기(2024년 12월말)"])
    assert r[0]["fiscal_year"] == 2024 and not r[0]["period_irregular"]


def test_period_missing_header_infers_current():
    r = _per([None])
    assert r[0]["period_rel"] == 0 and r[0]["fiscal_year"] == 2024 and r[0]["period_inferred"]
    r = _per([None, "당기"])  # 일부만 비면 지어내지 않는다
    assert r[0]["fiscal_year"] is None and not r[0]["period_inferred"]


def test_fy0_from_report_name():
    assert bi.fy0_of("사업보고서 (2024.12)", "2025-03-10") == (2024, 12, "report_nm")
    assert bi.fy0_of("사업보고서 (2023.06)", "2023-09-01") == (2023, 6, "report_nm")
    assert bi.fy0_of("사업보고서", "2025-03-10")[0] == 2024


# ---------------------------------------------------------------- 고객 주석 어댑터


def _note(paths, body, unit="(단위 : 천원)"):
    return mk(paths, body, unit)


def test_customer_c_a2_long_form_and_total():
    cells = _note(
        ["구분", "당기", "전기"],
        [["A사", "1,000", "800"], ["B사", "500", "-"], ["합계", "1,500", "800"]],
    )
    a, rows = run("major_customer_note_consol", cells)
    assert (a["target"], a["adapter"], a["judged"]) == ("customer", "C-A*2", True)
    assert a["n_periods"] == 2
    assert len(rows) == 6  # 3행 × 2기간
    r = [x for x in rows if x["name_raw"] == "A사" and x["period_rel"] == 0][0]
    assert (r["amount_num"], r["amount_krw"], r["fiscal_year"], r["anonymized"]) == (
        1000.0,
        1_000_000.0,
        2024,
        True,
    )
    dash = [x for x in rows if x["name_raw"] == "B사" and x["period_rel"] == -1][0]
    assert dash["amount_num"] is None and dash["amount_raw"] == "-" and dash["amount_krw"] is None
    kinds = {x["name_raw"]: x["row_kind"] for x in rows}
    assert kinds == {"A사": "item", "B사": "item", "합계": "total"}
    assert all(
        x["parser_version"] == "disclosure_bizinterp_1.0.0" and x["source"] == "note_consol"
        for x in rows
    )
    assert all(x["available_from"] == "2025-03-11" for x in rows)


def test_customer_c_ap2_pairs():
    cells = _note(
        ["구분", "당기|금액", "당기|비율", "전기|금액", "전기|비율"],
        [
            ["삼성전자", "1,000", "50.0%", "900", "45.0%"],
            ["기타", "1,000", "50.0", "1,100", "55.0"],
        ],
    )
    a, rows = run("major_customer_note_consol", cells)
    assert a["adapter"] == "C-AP*2" and a["judged"]
    assert len(rows) == 4
    r = [x for x in rows if x["name_raw"] == "삼성전자" and x["period_rel"] == -1][0]
    assert (r["amount_num"], r["ratio_pct"], r["ratio_raw"]) == (900.0, 45.0, "45.0%")
    assert not r["anonymized"]


def test_customer_c_a_single_period_inferred():
    cells = _note(["구분", "금액"], [["A사", "100"], ["B사", "50"]])
    a, rows = run("major_customer_note_sep", cells)
    assert a["adapter"] == "C-A" and a["judged"]
    assert all(x["period_inferred"] and x["fiscal_year"] == 2024 for x in rows)
    assert rows[0]["scope"] == "sep"


def test_customer_c_ap_with_segment_not_judged():
    cells = _note(
        ["구분", "영업부문", "금액", "비율"],
        [["A사", "반도체", "100", "60%"], ["B사", "반도체", "50", "40%"]],
    )
    a, rows = run("major_customer_note_consol", cells)
    assert a["adapter"] == "C-AP+L" and not a["judged"]
    assert rows[0]["segment_raw"] == "반도체" and rows[0]["ratio_pct"] == 60.0


def test_customer_extra_text_column_marks_plus_l():
    cells = _note(["구분", "영업부문", "당기", "전기"], [["A사", "X", "1", "2"]])
    a, _ = run("major_customer_note_consol", cells)
    assert a["adapter"] == "C-A*2+L" and not a["judged"]


def test_customer_name_rowspan_carried():
    cells = _note(["구분", "매출구분", "금액"], [[("A사", 2, 1), "제품", "10"], ["상품", "5"]])
    a, rows = run("major_customer_note_consol", cells)
    assert [x["name_raw"] for x in rows] == ["A사", "A사"]
    assert [x["segment_raw"] for x in rows] == ["제품", "상품"]


def test_customer_dup_of_consol():
    body = [["A사", "1,000", "800"], ["합계", "1,000", "800"]]
    paths = ["구분", "당기", "전기"]
    rows = []
    for sec, ti in (("major_customer_note_consol", 0), ("major_customer_note_sep", 1)):
        cells = _note(paths, body)
        a = bi.assign_table(sec, cells)
        rows += bi.interp_table(META, sec, ti, cells, a)
    # 별도에서 금액 하나가 다르면 중복이 아니다
    cells = _note(paths, [["A사", "1,000", "700"], ["합계", "1,000", "700"]])
    a = bi.assign_table("major_customer_note_sep", cells)
    other = bi.interp_table(META, "major_customer_note_sep", 2, cells, a)
    bi.mark_customer_dups(rows + other)
    sep_same = [x for x in rows if x["source"] == "note_sep"]
    assert all(x["dup_of_consol"] and x["dup_group_id"] for x in sep_same)
    assert not any(x["dup_of_consol"] for x in rows if x["source"] == "note_consol")
    assert not any(x["dup_of_consol"] for x in other)


def test_customer_unit_missing_is_not_borrowed():
    cells = _note(["구분", "당기", "전기"], [["A사", "1,000", "800"]], unit=None)
    a, rows = run("major_customer_note_consol", cells)
    assert a["unit_status"] == "missing"
    assert all(x["amount_krw"] is None and x["amount_num"] is not None for x in rows)


def test_customer_unit_foreign_and_conflict_null_krw():
    for u, st in (
        ("(단위 : USD)", "foreign"),
        ("(단위 : 백만원, 천원)", "conflict"),
        ("(단위 : 톤)", "nomoney"),
    ):
        cells = _note(["구분", "당기", "전기"], [["A사", "10", "8"]], unit=u)
        a, rows = run("major_customer_note_consol", cells)
        assert (
            rows[0]["unit_status"] == st
            and rows[0]["amount_krw"] is None
            and rows[0]["amount_num"] == 10.0
        )


def test_customer_not_customer_table_excluded():
    # 숫자 열이 없는 글자 표
    cells = _note(["구분", "내용"], [["가", "나"], ["다", "라"]])
    assert bi.assign_table("major_customer_note_consol", cells) is None


def test_customer_empty_heading_row_skipped():
    cells = _note(["구분", "당기", "전기"], [["재화의 판매", "", ""], ["주요고객", "77", "69"]])
    _, rows = run("major_customer_note_consol", cells)
    assert {x["name_raw"] for x in rows} == {"주요고객"}


def test_customer_no_adapter():
    cells = _note(["구분", "국내", "국외", "합계"], [["A", "1", "2", "3"]])
    a = bi.assign_table("major_customer_note_consol", cells)
    assert (
        a["adapter"] == "no_adapter"
        and bi.interp_table(META, "major_customer_note_consol", 0, cells, a) == []
    )


# ---------------------------------------------------------------- 수주잔고 어댑터

ORD_PATHS = [
    "품목",
    "수주일자",
    "납기",
    "수주총액|수량",
    "수주총액|금액",
    "기납품액|수량",
    "기납품액|금액",
    "수주잔고|수량",
    "수주잔고|금액",
]


def test_orderbook_o_qa3():
    cells = mk(
        ORD_PATHS,
        [
            ["공사 가", "2023.03.20", "2025.12.31", "-", "2,404", "-", "694", "-", "1,710"],
            ["합 계", "", "", "-", "2,404", "-", "694", "-", "1,710"],
        ],
        "(단위 : 백만원)",
    )
    a, rows = run("orders", cells)
    assert (a["target"], a["adapter"], a["judged"]) == ("orderbook", "O-QA3", True)
    assert len(rows) == 2
    r = rows[0]
    assert (r["item_raw"], r["order_date"], r["due_date"]) == (
        "공사 가",
        "2023-03-20",
        "2025-12-31",
    )
    assert (r["total_amt_num"], r["backlog_amt_krw"], r["total_qty"], r["total_qty_raw"]) == (
        2404.0,
        1_710_000_000.0,
        None,
        "-",
    )
    assert r["fiscal_year"] == 2024 and r["row_kind"] == "item"
    assert rows[1]["row_kind"] == "total"
    assert r["total_amt_num"] - r["delivered_amt_num"] == r["backlog_amt_num"]


def test_orderbook_sales_section_o_a3_and_o_ta():
    cells = mk(
        ["품목", "수주총액", "기납품액", "수주잔고"], [["가", "100", "60", "40"]], "(단위 : 천원)"
    )
    a, rows = run("sales", cells)
    assert a["adapter"] == "O-A3" and a["judged"] and rows[0]["backlog_amt_krw"] == 40_000.0
    cells = mk(["품목", "수주총액", "수주잔고"], [["가", "100", "40"]], "(단위 : 천원)")
    a, rows = run("sales", cells)
    assert a["adapter"] == "O-TA" and not a["judged"] and rows[0]["delivered_amt_num"] is None


def test_orderbook_foreign_unit_keeps_currency_text():
    cells = mk(
        ["품목", "통화", "수주총액", "기납품액", "수주잔고"],
        [["가", "USD", "100", "60", "40"]],
        "(단위 : K$ )",
    )
    a, rows = run("orders", cells)
    assert a["adapter"] == "O-A3"
    assert (
        rows[0]["currency_raw"] == "USD"
        and rows[0]["total_amt_krw"] is None
        and rows[0]["total_amt_num"] == 100.0
    )


def test_orderbook_no_adapter_when_group_unknown():
    cells = mk(["품목", "수주총액", "비고2"], [["가", "100", "5"]])
    a = bi.assign_table("orders", cells)
    assert a["target"] == "orderbook" and a["adapter"] == "no_adapter"


# ---------------------------------------------------------------- 매출 어댑터


def test_segment_s_a_n_rowspan_text_carried_and_long_form():
    cells = mk(
        ["사업부문", "품목", "매출액|제 55 기", "매출액|제 54 기", "매출액|제 53 기"],
        [
            [("제빵", 3, 1), "빵류", "10", "20", "30"],
            ["과자류", "1", "2", "3"],
            ["합 계", "11", "22", "33"],
        ],
        "(단위 : 백만원)",
    )
    a, rows = run("sales", cells)
    assert (a["target"], a["adapter"], a["judged"], a["n_periods"]) == (
        "segment_sales",
        "S-A*n",
        True,
        3,
    )
    assert len(rows) == 9
    assert {x["cat1_raw"] for x in rows} == {"제빵"}  # rowspan 글자 칸이 모든 행에 이어짐
    first = [x for x in rows if x["row_idx"] == 1]
    assert [x["fiscal_year"] for x in first] == [2024, 2023, 2022]
    tot = [x for x in rows if x["row_kind"] == "total"]
    assert len(tot) == 3 and tot[0]["amount_krw"] == 11_000_000.0


def test_segment_s_ap_and_inferred_period():
    cells = mk(
        ["구분", "매출액", "비율"],
        [["국내", "70", "70%"], ["수출", "30", "30%"], ["합계", "100", "100%"]],
        "(단위 : 억원)",
    )
    a, rows = run("sales", cells)
    assert a["adapter"] == "S-AP" and a["judged"] and len(rows) == 3
    assert rows[0]["period_inferred"] and rows[0]["fiscal_year"] == 2024
    assert (rows[0]["amount_krw"], rows[0]["ratio_pct"]) == (7e9, 70.0)


def test_segment_s_ap_n_pairs():
    cells = mk(
        ["구분", "2024년|매출액", "2024년|비율", "2023년|매출액", "2023년|비율"],
        [["가", "10", "50%", "8", "40%"], ["나", "10", "50%", "12", "60%"]],
    )
    a, rows = run("sales", cells)
    assert a["adapter"] == "S-AP*n" and a["judged"] and len(rows) == 4
    r = [x for x in rows if x["cat1_raw"] == "가" and x["fiscal_year"] == 2023][0]
    assert (r["amount_num"], r["ratio_pct"]) == (8.0, 40.0)


def test_segment_s_a_np_n_not_judged():
    cells = mk(
        ["구분", "매출|2024년", "매출|2023년", "비율|2024년", "비율|2023년"],
        [["가", "10", "8", "55%", "45%"], ["나", "8", "10", "45%", "55%"]],
    )
    a, rows = run("sales", cells)
    assert a["adapter"] == "S-A*nP*n" and not a["judged"]
    assert sorted(
        (x["fiscal_year"], x["amount_num"], x["ratio_pct"]) for x in rows if x["cat1_raw"] == "가"
    ) == [(2023, 8.0, 45.0), (2024, 10.0, 55.0)]


def test_segment_ap_in_cell():
    cells = mk(
        ["구분", "2024년 매출액", "2023년 매출액"],
        [["가", "1,000 (3.4%)", "900 (3.0%)"], ["나", "2,000 (6.8%)", "1,900 (6.3%)"]],
    )
    a, rows = run("sales", cells)
    assert a["adapter"] == "S-AP-in-cell" and a["judged"]
    r = [x for x in rows if x["fiscal_year"] == 2024][0]
    assert (r["amount_num"], r["ratio_pct"], r["amount_raw"]) == (1000.0, 3.4, "1,000 (3.4%)")


def test_segment_numeric_rowspan_only_first_row():
    # 한 값이 두 품목 행에 걸침 — 값은 첫 행에만, 둘째 행은 값 칸이 없어 행을 만들지 않는다
    cells = mk(
        ["구분", "품목", "매출액", "비율"],
        [["전문의약품", ("A약", 1, 1), ("100", 2, 1), ("50%", 2, 1)], ["B약"]],
    )
    a, rows = run("sales", cells)
    assert len(rows) == 1 and rows[0]["value_rowspan"] == 2 and rows[0]["amount_num"] == 100.0


def test_segment_adjust_row_flag_and_desc_columns():
    cells = mk(
        ["사업부문", "주요 상표", "매출|2024년", "매출|2023년"],
        [["가", "브랜드X", "10", "9"], ["연결조정", "", "-1", "-1"]],
    )
    a, rows = run("sales", cells)
    assert a["adapter"] == "S-A*n"
    adj = [x for x in rows if x["is_adjust_row"]]
    assert len(adj) == 2 and adj[0]["row_kind"] == "adjust"
    assert rows[0]["desc_raw"] == "브랜드X" and rows[0]["cat1_raw"] == "가"


def test_segment_no_adapter_single_amount_and_quantity_columns():
    cells = mk(["구분", "매출액"], [["가", "1"], ["나", "2"]])
    assert bi.assign_table("sales", cells)["adapter"] == "no_adapter"
    cells = mk(
        ["구분", "매출|2024년|수량", "매출|2024년|금액"], [["가", "1", "2"], ["나", "2", "3"]]
    )
    a = bi.assign_table("sales", cells)
    assert a["adapter"] == "no_adapter"


def test_segment_missing_unit_nulls_krw_only():
    cells = mk(
        ["구분", "매출|2024년", "매출|2023년"], [["가", "10", "9"], ["나", "1", "2"]], unit=None
    )
    _, rows = run("sales", cells)
    assert all(
        x["unit_status"] == "missing" and x["amount_krw"] is None and x["amount_num"] is not None
        for x in rows
    )


def test_segment_dup_tables_marked_not_merged():
    rows = []
    for ti in (0, 1):
        cells = mk(["구분", "매출|2024년", "매출|2023년"], [["가", "10", "9"], ["나", "1", "2"]])
        a = bi.assign_table("sales", cells)
        rows += bi.interp_table(META, "sales", ti, cells, a)
    bi.mark_table_dups(rows, ("cat_path_raw", "period_idx", "amount_raw", "row_kind"))
    assert len(rows) == 8 and len({x["dup_group_id"] for x in rows}) == 1


# ---------------------------------------------------------------- 대상 탐지·범위 밖


def test_biz_customer_and_price_tables_are_out_of_scope():
    cells = mk(["고객", "매출액", "비율"], [["가", "1", "10%"], ["나", "2", "20%"]])
    a = bi.assign_table("sales", cells)
    assert a["target"] == "customer_biz" and a["adapter"] == "not_in_scope"
    assert bi.interp_table(META, "sales", 0, cells, a) == []
    cells = mk(["품목", "2024년 평균단가", "2023년 평균단가"], [["철", "10", "9"]])
    a = bi.assign_table("raw_materials", cells)
    assert a["target"] == "rawmat_price" and a["adapter"] == "not_in_scope"


def test_raw_materials_customer_like_column_not_customer():
    cells = mk(["거래처", "매입액", "비율"], [["가", "1", "10%"], ["나", "2", "20%"]])
    a = bi.assign_table("raw_materials", cells)
    assert a is None or a["target"] != "customer_biz"


def test_adapter_catalog_is_closed():
    assert set(bi.ADAPTERS) == {"customer", "orderbook", "segment_sales"}
    assert {a for t, a in bi.JUDGED} <= {x for v in bi.ADAPTERS.values() for x in v}


# ---------------------------------------------------------------- 문서 단위 상태 (§2.7)


def test_interp_doc_status():
    asg = [
        {
            "rcept_no": "d1",
            "section_key": "sales",
            "table_idx": 0,
            "target": "segment_sales",
            "adapter": "S-AP",
            "unit_status": "krw",
            "parse_status": "ok",
            "judged": True,
        },
        {
            "rcept_no": "d2",
            "section_key": "sales",
            "table_idx": 0,
            "target": "segment_sales",
            "adapter": "no_adapter",
            "unit_status": "krw",
            "parse_status": "ok",
            "judged": False,
        },
        {
            "rcept_no": "d3",
            "section_key": "sales",
            "table_idx": 0,
            "target": "segment_sales",
            "adapter": "S-AP",
            "unit_status": "missing",
            "parse_status": "ok",
            "judged": True,
        },
        {
            "rcept_no": "d4",
            "section_key": "sales",
            "table_idx": 0,
            "target": "segment_sales",
            "adapter": "S-AP",
            "unit_status": "krw",
            "parse_status": "partial",
            "judged": True,
        },
    ]
    out = bi.build_interp_doc(["d1", "d2", "d3", "d4", "d5"], asg, {("d1", "sales", 0): 7})
    st = {(r["rcept_no"], r["target"]): r for r in out}
    assert (
        st[("d1", "segment_sales")]["status"] == "ok" and st[("d1", "segment_sales")]["n_rows"] == 7
    )
    assert st[("d2", "segment_sales")]["status"] == "not_passed_template"
    assert st[("d3", "segment_sales")]["status"] == "unit_failed"
    assert st[("d4", "segment_sales")]["status"] == "parse_partial"
    assert st[("d5", "segment_sales")]["status"] == "no_candidate"
    assert st[("d1", "customer")]["status"] == "no_candidate"
    assert len(out) == 5 * 3
    # 판정 통과 군을 직접 주면 그 군만 쓴다
    out = bi.build_interp_doc(["d1"], asg, {("d1", "sales", 0): 7}, passed_groups=set())
    assert [r["status"] for r in out if r["target"] == "segment_sales"] == ["not_passed_template"]


def test_passed_subset_of_judged():
    assert bi.PASSED < bi.JUDGED
    assert bi.JUDGED - bi.PASSED == {
        ("customer", "C-A"),
        ("orderbook", "O-A3"),
        ("segment_sales", "S-A*n"),
    }


def _asg(adapter, target, rn="d1"):
    return {
        "rcept_no": rn,
        "section_key": "x",
        "table_idx": 0,
        "target": target,
        "adapter": adapter,
        "unit_status": "krw",
        "parse_status": "ok",
        "judged": True,
    }


def test_judged_but_not_passed_is_not_passed_template():
    for tg, ad in (("customer", "C-A"), ("segment_sales", "S-A*n"), ("orderbook", "O-A3")):
        out = bi.build_interp_doc(["d1"], [_asg(ad, tg)], {("d1", "x", 0): 5})
        r = [x for x in out if x["target"] == tg][0]
        assert r["status"] == "not_passed_template"
        assert (r["n_interpreted_tables"], r["n_rows"], r["tmpl_groups"]) == (0, 0, "")
    out = bi.build_interp_doc(["d1"], [_asg("C-AP*2", "customer")], {("d1", "x", 0): 5})
    r = [x for x in out if x["target"] == "customer"][0]
    assert (r["status"], r["n_interpreted_tables"], r["n_rows"]) == ("ok", 1, 5)
    assert r["tmpl_groups"] == "customer:C-AP*2"


def test_passed_flag_on_tables():
    # 합격 군
    cells = _note(
        ["구분", "당기|금액", "당기|비율", "전기|금액", "전기|비율"],
        [
            ["삼성전자", "1,000", "50.0%", "900", "45.0%"],
            ["기타", "1,000", "50.0", "1,100", "55.0"],
        ],
    )
    a, rows = run("major_customer_note_consol", cells)
    assert a["adapter"] == "C-AP*2" and a["judged"] and a["passed"]
    assert rows and all(r["passed"] and r["judged"] for r in rows)
    # 판정 못 한 군(C-A)은 judged지만 passed가 아니다
    cells = _note(["구분", "금액"], [["A사", "100"], ["B사", "50"]])
    a, rows = run("major_customer_note_sep", cells)
    assert a["adapter"] == "C-A" and a["judged"] and not a["passed"]
    assert rows and all(r["judged"] and not r["passed"] for r in rows)
    # 불합격 군(S-A*n)
    cells = mk(
        ["사업부문", "품목", "매출액|제 55 기", "매출액|제 54 기", "매출액|제 53 기"],
        [["제빵", "빵류", "10", "20", "30"], ["제과", "과자류", "1", "2", "3"]],
        "(단위 : 백만원)",
    )
    a, rows = run("sales", cells)
    assert a["adapter"] == "S-A*n" and a["judged"] and not a["passed"]
    assert rows and not any(r["passed"] for r in rows)
    # 접미사(+L)가 붙으면 합격 군이 아니다
    cells = _note(["구분", "영업부문", "당기", "전기"], [["A사", "X", "1", "2"]])
    a, rows = run("major_customer_note_consol", cells)
    assert a["adapter"] == "C-A*2+L" and not a["passed"]


def test_run_full_passed_only_outputs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # run_full과 같은 걸러내기: diag에서 passed==True만
    rows = []
    for sec, cells in (
        (
            "major_customer_note_consol",
            _note(["구분", "당기", "전기"], [["A사", "1", "2"]]),
        ),
        ("major_customer_note_sep", _note(["구분", "금액"], [["A사", "100"]])),
    ):
        _, rs = run(sec, cells)
        rows += rs
    cols = bi.TABLE_COLS["customer"]
    from modeler.etl.disclosures.biz_interp_build import _arrow_type

    schema = pa.schema([(c, _arrow_type(c)) for c in cols])
    t = pa.Table.from_pylist([{c: r.get(c) for c in cols} for r in rows], schema=schema)
    assert "passed" in t.column_names and "judged" in t.column_names
    sub = t.filter(pa.array([bool(x) for x in t["passed"].to_pylist()]))
    assert {g for g in sub["tmpl_group"].to_pylist()} == {"customer:C-A*2"}
    assert {g for g in t["tmpl_group"].to_pylist()} == {"customer:C-A*2", "customer:C-A"}
    pq.write_table(sub, tmp_path / "x.parquet")


# ---------------------------------------------------------------- 개발 표본 회귀 (사본이 있을 때만)

ROOT = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
SNAP = ROOT / "kr" / "derived" / "disclosure_struct" / "snapshot_date=2026-10-28"
DEV = ROOT / "kr" / "output" / "research_prep_20261027" / "b_stage2_interp" / "dev_sample_b102.tsv"
needs_data = pytest.mark.skipif(
    not (SNAP.exists() and DEV.exists()), reason="셀 표 사본·개발 표본 없음"
)


def _dev_tables(sections):
    import duckdb

    from modeler.etl.disclosures.biz_interp_build import iter_tables

    docs = sorted(
        {ln.split("\t")[1].strip() for ln in DEV.read_text().splitlines()[1:] if ln.strip()}
    )
    con = duckdb.connect()
    return list(
        iter_tables(con, str(SNAP / "kr_bizrpt_table_cell" / "**" / "*.parquet"), docs, sections)
    )


@needs_data
def test_dev_sample_regression():
    tabs = _dev_tables(bi_sections())
    by = {}
    for meta, sk, ti, cells in tabs:
        a = bi.assign_table(sk, cells)
        if a is None:
            continue
        rows = bi.interp_table(meta, sk, ti, cells, a)
        by[(meta["rcept_no"], sk, ti)] = (a, rows)
    # 모든 해석 행: 단위가 krw가 아니면 amount_krw가 없고, krw면 amount_num × 배수와 같다
    n = 0
    for a, rows in by.values():
        for r in rows:
            n += 1
            if r["unit_status"] != "krw":
                assert r.get("amount_krw") is None
            elif r.get("amount_num") is not None:
                assert r["amount_krw"] == r["amount_num"] * r["unit_factor"]
    assert n > 1000
    a, rows = by[("20260330001629", "sales", 1)]
    assert a["adapter"] == "S-AP*n"
    top = [x for x in rows if x["row_idx"] == 2 and x["fiscal_year"] == 2025][0]
    assert (top["cat1_raw"], top["cat2_raw"], top["amount_num"], top["ratio_pct"]) == (
        "조선 및 해양플랜트사업부문",
        "제품 매출",
        18738.0,
        75.6,
    )
    a, rows = by[("20240321002069", "orders", 3)]
    assert (
        a["adapter"] == "O-QA3"
        and rows[0]["backlog_amt_num"] == 1710.0
        and rows[0]["order_date"] == "2023-03-20"
    )
    a, rows = by[("20200330002064", "sales", 1)]
    assert a["adapter"] == "S-A*n" and {x["fiscal_year"] for x in rows} == {2019, 2018, 2017}
    a, rows = by[("20250320000731", "major_customer_note_consol", 0)]
    assert a["adapter"] == "C-A*2" and len(rows) == 4


def bi_sections():
    return bi_build_sections()


def bi_build_sections():
    from modeler.etl.disclosures.biz_interp_build import SECTIONS

    return SECTIONS
