"""사업보고서(b) 절 위치·표 셀 파서 단위 시험.

합성 HTML 조각으로 규칙을 고정한다. 마지막의 원문 고정 시험은 맥에 사본이 있을 때만 돈다
(06 표본 문서로, 정확도 검정이 아니라 회귀 고정이다).
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import zipfile
from pathlib import Path

import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from modeler.etl.disclosures import biz_build
from modeler.etl.disclosures.bizparse import (
    SECTION_KEYS,
    parse_biz_text,
    parse_num,
    scan_blocks,
    table_cells,
)


def _cells(out: dict, key: str) -> list[dict]:
    return [c for c in out["cells"] if c["section_key"] == key]


def _by_pos(cells: list[dict]) -> dict:
    return {(c["table_idx"], c["row_idx"], c["col_idx"]): c for c in cells}


def _doc(biz_inner: str, fin_inner: str = "", consol_title: str = "3. 연결재무제표 주석") -> str:
    return (
        "<DOCUMENT><BODY>"
        "<SECTION-1><TITLE>I. 회사의 개요</TITLE><P>가</P></SECTION-1>"
        f"<SECTION-1><TITLE>II. 사업의 내용</TITLE>{biz_inner}</SECTION-1>"
        "<SECTION-1><TITLE>III. 재무에 관한 사항</TITLE>"
        f"<SECTION-2><TITLE>{consol_title}</TITLE>{fin_inner}</SECTION-2>"
        "</SECTION-1></BODY></DOCUMENT>"
    )


# ---------------------------------------------------------------- 숫자


@pytest.mark.parametrize(
    "text,val,pct",
    [
        ("1,234", 1234.0, False),
        ("12.5", 12.5, False),
        ("12.5%", 12.5, True),
        ("-1,234", -1234.0, False),
        ("△1,234", -1234.0, False),
        ("▲ 5", -5.0, False),
        ("(1,234)", -1234.0, False),
        ("△(1,234)", -1234.0, False),
        ("−7", -7.0, False),
        ("0", 0.0, False),
        ("+3", 3.0, False),
    ],
)
def test_parse_num_values(text, val, pct):
    v, p, err = parse_num(text)
    assert v == val and p is pct and err is False


@pytest.mark.parametrize(
    "text",
    ["", "-", "–", "해당없음", "1,234원", "12,34", "1,2", "2024.12.31", "A사", "(주)", "1 234"],
)
def test_parse_num_not_a_number(text):
    assert parse_num(text) == (None, False, False)


# ---------------------------------------------------------------- 표 격자


def test_grid_rowspan_colspan_and_header_path():
    html = (
        "<TABLE><THEAD>"
        '<TR><TH ROWSPAN="2">구분</TH><TH COLSPAN="2">2023년</TH></TR>'
        "<TR><TH>수량</TH><TH>금액</TH></TR></THEAD><TBODY>"
        '<TR><TD ROWSPAN="2">제품</TD><TD>1</TD><TD>1,000</TD></TR>'
        "<TR><TD>2</TD><TD>(300)</TD></TR>"
        "</TBODY></TABLE>"
    )
    blocks = scan_blocks(html).blocks
    tb = [b for b in blocks if b[0] == "t"][0][1]
    info = table_cells(tb)
    assert (info["n_rows"], info["n_cols"], info["rule"]) == (4, 3, "tag")
    pos = {(c["row_idx"], c["col_idx"]): c for c in info["cells"]}
    assert len(info["cells"]) == 9  # 셀 하나 = 한 행
    assert pos[(0, 0)]["rowspan"] == 2 and pos[(0, 1)]["colspan"] == 2
    assert pos[(1, 1)]["is_header"] and pos[(1, 1)]["text_raw"] == "수량"
    # 왼쪽 위 좌표: 제품(rowspan 2)은 (2,0), 아래 행은 col 1부터
    assert pos[(2, 0)]["text_raw"] == "제품" and pos[(2, 0)]["rowspan"] == 2
    assert (3, 0) not in pos and pos[(3, 1)]["text_raw"] == "2"
    # 머리 경로: 열마다 0번 행부터 연속한 머리 셀
    assert pos[(2, 2)]["header_path_raw"] == "2023년|금액"
    assert pos[(2, 1)]["header_path_raw"] == "2023년|수량"
    assert pos[(2, 0)]["header_path_raw"] == "구분"  # rowspan 머리는 한 번만
    assert pos[(2, 2)]["num_value"] == 1000.0 and pos[(3, 2)]["num_value"] == -300.0
    assert (
        pos[(3, 1)]["row_label_raw"] == "제품"
    )  # 0번 열 격자 칸이 위 행에서 이어진 셀이어도 그 글자
    assert pos[(2, 1)]["row_label_raw"] == "제품" and pos[(2, 0)]["row_label_raw"] is None


def test_header_first_row_rule_when_no_th():
    html = "<TABLE><TR><TD>품목</TD><TD>금액</TD></TR><TR><TD>A</TD><TD>10</TD></TR></TABLE>"
    tb = [b for b in scan_blocks(html).blocks if b[0] == "t"][0][1]
    info = table_cells(tb)
    assert info["rule"] == "first_row"
    c = {(x["row_idx"], x["col_idx"]): x for x in info["cells"]}
    assert c[(0, 0)]["is_header"] and not c[(1, 1)]["is_header"]
    assert c[(1, 1)]["header_path_raw"] == "금액"


def test_header_none_when_first_row_numeric_or_single_row():
    html = "<TABLE><TR><TD>2023</TD><TD>10</TD></TR><TR><TD>A</TD><TD>10</TD></TR></TABLE>"
    tb = [b for b in scan_blocks(html).blocks if b[0] == "t"][0][1]
    assert table_cells(tb)["rule"] == "none"
    one = "<TABLE><TR><TD>(단위 : 원)</TD></TR></TABLE>"
    tb1 = [b for b in scan_blocks(one).blocks if b[0] == "t"][0][1]
    assert table_cells(tb1)["rule"] == "none"


def test_unit_cell_not_in_header_path():
    html = (
        '<TABLE><TR><TH COLSPAN="2">(단위 : 천원)</TH></TR>'
        "<TR><TH>구분</TH><TH>당기</TH></TR><TR><TD>A</TD><TD>5</TD></TR></TABLE>"
    )
    tb = [b for b in scan_blocks(html).blocks if b[0] == "t"][0][1]
    c = {(x["row_idx"], x["col_idx"]): x for x in table_cells(tb)["cells"]}
    assert c[(2, 1)]["header_path_raw"] == "당기"


def test_nested_table_text_merged_into_outer_cell():
    html = (
        "<TABLE><TR><TD>바깥"
        "<TABLE><TR><TD>안1</TD><TD>안2</TD></TR></TABLE></TD><TD>x</TD></TR></TABLE>"
    )
    blocks = scan_blocks(html).blocks
    tabs = [b for b in blocks if b[0] == "t"]
    assert len(tabs) == 1 and tabs[0][1].has_nested
    cells = table_cells(tabs[0][1])["cells"]
    assert len(cells) == 2 and cells[0]["text_raw"] == "바깥 안1 안2"


def test_text_raw_whitespace_only_normalised():
    html = "<TABLE><TR><TD>  가&cr; 나 다 </TD><TD></TD></TR></TABLE>"
    tb = [b for b in scan_blocks(html).blocks if b[0] == "t"][0][1]
    cells = table_cells(tb)["cells"]
    assert cells[0]["text_raw"] == "가 나 다" and cells[1]["text_raw"] == ""


# ---------------------------------------------------------------- 절 찾기


def test_section2_method_and_keys():
    biz = (
        "<SECTION-2><TITLE>1. 사업의 개요</TITLE><P>x</P></SECTION-2>"
        "<SECTION-2><TITLE>2. 주요 제품 및 서비스</TITLE><P>(단위 : 백만원)</P>"
        "<TABLE><TR><TH>제품</TH><TH>매출</TH></TR><TR><TD>A</TD><TD>10</TD></TR></TABLE></SECTION-2>"
        "<SECTION-2><TITLE>3. 원재료 및 생산설비</TITLE><P>y</P></SECTION-2>"
        "<SECTION-2><TITLE>4. 매출 및 수주상황</TITLE>"
        "<SECTION-3><TITLE>가. 매출실적</TITLE><TABLE><TR><TD>a</TD></TR></TABLE></SECTION-3>"
        "<SECTION-3><TITLE>다. 수주상황</TITLE><P>해당 없음</P></SECTION-3></SECTION-2>"
        "<SECTION-2><TITLE>5. 위험관리 및 파생거래</TITLE><P>z</P></SECTION-2>"
    )
    out = parse_biz_text(_doc(biz))
    s = out["sections"]
    assert [k for k in SECTION_KEYS if s[k]["found"]] == [
        "main_products",
        "raw_materials",
        "sales",
        "orders",
    ]
    assert all(s[k]["method"] == "section2" for k in ("main_products", "sales", "orders"))
    assert (
        s["main_products"]["n_tables"] == 1 and s["main_products"]["unit_raw"] == "(단위 : 백만원)"
    )
    assert s["raw_materials"]["n_tables"] == 0 and s["raw_materials"]["char_len"] == 1
    # sales는 수주상황 소제목 앞까지, orders는 그 뒤
    assert s["sales"]["n_tables"] == 1 and s["orders"]["n_tables"] == 0
    assert s["orders"]["heading_raw"] == "다. 수주상황"
    assert s["sales"]["char_end"] <= s["orders"]["char_start"]
    # 못 찾은 키는 이유와 함께 행이 있다
    assert s["major_customer_note_consol"]["miss_reason"] == "no_heading"
    assert s["major_customer_note_consol"]["found"] is False


def test_char_positions_point_at_tags():
    biz = "<SECTION-2><TITLE>2. 주요 제품 및 서비스</TITLE><P>가</P></SECTION-2>"
    text = _doc(biz)
    out = parse_biz_text(text)
    s = out["sections"]["main_products"]
    assert text[s["char_start"]] == "<" and text[s["char_start"] :].startswith("<SECTION-2>")
    assert text[s["char_end"]] == "<"


def test_p_split_with_spaced_glyphs_and_cr_lines():
    biz = (
        "<P><SPAN>1. 사 업의 개요&cr;</SPAN></P><P>개요 문장입니다.</P>"
        "<P><SPAN>2. 주요 제품 및 원재료 등&cr;</SPAN>"
        "가. 주요 제품 현황&cr;(단위 : 천원)</P>"
        "<TABLE><TR><TH>품목</TH><TH>금액</TH></TR><TR><TD>A</TD><TD>1,000</TD></TR></TABLE>"
        "<P>3. 생산 및 설비에 관한 사항</P><P>설비</P>"
        "<P>4. 매출에 관한 사항 당사의 매출은 다음과 같습니다.</P>"
        "<TABLE><TR><TD>매출</TD></TR></TABLE>"
        "<P>5. 수주상황</P><P>해당사항 없음</P>"
        "<P>6. 시장위험과 위험관리</P>"
    )
    out = parse_biz_text(_doc(biz))
    s = out["sections"]
    assert s["main_products"]["method"] == "p-split"
    assert s["main_products"]["heading_raw"].startswith("2. 주요 제품 및 원재료")
    assert s["raw_materials"]["found"]  # 같은 제목이 두 키에 모두 맞는다
    assert s["main_products"]["unit_raw"] == "(단위 : 천원)"
    assert s["sales"]["found"] and s["sales"]["n_tables"] == 1  # 제목 뒤에 본문이 붙은 줄
    assert s["orders"]["found"] and s["orders"]["n_tables"] == 0
    assert s["orders"]["heading_raw"].startswith("5. 수주상황")


def test_p_split_known_title_requires_increasing_number():
    # 4번 매출 절 안의 ``1. 매출실적`` 같은 소목록이 절을 끊지 않는다
    biz = (
        "<P>1. 사업의 개요</P><P>x</P>"
        "<P>4. 매출에 관한 사항</P><P>1. 매출실적</P>"
        "<TABLE><TR><TD>a</TD></TR></TABLE><P>2. 판매경로</P>"
        "<TABLE><TR><TD>b</TD></TR></TABLE>"
        "<P>5. 시장위험과 위험관리</P>"
    )
    s = parse_biz_text(_doc(biz))["sections"]
    assert s["sales"]["found"] and s["sales"]["n_tables"] == 2


def test_orders_inline_heading_inside_sales_line():
    biz = (
        "<SECTION-2><TITLE>4. 매출 및 수주상황</TITLE>"
        "<P>(4) 주요매출처 없습니다.(5) 수주상황</P>"
        "<TABLE><TR><TD>o</TD></TR></TABLE></SECTION-2>"
    )
    s = parse_biz_text(_doc(biz))["sections"]
    assert s["orders"]["found"] and s["orders"]["n_tables"] == 1 and s["sales"]["n_tables"] == 0


def test_industry_tag_prefix_2025_form():
    biz = (
        "<SECTION-2><TITLE>2. (제조서비스업)주요 제품 및 서비스</TITLE><P>가</P></SECTION-2>"
        "<SECTION-2><TITLE>4. (제조서비스업)매출 및 수주상황</TITLE><P>나</P></SECTION-2>"
        "<SECTION-2><TITLE>1. (금융업)사업의 개요</TITLE><P>다</P></SECTION-2>"
    )
    s = parse_biz_text(_doc(biz))["sections"]
    assert s["main_products"]["found"] and s["sales"]["found"]


def test_financial_form_has_no_target_sections():
    biz = (
        "<SECTION-2><TITLE>1. 사업의 개요</TITLE><P>x</P></SECTION-2>"
        "<SECTION-2><TITLE>2. 영업의 현황</TITLE><P>y</P></SECTION-2>"
        "<SECTION-2><TITLE>3. 영업설비</TITLE><P>z</P></SECTION-2>"
    )
    out = parse_biz_text(_doc(biz))
    assert out["biz_form"] == "financial"
    assert not any(
        out["sections"][k]["found"] for k in ("main_products", "raw_materials", "sales", "orders")
    )
    assert out["cells"] == []


def test_no_biz_section_flagged():
    out = parse_biz_text(
        "<DOCUMENT><SECTION-1><TITLE>I. 회사의 개요</TITLE></SECTION-1></DOCUMENT>"
    )
    assert "no_biz_section" in out["flags"] and out["biz_form"] == "no_biz"
    assert out["sections"]["sales"]["miss_reason"] == "no_biz_section"


# ---------------------------------------------------------------- 주요 고객 주석


def test_major_customer_note_heading_consol_vs_sep():
    fin_c = (
        "<P>(4) 주요 고객에 대한 정보</P><P>다음과 같습니다.</P><P>(단위: 천원)</P>"
        "<TABLE><TR><TH>구분</TH><TH>당기</TH></TR><TR><TD>A사</TD><TD>5</TD></TR></TABLE>"
        "<P>(5) 법인세</P><TABLE><TR><TD>무관</TD></TR></TABLE>"
    )
    text = (
        "<DOCUMENT><SECTION-1><TITLE>II. 사업의 내용</TITLE><P>x</P></SECTION-1>"
        "<SECTION-1><TITLE>III. 재무에 관한 사항</TITLE>"
        f"<SECTION-2><TITLE>3. 연결재무제표 주석</TITLE>{fin_c}</SECTION-2>"
        "<SECTION-2><TITLE>5. 재무제표 주석</TITLE>"
        "<P>(3) 당기 중 매출액의 10% 이상을 차지하는 고객은 1개사입니다.</P>"
        "<TABLE><TR><TD>B사</TD><TD>7</TD></TR></TABLE></SECTION-2>"
        "</SECTION-1></DOCUMENT>"
    )
    out = parse_biz_text(text)
    c = out["sections"]["major_customer_note_consol"]
    sp = out["sections"]["major_customer_note_sep"]
    assert c["found"] and c["n_tables"] == 1 and c["unit_raw"] == "(단위: 천원)"
    assert c["method"] == "note_heading" and c["heading_raw"] == "(4) 주요 고객에 대한 정보"
    assert sp["found"] and sp["n_tables"] == 1  # 제목 없이 문장으로 시작하는 항목
    assert [x["text_raw"] for x in _cells(out, "major_customer_note_sep")] == ["B사", "7"]


def test_customer_words_in_business_section_do_not_count_as_note():
    biz = "<P>1. 사업의 개요</P><P>(1) 주요 고객에 대한 정보</P><TABLE><TR><TD>a</TD></TR></TABLE>"
    s = parse_biz_text(_doc(biz))["sections"]
    assert (
        not s["major_customer_note_consol"]["found"] and not s["major_customer_note_sep"]["found"]
    )


# ---------------------------------------------------------------- 단위 줄


def _unit_of(inner: str) -> list:
    biz = f"<SECTION-2><TITLE>4. 매출 및 수주상황</TITLE>{inner}</SECTION-2>"
    out = parse_biz_text(_doc(biz))
    return [
        (c["unit_raw"], c["unit_src"])
        for c in _cells(out, "sales")
        if c["col_idx"] == 0 and c["row_idx"] == 0
    ]


def test_unit_p_before_and_not_inherited():
    inner = (
        "<P>(단위 : 백만원)</P>"
        "<TABLE><TR><TD>a</TD><TD>b</TD></TR></TABLE>"
        "<TABLE><TR><TD>c</TD><TD>d</TD></TR></TABLE>"
    )
    assert _unit_of(inner) == [("(단위 : 백만원)", "p_before"), (None, None)]


def test_unit_in_table_first_row_and_single_cell_table_before():
    inner = (
        '<TABLE><TR><TD COLSPAN="2">(단위: 천원)</TD></TR><TR><TD>a</TD><TD>b</TD></TR></TABLE>'
        "<TABLE><TR><TD>(단위 : 원)</TD></TR></TABLE>"
        "<TABLE><TR><TD>c</TD><TD>d</TD></TR></TABLE>"
    )
    got = _unit_of(inner)
    assert got[0] == ("(단위: 천원)", "in_table")
    assert got[2] == ("(단위 : 원)", "tbl_before")


def test_unit_search_stops_at_section_boundary_and_long_paragraph():
    inner = (
        "<P>(단위 : 원)</P><P>" + "가" * 250 + "</P>" "<TABLE><TR><TD>a</TD><TD>b</TD></TR></TABLE>"
    )
    assert _unit_of(inner) == [(None, None)]


# ---------------------------------------------------------------- 실행(zip → parquet)


def _zip_bytes(name: str, text: str, enc: str = "utf-8") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name, text.encode(enc))
    return buf.getvalue()


def test_parse_one_ok_and_fail_rows(tmp_path):
    meta = {
        "rcept_no": "20230315000001",
        "rcept_dt": "20230315",
        "corp_code": "00000001",
        "stock_code": "000001",
        "corp_cls": "Y",
        "report_nm": "사업보고서 (2022.12)",
        "available_from": "2023-03-16",
        "available_from_assumed": False,
    }
    biz = (
        "<SECTION-2><TITLE>2. 주요 제품 및 서비스</TITLE>"
        "<TABLE><TR><TH>a</TH><TH>b</TH></TR><TR><TD>1</TD><TD>2</TD></TR></TABLE></SECTION-2>"
    )
    good = tmp_path / "20230315000001.zip"
    good.write_bytes(_zip_bytes("20230315000001.xml", _doc(biz), "cp949"))
    r = biz_build.parse_one((meta, str(good), "abc1234"))
    assert r["doc"]["parse_status"] == "ok" and r["doc"]["src_encoding"] == "cp949"
    assert r["doc"]["parser_version"].endswith("+abc1234")
    assert len(r["index"]) == len(SECTION_KEYS) and len(r["cells"]) == 4
    assert r["index"][0]["found"] and r["index"][0]["n_cells"] == 4

    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    r = biz_build.parse_one(({**meta, "rcept_no": "20230315000002"}, str(bad), "abc1234"))
    assert r["doc"]["parse_status"] == "fail" and r["doc"]["parse_fail_reason"]
    assert len(r["index"]) == len(SECTION_KEYS)  # 실패해도 키마다 행이 있다
    assert all(not i["found"] and i["miss_reason"] == "parse_fail" for i in r["index"])
    assert r["cells"][0][0] == biz_build.DOC_SECTION  # 셀 표에는 자리표시 fail 행


def test_run_end_to_end_tmp_root(tmp_path, monkeypatch):
    root = tmp_path / "stock_data"
    b = root / "kr/raw/dart_doc_original/b"
    (b / "raw").mkdir(parents=True)
    rows = [
        ("20230315000001", "20230315", "Y"),
        ("20240315000002", "20240315", "K"),
        ("20240315000003", "20240315", "K"),
    ]
    with open(b / "targets.tsv", "w", encoding="utf-8") as f:
        f.write("rcept_no\tgroup\tyear\trcept_dt\tcorp_code\tstock_code\tcorp_cls\treport_nm\n")
        for rn, d, c in rows:
            f.write(f"{rn}\tb\t{d[:4]}\t{d}\t0000000{rn[-1]}\t00000{rn[-1]}\t{c}\t사업보고서\n")
    biz = (
        "<SECTION-2><TITLE>4. 매출 및 수주상황</TITLE><P>(단위 : 원)</P>"
        "<TABLE><TR><TH>a</TH><TH>b</TH></TR><TR><TD>1</TD><TD>2</TD></TR></TABLE></SECTION-2>"
    )
    (b / "raw" / "20230315000001.zip").write_bytes(_zip_bytes("20230315000001.xml", _doc(biz)))
    (b / "raw" / "20240315000002.zip").write_bytes(_zip_bytes("20240315000002.xml", _doc(biz)))
    (b / "raw" / "20240315000003.zip").write_bytes(b"broken")
    monkeypatch.setenv("STOCK_DATA_ROOT", str(root))
    monkeypatch.setattr(
        biz_build,
        "load_trade_dates",
        lambda _d: [
            dt.date(2023, 3, 15),
            dt.date(2023, 3, 16),
            dt.date(2024, 3, 15),
            dt.date(2024, 3, 18),
        ],
    )
    mf = biz_build.run(procs=2, out_name="t_out")
    out = root / "kr/output/t_out"
    idx = pq.read_table(out / "kr_bizrpt_section_index.parquet").to_pandas()
    assert len(idx) == 3 * len(SECTION_KEYS) and mf["parse_status"] == {"ok": 2, "fail": 1}
    assert idx.groupby("rcept_no").size().eq(len(SECTION_KEYS)).all()
    assert idx[idx.rcept_no == "20240315000002"].available_from.iloc[0] == "2024-03-18"
    cell = (
        ds.dataset(str(out / "kr_bizrpt_table_cell"), format="parquet", partitioning="hive")
        .to_table()
        .to_pandas()
    )
    assert sorted(cell.rcpt_year.astype(str).unique()) == ["2023", "2024"]
    ok = cell[cell.section_key == "sales"]
    assert len(ok) == 8 and ok.num_value.notna().sum() == 4
    fail = cell[cell.section_key == "_doc"]
    assert len(fail) == 1 and fail.parse_status.iloc[0] == "fail"
    key = ["rcept_no", "section_key", "table_idx", "row_idx", "col_idx"]
    assert not cell.duplicated(key).any()
    assert json.loads((out / "manifest_bizrpt.json").read_text())["rows"][
        "kr_bizrpt_table_cell"
    ] == len(cell)


# ---------------------------------------------------------------- 원문 회귀 고정(맥 사본이 있을 때)

_SAMPLE = (
    Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
    / "kr/output/research_prep_20261027/b_contract_fields/sample_raw/b"
)


@pytest.mark.skipif(not (_SAMPLE / "20230324000840.zip").exists(), reason="06 표본 원문 없음")
def test_sample_doc_regression_ge2023():
    from modeler.etl.disclosures.docparse import read_zip_bytes

    rn = "20230324000840"
    raw = read_zip_bytes((_SAMPLE / f"{rn}.zip").read_bytes(), rn)
    out = parse_biz_text(raw.text)
    s = out["sections"]
    assert (s["main_products"]["n_tables"], s["raw_materials"]["n_tables"]) == (2, 2)
    assert s["sales"]["method"] == "section2" and s["orders"]["heading_raw"] == "나. 수주상황"
    assert s["sales"]["n_tables"] == 1 and s["orders"]["n_tables"] == 4
    assert s["major_customer_note_consol"]["found"] and s["major_customer_note_sep"]["found"]
    # SECTION 제목으로 찾은 절의 위치는 태그 시작을 가리킨다
    for k in ("main_products", "raw_materials", "sales"):
        assert raw.text[s[k]["char_start"]] == "<"


@pytest.mark.skipif(not (_SAMPLE / "20150331001650.zip").exists(), reason="06 표본 원문 없음")
def test_sample_doc_regression_le2022_p_split():
    from modeler.etl.disclosures.docparse import read_zip_bytes

    rn = "20150331001650"
    raw = read_zip_bytes((_SAMPLE / f"{rn}.zip").read_bytes(), rn)
    s = parse_biz_text(raw.text)["sections"]
    assert s["main_products"]["method"] == "p-split"
    assert s["main_products"]["heading_raw"] == "2. 주요 제품 및 원재료 등"
    assert s["sales"]["heading_raw"] == "4. 매출에 관한 사항" and not s["orders"]["found"]
