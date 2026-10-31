"""정정 공시(c) 파서·연결 시험 (계획 13). 실제 데이터가 아니라 손으로 만든 최소 HTML을 쓴다."""

from __future__ import annotations

import io
import zipfile

import pandas as pd

from modeler.etl.disclosures.correction import (
    classify_item,
    extract_links,
    parse_correction_text,
    parse_correction_zip,
    parse_dates,
    title_family,
    to_num,
)
from modeler.etl.disclosures.correction_build import (
    Receipts,
    follow_chain,
    kind_variants,
    link_cancel_v2,
    link_correction,
)

HTML = """<html><body><table>
<tr><td>정정일자</td><td>2021-03-02</td></tr>
<tr><td>1. 정정관련 공시서류</td><td>단일판매ㆍ공급계약체결</td></tr>
<tr><td>2. 정정관련 공시서류제출일</td><td>{sub}</td></tr>
<tr><td>3. 정정사유</td><td>금액 변경</td></tr>
<tr><td>4. 정정사항</td></tr>
<tr><td>정정항목</td><td>정정전</td><td>정정후</td></tr>
<tr><td>2. 계약내역 계약금액(원)</td><td>1,000</td><td>2,000</td></tr>
<tr><td>5. 계약기간 - 종료일</td><td>2021-12-31</td><td>2022-06-30</td></tr>
<tr><td>-</td></tr>
<tr><td>1. 판매ㆍ공급계약 내용</td><td>본문</td></tr>
<tr><td>※ 관련공시</td><td>
<a href="http://k/x?acptno=20210201000011&amp;rcpno=20210201000011&amp;orgid=G">
2021-02-01 단일판매ㆍ공급계약체결</a></td></tr>
</table></body></html>"""


def _zip(rcept_no: str, text: str) -> bytes:
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr(f"{rcept_no}.xml", text.encode("utf-8"))
    return b.getvalue()


def test_parse_dates_formats():
    assert parse_dates("2020-03-05") == ["2020-03-05"]
    assert parse_dates("2020.03.05") == ["2020-03-05"]
    assert parse_dates("2019년 11월 14일") == ["2019-11-14"]
    assert parse_dates("20191114") == ["2019-11-14"]
    assert parse_dates("2016.11.28(최근 정정일 : 2020.06.05)") == ["2016-11-28", "2020-06-05"]
    assert parse_dates("2020-13-45") == []
    assert parse_dates(None) == []


def test_to_num_only_when_cell_is_number():
    assert to_num("1,576,260,000") == 1576260000.0
    assert to_num("0.71") == 0.71
    assert to_num("계약금액(원): 110,7") is None
    assert to_num("-") is None


def test_classify_item():
    assert classify_item("5. 계약기간 - 종료일") == ("period_end", False)
    assert classify_item("5. 계약기간(시작일)") == ("period_start", False)
    assert classify_item("2. 계약내역 확정 계약금액 매출액 대비(%)") == ("amount", True)
    assert classify_item("3. 계약상대") == ("counterparty", False)
    assert classify_item("공시유보 사유")[0] == "hold"
    assert classify_item("10. 기타 투자판단에 참고할 사항") == ("other_text", False)
    assert classify_item("6. 주요 계약조건") == ("unmapped", False)


def test_title_family():
    assert title_family("단일판매ㆍ공급계약체결(자율공시)") == "original"
    assert title_family("[기재정정]단일판매ㆍ공급계약 체결") == "correction"
    assert title_family("[첨부추가]단일판매ㆍ공급계약체결") == "attach"
    assert title_family("단일판매ㆍ공급계약해지") == "cancel"
    assert title_family("[기재정정]단일판매ㆍ공급계약해지") == "cancel"
    assert title_family("단일판매ㆍ공급계약(진행사항)") == "other"


def test_parse_correction_text_fields():
    f = parse_correction_text(HTML.format(sub="2021년 2월 1일"), "[기재정정]단일판매ㆍ공급계약체결")
    assert f["corr_kind"] == "value_correction"
    assert f["corr_date"] == "2021-03-02"
    assert f["ref_submit_date"] == "2021-02-01"
    assert f["n_items"] == 2  # '-' 한 칸 행에서 끝난다. 본문 행은 안 센다
    assert f["items"][0] == ("2. 계약내역 계약금액(원)", "1,000", "2,000")
    assert f["note_raw"] == "-"
    assert f["rel_links"] == [("20210201000011", "2021-02-01 단일판매ㆍ공급계약체결")]
    assert f["parse_status"] == "ok"


def test_latest_corr_date_only_with_marker():
    f = parse_correction_text(HTML.format(sub="2016.11.28(최근 정정일 : 2020.06.05)"))
    assert f["ref_submit_date"] == "2016-11-28"
    assert f["ref_latest_corr_date"] == "2020-06-05"
    assert f["ref_submit_n_dates"] == 2


def test_unreadable_submit_date_is_partial():
    f = parse_correction_text(HTML.format(sub="단일판매ㆍ공급계약체결"))
    assert f["ref_submit_date"] is None
    assert f["parse_status"] == "partial"
    assert "ref_submit_date" in f["parse_err"]


def test_attach_only_has_no_block():
    f = parse_correction_text(
        "<html><body><table><tr><td>본문</td></tr></table></body></html>",
        "[첨부추가]단일판매ㆍ공급계약체결",
    )
    assert f["corr_kind"] == "attach_only" and f["parse_status"] == "ok"
    f = parse_correction_text(
        "<html><body><table><tr><td>본문</td></tr></table></body></html>",
        "[기재정정]단일판매ㆍ공급계약체결",
    )
    assert f["corr_kind"] == "no_block" and f["parse_status"] == "partial"


def test_extract_links_only_after_related_marker():
    t = '<a href="x?rcpno=11111111111111">a</a> ※ 관련공시 <a href="x?rcpno=22222222222222">b</a>'
    assert extract_links(t) == [("22222222222222", "b")]


def test_parse_zip_roundtrip():
    f = parse_correction_zip(
        _zip("20210302800001", HTML.format(sub="2021-02-01")), "20210302800001"
    )
    assert f["src_encoding"] == "utf-8" and f["n_items"] == 2


def test_kind_variants_changes_only_ninth_digit():
    v = kind_variants("20191114000058")
    assert v == ["20191114800058", "20191114900058", "20191114600058", "20191114000058"]


def _rc(rows):
    return Receipts(
        pd.DataFrame(rows, columns=["rcept_no", "corp_code", "report_nm", "rcept_dt", "corp_cls"])
    )


def _row(no, dt_, link=None, sub="2021-02-01", corp="C1", cls="Y"):
    return dict(
        rcept_no=no,
        rcept_dt=dt_,
        corp_code=corp,
        corp_cls=cls,
        corr_kind="value_correction",
        parse_status="ok",
        ref_submit_date=sub,
        rel_links=link or [],
    )


ORIG = "단일판매ㆍ공급계약체결"
CORR = "[기재정정]단일판매ㆍ공급계약체결"


def test_link_kind_requires_same_corp_and_contract_title():
    rc = _rc(
        [
            ("20210201800011", "C1", ORIG, "2021-02-01", "Y"),
            ("20210302800001", "C1", CORR, "2021-03-02", "Y"),
        ]
    )
    r = _row("20210302800001", "20210302", [("20210201000011", "2021-02-01 x")])
    res = link_correction(r, rc)
    assert res["link_method"] == "link_kind" and res["target_rcept_no"] == "20210201800011"
    # 같은 번호가 다른 회사 것이면 연결 안 함(날짜 보조도 후보 없음)
    r2 = _row("20210302800001", "20210302", [("20210201000011", "x")], corp="C2")
    assert link_correction(r2, rc)["target_rcept_no"] is None


def test_link_to_non_contract_title_is_not_linked_and_skips_date_stage():
    rc = _rc(
        [
            ("20210201800011", "C1", "투자판단관련주요경영사항", "2021-02-01", "Y"),
            ("20210201800012", "C1", ORIG, "2021-02-01", "Y"),
            ("20210302800001", "C1", CORR, "2021-03-02", "Y"),
        ]
    )
    r = _row("20210302800001", "20210302", [("20210201000011", "x")])
    res = link_correction(r, rc)
    assert res["link_method"] == "link_non_contract" and res["target_rcept_no"] is None


def test_date_stage_unique_multi_and_none():
    base = [("20210302800001", "C1", CORR, "2021-03-02", "Y")]
    one = _rc(
        base
        + [
            ("20210201800011", "C1", ORIG, "2021-02-01", "Y"),
            ("20210201800012", "C1", "단일판매ㆍ공급계약해지", "2021-02-01", "Y"),
        ]
    )
    assert link_correction(_row("20210302800001", "20210302"), one)["link_method"] == "date_unique"
    two = _rc(
        base
        + [
            ("20210201800011", "C1", ORIG, "2021-02-01", "Y"),
            ("20210201800013", "C1", CORR, "2021-02-01", "Y"),
        ]
    )
    assert link_correction(_row("20210302800001", "20210302"), two)["link_method"] == "date_multi"
    assert (
        link_correction(_row("20210302800001", "20210302"), _rc(base))["link_method"]
        == "no_candidate"
    )
    assert (
        link_correction(_row("20210302800001", "20210302", sub=None), one)["link_method"]
        == "no_submit_date"
    )


def test_chain_reaches_original_breaks_and_loops():
    rc = _rc(
        [
            ("20210101800001", "C1", ORIG, "2021-01-01", "Y"),
            ("20210201800002", "C1", CORR, "2021-02-01", "Y"),
            ("20210301800003", "C1", CORR, "2021-03-01", "Y"),
            ("20200101800009", "C1", CORR, "2020-01-01", "Y"),
        ]
    )
    ct = {"20210201800002": "20210101800001", "20210301800003": "20210201800002"}
    assert follow_chain("20210201800002", "20210301800003", ct, rc)[:2] == (
        "reached_original",
        "20210101800001",
    )
    st = follow_chain("20200101800009", "20210301800003", ct, rc)
    assert st[0] == "chain_broken_pre_2020_03_05" and st[1] is None and st[2] == "20200101800009"
    loop = {"20210201800002": "20210301800003", "20210301800003": "20210201800002"}
    assert follow_chain("20210201800002", "20210301800003", loop, rc)[0] == "chain_loop"
    assert (
        follow_chain("20210201800002", "x", {"20210201800002": None}, rc)[0]
        == "chain_broken_unlinked"
    )
    assert follow_chain(None, "x", {}, rc)[0] == "unlinked"


def test_cancel_v2_original_chain_and_missing():
    rc = _rc(
        [
            ("20210101800001", "C1", ORIG, "2021-01-01", "Y"),
            ("20210201800002", "C1", CORR, "2021-02-01", "Y"),
            ("20210501800004", "C1", "단일판매ㆍ공급계약해지", "2021-05-01", "Y"),
        ]
    )
    chain = {"20210201800002": ("reached_original", "20210101800001", "20210101800001")}
    cancel = {"rcept_no": "20210501800004", "corp_code": "C1"}
    res = link_cancel_v2(cancel, [("20210201000002", "정정")], rc, chain)
    assert res["first_rcept_no_v2"] == "20210101800001" and res["link_method_v2"] == "v2_link_chain"
    res = link_cancel_v2(cancel, [("20210101000001", "원")], rc, chain)
    assert res["link_method_v2"] == "v2_link_original"
    assert link_cancel_v2(cancel, [], rc, chain)["link_method_v2"] == "v2_no_link"
    assert (
        link_cancel_v2(cancel, [("20190101000001", "x")], rc, chain)["link_method_v2"]
        == "v2_link_no_dart_match"
    )


def test_chain_longer_than_10_reaches_original_and_cycle_still_detected():
    rows = [("20210101800001", "C1", ORIG, "2021-01-01", "Y")]
    ct = {}
    prev = "20210101800001"
    for i in range(2, 24):  # 정정 22개 → 사슬 길이 23
        no = f"2021{i // 28 + 1:02d}{i % 28 + 1:02d}800{i:03d}"
        rows.append((no, "C1", CORR, f"{no[:4]}-{no[4:6]}-{no[6:8]}", "Y"))
        ct[no] = prev
        prev = no
    rc = _rc(rows)
    st = follow_chain(prev, "99999999800999", ct, rc)
    assert st[0] == "reached_original" and st[1] == "20210101800001" and st[3] == 23
    a, b = "20210301800002", "20210302800003"
    rc2 = _rc([(a, "C1", CORR, "2021-03-01", "Y"), (b, "C1", CORR, "2021-03-02", "Y")])
    assert follow_chain(a, "x", {a: b, b: a}, rc2)[0] == "chain_loop"


def test_date_self_not_linked_by_date_but_link_kept():
    rc = _rc(
        [
            ("20210302800000", "C1", ORIG, "2021-03-02", "Y"),
            ("20210302800001", "C1", CORR, "2021-03-02", "Y"),
        ]
    )
    # 제출일 = 접수일, 링크 없음 → 날짜 보조로 안 잇는다
    r = _row("20210302800001", "20210302", sub="2021-03-02")
    res = link_correction(r, rc)
    assert res["link_method"] == "date_self" and res["target_rcept_no"] is None
    # 같은 상황에서 링크가 있으면 link_kind 유지
    r = _row("20210302800001", "20210302", [("20210302000000", "x")], sub="2021-03-02")
    res = link_correction(r, rc)
    assert res["link_method"] == "link_kind" and res["target_rcept_no"] == "20210302800000"
