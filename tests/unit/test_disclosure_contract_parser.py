"""공급계약 공시 파서 단위 시험. 원문 zip은 저장소에 두지 않고 합성 HTML 조각으로 고정한다."""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import pandas as pd
import pytest

from modeler.etl.disclosures.calendar_util import NextTradingDay
from modeler.etl.disclosures.contract import (
    core_status,
    detect_form,
    parse_contract_html,
    parse_date,
    parse_number,
    split_related,
)
from modeler.etl.disclosures.counterparty import (
    build_index,
    classify_counterparty,
    split_parts,
)
from modeler.etl.disclosures.docparse import decode_bytes, read_zip_bytes
from modeler.etl.disclosures.linking import Linker, title_class


def _html(rows: list[list[str]], charset: str = "euc-kr") -> str:
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (
        f'<html><head><meta content="text/html; charset={charset}" '
        'http-equiv="Content-Type"></head>'
        f"<body><table>{body}</table></body></html>"
    )


V1 = [
    ["1. 판매ㆍ공급계약 내용", "반도체 제조장비"],
    ["2. 계약내역", "계약금액(원)", "10,440,000,000"],
    ["최근매출액(원)", "287,069,351,271"],
    ["매출액대비(%)", "3.64"],
    ["3. 계약상대방", "SK하이닉스"],
    ["4. 판매ㆍ공급지역", "국내"],
    ["5. 계약기간", "시작일", "2020-03-05"],
    ["종료일", "2020-06-30"],
    ["6. 주요 계약조건", "-"],
    ["7. 계약(수주)일자", "2020-03-05"],
    ["8. 기타 투자판단에 참고할 사항", "-USD 11,858,078 을 환산한 금액입니다. 부가가치세 별도"],
    ["※ 관련공시", "-"],
]
V2 = [
    ["1. 판매ㆍ공급계약 구분", "공사수주"],
    ["- 체결계약명", "플랜트 공사"],
    ["2. 계약내역", "계약금액(원)", "5,000,000,000"],
    ["최근 매출액(원)", "100,000,000,000"],
    ["매출액 대비(%)", "5.00"],
    ["대규모법인여부", "미해당"],
    ["3. 계약상대", "(주)한화"],
    ["4. 판매ㆍ공급지역", "미국"],
    ["5. 계약기간", "시작일", "-"],
    ["종료일", "-"],
    ["6. 주요 계약조건", "착공일로부터 39개월"],
    ["7. 계약(수주)일자", "2019-01-02"],
    ["공시유보 관련내용", "유보사유", "영업비밀"],
    ["유보기한", "2025-12-07"],
    ["8. 기타 투자판단에 참고할 사항", "-"],
    ["※관련공시", "-"],
]
V3 = [
    ["1. 판매ㆍ공급계약 구분", "상품공급"],
    ["- 체결계약명", "반도체 장비"],
    ["2. 계약내역", "조건부 계약여부", "미해당"],
    ["확정 계약금액", "7,000,000,000"],
    ["조건부 계약금액", "-"],
    ["계약금액 총액(원)", "7,000,000,000"],
    ["최근 매출액(원)", "70,000,000,000"],
    ["매출액 대비(%)", "10.0"],
    ["3. 계약상대방", "삼성전자"],
    ["- 회사와의 관계", "계열회사"],
    ["- 회사와 최근 3년간 동종계약 이행여부", "해당"],
    ["4. 판매ㆍ공급지역", "국내"],
    ["5. 계약기간", "시작일", "2024-01-01"],
    ["종료일", "2024-12-31"],
    ["7. 계약(수주)일자", "2023-12-28"],
    ["8. 기타 투자판단에 참고할 사항", "-"],
]
CANCEL = [
    ["1. 판매ㆍ공급계약 해지 구분", "상품공급"],
    ["- 해지계약명", "LED 조명 공급"],
    ["2. 해지내역", "해지금액(원)", "1,422,500,000"],
    ["최근매출액(원)", "100,000,000,000"],
    ["매출액대비(%)", "1.42"],
    ["3. 계약상대방", "(주)에이비씨"],
    ["4. 해지일자", "2019-02-08"],
    ["5. 주요 해지사유", "계약상대방의 계약미이행"],
    ["※관련공시", "2018-02-09 단일판매ㆍ공급계약체결(자율공시)"],
]


def test_parse_number_and_date():
    assert parse_number("1,234") == (1234.0, False)
    assert parse_number("-") == (None, False)
    assert parse_number("미정") == (None, False)
    assert parse_number("3.64%") == (3.64, False)
    assert parse_number("△100") == (-100.0, False)
    assert parse_number("백억원") == (None, True)
    assert parse_date("2020-03-05") == ("2020-03-05", False)
    assert parse_date("2020년 3월 5일") == ("2020-03-05", False)
    assert parse_date("-") == (None, False)
    assert parse_date("2020-02-31") == (None, True)
    assert parse_date("1900-01-01") == (None, True)


def test_form_v1_fields():
    f = parse_contract_html(_html(V1))
    assert f["form_variant"] == "v1"
    assert f["amount"] == 10_440_000_000 and f["amount_unit"] == "원"
    assert f["recent_sales"] == 287_069_351_271 and f["ratio_pct"] == 3.64
    assert f["counterparty_raw"] == "SK하이닉스"
    assert (f["period_start"], f["period_end"]) == ("2020-03-05", "2020-06-30")
    assert f["contract_date"] == "2020-03-05"
    assert (
        f["fx_hint"] == "USD" and f["vat_hint"] == "별도" and f["fx_amount_raw"] == "USD 11,858,078"
    )
    assert f["ratio_check_ok"] is True
    assert f["related_pairs"] == [] and core_status(f, False) == "ok"


def test_form_v2_dash_and_hold():
    f = parse_contract_html(_html(V2))
    assert f["form_variant"] == "v2" and f["contract_kind"] == "공사수주"
    assert f["contract_name"] == "플랜트 공사"
    assert f["period_start"] is None and f["period_start_raw"] == "-"  # `-`는 NULL, raw는 남긴다
    assert f["terms"] == "착공일로부터 39개월"
    assert f["hold_reason"] == "영업비밀" and f["hold_until"] == "2025-12-07"
    assert f["large_corp"] == "미해당" and f["counterparty_raw"] == "(주)한화"


def test_form_v3_conditional():
    f = parse_contract_html(_html(V3))
    assert f["form_variant"] == "v3"
    assert f["amount_fixed"] == 7_000_000_000 and f["amount_cond"] is None
    assert f["amount"] == 7_000_000_000 and f["cp_relation"] == "계열회사"
    assert f["cp_prior_3y"] == "해당" and f["is_conditional"] == "미해당"


def test_detect_form_priority():
    assert detect_form({"계약금액총액(원)", "판매ㆍ공급계약구분"}) == "v3"
    assert detect_form({"판매ㆍ공급계약구분", "계약금액(원)"}) == "v2"
    assert detect_form({"판매ㆍ공급계약내용", "계약금액(원)"}) == "v1"
    assert detect_form(set()) == "unknown"


def test_cancel_fields_and_related():
    f = parse_contract_html(_html(CANCEL))
    assert f["cancel_amount"] == 1_422_500_000 and f["cancel_date"] == "2019-02-08"
    assert f["cancel_reason"] == "계약상대방의 계약미이행" and f["contract_name"] == "LED 조명 공급"
    assert f["related_pairs"] == [("2018-02-09", "단일판매ㆍ공급계약체결(자율공시)")]
    assert core_status(f, True) == "ok"


def test_amount_unit_label_scales_and_foreign_label():
    rows = [r[:] for r in V1]
    rows[1] = ["2. 계약내역", "계약금액(백만원)", "10,440"]
    f = parse_contract_html(_html(rows))
    assert f["amount"] == 10_440_000_000 and f["amount_unit"] == "백만원"
    rows[1] = ["2. 계약내역", "계약금액(USD)", "10,440"]
    f = parse_contract_html(_html(rows))
    assert f["amount"] is None and "amount_currency" in f["parse_err"]
    rows[1] = [
        "2. 계약내역",
        "계약금액",
        "10,440,000,000",
    ]  # 단위 표기가 없으면 원으로 가정하고 표시
    f = parse_contract_html(_html(rows))
    assert f["amount"] == 10_440_000_000 and f["amount_unit_assumed"] is True


def test_number_error_is_recorded_not_silenced():
    rows = [r[:] for r in V1]
    rows[3] = ["매출액대비(%)", "약 3.6"]
    f = parse_contract_html(_html(rows))
    assert (
        f["ratio_pct"] is None and f["ratio_pct_raw"] == "약 3.6" and "ratio_pct" in f["parse_err"]
    )
    assert core_status(f, False) == "partial"


def test_ratio_check_flags_mismatch():
    rows = [r[:] for r in V1]
    rows[3] = ["매출액대비(%)", "9.99"]
    assert parse_contract_html(_html(rows))["ratio_check_ok"] is False


def test_split_related_formats():
    assert split_related("-") == []
    s = "2012.12.31 (정정) 단일판매·공급계약 체결(자율공시) 2013.12.31 기타 주요경영사항(자율공시)"
    assert split_related(s) == [
        ("2012-12-31", "(정정) 단일판매·공급계약 체결(자율공시)"),
        ("2013-12-31", "기타 주요경영사항(자율공시)"),
    ]
    s = "[2012.01.13]단일판매ㆍ공급계약 체결 [2013.10.31][기재정정]단일판매ㆍ공급계약 체결"
    assert [d for d, _ in split_related(s)] == ["2012-01-13", "2013-10-31"]
    s = "- 2012년 1월 4일 기타 주요경영사항 - 2013년 1월 4일 단일판매 공급계약체결(자율공시)"
    assert [d for d, _ in split_related(s)] == ["2012-01-04", "2013-01-04"]


def test_encoding_utf8_cp949_and_declared_mismatch():
    text = _html(V1)  # 선언은 euc-kr
    for enc in ("utf-8", "cp949"):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("20200305000001.xml", text.encode(enc))
        doc = read_zip_bytes(buf.getvalue(), "20200305000001.xml".split(".")[0])
        assert doc.encoding == enc and doc.declared_encoding == "cp949"
        assert doc.encoding_mismatch is (enc == "utf-8")
        assert "반도체 제조장비" in doc.text
    assert decode_bytes(b"\xff\xfe\x00")[1] == "replace"


def test_zip_picks_body_member():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("20200305000001_00760.xml", b"<html><body>attach</body></html>")
        z.writestr("20200305000001.xml", _html(V1).encode("utf-8"))
    assert read_zip_bytes(buf.getvalue(), "20200305000001").member == "20200305000001.xml"


def test_unknown_form_is_fail():
    f = parse_contract_html(
        "<html><body><table><tr><td>아무 표</td><td>값</td></tr></table></body></html>"
    )
    assert f["form_variant"] == "unknown" and core_status(f, False) == "fail"


def test_next_trading_day():
    cal = NextTradingDay([dt.date(2026, 10, 22), dt.date(2026, 10, 23)])
    cal2 = NextTradingDay([dt.date(2026, 10, 15), dt.date(2026, 10, 19)])
    assert cal2.next_after(dt.date(2026, 10, 15)) == (dt.date(2026, 10, 19), False)
    assert cal2.next_after(dt.date(2026, 10, 16)) == (
        dt.date(2026, 10, 19),
        False,
    )  # 접수일이 휴장일
    assert cal.next_after(dt.date(2026, 10, 23)) == (dt.date(2026, 10, 26), True)  # 금 → 월, 가정


def _idx():
    dm = pd.DataFrame(
        {
            "corp_code": ["1", "2", "3", "4", "5"],
            "ticker": ["000660", None, None, "111111", "222222"],
            "corp_name": ["SK하이닉스", "한국서부발전", "(주)동명", "동명", "동명"],
            "stock_name": ["SK하이닉스", None, None, "동명", "동명"],
        }
    )
    sm = pd.DataFrame({"ticker": ["000660"], "name": ["SK하이닉스"]})
    return build_index(dm, sm, "2026-10-23")


def test_counterparty_exact_normalized_no_fuzzy():
    idx = _idx()
    r = classify_counterparty("SK하이닉스", idx)[0]
    assert (
        r["cp_class"] == "listed_exact"
        and r["matched_ticker"] == "000660"
        and r["matched_corp_code"] == "1"
    )
    r = classify_counterparty("한국서부발전(주)", idx)[0]
    assert r["cp_class"] == "unlisted_dart_normalized" and r["matched_ticker"] is None
    r = classify_counterparty("에스케이하이닉스(주)", idx)[0]  # 퍼지 금지: 못 맞춘다
    assert r["cp_class"] == "unmatched_korean_name" and r["matched_ticker"] is None
    r = classify_counterparty("동명", idx)[0]  # 동명 후보 여럿이면 연결하지 않는다
    assert (
        r["cp_class"] == "ambiguous_name"
        and r["matched_corp_code"] is None
        and r["n_candidates"] == 2
    )
    assert classify_counterparty("-", idx)[0]["cp_class"] == "missing"
    assert classify_counterparty("Nordex USA, Inc.", idx)[0]["cp_class"] == "foreign_or_latin_name"
    assert classify_counterparty("아시아 소재 선사", idx)[0]["cp_class"] == "anonymous_descriptor"
    assert classify_counterparty("방위사업청", idx)[0]["cp_class"] == "public_sector_unmatched"


def test_counterparty_split_parts():
    assert split_parts("무림페이퍼(주), 무림SP(주)") == ["무림페이퍼(주)", "무림SP(주)"]
    assert split_parts("ABC Co., Ltd.") == ["ABC Co., Ltd."]
    assert split_parts("A사(서울, 부산)") == ["A사(서울, 부산)"]
    idx = _idx()
    parts = classify_counterparty("SK하이닉스, 한국서부발전", idx)
    assert [p["part_idx"] for p in parts] == [0, 1]
    assert [p["cp_class"] for p in parts] == ["listed_exact", "unlisted_dart_exact"]


def _ev(rcept, corp, d, name, amt, nm="단일판매ㆍ공급계약체결"):
    return dict(
        rcept_no=rcept, corp_code=corp, rcept_dt=d, contract_name=name, amount=amt, report_nm=nm
    )


def test_cancel_link_rules():
    e1 = _ev("E1", "C", "20180209", "LED", 100.0)
    e2 = _ev("E2", "C", "20180209", "다른 계약", 200.0)
    e3 = _ev("E3", "C", "20180210", "LED", 100.0)
    lk = Linker([e1, e2, e3], [e1, e2, e3])
    base = {"corp_code": "C", "contract_name": "LED", "cancel_amount": 100.0, "amount": None}
    r = lk.link_cancel(base, [("2018-02-10", "단일판매ㆍ공급계약체결")])
    assert r["first_rcept_no"] == "E3" and r["link_method"] == "unique_corp_date"
    r = lk.link_cancel(
        base, [("2018-02-09", "단일판매ㆍ공급계약체결")]
    )  # 후보 2 → 이름 일치로 줄임
    assert r["first_rcept_no"] == "E1" and r["link_method"] == "name_match"
    r = lk.link_cancel(
        {**base, "contract_name": "없는 이름"}, [("2018-02-09", "단일판매ㆍ공급계약체결")]
    )
    assert r["first_rcept_no"] == "E1" and r["link_method"] == "amount_match"
    r = lk.link_cancel(
        {**base, "contract_name": "x", "cancel_amount": 999.0},
        [("2018-02-09", "단일판매ㆍ공급계약체결")],
    )
    assert r["first_rcept_no"] is None and r["link_method"] == "ambiguous_multiple_candidates"
    r = lk.link_cancel(base, [("2017-01-01", "단일판매ㆍ공급계약체결")])
    assert r["link_method"] == "unlinked_first_not_collected"
    r = lk.link_cancel(
        base, [("2018-02-10", "단일판매ㆍ공급계약체결"), ("2018-02-09", "단일판매ㆍ공급계약체결")]
    )
    assert r["first_pair_date"] == "2018-02-09"  # 가장 이른 날짜
    assert lk.link_cancel(base, [])["link_method"] == "no_related"
    assert (
        lk.link_cancel(base, [("2018-02-09", "기타 경영사항(자율공시)")])["link_method"]
        == "no_contract_pair"
    )
    r = lk.link_cancel(base, [("2018-02-09", "[기재정정]단일판매ㆍ공급계약 체결")])
    assert r["link_method"] == "only_correction_pairs"


def test_title_class():
    assert title_class("단일판매ㆍ공급계약체결(자율공시)") == "conclusion"
    assert title_class("단일판매ㆍ공급계약 해지") == "cancel"
    assert title_class("[기재정정]단일판매ㆍ공급계약 체결") == "conclusion_correction"
    assert title_class("기타 경영사항(자율공시)") == "other"


@pytest.mark.parametrize("val,n", [("2018-02-09 단일판매ㆍ공급계약체결(자율공시)", 1), ("-", 0)])
def test_related_pair_counts(val, n):
    assert len(split_related(val)) == n
