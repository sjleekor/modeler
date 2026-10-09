"""사전등록 §11.1 규칙마다 작은 합성 예시 하나씩. 점수·결과와 무관하다."""

import polars as pl

from modeler.scores.quality import etf_classify as ec
from modeler.scores.quality import etf_report as er
from modeler.scores.quality import power  # noqa: F401  (임포트 가능 확인)


def test_active_token():
    assert ec.parse_name("KODEX 바이오액티브").active
    assert ec.parse_name("ACE Korea Active").active
    assert not ec.parse_name("KODEX 200").active


def test_hedge_tokens():
    assert ec.parse_name("TIGER 미국S&P500(H)").hedge == "H"
    assert ec.parse_name("ACE 일본TOPIX레버리지(합성 H)").hedge == "H"
    assert ec.parse_name("TIGER 미국나스닥100(UH)").hedge == "UH"
    assert ec.parse_name("TIGER 미국나스닥100").hedge == "none"


def test_leverage_and_inverse_multiplier():
    assert ec.parse_name("KODEX 레버리지").multiplier == 2.0
    assert ec.parse_name("KODEX 인버스").multiplier == -1.0
    assert ec.parse_name("KODEX 200선물인버스2X").multiplier == -2.0
    assert ec.parse_name("KODEX 코스닥150선물인버스").multiplier == -1.0
    assert ec.parse_name("TIGER 200").multiplier == 1.0
    assert ec.parse_name("KBSTAR 국채선물10년 레버리지").pension_ineligible_candidate
    assert not ec.parse_name("TIGER 200").pension_ineligible_candidate


def test_option_strategy():
    assert ec.parse_name("KODEX 200타겟위클리커버드콜").option == "covered_call"
    assert ec.parse_name("X 버퍼3월액티브").option == "premium_option"
    assert ec.parse_name("ACE 지수", "코스피 200 커버드콜 5% OTM").option == "covered_call"
    assert ec.parse_name("KODEX 200").option == "none"


def test_maturity_yymm():
    assert ec.parse_name("TIGER 26-04 회사채(A+이상)액티브").maturity == "26-04"
    assert ec.parse_name("RISE 27-12 국고채").maturity == "27-12"
    assert ec.parse_name("KODEX 코스닥150").maturity == ""
    assert ec.parse_name("X 26-13").maturity == ""  # 13월은 아니다


def test_return_type_and_currency():
    assert ec.parse_index("FnGuide 반도체 지수(PR)")["return_type"] == "PR"
    assert ec.parse_index("KAP 국고채10년지수(총수익지수)")["return_type"] == "TR"
    assert ec.parse_index("KEDI 미국 퀄리티500 지수(NTR)")["return_type"] == "NTR"
    assert ec.parse_index("코스피 200")["return_type"] == "unspecified"
    assert ec.parse_index("A Index PR/TR Hybrid")["return_type"] == "ambiguous"
    assert ec.parse_index("Solactive Aero Index PR USD")["currency"] == "USD"
    assert ec.parse_index("코스피 200")["currency"] == "unspecified"


def test_base_index_strips_return_type_and_whitespace():
    a = ec.parse_index("FnGuide K-푸드 지수 (PR)")["base_index"]
    b = ec.parse_index("FnGuide K-푸드 지수(시장가격)")["base_index"]
    assert a == b


def test_region_lists():
    assert ec.classify_region("코스피 200")[0] == "domestic"
    assert ec.classify_region("KRX 300")[0] == "domestic"
    assert ec.classify_region("S&P 500")[0] == "foreign"
    assert ec.classify_region("MSCI Korea Index")[0] == "domestic"
    assert ec.classify_region("KAP 미국 국채 10년 지수(총수익)")[0] == "foreign"
    assert ec.classify_region("KRX 300 미국달러 선물혼합지수")[0] == "unknown"
    assert ec.classify_region("미국달러선물지수")[0] == "unknown"  # 명시 목록
    assert ec.classify_region("Samsung Korea Target Date 2030 Index")[0] == "unknown"
    assert ec.classify_region(None)[0] == "unknown"


def _frame(rows):
    return pl.DataFrame(
        rows,
        schema=["BAS_DD", "ISU_CD", "ISU_NM", "IDX_IND_NM", "TDD_CLSPRC"],
        orient="row",
    )


def test_trading_day_is_a_day_with_any_close():
    df = _frame(
        [
            ("20240102", "A", "KODEX 200", "코스피 200", "100"),
            ("20240103", "A", "KODEX 200", "코스피 200", None),  # 휴장일 행
            ("20240103", "B", "TIGER 200", "코스피 200", None),
            ("20240104", "A", "KODEX 200", "코스피 200", "101"),
        ]
    )
    assert ec.trading_days(df) == ["20240102", "20240104"]


def test_last_trading_day_name_is_fixed_and_closed_rows_only():
    df = _frame(
        [
            ("20240102", "A", "OLD 이름", "코스피 200", "100"),
            ("20240103", "A", "NEW 이름액티브", "코스피 200", "101"),
            (
                "20240104",
                "A",
                "휴장일이름 인버스",
                "코스피 200",
                None,
            ),  # 종가 없음 -> 마지막 거래일 아님
        ]
    )
    c = ec.classification_table(df).row(0, named=True)
    assert c["last_trade_date"] == "20240103"
    assert c["isu_nm"] == "NEW 이름액티브"
    assert c["active"] and c["multiplier"] == 1.0


def test_alphanumeric_code_kept_as_string():
    df = _frame([("20240102", "0123A0", "KODEX X", "코스피 200", "5")])
    assert ec.classification_table(df)["isu_cd"].to_list() == ["0123A0"]


def test_group_key_splits_on_each_component():
    base = ("20240102", "KODEX", "코스피 200", "5")

    def key(nm, ix="코스피 200"):
        df = _frame([("20240102", "A", nm, ix, "5")])
        return ec.classification_table(df)["group_key"][0]

    k0 = key("KODEX 200")
    assert key("KODEX 200액티브") != k0
    assert key("KODEX 200(H)") != k0
    assert key("KODEX 200레버리지") != k0
    assert key("KODEX 200 커버드콜") != k0
    assert key("KODEX 200", "코스피 200 TR") != k0
    assert key("KODEX 200", "코스피 200 (USD)") != k0
    assert key("KODEX 200") == k0 and base


def test_distribution_counts_groups_by_year_first_trading_day():
    rows = []
    for i in range(5):  # 같은 키 5개 -> 그룹 크기 5
        rows.append(("20150102", f"{i:06d}", f"ETF{i}", "코스피 200", "10"))
    rows.append(("20150102", "999999", "혼자", "KRX 300", "10"))
    rows.append(("20150105", "999999", "혼자", "KRX 300", "10"))
    df = _frame(rows)
    cls = ec.classification_table(df)
    d = er.distribution(df, cls).row(0, named=True)
    assert d["listed"] == 6
    assert (d["fixed_ga_g5"], d["fixed_ga_e5"]) == (1, 5)
    assert (d["repro_ga_g3"], d["repro_ga_e3"]) == (1, 5)


def test_exclude_maturity_by_name_or_index_and_region_unknown_flag():
    df = _frame(
        [
            ("20240102", "A", "TIGER 26-04 회사채액티브", "KIS 회사채 지수", "5"),
            ("20240102", "B", "KODEX 회사채 액티브", "KIS 회사채2604만기형 지수", "5"),
            ("20240102", "C", "KODEX 200", "코스피 200", "5"),
            ("20240102", "D", "X 혼합", "Solactive Physical AI Index", "5"),
        ]
    )
    c = {r["isu_cd"]: r for r in ec.classification_table(df).iter_rows(named=True)}
    assert c["A"]["exclude_maturity"] and not c["A"]["maturity_by_index_name"]
    assert c["B"]["exclude_maturity"] and c["B"]["maturity_by_index_name"]
    assert not c["C"]["exclude_maturity"] and not c["C"]["exclude_e1_region_unknown"]
    assert c["D"]["exclude_e1_region_unknown"]
    d = er.distribution(df, ec.classification_table(df)).row(0, named=True)
    assert d["full_lenient_held"] == 2 and d["full_lenient_incl_maturity_held"] == 0
