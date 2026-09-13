"""B-10 Stage 2 — filing_receipt_quality / capital_change_quality diagnostics."""

from __future__ import annotations

from datetime import date

import duckdb
from modeler.etl.phase_b_quality import (
    register_capital_change_quality_view,
    register_filing_receipt_quality_view,
)


def _receipt_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE dart_filing_receipt_raw ("
        "corp_code VARCHAR, ticker VARCHAR, report_nm VARCHAR, rcept_no VARCHAR, "
        "rcept_dt DATE, rm VARCHAR)"
    )
    return con


def _add_receipt(
    con: duckdb.DuckDBPyConnection,
    *,
    corp_code: str = "00126380",
    ticker: str | None = "005930",
    report_nm: str = "사업보고서 (2023.12)",
    rcept_no: str = "20240310000001",
    rcept_dt: date = date(2024, 3, 10),
    rm: str = "",
) -> None:
    con.execute(
        "INSERT INTO dart_filing_receipt_raw VALUES (?,?,?,?,?,?)",
        [corp_code, ticker, report_nm, rcept_no, rcept_dt, rm],
    )


def _receipt_rows(con: duckdb.DuckDBPyConnection) -> list[dict]:
    register_filing_receipt_quality_view(con)
    result = con.execute("SELECT * FROM filing_receipt_quality")
    columns = [d[0] for d in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def test_amendment_and_later_corrected_are_counted_separately() -> None:
    con = _receipt_con()
    # An original that was corrected later (rm carries '정') ...
    _add_receipt(con, rcept_no="20240310000001", rm="유정")
    # ... and the correction itself, which is a separate receipt.
    _add_receipt(
        con,
        report_nm="[기재정정]사업보고서 (2023.12)",
        rcept_no="20240515000002",
        rcept_dt=date(2024, 5, 15),
        rm="유",
    )
    _add_receipt(
        con, report_nm="주주명부폐쇄", rcept_no="20240601000003", rcept_dt=date(2024, 6, 1)
    )

    row = _receipt_rows(con)[0]
    assert row["receipt_year"] == 2024
    assert row["receipts"] == 3
    assert row["amendment_receipts"] == 1
    assert row["later_corrected_receipts"] == 1
    assert row["periodic_report_receipts"] == 2  # the original and its amendment
    assert row["periodic_amendment_receipts"] == 1
    assert row["amendment_ratio"] == 1 / 3


def test_only_periodic_amendments_size_the_xbrl_backfill() -> None:
    con = _receipt_con()
    _add_receipt(
        con,
        report_nm="[기재정정]단일판매ㆍ공급계약체결",
        rcept_no="20240401000001",
        rcept_dt=date(2024, 4, 1),
    )
    _add_receipt(
        con,
        report_nm="[기재정정]반기보고서 (2024.06)",
        rcept_no="20240901000002",
        rcept_dt=date(2024, 9, 1),
    )

    row = _receipt_rows(con)[0]
    assert row["amendment_receipts"] == 2
    # A corrected supply contract has no financial statement to re-fetch.
    assert row["periodic_amendment_receipts"] == 1


def test_missing_backfill_year_is_an_absent_row_not_a_zero_row() -> None:
    con = _receipt_con()
    _add_receipt(con, rcept_no="20230310000001", rcept_dt=date(2023, 3, 10))
    _add_receipt(con, rcept_no="20250310000002", rcept_dt=date(2025, 3, 10))

    assert [row["receipt_year"] for row in _receipt_rows(con)] == [2023, 2025]


def test_receipts_that_cannot_join_the_panel_are_surfaced() -> None:
    con = _receipt_con()
    _add_receipt(con, ticker=None, rcept_no="20240310000001")
    _add_receipt(con, ticker="", rcept_no="20240310000002")
    _add_receipt(con, rcept_no="not-a-receipt-no")

    row = _receipt_rows(con)[0]
    assert row["receipts_without_ticker"] == 2
    assert row["tickers"] == 1
    assert row["receipts_with_malformed_rcept_no"] == 1


def _capital_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE dart_corp_master (ticker VARCHAR, market VARCHAR, corp_code VARCHAR)")
    con.execute(
        "CREATE TABLE dart_share_count_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, se VARCHAR, istc_totqy BIGINT, tesstk_co BIGINT, "
        "now_to_isu_stock_totqy BIGINT, now_to_dcrs_stock_totqy BIGINT, stlm_dt DATE)"
    )
    con.execute(
        "CREATE TABLE dart_capital_change_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, isu_dcrs_de DATE, isu_dcrs_stle VARCHAR, "
        "isu_dcrs_stock_knd VARCHAR, isu_dcrs_qy BIGINT)"
    )
    con.execute("INSERT INTO dart_corp_master VALUES ('000001','KOSPI','corp1')")
    for year in range(2019, 2024):
        con.execute(
            "INSERT INTO dart_share_count_raw VALUES "
            "('corp1','000001',?,'11011',?,'합계',1000,0,500,0,?)",
            [year, f"{year + 1}0310000001", date(year, 12, 31)],
        )
    return con


def _add_capital_row(
    con: duckdb.DuckDBPyConnection,
    *,
    bsns_year: int,
    reprt_code: str,
    rcept_no: str,
    isu_dcrs_de: date | None,
    isu_dcrs_stle: str,
    isu_dcrs_qy: int | None,
) -> None:
    con.execute(
        "INSERT INTO dart_capital_change_raw VALUES ('corp1','000001',?,?,?,?,?,'보통주',?)",
        [bsns_year, reprt_code, rcept_no, isu_dcrs_de, isu_dcrs_stle, isu_dcrs_qy],
    )


def _capital_rows(con: duckdb.DuckDBPyConnection) -> dict[tuple[int, str], dict]:
    register_capital_change_quality_view(con)
    result = con.execute("SELECT * FROM capital_change_quality")
    columns = [d[0] for d in result.description]
    rows = [dict(zip(columns, row, strict=True)) for row in result.fetchall()]
    return {(r["bsns_year"], r["reprt_code"]): r for r in rows}


def test_placeholder_and_unclassified_rows_are_split_out() -> None:
    con = _capital_con()
    _add_capital_row(
        con,
        bsns_year=2022,
        reprt_code="11011",
        rcept_no="20230310000002",
        isu_dcrs_de=date(2022, 6, 15),
        isu_dcrs_stle="유상증자(일반공모)",
        isu_dcrs_qy=100,
    )
    _add_capital_row(
        con,
        bsns_year=2022,
        reprt_code="11011",
        rcept_no="20230310000002",
        isu_dcrs_de=date(2022, 7, 15),
        isu_dcrs_stle="기타",
        isu_dcrs_qy=50,
    )
    # A quarterly report that carried nothing but the '-' placeholder.
    _add_capital_row(
        con,
        bsns_year=2023,
        reprt_code="11013",
        rcept_no="20230515000004",
        isu_dcrs_de=None,
        isu_dcrs_stle="-",
        isu_dcrs_qy=None,
    )

    rows = _capital_rows(con)
    annual = rows[(2022, "11011")]
    assert annual["is_annual"] is True
    assert annual["event_rows"] == 2
    assert annual["unclassified_rows"] == 1
    assert annual["unclassified_ratio"] == 0.5
    assert annual["economic_rows"] == 1

    quarterly = rows[(2023, "11013")]
    assert quarterly["is_annual"] is False
    assert quarterly["placeholder_rows"] == 1
    assert quarterly["placeholder_ratio"] == 1.0
    assert quarterly["event_rows"] == 0
    # A placeholder row is not an unclassified event -- it is no event at all.
    assert quarterly["unclassified_rows"] == 0
    assert quarterly["unclassified_ratio"] is None


def test_annual_vintage_carries_its_disagreement_with_the_newest_vintage() -> None:
    con = _capital_con()
    # Same event, re-dated across the fiscal-year boundary by the newer report.
    _add_capital_row(
        con,
        bsns_year=2022,
        reprt_code="11011",
        rcept_no="20230310000002",
        isu_dcrs_de=date(2022, 1, 15),
        isu_dcrs_stle="유상증자(일반공모)",
        isu_dcrs_qy=100,
    )
    _add_capital_row(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000003",
        isu_dcrs_de=date(2021, 12, 20),
        isu_dcrs_stle="유상증자(일반공모)",
        isu_dcrs_qy=100,
    )

    rows = _capital_rows(con)
    old = rows[(2022, "11011")]
    assert old["vintage_distance_years"] == 1
    assert old["compared_windows"] == 3
    assert old["feature_changing_windows"] == 2  # the event left 2022 for 2021

    # The newest vintage has nothing newer to be compared against.
    newest = rows[(2023, "11011")]
    assert newest["compared_windows"] is None
    assert newest["feature_changing_rate"] is None
