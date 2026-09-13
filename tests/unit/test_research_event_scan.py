from __future__ import annotations

from datetime import date

import duckdb
import pytest

from modeler.etl.features.event_scan import (
    ECONOMIC_DECREASE_REASONS,
    ECONOMIC_INCREASE_REASONS,
    EVENT_SCAN_TABLE,
    MECHANICAL_DECREASE_REASONS,
    MECHANICAL_INCREASE_REASONS,
    build_issuance_sql,
    register_event_scan_daily_view,
)
from modeler.etl.marts.financial_quarters import register_fin_quarterly_metric_vintage_view
from modeler.etl.marts.metric_vintages import register_stock_metric_vintage_fact_view


def _weekdays_in(year: int, month: int) -> list[date]:
    days = []
    day = 1
    while True:
        try:
            d = date(year, month, day)
        except ValueError:
            break
        if d.weekday() < 5:
            days.append(d)
        day += 1
    return days


_TRADING_DAYS = [d for month in range(1, 13) for d in _weekdays_in(2023, month)] + [
    d for month in range(1, 13) for d in _weekdays_in(2024, month)
]


def _base_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE dart_corp_master (ticker VARCHAR, market VARCHAR, corp_code VARCHAR)")
    con.execute(
        "CREATE TABLE dart_financial_statement_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "fs_div VARCHAR, sj_div VARCHAR, account_id VARCHAR, account_nm VARCHAR, "
        "ord INTEGER, thstrm_amount DECIMAL(30,4), currency VARCHAR, rcept_no VARCHAR, "
        "thstrm_add_amount DECIMAL(30,4), frmtrm_q_amount DECIMAL(30,4))"
    )
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
    con.execute(
        "CREATE TABLE dart_shareholder_return_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, statement_type VARCHAR, row_name VARCHAR, stock_knd VARCHAR, "
        "dim1 VARCHAR, dim2 VARCHAR, dim3 VARCHAR, metric_code VARCHAR, "
        "value_numeric DECIMAL(30,4), value_text VARCHAR, stlm_dt DATE)"
    )
    con.execute(
        "CREATE TABLE dart_xbrl_fact_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, concept_id VARCHAR, concept_name VARCHAR, context_id VARCHAR, "
        "context_type VARCHAR, period_start DATE, period_end DATE, instant_date DATE, "
        "dimensions VARCHAR, value_numeric DECIMAL(30,4), label_ko VARCHAR)"
    )
    con.execute(
        "CREATE TABLE dart_filing_receipt_raw "
        "(corp_code VARCHAR, rcept_no VARCHAR, report_nm VARCHAR)"
    )
    con.execute(
        "CREATE TABLE dim_stock_pit_daily ("
        "trade_date DATE, ticker VARCHAR, market VARCHAR, market_cap_pit DOUBLE, "
        "shares_is_available BOOLEAN, shares_invalid_flag BOOLEAN)"
    )
    con.execute(
        "CREATE TABLE dim_price_quality_daily ("
        "trade_date DATE, ticker VARCHAR, market VARCHAR, is_halted BOOLEAN, "
        "valid_session_idx BIGINT)"
    )
    con.execute("INSERT INTO dart_corp_master VALUES ('005930', 'KOSPI', '00126380')")
    return con


def _insert_share_count(
    con: duckdb.DuckDBPyConnection,
    *,
    bsns_year: int,
    reprt_code: str,
    rcept_no: str,
    istc_totqy: int,
    now_to_isu: int,
    now_to_dcrs: int,
    stlm_dt: date,
) -> None:
    con.execute(
        "INSERT INTO dart_share_count_raw VALUES " "('00126380','005930',?,?,?,'합계',?,0,?,?,?)",
        [bsns_year, reprt_code, rcept_no, istc_totqy, now_to_isu, now_to_dcrs, stlm_dt],
    )


def _insert_capital_change(
    con: duckdb.DuckDBPyConnection,
    *,
    bsns_year: int,
    reprt_code: str,
    rcept_no: str,
    isu_dcrs_de: date,
    isu_dcrs_stle: str,
    isu_dcrs_qy: int,
) -> None:
    con.execute(
        "INSERT INTO dart_capital_change_raw VALUES " "('00126380','005930',?,?,?,?,?,'보통주',?)",
        [bsns_year, reprt_code, rcept_no, isu_dcrs_de, isu_dcrs_stle, isu_dcrs_qy],
    )


def _insert_dividend_row(
    con: duckdb.DuckDBPyConnection,
    *,
    bsns_year: int,
    rcept_no: str,
    row_name: str,
    value_numeric: float | None,
    value_text: str = "",
    stock_knd: str = "보통주",
) -> None:
    con.execute(
        "INSERT INTO dart_shareholder_return_raw VALUES "
        "('00126380','005930',?,'11011',?,'dividend',?,?,'','','','thstrm',?,?,NULL)",
        [bsns_year, rcept_no, row_name, stock_knd, value_numeric, value_text],
    )


def _insert_shares_instant(
    con: duckdb.DuckDBPyConnection,
    *,
    bsns_year: int,
    rcept_no: str,
    issued: int,
    treasury: int,
) -> None:
    con.execute(
        "INSERT INTO dart_share_count_raw VALUES "
        "('00126380','005930',?,'11011',?,'합계',?,?,NULL,NULL,NULL)",
        [bsns_year, rcept_no, issued, treasury],
    )


def _insert_pit(con: duckdb.DuckDBPyConnection, *, trade_date: date, market_cap: float) -> None:
    con.execute(
        "INSERT INTO dim_stock_pit_daily VALUES (?, '005930', 'KOSPI', ?, TRUE, FALSE)",
        [trade_date, market_cap],
    )
    con.execute(
        "INSERT INTO dim_price_quality_daily VALUES (?, '005930', 'KOSPI', FALSE, 1)",
        [trade_date],
    )


def _register(con: duckdb.DuckDBPyConnection, **views: str) -> None:
    register_stock_metric_vintage_fact_view(con, trading_days=_TRADING_DAYS)
    register_fin_quarterly_metric_vintage_view(con)
    register_event_scan_daily_view(con, trading_days=_TRADING_DAYS, **views)


def _row(con: duckdb.DuckDBPyConnection, trade_date: date) -> tuple | None:
    return con.execute(
        f"SELECT * FROM {EVENT_SCAN_TABLE} WHERE ticker='005930' AND trade_date=?",
        [trade_date],
    ).fetchone()


def _columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE {EVENT_SCAN_TABLE}").fetchall()]


def test_economic_issuance_is_identified_and_counted() -> None:
    con = _base_con()
    _insert_share_count(
        con,
        bsns_year=2022,
        reprt_code="11011",
        rcept_no="20230310000001",
        istc_totqy=1000,
        now_to_isu=500,
        now_to_dcrs=0,
        stlm_dt=date(2022, 12, 31),
    )
    _insert_share_count(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        istc_totqy=1100,
        now_to_isu=600,
        now_to_dcrs=0,
        stlm_dt=date(2023, 12, 31),
    )
    _insert_capital_change(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        isu_dcrs_de=date(2023, 6, 15),
        isu_dcrs_stle="유상증자(일반공모)",
        isu_dcrs_qy=100,
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["issuance_identity_ok"]] is True
    assert row[idx["issuance_classification_complete"]] is True
    assert row[idx["ev_net_share_issuance_yoy"]] == 100 / 1000


def test_mechanical_action_is_excluded_from_economic_issuance() -> None:
    con = _base_con()
    _insert_share_count(
        con,
        bsns_year=2022,
        reprt_code="11011",
        rcept_no="20230310000001",
        istc_totqy=1000,
        now_to_isu=500,
        now_to_dcrs=0,
        stlm_dt=date(2022, 12, 31),
    )
    # A 2:1 stock split doubles shares mechanically -- not economic issuance.
    _insert_share_count(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        istc_totqy=2000,
        now_to_isu=1500,
        now_to_dcrs=0,
        stlm_dt=date(2023, 12, 31),
    )
    _insert_capital_change(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        isu_dcrs_de=date(2023, 6, 15),
        isu_dcrs_stle="주식분할",
        isu_dcrs_qy=1000,
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["issuance_identity_ok"]] is True
    assert row[idx["ev_net_share_issuance_yoy"]] == 0.0


def test_unclassified_reason_blocks_issuance_feature() -> None:
    con = _base_con()
    _insert_share_count(
        con,
        bsns_year=2022,
        reprt_code="11011",
        rcept_no="20230310000001",
        istc_totqy=1000,
        now_to_isu=500,
        now_to_dcrs=0,
        stlm_dt=date(2022, 12, 31),
    )
    _insert_share_count(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        istc_totqy=1100,
        now_to_isu=600,
        now_to_dcrs=0,
        stlm_dt=date(2023, 12, 31),
    )
    _insert_capital_change(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        isu_dcrs_de=date(2023, 6, 15),
        isu_dcrs_stle="기타",
        isu_dcrs_qy=100,
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["issuance_classification_complete"]] is False
    assert row[idx["ev_net_share_issuance_yoy"]] is None


def test_identity_mismatch_blocks_issuance_feature() -> None:
    con = _base_con()
    _insert_share_count(
        con,
        bsns_year=2022,
        reprt_code="11011",
        rcept_no="20230310000001",
        istc_totqy=1000,
        now_to_isu=500,
        now_to_dcrs=0,
        stlm_dt=date(2022, 12, 31),
    )
    _insert_share_count(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        istc_totqy=1100,
        now_to_isu=600,
        now_to_dcrs=0,
        stlm_dt=date(2023, 12, 31),
    )
    # Capital-change rows under-report the actual increase -> identity fails.
    _insert_capital_change(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        isu_dcrs_de=date(2023, 6, 15),
        isu_dcrs_stle="유상증자(일반공모)",
        isu_dcrs_qy=50,
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["issuance_identity_ok"]] is False
    assert row[idx["ev_net_share_issuance_yoy"]] is None


def test_dividend_direct_total_row_is_preferred_when_present() -> None:
    con = _base_con()
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="현금배당금총액(백만원)",
        value_numeric=10,
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["dividend_source"]] == "direct_total"
    assert row[idx["cash_dividends_total"]] == 10 * 1_000_000


def test_dividend_falls_back_to_dps_proxy_when_total_row_missing() -> None:
    con = _base_con()
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="주당 현금배당금(원)",
        value_numeric=500,
    )
    _insert_shares_instant(
        con, bsns_year=2023, rcept_no="20240310000002", issued=1000, treasury=100
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["dividend_source"]] == "dps_proxy"
    assert row[idx["cash_dividends_total"]] == 500 * (1000 - 100)


def test_explicit_dash_normalizes_to_zero_not_null() -> None:
    con = _base_con()
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="주당 현금배당금(원)",
        value_numeric=None,
        value_text="-",
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["cash_dividends_total"]] == 0


def test_payout_yield_combines_dividends_and_buyback() -> None:
    con = _base_con()
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="현금배당금총액(백만원)",
        value_numeric=1,
    )
    # treasury_share_acquisition_amount is cumulative_reported (CF-sourced):
    # each filing's thstrm_amount is cumulative-through-that-quarter, so 4
    # equal 5-unit standalone quarters means Q1=5, Q2=10, Q3=15, Q4(annual)=20.
    for reprt_code, amount, rcept_no in [
        ("11013", 5, "20230410000001"),
        ("11012", 10, "20230810000001"),
        ("11014", 15, "20231110000001"),
        ("11011", 20, "20240310000002"),
    ]:
        con.execute(
            "INSERT INTO dart_financial_statement_raw "
            "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
            "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
            "('00126380','005930',2023,?,'CFS','CF','dart_AcquisitionOfTreasuryShares','',1,?,'KRW',?)",
            [reprt_code, amount, rcept_no],
        )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["buyback_cash_ttm"]] == 20
    dividends = 1 * 1_000_000
    assert row[idx["ev_payout_yield"]] == (dividends + 20) / 1_000_000_000


def test_lag1_variant_and_grain_uniqueness() -> None:
    con = _base_con()
    _insert_share_count(
        con,
        bsns_year=2022,
        reprt_code="11011",
        rcept_no="20230310000001",
        istc_totqy=1000,
        now_to_isu=500,
        now_to_dcrs=0,
        stlm_dt=date(2022, 12, 31),
    )
    _insert_share_count(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        istc_totqy=1100,
        now_to_isu=600,
        now_to_dcrs=0,
        stlm_dt=date(2023, 12, 31),
    )
    _insert_capital_change(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        isu_dcrs_de=date(2023, 6, 15),
        isu_dcrs_stle="유상증자(일반공모)",
        isu_dcrs_qy=100,
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _insert_pit(con, trade_date=date(2024, 3, 18), market_cap=1_000_000_000)
    _register(con)

    dupes = con.execute(
        f"SELECT trade_date, ticker, market, COUNT(*) "
        f"FROM {EVENT_SCAN_TABLE} GROUP BY 1, 2, 3 HAVING COUNT(*) > 1"
    ).fetchall()
    assert dupes == []

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row_prior = _row(con, date(2024, 3, 15))
    row_next = _row(con, date(2024, 3, 18))
    assert (
        row_next[idx["ev_net_share_issuance_yoy_lag1"]]
        == row_prior[idx["ev_net_share_issuance_yoy"]]
    )


# --- §4.4.1 capital-change vintage selection ------------------------------
#
# irdsSttus reprints the whole since-listing history in every report, so one
# real event is stored once per vintage. These cover the dedup rule and the two
# candidate policies the vintage distance probe chooses between.


def _two_annual_positions(
    con: duckdb.DuckDBPyConnection,
    *,
    prior_year: int,
    prior_rcept: str,
    current_year: int,
    current_rcept: str,
) -> None:
    """1000 shares growing to 1100, so a single 100-share issuance reconciles."""
    _insert_share_count(
        con,
        bsns_year=prior_year,
        reprt_code="11011",
        rcept_no=prior_rcept,
        istc_totqy=1000,
        now_to_isu=500,
        now_to_dcrs=0,
        stlm_dt=date(prior_year, 12, 31),
    )
    _insert_share_count(
        con,
        bsns_year=current_year,
        reprt_code="11011",
        rcept_no=current_rcept,
        istc_totqy=1100,
        now_to_isu=600,
        now_to_dcrs=0,
        stlm_dt=date(current_year, 12, 31),
    )


def test_same_event_in_two_vintages_is_counted_once() -> None:
    con = _base_con()
    _two_annual_positions(
        con,
        prior_year=2022,
        prior_rcept="20230310000001",
        current_year=2023,
        current_rcept="20240310000002",
    )
    # The FY2023 report and the FY2024 report both list the same 2023 issuance.
    for bsns_year, rcept_no in ((2023, "20240310000002"), (2024, "20250310000003")):
        _insert_capital_change(
            con,
            bsns_year=bsns_year,
            reprt_code="11011",
            rcept_no=rcept_no,
            isu_dcrs_de=date(2023, 6, 15),
            isu_dcrs_stle="유상증자(일반공모)",
            isu_dcrs_qy=100,
        )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    # Summing both copies would give 1000 + 200 != 1100 and NULL the feature.
    assert row[idx["issuance_identity_ok"]] is True
    assert row[idx["ev_net_share_issuance_yoy"]] == 100 / 1000


def test_quarterly_placeholder_vintage_does_not_blank_the_event_list() -> None:
    con = _base_con()
    _two_annual_positions(
        con,
        prior_year=2022,
        prior_rcept="20230310000001",
        current_year=2023,
        current_rcept="20240310000002",
    )
    _insert_capital_change(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        isu_dcrs_de=date(2023, 6, 15),
        isu_dcrs_stle="유상증자(일반공모)",
        isu_dcrs_qy=100,
    )
    # Quarterly reports come back as a single '-' placeholder carrying no
    # history. Newest-first selection must not pick one and see zero events.
    con.execute(
        "INSERT INTO dart_capital_change_raw VALUES "
        "('00126380','005930',2024,'11013','20240515000004',NULL,'-','',NULL)"
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["issuance_identity_ok"]] is True
    assert row[idx["ev_net_share_issuance_yoy"]] == 100 / 1000


def test_strict_pit_ignores_a_vintage_disclosed_after_the_position() -> None:
    con = _base_con()
    _two_annual_positions(
        con,
        prior_year=2021,
        prior_rcept="20220310000001",
        current_year=2022,
        current_rcept="20230310000002",
    )
    # The 2022 issuance is only listed by the FY2023 report, filed 2024-03-10 --
    # a year after the FY2022 position it would be used for.
    _insert_capital_change(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000003",
        isu_dcrs_de=date(2022, 6, 15),
        isu_dcrs_stle="유상증자(일반공모)",
        isu_dcrs_qy=100,
    )
    _insert_pit(con, trade_date=date(2023, 3, 15), market_cap=1_000_000_000)

    _register(con, vintage_policy="latest_vintage")
    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2023, 3, 15))
    assert row[idx["ev_net_share_issuance_yoy"]] == 100 / 1000

    _register(con, vintage_policy="strict_pit")
    row = _row(con, date(2023, 3, 15))
    assert row[idx["issuance_identity_ok"]] is False
    assert row[idx["ev_net_share_issuance_yoy"]] is None


def test_strict_pit_reads_the_figure_the_position_could_actually_see() -> None:
    con = _base_con()
    _two_annual_positions(
        con,
        prior_year=2021,
        prior_rcept="20220310000001",
        current_year=2022,
        current_rcept="20230310000002",
    )
    # Same event, quantity later corrected 100 -> 150 by the FY2023 report.
    _insert_capital_change(
        con,
        bsns_year=2022,
        reprt_code="11011",
        rcept_no="20230310000002",
        isu_dcrs_de=date(2022, 6, 15),
        isu_dcrs_stle="유상증자(일반공모)",
        isu_dcrs_qy=100,
    )
    _insert_capital_change(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000003",
        isu_dcrs_de=date(2022, 6, 15),
        isu_dcrs_stle="유상증자(일반공모)",
        isu_dcrs_qy=150,
    )
    _insert_pit(con, trade_date=date(2023, 3, 15), market_cap=1_000_000_000)

    # strict_pit reads the FY2022 report the position could see: 1000+100=1100.
    _register(con, vintage_policy="strict_pit")
    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2023, 3, 15))
    assert row[idx["issuance_identity_ok"]] is True
    assert row[idx["ev_net_share_issuance_yoy"]] == 100 / 1000

    # latest_vintage applies the later correction to the older share counts,
    # so 1000+150 != 1100 and the identity guard blanks the feature.
    _register(con, vintage_policy="latest_vintage")
    row = _row(con, date(2023, 3, 15))
    assert row[idx["issuance_identity_ok"]] is False
    assert row[idx["ev_net_share_issuance_yoy"]] is None


def test_unknown_vintage_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown vintage_policy"):
        build_issuance_sql(vintage_policy="whatever")


# --- §4.4 isu_dcrs_stle catalog (v2) --------------------------------------


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        # v2 additions. Each is a reason the collected history actually returns
        # that v1 had no entry for, so every one of these used to NULL its
        # whole trailing-year window.
        ("신주인수권행사", 100 / 1000),
        ("유상증자(주주우선공모)", 100 / 1000),
        ("출자전환", 100 / 1000),
        # A reason that stays outside the catalog must still block the window.
        ("기타", None),
    ],
)
def test_v2_catalog_classifies_the_reasons_the_source_actually_returns(
    reason: str, expected: float | None
) -> None:
    con = _base_con()
    _two_annual_positions(
        con,
        prior_year=2022,
        prior_rcept="20230310000001",
        current_year=2023,
        current_rcept="20240310000002",
    )
    _insert_capital_change(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        isu_dcrs_de=date(2023, 6, 15),
        isu_dcrs_stle=reason,
        isu_dcrs_qy=100,
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["ev_net_share_issuance_yoy"]] == expected


def test_musang_gamja_is_mechanical_under_either_spelling() -> None:
    # 감자(무상) and 무상감자 are the same action; both must reconcile the
    # identity without counting as economic issuance.
    for reason in ("감자(무상)", "무상감자"):
        con = _base_con()
        _insert_share_count(
            con,
            bsns_year=2022,
            reprt_code="11011",
            rcept_no="20230310000001",
            istc_totqy=1000,
            now_to_isu=500,
            now_to_dcrs=0,
            stlm_dt=date(2022, 12, 31),
        )
        _insert_share_count(
            con,
            bsns_year=2023,
            reprt_code="11011",
            rcept_no="20240310000002",
            istc_totqy=900,
            now_to_isu=500,
            now_to_dcrs=100,
            stlm_dt=date(2023, 12, 31),
        )
        _insert_capital_change(
            con,
            bsns_year=2023,
            reprt_code="11011",
            rcept_no="20240310000002",
            isu_dcrs_de=date(2023, 6, 15),
            isu_dcrs_stle=reason,
            isu_dcrs_qy=100,
        )
        _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
        _register(con)

        idx = {c: i for i, c in enumerate(_columns(con))}
        row = _row(con, date(2024, 3, 15))
        assert row[idx["issuance_identity_ok"]] is True, reason
        assert row[idx["ev_net_share_issuance_yoy"]] == 0.0, reason


def test_paid_capital_reduction_makes_net_issuance_negative() -> None:
    """Regression (issuance_v3): 유상감자 must count as an economic decrease.

    The catalog only held ``감자(유상)``, which this source never writes — it
    writes ``유상감자``. Until v3 the economic-decrease set matched nothing, so
    the feature could not go negative and every window holding one of those 699
    rows was dropped as unclassified. See 10_known_issues.md I3.
    """
    con = _base_con()
    _insert_share_count(
        con,
        bsns_year=2022,
        reprt_code="11011",
        rcept_no="20230310000001",
        istc_totqy=1000,
        now_to_isu=0,
        now_to_dcrs=0,
        stlm_dt=date(2022, 12, 31),
    )
    _insert_share_count(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        istc_totqy=900,
        now_to_isu=0,
        now_to_dcrs=100,
        stlm_dt=date(2023, 12, 31),
    )
    _insert_capital_change(
        con,
        bsns_year=2023,
        reprt_code="11011",
        rcept_no="20240310000002",
        isu_dcrs_de=date(2023, 6, 15),
        isu_dcrs_stle="유상감자",
        isu_dcrs_qy=100,
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["issuance_classification_complete"]] is True
    assert row[idx["issuance_identity_ok"]] is True
    assert row[idx["ev_net_share_issuance_yoy"]] == -100 / 1000


def test_dividend_total_with_unit_slip_falls_back_to_dps_proxy() -> None:
    """Regression (payout_v2): a 원-denominated total must not be scaled by 1e6.

    Some filers write the raw won amount into the 백만원-labelled field
    (일양약품 2019). DPS x (issued - treasury) from the same filing catches the
    1e6 gap. See 10_known_issues.md I5.
    """
    con = _base_con()
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="현금배당금총액(백만원)",
        value_numeric=450_000,  # already in won: DPS 500 x 900 shares
    )
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="주당 현금배당금(원)",
        value_numeric=500,
    )
    _insert_shares_instant(
        con, bsns_year=2023, rcept_no="20240310000002", issued=1000, treasury=100
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["dividend_source"]] == "dps_proxy"
    assert row[idx["cash_dividends_total"]] == 500 * (1000 - 100)


def test_dividend_total_within_sanity_multiple_is_still_preferred() -> None:
    """The unit check must not reject ordinary disagreement with the DPS proxy.

    Share classes and timing make the two estimates differ by a small factor;
    only a ~1e6 gap is a unit slip.
    """
    con = _base_con()
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="현금배당금총액(백만원)",
        value_numeric=2,  # 2,000,000 won vs a 450,000 won DPS proxy -> 4.4x
    )
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="주당 현금배당금(원)",
        value_numeric=500,
    )
    _insert_shares_instant(
        con, bsns_year=2023, rcept_no="20240310000002", issued=1000, treasury=100
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["dividend_source"]] == "direct_total"
    assert row[idx["cash_dividends_total"]] == 2 * 1_000_000


def test_negative_dividend_total_is_rejected() -> None:
    """Regression (payout_v2): a negative cash dividend is not a quantity."""
    con = _base_con()
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="현금배당금총액(백만원)",
        value_numeric=-3414,
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["cash_dividends_total"]] is None
    assert row[idx["dividend_source"]] is None
    assert row[idx["ev_payout_yield"]] is None


def test_catalog_classes_stay_disjoint() -> None:
    sets = {
        "economic_increase": ECONOMIC_INCREASE_REASONS,
        "economic_decrease": ECONOMIC_DECREASE_REASONS,
        "mechanical_increase": MECHANICAL_INCREASE_REASONS,
        "mechanical_decrease": MECHANICAL_DECREASE_REASONS,
    }
    total = sum(len(s) for s in sets.values())
    union = set().union(*sets.values())
    assert len(union) == total, f"a reason appears in two classes: {sets}"


def test_impossible_payout_yield_is_rejected_without_a_dps_row() -> None:
    """payout_v2 backstop: a unit slip with no DPS row to check it against.

    Ticker 344860 filed 358,740,000 into the 백만원 field; with no DPS row the
    upstream cross-check has nothing to compare against, so the yield (1.8m%)
    has to be caught on magnitude alone. See 10_known_issues.md I5.
    """
    con = _base_con()
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="현금배당금총액(백만원)",
        value_numeric=1_000_000,  # -> 1e12 won against a 1e9 market cap
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["ev_payout_yield"]] is None


def test_negative_buyback_cash_is_dropped_not_summed() -> None:
    """payout_v2: a negative treasury acquisition must not drag payout negative.

    The dividend leg stays, so the row keeps a payout — it just loses the
    component the source could not state as a quantity.
    """
    con = _base_con()
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="현금배당금총액(백만원)",
        value_numeric=10,
    )
    for reprt_code, amount, rcept_no in [
        ("11013", -1, "20230410000001"),
        ("11012", -2, "20230810000001"),
        ("11014", -3, "20231110000001"),
        ("11011", -4, "20240310000002"),
    ]:
        con.execute(
            "INSERT INTO dart_financial_statement_raw "
            "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
            "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
            "('00126380','005930',2023,?,'CFS','CF','dart_AcquisitionOfTreasuryShares','',1,?,'KRW',?)",
            [reprt_code, amount, rcept_no],
        )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["buyback_cash_ttm"]] is None
    assert row[idx["ev_payout_yield"]] == pytest.approx(10 * 1_000_000 / 1_000_000_000)


def test_dps_proxy_uses_the_common_share_row_not_the_preferred_one() -> None:
    """payout_v3: stock_knd is not in the row key, so both classes compete.

    The proxy multiplies the winning per-share figure by all shares outstanding,
    so picking the preferred row prices common shares at the preferred dividend.
    Both rows come from one filing and tie on every other key, which is also
    what made the mart depend on scan order. See 10_known_issues.md I12.
    """
    con = _base_con()
    for stock_knd, dps in [("우선주", 900), ("보통주", 500)]:
        _insert_dividend_row(
            con,
            bsns_year=2023,
            rcept_no="20240310000001",
            row_name="주당 현금배당금(원)",
            value_numeric=dps,
            stock_knd=stock_knd,
        )
    _insert_shares_instant(
        con, bsns_year=2023, rcept_no="20240310000002", issued=1000, treasury=100
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["dividend_source"]] == "dps_proxy"
    assert row[idx["cash_dividends_total"]] == 500 * (1000 - 100)


def test_dividend_row_of_an_unknown_share_class_is_not_used() -> None:
    """A 종류주식 row cannot be matched to a share count, so it is dropped."""
    con = _base_con()
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="주당 현금배당금(원)",
        value_numeric=500,
        stock_knd="종류주식",
    )
    _insert_shares_instant(
        con, bsns_year=2023, rcept_no="20240310000002", issued=1000, treasury=100
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["cash_dividends_total"]] is None
    assert row[idx["ev_payout_yield"]] is None


def test_unsplit_dividend_row_is_treated_as_the_common_share() -> None:
    """A filer with one share class writes '-'; that figure is still usable."""
    con = _base_con()
    _insert_dividend_row(
        con,
        bsns_year=2023,
        rcept_no="20240310000001",
        row_name="주당 현금배당금(원)",
        value_numeric=500,
        stock_knd="-",
    )
    _insert_shares_instant(
        con, bsns_year=2023, rcept_no="20240310000002", issued=1000, treasury=100
    )
    _insert_pit(con, trade_date=date(2024, 3, 15), market_cap=1_000_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _row(con, date(2024, 3, 15))
    assert row[idx["cash_dividends_total"]] == 500 * (1000 - 100)
