from __future__ import annotations

from datetime import date

import duckdb
from modeler.etl.marts.financial_quarters import (
    FQMV_TABLE,
    register_fin_quarterly_metric_vintage_view,
)
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
        "rcept_no VARCHAR, se VARCHAR, istc_totqy BIGINT, tesstk_co BIGINT, stlm_dt DATE)"
    )
    con.execute(
        "CREATE TABLE dart_shareholder_return_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, statement_type VARCHAR, row_name VARCHAR, stock_knd VARCHAR, "
        "dim1 VARCHAR, dim2 VARCHAR, dim3 VARCHAR, metric_code VARCHAR, "
        "value_numeric DECIMAL(30,4), stlm_dt DATE)"
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
    con.execute("INSERT INTO dart_corp_master VALUES ('005930', 'KOSPI', '00126380')")
    return con


def _insert_fs_row(
    con: duckdb.DuckDBPyConnection,
    *,
    bsns_year: int,
    reprt_code: str,
    account_id: str,
    thstrm_amount: float,
    rcept_no: str,
    thstrm_add_amount: float | None = None,
    sj_div: str = "IS",
    fs_div: str = "CFS",
) -> None:
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no, thstrm_add_amount) "
        "VALUES ('00126380', '005930', ?, ?, ?, ?, ?, '', 1, ?, 'KRW', ?, ?)",
        [
            bsns_year,
            reprt_code,
            fs_div,
            sj_div,
            account_id,
            thstrm_amount,
            rcept_no,
            thstrm_add_amount,
        ],
    )


def _register(con: duckdb.DuckDBPyConnection) -> None:
    register_stock_metric_vintage_fact_view(con, trading_days=_TRADING_DAYS)
    register_fin_quarterly_metric_vintage_view(con)


def _revenue_row(bsns_year: int, quarter: str, con: duckdb.DuckDBPyConnection) -> tuple | None:
    return con.execute(
        f"SELECT standalone_value, standalone_source_conflict, ttm_value, ttm_complete "
        f"FROM {FQMV_TABLE} WHERE metric_code='revenue' AND bsns_year=? AND quarter=?",
        [bsns_year, quarter],
    ).fetchone()


def test_direct_interim_standalone_matches_doc_worked_example() -> None:
    """04_specific_plan_B.md §6 B-3 test list: Q1=100, half=250(cumulative),
    q3=390(cumulative), annual=560 -> standalone 100/150/140/170."""
    con = _base_con()
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11013",
        account_id="ifrs-full_Revenue",
        thstrm_amount=100,
        rcept_no="20230410000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11012",
        account_id="ifrs-full_Revenue",
        thstrm_amount=150,
        thstrm_add_amount=250,
        rcept_no="20230810000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11014",
        account_id="ifrs-full_Revenue",
        thstrm_amount=140,
        thstrm_add_amount=390,
        rcept_no="20231110000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11011",
        account_id="ifrs-full_Revenue",
        thstrm_amount=560,
        rcept_no="20240310000001",
    )
    _register(con)

    for quarter, expected in [("Q1", 100), ("Q2", 150), ("Q3", 140), ("Q4", 170)]:
        row = _revenue_row(2023, quarter, con)
        assert row[0] == expected, f"{quarter}: {row}"
        assert row[1] in (None, False), f"{quarter} conflict: {row}"


def test_standalone_source_conflict_excludes_value() -> None:
    con = _base_con()
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11013",
        account_id="ifrs-full_Revenue",
        thstrm_amount=100,
        rcept_no="20230410000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11012",
        account_id="ifrs-full_Revenue",
        thstrm_amount=150,
        thstrm_add_amount=999,
        rcept_no="20230810000001",
    )
    _register(con)

    row = _revenue_row(2023, "Q2", con)
    assert row[0] is None
    assert row[1] is True


def test_missing_interim_quarter_blocks_q4_computation() -> None:
    con = _base_con()
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11013",
        account_id="ifrs-full_Revenue",
        thstrm_amount=100,
        rcept_no="20230410000001",
    )
    # Q2 (half) never filed.
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11014",
        account_id="ifrs-full_Revenue",
        thstrm_amount=140,
        rcept_no="20231110000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11011",
        account_id="ifrs-full_Revenue",
        thstrm_amount=560,
        rcept_no="20240310000001",
    )
    _register(con)

    assert _revenue_row(2023, "Q4", con) is None


def test_cumulative_reported_metric_differencing() -> None:
    con = _base_con()
    for reprt_code, amount, rcept_no in [
        ("11013", 500, "20230410000001"),
        ("11012", 900, "20230810000001"),
        ("11014", 1250, "20231110000001"),
        ("11011", 1600, "20240310000001"),
    ]:
        _insert_fs_row(
            con,
            bsns_year=2023,
            reprt_code=reprt_code,
            account_id="ifrs-full_CashFlowsFromUsedInOperatingActivities",
            thstrm_amount=amount,
            rcept_no=rcept_no,
            sj_div="CF",
        )
    _register(con)

    rows = con.execute(
        f"SELECT quarter, standalone_value FROM {FQMV_TABLE} "
        f"WHERE metric_code='operating_cash_flow' AND bsns_year=2023 ORDER BY quarter"
    ).fetchall()
    assert dict(rows) == {"Q1": 500.0, "Q2": 400.0, "Q3": 350.0, "Q4": 350.0}


def test_ttm_identity_for_four_contiguous_quarters() -> None:
    con = _base_con()
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11013",
        account_id="ifrs-full_Revenue",
        thstrm_amount=100,
        rcept_no="20230410000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11012",
        account_id="ifrs-full_Revenue",
        thstrm_amount=150,
        thstrm_add_amount=250,
        rcept_no="20230810000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11014",
        account_id="ifrs-full_Revenue",
        thstrm_amount=140,
        thstrm_add_amount=390,
        rcept_no="20231110000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11011",
        account_id="ifrs-full_Revenue",
        thstrm_amount=560,
        rcept_no="20240310000001",
    )
    _register(con)

    row = _revenue_row(2023, "Q4", con)
    assert row[2] == 100 + 150 + 140 + 170
    assert row[3] is True


def test_ttm_null_when_a_quarter_is_missing() -> None:
    con = _base_con()
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11013",
        account_id="ifrs-full_Revenue",
        thstrm_amount=100,
        rcept_no="20230410000001",
    )
    # Q2 missing entirely.
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11014",
        account_id="ifrs-full_Revenue",
        thstrm_amount=140,
        rcept_no="20231110000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11011",
        account_id="ifrs-full_Revenue",
        thstrm_amount=560,
        rcept_no="20240310000001",
    )
    _register(con)

    row = _revenue_row(2023, "Q3", con)
    assert row[0] == 140
    assert row[2] is None  # TTM at Q3 needs Q3,Q2,Q1,Q4(prior year) — Q2 missing


def test_cfs_ofs_mixed_quarters_never_produce_a_ttm() -> None:
    con = _base_con()
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11013",
        account_id="ifrs-full_Revenue",
        thstrm_amount=100,
        rcept_no="20230410000001",
        fs_div="CFS",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11012",
        account_id="ifrs-full_Revenue",
        thstrm_amount=150,
        thstrm_add_amount=250,
        rcept_no="20230810000001",
        fs_div="CFS",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11014",
        account_id="ifrs-full_Revenue",
        thstrm_amount=140,
        thstrm_add_amount=390,
        rcept_no="20231110000001",
        fs_div="CFS",
    )
    # Q4 (annual) only exists under OFS -> fs_basis partition breaks contiguity.
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11011",
        account_id="ifrs-full_Revenue",
        thstrm_amount=560,
        rcept_no="20240310000001",
        fs_div="OFS",
    )
    _register(con)

    cfs_rows = con.execute(
        f"SELECT quarter, ttm_value FROM {FQMV_TABLE} "
        f"WHERE metric_code='revenue' AND fs_basis='CFS' AND bsns_year=2023"
    ).fetchall()
    assert all(ttm is None for _q, ttm in cfs_rows)
    ofs_rows = con.execute(
        f"SELECT quarter, standalone_value FROM {FQMV_TABLE} "
        f"WHERE metric_code='revenue' AND fs_basis='OFS' AND bsns_year=2023"
    ).fetchall()
    # Q4 alone can't be computed without Q1-Q3 -> no row at all (same
    # convention as test_missing_interim_quarter_blocks_q4_computation).
    assert ofs_rows == []


def test_weighted_average_shares_reconstruction() -> None:
    con = _base_con()
    # Each filing's context is its own year-to-date duration — a Q1 report
    # cannot carry a Jan-Dec context, and B-2 now refuses to read one as this
    # filing's period (see test_metric_vintages.py, 08 §4.3.2).
    for reprt_code, amount, rcept_no, period_end in [
        ("11013", 1000, "20230410000001", "2023-03-31"),
        ("11012", 1050, "20230810000001", "2023-06-30"),
        ("11014", 1080, "20231110000001", "2023-09-30"),
        ("11011", 1100, "20240310000001", "2023-12-31"),
    ]:
        con.execute(
            "INSERT INTO dart_xbrl_fact_raw VALUES "
            "('00126380','005930',2023,?,?,"
            "'ifrs-full_WeightedAverageNumberOfOrdinarySharesOutstandingBasic','가중평균주식수',"
            "'ctx1','duration',DATE '2023-01-01',?::DATE,NULL,'[]',?,'가중평균주식수')",
            [reprt_code, rcept_no, period_end, amount],
        )
    _register(con)

    rows = con.execute(
        f"SELECT quarter, standalone_value, ttm_value FROM {FQMV_TABLE} "
        f"WHERE metric_code='weighted_avg_shares' AND bsns_year=2023 ORDER BY quarter"
    ).fetchall()
    by_quarter = {q: v for q, v, _ttm in rows}
    assert by_quarter["Q1"] == 1000
    assert by_quarter["Q2"] == 2 * 1050 - 1000
    assert by_quarter["Q3"] == 3 * 1080 - 2 * 1050
    assert by_quarter["Q4"] == 4 * 1100 - 3 * 1080
    # Weighted-share averages are never TTM-summed.
    assert all(ttm is None for _q, _v, ttm in rows)


def test_instant_metric_passthrough_and_value_lag_4q() -> None:
    con = _base_con()
    # Same quarter position (annual, 11011) two years running.
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
        "('00126380','005930',2022,'11011','CFS','BS','ifrs-full_Assets','',1,"
        "9000,'KRW','20230310000001'),"
        "('00126380','005930',2023,'11011','CFS','BS','ifrs-full_Assets','',1,"
        "10000,'KRW','20240310000001')"
    )
    _register(con)

    row_2023 = con.execute(
        f"SELECT standalone_value, value_lag_4q FROM {FQMV_TABLE} "
        f"WHERE metric_code='total_assets' AND bsns_year=2023 AND quarter='Q4'"
    ).fetchone()
    assert row_2023 == (10000, 9000)


def test_non_december_fiscal_year_end_does_not_break_quarter_ordering() -> None:
    """quarter_ordinal is driven by (bsns_year, reprt_code), not calendar month
    — a March fiscal year-end company must still get standalone 100/150/140/170."""
    con = _base_con()
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11013",
        account_id="ifrs-full_Revenue",
        thstrm_amount=100,
        rcept_no="20230710000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11012",
        account_id="ifrs-full_Revenue",
        thstrm_amount=150,
        thstrm_add_amount=250,
        rcept_no="20231010000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11014",
        account_id="ifrs-full_Revenue",
        thstrm_amount=140,
        thstrm_add_amount=390,
        rcept_no="20240110000001",
    )
    _insert_fs_row(
        con,
        bsns_year=2023,
        reprt_code="11011",
        account_id="ifrs-full_Revenue",
        thstrm_amount=560,
        rcept_no="20240610000001",
    )
    _register(con)

    for quarter, expected in [("Q1", 100), ("Q2", 150), ("Q3", 140), ("Q4", 170)]:
        row = _revenue_row(2023, quarter, con)
        assert row[0] == expected, f"{quarter}: {row}"
