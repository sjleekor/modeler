from __future__ import annotations

from datetime import date

import duckdb
from modeler.etl.features.fin_scan import FIN_SCAN_TABLE, register_fin_scan_daily_view
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
    con.execute(
        "CREATE TABLE dim_stock_pit_daily ("
        "trade_date DATE, ticker VARCHAR, market VARCHAR, market_cap_pit DOUBLE, "
        "issued_shares_pit DOUBLE, shares_is_available BOOLEAN, "
        "shares_invalid_flag BOOLEAN, shares_available_from DATE)"
    )
    con.execute(
        "CREATE TABLE dim_price_quality_daily ("
        "trade_date DATE, ticker VARCHAR, market VARCHAR, is_halted BOOLEAN, "
        "valid_session_idx BIGINT)"
    )
    return con


def _insert_corp(con: duckdb.DuckDBPyConnection, corp_code: str, ticker: str) -> None:
    con.execute("INSERT INTO dart_corp_master VALUES (?, 'KOSPI', ?)", [ticker, corp_code])


def _insert_fs(
    con: duckdb.DuckDBPyConnection,
    *,
    corp_code: str,
    ticker: str,
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
        "VALUES (?, ?, ?, ?, ?, ?, ?, '', 1, ?, 'KRW', ?, ?)",
        [
            corp_code,
            ticker,
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


def _insert_quarter_set(
    con: duckdb.DuckDBPyConnection,
    *,
    corp_code: str,
    ticker: str,
    account_id: str,
    q1: float,
    q2: float,
    q3: float,
    annual: float,
    sj_div: str = "IS",
    fs_div: str = "CFS",
    rcept_prefix: str = "2023",
) -> None:
    """A full Q1-Q4 direct-interim quarter set for one bsns_year=2023 metric,
    with self-consistent thstrm_add_amount cumulative cross-checks."""
    _insert_fs(
        con,
        corp_code=corp_code,
        ticker=ticker,
        bsns_year=2023,
        reprt_code="11013",
        account_id=account_id,
        thstrm_amount=q1,
        rcept_no=f"{rcept_prefix}0410000001",
        sj_div=sj_div,
        fs_div=fs_div,
    )
    _insert_fs(
        con,
        corp_code=corp_code,
        ticker=ticker,
        bsns_year=2023,
        reprt_code="11012",
        account_id=account_id,
        thstrm_amount=q2,
        thstrm_add_amount=q1 + q2,
        rcept_no=f"{rcept_prefix}0810000001",
        sj_div=sj_div,
        fs_div=fs_div,
    )
    _insert_fs(
        con,
        corp_code=corp_code,
        ticker=ticker,
        bsns_year=2023,
        reprt_code="11014",
        account_id=account_id,
        thstrm_amount=q3,
        thstrm_add_amount=q1 + q2 + q3,
        rcept_no=f"{rcept_prefix}1110000001",
        sj_div=sj_div,
        fs_div=fs_div,
    )
    _insert_fs(
        con,
        corp_code=corp_code,
        ticker=ticker,
        bsns_year=2023,
        reprt_code="11011",
        account_id=account_id,
        thstrm_amount=q1 + q2 + q3 + annual,
        rcept_no=f"{rcept_prefix}0311000001",
        sj_div=sj_div,
        fs_div=fs_div,
    )


def _insert_instant(
    con: duckdb.DuckDBPyConnection,
    *,
    corp_code: str,
    ticker: str,
    bsns_year: int,
    account_id: str,
    value: float,
    rcept_no: str,
    fs_div: str = "CFS",
) -> None:
    _insert_fs(
        con,
        corp_code=corp_code,
        ticker=ticker,
        bsns_year=bsns_year,
        reprt_code="11011",
        account_id=account_id,
        thstrm_amount=value,
        rcept_no=rcept_no,
        sj_div="BS",
        fs_div=fs_div,
    )


def _insert_pit(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str,
    trade_date: date,
    market_cap: float | None,
    issued_shares: float = 1_000_000,
    is_available: bool = True,
    invalid: bool = False,
) -> None:
    con.execute(
        "INSERT INTO dim_stock_pit_daily VALUES (?, ?, 'KOSPI', ?, ?, ?, ?, ?)",
        [trade_date, ticker, market_cap, issued_shares, is_available, invalid, trade_date],
    )
    con.execute(
        "INSERT INTO dim_price_quality_daily VALUES (?, ?, 'KOSPI', FALSE, 1)",
        [trade_date, ticker],
    )


def _register(con: duckdb.DuckDBPyConnection) -> None:
    register_stock_metric_vintage_fact_view(con, trading_days=_TRADING_DAYS)
    register_fin_quarterly_metric_vintage_view(con)
    register_fin_scan_daily_view(con)


def _feature_row(con: duckdb.DuckDBPyConnection, ticker: str, trade_date: date) -> tuple | None:
    return con.execute(
        f"SELECT * FROM {FIN_SCAN_TABLE} WHERE ticker=? AND trade_date=?",
        [ticker, trade_date],
    ).fetchone()


def _columns(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [row[0] for row in con.execute(f"DESCRIBE {FIN_SCAN_TABLE}").fetchall()]


def _setup_full_ticker(con: duckdb.DuckDBPyConnection, corp_code: str, ticker: str) -> None:
    _insert_corp(con, corp_code, ticker)
    _insert_quarter_set(
        con,
        corp_code=corp_code,
        ticker=ticker,
        account_id="ifrs-full_Revenue",
        q1=100,
        q2=150,
        q3=140,
        annual=170,
    )
    _insert_quarter_set(
        con,
        corp_code=corp_code,
        ticker=ticker,
        account_id="ifrs-full_ProfitLossAttributableToOwnersOfParent",
        q1=10,
        q2=15,
        q3=14,
        annual=17,
    )
    _insert_quarter_set(
        con,
        corp_code=corp_code,
        ticker=ticker,
        account_id="ifrs-full_ProfitLoss",
        q1=10,
        q2=15,
        q3=14,
        annual=17,
    )
    _insert_quarter_set(
        con,
        corp_code=corp_code,
        ticker=ticker,
        account_id="ifrs-full_CashFlowsFromUsedInOperatingActivities",
        q1=20,
        q2=15,
        q3=15,
        annual=20,
        sj_div="CF",
    )
    _insert_quarter_set(
        con,
        corp_code=corp_code,
        ticker=ticker,
        account_id="ifrs-full_GrossProfit",
        q1=40,
        q2=50,
        q3=45,
        annual=55,
    )
    _insert_instant(
        con,
        corp_code=corp_code,
        ticker=ticker,
        bsns_year=2022,
        account_id="ifrs-full_Assets",
        value=900,
        rcept_no="20230310000001",
    )
    _insert_instant(
        con,
        corp_code=corp_code,
        ticker=ticker,
        bsns_year=2023,
        account_id="ifrs-full_Assets",
        value=1000,
        rcept_no="20240310000002",
    )
    _insert_instant(
        con,
        corp_code=corp_code,
        ticker=ticker,
        bsns_year=2023,
        account_id="ifrs-full_Equity",
        value=500,
        rcept_no="20240310000002",
    )


def test_full_feature_row_computes_after_annual_filing_is_available() -> None:
    con = _base_con()
    _setup_full_ticker(con, "00126380", "005930")
    _insert_pit(con, ticker="005930", trade_date=date(2024, 3, 8), market_cap=500_000_000)
    _insert_pit(con, ticker="005930", trade_date=date(2024, 3, 15), market_cap=500_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}

    before = _feature_row(con, "005930", date(2024, 3, 8))
    assert before[idx["fin_gross_profitability"]] is None  # annual filing not yet available

    row = _feature_row(con, "005930", date(2024, 3, 15))
    assert row is not None
    assert row[idx["fs_basis_used"]] == "CFS"
    assert row[idx["negative_equity"]] is False

    market_cap = 500_000_000
    expected_bm = 500 / market_cap
    controlling_net_income_ttm = 10 + 15 + 14 + 17
    assert row[idx["fin_book_to_market"]] == expected_bm
    assert row[idx["fin_earnings_yield"]] == controlling_net_income_ttm / market_cap
    assert row[idx["fin_sales_to_price"]] == 560 / market_cap  # revenue TTM 100+150+140+170

    avg_assets = (1000 + 900) / 2
    gross_profit_ttm = 40 + 50 + 45 + 55
    assert row[idx["fin_gross_profitability"]] == gross_profit_ttm / avg_assets
    assert row[idx["gross_profit_source"]] == "direct"
    assert row[idx["fin_asset_growth_yoy"]] == 1000 / 900 - 1

    net_income_ttm = 10 + 15 + 14 + 17
    cfo_ttm = 20 + 15 + 15 + 20
    assert row[idx["fin_accruals_to_assets"]] == (net_income_ttm - cfo_ttm) / avg_assets
    assert row[idx["fin_cfo_yield"]] == cfo_ttm / market_cap


def test_negative_equity_nulls_book_to_market_only() -> None:
    con = _base_con()
    _setup_full_ticker(con, "00126380", "005930")
    # Overwrite equity with a negative value (later rcept_no wins B-3's tie-break
    # only if it's earliest/non-revision; use the same rcept_no+value instead).
    con.execute(
        "UPDATE dart_financial_statement_raw SET thstrm_amount = -200 "
        "WHERE account_id = 'ifrs-full_Equity'"
    )
    _insert_pit(con, ticker="005930", trade_date=date(2024, 3, 15), market_cap=500_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row = _feature_row(con, "005930", date(2024, 3, 15))
    assert row[idx["negative_equity"]] is True
    assert row[idx["fin_book_to_market"]] is None
    # Other ratios are unaffected by negative equity.
    assert row[idx["fin_earnings_yield"]] is not None


def test_invalid_shares_nulls_size_and_value_but_not_profitability() -> None:
    con = _base_con()
    _setup_full_ticker(con, "00126380", "005930")
    _insert_pit(
        con,
        ticker="005930",
        trade_date=date(2024, 3, 15),
        market_cap=None,
        invalid=True,
    )
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row = _feature_row(con, "005930", date(2024, 3, 15))
    assert row[idx["fin_log_mcap"]] is None
    assert row[idx["fin_book_to_market"]] is None
    assert row[idx["fin_earnings_yield"]] is None
    # Profitability/asset-growth/accruals never touch market cap or shares.
    assert row[idx["fin_gross_profitability"]] is not None
    assert row[idx["fin_asset_growth_yoy"]] is not None


def test_shared_fs_basis_never_mixes_cfs_and_ofs_for_one_ticker_date() -> None:
    con = _base_con()
    _insert_corp(con, "00126380", "005930")
    _insert_quarter_set(
        con,
        corp_code="00126380",
        ticker="005930",
        account_id="ifrs-full_ProfitLoss",
        q1=10,
        q2=15,
        q3=14,
        annual=17,
        fs_div="CFS",
    )
    # Equity exists under BOTH bases with different values; CFS must win
    # because net_income (the basis anchor) only exists under CFS.
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
        "('00126380','005930',2023,'11011','CFS','BS','ifrs-full_Equity','',1,"
        "500,'KRW','20240310000002'),"
        "('00126380','005930',2023,'11011','OFS','BS','ifrs-full_Equity','',1,"
        "999,'KRW','20240310000003')"
    )
    _insert_pit(con, ticker="005930", trade_date=date(2024, 3, 15), market_cap=500_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row = _feature_row(con, "005930", date(2024, 3, 15))
    assert row[idx["fs_basis_used"]] == "CFS"
    assert row[idx["fin_book_to_market"]] == 500 / 500_000_000


def test_fin_value_z_requires_at_least_two_components_and_cross_sections() -> None:
    con = _base_con()
    _setup_full_ticker(con, "00126380", "005930")
    _setup_full_ticker(con, "00164779", "000660")
    trade_date = date(2024, 3, 15)
    _insert_pit(con, ticker="005930", trade_date=trade_date, market_cap=500_000_000)
    _insert_pit(con, ticker="000660", trade_date=trade_date, market_cap=250_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row_a = _feature_row(con, "005930", trade_date)
    row_b = _feature_row(con, "000660", trade_date)
    assert row_a[idx["value_component_count"]] == 4
    assert row_a[idx["fin_value_z"]] is not None
    assert row_b[idx["fin_value_z"]] is not None
    # Two names cross-sectionally z-scored around 0 -> they are not both equal
    # unless their raw ratios are identical (different market caps here).
    assert row_a[idx["fin_value_z"]] != row_b[idx["fin_value_z"]]


def test_missing_components_stay_null_through_winsorize() -> None:
    """Regression (fin_v2): a name with no financials must not get a value_z.

    DuckDB's GREATEST/LEAST skip NULL arguments, so an unguarded
    ``LEAST(GREATEST(NULL, p01), p99)`` returns p01 — every missing component
    would be imputed to the market's 1st percentile, counted as valid, and the
    name pinned to the "most expensive" end of the cross-section. See
    docs/dev/20260731_raw_features/01_feature_candidate/10_known_issues.md I1.
    """
    con = _base_con()
    # Two fully-populated names give the cross-section its 1/99% bounds...
    _setup_full_ticker(con, "00126380", "005930")
    _setup_full_ticker(con, "00164779", "000660")
    trade_date = date(2024, 3, 15)
    _insert_pit(con, ticker="005930", trade_date=trade_date, market_cap=500_000_000)
    _insert_pit(con, ticker="000660", trade_date=trade_date, market_cap=250_000_000)
    # ...and this one is priced but has no financial statements at all.
    _insert_pit(con, ticker="900000", trade_date=trade_date, market_cap=100_000_000)
    _register(con)

    idx = {c: i for i, c in enumerate(_columns(con))}
    row = _feature_row(con, "900000", trade_date)
    assert row is not None, "a priced name should still get a row"
    for component in (
        "fin_book_to_market",
        "fin_earnings_yield",
        "fin_cfo_yield",
        "fin_sales_to_price",
    ):
        assert row[idx[component]] is None
    assert row[idx["value_component_count"]] == 0
    assert row[idx["fin_value_z"]] is None


def test_lag1_variant_matches_prior_valid_session() -> None:
    con = _base_con()
    _setup_full_ticker(con, "00126380", "005930")
    _insert_pit(con, ticker="005930", trade_date=date(2024, 3, 15), market_cap=500_000_000)
    _insert_pit(con, ticker="005930", trade_date=date(2024, 3, 18), market_cap=600_000_000)
    _register(con)

    cols = _columns(con)
    idx = {c: i for i, c in enumerate(cols)}
    row_prior = _feature_row(con, "005930", date(2024, 3, 15))
    row_next = _feature_row(con, "005930", date(2024, 3, 18))
    assert row_next[idx["fin_log_mcap_lag1"]] == row_prior[idx["fin_log_mcap"]]
    assert (
        row_next[idx["fin_gross_profitability_lag1"]] == row_prior[idx["fin_gross_profitability"]]
    )


def test_output_grain_is_unique_per_trade_date_ticker_market() -> None:
    con = _base_con()
    _setup_full_ticker(con, "00126380", "005930")
    _insert_pit(con, ticker="005930", trade_date=date(2024, 3, 15), market_cap=500_000_000)
    _register(con)

    dupes = con.execute(
        f"SELECT trade_date, ticker, market, COUNT(*) "
        f"FROM {FIN_SCAN_TABLE} GROUP BY 1, 2, 3 HAVING COUNT(*) > 1"
    ).fetchall()
    assert dupes == []


def test_no_feature_is_available_before_its_own_available_from() -> None:
    con = _base_con()
    _setup_full_ticker(con, "00126380", "005930")
    _insert_pit(con, ticker="005930", trade_date=date(2024, 3, 15), market_cap=500_000_000)
    _register(con)

    rows = con.execute(
        f"SELECT value_available_from, profitability_available_from, "
        f"asset_growth_available_from, accruals_available_from, trade_date "
        f"FROM {FIN_SCAN_TABLE} WHERE ticker='005930' AND trade_date=?",
        [date(2024, 3, 15)],
    ).fetchone()
    trade_date = rows[-1]
    for available_from in rows[:-1]:
        assert available_from is None or available_from <= trade_date
