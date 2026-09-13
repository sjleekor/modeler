from __future__ import annotations

import statistics
from datetime import date

import duckdb

from modeler.etl.features.sue_event import SUE_EVENT_TABLE, register_sue_event_view
from modeler.etl.marts.financial_quarters import register_fin_quarterly_metric_vintage_view
from modeler.etl.marts.metric_vintages import register_stock_metric_vintage_fact_view

CORP_CODE = "00126380"
TICKER = "005930"

_NI_CONCEPT = "ifrs-full_ProfitLossAttributableToOwnersOfParent"
_SHARES_CONCEPT = "ifrs-full_WeightedAverageNumberOfOrdinarySharesOutstandingBasic"


def _weekdays_from(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current = date.fromordinal(current.toordinal() + 1)
    return days


# A long, uninterrupted weekday calendar spanning every rcept_no/close-price
# date used below (2020-01-01 .. ~2025-01-01).
_TRADING_DAYS = _weekdays_from(date(2020, 1, 1), 1400)


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
        "CREATE TABLE daily_ohlcv (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "open BIGINT, high BIGINT, low BIGINT, close BIGINT, volume BIGINT)"
    )
    con.execute(
        "CREATE TABLE dim_price_quality_daily ("
        "trade_date DATE, ticker VARCHAR, market VARCHAR, ca_mask BOOLEAN)"
    )
    con.execute(f"INSERT INTO dart_corp_master VALUES ('{TICKER}', 'KOSPI', '{CORP_CODE}')")
    return con


def _insert_ni(
    con: duckdb.DuckDBPyConnection,
    *,
    bsns_year: int,
    reprt_code: str,
    rcept_no: str,
    thstrm_amount: float,
    report_nm: str | None = "사업보고서",
) -> None:
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
        f"('{CORP_CODE}','{TICKER}',?,?,'CFS','CIS','{_NI_CONCEPT}','',1,?,'KRW',?)",
        [bsns_year, reprt_code, thstrm_amount, rcept_no],
    )
    if report_nm is not None:
        con.execute(
            "INSERT INTO dart_filing_receipt_raw VALUES (?, ?, ?)",
            [CORP_CODE, rcept_no, report_nm],
        )


def _insert_shares(
    con: duckdb.DuckDBPyConnection,
    *,
    bsns_year: int,
    reprt_code: str,
    rcept_no: str,
    cumulative_average: float,
) -> None:
    con.execute(
        "INSERT INTO dart_xbrl_fact_raw VALUES "
        f"('{CORP_CODE}','{TICKER}',?,?,?,'{_SHARES_CONCEPT}','가중평균주식수',"
        "'ctx1','duration',DATE '2000-01-01',DATE '2000-01-01',NULL,'[]',?,'가중평균주식수')",
        [bsns_year, reprt_code, rcept_no, cumulative_average],
    )


def _insert_close(con: duckdb.DuckDBPyConnection, trade_date: date, close: int) -> None:
    con.execute(
        f"INSERT INTO daily_ohlcv VALUES (?, '{TICKER}', 'KOSPI', ?, ?, ?, ?, 1000)",
        [trade_date, close, close, close, close],
    )


def _register(con: duckdb.DuckDBPyConnection) -> None:
    register_stock_metric_vintage_fact_view(con, trading_days=_TRADING_DAYS)
    register_fin_quarterly_metric_vintage_view(con)
    register_sue_event_view(con)


def _insert_full_year(
    con: duckdb.DuckDBPyConnection,
    *,
    bsns_year: int,
    q1: float,
    q2: float,
    q3: float,
    q4: float,
    rcept_prefix: str,
) -> None:
    """One full Q1-Q4 year of controlling_net_income + constant-1000 shares."""
    _insert_ni(
        con,
        bsns_year=bsns_year,
        reprt_code="11013",
        rcept_no=f"{rcept_prefix}0410000001",
        thstrm_amount=q1,
    )
    _insert_ni(
        con,
        bsns_year=bsns_year,
        reprt_code="11012",
        rcept_no=f"{rcept_prefix}0810000001",
        thstrm_amount=q2,
    )
    _insert_ni(
        con,
        bsns_year=bsns_year,
        reprt_code="11014",
        rcept_no=f"{rcept_prefix}1110000001",
        thstrm_amount=q3,
    )
    _insert_ni(
        con,
        bsns_year=bsns_year,
        reprt_code="11011",
        rcept_no=f"{rcept_prefix}0311000001",
        thstrm_amount=q1 + q2 + q3 + q4,
    )
    for reprt_code, rcept_no in [
        ("11013", f"{rcept_prefix}0410000001"),
        ("11012", f"{rcept_prefix}0810000001"),
        ("11014", f"{rcept_prefix}1110000001"),
        ("11011", f"{rcept_prefix}0311000001"),
    ]:
        _insert_shares(
            con,
            bsns_year=bsns_year,
            reprt_code=reprt_code,
            rcept_no=rcept_no,
            cumulative_average=1000,
        )


def test_sue_uses_exactly_8_trailing_original_events_and_matches_python_stddev() -> None:
    con = _base_con()
    # Y1 (2020): base year, no seasonal_change of its own in this fixture.
    _insert_full_year(con, bsns_year=2020, q1=100, q2=100, q3=100, q4=100, rcept_prefix="2020")
    # Y2 (2021): seasonal changes vs Y1 = +10, -10, +20, -5
    _insert_full_year(con, bsns_year=2021, q1=110, q2=90, q3=120, q4=95, rcept_prefix="2021")
    # Y3 (2022): seasonal changes vs Y2 = +5, -10, +20, -5
    _insert_full_year(con, bsns_year=2022, q1=115, q2=80, q3=140, q4=90, rcept_prefix="2022")
    # Y4 Q1 only (2023): the event under test, seasonal change vs Y3 Q1 = +5
    _insert_ni(
        con, bsns_year=2023, reprt_code="11013", rcept_no="20230410000001", thstrm_amount=120
    )
    _insert_shares(
        con, bsns_year=2023, reprt_code="11013", rcept_no="20230410000001", cumulative_average=1000
    )

    # Enough close prices around the 2023-Q1 event's formation for all 6 buckets.
    formation = date(2023, 4, 11)
    trading_days_from_formation = [d for d in _TRADING_DAYS if d >= formation][:65]
    for i, d in enumerate(trading_days_from_formation):
        _insert_close(con, d, 100 + i)

    _register(con)

    row = con.execute(
        f"SELECT sue_history_count, seasonal_change, fin_sue "
        f"FROM {SUE_EVENT_TABLE} WHERE bsns_year=2023 AND reprt_code='11013'"
    ).fetchone()
    assert row is not None
    history_count, seasonal_change, fin_sue = row

    trailing_seasonal_changes = [10, -10, 20, -5, 5, -10, 20, -5]  # Y2 Q1-4, Y3 Q1-4
    assert history_count == 8
    # seasonal_change is an EPS (per-share) difference, not a raw net-income
    # difference -- shares are constant at 1000, so (120-115)/1000 = 0.005.
    assert abs(seasonal_change - 5 / 1000) < 1e-9
    expected_stddev = statistics.stdev([v / 1000 for v in trailing_seasonal_changes])
    expected_sue = (5 / 1000) / expected_stddev
    assert abs(fin_sue - expected_sue) < 1e-9


def test_history_below_8_leaves_sue_null_but_seasonal_change_present() -> None:
    con = _base_con()
    _insert_full_year(con, bsns_year=2020, q1=100, q2=100, q3=100, q4=100, rcept_prefix="2020")
    _insert_ni(
        con, bsns_year=2021, reprt_code="11013", rcept_no="20210410000001", thstrm_amount=110
    )
    _insert_shares(
        con, bsns_year=2021, reprt_code="11013", rcept_no="20210410000001", cumulative_average=1000
    )
    formation = date(2021, 4, 12)
    for i, d in enumerate([d for d in _TRADING_DAYS if d >= formation][:65]):
        _insert_close(con, d, 100 + i)
    _register(con)

    row = con.execute(
        f"SELECT sue_history_count, seasonal_change, fin_sue "
        f"FROM {SUE_EVENT_TABLE} WHERE bsns_year=2021 AND reprt_code='11013'"
    ).fetchone()
    assert row is not None
    assert row[0] < 8
    assert row[1] is not None
    assert row[2] is None


def test_original_event_requires_filing_receipt_confirmation() -> None:
    con = _base_con()
    _insert_full_year(con, bsns_year=2020, q1=100, q2=100, q3=100, q4=100, rcept_prefix="2020")
    # 2021 Q1 has a financial-statement row but NO filing_receipt_raw entry.
    _insert_ni(
        con,
        bsns_year=2021,
        reprt_code="11013",
        rcept_no="20210410000001",
        thstrm_amount=110,
        report_nm=None,
    )
    _insert_shares(
        con, bsns_year=2021, reprt_code="11013", rcept_no="20210410000001", cumulative_average=1000
    )
    _register(con)

    rows = con.execute(
        f"SELECT * FROM {SUE_EVENT_TABLE} WHERE bsns_year=2021 AND reprt_code='11013'"
    ).fetchall()
    assert rows == []


def test_revision_within_60_sessions_blocks_primary_sample() -> None:
    con = _base_con()
    _insert_full_year(con, bsns_year=2020, q1=100, q2=100, q3=100, q4=100, rcept_prefix="2020")
    _insert_ni(
        con, bsns_year=2021, reprt_code="11013", rcept_no="20210410000001", thstrm_amount=110
    )
    _insert_shares(
        con, bsns_year=2021, reprt_code="11013", rcept_no="20210410000001", cumulative_average=1000
    )
    # A correction lands 10 sessions after formation -- well within 60.
    formation = date(2021, 4, 12)
    revision_disclosed = [d for d in _TRADING_DAYS if d >= formation][9]
    _insert_ni(
        con,
        bsns_year=2021,
        reprt_code="11013",
        rcept_no=revision_disclosed.strftime("%Y%m%d") + "000002",
        thstrm_amount=115,
        report_nm="[기재정정]사업보고서",
    )
    for i, d in enumerate([d for d in _TRADING_DAYS if d >= formation][:65]):
        _insert_close(con, d, 100)
    _register(con)

    row = con.execute(
        f"SELECT revision_within_60_sessions, is_primary_constant_sample "
        f"FROM {SUE_EVENT_TABLE} WHERE bsns_year=2021 AND reprt_code='11013'"
    ).fetchone()
    assert row[0] is True
    assert row[1] is False


def test_bucket_returns_ca_contamination_and_grain_uniqueness() -> None:
    con = _base_con()
    _insert_full_year(con, bsns_year=2020, q1=100, q2=100, q3=100, q4=100, rcept_prefix="2020")
    _insert_ni(
        con, bsns_year=2021, reprt_code="11013", rcept_no="20210410000001", thstrm_amount=110
    )
    _insert_shares(
        con, bsns_year=2021, reprt_code="11013", rcept_no="20210410000001", cumulative_average=1000
    )
    formation = date(2021, 4, 12)
    window_days = [d for d in _TRADING_DAYS if d >= formation][:65]
    for i, d in enumerate(window_days):
        _insert_close(con, d, 100 + i)
    # Flag the 4th trading day after formation (inside bucket (3,5]) as a CA event.
    con.execute(
        "INSERT INTO dim_price_quality_daily VALUES (?, ?, 'KOSPI', TRUE)",
        [window_days[4], TICKER],
    )
    _register(con)

    row = con.execute(
        f"SELECT bucket_0_3_raw, bucket_0_3_ca_contaminated, "
        f"bucket_3_5_ca_contaminated, is_primary_constant_sample "
        f"FROM {SUE_EVENT_TABLE} WHERE bsns_year=2021 AND reprt_code='11013'"
    ).fetchone()
    assert row[0] == (100 + 3) / 100 - 1
    assert row[1] is False
    assert row[2] is True
    assert row[3] is False  # CA contamination disqualifies the primary sample

    dupes = con.execute(
        f"SELECT ticker, original_rcept_no, event_formation_date, market, COUNT(*) "
        f"FROM {SUE_EVENT_TABLE} GROUP BY 1, 2, 3, 4 HAVING COUNT(*) > 1"
    ).fetchall()
    assert dupes == []
