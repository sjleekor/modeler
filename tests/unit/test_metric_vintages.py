from __future__ import annotations

from datetime import date

import duckdb
from modeler.etl.marts.metric_vintages import (
    SMVF_TABLE,
    register_stock_metric_vintage_fact_view,
)


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


_TRADING_DAYS = [
    d for year in (2025, 2026) for month in range(1, 13) for d in _weekdays_in(year, month)
]

# Real filings always name the consolidation basis on a dimension axis — all
# 2,545 FY2024 annual filings in the 2026-08-09 lake carry axis-marked
# contexts. A fact without one cannot be paired against a specific fs_div, so
# fixtures that leave dimensions empty describe a filing that does not exist
# and hid the pairing defect in 08 §4.3.2.
_AXIS = "ifrs-full:ConsolidatedAndSeparateFinancialStatementsAxis"
_DIM_CFS = f'["{_AXIS}=ifrs-full:ConsolidatedMember"]'
_DIM_OFS = f'["{_AXIS}=ifrs-full:SeparateMember"]'


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


def _register(con: duckdb.DuckDBPyConnection) -> None:
    register_stock_metric_vintage_fact_view(con, trading_days=_TRADING_DAYS)


def test_baseline_identity_transform_no_receipt_history() -> None:
    """Pre-B-1 state: single captured rcept_no per filing, no receipt list at
    all — every row must land as captured_vintages_only, not blocked."""
    con = _base_con()
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
        "('00126380','005930',2025,'11011','CFS','IS','ifrs-full_Revenue','매출액',1,"
        "1000.0000,'KRW','20260310000001')"
    )
    con.execute(
        "INSERT INTO dart_xbrl_fact_raw VALUES "
        "('00126380','005930',2025,'11011','20260310000001','ifrs-full_Revenue','매출액',"
        "'ctx1','duration',DATE '2025-01-01',DATE '2025-12-31',NULL,'"
        + _DIM_CFS
        + "',1000.0000,'매출액')"
    )
    con.execute(
        "INSERT INTO dart_share_count_raw VALUES "
        "('00126380','005930',2025,'11011','20260310000001','합계',5000,100,DATE '2025-12-31')"
    )
    _register(con)

    rows = con.execute(
        f"SELECT metric_code, rcept_no, disclosed_date, available_from, availability_source, "
        f"captured_vintage_status, receipt_value_pairing_status, period_end_source, "
        f"statement_period_end FROM {SMVF_TABLE} ORDER BY metric_code"
    ).fetchall()
    by_metric = {r[0]: r for r in rows}

    revenue = by_metric["revenue"]
    assert revenue[1] == "20260310000001"
    assert revenue[2] == date(2026, 3, 10)
    assert revenue[3] == date(2026, 3, 11)  # next KRX session after 2026-03-10 (Tue)
    assert revenue[4] == "rcept_no"
    assert revenue[5] == "captured_vintages_only"
    assert revenue[6] == "verified_same_receipt"
    assert revenue[7] == "xbrl"
    assert revenue[8] == date(2025, 12, 31)

    issued_shares = by_metric["issued_shares"]
    assert issued_shares[5] == "captured_vintages_only"
    assert issued_shares[6] == "not_applicable"


def test_missing_rcept_no_uses_synthetic_fallback() -> None:
    con = _base_con()
    con.execute(
        "INSERT INTO dart_share_count_raw VALUES "
        "('00126380','005930',2025,'11011','','합계',5000,100,DATE '2025-12-31')"
    )
    _register(con)

    row = con.execute(
        f"SELECT availability_source, available_from, statement_period_end "
        f"FROM {SMVF_TABLE} WHERE metric_code = 'issued_shares'"
    ).fetchone()
    assert row[0] == "synthetic_fallback"
    # annual report -> period_end + 90 days
    assert row[1] == date(2025, 12, 31) + __import__("datetime").timedelta(days=90)
    assert row[2] == date(2025, 12, 31)


def test_multi_vintage_preserves_both_receipts_after_backfill() -> None:
    """B-1 backfill scenario: original + a later correction, both linked via
    dart_filing_receipt_raw. Grain must NOT collapse to one winner."""
    con = _base_con()
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
        "('00126380','005930',2025,'11011','CFS','IS','ifrs-full_Revenue','매출액',1,"
        "1000.0000,'KRW','20260310000001'),"
        "('00126380','005930',2025,'11011','CFS','IS','ifrs-full_Revenue','매출액',1,"
        "1100.0000,'KRW','20260405000002')"
    )
    con.execute(
        "INSERT INTO dart_filing_receipt_raw VALUES "
        "('00126380','20260310000001','사업보고서'),"
        "('00126380','20260405000002','[기재정정]사업보고서')"
    )
    _register(con)

    rows = con.execute(
        f"SELECT rcept_no, disclosed_date, available_from, is_revision, original_rcept_no, "
        f"captured_vintage_status FROM {SMVF_TABLE} "
        f"WHERE metric_code = 'revenue' ORDER BY rcept_no"
    ).fetchall()
    assert len(rows) == 2

    original, revision = rows
    assert original[0] == "20260310000001"
    assert original[3] is False
    assert original[4] == "20260310000001"
    assert original[5] == "original_confirmed_revisions_partial"

    assert revision[0] == "20260405000002"
    assert revision[3] is True
    assert revision[4] is None
    assert revision[5] == "original_confirmed_revisions_partial"

    # Each vintage's own availability, independently computed.
    assert original[2] == date(2026, 3, 11)
    assert revision[2] == date(2026, 4, 6)


def test_row_without_receipt_match_is_unlinked_when_siblings_are_matched() -> None:
    con = _base_con()
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
        "('00126380','005930',2025,'11011','CFS','IS','ifrs-full_Revenue','매출액',1,"
        "1000.0000,'KRW','20260310000001'),"
        "('00126380','005930',2025,'11011','CFS','IS','ifrs-full_Revenue','매출액',1,"
        "1100.0000,'KRW','20260405000002')"
    )
    # Only the first receipt is in the receipt list; the second is unlinkable.
    con.execute(
        "INSERT INTO dart_filing_receipt_raw VALUES ('00126380','20260310000001','사업보고서')"
    )
    _register(con)

    rows = con.execute(
        f"SELECT rcept_no, captured_vintage_status FROM {SMVF_TABLE} "
        f"WHERE metric_code = 'revenue' ORDER BY rcept_no"
    ).fetchall()
    status_by_rcept = dict(rows)
    assert status_by_rcept["20260310000001"] == "original_confirmed_revisions_partial"
    assert status_by_rcept["20260405000002"] == "unlinked_receipt"


def test_same_day_filings_collapse_to_latest_rcept_no() -> None:
    con = _base_con()
    con.execute(
        "INSERT INTO dart_share_count_raw VALUES "
        "('00126380','005930',2025,'11011','20260310000001','합계',5000,100,DATE '2025-12-31'),"
        "('00126380','005930',2025,'11011','20260310000009','합계',5000,100,DATE '2025-12-31')"
    )
    _register(con)

    rows = con.execute(
        f"SELECT rcept_no, same_day_effective_rcept_no FROM {SMVF_TABLE} "
        f"WHERE metric_code = 'issued_shares' ORDER BY rcept_no"
    ).fetchall()
    assert len(rows) == 2
    assert all(effective == "20260310000009" for _rcept, effective in rows)


def test_period_end_conflict_is_flagged_and_xbrl_wins() -> None:
    con = _base_con()
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
        "('00126380','005930',2025,'11011','CFS','IS','ifrs-full_Revenue','매출액',1,"
        "1000.0000,'KRW','20260310000001')"
    )
    # XBRL says period end 2025-12-31; share_count's stlm_dt disagrees (2025-12-30).
    con.execute(
        "INSERT INTO dart_xbrl_fact_raw VALUES "
        "('00126380','005930',2025,'11011','20260310000001','ifrs-full_Revenue','매출액',"
        "'ctx1','duration',DATE '2025-01-01',DATE '2025-12-31',NULL,'"
        + _DIM_CFS
        + "',1000.0000,'매출액')"
    )
    con.execute(
        "INSERT INTO dart_share_count_raw VALUES "
        "('00126380','005930',2025,'11011','20260310000001','합계',5000,100,DATE '2025-12-30')"
    )
    _register(con)

    row = con.execute(
        f"SELECT period_end_source, period_end_conflict, statement_period_end "
        f"FROM {SMVF_TABLE} WHERE metric_code = 'revenue'"
    ).fetchone()
    assert row[0] == "xbrl"
    assert row[1] is True
    assert row[2] == date(2025, 12, 31)


def test_cfs_and_ofs_are_preserved_as_distinct_rows() -> None:
    con = _base_con()
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
        "('00126380','005930',2025,'11011','CFS','IS','ifrs-full_Revenue','매출액',1,"
        "1000.0000,'KRW','20260310000001'),"
        "('00126380','005930',2025,'11011','OFS','IS','ifrs-full_Revenue','매출액',1,"
        "900.0000,'KRW','20260310000001')"
    )
    _register(con)

    rows = con.execute(
        f"SELECT fs_basis, value_numeric FROM {SMVF_TABLE} "
        f"WHERE metric_code = 'revenue' ORDER BY fs_basis"
    ).fetchall()
    assert rows == [("CFS", 1000.0), ("OFS", 900.0)]


def test_xbrl_value_mismatch_is_flagged_not_silently_accepted() -> None:
    con = _base_con()
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
        "('00126380','005930',2025,'11011','CFS','IS','ifrs-full_Revenue','매출액',1,"
        "1000.0000,'KRW','20260310000001')"
    )
    con.execute(
        "INSERT INTO dart_xbrl_fact_raw VALUES "
        "('00126380','005930',2025,'11011','20260310000001','ifrs-full_Revenue','매출액',"
        "'ctx1','duration',DATE '2025-01-01',DATE '2025-12-31',NULL,'"
        + _DIM_CFS
        + "',999.0000,'매출액')"
    )
    _register(con)

    row = con.execute(
        f"SELECT receipt_value_pairing_status, pairing_tolerance FROM {SMVF_TABLE} "
        f"WHERE metric_code = 'revenue'"
    ).fetchone()
    assert row[0] == "value_mismatch"
    assert row[1] == 0.0


def test_no_is_active_dependency_on_corp_master() -> None:
    """corp_view is only consulted for ticker/market/corp_code — passing a
    corp master without an is_active column at all must not raise (§3.4)."""
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
    con.execute(
        "INSERT INTO dart_share_count_raw VALUES "
        "('00126380','005930',2025,'11011','20260310000001','합계',5000,100,DATE '2025-12-31')"
    )
    _register(con)
    # istc_totqy and tesstk_co are both non-null on the one raw row, so two
    # metric candidates (issued_shares, treasury_shares) are correctly emitted.
    row_count = con.execute(f"SELECT COUNT(*) FROM {SMVF_TABLE}").fetchone()[0]
    assert row_count == 2


# --- 08 §4.3.2: an XBRL fact belongs to a period and a consolidation basis ---
#
# The fixtures above describe a filing whose XBRL holds exactly one context.
# Real filings hold three comparative years on two consolidation axes, and that
# is the shape in which all three defects below appear.


def _add_financial(
    con: duckdb.DuckDBPyConnection,
    *,
    amount: float,
    fs_div: str = "CFS",
    sj_div: str = "IS",
    account_id: str = "ifrs-full_Revenue",
    account_nm: str = "매출액",
    bsns_year: int = 2025,
    reprt_code: str = "11011",
    rcept_no: str = "20260310000001",
    ord_: int = 1,
) -> None:
    con.execute(
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no) "
        "VALUES ('00126380','005930',?,?,?,?,?,?,?,?, 'KRW', ?)",
        [bsns_year, reprt_code, fs_div, sj_div, account_id, account_nm, ord_, amount, rcept_no],
    )


def _add_xbrl(
    con: duckdb.DuckDBPyConnection,
    *,
    value: float,
    context_id: str,
    period_start: date | None = None,
    period_end: date | None = None,
    instant_date: date | None = None,
    dimensions: str = _DIM_CFS,
    concept_id: str = "ifrs-full_Revenue",
    label: str = "매출액",
    bsns_year: int = 2025,
    reprt_code: str = "11011",
    rcept_no: str = "20260310000001",
) -> None:
    con.execute(
        "INSERT INTO dart_xbrl_fact_raw VALUES " "('00126380','005930',?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            bsns_year,
            reprt_code,
            rcept_no,
            concept_id,
            label,
            context_id,
            "instant" if instant_date is not None else "duration",
            period_start,
            period_end,
            instant_date,
            dimensions,
            value,
            label,
        ],
    )


def _three_comparative_years(con: duckdb.DuckDBPyConnection, dimensions: str = _DIM_CFS) -> None:
    """What a FY2025 annual filing's XBRL actually contains for one concept."""
    for year, value in ((2023, 700.0), (2024, 850.0), (2025, 1000.0)):
        _add_xbrl(
            con,
            value=value,
            context_id=f"ctx{year}",
            period_start=date(year, 1, 1),
            period_end=date(year, 12, 31),
            dimensions=dimensions,
        )


def _revenue_row(con: duckdb.DuckDBPyConnection, fs_basis: str = "CFS") -> tuple:
    return con.execute(
        f"SELECT statement_period_end, receipt_value_pairing_status, period_end_source, "
        f"period_end_conflict, value_numeric FROM {SMVF_TABLE} "
        f"WHERE metric_code = 'revenue' AND fs_basis = '{fs_basis}'"
    ).fetchone()


def test_statement_period_end_is_the_filings_own_period_not_the_oldest_comparative() -> None:
    con = _base_con()
    _add_financial(con, amount=1000.0)
    _three_comparative_years(con)
    _register(con)

    period_end, pairing, source, conflict, _ = _revenue_row(con)

    # Taking MIN() over the three contexts put this on 2023-12-31, and
    # statement_period_end is the mart's grain.
    assert period_end == date(2025, 12, 31)
    assert source == "xbrl"
    assert conflict is False
    # And the value now pairs against its own year rather than an arbitrary one.
    assert pairing == "verified_same_receipt"


def test_a_context_dated_after_the_receipt_cannot_be_the_period_end() -> None:
    con = _base_con()
    _add_financial(con, amount=1000.0)
    _three_comparative_years(con)
    # A forward-looking context: this filing was submitted 2026-03-10, so it
    # does not report on a period ending 2026-12-31.
    _add_xbrl(
        con,
        value=1200.0,
        context_id="ctx_future",
        period_start=date(2026, 1, 1),
        period_end=date(2026, 12, 31),
    )
    _register(con)

    period_end, pairing, _, _, _ = _revenue_row(con)

    assert period_end == date(2025, 12, 31)
    assert pairing == "verified_same_receipt"


def test_stlm_date_agrees_with_the_xbrl_period_once_the_right_context_is_picked() -> None:
    con = _base_con()
    _add_financial(con, amount=1000.0)
    _three_comparative_years(con)
    con.execute(
        "INSERT INTO dart_share_count_raw VALUES "
        "('00126380','005930',2025,'11011','20260310000001','합계',5000,100,DATE '2025-12-31')"
    )
    _register(con)

    _, _, source, conflict, _ = _revenue_row(con)

    # period_end_conflict only means something once both sides name the same
    # period; against the oldest comparative it fired on 98.3% of real rows.
    assert source == "xbrl"
    assert conflict is False


def test_a_separate_statement_row_pairs_against_the_separate_context() -> None:
    con = _base_con()
    _add_financial(con, amount=1000.0, fs_div="CFS")
    _add_financial(con, amount=400.0, fs_div="OFS", ord_=2)
    _three_comparative_years(con, dimensions=_DIM_CFS)
    _three_comparative_years_ofs(con)
    _register(con)

    assert _revenue_row(con, "CFS")[1] == "verified_same_receipt"
    # SeparateMember always loses the dimension tie-break, so before the fix
    # this row was compared against the consolidated 1000 and could never
    # verify — that alone made value_mismatch unavoidable for half the rows.
    assert _revenue_row(con, "OFS")[1] == "verified_same_receipt"


def _three_comparative_years_ofs(con: duckdb.DuckDBPyConnection) -> None:
    for year, value in ((2023, 280.0), (2024, 340.0), (2025, 400.0)):
        _add_xbrl(
            con,
            value=value,
            context_id=f"ctx{year}_ofs",
            period_start=date(year, 1, 1),
            period_end=date(year, 12, 31),
            dimensions=_DIM_OFS,
        )


def test_interim_income_statement_pairs_against_the_three_month_duration() -> None:
    con = _base_con()
    # OpenDART's thstrm_amount for an interim IS is the 3-month figure.
    _add_financial(con, amount=300.0, sj_div="IS", reprt_code="11012", rcept_no="20250814000001")
    _add_xbrl(
        con,
        value=300.0,
        context_id="ctx_q2",
        period_start=date(2025, 4, 1),
        period_end=date(2025, 6, 30),
        reprt_code="11012",
        rcept_no="20250814000001",
    )
    _add_xbrl(
        con,
        value=550.0,
        context_id="ctx_ytd",
        period_start=date(2025, 1, 1),
        period_end=date(2025, 6, 30),
        reprt_code="11012",
        rcept_no="20250814000001",
    )
    _register(con)

    period_end, pairing, _, _, value = _revenue_row(con)

    assert period_end == date(2025, 6, 30)
    assert value == 300
    assert pairing == "verified_same_receipt"


def test_interim_cash_flow_pairs_against_the_cumulative_duration() -> None:
    con = _base_con()
    # ... while a CF thstrm_amount is year-to-date, so the same filing needs
    # the opposite context. One rule for both is what makes this a defect
    # rather than a preference.
    _add_financial(
        con,
        amount=550.0,
        sj_div="CF",
        account_id="ifrs-full_CashFlowsFromUsedInOperatingActivities",
        account_nm="영업활동현금흐름",
        reprt_code="11012",
        rcept_no="20250814000001",
    )
    for context_id, start, value in (
        ("ctx_q2", date(2025, 4, 1), 300.0),
        ("ctx_ytd", date(2025, 1, 1), 550.0),
    ):
        _add_xbrl(
            con,
            value=value,
            context_id=context_id,
            period_start=start,
            period_end=date(2025, 6, 30),
            concept_id="ifrs-full_CashFlowsFromUsedInOperatingActivities",
            label="영업활동현금흐름",
            reprt_code="11012",
            rcept_no="20250814000001",
        )
    _register(con)

    row = con.execute(
        f"SELECT value_numeric, receipt_value_pairing_status FROM {SMVF_TABLE} "
        f"WHERE metric_code = 'operating_cash_flow'"
    ).fetchone()

    assert row[0] == 550
    assert row[1] == "verified_same_receipt"


def test_an_xbrl_sourced_metric_takes_the_current_period_not_a_comparative() -> None:
    con = _base_con()
    # No financial-statement counterpart: amortization comes straight from XBRL,
    # so nothing else pins its period down.
    for year, value in ((2023, 70.0), (2024, 85.0), (2025, 100.0)):
        _add_xbrl(
            con,
            value=value,
            context_id=f"ctx_amort_{year}",
            period_start=date(year, 1, 1),
            period_end=date(year, 12, 31),
            concept_id="ifrs-full_AmortisationExpense",
            label="무형자산상각비",
        )
    _register(con)

    row = con.execute(
        f"SELECT statement_period_end, value_numeric FROM {SMVF_TABLE} "
        f"WHERE metric_code = 'amortization_intangible_assets'"
    ).fetchone()

    assert row[0] == date(2025, 12, 31)
    # A two-year-old number carrying the current period's date is the worst of
    # the three defects: it is silently wrong rather than missing.
    assert row[1] == 100


def test_pairing_is_unlinked_when_no_context_matches_the_basis() -> None:
    con = _base_con()
    _add_financial(con, amount=400.0, fs_div="OFS")
    # Consolidated-only XBRL — a standalone-only filer's opposite case.
    _three_comparative_years(con, dimensions=_DIM_CFS)
    _register(con)

    # Not "value_mismatch": the two numbers were never comparable, and calling
    # that a mismatch is what made the frozen tolerance unusable as a gate.
    assert _revenue_row(con, "OFS")[1] == "unlinked_receipt"


# --------------------------------------------------------------------------
# I7 — the XBRL fallback for financial-statement metrics
# --------------------------------------------------------------------------


def _xbrl_row(concept: str, dims: str, value: float, rcept: str = "20260310000001") -> str:
    return (
        "INSERT INTO dart_xbrl_fact_raw VALUES "
        f"('00126380','005930',2025,'11011','{rcept}','{concept}','x',"
        f"'ctx_{concept}_{dims[-14:-2]}','duration',DATE '2025-01-01',DATE '2025-12-31',NULL,'"
        + dims
        + f"',{value},'x')"
    )


def _statement_row(account_id: str, fs_div: str, value: float) -> str:
    return (
        "INSERT INTO dart_financial_statement_raw "
        "(corp_code, ticker, bsns_year, reprt_code, fs_div, sj_div, account_id, "
        "account_nm, ord, thstrm_amount, currency, rcept_no) VALUES "
        f"('00126380','005930',2025,'11011','{fs_div}','IS','{account_id}','x',1,"
        f"{value},'KRW','20260310000001')"
    )


def test_the_fallback_fills_a_metric_the_statement_api_never_reported() -> None:
    # The whole point of I7. `revenue` had 8,103 canonical rows against
    # `net_income`'s 141,011, because the financial-statement API simply does
    # not carry it for most filers — while dart_xbrl_fact_raw does.
    con = _base_con()
    con.execute(_xbrl_row("ifrs-full_Revenue", _DIM_CFS, 1000.0))
    _register(con)

    rows = con.execute(
        f"SELECT fs_basis, value_numeric, source_table FROM {SMVF_TABLE} "
        "WHERE metric_code = 'revenue'"
    ).fetchall()

    assert rows == [("CFS", 1000, "dart_xbrl_fact_raw")]


def test_the_fallback_lands_in_the_same_partition_as_the_statement_rule() -> None:
    # The defect the first attempt hit: the XBRL branch hardcoded fs_basis to
    # '', the winner window partitions by fs_basis, and so a fallback never
    # competed with a CFS statement row — it added a second row for the same
    # metric and period instead of filling a gap.
    con = _base_con()
    con.execute(_statement_row("ifrs-full_Revenue", "CFS", 1000.0))
    con.execute(_xbrl_row("ifrs-full_Revenue", _DIM_CFS, 1000.0))
    _register(con)

    rows = con.execute(
        f"SELECT fs_basis, source_table FROM {SMVF_TABLE} WHERE metric_code = 'revenue'"
    ).fetchall()

    assert len(rows) == 1
    assert rows[0] == ("CFS", "dart_financial_statement_raw")


def test_a_reported_figure_always_outranks_the_fallback() -> None:
    # Priority 100/200 sits below the statement rules' 10/20, so the fallback
    # only ever fills a gap. If it could win, an XBRL context would silently
    # replace the number OpenDART actually reported.
    con = _base_con()
    con.execute(_statement_row("ifrs-full_Revenue", "CFS", 1000.0))
    con.execute(_xbrl_row("ifrs-full_Revenue", _DIM_CFS, 9999.0))
    _register(con)

    value = con.execute(
        f"SELECT value_numeric FROM {SMVF_TABLE} WHERE metric_code = 'revenue'"
    ).fetchone()[0]

    assert value == 1000


def test_the_basis_comes_from_the_xbrl_dimensions() -> None:
    # ConsolidatedMember -> CFS, SeparateMember -> OFS. Without this the two
    # bases would collapse onto one row and one of them would be lost.
    con = _base_con()
    con.execute(_xbrl_row("ifrs-full_Revenue", _DIM_CFS, 1000.0))
    con.execute(_xbrl_row("ifrs-full_Revenue", _DIM_OFS, 700.0))
    _register(con)

    rows = con.execute(
        f"SELECT fs_basis, value_numeric FROM {SMVF_TABLE} "
        "WHERE metric_code = 'revenue' ORDER BY fs_basis"
    ).fetchall()

    assert rows == [("CFS", 1000), ("OFS", 700)]


def test_a_fact_with_no_basis_dimension_is_not_used_as_a_fallback() -> None:
    # A context that names no consolidation axis cannot be assigned to CFS or
    # OFS, and guessing would put a separate-basis figure in the consolidated
    # row. Every real filing marks the axis (all 2,545 FY2024 annuals do).
    con = _base_con()
    con.execute(_xbrl_row("ifrs-full_Revenue", "[]", 1000.0))
    _register(con)

    rows = con.execute(
        f"SELECT count(*) FROM {SMVF_TABLE} WHERE metric_code = 'revenue'"
    ).fetchone()

    assert rows[0] == 0


def test_the_legacy_xbrl_metrics_keep_their_empty_basis() -> None:
    # weighted_avg_shares and friends have always lived at fs_basis = ''.
    # Moving them would break parity well outside I7's scope, which is why the
    # basis follows the RULE rather than the source table.
    con = _base_con()
    con.execute(_xbrl_row("ifrs-full_WeightedAverageShares", _DIM_CFS, 500.0))
    _register(con)

    rows = con.execute(
        f"SELECT fs_basis FROM {SMVF_TABLE} WHERE metric_code = 'weighted_avg_shares'"
    ).fetchall()

    assert rows == [("",)]


def test_both_ifrs_spellings_are_mapped() -> None:
    # `ifrs_Revenue` carries 184,846 facts against `ifrs-full_Revenue`'s
    # 555,934; mapping only one spelling leaves a quarter of them unread.
    from collector.kr.definitions.metric_rules import default_metric_mapping_rules

    fallback = [r for r in default_metric_mapping_rules() if r.rule_code.startswith("xbrlfb.")]
    revenue_concepts = {r.account_id for r in fallback if r.metric_code == "revenue"}

    assert revenue_concepts == {"ifrs-full_Revenue", "ifrs_Revenue"}


def test_every_fallback_rule_names_a_basis() -> None:
    # A fallback rule without fs_div would be stranded at fs_basis='' and could
    # never fill the gap it exists for.
    from collector.kr.definitions.metric_rules import default_metric_mapping_rules

    fallback = [r for r in default_metric_mapping_rules() if r.rule_code.startswith("xbrlfb.")]

    assert fallback
    assert all(r.fs_div in {"CFS", "OFS"} for r in fallback)
    assert all(r.priority >= 100 for r in fallback)
