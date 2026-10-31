"""``fin_quarterly_metric_vintage_v4`` -- fiscal renumbering, TTM continuity, outlier and Q4 blocks.

The vintage fact is faked with a plain table (the quarterly statement only reads a handful of
its columns), so each case states exactly the filings it needs. Default v1/v2 text is pinned
in ``test_metric_vintages_income_v3.py``; here a December-end series is also checked to be
unchanged by v4.
"""

from __future__ import annotations

from datetime import date

import duckdb

from modeler.etl.marts import financial_quarters as fq
from modeler.etl.marts import financial_quarters_v4 as fq4

_REPRT = {1: "11013", 2: "11012", 3: "11014", 4: "11011"}
_FACT = "fake_fact"


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(
        f"CREATE TABLE {_FACT} (ticker VARCHAR, market VARCHAR, corp_code VARCHAR, "
        "metric_code VARCHAR, fs_basis VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "value_numeric DECIMAL(30,4), cumulative_value_numeric DECIMAL(30,4), "
        "comparative_q_amount DECIMAL(30,4), available_from DATE, rcept_no VARCHAR, "
        "statement_period_end DATE, is_revision BOOLEAN)"
    )
    return con


def _add(
    con, *, ticker="T", metric="net_income", cal_year, ordinal, pe, value, cum=None, rcept=None
):
    """One filing. ``cal_year`` is the (calendar-year) ``bsns_year`` the fact carries today."""
    con.execute(
        f"INSERT INTO {_FACT} VALUES (?, 'KOSPI', 'C', ?, 'CFS', ?, ?, ?, ?, NULL, ?, ?, ?, FALSE)",
        [
            ticker, metric, cal_year, _REPRT[ordinal], value, cum,
            date(pe.year + 1, 1, 1) if ordinal == 4 else date(pe.year, pe.month, 28),
            rcept or f"{ticker}{pe}{ordinal}", pe,
        ],
    )  # fmt: skip


def _march_company(con, *, ticker="M", metric="net_income", extra=None):
    """FY Apr 2020 - Mar 2022: fiscal years 2021 and 2022, quarters 10/20/30/40 (+4 each year)."""
    series = [
        # (fiscal ordinal, period end, value, annual?)
        (1, date(2020, 6, 30), 10), (2, date(2020, 9, 30), 20), (3, date(2020, 12, 31), 30),
        (4, date(2021, 3, 31), 100),   # FY2021 annual: Q4 = 100 - 60 = 40
        (1, date(2021, 6, 30), 11), (2, date(2021, 9, 30), 21), (3, date(2021, 12, 31), 31),
        (4, date(2022, 3, 31), 110),   # FY2022 annual: Q4 = 110 - 63 = 47
    ]  # fmt: skip
    for ordinal, pe, value in series:
        _add(
            con, ticker=ticker, metric=metric, cal_year=pe.year, ordinal=ordinal, pe=pe, value=value
        )


def _rows(con, ticker="M", metric="net_income"):
    fq4.register_fin_quarterly_metric_vintage_v4_view(con, vintage_view=_FACT)
    return {
        (r[0], r[1]): r
        for r in con.execute(
            "SELECT bsns_year, quarter, standalone_value, ttm_value, ttm_complete, "
            "standalone_null_reason, statement_period_end, seq_key "
            f"FROM {fq4.FQMV_V4_TABLE} WHERE ticker=? AND metric_code=? "
            "ORDER BY seq_key",
            [ticker, metric],
        ).fetchall()
    }


def test_march_year_end_is_renumbered_by_fiscal_year() -> None:
    con = _con()
    _march_company(con)
    rows = _rows(con)
    # Q4 with period end 2021-03 belongs to fiscal 2021 together with Q1..Q3 ending 2020-06..12.
    assert rows[(2021, "Q4")][6] == date(2021, 3, 31)
    assert [rows[(2021, q)][6].month for q in ("Q1", "Q2", "Q3", "Q4")] == [6, 9, 12, 3]
    assert [rows[(2021, q)][2] for q in ("Q1", "Q2", "Q3", "Q4")] == [10, 20, 30, 40]
    assert [rows[(2022, q)][2] for q in ("Q1", "Q2", "Q3", "Q4")] == [11, 21, 31, 47]


def test_old_numbering_mixed_years_and_v4_ttm_is_four_consecutive_quarters() -> None:
    con = _con()
    _march_company(con)
    # Contrast: the unchanged v2 statement keys by the fact's own bsns_year (calendar of period
    # end), so Q4 (2021-03) sits in 2021 with Q1..Q3 ending 2021-06..12 -> Q4 = 100 - 63.
    fq.register_fin_quarterly_metric_vintage_view(
        con, view_name="old_fq", vintage_view=_FACT, semantics="v2"
    )
    old_q4 = con.execute(
        "SELECT standalone_value FROM old_fq WHERE bsns_year=2021 AND quarter='Q4'"
    ).fetchone()[0]
    assert old_q4 == 100 - 63
    rows = _rows(con)
    # TTM at fiscal 2022 Q4 = Q1..Q4 of fiscal 2022 = 11+21+31+47; at 2022 Q1 = 20+30+40+11.
    assert rows[(2022, "Q4")][3] == 11 + 21 + 31 + 47 and rows[(2022, "Q4")][4]
    assert rows[(2022, "Q1")][3] == 20 + 30 + 40 + 11
    assert rows[(2022, "Q3")][3] == 40 + 11 + 21 + 31
    # seq_key steps by exactly 1 over the whole series.
    keys = [r[7] for r in sorted(rows.values(), key=lambda r: r[7])]
    assert keys == list(range(keys[0], keys[0] + len(keys)))


def test_december_year_end_is_unchanged_by_v4() -> None:
    con = _con()
    for year, base in ((2020, 0), (2021, 100)):
        for ordinal, value in ((1, 10), (2, 20), (3, 30), (4, 100)):
            pe = date(year, (3, 6, 9, 12)[ordinal - 1], (31, 30, 30, 31)[ordinal - 1])
            _add(con, ticker="D", cal_year=year, ordinal=ordinal, pe=pe, value=value + base)
    fq.register_fin_quarterly_metric_vintage_view(
        con, view_name="old_fq", vintage_view=_FACT, semantics="v2"
    )
    fq4.register_fin_quarterly_metric_vintage_v4_view(con, vintage_view=_FACT)
    cols = (
        "ticker, metric_code, fs_basis, bsns_year, quarter, seq_key, standalone_value, "
        "ttm_value, ttm_complete, ttm_available_from, value_lag_4q, available_from, rcept_no"
    )
    old = con.execute(f"SELECT {cols} FROM old_fq ORDER BY ALL").fetchall()
    new = con.execute(f"SELECT {cols} FROM {fq4.FQMV_V4_TABLE} ORDER BY ALL").fetchall()
    assert old == new and len(old) == 8


def test_outlier_standalone_value_is_nulled_with_reason_and_ttm_follows() -> None:
    con = _con()
    _march_company(con)
    # Unit error: fiscal 2022 Q2 filed as 5e14 (also breaks the Q4 difference).
    con.execute(f"UPDATE {_FACT} SET value_numeric = 500000000000000 "
                "WHERE reprt_code='11012' AND statement_period_end=DATE '2021-09-30'")  # fmt: skip
    rows = _rows(con)
    q2 = rows[(2022, "Q2")]
    assert q2[2] is None and q2[5] == fq4.REASON_OUTLIER
    # The Q4 that subtracts it is blocked too, and so is every TTM window with a NULL quarter.
    assert rows[(2022, "Q4")][2] is None and rows[(2022, "Q4")][5] == fq4.REASON_Q4_OUTLIER
    assert rows[(2022, "Q3")][3] is None and rows[(2022, "Q4")][3] is None
    # Values just under the limit stay.
    assert rows[(2021, "Q4")][2] == 40 and rows[(2021, "Q4")][5] is None


def test_outlier_in_annual_makes_q4_null_and_cash_flow_is_also_checked() -> None:
    con = _con()
    _march_company(con)
    con.execute(f"UPDATE {_FACT} SET value_numeric = 9.04e16 "
                "WHERE reprt_code='11011' AND statement_period_end=DATE '2022-03-31'")  # fmt: skip
    rows = _rows(con)
    assert rows[(2022, "Q4")][2] is None and rows[(2022, "Q4")][5] == fq4.REASON_OUTLIER

    con = _con()
    # Cash flow: cumulative filings 5, 12, 20, 6e16 -> Q4 = 6e16 - 20 is over the limit.
    cash = [(1, date(2021, 3, 31), 5), (2, date(2021, 6, 30), 12)]
    cash += [(3, date(2021, 9, 30), 20), (4, date(2021, 12, 31), 6e16)]
    for ordinal, pe, cum in cash:
        _add(con, ticker="C", metric="operating_cash_flow", cal_year=2021, ordinal=ordinal,
             pe=pe, value=cum)  # fmt: skip
    r = _rows(con, ticker="C", metric="operating_cash_flow")
    assert r[(2021, "Q2")][2] == 7 and r[(2021, "Q4")][2] is None
    assert r[(2021, "Q4")][5] == fq4.REASON_OUTLIER


def test_q4_is_null_when_q2_or_q3_has_a_source_conflict() -> None:
    for conflict_ordinal in (2, 3):
        con = _con()
        _march_company(con)
        # Reported cumulative disagrees with the sum of the direct quarters at that filing.
        con.execute(
            f"UPDATE {_FACT} SET cumulative_value_numeric = 999 "
            "WHERE reprt_code=? AND statement_period_end=?",
            [
                _REPRT[conflict_ordinal],
                {2: date(2021, 9, 30), 3: date(2021, 12, 31)}[conflict_ordinal],
            ],
        )
        rows = _rows(con)
        blocked = rows[(2022, f"Q{conflict_ordinal}")]
        assert blocked[2] is None and blocked[5] == fq4.REASON_SOURCE_CONFLICT
        q4 = rows[(2022, "Q4")]
        assert q4[2] is None and q4[5] == fq4.REASON_Q4_CONFLICT
        # The other fiscal year is untouched.
        assert rows[(2021, "Q4")][2] == 40 and rows[(2021, "Q4")][5] is None
        # The unchanged statement (old numbering) still derives a Q4 for that conflict.
        fq.register_fin_quarterly_metric_vintage_view(
            con, view_name="old_fq", vintage_view=_FACT, semantics="v2"
        )
        assert (
            con.execute(
                "SELECT standalone_value FROM old_fq WHERE bsns_year=2021 AND quarter='Q4'"
            ).fetchone()[0]
            is not None
        )


def test_defaults_and_names_do_not_touch_existing_marts() -> None:
    assert fq4.FQMV_V4_TABLE == "fin_quarterly_metric_vintage_v4"
    assert fq4.FQMV_V4_TABLE != fq.FQMV_TABLE
    assert fq4.DEFAULT_VINTAGE_VIEW == "stock_metric_vintage_fact_v3"
    sql = fq4.build_fin_quarterly_metric_vintage_v4_sql()
    assert "fiscal_fact" in sql and "standalone_null_reason" in sql
