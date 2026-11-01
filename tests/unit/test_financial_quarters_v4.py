"""``fin_quarterly_metric_vintage_v4`` -- fiscal renumbering, TTM continuity, outlier and Q4 blocks.

The vintage fact is faked with a plain table (the quarterly statement only reads a handful of
its columns), so each case states exactly the filings it needs. Default v1/v2 text is pinned
in ``test_metric_vintages_income_v3.py``; here a December-end series is also checked to be
unchanged by v4.
"""

from __future__ import annotations

import json
from datetime import date

import duckdb
import pytest

from modeler.etl.config import DataRoot, LakeConfig
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
        "statement_period_end DATE, is_revision BOOLEAN, period_end_source VARCHAR, "
        "source_key VARCHAR)"
    )
    # Empty unless a case adds report names: a company without any annual-report name keeps
    # the code-based position, so the cases that predate the receipt rule need no receipts.
    con.execute(
        f"CREATE TABLE {fq4.DEFAULT_RECEIPT_VIEW} "
        "(corp_code VARCHAR, rcept_no VARCHAR, report_nm VARCHAR)"
    )
    return con


_KIND = {1: "분기보고서", 2: "반기보고서", 3: "분기보고서", 4: "사업보고서"}


def _add(
    con, *, ticker="T", metric="net_income", cal_year, ordinal, pe, value, cum=None, rcept=None,
    code=None, src="xbrl", name=None, corp="C",
):  # fmt: skip
    """One filing. ``cal_year`` is the (calendar-year) ``bsns_year`` the fact carries today.

    ``ordinal`` is the report the filing really is; ``code`` the ordinal of the ``reprt_code``
    the fact stored it under (default: the same). ``name=True`` also puts the report's name
    (``분기보고서 (YYYY.MM)`` ...) on the receipt list, a string puts that name.
    """
    rcept = rcept or f"{ticker}{pe}{ordinal}"
    con.execute(
        f"INSERT INTO {_FACT} VALUES "
        "(?, 'KOSPI', ?, ?, 'CFS', ?, ?, ?, ?, NULL, ?, ?, ?, FALSE, ?, ?)",
        [
            ticker, corp, metric, cal_year, _REPRT[code or ordinal], value, cum,
            date(pe.year + 1, 1, 1) if ordinal == 4 else date(pe.year, pe.month, 28),
            rcept, pe, src, f"{rcept}:{metric}",
        ],
    )  # fmt: skip
    if name:
        nm = name if isinstance(name, str) else f"{_KIND[ordinal]} ({pe.year}.{pe.month:02d})"
        _receipt(con, rcept, nm, corp=corp)


def _receipt(con, rcept, name, *, corp="C"):
    con.execute(f"INSERT INTO {fq4.DEFAULT_RECEIPT_VIEW} VALUES (?, ?, ?)", [corp, rcept, name])


def _march_company(con, *, ticker="M", metric="net_income", names=False):
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
            con,
            ticker=ticker,
            metric=metric,
            cal_year=pe.year,
            ordinal=ordinal,
            pe=pe,
            value=value,
            name=names,
        )


def _rows(con, ticker="M", metric="net_income"):
    fq4.register_fin_quarterly_metric_vintage_v4_view(con, vintage_view=_FACT)
    return {
        (r[0], r[1]): r
        for r in con.execute(
            "SELECT bsns_year, quarter, standalone_value, ttm_value, ttm_complete, "
            "standalone_null_reason, statement_period_end, seq_key, value_lag_4q, rcept_no "
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


@pytest.mark.parametrize("names", [False, True])
def test_december_year_end_is_unchanged_by_v4(names: bool) -> None:
    con = _con()
    for year, base in ((2020, 0), (2021, 100)):
        for ordinal, value in ((1, 10), (2, 20), (3, 30), (4, 100)):
            pe = date(year, (3, 6, 9, 12)[ordinal - 1], (31, 30, 30, 31)[ordinal - 1])
            _add(
                con, ticker="D", cal_year=year, ordinal=ordinal, pe=pe, value=value + base,
                name=names,
            )  # fmt: skip
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
    assert fq4.FQMV_DL_V4_TABLE == "fin_quarterly_metric_vintage_dl_v4"
    assert fq4.DEFAULT_DL_VINTAGE_VIEW == "stock_metric_vintage_fact_dl_v3"
    assert fq4.DEFAULT_RECEIPT_VIEW == "dart_filing_receipt_raw"
    sql = fq4.build_fin_quarterly_metric_vintage_v4_sql()
    assert "fiscal_fact" in sql and "standalone_null_reason" in sql


def _audit(con):
    return {
        (r[0], r[1], r[2]): r[3:]
        for r in con.execute(
            "SELECT ticker, rcept_no, metric_code, fiscal_position_status, fiscal_bsns_year, "
            f"fiscal_reprt_code FROM ({fq4.build_fiscal_position_audit_sql(vintage_view=_FACT)})"
        ).fetchall()
    }


def test_report_names_give_the_same_march_numbering_as_the_codes() -> None:
    plain, named = _con(), _con()
    _march_company(plain)
    _march_company(named, names=True)
    assert _rows(named) == _rows(plain)
    assert {v[0] for v in _audit(named).values()} == {fq4.POSITION_OK}


def test_interim_filing_stored_under_another_interim_code_is_moved() -> None:
    con = _con()
    _march_company(con, names=True)
    # The fiscal-2022 Q1 report (분기보고서 2021.06) stored under the half-year code.
    con.execute(f"UPDATE {_FACT} SET reprt_code='11012' WHERE statement_period_end='2021-06-30'")
    rows = _rows(con)
    assert [rows[(2022, q)][2] for q in ("Q1", "Q2", "Q3", "Q4")] == [11, 21, 31, 47]
    assert rows[(2022, "Q4")][3] == 11 + 21 + 31 + 47
    status = _audit(con)[("M", "M2021-06-301", "net_income")]
    assert status == (fq4.POSITION_RELABELLED, 2022, "11013")


def test_annual_stored_under_an_interim_code_keeps_balance_items_and_drops_income() -> None:
    con = _con()
    _march_company(con, names=True)
    # An annual report whose income row sits under Q1's code: extracted as a 3-month figure,
    # so it cannot be the annual. Its balance-sheet row is the same in every code.
    _add(con, ticker="M", metric="total_assets", cal_year=2022, ordinal=4, code=1,
         pe=date(2022, 3, 31), value=500, rcept="M2022-03-314")  # fmt: skip
    con.execute(f"UPDATE {_FACT} SET reprt_code='11013' "
                "WHERE metric_code='net_income' AND statement_period_end='2022-03-31'")  # fmt: skip
    audit = _audit(con)
    assert audit[("M", "M2022-03-314", "net_income")][0] == fq4.POSITION_CODE_MISMATCH
    assert audit[("M", "M2022-03-314", "total_assets")] == (fq4.POSITION_RELABELLED, 2022, "11011")
    rows = _rows(con)
    assert (2022, "Q4") not in rows  # no annual -> no derived Q4, and nothing at Q1 either
    assert rows[(2022, "Q1")][2] == 11 and rows[(2022, "Q1")][6] == date(2021, 6, 30)
    bs = _rows(con, metric="total_assets")
    assert bs[(2022, "Q4")][2] == 500 and bs[(2022, "Q4")][6] == date(2022, 3, 31)


def _year_end_change(con, ticker="X"):
    """March year end until 2021-03, a 9-month year to 2021-12, December from 2022."""
    series = [
        (1, date(2020, 6, 30), 10), (2, date(2020, 9, 30), 20), (3, date(2020, 12, 31), 30),
        (4, date(2021, 3, 31), 100),  # FY2021 (Mar): Q4 = 40
        # transition Apr-Dec 2021: what DART calls Q1, half and annual
        (1, date(2021, 6, 30), 5), (2, date(2021, 9, 30), 6), (4, date(2021, 12, 31), 18),
        (1, date(2022, 3, 31), 1), (2, date(2022, 6, 30), 2), (3, date(2022, 9, 30), 3),
        (4, date(2022, 12, 31), 10),  # FY2022 (Dec): Q4 = 4
    ]  # fmt: skip
    for ordinal, pe, value in series:
        _add(con, ticker=ticker, cal_year=pe.year, ordinal=ordinal, pe=pe, value=value,
             name=True, corp=ticker)  # fmt: skip
    _receipt(con, f"{ticker}-fy2020", "사업보고서 (2020.03)", corp=ticker)


def test_fiscal_year_end_change_drops_the_transition_year_and_ttm_does_not_bridge_it() -> None:
    con = _con()
    _year_end_change(con)
    audit = _audit(con)
    dropped = {k[1] for k, v in audit.items() if v[0] == fq4.POSITION_TRANSITION_YEAR}
    assert dropped == {"X2021-06-301", "X2021-09-302", "X2021-12-314"}
    rows = _rows(con, ticker="X")
    assert [rows[(2021, q)][2] for q in ("Q1", "Q2", "Q3", "Q4")] == [10, 20, 30, 40]
    assert [rows[(2022, q)][2] for q in ("Q1", "Q2", "Q3", "Q4")] == [1, 2, 3, 4]
    assert all(r[6].year != 2021 or r[6].month == 3 for r in rows.values() if r[6].year >= 2021)
    # seq_key runs on, but the period ends jump 12 months: no window may reach across.
    assert [rows[(2022, q)][4] for q in ("Q1", "Q2", "Q3")] == [False, False, False]
    assert rows[(2022, "Q4")][3] == 10 and rows[(2022, "Q4")][4]
    assert rows[(2022, "Q1")][8] is None  # four quarters back is 2020-06, 21 months


def test_calendar_fallback_period_end_takes_the_month_of_the_report_name() -> None:
    con = _con()
    _receipt(con, "J-fy2025", "사업보고서 (2025.06)")
    series = [(1, date(2025, 9, 30), 1), (2, date(2025, 12, 31), 2)]
    for ordinal, pe, value in series:
        _add(con, ticker="J", cal_year=pe.year, ordinal=ordinal, pe=pe, value=value, name=True)
    # Q3 (분기보고서 2026.03) whose period end fell back to make_date(2026, 9, 30).
    _add(con, ticker="J", cal_year=2026, ordinal=3, pe=date(2026, 9, 30), value=3,
         src="calendar_fallback", rcept="J-q3", name="분기보고서 (2026.03)")  # fmt: skip
    rows = _rows(con, ticker="J")
    assert rows[(2026, "Q3")][6] == date(2026, 3, 31)
    assert rows[(2026, "Q3")][2] == 3


def test_cumulative_q4_without_q3_has_a_reason() -> None:
    con = _con()
    for ordinal, pe, cum in ((1, date(2021, 3, 31), 5), (2, date(2021, 6, 30), 12),
                             (4, date(2021, 12, 31), 30)):  # fmt: skip
        _add(con, ticker="C", metric="operating_cash_flow", cal_year=2021, ordinal=ordinal,
             pe=pe, value=cum)  # fmt: skip
    r = _rows(con, ticker="C", metric="operating_cash_flow")
    assert r[(2021, "Q4")][2] is None and r[(2021, "Q4")][5] == fq4.REASON_Q4_NO_Q3
    assert r[(2021, "Q2")][2] == 7 and r[(2021, "Q2")][5] is None


def test_negative_and_zero_quarters_are_kept() -> None:
    con = _con()
    for ordinal, value in ((1, -5), (2, 0), (3, 7), (4, 2)):
        pe = date(2021, (3, 6, 9, 12)[ordinal - 1], (31, 30, 30, 31)[ordinal - 1])
        _add(con, ticker="Z", cal_year=2021, ordinal=ordinal, pe=pe, value=value)
    r = _rows(con, ticker="Z")
    assert [r[(2021, q)][2] for q in ("Q1", "Q2", "Q3", "Q4")] == [-5, 0, 7, 0]
    assert all(r[(2021, q)][5] is None for q in ("Q1", "Q2", "Q3", "Q4"))
    assert r[(2021, "Q4")][3] == 2


def test_dl_v4_materializes_under_its_own_name_with_the_variant(tmp_path) -> None:
    con = _con()
    _march_company(con, names=True)
    con.execute(f"CREATE VIEW {fq4.DEFAULT_DL_VINTAGE_VIEW} AS SELECT * FROM {_FACT}")
    config = LakeConfig(root=DataRoot(tmp_path), snapshot_date="2026-10-23", source="test")
    view = fq4.materialize_fin_quarterly_metric_vintage_dl_v4(con, config)
    assert view == fq4.FQMV_DL_V4_TABLE
    out = config.feature_mart_root
    assert sorted(p.name for p in out.iterdir()) == [fq4.FQMV_DL_V4_TABLE]
    meta = json.loads((out / fq4.FQMV_DL_V4_TABLE / "_cache_metadata.json").read_text())
    assert meta["semantics_version"] == fq4.SEMANTICS_V4
    assert meta["variant"] == "keep_delisted_corps"
    assert meta["vintage_view"] == fq4.DEFAULT_DL_VINTAGE_VIEW
    n = con.execute(f"SELECT count(*) FROM {view} WHERE ticker='M'").fetchone()[0]
    fq4.register_fin_quarterly_metric_vintage_v4_view(con, vintage_view=_FACT)
    assert n == con.execute(f"SELECT count(*) FROM {fq4.FQMV_V4_TABLE}").fetchone()[0] == 8
