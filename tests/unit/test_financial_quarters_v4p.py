"""``fin_quarterly_metric_vintage_dlr_v4p`` -- P6 ``_dlr`` with PIT fiscal placement, R1, 1e14.

The vintage fact is faked with a plain table (the statements read only a handful of its columns)
and so is the receipt list. A filing's ``rcept_no`` starts with its receipt date, as in DART, so
the date a report name became known is the same date the fact's ``disclosed_date`` carries.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import duckdb
import pytest

from modeler.etl.config import DataRoot, LakeConfig
from modeler.etl.marts import financial_quarters_v4 as fq4
from modeler.etl.marts import financial_quarters_v4p as fq4p
from modeler.etl.marts import metric_vintages_delisted as dl

_REPRT = {1: "11013", 2: "11012", 3: "11014", 4: "11011"}
_KIND = {1: "분기보고서", 2: "반기보고서", 3: "분기보고서", 4: "사업보고서"}
_FACT = "fake_fact"
_RCPT = fq4p.DEFAULT_RECEIPT_VIEW
_COLS = (
    "ticker, metric_code, fs_basis, bsns_year, quarter, seq_key, statement_period_end, "
    "rcept_no, available_from, standalone_value, standalone_source_conflict, "
    "cumulative_derived_value, comparative_q_amount, value_lag_4q, ttm_value, ttm_complete, "
    "ttm_available_from, ttm_includes_conflict"
)


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(
        f"CREATE TABLE {_FACT} (ticker VARCHAR, market VARCHAR, corp_code VARCHAR, "
        "metric_code VARCHAR, fs_basis VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "value_numeric DECIMAL(30,4), cumulative_value_numeric DECIMAL(30,4), "
        "comparative_q_amount DECIMAL(30,4), disclosed_date DATE, available_from DATE, "
        "rcept_no VARCHAR, statement_period_end DATE, is_revision BOOLEAN, "
        "period_end_source VARCHAR, source_key VARCHAR)"
    )
    con.execute(
        f"CREATE TABLE {_RCPT} (corp_code VARCHAR, rcept_no VARCHAR, report_nm VARCHAR, "
        "rcept_dt DATE)"
    )
    return con


def _rcept(day: date, tail: str = "000000") -> str:
    """A 14-digit ``rcept_no`` that starts with its receipt date (deterministic)."""
    return f"{day:%Y%m%d}{tail}"


def _add(
    con, *, ticker="T", corp=None, metric="net_income", ordinal, pe, filed, value, cum=None,
    code=None, name=True, revision=False, rcept=None,
):  # fmt: skip
    """One filing of report ``ordinal`` (``code``: the code the fact stored it under), received
    on ``filed``. ``name=True`` puts ``<kind> (YYYY.MM)`` on the receipt list, a string that
    name, ``False`` nothing. Returns the ``rcept_no``."""
    rcept = rcept or _rcept(filed, f"{pe:%y%m}{ordinal}{code or ordinal}")
    con.execute(
        f"INSERT INTO {_FACT} VALUES "
        "(?, 'KOSPI', ?, ?, 'CFS', ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, 'xbrl', ?)",
        [
            ticker, corp or ticker, metric, pe.year, _REPRT[code or ordinal], value, cum,
            filed, filed + timedelta(days=1), rcept, pe, revision, f"{rcept}:{metric}",
        ],
    )  # fmt: skip
    if name:
        nm = name if isinstance(name, str) else f"{_KIND[ordinal]} ({pe.year}.{pe.month:02d})"
        con.execute(f"INSERT INTO {_RCPT} VALUES (?, ?, ?, ?)", [corp or ticker, rcept, nm, filed])
    return rcept


def _q_end(year: int, q: int) -> date:
    return date(year, 3 * q, (31, 30, 30, 31)[q - 1])


def _december_company(con, *, ticker="D", conflict_q2=False):
    """Calendar 2020-2021 with names; ``conflict_q2``: 2021 Q2's cumulative disagrees."""
    for year, base in ((2020, 0), (2021, 100)):
        for q, value in ((1, 10), (2, 20), (3, 30), (4, 100)):
            pe = _q_end(year, q)
            filed = date(year + 1, 3, 15) if q == 4 else pe + timedelta(days=40)
            cum = 999 if (conflict_q2 and year == 2021 and q == 2) else None
            _add(con, ticker=ticker, ordinal=q, pe=pe, filed=filed, value=value + base, cum=cum)


def _rows(con, ticker, metric="net_income"):
    fq4p.register_fin_quarterly_metric_vintage_dlr_v4p_view(con, vintage_view=_FACT)
    return con.execute(
        "SELECT fiscal_year_end, quarter, standalone_value, ttm_value, ttm_complete, "
        "standalone_null_reason, statement_period_end, seq_key, bsns_year, "
        "standalone_source_conflict, ttm_includes_conflict, value_lag_4q "
        f"FROM {fq4p.FQMV_DLR_V4P_TABLE} WHERE ticker=? AND metric_code=? ORDER BY seq_key",
        [ticker, metric],
    ).fetchall()


def _by_slot(rows):
    return {(r[0], r[1]): r for r in rows}


@pytest.mark.parametrize("conflict", [False, True])
def test_december_company_equals_the_p6_dlr_mart(conflict: bool) -> None:
    con = _con()
    _december_company(con, conflict_q2=conflict)
    sql = dl.build_fin_quarterly_metric_vintage_dlr_sql(vintage_view=_FACT, semantics="v1")
    p6 = con.execute(f"SELECT {_COLS} FROM ({sql}) ORDER BY ALL").fetchall()
    fq4p.register_fin_quarterly_metric_vintage_dlr_v4p_view(con, vintage_view=_FACT)
    new = con.execute(f"SELECT {_COLS} FROM {fq4p.FQMV_DLR_V4P_TABLE} ORDER BY ALL").fetchall()
    assert new == p6 and len(new) == 8
    if conflict:
        q2 = _by_slot(_rows(con, "D"))[(date(2021, 12, 31), "Q2")]
        assert q2[2] == 120 and q2[9] is True  # R1: reported value kept, flag set


def _march_to_december(con, ticker="X"):
    """March year end through 2021-03, a 9-month year to 2021-12, December from 2022.

    The 2021-06 and 2021-09 reports were filed before anything told the year end would move;
    the transition year's annual report (사업보고서 2021.12) arrived 2022-03-20.
    """
    con.execute(
        f"INSERT INTO {_RCPT} VALUES (?, ?, '사업보고서 (2020.03)', DATE '2020-06-20')",
        [ticker, _rcept(date(2020, 6, 20))],
    )
    series = [
        # (ordinal as DART filed it, period end, filed, value)
        (1, date(2020, 6, 30), date(2020, 8, 10), 10),
        (2, date(2020, 9, 30), date(2020, 11, 10), 20),
        (3, date(2020, 12, 31), date(2021, 2, 10), 30),
        (4, date(2021, 3, 31), date(2021, 6, 20), 100),   # FYE 2021-03: Q4 = 40
        (1, date(2021, 6, 30), date(2021, 8, 10), 5),
        (2, date(2021, 9, 30), date(2021, 11, 10), 6),
        (4, date(2021, 12, 31), date(2022, 3, 20), 18),   # 9-month transition year
        (1, date(2022, 3, 31), date(2022, 5, 10), 1),
        (2, date(2022, 6, 30), date(2022, 8, 10), 2),
        (3, date(2022, 9, 30), date(2022, 11, 10), 3),
        (4, date(2022, 12, 31), date(2023, 3, 20), 10),   # FYE 2022-12: Q4 = 4
    ]  # fmt: skip
    for ordinal, pe, filed, value in series:
        _add(con, ticker=ticker, ordinal=ordinal, pe=pe, filed=filed, value=value)


def test_a_later_year_end_change_does_not_reach_earlier_filings() -> None:
    con = _con()
    _march_to_december(con)
    slots = _by_slot(_rows(con, "X"))
    # Filed while the company was a March company: Q1 / Q2 of the year ending 2022-03.
    assert slots[(date(2022, 3, 31), "Q1")][2] == 5
    assert slots[(date(2022, 3, 31), "Q2")][2] == 6
    # The transition year's own annual report is known at its filing to be 9 months: dropped.
    assert all(r[6] != date(2021, 12, 31) for r in slots.values())
    # r2 places the same filings with the later name and drops them.
    audit = con.execute(
        "SELECT statement_period_end, fiscal_position_status FROM "
        f"({fq4.build_fiscal_position_audit_sql(vintage_view=_FACT)}) WHERE ticker='X'"
    ).fetchall()
    assert dict(audit)[date(2021, 6, 30)] == fq4.POSITION_TRANSITION_YEAR


def _audit_by_rcept(con):
    sql = fq4p.build_pit_fiscal_position_audit_sql(vintage_view=_FACT)
    return {
        r[0]: r[1:]
        for r in con.execute(
            f"SELECT rcept_no, fiscal_position_status, fiscal_year_end, fiscal_reprt_code "
            f"FROM ({sql})"
        ).fetchall()
    }


@pytest.mark.parametrize("cut", [date(2021, 12, 31), date(2022, 3, 19), date(2022, 6, 30)])
def test_placement_is_the_same_when_later_data_is_cut_off(cut: date) -> None:
    full = _con()
    _march_to_december(full)
    cut_con = _con()
    _march_to_december(cut_con)
    cut_con.execute(f"DELETE FROM {_FACT} WHERE disclosed_date > ?", [cut])
    cut_con.execute(f"DELETE FROM {_RCPT} WHERE rcept_dt > ?", [cut])
    # rcept_no carries its receipt date, so the two lists have the same numbers.
    whole, cut_off = _audit_by_rcept(full), _audit_by_rcept(cut_con)
    assert cut_off and all(whole[k] == v for k, v in cut_off.items())
    # Same for the mart rows (both TTM windows and values only look back).
    fq4p.register_fin_quarterly_metric_vintage_dlr_v4p_view(full, vintage_view=_FACT)
    fq4p.register_fin_quarterly_metric_vintage_dlr_v4p_view(cut_con, vintage_view=_FACT)
    q = f"SELECT {_COLS}, fiscal_year_end FROM {fq4p.FQMV_DLR_V4P_TABLE} ORDER BY ALL"
    kept = {r[7] for r in cut_con.execute(q).fetchall()}
    assert [r for r in full.execute(q).fetchall() if r[7] in kept] == cut_con.execute(q).fetchall()


def test_old_and_new_numbering_get_separate_slots() -> None:
    con = _con()
    _march_to_december(con)
    rows = _rows(con, "X")
    slots = _by_slot(rows)
    # "2022 Q1" twice: the year ending 2022-03 (filed 2021-08) and the one ending 2022-12.
    q1s = [r for r in rows if r[8] == 2022 and r[1] == "Q1"]
    assert {(r[0], r[2]) for r in q1s} == {(date(2022, 3, 31), 5), (date(2022, 12, 31), 1)}
    # Q4 of the new year subtracts its own Q1..Q3, not the old-numbering Q1.
    assert slots[(date(2022, 12, 31), "Q4")][2] == 10 - (1 + 2 + 3)
    # One row per calendar quarter, in time order.
    keys = [r[7] for r in rows]
    assert len(keys) == len(set(keys))
    assert [r[6] for r in rows] == sorted(r[6] for r in rows)
    # TTM never bridges the dropped 2021-12 quarter; the full new year is one window.
    assert [slots[(date(2022, 12, 31), q)][4] for q in ("Q1", "Q2", "Q3")] == [False] * 3
    assert slots[(date(2022, 12, 31), "Q4")][3] == 10 and slots[(date(2022, 12, 31), "Q4")][4]
    # The old year: Q4 2021-03 = 40, and 2021-09's window = 30 + 40 + 5 + 6.
    assert slots[(date(2021, 3, 31), "Q4")][2] == 40
    assert slots[(date(2022, 3, 31), "Q2")][3] == 30 + 40 + 5 + 6


def _march_company(con, *, ticker="M", metric="net_income", cum2=None):
    """FY Apr 2020 - Mar 2022 with names; ``cum2``: cumulative of the fiscal-2022 half year."""
    con.execute(
        f"INSERT INTO {_RCPT} VALUES (?, ?, '사업보고서 (2020.03)', DATE '2020-06-20')",
        [ticker, _rcept(date(2020, 6, 20))],
    )
    series = [
        (1, date(2020, 6, 30), 10), (2, date(2020, 9, 30), 20), (3, date(2020, 12, 31), 30),
        (4, date(2021, 3, 31), 100),
        (1, date(2021, 6, 30), 11), (2, date(2021, 9, 30), 21), (3, date(2021, 12, 31), 31),
        (4, date(2022, 3, 31), 110),
    ]  # fmt: skip
    for ordinal, pe, value in series:
        filed = pe + timedelta(days=80 if ordinal == 4 else 40)
        cum = cum2 if pe == date(2021, 9, 30) else None
        _add(con, ticker=ticker, metric=metric, ordinal=ordinal, pe=pe, filed=filed,
             value=value, cum=cum)  # fmt: skip


def test_r1_keeps_the_conflicting_quarter_and_flags_the_ttm() -> None:
    con = _con()
    _march_company(con, cum2=999)
    slots = _by_slot(_rows(con, "M"))
    fy = date(2022, 3, 31)
    assert slots[(fy, "Q2")][2] == 21 and slots[(fy, "Q2")][9] is True
    assert slots[(fy, "Q2")][5] is None
    # No Q4 block under R1: Q4 = 110 - (11 + 21 + 31).
    assert slots[(fy, "Q4")][2] == 47 and slots[(fy, "Q4")][5] is None
    # Windows with the conflict quarter are flagged, the one before it is not.
    assert slots[(fy, "Q2")][10] is True and slots[(fy, "Q4")][10] is True
    assert slots[(fy, "Q1")][10] is False
    assert slots[(date(2021, 3, 31), "Q4")][10] is False


def test_outlier_block_nulls_duration_values_and_the_q4_that_uses_them() -> None:
    con = _con()
    _march_company(con)
    con.execute(f"UPDATE {_FACT} SET value_numeric = 500000000000000 "
                "WHERE statement_period_end = DATE '2021-09-30'")  # fmt: skip
    slots = _by_slot(_rows(con, "M"))
    fy = date(2022, 3, 31)
    assert slots[(fy, "Q2")][2] is None and slots[(fy, "Q2")][5] == fq4p.REASON_OUTLIER
    assert slots[(fy, "Q4")][2] is None and slots[(fy, "Q4")][5] == fq4p.REASON_Q4_OUTLIER
    assert slots[(fy, "Q3")][3] is None  # every window with the blocked quarter
    assert slots[(date(2021, 3, 31), "Q4")][2] == 40
    # A balance-sheet value of that size is not a unit error candidate: kept.
    _add(con, ticker="B", metric="total_assets", ordinal=1, pe=date(2021, 3, 31),
         filed=date(2021, 5, 10), value=500000000000000)  # fmt: skip
    assert _rows(con, "B", "total_assets")[0][2] == 500000000000000


def test_without_the_outlier_block_the_value_stays() -> None:
    con = _con()
    _march_company(con)
    con.execute(f"UPDATE {_FACT} SET value_numeric = 500000000000000 "
                "WHERE statement_period_end = DATE '2021-09-30'")  # fmt: skip
    sql = fq4p.build_fin_quarterly_metric_vintage_dlr_v4p_sql(
        vintage_view=_FACT, outlier_block=False
    )
    vals = con.execute(
        f"SELECT standalone_value FROM ({sql}) WHERE statement_period_end = DATE '2021-09-30'"
    ).fetchall()
    assert vals == [(500000000000000,)]


def test_filings_before_the_first_annual_report_name() -> None:
    con = _con()
    # A June company listed in 2021: no annual-report name before 2021-09.
    q3 = _add(con, ticker="J", ordinal=3, pe=date(2021, 3, 31), filed=date(2021, 5, 14), value=3)
    # A half-year report stored under Q1's code: its own name says half year.
    half = _add(con, ticker="H", ordinal=2, code=1, pe=date(2020, 12, 31),
                filed=date(2021, 2, 14), value=2)  # fmt: skip
    # A quarterly report under the half-year code: Q1 or Q3 cannot be told -> dropped.
    odd = _add(con, ticker="K", ordinal=1, code=2, pe=date(2021, 3, 31),
               filed=date(2021, 5, 14), value=1)  # fmt: skip
    # A December company's first quarters: as in r2.
    dec = _add(con, ticker="N", ordinal=1, pe=date(2021, 3, 31), filed=date(2021, 5, 14), value=7)
    audit = _audit_by_rcept(con)
    assert audit[q3] == (fq4.POSITION_NO_ANNUAL_REFERENCE, date(2021, 6, 30), "11014")
    assert audit[half] == (fq4.POSITION_RELABELLED, date(2021, 6, 30), "11012")
    assert audit[odd][0] == fq4.POSITION_CODE_MISMATCH and audit[odd][1] is None
    assert audit[dec] == (fq4.POSITION_NO_ANNUAL_REFERENCE, date(2021, 12, 31), "11013")
    assert _rows(con, "N")[0][:2] == (date(2021, 12, 31), "Q1")


def test_a_period_end_keeps_the_slot_of_its_first_vintage() -> None:
    con = _con()
    # Original stored as Q1 (December assumption), revision as Q3 (June): different slots.
    first = _add(con, ticker="S", ordinal=1, pe=date(2021, 3, 31), filed=date(2021, 5, 28),
                 value=4, name="분기보고서 (2021.03)")  # fmt: skip
    _add(con, ticker="S", ordinal=3, pe=date(2021, 3, 31), filed=date(2021, 8, 12), value=4,
         name="[기재정정]분기보고서 (2021.03)", revision=True)  # fmt: skip
    rows = _rows(con, "S")
    assert len(rows) == 1 and rows[0][:2] == (date(2021, 12, 31), "Q1")
    fq4p.register_fin_quarterly_metric_vintage_dlr_v4p_view(con, vintage_view=_FACT)
    rcept = con.execute(f"SELECT rcept_no FROM {fq4p.FQMV_DLR_V4P_TABLE}").fetchone()[0]
    assert rcept == first


def test_half_year_report_three_months_after_a_year_end_is_dropped() -> None:
    con = _con()
    # A REIT with 6-month years: annual names 2020-12 and 2021-06 are known.
    for pe, filed in (
        (date(2020, 12, 31), date(2021, 3, 20)),
        (date(2021, 6, 30), date(2021, 9, 20)),
    ):
        _add(con, ticker="R", ordinal=4, pe=pe, filed=filed, value=50)
    # Its next report: 3 months into a 6-month year, named 반기보고서, filed under Q1's code.
    reit = _add(con, ticker="R", ordinal=1, pe=date(2021, 9, 30), filed=date(2021, 11, 14),
                value=9, name="반기보고서 (2021.09)")  # fmt: skip
    assert _audit_by_rcept(con)[reit][0] == fq4.POSITION_CODE_MISMATCH
    assert all(r[6] != date(2021, 9, 30) for r in _rows(con, "R"))


def test_columns_types_and_materialized_metadata(tmp_path) -> None:
    con = _con()
    _march_company(con)
    con.execute(f"CREATE VIEW {fq4p.DEFAULT_VINTAGE_VIEW} AS SELECT * FROM {_FACT}")
    p6 = con.execute(
        "DESCRIBE SELECT * FROM ("
        f"{dl.build_fin_quarterly_metric_vintage_dlr_sql(vintage_view=_FACT, semantics='v1')})"
    ).fetchall()
    config = LakeConfig(root=DataRoot(tmp_path), snapshot_date="2026-10-23", source="test")
    view = fq4p.materialize_fin_quarterly_metric_vintage_dlr_v4p(con, config)
    assert view == fq4p.FQMV_DLR_V4P_TABLE == "fin_quarterly_metric_vintage_dlr_v4p"
    new_sql = fq4p.build_fin_quarterly_metric_vintage_dlr_v4p_sql()
    new = con.execute(f"DESCRIBE SELECT * FROM ({new_sql})").fetchall()
    assert [(r[0], r[1]) for r in new[: len(p6)]] == [(r[0], r[1]) for r in p6]
    assert tuple(r[0] for r in new) == fq4p.DLR_COLUMNS + fq4p.EXTRA_COLUMNS
    out = config.feature_mart_root
    assert sorted(p.name for p in out.iterdir()) == [fq4p.FQMV_DLR_V4P_TABLE]
    meta = json.loads((out / view / "_cache_metadata.json").read_text())
    assert meta["semantics_version"] == fq4p.SEMANTICS_V4P
    assert meta["conflict_rule"] == "R1_keep_reported_quarter"
    assert meta["fiscal_placement"] == "pit_receipt_names"
    assert meta["variant"] == "keep_delisted_corps"
    assert meta["vintage_view"] == "stock_metric_vintage_fact_dl_v3"
    # The mart filters no company: a delisted name of the (delisted-keeping) fact stays.
    _add(con, ticker="GONE", ordinal=1, pe=date(2019, 3, 31), filed=date(2019, 5, 10), value=1)
    sql = fq4p.build_fin_quarterly_metric_vintage_dlr_v4p_sql(vintage_view=_FACT)
    assert con.execute(f"SELECT count(*) FROM ({sql}) WHERE ticker='GONE'").fetchone()[0] == 1


def test_r1_text_guard(monkeypatch) -> None:
    old, new = fq4p._R1_VALUE_REPLACEMENTS[0]
    assert "THEN NULL" in old and "THEN q2_value" in new
    monkeypatch.setattr(fq4p, "_R1_VALUE_REPLACEMENTS", (("text that is not there", "x"),))
    with pytest.raises(dl.OriginalSqlDriftError):
        fq4p._r1_statement("x", 0.0)
