"""``stock_pit_nonempty``: an empty later share filing must not override a complete one."""

from __future__ import annotations

import duckdb
import pytest

from modeler.etl.stock_pit import PLAN_ASOF_INTERVALS, build_stock_pit_sql
from modeler.etl.stock_pit_nonempty import (
    build_nonempty_share_view_sql,
    build_stock_pit_nonempty_sql,
)

_COLS = "ticker,bsns_year,reprt_code,rcept_no,se,istc_totqy,tesstk_co,distb_stock_co,stlm_dt"


def _con(share_rows: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        """CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES
        (DATE '2024-01-08', 'A', 'KOSPI', 100, 100, 100, 100, 10),
        (DATE '2024-01-09', 'A', 'KOSPI', 100, 100, 100, 100, 10),
        (DATE '2024-01-10', 'A', 'KOSPI', 100, 100, 100, 100, 10),
        (DATE '2024-01-11', 'A', 'KOSPI', 100, 100, 100, 100, 10),
        (DATE '2024-01-12', 'A', 'KOSPI', 100, 100, 100, 100, 10),
        (DATE '2024-01-08', 'B', 'KOSPI', 50, 50, 50, 50, 10),
        (DATE '2024-01-12', 'B', 'KOSPI', 50, 50, 50, 50, 10)
        ) AS t(trade_date,ticker,market,open,high,low,close,volume)"""
    )
    con.execute(
        f"CREATE VIEW dart_share_count_raw AS SELECT * FROM (VALUES {share_rows}) AS t({_COLS})"
    )
    return con


COMPLETE = [
    "('A', 2023, '11014', '20231115000001', '합계', 1000, 100, 900, DATE '2023-09-30')",
    "('B', 2023, '11014', '20231115000002', '합계', 500, 0, 500, DATE '2023-09-30')",
]
EMPTY_LATER = "('A', 2023, '11011', '20240110000001', '합계', NULL, NULL, NULL, DATE '2023-12-31')"


def _issued(con, sql: str, ticker: str = "A") -> list[tuple]:
    return con.execute(
        f"SELECT trade_date, issued_shares_pit, float_shares_pit FROM ({sql}) "
        "WHERE ticker = ? ORDER BY trade_date",
        [ticker],
    ).fetchall()


def test_empty_later_filing_overrides_in_the_frozen_text_but_not_here() -> None:
    con = _con(",".join([*COMPLETE, EMPTY_LATER]))
    frozen = _issued(con, build_stock_pit_sql())
    fixed = _issued(con, build_stock_pit_nonempty_sql())
    # the empty filing (disclosed 01-10) is visible from 01-11 and blanks the shares
    assert [r[1] for r in frozen] == [1000, 1000, 1000, None, None]
    assert [r[1] for r in fixed] == [1000, 1000, 1000, 1000, 1000]
    assert fixed[-1][2] == pytest.approx(900)


def test_a_later_complete_filing_still_wins() -> None:
    later = "('A', 2023, '11011', '20240110000001', '합계', 1200, 0, 1200, DATE '2023-12-31')"
    con = _con(",".join([*COMPLETE, later]))
    assert [r[1] for r in _issued(con, build_stock_pit_nonempty_sql())] == [
        1000,
        1000,
        1000,
        1200,
        1200,
    ]


def test_identical_to_the_frozen_sql_without_empty_filings() -> None:
    con = _con(",".join(COMPLETE))
    for plan_kwargs in ({}, {"plan": PLAN_ASOF_INTERVALS}):
        a = con.execute(
            f"SELECT * FROM ({build_stock_pit_sql(**plan_kwargs)}) ORDER BY ticker, trade_date"
        ).fetchall()
        b = con.execute(
            f"SELECT * FROM ({build_stock_pit_nonempty_sql(**plan_kwargs)}) "
            "ORDER BY ticker, trade_date"
        ).fetchall()
        assert a == b
        assert len(a) == 7


def test_filter_view_keeps_all_columns_and_only_positive_rows() -> None:
    con = _con(",".join([*COMPLETE, EMPTY_LATER]))
    cols = [d[0] for d in con.execute("SELECT * FROM dart_share_count_raw").description]
    cur = con.execute(build_nonempty_share_view_sql())
    assert [d[0] for d in cur.description] == cols
    assert len(cur.fetchall()) == 2
    assert "FROM my_view" in build_nonempty_share_view_sql("my_view")
