from __future__ import annotations

import datetime

import duckdb
import pytest

from modeler.etl.stock_pit import build_stock_pit_sql
from modeler.etl.trading_panel import build_full_panel_sql, build_valid_session_sql


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        """CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES
        (DATE '2024-01-02', 'A', 'KOSPI', 100, 100, 100, 100, 10),
        (DATE '2024-01-03', 'A', 'KOSPI', 100, 100, 100, 110, 10),
        (DATE '2024-01-04', 'A', 'KOSPI', 0, 0, 0, 110, 0),
        (DATE '2024-01-05', 'A', 'KOSPI', 100, 100, 100, 121, 10)
        ) AS t(trade_date,ticker,market,open,high,low,close,volume)"""
    )
    con.execute(
        """CREATE VIEW dart_share_count_raw AS SELECT * FROM (VALUES
        ('A', 2023, '11011', '20240102000001', '합계', 1000, NULL, 700, DATE '2023-12-31'),
        ('A', 2023, '11011', '20240104000001', '합계', 2000, 100, NULL, DATE '2023-12-31'),
        ('A', 2022, '11011', '20240106000001', '합계', 3000, 100, 2800, DATE '2022-12-31')
        ) AS t(ticker,bsns_year,reprt_code,rcept_no,se,istc_totqy,tesstk_co,
        distb_stock_co,stlm_dt)"""
    )
    return con


def test_valid_session_skips_halt_and_preserves_full_panel() -> None:
    con = _con()
    con.execute(f"CREATE VIEW valid AS {build_valid_session_sql()}")
    assert con.execute("SELECT count(*) FROM valid").fetchone()[0] == 3
    assert con.execute("SELECT valid_session_idx FROM valid ORDER BY trade_date").fetchall() == [
        (1,),
        (2,),
        (3,),
    ]
    con.execute(f"CREATE VIEW panel AS {build_full_panel_sql()}")
    halt = con.execute(
        "SELECT is_halted, valid_session_idx FROM panel WHERE trade_date=DATE '2024-01-04'"
    ).fetchone()
    assert halt == (True, None)


def test_pit_uses_next_session_and_latest_period_without_backward_fill() -> None:
    con = _con()
    con.execute(f"CREATE VIEW pit AS {build_stock_pit_sql()}")
    rows = con.execute(
        "SELECT trade_date, issued_shares_pit, float_shares_pit, shares_available_from "
        "FROM pit ORDER BY trade_date"
    ).fetchall()
    # 1/2 filing becomes visible on next KRX session, 1/4 correction on 1/5.
    assert rows[0][1] is None
    assert rows[1][1] == pytest.approx(1000)
    assert rows[2][1] == pytest.approx(1000)  # halt remains a price row
    assert rows[3][1] == pytest.approx(2000)
    assert rows[1][2] == pytest.approx(700)
    assert rows[3][2] == pytest.approx(1900)  # issued - treasury fallback
    assert rows[1][3] == datetime.date(2024, 1, 3)


def test_pit_does_not_let_late_old_period_override_new_period() -> None:
    con = _con()
    con.execute(f"CREATE VIEW pit AS {build_stock_pit_sql()}")
    (issued,) = con.execute(
        "SELECT issued_shares_pit FROM pit WHERE trade_date=DATE '2024-01-05'"
    ).fetchone()
    assert issued == pytest.approx(2000)
