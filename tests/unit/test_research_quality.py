from __future__ import annotations

import json

import duckdb

from modeler.etl.quality import (
    build_price_quality_sql,
    diagnose_publication_lag,
    short_regime_sql,
    write_publication_lag_evidence,
)


def _views() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        """CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES
        (DATE '2015-06-14','A','KOSPI',100,100,100,100,10),
        (DATE '2015-06-15','A','KOSPI',100,100,100,140,10),
        (DATE '2015-06-16','A','KOSPI',100,100,100,140,10),
        (DATE '2024-01-02','A','KOSPI',100,100,100,100,10),
        (DATE '2024-01-03','A','KOSPI',100,100,100,50,10)
        ) AS t(trade_date,ticker,market,open,high,low,close,volume)"""
    )
    con.execute(
        """CREATE VIEW dim_stock_pit_daily AS SELECT * FROM (VALUES
        (DATE '2015-06-14','A','KOSPI',1000.0),
        (DATE '2015-06-15','A','KOSPI',1000.0),
        (DATE '2015-06-16','A','KOSPI',1000.0),
        (DATE '2024-01-02','A','KOSPI',1000.0),
        (DATE '2024-01-03','A','KOSPI',2000.0)
        ) AS t(trade_date,ticker,market,issued_shares_pit)"""
    )
    return con


def test_regime_boundaries() -> None:
    con = duckdb.connect()
    rows = con.execute(
        "SELECT d, " + short_regime_sql("d") + " FROM (VALUES "
        "(DATE '2020-03-13'),(DATE '2020-03-16'),(DATE '2021-05-03'),"
        "(DATE '2023-11-06'),(DATE '2025-03-31')) t(d)"
    ).fetchall()
    assert [r[1] for r in rows] == ["allowed", "banned", "partial", "banned", "allowed"]


def test_jump_and_share_confirmation_are_separate() -> None:
    con = _views()
    con.execute(f"CREATE VIEW q AS {build_price_quality_sql()}")
    before, after = con.execute(
        "SELECT ca_price_jump_suspect FROM q WHERE trade_date IN "
        "(DATE '2015-06-14', DATE '2015-06-15') ORDER BY trade_date"
    ).fetchall()[0][0], con.execute(
        "SELECT ca_price_jump_suspect FROM q WHERE trade_date=DATE '2015-06-15'"
    ).fetchone()[0]
    assert before is False
    assert after is True
    confirmed = con.execute(
        "SELECT ca_share_change_confirmed FROM q WHERE trade_date=DATE '2024-01-03'"
    ).fetchone()[0]
    assert confirmed is True


def test_quality_keeps_halt_out_of_valid_index() -> None:
    con = _views()
    con.execute(f"CREATE VIEW q AS {build_price_quality_sql()}")
    row = con.execute(
        "SELECT is_halted, valid_session_idx FROM q WHERE trade_date=DATE '2024-01-02'"
    ).fetchone()
    assert row == (False, 4)


def test_unknown_ca_applicability_is_masked_conservatively() -> None:
    con = duckdb.connect()
    con.execute(
        """CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES
        (DATE '2013-01-01','A','KOSPI',100,100,100,100,10),
        (DATE '2013-01-02','A','KOSPI',100,100,100,101,10)
        ) AS t(trade_date,ticker,market,open,high,low,close,volume)"""
    )
    con.execute(
        "CREATE VIEW dim_stock_pit_daily AS SELECT * FROM (VALUES "
        "(DATE '2013-01-01','A','KOSPI',1000.0),"
        "(DATE '2013-01-02','A','KOSPI',1000.0) "
        ") AS t(trade_date,ticker,market,issued_shares_pit)"
    )
    con.execute(f"CREATE VIEW q AS {build_price_quality_sql()}")
    row = con.execute(
        "SELECT ca_price_jump_suspect, ca_rule_applicability_unknown, ca_event, ca_mask "
        "FROM q WHERE trade_date=DATE '2013-01-02'"
    ).fetchone()
    assert row == (None, True, True, True)


def _flow_lag_fixture(balance_max: str, reference_max: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        """CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES
        (DATE '2024-01-02','A','KOSPI'),
        (DATE '2024-01-03','A','KOSPI'),
        (DATE '2024-01-04','A','KOSPI'),
        (DATE '2024-01-05','A','KOSPI'),
        (DATE '2024-01-08','A','KOSPI')
        ) AS t(trade_date,ticker,market)"""
    )
    con.execute(
        f"""CREATE VIEW krx_security_flow_raw AS SELECT * FROM (VALUES
        (DATE '{reference_max}','A','KOSPI','short_selling_volume',100.0),
        (DATE '{reference_max}','A','KOSPI','short_selling_value',100.0),
        (DATE '{balance_max}','A','KOSPI','short_selling_balance_quantity',100.0)
        ) AS t(trade_date,ticker,market,metric_code,value)"""
    )
    return con


def test_diagnose_publication_lag_detects_two_session_gap() -> None:
    con = _flow_lag_fixture("2024-01-04", "2024-01-08")
    diagnosis = diagnose_publication_lag(con)
    assert diagnosis["verified"] is True
    assert diagnosis["public_lag_sessions"] == 2
    assert diagnosis["balance_max_trade_date"] == "2024-01-04"
    assert diagnosis["reference_max_trade_date"] == "2024-01-08"


def test_diagnose_publication_lag_unresolved_when_no_gap() -> None:
    con = _flow_lag_fixture("2024-01-08", "2024-01-08")
    diagnosis = diagnose_publication_lag(con)
    assert diagnosis["verified"] is False
    assert diagnosis["public_lag_sessions"] is None
    assert diagnosis["reason"] == "insufficient_raw_coverage_or_no_observed_gap"


def test_diagnose_publication_lag_unresolved_when_gap_implausible() -> None:
    con = _flow_lag_fixture("2024-01-02", "2024-01-08")
    diagnosis = diagnose_publication_lag(con, max_plausible_lag_sessions=1)
    assert diagnosis["verified"] is False
    assert diagnosis["public_lag_sessions"] is None
    assert "outside_plausible_range" in diagnosis["reason"]


def test_write_publication_lag_evidence_verified(tmp_path) -> None:
    con = _flow_lag_fixture("2024-01-04", "2024-01-08")
    path = write_publication_lag_evidence(tmp_path / "short_balance_publication_lag.json", con)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "verified"
    assert payload["public_lag_sessions"] == 2
    assert payload["measurement_date_field"] == "trade_date (sourced from KRX RPT_DUTY_OCCR_DD)"
    assert payload["evidence_basis"]


def test_write_publication_lag_evidence_without_connection_is_unresolved(tmp_path) -> None:
    path = write_publication_lag_evidence(tmp_path / "short_balance_publication_lag.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "unresolved"
    assert payload["public_lag_sessions"] is None
