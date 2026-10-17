"""``feat_fin_v2.fin_roe_v2`` (15_quality_fix_plan §4.4) and the frozen fin_scan text."""

from __future__ import annotations

import datetime as dt
import hashlib

import duckdb
import pytest

from modeler.etl.features.fin_scan import build_fin_scan_daily_sql
from modeler.etl.features.fin_v2 import build_fin_v2_sql

FILED = dt.date(2020, 5, 1)
ASOF = dt.date(2020, 6, 1)


def _row(ticker, metric, value, rcept, *, basis="CFS", seq=1, filed=FILED):
    flow = metric in ("net_income", "controlling_net_income")
    return (
        ticker,
        metric,
        basis,
        seq,
        rcept,
        "direct_interim" if flow else "instant",
        value if flow else None,
        None if flow else value,
        filed if flow else None,
        None if flow else filed,
        None,
    )


def _con(rows, tickers):
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE fin_quarterly_metric_vintage (ticker VARCHAR, metric_code VARCHAR, "
        "fs_basis VARCHAR, seq_key BIGINT, rcept_no VARCHAR, metric_kind VARCHAR, "
        "ttm_value DOUBLE, standalone_value DOUBLE, ttm_available_from DATE, "
        "available_from DATE, value_lag_4q DOUBLE)"
    )
    con.executemany("INSERT INTO fin_quarterly_metric_vintage VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.execute(
        "CREATE TABLE dim_stock_pit_daily (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "market_cap_pit DOUBLE, issued_shares_pit DOUBLE, shares_is_available BOOLEAN, "
        "shares_invalid_flag BOOLEAN, shares_available_from DATE)"
    )
    con.execute(
        "CREATE TABLE dim_price_quality_daily (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "is_halted BOOLEAN, valid_session_idx BIGINT)"
    )
    for t in tickers:
        con.execute(
            "INSERT INTO dim_stock_pit_daily VALUES (?, ?, 'KOSPI', 1e6, 100, TRUE, FALSE, ?)",
            [ASOF, t, FILED],
        )
        con.execute(
            "INSERT INTO dim_price_quality_daily VALUES (?, ?, 'KOSPI', FALSE, 1)", [ASOF, t]
        )
    return con


def _roe(con) -> dict[str, float | None]:
    sql = build_fin_v2_sql()
    return {t: v for t, v in con.execute(f"SELECT ticker, fin_roe_v2 FROM ({sql})").fetchall()}


def test_ratio_and_equity_guard() -> None:
    rows = []
    for t, equity in (("POS", 200.0), ("NEG", -200.0), ("ZERO", 0.0)):
        rows += [
            _row(t, "net_income", 30.0, f"R{t}"),
            _row(t, "controlling_net_income", 20.0, f"R{t}"),
            _row(t, "total_equity", equity, f"R{t}"),
        ]
    rows += [
        _row("NOEQ", "net_income", 30.0, "RNOEQ"),
        _row("NOEQ", "controlling_net_income", 20.0, "RNOEQ"),
    ]
    out = _roe(_con(rows, ["POS", "NEG", "ZERO", "NOEQ"]))
    assert out["POS"] == pytest.approx(0.1)  # controlling 20 / 200
    assert out["NEG"] is None
    assert out["ZERO"] is None
    assert out["NOEQ"] is None


def test_falls_back_to_net_income_when_controlling_is_missing() -> None:
    rows = [_row("A", "net_income", 30.0, "R1"), _row("A", "total_equity", 300.0, "R1")]
    assert _roe(_con(rows, ["A"]))["A"] == pytest.approx(0.1)


def test_numerator_and_denominator_must_share_a_filing() -> None:
    rows = [
        _row("A", "net_income", 30.0, "R1"),
        _row("A", "controlling_net_income", 20.0, "R1"),
        _row("A", "total_equity", 200.0, "R2"),  # same quarter re-filed elsewhere
        _row("B", "net_income", 30.0, "R1"),
        _row("B", "controlling_net_income", 20.0, "R1"),
        _row("B", "total_equity", 200.0, "R1"),
    ]
    out = _roe(_con(rows, ["A", "B"]))
    assert out["A"] is None
    assert out["B"] == pytest.approx(0.1)


def test_one_basis_per_ticker_date() -> None:
    # CFS net income exists -> CFS is used even though an OFS equity exists with its own rcept.
    rows = [
        _row("A", "net_income", 30.0, "R1", basis="CFS"),
        _row("A", "total_equity", 300.0, "R1", basis="CFS"),
        _row("A", "net_income", 50.0, "R9", basis="OFS"),
        _row("A", "total_equity", 100.0, "R9", basis="OFS"),
    ]
    assert _roe(_con(rows, ["A"]))["A"] == pytest.approx(0.1)


def test_base_ok_gate() -> None:
    rows = [_row("A", "net_income", 30.0, "R1"), _row("A", "total_equity", 300.0, "R1")]
    con = _con(rows, ["A"])
    con.execute("UPDATE dim_price_quality_daily SET is_halted = TRUE")
    assert _roe(con)["A"] is None


def test_fin_scan_sql_text_is_unchanged() -> None:
    assert hashlib.sha256(build_fin_scan_daily_sql().encode()).hexdigest() == (
        "9656de5eb96b9b1498187308a9ea3831a3f0a2f4e29ff45cca7fb061fdd7fb4a"
    )
