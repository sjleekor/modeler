"""Unit tests for P3 — feat_price and feat_flow (synthetic fixtures, no lake).

feat_price: returns/vol/turnover/halt-flag arithmetic on a clean ramp.
feat_flow: KRX-first dedup (zero-conflict), wide pivot, cumulative net-buy, and
the short-balance NULL gap (coverage asymmetry, etl_00 §3.2).
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path

import duckdb
import pytest
from modeler.etl.features import flow, price


def _write_parquet(path: Path, select_values_sql: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"COPY ({select_values_sql}) TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()


# --- feat_price -------------------------------------------------------------


@pytest.fixture()
def price_con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    # 5 sessions: close 100,110,121,0(halt),133.1 ; halt row carries stale close.
    pq = tmp_path / "ohlcv.parquet"
    rows = [
        ("2020-01-01", 100, 110, 90, 100, 10),
        ("2020-01-02", 100, 120, 95, 110, 12),
        ("2020-01-03", 110, 130, 100, 121, 14),
        ("2020-01-04", 0, 0, 0, 121, 0),  # halted: OHL=0, stale close
        ("2020-01-05", 120, 140, 110, 133, 16),
    ]
    values = ",".join(
        f"(DATE '{d}', '005930', 'KOSPI', {o}, {h}, {lo}, {c}, {v})" for (d, o, h, lo, c, v) in rows
    )
    _write_parquet(
        pq,
        "SELECT * FROM (VALUES " + values + ") AS t"
        "(trade_date, ticker, market, open, high, low, close, volume)",
    )
    con = duckdb.connect()
    src = f"read_parquet('{pq.as_posix()}', hive_partitioning=false)"
    con.execute(f"CREATE VIEW daily_ohlcv AS SELECT * FROM {src}")
    con.execute(f"CREATE VIEW feat_price AS {price.build_price_sql('daily_ohlcv')}")
    return con


def test_price_ret_1d_log_return(price_con) -> None:
    (ret,) = price_con.execute(
        "SELECT px_ret_1d FROM feat_price WHERE trade_date = DATE '2020-01-02'"
    ).fetchone()
    assert ret == pytest.approx(math.log(110 / 100))


def test_price_turnover_is_close_times_volume(price_con) -> None:
    (turnover,) = price_con.execute(
        "SELECT px_turnover FROM feat_price WHERE trade_date = DATE '2020-01-03'"
    ).fetchone()
    assert turnover == pytest.approx(121 * 14)


def test_price_halt_flag_and_ratio(price_con) -> None:
    rows = dict(
        price_con.execute(
            "SELECT trade_date, px_is_halted FROM feat_price ORDER BY trade_date"
        ).fetchall()
    )
    assert rows[datetime.date(2020, 1, 4)] is True
    assert rows[datetime.date(2020, 1, 3)] is False

    (ratio,) = price_con.execute(
        "SELECT px_halt_ratio_20d FROM feat_price WHERE trade_date = DATE '2020-01-05'"
    ).fetchone()
    assert ratio == pytest.approx(1 / 5)  # 1 halt of 5 trailing rows


def test_price_rows_preserved(price_con) -> None:
    (n,) = price_con.execute("SELECT count(*) FROM feat_price").fetchone()
    assert n == 5


def test_amihud_lag1_matches_prior_session_native_value() -> None:
    """px_amihud_20d is joined in via a separate subquery from the other lag1
    columns (research/etl/features/price.py) — its own _lag1 must still obey
    the same "prior valid session" contract §A-1 requires of every family."""
    con = duckdb.connect()
    rows = [
        f"(DATE '2020-01-01' + {i}, '005930', 'KOSPI', 100, 105, 95, {100 + i}, {10 + i})"
        for i in range(25)
    ]
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + ",".join(rows)
        + ") t(trade_date, ticker, market, open, high, low, close, volume)"
    )
    con.execute(f"CREATE VIEW feat_price AS {price.build_price_sql('daily_ohlcv')}")
    day_n, day_n_minus_1_lag1 = con.execute("""
        SELECT
            (SELECT px_amihud_20d FROM feat_price WHERE trade_date = DATE '2020-01-01' + 23),
            (SELECT px_amihud_20d_lag1 FROM feat_price WHERE trade_date = DATE '2020-01-01' + 24)
    """).fetchone()
    assert day_n is not None
    assert day_n_minus_1_lag1 == pytest.approx(day_n)


def test_residual_momentum_waits_for_full_regression_and_momentum_windows() -> None:
    con = duckdb.connect()
    rows = []
    for i in range(510):
        rows.extend([
            f"(DATE '2020-01-01'+{i}, 'A','KOSPI',1,1,1,{100+i},100)",
            f"(DATE '2020-01-01'+{i}, 'B','KOSPI',1,1,1,{200+2*i},100)",
        ])
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + ",".join(rows)
        + ") t(trade_date,ticker,market,open,high,low,close,volume)"
    )
    con.execute(f"CREATE VIEW feat_price AS {price.build_price_sql()}")
    values = con.execute(
        "SELECT px_resid_mom_12_1 FROM feat_price WHERE ticker='A' "
        "ORDER BY trade_date OFFSET 500 LIMIT 6"
    ).fetchall()
    assert all(row[0] is None for row in values[:5])
    assert values[5][0] is not None


def test_turnover_shock_nulls_zero_volume_instead_of_crashing() -> None:
    # A non-halted zero-volume session (open/high/low != 0, volume=0) must not
    # trip DuckDB's "cannot take logarithm of zero" on the LN(turnover/...) arg.
    con = duckdb.connect()
    rows = [f"(DATE '2020-01-01'+{i}, 'A','KOSPI',100,105,95,100,10)" for i in range(65)]
    rows[64] = "(DATE '2020-01-01'+64, 'A','KOSPI',100,105,95,100,0)"
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + ",".join(rows)
        + ") t(trade_date,ticker,market,open,high,low,close,volume)"
    )
    con.execute(f"CREATE VIEW feat_price AS {price.build_price_sql()}")
    (shock,) = con.execute(
        "SELECT px_turnover_shock FROM feat_price WHERE trade_date = DATE '2020-01-01'+64"
    ).fetchone()
    assert shock is None


# --- feat_flow --------------------------------------------------------------


@pytest.fixture()
def flow_con(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """Two sessions of foreign net-buy with a KRX/PYKRX duplicate (zero conflict),
    plus a short-balance value only on the 2nd day (coverage gap)."""
    pq = tmp_path / "flow.parquet"
    # (trade_date, ticker, market, metric_code, value, source)
    rows = [
        ("2020-01-01", "005930", "KOSPI", "foreign_net_buy_volume", 100.0, "KRX"),
        ("2020-01-01", "005930", "KOSPI", "foreign_net_buy_volume", 100.0, "PYKRX"),  # dup
        ("2020-01-02", "005930", "KOSPI", "foreign_net_buy_volume", 50.0, "PYKRX"),  # only PYKRX
        ("2020-01-01", "005930", "KOSPI", "short_selling_volume", 7.0, "KRX"),
        ("2020-01-02", "005930", "KOSPI", "short_selling_balance_quantity", 999.0, "KRX"),
    ]
    values = ",".join(
        f"(DATE '{d}', '{tk}', '{mk}', '{mc}', {val}, '{src}')"
        for (d, tk, mk, mc, val, src) in rows
    )
    _write_parquet(
        pq,
        "SELECT * FROM (VALUES " + values + ") AS t"
        "(trade_date, ticker, market, metric_code, value, source)",
    )
    con = duckdb.connect()
    con.execute(
        "CREATE VIEW krx_security_flow_raw AS "
        f"SELECT * FROM read_parquet('{pq.as_posix()}', hive_partitioning=false)"
    )
    return con


def test_flow_dedup_keeps_krx_and_dedups(flow_con) -> None:
    dedup = flow.build_dedup_sql("krx_security_flow_raw")
    rows = flow_con.execute(
        f"SELECT trade_date, metric_code, value FROM ({dedup}) "
        "WHERE metric_code='foreign_net_buy_volume' ORDER BY trade_date"
    ).fetchall()
    # d1: KRX/PYKRX duplicate collapses to one row (KRX wins, value identical).
    # d2: single PYKRX row survives (KRX absent).
    assert len(rows) == 2
    assert rows[0][2] == pytest.approx(100.0)
    assert rows[1][2] == pytest.approx(50.0)


def test_flow_pivot_and_cumulative_netbuy(flow_con) -> None:
    flow_con.execute(f"CREATE VIEW feat_flow AS {flow.build_flow_sql('krx_security_flow_raw')}")
    rows = flow_con.execute(
        "SELECT trade_date, flow_foreign_netbuy_sum_5d FROM feat_flow ORDER BY trade_date"
    ).fetchall()
    assert rows[0][1] == pytest.approx(100.0)
    assert rows[1][1] == pytest.approx(150.0)  # 100 + 50 cumulative


def test_flow_short_balance_null_before_coverage(flow_con) -> None:
    flow_con.execute(f"CREATE VIEW feat_flow AS {flow.build_flow_sql('krx_security_flow_raw')}")
    rows = dict(
        flow_con.execute(
            "SELECT trade_date, flow_short_balance_qty FROM feat_flow ORDER BY trade_date"
        ).fetchall()
    )
    # d1 has no balance metric -> NULL; d2 has 999 (coverage starts later, §3.2).
    assert rows[datetime.date(2020, 1, 1)] is None
    assert rows[datetime.date(2020, 1, 2)] == pytest.approx(999.0)


def test_flow_metric_codes_count() -> None:
    assert len(flow.METRIC_CODES) == 7


# --- A0 input lineage: feat_price's SQL text is a cache key ---


def test_feat_price_sql_text_is_frozen() -> None:
    """``mart._sql_hash`` keys the A0 mart cache on this exact string, and every
    published Phase A/B run records the lineage that follows from it.

    So a change here is never cosmetic: it invalidates the materialized mart
    and detaches new runs from the canonical ones. These two hashes are the
    text as of the Stage 1a work, which extracted the shared market-model CTE
    into ``trading_panel.build_market_model_sql`` — extraction only, byte for
    byte. Update them deliberately, with the rebuild that implies.
    """
    import hashlib

    def _hash(**kwargs) -> str:
        return hashlib.sha256(price.build_price_sql(**kwargs).encode()).hexdigest()

    assert _hash() == "6886ddd45d33c6a13f0d1fd1653a8dd37518bfaa5189ae1227f853bb3b2a4bf3"
    assert (
        _hash(quality_view="dim_price_quality_daily")
        == "110617482d64a50d4ef92ad70defcc23b5b276e2a025452f92a845ccbacec228"
    )


def test_feat_price_and_feat_macro_exposure_share_one_market_model() -> None:
    """``px_idio_vol_60d`` and every ``macro_beta_*`` are statements about the
    same ``resid_ret``; they must not drift into two definitions of it."""
    from modeler.etl.features.macro_exposure import build_macro_exposure_sql
    from modeler.etl.trading_panel import build_market_model_sql

    assert build_market_model_sql("valid_q") in price.build_price_sql()
    assert build_market_model_sql("valid") in build_macro_exposure_sql()
