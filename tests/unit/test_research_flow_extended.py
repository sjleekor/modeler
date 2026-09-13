from __future__ import annotations

import duckdb
import pytest
from modeler.etl.features.flow import build_flow_sql


def test_flow_ratio_joins_price_and_float_without_zero_fill() -> None:
    con = duckdb.connect()
    flow_rows = []
    for i in range(1, 21):
        d = f"2024-02-{i:02d}"
        for code, value in (
            ("foreign_net_buy_volume", 10.0),
            ("institution_net_buy_volume", 5.0),
            ("individual_net_buy_volume", -2.0),
            ("foreign_holding_shares", 100.0 + i),
            ("short_selling_volume", 2.0),
            ("short_selling_balance_quantity", 20.0),
        ):
            flow_rows.append(f"(DATE '{d}', 'A', 'KOSPI', '{code}', {value}, 'KRX')")
    con.execute(
        "CREATE VIEW krx_security_flow_raw AS SELECT * FROM (VALUES "
        + ",".join(flow_rows)
        + ") t(trade_date,ticker,market,metric_code,value,source)"
    )
    price_rows = ",".join(
        f"(DATE '2024-02-{i:02d}', 'A', 'KOSPI', 100,100,100,100,{1000 + i})"
        for i in range(1, 21)
    )
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + price_rows
        + ") t(trade_date,ticker,market,open,high,low,close,volume)"
    )
    pit_rows = ",".join(
        f"(DATE '2024-02-{i:02d}', 'A', 'KOSPI', 10000.0)" for i in range(1, 21)
    )
    con.execute(
        "CREATE VIEW dim_stock_pit_daily AS SELECT * FROM (VALUES "
        + pit_rows
        + ") t(trade_date,ticker,market,float_shares_pit)"
    )
    con.execute(
        "CREATE VIEW dim_price_quality_daily AS SELECT trade_date,ticker,market, "
        "FALSE AS short_balance_is_available, 'allowed' AS short_regime, "
        "ROW_NUMBER() OVER (PARTITION BY ticker, market ORDER BY trade_date) AS valid_session_idx "
        "FROM daily_ohlcv"
    )
    con.execute(
        "CREATE VIEW feat_flow AS "
        + build_flow_sql(
            price_view="daily_ohlcv",
            pit_view="dim_stock_pit_daily",
            quality_view="dim_price_quality_daily",
        )
    )
    row = con.execute(
        "SELECT flow_foreign_netbuy_to_volume_5d, flow_short_turnover_20d, "
        "flow_foreign_netbuy_to_volume_5d_lag1 FROM feat_flow "
        "WHERE trade_date=DATE '2024-02-20'"
    ).fetchone()
    assert row[0] == pytest.approx(50 / sum(1000 + i for i in range(16, 21)))
    assert row[1] == pytest.approx(40 / sum(1000 + i for i in range(1, 21)))
    assert row[2] == pytest.approx(50 / sum(1000 + i for i in range(15, 20)))


def test_flow_rolling_window_skips_halt_days_like_price_features_do() -> None:
    """A raw flow row present on a price-halt day must not consume one of the
    20 rolling-window slots, and must not become anyone's "prior valid
    session" for _lag1 — mirrors research/etl/trading_panel.py's halt
    exclusion for price features (04_specific_plan_A.md §A-1 test list:
    "A0 native/lag1 mapping과 valid-session shift invariant")."""
    con = duckdb.connect()
    # 22 consecutive calendar days; day 10 is a halt in price terms but KRX
    # still recorded a flow observation for it (the real-world discrepancy).
    flow_rows = [
        f"(DATE '2024-04-{i:02d}', 'A', 'KOSPI', 'foreign_net_buy_volume', {float(i)}, 'KRX')"
        for i in range(1, 23)
    ]
    con.execute(
        "CREATE VIEW krx_security_flow_raw AS SELECT * FROM (VALUES "
        + ",".join(flow_rows)
        + ") t(trade_date,ticker,market,metric_code,value,source)"
    )
    price_rows = ",".join(
        (
            f"(DATE '2024-04-{i:02d}', 'A', 'KOSPI', 0,0,0,100,0)"
            if i == 10
            else f"(DATE '2024-04-{i:02d}', 'A', 'KOSPI', 100,100,100,100,100)"
        )
        for i in range(1, 23)
    )
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + price_rows
        + ") t(trade_date,ticker,market,open,high,low,close,volume)"
    )
    con.execute(
        "CREATE VIEW dim_stock_pit_daily AS SELECT trade_date, ticker, market, "
        "10000.0 AS float_shares_pit FROM daily_ohlcv"
    )
    con.execute("""
        CREATE VIEW dim_price_quality_daily AS
        WITH valid AS (
            SELECT trade_date, ticker, market,
                   ROW_NUMBER() OVER (PARTITION BY ticker, market ORDER BY trade_date)
                       AS valid_session_idx
            FROM daily_ohlcv
            WHERE NOT (open = 0 AND high = 0 AND low = 0)
        )
        SELECT d.trade_date, d.ticker, d.market,
               FALSE AS short_balance_is_available, 'allowed' AS short_regime,
               v.valid_session_idx
        FROM daily_ohlcv d
        LEFT JOIN valid v USING (trade_date, ticker, market)
    """)
    con.execute(
        "CREATE VIEW feat_flow AS "
        + build_flow_sql(
            price_view="daily_ohlcv",
            pit_view="dim_stock_pit_daily",
            quality_view="dim_price_quality_daily",
        )
    )

    # The halt day itself must not appear in feat_flow at all (no valid session).
    (halt_day_count,) = con.execute(
        "SELECT count(*) FROM feat_flow WHERE trade_date = DATE '2024-04-10'"
    ).fetchone()
    assert halt_day_count == 0

    # 20-valid-session window at day 22 must be days {2..9, 11..22} (skipping
    # the halt at day 10), sum = 242 — NOT 250, which is what the last 20
    # *rows* (days 3..22, including the halt) would wrongly give.
    (sum_20d, native_lag1) = con.execute(
        "SELECT flow_foreign_netbuy_sum_20d, flow_foreign_netbuy_to_volume_20d_lag1 "
        "FROM feat_flow WHERE trade_date = DATE '2024-04-22'"
    ).fetchone()
    assert sum_20d == pytest.approx(242.0)

    # day 11's lag1 (its "prior valid session") must be day 9's native ratio,
    # not day 10's (which doesn't exist as a valid session at all).
    day9_native, day11_lag1 = con.execute("""
        SELECT
            (SELECT flow_foreign_netbuy_to_volume_5d FROM feat_flow
             WHERE trade_date = DATE '2024-04-09'),
            (SELECT flow_foreign_netbuy_to_volume_5d_lag1 FROM feat_flow
             WHERE trade_date = DATE '2024-04-11')
    """).fetchone()
    assert day11_lag1 == pytest.approx(day9_native)


def test_nat_proxy_excludes_unavailable_names_from_cross_sectional_rank() -> None:
    con = duckdb.connect()
    flow_rows = []
    for i in range(1, 22):
        d = f"2024-03-{i:02d}"
        for ticker, holding in (("A", 100.0 + i), ("B", 200.0 + i)):
            for code, value in (
                ("foreign_holding_shares", holding),
                ("short_selling_balance_quantity", 20.0 + i),
                ("short_selling_volume", 2.0),
            ):
                flow_rows.append(f"(DATE '{d}', '{ticker}', 'KOSPI', '{code}', {value}, 'KRX')")
    con.execute(
        "CREATE VIEW krx_security_flow_raw AS SELECT * FROM (VALUES "
        + ",".join(flow_rows)
        + ") t(trade_date,ticker,market,metric_code,value,source)"
    )
    price_rows = ",".join(
        f"(DATE '2024-03-{i:02d}', '{ticker}', 'KOSPI', 100,100,100,100,1000)"
        for i in range(1, 22) for ticker in ("A", "B")
    )
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + price_rows
        + ") t(trade_date,ticker,market,open,high,low,close,volume)"
    )
    pit_rows = ",".join(
        f"(DATE '2024-03-{i:02d}', '{ticker}', 'KOSPI', 10000.0)"
        for i in range(1, 22) for ticker in ("A", "B")
    )
    con.execute(
        "CREATE VIEW dim_stock_pit_daily AS SELECT * FROM (VALUES "
        + pit_rows
        + ") t(trade_date,ticker,market,float_shares_pit)"
    )
    con.execute(
        "CREATE VIEW dim_price_quality_daily AS SELECT trade_date,ticker,market, "
        "(ticker='A') AS short_balance_is_available, 'allowed' AS short_regime, "
        "ROW_NUMBER() OVER (PARTITION BY ticker, market ORDER BY trade_date) AS valid_session_idx "
        "FROM daily_ohlcv"
    )
    con.execute(
        "CREATE VIEW feat_flow AS "
        + build_flow_sql(
            price_view="daily_ohlcv",
            pit_view="dim_stock_pit_daily",
            quality_view="dim_price_quality_daily",
        )
    )
    rows = con.execute(
        "SELECT ticker, flow_nat_proxy_20d FROM feat_flow "
        "WHERE trade_date=DATE '2024-03-21' ORDER BY ticker"
    ).fetchall()
    assert rows[0][1] is not None
    assert rows[1][1] is None
