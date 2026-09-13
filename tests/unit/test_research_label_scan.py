from __future__ import annotations

import duckdb
import pytest
from modeler.etl.labels import build_label_scan_sql


def test_scan_bucket_identity_and_null_rank() -> None:
    con = duckdb.connect()
    rows = []
    for i, close in enumerate((100, 110, 121, 133.1, 146.41, 161.051), start=1):
        rows.append((f"2024-01-{i:02d}", "A", "KOSPI", close))
        rows.append((f"2024-01-{i:02d}", "B", "KOSPI", 100.0))
    values = ",".join(
        f"(DATE '{d}', '{t}', '{m}', 1, 1, 1, {c}, 10)" for d, t, m, c in rows
    )
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + values
        + ") t(trade_date,ticker,market,open,high,low,close,volume)"
    )
    sql = build_label_scan_sql(
        horizons=(1, 2, 3, 5),
        buckets=((0, 1), (1, 2), (2, 3)),
        holdout_start="2025-08-01",
    )
    con.execute(f"CREATE VIEW label_scan AS {sql}")
    row = con.execute(
        "SELECT fwd_ret_2d, bucket_ret_1_2d, y_rank_2d FROM label_scan "
        "WHERE ticker='A' AND trade_date=DATE '2024-01-01'"
    ).fetchone()
    assert row[0] == pytest.approx(1.21 - 1)
    assert row[1] == pytest.approx((1 + row[0]) / (1 + (1.1 - 1)) - 1)
    assert row[2] is not None
    null_row = con.execute(
        "SELECT y_rank_5d FROM label_scan WHERE trade_date=DATE '2024-01-06' AND ticker='A'"
    ).fetchone()
    assert null_row[0] is None


def test_scan_keeps_market_excess_mean_zero() -> None:
    con = duckdb.connect()
    rows = []
    for i, values in enumerate(((100, 100), (110, 105), (121, 110), (133, 120)), start=1):
        for ticker, close in zip(("A", "B"), values):
            rows.append((f"2024-02-{i:02d}", ticker, close))
    values = ",".join(f"(DATE '{d}', '{t}', 'KOSPI', 1,1,1,{c},10)" for d, t, c in rows)
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + values
        + ") t(trade_date,ticker,market,open,high,low,close,volume)"
    )
    con.execute(
        "CREATE VIEW label_scan AS "
        + build_label_scan_sql(horizons=(1, 2), buckets=((0, 1),), holdout_start="2025-08-01")
    )
    (mean_excess,) = con.execute(
        "SELECT avg(raw_label_1d) FROM label_scan WHERE trade_date=DATE '2024-02-01'"
    ).fetchone()
    assert mean_excess == pytest.approx(0.0, abs=1e-12)


def test_rank_population_excludes_masked_endpoint_rows() -> None:
    con = duckdb.connect()
    rows = []
    for ticker, closes in {
        "A": (100, 110),
        "B": (100, 105),
        "C": (100, 95),
    }.items():
        for i, close in enumerate(closes, start=1):
            rows.append(f"(DATE '2024-03-0{i}', '{ticker}', 'KOSPI', 1,1,1,{close},10)")
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + ",".join(rows)
        + ") t(trade_date,ticker,market,open,high,low,close,volume)"
    )
    con.execute(
        "CREATE VIEW quality AS SELECT trade_date,ticker,market, "
        "(ticker='A' AND trade_date=DATE '2024-03-02') AS ca_event "
        "FROM daily_ohlcv"
    )
    con.execute(
        "CREATE VIEW label_scan AS "
        + build_label_scan_sql(
            horizons=(1,), quality_view="quality", buckets=((0, 1),), holdout_start="2025-08-01"
        )
    )
    rows = con.execute(
        "SELECT ticker, y_rank_1d FROM label_scan "
        "WHERE trade_date=DATE '2024-03-01' ORDER BY ticker"
    ).fetchall()
    assert rows == [("A", None), ("B", pytest.approx(1.0)), ("C", pytest.approx(0.0))]
