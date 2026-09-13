"""Unit tests for P2 — dim_trading_calendar, dim_universe_daily, mart helpers.

A tiny synthetic ``daily_ohlcv`` lake is built on disk (no DB, no real lake).
Thresholds are shrunk via :class:`UniverseFilter` so each filter boundary is
checkable on a handful of rows.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import duckdb
import pytest

from modeler.etl import calendar, mart, universe
from modeler.etl.config import DataRoot, EngineOptions, LakeConfig
from modeler.etl.lake import connect, register_views

# (trade_date, ticker, market, open, high, low, close, volume)
Row = tuple[str, str, str, int, int, int, int, int]


def _write_ohlcv(lake_root: Path, snapshot_date: str, source: str, rows: list[Row]) -> None:
    table_dir = (
        lake_root
        / "raw"
        / "raw_postgres"
        / f"snapshot_date={snapshot_date}"
        / f"source={source}"
        / "daily_ohlcv"
        / "schema_version=1"
        / "year=2020"
        / "month=01"
    )
    table_dir.mkdir(parents=True, exist_ok=True)
    out = (table_dir / "part-000000.parquet").as_posix()

    values = ",\n".join(
        f"(DATE '{d}', '{tk}', '{mk}', {o}, {h}, {lo}, {c}, {v})"
        for (d, tk, mk, o, h, lo, c, v) in rows
    )
    con = duckdb.connect()
    con.execute(f"""
        COPY (
          SELECT * FROM (VALUES
            {values}
          ) AS t(trade_date, ticker, market, open, high, low, close, volume)
        ) TO '{out}' (FORMAT PARQUET)
        """)
    con.close()


def _dates(n: int) -> list[str]:
    # n consecutive January 2020 days (ordering is all that matters).
    return [f"2020-01-{i:02d}" for i in range(1, n + 1)]


@pytest.fixture()
def lake_with_prices(tmp_path: Path):
    """Two tickers: A (clean) on KOSPI, B (halt + low-liquidity) on KOSDAQ."""
    d = _dates(6)
    rows: list[Row] = []
    # Ticker A: 6 clean sessions, turnover 200 each (close 10 * volume 20).
    for day in d:
        rows.append((day, "A", "KOSPI", 10, 11, 9, 10, 20))
    # Ticker B (KOSDAQ): d1 halted (OHL=0), d2 low-liquidity (turnover 1), rest ok.
    rows.append((d[0], "B", "KOSDAQ", 0, 0, 0, 5, 10))  # halted
    rows.append((d[1], "B", "KOSDAQ", 1, 1, 1, 1, 1))  # turnover 1
    for day in d[2:]:
        rows.append((day, "B", "KOSDAQ", 10, 11, 9, 10, 20))

    snapshot_date, source = "2020-01-01", "local_mydb"
    _write_ohlcv(tmp_path, snapshot_date, source, rows)
    cfg = LakeConfig(
        root=DataRoot(tmp_path),
        snapshot_date=snapshot_date,
        source=source,
        engine=EngineOptions(threads=2),
    )
    con = connect(cfg)
    register_views(con, cfg, tables=["daily_ohlcv"])
    return con, cfg


# --- calendar ---------------------------------------------------------------


def test_calendar_d_idx_is_dense_per_market(lake_with_prices) -> None:
    con, cfg = lake_with_prices
    calendar.materialize_calendar(con, cfg)

    rows = con.execute(
        "SELECT market, trade_date, d_idx FROM dim_trading_calendar " "ORDER BY market, trade_date"
    ).fetchall()

    kospi = [r for r in rows if r[0] == "KOSPI"]
    kosdaq = [r for r in rows if r[0] == "KOSDAQ"]
    # Both markets observe the same 6 trading dates -> dense 1..6 each.
    assert [r[2] for r in kospi] == [1, 2, 3, 4, 5, 6]
    assert [r[2] for r in kosdaq] == [1, 2, 3, 4, 5, 6]


def test_calendar_dedupes_dates_across_tickers(lake_with_prices) -> None:
    con, cfg = lake_with_prices
    calendar.materialize_calendar(con, cfg)
    (n,) = con.execute(
        "SELECT count(*) FROM dim_trading_calendar WHERE market = 'KOSPI'"
    ).fetchone()
    assert n == 6  # 6 distinct dates, not 6 rows-per-ticker


# --- universe ---------------------------------------------------------------


@pytest.fixture()
def small_filter() -> universe.UniverseFilter:
    return universe.UniverseFilter(
        warmup_window=3,
        warmup_min_valid=2,
        liquidity_window=3,
        min_liquidity_krw=100.0,
        label_horizon=2,
    )


def _flags(con, ticker: str) -> dict[str, dict]:
    rows = con.execute(
        "SELECT trade_date, not_halted, warmup_ok, liquidity_ok, label_ok, in_universe "
        f"FROM dim_universe_daily WHERE ticker = '{ticker}' ORDER BY trade_date"
    ).fetchall()
    cols = ["not_halted", "warmup_ok", "liquidity_ok", "label_ok", "in_universe"]
    return {str(r[0]): dict(zip(cols, r[1:])) for r in rows}


def test_universe_warmup_and_label_boundaries(lake_with_prices, small_filter) -> None:
    con, cfg = lake_with_prices
    universe.materialize_universe(con, cfg, small_filter)
    a = _flags(con, "A")
    days = sorted(a)

    # warmup_min_valid=2, window=3: first day has only 1 valid day -> false.
    assert a[days[0]]["warmup_ok"] is False
    assert a[days[1]]["warmup_ok"] is True

    # label_horizon=2 over 6 non-halt sessions: d_idx_nh <= 4 has t+2 -> true;
    # last two sessions cannot produce a label.
    assert a[days[3]]["label_ok"] is True
    assert a[days[4]]["label_ok"] is False
    assert a[days[5]]["label_ok"] is False

    # Clean ticker is in-universe once warmed up (label_ok excluded from gate).
    assert a[days[0]]["in_universe"] is False  # warmup not met
    assert a[days[2]]["in_universe"] is True


def test_universe_halt_and_liquidity_flags(lake_with_prices, small_filter) -> None:
    con, cfg = lake_with_prices
    universe.materialize_universe(con, cfg, small_filter)
    b = _flags(con, "B")
    days = sorted(b)

    # d1 halted (OHL=0): not_halted false, label_ok false, never in-universe.
    assert b[days[0]]["not_halted"] is False
    assert b[days[0]]["label_ok"] is False
    assert b[days[0]]["in_universe"] is False

    # d2 turnover=1 (< floor 100): liquidity_ok false -> out.
    assert b[days[1]]["liquidity_ok"] is False
    assert b[days[1]]["in_universe"] is False


# --- mart helpers -----------------------------------------------------------


def test_mart_materialize_is_idempotent_and_force_rebuilds(lake_with_prices) -> None:
    con, cfg = lake_with_prices
    assert mart.is_materialized(cfg, calendar.CALENDAR_TABLE) is False

    calendar.materialize_calendar(con, cfg)
    assert mart.is_materialized(cfg, calendar.CALENDAR_TABLE) is True
    first = sorted(mart.mart_table_dir(cfg, calendar.CALENDAR_TABLE).rglob("*.parquet"))
    assert first  # at least one part file

    # Second call without force is a no-op (skip): files unchanged.
    calendar.materialize_calendar(con, cfg)
    again = sorted(mart.mart_table_dir(cfg, calendar.CALENDAR_TABLE).rglob("*.parquet"))
    assert again == first

    # force rebuilds without error and stays queryable.
    calendar.materialize_calendar(con, cfg, force=True)
    (n,) = con.execute("SELECT count(*) FROM dim_trading_calendar").fetchone()
    assert n == 12  # 6 dates x 2 markets


def test_register_mart_view_before_materialize_raises(lake_with_prices) -> None:
    con, cfg = lake_with_prices
    with pytest.raises(FileNotFoundError):
        mart.register_mart_view(con, cfg, "dim_universe_daily")


def test_mart_cache_hash_mismatch_requires_rebuild(lake_with_prices) -> None:
    con, cfg = lake_with_prices
    mart.materialize(con, cfg, "cache_contract", "SELECT 1 AS value")
    changed = replace(cfg, analysis_config_hash="changed")
    with pytest.raises(RuntimeError, match="contract mismatch"):
        mart.materialize(con, changed, "cache_contract", "SELECT 1 AS value")
