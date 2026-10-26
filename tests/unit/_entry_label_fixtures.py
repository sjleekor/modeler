"""Shared synthetic price panels for the entry-label helper tests.

Used by ``test_splits_label_end.py``, ``test_label_gate.py`` and
``test_universe_terminal.py`` to get *real* ``labels_entry`` output (column names,
kinds, ``label_end_date``, ``in_terminal_tail``) on a few hand-made tickers.
"""

from __future__ import annotations

import datetime

import duckdb
import polars as pl

from modeler.etl.labels import LabelSpec
from modeler.etl.labels_entry import build_entry_label_sql


def price_con(rows: list[tuple]) -> duckdb.DuckDBPyConnection:
    """rows: (date, ticker, open, high, low, close); market KOSPI, volume 1."""
    con = duckdb.connect()
    values = ",".join(
        f"(DATE '{d}', '{t}', 'KOSPI', {o}, {h}, {lo}, {c}, 1)" for d, t, o, h, lo, c in rows
    )
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + values
        + ") AS t(trade_date, ticker, market, open, high, low, close, volume)"
    )
    return con


def series(ticker: str, start: str, closes: list) -> list[tuple]:
    """Consecutive calendar days; ``None`` = halted day (0, 0, 0, 0)."""
    first = datetime.date.fromisoformat(start)
    out = []
    for i, c in enumerate(closes):
        d = (first + datetime.timedelta(days=i)).isoformat()
        bar = (0, 0, 0, 0) if c is None else (c - 1, c + 1, c - 2, c)
        out.append((d, ticker, *bar))
    return out


def delisted_after_long_halt(n_market: int = 60) -> list[tuple]:
    """A: 5 trading days, 30 halted days, then 7 tail days (정리매매), then gone.
    B: a flat ticker that keeps the market calendar going for ``n_market`` days."""
    tail = [104 * 0.6**k for k in range(1, 8)]
    rows = series("A", "2020-01-02", [100, 101, 102, 103, 104] + [None] * 30 + tail)
    return rows + series("B", "2020-01-02", [10] * n_market)


def entry_frame(
    con: duckdb.DuckDBPyConnection, spec: LabelSpec, entry: str = "c1", **kw: object
) -> pl.DataFrame:
    """``labels_entry`` output as a polars frame ordered by (ticker, trade_date)."""
    sql = build_entry_label_sql(spec, entry, **kw)  # type: ignore[arg-type]
    return pl.from_arrow(
        con.execute(f"SELECT * FROM ({sql}) ORDER BY ticker, trade_date").to_arrow_table()
    )  # type: ignore[return-value]
