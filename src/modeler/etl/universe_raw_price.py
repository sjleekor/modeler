"""Formation universe with the price floor on the KRX *raw* close — C1-4a.

``etl/universe.py`` reads ``daily_ohlcv`` and applies ``close >= min_close_krw``
(1,000원 in ``tradable_universe_filter``) to ``daily_ohlcv.close``. That column
is the Naver/pykrx **adjusted** close, retro-adjusted every time a later reverse
split, capital reduction (감자), merger-ratio change or split happens. A row's
price level therefore carries information from after the row's date:

  * a name that later does a reverse split or 감자 (usually a distressed penny
    stock) has a history whose adjusted close is a multiple of the price people
    actually paid, so its raw-800원 rows pass the 1,000원 floor;
  * a name that later splits has adjusted closes below the raw ones, so rows
    that were raw-1,000원-or-more are wrongly dropped.

On the 2026-09-30 lake (2015-01..2025-07, non-halt) the first group is 212,036
rows and the second 12,966 rows (``my/.../16_c1_label_benchmark.md`` §9,
``17_c2_feature_set.md`` §8.1). KRX's own ``daily_market_cap.source_close`` is the
close as traded that day and is never revised, so the floor is applied there.

What changes. The price floor (``close_ok``, ``tradable_ok``, ``in_universe`` when
``min_close_krw > 0``) and, as a side effect, the liquidity floor. The raw view built
here replaces ``close`` by the KRX raw close and ``volume`` by the KRX raw volume as
a *pair*, so ``turnover = close * volume`` is raw close x raw volume, which is the KRX
trading value (median |log error| 0.6%). For a pure split or reverse split the adjusted
close x adjusted volume is the same number (price and volume are adjusted in inverse
proportion), so nothing changes there (a test pins it). For rights and bonus issues the
price is adjusted but the volume is not (2015-01..2025-07: median price ratio 0.947,
volume ratio 1.0), so the adjusted product is more than 5% off the trading value in
about 1.0M of 5.8M non-halt rows (20% of the rows in 2015, about 0 from 2023). On those
rows the liquidity flag can flip: 60,843 of 6.08M rows (1.0%, tradable filter,
2015-01..2025-07), mostly "raw passes, adjusted fails" in the early years. ``open``/``high``/
``low`` stay from ``daily_ohlcv``: the halt flag is ``open = high = low = 0``, and the
halt rows of the two sources do not line up exactly.

Where the KRX row is missing (``daily_market_cap`` starts 2014-06-02 and skips
about 1% of later rows) or has no positive ``source_close`` or no volume, the pair
falls back to the adjusted ``daily_ohlcv`` values and ``raw_price_missing`` is TRUE.
For those rows the floor is still the old, adjusted one.

Join key. ``daily_market_cap`` is joined to ``daily_ohlcv`` on ``(trade_date,
ticker)`` only, not ``market``: a name that moves between KOSDAQ and KOSPI carries
one market label in ``daily_ohlcv`` (the current one, all history) and the old one
in ``daily_market_cap`` (25,407 rows to 2025-07), and the universe partitions on
the ``daily_ohlcv`` label. Each source has one row per ``(trade_date, ticker)``;
:func:`assert_raw_price_inputs_unique` checks it, because a duplicate would
multiply universe rows.

TRAIN vs SERVING. On the serving day t the adjustment factor is 1, so adjusted and
raw closes are the same number and today's ranking is unchanged. Only the training
history differs (later splits/reductions are not yet known on day t, but they are
in the history a model trains on). The serving side still needs the day-t
``daily_market_cap`` row, otherwise it falls back to the adjusted value, which is
the same number that day.

NEXT PREREGISTRATION. The next model's universe is this module together with
``universe_terminal`` (정리매매 tail exclusion)::

    sql = build_universe_v2_sql(build_universe_raw_price_sql(flt), label_view)

``universe.py`` itself is not touched (it is a model02 code-hash file).
"""

from __future__ import annotations

import re

import duckdb

from modeler.etl.universe import UniverseFilter, build_universe_sql

KEYS: tuple[str, ...] = ("trade_date", "ticker", "market")
RAW_MISSING_COL = "raw_price_missing"
_PAIR_KEYS = ("trade_date", "ticker")
_RAW_ALIAS = "raw_px"

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


def _ident(value: str, what: str) -> str:
    if not _IDENT.match(value):
        raise ValueError(f"{what} must be a plain SQL identifier, got {value!r}")
    return value


def build_raw_price_view_sql(
    price_view: str = "daily_ohlcv",
    market_cap_view: str = "daily_market_cap",
) -> str:
    """SELECT with the columns ``universe.build_universe_sql`` reads from its price view.

    One row per ``price_view`` row, same keys (``trade_date``, ``ticker``, ``market``)
    and ``open``/``high``/``low``; ``close`` and ``volume`` are the KRX raw pair where
    ``market_cap_view`` has ``source_close > 0`` and a ``volume``, else the adjusted
    ``price_view`` values. ``raw_price_missing`` is TRUE for the fallback rows.
    """
    pv = _ident(price_view, "price_view")
    mv = _ident(market_cap_view, "market_cap_view")
    raw_ok = "COALESCE(m.source_close > 0 AND m.volume IS NOT NULL, FALSE)"
    return f"""
        SELECT
            o.trade_date, o.ticker, o.market,
            o.open, o.high, o.low,
            CASE WHEN {raw_ok} THEN m.source_close ELSE o.close END AS close,
            CASE WHEN {raw_ok} THEN m.volume ELSE o.volume END AS volume,
            NOT {raw_ok} AS {RAW_MISSING_COL}
        FROM {pv} o
        LEFT JOIN {mv} m
          ON m.trade_date = o.trade_date AND m.ticker = o.ticker
    """


def build_universe_raw_price_sql(
    flt: UniverseFilter,
    *,
    price_view: str = "daily_ohlcv",
    market_cap_view: str = "daily_market_cap",
) -> str:
    """``universe.build_universe_sql(flt)`` over the raw-price view, plus ``raw_price_missing``.

    Same columns as ``build_universe_sql`` (same names and meaning, including
    ``in_universe``) followed by ``raw_price_missing`` (BOOLEAN, TRUE where the price
    floor and turnover had to use the adjusted values). One row per
    ``(trade_date, ticker, market)`` of ``price_view``. The result is a WITH
    statement, so ``universe_terminal.build_universe_v2_sql`` accepts it as is.
    """
    raw = build_raw_price_view_sql(price_view, market_cap_view)
    universe = build_universe_sql(flt, price_view=_RAW_ALIAS)
    on = " AND ".join(f"f.{k} = u.{k}" for k in KEYS)
    return f"""
        WITH {_RAW_ALIAS} AS (
            {raw.strip()}
        )
        SELECT u.*, f.{RAW_MISSING_COL}
        FROM (
            {universe.strip()}
        ) u
        LEFT JOIN {_RAW_ALIAS} f ON {on}
    """


def assert_raw_price_inputs_unique(
    con: duckdb.DuckDBPyConnection,
    price_view: str = "daily_ohlcv",
    market_cap_view: str = "daily_market_cap",
) -> None:
    """Fail closed if either input has two rows for one key (they would multiply rows).

    ``price_view`` must be unique on ``(trade_date, ticker, market)`` and
    ``market_cap_view`` on ``(trade_date, ticker)``.
    """
    for view, keys in (
        (_ident(price_view, "price_view"), KEYS),
        (_ident(market_cap_view, "market_cap_view"), _PAIR_KEYS),
    ):
        key_list = ", ".join(keys)
        extra = con.execute(
            f"SELECT count(*) - count(DISTINCT ({key_list})) FROM {view}"  # noqa: S608
        ).fetchone()[0]
        if extra:
            raise AssertionError(f"{view} has {extra} duplicate ({key_list}) rows")
