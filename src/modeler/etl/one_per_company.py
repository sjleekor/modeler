"""One listed line per company and date — C3 ``keep_one_per_company``.

A Korean company can have several listed lines on one date: the common stock
(보통주) and one or more preferred stocks (우선주, 2우B ...). They move together,
so in a cross-section they are close to duplicate rows of one issuer. The rule
(``08_decisions_and_execution.md`` §3, §7.5) keeps one row per ``(trade_date,
corp_code)``:

  1. the common stock, if it is in the universe that day;
  2. otherwise the in-universe line with the larger trailing 20-session KRX
     trading value (ties by ticker, then market).

``keep_one_per_company`` is TRUE for exactly that row and FALSE for every other
row, including rows that are not in the universe. It never turns a row on that
the universe left out.

corp_code mapping (checked on the 2026-09-30 lake)
--------------------------------------------------
``dart_corp_master`` holds one row per ``ticker`` and carries delisted companies
too (3,959 tickers, ``ticker`` unique), so historical tickers map; the table is
"current" only in that a ticker is never re-assigned to another corp_code, which
the master cannot show. **Only common-stock tickers are in it**: of the 3,213
tickers of ``daily_ohlcv`` the 3,051 ending in ``0`` map 3,019 and **all 162
others map 0** (OpenDART gives a corp code to the common stock's ``stock_code``
only). A preferred stock therefore takes the corp_code of its common stock, whose
ticker is the same first five characters plus ``0``::

    005935 (삼성전자우)  -> 005930          00680K (미래에셋증권2우B) -> 006800

That sibling exists in the master for 161 of the 162 non-``0`` tickers. A ticker
with no corp_code (neither itself nor its sibling in the master) gets
``corp_code = NULL`` and ``corp_code_missing = TRUE``; it is kept (one row per
ticker) because there is nothing to group it by. Those are mostly listings newer
than the master (SPACs, 2024-25 IPOs) and the odd preferred whose common stock
is itself unmapped (``0220WL`` / ``0220W0``); two such lines of one company stay
two rows.

Common vs preferred. ``is_common_stock`` = the ticker's last character is ``0``.
In the lake the last character is ``0`` for 3,051 tickers (common) and ``5``
(112), ``7`` (19), ``9`` (6), ``K`` (20), ``L`` (4), ``M`` (1) for preferred and
the like (``…5`` first preferred, ``…7`` / ``…9`` 2nd/3rd, ``…K`` / ``…L`` /
``…M`` the alphanumeric codes of newer preferred lines). It is a ticker-format
rule, not a name or share-class field; the master has no class column.

Trailing trading value. ``trailing_tv20`` is the mean of the KRX
``daily_market_cap.trading_value`` over the 20 most recent rows of that ticker up
to and including the date (rows missing in the lake are skipped, null values are
ignored, so a short window is a mean of what exists). ``daily_market_cap`` is
joined on ``(trade_date, ticker)`` only, not ``market``, like
``universe_raw_price``. It is only a tie-breaker between lines without a common
stock, so a mean and a sum rank identically except at the start of a series.

Two forms, same result (pinned by a test): :func:`build_one_per_company_sql`
(wraps a universe view or ``build_universe_v2_sql`` statement) and
:func:`with_one_per_company` (polars). The next model's universe::

    sql = build_one_per_company_sql(
        build_universe_v2_sql(build_universe_raw_price_sql(flt), label_view)
    )

TRAIN vs SERVING. Nothing here looks forward: corp_code is a static map and the
trading value is trailing, so day-t serving gets the same rule. The only
caveat is the master's currentness above.
"""

from __future__ import annotations

import re

import duckdb
import polars as pl

UNIVERSE_COL = "in_universe_v2"
KEEP_COL = "keep_one_per_company"
CORP_COL = "corp_code"
MISSING_COL = "corp_code_missing"
COMMON_COL = "is_common_stock"
TV_COL = "trailing_tv20"
TV_WINDOW = 20
KEYS: tuple[str, ...] = ("trade_date", "ticker", "market")

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_STATEMENT = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def _ident(value: str, what: str) -> str:
    if not _IDENT.match(value):
        raise ValueError(f"{what} must be a plain SQL identifier, got {value!r}")
    return value


def _relation(value: str, what: str) -> str:
    if _IDENT.match(value):
        return value
    if _STATEMENT.match(value):
        return f"(\n{value.strip()}\n)"
    raise ValueError(f"{what} must be a view name or a SELECT/WITH statement")


def build_one_per_company_sql(
    universe: str,
    *,
    corp_view: str = "dart_corp_master",
    market_cap_view: str = "daily_market_cap",
    universe_col: str = UNIVERSE_COL,
    window: int = TV_WINDOW,
) -> str:
    """SQL: every universe column plus the one-per-company columns.

    Added columns: ``corp_code`` (VARCHAR, NULL when unmapped), ``is_common_stock``,
    ``trailing_tv20`` (DOUBLE, NULL without a market-cap row), ``corp_code_missing`` and
    ``keep_one_per_company`` (BOOLEAN, never NULL). One row per universe row.

    ``universe``: a view/table name or a SELECT/WITH statement with ``trade_date``,
    ``ticker``, ``market`` and ``universe_col``. ``corp_view`` needs ``ticker`` and
    ``corp_code``; ``market_cap_view`` needs ``trade_date``, ``ticker``,
    ``trading_value`` and must be unique on ``(trade_date, ticker)``
    (:func:`assert_one_per_company_inputs_unique`).
    """
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    uni = _relation(universe, "universe")
    cv = _ident(corp_view, "corp_view")
    mv = _ident(market_cap_view, "market_cap_view")
    u_col = _ident(universe_col, "universe_col")
    w = int(window) - 1
    return f"""
        WITH corp AS (
            SELECT ticker, MIN(corp_code) AS corp_code
            FROM {cv}
            WHERE ticker IS NOT NULL AND ticker <> ''
              AND corp_code IS NOT NULL AND corp_code <> ''
            GROUP BY ticker
        ),
        tv AS (
            SELECT trade_date, ticker,
                   AVG(CAST(trading_value AS DOUBLE)) OVER (
                       PARTITION BY ticker ORDER BY trade_date
                       ROWS BETWEEN {w} PRECEDING AND CURRENT ROW
                   ) AS {TV_COL}
            FROM {mv}
        ),
        keyed AS (
            SELECT u.*,
                   COALESCE(c1.corp_code, c2.corp_code) AS {CORP_COL},
                   (RIGHT(u.ticker, 1) = '0') AS {COMMON_COL},
                   t.{TV_COL} AS {TV_COL}
            FROM {uni} u
            LEFT JOIN corp c1 ON c1.ticker = u.ticker
            LEFT JOIN corp c2
              ON RIGHT(u.ticker, 1) <> '0' AND c2.ticker = SUBSTR(u.ticker, 1, 5) || '0'
            LEFT JOIN tv t ON t.trade_date = u.trade_date AND t.ticker = u.ticker
        ),
        ranked AS (
            SELECT k.*,
                   COALESCE(k.{u_col}, FALSE) AS _in_u,
                   ROW_NUMBER() OVER (
                       PARTITION BY k.trade_date, k.{CORP_COL}
                       ORDER BY COALESCE(k.{COMMON_COL}, FALSE) DESC,
                                k.{TV_COL} DESC NULLS LAST,
                                k.ticker ASC, k.market ASC
                   ) AS _rn
            FROM keyed k
            WHERE COALESCE(k.{u_col}, FALSE) AND k.{CORP_COL} IS NOT NULL
        )
        SELECT
            k.* EXCLUDE ({COMMON_COL}),
            COALESCE(k.{COMMON_COL}, FALSE) AS {COMMON_COL},
            (k.{CORP_COL} IS NULL) AS {MISSING_COL},
            CASE
                WHEN NOT COALESCE(k.{u_col}, FALSE) THEN FALSE
                WHEN k.{CORP_COL} IS NULL THEN TRUE
                ELSE COALESCE(r._rn = 1, FALSE)
            END AS {KEEP_COL}
        FROM keyed k
        LEFT JOIN ranked r
          ON r.trade_date = k.trade_date AND r.ticker = k.ticker AND r.market = k.market
    """


def assert_one_per_company_inputs_unique(
    con: duckdb.DuckDBPyConnection,
    universe: str = "dim_universe_daily",
    market_cap_view: str = "daily_market_cap",
) -> None:
    """Fail closed if ``universe`` has two rows for one ``(trade_date, ticker, market)``
    or ``market_cap_view`` two for one ``(trade_date, ticker)`` (they would multiply rows)."""
    for view, keys in (
        (_ident(universe, "universe"), KEYS),
        (_ident(market_cap_view, "market_cap_view"), ("trade_date", "ticker")),
    ):
        key_list = ", ".join(keys)
        extra = con.execute(
            f"SELECT count(*) - count(DISTINCT ({key_list})) FROM {view}"  # noqa: S608
        ).fetchone()[0]
        if extra:
            raise AssertionError(f"{view} has {extra} duplicate ({key_list}) rows")


def with_one_per_company[F: (pl.DataFrame, pl.LazyFrame)](
    universe: F,
    corp_master: pl.DataFrame | pl.LazyFrame,
    market_cap: pl.DataFrame | pl.LazyFrame,
    *,
    universe_col: str = UNIVERSE_COL,
    window: int = TV_WINDOW,
) -> F:
    """Polars form of :func:`build_one_per_company_sql` (same columns and values).

    ``corp_master``: ``ticker``, ``corp_code``. ``market_cap``: ``trade_date``,
    ``ticker``, ``trading_value``. The universe's row order is kept; same type back.
    """
    if int(window) < 1:
        raise ValueError("window must be >= 1")
    lazy_in = isinstance(universe, pl.LazyFrame)
    u = universe.lazy()
    names = set(u.collect_schema().names())
    for c in (CORP_COL, KEEP_COL, MISSING_COL, COMMON_COL, TV_COL):
        if c in names:
            raise ValueError(f"universe already has a column {c!r}")

    corp = (
        corp_master.lazy()
        .filter(
            pl.col("ticker").is_not_null()
            & (pl.col("ticker") != "")
            & pl.col(CORP_COL).is_not_null()
            & (pl.col(CORP_COL) != "")
        )
        .group_by("ticker")
        .agg(pl.col(CORP_COL).min())
    )
    c1 = corp.rename({CORP_COL: "_c1"})
    c2 = corp.rename({"ticker": "_sib", CORP_COL: "_c2"})
    tv = (
        market_cap.lazy()
        .sort(["ticker", "trade_date"])
        .select(
            "trade_date",
            "ticker",
            pl.col("trading_value")
            .cast(pl.Float64)
            .rolling_mean(window_size=int(window), min_samples=1)
            .over("ticker")
            .alias(TV_COL),
        )
    )
    is_common = pl.col("ticker").str.ends_with("0")
    k = (
        u.with_row_index("_i")
        .with_columns(
            is_common.alias(COMMON_COL),
            (pl.col("ticker").str.slice(0, 5) + "0").alias("_sib"),
        )
        .join(c1, on="ticker", how="left", maintain_order="left")
        .join(c2, on="_sib", how="left", maintain_order="left")
        .join(tv, on=["trade_date", "ticker"], how="left", maintain_order="left")
        .with_columns(
            pl.when(is_common)
            .then(pl.col("_c1"))
            .otherwise(pl.coalesce("_c1", "_c2"))
            .alias(CORP_COL)
        )
    )
    in_u = pl.col(universe_col).fill_null(False)
    winners = (
        k.filter(in_u & pl.col(CORP_COL).is_not_null())
        .sort(
            ["trade_date", CORP_COL, COMMON_COL, TV_COL, "ticker", "market"],
            descending=[False, False, True, True, False, False],
            nulls_last=True,
        )
        .group_by(["trade_date", CORP_COL], maintain_order=True)
        .agg(pl.col("_i").first())
        .select("_i")
        .with_columns(pl.lit(True).alias("_win"))
    )
    out = (
        k.join(winners, on="_i", how="left", maintain_order="left")
        .with_columns(
            pl.col(CORP_COL).is_null().alias(MISSING_COL),
            pl.when(~in_u)
            .then(False)
            .when(pl.col(CORP_COL).is_null())
            .then(True)
            .otherwise(pl.col("_win").fill_null(False))
            .alias(KEEP_COL),
        )
        .drop("_i", "_sib", "_c1", "_c2", "_win")
    )
    return out if lazy_in else out.collect()  # type: ignore[return-value]
