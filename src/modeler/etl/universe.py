# ruff: noqa: E501
"""dim_universe_daily — per-(trade_date, ticker, market) eligibility flags.

Implements the etl_00 §1.2 universe filter as *auditable boolean columns* (one
per condition) plus a combined ``in_universe`` gate, so the reason a name is
in/out on a given day is inspectable rather than hidden in a WHERE clause.

Filter conditions (etl_00 §1.2):
  1) row exists at t AND not halted        -> ``not_halted``
  2) >= min_valid of trailing window valid  -> ``warmup_ok``   (new-listing/long-halt guard)
  3) trailing avg turnover >= floor         -> ``liquidity_ok`` (low-liquidity noise guard)
  4) t+H close exists                        -> ``label_ok``     (label producible; TRAIN ONLY)

``in_universe`` = not_halted AND warmup_ok AND liquidity_ok. Condition 4 is kept
*separate* because it must only be applied to the train/validation set, never at
inference time where the future does not exist (etl_00 §1.2).

Conventions (verified on the lake):
  - ``is_halted := (open=0 AND high=0 AND low=0)`` — pykrx halt-day convention, ~1.6%.
  - ``turnover  := close * volume`` cast to DOUBLE (Decimal128 overflow guard, etl_01 §3).
  - Forward existence uses the *per-ticker non-halt trading-day index* (etl_00 §2.1),
    so t+H means H of the ticker's own real sessions later — consistent with the label.

See ``docs/target/01_20_access_return_rank/etl_00_ridge_elasticnet.md`` §1.2 and
``etl_03_implementation_plan.md`` §4 (P2).
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view

UNIVERSE_TABLE = "dim_universe_daily"
BROAD_UNIVERSE_TABLE = "dim_universe_broad_daily"
TRADABLE_UNIVERSE_TABLE = "dim_universe_tradable_daily"


def assert_tradable_subset(
    con: duckdb.DuckDBPyConnection,
    broad_view: str = BROAD_UNIVERSE_TABLE,
    tradable_view: str = TRADABLE_UNIVERSE_TABLE,
) -> None:
    """Fail closed if the tradable formation population escapes the broad one."""
    count = con.execute(f"""
        SELECT count(*)
        FROM {tradable_view} t
        LEFT JOIN {broad_view} b USING (trade_date, ticker, market)
        WHERE t.in_universe AND (b.in_universe IS NULL OR NOT b.in_universe)
    """).fetchone()[0]
    if count:
        raise AssertionError(f"tradable universe is not a subset of broad universe: {count} rows")


@dataclass(frozen=True)
class UniverseFilter:
    """Tunable thresholds for the universe gate (etl_00 §1.2 defaults)."""

    warmup_window: int = 60  # trailing rows examined for warm-up
    warmup_min_valid: int = 40  # >= this many non-halt days in the window
    liquidity_window: int = 60  # trailing rows for avg turnover
    min_liquidity_krw: float = 1e8  # 1억원 floor
    label_horizon: int = 20  # H trading days forward for label existence
    min_close_krw: float = 0.0
    apply_liquidity_filter: bool = True
    membership_reconstruction_available: bool = False


def broad_universe_filter() -> UniverseFilter:
    """Broad observed-OHLCV universe; liquidity is diagnostic only."""
    return UniverseFilter(
        liquidity_window=20,
        min_liquidity_krw=0.0,
        min_close_krw=0.0,
        apply_liquidity_filter=False,
    )


def tradable_universe_filter() -> UniverseFilter:
    """Tradable screen kept separate from the broad formation population."""
    return UniverseFilter(
        liquidity_window=20,
        min_liquidity_krw=1e8,
        min_close_krw=1000.0,
        apply_liquidity_filter=True,
    )


def build_universe_sql(
    flt: UniverseFilter,
    *,
    price_view: str = "daily_ohlcv",
) -> str:
    """SQL producing one flagged row per (trade_date, ticker, market).

    ``price_view`` must already be registered on the connection. Both rolling
    windows use ``warmup_window``/``liquidity_window`` row counts; they share a
    frame here since both default to 60 (etl_00 §1.2) — kept as separate columns
    for clarity.
    """
    w = flt.warmup_window
    lw = flt.liquidity_window
    return f"""
        WITH base AS (
            SELECT
                trade_date, ticker, market,
                CAST(close AS DOUBLE) AS close_d,
                (open = 0 AND high = 0 AND low = 0) AS is_halted,
                CAST(close AS DOUBLE) * CAST(volume AS DOUBLE) AS turnover
            FROM {price_view}
        ),
        win AS (
            SELECT
                trade_date, ticker, market, close_d, is_halted,
                COALESCE(SUM(CASE WHEN NOT is_halted THEN 1 ELSE 0 END)
                    OVER (PARTITION BY ticker, market ORDER BY trade_date
                          ROWS BETWEEN {w - 1} PRECEDING AND CURRENT ROW), 0) AS valid_days,
                AVG(CASE WHEN NOT is_halted THEN turnover END)
                    OVER (PARTITION BY ticker, market ORDER BY trade_date
                          ROWS BETWEEN {lw - 1} PRECEDING AND CURRENT ROW) AS avg_turnover
            FROM base
        ),
        nonhalt AS (
            SELECT
                trade_date, ticker, market,
                ROW_NUMBER() OVER (PARTITION BY ticker, market ORDER BY trade_date) AS d_idx_nh,
                COUNT(*) OVER (PARTITION BY ticker, market) AS n_nonhalt
            FROM base
            WHERE NOT is_halted
        )
        SELECT
            w.trade_date,
            w.ticker,
            w.market,
            w.is_halted,
            w.valid_days,
            w.avg_turnover,
            nh.d_idx_nh,
            (NOT w.is_halted) AS not_halted,
            (w.valid_days >= {flt.warmup_min_valid}) AS warmup_ok,
            (COALESCE(w.avg_turnover, 0) >= {flt.min_liquidity_krw}) AS liquidity_ok,
            (w.close_d >= {flt.min_close_krw}) AS close_ok,
            (w.close_d >= {flt.min_close_krw}
                AND COALESCE(w.avg_turnover, 0) >= {flt.min_liquidity_krw}) AS tradable_ok,
            (nh.d_idx_nh IS NOT NULL
                AND nh.d_idx_nh + {flt.label_horizon} <= nh.n_nonhalt) AS label_ok,
            {', '.join(
                f"(nh.d_idx_nh IS NOT NULL AND nh.d_idx_nh + {h} <= nh.n_nonhalt) AS label_ok_{h}d"
                for h in (1, 2, 3, 5, 10, 20, 40, 60, 120)
            )},
            {str(flt.membership_reconstruction_available).upper()} AS membership_reconstruction_available,
            FALSE AS management_filter_available,
            (
                NOT w.is_halted
                AND w.valid_days >= {flt.warmup_min_valid}
                AND w.close_d >= {flt.min_close_krw}
                AND ({str(not flt.apply_liquidity_filter).upper()}
                     OR COALESCE(w.avg_turnover, 0) >= {flt.min_liquidity_krw})
            ) AS in_universe
        FROM win w
        LEFT JOIN nonhalt nh USING (trade_date, ticker, market)
    """


def materialize_universe(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    flt: UniverseFilter | None = None,
    *,
    price_view: str = "daily_ohlcv",
    force: bool = False,
) -> str:
    """Build + register ``dim_universe_daily`` mart view. Returns view name.

    Requires ``price_view`` registered on ``con``.
    """
    flt = flt or UniverseFilter()
    materialize(
        con,
        config,
        UNIVERSE_TABLE,
        build_universe_sql(flt, price_view=price_view),
        force=force,
    )
    return register_mart_view(con, config, UNIVERSE_TABLE)


def materialize_named_universes(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    price_view: str = "daily_ohlcv",
    force: bool = False,
) -> tuple[str, str]:
    """Materialize the broad and tradable universes used by horizon scans."""
    for name, flt in (
        (BROAD_UNIVERSE_TABLE, broad_universe_filter()),
        (TRADABLE_UNIVERSE_TABLE, tradable_universe_filter()),
    ):
        materialize(
            con,
            config,
            name,
            build_universe_sql(flt, price_view=price_view),
            force=force,
        )
        register_mart_view(con, config, name)
    return BROAD_UNIVERSE_TABLE, TRADABLE_UNIVERSE_TABLE
