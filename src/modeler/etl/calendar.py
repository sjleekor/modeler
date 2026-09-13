"""dim_trading_calendar — the market-level trading-day index (``d_idx``).

This is the single source of truth for "what is N trading days later" across the
whole pipeline. Forward returns (label horizon) and walk-forward embargo/purge
must all be expressed in ``d_idx``, never calendar days, so that market holidays
and per-ticker halts are absorbed (etl_00 §2.1, §5 and the §9 checklist:
"t+20는 캘린더가 아니라 거래일 인덱스 기준인가").

Grain: ``(market, trade_date)`` with a dense per-market ``d_idx`` starting at 1.
A market trading day is any date present in ``daily_ohlcv`` for that market.

See ``docs/target/00_shared_etl_platform.md`` §1 (dim_trading_calendar) and
``docs/target/01_20_access_return_rank/etl_03_implementation_plan.md`` §4 (P2).
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view

CALENDAR_TABLE = "dim_trading_calendar"


def build_calendar_sql(source_view: str = "daily_ohlcv") -> str:
    """SQL producing ``(market, trade_date, d_idx)`` from a price view.

    ``d_idx`` is a dense 1-based index over each market's distinct trading days
    (ascending date). It is market-wide, independent of any single ticker — so
    "20 trading days later" means 20 *market* sessions, even across a ticker's
    own halts.
    """
    return f"""
        SELECT
            market,
            trade_date,
            CAST(
                ROW_NUMBER() OVER (PARTITION BY market ORDER BY trade_date)
                AS BIGINT
            ) AS d_idx
        FROM (SELECT DISTINCT market, trade_date FROM {source_view})
    """


def materialize_calendar(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    source_view: str = "daily_ohlcv",
    force: bool = False,
) -> str:
    """Build + register ``dim_trading_calendar`` mart view. Returns view name.

    Requires ``source_view`` (default ``daily_ohlcv``) to already be registered
    on ``con`` (see :func:`modeler.etl.lake.register_views`).
    """
    materialize(
        con,
        config,
        CALENDAR_TABLE,
        build_calendar_sql(source_view),
        force=force,
    )
    return register_mart_view(con, config, CALENDAR_TABLE)
