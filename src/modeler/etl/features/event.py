"""feat_event — sparse corporate-action flags (etl_00 §3.4, selective / 2nd-pass).

Source: ``dart_share_count_raw`` (issued/treasury shares). The DART event tables
are non-standard (``se``/``stock_knd`` free-text, ~50-139 variants) and not yet
normalized (etl_00 §3.4), so the 1st pass uses ONLY robust low-frequency flags
anchored on the ``se = '합계'`` (total) rows:

  - ev_treasury_ratio        : treasury shares / issued shares (buyback intensity)
  - ev_has_treasury          : treasury shares > 0 flag
  - ev_shares_chg_yoy        : YoY change in issued shares (dilution / buyback)

PIT lag: filings are annual (``bsns_year``); we make a row available at
``bsns_year-end + 90 days`` (same conservative lag as financials, etl_00 §3.3),
then interval-join to the universe so every (t, ticker) gets the latest available
year's flags. ``ev_`` prefix. NULL before first availability -> ``*_isna`` (P5).

Dividend features are deferred: ``dart_shareholder_return_raw`` needs the se/
stock_knd canonical mapping (etl_00 §3.4) before its values are trustworthy.

See ``etl_00`` §3.4, ``etl_03_implementation_plan.md`` §4 (P8).
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view

EVENT_TABLE = "feat_event"

ANNUAL_LAG_DAYS = 90


def build_event_sql(
    universe_view: str = "dim_universe_daily",
    share_count_view: str = "dart_share_count_raw",
) -> str:
    """SQL producing ``feat_event`` at (trade_date, ticker, market) grain.

    Uses ``se = '합계'`` total rows per (ticker, bsns_year); makes each year
    available at year-end + 90d; interval-joins to the universe (PIT as-of).
    """
    return f"""
        WITH totals AS (
            -- one total row per (ticker, bsns_year): issued + treasury shares.
            SELECT ticker, bsns_year,
                   CAST(istc_totqy AS DOUBLE) AS issued_shares,
                   CAST(tesstk_co AS DOUBLE) AS treasury_shares,
                   make_date(bsns_year, 12, 31)
                       + INTERVAL '{ANNUAL_LAG_DAYS} days' AS available_from
            FROM {share_count_view}
            WHERE se = '합계'
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY ticker, bsns_year ORDER BY istc_totqy DESC NULLS LAST
            ) = 1
        ),
        derived AS (
            SELECT ticker, bsns_year, issued_shares, treasury_shares, available_from,
                   LAG(issued_shares) OVER (PARTITION BY ticker ORDER BY bsns_year)
                       AS prev_issued_shares,
                   LEAD(available_from) OVER (PARTITION BY ticker ORDER BY available_from)
                       AS next_from
            FROM totals
        ),
        pit_asof AS (
            SELECT u.trade_date, u.ticker, u.market,
                   d.issued_shares, d.treasury_shares, d.prev_issued_shares
            FROM {universe_view} u
            JOIN derived d
              ON d.ticker = u.ticker
             AND d.available_from <= u.trade_date
             AND (d.next_from IS NULL OR u.trade_date < d.next_from)
            WHERE u.in_universe
        )
        SELECT
            trade_date, ticker, market,
            treasury_shares / NULLIF(issued_shares, 0) AS ev_treasury_ratio,
            (treasury_shares > 0) AS ev_has_treasury,
            issued_shares / NULLIF(prev_issued_shares, 0) - 1 AS ev_shares_chg_yoy
        FROM pit_asof
    """


def materialize_event(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    universe_view: str = "dim_universe_daily",
    share_count_view: str = "dart_share_count_raw",
    force: bool = False,
) -> str:
    """Build + register ``feat_event`` mart view. Returns the view name.

    Requires ``universe_view`` and ``share_count_view`` registered on ``con``.
    """
    materialize(
        con,
        config,
        EVENT_TABLE,
        build_event_sql(universe_view, share_count_view),
        force=force,
    )
    return register_mart_view(con, config, EVENT_TABLE)
