"""``feat_flow_v2`` — flow features that fix the gaps found in 15_quality_fix_plan §4.2-§4.3.

A new mart beside ``feat_flow``; nothing in ``flow.py`` changes, so the frozen
``feat_flow`` ``sql_hash`` is untouched. The wide pivot, KRX-first dedup and the
valid-session filter are ``flow.py``'s own (imported, not copied).

Columns (all keyed ``(trade_date, ticker, market)``):

* ``flow_{foreign,inst,indiv}_netbuy_to_float_20d_v2`` — 20-session net-buy share sum over
  the same-day PIT float shares (``dim_stock_pit_daily.float_shares_pit``). NULL unless
  all 20 window rows carry a net-buy value.
* ``flow_{foreign,inst,indiv}_netbuy_to_float_5d_v2`` — the same ratio over a 5-row window
  (``COUNT = 5``), NULL unless all 5 window rows carry a net-buy value.
* ``flow_{foreign,inst}_netbuy_z_20d_v2`` — ``flow.py``'s z-score, NULL unless the window
  holds 20 non-NULL observations (the original emits a value from a 2-row window).
* ``flow_short_{avg_price,selling_volume,selling_value}_v2`` — ``flow.py``'s values, NULL
  unless ``short_regime_sql()`` says ``allowed`` for the trade date.
* ``flow_individual_netbuy_to_volume_{5,20}d_v2`` (C2-7) — individual net-buy share sum over
  the window divided by the KRX **raw** volume sum over the same window
  (``daily_market_cap.volume``). NULL unless both counts equal the window length and the
  denominator is positive. It is ``flow.py``'s ``flow_individual_netbuy_to_volume_{5,20}d``
  with the denominator swapped.
* ``flow_{foreign,inst,indiv}_netbuy_to_listed_{5,20}d_v2`` (C2-8) — net-buy share sum over
  the window divided by the same-day KRX ``daily_market_cap.listed_shares``. NULL unless all
  window rows carry a net-buy value and ``listed_shares > 0``.

Why the last two groups exist (17 §8 / §8.1, 15 §10.7). ``daily_ohlcv.volume`` (and close) is
Naver's *adjusted* series: a later split, reverse split or capital reduction rewrites the whole
past. KRX net-buy quantities are raw share counts, so ``net_buy / adjusted_volume`` carries the
future event into past rows (for tickers that later merge or reduce capital, ``|value| > 1`` in
6.8% of rows against 0.02% for the rest). ``daily_market_cap.volume`` is the KRX regular-session
raw count — the same scope as the Naver volume (median ratio 1.000 where the adjustment factor
is 1, before and after NXT) — so raw/raw has no future information.

The ``to_float`` denominator has a different defect, not a leak. DART float
(``float_shares_pit``) is wrong by the event ratio for about two months after a split or
reverse split (median 63 days), is empty in many Q1/Q3 filings, and has 1,000x unit errors.
DART's "float" is ``distb_stock_co``, or issued minus treasury shares when that is empty
(``etl/stock_pit.py``), so it is close in meaning to KRX listed shares (they differ by
roughly the treasury stock). The KRX listed-share count is the same day's value and changes
on the listing date, so it is right straight after an event; hence the ``to_listed`` columns.
The ``to_float`` columns are unchanged and stay in the mart for the record only; the next KR
model reads ``to_listed``.

``daily_market_cap`` is joined on ``(trade_date, ticker)`` only, not on ``market``. Its
``market`` is the listing market *at that time*, while the flow rows (like ``daily_ohlcv``)
carry the ticker's *current* market for the whole history, so a ticker that moved between
KOSPI and KOSDAQ would lose every KRX row before the transfer (25,407 rows through 2025-07,
e.g. 034230, 066970). The primary key of ``daily_market_cap`` is ``(trade_date, ticker, market)``,
so two rows for one ticker-day are possible in principle; the pick is deterministic: the row
whose market equals the flow row's market, else the smallest market name (then the smaller
volume and listed shares, so even a duplicated key resolves the same way on every run).
A ticker-day without any row gets NULL raw volume and listed shares. Coverage is 98.5-99.5%.
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.features.flow import METRIC_CODES, _pivot_expr, build_dedup_sql
from modeler.etl.mart import materialize, register_mart_view
from modeler.etl.quality import short_regime_sql

FLOW_V2_TABLE = "feat_flow_v2"

FLOW_V2_COLUMNS: tuple[str, ...] = (
    "flow_foreign_netbuy_to_float_20d_v2",
    "flow_inst_netbuy_to_float_20d_v2",
    "flow_indiv_netbuy_to_float_20d_v2",
    "flow_foreign_netbuy_z_20d_v2",
    "flow_inst_netbuy_z_20d_v2",
    "flow_short_avg_price_v2",
    "flow_short_selling_volume_v2",
    "flow_short_selling_value_v2",
    "flow_foreign_netbuy_to_float_5d_v2",
    "flow_inst_netbuy_to_float_5d_v2",
    "flow_indiv_netbuy_to_float_5d_v2",
    # C2-7: individual net buy over the KRX raw (unadjusted) volume
    "flow_individual_netbuy_to_volume_5d_v2",
    "flow_individual_netbuy_to_volume_20d_v2",
    # C2-8: net buy over the same-day KRX listed shares
    "flow_foreign_netbuy_to_listed_5d_v2",
    "flow_foreign_netbuy_to_listed_20d_v2",
    "flow_inst_netbuy_to_listed_5d_v2",
    "flow_inst_netbuy_to_listed_20d_v2",
    "flow_indiv_netbuy_to_listed_5d_v2",
    "flow_indiv_netbuy_to_listed_20d_v2",
)

assert set(METRIC_CODES) >= {
    "foreign_net_buy_volume",
    "institution_net_buy_volume",
    "individual_net_buy_volume",
    "short_selling_volume",
    "short_selling_value",
}


def _to_float(raw: str, name: str, window: str = "w20", rows: int = 20) -> str:
    return f"""CASE WHEN COUNT({raw}) OVER {window} = {rows}
                     THEN SUM({raw}) OVER {window} / NULLIF(float_shares_pit, 0) END
                AS {name}"""


def _to_volume(raw: str, name: str, window: str, rows: int) -> str:
    """Net-buy sum over the KRX raw volume sum of the same window (both counts full)."""
    return f"""CASE WHEN COUNT({raw}) OVER {window} = {rows}
                     AND COUNT(krx_raw_volume) OVER {window} = {rows}
                     AND SUM(krx_raw_volume) OVER {window} > 0
                     THEN SUM({raw}) OVER {window} / SUM(krx_raw_volume) OVER {window} END
                AS {name}"""


def _to_listed(raw: str, name: str, window: str, rows: int) -> str:
    """Net-buy sum over the same-day KRX listed shares (window full, listed shares > 0)."""
    return f"""CASE WHEN COUNT({raw}) OVER {window} = {rows} AND krx_listed_shares > 0
                     THEN SUM({raw}) OVER {window} / krx_listed_shares END
                AS {name}"""


def _z20(raw: str, name: str) -> str:
    return f"""CASE WHEN COUNT({raw}) OVER w20 = 20
                     THEN ({raw} - AVG({raw}) OVER w20)
                          / NULLIF(STDDEV_SAMP({raw}) OVER w20, 0) END
                AS {name}"""


def build_flow_v2_sql(
    flow_view: str = "krx_security_flow_raw",
    *,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    market_cap_view: str = "daily_market_cap",
) -> str:
    """SQL producing ``feat_flow_v2``.

    ``quality_view`` supplies ``valid_session_idx``: as in ``flow.py``, halt-day rows are
    dropped before any window runs so a window of 20 rows means 20 valid sessions.
    ``pit_view`` supplies ``float_shares_pit`` (``etl/stock_pit.py``).
    ``market_cap_view`` supplies the KRX raw ``volume`` and ``listed_shares`` for the
    ``to_volume`` / ``to_listed`` columns. It is joined on ``(trade_date, ticker)`` because its
    ``market`` is the market at the time, not the current one the flow rows carry (see the
    module docstring); one row per flow row is kept, preferring the same market.
    """
    return f"""
        WITH dedup AS (
            {build_dedup_sql(flow_view)}
        ),
        wide AS (
            SELECT
                trade_date, ticker, market,
                {_pivot_expr()}
            FROM dedup
            GROUP BY trade_date, ticker, market
        ),
        krx_raw AS (
            SELECT trade_date, ticker, market AS krx_market,
                   CAST(volume AS DOUBLE) AS krx_raw_volume,
                   CAST(listed_shares AS DOUBLE) AS krx_listed_shares
            FROM {market_cap_view}
        ),
        sessioned AS (
            SELECT wide.*, s.float_shares_pit,
                   mc.krx_raw_volume,
                   mc.krx_listed_shares,
                   {short_regime_sql("wide.trade_date")} AS short_regime_v2
            FROM wide
            LEFT JOIN {pit_view} s USING (trade_date, ticker, market)
            LEFT JOIN krx_raw mc ON mc.trade_date = wide.trade_date AND mc.ticker = wide.ticker
            LEFT JOIN {quality_view} q USING (trade_date, ticker, market)
            WHERE q.valid_session_idx IS NOT NULL
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY wide.trade_date, wide.ticker, wide.market
                ORDER BY (mc.krx_market = wide.market) DESC NULLS LAST,
                         mc.krx_market ASC NULLS LAST,
                         mc.krx_raw_volume ASC NULLS LAST,
                         mc.krx_listed_shares ASC NULLS LAST
            ) = 1
        )
        SELECT
            trade_date, ticker, market,
            {_to_float("foreign_net_buy_volume", FLOW_V2_COLUMNS[0])},
            {_to_float("institution_net_buy_volume", FLOW_V2_COLUMNS[1])},
            {_to_float("individual_net_buy_volume", FLOW_V2_COLUMNS[2])},
            {_z20("foreign_net_buy_volume", FLOW_V2_COLUMNS[3])},
            {_z20("institution_net_buy_volume", FLOW_V2_COLUMNS[4])},
            CASE WHEN short_regime_v2 = 'allowed'
                 THEN short_selling_value / NULLIF(short_selling_volume, 0) END
                AS {FLOW_V2_COLUMNS[5]},
            CASE WHEN short_regime_v2 = 'allowed' THEN short_selling_volume END
                AS {FLOW_V2_COLUMNS[6]},
            CASE WHEN short_regime_v2 = 'allowed' THEN short_selling_value END
                AS {FLOW_V2_COLUMNS[7]},
            {_to_float("foreign_net_buy_volume", FLOW_V2_COLUMNS[8], "w5", 5)},
            {_to_float("institution_net_buy_volume", FLOW_V2_COLUMNS[9], "w5", 5)},
            {_to_float("individual_net_buy_volume", FLOW_V2_COLUMNS[10], "w5", 5)},
            {_to_volume("individual_net_buy_volume", FLOW_V2_COLUMNS[11], "w5", 5)},
            {_to_volume("individual_net_buy_volume", FLOW_V2_COLUMNS[12], "w20", 20)},
            {_to_listed("foreign_net_buy_volume", FLOW_V2_COLUMNS[13], "w5", 5)},
            {_to_listed("foreign_net_buy_volume", FLOW_V2_COLUMNS[14], "w20", 20)},
            {_to_listed("institution_net_buy_volume", FLOW_V2_COLUMNS[15], "w5", 5)},
            {_to_listed("institution_net_buy_volume", FLOW_V2_COLUMNS[16], "w20", 20)},
            {_to_listed("individual_net_buy_volume", FLOW_V2_COLUMNS[17], "w5", 5)},
            {_to_listed("individual_net_buy_volume", FLOW_V2_COLUMNS[18], "w20", 20)}
        FROM sessioned
        WINDOW w20 AS (
            PARTITION BY ticker, market ORDER BY trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        ),
        w5 AS (
            PARTITION BY ticker, market ORDER BY trade_date
            ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
        )
    """


def materialize_flow_v2(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    flow_view: str = "krx_security_flow_raw",
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    market_cap_view: str = "daily_market_cap",
    force: bool = False,
) -> str:
    """Build + register the ``feat_flow_v2`` mart view. Returns the view name."""
    materialize(
        con,
        config,
        FLOW_V2_TABLE,
        build_flow_v2_sql(
            flow_view,
            pit_view=pit_view,
            quality_view=quality_view,
            market_cap_view=market_cap_view,
        ),
        force=force,
    )
    return register_mart_view(con, config, FLOW_V2_TABLE)
