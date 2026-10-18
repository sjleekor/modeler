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
) -> str:
    """SQL producing ``feat_flow_v2``.

    ``quality_view`` supplies ``valid_session_idx``: as in ``flow.py``, halt-day rows are
    dropped before any window runs so a window of 20 rows means 20 valid sessions.
    ``pit_view`` supplies ``float_shares_pit`` (``etl/stock_pit.py``).
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
        sessioned AS (
            SELECT wide.*, s.float_shares_pit,
                   {short_regime_sql("wide.trade_date")} AS short_regime_v2
            FROM wide
            LEFT JOIN {pit_view} s USING (trade_date, ticker, market)
            LEFT JOIN {quality_view} q USING (trade_date, ticker, market)
            WHERE q.valid_session_idx IS NOT NULL
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
            {_to_float("individual_net_buy_volume", FLOW_V2_COLUMNS[10], "w5", 5)}
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
    force: bool = False,
) -> str:
    """Build + register the ``feat_flow_v2`` mart view. Returns the view name."""
    materialize(
        con,
        config,
        FLOW_V2_TABLE,
        build_flow_v2_sql(flow_view, pit_view=pit_view, quality_view=quality_view),
        force=force,
    )
    return register_mart_view(con, config, FLOW_V2_TABLE)
