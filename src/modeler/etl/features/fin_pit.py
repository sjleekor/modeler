"""feat_fin_pit — point-in-time financial features (etl_00 §3.3, etl_01 §6).

Source: ``stock_metric_fact`` (canonical lake). Financials carry no disclosure
date, so a conservative PIT lag stands in for it (etl_00 §3.3):

    available_from := period_end + 90d   (annual reports)
                      period_end + 45d   (quarterly: q1/half/q3)

A feature row at trade_date ``t`` may only use reports with
``available_from <= t`` (no look-ahead). For each (ticker, metric_code) we build
disclosure *intervals* ``[available_from, next_available_from)`` and join the
universe by interval — this preserves ALL ~26 metric_codes per name (a naive
``ASOF JOIN`` returns only one right row per left row and would drop 25 metrics;
etl_01 §6 verified: Samsung 2024-06-03 -> 26 rows). When two reports map to the
same available_from we keep the latest ``period_end`` (restatement-safe).

The wide metrics are turned into ratios/growth (level amounts are not used
directly, L3/L5, etl_00 §3.3): roa, debt_to_equity, equity_ratio, ocf_to_assets,
cash_ratio. Capital impairment (``total_equity <= 0``, ~12 names) flips ratios,
so a ``fin_is_negative_equity`` flag is set and the affected ratio is clipped.
``fin_has_fs`` marks names with any balance-sheet metric (vs SC/SR-only corps).

Output grain: (trade_date, ticker, market), ``fin_`` prefix. NULL where no report
is yet available at ``t`` (early history / warm-up) -> ``*_isna`` flags in P5.

See ``etl_00`` §3.3, ``etl_01`` §6, ``etl_03_implementation_plan.md`` §4 (P7).
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view

FIN_TABLE = "feat_fin_pit"

# Balance-sheet / cash-flow / share metrics used by the derived ratios.
# (IS metrics like revenue/net_income are sparse; included where present.)
_PIVOT_METRICS = (
    "total_assets",
    "total_liabilities",
    "total_equity",
    "cash_and_cash_equivalents",
    "operating_cash_flow",
    "net_income",
    "controlling_net_income",
    "revenue",
    "operating_income",
    "issued_shares",
)

ANNUAL_LAG_DAYS = 90
QUARTERLY_LAG_DAYS = 45


def build_available_sql(smf_view: str = "stock_metric_fact") -> str:
    """CTE-body SQL adding ``available_from`` and deduping to latest period_end.

    ``available_from`` = period_end + (90d annual / 45d quarterly). For a given
    (ticker, metric_code, available_from) the latest period_end wins (handles
    restatements landing on the same disclosure lag).
    """
    lag = (
        f"CASE WHEN period_type = 'annual' THEN period_end + INTERVAL '{ANNUAL_LAG_DAYS} days' "
        f"ELSE period_end + INTERVAL '{QUARTERLY_LAG_DAYS} days' END"
    )
    return f"""
        SELECT ticker, market, metric_code,
               CAST(value_numeric AS DOUBLE) AS value,
               {lag} AS available_from
        FROM {smf_view}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY ticker, metric_code, {lag}
            ORDER BY period_end DESC
        ) = 1
    """


def _pivot_expr() -> str:
    cols = [f"MAX(CASE WHEN metric_code = '{m}' THEN value END) AS {m}" for m in _PIVOT_METRICS]
    return ",\n            ".join(cols)


def build_fin_pit_sql(
    universe_view: str = "dim_universe_daily",
    smf_view: str = "stock_metric_fact",
) -> str:
    """SQL producing ``feat_fin_pit`` at (trade_date, ticker, market) grain.

    Joins the universe to disclosure intervals (PIT as-of, all metrics preserved),
    wide-pivots the metrics, and derives ratios. ``universe_view`` and
    ``smf_view`` must already be registered on the connection.
    """
    avail = build_available_sql(smf_view)
    return f"""
        WITH avail AS (
            {avail}
        ),
        intervals AS (
            SELECT ticker, market, metric_code, value, available_from,
                   LEAD(available_from) OVER (
                       PARTITION BY ticker, metric_code ORDER BY available_from
                   ) AS next_from
            FROM avail
        ),
        pit_asof AS (
            -- PIT as-of by interval: keeps every metric_code per (t, ticker).
            SELECT u.trade_date, u.ticker, u.market, f.metric_code, f.value
            FROM {universe_view} u
            JOIN intervals f
              ON f.ticker = u.ticker
             AND f.available_from <= u.trade_date
             AND (f.next_from IS NULL OR u.trade_date < f.next_from)
            WHERE u.in_universe
        ),
        wide AS (
            SELECT trade_date, ticker, market,
                {_pivot_expr()}
            FROM pit_asof
            GROUP BY trade_date, ticker, market
        )
        SELECT
            trade_date, ticker, market,
            -- capital-impairment flag (total_equity <= 0); ratios clipped below.
            (total_equity IS NOT NULL AND total_equity <= 0) AS fin_is_negative_equity,
            (total_assets IS NOT NULL) AS fin_has_fs,
            net_income / NULLIF(total_assets, 0) AS fin_roa,
            -- debt/equity clipped when equity <= 0 (avoid sign flip / inf)
            CASE WHEN total_equity > 0
                 THEN total_liabilities / total_equity END AS fin_debt_to_equity,
            total_equity / NULLIF(total_assets, 0) AS fin_equity_ratio,
            operating_cash_flow / NULLIF(total_assets, 0) AS fin_ocf_to_assets,
            cash_and_cash_equivalents / NULLIF(total_assets, 0) AS fin_cash_ratio,
            COALESCE(controlling_net_income, net_income)
                / NULLIF(total_equity, 0) AS fin_roe,
            revenue / NULLIF(total_assets, 0) AS fin_asset_turnover,
            operating_income / NULLIF(revenue, 0) AS fin_operating_margin
        FROM wide
    """


def materialize_fin_pit(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    universe_view: str = "dim_universe_daily",
    smf_view: str = "stock_metric_fact",
    force: bool = False,
) -> str:
    """Build + register ``feat_fin_pit`` mart view. Returns the view name.

    Requires ``universe_view`` and ``smf_view`` registered on ``con`` (the latter
    from the canonical lake).
    """
    materialize(
        con,
        config,
        FIN_TABLE,
        build_fin_pit_sql(universe_view, smf_view),
        force=force,
    )
    return register_mart_view(con, config, FIN_TABLE)
