"""feat_common — market/macro common features, PIT broadcast (etl_00 §3.5).

Source: ``common_feature_daily_fact`` (canonical lake). These are stock-agnostic
market/macro series (indices, breadth, rates, FX, commodities, macro) that get
broadcast to every name on a given ``feature_date`` (etl_00 §3.5). PIT is already
guaranteed by the table's ``asof_available_date`` (look-ahead violations = 0,
verified), and we additionally filter ``asof_available_date <= t`` so only rows
known by ``t`` are used.

Grain after pivot: one row per ``feature_date`` (= trade_date ``t``); the panel
join broadcasts it across all tickers on that date (etl_00 §4.1). ``cf_`` prefix.

Coverage (measured on snapshot 2026-08-23, not the stale figure this docstring
carried): the daily series start **2014-06-16** and the monthly macro ones
**2013-06-20** — the backfill this file was written before has since happened.
So over the 2015+ training panel these are populated, not mostly NULL. The
``*_isna`` flags (P5) still carry per-feature gaps, which are now warm-up
windows (a 20-day return needs 20 prior sessions) rather than a missing history.

See ``etl_00`` §3.5, ``etl_03_implementation_plan.md`` §4 (P8).
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view

COMMON_TABLE = "feat_common"

# Curated core set (etl_00 §3.5): market direction, rates/term-spread, FX, risk.
# A superset is available in the table; pass ``feature_codes`` to override.
DEFAULT_FEATURE_CODES: tuple[str, ...] = (
    "market_kospi_ret_5d",
    "market_kospi_ret_20d",
    "market_kosdaq_ret_1d",
    "market_kospi200_ret_1d",
    "rate_kr_gov3y_level",
    "rate_kr_gov10y_level",
    "rate_kr_term_spread_10y_3y",
    "rate_us_term_spread_10y_2y",
    "fx_usdkrw_ret_5d",
    "global_vix_level",
    "global_sp500_ret_1d",
    "commodity_wti_ret_20d",
)


def _safe_col(code: str) -> str:
    """Map a feature_code to a safe ``cf_`` column name."""
    return "cf_" + code


def build_common_sql(
    cfdf_view: str = "common_feature_daily_fact",
    feature_codes: tuple[str, ...] = DEFAULT_FEATURE_CODES,
) -> str:
    """SQL producing ``feat_common`` (one row per feature_date, ``cf_`` columns).

    PIT: keeps only ``asof_available_date <= feature_date`` rows, then per
    (feature_date, feature_code) the latest available row wins. ``cfdf_view``
    must already be registered on the connection.
    """
    pivot_cols = ",\n            ".join(
        f"MAX(CASE WHEN feature_code = '{code}' THEN value END) AS {_safe_col(code)}"
        for code in feature_codes
    )
    return f"""
        WITH pit AS (
            SELECT feature_date, feature_code,
                   CAST(value_numeric AS DOUBLE) AS value
            FROM {cfdf_view}
            WHERE asof_available_date <= feature_date
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY feature_date, feature_code
                ORDER BY asof_available_date DESC
            ) = 1
        )
        SELECT
            feature_date AS trade_date,
            {pivot_cols}
        FROM pit
        GROUP BY feature_date
    """


def materialize_common(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    cfdf_view: str = "common_feature_daily_fact",
    feature_codes: tuple[str, ...] = DEFAULT_FEATURE_CODES,
    force: bool = False,
) -> str:
    """Build + register ``feat_common`` mart view. Returns the view name.

    Requires ``cfdf_view`` registered on ``con`` (from the canonical lake).
    """
    materialize(
        con,
        config,
        COMMON_TABLE,
        build_common_sql(cfdf_view, feature_codes),
        force=force,
    )
    return register_mart_view(con, config, COMMON_TABLE)
