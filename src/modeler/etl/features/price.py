# ruff: noqa: E501
"""feat_price — derived price/momentum/volatility/liquidity features (etl_00 §3.1).

Source: ``daily_ohlcv`` only (always available). Produces *pre-standardization*
columns at grain ``(trade_date, ticker, market)`` with the ``px_`` prefix. Level
prices are deliberately turned into returns/ratios (etl_00 §3.1, L3/L5); winsor/
log/z-score is the model preprocess step (P5), not here.

Windows use trading-row counts (``ROWS BETWEEN n PRECEDING``) over the per-ticker
ordering, which absorbs halts/holidays consistently with the d_idx label logic.
Halt days (``open=high=low=0``) carry a stale close, so returns spanning a halt
are slightly distorted — flagged via ``px_is_halted`` / ``px_halt_ratio_20d`` so
the model preprocess can mask them (etl_00 §3.1, §1.2).

Conventions (etl_01 §3): all price arithmetic casts BIGINT prices to DOUBLE
before division/log to avoid integer truncation and Decimal overflow.

See ``etl_00`` §3.1 and ``etl_03_implementation_plan.md`` §4 (P3).
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view
from modeler.etl.trading_panel import build_market_model_sql, build_valid_session_sql

PRICE_TABLE = "feat_price"


def build_price_sql(
    price_view: str = "daily_ohlcv",
    *,
    quality_view: str | None = None,
) -> str:
    """SQL producing ``feat_price`` from a daily OHLCV view.

    ``price_view`` must already be registered on the connection. Log returns use
    ``ln(close_t / close_{t-n})``; momentum/vol use day-over-day log returns.
    """
    quality_join = (
        f"LEFT JOIN {quality_view} q USING (trade_date, ticker, market)"
        if quality_view
        else ""
    )
    ca_event = "COALESCE(q.ca_event, FALSE)" if quality_view else "FALSE"
    return f"""
        WITH base AS (
            SELECT trade_date, ticker, market,
                   CAST(close AS DOUBLE) AS close_d,
                   CAST(high AS DOUBLE) AS high_d,
                   CAST(low AS DOUBLE) AS low_d,
                   CAST(volume AS DOUBLE) AS volume_d,
                   (open = 0 AND high = 0 AND low = 0) AS is_halted
            FROM {price_view}
        ),
        valid AS (
            {build_valid_session_sql(price_view)}
        ),
        valid_q AS (
            SELECT v.*, {ca_event} AS ca_event
            FROM valid v {quality_join}
        ),
        {build_market_model_sql("valid_q")},
        features AS (
            SELECT
                trade_date, ticker, market, valid_session_idx,
                -- All rows in these windows are valid sessions by construction.
                -SUM(log_ret) OVER w5 AS px_reversal_5d,
                LN(LAG(close_d, 21) OVER w / NULLIF(LAG(close_d, 126) OVER w, 0))
                    AS px_mom_6_1,
                LN(LAG(close_d, 21) OVER w / NULLIF(LAG(close_d, 252) OVER w, 0))
                    AS px_mom_12_1,
                CASE WHEN COUNT(resid_ret) OVER (
                    PARTITION BY ticker, market ORDER BY trade_date
                    ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING
                ) = 232 THEN SUM(resid_ret) OVER (
                    PARTITION BY ticker, market ORDER BY trade_date
                    ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING
                ) END AS px_resid_mom_12_1,
                close_d / NULLIF(MAX(close_d) OVER w252, 0) - 1 AS px_near_52w_high,
                MAX(simple_ret) OVER w20 AS px_maxret_20d,
                CASE WHEN COUNT(resid_ret) OVER w126 >= 126
                     THEN STDDEV_SAMP(resid_ret) OVER w60 END AS px_idio_vol_60d,
                LN(NULLIF(turnover, 0) / NULLIF(
                    QUANTILE_CONT(turnover, 0.5) OVER (
                        PARTITION BY ticker, market ORDER BY trade_date
                        ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING
                    ), 0)) AS px_turnover_shock,
                AVG(CASE WHEN log_ret = 0 OR volume_d = 0 THEN 1.0 ELSE 0.0 END) OVER w20
                    AS px_zero_ret_ratio_20d,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w5 AS ca_count_5,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w20 AS ca_count_20,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w60 AS ca_count_60,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w252 AS ca_count_252,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w252_skip21
                    AS ca_count_252_skip21,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w60_prior AS ca_count_60_prior,
                close_d AS close_for_legacy,
                log_ret AS ret_1d,
                turnover,
                ca_event
            FROM residuals
            WINDOW
                w AS (PARTITION BY ticker, market ORDER BY trade_date),
                w5 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
                w20 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
                w60 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),
                w126 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 125 PRECEDING AND CURRENT ROW),
                w252 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW),
                w252_skip21 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 252 PRECEDING AND 21 PRECEDING),
                w60_prior AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING)
        ),
        masked AS (
            SELECT *,
                CASE WHEN ca_count_5 > 0 THEN NULL ELSE px_reversal_5d END AS m_reversal,
                CASE WHEN ca_count_252_skip21 > 0 THEN NULL ELSE px_mom_6_1 END AS m_mom6,
                CASE WHEN ca_count_252_skip21 > 0 THEN NULL ELSE px_mom_12_1 END AS m_mom12,
                CASE WHEN ca_count_252_skip21 > 0 THEN NULL ELSE px_resid_mom_12_1 END AS m_resid,
                CASE WHEN ca_count_252 > 0 THEN NULL ELSE px_near_52w_high END AS m_high,
                CASE WHEN ca_count_20 > 0 THEN NULL ELSE px_maxret_20d END AS m_maxret,
                CASE WHEN ca_count_60 > 0 THEN NULL ELSE px_idio_vol_60d END AS m_idio,
                CASE WHEN ca_count_60_prior > 0 THEN NULL ELSE px_turnover_shock END AS m_shock,
                CASE WHEN ca_count_20 > 0 THEN NULL ELSE px_zero_ret_ratio_20d END AS m_zero
            FROM features
        ),
        variants AS (
            SELECT *,
                m_reversal AS px_reversal_5d,
                m_mom6 AS px_mom_6_1,
                m_mom12 AS px_mom_12_1,
                m_resid AS px_resid_mom_12_1,
                m_high AS px_near_52w_high,
                m_maxret AS px_maxret_20d,
                m_idio AS px_idio_vol_60d,
                m_shock AS px_turnover_shock,
                m_zero AS px_zero_ret_ratio_20d,
                LAG(m_reversal) OVER (PARTITION BY ticker, market ORDER BY trade_date)
                    AS px_reversal_5d_lag1,
                LAG(m_mom6) OVER (PARTITION BY ticker, market ORDER BY trade_date)
                    AS px_mom_6_1_lag1,
                LAG(m_mom12) OVER (PARTITION BY ticker, market ORDER BY trade_date)
                    AS px_mom_12_1_lag1,
                LAG(m_resid) OVER (PARTITION BY ticker, market ORDER BY trade_date)
                    AS px_resid_mom_12_1_lag1,
                LAG(m_high) OVER (PARTITION BY ticker, market ORDER BY trade_date)
                    AS px_near_52w_high_lag1,
                LAG(m_maxret) OVER (PARTITION BY ticker, market ORDER BY trade_date)
                    AS px_maxret_20d_lag1,
                LAG(m_idio) OVER (PARTITION BY ticker, market ORDER BY trade_date)
                    AS px_idio_vol_60d_lag1,
                LAG(m_shock) OVER (PARTITION BY ticker, market ORDER BY trade_date)
                    AS px_turnover_shock_lag1,
                LAG(m_zero) OVER (PARTITION BY ticker, market ORDER BY trade_date)
                    AS px_zero_ret_ratio_20d_lag1
            FROM masked
        ),
        legacy AS (
            SELECT *,
                LAG(close_d, 5) OVER w AS close_lag5,
                LAG(close_d, 20) OVER w AS close_lag20,
                LAG(close_d, 60) OVER w AS close_lag60,
                AVG(close_d) OVER w20 AS ma20,
                MAX(high_d) OVER w20 AS hi20,
                MIN(low_d) OVER w20 AS lo20,
                MAX(close_d) OVER w252 AS hi52w,
                AVG(CASE WHEN is_halted THEN 1.0 ELSE 0.0 END) OVER w20 AS halt_ratio
            FROM base
            WINDOW
                w AS (PARTITION BY ticker, market ORDER BY trade_date),
                w20 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
                w252 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
        )
        SELECT
            l.trade_date, l.ticker, l.market,
            v.ret_1d AS px_ret_1d,
            LN(l.close_d / NULLIF(l.close_lag5, 0)) AS px_ret_5d,
            LN(l.close_d / NULLIF(l.close_lag20, 0)) AS px_ret_20d,
            LN(l.close_d / NULLIF(l.close_lag60, 0)) AS px_ret_60d,
            (LN(l.close_d / NULLIF(l.close_lag20, 0)) - LN(l.close_d / NULLIF(l.close_lag60, 0))) AS px_mom_20_60,
            STDDEV_SAMP(v.ret_1d) OVER (PARTITION BY l.ticker, l.market ORDER BY l.trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS px_vol_20d,
            STDDEV_SAMP(v.ret_1d) OVER (PARTITION BY l.ticker, l.market ORDER BY l.trade_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS px_vol_60d,
            (l.hi20 - l.lo20) / NULLIF(l.close_d, 0) AS px_high_low_range_20d,
            v.turnover AS px_turnover,
            AVG(v.turnover) OVER (PARTITION BY l.ticker, l.market ORDER BY l.trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS px_turnover_ma20,
            v.px_amihud_20d,
            (l.close_d / NULLIF(l.ma20, 0) - 1) AS px_gap_vs_ma20,
            (l.close_d / NULLIF(l.hi52w, 0) - 1) AS px_dist_52w_high,
            l.is_halted AS px_is_halted,
            l.halt_ratio AS px_halt_ratio_20d,
            v.px_reversal_5d, v.px_mom_6_1, v.px_mom_12_1,
            v.px_resid_mom_12_1, v.px_near_52w_high, v.px_maxret_20d,
            v.px_idio_vol_60d, v.px_turnover_shock, v.px_zero_ret_ratio_20d,
            v.px_reversal_5d_lag1, v.px_mom_6_1_lag1, v.px_mom_12_1_lag1,
            v.px_resid_mom_12_1_lag1, v.px_near_52w_high_lag1, v.px_maxret_20d_lag1,
            v.px_idio_vol_60d_lag1, v.px_turnover_shock_lag1, v.px_zero_ret_ratio_20d_lag1,
            v.px_amihud_20d_lag1
        FROM legacy l
        LEFT JOIN (
            SELECT *,
                LAG(px_amihud_20d) OVER (PARTITION BY ticker, market ORDER BY trade_date)
                    AS px_amihud_20d_lag1
            FROM (
                SELECT *,
                    AVG(ABS(ret_1d) / NULLIF(turnover, 0)) OVER (
                        PARTITION BY ticker, market ORDER BY trade_date
                        ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS px_amihud_20d
                FROM variants
            )
        ) v USING (trade_date, ticker, market)
    """


def materialize_price(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    price_view: str = "daily_ohlcv",
    quality_view: str | None = None,
    force: bool = False,
) -> str:
    """Build + register ``feat_price`` mart view. Returns the view name.

    Requires ``price_view`` registered on ``con``.
    """
    materialize(
        con,
        config,
        PRICE_TABLE,
        build_price_sql(price_view, quality_view=quality_view),
        force=force,
    )
    return register_mart_view(con, config, PRICE_TABLE)
