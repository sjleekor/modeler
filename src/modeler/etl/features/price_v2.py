# ruff: noqa: E501
"""feat_price_v2 — price features on valid sessions with corporate-action masks.

Plan: ``my/milestones/common/20261012_feature_explainer/15_quality_fix_plan.md`` §4.1.
A new mart beside ``feat_price``; ``price.py`` and its SQL text are untouched (the
SQL text is a hash contract). Every column ends in ``_v2``.

Unified rules:

* Windows are counted over valid (non-halt) sessions from
  :func:`~modeler.etl.trading_panel.build_valid_session_sql`; halt rows get no value.
* A value exists only when the window holds its full ``n`` valid sessions
  (``LAG``/``COUNT`` over valid sessions), with no calendar-length cap.
* NULL when any ``ca_event`` falls inside the window. ``ca_event`` comes from
  ``quality_view``; without it ``ca_event`` is ``FALSE`` (same as ``price.py``).

CA mask windows: price-level windows (MA20, 52w high, range) mask their own rows;
return windows mask ``n`` rows back to ``t`` (``ROWS n PRECEDING``), covering both
endpoints; ``px_ret_60_20_v2`` masks ``t-60..t`` as the plan states; rolling
statistics of daily returns (vol, amihud, limit-hit) mask their ``n`` rows;
``px_ret_1d_v2`` (the single-day log return itself) is masked only on a ``ca_event``
day ``t`` — the corporate-action jump — and carries no window.
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import MartPlan, materialize, register_mart_view
from modeler.etl.trading_panel import build_valid_session_sql

PRICE_V2_TABLE = "feat_price_v2"
PRICE_V2_SEMANTICS = "price_v2"

# Same switch date as etl/quality.py (not exposed as a constant there).
PRICE_LIMIT_CHANGE_DATE = "2015-06-15"
PRICE_LIMIT_BEFORE = 0.15
PRICE_LIMIT_AFTER = 0.30
PRICE_LIMIT_MARGIN = 0.01

PRICE_V2_COLUMNS = (
    "px_ret_60_20_v2",
    "px_high_low_range_20d_v2",
    "px_ret_5d_v2",
    "px_ret_20d_v2",
    "px_ret_60d_v2",
    "px_vol_20d_v2",
    "px_vol_60d_v2",
    "px_gap_vs_ma20_v2",
    "px_dist_52w_high_v2",
    "px_amihud_20d_v2",
    "px_limit_hit_ratio_20d_v2",
    "px_ret_1d_v2",
)


def build_price_v2_sql(
    price_view: str = "daily_ohlcv",
    quality_view: str | None = None,
) -> str:
    """SQL producing ``feat_price_v2`` from a daily OHLCV view (keys: trade_date, ticker, market)."""
    quality_join = (
        f"LEFT JOIN {quality_view} q USING (trade_date, ticker, market)" if quality_view else ""
    )
    ca_event = "COALESCE(q.ca_event, FALSE)" if quality_view else "FALSE"
    part = "PARTITION BY ticker, market ORDER BY valid_session_idx"

    def win(n_preceding: int) -> str:
        return f"{part} ROWS BETWEEN {n_preceding} PRECEDING AND CURRENT ROW"

    return f"""
        WITH valid AS (
            {build_valid_session_sql(price_view)}
        ),
        vq AS (
            SELECT v.trade_date, v.ticker, v.market, v.valid_session_idx,
                   v.close_d, v.turnover, v.simple_ret, v.log_ret,
                   CAST(p.high AS DOUBLE) AS high_d,
                   CAST(p.low AS DOUBLE) AS low_d,
                   CASE WHEN v.trade_date < DATE '{PRICE_LIMIT_CHANGE_DATE}'
                        THEN {PRICE_LIMIT_BEFORE} ELSE {PRICE_LIMIT_AFTER} END
                        AS limit_pct,
                   {ca_event} AS ca_event
            FROM valid v
            JOIN {price_view} p USING (trade_date, ticker, market)
            {quality_join}
        ),
        raw AS (
            SELECT
                trade_date, ticker, market,
                CASE WHEN ca_event THEN NULL ELSE log_ret END AS r_ret1,
                LN(LAG(close_d, 20) OVER w / NULLIF(LAG(close_d, 60) OVER w, 0)) AS r_60_20,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w60 AS ca_60,
                CASE WHEN COUNT(high_d) OVER w19 = 20 AND MIN(low_d) OVER w19 > 0
                     THEN (MAX(high_d) OVER w19 - MIN(low_d) OVER w19) / NULLIF(close_d, 0) END
                    AS r_range,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w19 AS ca_19,
                LN(close_d / NULLIF(LAG(close_d, 5) OVER w, 0)) AS r_ret5,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w5 AS ca_5,
                LN(close_d / NULLIF(LAG(close_d, 20) OVER w, 0)) AS r_ret20,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w20 AS ca_20,
                LN(close_d / NULLIF(LAG(close_d, 60) OVER w, 0)) AS r_ret60,
                CASE WHEN COUNT(log_ret) OVER w19 = 20 THEN STDDEV_SAMP(log_ret) OVER w19 END
                    AS r_vol20,
                CASE WHEN COUNT(log_ret) OVER w59 = 60 THEN STDDEV_SAMP(log_ret) OVER w59 END
                    AS r_vol60,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w59 AS ca_59,
                CASE WHEN COUNT(close_d) OVER w19 = 20
                     THEN close_d / NULLIF(AVG(close_d) OVER w19, 0) - 1 END AS r_gap,
                CASE WHEN COUNT(close_d) OVER w251 = 252
                     THEN close_d / NULLIF(MAX(close_d) OVER w251, 0) - 1 END AS r_dist,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER w251 AS ca_251,
                CASE WHEN COUNT(log_ret) OVER w19 = 20
                     THEN AVG(ABS(log_ret) / NULLIF(turnover, 0)) OVER w19 END AS r_amihud,
                CASE WHEN COUNT(simple_ret) OVER w19 = 20
                     THEN AVG(CASE WHEN ABS(simple_ret) >= limit_pct - {PRICE_LIMIT_MARGIN}
                                   THEN 1.0 ELSE 0.0 END) OVER w19 END AS r_limit
            FROM vq
            WINDOW
                w AS ({part}),
                w5 AS ({win(5)}),
                w19 AS ({win(19)}),
                w20 AS ({win(20)}),
                w59 AS ({win(59)}),
                w60 AS ({win(60)}),
                w251 AS ({win(251)})
        )
        SELECT
            p.trade_date, p.ticker, p.market,
            CASE WHEN r.ca_60 = 0 THEN r.r_60_20 END AS px_ret_60_20_v2,
            CASE WHEN r.ca_19 = 0 THEN r.r_range END AS px_high_low_range_20d_v2,
            CASE WHEN r.ca_5 = 0 THEN r.r_ret5 END AS px_ret_5d_v2,
            CASE WHEN r.ca_20 = 0 THEN r.r_ret20 END AS px_ret_20d_v2,
            CASE WHEN r.ca_60 = 0 THEN r.r_ret60 END AS px_ret_60d_v2,
            CASE WHEN r.ca_19 = 0 THEN r.r_vol20 END AS px_vol_20d_v2,
            CASE WHEN r.ca_59 = 0 THEN r.r_vol60 END AS px_vol_60d_v2,
            CASE WHEN r.ca_19 = 0 THEN r.r_gap END AS px_gap_vs_ma20_v2,
            CASE WHEN r.ca_251 = 0 THEN r.r_dist END AS px_dist_52w_high_v2,
            CASE WHEN r.ca_19 = 0 THEN r.r_amihud END AS px_amihud_20d_v2,
            CASE WHEN r.ca_19 = 0 THEN r.r_limit END AS px_limit_hit_ratio_20d_v2,
            r.r_ret1 AS px_ret_1d_v2
        FROM {price_view} p
        LEFT JOIN raw r USING (trade_date, ticker, market)
    """


def materialize_price_v2(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    price_view: str = "daily_ohlcv",
    quality_view: str | None = None,
    force: bool = False,
) -> str:
    """Build + register the ``feat_price_v2`` mart view. Returns the view name."""
    sql = build_price_v2_sql(price_view, quality_view=quality_view)
    plan = MartPlan(plan_id="single", final_sql=sql, semantics_version=PRICE_V2_SEMANTICS)
    materialize(con, config, PRICE_V2_TABLE, sql, force=force, plan=plan)
    return register_mart_view(con, config, PRICE_V2_TABLE)
