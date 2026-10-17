"""``feat_fin_v2`` — ROE on the vintage path (15_quality_fix_plan §4.4).

``fin_pit.py``'s ``fin_roe`` divides by ``NULLIF(equity, 0)``, so a negative equity flips
the sign, and each metric is as-of'd on its own (a net income and an equity from different
filings can meet on one day). ``fin_roe_v2`` reads ``fin_quarterly_metric_vintage`` through
the shared primitives in ``fin_vintage`` — one ``fs_basis`` per ticker-date (CFS if
``net_income`` has a CFS value, else OFS, exactly as ``fin_scan.py`` decides), TTM for flow
metrics, same-day candidates resolved to the latest fiscal period — and adds three rules:

* numerator = ``COALESCE(controlling_net_income, net_income)`` TTM, denominator =
  ``total_equity`` instant, both under that one basis;
* the two must come from the **same filing**: the vintage keeps a per-metric interval, so on
  a given day the numerator and denominator can be the current figure of different filings
  (one metric re-filed, the other not yet). ``rcept_no`` is in the vintage rows but dropped
  by ``metric_intervals``; it is carried through here and the ratio is NULL unless equal;
* NULL unless ``total_equity > 0`` and ``BASE_OK_SQL`` (the same gate as ``fin_scan``).

``fin_vintage.py`` and ``fin_scan.py`` are not modified; the interval CTE text is theirs,
with one column added.
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.features.fin_scan import _panel_source
from modeler.etl.features.fin_vintage import (
    BASE_OK_SQL,
    FS_BASES,
    JOIN_PLAN_OR,
    build_metric_intervals_cte,
    build_metric_joins,
)
from modeler.etl.mart import materialize, register_mart_view

FIN_V2_TABLE = "feat_fin_v2"

_METRICS = ("total_equity", "controlling_net_income", "net_income")

_INTERVAL_SELECT = "ticker, metric_code, fs_basis, daily_value, value_lag_4q,"


def _intervals_with_rcept(vintage_view: str) -> str:
    """``build_metric_intervals_cte`` with ``rcept_no`` added to ``metric_intervals``."""
    cte = build_metric_intervals_cte(vintage_view, _METRICS)
    if cte.count(_INTERVAL_SELECT) != 1:
        raise RuntimeError("fin_vintage metric_intervals text changed; fin_v2 must be updated")
    return cte.replace(
        _INTERVAL_SELECT, "ticker, metric_code, fs_basis, daily_value, value_lag_4q, rcept_no,"
    )


def build_fin_v2_sql(
    *,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    vintage_view: str = "fin_quarterly_metric_vintage",
    join_plan: str = JOIN_PLAN_OR,
) -> str:
    """SQL producing ``feat_fin_v2`` (``fin_roe_v2``)."""
    joins = build_metric_joins(_METRICS, FS_BASES, join_plan)
    value_cols = ",\n            ".join(
        f"m_{m}_{b.lower()}.daily_value AS v_{m}_{b.lower()}, "
        f"m_{m}_{b.lower()}.rcept_no AS r_{m}_{b.lower()}"
        for m in _METRICS
        for b in FS_BASES
    )
    return f"""
    WITH {_intervals_with_rcept(vintage_view)},
    panel AS (
        SELECT
            pit.trade_date, pit.ticker, pit.market,
            pit.market_cap_pit, pit.shares_is_available, pit.shares_invalid_flag,
            q.is_halted, q.valid_session_idx
        {_panel_source(pit_view, quality_view)}
    ),
    joined AS (
        SELECT panel.*,
            {value_cols}
        FROM panel
        {joins}
    ),
    resolved AS (
        SELECT
            trade_date, ticker, market,
            market_cap_pit, shares_is_available, shares_invalid_flag,
            is_halted, valid_session_idx,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_total_equity_cfs
                 ELSE v_total_equity_ofs END AS equity,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN r_total_equity_cfs
                 ELSE r_total_equity_ofs END AS equity_rcept,
            CASE WHEN v_net_income_cfs IS NOT NULL
                 THEN COALESCE(v_controlling_net_income_cfs, v_net_income_cfs)
                 ELSE COALESCE(v_controlling_net_income_ofs, v_net_income_ofs) END AS earnings,
            CASE WHEN v_net_income_cfs IS NOT NULL
                 THEN CASE WHEN v_controlling_net_income_cfs IS NOT NULL
                           THEN r_controlling_net_income_cfs ELSE r_net_income_cfs END
                 ELSE CASE WHEN v_controlling_net_income_ofs IS NOT NULL
                           THEN r_controlling_net_income_ofs ELSE r_net_income_ofs END
            END AS earnings_rcept
        FROM joined
    ),
    scored AS (
        SELECT *, {BASE_OK_SQL} AS base_ok
        FROM resolved
    )
    SELECT
        trade_date, ticker, market,
        CASE WHEN base_ok AND equity > 0 AND earnings IS NOT NULL
                  AND earnings_rcept IS NOT NULL AND earnings_rcept = equity_rcept
             THEN earnings / equity END AS fin_roe_v2
    FROM scored
    """


def materialize_fin_v2(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    vintage_view: str = "fin_quarterly_metric_vintage",
    force: bool = False,
) -> str:
    """Build + register the ``feat_fin_v2`` mart view. Returns the view name."""
    materialize(
        con,
        config,
        FIN_V2_TABLE,
        build_fin_v2_sql(pit_view=pit_view, quality_view=quality_view, vintage_view=vintage_view),
        force=force,
    )
    return register_mart_view(con, config, FIN_V2_TABLE)
