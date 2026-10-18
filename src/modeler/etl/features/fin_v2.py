"""``feat_fin_v2`` — ROE on the vintage path (15_quality_fix_plan §4.4).

``fin_pit.py``'s ``fin_roe`` divides by ``NULLIF(equity, 0)``, so a negative equity flips
the sign, and each metric is as-of'd on its own (a net income and an equity from different
filings can meet on one day). ``fin_roe_v2`` reads ``fin_quarterly_metric_vintage`` through
the shared primitives in ``fin_vintage`` — one ``fs_basis`` per ticker-date (CFS if
``net_income`` has a CFS value, else OFS, exactly as ``fin_scan.py`` decides), TTM for flow
metrics, same-day candidates resolved to the latest fiscal period — and adds three rules:

* numerator = TTM ``controlling_net_income`` if its ``rcept_no`` equals the equity's, else TTM
  ``net_income`` if its ``rcept_no`` does, else NULL; denominator = ``total_equity`` instant;
  both under that one basis;
* the two must come from the **same filing**: the vintage keeps a per-metric interval, so on
  a given day the numerator and denominator can be the current figure of different filings
  (one metric re-filed, the other not yet). ``rcept_no`` is in the vintage rows but dropped
  by ``metric_intervals``; it is carried through here and the numerator is picked only
  from the equity's own filing;
* NULL unless ``total_equity > 0`` and ``BASE_OK_SQL`` (the same gate as ``fin_scan``).

Two more columns, from the same selected basis and gate:

* ``fin_roa_v2`` = TTM earnings / ``total_assets``, with the same "same report" rule against the
  ``total_assets`` ``rcept_no`` (controlling, else net income, else NULL); NULL unless
  ``total_assets > 0``. ROE mixes leverage into the ratio and drops every firm with
  ``equity <= 0``; assets are positive for firms ROE cannot score, so ROA keeps the deficit tail.
* ``fin_capital_impaired_v2`` = ``total_equity <= 0`` (TRUE / FALSE; NULL when equity is
  missing or ``BASE_OK`` fails), so the tail can include the capital-impaired firms that
  ``fin_roe_v2`` drops.

The vintage mart only holds firms still in ``stock_master``; the next preregistration passes
``vintage_view="fin_quarterly_metric_vintage_dl"`` (``etl/marts/metric_vintages_delisted.py``)
so delisted firms are covered too. The join is on ``ticker`` only, so nothing else changes.

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
    JOIN_PLAN_COALESCE,
    build_metric_intervals_cte,
    build_metric_joins,
)
from modeler.etl.mart import materialize, register_mart_view

FIN_V2_TABLE = "feat_fin_v2"

_METRICS = ("total_equity", "controlling_net_income", "net_income", "total_assets")

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
    join_plan: str = JOIN_PLAN_COALESCE,
) -> str:
    """SQL producing ``feat_fin_v2`` (``fin_roe_v2``).

    ``join_plan`` defaults to ``coalesce_join`` (hash join on ticker; 0.5 s against 361.7 s for
    the ``or`` text on the dev data, same rows). This is a new mart, so there is no hash
    contract to keep.
    """
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
            -- numerator: controlling net income if it comes from the equity's filing, else
            -- net income if that does; never a figure from another filing (§4.4 "same report")
            CASE WHEN v_net_income_cfs IS NOT NULL
                 THEN CASE WHEN r_controlling_net_income_cfs = r_total_equity_cfs
                           THEN v_controlling_net_income_cfs
                           WHEN r_net_income_cfs = r_total_equity_cfs THEN v_net_income_cfs END
                 ELSE CASE WHEN r_controlling_net_income_ofs = r_total_equity_ofs
                           THEN v_controlling_net_income_ofs
                           WHEN r_net_income_ofs = r_total_equity_ofs THEN v_net_income_ofs END
            END AS earnings,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_total_assets_cfs
                 ELSE v_total_assets_ofs END AS assets,
            -- ROA numerator: same rule, against the total_assets filing
            CASE WHEN v_net_income_cfs IS NOT NULL
                 THEN CASE WHEN r_controlling_net_income_cfs = r_total_assets_cfs
                           THEN v_controlling_net_income_cfs
                           WHEN r_net_income_cfs = r_total_assets_cfs THEN v_net_income_cfs END
                 ELSE CASE WHEN r_controlling_net_income_ofs = r_total_assets_ofs
                           THEN v_controlling_net_income_ofs
                           WHEN r_net_income_ofs = r_total_assets_ofs THEN v_net_income_ofs END
            END AS earnings_for_assets
        FROM joined
    ),
    scored AS (
        SELECT *, {BASE_OK_SQL} AS base_ok
        FROM resolved
    )
    SELECT
        trade_date, ticker, market,
        CASE WHEN base_ok AND equity > 0 AND earnings IS NOT NULL
             THEN earnings / equity END AS fin_roe_v2,
        CASE WHEN base_ok AND assets > 0 AND earnings_for_assets IS NOT NULL
             THEN earnings_for_assets / assets END AS fin_roa_v2,
        CASE WHEN base_ok AND equity IS NOT NULL THEN equity <= 0 END AS fin_capital_impaired_v2
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
    join_plan: str = JOIN_PLAN_COALESCE,
) -> str:
    """Build + register the ``feat_fin_v2`` mart view. Returns the view name."""
    materialize(
        con,
        config,
        FIN_V2_TABLE,
        build_fin_v2_sql(
            pit_view=pit_view,
            quality_view=quality_view,
            vintage_view=vintage_view,
            join_plan=join_plan,
        ),
        force=force,
    )
    return register_mart_view(con, config, FIN_V2_TABLE)
