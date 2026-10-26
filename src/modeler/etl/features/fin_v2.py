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

Six more ratios replace the remaining legacy ``fin_pit.py:145-165`` ones (same meaning, vintage
path, same-report rule): ``fin_operating_margin_v2`` (TTM operating income / TTM revenue,
revenue > 0), ``fin_debt_to_equity_v2`` (total liabilities / equity, equity > 0),
``fin_equity_ratio_v2`` (equity / assets, assets > 0, negative equity gives a negative ratio),
``fin_ocf_to_assets_v2`` (TTM operating cash flow / assets), ``fin_cash_ratio_v2`` (cash and
equivalents / assets), ``fin_asset_turnover_v2`` (TTM revenue / assets). Each is NULL unless
numerator and denominator carry the same ``rcept_no`` and ``BASE_OK`` holds. Flow vs. instant
is already settled by ``fin_vintage`` (``daily_value`` is the TTM value for flow metrics and
the quarter-end value for balance-sheet metrics), so the ratios use ``daily_value`` as is and
a flow/flow ratio has both sides TTM.

The vintage mart only holds firms still in ``stock_master``; the next preregistration passes
``vintage_view="fin_quarterly_metric_vintage_dl"`` (``etl/marts/metric_vintages_delisted.py``)
so delisted firms are covered too. The join is on ``ticker`` only, so nothing else changes.

One sensitivity column, ``fin_ttm_conflict_v2`` (G-14, R1): TRUE when the TTM earnings value behind
``fin_roe_v2`` / ``fin_roa_v2`` on that row includes a conflict quarter, read from
``ttm_includes_conflict`` of the vintage row the value came from (same basis, same ``rcept_no``
selection as the ratio itself; the flag of the numerator that was picked). It is defined where
at least one of the two ratios is non-NULL: TRUE if the earnings of a non-NULL ratio carry a
conflict, else FALSE. With ``vintage_has_ttm_conflict=False`` (``_dl``, no such column) the column
is NULL. It is not a model feature.

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

_METRICS = (
    "total_equity",
    "controlling_net_income",
    "net_income",
    "total_assets",
    "total_liabilities",
    "operating_income",
    "revenue",
    "operating_cash_flow",
    "cash_and_cash_equivalents",
)


def _selected(metric: str, alias: str) -> str:
    """``metric``'s value and ``rcept_no`` under the ticker-date's one selected basis."""
    return f"""CASE WHEN v_net_income_cfs IS NOT NULL THEN v_{metric}_cfs
                 ELSE v_{metric}_ofs END AS {alias},
            CASE WHEN v_net_income_cfs IS NOT NULL THEN r_{metric}_cfs
                 ELSE r_{metric}_ofs END AS {alias}_rcept"""


_INTERVAL_SELECT = "ticker, metric_code, fs_basis, daily_value, value_lag_4q,"
_POINTS_TAIL = "            value_lag_4q\n        FROM "
#: metrics whose TTM value can include a conflict quarter (the ROE / ROA numerators)
_CONFLICT_METRICS = ("controlling_net_income", "net_income")


def _intervals_with_rcept(vintage_view: str, ttm_conflict: bool = False) -> str:
    """``build_metric_intervals_cte`` with ``rcept_no`` (and ``conflict_flag``) added.

    ``conflict_flag`` is the vintage row's ``ttm_includes_conflict``; it travels with
    ``daily_value`` through ``metric_points`` and ``metric_intervals``, so it is the flag of the
    very row the interval's value came from.
    """
    cte = build_metric_intervals_cte(vintage_view, _METRICS)
    if cte.count(_INTERVAL_SELECT) != 1:
        raise RuntimeError("fin_vintage metric_intervals text changed; fin_v2 must be updated")
    extra = ""
    if ttm_conflict:
        tail = f"{_POINTS_TAIL}{vintage_view}"
        if cte.count(tail) != 1:
            raise RuntimeError("fin_vintage metric_points text changed; fin_v2 must be updated")
        cte = cte.replace(
            tail,
            "            value_lag_4q,\n"
            "            ttm_includes_conflict AS conflict_flag\n"
            f"        FROM {vintage_view}",
        )
        extra = " conflict_flag,"
    return cte.replace(
        _INTERVAL_SELECT,
        f"ticker, metric_code, fs_basis, daily_value, value_lag_4q, rcept_no,{extra}",
    )


def build_fin_v2_sql(
    *,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    vintage_view: str = "fin_quarterly_metric_vintage",
    join_plan: str = JOIN_PLAN_COALESCE,
    vintage_has_ttm_conflict: bool = False,
) -> str:
    """SQL producing ``feat_fin_v2`` (``fin_roe_v2``).

    ``vintage_has_ttm_conflict``: the vintage view has ``ttm_includes_conflict`` (the ``_dlr``
    mart); ``fin_ttm_conflict_v2`` is then filled, else it is a NULL boolean column.

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
    if vintage_has_ttm_conflict:
        value_cols += ",\n            " + ",\n            ".join(
            f"m_{m}_{b.lower()}.conflict_flag AS c_{m}_{b.lower()}"
            for m in _CONFLICT_METRICS
            for b in FS_BASES
        )
        conflict_resolved = """,
            -- conflict flag of the numerator each ratio picked (same branch as above)
            CASE WHEN v_net_income_cfs IS NOT NULL
                 THEN CASE WHEN r_controlling_net_income_cfs = r_total_equity_cfs
                           THEN c_controlling_net_income_cfs
                           WHEN r_net_income_cfs = r_total_equity_cfs THEN c_net_income_cfs END
                 ELSE CASE WHEN r_controlling_net_income_ofs = r_total_equity_ofs
                           THEN c_controlling_net_income_ofs
                           WHEN r_net_income_ofs = r_total_equity_ofs THEN c_net_income_ofs END
            END AS earnings_conflict,
            CASE WHEN v_net_income_cfs IS NOT NULL
                 THEN CASE WHEN r_controlling_net_income_cfs = r_total_assets_cfs
                           THEN c_controlling_net_income_cfs
                           WHEN r_net_income_cfs = r_total_assets_cfs THEN c_net_income_cfs END
                 ELSE CASE WHEN r_controlling_net_income_ofs = r_total_assets_ofs
                           THEN c_controlling_net_income_ofs
                           WHEN r_net_income_ofs = r_total_assets_ofs THEN c_net_income_ofs END
            END AS earnings_for_assets_conflict"""
        conflict_out = """,
        CASE WHEN (base_ok AND equity > 0 AND earnings IS NOT NULL)
                  OR (base_ok AND assets > 0 AND earnings_for_assets IS NOT NULL)
             THEN (COALESCE(earnings_conflict, FALSE)
                   AND base_ok AND equity > 0 AND earnings IS NOT NULL)
                  OR (COALESCE(earnings_for_assets_conflict, FALSE)
                      AND base_ok AND assets > 0 AND earnings_for_assets IS NOT NULL)
        END AS fin_ttm_conflict_v2"""
    else:
        conflict_resolved = ""
        conflict_out = ",\n        CAST(NULL AS BOOLEAN) AS fin_ttm_conflict_v2"
    return f"""
    WITH {_intervals_with_rcept(vintage_view, vintage_has_ttm_conflict)},
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
            END AS earnings_for_assets,
            {_selected("total_liabilities", "liabilities")},
            {_selected("operating_income", "op_income")},
            {_selected("revenue", "revenue")},
            {_selected("operating_cash_flow", "ocf")},
            {_selected("cash_and_cash_equivalents", "cash")},
            CASE WHEN v_net_income_cfs IS NOT NULL THEN r_total_assets_cfs
                 ELSE r_total_assets_ofs END AS assets_rcept{conflict_resolved}
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
        CASE WHEN base_ok AND equity IS NOT NULL THEN equity <= 0 END AS fin_capital_impaired_v2,
        CASE WHEN base_ok AND revenue > 0 AND op_income IS NOT NULL
                  AND op_income_rcept = revenue_rcept
             THEN op_income / revenue END AS fin_operating_margin_v2,
        CASE WHEN base_ok AND equity > 0 AND liabilities IS NOT NULL
                  AND liabilities_rcept = equity_rcept
             THEN liabilities / equity END AS fin_debt_to_equity_v2,
        CASE WHEN base_ok AND assets > 0 AND equity IS NOT NULL
                  AND equity_rcept = assets_rcept
             THEN equity / assets END AS fin_equity_ratio_v2,
        CASE WHEN base_ok AND assets > 0 AND ocf IS NOT NULL
                  AND ocf_rcept = assets_rcept
             THEN ocf / assets END AS fin_ocf_to_assets_v2,
        CASE WHEN base_ok AND assets > 0 AND cash IS NOT NULL
                  AND cash_rcept = assets_rcept
             THEN cash / assets END AS fin_cash_ratio_v2,
        CASE WHEN base_ok AND assets > 0 AND revenue IS NOT NULL
                  AND revenue_rcept = assets_rcept
             THEN revenue / assets END AS fin_asset_turnover_v2{conflict_out}
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
    vintage_has_ttm_conflict: bool = False,
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
            vintage_has_ttm_conflict=vintage_has_ttm_conflict,
        ),
        force=force,
    )
    return register_mart_view(con, config, FIN_V2_TABLE)
