"""``feat_fin_scan_daily`` — Phase B B-4 (04_specific_plan_B.md §4.1-§4.3, B-4).

Daily PIT materialization of the 5 continuous financial families (size,
value, profitability, asset growth, accruals) from ``fin_quarterly_metric_vintage``
(B-3) interval-joined onto A0's own PIT shares/market-cap
(``dim_stock_pit_daily``) and price-quality (``dim_price_quality_daily``) marts.

Grain: ``(trade_date, ticker, market)``, one row per raw ``daily_ohlcv`` row
(broad/tradable filtering happens downstream, same layering as ``feat_price``/
``feat_flow`` — §5.1: "official sample은 ... broad + common formation + ...").

Shared fs_basis per ticker-date (§3.6): rather than resolving CFS/OFS
independently per metric (which could silently mix bases across a single
day's feature bundle), every metric is interval-joined under *both* bases,
then one ``fs_basis_used`` decision — CFS if ``net_income`` has a CFS value at
this date, else OFS — selects every other metric's value for that date. This
is what makes ``fin_accruals_to_assets``'s net_income/CFO/avg_assets a genuine
"same four-quarter set, same fs basis" computation (§4.3), not three
independently-chosen bases that happen to collide.
"""

from __future__ import annotations

from collections.abc import Sequence

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.features.fin_vintage import (
    BASE_OK_SQL,
    FS_BASES,
    JOIN_PLAN_OR,
    build_metric_intervals_cte,
    build_metric_joins,
)
from modeler.etl.mart import MartPlan, materialize, register_mart_view

FIN_SCAN_TABLE = "feat_fin_scan_daily"

SEMANTICS_V1 = "v1"
SEMANTICS_V2 = "v2"
SEMANTICS_VERSIONS = (SEMANTICS_V1, SEMANTICS_V2)
#: ``v2`` takes each cross-sectional mean and standard deviation over the whole
#: ``(trade_date, market)`` partition in ``ticker`` order. The frame is unchanged
#: (all rows, so the value is the same mean), only the order of the float additions is
#: fixed. Without it the additions follow the partition's arrival order, which varies
#: from run to run and moves ``fin_value_z`` by about 1e-14.
_ORDERED_WHOLE_PARTITION = (
    "ORDER BY ticker ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING"
)

#: N2-9 diagnostic table name. A separate mart, never a replacement: the
#: industry variant is not PIT, so both have to exist side by side to be
#: comparable at all.
FIN_SCAN_INDUSTRY_TABLE = "feat_fin_scan_daily_ind"

#: Cross-section the industry-neutral variant normalises within. Banks, biotech,
#: shipbuilders and game studios currently z-score against one KOSPI pool; Barra
#: keeps industry as a first-class block for exactly this reason.
CROSS_SECTION_WITH_INDUSTRY = "trade_date, market, industry_group"

# §1.3 fingerprint, same role as ``EVENT_FEATURE_FORMULA_VERSION`` for the event
# features: the formula/handling rules below are not covered by ``config_hash``
# (the scan YAML) or ``phase_b_code_hash`` (which only hashes
# ``horizon_scan_phase_b*.py``), so a change here would otherwise produce
# different numbers under identical run-spec fingerprints. Bump it whenever the
# ratio definitions, the winsorize/z-score handling, or the fs_basis rule change,
# and do not reuse an existing artifact of the same snapshot across a bump.
#
#   fin_v1 — the 2026-08 rules as first scanned (§4.1-§4.3).
#   fin_v2 — value components keep NULL through the winsorize step, so the
#            ">= 2 valid components" rule actually binds (10_known_issues.md I1).
#   fin_v3 — same-day metric candidates resolve to the latest fiscal period
#            instead of scan order (10_known_issues.md I12).
#   fin_v4 — I7. The canonical metric rules gained an XBRL fallback for the
#            financial-statement metrics, and the vintage mart now takes an
#            XBRL candidate's fs_basis from the rule instead of hardcoding ''.
#            revenue goes from 8,103 rows to ~148,000, so fin_sales_to_price
#            and fin_gross_profitability change by more than any tweak this
#            fingerprint has covered before. The mapping rules live in
#            collector.kr.definitions and are covered by neither config_hash
#            (the scan YAML) nor phase_b_code_hash (the analysis modules), so
#            without this bump a re-run would reuse the old Phase B artifact
#            under an identical fingerprint while producing different numbers.
FIN_FEATURE_FORMULA_VERSION = "fin_v4"

# Metrics interval-joined onto the daily panel; total_assets additionally
# carries its own value_lag_4q (B-3) for avg_assets / asset growth.
_METRICS = (
    "total_equity",
    "controlling_net_income",
    "operating_cash_flow",
    "revenue",
    "cogs",
    "gross_profit",
    "operating_income",
    "net_income",
    "total_assets",
)
#: F-4.1 moved the interval/join/base_ok builders to ``fin_vintage`` so
#: ``feat_fin_risk`` reads the vintage under the same rules. Extraction only —
#: the emitted text is byte-identical, which
#: ``test_fin_risk.py::test_the_fin_scan_sql_is_byte_identical`` pins.
_BASES = FS_BASES

#: Key columns of the mart, in output order.
FIN_SCAN_KEY_COLUMNS = ("trade_date", "ticker", "market")

#: ``fin_log_mcap`` is defined here once. The full mart and the column projection below
#: both embed this text (``base_ok`` is :data:`BASE_OK_SQL`), so the two cannot disagree.
FIN_LOG_MCAP_SQL = "CASE WHEN base_ok THEN ln(market_cap_pit) END"

#: Columns the projection can produce, in the full mart's output order, with their
#: expressions over the ``scored`` relation. A column without an entry is not supported:
#: the projection fails instead of dropping it.
PROJECTION_EXPRESSIONS: dict[str, str] = {"fin_log_mcap": FIN_LOG_MCAP_SQL}

PLAN_PROJECTION = "projection"


def _panel_source(pit_view: str, quality_view: str) -> str:
    """The ``FROM`` of the ``panel`` CTE (before the optional industry join)."""
    return f"""FROM {pit_view} pit
        LEFT JOIN {quality_view} q USING (trade_date, ticker, market)"""


def build_fin_scan_daily_sql(
    *,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    vintage_view: str = "fin_quarterly_metric_vintage",
    industry_view: str | None = None,
    join_plan: str = JOIN_PLAN_OR,
    semantics: str = SEMANTICS_V1,
) -> str:
    """SQL producing ``feat_fin_scan_daily`` from B-3's quarterly vintage mart.

    Args:
        pit_view: A0's PIT shares/market-cap mart.
        quality_view: A0's price-quality mart.
        vintage_view: B-3's quarterly metric vintage mart.
        industry_view: When given, a ``(ticker, industry_group)`` view; every
            cross-sectional winsorize and z-score then partitions on
            ``(trade_date, market, industry_group)`` instead of
            ``(trade_date, market)``. This is the N2-9 **diagnostic** variant
            and is not point-in-time — ``induty_code`` carries today's industry
            backwards — so it belongs beside the existing path, never in place
            of it. Left ``None`` the emitted SQL is unchanged, which is what
            keeps the frozen parity tests meaningful.
        join_plan: How the 18 interval joins state their upper bound
            (:func:`~modeler.etl.features.fin_vintage.build_metric_join`). The
            default is the text every frozen mart was written under; the other
            plan returns the same rows with a different text, so it is a
            ``MartPlan`` and never the cache contract.
        semantics: ``"v2"`` fixes the float summation order of the cross-sectional
            z-scores (see :data:`_ORDERED_WHOLE_PARTITION`). It is a different meaning
            (``fin_value_z`` moves in the last digits), hence a different ``sql_hash``.
    """
    if semantics not in SEMANTICS_VERSIONS:
        raise ValueError(
            f"unknown fin_scan semantics {semantics!r}; expected {SEMANTICS_VERSIONS}"
        )
    joins = build_metric_joins(_METRICS, _BASES, join_plan)

    # One string, used by every winsorize percentile and every z-score, so the
    # variant cannot end up neutralising some components and not others.
    cross_section = "trade_date, market" if industry_view is None else CROSS_SECTION_WITH_INDUSTRY
    # The window of every mean and standard deviation. v1 states only the partition.
    moments = f"PARTITION BY {cross_section}" + (
        f" {_ORDERED_WHOLE_PARTITION}" if semantics == SEMANTICS_V2 else ""
    )
    industry_join = (
        "" if industry_view is None else f"\n        LEFT JOIN {industry_view} ind USING (ticker)"
    )
    industry_column = "" if industry_view is None else ",\n            ind.industry_group"
    industry_passthrough = "" if industry_view is None else ", industry_group"

    return f"""
    WITH {build_metric_intervals_cte(vintage_view, _METRICS)},
    panel AS (
        SELECT
            pit.trade_date, pit.ticker, pit.market,
            pit.market_cap_pit, pit.issued_shares_pit,
            pit.shares_is_available, pit.shares_invalid_flag, pit.shares_available_from,
            q.is_halted, q.valid_session_idx{industry_column}
        {_panel_source(pit_view, quality_view)}{industry_join}
    ),
    joined AS (
        SELECT panel.*,
            {",\n            ".join(
                f"m_{m}_{b.lower()}.daily_value AS v_{m}_{b.lower()}, "
                f"m_{m}_{b.lower()}.daily_available_from AS a_{m}_{b.lower()}"
                for m in _METRICS for b in _BASES
            )},
            m_total_assets_cfs.value_lag_4q AS v_total_assets_cfs_lag4q,
            m_total_assets_ofs.value_lag_4q AS v_total_assets_ofs_lag4q
        FROM panel
        {joins}
    ),
    resolved AS (
        SELECT
            trade_date, ticker, market{industry_passthrough},
            market_cap_pit, shares_is_available, shares_invalid_flag,
            is_halted, valid_session_idx,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN 'CFS'
                 WHEN v_net_income_ofs IS NOT NULL THEN 'OFS' END AS fs_basis_used,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_total_equity_cfs
                 ELSE v_total_equity_ofs END AS total_equity_selected,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_controlling_net_income_cfs
                 ELSE v_controlling_net_income_ofs END AS controlling_net_income_selected,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_operating_cash_flow_cfs
                 ELSE v_operating_cash_flow_ofs END AS operating_cash_flow_selected,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_revenue_cfs
                 ELSE v_revenue_ofs END AS revenue_selected,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_cogs_cfs
                 ELSE v_cogs_ofs END AS cogs_selected,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_gross_profit_cfs
                 ELSE v_gross_profit_ofs END AS gross_profit_selected,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_operating_income_cfs
                 ELSE v_operating_income_ofs END AS operating_income_selected,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_net_income_cfs
                 ELSE v_net_income_ofs END AS net_income_selected,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_total_assets_cfs
                 ELSE v_total_assets_ofs END AS total_assets_selected,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN v_total_assets_cfs_lag4q
                 ELSE v_total_assets_ofs_lag4q END AS total_assets_lag4q_selected,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN a_net_income_cfs
                 ELSE a_net_income_ofs END AS net_income_available_from,
            CASE WHEN v_net_income_cfs IS NOT NULL
                 THEN greatest(a_total_equity_cfs, a_net_income_cfs)
                 ELSE greatest(a_total_equity_ofs, a_net_income_ofs) END
                AS value_available_from,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN a_gross_profit_cfs
                 ELSE a_gross_profit_ofs END AS profitability_available_from,
            CASE WHEN v_net_income_cfs IS NOT NULL THEN a_total_assets_cfs
                 ELSE a_total_assets_ofs END AS asset_growth_available_from,
            CASE WHEN v_net_income_cfs IS NOT NULL
                 THEN greatest(a_net_income_cfs, a_operating_cash_flow_cfs, a_total_assets_cfs)
                 ELSE greatest(a_net_income_ofs, a_operating_cash_flow_ofs, a_total_assets_ofs) END
                AS accruals_available_from
        FROM joined
    ),
    scored AS (
        SELECT
            *,
            {BASE_OK_SQL} AS base_ok,
            (total_equity_selected IS NOT NULL AND total_equity_selected <= 0) AS negative_equity,
            CASE WHEN total_assets_selected > 0 AND total_assets_lag4q_selected > 0
                 THEN (total_assets_selected + total_assets_lag4q_selected) / 2 END AS avg_assets,
            CASE WHEN gross_profit_selected IS NOT NULL THEN gross_profit_selected
                 WHEN revenue_selected IS NOT NULL AND cogs_selected IS NOT NULL
                      THEN revenue_selected - cogs_selected
            END AS gross_profit_effective,
            CASE WHEN gross_profit_selected IS NOT NULL THEN 'direct'
                 WHEN revenue_selected IS NOT NULL AND cogs_selected IS NOT NULL
                      THEN 'revenue_minus_cogs_fallback'
            END AS gross_profit_source
        FROM resolved
    ),
    ratios AS (
        SELECT
            trade_date, ticker, market{industry_passthrough}, fs_basis_used, negative_equity,
            gross_profit_source, value_available_from, profitability_available_from,
            asset_growth_available_from, accruals_available_from,
            {FIN_LOG_MCAP_SQL} AS fin_log_mcap,
            CASE WHEN base_ok AND total_equity_selected > 0
                 THEN total_equity_selected / market_cap_pit END AS fin_book_to_market,
            CASE WHEN base_ok THEN controlling_net_income_selected / market_cap_pit
            END AS fin_earnings_yield,
            CASE WHEN base_ok THEN operating_cash_flow_selected / market_cap_pit
            END AS fin_cfo_yield,
            CASE WHEN base_ok THEN revenue_selected / market_cap_pit
            END AS fin_sales_to_price,
            CASE WHEN avg_assets > 0 THEN gross_profit_effective / avg_assets END
                AS fin_gross_profitability,
            CASE WHEN avg_assets > 0 THEN operating_income_selected / avg_assets END
                AS fin_operating_profitability,
            CASE WHEN total_assets_lag4q_selected > 0
                 THEN total_assets_selected / total_assets_lag4q_selected - 1
            END AS fin_asset_growth_yoy,
            CASE WHEN avg_assets > 0 AND net_income_selected IS NOT NULL
                      AND operating_cash_flow_selected IS NOT NULL
                 THEN (net_income_selected - operating_cash_flow_selected) / avg_assets
            END AS fin_accruals_to_assets
        FROM scored
    ),
    -- §4.1: winsorize each value component at its own (trade_date, market)
    -- 1st/99th percentile, then z-score the winsorized series.
    --
    -- Each clip is guarded by ``WHEN <ratio> IS NULL THEN NULL``: DuckDB's
    -- GREATEST/LEAST *skip* NULL arguments, so a bare
    -- ``LEAST(GREATEST(NULL, p01), p99)`` returns p01. Without the guard a
    -- company with no financials at all is silently imputed to the market's
    -- 1st percentile on every component, counted as 4 valid components, and
    -- pinned to the "most expensive" end of the cross-section (fin_v1
    -- behaviour: 29.2% of emitted values broke the >= 2 component rule — see
    -- docs/dev/20260731_raw_features/01_feature_candidate/10_known_issues.md I1).
    winsorized AS (
        SELECT
            *,
            CASE WHEN fin_book_to_market IS NULL THEN NULL ELSE
                LEAST(GREATEST(fin_book_to_market,
                    quantile_cont(fin_book_to_market, 0.01)
                        OVER (PARTITION BY {cross_section})),
                    quantile_cont(fin_book_to_market, 0.99)
                        OVER (PARTITION BY {cross_section}))
            END AS w_bm,
            CASE WHEN fin_earnings_yield IS NULL THEN NULL ELSE
                LEAST(GREATEST(fin_earnings_yield,
                    quantile_cont(fin_earnings_yield, 0.01)
                        OVER (PARTITION BY {cross_section})),
                    quantile_cont(fin_earnings_yield, 0.99)
                        OVER (PARTITION BY {cross_section}))
            END AS w_ep,
            CASE WHEN fin_cfo_yield IS NULL THEN NULL ELSE
                LEAST(GREATEST(fin_cfo_yield,
                    quantile_cont(fin_cfo_yield, 0.01)
                        OVER (PARTITION BY {cross_section})),
                    quantile_cont(fin_cfo_yield, 0.99)
                        OVER (PARTITION BY {cross_section}))
            END AS w_cfop,
            CASE WHEN fin_sales_to_price IS NULL THEN NULL ELSE
                LEAST(GREATEST(fin_sales_to_price,
                    quantile_cont(fin_sales_to_price, 0.01)
                        OVER (PARTITION BY {cross_section})),
                    quantile_cont(fin_sales_to_price, 0.99)
                        OVER (PARTITION BY {cross_section}))
            END AS w_sp
        FROM ratios
    ),
    zscored AS (
        SELECT
            *,
            (w_bm - AVG(w_bm) OVER ({moments}))
                / NULLIF(STDDEV_SAMP(w_bm) OVER ({moments}), 0) AS z_bm,
            (w_ep - AVG(w_ep) OVER ({moments}))
                / NULLIF(STDDEV_SAMP(w_ep) OVER ({moments}), 0) AS z_ep,
            (w_cfop - AVG(w_cfop) OVER ({moments}))
                / NULLIF(STDDEV_SAMP(w_cfop) OVER ({moments}), 0) AS z_cfop,
            (w_sp - AVG(w_sp) OVER ({moments}))
                / NULLIF(STDDEV_SAMP(w_sp) OVER ({moments}), 0) AS z_sp
        FROM winsorized
    ),
    value_combined AS (
        SELECT
            *,
            (CASE WHEN z_bm IS NOT NULL THEN 1 ELSE 0 END
             + CASE WHEN z_ep IS NOT NULL THEN 1 ELSE 0 END
             + CASE WHEN z_cfop IS NOT NULL THEN 1 ELSE 0 END
             + CASE WHEN z_sp IS NOT NULL THEN 1 ELSE 0 END) AS value_component_count
        FROM zscored
    ),
    final AS (
        SELECT
            *,
            CASE WHEN value_component_count >= 2 THEN
                (COALESCE(z_bm, 0) + COALESCE(z_ep, 0) + COALESCE(z_cfop, 0) + COALESCE(z_sp, 0))
                / value_component_count
            END AS fin_value_z
        FROM value_combined
    )
    SELECT
        trade_date, ticker, market{industry_passthrough},
        fin_log_mcap,
        LAG(fin_log_mcap) OVER (PARTITION BY ticker, market ORDER BY trade_date)
            AS fin_log_mcap_lag1,
        fin_book_to_market, fin_earnings_yield, fin_cfo_yield, fin_sales_to_price,
        negative_equity, value_component_count, fs_basis_used,
        fin_value_z,
        LAG(fin_value_z) OVER (PARTITION BY ticker, market ORDER BY trade_date)
            AS fin_value_z_lag1,
        fin_gross_profitability,
        LAG(fin_gross_profitability) OVER (PARTITION BY ticker, market ORDER BY trade_date)
            AS fin_gross_profitability_lag1,
        gross_profit_source,
        fin_operating_profitability,
        LAG(fin_operating_profitability) OVER (PARTITION BY ticker, market ORDER BY trade_date)
            AS fin_operating_profitability_lag1,
        fin_asset_growth_yoy,
        LAG(fin_asset_growth_yoy) OVER (PARTITION BY ticker, market ORDER BY trade_date)
            AS fin_asset_growth_yoy_lag1,
        fin_accruals_to_assets,
        LAG(fin_accruals_to_assets) OVER (PARTITION BY ticker, market ORDER BY trade_date)
            AS fin_accruals_to_assets_lag1,
        value_available_from,
        (trade_date - value_available_from) AS value_fin_age_days,
        profitability_available_from,
        (trade_date - profitability_available_from) AS profitability_fin_age_days,
        asset_growth_available_from,
        (trade_date - asset_growth_available_from) AS asset_growth_fin_age_days,
        accruals_available_from,
        (trade_date - accruals_available_from) AS accruals_fin_age_days
    FROM final
    """


def check_projection_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """The requested non-key columns in the mart's output order; unsupported ones fail.

    Raises:
        ValueError: A column has no projection support, or nothing is requested.
    """
    wanted = {column for column in columns if column not in FIN_SCAN_KEY_COLUMNS}
    unsupported = sorted(wanted - set(PROJECTION_EXPRESSIONS))
    if unsupported:
        raise ValueError(
            f"feat_fin_scan_daily projection does not support {unsupported}; "
            f"supported: {list(PROJECTION_EXPRESSIONS)}. Build the full mart instead."
        )
    if not wanted:
        raise ValueError("feat_fin_scan_daily projection needs at least one column")
    return tuple(name for name in PROJECTION_EXPRESSIONS if name in wanted)


def build_fin_scan_projection_sql(
    columns: Sequence[str],
    *,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
) -> str:
    """The key columns plus ``columns``, with the full mart's definitions and none of its cost.

    Same grain and rows as :func:`build_fin_scan_daily_sql` (one per PIT row), the same
    ``panel`` source and ``base_ok``, and the same expression for each column. It reads
    neither the quarterly vintage mart nor any interval join.
    """
    names = check_projection_columns(columns)
    selects = ",\n        ".join(
        f"{PROJECTION_EXPRESSIONS[name]} AS {name}" for name in names
    )
    return f"""
    WITH panel AS (
        SELECT
            pit.trade_date, pit.ticker, pit.market,
            pit.market_cap_pit, pit.shares_is_available, pit.shares_invalid_flag,
            q.is_halted, q.valid_session_idx
        {_panel_source(pit_view, quality_view)}
    ),
    scored AS (
        SELECT
            *,
            {BASE_OK_SQL} AS base_ok
        FROM panel
    )
    SELECT
        trade_date, ticker, market,
        {selects}
    FROM scored
    """


def register_fin_scan_daily_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = FIN_SCAN_TABLE,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    vintage_view: str = "fin_quarterly_metric_vintage",
    industry_view: str | None = None,
) -> str:
    """Register a DuckDB view over the SQL above (no parquet — tests/parity)."""
    sql = build_fin_scan_daily_sql(
        pit_view=pit_view,
        quality_view=quality_view,
        vintage_view=vintage_view,
        industry_view=industry_view,
    )
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_fin_scan_daily_industry(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    vintage_view: str = "fin_quarterly_metric_vintage",
    corp_master_view: str = "dart_corp_master",
    force: bool = False,
) -> str:
    """Build the N2-9 industry-neutral diagnostic mart beside the plain one.

    Every winsorize percentile and z-score normalises within
    ``(trade_date, market, industry_group)`` rather than ``(trade_date,
    market)``, so the same features are recomputed against industry peers.
    Nothing else changes: same ratios, same fs_basis rule, same component
    count.

    It is a *diagnostic*, and the reason is not caution. ``induty_code`` is
    today's industry with no change history, so a company that switched
    business lines carries its present industry backwards through the whole
    sample. That is a look-ahead, which disqualifies the output as an alpha
    feature while leaving it perfectly good for asking whether a result
    survives industry neutralisation.

    Raises:
        RuntimeError: When the corp master carries no ``induty_code``.
    """
    from modeler.etl.industry import register_industry_group_view

    industry_view = register_industry_group_view(con, corp_master_view=corp_master_view)
    materialize(
        con,
        config,
        FIN_SCAN_INDUSTRY_TABLE,
        build_fin_scan_daily_sql(
            pit_view=pit_view,
            quality_view=quality_view,
            vintage_view=vintage_view,
            industry_view=industry_view,
        ),
        force=force,
    )
    return register_mart_view(con, config, FIN_SCAN_INDUSTRY_TABLE)


def materialize_fin_scan_daily(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    pit_view: str = "dim_stock_pit_daily",
    quality_view: str = "dim_price_quality_daily",
    vintage_view: str = "fin_quarterly_metric_vintage",
    force: bool = False,
    join_plan: str = JOIN_PLAN_OR,
    semantics: str = SEMANTICS_V1,
    columns: Sequence[str] | None = None,
) -> str:
    """Build + register ``feat_fin_scan_daily`` as a cached parquet mart.

    Requires ``pit_view``, ``quality_view`` (A0) and ``vintage_view`` (B-3)
    already registered on ``con``.

    ``join_plan="coalesce_join"`` writes the mart from the hash-joinable text
    (Mac, 2026-09-29 snapshot: about 23 s against about 50 min). The cache
    contract stays the default text, so ``sql_hash`` is unchanged; the metadata
    additionally carries ``plan`` and ``plan_hash`` and the mart is only reused
    by a caller that asks for the same plan.

    ``semantics="v2"`` is a different meaning (a fixed summation order for the
    z-scores), so the contract text, and with it ``sql_hash``, is the v2 text and
    the metadata also records ``semantics_version``. It combines with any join plan.

    ``columns`` builds the projection instead (:func:`build_fin_scan_projection_sql`): the
    keys plus those columns only, which makes it a different mart. Its contract text is the
    projection statement, so ``sql_hash`` and ``schema_hash`` describe what is on disk, and
    the metadata records ``plan="projection"`` and the ``projected_columns``. Neither a
    full mart is reused for it nor it for a full-mart caller. There is no join plan or
    z-score semantics to choose, so a non-default value is an error.
    """
    if columns is not None:
        if join_plan != JOIN_PLAN_OR or semantics != SEMANTICS_V1:
            raise ValueError("a feat_fin_scan_daily projection takes no join_plan or semantics")
        projected = check_projection_columns(columns)
        sql = build_fin_scan_projection_sql(
            projected, pit_view=pit_view, quality_view=quality_view)
        materialize(
            con, config, FIN_SCAN_TABLE, sql, force=force,
            plan=MartPlan(
                plan_id=PLAN_PROJECTION, final_sql=sql,
                metadata=(("projected_columns", list(projected)),)),
        )
        return register_mart_view(con, config, FIN_SCAN_TABLE)
    kwargs = dict(pit_view=pit_view, quality_view=quality_view, vintage_view=vintage_view)
    plan = None
    if join_plan != JOIN_PLAN_OR or semantics != SEMANTICS_V1:
        plan = MartPlan(
            plan_id=join_plan if join_plan != JOIN_PLAN_OR else "single",
            final_sql=build_fin_scan_daily_sql(**kwargs, join_plan=join_plan, semantics=semantics),
            semantics_version=semantics,
        )
    materialize(
        con,
        config,
        FIN_SCAN_TABLE,
        build_fin_scan_daily_sql(**kwargs, semantics=semantics),
        force=force,
        plan=plan,
    )
    return register_mart_view(con, config, FIN_SCAN_TABLE)
