# ruff: noqa: E501
"""feat_flow — investor-flow / short-selling features (etl_00 §3.2).

Source: ``krx_security_flow_raw`` (the heaviest table, 76M rows). Pipeline:

  1. KRX-first dedup. KRX/PYKRX carry the same natural key with zero value
     conflicts (etl_00 §3.2). ``QUALIFY ROW_NUMBER() ... ORDER BY (KRX first)``
     keeps the KRX row. Result = 55,918,702 distinct rows (etl_01 §4.2, pinned
     as a regression guard in tests). NOTE: requires the view be read with
     ``hive_partitioning=false`` so the real ``source`` column survives
     (etl_01 §4.2) — the lake reader enforces this.
  2. Wide pivot the 7 metric_codes to one row per (trade_date, ticker, market).
  3. Derive cumulative / change / z-score features (``flow_`` prefix). Net-buy
     volumes are accumulated and z-scored (not used as raw share counts).

Short-vs-volume ratios (``short_selling_volume / daily volume``, etc.) need the
daily traded ``volume``/``turnover`` which live in ``daily_ohlcv``, not here, so
they are computed at panel assembly (P5) where price and flow are joined. This
builder passes through ``flow_short_selling_volume`` / ``flow_short_selling_value``
for that step and keeps only flow-internal derivations.

Coverage asymmetry: ``short_selling_balance_quantity`` starts 2016-06-30 (others
2007-06-05), so balance-derived features are NULL before then -> the preprocess
stage adds ``*_isna`` flags (etl_00 §3.2, L1). The 3-investor net-buy sum is NOT
an identity (excludes 기타법인) so we never derive a "closes-to-zero" feature
from it (etl_00 §3.2).

See ``etl_00`` §3.2 and ``etl_03_implementation_plan.md`` §4 (P3).
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import MartPlan, materialize, register_mart_view
from modeler.etl.quality import short_regime_sql

FLOW_TABLE = "feat_flow"

#: Default: KRX-first ``ROW_NUMBER`` dedup over every raw row, then a conditional-aggregation
#: pivot. This is the text every frozen mart was written under.
PIVOT_PLAN_WINDOW = "window_dedup"
#: Same rows from one ``GROUP BY`` with ``arg_min_null`` per metric code. It skips the
#: window sort of the ~78M-row dedup. ``arg_min_null`` (not ``arg_min``) is required: the
#: window version keeps a KRX row whose value is NULL, and ``arg_min`` would skip it and
#: fall through to the PYKRX value.
PIVOT_PLAN_ARGMIN = "argmin_pivot"
PIVOT_PLANS = (PIVOT_PLAN_WINDOW, PIVOT_PLAN_ARGMIN)

# The 7 metric_codes carried by krx_security_flow_raw (verified on the lake).
METRIC_CODES: tuple[str, ...] = (
    "foreign_net_buy_volume",
    "institution_net_buy_volume",
    "individual_net_buy_volume",
    "foreign_holding_shares",
    "short_selling_volume",
    "short_selling_value",
    "short_selling_balance_quantity",
)


def build_dedup_sql(flow_view: str = "krx_security_flow_raw") -> str:
    """SQL for KRX-first dedup of the raw flow view (etl_00 §3.2).

    Emits ``(trade_date, ticker, market, metric_code, value)`` with one row per
    natural key, KRX winning over PYKRX. ``value`` cast to DOUBLE (etl_01 §3).
    """
    return f"""
        SELECT trade_date, ticker, market, metric_code,
               CAST(value AS DOUBLE) AS value
        FROM {flow_view}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY trade_date, ticker, market, metric_code
            ORDER BY CASE source WHEN 'KRX' THEN 0 ELSE 1 END
        ) = 1
    """


def _pivot_expr() -> str:
    """Conditional-aggregation pivot of the 7 metric_codes into wide columns."""
    cols = []
    for code in METRIC_CODES:
        cols.append(f"MAX(CASE WHEN metric_code = '{code}' THEN value END) AS {code}")
    return ",\n            ".join(cols)


def _argmin_pivot_expr() -> str:
    """KRX-first value per metric code, picked inside the ``GROUP BY`` (no dedup pass)."""
    cols = []
    for code in METRIC_CODES:
        cols.append(
            "arg_min_null(CAST(value AS DOUBLE), CASE source WHEN 'KRX' THEN 0 ELSE 1 END) "
            f"FILTER (WHERE metric_code = '{code}') AS {code}"
        )
    return ",\n                ".join(cols)


def build_flow_sql(
    flow_view: str = "krx_security_flow_raw",
    *,
    price_view: str | None = None,
    pit_view: str | None = None,
    quality_view: str | None = None,
    pivot_plan: str = PIVOT_PLAN_WINDOW,
    short_balance_lag_sessions: int = 0,
) -> str:
    """SQL producing ``feat_flow`` (dedup -> wide pivot -> derived features).

    ``flow_view`` must already be registered on the connection (hive=false).

    ``pivot_plan`` only changes how the wide table is reached. ``window_dedup`` (the
    default) is the original text; ``argmin_pivot`` returns the same rows with a
    different text, so it is a :class:`~modeler.etl.mart.MartPlan`, never the cache
    contract.

    ``short_balance_lag_sessions`` (KR 10월 계획 4.5) pushes the two baseline balance
    columns, ``flow_short_balance_qty`` and ``flow_short_balance_chg_20d``, back by that
    many sessions: row t then carries the balance *measured* at session t-n, which is
    what was public by the close of t when the publication lag is n sessions. ``0`` (the
    default) emits the text every frozen mart was written under, byte for byte. The other
    balance columns (``flow_short_interest_ratio`` ...) are not touched — they have
    their own availability gate. The shift counts market sessions, not rows: a ticker
    with a missing flow row gets NULL rather than a value from the wrong day.
    """
    if pivot_plan not in PIVOT_PLANS:
        raise ValueError(f"unknown feat_flow pivot plan {pivot_plan!r}; expected {PIVOT_PLANS}")
    if short_balance_lag_sessions < 0:
        raise ValueError(
            f"short_balance_lag_sessions must be >= 0, got {short_balance_lag_sessions}"
        )
    dedup = build_dedup_sql(flow_view)
    if price_view is None:
        # Degraded path (no price/quality data available): keep every flow
        # row and every downstream window ordered by raw trade_date, exactly
        # as before — legacy consumers (research/models/_01_20_access_return_rank)
        # call materialize_flow with no views at all and must not change.
        session_cte = """
        sessioned AS (
            SELECT wide.*, NULL::DOUBLE AS total_volume, NULL::DOUBLE AS float_shares_pit,
                   NULL::BOOLEAN AS is_halted, NULL::BOOLEAN AS short_balance_is_available,
                   'unknown'::VARCHAR AS short_regime, NULL::BIGINT AS valid_session_idx
            FROM wide
        )"""
    else:
        pit_join = (
            f"LEFT JOIN {pit_view} s USING (trade_date, ticker, market)"
            if pit_view
            else ""
        )
        q_join = (
            f"LEFT JOIN {quality_view} q USING (trade_date, ticker, market)"
            if quality_view
            else ""
        )
        balance_available = (
            "COALESCE(q.short_balance_is_available, trade_date >= DATE '2016-06-30')"
            if quality_view
            else "trade_date >= DATE '2016-06-30'"
        )
        regime = "COALESCE(q.short_regime, 'unknown')" if quality_view else short_regime_sql()
        # Only quality_view can tell us which rows are real (non-halt) trading
        # sessions (dim_price_quality_daily.valid_session_idx is NULL on halt
        # days, per research/etl/trading_panel.py). Filtering to those rows
        # *before* any window runs is what price.py already does for price
        # features (build_valid_session_sql filters halts out first, then
        # ORDER BY trade_date is session-safe) — flow's windows must match,
        # or a halt-day row silently consumes one of the N rolling slots and
        # both the native ratio and its _lag1 stop meaning "N valid sessions."
        session_filter = "WHERE q.valid_session_idx IS NOT NULL" if quality_view else ""
        valid_session_idx = "q.valid_session_idx" if quality_view else "NULL::BIGINT"
        session_cte = f"""
        sessioned AS (
            SELECT wide.*, CAST(p.volume AS DOUBLE) AS total_volume,
                   s.float_shares_pit,
                   (p.open = 0 AND p.high = 0 AND p.low = 0) AS is_halted,
                   {balance_available} AS short_balance_is_available,
                   {regime} AS short_regime,
                   {valid_session_idx} AS valid_session_idx
            FROM wide
            LEFT JOIN {price_view} p USING (trade_date, ticker, market)
            {pit_join}
            {q_join}
            {session_filter}
        )"""
    ratio_columns = (
        "flow_foreign_netbuy_to_volume_5d, flow_foreign_netbuy_to_volume_20d, flow_foreign_netbuy_to_volume_60d, "
        "flow_inst_netbuy_to_volume_5d, flow_inst_netbuy_to_volume_20d, flow_inst_netbuy_to_volume_60d, "
        "flow_individual_netbuy_to_volume_5d, flow_individual_netbuy_to_volume_20d, flow_individual_netbuy_to_volume_60d, "
        "flow_foreign_holding_ratio_chg_5d, flow_foreign_holding_ratio_chg_20d, flow_foreign_holding_ratio_chg_60d, "
        "flow_short_turnover_20d, flow_short_interest_ratio, flow_short_interest_ratio_chg_20d, "
        "flow_days_to_cover, flow_nat_proxy_20d"
    )
    lag_columns = ", ".join(
        f"LAG({c}) OVER (PARTITION BY ticker, market ORDER BY trade_date) AS {c}_lag1"
        for c in ratio_columns.replace(" ", "").split(",")
    )
    if pivot_plan == PIVOT_PLAN_ARGMIN:
        wide_ctes = f"""wide AS (
            -- one hash aggregation: the KRX-first value per (key, metric_code), no window sort
            SELECT
                trade_date, ticker, market,
                {_argmin_pivot_expr()}
            FROM {flow_view}
            GROUP BY trade_date, ticker, market
        )"""
    else:
        wide_ctes = f"""dedup AS (
            {dedup}
        ),
        wide AS (
            SELECT
                trade_date, ticker, market,
                {_pivot_expr()}
            FROM dedup
            GROUP BY trade_date, ticker, market
        )"""
    if short_balance_lag_sessions:
        n = short_balance_lag_sessions
        # Market-session number of each row (the flow table's own calendar). The frame
        # [n PRECEDING, n PRECEDING] over that number is the row exactly n sessions back,
        # or empty (NULL) when the ticker has no flow row on that session.
        balance_lag_cte = f""",
        balance_lag AS (
            SELECT variants.*,
                DENSE_RANK() OVER (ORDER BY trade_date) AS balance_sess_no
            FROM variants
        ),
        balance_lagged AS (
            SELECT balance_lag.* REPLACE (
                MAX(flow_short_balance_qty) OVER wl AS flow_short_balance_qty,
                MAX(flow_short_balance_chg_20d) OVER wl AS flow_short_balance_chg_20d
            )
            FROM balance_lag
            WINDOW wl AS (
                PARTITION BY ticker, market ORDER BY balance_sess_no
                RANGE BETWEEN {n} PRECEDING AND {n} PRECEDING
            )
        )"""
        final_source = "balance_lagged"
    else:
        balance_lag_cte = ""
        final_source = "variants"
    return f"""
        WITH {wide_ctes},
        {session_cte},
        flow_base AS (
            SELECT sessioned.*,
                SUM(foreign_net_buy_volume) OVER w5 AS flow_foreign_netbuy_sum_5d,
                SUM(foreign_net_buy_volume) OVER w20 AS flow_foreign_netbuy_sum_20d,
                SUM(institution_net_buy_volume) OVER w5 AS flow_inst_netbuy_sum_5d,
                SUM(institution_net_buy_volume) OVER w20 AS flow_inst_netbuy_sum_20d,
                SUM(individual_net_buy_volume) OVER w5 AS flow_indiv_netbuy_sum_5d,
                SUM(individual_net_buy_volume) OVER w20 AS flow_indiv_netbuy_sum_20d,
                foreign_holding_shares - LAG(foreign_holding_shares, 5) OVER w
                    AS flow_foreign_holding_chg_5d,
                foreign_holding_shares - LAG(foreign_holding_shares, 20) OVER w
                    AS flow_foreign_holding_chg_20d,
                short_selling_balance_quantity - LAG(short_selling_balance_quantity, 20) OVER w
                    AS flow_short_balance_chg_20d,
                (foreign_net_buy_volume - AVG(foreign_net_buy_volume) OVER w20)
                    / NULLIF(STDDEV_SAMP(foreign_net_buy_volume) OVER w20, 0)
                    AS flow_foreign_netbuy_z_20d,
                (institution_net_buy_volume - AVG(institution_net_buy_volume) OVER w20)
                    / NULLIF(STDDEV_SAMP(institution_net_buy_volume) OVER w20, 0)
                    AS flow_inst_netbuy_z_20d,
                short_selling_value / NULLIF(short_selling_volume, 0) AS flow_short_avg_price,
                short_selling_volume AS flow_short_selling_volume,
                short_selling_value AS flow_short_selling_value,
                short_selling_balance_quantity AS flow_short_balance_qty
            FROM sessioned
            WINDOW
                w AS (PARTITION BY ticker, market ORDER BY trade_date),
                w5 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
                w20 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
                w60 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
        ),
        ratios AS (
            SELECT flow_base.*,
                CASE WHEN COUNT(foreign_net_buy_volume) OVER w5 = 5
                          AND COUNT(total_volume) OVER w5 = 5
                     THEN SUM(foreign_net_buy_volume) OVER w5 / NULLIF(SUM(total_volume) OVER w5, 0) END
                    AS flow_foreign_netbuy_to_volume_5d,
                CASE WHEN COUNT(foreign_net_buy_volume) OVER w20 = 20
                          AND COUNT(total_volume) OVER w20 = 20
                     THEN SUM(foreign_net_buy_volume) OVER w20 / NULLIF(SUM(total_volume) OVER w20, 0) END
                    AS flow_foreign_netbuy_to_volume_20d,
                CASE WHEN COUNT(foreign_net_buy_volume) OVER w60 = 60
                          AND COUNT(total_volume) OVER w60 = 60
                     THEN SUM(foreign_net_buy_volume) OVER w60 / NULLIF(SUM(total_volume) OVER w60, 0) END
                    AS flow_foreign_netbuy_to_volume_60d,
                CASE WHEN COUNT(institution_net_buy_volume) OVER w5 = 5
                          AND COUNT(total_volume) OVER w5 = 5
                     THEN SUM(institution_net_buy_volume) OVER w5 / NULLIF(SUM(total_volume) OVER w5, 0) END
                    AS flow_inst_netbuy_to_volume_5d,
                CASE WHEN COUNT(institution_net_buy_volume) OVER w20 = 20
                          AND COUNT(total_volume) OVER w20 = 20
                     THEN SUM(institution_net_buy_volume) OVER w20 / NULLIF(SUM(total_volume) OVER w20, 0) END
                    AS flow_inst_netbuy_to_volume_20d,
                CASE WHEN COUNT(institution_net_buy_volume) OVER w60 = 60
                          AND COUNT(total_volume) OVER w60 = 60
                     THEN SUM(institution_net_buy_volume) OVER w60 / NULLIF(SUM(total_volume) OVER w60, 0) END
                    AS flow_inst_netbuy_to_volume_60d,
                CASE WHEN COUNT(individual_net_buy_volume) OVER w5 = 5
                          AND COUNT(total_volume) OVER w5 = 5
                     THEN SUM(individual_net_buy_volume) OVER w5 / NULLIF(SUM(total_volume) OVER w5, 0) END
                    AS flow_individual_netbuy_to_volume_5d,
                CASE WHEN COUNT(individual_net_buy_volume) OVER w20 = 20
                          AND COUNT(total_volume) OVER w20 = 20
                     THEN SUM(individual_net_buy_volume) OVER w20 / NULLIF(SUM(total_volume) OVER w20, 0) END
                    AS flow_individual_netbuy_to_volume_20d,
                CASE WHEN COUNT(individual_net_buy_volume) OVER w60 = 60
                          AND COUNT(total_volume) OVER w60 = 60
                     THEN SUM(individual_net_buy_volume) OVER w60 / NULLIF(SUM(total_volume) OVER w60, 0) END
                    AS flow_individual_netbuy_to_volume_60d,
                foreign_holding_shares / NULLIF(float_shares_pit, 0)
                    AS flow_foreign_holding_ratio,
                foreign_holding_shares / NULLIF(float_shares_pit, 0)
                    - LAG(foreign_holding_shares / NULLIF(float_shares_pit, 0), 5) OVER w
                    AS flow_foreign_holding_ratio_chg_5d,
                foreign_holding_shares / NULLIF(float_shares_pit, 0)
                    - LAG(foreign_holding_shares / NULLIF(float_shares_pit, 0), 20) OVER w
                    AS flow_foreign_holding_ratio_chg_20d,
                foreign_holding_shares / NULLIF(float_shares_pit, 0)
                    - LAG(foreign_holding_shares / NULLIF(float_shares_pit, 0), 60) OVER w
                    AS flow_foreign_holding_ratio_chg_60d,
                CASE WHEN short_regime = 'allowed'
                          AND COUNT(total_volume) OVER w20 = 20
                          AND COUNT(short_selling_volume) OVER w20 = 20
                     THEN SUM(short_selling_volume) OVER w20 / NULLIF(SUM(total_volume) OVER w20, 0) END
                    AS flow_short_turnover_20d,
                CASE WHEN short_regime = 'allowed' AND short_balance_is_available
                     THEN short_selling_balance_quantity / NULLIF(float_shares_pit, 0) END
                    AS flow_short_interest_ratio,
                CASE WHEN short_regime = 'allowed' AND short_balance_is_available
                     THEN short_selling_balance_quantity / NULLIF(float_shares_pit, 0)
                          - LAG(short_selling_balance_quantity / NULLIF(float_shares_pit, 0), 20) OVER w END
                    AS flow_short_interest_ratio_chg_20d,
                CASE WHEN short_regime = 'allowed' AND short_balance_is_available
                     THEN short_selling_balance_quantity / NULLIF(AVG(total_volume) OVER w20, 0) END
                    AS flow_days_to_cover,
                CASE WHEN short_regime = 'allowed'
                          AND short_balance_is_available
                          AND LAG(short_balance_is_available, 20) OVER w
                     THEN foreign_holding_shares / NULLIF(float_shares_pit, 0)
                          - LAG(foreign_holding_shares / NULLIF(float_shares_pit, 0), 20) OVER w
                END AS nat_ahf_20,
                CASE WHEN short_regime = 'allowed' AND short_balance_is_available
                     THEN short_selling_balance_quantity / NULLIF(float_shares_pit, 0)
                END AS nat_asi_20
            FROM flow_base
            WINDOW
                w AS (PARTITION BY ticker, market ORDER BY trade_date),
                w5 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
                w20 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW),
                w60 AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
        ),
        nat_ranked AS (
            SELECT trade_date, ticker, market,
                PERCENT_RANK() OVER (
                    PARTITION BY trade_date, market ORDER BY nat_ahf_20
                ) - PERCENT_RANK() OVER (
                    PARTITION BY trade_date, market ORDER BY nat_asi_20
                ) AS flow_nat_proxy_20d
            FROM ratios
            WHERE nat_ahf_20 IS NOT NULL AND nat_asi_20 IS NOT NULL
        ),
        ranked AS (
            SELECT ratios.*, n.flow_nat_proxy_20d
            FROM ratios
            LEFT JOIN nat_ranked n USING (trade_date, ticker, market)
        ),
        variants AS (
            SELECT ranked.*, {lag_columns}
            FROM ranked
        ){balance_lag_cte}
        SELECT
            trade_date, ticker, market,
            flow_foreign_netbuy_sum_5d, flow_foreign_netbuy_sum_20d,
            flow_inst_netbuy_sum_5d, flow_inst_netbuy_sum_20d,
            flow_indiv_netbuy_sum_5d, flow_indiv_netbuy_sum_20d,
            flow_foreign_holding_chg_5d, flow_foreign_holding_chg_20d,
            flow_short_balance_chg_20d, flow_foreign_netbuy_z_20d,
            flow_inst_netbuy_z_20d, flow_short_avg_price,
            flow_short_selling_volume, flow_short_selling_value, flow_short_balance_qty,
            {ratio_columns}, {lag_columns},
            short_balance_is_available, short_regime
        FROM {final_source}
    """


def materialize_flow(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    flow_view: str = "krx_security_flow_raw",
    price_view: str | None = None,
    pit_view: str | None = None,
    quality_view: str | None = None,
    force: bool = False,
    pivot_plan: str = PIVOT_PLAN_WINDOW,
    short_balance_lag_sessions: int = 0,
) -> str:
    """Build + register ``feat_flow`` mart view. Returns the view name.

    Requires ``flow_view`` registered on ``con`` (hive=false — the lake reader
    enforces this so the KRX-first dedup is not neutralized, etl_01 §4.2).

    ``pivot_plan="argmin_pivot"`` writes the mart from the single-aggregation text. The
    cache contract stays the default text (``sql_hash`` unchanged); the metadata also
    carries ``plan`` / ``plan_hash`` and the mart is only reused by a caller asking for
    the same plan.

    ``short_balance_lag_sessions > 0`` writes a different mart under the same name, so it
    belongs in a lake root of its own (see ``experiments/flow_lag_v2``), never the shared
    one — ``force`` is refused with it so a frozen ``feat_flow`` cannot be overwritten.
    """
    if short_balance_lag_sessions and force:
        raise ValueError(
            "force=True with short_balance_lag_sessions would rebuild feat_flow in place; "
            "use a separate lake root (experiments/flow_lag_v2) and no force"
        )
    kwargs = dict(
        price_view=price_view, pit_view=pit_view, quality_view=quality_view,
        short_balance_lag_sessions=short_balance_lag_sessions,
    )
    plan = None
    if pivot_plan != PIVOT_PLAN_WINDOW:
        plan = MartPlan(
            plan_id=pivot_plan,
            final_sql=build_flow_sql(flow_view, **kwargs, pivot_plan=pivot_plan),
        )
    materialize(
        con,
        config,
        FLOW_TABLE,
        build_flow_sql(flow_view, **kwargs),
        force=force,
        plan=plan,
    )
    return register_mart_view(con, config, FLOW_TABLE)
