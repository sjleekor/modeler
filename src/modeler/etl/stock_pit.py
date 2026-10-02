"""Point-in-time issued/float shares and approximate market-cap mart."""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import MartPlan, materialize, register_mart_view

PIT_TABLE = "dim_stock_pit_daily"

#: Default: every price row joined to every filing available by then, ``ROW_NUMBER`` keeps
#: the best. This is the text every frozen mart was written under.
PLAN_SINGLE = "single"
#: Same rows: the winning filing per ticker becomes a validity interval, and the prices
#: are matched with an ``ASOF`` join. The first-session lookup uses a distinct-date table.
PLAN_ASOF_INTERVALS = "asof_intervals"
PLANS = (PLAN_SINGLE, PLAN_ASOF_INTERVALS)


def build_stock_pit_sql(
    *,
    price_view: str = "daily_ohlcv",
    share_view: str = "dart_share_count_raw",
    plan: str = PLAN_SINGLE,
) -> str:
    """Build a filing-date PIT join without backward filling pre-first filing.

    ``rcept_no`` is the disclosure date source. Rows with malformed/missing
    receipt numbers use the documented lag fallback only; they are marked so a
    smoke report can quantify the fallback path.

    ``plan`` picks how the same rows are reached (see :data:`PLAN_ASOF_INTERVALS`). The
    default text is unchanged. The one place the plans may differ is a tie on the whole
    ordering key ``(stlm_dt, disclosed_date, rcept_no)``: the default picks arbitrarily,
    ``asof_intervals`` breaks it by ``available_from, issued_raw, treasury_raw, float_raw``
    so the result no longer depends on thread scheduling.
    """
    if plan not in PLANS:
        raise ValueError(f"unknown dim_stock_pit_daily plan {plan!r}; expected {PLANS}")
    if plan == PLAN_ASOF_INTERVALS:
        return _build_asof_intervals_sql(price_view, share_view)
    return f"""
        WITH prices AS (
            SELECT
                trade_date, ticker, market,
                CAST(close AS DOUBLE) AS close_d
            FROM {price_view}
        ),
        raw AS (
            SELECT
                ticker, bsns_year, reprt_code, rcept_no, se,
                TRY_CAST(istc_totqy AS DOUBLE) AS issued_raw,
                TRY_CAST(tesstk_co AS DOUBLE) AS treasury_raw,
                TRY_CAST(distb_stock_co AS DOUBLE) AS float_raw,
                stlm_dt,
                TRY_STRPTIME(NULLIF(SUBSTR(CAST(rcept_no AS VARCHAR), 1, 8), ''), '%Y%m%d')::DATE
                    AS disclosed_date
            FROM {share_view}
            WHERE se = '합계'
        ),
        filings AS (
            SELECT
                r.*,
                CASE WHEN r.disclosed_date IS NOT NULL THEN
                    (SELECT MIN(p.trade_date) FROM prices p
                     WHERE p.trade_date > r.disclosed_date)
                ELSE COALESCE(
                    (SELECT MIN(p.trade_date) FROM prices p
                     WHERE p.trade_date >= r.stlm_dt + CASE
                         WHEN r.reprt_code = '11011' THEN INTERVAL '90 days'
                         ELSE INTERVAL '45 days' END),
                    (SELECT MIN(p.trade_date) FROM prices p
                     WHERE p.trade_date >= r.stlm_dt)
                ) END AS available_from,
                (r.disclosed_date IS NULL) AS used_fallback_lag
            FROM raw r
        ),
        candidates AS (
            SELECT
                p.trade_date, p.ticker, p.market, p.close_d,
                f.issued_raw, f.treasury_raw, f.float_raw,
                f.available_from, f.disclosed_date, f.rcept_no,
                f.used_fallback_lag,
                ROW_NUMBER() OVER (
                    PARTITION BY p.trade_date, p.ticker, p.market
                    ORDER BY f.stlm_dt DESC NULLS LAST,
                             f.disclosed_date DESC NULLS LAST,
                             f.rcept_no DESC
                ) AS filing_rank
            FROM prices p
            LEFT JOIN filings f
              ON f.ticker = p.ticker
             AND f.available_from <= p.trade_date
        ),
        selected AS (
            SELECT * FROM candidates WHERE filing_rank = 1
        )
        SELECT
            trade_date, ticker, market,
            CASE WHEN issued_raw > 0 THEN issued_raw END AS issued_shares_pit,
            CASE WHEN treasury_raw >= 0 THEN treasury_raw END AS treasury_shares_pit,
            CASE
                WHEN float_raw > 0 AND issued_raw > 0 AND float_raw <= issued_raw
                    THEN float_raw
                WHEN float_raw IS NULL AND issued_raw > 0 AND treasury_raw IS NOT NULL
                     AND treasury_raw >= 0 AND issued_raw - treasury_raw > 0
                    THEN issued_raw - treasury_raw
            END AS float_shares_pit,
            CASE WHEN issued_raw > 0 THEN close_d * issued_raw END AS market_cap_pit,
            available_from AS shares_available_from,
            CASE WHEN available_from IS NOT NULL THEN trade_date - available_from END
                AS shares_age_days,
            CASE WHEN available_from IS NOT NULL THEN 'dart_share_count_raw' END AS shares_source,
            (available_from IS NOT NULL AND issued_raw > 0) AS shares_is_available,
            (issued_raw IS NOT NULL AND issued_raw <= 0)
                OR (float_raw IS NOT NULL AND float_raw <= 0)
                OR (float_raw IS NOT NULL AND issued_raw IS NOT NULL AND float_raw > issued_raw)
                AS shares_invalid_flag,
            (treasury_raw IS NULL AND issued_raw IS NOT NULL) AS treasury_missing_flag,
            (float_raw IS NULL AND issued_raw IS NOT NULL AND treasury_raw IS NOT NULL)
                AS float_fallback_used,
            (used_fallback_lag IS TRUE) AS shares_used_fallback_lag
        FROM selected
    """


def _build_asof_intervals_sql(price_view: str, share_view: str) -> str:
    return f"""
        WITH prices AS (
            SELECT
                trade_date, ticker, market,
                CAST(close AS DOUBLE) AS close_d
            FROM {price_view}
        ),
        days AS (SELECT DISTINCT trade_date FROM prices),
        raw AS (
            SELECT
                ticker, bsns_year, reprt_code, rcept_no, se,
                TRY_CAST(istc_totqy AS DOUBLE) AS issued_raw,
                TRY_CAST(tesstk_co AS DOUBLE) AS treasury_raw,
                TRY_CAST(distb_stock_co AS DOUBLE) AS float_raw,
                stlm_dt,
                TRY_STRPTIME(NULLIF(SUBSTR(CAST(rcept_no AS VARCHAR), 1, 8), ''), '%Y%m%d')::DATE
                    AS disclosed_date
            FROM {share_view}
            WHERE se = '합계'
        ),
        filings AS (
            SELECT
                r.*,
                CASE WHEN r.disclosed_date IS NOT NULL THEN
                    (SELECT MIN(p.trade_date) FROM days p
                     WHERE p.trade_date > r.disclosed_date)
                ELSE COALESCE(
                    (SELECT MIN(p.trade_date) FROM days p
                     WHERE p.trade_date >= r.stlm_dt + CASE
                         WHEN r.reprt_code = '11011' THEN INTERVAL '90 days'
                         ELSE INTERVAL '45 days' END),
                    (SELECT MIN(p.trade_date) FROM days p
                     WHERE p.trade_date >= r.stlm_dt)
                ) END AS available_from,
                (r.disclosed_date IS NULL) AS used_fallback_lag
            FROM raw r
        ),
        ranked AS (
            -- the default's ordering; the trailing columns only break exact-key ties
            SELECT f.*,
                ROW_NUMBER() OVER (
                    PARTITION BY f.ticker
                    ORDER BY f.stlm_dt DESC NULLS LAST,
                             f.disclosed_date DESC NULLS LAST,
                             f.rcept_no DESC,
                             f.available_from, f.issued_raw, f.treasury_raw, f.float_raw
                ) AS prio_rank
            FROM filings f
            WHERE f.available_from IS NOT NULL
        ),
        ended AS (
            -- a filing wins from its available_from until the first availability of any
            -- filing that outranks it (smaller prio_rank); NULL = never displaced
            SELECT r.*,
                MIN(r.available_from) OVER (
                    PARTITION BY r.ticker ORDER BY r.prio_rank
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS displaced_from
            FROM ranked r
        ),
        winner_intervals AS (
            SELECT * FROM ended
            WHERE displaced_from IS NULL OR available_from < displaced_from
        ),
        selected AS (
            SELECT
                p.trade_date, p.ticker, p.market, p.close_d,
                w.issued_raw, w.treasury_raw, w.float_raw,
                w.available_from, w.disclosed_date, w.rcept_no,
                w.used_fallback_lag
            FROM prices p
            ASOF LEFT JOIN winner_intervals w
              ON w.ticker = p.ticker
             AND p.trade_date >= w.available_from
        )
        SELECT
            trade_date, ticker, market,
            CASE WHEN issued_raw > 0 THEN issued_raw END AS issued_shares_pit,
            CASE WHEN treasury_raw >= 0 THEN treasury_raw END AS treasury_shares_pit,
            CASE
                WHEN float_raw > 0 AND issued_raw > 0 AND float_raw <= issued_raw
                    THEN float_raw
                WHEN float_raw IS NULL AND issued_raw > 0 AND treasury_raw IS NOT NULL
                     AND treasury_raw >= 0 AND issued_raw - treasury_raw > 0
                    THEN issued_raw - treasury_raw
            END AS float_shares_pit,
            CASE WHEN issued_raw > 0 THEN close_d * issued_raw END AS market_cap_pit,
            available_from AS shares_available_from,
            CASE WHEN available_from IS NOT NULL THEN trade_date - available_from END
                AS shares_age_days,
            CASE WHEN available_from IS NOT NULL THEN 'dart_share_count_raw' END AS shares_source,
            (available_from IS NOT NULL AND issued_raw > 0) AS shares_is_available,
            (issued_raw IS NOT NULL AND issued_raw <= 0)
                OR (float_raw IS NOT NULL AND float_raw <= 0)
                OR (float_raw IS NOT NULL AND issued_raw IS NOT NULL AND float_raw > issued_raw)
                AS shares_invalid_flag,
            (treasury_raw IS NULL AND issued_raw IS NOT NULL) AS treasury_missing_flag,
            (float_raw IS NULL AND issued_raw IS NOT NULL AND treasury_raw IS NOT NULL)
                AS float_fallback_used,
            (used_fallback_lag IS TRUE) AS shares_used_fallback_lag
        FROM selected
    """


def materialize_stock_pit(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    price_view: str = "daily_ohlcv",
    share_view: str = "dart_share_count_raw",
    force: bool = False,
    plan: str = PLAN_SINGLE,
) -> str:
    """Build + register ``dim_stock_pit_daily``.

    ``plan="asof_intervals"`` writes the same rows from the interval text (Mac, 2026-09-29
    snapshot: 1.6 s against 87.7 s and 9 GB of spill). The cache contract stays the default
    text, so ``sql_hash`` is unchanged; the metadata also carries ``plan`` / ``plan_hash``.
    """
    mart_plan = None
    if plan != PLAN_SINGLE:
        mart_plan = MartPlan(
            plan_id=plan,
            final_sql=build_stock_pit_sql(price_view=price_view, share_view=share_view, plan=plan),
        )
    materialize(
        con,
        config,
        PIT_TABLE,
        build_stock_pit_sql(price_view=price_view, share_view=share_view),
        force=force,
        plan=mart_plan,
    )
    return register_mart_view(con, config, PIT_TABLE)
