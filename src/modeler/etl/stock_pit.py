"""Point-in-time issued/float shares and approximate market-cap mart."""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view

PIT_TABLE = "dim_stock_pit_daily"


def build_stock_pit_sql(
    *,
    price_view: str = "daily_ohlcv",
    share_view: str = "dart_share_count_raw",
) -> str:
    """Build a filing-date PIT join without backward filling pre-first filing.

    ``rcept_no`` is the disclosure date source. Rows with malformed/missing
    receipt numbers use the documented lag fallback only; they are marked so a
    smoke report can quantify the fallback path.
    """
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


def materialize_stock_pit(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    price_view: str = "daily_ohlcv",
    share_view: str = "dart_share_count_raw",
    force: bool = False,
) -> str:
    materialize(
        con,
        config,
        PIT_TABLE,
        build_stock_pit_sql(price_view=price_view, share_view=share_view),
        force=force,
    )
    return register_mart_view(con, config, PIT_TABLE)
