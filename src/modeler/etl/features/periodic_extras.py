"""Daily PIT features from OpenDART periodic employee/governance responses (N6).

The DS002 endpoints expose final-vintage values.  Every feature therefore starts
only after the retained receipt becomes public; annual values are never backdated
to the business-year end.  Growth/change values use the later of the current and
previous receipt, and productivity also waits for the annual revenue vintage.
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view

PERIODIC_EXTRAS_TABLE = "feat_periodic_extras"
PERIODIC_EXTRAS_FORMULA_VERSION = "periodic_extras_v2"
FINAL_VINTAGE_CAPTURE_RATIO = 0.0184


def build_periodic_extras_sql(
    universe_view: str = "dim_universe_broad_daily",
    employee_view: str = "dart_employee_raw",
    governance_view: str = "dart_governance_raw",
    filing_receipt_view: str = "dart_filing_receipt_raw",
    vintage_view: str = "stock_metric_vintage_fact",
) -> str:
    """Return the N6 daily PIT feature query at ticker-session grain."""
    return f"""
        WITH trading_days AS (
            SELECT DISTINCT trade_date FROM {universe_view}
        ),
        receipt_dates AS (
            SELECT rcept_no, max(rcept_dt) AS rcept_dt
            FROM {filing_receipt_view}
            WHERE rcept_no IS NOT NULL AND rcept_dt IS NOT NULL
            GROUP BY rcept_no
        ),
        next_receipt_session AS (
            SELECT r.rcept_no, min(t.trade_date) AS available_from
            FROM receipt_dates r
            JOIN trading_days t ON t.trade_date > r.rcept_dt
            GROUP BY r.rcept_no
        ),
        employee_latest AS (
            SELECT corp_code, bsns_year, max(rcept_no) AS rcept_no
            FROM {employee_view}
            WHERE statement_type = 'employee'
            GROUP BY corp_code, bsns_year
        ),
        employee_parsed AS (
            SELECT
                e.corp_code, e.ticker, e.bsns_year, e.rcept_no,
                trim(json_extract_string(e.raw_payload, '$.fo_bbm')) AS division,
                try_cast(regexp_replace(
                    json_extract_string(e.raw_payload, '$.sm'), '[^0-9.-]', '', 'g'
                ) AS DOUBLE) AS headcount
            FROM {employee_view} e
            JOIN employee_latest l USING (corp_code, bsns_year, rcept_no)
            WHERE e.statement_type = 'employee'
        ),
        employee_flagged AS (
            SELECT *,
                division LIKE '%합계%' AS is_summary,
                max(CASE WHEN division LIKE '%합계%' AND headcount IS NOT NULL
                         THEN 1 ELSE 0 END)
                    OVER (PARTITION BY corp_code, bsns_year, rcept_no) AS has_summary
            FROM employee_parsed
        ),
        employee_annual AS (
            SELECT
                corp_code, any_value(ticker) AS ticker, bsns_year, any_value(rcept_no) AS rcept_no,
                sum(headcount) FILTER (
                    WHERE headcount IS NOT NULL AND (has_summary = 0 OR is_summary)
                ) AS headcount,
                any_value(n.available_from) AS employee_available_from
            FROM employee_flagged e
            LEFT JOIN next_receipt_session n USING (rcept_no)
            GROUP BY corp_code, bsns_year
        ),
        structural_change_year AS (
            SELECT DISTINCT corp_code, year(rcept_dt)::INTEGER AS bsns_year
            FROM {filing_receipt_view}
            WHERE regexp_matches(report_nm, '합병등종료보고서\\((분할|합병)\\)')
        ),
        employee_lagged AS (
            SELECT *,
                lag(bsns_year) OVER (PARTITION BY corp_code ORDER BY bsns_year) AS previous_year,
                lag(headcount) OVER (PARTITION BY corp_code ORDER BY bsns_year)
                    AS previous_headcount,
                lag(employee_available_from) OVER (PARTITION BY corp_code ORDER BY bsns_year)
                    AS previous_available_from
            FROM employee_annual
        ),
        employee_growth AS (
            SELECT
                e.ticker, e.bsns_year,
                greatest(e.employee_available_from, e.previous_available_from) AS available_from,
                CASE
                    WHEN e.previous_year = e.bsns_year - 1
                     AND e.headcount >= 1 AND e.previous_headcount >= 1
                     AND NOT (
                         s.corp_code IS NOT NULL
                         AND abs(e.headcount / nullif(e.previous_headcount, 0) - 1) >= 0.30
                     )
                    THEN e.headcount / nullif(e.previous_headcount, 0) - 1
                END AS hc_employee_growth_yoy
            FROM employee_lagged e
            LEFT JOIN structural_change_year s USING (corp_code, bsns_year)
        ),
        annual_revenue AS (
            SELECT
                ticker, bsns_year, value_numeric::DOUBLE AS revenue,
                available_from AS revenue_available_from
            FROM {vintage_view}
            WHERE metric_code = 'revenue' AND reprt_code = '11011' AND value_numeric > 0
            QUALIFY row_number() OVER (
                PARTITION BY ticker, bsns_year
                ORDER BY CASE WHEN fs_basis = 'CFS' THEN 0 ELSE 1 END,
                         available_from DESC NULLS LAST, rcept_no DESC
            ) = 1
        ),
        productivity AS (
            SELECT
                e.ticker, e.bsns_year,
                greatest(e.employee_available_from, r.revenue_available_from) AS available_from,
                ln(r.revenue / e.headcount) AS hc_revenue_per_employee
            FROM employee_annual e
            JOIN annual_revenue r USING (ticker, bsns_year)
            WHERE e.headcount > 0
              AND e.employee_available_from IS NOT NULL
              AND r.revenue_available_from IS NOT NULL
        ),
        major_latest AS (
            SELECT corp_code, bsns_year, max(rcept_no) AS rcept_no
            FROM {governance_view}
            WHERE statement_type = 'major_shareholder'
            GROUP BY corp_code, bsns_year
        ),
        major_rows AS (
            SELECT
                g.corp_code, g.ticker, g.bsns_year, g.rcept_no,
                try_cast(regexp_replace(
                    json_extract_string(g.raw_payload, '$.trmend_posesn_stock_qota_rt'),
                    '[^0-9.-]', '', 'g'
                ) AS DOUBLE) AS stake,
                CASE
                    WHEN trim(json_extract_string(g.raw_payload, '$.nm')) = '계'
                     AND trim(json_extract_string(g.raw_payload, '$.stock_knd')) = '합계' THEN 1
                    WHEN trim(json_extract_string(g.raw_payload, '$.nm')) = '계'
                     AND (json_extract_string(g.raw_payload, '$.stock_knd') LIKE '%보통%'
                       OR json_extract_string(g.raw_payload, '$.stock_knd') LIKE '%의결권%') THEN 2
                    WHEN trim(json_extract_string(g.raw_payload, '$.nm')) = '계' THEN 3
                    ELSE 4
                END AS selection_priority
            FROM {governance_view} g
            JOIN major_latest l USING (corp_code, bsns_year, rcept_no)
            WHERE g.statement_type = 'major_shareholder'
        ),
        major_ranked AS (
            SELECT *, min(selection_priority) OVER (
                PARTITION BY corp_code, bsns_year
            ) AS best_priority
            FROM major_rows
            WHERE stake BETWEEN 0 AND 100
        ),
        major_annual AS (
            SELECT
                corp_code, any_value(ticker) AS ticker, bsns_year, any_value(rcept_no) AS rcept_no,
                max(stake) FILTER (WHERE selection_priority = best_priority) AS own_major_stake,
                any_value(n.available_from) AS stake_available_from
            FROM major_ranked m
            LEFT JOIN next_receipt_session n USING (rcept_no)
            GROUP BY corp_code, bsns_year
        ),
        major_features AS (
            SELECT
                ticker, bsns_year, own_major_stake,
                stake_available_from AS level_available_from,
                CASE WHEN previous_year = bsns_year - 1
                     THEN own_major_stake - previous_stake END AS own_major_stake_chg,
                CASE WHEN previous_year = bsns_year - 1
                     THEN greatest(stake_available_from, previous_available_from) END
                    AS change_available_from
            FROM (
                SELECT *,
                    lag(bsns_year) OVER (PARTITION BY corp_code ORDER BY bsns_year)
                        AS previous_year,
                    lag(own_major_stake) OVER (PARTITION BY corp_code ORDER BY bsns_year)
                        AS previous_stake,
                    lag(stake_available_from) OVER (PARTITION BY corp_code ORDER BY bsns_year)
                        AS previous_available_from
                FROM major_annual
            )
        ),
        daily AS (
            SELECT
                u.trade_date, u.ticker, u.market,
                eg.hc_employee_growth_yoy,
                p.hc_revenue_per_employee,
                ml.own_major_stake,
                mc.own_major_stake_chg,
                eg.available_from AS hc_employee_growth_available_from,
                p.available_from AS hc_productivity_available_from,
                ml.level_available_from AS own_major_stake_available_from,
                mc.change_available_from AS own_major_stake_chg_available_from,
                'final_vintage' AS periodic_source_warning,
                {FINAL_VINTAGE_CAPTURE_RATIO}::DOUBLE AS vintage_capture_ratio
            FROM {universe_view} u
            ASOF LEFT JOIN employee_growth eg
                ON u.ticker = eg.ticker AND u.trade_date >= eg.available_from
            ASOF LEFT JOIN productivity p
                ON u.ticker = p.ticker AND u.trade_date >= p.available_from
            ASOF LEFT JOIN major_features ml
                ON u.ticker = ml.ticker AND u.trade_date >= ml.level_available_from
            ASOF LEFT JOIN major_features mc
                ON u.ticker = mc.ticker AND u.trade_date >= mc.change_available_from
            WHERE u.in_universe
        )
        SELECT
            *,
            lag(hc_employee_growth_yoy) OVER w AS hc_employee_growth_yoy_lag1,
            lag(hc_revenue_per_employee) OVER w AS hc_revenue_per_employee_lag1,
            lag(own_major_stake) OVER w AS own_major_stake_lag1,
            lag(own_major_stake_chg) OVER w AS own_major_stake_chg_lag1
        FROM daily
        WINDOW w AS (PARTITION BY ticker, market ORDER BY trade_date)
    """


def materialize_periodic_extras(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    universe_view: str = "dim_universe_broad_daily",
    employee_view: str = "dart_employee_raw",
    governance_view: str = "dart_governance_raw",
    filing_receipt_view: str = "dart_filing_receipt_raw",
    vintage_view: str = "stock_metric_vintage_fact",
    force: bool = False,
) -> str:
    materialize(
        con,
        config,
        PERIODIC_EXTRAS_TABLE,
        build_periodic_extras_sql(
            universe_view,
            employee_view,
            governance_view,
            filing_receipt_view,
            vintage_view,
        ),
        force=force,
    )
    return register_mart_view(con, config, PERIODIC_EXTRAS_TABLE)
