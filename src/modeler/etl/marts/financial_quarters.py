"""``fin_quarterly_metric_vintage`` — Phase B B-3 (04_specific_plan_B.md §3.7, B-3).

Turns ``stock_metric_vintage_fact`` (B-2) instant/cumulative raw filing values
into a PIT-consistent standalone-quarter and TTM history, per metric "kind":

    direct_interim      revenue, net_income, ... — an interim (Q1/half/Q3)
                         IS/CIS filing's own ``thstrm_amount`` is *already* the
                         direct 3-month value for that quarter (OpenDART
                         convention); only Q4 needs computing, as the annual
                         filing's total minus the three known interim quarters.
    cumulative_reported  operating_cash_flow, ... — a CF filing's
                         ``thstrm_amount`` is the *cumulative* year-to-date
                         value regardless of quarter; every quarter here is a
                         sequential difference of two adjacent cumulative
                         filings.
    instant              total_assets, issued_shares, ... — a period-end
                         balance, passed through unchanged (no differencing).
    weighted_share       weighted_avg_shares, diluted_shares — a cumulative
                         time-weighted average, reconstructed via §3.7's
                         ``n * cumulative_average - (n-1) * prior_cumulative_average``
                         algebra rather than simple differencing.

Grain: ``(ticker, metric_code, fs_basis, bsns_year, quarter)``.

Point-in-time / multi-vintage scope (§1.1 condition 3, honest limitation):
    When B-1's real backfill eventually produces more than one vintage for the
    same (ticker, metric_code, fs_basis, bsns_year, reprt_code) position, this
    mart picks exactly one "primary" vintage per position — the earliest
    non-revision vintage, or the earliest known vintage if none is confirmed
    original — rather than generating the full combinatorial set of "effective
    rows" the plan's completion criterion describes. There is no real
    multi-vintage data yet to validate a fuller temporal join against; this is
    the same honest deferral B-2 makes for ``complete_original_and_revisions``.
    The dominant single-vintage case (today's raw) is unaffected either way.
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view

FQMV_TABLE = "fin_quarterly_metric_vintage"

DIRECT_INTERIM_METRICS = frozenset(
    {
        "revenue",
        "cogs",
        "gross_profit",
        "sga",
        "operating_income",
        "net_income",
        "controlling_net_income",
    }
)
CUMULATIVE_REPORTED_METRICS = frozenset(
    {
        "operating_cash_flow",
        "investing_cash_flow",
        "financing_cash_flow",
        "interest_received",
        "interest_paid",
        "dividends_paid",
        "capex_ppe",
        "capex_intangible",
        "borrowing_proceeds_long_term",
        "borrowing_repayments_long_term",
        "treasury_share_acquisition_amount",
    }
)
INSTANT_METRICS = frozenset(
    {
        "total_assets",
        "total_liabilities",
        "total_equity",
        "cash_and_cash_equivalents",
        "issued_shares",
        "treasury_shares",
        # F-5.1/F-5.2. All five are balance-sheet lines, so they take the
        # instant path: one quarter is enough and no TTM accumulation applies.
        # Registering the metric in `metric_rules` is not sufficient on its own
        # -- `_ALL_QUARTERED_METRICS` filters this mart's input, so a metric
        # absent here never reaches `feat_fin_risk` at all.
        "current_assets",
        "current_liabilities",
        "retained_earnings",
        "borrowings_short_term",
        "borrowings_long_term",
    }
)
WEIGHTED_SHARE_METRICS = frozenset({"weighted_avg_shares", "diluted_shares"})

DURATION_METRICS = DIRECT_INTERIM_METRICS | CUMULATIVE_REPORTED_METRICS
_ALL_QUARTERED_METRICS = DURATION_METRICS | INSTANT_METRICS | WEIGHTED_SHARE_METRICS

# reprt_code -> quarter ordinal within its own fiscal year (bsns_year).
_REPRT_TO_ORDINAL = {"11013": 1, "11012": 2, "11014": 3, "11011": 4}
_ORDINAL_TO_QUARTER = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}


def _in_list(codes: frozenset[str]) -> str:
    return "(" + ", ".join(f"'{c}'" for c in sorted(codes)) + ")"


def build_fin_quarterly_metric_vintage_sql(
    *,
    vintage_view: str = "stock_metric_vintage_fact",
    pairing_tolerance: float = 0.0,
) -> str:
    """SQL producing ``fin_quarterly_metric_vintage`` from B-2's vintage fact."""
    metrics_list = _in_list(_ALL_QUARTERED_METRICS)
    direct_list = _in_list(DIRECT_INTERIM_METRICS)
    cumulative_list = _in_list(CUMULATIVE_REPORTED_METRICS)
    instant_list = _in_list(INSTANT_METRICS)
    weighted_list = _in_list(WEIGHTED_SHARE_METRICS)
    duration_list = _in_list(DURATION_METRICS)
    quarter_label_case = (
        "CASE quarter_ordinal WHEN 1 THEN 'Q1' WHEN 2 THEN 'Q2' "
        "WHEN 3 THEN 'Q3' WHEN 4 THEN 'Q4' END"
    )

    return f"""
    WITH winners AS (
        -- §1.1: exactly one primary vintage per filing position (see module
        -- docstring's point-in-time / multi-vintage scope note).
        SELECT *
        FROM {vintage_view}
        WHERE metric_code IN {metrics_list}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY ticker, metric_code, fs_basis, bsns_year, reprt_code
            ORDER BY COALESCE(is_revision, FALSE) ASC, available_from ASC, rcept_no ASC
        ) = 1
    ),
    wide AS (
        SELECT
            ticker, market, corp_code, metric_code, fs_basis, bsns_year,
            MAX(CASE WHEN reprt_code = '11013' THEN value_numeric END) AS q1_value,
            MAX(CASE WHEN reprt_code = '11012' THEN value_numeric END) AS q2_value,
            MAX(CASE WHEN reprt_code = '11014' THEN value_numeric END) AS q3_value,
            MAX(CASE WHEN reprt_code = '11011' THEN value_numeric END) AS q4_value,
            MAX(CASE WHEN reprt_code = '11013' THEN available_from END) AS q1_avail,
            MAX(CASE WHEN reprt_code = '11012' THEN available_from END) AS q2_avail,
            MAX(CASE WHEN reprt_code = '11014' THEN available_from END) AS q3_avail,
            MAX(CASE WHEN reprt_code = '11011' THEN available_from END) AS q4_avail,
            MAX(CASE WHEN reprt_code = '11013' THEN rcept_no END) AS q1_rcept,
            MAX(CASE WHEN reprt_code = '11012' THEN rcept_no END) AS q2_rcept,
            MAX(CASE WHEN reprt_code = '11014' THEN rcept_no END) AS q3_rcept,
            MAX(CASE WHEN reprt_code = '11011' THEN rcept_no END) AS q4_rcept,
            MAX(CASE WHEN reprt_code = '11013' THEN statement_period_end END) AS q1_pe,
            MAX(CASE WHEN reprt_code = '11012' THEN statement_period_end END) AS q2_pe,
            MAX(CASE WHEN reprt_code = '11014' THEN statement_period_end END) AS q3_pe,
            MAX(CASE WHEN reprt_code = '11011' THEN statement_period_end END) AS q4_pe,
            MAX(CASE WHEN reprt_code = '11012' THEN cumulative_value_numeric END)
                AS q2_cumulative_reported,
            MAX(CASE WHEN reprt_code = '11014' THEN cumulative_value_numeric END)
                AS q3_cumulative_reported,
            MAX(CASE WHEN reprt_code = '11013' THEN comparative_q_amount END) AS q1_comparative,
            MAX(CASE WHEN reprt_code = '11012' THEN comparative_q_amount END) AS q2_comparative,
            MAX(CASE WHEN reprt_code = '11014' THEN comparative_q_amount END) AS q3_comparative,
            MAX(CASE WHEN reprt_code = '11011' THEN comparative_q_amount END) AS q4_comparative
        FROM winners
        WHERE metric_code IN {duration_list} OR metric_code IN {weighted_list}
        GROUP BY ticker, market, corp_code, metric_code, fs_basis, bsns_year
    ),
    -- Q1 is trivial for every non-instant kind: the filing's own value already
    -- *is* Q1 standalone (direct interim value, cumulative-through-Q1, or the
    -- Q1 cumulative-average itself).
    quarters AS (
        SELECT
            ticker, market, corp_code, metric_code, fs_basis, bsns_year,
            1 AS quarter_ordinal, q1_pe AS statement_period_end,
            q1_rcept AS rcept_no, q1_avail AS available_from,
            q1_value AS standalone_value,
            CAST(NULL AS BOOLEAN) AS standalone_source_conflict,
            CAST(NULL AS DECIMAL(30,4)) AS cumulative_derived_value,
            q1_comparative AS comparative_q_amount
        FROM wide
        WHERE q1_value IS NOT NULL

        UNION ALL
        SELECT
            ticker, market, corp_code, metric_code, fs_basis, bsns_year,
            2, q2_pe, q2_rcept,
            CASE WHEN metric_code IN {direct_list} THEN q2_avail
                 ELSE greatest(q2_avail, q1_avail) END,
            -- §3.7: a direct/cumulative-derived conflict excludes the value
            -- from official output (NULL), keeping only the diagnostic flag.
            CASE
                WHEN metric_code IN {direct_list} THEN
                    CASE WHEN q2_cumulative_reported IS NOT NULL
                              AND abs((q1_value + q2_value) - q2_cumulative_reported)
                                  > {pairing_tolerance}
                         THEN NULL
                         ELSE q2_value END
                WHEN metric_code IN {cumulative_list} THEN q2_value - q1_value
                WHEN metric_code IN {weighted_list} THEN 2 * q2_value - q1_value
            END,
            CASE WHEN metric_code IN {direct_list} AND q2_cumulative_reported IS NOT NULL
                 THEN abs((q1_value + q2_value) - q2_cumulative_reported) > {pairing_tolerance}
            END,
            CASE WHEN metric_code IN {direct_list} THEN q1_value + q2_value END,
            q2_comparative
        FROM wide
        -- direct_interim's own Q2 value never needs Q1 (only the diagnostic
        -- cross-check does, which degrades to NULL — not missing — without it).
        WHERE q2_value IS NOT NULL AND (metric_code IN {direct_list} OR q1_value IS NOT NULL)

        UNION ALL
        SELECT
            ticker, market, corp_code, metric_code, fs_basis, bsns_year,
            3, q3_pe, q3_rcept,
            CASE WHEN metric_code IN {direct_list} THEN q3_avail
                 ELSE greatest(q3_avail, q2_avail) END,
            CASE
                WHEN metric_code IN {direct_list} THEN
                    CASE WHEN q3_cumulative_reported IS NOT NULL
                              AND abs((q1_value + q2_value + q3_value) - q3_cumulative_reported)
                                  > {pairing_tolerance}
                         THEN NULL
                         ELSE q3_value END
                WHEN metric_code IN {cumulative_list} THEN q3_value - q2_value
                WHEN metric_code IN {weighted_list} THEN 3 * q3_value - 2 * q2_value
            END,
            CASE WHEN metric_code IN {direct_list} AND q3_cumulative_reported IS NOT NULL
                 THEN abs((q1_value + q2_value + q3_value) - q3_cumulative_reported)
                      > {pairing_tolerance}
            END,
            CASE WHEN metric_code IN {direct_list} THEN q1_value + q2_value + q3_value END,
            q3_comparative
        FROM wide
        -- Same reasoning as Q2: direct_interim's Q3 value stands alone.
        WHERE q3_value IS NOT NULL AND (metric_code IN {direct_list} OR q2_value IS NOT NULL)

        UNION ALL
        SELECT
            ticker, market, corp_code, metric_code, fs_basis, bsns_year,
            4, q4_pe, q4_rcept,
            CASE WHEN metric_code IN {direct_list}
                 THEN greatest(q4_avail, q1_avail, q2_avail, q3_avail)
                 ELSE greatest(q4_avail, q3_avail) END,
            CASE
                WHEN metric_code IN {direct_list} THEN q4_value - (q1_value + q2_value + q3_value)
                WHEN metric_code IN {cumulative_list} THEN q4_value - q3_value
                WHEN metric_code IN {weighted_list} THEN 4 * q4_value - 3 * q3_value
            END,
            CAST(NULL AS BOOLEAN),
            CAST(NULL AS DECIMAL(30,4)),
            q4_comparative
        FROM wide
        WHERE q4_value IS NOT NULL
          AND (metric_code NOT IN {direct_list}
               OR (q1_value IS NOT NULL AND q2_value IS NOT NULL AND q3_value IS NOT NULL))
    ),
    instant_quarters AS (
        -- Instant (period-end balance) metrics are never differenced.
        SELECT
            ticker, market, corp_code, metric_code, fs_basis, bsns_year,
            CASE reprt_code
                WHEN '11013' THEN 1 WHEN '11012' THEN 2
                WHEN '11014' THEN 3 WHEN '11011' THEN 4
            END AS quarter_ordinal,
            statement_period_end, rcept_no, available_from,
            value_numeric AS standalone_value,
            CAST(NULL AS BOOLEAN) AS standalone_source_conflict,
            CAST(NULL AS DECIMAL(30,4)) AS cumulative_derived_value,
            CAST(NULL AS DECIMAL(30,4)) AS comparative_q_amount
        FROM winners
        WHERE metric_code IN {instant_list}
    ),
    combined AS (
        SELECT
            ticker, market, corp_code, metric_code, fs_basis, bsns_year, quarter_ordinal,
            statement_period_end, rcept_no, available_from, standalone_value,
            standalone_source_conflict, cumulative_derived_value, comparative_q_amount,
            CASE
                WHEN metric_code IN {direct_list} THEN 'direct_interim'
                WHEN metric_code IN {cumulative_list} THEN 'cumulative_reported'
                WHEN metric_code IN {weighted_list} THEN 'weighted_share'
                ELSE 'instant'
            END AS metric_kind,
            bsns_year * 4 + quarter_ordinal AS seq_key
        FROM quarters
        UNION ALL
        SELECT
            ticker, market, corp_code, metric_code, fs_basis, bsns_year, quarter_ordinal,
            statement_period_end, rcept_no, available_from, standalone_value,
            standalone_source_conflict, cumulative_derived_value, comparative_q_amount,
            'instant' AS metric_kind,
            bsns_year * 4 + quarter_ordinal AS seq_key
        FROM instant_quarters
    ),
    -- Exact seq_key self-joins, not LAG(n): LAG counts *rows*, which only
    -- equals "n quarters back" when every quarter in between has a row. An
    -- instant metric with sparse filings (or any gap) would silently pull the
    -- wrong quarter under LAG; a join on ``seq_key = current - n`` either
    -- finds the exact quarter or (correctly) finds nothing.
    with_neighbors AS (
        SELECT
            c.*,
            n1.standalone_value AS neighbor1_value, n1.available_from AS neighbor1_avail,
            n2.standalone_value AS neighbor2_value, n2.available_from AS neighbor2_avail,
            n3.standalone_value AS neighbor3_value, n3.available_from AS neighbor3_avail,
            n4.standalone_value AS neighbor4_value
        FROM combined c
        LEFT JOIN combined n1
          ON n1.ticker = c.ticker AND n1.metric_code = c.metric_code
         AND n1.fs_basis = c.fs_basis AND n1.seq_key = c.seq_key - 1
        LEFT JOIN combined n2
          ON n2.ticker = c.ticker AND n2.metric_code = c.metric_code
         AND n2.fs_basis = c.fs_basis AND n2.seq_key = c.seq_key - 2
        LEFT JOIN combined n3
          ON n3.ticker = c.ticker AND n3.metric_code = c.metric_code
         AND n3.fs_basis = c.fs_basis AND n3.seq_key = c.seq_key - 3
        LEFT JOIN combined n4
          ON n4.ticker = c.ticker AND n4.metric_code = c.metric_code
         AND n4.fs_basis = c.fs_basis AND n4.seq_key = c.seq_key - 4
    )
    SELECT
        ticker, market, corp_code, metric_code, fs_basis, bsns_year,
        {quarter_label_case} AS quarter,
        quarter_ordinal, seq_key, statement_period_end, rcept_no, available_from,
        metric_kind, standalone_value, standalone_source_conflict,
        cumulative_derived_value, comparative_q_amount,
        neighbor4_value AS value_lag_4q,
        CASE
            WHEN metric_kind IN ('direct_interim', 'cumulative_reported')
             AND standalone_value IS NOT NULL AND neighbor1_value IS NOT NULL
             AND neighbor2_value IS NOT NULL AND neighbor3_value IS NOT NULL
            THEN standalone_value + neighbor1_value + neighbor2_value + neighbor3_value
        END AS ttm_value,
        (
            metric_kind IN ('direct_interim', 'cumulative_reported')
            AND standalone_value IS NOT NULL AND neighbor1_value IS NOT NULL
            AND neighbor2_value IS NOT NULL AND neighbor3_value IS NOT NULL
        ) AS ttm_complete,
        CASE
            WHEN metric_kind IN ('direct_interim', 'cumulative_reported')
             AND standalone_value IS NOT NULL AND neighbor1_value IS NOT NULL
             AND neighbor2_value IS NOT NULL AND neighbor3_value IS NOT NULL
            THEN greatest(available_from, neighbor1_avail, neighbor2_avail, neighbor3_avail)
        END AS ttm_available_from
    FROM with_neighbors
    """


def register_fin_quarterly_metric_vintage_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = FQMV_TABLE,
    vintage_view: str = "stock_metric_vintage_fact",
    pairing_tolerance: float = 0.0,
) -> str:
    """Register a DuckDB view over the SQL above (no parquet — tests/parity)."""
    sql = build_fin_quarterly_metric_vintage_sql(
        vintage_view=vintage_view, pairing_tolerance=pairing_tolerance
    )
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_fin_quarterly_metric_vintage(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    vintage_view: str = "stock_metric_vintage_fact",
    pairing_tolerance: float = 0.0,
    force: bool = False,
) -> str:
    """Build + register ``fin_quarterly_metric_vintage`` as a cached parquet mart.

    Requires ``vintage_view`` (B-2's ``stock_metric_vintage_fact``) already
    registered on ``con``.
    """
    materialize(
        con,
        config,
        FQMV_TABLE,
        build_fin_quarterly_metric_vintage_sql(
            vintage_view=vintage_view, pairing_tolerance=pairing_tolerance
        ),
        force=force,
    )
    return register_mart_view(con, config, FQMV_TABLE)
