"""``fin_quarterly_metric_vintage_v4`` -- opt-in quarterly mart that fixes three defects.

Decided 2026-11-01 (05_prep_a_financial_target.md, "마트에 남은 결함" / "결정"). It lives in its
own module so the closed kr_next files (``financial_quarters.py``, ``metric_vintages.py``,
``metric_vintages_delisted.py``) stay byte-identical and their default output cannot move. The
v4 statement *wraps* the unchanged ``v2`` quarterly statement; nothing is copied from it.

1. **Fiscal renumbering.** ``bsns_year`` of the vintage fact is the calendar year of the period
   end, so a March-end company has its Q4 (period end 2022-03) and its next Q1-Q3 (2022-06...)
   in the same ``bsns_year``. Q4 then became ``annual - next year's Q1..Q3`` and no TTM window
   was four consecutive quarters. v4 relabels every filing with the fiscal year its *fiscal*
   year ends in: ``year(period_end + 3 * (4 - ordinal) months)`` where ``ordinal`` is the
   report's own fiscal quarter (Q1 11013 .. annual 11011, DART numbers them by fiscal quarter).
   A December-end company gets exactly its old ``bsns_year``. A filing without a
   ``statement_period_end`` keeps its ``bsns_year`` (treated as calendar).
2. **Outlier block.** A duration metric's standalone quarter value with ``abs > 1e14`` KRW
   (XBRL unit errors, 10^6 / 10^3 times too large) is set to NULL with a reason.
3. **Q4 propagation.** A direct-interim Q4 is ``annual - (Q1+Q2+Q3)``. If Q2 or Q3 is NULL
   because of ``standalone_source_conflict`` the Q4 is NULL too. Beyond what was decided, a Q4
   is also NULL when one of its Q1..Q3 direct values is itself over the limit, since that
   quarter feeds the difference.

The correction-version policy is unchanged (earliest non-revision vintage). TTM is recomputed
from the adjusted standalone values, so a NULLed quarter leaves every TTM window it is in NULL.
Extra column: ``standalone_null_reason`` (NULL when the value was not blocked here or upstream).
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import MartPlan, materialize, register_mart_view
from modeler.etl.marts import financial_quarters as _fq

FQMV_V4_TABLE = "fin_quarterly_metric_vintage_v4"
SEMANTICS_V4 = "v4"
#: The v4 mart reads the v3 fact (3-month income contexts). The quarterly base is ``v2``.
DEFAULT_VINTAGE_VIEW = "stock_metric_vintage_fact_v3"
#: Largest absolute standalone quarter value kept (KRW).
OUTLIER_ABS_LIMIT = 100_000_000_000_000

REASON_SOURCE_CONFLICT = "standalone_source_conflict"
REASON_OUTLIER = "abs_over_1e14"
REASON_Q4_CONFLICT = "q4_blocked_q2_q3_source_conflict"
REASON_Q4_OUTLIER = "q4_blocked_prior_quarter_abs_over_1e14"

#: Fiscal-year relabelling of the vintage fact (only ``bsns_year`` changes).
_FISCAL_FACT_TEMPLATE = (
    "(SELECT * REPLACE (CAST(CASE WHEN statement_period_end IS NULL OR "
    "reprt_code NOT IN ('11013', '11012', '11014', '11011') THEN bsns_year "
    "ELSE year(statement_period_end + to_months(3 * (4 - CASE reprt_code "
    "WHEN '11013' THEN 1 WHEN '11012' THEN 2 WHEN '11014' THEN 3 ELSE 4 END))) END "
    "AS INTEGER) AS bsns_year) FROM {view}) AS fiscal_fact"
)


def build_fin_quarterly_metric_vintage_v4_sql(
    *,
    vintage_view: str = DEFAULT_VINTAGE_VIEW,
    pairing_tolerance: float = 0.0,
) -> str:
    """SQL producing ``fin_quarterly_metric_vintage_v4`` (see the module docstring)."""
    base_sql = _fq.build_fin_quarterly_metric_vintage_sql(
        vintage_view=_FISCAL_FACT_TEMPLATE.format(view=vintage_view),
        pairing_tolerance=pairing_tolerance,
        semantics=_fq.SEMANTICS_V2,
    )
    duration = _fq._in_list(_fq.DURATION_METRICS)
    direct = _fq._in_list(_fq.DIRECT_INTERIM_METRICS)
    lim = OUTLIER_ABS_LIMIT

    def over(col: str) -> str:
        return f"({col} IS NOT NULL AND abs({col}) > {lim})"

    return f"""
    WITH base AS MATERIALIZED (
        {base_sql}
    ),
    adjusted_in AS (
        -- Q4's own Q1..Q3 are the quarters with seq_key 3, 2 and 1 below it (same company,
        -- metric and basis); a RANGE frame is exact on the integer key, gaps included.
        SELECT
            b.*,
            bool_or(COALESCE(standalone_source_conflict, FALSE)) OVER prev3 AS prior_conflict,
            bool_or({over("standalone_value")}) OVER prev3 AS prior_outlier
        FROM base b
        WINDOW prev3 AS (
            PARTITION BY ticker, metric_code, fs_basis ORDER BY seq_key
            RANGE BETWEEN 3 PRECEDING AND 1 PRECEDING
        )
    ),
    adjusted AS MATERIALIZED (
        SELECT
            ticker, market, corp_code, metric_code, fs_basis, bsns_year, quarter,
            quarter_ordinal, seq_key, statement_period_end, rcept_no, available_from,
            metric_kind,
            CASE
                WHEN metric_kind = 'direct_interim' AND quarter_ordinal = 4
                 AND (prior_conflict OR prior_outlier)
                THEN NULL
                WHEN metric_code IN {duration} AND {over("standalone_value")} THEN NULL
                ELSE standalone_value
            END AS standalone_value,
            standalone_source_conflict, cumulative_derived_value, comparative_q_amount,
            CASE
                WHEN metric_kind = 'direct_interim' AND quarter_ordinal = 4
                 AND prior_conflict
                THEN '{REASON_Q4_CONFLICT}'
                WHEN metric_kind = 'direct_interim' AND quarter_ordinal = 4
                 AND prior_outlier
                THEN '{REASON_Q4_OUTLIER}'
                WHEN metric_code IN {duration} AND {over("standalone_value")}
                THEN '{REASON_OUTLIER}'
                WHEN metric_code IN {direct} AND standalone_source_conflict IS TRUE
                 AND standalone_value IS NULL
                THEN '{REASON_SOURCE_CONFLICT}'
            END AS standalone_null_reason
        FROM adjusted_in
    ),
    with_neighbors AS (
        SELECT
            c.*,
            n1.standalone_value AS neighbor1_value, n1.available_from AS neighbor1_avail,
            n2.standalone_value AS neighbor2_value, n2.available_from AS neighbor2_avail,
            n3.standalone_value AS neighbor3_value, n3.available_from AS neighbor3_avail,
            n4.standalone_value AS neighbor4_value
        FROM adjusted c
        LEFT JOIN adjusted n1
          ON n1.ticker = c.ticker AND n1.metric_code = c.metric_code
         AND n1.fs_basis = c.fs_basis AND n1.seq_key = c.seq_key - 1
        LEFT JOIN adjusted n2
          ON n2.ticker = c.ticker AND n2.metric_code = c.metric_code
         AND n2.fs_basis = c.fs_basis AND n2.seq_key = c.seq_key - 2
        LEFT JOIN adjusted n3
          ON n3.ticker = c.ticker AND n3.metric_code = c.metric_code
         AND n3.fs_basis = c.fs_basis AND n3.seq_key = c.seq_key - 3
        LEFT JOIN adjusted n4
          ON n4.ticker = c.ticker AND n4.metric_code = c.metric_code
         AND n4.fs_basis = c.fs_basis AND n4.seq_key = c.seq_key - 4
    )
    SELECT
        ticker, market, corp_code, metric_code, fs_basis, bsns_year, quarter,
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
             AND available_from IS NOT NULL AND neighbor1_avail IS NOT NULL
             AND neighbor2_avail IS NOT NULL AND neighbor3_avail IS NOT NULL
            THEN greatest(available_from, neighbor1_avail, neighbor2_avail, neighbor3_avail)
        END AS ttm_available_from,
        standalone_null_reason
    FROM with_neighbors
    """


def register_fin_quarterly_metric_vintage_v4_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = FQMV_V4_TABLE,
    vintage_view: str = DEFAULT_VINTAGE_VIEW,
    pairing_tolerance: float = 0.0,
) -> str:
    """Register a DuckDB view over the v4 SQL (no parquet -- tests/parity)."""
    sql = build_fin_quarterly_metric_vintage_v4_sql(
        vintage_view=vintage_view, pairing_tolerance=pairing_tolerance
    )
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_fin_quarterly_metric_vintage_v4(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    vintage_view: str = DEFAULT_VINTAGE_VIEW,
    pairing_tolerance: float = 0.0,
    force: bool = False,
    table_name: str = FQMV_V4_TABLE,
) -> str:
    """Build + register the v4 quarterly mart as parquet under a new name.

    ``vintage_view`` must already be registered on ``con`` (default: the v3 fact). The cache
    metadata records ``semantics_version="v4"``. The default name never collides with the
    v1/v2/v3 marts, so nothing existing is overwritten.
    """
    sql = build_fin_quarterly_metric_vintage_v4_sql(
        vintage_view=vintage_view, pairing_tolerance=pairing_tolerance
    )
    plan = MartPlan(
        plan_id="single",
        final_sql=sql,
        semantics_version=SEMANTICS_V4,
        metadata=(("fiscal_year_renumber", True), ("outlier_abs_limit", OUTLIER_ABS_LIMIT)),
    )
    materialize(con, config, table_name, sql, force=force, plan=plan)
    return register_mart_view(con, config, table_name)
