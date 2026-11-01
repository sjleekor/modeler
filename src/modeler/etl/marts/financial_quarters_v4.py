"""``fin_quarterly_metric_vintage_v4`` -- opt-in quarterly mart that fixes three defects.

Decided 2026-11-01 (05_prep_a_financial_target.md, "마트에 남은 결함" / "결정"). It lives in its
own module so the closed kr_next files (``financial_quarters.py``, ``metric_vintages.py``,
``metric_vintages_delisted.py``) stay byte-identical and their default output cannot move. The
v4 statement *wraps* the unchanged ``v2`` quarterly statement; nothing is copied from it.

1. **Fiscal renumbering.** ``bsns_year`` of the vintage fact is the calendar year of the period
   end, so a March-end company has its Q4 (period end 2022-03) and its next Q1-Q3 (2022-06...)
   in the same ``bsns_year``. Q4 then became ``annual - next year's Q1..Q3`` and no TTM window
   was four consecutive quarters.

   v4 places every filing by its period end and the company's own fiscal year ends, taken from
   the annual-report names (``사업보고서 (YYYY.MM)``) of the receipt list. With ``next`` the first
   fiscal year end on or after the period end, ``k`` the months between them and ``prev`` the
   fiscal year end before it:

   * fiscal year ``year(next) - k // 12``, fiscal quarter ``4 - (k % 12) / 3`` (``k % 12`` must
     be 0, 3, 6 or 9);
   * after the newest fiscal year end on the list the same is counted forward from ``prev``
     (a 12-month year is assumed);
   * the fact's ``reprt_code`` should name that same fiscal quarter, but does not always
     (non-December companies, revisions: e.g. a ``분기보고서 (2015.06)`` of a March company
     stored under ``11012``, or an annual report's balance sheet stored under ``11013``). Such
     a row is moved to the right code when its value means the same there: the report kind of
     its receipt name must match the fiscal quarter, and either both codes are interim (the
     fact extracts every interim code alike) or the metric is not an income item
     (balance-sheet and cash-flow items are extracted alike in every code). An income item
     across interim <-> annual was extracted with the other period type (3-month vs. longest
     context) and is dropped;
   * a fiscal year whose length is not a multiple of 12 months (a fiscal-year-end change, a
     REIT's 6-month year) is dropped as a whole: its quarters do not fit Q1..Q4;
   * a filing whose period end came from the calendar fallback (``make_date(bsns_year, 3/6/9/12,
     ...)``) takes the month of its own report name instead;
   * a company with no annual-report name on the list keeps the code-based position
     ``year(period_end + 3 * (4 - quarter) months)``.

   A December-end company keeps exactly its old ``bsns_year`` and quarter. The placement of a
   filing reads annual-report names that may postdate it (a later fiscal-year-end change decides
   that the filings in between are a transition year); values and ``available_from`` are not
   affected, only which rows exist. :func:`build_fiscal_position_audit_sql` shows the decision
   for every fact row.
2. **Outlier block.** A duration metric's standalone quarter value with ``abs > 1e14`` KRW
   (XBRL unit errors, 10^6 / 10^3 times too large) is set to NULL with a reason.
3. **Q4 propagation.** A direct-interim Q4 is ``annual - (Q1+Q2+Q3)``. If Q2 or Q3 is NULL
   because of ``standalone_source_conflict`` the Q4 is NULL too. Beyond what was decided, a Q4
   is also NULL when one of its Q1..Q3 direct values is itself over the limit, since that
   quarter feeds the difference.

TTM (and ``value_lag_4q``) is recomputed from the adjusted standalone values and only when the
four period ends are consecutive calendar quarters (3, 6 and 9 months apart; the lag 12), so a
NULLed or dropped quarter leaves every window it is in NULL instead of reaching past it.

The correction-version policy is unchanged (earliest non-revision vintage).
Extra column: ``standalone_null_reason`` -- why ``standalone_value`` is NULL (the three blocks
above, an upstream ``standalone_source_conflict``, or a cumulative Q4 without its Q3); NULL when
the value is present. A reported 0 or negative value is never NULLed.

``fin_quarterly_metric_vintage_dl_v4`` is the same statement over the delisted-keeping ``v3``
fact (``stock_metric_vintage_fact_dl_v3`` of ``metric_vintages_delisted``).
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import MartPlan, materialize, register_mart_view
from modeler.etl.marts import financial_quarters as _fq

FQMV_V4_TABLE = "fin_quarterly_metric_vintage_v4"
FQMV_DL_V4_TABLE = "fin_quarterly_metric_vintage_dl_v4"
SEMANTICS_V4 = "v4"
#: The v4 mart reads the v3 fact (3-month income contexts). The quarterly base is ``v2``.
DEFAULT_VINTAGE_VIEW = "stock_metric_vintage_fact_v3"
#: The delisted-keeping v3 fact (``metric_vintages_delisted`` with ``semantics="v3"``).
DEFAULT_DL_VINTAGE_VIEW = "stock_metric_vintage_fact_dl_v3"
DEFAULT_RECEIPT_VIEW = "dart_filing_receipt_raw"
#: Largest absolute standalone quarter value kept (KRW).
OUTLIER_ABS_LIMIT = 100_000_000_000_000

REASON_SOURCE_CONFLICT = "standalone_source_conflict"
REASON_OUTLIER = "abs_over_1e14"
REASON_Q4_CONFLICT = "q4_blocked_q2_q3_source_conflict"
REASON_Q4_OUTLIER = "q4_blocked_prior_quarter_abs_over_1e14"
#: Upstream (unchanged ``v2`` statement): a cumulative/weighted Q4 is ``annual - Q3`` and is
#: NULL when that fiscal year's Q3 filing is missing. Only the reason is added here.
REASON_Q4_NO_Q3 = "q4_missing_q3_cumulative"

#: ``fiscal_position_status`` values. Rows with a status in :data:`KEPT_POSITIONS` reach the mart.
POSITION_OK = "ok"
POSITION_OK_AFTER_LAST_ANNUAL = "ok_after_last_annual"
POSITION_NO_ANNUAL_REFERENCE = "kept_no_annual_reference"
POSITION_UNPOSITIONED = "kept_unpositioned"
POSITION_RELABELLED = "relabelled_report_code"
POSITION_CODE_MISMATCH = "dropped_report_code_mismatch"
POSITION_IRREGULAR = "dropped_irregular_period_end"
POSITION_TRANSITION_YEAR = "dropped_non_12_month_fiscal_year"
KEPT_POSITIONS = (
    POSITION_OK,
    POSITION_OK_AFTER_LAST_ANNUAL,
    POSITION_RELABELLED,
    POSITION_NO_ANNUAL_REFERENCE,
    POSITION_UNPOSITIONED,
)

#: ``분기보고서 (2024.03)``, ``[기재정정]사업보고서 (2023.12)`` ... (not ``...연장신고서``).
_PERIODIC_NAME_RE = r"^(\[[^\]]*\])?\s*(사업|반기|분기)보고서\s*\((\d{4})\.(\d{2})\)\s*$"

_CODE_ORDINAL = (
    "CASE reprt_code WHEN '11013' THEN 1 WHEN '11012' THEN 2 "
    "WHEN '11014' THEN 3 WHEN '11011' THEN 4 END"
)


def _in_sql(values: tuple[str, ...]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def build_fiscal_position_audit_sql(
    *,
    vintage_view: str = DEFAULT_VINTAGE_VIEW,
    filing_receipt_view: str = DEFAULT_RECEIPT_VIEW,
) -> str:
    """Every fact row with its fiscal placement (see point 1 of the module docstring).

    Adds ``report_name_period_end``, ``report_kind``, ``fiscal_period_end``,
    ``next_fiscal_year_end``, ``prev_fiscal_year_end``, ``fiscal_ordinal``,
    ``fiscal_position_status``, ``fiscal_bsns_year`` and ``fiscal_reprt_code`` to the fact's own
    columns. The mart keeps only rows whose status is in :data:`KEPT_POSITIONS`, with
    ``fiscal_bsns_year`` / ``fiscal_period_end`` / ``fiscal_reprt_code`` in place of the
    fact's ``bsns_year`` / ``statement_period_end`` / ``reprt_code``.
    """
    name_end = (
        f"last_day(make_date("
        f"CAST(regexp_extract(report_nm, '{_PERIODIC_NAME_RE}', 3) AS INTEGER), "
        f"CAST(regexp_extract(report_nm, '{_PERIODIC_NAME_RE}', 4) AS INTEGER), 1))"
    )
    return f"""
    WITH periodic_receipts AS (
        SELECT corp_code, rcept_no,
               regexp_extract(report_nm, '{_PERIODIC_NAME_RE}', 2) AS report_kind,
               {name_end} AS name_period_end
        FROM {filing_receipt_view}
        WHERE regexp_matches(report_nm, '{_PERIODIC_NAME_RE}')
    ),
    fiscal_year_ends AS (
        SELECT DISTINCT corp_code, name_period_end AS fiscal_year_end
        FROM periodic_receipts
        WHERE report_kind = '사업'
    ),
    receipt_period AS (
        SELECT rcept_no, max(name_period_end) AS report_name_period_end,
               CASE WHEN count(DISTINCT report_kind) = 1 THEN any_value(report_kind) END
                   AS report_kind
        FROM periodic_receipts
        GROUP BY rcept_no
    ),
    src AS (
        SELECT f.*, rp.report_name_period_end, rp.report_kind,
               {_CODE_ORDINAL} AS code_ordinal,
               CASE WHEN f.period_end_source = 'calendar_fallback'
                         AND rp.report_name_period_end IS NOT NULL
                    THEN rp.report_name_period_end
                    ELSE f.statement_period_end END AS fiscal_period_end
        FROM {vintage_view} f
        LEFT JOIN receipt_period rp ON rp.rcept_no = f.rcept_no
    ),
    with_next AS (
        -- the first fiscal year end on or after the period end
        SELECT s.*, n.fiscal_year_end AS next_fiscal_year_end
        FROM src s
        ASOF LEFT JOIN fiscal_year_ends n
          ON n.corp_code = s.corp_code AND n.fiscal_year_end >= s.fiscal_period_end
    ),
    with_both AS (
        -- the last fiscal year end before it
        SELECT w.*, p.fiscal_year_end AS prev_fiscal_year_end
        FROM with_next w
        ASOF LEFT JOIN fiscal_year_ends p
          ON p.corp_code = w.corp_code AND p.fiscal_year_end < w.fiscal_period_end
    ),
    measured AS (
        SELECT *,
               date_diff('month', fiscal_period_end, next_fiscal_year_end) AS months_to_next,
               date_diff('month', prev_fiscal_year_end, fiscal_period_end) AS months_from_prev,
               date_diff('month', prev_fiscal_year_end, next_fiscal_year_end) AS fiscal_year_months
        FROM with_both
    ),
    placed AS (
        SELECT *,
            CASE
                WHEN fiscal_period_end IS NULL OR code_ordinal IS NULL THEN NULL
                WHEN next_fiscal_year_end IS NOT NULL THEN
                    CASE WHEN months_to_next % 12 IN (0, 3, 6, 9)
                         THEN 4 - (months_to_next % 12) // 3 END
                WHEN prev_fiscal_year_end IS NOT NULL THEN
                    CASE WHEN months_from_prev > 0 AND months_from_prev % 12 IN (0, 3, 6, 9)
                         THEN ((months_from_prev // 3) - 1) % 4 + 1 END
            END AS fiscal_ordinal,
            CASE
                WHEN next_fiscal_year_end IS NOT NULL
                THEN year(next_fiscal_year_end) - months_to_next // 12
                WHEN prev_fiscal_year_end IS NOT NULL
                THEN year(prev_fiscal_year_end) + (months_from_prev + 11) // 12
            END AS placed_year
        FROM measured
    ),
    classified AS (
        SELECT *,
            CASE
                WHEN fiscal_period_end IS NULL OR code_ordinal IS NULL
                THEN '{POSITION_UNPOSITIONED}'
                WHEN next_fiscal_year_end IS NULL AND prev_fiscal_year_end IS NULL
                THEN '{POSITION_NO_ANNUAL_REFERENCE}'
                WHEN fiscal_ordinal IS NULL
                THEN '{POSITION_IRREGULAR}'
                WHEN next_fiscal_year_end IS NOT NULL AND prev_fiscal_year_end IS NOT NULL
                 AND fiscal_year_months % 12 <> 0
                THEN '{POSITION_TRANSITION_YEAR}'
                WHEN fiscal_ordinal = code_ordinal AND next_fiscal_year_end IS NOT NULL
                THEN '{POSITION_OK}'
                WHEN fiscal_ordinal = code_ordinal
                THEN '{POSITION_OK_AFTER_LAST_ANNUAL}'
                -- The fact stored the filing under another report code. Its value was
                -- extracted by that code's period type; the same rule applies to every
                -- interim code, and to balance-sheet / cash-flow items in every code. Only
                -- an income item across interim <-> annual was extracted differently.
                WHEN report_kind IS NOT NULL
                 AND report_kind = CASE fiscal_ordinal WHEN 4 THEN '사업'
                                                       WHEN 2 THEN '반기' ELSE '분기' END
                 AND ((fiscal_ordinal < 4 AND code_ordinal < 4)
                      OR metric_code NOT IN {_fq._in_list(_fq.DIRECT_INTERIM_METRICS)})
                THEN '{POSITION_RELABELLED}'
                ELSE '{POSITION_CODE_MISMATCH}'
            END AS fiscal_position_status
        FROM placed
    )
    SELECT * EXCLUDE (code_ordinal, months_to_next, months_from_prev, fiscal_year_months,
                      placed_year),
        CAST(CASE
            WHEN fiscal_position_status IN ('{POSITION_OK}', '{POSITION_OK_AFTER_LAST_ANNUAL}',
                                            '{POSITION_RELABELLED}')
            THEN placed_year
            WHEN fiscal_position_status = '{POSITION_NO_ANNUAL_REFERENCE}'
            THEN year(fiscal_period_end + to_months(3 * (4 - code_ordinal)))
            ELSE bsns_year
        END AS INTEGER) AS fiscal_bsns_year,
        CASE WHEN fiscal_position_status = '{POSITION_RELABELLED}'
             THEN CASE fiscal_ordinal WHEN 1 THEN '11013' WHEN 2 THEN '11012'
                                      WHEN 3 THEN '11014' ELSE '11011' END
             ELSE reprt_code END AS fiscal_reprt_code
    FROM classified
    """


def _fiscal_fact_relation(vintage_view: str, filing_receipt_view: str) -> str:
    """The fact with fiscal ``bsns_year`` / period end, dropped rows removed (a FROM item)."""
    audit = build_fiscal_position_audit_sql(
        vintage_view=vintage_view, filing_receipt_view=filing_receipt_view
    )
    return (
        "(SELECT * EXCLUDE (report_name_period_end, report_kind, fiscal_period_end, "
        "next_fiscal_year_end, prev_fiscal_year_end, fiscal_ordinal, fiscal_position_status, "
        "fiscal_bsns_year, fiscal_reprt_code) "
        "REPLACE (fiscal_bsns_year AS bsns_year, fiscal_period_end AS statement_period_end, "
        "fiscal_reprt_code AS reprt_code) "
        f"FROM ({audit}) AS fiscal_audit "
        f"WHERE fiscal_position_status IN {_in_sql(KEPT_POSITIONS)} "
        # One receipt can reach the same slot twice (stored under two codes, one of them
        # moved here). Keep the row that was already there, then the non-fallback one, so
        # the quarterly statement's own tie-break never sees two rows of one receipt.
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, metric_code, fs_basis, "
        "fiscal_bsns_year, fiscal_reprt_code, rcept_no "
        f"ORDER BY (fiscal_position_status = '{POSITION_RELABELLED}') ASC, "
        "(period_end_source = 'calendar_fallback') ASC, source_key ASC) = 1"
        ") AS fiscal_fact"
    )


def build_fin_quarterly_metric_vintage_v4_sql(
    *,
    vintage_view: str = DEFAULT_VINTAGE_VIEW,
    filing_receipt_view: str = DEFAULT_RECEIPT_VIEW,
    pairing_tolerance: float = 0.0,
) -> str:
    """SQL producing ``fin_quarterly_metric_vintage_v4`` (see the module docstring)."""
    base_sql = _fq.build_fin_quarterly_metric_vintage_sql(
        vintage_view=_fiscal_fact_relation(vintage_view, filing_receipt_view),
        pairing_tolerance=pairing_tolerance,
        semantics=_fq.SEMANTICS_V2,
    )
    duration = _fq._in_list(_fq.DURATION_METRICS)
    direct = _fq._in_list(_fq.DIRECT_INTERIM_METRICS)
    lim = OUTLIER_ABS_LIMIT

    def over(col: str) -> str:
        return f"({col} IS NOT NULL AND abs({col}) > {lim})"

    def step(n: int) -> str:
        # neighbour n ends exactly 3*n months earlier (consecutive calendar quarters)
        return f"date_diff('month', neighbor{n}_pe, statement_period_end) = {3 * n}"

    ttm_ok = (
        "(metric_kind IN ('direct_interim', 'cumulative_reported')\n"
        "             AND standalone_value IS NOT NULL AND neighbor1_value IS NOT NULL\n"
        "             AND neighbor2_value IS NOT NULL AND neighbor3_value IS NOT NULL\n"
        f"             AND {step(1)} AND {step(2)} AND {step(3)})"
    )

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
                WHEN metric_kind IN ('cumulative_reported', 'weighted_share')
                 AND quarter_ordinal = 4 AND standalone_value IS NULL
                THEN '{REASON_Q4_NO_Q3}'
            END AS standalone_null_reason
        FROM adjusted_in
    ),
    with_neighbors AS (
        SELECT
            c.*,
            n1.standalone_value AS neighbor1_value, n1.available_from AS neighbor1_avail,
            n1.statement_period_end AS neighbor1_pe,
            n2.standalone_value AS neighbor2_value, n2.available_from AS neighbor2_avail,
            n2.statement_period_end AS neighbor2_pe,
            n3.standalone_value AS neighbor3_value, n3.available_from AS neighbor3_avail,
            n3.statement_period_end AS neighbor3_pe,
            n4.standalone_value AS neighbor4_value, n4.statement_period_end AS neighbor4_pe
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
        CASE WHEN date_diff('month', neighbor4_pe, statement_period_end) = 12
             THEN neighbor4_value END AS value_lag_4q,
        CASE
            WHEN {ttm_ok}
            THEN standalone_value + neighbor1_value + neighbor2_value + neighbor3_value
        END AS ttm_value,
        COALESCE({ttm_ok}, FALSE) AS ttm_complete,
        CASE
            WHEN {ttm_ok}
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
    filing_receipt_view: str = DEFAULT_RECEIPT_VIEW,
    pairing_tolerance: float = 0.0,
) -> str:
    """Register a DuckDB view over the v4 SQL (no parquet -- tests/parity)."""
    sql = build_fin_quarterly_metric_vintage_v4_sql(
        vintage_view=vintage_view,
        filing_receipt_view=filing_receipt_view,
        pairing_tolerance=pairing_tolerance,
    )
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def _materialize(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    vintage_view: str,
    filing_receipt_view: str,
    pairing_tolerance: float,
    force: bool,
    table_name: str,
    variant: tuple[tuple[str, object], ...],
) -> str:
    sql = build_fin_quarterly_metric_vintage_v4_sql(
        vintage_view=vintage_view,
        filing_receipt_view=filing_receipt_view,
        pairing_tolerance=pairing_tolerance,
    )
    plan = MartPlan(
        plan_id="single",
        final_sql=sql,
        semantics_version=SEMANTICS_V4,
        metadata=(
            ("fiscal_year_renumber", "receipt_annual_report_period_ends"),
            ("outlier_abs_limit", OUTLIER_ABS_LIMIT),
            ("ttm_requires_consecutive_period_ends", True),
            ("vintage_view", vintage_view),
            ("filing_receipt_view", filing_receipt_view),
            *variant,
        ),
    )
    materialize(con, config, table_name, sql, force=force, plan=plan)
    return register_mart_view(con, config, table_name)


def materialize_fin_quarterly_metric_vintage_v4(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    vintage_view: str = DEFAULT_VINTAGE_VIEW,
    filing_receipt_view: str = DEFAULT_RECEIPT_VIEW,
    pairing_tolerance: float = 0.0,
    force: bool = False,
    table_name: str = FQMV_V4_TABLE,
) -> str:
    """Build + register the v4 quarterly mart as parquet under a new name.

    ``vintage_view`` (default: the v3 fact) and ``filing_receipt_view`` must already be
    registered on ``con``. The cache metadata records ``semantics_version="v4"``. The default
    name never collides with the v1/v2/v3 marts, so nothing existing is overwritten.
    """
    return _materialize(
        con,
        config,
        vintage_view=vintage_view,
        filing_receipt_view=filing_receipt_view,
        pairing_tolerance=pairing_tolerance,
        force=force,
        table_name=table_name,
        variant=(),
    )


def materialize_fin_quarterly_metric_vintage_dl_v4(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    vintage_view: str = DEFAULT_DL_VINTAGE_VIEW,
    filing_receipt_view: str = DEFAULT_RECEIPT_VIEW,
    pairing_tolerance: float = 0.0,
    force: bool = False,
    table_name: str = FQMV_DL_V4_TABLE,
) -> str:
    """The v4 mart over the delisted-keeping v3 fact (``stock_metric_vintage_fact_dl_v3``).

    Build that fact first with
    ``metric_vintages_delisted.materialize_stock_metric_vintage_fact_dl(semantics="v3")``. Only
    the corp relation of the fact differs from :func:`materialize_fin_quarterly_metric_vintage_v4`
    (the v4 statement is the same); the cache metadata records the variant.
    """
    return _materialize(
        con,
        config,
        vintage_view=vintage_view,
        filing_receipt_view=filing_receipt_view,
        pairing_tolerance=pairing_tolerance,
        force=force,
        table_name=table_name,
        variant=(
            ("variant", "keep_delisted_corps"),
            ("corp_market_sources", "dart_corp_master,stock_master,daily_ohlcv"),
        ),
    )
