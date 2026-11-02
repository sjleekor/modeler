"""``fin_quarterly_metric_vintage_dlr_v4p`` -- the P6 ``_dlr`` input, three fixes, point in time.

Decided 2026-11-02 (05_prep_a_financial_target.md, "P6′ 범위 결정" S1-S4). P6 reads
``fin_quarterly_metric_vintage_dlr``: the delisted-keeping fact, rule R1, quarterly statement
``v1``. P6′ keeps the model, the features and rule R1 and changes only this mart:

* **KR-FB-0002** -- the fact is ``stock_metric_vintage_fact_dl_v3`` (3-month XBRL income
  contexts, delisted corps kept).
* **KR-FB-0004** -- a duration metric's standalone value with ``abs > 1e14`` KRW is NULL with a
  reason, and so is a direct-interim Q4 whose Q1..Q3 value is (the r2 rule of
  :mod:`financial_quarters_v4`).
* **KR-FB-0003** -- fiscal renumbering, **point in time**. :mod:`financial_quarters_v4` (r2)
  places a filing with every annual-report name of the receipt list, including names filed
  after it: a later fiscal-year-end change turns the quarters before it into a dropped
  transition year, so "this company will change its year end" reaches the past as missing
  rows. Here a filing is placed only with the annual-report names (``사업보고서 (YYYY.MM)``)
  whose receipt date is on or before the filing's own ``disclosed_date`` (the receipt date of
  ``rcept_no``, the same date ``available_from`` is computed from). Before the first such
  name the code-based position ``year(period_end + 3 * (4 - quarter))`` is used. The
  company's current year end (``dart_corp_master``) is not read: it is today's.

Rule R1 (``metric_vintages_delisted._R1_REPLACEMENTS``): a direct-interim Q2/Q3 whose quarterly
and cumulative values disagree keeps the reported quarterly value (``standalone_source_conflict``
stays TRUE), and ``ttm_includes_conflict`` marks a TTM with such a quarter. Because no Q2/Q3 is
NULLed for a conflict, the Q4 conflict block of r2 (KR-FB-0005) has no condition to act on and
is not applied.

**Fiscal key.** With names known at the time, a filing inside a coming transition year is
placed by the old year end: a March company's 2022-06 report is Q1 of the year ending
2023-03. If the company moves to December, its 2023-03 report is Q1 of the year ending
2023-12 -- both "2023 Q1". Keyed by year number the two would share one slot and the
quarterly statement's first-vintage tie-break would keep the older row. So the statement's
``bsns_year`` is fed the *assumed fiscal year end* as ``yyyymm`` (202303 vs. 202312), and the
output gets ``bsns_year = yyyymm // 100`` back plus the ``fiscal_year_end`` date. ``seq_key``
is the calendar quarter of ``statement_period_end`` (``year * 4 + (month - 1) // 3 + 1``),
which equals ``bsns_year * 4 + quarter`` for a December company and keeps the old-numbering
and new-numbering rows of a changing company in time order. The quarterly statement's own
``seq_key``/TTM/lag columns (computed from the ``yyyymm`` key) are discarded and recomputed:
TTM and ``value_lag_4q`` only over period ends 3, 6, 9 (12) months apart, as in r2.

A December-end company is placed exactly as in r2. The transition year itself (an annual
report of a non-12-month year) is still dropped as soon as its own name is on the list: at its
filing the year is known not to be 12 months. Three rules beyond r2, all reading only the
filing's own report name or earlier filings:

* a kept position must not contradict the filing's own report name either (r2 checks the name
  only when the code disagrees). Counting forward from the last known year end assumes a
  12-month year; a 반기보고서 three months after a year end (a REIT with 6-month years) shows at
  its own filing that it does not hold, and is dropped;
* before the first annual-report name, a filing named 반기보고서 is Q2 whatever its code (moved
  like r2's relabelled rows), and a 분기보고서 under the half-year or annual code is dropped
  (Q1 or Q3 cannot be told);
* a period end has one slot: that of its first vintage (the statement's own winner order). A
  later vintage placed in another slot -- only possible before the first annual-report name,
  when the report code changed between original and revision -- is dropped. Without this two
  rows would share one calendar ``seq_key``.

A non-December company whose year end never changed is placed as in r2 except for filings
before its first annual-report name whose code is wrong in a way the name cannot show
(a ``분기보고서`` stored as Q1 that is Q3): r2 used the later name, which this mart cannot.

Columns are those of ``fin_quarterly_metric_vintage_dlr`` (same names, order and types), then
``standalone_null_reason`` and ``fiscal_year_end``.
:func:`build_pit_fiscal_position_audit_sql` shows the placement of every fact row.
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import MartPlan, materialize, register_mart_view
from modeler.etl.marts import financial_quarters as _fq
from modeler.etl.marts import financial_quarters_v4 as _fq4
from modeler.etl.marts import metric_vintages_delisted as _dl

FQMV_DLR_V4P_TABLE = "fin_quarterly_metric_vintage_dlr_v4p"
SEMANTICS_V4P = "v4p"
DEFAULT_VINTAGE_VIEW = _fq4.DEFAULT_DL_VINTAGE_VIEW
DEFAULT_RECEIPT_VIEW = _fq4.DEFAULT_RECEIPT_VIEW
OUTLIER_ABS_LIMIT = _fq4.OUTLIER_ABS_LIMIT
CONFLICT_RULE = "R1_keep_reported_quarter"
FISCAL_PLACEMENT = "pit_receipt_names"

REASON_OUTLIER = _fq4.REASON_OUTLIER
REASON_Q4_OUTLIER = _fq4.REASON_Q4_OUTLIER
REASON_Q4_NO_Q3 = _fq4.REASON_Q4_NO_Q3

#: Only the two Q2/Q3 replacements of R1 are applied to the statement text; the TTM (and its
#: conflict flag) is recomputed below over the calendar ``seq_key``.
_R1_VALUE_REPLACEMENTS = _dl._R1_REPLACEMENTS[:2]

#: ``fin_quarterly_metric_vintage_dlr`` columns, in order (the P6 input), then the extras.
DLR_COLUMNS = (
    "ticker", "market", "corp_code", "metric_code", "fs_basis", "bsns_year", "quarter",
    "quarter_ordinal", "seq_key", "statement_period_end", "rcept_no", "available_from",
    "metric_kind", "standalone_value", "standalone_source_conflict", "cumulative_derived_value",
    "comparative_q_amount", "value_lag_4q", "ttm_value", "ttm_complete", "ttm_available_from",
    "ttm_includes_conflict",
)  # fmt: skip
EXTRA_COLUMNS = ("standalone_null_reason", "fiscal_year_end")

_RCEPT_DATE = "CASE WHEN {col} ~ '^[0-9]{{14}}$' THEN strptime(left({col}, 8), '%Y%m%d')::DATE END"


#: Placements that reach the mart. ``kept_unpositioned`` (no period end or report code) of r2
#: is not among them: without a period end there is no year end to key by nor a calendar
#: ``seq_key``.
KEPT_POSITIONS = (
    _fq4.POSITION_OK,
    _fq4.POSITION_OK_AFTER_LAST_ANNUAL,
    _fq4.POSITION_RELABELLED,
    _fq4.POSITION_NO_ANNUAL_REFERENCE,
)


def build_pit_fiscal_position_audit_sql(
    *,
    vintage_view: str = DEFAULT_VINTAGE_VIEW,
    filing_receipt_view: str = DEFAULT_RECEIPT_VIEW,
) -> str:
    """Every fact row with its point-in-time fiscal placement.

    The placement rules (``fiscal_position_status``) are those of
    :func:`financial_quarters_v4.build_fiscal_position_audit_sql` plus the report-name rules of
    the module docstring. The set of fiscal year ends differs: per fact row, the annual-report
    names whose receipt date (the date of ``rcept_no``, else ``rcept_dt``) is on or before the
    row's ``disclosed_date``. The one-slot-per-period-end rule is applied later (in the mart's
    fact relation), so a row it drops still shows a kept status here. Adds
    ``report_name_period_end``, ``report_kind``, ``fiscal_period_end``, ``next_fiscal_year_end``,
    ``prev_fiscal_year_end``, ``fiscal_ordinal``, ``fiscal_position_status``,
    ``fiscal_year_end`` (the year end assumed at the time), ``fiscal_key`` (its ``yyyymm``) and
    ``fiscal_reprt_code``.
    """
    rx = _fq4._PERIODIC_NAME_RE
    name_end = (
        f"last_day(make_date("
        f"CAST(regexp_extract(report_nm, '{rx}', 3) AS INTEGER), "
        f"CAST(regexp_extract(report_nm, '{rx}', 4) AS INTEGER), 1))"
    )
    rcept_date = _RCEPT_DATE.format(col="rcept_no")
    expected_kind = "CASE fiscal_ordinal WHEN 4 THEN '사업' WHEN 2 THEN '반기' ELSE '분기' END"
    kind_ok = f"(report_kind IS NULL OR report_kind = {expected_kind})"
    ok, ok_after, relabelled = (
        _fq4.POSITION_OK,
        _fq4.POSITION_OK_AFTER_LAST_ANNUAL,
        _fq4.POSITION_RELABELLED,
    )
    return f"""
    WITH periodic_receipts AS (
        SELECT corp_code, rcept_no,
               regexp_extract(report_nm, '{rx}', 2) AS report_kind,
               {name_end} AS name_period_end,
               COALESCE({rcept_date}, rcept_dt) AS known_from
        FROM {filing_receipt_view}
        WHERE regexp_matches(report_nm, '{rx}')
    ),
    fiscal_year_ends AS (
        -- a year end is known from its first annual-report name (corrections come later)
        SELECT corp_code, name_period_end AS fiscal_year_end, min(known_from) AS known_from
        FROM periodic_receipts
        WHERE report_kind = '사업' AND known_from IS NOT NULL
        GROUP BY corp_code, name_period_end
    ),
    receipt_period AS (
        -- the filing's own report name: known when the filing is
        SELECT rcept_no, max(name_period_end) AS report_name_period_end,
               CASE WHEN count(DISTINCT report_kind) = 1 THEN any_value(report_kind) END
                   AS report_kind
        FROM periodic_receipts
        GROUP BY rcept_no
    ),
    src AS (
        SELECT f.*, rp.report_name_period_end, rp.report_kind,
               {_fq4._CODE_ORDINAL} AS code_ordinal,
               CASE WHEN f.period_end_source = 'calendar_fallback'
                         AND rp.report_name_period_end IS NOT NULL
                    THEN rp.report_name_period_end
                    ELSE f.statement_period_end END AS fiscal_period_end
        FROM {vintage_view} f
        LEFT JOIN receipt_period rp ON rp.rcept_no = f.rcept_no
    ),
    bundles AS (
        SELECT DISTINCT corp_code, fiscal_period_end, disclosed_date
        FROM src
        WHERE fiscal_period_end IS NOT NULL AND disclosed_date IS NOT NULL
    ),
    bundle_reference AS (
        -- the year ends known on the filing's own disclosure date, nearest on each side
        SELECT b.corp_code, b.fiscal_period_end, b.disclosed_date,
               min(y.fiscal_year_end) FILTER (WHERE y.fiscal_year_end >= b.fiscal_period_end)
                   AS next_fiscal_year_end,
               max(y.fiscal_year_end) FILTER (WHERE y.fiscal_year_end < b.fiscal_period_end)
                   AS prev_fiscal_year_end
        FROM bundles b
        LEFT JOIN fiscal_year_ends y
          ON y.corp_code = b.corp_code AND y.known_from <= b.disclosed_date
        GROUP BY b.corp_code, b.fiscal_period_end, b.disclosed_date
    ),
    with_both AS (
        SELECT s.*, br.next_fiscal_year_end, br.prev_fiscal_year_end
        FROM src s
        LEFT JOIN bundle_reference br
          ON br.corp_code = s.corp_code AND br.fiscal_period_end = s.fiscal_period_end
         AND br.disclosed_date = s.disclosed_date
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
                -- no year end known yet: the report code, unless the filing's own name is a
                -- half-year report (a 분기보고서 name cannot tell Q1 from Q3)
                WHEN report_kind = '반기' THEN 2
                ELSE code_ordinal
            END AS fiscal_ordinal,
            -- the year end this filing belongs to, as known at its disclosure
            CASE
                WHEN next_fiscal_year_end IS NOT NULL
                THEN last_day(next_fiscal_year_end - to_months(12 * (months_to_next // 12)))
                WHEN prev_fiscal_year_end IS NOT NULL
                THEN last_day(prev_fiscal_year_end
                              + to_months(12 * ((months_from_prev + 11) // 12)))
            END AS reference_year_end
        FROM measured
    ),
    placed_end AS (
        SELECT * EXCLUDE (reference_year_end),
            CASE
                WHEN next_fiscal_year_end IS NOT NULL OR prev_fiscal_year_end IS NOT NULL
                THEN reference_year_end
                -- the code-based position of r2 (``kept_no_annual_reference``)
                ELSE last_day(fiscal_period_end + to_months(3 * (4 - fiscal_ordinal)))
            END AS placed_year_end
        FROM placed
    ),
    classified AS (
        SELECT *,
            CASE
                WHEN fiscal_period_end IS NULL OR code_ordinal IS NULL
                THEN '{_fq4.POSITION_UNPOSITIONED}'
                WHEN fiscal_ordinal IS NULL
                THEN '{_fq4.POSITION_IRREGULAR}'
                WHEN next_fiscal_year_end IS NOT NULL AND prev_fiscal_year_end IS NOT NULL
                 AND fiscal_year_months % 12 <> 0
                THEN '{_fq4.POSITION_TRANSITION_YEAR}'
                -- Beyond r2: the filing's own report name must not contradict the position
                -- either. Counting forward from the last known year end assumes a 12-month
                -- year; a filing named 반기보고서 three months after a year end (a 6-month-year
                -- REIT) shows at its own filing that the assumption does not hold.
                WHEN fiscal_ordinal = code_ordinal AND {kind_ok}
                 AND next_fiscal_year_end IS NOT NULL
                THEN '{ok}'
                WHEN fiscal_ordinal = code_ordinal AND {kind_ok}
                 AND prev_fiscal_year_end IS NOT NULL
                THEN '{ok_after}'
                WHEN fiscal_ordinal = code_ordinal AND {kind_ok}
                THEN '{_fq4.POSITION_NO_ANNUAL_REFERENCE}'
                WHEN report_kind IS NOT NULL AND report_kind = {expected_kind}
                 AND ((fiscal_ordinal < 4 AND code_ordinal < 4)
                      OR metric_code NOT IN {_fq._in_list(_fq.DIRECT_INTERIM_METRICS)})
                THEN '{relabelled}'
                ELSE '{_fq4.POSITION_CODE_MISMATCH}'
            END AS fiscal_position_status
        FROM placed_end
    ),
    keyed AS (
        SELECT * EXCLUDE (code_ordinal, months_to_next, months_from_prev, fiscal_year_months,
                          placed_year_end),
            CASE
                WHEN fiscal_position_status IN {_fq4._in_sql(KEPT_POSITIONS)}
                THEN placed_year_end
            END AS fiscal_year_end,
            CASE WHEN fiscal_position_status = '{relabelled}'
                 THEN CASE fiscal_ordinal WHEN 1 THEN '11013' WHEN 2 THEN '11012'
                                          WHEN 3 THEN '11014' ELSE '11011' END
                 ELSE reprt_code END AS fiscal_reprt_code
        FROM classified
    )
    SELECT *,
        CAST(year(fiscal_year_end) * 100 + month(fiscal_year_end) AS INTEGER) AS fiscal_key
    FROM keyed
    """


_AUDIT_EXTRA = (
    "report_name_period_end, report_kind, fiscal_period_end, next_fiscal_year_end, "
    "prev_fiscal_year_end, fiscal_ordinal, fiscal_position_status, fiscal_year_end, "
    "fiscal_key, fiscal_reprt_code"
)


def _pit_fact_relation(vintage_view: str, filing_receipt_view: str) -> str:
    """The fact keyed by ``fiscal_key`` (as ``bsns_year``), dropped rows removed (a FROM item)."""
    audit = build_pit_fiscal_position_audit_sql(
        vintage_view=vintage_view, filing_receipt_view=filing_receipt_view
    )
    return (
        f"(SELECT * EXCLUDE ({_AUDIT_EXTRA}) "
        "REPLACE (fiscal_key AS bsns_year, fiscal_period_end AS statement_period_end, "
        "fiscal_reprt_code AS reprt_code) "
        "FROM ("
        f"SELECT * FROM ({audit}) AS fiscal_audit "
        f"WHERE fiscal_position_status IN {_fq4._in_sql(KEPT_POSITIONS)} "
        # One receipt can reach the same slot twice (stored under two codes, one moved here):
        # keep the row that was already there, then the non-fallback one (as in r2).
        "QUALIFY ROW_NUMBER() OVER (PARTITION BY ticker, metric_code, fs_basis, "
        "fiscal_key, fiscal_reprt_code, rcept_no "
        f"ORDER BY (fiscal_position_status = '{_fq4.POSITION_RELABELLED}') ASC, "
        "(period_end_source = 'calendar_fallback') ASC, source_key ASC) = 1"
        ") AS slot_rows "
        # A period end has one slot: the one its first vintage (the statement's own winner
        # order) was placed in at its disclosure. A later vintage placed elsewhere -- a code
        # that changed before the first annual-report name -- is dropped.
        "QUALIFY fiscal_key = first_value(fiscal_key) OVER ("
        "PARTITION BY ticker, metric_code, fs_basis, fiscal_period_end "
        "ORDER BY COALESCE(is_revision, FALSE) ASC, available_from ASC, rcept_no ASC, "
        "source_key ASC)"
        ") AS fiscal_fact"
    )


def _r1_statement(fact_relation: str, pairing_tolerance: float) -> str:
    """The unchanged ``v2`` quarterly statement with R1's two Q2/Q3 value replacements."""
    sql = _fq.build_fin_quarterly_metric_vintage_sql(
        vintage_view=fact_relation,
        pairing_tolerance=pairing_tolerance,
        semantics=_fq.SEMANTICS_V2,
    )
    for old, new in _R1_VALUE_REPLACEMENTS:
        hits = sql.count(old)
        if hits != 1:
            raise _dl.OriginalSqlDriftError(
                f"{FQMV_DLR_V4P_TABLE}: expected the original text {old!r} exactly once, "
                f"found {hits}x; financial_quarters changed, so rule R1 must be rewritten"
            )
        sql = sql.replace(old, new)
    return sql


def build_fin_quarterly_metric_vintage_dlr_v4p_sql(
    *,
    vintage_view: str = DEFAULT_VINTAGE_VIEW,
    filing_receipt_view: str = DEFAULT_RECEIPT_VIEW,
    pairing_tolerance: float = 0.0,
    outlier_block: bool = True,
) -> str:
    """SQL producing ``fin_quarterly_metric_vintage_dlr_v4p`` (see the module docstring).

    ``outlier_block=False`` leaves out the KR-FB-0004 block (reason column still filled for the
    other reasons); it exists to measure the block's effect, not for P6′.
    """
    base_sql = _r1_statement(_pit_fact_relation(vintage_view, filing_receipt_view),
                             pairing_tolerance)  # fmt: skip
    duration = _fq._in_list(_fq.DURATION_METRICS)
    lim = OUTLIER_ABS_LIMIT

    def over(col: str) -> str:
        if not outlier_block:
            return "FALSE"
        return f"({col} IS NOT NULL AND abs({col}) > {lim})"

    def step(n: int) -> str:
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
    rekeyed AS (
        SELECT
            b.* EXCLUDE (bsns_year, seq_key, value_lag_4q, ttm_value, ttm_complete,
                         ttm_available_from),
            b.bsns_year AS fiscal_key,
            CAST(b.bsns_year // 100 AS INTEGER) AS bsns_year,
            last_day(make_date(b.bsns_year // 100, b.bsns_year % 100, 1)) AS fiscal_year_end,
            CAST(year(b.statement_period_end) * 4
                 + (month(b.statement_period_end) - 1) // 3 + 1 AS INTEGER) AS seq_key,
            -- Q4's own Q1..Q3: the same fiscal key (the quarters its difference subtracts)
            bool_or(CASE WHEN b.quarter_ordinal < 4 THEN {over("b.standalone_value")} END)
                OVER (PARTITION BY b.ticker, b.metric_code, b.fs_basis, b.bsns_year)
                AS prior_outlier
        FROM base b
    ),
    adjusted AS MATERIALIZED (
        SELECT
            ticker, market, corp_code, metric_code, fs_basis, bsns_year, quarter,
            quarter_ordinal, seq_key, statement_period_end, rcept_no, available_from,
            metric_kind,
            CASE
                WHEN metric_kind = 'direct_interim' AND quarter_ordinal = 4
                 AND COALESCE(prior_outlier, FALSE)
                THEN NULL
                WHEN metric_code IN {duration} AND {over("standalone_value")} THEN NULL
                ELSE standalone_value
            END AS standalone_value,
            standalone_source_conflict, cumulative_derived_value, comparative_q_amount,
            CASE
                WHEN metric_kind = 'direct_interim' AND quarter_ordinal = 4
                 AND COALESCE(prior_outlier, FALSE)
                THEN '{REASON_Q4_OUTLIER}'
                WHEN metric_code IN {duration} AND {over("standalone_value")}
                THEN '{REASON_OUTLIER}'
                WHEN metric_kind IN ('cumulative_reported', 'weighted_share')
                 AND quarter_ordinal = 4 AND standalone_value IS NULL
                THEN '{REASON_Q4_NO_Q3}'
            END AS standalone_null_reason,
            fiscal_year_end
        FROM rekeyed
    ),
    with_neighbors AS (
        SELECT
            c.*,
            n1.standalone_value AS neighbor1_value, n1.available_from AS neighbor1_avail,
            n1.statement_period_end AS neighbor1_pe,
            n1.standalone_source_conflict AS neighbor1_conflict,
            n2.standalone_value AS neighbor2_value, n2.available_from AS neighbor2_avail,
            n2.statement_period_end AS neighbor2_pe,
            n2.standalone_source_conflict AS neighbor2_conflict,
            n3.standalone_value AS neighbor3_value, n3.available_from AS neighbor3_avail,
            n3.statement_period_end AS neighbor3_pe,
            n3.standalone_source_conflict AS neighbor3_conflict,
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
        CASE
            WHEN {ttm_ok}
            THEN COALESCE(standalone_source_conflict, FALSE)
                 OR COALESCE(neighbor1_conflict, FALSE)
                 OR COALESCE(neighbor2_conflict, FALSE)
                 OR COALESCE(neighbor3_conflict, FALSE)
        END AS ttm_includes_conflict,
        standalone_null_reason,
        fiscal_year_end
    FROM with_neighbors
    """


def register_fin_quarterly_metric_vintage_dlr_v4p_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = FQMV_DLR_V4P_TABLE,
    vintage_view: str = DEFAULT_VINTAGE_VIEW,
    filing_receipt_view: str = DEFAULT_RECEIPT_VIEW,
    pairing_tolerance: float = 0.0,
) -> str:
    """Register a DuckDB view over the v4p SQL (no parquet -- tests/parity)."""
    sql = build_fin_quarterly_metric_vintage_dlr_v4p_sql(
        vintage_view=vintage_view,
        filing_receipt_view=filing_receipt_view,
        pairing_tolerance=pairing_tolerance,
    )
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name


def materialize_fin_quarterly_metric_vintage_dlr_v4p(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    vintage_view: str = DEFAULT_VINTAGE_VIEW,
    filing_receipt_view: str = DEFAULT_RECEIPT_VIEW,
    pairing_tolerance: float = 0.0,
    force: bool = False,
    table_name: str = FQMV_DLR_V4P_TABLE,
) -> str:
    """Build + register the P6′ quarterly mart as parquet under its own name.

    ``vintage_view`` (default: ``stock_metric_vintage_fact_dl_v3``) and ``filing_receipt_view``
    must already be registered on ``con``. Nothing existing is overwritten by the default name.
    """
    sql = build_fin_quarterly_metric_vintage_dlr_v4p_sql(
        vintage_view=vintage_view,
        filing_receipt_view=filing_receipt_view,
        pairing_tolerance=pairing_tolerance,
    )
    plan = MartPlan(
        plan_id="single",
        final_sql=sql,
        semantics_version=SEMANTICS_V4P,
        metadata=(
            ("variant", "keep_delisted_corps"),
            ("corp_market_sources", "dart_corp_master,stock_master,daily_ohlcv"),
            ("conflict_rule", CONFLICT_RULE),
            ("fiscal_placement", FISCAL_PLACEMENT),
            ("quarterly_statement_semantics", _fq.SEMANTICS_V2),
            ("outlier_abs_limit", OUTLIER_ABS_LIMIT),
            ("ttm_requires_consecutive_period_ends", True),
            ("seq_key", "calendar_quarter_of_statement_period_end"),
            ("vintage_view", vintage_view),
            ("filing_receipt_view", filing_receipt_view),
        ),
    )
    materialize(con, config, table_name, sql, force=force, plan=plan)
    return register_mart_view(con, config, table_name)
