"""§7.1 raw-side quality diagnostics for Phase B (B-10 Stage 2).

Two of the seven ``*_quality``/``*_coverage`` artifacts the run directory
contract lists. Both sit directly on raw, before any feature mart, and answer
"is the source good enough to build on" rather than "is the feature any good":

``filing_receipt_quality``   per receipt year — how much of the disclosure
                             receipt history landed, how much of it is a
                             correction rather than an original filing, and how
                             much was corrected later. B-1 step 6 needs the
                             correction count to size the receipt-targeted XBRL
                             backfill; SUE needs it because a revision must
                             never be counted as a new event (§3.5).

``capital_change_quality``   per (vintage year, report code) — placeholder vs
                             real event rows, unclassified ``isu_dcrs_stle``
                             coverage, and, for annual vintages, how much this
                             vintage disagrees with the newest one. The last
                             part reuses the §4.4.1 probe so vintage duplication
                             stays a standing measurement instead of a one-off.

Both are pure SQL over registered views, so the phase B runner can materialize
them the same way it materializes everything else in the run directory.
"""

from __future__ import annotations

import duckdb

from modeler.etl.features.event_scan import ANNUAL_REPRT_CODE, _classification_case
from modeler.etl.vintage_probe import build_vintage_window_diff_sql

FILING_RECEIPT_QUALITY_TABLE = "filing_receipt_quality"
CAPITAL_CHANGE_QUALITY_TABLE = "capital_change_quality"

# OpenDART list.json prefixes a corrected re-filing's report_nm with a bracketed
# marker. These are the ones seen in the collected history; anything else
# bracketed still counts as "amended" via the generic pattern below, so a new
# marker inflates no ratio silently.
AMENDMENT_MARKERS: tuple[str, ...] = ("기재정정", "첨부정정", "첨부추가", "변경등록")

# list.json's ``rm`` column is a concatenation of single-character flags; '정'
# means a correction to THIS receipt was filed later. Distinct from the markers
# above, which mean this receipt IS the correction.
LATER_CORRECTED_FLAG = "정"

PERIODIC_REPORT_NAMES: tuple[str, ...] = ("사업보고서", "반기보고서", "분기보고서")


def _any_contains(column: str, needles: tuple[str, ...]) -> str:
    return "(" + " OR ".join(f"{column} LIKE '%{needle}%'" for needle in needles) + ")"


def build_filing_receipt_quality_sql(
    *,
    filing_receipt_view: str = "dart_filing_receipt_raw",
) -> str:
    """One row per receipt year over ``dart_filing_receipt_raw``.

    Years with no rows simply do not appear — a gap in this table is how a
    missing backfill year shows up, so it must not be filled with zeros.
    """
    is_amendment = _any_contains("report_nm", AMENDMENT_MARKERS)
    is_periodic = _any_contains("report_nm", PERIODIC_REPORT_NAMES)
    return f"""
    SELECT
        CAST(EXTRACT(year FROM rcept_dt) AS INTEGER) AS receipt_year,
        COUNT(*) AS receipts,
        COUNT(DISTINCT corp_code) AS corps,
        COUNT(DISTINCT NULLIF(ticker, '')) AS tickers,
        MIN(rcept_dt) AS min_rcept_dt,
        MAX(rcept_dt) AS max_rcept_dt,
        COUNT(*) FILTER (WHERE {is_periodic}) AS periodic_report_receipts,
        -- B-1 step 6 sizes the receipt-targeted original-XBRL backfill off this
        -- one: a corrected periodic report is a filing whose original value the
        -- lake does not have, not just any amended disclosure.
        COUNT(*) FILTER (WHERE {is_periodic} AND {is_amendment})
            AS periodic_amendment_receipts,
        COUNT(*) FILTER (WHERE {is_amendment}) AS amendment_receipts,
        COUNT(*) FILTER (WHERE {is_amendment})::DOUBLE
            / NULLIF(COUNT(*), 0) AS amendment_ratio,
        COUNT(*) FILTER (WHERE rm LIKE '%{LATER_CORRECTED_FLAG}%') AS later_corrected_receipts,
        COUNT(*) FILTER (WHERE rm LIKE '%{LATER_CORRECTED_FLAG}%')::DOUBLE
            / NULLIF(COUNT(*), 0) AS later_corrected_ratio,
        COUNT(*) FILTER (WHERE ticker IS NULL OR ticker = '') AS receipts_without_ticker,
        COUNT(*) FILTER (WHERE rcept_no IS NULL OR NOT rcept_no ~ '^[0-9]{{14}}$')
            AS receipts_with_malformed_rcept_no
    FROM {filing_receipt_view}
    WHERE rcept_dt IS NOT NULL
    GROUP BY 1
    ORDER BY 1
    """


def build_capital_change_quality_sql(
    *,
    capital_change_view: str = "dart_capital_change_raw",
    share_count_view: str = "dart_share_count_raw",
    corp_view: str = "dart_corp_master",
) -> str:
    """One row per (vintage year, report code) over ``dart_capital_change_raw``.

    ``compared_windows``/``feature_changing_windows`` are only populated for
    annual vintages that have a newer annual vintage to be compared against —
    NULL elsewhere, which is the honest representation of "not measurable here"
    rather than "measured as zero".
    """
    window_diff = build_vintage_window_diff_sql(
        capital_change_view=capital_change_view,
        share_count_view=share_count_view,
        corp_view=corp_view,
    )
    classification = _classification_case("cc")
    return f"""
    WITH classified AS (
        SELECT
            cc.bsns_year, cc.reprt_code, cc.ticker, cc.isu_dcrs_de, cc.isu_dcrs_qy,
            CASE WHEN cc.isu_dcrs_de IS NULL THEN NULL ELSE {classification} END AS action_class
        FROM {capital_change_view} cc
    ),
    raw_quality AS (
        SELECT
            bsns_year,
            reprt_code,
            (reprt_code = '{ANNUAL_REPRT_CODE}') AS is_annual,
            COUNT(*) AS rows,
            COUNT(DISTINCT ticker) AS tickers,
            COUNT(*) FILTER (WHERE isu_dcrs_de IS NULL) AS placeholder_rows,
            COUNT(*) FILTER (WHERE isu_dcrs_de IS NULL)::DOUBLE
                / NULLIF(COUNT(*), 0) AS placeholder_ratio,
            COUNT(*) FILTER (WHERE isu_dcrs_de IS NOT NULL) AS event_rows,
            COUNT(DISTINCT ticker) FILTER (WHERE isu_dcrs_de IS NOT NULL) AS tickers_with_events,
            COUNT(*) FILTER (WHERE action_class = 'unclassified') AS unclassified_rows,
            COUNT(*) FILTER (WHERE action_class = 'unclassified')::DOUBLE
                / NULLIF(COUNT(*) FILTER (WHERE isu_dcrs_de IS NOT NULL), 0)
                AS unclassified_ratio,
            COUNT(*) FILTER (
                WHERE action_class IN ('economic_increase', 'economic_decrease')
            ) AS economic_rows,
            COUNT(*) FILTER (
                WHERE action_class IN ('mechanical_increase', 'mechanical_decrease')
            ) AS mechanical_rows,
            COUNT(*) FILTER (WHERE isu_dcrs_de IS NOT NULL AND isu_dcrs_qy IS NULL)
                AS event_rows_without_quantity,
            MIN(isu_dcrs_de) AS min_isu_dcrs_de,
            MAX(isu_dcrs_de) AS max_isu_dcrs_de
        FROM classified
        GROUP BY 1, 2
    ),
    vintage_agreement AS (
        SELECT
            old_bsns_year AS bsns_year,
            MAX(vintage_distance_years) AS vintage_distance_years,
            COUNT(*) AS compared_windows,
            COUNT(*) FILTER (WHERE feature_changing) AS feature_changing_windows,
            COUNT(*) FILTER (WHERE feature_changing)::DOUBLE
                / NULLIF(COUNT(*), 0) AS feature_changing_rate
        FROM ({window_diff})
        GROUP BY 1
    )
    SELECT
        q.*,
        v.vintage_distance_years,
        v.compared_windows,
        v.feature_changing_windows,
        v.feature_changing_rate
    FROM raw_quality q
    LEFT JOIN vintage_agreement v ON v.bsns_year = q.bsns_year AND q.is_annual
    ORDER BY q.bsns_year, q.reprt_code
    """


def register_filing_receipt_quality_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = FILING_RECEIPT_QUALITY_TABLE,
    **views: str,
) -> str:
    con.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS {build_filing_receipt_quality_sql(**views)}"
    )
    return view_name


def register_capital_change_quality_view(
    con: duckdb.DuckDBPyConnection,
    *,
    view_name: str = CAPITAL_CHANGE_QUALITY_TABLE,
    **views: str,
) -> str:
    con.execute(
        f"CREATE OR REPLACE VIEW {view_name} AS {build_capital_change_quality_sql(**views)}"
    )
    return view_name
