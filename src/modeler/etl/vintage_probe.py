"""§4.4.1 vintage distance probe — how much do capital-change vintages disagree?

``irdsSttus`` reprints the whole since-listing capital-action history in every
annual report, so ``dart_capital_change_raw`` holds one copy of each real event
per report vintage. §4.4.1 fixes the dedup rule (one vintage per ticker, used
whole) but leaves the *choice* of vintage to measurement:

    latest_vintage  the newest report, used for every filing position
    strict_pit      per position, the newest report already disclosed then

This module computes the two pre-registered metrics that decide between them.

Metric 1 — ``feature-changing 불일치율``. Rather than matching events between
vintages one by one (a corrected date or quantity makes "the same event" two
different rows, and any matching rule for that is a judgement call), this
compares what the feature actually consumes: the per-class quantity sums inside
each trailing-year window. A date correction that stays inside its window
changes nothing and is correctly counted as agreement; one that crosses a
window boundary shows up as disagreement in two windows, which is exactly its
effect on ``ev_net_share_issuance_yoy``.

Metric 2 — identity pass rate. The share of filing positions whose
``기초 + 증가 - 감소 = 기말`` identity holds, and so produce a non-NULL feature,
under each policy. A policy that never looks ahead but blanks the whole panel
has not produced a usable candidate.

Comparison scope: an older report cannot know about events after its own
settlement date, so every comparison is cut at the older vintage's ``stlm_dt``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import duckdb

from modeler.etl.features.event_scan import (
    ANNUAL_REPRT_CODE,
    VINTAGE_POLICIES,
    _classification_case,
    build_issuance_sql,
    register_event_scan_calendar,
)

# The five class sums build_issuance_sql aggregates per window. Two vintages
# agree on a window exactly when all five (and the event count) match.
_CLASS_SUMS = (
    "economic_increase",
    "mechanical_increase",
    "economic_decrease",
    "mechanical_decrease",
    "unclassified",
)


def _side_sums(side: str, rcept_col: str) -> str:
    parts = [
        f"COALESCE(SUM(CASE WHEN e.vintage_rcept_no = cw.{rcept_col} "
        f"AND e.action_class = '{cls}' THEN e.isu_dcrs_qy END), 0) AS {side}_{cls}"
        for cls in _CLASS_SUMS
    ]
    parts.append(
        f"COUNT(*) FILTER (WHERE e.vintage_rcept_no = cw.{rcept_col}) AS {side}_event_count"
    )
    return ",\n            ".join(parts)


def _window_agreement_predicate() -> str:
    checks = [
        f"old_{cls} IS DISTINCT FROM new_{cls}" for cls in (*_CLASS_SUMS, "event_count")
    ]
    return " OR ".join(checks)


def build_vintage_window_diff_sql(
    *,
    capital_change_view: str = "dart_capital_change_raw",
    share_count_view: str = "dart_share_count_raw",
    corp_view: str = "dart_corp_master",
) -> str:
    """One row per (ticker, older vintage, trailing-year window) comparison.

    ``feature_changing`` is TRUE when the older report and the newest report
    disagree on any class sum the issuance feature consumes for that window.
    """
    return f"""
    WITH corp AS (
        SELECT ticker FROM {corp_view}
        WHERE ticker IS NOT NULL AND ticker <> '' AND market IS NOT NULL
    ),
    annual_positions AS (
        SELECT DISTINCT s.ticker, s.bsns_year, s.stlm_dt
        FROM {share_count_view} s
        JOIN corp c ON c.ticker = s.ticker
        WHERE s.reprt_code = '{ANNUAL_REPRT_CODE}' AND s.se = '합계' AND s.stlm_dt IS NOT NULL
    ),
    windows AS (
        SELECT
            cur.ticker, cur.bsns_year AS window_bsns_year,
            prev.stlm_dt AS window_start, cur.stlm_dt AS window_end
        FROM annual_positions cur
        JOIN annual_positions prev
          ON prev.ticker = cur.ticker AND prev.bsns_year = cur.bsns_year - 1
    ),
    vintages AS (
        SELECT DISTINCT
            cc.ticker,
            cc.bsns_year AS vintage_bsns_year,
            cc.rcept_no AS vintage_rcept_no
        FROM {capital_change_view} cc
        JOIN corp c ON c.ticker = cc.ticker
        WHERE cc.reprt_code = '{ANNUAL_REPRT_CODE}'
    ),
    vintage_cut AS (
        SELECT v.*, ap.stlm_dt AS cut_date
        FROM vintages v
        LEFT JOIN annual_positions ap
          ON ap.ticker = v.ticker AND ap.bsns_year = v.vintage_bsns_year
    ),
    newest AS (
        SELECT ticker, vintage_bsns_year, vintage_rcept_no
        FROM vintage_cut
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY ticker
            ORDER BY vintage_bsns_year DESC, vintage_rcept_no DESC
        ) = 1
    ),
    events AS (
        SELECT
            cc.ticker,
            cc.rcept_no AS vintage_rcept_no,
            cc.isu_dcrs_de,
            cc.isu_dcrs_qy,
            {_classification_case("cc")} AS action_class
        FROM {capital_change_view} cc
        JOIN corp c ON c.ticker = cc.ticker
        WHERE cc.reprt_code = '{ANNUAL_REPRT_CODE}' AND cc.isu_dcrs_de IS NOT NULL
    ),
    pairs AS (
        SELECT
            o.ticker,
            o.vintage_bsns_year AS old_bsns_year, o.vintage_rcept_no AS old_rcept_no,
            n.vintage_bsns_year AS new_bsns_year, n.vintage_rcept_no AS new_rcept_no,
            o.cut_date
        FROM vintage_cut o
        JOIN newest n ON n.ticker = o.ticker
        WHERE o.vintage_bsns_year < n.vintage_bsns_year AND o.cut_date IS NOT NULL
    ),
    comparable_windows AS (
        SELECT p.*, w.window_bsns_year, w.window_start, w.window_end
        FROM pairs p
        JOIN windows w ON w.ticker = p.ticker AND w.window_end <= p.cut_date
    ),
    agg AS (
        SELECT
            cw.ticker, cw.old_bsns_year, cw.new_bsns_year, cw.cut_date,
            cw.window_bsns_year, cw.window_start, cw.window_end,
            {_side_sums("old", "old_rcept_no")},
            {_side_sums("new", "new_rcept_no")}
        FROM comparable_windows cw
        LEFT JOIN events e
          ON e.ticker = cw.ticker
         AND e.vintage_rcept_no IN (cw.old_rcept_no, cw.new_rcept_no)
         AND e.isu_dcrs_de > cw.window_start AND e.isu_dcrs_de <= cw.window_end
        GROUP BY
            cw.ticker, cw.old_bsns_year, cw.new_bsns_year, cw.cut_date,
            cw.window_bsns_year, cw.window_start, cw.window_end,
            cw.old_rcept_no, cw.new_rcept_no
    )
    SELECT
        *,
        new_bsns_year - old_bsns_year AS vintage_distance_years,
        ({_window_agreement_predicate()}) AS feature_changing
    FROM agg
    """


def build_vintage_diff_summary_sql(**views: str) -> str:
    """Metric 1, aggregated by vintage distance — the probe's headline table."""
    detail = build_vintage_window_diff_sql(**views)
    return f"""
    SELECT
        vintage_distance_years,
        COUNT(DISTINCT ticker) AS tickers,
        COUNT(*) AS compared_windows,
        COUNT(*) FILTER (WHERE feature_changing) AS changed_windows,
        COUNT(*) FILTER (WHERE feature_changing)::DOUBLE
            / NULLIF(COUNT(*), 0) AS feature_changing_rate
    FROM ({detail})
    GROUP BY 1
    ORDER BY 1
    """


def build_vintage_row_diff_sql(
    *,
    capital_change_view: str = "dart_capital_change_raw",
    share_count_view: str = "dart_share_count_raw",
    corp_view: str = "dart_corp_master",
) -> str:
    """Row-level colour: older-vintage events with no identical row in the newest.

    Deliberately not a bijective matching — it answers "is this exact
    (date, reason, kind, quantity) still there", which is enough to tell an
    append-only history apart from one that gets silently rewritten.
    """
    window_sql = build_vintage_window_diff_sql(
        capital_change_view=capital_change_view,
        share_count_view=share_count_view,
        corp_view=corp_view,
    )
    return f"""
    WITH pairs AS (
        SELECT DISTINCT ticker, old_bsns_year, new_bsns_year, cut_date,
               vintage_distance_years
        FROM ({window_sql})
    ),
    vintage_rows AS (
        SELECT
            cc.ticker, cc.bsns_year AS vintage_bsns_year, cc.rcept_no AS vintage_rcept_no,
            cc.isu_dcrs_de, cc.isu_dcrs_stle, cc.isu_dcrs_stock_knd, cc.isu_dcrs_qy
        FROM {capital_change_view} cc
        WHERE cc.reprt_code = '{ANNUAL_REPRT_CODE}' AND cc.isu_dcrs_de IS NOT NULL
    )
    SELECT
        p.vintage_distance_years,
        COUNT(*) AS old_events,
        COUNT(*) FILTER (
            WHERE NOT EXISTS (
                SELECT 1 FROM vintage_rows n
                WHERE n.ticker = o.ticker
                  AND n.vintage_bsns_year = p.new_bsns_year
                  AND n.isu_dcrs_de = o.isu_dcrs_de
                  AND n.isu_dcrs_stle = o.isu_dcrs_stle
                  AND n.isu_dcrs_stock_knd = o.isu_dcrs_stock_knd
                  AND n.isu_dcrs_qy = o.isu_dcrs_qy
            )
        ) AS old_events_absent_from_newest
    FROM pairs p
    JOIN vintage_rows o
      ON o.ticker = p.ticker
     AND o.vintage_bsns_year = p.old_bsns_year
     AND o.isu_dcrs_de <= p.cut_date
    GROUP BY 1
    ORDER BY 1
    """


def build_identity_pass_rate_sql(*, vintage_policy: str, **views: str) -> str:
    """Metric 2 for one policy — how many filing positions survive the identity."""
    issuance = build_issuance_sql(vintage_policy=vintage_policy, **views)
    return f"""
    SELECT
        '{vintage_policy}' AS vintage_policy,
        COUNT(*) AS positions,
        COUNT(*) FILTER (WHERE has_prior_year) AS positions_with_prior_year,
        COUNT(*) FILTER (WHERE issuance_identity_ok) AS identity_ok,
        COUNT(ev_net_share_issuance_yoy) AS feature_available,
        COUNT(ev_net_share_issuance_yoy)::DOUBLE
            / NULLIF(COUNT(*) FILTER (WHERE has_prior_year), 0) AS feature_available_rate
    FROM ({issuance})
    """


def measure_identity_pass_rate(
    con: duckdb.DuckDBPyConnection,
    *,
    trading_days: Sequence[date],
    policies: Sequence[str] = VINTAGE_POLICIES,
    **views: str,
) -> list[dict[str, object]]:
    """Run metric 2 for each policy against an already-attached lake."""
    register_event_scan_calendar(con, trading_days)
    rows: list[dict[str, object]] = []
    for policy in policies:
        sql = build_identity_pass_rate_sql(vintage_policy=policy, **views)
        result = con.execute(sql)
        columns = [d[0] for d in result.description]
        rows.append(dict(zip(columns, result.fetchone(), strict=True)))
    return rows
