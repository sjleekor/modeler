"""feat_filing_activity — disclosure-activity features from receipts already held (N5-7).

Source: ``dart_filing_receipt_raw``, 1,201,866 rows over 2015-01 .. 2026-08 for
2,657 tickers. **Nothing new is collected here.** The table has been in the
pipeline since Phase B and is used only for ``fin_sue_event`` and for quality
diagnostics; not one alpha feature reads it.

This is what replaced the ``elestock`` collection plan (N5-6). The DS004
ownership APIs answer with a rolling two-year window — even Samsung Electronics,
listed in 1975, starts at 2024-08-26 — and take ``corp_code`` as their only
argument, so there is no year loop that reaches further back. The receipts carry
the same events over **10.5 years** at 98.1% coverage. The trade is quantity for
frequency: the collected path could have given holdings changes, this one gives
filing counts. A ten-year sample of a weaker measure beats a one-year sample of
a stronger one.

Two definitions are shared rather than re-derived, because a second definition
of the same concept is worse than either one:

* amendment markers come from :mod:`modeler.etl.phase_b_quality`, which already
  computes ``revision_ratio`` from them for Phase B source quality.
* the trading calendar is the universe panel's own ``trade_date`` set, so
  windows count sessions the way every other feature family counts them.

PIT: a receipt filed on day D is exposed on the **next trading day**, never on
D itself. DART publishes through the day, and same-day exposure would let an
evening filing predict that afternoon's return.

``ev_material_event_flag`` (`09` §3) is deliberately absent. It needs a
materiality list — which ``report_nm`` values count as material — and that
choice is not made yet. Adding it with a provisional list would put an
unregistered judgement inside a pre-registered bundle.
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view
from modeler.etl.phase_b_quality import AMENDMENT_MARKERS

__all__ = [
    "AMENDMENT_MARKERS",
    "BURST_BASELINE",
    "FILING_ACTIVITY_FORMULA_VERSION",
    "FILING_ACTIVITY_TABLE",
    "INSIDER_MARKER",
    "MAJOR_HOLDER_MARKER",
    "RATIO_WINDOW",
    "WINDOWS",
    "build_filing_activity_sql",
    "materialize_filing_activity",
]

FILING_ACTIVITY_TABLE = "feat_filing_activity"

#: Bump when a definition below changes, for the same reason
#: ``FIN_FEATURE_FORMULA_VERSION`` exists: these rules are not covered by
#: ``config_hash`` or the Phase B code hash, so a silent change would produce
#: different numbers under an identical run fingerprint.
#:
#:   filing_v1 — the 2026-08 definitions as first written (N5-7).
#:   filing_v2 — expose one-session-lag variants used by the expansion Scan.
#:   filing_v3 — use the registered broad Horizon Scan universe mart.
FILING_ACTIVITY_FORMULA_VERSION = "filing_v3"

#: 임원ㆍ주요주주 특정증권등 소유상황보고서 — the ``elestock`` event, as a receipt.
#: The separator is U+318D (ㆍ), not a middle dot, which is why this matches on a
#: substring rather than on equality: 139,697 receipts across 2,607 tickers.
INSIDER_MARKER = "주요주주특정증권등소유상황보고서"

#: 주식등의 대량보유상황보고서 — the 5% rule. 101,656 receipts (일반 + 약식).
MAJOR_HOLDER_MARKER = "주식등의대량보유상황보고서"

#: Trailing session counts. Both are pre-registered (N5-6): 60 as the primary
#: window, 120 as the declared variant, so choosing between them after seeing
#: results is not available.
WINDOWS: tuple[int, ...] = (60, 120)

#: Window for the amendment ratios. A ratio needs enough filings underneath it
#: to be a ratio at all, and filings run about 8 per company-year.
RATIO_WINDOW = 250

#: Sessions of history the burst ratio's baseline needs. Below this the median
#: is being taken over a partial window and the ratio is not comparable.
BURST_BASELINE = 250


def _classification_sql() -> str:
    """CASE expressions splitting a receipt into the classes we count."""
    amendment = " OR ".join(f"report_nm LIKE '%{marker}%'" for marker in AMENDMENT_MARKERS)
    return f"""
        ({amendment}) AS is_amendment,
        (report_nm LIKE '%{INSIDER_MARKER}%') AS is_insider,
        (report_nm LIKE '%{MAJOR_HOLDER_MARKER}%') AS is_major_holder
    """


def build_filing_activity_sql(
    universe_view: str = "dim_universe_broad_daily",
    filing_receipt_view: str = "dart_filing_receipt_raw",
) -> str:
    """SQL producing ``feat_filing_activity`` at (trade_date, ticker, market) grain.

    Args:
        universe_view: Daily panel; also supplies the trading calendar.
        filing_receipt_view: Raw receipt history.

    Returns:
        A SELECT statement.
    """
    window_columns = []
    for window in WINDOWS:
        offset = window - 1
        window_columns.append(f"""
            SUM(filings) OVER (
                PARTITION BY ticker, market ORDER BY trade_date
                ROWS BETWEEN {offset} PRECEDING AND CURRENT ROW
            ) AS ev_filing_count_{window}d,
            SUM(insider_filings) OVER (
                PARTITION BY ticker, market ORDER BY trade_date
                ROWS BETWEEN {offset} PRECEDING AND CURRENT ROW
            ) AS own_insider_filing_{window}d,
            SUM(major_filings) OVER (
                PARTITION BY ticker, market ORDER BY trade_date
                ROWS BETWEEN {offset} PRECEDING AND CURRENT ROW
            ) AS own_major_filing_{window}d""")

    burst_columns = []
    for window in WINDOWS:
        for prefix, column in (
            ("ev_filing", f"ev_filing_count_{window}d"),
            ("own_insider_filing", f"own_insider_filing_{window}d"),
        ):
            burst_columns.append(f"""
            CASE WHEN session_ordinal >= {BURST_BASELINE} THEN
                {column} / NULLIF(
                    quantile_cont({column}, 0.5) OVER (
                        PARTITION BY ticker, market ORDER BY trade_date
                        ROWS BETWEEN {BURST_BASELINE - 1} PRECEDING AND CURRENT ROW
                    ), 0)
            END AS {prefix}_burst_{window}d""")

    return f"""
        WITH receipts AS (
            -- One row per (ticker, receipt day) with the counts we need. The
            -- receipt table has no duplicate rcept_no, so a plain COUNT is the
            -- filing count.
            SELECT
                ticker,
                rcept_dt,
                COUNT(*) AS filings,
                COUNT(*) FILTER (WHERE is_amendment) AS amendments,
                COUNT(*) FILTER (WHERE is_insider) AS insider_filings,
                COUNT(*) FILTER (WHERE is_major_holder) AS major_filings,
                COUNT(*) FILTER (WHERE is_insider OR is_major_holder) AS ownership_filings,
                COUNT(*) FILTER (
                    WHERE is_amendment AND (is_insider OR is_major_holder)
                ) AS ownership_amendments
            FROM (
                SELECT ticker, rcept_dt, {_classification_sql()}
                FROM {filing_receipt_view}
                WHERE ticker IS NOT NULL AND rcept_dt IS NOT NULL
            )
            GROUP BY ticker, rcept_dt
        ),
        trading_days AS (
            SELECT DISTINCT trade_date FROM {universe_view}
        ),
        next_session AS (
            -- A receipt filed on D is exposed on the first session AFTER D.
            -- DART publishes through the day, so exposing it on D itself would
            -- let an evening filing predict that afternoon's return.
            --
            -- Resolved over DISTINCT receipt dates (a few thousand) rather than
            -- over receipts (over a million), which is the difference between a
            -- cheap join and a very expensive one.
            SELECT rd.rcept_dt, MIN(s.trade_date) AS available_date
            FROM (SELECT DISTINCT rcept_dt FROM receipts) rd
            JOIN trading_days s ON s.trade_date > rd.rcept_dt
            GROUP BY rd.rcept_dt
        ),
        daily AS (
            SELECT r.ticker, n.available_date,
                   SUM(r.filings) AS filings,
                   SUM(r.amendments) AS amendments,
                   SUM(r.insider_filings) AS insider_filings,
                   SUM(r.major_filings) AS major_filings,
                   SUM(r.ownership_filings) AS ownership_filings,
                   SUM(r.ownership_amendments) AS ownership_amendments
            FROM receipts r
            JOIN next_session n ON n.rcept_dt = r.rcept_dt
            GROUP BY r.ticker, n.available_date
        ),
        panel AS (
            -- Every universe row, with zeros on days nothing was filed. Without
            -- the zero rows a ROWS window would count filing days rather than
            -- sessions, and a quiet company's window would silently stretch
            -- across years.
            SELECT
                u.trade_date, u.ticker, u.market,
                COALESCE(d.filings, 0) AS filings,
                COALESCE(d.amendments, 0) AS amendments,
                COALESCE(d.insider_filings, 0) AS insider_filings,
                COALESCE(d.major_filings, 0) AS major_filings,
                COALESCE(d.ownership_filings, 0) AS ownership_filings,
                COALESCE(d.ownership_amendments, 0) AS ownership_amendments
            FROM {universe_view} u
            LEFT JOIN daily d
                   ON d.ticker = u.ticker AND d.available_date = u.trade_date
            WHERE u.in_universe
        ),
        windowed AS (
            SELECT
                trade_date, ticker, market,
                COUNT(*) OVER (
                    PARTITION BY ticker, market ORDER BY trade_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS session_ordinal,
                {",".join(window_columns)},
                SUM(amendments) OVER w_ratio AS amendments_{RATIO_WINDOW}d,
                SUM(filings) OVER w_ratio AS filings_{RATIO_WINDOW}d,
                SUM(ownership_amendments) OVER w_ratio AS own_amendments_{RATIO_WINDOW}d,
                SUM(ownership_filings) OVER w_ratio AS own_filings_{RATIO_WINDOW}d
            FROM panel
            WINDOW w_ratio AS (
                PARTITION BY ticker, market ORDER BY trade_date
                ROWS BETWEEN {RATIO_WINDOW - 1} PRECEDING AND CURRENT ROW
            )
        )
        , features AS (
        SELECT
            trade_date, ticker, market,
            {", ".join(
                f"ev_filing_count_{w}d, own_insider_filing_{w}d, own_major_filing_{w}d"
                for w in WINDOWS
            )},
            {",".join(burst_columns)},
            -- A ratio with no denominator is not zero, it is unknown. Returning
            -- 0 here would read as "this company never amends" for companies
            -- that simply have not filed yet.
            amendments_{RATIO_WINDOW}d / NULLIF(filings_{RATIO_WINDOW}d, 0)
                AS ev_amendment_ratio_1y,
            own_amendments_{RATIO_WINDOW}d / NULLIF(own_filings_{RATIO_WINDOW}d, 0)
                AS own_amendment_ratio_1y
        FROM windowed
        )
        SELECT
            *,
            lag(ev_filing_burst_60d) OVER w AS ev_filing_burst_60d_lag1,
            lag(ev_amendment_ratio_1y) OVER w AS ev_amendment_ratio_1y_lag1,
            lag(own_insider_filing_burst_60d) OVER w
                AS own_insider_filing_burst_60d_lag1,
            lag(own_major_filing_60d) OVER w AS own_major_filing_60d_lag1,
            lag(own_amendment_ratio_1y) OVER w AS own_amendment_ratio_1y_lag1
        FROM features
        WINDOW w AS (PARTITION BY ticker, market ORDER BY trade_date)
    """


def materialize_filing_activity(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    universe_view: str = "dim_universe_broad_daily",
    filing_receipt_view: str = "dart_filing_receipt_raw",
    force: bool = False,
) -> str:
    """Build + register the ``feat_filing_activity`` mart view.

    Args:
        con: DuckDB connection with the source views registered.
        config: Lake configuration.
        universe_view: Daily panel view name.
        filing_receipt_view: Raw receipt view name.
        force: Rebuild even when the mart already exists.

    Returns:
        The registered view name.
    """
    materialize(
        con,
        config,
        FILING_ACTIVITY_TABLE,
        build_filing_activity_sql(universe_view, filing_receipt_view),
        force=force,
    )
    return register_mart_view(con, config, FILING_ACTIVITY_TABLE)
