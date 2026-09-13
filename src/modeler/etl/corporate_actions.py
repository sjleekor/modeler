# ruff: noqa: E501
"""``dim_mcap_distortion`` — sessions where market cap is arithmetically right and economically wrong.

`market_cap` is `close x listed_shares` and the identity holds exactly (N1-9 V1,
7,060,600 rows, 0 violations). The problem is that the two factors do not react
to a corporate action at the same moment.

A 무상증자 adjusts the **price** on the 권리락 date and lists the **new shares**
about three weeks later. In between, an adjusted price is multiplied by a
pre-issue share count, so market cap reads roughly `1/f` of the company's equity
value. Measured on prod 2026-08-19: **553 events, 23 days on average, 85 at the
longest, 12,739 ticker-days** — 0.18% of the panel.

KRX is not wrong. Unlisted shares genuinely are not in the listed market cap.
But a size or value feature reading it sees a company halve and then double back,
and that swing lands exactly on the names doing bonus issues — an event
correlated with past returns and small size. That is a manufactured interaction,
not noise, which is why these sessions are masked rather than kept.

**The condition is an event, not staleness.** A flat price on the day the share
count rises is the normal case: conversions, warrant exercises and paid-in issues
are real issuance and market cap *should* grow (about 80% of share increases
between 1.05x and 1.8x look like this). `shares_age_days` cannot stand in for the
condition either — a five-month-old share count is correct if nothing happened,
and a one-day-old one is wrong if yesterday was an ex-date.

**It is point-in-time.** Both inputs are observable at t: the 권리락 filing is
public on its receipt date, and "the share count has not moved since" is a
statement about the past. Nothing here looks forward.

Scope is deliberately 무상증자 only. 유상증자 and 전환권행사 share the two-date
shape but adjust the price by far less, and only the bonus-issue case has been
measured (`poc/n1_validation.md` §5.5).
"""

from __future__ import annotations

import duckdb

# The exchange files 권리락 on the session *before* the price moves; measured on
# 2026-08-19, using the receipt date itself finds no drop at all (average factor
# 1.000) while the next session shows 2.023 against a later share rise of 2.127.
BONUS_EX_MARKERS: tuple[str, ...] = ("권리락", "무상증자")

# Longest observed gap between 권리락 and the new listing is 85 days. Past this a
# filed issue is treated as never having landed, so the mask does not run forever.
MAX_WINDOW_DAYS = 120

# Below this a share-count move is rounding or a trivial issuance, not the
# listing that resolves a bonus issue.
SHARE_CHANGE_THRESHOLD = 0.05

MCAP_DISTORTION_VIEW = "dim_mcap_distortion"

__all__ = [
    "BONUS_EX_MARKERS",
    "MAX_WINDOW_DAYS",
    "MCAP_DISTORTION_VIEW",
    "SHARE_CHANGE_THRESHOLD",
    "build_mcap_distortion_sql",
    "register_mcap_distortion_view",
]


def build_mcap_distortion_sql(
    receipt_view: str = "dart_filing_receipt_raw",
    market_cap_view: str = "daily_market_cap",
    *,
    max_window_days: int = MAX_WINDOW_DAYS,
    share_change_threshold: float = SHARE_CHANGE_THRESHOLD,
) -> str:
    """SQL listing ``(trade_date, ticker)`` where market cap must not be trusted.

    Only the affected sessions are emitted, so callers ``LEFT JOIN`` and
    ``COALESCE`` to ``FALSE``. Emitting a row per panel session instead would
    make the view as large as the panel to carry 0.18% of the information.
    """
    marker_predicate = " AND ".join(
        f"regexp_replace(report_nm, '\\s+', '', 'g') LIKE '%{marker}%'" for marker in BONUS_EX_MARKERS
    )
    return f"""
        WITH filings AS (
            -- 권리락(무상증자). The whitespace in report_nm is not stable across
            -- years -- '권리락(무상증자)' and '권리락              (무상증자)' are
            -- the same disclosure -- so it is stripped before matching.
            SELECT DISTINCT ticker, rcept_dt
            FROM {receipt_view}
            WHERE ticker IS NOT NULL AND ticker <> ''
              AND {marker_predicate}
        ),
        sessions AS (
            SELECT trade_date, ticker, listed_shares,
                   lag(listed_shares) OVER (PARTITION BY ticker ORDER BY trade_date) AS prev_shares
            FROM {market_cap_view}
        ),
        share_moves AS (
            SELECT ticker, trade_date
            FROM sessions
            WHERE prev_shares IS NOT NULL AND prev_shares > 0
              AND abs(listed_shares * 1.0 / prev_shares - 1) > {share_change_threshold}
        ),
        ex_dates AS (
            -- The price moves on the first session *after* the filing.
            SELECT f.ticker, f.rcept_dt,
                   (SELECT min(s.trade_date) FROM sessions s
                     WHERE s.ticker = f.ticker AND s.trade_date > f.rcept_dt) AS ex_date
            FROM filings f
        ),
        windows AS (
            SELECT e.ticker, e.ex_date,
                   -- The session the new shares list. Market cap is right again
                   -- from that session on, so it is the exclusive end of the window.
                   COALESCE(
                       (SELECT min(m.trade_date) FROM share_moves m
                         WHERE m.ticker = e.ticker AND m.trade_date >= e.ex_date),
                       DATE '9999-12-31'
                   ) AS resolve_date
            FROM ex_dates e
            WHERE e.ex_date IS NOT NULL
        )
        SELECT DISTINCT s.trade_date, s.ticker, TRUE AS mcap_unreliable
        FROM windows w
        JOIN sessions s
          ON s.ticker = w.ticker
         AND s.trade_date >= w.ex_date
         AND s.trade_date < w.resolve_date
         AND s.trade_date <= w.ex_date + INTERVAL '{max_window_days}' DAY
    """


def register_mcap_distortion_view(
    con: duckdb.DuckDBPyConnection,
    *,
    receipt_view: str = "dart_filing_receipt_raw",
    market_cap_view: str = "daily_market_cap",
    view_name: str = MCAP_DISTORTION_VIEW,
    max_window_days: int = MAX_WINDOW_DAYS,
    share_change_threshold: float = SHARE_CHANGE_THRESHOLD,
) -> str:
    """Register the distortion view and return its name."""
    sql = build_mcap_distortion_sql(
        receipt_view,
        market_cap_view,
        max_window_days=max_window_days,
        share_change_threshold=share_change_threshold,
    )
    con.execute(f"CREATE OR REPLACE VIEW {view_name} AS {sql}")
    return view_name
