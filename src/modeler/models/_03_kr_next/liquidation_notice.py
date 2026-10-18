"""Serving rule for the 정리매매 (liquidation trading) notice (G-21, prereg draft §5.3 / §16).

Training removes the terminal tail by looking *forward* (``in_terminal_tail``). Serving cannot, so
a name is flagged from what DART had published by the signal date ``t``:

* **start receipt** = a ``dart_filing_receipt_raw`` row whose whitespace-normalized ``report_nm``
  contains ``정리매매`` AND (``개시`` OR ``재개``). The latest one with ``rcept_dt <= t`` counts;
  two receipts on the same day: the larger ``rcept_no``.
* **hold receipt** = normalized ``report_nm`` contains ``정리매매`` AND ``보류``. A hold that is
  *later* than the start (``rcept_dt <= t`` and ``rcept_no`` larger than the start's; ``rcept_no``
  begins with the receipt date, so this is chronological) cancels the flag.
* **15 sessions** — the stock's own non-halt sessions after ``rcept_dt`` up to and including ``t``
  must be fewer than 15. Non-halt is ``NOT (open = 0 AND high = 0 AND low = 0)`` on
  ``daily_ohlcv``, the definition ``universe`` and ``quality`` use. A start received on a
  non-session day counts from the last non-halt session before it.
* **60 calendar days** (added after M-2) — ``t - rcept_dt <= 60``. A notice whose sessions never
  ran (long halt, trading resumed months later) would otherwise stay on for good.

``liq_notice_flag = start exists AND NOT cleared AND sessions < 15 AND days <= 60``;
``liq_notice_rcept_no`` is the start's ``rcept_no`` when the flag is TRUE, else NULL.

The relation has one row per ``(trade_date, ticker)`` that is a non-halt session of the stock.
Joiners should ``COALESCE(liq_notice_flag, FALSE)``. It reads receipts and prices only: no label,
no return. The consumer, not the dataset, drops flagged names from the candidates.
"""

from __future__ import annotations

NOTICE_SESSIONS = 15
NOTICE_CALENDAR_DAYS = 60

#: whitespace-normalized report name (ASCII, ideographic and no-break spaces)
NORMALIZED_REPORT_NM = r"regexp_replace(report_nm, '[\s\x{3000}\x{00a0}]+', '', 'g')"


def build_liquidation_notice_sql(
    receipt_view: str = "dart_filing_receipt_raw",
    price_view: str = "daily_ohlcv",
    *,
    sessions: int = NOTICE_SESSIONS,
    calendar_days: int = NOTICE_CALENDAR_DAYS,
) -> str:
    """SELECT ``trade_date, ticker, liq_notice_flag, liq_notice_rcept_no`` (see module doc)."""
    if not isinstance(sessions, int) or sessions < 1:
        raise ValueError(f"sessions must be a positive int, got {sessions!r}")
    if not isinstance(calendar_days, int) or calendar_days < 1:
        raise ValueError(f"calendar_days must be a positive int, got {calendar_days!r}")
    return f"""
    WITH ln_session AS (
        SELECT trade_date, ticker,
               DENSE_RANK() OVER (PARTITION BY ticker ORDER BY trade_date) AS d_idx
        FROM {price_view}
        WHERE NOT (open = 0 AND high = 0 AND low = 0)
    ),
    ln_session_day AS (
        SELECT trade_date, ticker, MIN(d_idx) AS d_idx FROM ln_session GROUP BY trade_date, ticker
    ),
    ln_receipt AS (
        SELECT ticker, rcept_dt, rcept_no, {NORMALIZED_REPORT_NM} AS norm
        FROM {receipt_view}
        WHERE ticker IS NOT NULL AND rcept_dt IS NOT NULL AND rcept_no IS NOT NULL
    ),
    ln_start AS (
        SELECT ticker, rcept_dt, MAX(rcept_no) AS rcept_no
        FROM ln_receipt
        WHERE norm LIKE '%정리매매%' AND (norm LIKE '%개시%' OR norm LIKE '%재개%')
        GROUP BY ticker, rcept_dt
    ),
    ln_start_base AS (
        -- non-halt sessions the stock had up to and including the receipt date
        SELECT s.ticker, s.rcept_dt, s.rcept_no, COALESCE(d.d_idx, 0) AS base_idx
        FROM ln_start s
        ASOF LEFT JOIN ln_session_day d ON d.ticker = s.ticker AND s.rcept_dt >= d.trade_date
    ),
    ln_hold AS (
        SELECT ticker, rcept_dt, MAX(rcept_no) AS rcept_no
        FROM ln_receipt
        WHERE norm LIKE '%정리매매%' AND norm LIKE '%보류%'
        GROUP BY ticker, rcept_dt
    ),
    ln_joined AS (
        SELECT d.trade_date, d.ticker, d.d_idx,
               s.rcept_dt AS start_dt, s.rcept_no AS start_no, s.base_idx,
               h.rcept_no AS hold_no
        FROM ln_session_day d
        ASOF LEFT JOIN ln_start_base s ON d.ticker = s.ticker AND d.trade_date >= s.rcept_dt
        ASOF LEFT JOIN ln_hold h ON d.ticker = h.ticker AND d.trade_date >= h.rcept_dt
    ),
    ln_flag AS (
        SELECT trade_date, ticker, start_no,
               (start_no IS NOT NULL
                AND NOT COALESCE(hold_no > start_no, FALSE)
                AND (d_idx - base_idx) < {sessions}
                AND (trade_date - start_dt) <= {calendar_days}) AS flag
        FROM ln_joined
    )
    SELECT trade_date, ticker,
           flag AS liq_notice_flag,
           CASE WHEN flag THEN start_no END AS liq_notice_rcept_no
    FROM ln_flag
    """
