"""Serving 정리매매 rule (G-21): semantics on synthetic receipts and prices."""

from __future__ import annotations

import datetime as dt

import duckdb
import polars as pl
import pytest

from modeler.models._03_kr_next.liquidation_notice import build_liquidation_notice_sql

START = dt.date(2020, 1, 1)  # a Wednesday; day i = START + i days (weekends are sessions here)


def d(i: int) -> dt.date:
    return START + dt.timedelta(days=i)


def _run(receipts, bars, **kw):
    """``bars``: {ticker: set of halted day indexes}; days 0..99 otherwise trade."""
    con = duckdb.connect()
    rows = []
    for t, halted in bars.items():
        for i in range(100):
            z = i in halted
            rows.append(
                (d(i), t, "KOSPI", *(0.0,) * 3, 0.0)
                if z
                else (d(i), t, "KOSPI", 9.0, 11.0, 8.0, 10.0)
            )
    con.register(
        "daily_ohlcv",
        pl.DataFrame(
            [r if len(r) == 7 else (*r,) for r in rows],
            schema=["trade_date", "ticker", "market", "open", "high", "low", "close"],
            orient="row",
        ),
    )
    con.register(
        "dart_filing_receipt_raw",
        pl.DataFrame(
            receipts,
            schema={
                "ticker": pl.String,
                "rcept_no": pl.String,
                "rcept_dt": pl.Date,
                "report_nm": pl.String,
            },
            orient="row",
        ),
    )
    out = con.execute(build_liquidation_notice_sql(**kw)).fetchall()
    return {(r[0], r[1]): (r[2], r[3]) for r in out}


def _flagged_days(res, ticker):
    return sorted((day - START).days for (day, t), (f, _) in res.items() if t == ticker and f)


def test_flag_runs_for_fifteen_sessions_after_the_receipt() -> None:
    res = _run(
        [("A", "R1", d(10), "주권매매거래정지해제(상장폐지에따른정리매매개시)")], {"A": set()}
    )
    # the receipt day itself is session 0 (not counted); sessions 1..14 after it are flagged
    assert _flagged_days(res, "A") == list(range(10, 25))
    assert res[(d(10), "A")] == (True, "R1")
    assert res[(d(25), "A")] == (False, None)
    assert res[(d(5), "A")] == (False, None)


def test_halt_days_do_not_count_and_have_no_row() -> None:
    res = _run([("A", "R1", d(10), "정리매매 개시")], {"A": {12, 13, 14}})
    assert (d(12), "A") not in res
    # three halted days push the end out by three
    assert _flagged_days(res, "A") == [10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]


def test_sixty_calendar_day_expiry_after_a_long_halt() -> None:
    # halted from day 11 to day 79: the non-halt count never reaches 15 but the date cap does
    res = _run([("A", "R1", d(10), "정리매매개시")], {"A": set(range(11, 80))})
    assert _flagged_days(res, "A") == [10]
    res = _run([("A", "R1", d(10), "정리매매개시")], {"A": set(range(11, 60))})
    # trading resumes on day 60; the 60-calendar-day cap (day 70) ends it before 15 sessions
    assert _flagged_days(res, "A") == [10, *range(60, 71)]
    custom = _run([("A", "R1", d(10), "정리매매개시")], {"A": set()}, calendar_days=3)
    assert _flagged_days(custom, "A") == [10, 11, 12, 13]


def test_a_later_hold_clears_and_a_restart_flags_again() -> None:
    receipts = [
        ("A", "20200111000001", d(10), "정리매매개시"),
        ("A", "20200116000002", d(15), "기타시장안내(정리매매보류관련)"),
        ("A", "20200125000003", d(24), "기타시장안내(가처분신청기각에따른정리매매절차재개)"),
    ]
    res = _run(receipts, {"A": set()})
    days = _flagged_days(res, "A")
    assert days[:5] == [10, 11, 12, 13, 14]
    assert 15 not in days and 23 not in days
    assert res[(d(24), "A")] == (True, "20200125000003")
    assert max(days) == 38  # 14 sessions after the restart


def test_same_day_ties_take_the_larger_receipt_no_and_hold_before_start_does_not_clear() -> None:
    receipts = [
        ("A", "20200105000009", d(4), "정리매매보류"),
        ("A", "20200111000001", d(10), "정리매매개시"),
        ("A", "20200111000002", d(10), "[기재정정]정리매매개시"),
    ]
    res = _run(receipts, {"A": set()})
    assert res[(d(10), "A")] == (True, "20200111000002")


def test_report_name_is_whitespace_normalized_and_needs_both_words() -> None:
    receipts = [
        ("A", "R1", d(10), "정리 매 매　개 시"),
        ("B", "R2", d(10), "정리매매 안내"),  # no 개시/재개
        ("C", "R3", d(10), "상장폐지 개시"),  # no 정리매매
        (None, "R4", d(10), "정리매매개시"),  # no ticker: ignored
    ]
    res = _run(receipts, {"A": set(), "B": set(), "C": set()})
    assert _flagged_days(res, "A")[0] == 10
    assert _flagged_days(res, "B") == []
    assert _flagged_days(res, "C") == []


def test_receipt_on_a_non_session_day_counts_from_the_last_session_before_it() -> None:
    # day 10 halted: a receipt filed on day 10 starts counting from session 9, so the receipt
    # day gives no row and sessions 1..14 (days 11..24) are flagged
    res = _run([("A", "R1", d(10), "정리매매개시")], {"A": {10}})
    assert _flagged_days(res, "A") == list(range(11, 25))


def test_never_null_flag_and_no_label_columns() -> None:
    res = _run([("A", "R1", d(10), "정리매매개시")], {"A": set(), "B": set()})
    assert all(f in (True, False) for f, _ in res.values())
    sql = build_liquidation_notice_sql()
    for needle in ("fwd_ret", "label", "close"):
        assert needle not in sql


def test_bad_arguments() -> None:
    with pytest.raises(ValueError):
        build_liquidation_notice_sql(sessions=0)
    with pytest.raises(ValueError):
        build_liquidation_notice_sql(calendar_days=0)
