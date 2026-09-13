"""``feat_filing_activity`` — disclosure activity from receipts we already hold (N5-7).

The features themselves are counts, so most of what can go wrong is timing.
These tests run the real SQL over small synthetic panels where the correct
answer is countable by hand, and the ones that matter pin the point-in-time
rule: a receipt filed on day D must not be visible on day D.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pytest

from modeler.etl.features.filing_activity import (
    AMENDMENT_MARKERS,
    INSIDER_MARKER,
    MAJOR_HOLDER_MARKER,
    RATIO_WINDOW,
    WINDOWS,
    build_filing_activity_sql,
)

TICKER = "005930"
MARKET = "KOSPI"
START = date(2020, 1, 6)  # a Monday

INSIDER_NAME = f"임원ㆍ{INSIDER_MARKER}"
MAJOR_NAME = f"{MAJOR_HOLDER_MARKER}(일반)"
AMENDED_INSIDER_NAME = f"[{AMENDMENT_MARKERS[0]}]임원ㆍ{INSIDER_MARKER}"
PLAIN_NAME = "현금ㆍ현물배당결정"


def _sessions(count: int, start: date = START) -> list[date]:
    """`count` consecutive weekday sessions."""
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _run(
    sessions: list[date],
    receipts: list[tuple[date, str]],
    *,
    ticker: str = TICKER,
) -> list[dict]:
    """Run the mart SQL over a synthetic panel and return rows as dicts."""
    con = duckdb.connect()
    con.execute("CREATE TABLE universe(trade_date DATE, ticker VARCHAR, market VARCHAR)")
    con.executemany(
        "INSERT INTO universe VALUES (?, ?, ?)",
        [(day, ticker, MARKET) for day in sessions],
    )
    con.execute(
        "CREATE VIEW dim_universe_daily AS "
        "SELECT trade_date, ticker, market, TRUE AS in_universe FROM universe"
    )
    con.execute("CREATE TABLE receipts(ticker VARCHAR, rcept_dt DATE, report_nm VARCHAR)")
    if receipts:
        con.executemany(
            "INSERT INTO receipts VALUES (?, ?, ?)",
            [(ticker, day, name) for day, name in receipts],
        )
    con.execute("CREATE VIEW dart_filing_receipt_raw AS SELECT * FROM receipts")

    result = con.execute(build_filing_activity_sql(universe_view="dim_universe_daily")).fetchall()
    columns = [description[0] for description in con.description]
    rows = [dict(zip(columns, row, strict=True)) for row in result]
    return sorted(rows, key=lambda row: row["trade_date"])


def _by_date(rows: list[dict]) -> dict[date, dict]:
    return {row["trade_date"]: row for row in rows}


# --------------------------------------------------------------------------
# point in time
# --------------------------------------------------------------------------


def test_a_filing_is_not_visible_on_the_day_it_was_filed() -> None:
    # The test this module exists for. DART publishes through the day, so
    # counting a filing on its own receipt date would let an evening
    # disclosure predict that afternoon's return.
    sessions = _sessions(5)
    rows = _by_date(_run(sessions, [(sessions[2], PLAIN_NAME)]))

    assert rows[sessions[2]]["ev_filing_count_60d"] == 0
    assert rows[sessions[3]]["ev_filing_count_60d"] == 1


def test_a_filing_on_a_non_session_day_lands_on_the_next_session() -> None:
    # Receipts arrive on weekends and holidays too; they must not vanish.
    sessions = _sessions(5)  # Mon..Fri
    saturday = sessions[4] + timedelta(days=1)
    later = _sessions(3, start=sessions[4] + timedelta(days=3))

    rows = _by_date(_run(sessions + later, [(saturday, PLAIN_NAME)]))

    assert rows[sessions[4]]["ev_filing_count_60d"] == 0
    assert rows[later[0]]["ev_filing_count_60d"] == 1


def test_a_filing_after_the_last_session_is_simply_not_counted() -> None:
    # Not an error and not back-dated: there is no session to expose it on yet.
    sessions = _sessions(5)
    rows = _by_date(_run(sessions, [(sessions[-1], PLAIN_NAME)]))

    assert all(row["ev_filing_count_60d"] == 0 for row in rows.values())


# --------------------------------------------------------------------------
# counting and classification
# --------------------------------------------------------------------------


def test_the_window_counts_sessions_not_filing_days() -> None:
    # Without a zero row on quiet days a ROWS window would count filing days,
    # and a company that files twice a decade would carry a decade-long window.
    sessions = _sessions(WINDOWS[0] + 10)
    rows = _by_date(_run(sessions, [(sessions[0], PLAIN_NAME)]))

    # Visible from the next session, and gone once 60 sessions have passed.
    assert rows[sessions[1]]["ev_filing_count_60d"] == 1
    assert rows[sessions[WINDOWS[0]]]["ev_filing_count_60d"] == 1
    assert rows[sessions[WINDOWS[0] + 1]]["ev_filing_count_60d"] == 0


def test_insider_and_five_percent_filings_are_counted_separately() -> None:
    sessions = _sessions(6)
    receipts = [
        (sessions[0], INSIDER_NAME),
        (sessions[0], MAJOR_NAME),
        (sessions[0], PLAIN_NAME),
    ]
    row = _by_date(_run(sessions, receipts))[sessions[1]]

    assert row["ev_filing_count_60d"] == 3
    assert row["own_insider_filing_60d"] == 1
    assert row["own_major_filing_60d"] == 1


def test_an_amended_ownership_filing_counts_in_both_places() -> None:
    # An amendment is still a filing; it is also an amendment. Counting it in
    # only one place would make the ratio's denominator disagree with the count.
    sessions = _sessions(6)
    receipts = [(sessions[0], INSIDER_NAME), (sessions[0], AMENDED_INSIDER_NAME)]
    row = _by_date(_run(sessions, receipts))[sessions[1]]

    assert row["own_insider_filing_60d"] == 2
    assert row["own_amendment_ratio_1y"] == pytest.approx(0.5)
    assert row["ev_amendment_ratio_1y"] == pytest.approx(0.5)


def test_the_ownership_amendment_ratio_ignores_unrelated_amendments() -> None:
    # Its denominator is ownership filings, so an amended dividend resolution
    # must not move it — that is what separates it from ev_amendment_ratio_1y.
    sessions = _sessions(6)
    receipts = [
        (sessions[0], INSIDER_NAME),
        (sessions[0], f"[{AMENDMENT_MARKERS[0]}]{PLAIN_NAME}"),
    ]
    row = _by_date(_run(sessions, receipts))[sessions[1]]

    assert row["own_amendment_ratio_1y"] == pytest.approx(0.0)
    assert row["ev_amendment_ratio_1y"] == pytest.approx(0.5)


def test_the_amendment_markers_are_the_ones_phase_b_already_uses() -> None:
    # A second definition of the same concept is worse than either one:
    # revision_ratio and these features would drift apart silently.
    from modeler.etl.phase_b_quality import AMENDMENT_MARKERS as PHASE_B_MARKERS

    assert AMENDMENT_MARKERS is PHASE_B_MARKERS


# --------------------------------------------------------------------------
# ratios and bursts
# --------------------------------------------------------------------------


def test_a_company_with_no_filings_gets_null_not_zero_for_the_ratio() -> None:
    # Zero would read as "this company never amends", which is a claim. No
    # denominator is not a claim.
    sessions = _sessions(10)
    rows = _run(sessions, [])

    assert all(row["ev_amendment_ratio_1y"] is None for row in rows)
    assert all(row["ev_filing_count_60d"] == 0 for row in rows)


def test_the_burst_ratio_is_null_until_its_baseline_window_is_full() -> None:
    # A median over a partial window is not comparable with one over a full
    # window, and the ratio would look like a burst wherever history is short.
    sessions = _sessions(RATIO_WINDOW + 20)
    # File steadily, so the baseline median is non-zero and the only reason a
    # burst can be NULL is the window not being full yet.
    steady = [(day, PLAIN_NAME) for day in sessions[::5]]
    rows = _by_date(_run(sessions, steady))

    assert rows[sessions[10]]["ev_filing_burst_60d"] is None
    assert rows[sessions[-1]]["ev_filing_burst_60d"] == pytest.approx(1.0, abs=0.3)


def test_a_quiet_history_leaves_the_burst_undefined_rather_than_infinite() -> None:
    # A median of zero is the common case for a company that rarely files;
    # dividing by it must not produce an infinity that dominates a z-score.
    sessions = _sessions(RATIO_WINDOW + 20)
    rows = _by_date(_run(sessions, [(sessions[-3], PLAIN_NAME)]))

    assert rows[sessions[-1]]["ev_filing_burst_60d"] is None


# --------------------------------------------------------------------------
# shape
# --------------------------------------------------------------------------


def test_both_pre_registered_windows_are_emitted() -> None:
    # 60 and 120 are both pre-registered (N5-6), so picking between them after
    # seeing results is not available.
    rows = _run(_sessions(5), [])

    for window in WINDOWS:
        assert f"ev_filing_count_{window}d" in rows[0]
        assert f"own_insider_filing_{window}d" in rows[0]
        assert f"own_major_filing_{window}d" in rows[0]


def test_the_grain_is_one_row_per_ticker_session() -> None:
    sessions = _sessions(7)
    rows = _run(sessions, [(sessions[0], PLAIN_NAME)])

    assert len(rows) == len(sessions)
    assert {row["trade_date"] for row in rows} == set(sessions)
    assert {row["market"] for row in rows} == {MARKET}
