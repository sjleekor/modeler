"""The market-cap distortion window and the feature that masks it (N1-9 §5.5).

`market_cap = close x listed_shares` is exact, and still economically wrong for
about three weeks after a 무상증자: the price adjusts on the 권리락 date and the
new shares list weeks later, so an adjusted price is multiplied by a pre-issue
share count. These tests run the real SQL over panels small enough to count by
hand, and the ones that matter pin the two things easy to get wrong — the window
must start the session *after* the filing, and it must end when the shares list,
not when someone decides enough time has passed.
"""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pytest
from modeler.etl.corporate_actions import (
    MAX_WINDOW_DAYS,
    build_mcap_distortion_sql,
    register_mcap_distortion_view,
)
from modeler.etl.features.market_cap import build_market_cap_sql

TICKER = "005930"
MARKET = "KOSPI"
START = date(2026, 1, 5)  # a Monday
BONUS_FILING = "권리락(무상증자)"
SPACED_FILING = "권리락              (무상증자)"


def _sessions(count: int, start: date = START) -> list[date]:
    days: list[date] = []
    cursor = start
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def _con(
    rows: list[tuple[date, int, int]],
    receipts: list[tuple[date, str]],
    *,
    ticker: str = TICKER,
) -> duckdb.DuckDBPyConnection:
    """`rows` are (trade_date, close, listed_shares)."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE daily_market_cap("
        "trade_date DATE, ticker VARCHAR, market VARCHAR,"
        "source_close BIGINT, listed_shares BIGINT, market_cap BIGINT)"
    )
    con.executemany(
        "INSERT INTO daily_market_cap VALUES (?, ?, ?, ?, ?, ?)",
        [(d, ticker, MARKET, c, s, c * s) for d, c, s in rows],
    )
    con.execute(
        "CREATE TABLE dart_filing_receipt_raw("
        "ticker VARCHAR, rcept_dt DATE, report_nm VARCHAR)"
    )
    if receipts:
        con.executemany(
            "INSERT INTO dart_filing_receipt_raw VALUES (?, ?, ?)",
            [(ticker, d, name) for d, name in receipts],
        )
    return con


def _flagged(con: duckdb.DuckDBPyConnection) -> set[date]:
    rows = con.execute(build_mcap_distortion_sql()).fetchall()
    return {row[0] for row in rows}


def _bonus_panel(
    sessions: list[date],
    *,
    ex_idx: int,
    listing_idx: int | None,
) -> list[tuple[date, int, int]]:
    """Price halves at `ex_idx`; shares double at `listing_idx` (None = never)."""
    rows = []
    for i, day in enumerate(sessions):
        close = 10_000 if i < ex_idx else 5_000
        shares = 1_000_000
        if listing_idx is not None and i >= listing_idx:
            shares = 2_000_000
        rows.append((day, close, shares))
    return rows


# --------------------------------------------------------------------------
# where the window starts and ends
# --------------------------------------------------------------------------


def test_the_window_starts_the_session_after_the_filing() -> None:
    # The exchange files 권리락 before the price moves. Measured on prod: using
    # the receipt date itself finds no drop at all (average factor 1.000) while
    # the next session shows 2.023. Starting a session early would mask a
    # session whose market cap is still correct.
    sessions = _sessions(10)
    con = _con(_bonus_panel(sessions, ex_idx=4, listing_idx=8), [(sessions[3], BONUS_FILING)])

    flagged = _flagged(con)

    assert sessions[3] not in flagged  # filing day: price has not moved yet
    assert sessions[4] in flagged  # ex-date


def test_the_window_ends_when_the_new_shares_list() -> None:
    # From the listing session on, both factors are post-issue again and market
    # cap is correct. Masking it would throw away good data.
    sessions = _sessions(10)
    con = _con(_bonus_panel(sessions, ex_idx=4, listing_idx=8), [(sessions[3], BONUS_FILING)])

    flagged = _flagged(con)

    assert flagged == set(sessions[4:8])
    assert sessions[8] not in flagged


def test_a_filed_issue_that_never_lists_stops_at_the_cap() -> None:
    # 66 of 619 filings had no matching share move. Without a cap those tickers
    # would be masked for the rest of the panel.
    sessions = _sessions(200)
    con = _con(_bonus_panel(sessions, ex_idx=1, listing_idx=None), [(sessions[0], BONUS_FILING)])

    flagged = _flagged(con)

    assert flagged  # it does mask
    assert max(flagged) <= sessions[1] + timedelta(days=MAX_WINDOW_DAYS)
    assert sessions[-1] not in flagged


# --------------------------------------------------------------------------
# what must not be flagged
# --------------------------------------------------------------------------


def test_a_share_rise_with_no_bonus_filing_is_not_flagged() -> None:
    # The common case by far: conversions, warrant exercises and paid-in issues
    # list new shares without a price adjustment, and market cap *should* grow.
    # About 80% of share increases between 1.05x and 1.8x look like this.
    sessions = _sessions(10)
    rows = [(day, 10_000, 1_000_000 if i < 5 else 2_000_000) for i, day in enumerate(sessions)]
    con = _con(rows, [])

    assert _flagged(con) == set()


def test_an_unrelated_filing_does_not_open_a_window() -> None:
    sessions = _sessions(10)
    con = _con(
        _bonus_panel(sessions, ex_idx=4, listing_idx=8),
        [(sessions[3], "현금ㆍ현물배당결정"), (sessions[3], "주요사항보고서(무상증자결정)")],
    )

    # 무상증자결정 is the decision, not the ex-date — it carries no 권리락.
    assert _flagged(con) == set()


def test_the_filing_name_matches_despite_padding() -> None:
    # report_nm spacing is not stable across years; both spellings are the same
    # disclosure and 155 of the 액면병합 rows alone use the padded form.
    sessions = _sessions(10)
    con = _con(_bonus_panel(sessions, ex_idx=4, listing_idx=8), [(sessions[3], SPACED_FILING)])

    assert _flagged(con) == set(sessions[4:8])


def test_a_tiny_share_move_does_not_close_the_window() -> None:
    # A sub-threshold wobble is rounding or a trivial issuance, not the listing
    # that resolves the bonus issue. Treating it as the end would unmask sessions
    # whose share count is still pre-issue.
    sessions = _sessions(10)
    rows = []
    for i, day in enumerate(sessions):
        close = 10_000 if i < 4 else 5_000
        shares = 1_000_000
        if 5 <= i < 8:
            shares = 1_010_000  # +1%, below the 5% threshold
        elif i >= 8:
            shares = 2_000_000
        rows.append((day, close, shares))
    con = _con(rows, [(sessions[3], BONUS_FILING)])

    assert _flagged(con) == set(sessions[4:8])


# --------------------------------------------------------------------------
# the feature
# --------------------------------------------------------------------------


def _feature_rows(con: duckdb.DuckDBPyConnection) -> dict[date, dict]:
    register_mcap_distortion_view(con)
    result = con.execute(build_market_cap_sql()).fetchall()
    columns = [d[0] for d in con.description]
    return {row[0]: dict(zip(columns, row, strict=True)) for row in result}


def test_the_modelling_column_is_null_inside_the_window() -> None:
    sessions = _sessions(10)
    con = _con(_bonus_panel(sessions, ex_idx=4, listing_idx=8), [(sessions[3], BONUS_FILING)])

    rows = _feature_rows(con)

    assert rows[sessions[4]]["mcap_krx_log"] is None
    assert rows[sessions[3]]["mcap_krx_log"] is not None
    assert rows[sessions[8]]["mcap_krx_log"] is not None


def test_the_raw_value_survives_so_the_mask_can_be_audited() -> None:
    # Dropping the rows entirely would make "masked" and "no data" the same
    # thing, and there would be no way to measure what the mask removed.
    sessions = _sessions(10)
    con = _con(_bonus_panel(sessions, ex_idx=4, listing_idx=8), [(sessions[3], BONUS_FILING)])

    rows = _feature_rows(con)
    masked = rows[sessions[4]]

    assert masked["mcap_unreliable"] is True
    assert masked["mcap_krx"] == 5_000 * 1_000_000
    assert len(rows) == len(sessions)


def test_the_flag_is_false_rather_than_null_outside_the_window() -> None:
    # A NULL flag would propagate into any downstream boolean and silently turn
    # a filter into "unknown".
    sessions = _sessions(6)
    rows = [(day, 10_000, 1_000_000) for day in sessions]
    con = _con(rows, [])

    assert all(row["mcap_unreliable"] is False for row in _feature_rows(con).values())


def test_the_unmasked_variant_is_available_for_measuring_the_mask() -> None:
    sessions = _sessions(10)
    con = _con(_bonus_panel(sessions, ex_idx=4, listing_idx=8), [(sessions[3], BONUS_FILING)])
    register_mcap_distortion_view(con)

    unmasked = con.execute(build_market_cap_sql(distortion_view=None)).fetchall()
    columns = [d[0] for d in con.description]
    rows = {row[0]: dict(zip(columns, row, strict=True)) for row in unmasked}

    assert rows[sessions[4]]["mcap_krx_log"] is not None
    assert all(row["mcap_unreliable"] is False for row in rows.values())


def test_a_non_positive_market_cap_is_null_not_an_error() -> None:
    sessions = _sessions(4)
    con = _con([(day, 0, 1_000_000) for day in sessions], [])

    assert all(row["mcap_krx_log"] is None for row in _feature_rows(con).values())


@pytest.mark.parametrize("column", ["trade_date", "ticker", "market", "mcap_krx", "mcap_krx_log"])
def test_the_grain_and_columns_are_stable(column: str) -> None:
    sessions = _sessions(3)
    con = _con([(day, 10_000, 1_000_000) for day in sessions], [])

    assert column in next(iter(_feature_rows(con).values()))
