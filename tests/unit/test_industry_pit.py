"""``dim_industry_pit_daily`` — F-1.5 (01_industry_pit §2.3).

What has to hold for a point-in-time industry dimension to be worth having:

* an observation is not readable on the day it was observed — the monthly run
  has landed at 23:10 and at 00:14 KST, so "observed on t" says nothing about
  whether it was known during session t;
* the backcast segment is *marked*, and marked apart from "this ticker was
  never profiled at all", because a consumer that must drop look-ahead rows
  has to be able to find them;
* the fold-up is ``industry_groups.resolve_groups``, not a second
  implementation of it in SQL — the point of ``01`` §7 ("업종 그룹 깊이 재선택
  하지 않는다") — and it is re-resolved per month, because membership moves;
* a change flag reads NULL while a change is undetectable, rather than FALSE.
  The seed rows carry ``dart_corp_master``'s current state, so the first pair
  can differ for reasons that predate both snapshots.
"""

from __future__ import annotations

import datetime as dt
import json

import duckdb
import pytest
from collector.kr.shared import (
    MIN_GROUP_SIZE,
    OTHER_GROUP,
    UNKNOWN_GROUP,
    resolve_groups,
)

from modeler.etl.config import DataRoot, LakeConfig
from modeler.etl.features.industry_pit import (
    CHANGE_LOOKBACK_SESSIONS,
    FORMULA_VERSION,
    INDUSTRY_PIT_TABLE,
    build_industry_pit_sql,
    compute_industry_observations,
    materialize_industry_pit,
)
from modeler.etl.mart import is_materialized

# A run of consecutive weekday sessions is enough: every window in this mart is
# ROWS-based over a ticker's own valid sessions, so the calendar never enters.
_START = dt.date(2026, 1, 5)  # a Monday

Observation = tuple[str, dt.date, dt.datetime, str | None, bool]


def _sessions(count: int, start: dt.date = _START) -> list[dt.date]:
    days: list[dt.date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += dt.timedelta(days=1)
    return days


def _at(session: dt.date, hour: int = 0, minute: int = 14) -> dt.datetime:
    """The moment the monthly run read a profile, on a session's own date."""
    return dt.datetime(session.year, session.month, session.day, hour, minute)


def _con(
    prices: list[tuple[dt.date, str, str]],
    observations: list[Observation],
    *,
    halted: list[tuple[dt.date, str, str]] | None = None,
) -> duckdb.DuckDBPyConnection:
    """A lake with just the two views this mart reads.

    ``observations`` are ``(ticker, observed_month, observed_at, induty_code,
    is_seed)``; ``halted`` rows are written with ``open=high=low=0``, the halt
    encoding ``build_valid_session_sql`` filters on.
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE daily_ohlcv (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume DOUBLE)"
    )
    con.executemany("INSERT INTO daily_ohlcv VALUES (?,?,?,100,100,100,100,1000)", prices)
    if halted:
        con.executemany("INSERT INTO daily_ohlcv VALUES (?,?,?,0,0,0,100,0)", halted)
    con.execute(
        "CREATE TABLE dart_corp_profile_history (corp_code VARCHAR, ticker VARCHAR, "
        "observed_month DATE, observed_at TIMESTAMP WITH TIME ZONE, induty_code VARCHAR, "
        "is_seed BOOLEAN)"
    )
    if observations:
        con.executemany(
            "INSERT INTO dart_corp_profile_history VALUES (?,?,?,?,?,?)",
            [
                (f"c{ticker}", ticker, month, at, code, seed)
                for ticker, month, at, code, seed in observations
            ],
        )
    return con


def _rows_by_date(con: duckdb.DuckDBPyConnection, ticker: str) -> dict[dt.date, tuple]:
    """``trade_date -> (code, group, observed_at, backcast, changed, group_changed)``."""
    staging = compute_industry_observations(con)
    rows = con.execute(
        "SELECT trade_date, ind_ksic_code, ind_group, ind_observed_at, ind_is_backcast, "
        "ind_changed_recent_252, ind_group_changed_recent_252 "
        f"FROM ({build_industry_pit_sql(staging)}) WHERE ticker = ? ORDER BY trade_date",
        [ticker],
    ).fetchall()
    return {row[0]: row[1:] for row in rows}


# --------------------------------------------------------------------------
# the as-of rule
# --------------------------------------------------------------------------


def test_an_observation_is_not_readable_on_its_own_observation_day() -> None:
    # The 2026-09 snapshot was read at 00:14 KST and the seed at 23:10. If the
    # mart exposed an observation on date(observed_at), the first of those would
    # be legitimate and the second a look-ahead -- and nothing in the row says
    # which one happened. One session later is correct for both.
    sessions = _sessions(6)
    observed_on = sessions[2]
    con = _con(
        [(d, "000100", "KOSPI") for d in sessions],
        [
            ("000100", dt.date(2025, 12, 1), dt.datetime(2025, 12, 20, 23, 10), "26", True),
            ("000100", dt.date(2026, 1, 1), _at(observed_on, 23, 10), "58", False),
        ],
    )

    rows = _rows_by_date(con, "000100")

    assert rows[observed_on][0] == "26"  # still the previous code
    assert rows[sessions[3]][0] == "58"  # readable from the next session


def test_the_backcast_segment_carries_the_earliest_code_and_says_so() -> None:
    sessions = _sessions(5)
    con = _con(
        [(d, "000100", "KOSPI") for d in sessions],
        [("000100", dt.date(2026, 1, 1), _at(sessions[2], 23, 10), "26", True)],
    )

    rows = _rows_by_date(con, "000100")

    assert [rows[d][3] for d in sessions[:3]] == [True, True, True]
    assert rows[sessions[3]][3] is False
    # The same code on both sides -- the backcast IS the first observation
    # applied backwards, so the boundary must not read as an industry change.
    assert {rows[d][0] for d in sessions} == {"26"}


def test_a_ticker_that_was_never_profiled_is_null_not_backcast() -> None:
    # 365,397 sessions on snapshot 2026-09-08 are this case: delisted before
    # OpenDART was collected. "No industry known" and "industry known but
    # backcast" call for different handling, so they cannot share a value.
    sessions = _sessions(3)
    con = _con(
        [(d, t, "KOSPI") for d in sessions for t in ("000100", "000200")],
        [("000100", dt.date(2026, 1, 1), _at(sessions[1], 23, 10), "26", True)],
    )

    absent = _rows_by_date(con, "000200")

    assert {row[3] for row in absent.values()} == {None}  # ind_is_backcast
    assert {row[0] for row in absent.values()} == {None}  # ind_ksic_code
    assert {row[2] for row in absent.values()} == {None}  # ind_observed_at


def test_halt_rows_never_enter_the_grain() -> None:
    sessions = _sessions(4)
    con = _con(
        [(d, "000100", "KOSPI") for d in sessions[:2]],
        [("000100", dt.date(2026, 1, 1), dt.datetime(2026, 1, 1, 9, 0), "26", True)],
        halted=[(d, "000100", "KOSPI") for d in sessions[2:]],
    )

    rows = _rows_by_date(con, "000100")

    assert set(rows) == set(sessions[:2])


# --------------------------------------------------------------------------
# the fold-up is the shared rule, per observed month
# --------------------------------------------------------------------------


def test_the_group_is_resolve_groups_answer_not_a_sql_reimplementation() -> None:
    # 25 names in KSIC 26 clear MIN_GROUP_SIZE and keep their own group; the
    # three in 01 do not, and their section (A) is short too, so they land in
    # OTHER. Both facts come from resolve_groups; this test demands the mart
    # agree with it rather than restating the thresholds.
    sessions = _sessions(2)
    big = [f"1000{i:02d}" for i in range(MIN_GROUP_SIZE + 5)]
    small = ["200001", "200002", "200003"]
    codes: dict[str, str | None] = {
        **{t: "26100" for t in big},
        **{t: "01120" for t in small},
    }
    con = _con(
        [(d, t, "KOSPI") for d in sessions for t in codes],
        [
            (t, dt.date(2026, 1, 1), dt.datetime(2026, 1, 1, 9, 0), code, True)
            for t, code in codes.items()
        ],
    )
    staging = compute_industry_observations(con)

    got = dict(
        con.execute(
            f"SELECT DISTINCT ticker, ind_group FROM ({build_industry_pit_sql(staging)})"
        ).fetchall()
    )

    assert got == resolve_groups(codes)
    assert got[big[0]] == "26"
    assert got[small[0]] == OTHER_GROUP


def test_the_fold_up_is_re_resolved_each_month_as_membership_moves() -> None:
    # A group of exactly MIN_GROUP_SIZE stands on its own in January. One name
    # delists, so in February the group is 19 -- below the minimum -- and folds
    # up. Resolving once over the whole history would freeze January's answer
    # onto February, which is the mistake `resolve_groups` documents.
    sessions = _sessions(4)
    tickers = [f"1000{i:02d}" for i in range(MIN_GROUP_SIZE)]
    january: dict[str, str | None] = {t: "26100" for t in tickers}
    february: dict[str, str | None] = {t: "26100" for t in tickers[:-1]}
    con = _con(
        [(d, t, "KOSPI") for d in sessions for t in tickers],
        [
            (t, dt.date(2026, 1, 1), dt.datetime(2026, 1, 1, 9, 0), code, True)
            for t, code in january.items()
        ]
        + [
            (t, dt.date(2026, 2, 1), _at(sessions[1]), code, False)
            for t, code in february.items()
        ],
    )
    staging = compute_industry_observations(con)

    staged = {
        (ticker, month): group
        for ticker, month, group in con.execute(
            f"SELECT ticker, observed_month, ind_group FROM {staging}"
        ).fetchall()
    }

    survivor = tickers[0]
    assert staged[(survivor, dt.date(2026, 1, 1))] == resolve_groups(january)[survivor]
    assert staged[(survivor, dt.date(2026, 2, 1))] == resolve_groups(february)[survivor]
    assert staged[(survivor, dt.date(2026, 1, 1))] == "26"
    assert staged[(survivor, dt.date(2026, 2, 1))] == OTHER_GROUP


def test_a_missing_code_stays_unknown_and_is_never_folded_into_a_real_group() -> None:
    sessions = _sessions(2)
    con = _con(
        [(d, "000100", "KOSPI") for d in sessions],
        [("000100", dt.date(2026, 1, 1), dt.datetime(2026, 1, 1, 9, 0), None, True)],
    )

    rows = _rows_by_date(con, "000100")

    assert {row[1] for row in rows.values()} == {UNKNOWN_GROUP}
    # The observation exists, so this is a backcast row with a known-empty code
    # -- not the "never profiled" case above.
    assert rows[sessions[1]][3] is not None


# --------------------------------------------------------------------------
# the change flags
# --------------------------------------------------------------------------


def test_a_change_is_undetectable_until_two_non_seed_observations_are_exposed() -> None:
    # Snapshot 2026-09-08 is exactly this state: one seed month plus one real
    # month. F-1.4 found two code differences across that pair and established
    # that one of them was a precision change, not an industry change -- and
    # that the seed's observed_at cannot date either of them. FALSE would claim
    # more than we know.
    sessions = _sessions(6)
    con = _con(
        [(d, "000100", "KOSPI") for d in sessions],
        [
            ("000100", dt.date(2025, 12, 1), dt.datetime(2025, 12, 20, 23, 10), "26", True),
            ("000100", dt.date(2026, 1, 1), _at(sessions[1]), "58", False),
        ],
    )

    rows = _rows_by_date(con, "000100")

    assert rows[sessions[3]][0] == "58"  # the new code is exposed...
    assert rows[sessions[3]][4] is None  # ...but nothing is claimed about when it moved
    assert rows[sessions[3]][5] is None


def test_a_change_between_two_real_snapshots_is_flagged() -> None:
    # Both groups are populated above MIN_GROUP_SIZE on both sides of the move,
    # so `ind_group` is "26" then "58" rather than the OTHER a thin fixture
    # would fold everything into -- the group flag has to be a statement about
    # the industry, not about the fixture's size.
    sessions = _sessions(10)
    subject = "100000"
    manufacturing = [subject] + [f"1000{i:02d}" for i in range(1, MIN_GROUP_SIZE)]
    software = [f"2000{i:02d}" for i in range(MIN_GROUP_SIZE)]
    tickers = manufacturing + software

    def month(observed_month: dt.date, at: dt.datetime, subject_code: str, seed: bool):
        codes = {t: "26100" for t in manufacturing} | {t: "58200" for t in software}
        codes[subject] = subject_code
        return [(t, observed_month, at, code, seed) for t, code in codes.items()]

    con = _con(
        [(d, t, "KOSPI") for d in sessions for t in tickers],
        month(dt.date(2025, 12, 1), dt.datetime(2025, 12, 20, 23, 10), "26100", True)
        + month(dt.date(2026, 1, 1), _at(sessions[1]), "26100", False)
        + month(dt.date(2026, 2, 1), _at(sessions[4]), "58200", False)
        + month(dt.date(2026, 3, 1), _at(sessions[7]), "58200", False),
    )

    rows = _rows_by_date(con, subject)

    # The 2026-01 snapshot's only predecessor is the seed, so nothing is
    # measurable yet even though the code is exposed.
    assert rows[sessions[3]][4] is None
    assert (rows[sessions[3]][0], rows[sessions[3]][1]) == ("26100", "26")
    assert (rows[sessions[5]][0], rows[sessions[5]][1]) == ("58200", "58")
    assert rows[sessions[5]][4] is True  # the 26 -> 58 move, dated to its own snapshot
    assert rows[sessions[5]][5] is True  # and a group change too
    # 2026-03 repeats 58200: measurable, and no change of its own -- but the
    # February move is still inside the lookback, so the flag stays TRUE.
    assert rows[sessions[8]][4] is True


def test_a_precision_change_moves_the_code_flag_but_not_the_group_flag() -> None:
    # F-1.4's measured case: 262 -> 26293 is the same industry reported at a
    # finer precision. The code flag has to fire (the string did change) and the
    # group flag must not -- which is why D-F1's 0.3% threshold is stated on the
    # group rate.
    sessions = _sessions(8)
    con = _con(
        [(d, "000100", "KOSPI") for d in sessions],
        [
            ("000100", dt.date(2025, 12, 1), dt.datetime(2025, 12, 20, 23, 10), "262", True),
            ("000100", dt.date(2026, 1, 1), _at(sessions[1]), "262", False),
            ("000100", dt.date(2026, 2, 1), _at(sessions[4]), "26293", False),
        ],
    )

    rows = _rows_by_date(con, "000100")

    assert rows[sessions[5]][4] is True
    assert rows[sessions[5]][5] is False


def test_the_flag_expires_after_the_lookback_window() -> None:
    sessions = _sessions(CHANGE_LOOKBACK_SESSIONS + 5)
    con = _con(
        [(d, "000100", "KOSPI") for d in sessions],
        [
            ("000100", dt.date(2025, 11, 1), dt.datetime(2025, 11, 20, 23, 10), "26", True),
            ("000100", dt.date(2025, 12, 1), _at(sessions[0]), "26", False),
            ("000100", dt.date(2026, 1, 1), _at(sessions[1]), "58", False),
        ],
    )

    rows = _rows_by_date(con, "000100")

    assert rows[sessions[2]][4] is True  # the session the change becomes readable
    assert rows[sessions[2 + CHANGE_LOOKBACK_SESSIONS - 1]][4] is True
    assert rows[sessions[2 + CHANGE_LOOKBACK_SESSIONS]][4] is False


# --------------------------------------------------------------------------
# the cache contract
# --------------------------------------------------------------------------


def test_the_frozen_rules_are_part_of_the_cache_contract() -> None:
    sql = build_industry_pit_sql()

    assert FORMULA_VERSION in sql
    assert f"min_group_size={MIN_GROUP_SIZE}" in sql
    assert f"lookback={CHANGE_LOOKBACK_SESSIONS}" in sql


def test_materialize_is_idempotent_and_rebuilds_a_stale_contract(tmp_path) -> None:
    sessions = _sessions(3)
    con = _con(
        [(d, "000100", "KOSPI") for d in sessions],
        [("000100", dt.date(2026, 1, 1), dt.datetime(2026, 1, 1, 9, 0), "26", True)],
    )
    config = LakeConfig(root=DataRoot(tmp_path), snapshot_date="2026-01-09", source="test")

    assert materialize_industry_pit(con, config) == INDUSTRY_PIT_TABLE
    assert is_materialized(config, INDUSTRY_PIT_TABLE)
    assert materialize_industry_pit(con, config) == INDUSTRY_PIT_TABLE  # cached

    path = (
        tmp_path
        / "derived"
        / "feature"
        / "snapshot_date=2026-01-09"
        / "source=test"
        / INDUSTRY_PIT_TABLE
        / "_cache_metadata.json"
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["sql_hash"] = "0" * 64  # as if the fold-up rule had moved under it
    path.write_text(json.dumps(stored, sort_keys=True), encoding="utf-8")

    # A mart written under another contract is rebuilt, never reused. The
    # currency check owns that decision here, so the caller does not have to
    # pass force -- the same shape as dim_peer_monthly.
    assert materialize_industry_pit(con, config) == INDUSTRY_PIT_TABLE
    assert json.loads(path.read_text(encoding="utf-8"))["sql_hash"] != "0" * 64


def test_a_config_hash_change_rebuilds_rather_than_binding_the_old_mart(tmp_path) -> None:
    sessions = _sessions(3)
    con = _con(
        [(d, "000100", "KOSPI") for d in sessions],
        [("000100", dt.date(2026, 1, 1), dt.datetime(2026, 1, 1, 9, 0), "26", True)],
    )
    unpinned = LakeConfig(root=DataRoot(tmp_path), snapshot_date="2026-01-09", source="test")
    materialize_industry_pit(con, unpinned)

    pinned = LakeConfig(
        root=DataRoot(tmp_path),
        snapshot_date="2026-01-09",
        source="test",
        analysis_config_hash="3ca949e6",
    )

    # register_mart_view refuses a mart stamped with another run's hash, so the
    # currency check has to catch the mismatch before it gets there.
    assert materialize_industry_pit(con, pinned) == INDUSTRY_PIT_TABLE
    path = (
        tmp_path
        / "derived"
        / "feature"
        / "snapshot_date=2026-01-09"
        / "source=test"
        / INDUSTRY_PIT_TABLE
        / "_cache_metadata.json"
    )
    assert json.loads(path.read_text(encoding="utf-8"))["analysis_config_hash"] == "3ca949e6"


def test_an_empty_history_fails_loudly_rather_than_grouping_everything_unknown() -> None:
    con = _con([(d, "000100", "KOSPI") for d in _sessions(2)], [])

    with pytest.raises(RuntimeError, match="no ticker-bearing row"):
        compute_industry_observations(con)
