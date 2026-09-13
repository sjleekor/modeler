"""F-1.7 — the ``induty_code`` change-rate report (01_industry_pit §4).

This report produces the number D-F1 is decided on, so the two ways it could
be quietly wrong are the two things tested here:

* a seed pair must never reach the average. The seed rows are copies of
  ``dart_corp_master``'s current state, so a difference across that boundary
  cannot be dated, and averaging it in would let the 2026-09 snapshot alone
  settle a decision that needs three real observation windows;
* a precision change (``262`` -> ``26293``) must not be counted as an industry
  change. F-1.4 measured one of those in the very first pair.

The threshold itself (0.3%) was fixed before any result and is asserted as a
constant, not recomputed.
"""

from __future__ import annotations

import datetime as dt

import duckdb

from modeler.analysis.industry_change_report import (
    GROUP_RATE_THRESHOLD_PCT,
    REQUIRED_PAIRS,
    _pair_rows,
    cumulative_section,
    direction_section,
    precision_section,
    verdict_section,
)

_STAGING = "_industry_pit_staging"


def _staged(rows: list[tuple[str, dt.date, str, str, bool]]) -> duckdb.DuckDBPyConnection:
    """``(ticker, observed_month, induty_code, ind_group, is_seed)`` as staged."""
    con = duckdb.connect()
    con.execute(
        f"CREATE TABLE {_STAGING} (ticker VARCHAR, observed_month DATE, "
        "observed_at TIMESTAMP WITH TIME ZONE, observed_date DATE, induty_code VARCHAR, "
        "ind_group VARCHAR, is_seed BOOLEAN)"
    )
    con.executemany(
        f"INSERT INTO {_STAGING} VALUES (?, ?, NULL, NULL, ?, ?, ?)",
        rows,
    )
    return con


def _months(count: int) -> list[dt.date]:
    return [dt.date(2026, 8 + i, 1) for i in range(count)]


def _cohort(
    months: list[dt.date],
    codes: dict[str, list[str]],
    groups: dict[str, list[str]],
    *,
    seed_first: bool = True,
) -> list[tuple[str, dt.date, str, str, bool]]:
    return [
        (ticker, month, codes[ticker][i], groups[ticker][i], seed_first and i == 0)
        for ticker in codes
        for i, month in enumerate(months)
    ]


def test_the_threshold_is_the_one_fixed_before_any_result() -> None:
    assert GROUP_RATE_THRESHOLD_PCT == 0.3
    assert REQUIRED_PAIRS == 3


def test_a_seed_pair_is_reported_and_kept_out_of_the_verdict() -> None:
    # Snapshot 2026-09-08's actual shape: one seed month, one real month, one
    # group change. 1/3959 is 0.0253%, far under the threshold -- and it must
    # still not decide anything.
    months = _months(2)
    rows = _cohort(
        months,
        {"000100": ["47100", "46100"], "000200": ["26100", "26100"]},
        {"000100": ["47", "46"], "000200": ["26", "26"]},
    )
    pairs = _pair_rows(_staged(rows), _STAGING)

    assert len(pairs) == 1
    assert pairs[0][2] is True  # involves_seed
    assert (pairs[0][3], pairs[0][7]) == (2, 1)  # n_both, group_changes

    verdict = verdict_section(pairs)

    assert "판정 보류" in verdict
    assert "비-seed 구간이 0/3개" in verdict
    assert "판정에 넣지 않는다" in verdict


def test_a_precision_change_is_split_out_of_the_code_rate() -> None:
    months = _months(2)
    rows = _cohort(
        months,
        {"000100": ["262", "26293"], "000200": ["47100", "46100"]},
        {"000100": ["26", "26"], "000200": ["47", "46"]},
    )
    pairs = _pair_rows(_staged(rows), _STAGING)

    _, _, _, n_both, code, precision, genuine, group = pairs[0]

    assert (n_both, code, precision, genuine, group) == (2, 2, 1, 1, 1)
    assert "정밀도 변경 1건 + 실제 변경 1건" in precision_section(pairs)


def test_three_real_pairs_under_the_threshold_release_f3() -> None:
    # Four snapshots: the seed plus three real ones, one group change in 1,000
    # names per pair -> 0.1% a month, under 0.3%.
    months = _months(4)
    codes: dict[str, list[str]] = {}
    groups: dict[str, list[str]] = {}
    for i in range(1000):
        ticker = f"{i:06d}"
        codes[ticker] = ["26100"] * 4
        groups[ticker] = ["26"] * 4
    # One name moves in each of the three real transitions.
    for pair_index, ticker in enumerate(("000001", "000002", "000003")):
        for i in range(pair_index + 1, 4):
            codes[ticker][i] = "58200"
            groups[ticker][i] = "58"

    pairs = _pair_rows(_staged(_cohort(months, codes, groups)), _STAGING)
    verdict = verdict_section(pairs)

    assert len(pairs) == 3
    assert [row[2] for row in pairs] == [True, False, False]
    # Only two pairs are seed-free, so the verdict still cannot be reached --
    # which is the point: three real pairs need four real snapshots.
    assert "판정 보류" in verdict
    assert "비-seed 구간이 2/3개" in verdict


def test_a_rate_over_the_threshold_keeps_industry_diagnostic_only() -> None:
    months = _months(5)
    codes: dict[str, list[str]] = {}
    groups: dict[str, list[str]] = {}
    for i in range(100):
        ticker = f"{i:06d}"
        codes[ticker] = ["26100"] * 5
        groups[ticker] = ["26"] * 5
    # One of 100 moves at every real transition: 1% a month, well over 0.3%.
    for pair_index in range(4):
        ticker = f"{pair_index + 1:06d}"
        for i in range(pair_index + 1, 5):
            codes[ticker][i] = "58200"
            groups[ticker][i] = "58"

    pairs = _pair_rows(_staged(_cohort(months, codes, groups)), _STAGING)
    verdict = verdict_section(pairs)

    assert len([row for row in pairs if not row[2]]) == 3
    assert "문턱 초과" in verdict
    assert "정규화·진단 전용을 유지" in verdict


def test_a_low_enough_rate_over_three_real_pairs_allows_f3() -> None:
    months = _months(5)
    codes: dict[str, list[str]] = {}
    groups: dict[str, list[str]] = {}
    for i in range(2000):
        ticker = f"{i:06d}"
        codes[ticker] = ["26100"] * 5
        groups[ticker] = ["26"] * 5
    # One of 2,000 at each real transition: 0.05% a month.
    for pair_index in range(4):
        ticker = f"{pair_index + 1:06d}"
        for i in range(pair_index + 1, 5):
            codes[ticker][i] = "58200"
            groups[ticker][i] = "58"

    pairs = _pair_rows(_staged(_cohort(months, codes, groups)), _STAGING)
    verdict = verdict_section(pairs)

    assert "문턱 미달" in verdict
    assert "ind_is_backcast" in verdict  # the label the card must still carry


def test_the_cumulative_rate_is_measured_against_the_first_snapshot() -> None:
    # A name that moves and moves back reads as 0 cumulative change even though
    # two monthly transitions fired -- that is the definition in §4.1, and the
    # difference between the two rates is itself the interesting quantity.
    months = _months(3)
    rows = _cohort(
        months,
        {"000100": ["26100", "58200", "26100"]},
        {"000100": ["26", "58", "26"]},
    )
    con = _staged(rows)

    monthly = _pair_rows(con, _STAGING)
    cumulative = cumulative_section(con, _STAGING)

    assert [row[6] for row in monthly] == [1, 1]  # both transitions are real moves
    assert "| 2026-10-01 | 1 | 0 | 0.0000% | 0 | 0.0000% |" in cumulative


def test_the_direction_table_names_both_groups() -> None:
    months = _months(2)
    rows = _cohort(
        months,
        {"000100": ["47100", "46100"]},
        {"000100": ["47", "46"]},
    )

    table = direction_section(_staged(rows), _STAGING)

    assert "| 47 | 46 | 1 |" in table


def test_no_change_says_so_instead_of_rendering_an_empty_table() -> None:
    months = _months(2)
    rows = _cohort(months, {"000100": ["26100", "26100"]}, {"000100": ["26", "26"]})
    con = _staged(rows)

    assert "그룹이 바뀐 법인이 없다" in direction_section(con, _STAGING)
    assert "정밀도 변경과 실제 변경을 나눌 것도 없다" in precision_section(
        _pair_rows(con, _STAGING)
    )
