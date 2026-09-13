"""F-6.1 — which receipts the `fin_sue` backfill should actually fetch.

The narrowing rule is the whole substance of this script, and it rests on one
fact read out of `features/sue_event.py` rather than out of prose:
`comparative_eps` comes from `value_lag_4q`, the comparative column inside the
*same* filing, so one filing yields one `seasonal_change` and a `fin_sue` value
needs nine consecutive filings -- not thirteen.

Two ways this could be quietly wrong, both tested:

* fetching everything missing. 25,370 receipts lack XBRL but 2,670 of them sit
  in windows containing a quarter that was never filed, and no fetch closes
  those;
* fetching too little. A receipt that participates in *any* completable window
  has to be a target even if most of its windows are hopeless.
"""

from __future__ import annotations

import datetime as dt

from modeler.analysis.sue_backfill_targets import RUN_LENGTH, select_targets
from modeler.etl.features.sue_event import MIN_SUE_HISTORY


def _row(corp: str, period: int, *, has_xbrl: bool, ticker: str = "000100") -> tuple:
    """One original periodic filing at a quarter index."""
    year, quarter = 2015 + period // 4, period % 4
    return (
        corp,
        ticker,
        f"{corp}-{period:03d}",
        dt.date(year, 3 * quarter + 3, 15),
        year,
        ("11013", "11012", "11014", "11011")[quarter],
        period,
        has_xbrl,
    )


def _run(corp: str, count: int, missing: set[int]) -> list[tuple]:
    return [_row(corp, p, has_xbrl=p not in missing) for p in range(count)]


def test_the_run_length_comes_from_the_feature_code() -> None:
    # Nine, because eight rows of history plus the event's own filing. If
    # sue_event's MIN_SUE_HISTORY moves, this moves with it rather than drifting.
    assert RUN_LENGTH == MIN_SUE_HISTORY + 1 == 9


def test_a_gap_inside_a_completable_window_is_a_target() -> None:
    rows = _run("A", RUN_LENGTH, missing={4})

    targets, stats = select_targets(rows)

    assert [t["rcept_no"] for t in targets] == ["A-004"]
    assert stats["windows_completable"] == 1
    assert stats["windows_unfilable"] == 0


def test_a_window_with_an_unfiled_quarter_yields_nothing() -> None:
    # Period 3 was never filed at all -- there is no receipt to fetch, so the
    # other gaps in that window are unreachable and must not be requested.
    rows = [r for r in _run("A", RUN_LENGTH, missing={4}) if r[6] != 3]

    targets, stats = select_targets(rows)

    assert targets == []
    assert stats["windows_completable"] == 0
    assert stats["windows_unfilable"] >= 1
    # The receipt is still counted as missing; it is just not a target.
    assert stats["receipts_missing_xbrl"] == 1


def test_a_run_shorter_than_nine_quarters_yields_nothing() -> None:
    rows = _run("A", RUN_LENGTH - 1, missing={2})

    targets, stats = select_targets(rows)

    assert targets == []
    assert stats["windows_completable"] == 0
    assert stats["windows_complete"] == 0


def test_an_already_complete_window_asks_for_nothing() -> None:
    rows = _run("A", RUN_LENGTH, missing=set())

    targets, stats = select_targets(rows)

    assert targets == []
    assert stats["windows_complete"] == 1


def test_a_receipt_is_a_target_if_any_one_of_its_windows_is_completable() -> None:
    # Eleven quarters with period 1 never filed and period 6 missing XBRL. The
    # windows starting at 0 and 1 both straddle the unfiled quarter and are
    # hopeless; the one starting at 2 is complete except for period 6. Period 6
    # belongs to all three, so a rule that gave up on a receipt because *a*
    # window was unfilable would drop a fetch that does close a window.
    rows = [r for r in _run("A", RUN_LENGTH + 2, missing={6}) if r[6] != 1]

    targets, stats = select_targets(rows)

    assert [t["rcept_no"] for t in targets] == ["A-006"]
    assert stats["windows_unfilable"] == 2
    assert stats["windows_completable"] == 1


def test_every_gap_in_a_window_is_requested_not_just_the_first() -> None:
    # Filling one of two holes still leaves the window short, so a rule that
    # stopped at the first gap would spend calls and change nothing.
    rows = _run("A", RUN_LENGTH, missing={2, 6})

    targets, _ = select_targets(rows)

    assert sorted(t["rcept_no"] for t in targets) == ["A-002", "A-006"]


def test_corporations_are_judged_independently() -> None:
    rows = _run("A", RUN_LENGTH, missing={4}) + _run("B", RUN_LENGTH - 1, missing={4})

    targets, stats = select_targets(rows)

    assert [t["corp_code"] for t in targets] == ["A"]
    assert stats["corps_seen"] == 2
    assert stats["target_corps"] == 1


def test_the_target_carries_the_fields_the_backfill_command_parses() -> None:
    # cli/app.py reads exactly these five keys per JSON line.
    targets, _ = select_targets(_run("A", RUN_LENGTH, missing={0}))

    assert set(targets[0]) == {"ticker", "corp_code", "bsns_year", "reprt_code", "rcept_no"}
    assert targets[0]["reprt_code"] == "11013"  # period 0 is a Q1 report
    assert targets[0]["bsns_year"] == 2015


def test_a_missing_ticker_becomes_an_empty_string_not_none() -> None:
    # The command does str(row["ticker"]); a None would arrive as the literal
    # "None" and be sent to OpenDART as a stock code.
    rows = [
        (r[0], None, r[2], r[3], r[4], r[5], r[6], r[7])
        for r in _run("A", RUN_LENGTH, missing={0})
    ]

    targets, _ = select_targets(rows)

    assert targets[0]["ticker"] == ""


def test_targets_are_deduplicated_across_overlapping_windows() -> None:
    # A gap in the middle of a long run belongs to many windows. It must appear
    # once: the file is fed straight to the fetcher.
    rows = _run("A", RUN_LENGTH + 5, missing={7})

    targets, _ = select_targets(rows)

    assert len(targets) == 1


def test_an_empty_input_is_not_an_error() -> None:
    targets, stats = select_targets([])

    assert targets == []
    assert stats["corps_seen"] == 0
    assert stats["receipts_seen"] == 0
