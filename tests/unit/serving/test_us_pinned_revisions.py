"""US prepare: pinned universe-input revisions and the newest complete session A' (2026-10-05)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
from collector.lake import DataRoot

from modeler.serving import us_daily
from modeler.serving.us_daily import (
    UNIVERSE_INPUT_TABLES,
    collect_pinned_revisions,
    latest_complete_session,
    sha256_file,
    verify_universe_completion,
)
from modeler.us.lake import US_TABLES, UsLake

SESSIONS = [date(2026, 9, 30), date(2026, 10, 1), date(2026, 10, 2)]  # Wed, Thu, Fri


def _snapshot(root: DataRoot, table: str, snapshot: str, frame: pl.DataFrame | None = None) -> Path:
    directory = root.derived / "snapshots" / table / f"snapshot_date={snapshot}"
    directory.mkdir(parents=True, exist_ok=True)
    part = directory / "part.parquet"
    (frame if frame is not None else pl.DataFrame({"x": [1]})).write_parquet(part)
    return part


def _prices(days: list[date]) -> pl.DataFrame:
    return pl.DataFrame({"date": days, "symbol": ["AAA"] * len(days), "close": [10.0] * len(days)})


def _lake(
    tmp_path: Path, *, universe_inputs_at: str, newest_prices: str, newest_prices_days=SESSIONS
):
    """A lake shaped like 2026-10-05 morning: the universe was built from the 10-03 prices snapshot,
    derive-daily wrote a new prices snapshot on 10-05 although no session arrived."""
    root = DataRoot(tmp_path / "us")
    for table in US_TABLES:
        _snapshot(root, table, "2026-10-01")
    calendar = pl.DataFrame(
        {
            "date": [*SESSIONS, date(2026, 10, 3)],  # Saturday: not a session of XNYS
            "exchange": ["XNYS"] * 3 + ["XCME"],
            "close_local": [None] * 4,
        }
    )
    _snapshot(root, "trading_calendar", "2026-10-05", calendar)
    prices_pinned = _snapshot(root, "prices_daily", universe_inputs_at, _prices(SESSIONS))
    if newest_prices != universe_inputs_at:
        _snapshot(root, "prices_daily", newest_prices, _prices(newest_prices_days))
    universe = pl.DataFrame({"date": SESSIONS, "symbol": ["AAA"] * 3, "in_universe": [True] * 3})
    universe_part = _snapshot(root, "universe_daily", universe_inputs_at, universe)
    inputs = {
        "prices_daily": universe_inputs_at,
        "listing_snapshots": "2026-10-01",
        "filings_sub": "2026-10-01",
        "midas_security_daily": "2026-10-01",
    }
    ticker_source = root.base / "raw" / "tickers.json"
    ticker_source.parent.mkdir(parents=True, exist_ok=True)
    ticker_source.write_text("{}")
    record = {
        "schema_version": 1,
        "table": "universe_daily",
        "snapshot_date": universe_inputs_at,
        "snapshot_sha256": sha256_file(universe_part),
        "unjudged_months": [],
        "input_snapshots": inputs,
        "input_snapshot_sha256": {
            table: sha256_file(
                root.derived / "snapshots" / table / f"snapshot_date={day}" / "part.parquet"
            )
            for table, day in inputs.items()
        },
        "ticker_source_sha256": {"raw/tickers.json": sha256_file(ticker_source)},
        "previous_month_seed_count": 3,
    }
    (universe_part.parent / "completion.json").write_text(json.dumps(record))
    return root, prices_pinned


def test_newest_of_every_table_fails_the_universe_check_but_the_pinned_set_passes(
    tmp_path: Path,
) -> None:
    """The 2026-10-05 22:5x defect: A=10-02 prepare died in seconds on the revision check."""
    root, _ = _lake(tmp_path, universe_inputs_at="2026-10-03", newest_prices="2026-10-05")
    newest = UsLake(root).snapshot_manifest()
    assert newest["prices_daily"] == "2026-10-05" and newest["universe_daily"] == "2026-10-03"
    with pytest.raises(ValueError, match="input revision differs"):
        verify_universe_completion(root, newest["universe_daily"], newest)
    pinned = collect_pinned_revisions(root, UsLake(root))
    assert pinned["prices_daily"] == "2026-10-03"
    assert pinned["universe_daily"] == "2026-10-03"
    assert {table: pinned[table] for table in UNIVERSE_INPUT_TABLES}["filings_sub"] == "2026-10-01"
    # Tables outside the universe inputs still follow their newest snapshot.
    assert pinned["trading_calendar"] == "2026-10-05"
    assert verify_universe_completion(root, pinned["universe_daily"], pinned)


def test_the_verification_itself_still_rejects_a_changed_pinned_input(tmp_path: Path) -> None:
    root, prices_part = _lake(tmp_path, universe_inputs_at="2026-10-03", newest_prices="2026-10-05")
    pinned = collect_pinned_revisions(root, UsLake(root))
    _prices(SESSIONS[:2]).write_parquet(prices_part)  # content changed after the universe was built
    with pytest.raises(ValueError, match="source snapshot changed: prices_daily"):
        verify_universe_completion(root, pinned["universe_daily"], pinned)


def test_a_completion_without_usable_input_revisions_changes_nothing(tmp_path: Path) -> None:
    root, _ = _lake(tmp_path, universe_inputs_at="2026-10-03", newest_prices="2026-10-05")
    marker = (
        root.derived
        / "snapshots"
        / "universe_daily"
        / "snapshot_date=2026-10-03"
        / "completion.json"
    )
    for broken in (
        "not json",
        json.dumps({"input_snapshots": "x"}),
        json.dumps({"input_snapshots": {"prices_daily": "not-a-date"}}),
    ):
        marker.write_text(broken)
        assert collect_pinned_revisions(root, UsLake(root)) == UsLake(root).snapshot_manifest()


def test_a_prime_is_a_when_the_lake_covers_it_and_older_when_it_does_not(
    tmp_path: Path, monkeypatch
) -> None:
    root, _ = _lake(tmp_path, universe_inputs_at="2026-10-03", newest_prices="2026-10-05")
    monkeypatch.setattr(us_daily.DataRoot, "resolve", classmethod(lambda cls, market: root))
    covered = latest_complete_session(date(2026, 10, 2))
    assert covered["resolved"] == "2026-10-02" and covered["stale"] is False
    # A = Monday 10-05 does not exist in the lake yet: A' is Friday 10-02, three days back.
    behind = latest_complete_session(date(2026, 10, 5))
    assert behind["requested"] == "2026-10-05" and behind["resolved"] == "2026-10-02"
    assert behind["stale"] is True and behind["pinned_inputs"]["prices_daily"] == "2026-10-03"
    # A on a non-session (Saturday): the session before it.
    assert latest_complete_session(date(2026, 10, 3))["resolved"] == "2026-10-02"
    assert latest_complete_session(date(2026, 10, 1))["resolved"] == "2026-10-01"


def test_a_newer_prices_snapshot_cannot_move_a_prime_past_the_universe(
    tmp_path: Path, monkeypatch
) -> None:
    """New prices (a session the universe has not seen) wait for the universe rebuild."""
    newer = [*SESSIONS, date(2026, 10, 5)]
    root, _ = _lake(
        tmp_path,
        universe_inputs_at="2026-10-03",
        newest_prices="2026-10-06",
        newest_prices_days=newer,
    )
    calendar = pl.DataFrame(
        {
            "date": [*SESSIONS, date(2026, 10, 5)],
            "exchange": ["XNYS"] * 4,
            "close_local": [None] * 4,
        }
    )
    _snapshot(root, "trading_calendar", "2026-10-06", calendar)
    monkeypatch.setattr(us_daily.DataRoot, "resolve", classmethod(lambda cls, market: root))
    found = latest_complete_session(date(2026, 10, 5))
    assert found["prices_max"] == "2026-10-02" and found["universe_max"] == "2026-10-02"
    assert found["resolved"] == "2026-10-02" and found["stale"] is True


def test_an_empty_lake_resolves_nothing(tmp_path: Path, monkeypatch) -> None:
    root, _ = _lake(tmp_path, universe_inputs_at="2026-10-03", newest_prices="2026-10-03")
    empty = pl.DataFrame({"date": [], "exchange": [], "close_local": []}).cast(
        {"date": pl.Date, "exchange": pl.String, "close_local": pl.Time}
    )
    _snapshot(root, "trading_calendar", "2026-10-06", empty)
    monkeypatch.setattr(us_daily.DataRoot, "resolve", classmethod(lambda cls, market: root))
    found = latest_complete_session(date(2026, 10, 2))
    assert found["resolved"] is None and found["stale"] is False


def test_prepare_reads_the_pinned_revisions_not_the_newest_of_every_table(
    tmp_path: Path, monkeypatch
) -> None:
    class Reached(Exception):
        pass

    root, _ = _lake(tmp_path, universe_inputs_at="2026-10-03", newest_prices="2026-10-05")
    monkeypatch.setattr(us_daily.DataRoot, "resolve", classmethod(lambda cls, market: root))
    seen = []

    def fake_pinned(given_root, lake):
        seen.append((given_root, lake))
        raise Reached

    monkeypatch.setattr(us_daily, "collect_pinned_revisions", fake_pinned)
    with pytest.raises(Reached):
        us_daily.prepare_daily_features(date(2026, 10, 2), diagnostic_only=True)
    assert seen and seen[0][0] is root
