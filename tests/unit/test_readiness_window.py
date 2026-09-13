"""F-9.9 — the readiness gate has to be passable at its own default.

`--required-coverage-ratio 1.0` is the documented default and it failed 33 of
38 features on snapshot 2026-09-08 for two reasons, neither of them a data
defect:

* the feature-date grid ran ten sessions past the end of the data, because
  `available_from_date` is an *announced* availability and `macro_cpi` /
  `macro_consumer_sentiment` legitimately carried 2026-09-21 while the last
  priced session was 2026-09-07. Every daily series was then charged with
  missing values for sessions that had not happened;
* a derived feature emits a row from its input's first date but cannot carry a
  value until its window fills, so a YoY series has a structurally NULL run at
  the front (253 dates for `macro_cpi_yoy_latest`). Judging from the first
  *fact* charged that run forever.

Both are fixed by narrowing what is judged, so the tests that matter most are
the ones proving the narrowing did not also swallow a real defect.
"""

from __future__ import annotations

from datetime import date

import duckdb
import pytest
from modeler.etl.lake import _common_feature_calendars
from modeler.etl.marts import reports

_CODE = "market_kospi_close"
_DERIVED = "macro_cpi_yoy_latest"


def _obs_con(
    rows: list[tuple[str, date, date]], ohlcv_max: date | None
) -> duckdb.DuckDBPyConnection:
    """A lake with just the two views the calendar helper reads."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE common_feature_observation_raw ("
        "series_id VARCHAR, available_from_date DATE, observation_date DATE)"
    )
    con.executemany("INSERT INTO common_feature_observation_raw VALUES (?,?,?)", rows)
    if ohlcv_max is not None:
        con.execute("CREATE TABLE daily_ohlcv (trade_date DATE)")
        con.execute("INSERT INTO daily_ohlcv VALUES (?)", [ohlcv_max])
    return con


def _fact_con(rows: list[tuple[str, date, float | None, date]]) -> duckdb.DuckDBPyConnection:
    """``(feature_code, feature_date, value_numeric, asof_available_date)``."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE common_feature_daily_fact ("
        "feature_code VARCHAR, feature_date DATE, value_numeric DOUBLE, "
        "asof_available_date DATE)"
    )
    if rows:
        con.executemany("INSERT INTO common_feature_daily_fact VALUES (?,?,?,?)", rows)
    return con


# --------------------------------------------------------------------------
# (a) the grid must not run past the data
# --------------------------------------------------------------------------


def test_an_announced_future_availability_does_not_stretch_the_grid() -> None:
    # The measured case: macro_cpi says 2026-09-21, the last session is 09-07.
    con = _obs_con(
        [
            ("market_kospi", date(2026, 9, 1), date(2026, 9, 1)),
            ("macro_cpi", date(2026, 9, 21), date(2026, 8, 31)),
        ],
        ohlcv_max=date(2026, 9, 7),
    )

    sessions, feature_dates = _common_feature_calendars(con)

    assert sessions == feature_dates
    assert max(feature_dates) == date(2026, 9, 7)


def test_the_clamp_only_lowers_the_end_it_never_extends_it() -> None:
    # Prices running ahead of the observations is not a reason to invent
    # feature dates no series could cover.
    con = _obs_con(
        [("market_kospi", date(2026, 9, 1), date(2026, 9, 1))],
        ohlcv_max=date(2026, 12, 31),
    )

    _, feature_dates = _common_feature_calendars(con)

    assert max(feature_dates) == date(2026, 9, 1)


def test_a_lake_without_prices_is_left_unclamped_rather_than_failed() -> None:
    con = _obs_con(
        [("macro_cpi", date(2026, 9, 21), date(2026, 8, 31))],
        ohlcv_max=None,
    )

    _, feature_dates = _common_feature_calendars(con)

    assert max(feature_dates) == date(2026, 9, 21)


# --------------------------------------------------------------------------
# (b) a warm-up is not a hole
# --------------------------------------------------------------------------


def test_a_derivation_warmup_does_not_block_readiness_and_is_reported() -> None:
    grid = [date(2026, 1, d) for d in (5, 6, 7, 8, 9)]
    # Rows exist from the start; the first two carry no value, as a YoY series
    # cannot until twelve months of input have accumulated.
    rows = [(_DERIVED, d, None if i < 2 else 1.0, d) for i, d in enumerate(grid)]

    row = _row(_fact_con(rows), grid, _DERIVED)

    assert row.window_start == date(2026, 1, 7)
    assert row.warmup_null_count == 2
    assert row.coverage_ratio == 1.0
    assert row.ready, row.blockers


def test_a_null_inside_the_span_still_blocks() -> None:
    # The distinction the whole fix rests on: leading nulls are a warm-up, an
    # interior null is the 2017-10-10 Chuseok closure and must stay visible.
    grid = [date(2026, 1, d) for d in (5, 6, 7, 8, 9)]
    rows = [
        (_DERIVED, grid[0], None, grid[0]),
        (_DERIVED, grid[1], 1.0, grid[1]),
        (_DERIVED, grid[2], None, grid[2]),
        (_DERIVED, grid[3], 1.0, grid[3]),
        (_DERIVED, grid[4], 1.0, grid[4]),
    ]

    row = _row(_fact_con(rows), grid, _DERIVED)

    assert row.window_start == date(2026, 1, 6)
    assert row.warmup_null_count == 1
    assert not row.ready
    assert any("null_count=1" in b for b in row.blockers)


def test_a_date_missing_inside_the_span_still_blocks() -> None:
    grid = [date(2026, 1, d) for d in (5, 6, 7, 8)]
    rows = [
        (_CODE, grid[0], 1.0, grid[0]),
        (_CODE, grid[1], 1.0, grid[1]),
        # grid[2] has no row at all
        (_CODE, grid[3], 1.0, grid[3]),
    ]

    row = _row(_fact_con(rows), grid, _CODE)

    assert not row.ready
    assert any("missing_count=1" in b for b in row.blockers)


def test_a_pit_violation_inside_the_window_still_blocks() -> None:
    grid = [date(2026, 1, d) for d in (5, 6, 7)]
    rows = [
        (_CODE, grid[0], 1.0, grid[0]),
        (_CODE, grid[1], 1.0, date(2026, 2, 1)),  # available after the feature date
        (_CODE, grid[2], 1.0, grid[2]),
    ]

    row = _row(_fact_con(rows), grid, _CODE)

    assert not row.ready
    assert any("pit_violation_count=1" in b for b in row.blockers)


def test_the_old_single_window_behaviour_is_still_reachable() -> None:
    grid = [date(2026, 1, d) for d in (5, 6, 7)]
    rows = [(_DERIVED, grid[0], None, grid[0])] + [
        (_DERIVED, d, 1.0, d) for d in grid[1:]
    ]

    row = _row(_fact_con(rows), grid, _DERIVED, per_feature_window=False)

    assert row.window_start is None
    assert row.warmup_null_count == 0
    assert not row.ready  # the leading null is charged, as before


def test_a_feature_with_no_value_at_all_is_judged_over_the_whole_calendar() -> None:
    grid = [date(2026, 1, d) for d in (5, 6, 7)]
    rows = [(_CODE, d, None, d) for d in grid]

    row = _row(_fact_con(rows), grid, _CODE)

    # No non-NULL value means no window to narrow to, so nothing is forgiven.
    assert row.window_start is None
    assert not row.ready
    assert any("null_count=3" in b for b in row.blockers)


def _row(
    con: duckdb.DuckDBPyConnection,
    grid: list[date],
    code: str,
    *,
    per_feature_window: bool = True,
) -> reports.ReadinessRow:
    rows = reports.readiness_report(
        con,
        feature_dates=grid,
        feature_codes=[code],
        per_feature_window=per_feature_window,
    )
    assert len(rows) == 1
    return rows[0]


def test_the_helpers_agree_on_the_window_start() -> None:
    # _first_feature_dates is the single definition of "where judging begins";
    # _window_counts must scope to exactly that date, or the two disagree and
    # the leading nulls come back.
    grid = [date(2026, 1, d) for d in (5, 6, 7, 8)]
    rows = [(_DERIVED, grid[0], None, grid[0])] + [(_DERIVED, d, 2.0, d) for d in grid[1:]]
    con = _fact_con(rows)

    starts = reports._first_feature_dates(
        con, cfdf_view="common_feature_daily_fact", codes=[_DERIVED]
    )
    counts = reports._window_counts(
        con,
        cfdf_view="common_feature_daily_fact",
        feature_dates=grid,
        window_starts=starts,
    )

    assert starts == {_DERIVED: date(2026, 1, 6)}
    assert counts[_DERIVED] == (3, 3, 0, 0)  # facts, non_null, null, pit


def test_window_counts_is_empty_without_any_window() -> None:
    con = _fact_con([])

    assert (
        reports._window_counts(
            con,
            cfdf_view="common_feature_daily_fact",
            feature_dates=[date(2026, 1, 5)],
            window_starts={},
        )
        == {}
    )


@pytest.mark.parametrize("ratio", [1.0, 0.9])
def test_the_gate_passes_a_clean_feature_at_any_required_ratio(ratio: float) -> None:
    grid = [date(2026, 1, d) for d in (5, 6, 7)]
    con = _fact_con([(_CODE, d, 1.0, d) for d in grid])

    rows = reports.readiness_report(
        con, feature_dates=grid, feature_codes=[_CODE], required_coverage_ratio=ratio
    )

    assert rows[0].ready, rows[0].blockers
