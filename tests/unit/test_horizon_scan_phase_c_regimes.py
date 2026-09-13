"""Unit tests for the Phase C regime builder (Stage 1b §2, §5, §6.2, §6.5).

Design: ``docs/dev/20260829_macro_features/01_design/03_stage1b_conditional_ic_phase_c.md``.

These are outcome-blind by construction — nothing here touches a label, a
return or an IC — which is what allows the G1/G2 feasibility numbers to be
computed and recorded before the overlay hash is fixed.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import duckdb
import polars as pl
import pytest
from modeler.analysis.horizon_scan_phase_c_regimes import (
    LONG_WINDOW,
    REGIME_IDS,
    REGIME_SPECS,
    SHORT_WINDOW,
    SOURCE_FEATURE_CODES,
    build_regime_series,
    regime_occupancy,
    regime_persistence,
    regime_subperiod_counts,
    render_regime_summary_md,
    summarize_regimes,
)


def _sessions(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


_DEFAULTS = {
    "global_vix_level": 20.0,
    "market_kospi_close": 2500.0,
    "market_kospi_turnover_value": 1.0e12,
    "market_kosdaq_turnover_value": 5.0e11,
    "rate_kr_term_spread_10y_3y": 0.5,
    "market_kosdaq_ret_1d": 0.0,
    "market_kospi_ret_1d": 0.0,
    "fx_usdkrw_level": 1300.0,
}


def _build(
    sessions: list[date],
    *,
    fact_dates: list[date] | None = None,
    **series: list[float],
) -> pl.DataFrame:
    """Regime series over ``sessions``, with any factor overridden by name.

    ``fact_dates`` defaults to ``sessions``; pass a longer list to put
    non-session rows into the fact and check they are dropped.
    """
    fact_dates = fact_dates or sessions
    con = duckdb.connect()
    con.execute("CREATE TABLE label_scan (trade_date DATE)")
    con.executemany("INSERT INTO label_scan VALUES (?)", [(d,) for d in sessions])
    con.execute(
        "CREATE TABLE common_feature_daily_fact (feature_date DATE, feature_code VARCHAR, "
        "value_numeric DOUBLE, asof_available_date DATE)"
    )
    rows = []
    for code in SOURCE_FEATURE_CODES:
        values = series.get(code, [_DEFAULTS[code]] * len(fact_dates))
        for d, value in zip(fact_dates, values, strict=True):
            rows.append((d, code, value, d))
    con.executemany("INSERT INTO common_feature_daily_fact VALUES (?,?,?,?)", rows)
    return build_regime_series(con)


# --- §2.3 definitions --------------------------------------------------------


def test_every_preregistered_regime_is_emitted_with_a_binary_and_an_alt_cut() -> None:
    sessions = _sessions(date(2020, 1, 1), 400)
    frame = _build(sessions)
    assert REGIME_IDS == (
        "vix_up",
        "vix_high",
        "market_up",
        "liq_high",
        "term_steep",
        "kosdaq_rel_up",
        "krw_weak_20d",
    )
    for rid in REGIME_IDS:
        assert f"z_{rid}" in frame.columns
        assert f"s_{rid}" in frame.columns
        assert f"alt_s_{rid}" in frame.columns
    assert frame["session_idx"].to_list() == list(range(1, len(sessions) + 1))


def test_vix_up_is_the_twenty_session_change() -> None:
    sessions = _sessions(date(2020, 1, 1), 60)
    vix = [20.0 + 0.5 * i for i in range(len(sessions))]
    frame = _build(sessions, global_vix_level=vix)
    row = frame.filter(pl.col("session_idx") == 30)
    assert row["z_vix_up"].item() == pytest.approx(vix[29] - vix[9])
    assert row["s_vix_up"].item() is True


def test_market_up_is_the_252_session_log_ratio() -> None:
    sessions = _sessions(date(2020, 1, 1), 400)
    close = [2000.0 * math.exp(0.0005 * i) for i in range(len(sessions))]
    frame = _build(sessions, market_kospi_close=close)
    row = frame.filter(pl.col("session_idx") == 300)
    assert row["z_market_up"].item() == pytest.approx(math.log(close[299] / close[47]))


def test_liq_high_sums_both_markets_turnover() -> None:
    """§2.3: the daily IC is an n-weighted blend of KOSPI and KOSDAQ, so the
    liquidity regime has to be market-wide rather than KOSPI-only."""
    sessions = _sessions(date(2020, 1, 1), 300)
    kospi = [1.0e12] * len(sessions)
    kosdaq = [5.0e11] * (len(sessions) - 20) + [4.0e12] * 20  # a KOSDAQ-only surge
    frame = _build(
        sessions,
        market_kospi_turnover_value=kospi,
        market_kosdaq_turnover_value=kosdaq,
    )
    assert frame.filter(pl.col("session_idx") == len(sessions))["s_liq_high"].item() is True


def test_a_regime_is_null_until_its_window_is_complete() -> None:
    """A 252-session median taken over 40 sessions is a different statistic —
    a regime that flips because its window was short is an artefact."""
    sessions = _sessions(date(2020, 1, 1), 400)
    frame = _build(sessions).sort("session_idx")
    for spec in REGIME_SPECS:
        values = frame[f"z_{spec.regime_id}"].to_list()
        assert all(v is None for v in values[: spec.min_sessions - 1]), spec.regime_id
        assert values[spec.min_sessions - 1] is not None, spec.regime_id
    # The short-window regimes start well before the long-window ones.
    assert SHORT_WINDOW < LONG_WINDOW


def test_a_difference_needs_one_more_session_than_a_rolling_window() -> None:
    """``VIX_t - VIX_{t-20}`` reaches back to row t-20, so it needs 21 rows; a
    20-session rolling sum needs 20. Conflating the two puts a regime's first
    value one session in the wrong place."""
    by_id = {spec.regime_id: spec for spec in REGIME_SPECS}
    assert by_id["vix_up"].min_sessions == SHORT_WINDOW + 1  # diff_20_sessions
    assert by_id["krw_weak_20d"].min_sessions == SHORT_WINDOW + 1  # log_ratio_20_sessions
    assert by_id["market_up"].min_sessions == LONG_WINDOW + 1  # log_ratio_252_sessions
    assert by_id["kosdaq_rel_up"].min_sessions == SHORT_WINDOW  # sum20_diff
    assert by_id["vix_high"].min_sessions == LONG_WINDOW  # minus_median_252_sessions
    assert by_id["liq_high"].min_sessions == LONG_WINDOW


def test_windows_are_counted_in_sessions_not_in_fact_dates() -> None:
    """§2.1: the fact's own axis is every weekday for 2014-2023, so a KRX
    holiday sits in it carrying the previous value. Differencing there would
    make the window a different length; this counts sessions."""
    weekdays = _sessions(date(2020, 1, 1), 60)
    holidays = {weekdays[5], weekdays[6], weekdays[7]}
    sessions = [d for d in weekdays if d not in holidays]
    vix = [20.0 + 0.5 * i for i in range(len(weekdays))]
    frame = _build(sessions, fact_dates=weekdays, global_vix_level=vix)

    assert frame.height == len(sessions)
    by_date = {d: v for d, v in zip(sessions, vix[:0] or [], strict=False)}
    del by_date
    session_vix = [vix[weekdays.index(d)] for d in sessions]
    row = frame.filter(pl.col("session_idx") == 40)
    # 20 *sessions* back, not 20 weekdays: the three holiday rows are gone.
    assert row["z_vix_up"].item() == pytest.approx(session_vix[39] - session_vix[19])


def test_the_alternative_cut_is_against_the_trailing_median_not_zero() -> None:
    """§6.2 (gate G6): a diagnostic partition, never the judged one."""
    sessions = _sessions(date(2020, 1, 1), 700)
    # A drifting VIX makes the 20-session change positive most of the time, so
    # "> 0" and "> its own trailing median" partition the sample differently —
    # which is the whole point of carrying both.
    vix = [20.0 + 0.05 * i + 0.3 * math.sin(i * 0.05) for i in range(len(sessions))]
    frame = _build(sessions, global_vix_level=vix).drop_nulls("alt_s_vix_up")
    assert frame.height > 0
    assert frame["s_vix_up"].mean() == 1.0  # always "up" against zero
    assert 0.3 < frame["alt_s_vix_up"].mean() < 0.7  # balanced against its median
    assert (frame["s_vix_up"] != frame["alt_s_vix_up"]).any()


# --- §5 G1 / G2, §6.5 persistence -------------------------------------------


def _regime_frame(values: list[bool | None], *, regime: str = "vix_up") -> pl.DataFrame:
    days = _sessions(date(2020, 1, 1), len(values))
    data = {"trade_date": days, "session_idx": list(range(1, len(values) + 1))}
    for rid in REGIME_IDS:
        data[f"s_{rid}"] = values if rid == regime else [None] * len(values)
    return pl.DataFrame(data)


def test_g1_needs_both_sides_to_clear_the_count_and_the_share() -> None:
    balanced = _regime_frame([i % 2 == 0 for i in range(1000)])
    lopsided = _regime_frame([i % 20 == 0 for i in range(1000)])

    by_id = {row["regime_id"]: row for row in regime_occupancy(balanced)}
    assert by_id["vix_up"]["g1_pass"] is True
    assert by_id["vix_up"]["n_dates_s1"] == 500
    assert by_id["vix_up"]["share_s1"] == pytest.approx(0.5)

    by_id = {row["regime_id"]: row for row in regime_occupancy(lopsided)}
    assert by_id["vix_up"]["g1_pass"] is False  # 50 of 1000 — 5% share


def test_g1_ignores_sessions_where_the_regime_is_undefined() -> None:
    frame = _regime_frame([None] * 300 + [i % 2 == 0 for i in range(1000)])
    row = next(r for r in regime_occupancy(frame) if r["regime_id"] == "vix_up")
    assert row["n_dates"] == 1000
    assert row["first_date"] is not None


def test_persistence_counts_transitions_and_mean_run_length() -> None:
    """§6.5: the number that says how small the effective sample really is."""
    # 1,1,1,0,0,1,1,1,0,0 -> runs of True: 3,3 ; runs of False: 2,2
    frame = _regime_frame([True] * 3 + [False] * 2 + [True] * 3 + [False] * 2)
    row = next(r for r in regime_persistence(frame) if r["regime_id"] == "vix_up")
    assert row["n_regime_transitions"] == 3
    assert row["n_runs_s1"] == 2
    assert row["n_runs_s0"] == 2
    assert row["mean_run_length_s1"] == pytest.approx(3.0)
    assert row["mean_run_length_s0"] == pytest.approx(2.0)


def test_persistence_of_a_never_switching_regime_is_one_long_run() -> None:
    frame = _regime_frame([True] * 100)
    row = next(r for r in regime_persistence(frame) if r["regime_id"] == "vix_up")
    assert row["n_regime_transitions"] == 0
    assert row["n_runs_s1"] == 1
    assert row["mean_run_length_s1"] == pytest.approx(100.0)
    assert math.isnan(row["mean_run_length_s0"])


_PERIODS = [
    {"id": "early", "start": "2020-01-01", "end": "2020-03-31"},
    {"id": "late", "start": "2020-04-01", "end": "common_formation_end"},
]


def test_g2_marks_a_subperiod_valid_only_when_both_sides_clear_the_floor() -> None:
    days = _sessions(date(2020, 1, 1), 200)
    # First 65 sessions alternate; the rest are constantly True.
    values = [i % 2 == 0 for i in range(65)] + [True] * (len(days) - 65)
    frame = _regime_frame(values)
    frame = frame.with_columns(pl.Series("trade_date", days))

    rows = regime_subperiod_counts(
        frame,
        _PERIODS,
        placeholders={"common_formation_end": days[-1]},
        min_dates_per_regime=25,
    )
    by_period = {r["period_id"]: r for r in rows if r["regime_id"] == "vix_up"}
    assert by_period["early"]["g2_valid"] is True
    # The late window is one-sided, so it cannot show a sign either way.
    assert by_period["late"]["n_dates_s0"] == 0
    assert by_period["late"]["g2_valid"] is False


def test_g2_resolves_the_common_formation_end_placeholder() -> None:
    days = _sessions(date(2020, 1, 1), 200)
    frame = _regime_frame([i % 2 == 0 for i in range(len(days))]).with_columns(
        pl.Series("trade_date", days)
    )
    rows = regime_subperiod_counts(
        frame, _PERIODS, placeholders={"common_formation_end": days[100]}
    )
    late = next(r for r in rows if r["period_id"] == "late" and r["regime_id"] == "vix_up")
    assert late["period_end"] == days[100]
    assert late["n_dates"] <= 101


def test_summary_renders_all_three_tables() -> None:
    days = _sessions(date(2020, 1, 1), 600)
    frame = _regime_frame([i % 2 == 0 for i in range(len(days))]).with_columns(
        pl.Series("trade_date", days)
    )
    summary = summarize_regimes(frame, _PERIODS, placeholders={"common_formation_end": days[-1]})
    assert set(summary) == {"occupancy", "persistence", "subperiods"}
    markdown = render_regime_summary_md(summary)
    assert "G1 국면 점유율" in markdown
    assert "국면 지속" in markdown
    assert "G2 구간별 유효 여부" in markdown
    for rid in REGIME_IDS:
        assert f"`{rid}`" in markdown


def test_the_series_can_be_trimmed_to_the_judged_window() -> None:
    """Every preregistered pair is a ``common_survivor`` cell, so its daily IC
    stops at ``common_formation_end``; occupancy past that counts sessions
    Phase C never conditions on."""
    sessions = _sessions(date(2020, 1, 1), 400)
    con = duckdb.connect()
    con.execute("CREATE TABLE label_scan (trade_date DATE)")
    con.executemany("INSERT INTO label_scan VALUES (?)", [(d,) for d in sessions])
    con.execute(
        "CREATE TABLE common_feature_daily_fact (feature_date DATE, feature_code VARCHAR, "
        "value_numeric DOUBLE, asof_available_date DATE)"
    )
    con.executemany(
        "INSERT INTO common_feature_daily_fact VALUES (?,?,?,?)",
        [(d, code, _DEFAULTS[code], d) for code in SOURCE_FEATURE_CODES for d in sessions],
    )
    full = build_regime_series(con)
    trimmed = build_regime_series(con, sample_start=sessions[100], sample_end=sessions[300])
    assert full.height == len(sessions)
    assert trimmed.height == 201
    assert trimmed["trade_date"].min() == sessions[100]
    assert trimmed["trade_date"].max() == sessions[300]
