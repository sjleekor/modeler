from __future__ import annotations

import math
from datetime import date, timedelta

import duckdb
import polars as pl
import pytest
from modeler.analysis.horizon_scan_daily_ic import (
    DAILY_IC_DIR_NAME,
    DAILY_SPREAD_DIR_NAME,
    ParquetDailyIcSink,
    reconcile_daily_ic,
)
from modeler.analysis.horizon_scan_phase_b_scan import (
    apply_combined_ab_bh,
    apply_phase_b_only_bh,
    assemble_phase_b_primary_table,
    assert_scan_matches_ready_population,
    build_event_cohort_frame_sql,
    build_phase_b_panel_sql,
    compute_phase_b_evidence_grade,
    compute_phase_b_period_sign_pass,
    compute_phase_b_screen_pass,
    phase_b_primary_stats_rows,
    register_phase_b_panel,
    run_phase_b_continuous_scan,
    run_phase_b_event_scan,
    scan_event_cohort_cell,
)
from modeler.analysis.horizon_scan_runner import register_analysis_panel

# --- panel join ---


def _seed_a0_marts(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE label_scan AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', 0.01, true, true),
            (DATE '2024-01-02', 'B', 'KOSDAQ', 0.02, true, false),
            (DATE '2024-01-03', 'A', 'KOSPI', -0.01, true, true)
        ) AS t(trade_date, ticker, market, raw_label_5d, common_formation_120d,
               common_survivor_120d)
    """)
    con.execute("""
        CREATE TABLE feat_price AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', 0.10, 0.09),
            (DATE '2024-01-02', 'B', 'KOSDAQ', 0.20, 0.19),
            (DATE '2024-01-03', 'A', 'KOSPI', 0.11, 0.10)
        ) AS t(trade_date, ticker, market, px_reversal_5d, px_reversal_5d_lag1)
    """)
    con.execute("""
        CREATE TABLE feat_flow AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', 0.5, 'allowed'),
            (DATE '2024-01-02', 'B', 'KOSDAQ', 0.6, 'banned'),
            (DATE '2024-01-03', 'A', 'KOSPI', 0.4, 'allowed')
        ) AS t(trade_date, ticker, market, flow_foreign_netbuy_to_volume_20d, short_regime)
    """)
    con.execute("ALTER TABLE feat_flow ADD COLUMN short_balance_is_available BOOLEAN DEFAULT true")
    con.execute("""
        CREATE TABLE dim_universe_broad_daily AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', true, false, true),
            (DATE '2024-01-02', 'B', 'KOSDAQ', true, false, true),
            (DATE '2024-01-03', 'A', 'KOSPI', true, false, true)
        ) AS t(trade_date, ticker, market, in_universe, membership_reconstruction_available,
               management_filter_available)
    """)
    con.execute("""
        CREATE TABLE dim_universe_tradable_daily AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', true, false, true),
            (DATE '2024-01-02', 'B', 'KOSDAQ', false, false, true),
            (DATE '2024-01-03', 'A', 'KOSPI', true, false, true)
        ) AS t(trade_date, ticker, market, in_universe, membership_reconstruction_available,
               management_filter_available)
    """)
    con.execute("""
        CREATE TABLE dim_stock_pit_daily AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', 1000.0, 100.0, 900.0, 90000.0, true, false),
            (DATE '2024-01-02', 'B', 'KOSDAQ', 2000.0, 0.0, 2000.0, 400000.0, true, false),
            (DATE '2024-01-03', 'A', 'KOSPI', 1000.0, 100.0, 900.0, 90900.0, true, false)
        ) AS t(trade_date, ticker, market, issued_shares_pit, treasury_shares_pit,
               float_shares_pit, market_cap_pit, shares_is_available, shares_invalid_flag)
    """)
    con.execute("""
        CREATE TABLE dim_price_quality_daily AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', 100, false, false, 0.01, 0.00995,
             false, false, 0.0, false, 'allowed', true),
            (DATE '2024-01-02', 'B', 'KOSDAQ', 50, false, false, 0.02, 0.0198,
             false, false, 0.0, false, 'banned', true),
            (DATE '2024-01-03', 'A', 'KOSPI', 101, false, false, -0.01, -0.01005,
             false, false, 0.0, false, 'allowed', true)
        ) AS t(trade_date, ticker, market, valid_session_idx, is_halted, volume_zero,
               simple_ret, log_ret, ca_mask, ca_event, ca_event_cumulative,
               ca_rule_applicability_unknown, short_regime, short_balance_is_available)
    """)


def test_build_phase_b_panel_sql_joins_fin_and_event_scan_marts() -> None:
    con = duckdb.connect()
    _seed_a0_marts(con)
    con.execute("""
        CREATE TABLE feat_fin_scan_daily AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', -3.5),
            (DATE '2024-01-02', 'B', 'KOSDAQ', -4.5),
            (DATE '2024-01-03', 'A', 'KOSPI', -3.4)
        ) AS t(trade_date, ticker, market, fin_log_mcap)
    """)
    con.execute("""
        CREATE TABLE feat_event_scan_daily AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', 0.01),
            (DATE '2024-01-02', 'B', 'KOSDAQ', 0.02),
            (DATE '2024-01-03', 'A', 'KOSPI', 0.015)
        ) AS t(trade_date, ticker, market, ev_payout_yield)
    """)
    register_analysis_panel(con)
    view = register_phase_b_panel(con)
    rows = con.execute(f"SELECT * FROM {view} ORDER BY trade_date, ticker").fetchdf()
    assert len(rows) == 3  # matches label_scan/analysis_panel's grain, no fan-out
    assert {"fin_log_mcap", "ev_payout_yield", "px_reversal_5d", "in_broad"} <= set(rows.columns)


def test_build_phase_b_panel_sql_accepts_view_name_overrides() -> None:
    sql = build_phase_b_panel_sql(fin_scan_view="custom_fin_scan")
    assert "LEFT JOIN custom_fin_scan fs USING" in sql
    assert "FROM analysis_panel ap" in sql


def test_build_phase_b_panel_sql_omits_join_when_view_is_none() -> None:
    """A mart the real-lake orchestrator could not materialize (e.g.
    ``feat_event_scan_daily`` blocked by a missing raw source) is passed as
    ``None`` rather than a name — the SQL must not reference it at all, so the
    *other* mart's families are not collaterally blocked by a join against a
    nonexistent view."""
    sql = build_phase_b_panel_sql(event_scan_view=None)
    assert "feat_event_scan_daily" not in sql
    assert "es.*" not in sql
    assert "LEFT JOIN feat_fin_scan_daily fs USING" in sql


def test_register_phase_b_panel_works_with_only_one_mart_available() -> None:
    con = duckdb.connect()
    _seed_a0_marts(con)
    con.execute("""
        CREATE TABLE feat_fin_scan_daily AS SELECT * FROM (VALUES
            (DATE '2024-01-02', 'A', 'KOSPI', -3.5),
            (DATE '2024-01-02', 'B', 'KOSDAQ', -4.5),
            (DATE '2024-01-03', 'A', 'KOSPI', -3.4)
        ) AS t(trade_date, ticker, market, fin_log_mcap)
    """)
    register_analysis_panel(con)
    view = register_phase_b_panel(con, event_scan_view=None)
    rows = con.execute(f"SELECT * FROM {view} ORDER BY trade_date, ticker").fetchdf()
    assert len(rows) == 3
    assert "fin_log_mcap" in rows.columns
    assert "ev_payout_yield" not in rows.columns


# --- continuous families: wrapper around Phase A's runner ---


def _seed_continuous_panel(con: duckdb.DuckDBPyConnection, *, n_sessions: int = 70) -> None:
    """Same shape/spirit as test_horizon_scan_scan_cell.py's _seed_scan_panel,
    with the feature column renamed to a Phase B name — this test only
    exercises the wrapper's scan_type derivation and pass-through, not
    scan_cell's own math (already covered by test_horizon_scan_scan_cell.py).
    """
    start = date(2024, 1, 1)
    rows = []
    for session in range(1, n_sessions + 1):
        d = start + timedelta(days=session - 1)
        for market in ("KOSPI", "KOSDAQ"):
            for t in range(10):
                ticker = f"{market[:1]}{t}"
                feature = float(t) + 0.01 * session
                wobble = 3.0 * math.sin(t + 0.7 * session)
                raw_label = float(t) * 2.0 + wobble
                rows.append(
                    (
                        d,
                        ticker,
                        market,
                        session,
                        feature,
                        raw_label,
                        raw_label,
                        True,
                        True,
                        t < 8,
                        True,
                        t != 9,
                        False,
                    )
                )
    con.execute("""
        CREATE TABLE analysis_panel_phase_b (
            trade_date DATE, ticker VARCHAR, market VARCHAR, formation_session_idx BIGINT,
            fin_log_mcap DOUBLE, y_rank_60d DOUBLE, raw_label_60d DOUBLE, label_ok_60d BOOLEAN,
            in_broad BOOLEAN, in_tradable BOOLEAN, common_formation_120d BOOLEAN,
            common_survivor_120d BOOLEAN, ca_mask BOOLEAN
        )
    """)
    con.executemany("INSERT INTO analysis_panel_phase_b VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)


def test_run_phase_b_continuous_scan_derives_scan_type_and_runs_all_combos() -> None:
    con = duckdb.connect()
    _seed_continuous_panel(con)
    cell = {
        "hypothesis_id": "fin_log_mcap|fin_log_mcap|cum|0|60",
        "family": "fin_log_mcap",
        "feature": "fin_log_mcap",
        "cell_type": "cumulative",
        "h_start": 0,
        "h_end": 60,
        "expected_sign": "-",
    }
    rows = run_phase_b_continuous_scan(
        con,
        [cell],
        sample_start="2024-01-01",
        min_names=5,
        min_names_for_spread=5,
        quantile_count=5,
        min_dates_per_cell=30,
    )
    assert len(rows) == 4  # broad/tradable x common_survivor/available
    assert {r["scan_type"] for r in rows} == {"cum"}
    assert {(r["universe"], r["sample_kind"]) for r in rows} == {
        ("broad", "common_survivor"),
        ("broad", "available"),
        ("tradable", "common_survivor"),
        ("tradable", "available"),
    }
    broad_common = next(
        r for r in rows if r["universe"] == "broad" and r["sample_kind"] == "common_survivor"
    )
    assert broad_common["status"] == "valid"
    assert broad_common["feature"] == "fin_log_mcap"


# --- SUE event cohort scan (§5.4) ---


def _weekdays(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current = current + timedelta(days=1)
    return days


def _seed_calendar(con: duckdb.DuckDBPyConnection, days: list[date]) -> None:
    con.execute("CREATE TABLE daily_ohlcv (trade_date DATE)")
    con.executemany("INSERT INTO daily_ohlcv VALUES (?)", [(d,) for d in days])


def _seed_event_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE fin_sue_event (
            ticker VARCHAR, market VARCHAR, event_formation_date DATE,
            bsns_year INTEGER, reprt_code VARCHAR,
            fin_sue DOUBLE, bucket_0_3_excess DOUBLE,
            is_primary_constant_sample BOOLEAN
        )
    """)


def _insert_events(con: duckdb.DuckDBPyConnection, events: list[tuple]) -> None:
    con.executemany("INSERT INTO fin_sue_event VALUES (?,?,?,?,?,?,?,?)", events)


def _cohort_events(
    formation_date: date,
    market: str,
    n: int,
    *,
    reversed_rank: bool,
    bsns_year: int,
    reprt_code: str,
) -> list[tuple]:
    out = []
    for i in range(n):
        sue = float(i)
        excess = float(n - 1 - i) if reversed_rank else float(i)
        ticker = f"{market[:1]}{i}"
        out.append((ticker, market, formation_date, bsns_year, reprt_code, sue, excess, True))
    return out


_SCAN_KWARGS = dict(
    h_start=0,
    h_end=3,
    sample_start="2020-01-01",
    min_events_per_market_contribution=10,
    min_events_per_cohort_total=30,
    min_event_cohorts=8,
)


def test_scan_event_cohort_cell_pools_market_neutral_ranks_and_computes_ic() -> None:
    con = duckdb.connect()
    days = _weekdays(date(2020, 1, 6), 10)
    _seed_calendar(con, days)
    _seed_event_table(con)
    events: list[tuple] = []
    for idx, d in enumerate(days):
        window = ("2020", "11013") if idx < 5 else ("2020", "11012")
        reversed_rank = idx in (2, 5, 8)
        for market in ("KOSPI", "KOSDAQ"):
            events += _cohort_events(
                d,
                market,
                16,
                reversed_rank=reversed_rank,
                bsns_year=int(window[0]),
                reprt_code=window[1],
            )
    _insert_events(con, events)

    result = scan_event_cohort_cell(con, **_SCAN_KWARGS)

    assert result["status"] == "valid", result["status_reason"]
    assert result["n_dates"] == 10
    assert result["n_obs"] == 10 * 2 * 16
    assert result["n_obs_mean"] == 16.0
    assert result["n_obs_min"] == 16
    assert result["n_obs_median"] == 16.0
    assert abs(result["kospi_weight_mean"] - 0.5) < 1e-9
    assert abs(result["kosdaq_weight_mean"] - 0.5) < 1e-9
    # 7 aligned dates (ic=+1) + 3 reversed dates (ic=-1) -> mean 0.4 exactly.
    assert abs(result["ic_mean"] - 0.4) < 1e-9
    assert result["n_independent_filing_windows"] == 2
    assert math.isfinite(result["t_nw"])
    assert 0.0 <= result["p_nw"] <= 1.0


def test_scan_event_cohort_cell_excludes_market_below_min_contribution() -> None:
    con = duckdb.connect()
    days = _weekdays(date(2020, 2, 3), 10)
    _seed_calendar(con, days)
    _seed_event_table(con)
    events: list[tuple] = []
    for d in days[:9]:
        for market in ("KOSPI", "KOSDAQ"):
            events += _cohort_events(
                d, market, 16, reversed_rank=False, bsns_year=2020, reprt_code="11013"
            )
    # 10th date: KOSPI has only 5 (< min_events_per_market_contribution) and
    # must be fully excluded from pooling; KOSDAQ alone has 30 (>= total gate).
    events += _cohort_events(
        days[9], "KOSPI", 5, reversed_rank=False, bsns_year=2020, reprt_code="11013"
    )
    events += _cohort_events(
        days[9], "KOSDAQ", 30, reversed_rank=False, bsns_year=2020, reprt_code="11013"
    )
    _insert_events(con, events)

    result = scan_event_cohort_cell(con, **_SCAN_KWARGS)

    assert result["status"] == "valid", result["status_reason"]
    assert result["n_dates"] == 10  # the 10th date still qualifies via KOSDAQ alone
    assert result["n_obs"] == 9 * 32 + 30  # the excluded 5-event KOSPI group never counts
    assert abs(result["kospi_weight_mean"] - 0.45) < 1e-9
    assert abs(result["kosdaq_weight_mean"] - 0.55) < 1e-9


def test_scan_event_cohort_cell_drops_date_below_pooled_total_gate() -> None:
    con = duckdb.connect()
    days = _weekdays(date(2020, 3, 2), 10)
    _seed_calendar(con, days)
    _seed_event_table(con)
    events: list[tuple] = []
    for d in days[:9]:
        for market in ("KOSPI", "KOSDAQ"):
            events += _cohort_events(
                d, market, 16, reversed_rank=False, bsns_year=2020, reprt_code="11013"
            )
    # 10th date: both markets individually clear the per-market gate (12>=10)
    # but the pooled total (24) never reaches min_events_per_cohort_total=30.
    for market in ("KOSPI", "KOSDAQ"):
        events += _cohort_events(
            days[9], market, 12, reversed_rank=False, bsns_year=2020, reprt_code="11013"
        )
    _insert_events(con, events)

    result = scan_event_cohort_cell(con, **_SCAN_KWARGS)

    assert result["status"] == "valid", result["status_reason"]
    assert result["n_dates"] == 9  # the 10th date is dropped from the IC series
    assert result["n_obs"] == 9 * 32 + 24  # but its qualifying groups still count as n_obs


def test_scan_event_cohort_cell_insufficient_when_cohorts_below_minimum() -> None:
    con = duckdb.connect()
    days = _weekdays(date(2020, 4, 1), 5)
    _seed_calendar(con, days)
    _seed_event_table(con)
    events: list[tuple] = []
    for d in days:
        for market in ("KOSPI", "KOSDAQ"):
            events += _cohort_events(
                d, market, 16, reversed_rank=False, bsns_year=2020, reprt_code="11013"
            )
    _insert_events(con, events)

    result = scan_event_cohort_cell(con, **_SCAN_KWARGS)

    assert result["status"] == "insufficient"
    assert result["status_reason"] == "insufficient_cohorts:5<8"
    assert result["n_dates"] == 5
    assert result["ic_mean"] is None


def test_scan_event_cohort_cell_secondary_sample_never_enters_primary_scan() -> None:
    con = duckdb.connect()
    days = _weekdays(date(2020, 5, 4), 10)
    _seed_calendar(con, days)
    _seed_event_table(con)
    events: list[tuple] = []
    for d in days:
        for market in ("KOSPI", "KOSDAQ"):
            for ticker, market_, formation, year, reprt, sue, excess, _primary in _cohort_events(
                d, market, 16, reversed_rank=False, bsns_year=2020, reprt_code="11013"
            ):
                events.append((ticker, market_, formation, year, reprt, sue, excess, False))
    _insert_events(con, events)

    result = scan_event_cohort_cell(con, **_SCAN_KWARGS)

    assert result["status"] == "insufficient"
    assert result["status_reason"] == "no_formation_rows"


def test_run_phase_b_event_scan_wraps_cell_registry_fields() -> None:
    con = duckdb.connect()
    days = _weekdays(date(2020, 6, 1), 10)
    _seed_calendar(con, days)
    _seed_event_table(con)
    events: list[tuple] = []
    for d in days:
        for market in ("KOSPI", "KOSDAQ"):
            events += _cohort_events(
                d, market, 16, reversed_rank=False, bsns_year=2020, reprt_code="11013"
            )
    _insert_events(con, events)
    cell = {
        "hypothesis_id": "fin_sue|fin_sue|event|0|3",
        "family": "fin_sue",
        "feature": "fin_sue",
        "cell_type": "event_bucket",
        "h_start": 0,
        "h_end": 3,
        "expected_sign": "+",
    }
    rows = run_phase_b_event_scan(
        con,
        [cell],
        sample_start="2020-01-01",
        min_events_per_market_contribution=10,
        min_events_per_cohort_total=30,
        min_event_cohorts=8,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["scan_type"] == "event_bucket"
    assert row["family"] == "fin_sue"
    assert row["hypothesis_id"] == "fin_sue|fin_sue|event|0|3"
    assert row["status"] == "valid"


# --- assembly across continuous + event cells ---


def _stat_row(hypothesis_id: str, **overrides) -> dict:
    # role="ready_primary" mirrors real usage: run_registry_scan/
    # run_phase_b_event_scan both spread the readiness cell's own fields
    # (including role) into every scanned row via {**cell, ...}.
    base = dict(
        hypothesis_id=hypothesis_id,
        family="fam",
        scan_type="cum",
        h_end=60,
        expected_sign="+",
        universe="broad",
        sample_kind="common_survivor",
        status="valid",
        p_nw=0.02,
        ic_mean=0.05,
        role="ready_primary",
    )
    base.update(overrides)
    return base


def test_assemble_phase_b_primary_table_picks_discovery_row_for_ready_cell() -> None:
    readiness_rows = [
        {"hypothesis_id": "a", "role": "ready_primary"},
        {"hypothesis_id": "b", "role": "blocked_exploratory"},
    ]
    scanned_rows = [
        _stat_row("a", universe="broad", sample_kind="common_survivor"),
        _stat_row("a", universe="tradable", sample_kind="available"),
    ]
    table = assemble_phase_b_primary_table(readiness_rows, scanned_rows)
    assert len(table) == 2
    assert table[0]["universe"] == "broad" and table[0]["sample_kind"] == "common_survivor"
    assert table[1]["status"] == "not_evaluated"
    assert table[1]["status_reason"] == "blocked_exploratory"

    primary = phase_b_primary_stats_rows(table)
    assert [r["hypothesis_id"] for r in primary] == ["a"]
    assert_scan_matches_ready_population(primary, readiness_rows)


def test_assemble_phase_b_primary_table_raises_on_missing_ready_scan() -> None:
    readiness_rows = [{"hypothesis_id": "a", "role": "ready_primary"}]
    with pytest.raises(ValueError, match="never scanned"):
        assemble_phase_b_primary_table(readiness_rows, [])


def test_assemble_phase_b_primary_table_uses_the_sole_event_row_directly() -> None:
    readiness_rows = [{"hypothesis_id": "fin_sue|fin_sue|event|0|3", "role": "ready_primary"}]
    scanned_rows = [_stat_row("fin_sue|fin_sue|event|0|3", scan_type="event_bucket")]
    table = assemble_phase_b_primary_table(readiness_rows, scanned_rows)
    assert table[0]["scan_type"] == "event_bucket"


# --- B-9 BH steps ---


def test_apply_phase_b_only_bh_renames_fields_and_matches_manual_bh() -> None:
    rows = [_stat_row(f"h{i}", p_nw=p, family=f"fam{i}") for i, p in enumerate([0.01, 0.5, 0.9])]
    out = apply_phase_b_only_bh(rows, q_threshold=0.10)
    assert {"q_fdr_phase_b", "bh_pass_phase_b", "primary_discovery_phase_b"} <= set(out[0].keys())
    assert "q_fdr_global" not in out[0]
    best = next(r for r in out if r["hypothesis_id"] == "h0")
    assert best["bh_pass_phase_b"] is True


def test_apply_combined_ab_bh_pools_both_phases_without_id_collision() -> None:
    phase_a_rows = [_stat_row(f"a{i}", family=f"px{i}", p_nw=p) for i, p in enumerate([0.001, 0.6])]
    phase_b_rows = [_stat_row(f"b{i}", family=f"fin{i}", p_nw=p) for i, p in enumerate([0.6, 0.7])]
    out = apply_combined_ab_bh(phase_a_rows, phase_b_rows, q_threshold=0.10)
    assert len(out) == 4
    assert {"q_fdr_global_ab", "bh_pass_ab", "primary_discovery_ab"} <= set(out[0].keys())
    strong = next(r for r in out if r["hypothesis_id"] == "a0")
    assert strong["bh_pass_ab"] is True
    weak_ids = {"a1", "b0", "b1"}
    by_id = {r["hypothesis_id"]: r for r in out}
    assert all(not by_id[wid]["bh_pass_ab"] for wid in weak_ids)


def test_build_event_cohort_frame_sql_uses_the_requested_bucket_columns() -> None:
    sql = build_event_cohort_frame_sql(h_start=5, h_end=10, sample_start="2020-01-01")
    assert "e.bucket_5_10_excess" in sql
    assert "e.is_primary_constant_sample" in sql


# --- screen_pass (§9 B-9) ---


def test_compute_phase_b_period_sign_pass_fails_hard_at_one_or_zero_valid_periods() -> None:
    """§9 rule 4's "1개 이하는 실패" — Phase A's own compute_period_sign_pass
    would pass a single sign-consistent period (consistent > 0.5); Phase B's
    variant must not."""
    zero = compute_phase_b_period_sign_pass([None, None], expected_sign="+")
    assert zero == {
        "valid_subperiods": 0,
        "sign_consistent_subperiods": 0,
        "period_sign_pass": False,
    }
    one_consistent = compute_phase_b_period_sign_pass([0.05, None], expected_sign="+")
    assert one_consistent["valid_subperiods"] == 1
    assert one_consistent["period_sign_pass"] is False


def test_compute_phase_b_period_sign_pass_two_periods_requires_both_to_agree() -> None:
    both_agree = compute_phase_b_period_sign_pass([0.05, 0.02], expected_sign="+")
    assert both_agree["period_sign_pass"] is True
    one_disagrees = compute_phase_b_period_sign_pass([0.05, -0.02], expected_sign="+")
    assert one_disagrees["period_sign_pass"] is False


def test_compute_phase_b_period_sign_pass_three_periods_uses_strict_majority() -> None:
    majority = compute_phase_b_period_sign_pass([0.05, 0.02, -0.01], expected_sign="+")
    assert majority["sign_consistent_subperiods"] == 2
    assert majority["period_sign_pass"] is True
    tie_not_possible_but_minority_fails = compute_phase_b_period_sign_pass(
        [0.05, -0.02, -0.01], expected_sign="+"
    )
    assert tie_not_possible_but_minority_fails["period_sign_pass"] is False


_ALL_PASS_KWARGS = dict(
    role="ready_primary",
    primary_discovery=True,
    isolated_spike=False,
    tradable_pass=True,
    period_sign_pass=True,
    available_direction_pass=True,
    robustness_required=False,
    robustness_pass=None,
)


def test_compute_phase_b_screen_pass_blocked_role_never_evaluated() -> None:
    out = compute_phase_b_screen_pass(**{**_ALL_PASS_KWARGS, "role": "blocked_exploratory"})
    assert out == {"screen_pass": False, "not_applicable_role": True, "failed_gates": []}


def test_compute_phase_b_screen_pass_all_gates_clear() -> None:
    out = compute_phase_b_screen_pass(**_ALL_PASS_KWARGS)
    assert out["screen_pass"] is True
    assert out["not_applicable_role"] is False
    assert out["failed_gates"] == []


@pytest.mark.parametrize(
    "overrides,expected_failed_gate",
    [
        ({"primary_discovery": False}, "primary_discovery"),
        ({"isolated_spike": True}, "isolated_spike_clear"),
        ({"tradable_pass": False}, "tradable_pass"),
        ({"period_sign_pass": False}, "period_sign_pass"),
        ({"available_direction_pass": False}, "available_direction_pass"),
    ],
)
def test_compute_phase_b_screen_pass_each_rule_can_fail_independently(
    overrides: dict, expected_failed_gate: str
) -> None:
    out = compute_phase_b_screen_pass(**{**_ALL_PASS_KWARGS, **overrides})
    assert out["screen_pass"] is False
    assert out["failed_gates"] == [expected_failed_gate]


def test_compute_phase_b_screen_pass_available_direction_none_is_skipped_not_failed() -> None:
    """A cell without a computable available-sample IC (e.g. insufficient
    coverage) skips rule 5 rather than failing it — mirrors Phase A's own
    ``compute_screen_pass`` treatment of the same optional gate."""
    out = compute_phase_b_screen_pass(**{**_ALL_PASS_KWARGS, "available_direction_pass": None})
    assert out["screen_pass"] is True
    assert "available_direction_pass" not in out["failed_gates"]


def test_compute_phase_b_screen_pass_robustness_gate_only_applies_when_required() -> None:
    not_required = compute_phase_b_screen_pass(
        **{**_ALL_PASS_KWARGS, "robustness_required": False, "robustness_pass": False}
    )
    assert not_required["screen_pass"] is True  # gate not evaluated at all

    required_and_failed = compute_phase_b_screen_pass(
        **{**_ALL_PASS_KWARGS, "robustness_required": True, "robustness_pass": False}
    )
    assert required_and_failed["screen_pass"] is False
    assert required_and_failed["failed_gates"] == ["robustness_pass"]

    required_and_passed = compute_phase_b_screen_pass(
        **{**_ALL_PASS_KWARGS, "robustness_required": True, "robustness_pass": True}
    )
    assert required_and_passed["screen_pass"] is True


def test_compute_phase_b_screen_pass_ca_holdout_policy_default_true() -> None:
    out = compute_phase_b_screen_pass(
        role="ready_primary",
        primary_discovery=True,
        isolated_spike=False,
        tradable_pass=True,
        period_sign_pass=True,
        available_direction_pass=None,
        robustness_required=False,
        robustness_pass=None,
    )
    assert out["screen_pass"] is True

    out_fail = compute_phase_b_screen_pass(**{**_ALL_PASS_KWARGS, "ca_holdout_policy_pass": False})
    assert out_fail["screen_pass"] is False
    assert out_fail["failed_gates"] == ["ca_holdout_policy_pass"]


# --- evidence grade (§9 B-9) ---

_GRADE_ALL_PASS_KWARGS = dict(
    role="ready_primary",
    family="fin_log_mcap",
    screen_pass=True,
    failed_gates=[],
    valid_subperiods=3,
    all_offsets_evaluable=True,
    # Stated rather than defaulted: the parameter is fail-closed, so leaving it
    # out is itself a grade-A cap (see the source-quality tests below).
    source_quality_status="ok",
)


def test_compute_phase_b_evidence_grade_blocked_role_is_not_evaluated() -> None:
    out = compute_phase_b_evidence_grade(
        **{**_GRADE_ALL_PASS_KWARGS, "role": "blocked_exploratory", "screen_pass": False}
    )
    assert out == "NE"


def test_compute_phase_b_evidence_grade_available_sign_flip_is_grade_c() -> None:
    out = compute_phase_b_evidence_grade(
        **{
            **_GRADE_ALL_PASS_KWARGS,
            "screen_pass": False,
            "failed_gates": ["available_direction_pass"],
        }
    )
    assert out == "C"


def test_compute_phase_b_evidence_grade_robustness_failure_is_grade_c() -> None:
    out = compute_phase_b_evidence_grade(
        **{**_GRADE_ALL_PASS_KWARGS, "screen_pass": False, "failed_gates": ["robustness_pass"]}
    )
    assert out == "C"


def test_compute_phase_b_evidence_grade_clean_pass_is_grade_a() -> None:
    assert compute_phase_b_evidence_grade(**_GRADE_ALL_PASS_KWARGS) == "A"


def test_compute_phase_b_evidence_grade_exactly_two_periods_caps_at_b() -> None:
    out = compute_phase_b_evidence_grade(**{**_GRADE_ALL_PASS_KWARGS, "valid_subperiods": 2})
    assert out == "B"


def test_compute_phase_b_evidence_grade_pit_industry_family_caps_at_b() -> None:
    out = compute_phase_b_evidence_grade(**{**_GRADE_ALL_PASS_KWARGS, "family": "fin_value_z"})
    assert out == "B"
    # size (not PIT-industry-dependent) is unaffected
    assert compute_phase_b_evidence_grade(**_GRADE_ALL_PASS_KWARGS) == "A"


def test_compute_phase_b_evidence_grade_source_quality_warning_caps_at_b() -> None:
    """§9's source warning is non-fatal: it removes grade A, nothing more."""
    out = compute_phase_b_evidence_grade(
        **{**_GRADE_ALL_PASS_KWARGS, "source_quality_status": "warn"}
    )
    assert out == "B"


def test_compute_phase_b_evidence_grade_unmeasured_source_quality_caps_at_b() -> None:
    # "We could not check" is not a basis for the strongest claim, so it caps
    # exactly like a breached threshold does.
    out = compute_phase_b_evidence_grade(
        **{**_GRADE_ALL_PASS_KWARGS, "source_quality_status": "unmeasured"}
    )
    assert out == "B"


def test_compute_phase_b_evidence_grade_absent_source_quality_caps_at_b() -> None:
    kwargs = {k: v for k, v in _GRADE_ALL_PASS_KWARGS.items() if k != "source_quality_status"}
    # A run that produced no diagnostic at all cannot use its absence as
    # evidence of quality — the parameter is fail-closed by default.
    assert compute_phase_b_evidence_grade(**kwargs) == "B"


def test_compute_phase_b_evidence_grade_not_applicable_source_quality_allows_a() -> None:
    # A family that never reads the metric layer has nothing to warn about.
    out = compute_phase_b_evidence_grade(
        **{**_GRADE_ALL_PASS_KWARGS, "source_quality_status": "not_applicable"}
    )
    assert out == "A"


def test_compute_phase_b_evidence_grade_source_warning_never_lowers_below_b() -> None:
    # Screen-pass failure already routes to D; a source warning on top must not
    # change that, and must not turn a C into something worse either.
    assert (
        compute_phase_b_evidence_grade(
            **{
                **_GRADE_ALL_PASS_KWARGS,
                "screen_pass": False,
                "failed_gates": ["primary_discovery"],
                "source_quality_status": "warn",
            }
        )
        == "D"
    )
    assert (
        compute_phase_b_evidence_grade(
            **{
                **_GRADE_ALL_PASS_KWARGS,
                "screen_pass": False,
                "failed_gates": ["robustness_pass"],
                "source_quality_status": "warn",
            }
        )
        == "C"
    )


def test_compute_phase_b_evidence_grade_offset_not_fully_evaluable_caps_at_b() -> None:
    out = compute_phase_b_evidence_grade(
        **{**_GRADE_ALL_PASS_KWARGS, "all_offsets_evaluable": False}
    )
    assert out == "B"


def test_compute_phase_b_evidence_grade_screen_fail_other_reason_is_grade_d() -> None:
    out = compute_phase_b_evidence_grade(
        **{**_GRADE_ALL_PASS_KWARGS, "screen_pass": False, "failed_gates": ["primary_discovery"]}
    )
    assert out == "D"


def test_compute_phase_b_evidence_grade_insufficient_filing_windows_caps_at_b() -> None:
    """§6 B-8 SUE point 5: too few independent filing windows blocks grade A
    even though every other condition (incl. screen_pass) is clean."""
    out = compute_phase_b_evidence_grade(
        **{
            **_GRADE_ALL_PASS_KWARGS,
            "n_independent_filing_windows": 10,
            "grade_a_min_independent_filing_windows": 20,
        }
    )
    assert out == "B"


def test_compute_phase_b_evidence_grade_sufficient_filing_windows_allows_a() -> None:
    out = compute_phase_b_evidence_grade(
        **{
            **_GRADE_ALL_PASS_KWARGS,
            "n_independent_filing_windows": 25,
            "grade_a_min_independent_filing_windows": 20,
        }
    )
    assert out == "A"


def test_compute_phase_b_evidence_grade_filing_windows_none_is_not_applicable() -> None:
    """Continuous cells never compute ``n_independent_filing_windows`` — the
    default ``None`` must not itself block grade A."""
    assert compute_phase_b_evidence_grade(**_GRADE_ALL_PASS_KWARGS) == "A"


def test_run_phase_b_continuous_scan_hands_the_daily_sink_through(tmp_path) -> None:
    """Stage 0 through the Phase B wrapper: the sink sees Phase-B-shaped
    registry rows (``cell_type``/``role``) and stores them under one normalized
    identity."""
    con = duckdb.connect()
    _seed_continuous_panel(con)
    sink = ParquetDailyIcSink(tmp_path)
    rows = run_phase_b_continuous_scan(
        con,
        [
            {
                "hypothesis_id": "fin_log_mcap|fin_log_mcap|cum|0|60",
                "family": "fin_log_mcap",
                "feature": "fin_log_mcap",
                "cell_type": "cumulative",
                "h_start": 0,
                "h_end": 60,
                "expected_sign": "-",
                "role": "ready_primary",
            }
        ],
        sample_start="2024-01-01",
        min_names=5,
        min_names_for_spread=5,
        quantile_count=5,
        min_dates_per_cell=30,
        daily_sink=sink,
    )
    summary = sink.finalize()
    assert summary.daily_ic.file_count == 1
    stored = pl.read_parquet(
        tmp_path / DAILY_IC_DIR_NAME / "family=fin_log_mcap" / "fin_log_mcap.parquet"
    )
    assert set(stored["hypothesis_role"].to_list()) == {"ready_primary"}
    assert set(stored["scan_type"].to_list()) == {"cum"}
    result = reconcile_daily_ic(
        rows,
        daily_ic_dir=tmp_path / DAILY_IC_DIR_NAME,
        daily_spread_dir=tmp_path / DAILY_SPREAD_DIR_NAME,
    )
    assert result["reconciled"] is True
    assert result["n_cells"] == 4
