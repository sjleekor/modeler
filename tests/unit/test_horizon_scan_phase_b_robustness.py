from __future__ import annotations

import math
from datetime import date, timedelta

import duckdb
import polars as pl
import pytest
from modeler.analysis.horizon_scan_phase_b_robustness import (
    _nonoverlap_min_dates_for_cell,
    compute_nonoverlap_robustness_pass,
    evaluate_sue_cluster_confirmation,
    run_filing_cycle_block_bootstrap,
    run_issuer_cluster_bootstrap,
    run_phase_b_continuous_nonoverlap,
    run_phase_b_temporal_placebo,
    select_phase_b_long_horizon_cells,
)
from modeler.analysis.horizon_scan_runner import scan_cell

# --- pure logic: non-overlap min-dates override / robustness gate ---


def test_nonoverlap_min_dates_for_cell_overrides_h120_cells() -> None:
    overrides = {"default": 20, "cumulative_120": 12, "bucket_60_120": 12}
    assert _nonoverlap_min_dates_for_cell({"h_start": 0, "h_end": 120}, "cum", overrides) == 12
    assert _nonoverlap_min_dates_for_cell({"h_start": 60, "h_end": 120}, "bucket", overrides) == 12
    assert _nonoverlap_min_dates_for_cell({"h_start": 0, "h_end": 60}, "cum", overrides) == 20
    assert _nonoverlap_min_dates_for_cell({"h_start": 40, "h_end": 60}, "bucket", overrides) == 20


def test_compute_nonoverlap_robustness_pass_thresholds() -> None:
    passing = compute_nonoverlap_robustness_pass(
        {"n_offsets_total": 5, "n_offsets_valid": 4, "offset_sign_agreement_ratio": 0.75}
    )
    assert passing == {"valid_offset_ratio": 0.8, "nonoverlap_robustness_pass": True}

    failing_ratio = compute_nonoverlap_robustness_pass(
        {"n_offsets_total": 5, "n_offsets_valid": 3, "offset_sign_agreement_ratio": 0.9}
    )
    assert failing_ratio["nonoverlap_robustness_pass"] is False

    failing_sign = compute_nonoverlap_robustness_pass(
        {"n_offsets_total": 5, "n_offsets_valid": 5, "offset_sign_agreement_ratio": 0.5}
    )
    assert failing_sign["nonoverlap_robustness_pass"] is False

    no_offsets = compute_nonoverlap_robustness_pass(
        {"n_offsets_total": 0, "n_offsets_valid": 0, "offset_sign_agreement_ratio": None}
    )
    assert no_offsets == {"valid_offset_ratio": 0.0, "nonoverlap_robustness_pass": False}


def test_select_phase_b_long_horizon_cells_matches_nw_lag_59_rule() -> None:
    cells = [
        {"hypothesis_id": "a", "cell_type": "cumulative", "h_start": 0, "h_end": 60},
        {"hypothesis_id": "b", "cell_type": "cumulative", "h_start": 0, "h_end": 120},
        {"hypothesis_id": "c", "cell_type": "bucket", "h_start": 40, "h_end": 60},
        {"hypothesis_id": "d", "cell_type": "bucket", "h_start": 60, "h_end": 120},
    ]
    selected = select_phase_b_long_horizon_cells(cells)
    assert {c["hypothesis_id"] for c in selected} == {"a", "b", "d"}


# --- continuous: non-overlap offsets + temporal placebo (reuse Phase A) ---


def _seed_phase_b_panel(
    con: duckdb.DuckDBPyConnection, *, n_sessions: int, n_tickers: int = 10
) -> None:
    rows: list[dict] = []
    base = date(2024, 1, 1)
    for session in range(1, n_sessions + 1):
        d = base + timedelta(days=session - 1)
        for market in ("KOSPI", "KOSDAQ"):
            for t in range(n_tickers):
                ticker = f"{market[:1]}{t}"
                wobble5 = 3.0 * math.sin(t + 0.7 * session)
                raw_5 = float(t) * 2.0 + wobble5
                wobble60 = 3.0 * math.sin(t + 0.05 * session)
                raw_60 = float(t) * 2.0 + wobble60
                rows.append(
                    {
                        "trade_date": d,
                        "ticker": ticker,
                        "market": market,
                        "formation_session_idx": session,
                        "fin_log_mcap": float(t) + 0.01 * session,
                        "y_rank_5d": raw_5,
                        "raw_label_5d": raw_5,
                        "label_ok_5d": True,
                        "y_rank_60d": raw_60,
                        "raw_label_60d": raw_60,
                        "label_ok_60d": True,
                        "y_rank_120d": raw_60,
                        "raw_label_120d": raw_60,
                        "label_ok_120d": True,
                        "in_broad": True,
                        "in_tradable": True,
                        "common_formation_120d": True,
                        "common_survivor_120d": True,
                        "ca_mask": False,
                    }
                )
    frame = pl.DataFrame(rows)
    con.register("_panel_source", frame)
    con.execute("CREATE OR REPLACE TABLE analysis_panel_phase_b AS SELECT * FROM _panel_source")
    con.unregister("_panel_source")


def test_run_phase_b_continuous_nonoverlap_computes_gate_per_cell() -> None:
    con = duckdb.connect()
    _seed_phase_b_panel(con, n_sessions=40)
    cell = {
        "hypothesis_id": "fin_log_mcap|fin_log_mcap|cum|0|5",
        "family": "fin_log_mcap",
        "feature": "fin_log_mcap",
        "cell_type": "cumulative",
        "h_start": 0,
        "h_end": 5,
        "expected_sign": "+",
    }
    rows = run_phase_b_continuous_nonoverlap(
        con,
        [cell],
        panel_view="analysis_panel_phase_b",
        sample_start="2024-01-01",
        min_names=5,
        nonoverlap_min_dates_overrides={"default": 5},
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["scan_type"] == "cum"
    assert row["n_offsets_total"] == 5  # stride = h_end for a cumulative cell
    assert row["nonoverlap_min_dates"] == 5
    assert row["n_offsets_valid"] == 5
    assert row["offset_sign_agreement_ratio"] > 0.5  # feature is strongly aligned with the label
    assert row["nonoverlap_robustness_pass"] is True


def test_run_phase_b_continuous_nonoverlap_uses_h120_override() -> None:
    con = duckdb.connect()
    _seed_phase_b_panel(con, n_sessions=10)  # far too few sessions for stride=120 to pass
    cell = {
        "hypothesis_id": "fin_log_mcap|fin_log_mcap|cum|0|120",
        "family": "fin_log_mcap",
        "feature": "fin_log_mcap",
        "cell_type": "cumulative",
        "h_start": 0,
        "h_end": 120,
        "expected_sign": "+",
    }
    rows = run_phase_b_continuous_nonoverlap(
        con, [cell], panel_view="analysis_panel_phase_b", sample_start="2024-01-01", min_names=5
    )
    assert rows[0]["nonoverlap_min_dates"] == 12  # cumulative_120 override, not the default 20


def test_run_phase_b_temporal_placebo_reuses_permutation_module_and_is_deterministic() -> None:
    con = duckdb.connect()
    _seed_phase_b_panel(con, n_sessions=90)
    cell = {
        "hypothesis_id": "fin_log_mcap|fin_log_mcap|cum|0|60",
        "family": "fin_log_mcap",
        "feature": "fin_log_mcap",
        "cell_type": "cumulative",
        "h_start": 0,
        "h_end": 60,
        "expected_sign": "-",
    }
    long_cells = select_phase_b_long_horizon_cells([cell])
    assert len(long_cells) == 1

    real_cell = scan_cell(
        con,
        panel_view="analysis_panel_phase_b",
        feature_col="fin_log_mcap",
        scan_type="cum",
        h_start=0,
        h_end=60,
        universe="broad",
        sample_kind="common_survivor",
        sample_start="2024-01-01",
        min_names=2,
        min_names_for_spread=2,
        quantile_count=5,
        min_dates_per_cell=5,
        compute_spread=False,
    )
    assert real_cell["status"] == "valid"
    hid = cell["hypothesis_id"]
    kwargs = dict(
        con=con,
        panel_view="analysis_panel_phase_b",
        long_horizon_cells=long_cells,
        real_t_nw_by_id={hid: real_cell["t_nw"]},
        config_hash="test-hash",
        sample_start="2024-01-01",
        min_names=2,
        min_names_for_spread=2,
        quantile_count=5,
        min_dates_per_cell=5,
        n_replicates=3,
        min_shift_sessions=10,
    )
    result_a = run_phase_b_temporal_placebo(**kwargs)
    result_b = run_phase_b_temporal_placebo(**kwargs)
    assert result_a["per_cell"] == result_b["per_cell"]
    assert 0 < result_a["per_cell"][hid]["p_temporal_nw"] <= 1.0


# --- SUE: issuer-cluster / filing-cycle block bootstrap ---


def _weekdays(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current = current + timedelta(days=1)
    return days


def _seed_bootstrap_calendar(con: duckdb.DuckDBPyConnection, days: list[date]) -> None:
    con.execute("CREATE TABLE daily_ohlcv (trade_date DATE)")
    con.executemany("INSERT INTO daily_ohlcv VALUES (?)", [(d,) for d in days])


def _seed_bootstrap_event_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE fin_sue_event (
            ticker VARCHAR, market VARCHAR, event_formation_date DATE,
            bsns_year INTEGER, reprt_code VARCHAR,
            fin_sue DOUBLE, bucket_0_3_excess DOUBLE,
            is_primary_constant_sample BOOLEAN
        )
    """)


def _aligned_events(
    formation_date: date, market: str, n: int, *, bsns_year: int, reprt_code: str
) -> list[tuple]:
    # market-qualified ticker (not market[:1]) so KOSPI/KOSDAQ tickers never
    # collide -- real tickers belong to exactly one market, and the issuer
    # cluster bootstrap resamples by ticker alone, so a collision here would
    # silently merge two markets' issuer histories into one fake cluster.
    return [
        (f"{market}_{i}", market, formation_date, bsns_year, reprt_code, float(i), float(i), True)
        for i in range(n)
    ]


def _seed_aligned_bootstrap_events(con: duckdb.DuckDBPyConnection, *, n_dates: int = 10) -> None:
    """16 tickers per market repeat their (perfectly aligned SUE/excess)
    history across every one of ``n_dates`` formation dates -- each date is
    also its own distinct (bsns_year, reprt_code) filing window, giving
    both cluster axes real diversity to resample from."""
    days = _weekdays(date(2021, 1, 4), n_dates)
    _seed_bootstrap_calendar(con, days)
    _seed_bootstrap_event_table(con)
    events: list[tuple] = []
    for idx, d in enumerate(days):
        for market in ("KOSPI", "KOSDAQ"):
            events += _aligned_events(d, market, 16, bsns_year=2020, reprt_code=f"W{idx}")
    con.executemany("INSERT INTO fin_sue_event VALUES (?,?,?,?,?,?,?,?)", events)


_BOOTSTRAP_KWARGS = dict(
    h_start=0,
    h_end=3,
    sample_start="2021-01-01",
    min_events_per_market_contribution=10,
    min_events_per_cohort_total=30,
)


def test_run_issuer_cluster_bootstrap_confirms_a_real_relationship() -> None:
    con = duckdb.connect()
    _seed_aligned_bootstrap_events(con)
    result = run_issuer_cluster_bootstrap(
        con,
        hypothesis_id="fin_sue|fin_sue|event|0|3",
        config_hash="test-hash",
        n_replicates=30,
        expected_sign="+",
        **_BOOTSTRAP_KWARGS,
    )
    assert result["n_clusters"] == 32  # 16 tickers x 2 markets
    assert result["n_valid_replicates"] == 30
    assert result["bootstrap_mean"] > 0.9
    assert result["bootstrap_p"] < 0.10
    assert result["cluster_confirm_pass"] is True


def test_run_filing_cycle_block_bootstrap_confirms_a_real_relationship() -> None:
    con = duckdb.connect()
    _seed_aligned_bootstrap_events(con)
    result = run_filing_cycle_block_bootstrap(
        con,
        hypothesis_id="fin_sue|fin_sue|event|0|3",
        config_hash="test-hash",
        n_replicates=30,
        expected_sign="+",
        **_BOOTSTRAP_KWARGS,
    )
    assert result["n_clusters"] == 10  # 10 distinct (bsns_year, reprt_code) windows
    assert result["n_valid_replicates"] == 30
    assert result["bootstrap_mean"] > 0.9
    assert result["bootstrap_p"] < 0.10
    assert result["cluster_confirm_pass"] is True


def test_cluster_bootstrap_fails_confirmation_on_expected_sign_mismatch() -> None:
    con = duckdb.connect()
    _seed_aligned_bootstrap_events(con)
    result = run_issuer_cluster_bootstrap(
        con,
        hypothesis_id="fin_sue|fin_sue|event|0|3",
        config_hash="test-hash",
        n_replicates=20,
        expected_sign="-",  # real relationship is positive -> sign check must fail
        **_BOOTSTRAP_KWARGS,
    )
    assert result["bootstrap_mean"] > 0.9
    assert result["cluster_confirm_pass"] is False


def test_cluster_bootstrap_is_deterministic_given_the_same_config_hash() -> None:
    con = duckdb.connect()
    _seed_aligned_bootstrap_events(con)
    kwargs = dict(
        hypothesis_id="fin_sue|fin_sue|event|0|3",
        config_hash="abc",
        n_replicates=10,
        expected_sign="+",
        **_BOOTSTRAP_KWARGS,
    )
    first = run_issuer_cluster_bootstrap(con, **kwargs)
    second = run_issuer_cluster_bootstrap(con, **kwargs)
    assert first["replicate_ic_means"] == second["replicate_ic_means"]


def test_cluster_bootstrap_resumes_from_checkpoint_without_recomputing(tmp_path) -> None:
    con = duckdb.connect()
    _seed_aligned_bootstrap_events(con)
    checkpoint_path = tmp_path / "issuer_bootstrap.jsonl"
    kwargs = dict(
        hypothesis_id="fin_sue|fin_sue|event|0|3",
        config_hash="abc",
        n_replicates=5,
        expected_sign="+",
        checkpoint_path=checkpoint_path,
        **_BOOTSTRAP_KWARGS,
    )
    first = run_issuer_cluster_bootstrap(con, **kwargs)
    assert len(checkpoint_path.read_text(encoding="utf-8").splitlines()) == 5

    lines = checkpoint_path.read_text(encoding="utf-8").splitlines()
    checkpoint_path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    second = run_issuer_cluster_bootstrap(con, **kwargs)
    assert second["replicate_ic_means"] == first["replicate_ic_means"]


def test_run_cluster_bootstrap_handles_no_formation_rows() -> None:
    con = duckdb.connect()
    _seed_bootstrap_calendar(con, _weekdays(date(2021, 1, 4), 5))
    _seed_bootstrap_event_table(con)
    result = run_issuer_cluster_bootstrap(
        con,
        hypothesis_id="x",
        config_hash="c",
        n_replicates=5,
        **_BOOTSTRAP_KWARGS,
    )
    assert result["status_reason"] == "no_formation_rows"
    assert result["cluster_confirm_pass"] is False


def test_evaluate_sue_cluster_confirmation_requires_both_bootstraps_to_pass() -> None:
    passing = {"cluster_confirm_pass": True, "bootstrap_p": 0.01, "bootstrap_mean": 0.5}
    failing = {"cluster_confirm_pass": False, "bootstrap_p": 0.5, "bootstrap_mean": -0.1}

    both_pass = evaluate_sue_cluster_confirmation(passing, passing)
    assert both_pass["sue_cluster_confirm_pass"] is True

    one_fails = evaluate_sue_cluster_confirmation(passing, failing)
    assert one_fails["sue_cluster_confirm_pass"] is False


def test_phase_b_nonoverlap_robustness_never_reaches_the_daily_ic_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase B's robustness diagnostics route through ``run_nonoverlap_offsets``
    and ``run_temporal_placebo``, neither of which accepts a Stage 0 sink —
    ``daily_ic.parquet`` holds the B-7 core scan's series only."""
    from modeler.analysis import horizon_scan_runner

    def _fail(*args, **kwargs):
        raise AssertionError("Phase B robustness must not call scan_cell")

    monkeypatch.setattr(horizon_scan_runner, "scan_cell", _fail)
    con = duckdb.connect()
    _seed_phase_b_panel(con, n_sessions=40)
    rows = run_phase_b_continuous_nonoverlap(
        con,
        [
            {
                "hypothesis_id": "fin_log_mcap|fin_log_mcap|cum|0|5",
                "family": "fin_log_mcap",
                "feature": "fin_log_mcap",
                "cell_type": "cumulative",
                "h_start": 0,
                "h_end": 5,
                "expected_sign": "+",
            }
        ],
        panel_view="analysis_panel_phase_b",
        sample_start="2024-01-01",
        min_names=5,
        nonoverlap_min_dates_overrides={"default": 5},
    )
    assert len(rows) == 1
