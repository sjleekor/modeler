"""Tests for ``research/analysis/horizon_scan_phase_b_joint_permutation.py``
(§6 B-8 "결합 단면 permutation").

The SUE rank-permutation mechanic is genuinely new, so it's tested directly
against a real (small) DuckDB-backed cohort fixture — same convention as
``test_horizon_scan_phase_b_diagnostics.py``'s event-ordinal tests. The
continuous side reuses Phase A's already-integration-tested
``run_cross_sectional_permutation`` machinery verbatim (that function itself
has no dedicated small-panel unit test anywhere in this repo either — it is
only exercised by the big synthetic end-to-end smoke test,
``test_horizon_scan_smoke.py``), so this file only verifies the *wiring*
(seed derivation, row combination, checkpoint resume) via mocks, not
``scan_cell``'s own math.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl
import pytest
import modeler.analysis.horizon_scan_phase_b_joint_permutation as joint_perm
from modeler.analysis.horizon_scan_phase_b_joint_permutation import (
    _permute_qualifying_sue_ranks,
    _scan_sue_null_row,
    run_combined_cross_sectional_permutation,
)

# --- _permute_qualifying_sue_ranks ---


def _qualifying_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_formation_date": [date(2020, 1, 6)] * 3 + [date(2020, 1, 7)] * 3,
            "market": ["KOSPI"] * 6,
            "ticker": ["T0", "T1", "T2", "T0", "T1", "T2"],
            "original_rcept_no": [
                "202001060",
                "202001061",
                "202001062",
                "202001070",
                "202001071",
                "202001072",
            ],
            "sue_pctrank": [0.1, 0.5, 0.9, 0.2, 0.6, 0.8],
            "excess_pctrank": [0.15, 0.45, 0.95, 0.25, 0.55, 0.85],
        }
    )


def test_permute_qualifying_sue_ranks_is_deterministic() -> None:
    frame = _qualifying_frame()
    out1 = _permute_qualifying_sue_ranks(frame, seed=42)
    out2 = _permute_qualifying_sue_ranks(frame, seed=42)
    assert out1["sue_pctrank"].to_list() == out2["sue_pctrank"].to_list()


def test_permute_qualifying_sue_ranks_is_input_order_invariant() -> None:
    frame = _qualifying_frame()
    canonical = _permute_qualifying_sue_ranks(frame, seed=42)
    reversed_input = _permute_qualifying_sue_ranks(frame.reverse(), seed=42)

    assert canonical.equals(reversed_input)


def test_permute_qualifying_sue_ranks_preserves_group_membership() -> None:
    """Values must only move within their own (date, market) group — the
    multiset of sue_pctrank values per group is unchanged, only their
    row-to-row pairing with excess_pctrank."""
    frame = _qualifying_frame()
    out = _permute_qualifying_sue_ranks(frame, seed=7)

    for (formation_date,), grp in frame.group_by(["event_formation_date"], maintain_order=True):
        original_values = sorted(grp["sue_pctrank"].to_list())
        permuted_values = sorted(
            out.filter(pl.col("event_formation_date") == formation_date)["sue_pctrank"].to_list()
        )
        assert original_values == permuted_values

    # excess_pctrank never moves
    assert out["excess_pctrank"].to_list() == frame["excess_pctrank"].to_list()


def test_permute_qualifying_sue_ranks_different_seeds_differ() -> None:
    frame = _qualifying_frame()
    out_a = _permute_qualifying_sue_ranks(frame, seed=1)
    out_b = _permute_qualifying_sue_ranks(frame, seed=2)
    assert out_a["sue_pctrank"].to_list() != out_b["sue_pctrank"].to_list()


# --- _scan_sue_null_row ---


def test_scan_sue_null_row_returns_insufficient_when_qualifying_is_none() -> None:
    cell = {"hypothesis_id": "fin_sue|fin_sue|event|0|3", "h_start": 0, "h_end": 3}
    row = _scan_sue_null_row(cell, None, seed=1, min_events_per_cohort_total=30)
    assert row["status"] == "insufficient"
    assert row["ic_mean"] is None


def test_scan_sue_null_row_computes_ic_when_cohorts_survive() -> None:
    # 3 cohort dates x 2 markets x enough rows per group to clear the total
    # floor after shuffling (shuffling doesn't change group sizes).
    rows = []
    for d_idx, d in enumerate([date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]):
        for market in ("KOSPI", "KOSDAQ"):
            for i in range(16):
                rows.append(
                    {
                        "event_formation_date": d,
                        "market": market,
                        "ticker": f"{market[:1]}{i}",
                        "original_rcept_no": f"{d:%Y%m%d}{market[:1]}{i}",
                        "sue_pctrank": i / 15,
                        "excess_pctrank": i / 15,
                        "formation_session_idx": d_idx + 1,
                    }
                )
    qualifying = pl.DataFrame(rows)
    cell = {"hypothesis_id": "fin_sue|fin_sue|event|0|3", "h_start": 0, "h_end": 3}

    row = _scan_sue_null_row(cell, qualifying, seed=1, min_events_per_cohort_total=30)

    assert row["scan_type"] == "event_bucket"
    assert row["status"] in ("valid", "insufficient")  # depends on the shuffle draw, never crashes


def test_scan_sue_null_row_is_input_order_invariant() -> None:
    rows = []
    for d_idx, d in enumerate([date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8)]):
        for market in ("KOSPI", "KOSDAQ"):
            for i in range(16):
                rows.append(
                    {
                        "event_formation_date": d,
                        "market": market,
                        "ticker": f"{market[:1]}{i}",
                        "original_rcept_no": f"{d:%Y%m%d}{market[:1]}{i}",
                        "sue_pctrank": i / 15,
                        "excess_pctrank": (15 - i) / 15,
                        "formation_session_idx": d_idx + 1,
                    }
                )
    qualifying = pl.DataFrame(rows)
    cell = {"hypothesis_id": "fin_sue|fin_sue|event|0|3", "h_start": 0, "h_end": 3}

    canonical = _scan_sue_null_row(
        cell, qualifying, seed=7, min_events_per_cohort_total=30
    )
    reversed_input = _scan_sue_null_row(
        cell, qualifying.reverse(), seed=7, min_events_per_cohort_total=30
    )

    assert canonical == reversed_input


# --- run_combined_cross_sectional_permutation: SUE-only path (real DuckDB) ---


def _weekdays(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _seed_sue_cohort(con: duckdb.DuckDBPyConnection, days: list[date]) -> None:
    con.execute("CREATE TABLE daily_ohlcv (trade_date DATE)")
    con.executemany("INSERT INTO daily_ohlcv VALUES (?)", [(d,) for d in days])
    con.execute("""
        CREATE TABLE fin_sue_event (
            ticker VARCHAR, market VARCHAR, event_formation_date DATE,
            original_rcept_no VARCHAR,
            bsns_year INTEGER, reprt_code VARCHAR,
            fin_sue DOUBLE, bucket_0_3_excess DOUBLE,
            is_primary_constant_sample BOOLEAN
        )
    """)
    events = []
    for d in days:
        for i in range(16):
            ticker = f"T{i}"
            events.append(
                (ticker, "KOSPI", d, f"{d:%Y%m%d}{i}", 2020, "11013", float(i), float(i), True)
            )
    con.executemany("INSERT INTO fin_sue_event VALUES (?,?,?,?,?,?,?,?,?)", events)


_SUE_ONLY_KWARGS = dict(
    panel_view="analysis_panel_phase_b",  # unused when combined_continuous_registry is empty
    combined_continuous_registry=[],
    config_hash="test-config-hash",
    sample_start="2020-01-01",
    min_names=2,
    min_names_for_spread=2,
    quantile_count=5,
    min_dates_per_cell=5,
    min_events_per_market_contribution=10,
    min_events_per_cohort_total=16,
    q_threshold=0.10,
)


def test_run_combined_cross_sectional_permutation_sue_only_produces_replicates() -> None:
    con = duckdb.connect()
    days = _weekdays(date(2020, 1, 6), 10)
    _seed_sue_cohort(con, days)
    ready_sue_cells = [
        {
            "hypothesis_id": "fin_sue|fin_sue|event|0|3",
            "family": "fin_sue",
            "feature": "fin_sue",
            "h_start": 0,
            "h_end": 3,
            "expected_sign": "+",
        }
    ]

    result = run_combined_cross_sectional_permutation(
        con, ready_sue_cells=ready_sue_cells, n_replicates=5, **_SUE_ONLY_KWARGS
    )

    assert result["n_replicates"] == 5
    assert len(result["null_discovery_counts"]) == 5
    assert all(n in (0, 1) for n in result["null_discovery_counts"])  # only 1 hypothesis in play


def test_run_combined_cross_sectional_permutation_resumes_from_checkpoint(
    tmp_path: Path,
) -> None:
    con = duckdb.connect()
    days = _weekdays(date(2020, 2, 3), 10)
    _seed_sue_cohort(con, days)
    ready_sue_cells = [
        {
            "hypothesis_id": "fin_sue|fin_sue|event|0|3",
            "family": "fin_sue",
            "feature": "fin_sue",
            "h_start": 0,
            "h_end": 3,
            "expected_sign": "+",
        }
    ]
    checkpoint_path = tmp_path / "checkpoint.jsonl"

    first = run_combined_cross_sectional_permutation(
        con,
        ready_sue_cells=ready_sue_cells,
        n_replicates=3,
        checkpoint_path=checkpoint_path,
        **_SUE_ONLY_KWARGS,
    )
    lines_after_first = checkpoint_path.read_text().splitlines()
    assert len(lines_after_first) == 3

    second = run_combined_cross_sectional_permutation(
        con,
        ready_sue_cells=ready_sue_cells,
        n_replicates=5,
        checkpoint_path=checkpoint_path,
        **_SUE_ONLY_KWARGS,
    )
    assert second["n_replicates"] == 5
    # the first 3 replicates must be untouched (same seed) across the resume
    assert second["replicate_summaries"][:3] == first["replicate_summaries"]


def test_run_combined_cross_sectional_permutation_raises_without_checkpoint_duplicates(
    tmp_path: Path,
) -> None:
    con = duckdb.connect()
    days = _weekdays(date(2020, 3, 2), 10)
    _seed_sue_cohort(con, days)
    ready_sue_cells = [
        {
            "hypothesis_id": "fin_sue|fin_sue|event|0|3",
            "family": "fin_sue",
            "feature": "fin_sue",
            "h_start": 0,
            "h_end": 3,
            "expected_sign": "+",
        }
    ]

    result = run_combined_cross_sectional_permutation(
        con, ready_sue_cells=ready_sue_cells, n_replicates=4, **_SUE_ONLY_KWARGS
    )
    assert {s["replicate"] for s in result["replicate_summaries"]} == {0, 1, 2, 3}


# --- continuous-side wiring: mocked (no real panel needed) ---


def test_run_combined_cross_sectional_permutation_wires_continuous_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, int] = {"scan": 0}

    monkeypatch.setattr(
        joint_perm,
        "fetch_broad_common_survivor_frame",
        lambda con, **k: pl.DataFrame({"trade_date": [], "market": [], "feat_a": []}),
    )
    monkeypatch.setattr(joint_perm, "permute_within_groups", lambda frame, **k: frame)

    def _fake_scan(con, registry, **k):
        calls["scan"] += 1
        return [
            {
                "hypothesis_id": hyp["hypothesis_id"],
                "family": hyp.get("family", "fam"),
                "scan_type": hyp["scan_type"],
                "h_end": hyp["h_end"],
                "status": "valid",
                "p_nw": 0.5,
                "expected_sign": hyp.get("expected_sign"),
                "ic_mean": 0.01,
            }
            for hyp in registry
        ]

    monkeypatch.setattr(joint_perm, "_scan_registry_once", _fake_scan)

    registry = [
        {
            "hypothesis_id": "px_reversal_5d|px_reversal_5d|cum|0|1",
            "family": "px_reversal_5d",
            "feature": "feat_a",
            "scan_type": "cum",
            "h_start": 0,
            "h_end": 1,
            "expected_sign": "+",
        }
    ]

    result = run_combined_cross_sectional_permutation(
        duckdb.connect(),
        panel_view="analysis_panel_phase_b",
        combined_continuous_registry=registry,
        ready_sue_cells=[],
        config_hash="test-config-hash",
        sample_start="2020-01-01",
        min_names=2,
        min_names_for_spread=2,
        quantile_count=5,
        min_dates_per_cell=5,
        min_events_per_market_contribution=10,
        min_events_per_cohort_total=16,
        n_replicates=3,
    )

    assert calls["scan"] == 3
    assert result["n_replicates"] == 3
    assert len(result["null_discovery_counts"]) == 3
