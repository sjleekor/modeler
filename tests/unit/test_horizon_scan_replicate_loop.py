from __future__ import annotations

import json
import math
from datetime import date, timedelta

import duckdb
import polars as pl
import pytest
import modeler.analysis.horizon_scan_permutation as permutation
from modeler.analysis.horizon_scan_checkpoint import build_checkpoint_fingerprint
from modeler.analysis.horizon_scan_permutation import (
    run_cross_sectional_permutation,
    run_lookahead_canary,
    run_temporal_placebo,
    select_long_horizon_hypotheses,
)
from modeler.analysis.horizon_scan_runner import scan_cell


def _seed_replicate_panel(
    con: duckdb.DuckDBPyConnection, *, n_sessions: int, n_tickers: int = 4
) -> None:
    """A broad/common-survivor-only panel carrying two co-movable primary
    features (feat_a/feat_b, h=5 cum/bucket), one long-horizon feature
    (feat_long, h=60 cum), and the fwd_ret_1d/raw_label_1d pair constructed so
    the look-ahead canary's IC=1.0 invariant holds by exact construction
    (raw_label_1d = fwd_ret_1d minus its per-(date,market) cross-sectional
    mean, so its rank order in t never differs from fwd_ret_1d's own).
    """
    rows: list[dict] = []
    base = date(2024, 1, 1)
    mean_t = (n_tickers - 1) / 2.0
    for session in range(1, n_sessions + 1):
        d = base + timedelta(days=session - 1)
        for market in ("KOSPI", "KOSDAQ"):
            for t in range(n_tickers):
                ticker = f"{market[:1]}{t}"
                wobble5 = 3.0 * math.sin(t + 0.7 * session)
                raw_5 = float(t) * 2.0 + wobble5
                wobble60 = 3.0 * math.sin(t + 0.05 * session)
                raw_60 = float(t) * 2.0 + wobble60
                fwd_ret_1d = float(t) + 0.01 * session
                rows.append(
                    {
                        "trade_date": d,
                        "ticker": ticker,
                        "market": market,
                        "formation_session_idx": session,
                        "ca_mask": False,
                        "in_broad": True,
                        "common_formation_120d": True,
                        "common_survivor_120d": True,
                        "feat_a": float(t) + 0.01 * session,
                        "feat_b": float(n_tickers - 1 - t) + 0.02 * session,
                        "y_rank_5d": raw_5,
                        "raw_label_5d": raw_5,
                        "label_ok_5d": True,
                        "y_rank_bucket_0_5d": raw_5,
                        "raw_bucket_label_0_5d": raw_5,
                        "bucket_ok_0_5d": True,
                        "feat_long": float(t) + 0.01 * session,
                        "y_rank_60d": raw_60,
                        "raw_label_60d": raw_60,
                        "label_ok_60d": True,
                        "fwd_ret_1d": fwd_ret_1d,
                        "y_rank_1d": float(t) - mean_t,
                        "raw_label_1d": float(t) - mean_t,
                        "label_ok_1d": True,
                    }
                )
    frame = pl.DataFrame(rows)
    con.register("_panel_source", frame)
    con.execute("CREATE OR REPLACE TABLE analysis_panel AS SELECT * FROM _panel_source")
    con.unregister("_panel_source")


def _cross_sectional_registry() -> list[dict]:
    return [
        {
            "hypothesis_id": "famA|feat_a|cum|0|5",
            "family": "famA",
            "feature": "feat_a",
            "scan_type": "cum",
            "h_start": 0,
            "h_end": 5,
            "expected_sign": "+",
        },
        {
            "hypothesis_id": "famB|feat_b|bucket|0|5",
            "family": "famB",
            "feature": "feat_b",
            "scan_type": "bucket",
            "h_start": 0,
            "h_end": 5,
            "expected_sign": "+",
        },
    ]


def _long_horizon_registry() -> list[dict]:
    return [
        {
            "hypothesis_id": "famC|feat_long|cum|0|60",
            "family": "famC",
            "feature": "feat_long",
            "scan_type": "cum",
            "h_start": 0,
            "h_end": 60,
            "expected_sign": "+",
        }
    ]


def test_select_long_horizon_hypotheses_matches_nw_lag_59_rule() -> None:
    registry = [
        {"hypothesis_id": "a", "scan_type": "cum", "h_start": 0, "h_end": 20},
        {"hypothesis_id": "b", "scan_type": "cum", "h_start": 0, "h_end": 60},
        {"hypothesis_id": "c", "scan_type": "cum", "h_start": 0, "h_end": 120},
        {"hypothesis_id": "d", "scan_type": "bucket", "h_start": 40, "h_end": 60},
        {"hypothesis_id": "e", "scan_type": "bucket", "h_start": 60, "h_end": 120},
    ]
    selected = select_long_horizon_hypotheses(registry)
    assert {h["hypothesis_id"] for h in selected} == {"b", "c", "e"}


def test_run_cross_sectional_permutation_is_complete_and_deterministic() -> None:
    con = duckdb.connect()
    _seed_replicate_panel(con, n_sessions=30)
    registry = _cross_sectional_registry()
    kwargs = dict(
        con=con,
        panel_view="analysis_panel",
        primary_registry=registry,
        real_discovery_count=1,
        config_hash="abc123",
        sample_start="2024-01-01",
        min_names=2,
        min_names_for_spread=2,
        quantile_count=5,
        min_dates_per_cell=5,
        n_replicates=3,
    )
    result_a = run_cross_sectional_permutation(**kwargs)
    result_b = run_cross_sectional_permutation(**kwargs)
    assert result_a["replicate_summaries"] == result_b["replicate_summaries"]
    assert len(result_a["replicate_summaries"]) == 3
    assert {s["replicate"] for s in result_a["replicate_summaries"]} == {0, 1, 2}


def test_run_cross_sectional_permutation_p_empirical_count_matches_formula() -> None:
    con = duckdb.connect()
    _seed_replicate_panel(con, n_sessions=30)
    registry = _cross_sectional_registry()
    result = run_cross_sectional_permutation(
        con=con,
        panel_view="analysis_panel",
        primary_registry=registry,
        real_discovery_count=2,
        config_hash="xyz",
        sample_start="2024-01-01",
        min_names=2,
        min_names_for_spread=2,
        quantile_count=5,
        min_dates_per_cell=5,
        n_replicates=5,
    )
    at_least = sum(1 for n in result["null_discovery_counts"] if n >= 2)
    assert result["p_empirical_count"] == pytest.approx((1 + at_least) / (5 + 1))


def test_run_cross_sectional_permutation_resumes_from_checkpoint_without_recomputing(
    tmp_path,
) -> None:
    con = duckdb.connect()
    _seed_replicate_panel(con, n_sessions=30)
    registry = _cross_sectional_registry()
    checkpoint = tmp_path / "checkpoint.jsonl"
    fake_row = {
        "replicate": 0,
        "seed": 999999,
        "n_valid_hypotheses": 2,
        "n_bh_pass": 2,
        "n_primary_discovery": 999,
        "min_p_nw": 0.0,
        "min_q_fdr_global": 0.0,
        "max_abs_t_nw": 999.0,
    }
    checkpoint.write_text(json.dumps(fake_row) + "\n", encoding="utf-8")
    result = run_cross_sectional_permutation(
        con=con,
        panel_view="analysis_panel",
        primary_registry=registry,
        real_discovery_count=0,
        config_hash="resume-test",
        sample_start="2024-01-01",
        min_names=2,
        min_names_for_spread=2,
        quantile_count=5,
        min_dates_per_cell=5,
        n_replicates=2,
        checkpoint_path=checkpoint,
    )
    by_index = {s["replicate"]: s for s in result["replicate_summaries"]}
    assert by_index[0]["n_primary_discovery"] == 999  # trusted from checkpoint, not recomputed
    assert by_index[1]["n_primary_discovery"] != 999  # replicate 1 was actually computed
    # the checkpoint file itself must now carry both replicates for a future resume
    persisted = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
    assert {row["replicate"] for row in persisted} == {0, 1}


def test_cross_sectional_directory_checkpoint_is_worker_count_invariant(tmp_path) -> None:
    registry = _cross_sectional_registry()
    fingerprint = build_checkpoint_fingerprint(
        registry=registry,
        a0_manifest_hash="a0",
        readiness_population_hash=None,
        smoke_family=None,
        requested_replicates=3,
        include_holdout=False,
        holdout_start=None,
        scan_engine="legacy",
        row_order_contract="row-v1",
        sue_nw_order_contract="sue-nw-v2",
        sue_permutation_order_contract="sue-rank-v2",
        mapping_contract_version="joint_cs_v2",
        analysis_kernel_hash="kernel",
        duckdb_version="duckdb",
        polars_version="polars",
        numpy_version="numpy",
    )
    kwargs = dict(
        panel_view="analysis_panel",
        primary_registry=registry,
        real_discovery_count=1,
        config_hash="workers",
        sample_start="2024-01-01",
        min_names=2,
        min_names_for_spread=2,
        quantile_count=5,
        min_dates_per_cell=5,
        n_replicates=3,
        mapping_contract_version="joint_cs_v2",
        collect_cell_stats=True,
        checkpoint_path=tmp_path / "cross-sectional",
        checkpoint_fingerprint=fingerprint,
    )
    con_one = duckdb.connect()
    _seed_replicate_panel(con_one, n_sessions=30)
    one = run_cross_sectional_permutation(
        con=con_one, workers=1, **{**kwargs, "checkpoint_path": tmp_path / "one"}
    )

    con_two = duckdb.connect()
    _seed_replicate_panel(con_two, n_sessions=30)
    two = run_cross_sectional_permutation(
        con=con_two, workers=2, **{**kwargs, "checkpoint_path": tmp_path / "two"}
    )

    assert one["replicate_summaries"] == two["replicate_summaries"]
    assert one["cell_stats"] == two["cell_stats"]


def test_cross_sectional_process_path_uses_positional_initializer_and_no_base_frame_pickle(
    monkeypatch,
):
    con = duckdb.connect()
    _seed_replicate_panel(con, n_sessions=30)
    registry = _cross_sectional_registry()
    base_frame = permutation.fetch_broad_common_survivor_frame(
        con,
        panel_view="analysis_panel",
        extra_cols=[
            "feat_a",
            "feat_b",
            "y_rank_5d",
            "raw_label_5d",
            "label_ok_5d",
            "y_rank_bucket_0_5d",
            "raw_bucket_label_0_5d",
            "bucket_ok_0_5d",
        ],
        sample_start="2024-01-01",
    )

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeProcessPool:
        def __init__(self, *, max_workers, initializer, initargs):
            assert max_workers == 2
            initializer(*initargs)

        def submit(self, function, replicate, kwargs):
            assert "base_frame" not in kwargs
            return _Future(function(replicate, kwargs))

        def shutdown(self, *, wait):
            assert wait is True

    def _fake_initializer(lake, panel_view, primary_registry, sample_start):
        assert lake == "synthetic-lake"
        assert panel_view == "analysis_panel"
        assert primary_registry == registry
        assert sample_start == "2024-01-01"
        permutation._CROSS_PROCESS_CONTEXT.clear()
        permutation._CROSS_PROCESS_CONTEXT.update({"con": con, "frame": base_frame})

    monkeypatch.setattr(permutation, "ProcessPoolExecutor", _FakeProcessPool)
    monkeypatch.setattr(permutation, "_init_cross_sectional_process_worker", _fake_initializer)

    result = permutation.run_cross_sectional_permutation(
        con=con,
        panel_view="analysis_panel",
        primary_registry=registry,
        real_discovery_count=0,
        config_hash="process-path",
        sample_start="2024-01-01",
        min_names=2,
        min_names_for_spread=2,
        quantile_count=5,
        min_dates_per_cell=5,
        n_replicates=1,
        workers=2,
        worker_lake="synthetic-lake",
    )

    assert result["n_replicates"] == 1


def test_run_temporal_placebo_is_complete_and_deterministic() -> None:
    con = duckdb.connect()
    _seed_replicate_panel(con, n_sessions=90)
    registry = _long_horizon_registry()
    real_cell = scan_cell(
        con,
        panel_view="analysis_panel",
        feature_col="feat_long",
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
    hid = registry[0]["hypothesis_id"]
    kwargs = dict(
        con=con,
        panel_view="analysis_panel",
        long_horizon_registry=registry,
        real_t_nw_by_id={hid: real_cell["t_nw"]},
        config_hash="abc",
        sample_start="2024-01-01",
        min_names=2,
        min_names_for_spread=2,
        quantile_count=5,
        min_dates_per_cell=5,
        n_replicates=3,
        min_shift_sessions=10,
    )
    result_a = run_temporal_placebo(**kwargs)
    result_b = run_temporal_placebo(**kwargs)
    result_parallel = run_temporal_placebo(**{**kwargs, "workers": 2})
    assert result_a["replicate_meta"] == result_b["replicate_meta"]
    assert result_a["replicate_meta"] == result_parallel["replicate_meta"]
    assert result_a["per_cell"] == result_b["per_cell"]
    assert result_a["per_cell"] == result_parallel["per_cell"]
    assert result_a["per_cell"][hid]["p_temporal_nw"] is not None
    assert 0 < result_a["per_cell"][hid]["p_temporal_nw"] <= 1.0
    for meta in result_a["replicate_meta"]:
        assert 10 <= meta["shift"] <= result_a["total_sessions"] - 10


def test_run_temporal_placebo_resumes_from_checkpoint_without_recomputing(tmp_path) -> None:
    con = duckdb.connect()
    _seed_replicate_panel(con, n_sessions=90)
    registry = _long_horizon_registry()
    hid = registry[0]["hypothesis_id"]
    checkpoint = tmp_path / "temporal_checkpoint.jsonl"
    fake_meta = {
        "replicate": 0,
        "seed": 12345,
        "shift": 42,
        "n_rows_after_join": 0,
        "n_tickers_after_join": 0,
        "abs_t_nw_by_id": {hid: 12345.0},
    }
    checkpoint.write_text(json.dumps(fake_meta) + "\n", encoding="utf-8")
    result = run_temporal_placebo(
        con=con,
        panel_view="analysis_panel",
        long_horizon_registry=registry,
        real_t_nw_by_id={hid: 5.0},
        config_hash="resume-temporal",
        sample_start="2024-01-01",
        min_names=2,
        min_names_for_spread=2,
        quantile_count=5,
        min_dates_per_cell=5,
        n_replicates=2,
        min_shift_sessions=10,
        checkpoint_path=checkpoint,
    )
    by_index = {m["replicate"]: m for m in result["replicate_meta"]}
    assert by_index[0]["abs_t_nw_by_id"][hid] == 12345.0  # trusted from checkpoint
    assert by_index[1]["abs_t_nw_by_id"][hid] != 12345.0


def test_run_lookahead_canary_passes_on_the_constructed_invariant() -> None:
    con = duckdb.connect()
    _seed_replicate_panel(con, n_sessions=10, n_tickers=6)
    result = run_lookahead_canary(
        con,
        panel_view="analysis_panel",
        sample_start="2024-01-01",
        min_names=2,
        min_dates_per_cell=5,
    )
    assert result["canary_pass"] is True
    assert result["ic_mean"] == pytest.approx(1.0, abs=1e-9)


def test_run_lookahead_canary_fails_when_label_does_not_track_the_feature() -> None:
    con = duckdb.connect()
    _seed_replicate_panel(con, n_sessions=10, n_tickers=6)
    con.execute("UPDATE analysis_panel SET raw_label_1d = -raw_label_1d, y_rank_1d = -y_rank_1d")
    result = run_lookahead_canary(
        con,
        panel_view="analysis_panel",
        sample_start="2024-01-01",
        min_names=2,
        min_dates_per_cell=5,
    )
    assert result["canary_pass"] is False
