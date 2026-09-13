from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import duckdb
import polars as pl
import pytest

from modeler.analysis.horizon_scan_daily_ic import (
    DAILY_IC_COLUMNS,
    DAILY_IC_DIR_NAME,
    DAILY_SPREAD_COLUMNS,
    DAILY_SPREAD_DIR_NAME,
    ParquetDailyIcSink,
    assert_daily_ic_reconciled,
    daily_ic_success_fields,
    normalize_cell_identity,
    reconcile_daily_ic,
)
from modeler.analysis.horizon_scan_runner import run_registry_scan

REGISTRY: list[dict[str, Any]] = [
    {
        "hypothesis_id": "fam|px_feature|cum|0|5",
        "family": "fam",
        "feature": "px_feature",
        "scan_type": "cum",
        "h_start": 0,
        "h_end": 5,
        "expected_sign": "+",
        "hypothesis_role": "primary",
    }
]

SCAN_KWARGS: dict[str, Any] = {
    "sample_start": "2024-01-01",
    "min_names": 5,
    "min_names_for_spread": 5,
    "quantile_count": 5,
    "min_dates_per_cell": 5,
}


def seed_scan_panel(
    con: duckdb.DuckDBPyConnection,
    *,
    n_sessions: int = 30,
    tied_label_sessions: tuple[int, ...] = (),
) -> None:
    """Same shape as ``test_horizon_scan_scan_cell``'s panel fixture.

    ``tied_label_sessions`` makes ``y_rank_5d`` constant across every name on
    those sessions while leaving ``raw_label_5d`` varying. That is the §2.3
    case the two-file layout exists for: the rank IC is NaN for both markets
    (zero variance in the realized ranks) so the date leaves ``daily``
    entirely, while the quantile spread — which reads the raw label — still
    has a value that ``q5_spread_raw`` averages in.
    """
    rows = []
    for session in range(1, n_sessions + 1):
        d = f"2024-01-01' + {session - 1}"
        for market in ("KOSPI", "KOSDAQ"):
            for t in range(10):
                ticker = f"{market[:1]}{t}"
                feature = float(t) + 0.01 * session
                wobble = 3.0 * math.sin(t + 0.7 * session)
                raw_label = float(t) * 2.0 + wobble
                rank_label = 0.0 if session in tied_label_sessions else raw_label
                rows.append(
                    f"(DATE '{d}, '{ticker}', '{market}', {session}, "
                    f"{feature}, {rank_label}, {raw_label}, true, "
                    f"true, {str(t < 8).lower()}, true, {str(t != 9).lower()}, false)"
                )
    con.execute(
        "CREATE TABLE analysis_panel AS SELECT "
        "trade_date, ticker, market, formation_session_idx, "
        "CAST(px_feature AS DOUBLE) AS px_feature, "
        "CAST(y_rank_5d AS DOUBLE) AS y_rank_5d, "
        "CAST(raw_label_5d AS DOUBLE) AS raw_label_5d, "
        "label_ok_5d, in_broad, in_tradable, common_formation_120d, "
        "common_survivor_120d, ca_mask "
        "FROM (VALUES " + ",".join(rows) + ") t(trade_date, ticker, market, formation_session_idx, "
        "px_feature, y_rank_5d, raw_label_5d, label_ok_5d, "
        "in_broad, in_tradable, common_formation_120d, common_survivor_120d, ca_mask)"
    )


def scan_with_sink(
    tmp_path: Path, *, tied_label_sessions: tuple[int, ...] = ()
) -> tuple[list[dict[str, Any]], ParquetDailyIcSink]:
    con = duckdb.connect()
    seed_scan_panel(con, tied_label_sessions=tied_label_sessions)
    sink = ParquetDailyIcSink(tmp_path)
    rows = run_registry_scan(con, REGISTRY, **SCAN_KWARGS, daily_sink=sink)
    return rows, sink


def read_daily_ic(tmp_path: Path) -> pl.DataFrame:
    return pl.read_parquet(tmp_path / DAILY_IC_DIR_NAME / "family=fam" / "px_feature.parquet")


def read_daily_spread(tmp_path: Path) -> pl.DataFrame:
    return pl.read_parquet(tmp_path / DAILY_SPREAD_DIR_NAME / "family=fam" / "px_feature.parquet")


def reconcile(tmp_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return reconcile_daily_ic(
        rows,
        daily_ic_dir=tmp_path / DAILY_IC_DIR_NAME,
        daily_spread_dir=tmp_path / DAILY_SPREAD_DIR_NAME,
    )


# --- cell identity normalization (§2.2) ---


def test_normalize_cell_identity_reads_a_phase_a_registry_row() -> None:
    identity = normalize_cell_identity(
        {
            "hypothesis_id": "px_mom_12_1|px_mom_12_1|cum|0|60",
            "family": "px_mom_12_1",
            "feature": "px_mom_12_1_lag1",
            "scan_type": "cum",
            "h_start": 0,
            "h_end": 60,
            "hypothesis_role": "primary",
        },
        universe="broad",
        sample_kind="common_survivor",
    )
    assert identity == {
        "hypothesis_id": "px_mom_12_1|px_mom_12_1|cum|0|60",
        "family": "px_mom_12_1",
        "feature": "px_mom_12_1_lag1",
        "scan_type": "cum",
        "h_start": 0,
        "h_end": 60,
        "universe": "broad",
        "sample_kind": "common_survivor",
        "hypothesis_role": "primary",
    }


def test_normalize_cell_identity_reads_a_phase_b_readiness_row() -> None:
    """Phase B rows name the same two fields differently: ``role`` instead of
    ``hypothesis_role``, ``cell_type`` instead of ``scan_type``."""
    identity = normalize_cell_identity(
        {
            "hypothesis_id": "fin_value_z|fin_value_z|bucket|20|60",
            "family": "fin_value_z",
            "feature": "fin_value_z",
            "cell_type": "bucket",
            "h_start": 20,
            "h_end": 60,
            "role": "ready_primary",
        },
        universe="tradable",
        sample_kind="available",
    )
    assert identity["scan_type"] == "bucket"
    assert identity["hypothesis_role"] == "ready_primary"
    assert identity["universe"] == "tradable"
    assert identity["sample_kind"] == "available"


def test_normalize_cell_identity_maps_cumulative_cell_type_to_cum() -> None:
    identity = normalize_cell_identity(
        {
            "hypothesis_id": "f|f|cum|0|20",
            "family": "f",
            "feature": "f",
            "cell_type": "cumulative",
            "h_start": 0,
            "h_end": 20,
            "role": "ready_primary",
        },
        universe="broad",
        sample_kind="common_survivor",
    )
    assert identity["scan_type"] == "cum"


# --- ParquetDailyIcSink layout / schema / summary (§2.3, §2.5) ---


def test_sink_writes_one_hive_partitioned_file_per_family_and_feature(tmp_path: Path) -> None:
    _rows, sink = scan_with_sink(tmp_path)
    sink.finalize()
    written = sorted(
        p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*.parquet") if p.is_file()
    )
    assert written == [
        f"{DAILY_IC_DIR_NAME}/family=fam/px_feature.parquet",
        f"{DAILY_SPREAD_DIR_NAME}/family=fam/px_feature.parquet",
    ]


def test_stored_frames_carry_exactly_the_preregistered_columns(tmp_path: Path) -> None:
    _rows, sink = scan_with_sink(tmp_path)
    sink.finalize()
    assert tuple(read_daily_ic(tmp_path).columns) == DAILY_IC_COLUMNS
    assert tuple(read_daily_spread(tmp_path).columns) == DAILY_SPREAD_COLUMNS


def test_every_scanned_universe_sample_combo_is_stored_once(tmp_path: Path) -> None:
    rows, sink = scan_with_sink(tmp_path)
    sink.finalize()
    stored = read_daily_ic(tmp_path)
    combos = set(
        zip(stored["universe"].to_list(), stored["sample_kind"].to_list()),
    )
    assert combos == {(r["universe"], r["sample_kind"]) for r in rows}
    for row in rows:
        group = stored.filter(
            (pl.col("universe") == row["universe"]) & (pl.col("sample_kind") == row["sample_kind"])
        )
        assert group.height == row["n_dates"]
        assert group["rank_ic"].mean() == pytest.approx(row["ic_mean"], abs=1e-12)


def test_finalize_reports_file_count_row_count_and_a_content_hash(tmp_path: Path) -> None:
    _rows, sink = scan_with_sink(tmp_path)
    summary = sink.finalize()
    artifacts = summary.as_manifest_artifacts()
    assert artifacts["daily_ic"]["file_count"] == 1
    assert artifacts["daily_ic"]["row_count"] == read_daily_ic(tmp_path).height
    assert artifacts["daily_spread"]["row_count"] == read_daily_spread(tmp_path).height
    assert len(artifacts["daily_ic"]["sha256"]) == 64
    assert artifacts["daily_ic"]["sha256"] != artifacts["daily_spread"]["sha256"]


def test_finalize_flushes_features_that_were_never_flushed_explicitly(tmp_path: Path) -> None:
    con = duckdb.connect()
    seed_scan_panel(con)
    sink = ParquetDailyIcSink(tmp_path)
    # Bypass run_registry_scan's own flush and rely on finalize alone.
    from modeler.analysis.horizon_scan_runner import scan_cell

    scan_cell(
        con,
        feature_col="px_feature",
        scan_type="cum",
        h_start=0,
        h_end=5,
        universe="broad",
        sample_kind="common_survivor",
        **SCAN_KWARGS,
        cell_identity=normalize_cell_identity(
            REGISTRY[0], universe="broad", sample_kind="common_survivor"
        ),
        daily_sink=sink,
    )
    assert not list(tmp_path.rglob("*.parquet"))
    summary = sink.finalize()
    assert summary.daily_ic.file_count == 1
    assert read_daily_ic(tmp_path).height == 30


def test_sink_refuses_to_overwrite_a_file_it_already_wrote(tmp_path: Path) -> None:
    _rows, sink = scan_with_sink(tmp_path)
    sink.finalize()
    sink.emit(
        normalize_cell_identity(REGISTRY[0], universe="broad", sample_kind="common_survivor"),
        daily=read_daily_ic(tmp_path).select(
            "trade_date", "formation_session_idx", "rank_ic", pl.col("n_obs").alias("n")
        ),
        market_ic=pl.DataFrame(
            {"trade_date": [], "market": [], "rank_ic": [], "n": []},
            schema={
                "trade_date": pl.Date,
                "market": pl.Utf8,
                "rank_ic": pl.Float64,
                "n": pl.Int64,
            },
        ),
    )
    with pytest.raises(FileExistsError, match="already written"):
        sink.flush_feature("px_feature")


def test_market_columns_are_zero_filled_when_a_market_is_absent(tmp_path: Path) -> None:
    con = duckdb.connect()
    seed_scan_panel(con)
    con.execute("DELETE FROM analysis_panel WHERE market = 'KOSDAQ' AND formation_session_idx = 3")
    sink = ParquetDailyIcSink(tmp_path)
    rows = run_registry_scan(con, REGISTRY, **SCAN_KWARGS, daily_sink=sink)
    sink.finalize()
    stored = read_daily_ic(tmp_path).filter(
        (pl.col("universe") == "broad")
        & (pl.col("sample_kind") == "common_survivor")
        & (pl.col("formation_session_idx") == 3)
    )
    assert stored.height == 1
    assert stored["n_kosdaq"].item() == 0
    assert stored["rank_ic_kosdaq"].item() is None
    assert stored["rank_ic"].item() == pytest.approx(stored["rank_ic_kospi"].item())
    assert reconcile(tmp_path, rows)["reconciled"] is True


# --- §4.1 reconciliation ---


def test_reconcile_accepts_a_run_whose_stored_series_rebuilds_every_summary(
    tmp_path: Path,
) -> None:
    rows, sink = scan_with_sink(tmp_path)
    sink.finalize()
    result = reconcile(tmp_path, rows)
    assert result["reconciled"] is True
    assert result["n_cells"] == 4
    assert result["max_abs_diff"] < 1e-12
    assert assert_daily_ic_reconciled(result) is result


def test_reconcile_handles_a_spread_only_date(tmp_path: Path) -> None:
    """The §2.3 reason the two artifacts are separate files.

    Session 7's realized ranks are constant, so both markets' rank ICs are NaN
    and the date is not in ``daily`` at all — but its quantile spread is real
    and is part of ``q5_spread_raw``. A single left-joined file would have
    dropped it and failed this reconciliation.
    """
    rows, sink = scan_with_sink(tmp_path, tied_label_sessions=(7,))
    sink.finalize()
    ic = read_daily_ic(tmp_path).filter(
        (pl.col("universe") == "broad") & (pl.col("sample_kind") == "common_survivor")
    )
    spread = read_daily_spread(tmp_path).filter(
        (pl.col("universe") == "broad") & (pl.col("sample_kind") == "common_survivor")
    )
    assert 7 not in ic["formation_session_idx"].to_list()
    assert spread.height == ic.height + 1
    broad = next(
        r for r in rows if r["universe"] == "broad" and r["sample_kind"] == "common_survivor"
    )
    assert spread["spread"].mean() == pytest.approx(broad["q5_spread_raw"], abs=1e-12)
    assert reconcile(tmp_path, rows)["reconciled"] is True


def test_reconcile_reports_a_summary_that_disagrees_with_the_stored_mean(
    tmp_path: Path,
) -> None:
    rows, sink = scan_with_sink(tmp_path)
    sink.finalize()
    rows[0] = {**rows[0], "ic_mean": rows[0]["ic_mean"] + 1e-6}
    result = reconcile(tmp_path, rows)
    assert result["reconciled"] is False
    assert {m["check"] for m in result["mismatches"]} == {"ic_mean"}
    assert result["max_abs_diff"] == pytest.approx(1e-6)
    with pytest.raises(RuntimeError, match="daily_ic reconciliation failed"):
        assert_daily_ic_reconciled(result)


@pytest.mark.parametrize(
    ("field", "delta", "check"),
    [
        ("ic_std", 1e-6, "ic_std"),
        ("t_nw", 1e-6, "t_nw"),
        ("n_obs", 1, "n_obs"),
        ("n_obs_min", 1, "n_obs_min"),
        ("n_obs_median", 1e-6, "n_obs_median"),
        ("n_hac_pairs_min", 1, "n_hac_pairs_min"),
        ("kospi_weight_mean", 1e-6, "kospi_weight_mean"),
        ("q5_spread_raw", 1e-6, "q5_spread_raw"),
    ],
)
def test_reconcile_checks_each_summary_field_it_promised(
    tmp_path: Path, field: str, delta: float, check: str
) -> None:
    rows, sink = scan_with_sink(tmp_path)
    sink.finalize()
    rows[0] = {**rows[0], field: rows[0][field] + delta}
    result = reconcile(tmp_path, rows)
    assert result["reconciled"] is False
    assert check in {m["check"] for m in result["mismatches"]}


def test_reconcile_rejects_a_valid_cell_with_no_stored_series(tmp_path: Path) -> None:
    rows, sink = scan_with_sink(tmp_path)
    sink.finalize()
    rows.append({**rows[0], "hypothesis_id": "other|px_feature|cum|0|5"})
    result = reconcile(tmp_path, rows)
    assert result["reconciled"] is False
    assert {m["check"] for m in result["mismatches"]} == {"missing_daily_ic"}


def test_reconcile_rejects_a_stored_cell_with_no_valid_summary_row(tmp_path: Path) -> None:
    rows, sink = scan_with_sink(tmp_path)
    sink.finalize()
    result = reconcile(tmp_path, rows[:-1])
    assert result["reconciled"] is False
    assert {m["check"] for m in result["mismatches"]} == {
        "unexpected_daily_ic",
        "unexpected_daily_spread",
    }


def test_reconcile_rejects_stored_rows_whose_market_weighting_is_inconsistent(
    tmp_path: Path,
) -> None:
    rows, sink = scan_with_sink(tmp_path)
    sink.finalize()
    target = tmp_path / DAILY_IC_DIR_NAME / "family=fam" / "px_feature.parquet"
    tampered = pl.read_parquet(target).with_columns(
        pl.when(pl.col("formation_session_idx") == 4)
        .then(pl.col("n_kospi") + 1)
        .otherwise(pl.col("n_kospi"))
        .alias("n_kospi")
    )
    tampered.write_parquet(target)
    result = reconcile(tmp_path, rows)
    assert result["reconciled"] is False
    assert "market_weighted_ic" in {m["check"] for m in result["mismatches"]}


def test_reconcile_ignores_insufficient_cells(tmp_path: Path) -> None:
    """An insufficient cell has no summary to check the stored series against,
    so §2.2 never emits one — and reconciliation must not then complain that
    it is missing."""
    con = duckdb.connect()
    seed_scan_panel(con)
    sink = ParquetDailyIcSink(tmp_path)
    rows = run_registry_scan(
        con, REGISTRY, **{**SCAN_KWARGS, "min_dates_per_cell": 60}, daily_sink=sink
    )
    summary = sink.finalize()
    assert all(r["status"] == "insufficient" for r in rows)
    assert summary.daily_ic.file_count == 0
    assert reconcile(tmp_path, rows) == {
        "reconciled": True,
        "n_cells": 0,
        "max_abs_diff": 0.0,
        "mismatches": [],
    }


def test_daily_ic_success_fields_are_the_two_documented_keys(tmp_path: Path) -> None:
    rows, sink = scan_with_sink(tmp_path)
    sink.finalize()
    fields = daily_ic_success_fields(reconcile(tmp_path, rows))
    assert fields == {"daily_ic_reconciled": True, "daily_ic_reconcile_max_abs_diff": 0.0}


# --- §2.1: run_registry_scan is the only scan_cell caller that may pass a sink ---


def test_only_run_registry_scan_hands_scan_cell_a_daily_sink() -> None:
    """The period-split scans (``compute_period_ics``,
    ``_compute_phase_b_period_ics``), the family lag1 calls and the replicate
    loops all call ``scan_cell`` directly. Stage 0 excludes them by never
    passing a sink — this pins that down at every call site at once, including
    ones added later.
    """
    import ast

    offenders: list[str] = []
    for path in sorted(Path("research").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if name != "scan_cell":
                    continue
                passes_sink = any(kw.arg == "daily_sink" for kw in call.keywords)
                if passes_sink and node.name != "run_registry_scan":
                    offenders.append(f"{path}:{call.lineno} in {node.name}()")
    assert offenders == []
