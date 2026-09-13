"""Unit tests for P6 — train (Ridge/ElasticNet walk-forward) on synthetic panels."""

from __future__ import annotations

import datetime

import numpy as np
import polars as pl
import pytest

from modeler.models._01_20_access_return_rank import train as tr


def _synthetic_panel(*, signal: bool, n_dates: int = 40, n_names: int = 30, seed: int = 0):
    """Std-like panel with fold metadata. If ``signal``, label depends on a feature."""
    rng = np.random.default_rng(seed)
    base = datetime.date(2020, 1, 1)
    rows = {
        "trade_date": [],
        "ticker": [],
        "market": [],
        "px_feat": [],
        "flow_feat": [],
        "y_rank_20d": [],
        "raw_label_20d": [],
        "fold_id": [],
        "fold_role": [],
    }

    # 2 folds: dates [0:18] train f1, [20:28] valid f1 ; [0:28] train f2, [30:38] valid f2.
    def role(i: int):
        if i < 18:
            return [(1, "train"), (2, "train")]
        if 20 <= i < 28:
            return [(1, "valid"), (2, "train")]
        if 30 <= i < 38:
            return [(2, "valid")]
        return []

    for i in range(n_dates):
        d = base + datetime.timedelta(days=i)
        assignments = role(i)
        if not assignments:
            continue
        px = rng.standard_normal(n_names)
        flow = rng.standard_normal(n_names)
        noise = rng.standard_normal(n_names)
        raw = (px * 0.5 + noise * 0.5) if signal else noise
        # per-date rank in [0,1]
        order = raw.argsort().argsort() / (n_names - 1)
        for fid, frole in assignments:
            for j in range(n_names):
                rows["trade_date"].append(d)
                rows["ticker"].append(f"T{j:02d}")
                rows["market"].append("KOSPI")
                rows["px_feat"].append(float(px[j]))
                rows["flow_feat"].append(float(flow[j]))
                rows["y_rank_20d"].append(float(order[j]))
                rows["raw_label_20d"].append(float(raw[j]))
                rows["fold_id"].append(fid)
                rows["fold_role"].append(frole)
    return pl.DataFrame(rows)


def test_design_columns_excludes_labels_and_meta() -> None:
    panel = _synthetic_panel(signal=True)
    cols = tr.design_columns(panel, "y_rank_20d")
    assert "px_feat" in cols
    assert "flow_feat" in cols
    for bad in ("y_rank_20d", "raw_label_20d", "fold_id", "trade_date"):
        assert bad not in cols


def test_walk_forward_runs_and_reports() -> None:
    panel = _synthetic_panel(signal=True)
    result = tr.walk_forward(panel, tr.TrainConfig(model="ridge"))
    assert result.fold_results
    assert "alpha" in result.best_params
    assert result.mean_rank_ic > 0.2  # strong feature -> high IC


def test_signal_beats_noise() -> None:
    sig = tr.walk_forward(_synthetic_panel(signal=True), tr.TrainConfig(model="ridge"))
    noise = tr.walk_forward(_synthetic_panel(signal=False, seed=1), tr.TrainConfig(model="ridge"))
    assert sig.mean_rank_ic > noise.mean_rank_ic
    assert abs(noise.mean_rank_ic) < 0.15  # no real signal -> IC near 0


def test_elasticnet_config_runs() -> None:
    panel = _synthetic_panel(signal=True)
    result = tr.walk_forward(
        panel, tr.TrainConfig(model="elasticnet", alphas=(0.01, 0.1), l1_ratios=(0.5,))
    )
    assert "l1_ratio" in result.best_params
    assert result.mean_rank_ic > 0.1


def test_target_nan_rows_dropped_in_fit() -> None:
    # inject NaN targets into train; fit must not raise (NaN guard).
    panel = _synthetic_panel(signal=True)
    poisoned = panel.with_columns(
        pl.when((pl.col("fold_role") == "train") & (pl.col("ticker") == "T00"))
        .then(float("nan"))
        .otherwise(pl.col("y_rank_20d"))
        .alias("y_rank_20d")
    )
    result = tr.walk_forward(poisoned, tr.TrainConfig(model="ridge"))
    assert result.fold_results  # completed without error


def test_invalid_model_raises() -> None:
    with pytest.raises(ValueError):
        tr.TrainConfig(model="svm")


def test_holdout_excluded_from_selection_and_used_once() -> None:
    """Fix #1: fold_role='holdout' rows are excluded from walk-forward selection
    and consumed only by evaluate_holdout()."""
    panel = _synthetic_panel(signal=True)
    # Re-label fold 2's valid rows as a holdout slice (its own fold_id retained).
    holdout = panel.with_columns(
        pl.when((pl.col("fold_id") == 2) & (pl.col("fold_role") == "valid"))
        .then(pl.lit("holdout"))
        .otherwise(pl.col("fold_role"))
        .alias("fold_role")
    )
    result = tr.walk_forward(holdout, tr.TrainConfig(model="ridge"))
    # Selection only saw fold 1's valid (the holdout fold has no 'valid' rows).
    assert all(fr.fold_id == 1 for fr in result.fold_results)

    report = tr.evaluate_holdout(holdout, result)
    assert report is not None  # holdout rows exist -> single eval runs
    assert report.n_dates > 0


def test_no_holdout_returns_none() -> None:
    panel = _synthetic_panel(signal=True)  # train/valid only
    result = tr.walk_forward(panel, tr.TrainConfig(model="ridge"))
    assert tr.evaluate_holdout(panel, result) is None
