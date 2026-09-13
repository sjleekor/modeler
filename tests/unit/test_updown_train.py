"""Unit tests for model 02's training, calibration and evaluation (PR6).

Everything is synthetic and tiny. The point is not to measure a model but to pin
the things that would silently corrupt a real run: the grid is the preregistered
one, the calibration slice never touches valid data, the same seed reproduces,
and a fold that cannot be fit is *reported* rather than crashing the run.
"""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modeler.models._02_updown_prob import calibrate as cal
from modeler.models._02_updown_prob import evaluate as ev
from modeler.models._02_updown_prob import features as fx
from modeler.models._02_updown_prob import train as tr


def _panel(
    *,
    horizon: int = 20,
    n_dates: int = 60,
    n_names: int = 30,
    signal: str = "perfect",
    n_folds: int = 2,
    seed: int = 7,
) -> pl.DataFrame:
    """A ``feat_panel_std``-shaped frame with a known relation to the label.

    Shaped like the real thing: expanding folds, each emitting its own train and
    valid slices, so one date appears as fold 1's valid and fold 2's train.

    ``signal="perfect"``: the label is a deterministic function of one feature.
    ``signal="none"``: the label is an independent coin flip — neither feature
    knows anything about it, which is what makes ln 2 the right answer.
    """
    # a fresh generator per call: a shared one would make each test's data
    # depend on how many tests ran before it
    rng = np.random.default_rng(seed)
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n_dates)]
    base: list[dict] = []
    for day in dates:
        for name in range(n_names):
            x = rng.uniform()
            noise = rng.uniform()
            coin = rng.uniform()
            up = int(x > 0.5) if signal == "perfect" else int(coin > 0.5)
            base.append(
                {
                    "trade_date": day,
                    "ticker": f"T{name:03d}",
                    "market": "KOSPI" if name % 2 else "KOSDAQ",
                    "px_ret_20d": x,
                    "px_vol_20d": noise,
                    f"y_up_{horizon}d": up,
                    f"y_top_{horizon}d": int(x > 0.8),
                    f"y_rank_{horizon}d": x,
                    f"raw_label_{horizon}d": (x - 0.5) * 0.1,
                }
            )
    frame = pl.DataFrame(base)

    per_fold = n_dates // (n_folds + 1)
    slices: list[pl.DataFrame] = []
    for fold_id in range(1, n_folds + 1):
        train_dates = dates[: fold_id * per_fold]
        valid_dates = dates[fold_id * per_fold : (fold_id + 1) * per_fold]
        for role, days in (("train", train_dates), ("valid", valid_dates)):
            slices.append(
                frame.filter(pl.col("trade_date").is_in(days)).with_columns(
                    pl.lit(fold_id).alias("fold_id"), pl.lit(role).alias("fold_role")
                )
            )
    return pl.concat(slices, how="vertical")


# --- the grids are the preregistered ones (`04` §2) --------------------------


def test_grid_sizes_and_the_e1_subset() -> None:
    assert len(tr.HGB_CLF_GRID) == 16
    assert len(tr.HGB_CLF_GRID_E1) == 4
    assert len(tr.LOGIT_GRID) == 3
    assert len(tr.HGB_REG_GRID) == 4
    assert set(map(str, tr.HGB_CLF_GRID_E1)) <= set(map(str, tr.HGB_CLF_GRID))
    for point in tr.HGB_CLF_GRID_E1:
        assert point["max_iter"] == 400
        assert point["l2_regularization"] == 0.0
    assert {p["max_leaf_nodes"] for p in tr.HGB_CLF_GRID} == {15, 31}
    assert {p["learning_rate"] for p in tr.HGB_CLF_GRID} == {0.03, 0.1}


def test_the_knobs_outside_the_grid_are_fixed() -> None:
    config = tr.TrainConfig(model="hgb_clf", target="y_up")
    model = tr.make_model(config, tr.HGB_CLF_GRID[0], ["px_ret_20d"])
    assert model.min_samples_leaf == 200
    assert model.max_bins == 255
    assert model.early_stopping is False
    assert model.class_weight is None  # reweighting positives breaks calibration
    assert model.random_state == 0


def test_monotonic_variant_passes_the_preregistered_signs() -> None:
    design = ["px_reversal_5d", "fin_log_mcap", "px_ret_1d"]
    config = tr.TrainConfig(model="hgb_clf", target="y_up", monotonic=True)
    model = tr.make_model(config, tr.HGB_CLF_GRID[0], design)
    assert model.monotonic_cst == [1, -1, 0]  # FS0's px_ret_1d has no prior sign
    assert fx.monotonic_constraints(tuple(design)) == (1, -1, 0)


def test_config_rejects_incoherent_combinations() -> None:
    with pytest.raises(ValueError, match="model must be"):
        tr.TrainConfig(model="lightgbm")
    with pytest.raises(ValueError, match="target must be"):
        tr.TrainConfig(target="y_absup")
    with pytest.raises(ValueError, match="rank baseline"):
        tr.TrainConfig(model="hgb_reg", target="y_up")
    with pytest.raises(ValueError, match="not a binary target"):
        tr.TrainConfig(model="hgb_clf", target="y_rank")
    with pytest.raises(ValueError, match="E4b"):
        tr.TrainConfig(model="logit", target="y_up", monotonic=True)
    with pytest.raises(ValueError, match="cal_target"):
        tr.TrainConfig(model="hgb_reg", target="y_rank", grid=tr.HGB_REG_GRID, calibrate="isotonic")
    with pytest.raises(ValueError, match="grid"):
        tr.TrainConfig(grid=())


# --- the calibration slice (`04` §3.2, §4) -----------------------------------


def test_cal_slice_is_the_trailing_fraction_with_an_embargo() -> None:
    dates = list(range(100))
    split = cal.split_cal_slice(dates, horizon=20, frac=0.2)
    assert split.cal_dates == list(range(80, 100))
    assert split.model_dates == list(range(60))
    # the 20 sessions between them belong to neither: a label formed on the last
    # model-train session resolves inside the calibration slice.
    assert set(split.model_dates) & set(split.cal_dates) == set()
    assert min(split.cal_dates) - max(split.model_dates) == 21


def test_cal_slice_refuses_to_leave_the_model_with_nothing() -> None:
    with pytest.raises(ValueError, match="no model-train sessions"):
        cal.split_cal_slice(list(range(20)), horizon=20, frac=0.2)
    with pytest.raises(ValueError, match="frac"):
        cal.split_cal_slice(list(range(100)), horizon=5, frac=1.0)


def test_a_single_class_calibration_slice_yields_no_calibrator() -> None:
    scores = np.array([0.2, 0.4, 0.6])
    assert cal.fit_calibrator("isotonic", scores, np.ones(3)) is None
    assert cal.fit_calibrator("none", scores, np.array([0.0, 1.0, 1.0])) is None
    assert cal.fit_calibrator("isotonic", scores, np.array([0.0, 1.0, 1.0])) is not None


def test_calibrators_stay_inside_the_log_loss_clip() -> None:
    scores = np.linspace(0, 1, 50)
    y = (scores > 0.5).astype(float)
    for method in ("isotonic", "platt"):
        calibrator = cal.fit_calibrator(method, scores, y)
        out = calibrator.transform(np.array([-5.0, 0.5, 5.0]))
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert not np.isclose(out.min(), 0.0, atol=0.0) or out.min() > 0.0


# --- extreme cases -----------------------------------------------------------


def test_a_perfectly_predictable_label_drives_log_loss_to_zero() -> None:
    result = tr.walk_forward(_panel(signal="perfect"), tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:1]))
    assert result.best_metric < 0.05
    assert result.config.selection_metric == "log_loss"


def test_an_unpredictable_label_lands_at_ln_two() -> None:
    result = tr.walk_forward(_panel(signal="none"), tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:1]))
    assert result.best_metric == pytest.approx(math.log(2), abs=0.05)


def test_logistic_model_runs_on_the_same_panel() -> None:
    config = tr.TrainConfig(model="logit", target="y_up", grid=tr.LOGIT_GRID)
    result = tr.walk_forward(_panel(signal="perfect"), config)
    assert result.best_params in tr.LOGIT_GRID
    assert result.best_metric < 0.35
    assert len(result.grid_metrics) == 3


def test_null_features_do_not_break_the_linear_model() -> None:
    panel = _panel(signal="perfect").with_columns(
        pl.when(pl.col("ticker") == "T000")
        .then(None)
        .otherwise(pl.col("px_vol_20d"))
        .alias("px_vol_20d")
    )
    config = tr.TrainConfig(model="logit", target="y_up", grid=tr.LOGIT_GRID[:1])
    result = tr.walk_forward(panel, config)
    assert result.best_metric == result.best_metric  # not NaN


def test_a_column_with_no_observed_value_does_not_break_the_binner() -> None:
    """The short-selling families start 2016-06-30, so an early fold has none.

    sklearn 1.9's histogram binner cannot bin an all-missing column: it raises
    numpy's "window shape cannot be larger than input array shape" from inside
    ``_BinMapper.fit``, naming nothing. Such a column is filled to a constant
    and reported, because a feature with no values carries no information
    either way and dropping it would change the design matrix per fold.
    """
    panel = _panel(signal="perfect").with_columns(
        pl.lit(None, pl.Float64).alias("flow_short_balance_qty")
    )
    result = tr.walk_forward(panel, tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:1]))
    assert result.best_metric < 0.05  # the informative feature still works
    reported = result.summary()["empty_design_columns"]
    assert reported and all("flow_short_balance_qty" in columns for columns in reported.values())


def test_empty_columns_finds_all_missing_columns_only() -> None:
    frame = pl.DataFrame(
        {
            "a": [1.0, 2.0, None],
            "b": [None, None, None],
            "c": [float("nan"), float("nan"), float("nan")],
            "d": [1, 2, 3],
        }
    )
    assert tr.empty_columns(frame, ["a", "b", "c", "d"]) == ["b", "c"]


# --- where the folds come from (the memory fix) ------------------------------


def _fold_table(panel: pl.DataFrame) -> pl.DataFrame:
    """The ``split_folds`` rows implied by a synthetic panel's fold tagging."""
    rows = []
    for fold_id in sorted(panel["fold_id"].unique().to_list()):
        fold = panel.filter(pl.col("fold_id") == fold_id)
        train = fold.filter(pl.col("fold_role") == "train")["trade_date"]
        valid = fold.filter(pl.col("fold_role") == "valid")["trade_date"]
        rows.append(
            {
                "fold_id": fold_id,
                "role": "fold",
                "train_start": train.min(),
                "train_end": train.max(),
                "valid_start": valid.min(),
                "valid_end": valid.max(),
            }
        )
    return pl.DataFrame(rows)


def _write_dataset(directory: Path, panel: pl.DataFrame, *, per_fold: bool) -> Path:
    std = directory / "feat_panel_std"
    std.mkdir(parents=True, exist_ok=True)
    _fold_table(panel).write_parquet(directory / "split_folds.parquet")
    if per_fold:
        for fold_id in sorted(panel["fold_id"].unique().to_list()):
            for role in ("train", "valid"):
                part = panel.filter((pl.col("fold_id") == fold_id) & (pl.col("fold_role") == role))
                part.write_parquet(std / f"fold{fold_id}_{role}.parquet")
    else:
        # the stateless layout: one panel, no fold columns
        panel.filter(pl.col("fold_id") == panel["fold_id"].max()).drop(
            ["fold_id", "fold_role"]
        ).unique(subset=["trade_date", "ticker"]).write_parquet(std / "part-000000.parquet")
    return directory


def test_dataset_folds_reads_the_per_fold_parts(tmp_path: Path) -> None:
    panel = _panel(signal="perfect")
    _write_dataset(tmp_path, panel, per_fold=True)
    source = tr.DatasetFolds(tmp_path)
    assert source.fold_ids() == sorted(panel["fold_id"].unique().to_list())
    train, valid = source.slices(2)
    assert train["fold_id"].unique().to_list() == [2]
    assert valid["fold_role"].unique().to_list() == ["valid"]
    assert "px_ret_20d" in source.design_columns("y_up_20d")
    assert "y_up_20d" not in source.design_columns("y_up_20d")


def test_dataset_folds_slices_the_single_layout_by_date_range(tmp_path: Path) -> None:
    """The stateless layout has no fold columns — the fold table supplies them."""
    panel = _panel(signal="perfect")
    _write_dataset(tmp_path, panel, per_fold=False)
    source = tr.DatasetFolds(tmp_path)
    table = _fold_table(panel)
    for fold_id in source.fold_ids():
        row = table.filter(pl.col("fold_id") == fold_id).row(0, named=True)
        train, valid = source.slices(fold_id)
        assert train["trade_date"].min() == row["train_start"]
        assert train["trade_date"].max() == row["train_end"]
        assert valid["trade_date"].min() == row["valid_start"]
        assert train["fold_role"].unique().to_list() == ["train"]


def test_a_run_gives_the_same_answer_from_either_fold_source(tmp_path: Path) -> None:
    """The memory fix must not change a single number."""
    panel = _panel(signal="perfect")
    _write_dataset(tmp_path, panel, per_fold=True)
    config = tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:2])
    in_memory = tr.walk_forward(panel, config)
    from_disk = tr.walk_forward(tr.DatasetFolds(tmp_path), config)
    assert from_disk.best_params == in_memory.best_params
    assert from_disk.best_metric == pytest.approx(in_memory.best_metric, abs=1e-12)
    assert [m["mean_metric"] for m in from_disk.grid_metrics] == pytest.approx(
        [m["mean_metric"] for m in in_memory.grid_metrics], abs=1e-12
    )
    columns = ["trade_date", "ticker", "p_raw"]
    assert (
        from_disk.predictions.select(columns)
        .sort(columns)
        .equals(in_memory.predictions.select(columns).sort(columns))
    )


def test_a_dataset_without_a_design_matrix_says_so(tmp_path: Path) -> None:
    (tmp_path / "feat_panel_std").mkdir()
    with pytest.raises(FileNotFoundError, match="no design matrix"):
        tr.DatasetFolds(tmp_path)


# --- determinism (`04` §6) ---------------------------------------------------


def test_the_same_seed_reproduces_the_same_metrics() -> None:
    panel = _panel(signal="none")
    config = tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:2], seed=0)
    first = tr.walk_forward(panel, config)
    second = tr.walk_forward(panel, config)
    assert [round(m["mean_metric"], 12) for m in first.grid_metrics] == [
        round(m["mean_metric"], 12) for m in second.grid_metrics
    ]
    assert first.predictions.equals(second.predictions)


def test_the_grid_sweep_and_the_refit_agree_on_the_winner() -> None:
    """The winner is refit to produce predictions; it must be the same model."""
    panel = _panel(signal="perfect")
    config = tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:2])
    result = tr.walk_forward(panel, config)
    swept = {str(m | {}): m["mean_metric"] for m in result.grid_metrics}
    refit = [f.metric for f in result.folds if f.skipped is None]
    assert result.best_metric == pytest.approx(float(np.mean(refit)), abs=1e-12)
    assert result.best_metric == pytest.approx(min(v for v in swept.values() if v == v), abs=1e-12)


# --- calibration end to end --------------------------------------------------


def test_isotonic_calibration_is_a_monotone_map_of_the_raw_probability() -> None:
    """Monotone *within a fold* — each fold fits its own calibrator (`04` §3.2).

    Pooling the folds and sorting globally would not be monotone, and should not
    be: two folds' calibrators are two different maps.
    """
    config = tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:1], calibrate="isotonic", horizon=5)
    result = tr.walk_forward(_panel(horizon=5, n_dates=90), config)
    assert result.predictions["p_cal"].null_count() == 0
    checked = 0
    for fold in result.folds:
        if fold.predictions is None:
            continue
        ordered = fold.predictions.sort("p_raw")
        diffs = ordered["p_cal"].diff().drop_nulls().to_numpy()
        assert (diffs >= -1e-12).all()
        checked += 1
    assert checked >= 2


def test_the_rank_baseline_gets_a_probability_only_through_calibration() -> None:
    """M-A: trained on y_rank, calibrated onto y_up (`04` §1, §3.3)."""
    config = tr.TrainConfig(
        model="hgb_reg",
        target="y_rank",
        grid=tr.HGB_REG_GRID[:1],
        calibrate="isotonic",
        cal_target="y_up",
        horizon=5,
    )
    result = tr.walk_forward(_panel(horizon=5, n_dates=90), config)
    assert config.selection_metric == "rank_ic"
    assert config.cal_target_column == "y_up_5d"
    predictions = result.predictions
    assert predictions["p_cal"].null_count() == 0
    assert predictions["p_cal"].max() <= 1.0

    report = ev.evaluate_run(result, k=5)
    # p_raw is a rank score: it must not be scored as a probability
    assert set(report.fold_metrics["pred_col"].drop_nulls().to_list()) == {"p_cal"}
    assert report.summary["primary_pred_col"] == "p_cal"


# --- a fold that cannot be fit is reported, not fatal ------------------------


def test_a_single_class_fold_is_recorded_as_skipped() -> None:
    panel = _panel(signal="perfect").with_columns(
        pl.when(pl.col("fold_id") == 1).then(1).otherwise(pl.col("y_up_20d")).alias("y_up_20d")
    )
    result = tr.walk_forward(panel, tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:1]))
    skipped = {f.fold_id: f.skipped for f in result.folds if f.skipped}
    assert skipped and "single class" in next(iter(skipped.values()))
    assert result.summary()["skipped_folds"] == skipped


def test_a_label_that_never_resolves_raises_instead_of_reporting_a_number() -> None:
    panel = _panel(signal="perfect").with_columns(pl.lit(None, pl.Int64).alias("y_up_20d"))
    with pytest.raises(ValueError, match="no grid point"):
        tr.walk_forward(panel, tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:1]))


# --- evaluation (`03` §3.2) --------------------------------------------------


def test_the_primary_metrics_are_looked_up_per_horizon() -> None:
    assert ev.PRIMARY_PROBABILITY == "log_loss"
    assert ev.PRIMARY_ECONOMIC == {
        5: "topk_cost_adjusted_return",
        20: "topk_cost_adjusted_return",
        60: "topk_cost_adjusted_return",
        120: "rank_ic_mean",
    }


def test_evaluate_run_reports_both_primaries_and_the_fold_spread() -> None:
    result = tr.walk_forward(_panel(signal="perfect"), tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:1]))
    report = ev.evaluate_run(result, k=5, cost_bps=60.0, tau=0.6)

    primary = report.primary()
    assert primary["probability_metric"] == "log_loss"
    assert primary["economic_metric"] == "topk_cost_adjusted_return"
    assert primary["probability_value"] < 0.1
    assert report.summary["k"] == 5
    assert report.summary["tau"] == 0.6
    assert report.summary["cost_bps_roundtrip"] == 60.0
    # the three portfolios all reported, per fold
    assert not report.economics.is_empty()
    for prefix in ("topk_", "tau_", "decile_"):
        assert any(c.startswith(prefix) for c in report.economics.columns)
    assert set(report.yearly["bucket"].to_list()) == {"2020"}
    assert set(report.by_market["bucket"].to_list()) == {"KOSPI", "KOSDAQ"}
    assert not report.reliability.is_empty()
    assert "pooled" in report.reliability["scope"].to_list()


def test_h120_is_judged_on_rank_ic_not_on_the_thin_topk_grid() -> None:
    panel = _panel(horizon=120, n_dates=90)
    config = tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:1], horizon=120)
    report = ev.evaluate_run(tr.walk_forward(panel, config), k=5)
    assert report.summary["economic_metric"] == "rank_ic_mean"
    assert report.summary["economic_value"] == report.summary["economic_value"]  # not NaN


def test_the_ece_ceiling_is_recorded_with_the_summary() -> None:
    result = tr.walk_forward(_panel(signal="none"), tr.TrainConfig(grid=tr.HGB_CLF_GRID_E1[:1]))
    report = ev.evaluate_run(result, k=5)
    assert report.summary["ece_ceiling"] == ev.ECE_CEILING == 0.03
    assert isinstance(report.summary["ece_over_ceiling"], bool)
