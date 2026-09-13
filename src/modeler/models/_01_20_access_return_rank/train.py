"""train — Ridge/ElasticNet walk-forward training + Rank IC report (etl_00 §6).

Reads the standardized panel (``feat_panel_std.parquet`` from P5), runs an
expanding walk-forward: for each fold, fit on the train slice and predict the
valid slice, then evaluate with the ranking metrics (etl_00 §6). Hyperparameters
(``alpha`` for Ridge, ``alpha``/``l1_ratio`` for ElasticNet) are selected by the
mean walk-forward Rank IC. The selected model can then be evaluated once on a
holdout fold (etl_00 §5, §6).

Engine boundary (etl_02 §6): the heavy lake/feature work is done (DuckDB) by the
time we get here; this stage is Polars -> numpy -> sklearn. ``*_isna`` flag
columns are included in the design matrix alongside the standardized features
(L1, etl_00 §4.2).

The target is the per-date rank label (``y_rank_20d``); the report ranks against
the raw excess return (``raw_label_20d``) so the top-decile spread is in return
units (etl_00 §6).

See ``etl_00`` §5, §6, ``etl_02`` §6, ``etl_03_implementation_plan.md`` §4 (P6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge

from modeler.etl.metrics import RankICReport, evaluate

DEFAULT_TARGET = "y_rank_20d"
DEFAULT_REALIZED = "raw_label_20d"


@dataclass(frozen=True)
class TrainConfig:
    """Model + hyperparameter grid for walk-forward selection (etl_00 §6)."""

    model: str = "ridge"  # "ridge" | "elasticnet" | "hgb"
    alphas: tuple[float, ...] = (0.1, 1.0, 10.0)
    l1_ratios: tuple[float, ...] = (0.1, 0.5, 0.9)  # elasticnet only
    max_iters: tuple[int, ...] = (100, 200)  # hgb only
    learning_rates: tuple[float, ...] = (0.01, 0.1)  # hgb only
    target: str = DEFAULT_TARGET
    realized: str = DEFAULT_REALIZED
    date_col: str = "trade_date"
    id_cols: tuple[str, ...] = ("ticker", "market")  # kept in prediction frames

    def __post_init__(self) -> None:
        if self.model not in ("ridge", "elasticnet", "hgb"):
            raise ValueError(f"model must be 'ridge', 'elasticnet', or 'hgb', got {self.model!r}")


@dataclass
class FoldResult:
    fold_id: int
    params: dict
    report: RankICReport


@dataclass
class TrainResult:
    """Walk-forward outcome + selected hyperparameters (etl_00 §6)."""

    config: TrainConfig
    fold_results: list[FoldResult]
    best_params: dict
    mean_rank_ic: float
    holdout_report: RankICReport | None = None
    feature_cols: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "model": self.config.model,
            "best_params": self.best_params,
            "mean_rank_ic": self.mean_rank_ic,
            "n_folds": len(self.fold_results),
            "per_fold_rank_ic": [fr.report.rank_ic_mean for fr in self.fold_results],
            "holdout": self.holdout_report.as_dict() if self.holdout_report else None,
        }


def design_columns(panel: pl.DataFrame, target: str) -> list[str]:
    """Model-input columns: standardized features + ``*_isna`` flags (etl_00 §4.2).

    Excludes keys, label columns, forward/bench helpers, and fold metadata.
    """
    drop_prefixes = ("y_", "raw_label", "fwd_ret_", "bench_ret_")
    drop_exact = {"trade_date", "ticker", "market", "fold_id", "fold_role", target}
    cols = []
    for name, dtype in zip(panel.columns, panel.dtypes):
        if name in drop_exact or any(name.startswith(p) for p in drop_prefixes):
            continue
        if dtype.is_numeric():
            cols.append(name)
    return cols


def _make_model(config: TrainConfig, params: dict):
    if config.model == "ridge":
        return Ridge(alpha=params["alpha"])
    if config.model == "elasticnet":
        return ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=5000)
    return HistGradientBoostingRegressor(
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        random_state=42,
    )


def _param_grid(config: TrainConfig) -> list[dict]:
    if config.model == "ridge":
        return [{"alpha": a} for a in config.alphas]
    if config.model == "elasticnet":
        return [{"alpha": a, "l1_ratio": r} for a in config.alphas for r in config.l1_ratios]
    return [
        {"max_iter": n, "learning_rate": lr}
        for n in config.max_iters
        for lr in config.learning_rates
    ]


def _fit_predict(
    train_df: pl.DataFrame,
    valid_df: pl.DataFrame,
    feature_cols: list[str],
    config: TrainConfig,
    params: dict,
) -> pl.DataFrame:
    """Fit on train, predict valid; returns valid rows + a ``pred`` column."""
    # Drop rows whose target is null OR NaN (t+H beyond the panel -> no label).
    tr = train_df.filter(pl.col(config.target).is_not_null() & pl.col(config.target).is_not_nan())
    x_tr = tr.select(feature_cols).to_numpy()
    y_tr = tr.get_column(config.target).to_numpy()
    model = _make_model(config, params)
    model.fit(x_tr, y_tr)

    x_va = valid_df.select(feature_cols).to_numpy()
    preds = model.predict(x_va)
    # Keep identifier columns (ticker/market) when present so downstream stock
    # selection can recover "which name"; metrics ignore the extra columns.
    id_cols = [c for c in config.id_cols if c in valid_df.columns]
    keep = [config.date_col, *id_cols, config.realized, config.target]
    return valid_df.select(keep).with_columns(pl.Series("pred", preds))


def walk_forward(
    panel_std: pl.DataFrame,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Run walk-forward training + hyperparameter selection on a std panel.

    The panel must carry ``fold_id``/``fold_role`` columns (from P5). For each
    candidate param set, every fold is fit on its train slice and scored on its
    valid slice; the param set with the highest mean Rank IC wins.
    """
    config = config or TrainConfig()
    feature_cols = design_columns(panel_std, config.target)

    wf = panel_std.filter(pl.col("fold_role").is_in(["train", "valid"]))
    fold_ids = sorted(wf.get_column("fold_id").unique().to_list())

    best = None
    for params in _param_grid(config):
        fold_results: list[FoldResult] = []
        for fid in fold_ids:
            train_df = wf.filter((pl.col("fold_id") == fid) & (pl.col("fold_role") == "train"))
            valid_df = wf.filter((pl.col("fold_id") == fid) & (pl.col("fold_role") == "valid"))
            if train_df.height == 0 or valid_df.height == 0:
                continue
            preds = _fit_predict(train_df, valid_df, feature_cols, config, params)
            report = evaluate(
                preds,
                pred_col="pred",
                realized_col=config.realized,
                date_col=config.date_col,
            )
            fold_results.append(FoldResult(fold_id=fid, params=params, report=report))

        ics = [fr.report.rank_ic_mean for fr in fold_results if not _isnan(fr.report.rank_ic_mean)]
        mean_ic = float(np.mean(ics)) if ics else float("-inf")
        if best is None or mean_ic > best[1]:
            best = (params, mean_ic, fold_results)

    assert best is not None  # _param_grid is always non-empty
    best_params, mean_ic, fold_results = best
    return TrainResult(
        config=config,
        fold_results=fold_results,
        best_params=best_params,
        mean_rank_ic=mean_ic,
        feature_cols=feature_cols,
    )


def evaluate_holdout(
    panel_std: pl.DataFrame,
    result: TrainResult,
) -> RankICReport | None:
    """Fit the selected model on the holdout fold's train slice, score the holdout.

    Returns None if the panel has no ``fold_role == 'holdout'`` rows. This is the
    single post-selection evaluation (etl_00 §5). The model is fit on the train
    rows of the SAME fold_id as the holdout (those rows were standardized with
    that fold's train-fit bounds), keeping the standardization consistent.
    """
    config = result.config
    holdout = panel_std.filter(pl.col("fold_role") == "holdout")
    if holdout.height == 0:
        return None
    holdout_fold_id = holdout.get_column("fold_id")[0]
    train_df = panel_std.filter(
        (pl.col("fold_role") == "train") & (pl.col("fold_id") == holdout_fold_id)
    )
    if train_df.height == 0:
        return None
    preds = _fit_predict(train_df, holdout, result.feature_cols, config, result.best_params)
    report = evaluate(
        preds, pred_col="pred", realized_col=config.realized, date_col=config.date_col
    )
    result.holdout_report = report
    return report


def predict_holdout_frame(
    panel_std: pl.DataFrame,
    result: TrainResult,
) -> pl.DataFrame | None:
    """Keyed predictions over the whole holdout slice (for stock selection).

    Mirrors :func:`evaluate_holdout` (fit the selected model on the holdout
    fold's train slice), but RETURNS the per-row prediction frame with identifier
    columns (``ticker``/``market``) instead of only the aggregated report. Rows
    whose realized label is still null (the most recent ~H sessions) are kept so
    the latest cross-section can be ranked for a live buy list.

    Returns None if the panel has no ``fold_role == 'holdout'`` rows.
    """
    config = result.config
    holdout = panel_std.filter(pl.col("fold_role") == "holdout")
    if holdout.height == 0:
        return None
    holdout_fold_id = holdout.get_column("fold_id")[0]
    train_df = panel_std.filter(
        (pl.col("fold_role") == "train") & (pl.col("fold_id") == holdout_fold_id)
    )
    if train_df.height == 0:
        return None
    return _fit_predict(train_df, holdout, result.feature_cols, config, result.best_params)


def train_from_dataset(
    dataset_dir: Path,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Load ``feat_panel_std.parquet`` from a built dataset dir and train.

    Convenience entrypoint pairing with :func:`build_dataset`. The std panel
    already carries fold metadata (P5), so this is read -> walk_forward.
    """
    panel_std = pl.read_parquet(dataset_dir / "feat_panel_std.parquet")
    result = walk_forward(panel_std, config)
    evaluate_holdout(panel_std, result)
    return result


def _isnan(x: float) -> bool:
    return x != x
