"""evaluate — turn a run's fold predictions into the preregistered report set.

Plan: ``docs/dev/20260907_model_experiment/03_validation_and_metrics.md`` §2
(the metrics), §3.2 (which two are primary at each h), §6 (what a run records).

Two things are deliberate.

**The primary metrics are looked up, not chosen.** ``PRIMARY_PROBABILITY`` is
valid log-loss at every horizon and ``PRIMARY_ECONOMIC`` is the cost-adjusted
top-k return except at h120, where 22 non-overlapping rebalances make that
number too noisy to judge on and Rank IC takes over. Everything else this
module computes is secondary and labelled as such, so a later reader cannot
quietly promote whichever number came out best.

**A rank score is not read as a probability.** M-A trains on ``y_rank`` and its
raw output is a score; only its isotonic-calibrated column gets log-loss, Brier
and ECE (`04` §3.3). The classifiers get both, with ``p_raw`` primary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from modeler.etl.metrics import (
    ClassificationReport,
    classification_report,
    economic_report,
    reliability_table,
    threshold_economic_report,
    topk_economic_report,
    topk_rebalance_series,
)
from modeler.etl.metrics import (
    evaluate as rank_evaluate,
)
from modeler.models._02_updown_prob.train import TrainResult

PRIMARY_PROBABILITY = "log_loss"
PRIMARY_ECONOMIC: dict[int, str] = {
    5: "topk_cost_adjusted_return",
    20: "topk_cost_adjusted_return",
    60: "topk_cost_adjusted_return",
    120: "rank_ic_mean",
}
# `03` §3.3's calibration condition: a candidate over this is re-judged on its
# calibrated column, and dropped if it is still over.
ECE_CEILING = 0.03


@dataclass
class RunEvaluation:
    """Everything one run records (`03` §6), as frames plus a summary dict."""

    fold_metrics: pl.DataFrame
    reliability: pl.DataFrame
    economics: pl.DataFrame
    rebalance_returns: pl.DataFrame
    yearly: pl.DataFrame
    by_market: pl.DataFrame
    summary: dict

    def primary(self) -> dict:
        """The two preregistered numbers, and the fold spread beside them."""
        return {
            key: self.summary[key]
            for key in (
                "horizon",
                "probability_metric",
                "probability_value",
                "probability_fold_std",
                "economic_metric",
                "economic_value",
                "economic_fold_std",
                "ece",
                "n_folds",
            )
        }


def _probability_columns(result: TrainResult) -> list[str]:
    """Which prediction columns may be read as probabilities.

    ``p_raw`` only for a classifier: for M-A it is a rank score, and log-loss on
    a rank score is a number with no meaning (`04` §3.3).
    """
    columns = ["p_raw"] if result.config.is_classifier else []
    if any(f.calibrated for f in result.folds):
        columns.append("p_cal")
    return columns


def _classification(frame: pl.DataFrame, pred: str, y: str, k: int) -> ClassificationReport:
    return classification_report(frame, pred_col=pred, y_col=y, k=k)


def fold_rows(result: TrainResult, *, k: int, cost_bps: float, tau: float) -> list[dict]:
    """One record per (fold, prediction column) with every metric on that slice."""
    config = result.config
    y_col = config.cal_target_column
    rows: list[dict] = []
    for fold in result.folds:
        if fold.predictions is None:
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "pred_col": None,
                    "skipped": fold.skipped,
                    "n_valid": fold.n_valid,
                }
            )
            continue
        frame = fold.predictions
        rank = rank_evaluate(
            frame,
            pred_col="p_raw",
            realized_col=config.realized_column,
            date_col=config.date_col,
        )
        for pred_col in _probability_columns(result):
            if frame.get_column(pred_col).null_count() == frame.height:
                continue
            report = _classification(frame, pred_col, y_col, k)
            topk = topk_economic_report(
                frame,
                pred_col=pred_col,
                realized_col=config.realized_column,
                horizon=config.horizon,
                k=k,
                date_col=config.date_col,
                cost_bps_roundtrip=cost_bps,
            )
            threshold = threshold_economic_report(
                frame,
                pred_col=pred_col,
                realized_col=config.realized_column,
                horizon=config.horizon,
                tau=tau,
                date_col=config.date_col,
                cost_bps_roundtrip=cost_bps,
            )
            decile = economic_report(
                frame,
                pred_col=pred_col,
                realized_col=config.realized_column,
                horizon=config.horizon,
                date_col=config.date_col,
                cost_bps_roundtrip=cost_bps,
            )
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "pred_col": pred_col,
                    "skipped": None,
                    "n_train": fold.n_train,
                    "n_valid": fold.n_valid,
                    "calibrated": fold.calibrated,
                    **report.as_dict(),
                    **rank.as_dict(),
                    **{f"topk_{key}": value for key, value in topk.as_dict().items()},
                    **{f"tau_{key}": value for key, value in threshold.as_dict().items()},
                    **{f"decile_{key}": value for key, value in decile.as_dict().items()},
                }
            )
    return rows


def _rebalance_returns(result: TrainResult, *, k: int, cost_bps: float) -> pl.DataFrame:
    """The top-k return path per fold, not just the fold's mean (`03` §6).

    ``economics.parquet`` records what the buy list earned over a fold; this
    records what it earned at each rebalance. Strategy drawdown, buy-hold
    hysteresis and CSCV all read a path, and a run that keeps only the means
    cannot be re-read for any of them without training again.

    Nothing here feeds a preregistered metric — the fold numbers still come from
    ``topk_economic_report``. It is a record, written once so it need not be
    recomputed.
    """
    config = result.config
    frames: list[pl.DataFrame] = []
    for fold in result.folds:
        if fold.predictions is None:
            continue
        for pred_col in _probability_columns(result):
            if fold.predictions.get_column(pred_col).null_count() == fold.predictions.height:
                continue
            series = topk_rebalance_series(
                fold.predictions,
                pred_col=pred_col,
                realized_col=config.realized_column,
                horizon=config.horizon,
                k=k,
                date_col=config.date_col,
                cost_bps_roundtrip=cost_bps,
            )
            if series.is_empty():
                continue
            frames.append(
                series.with_columns(
                    pl.lit(fold.fold_id).alias("fold_id"),
                    pl.lit(pred_col).alias("pred_col"),
                ).select(["fold_id", "pred_col", *series.columns])
            )
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()


def _breakdown(result: TrainResult, by: str, *, k: int) -> pl.DataFrame:
    """Log-loss / Rank IC per calendar year or per market (`03` §2.4)."""
    config = result.config
    predictions = result.predictions
    if predictions.is_empty():
        return pl.DataFrame()
    if by == "year":
        predictions = predictions.with_columns(
            pl.col(config.date_col).dt.year().alias("bucket").cast(pl.Utf8)
        )
    else:
        predictions = predictions.with_columns(pl.col("market").alias("bucket"))

    pred_cols = _probability_columns(result)
    rows: list[dict] = []
    for (bucket,), group in predictions.group_by(["bucket"], maintain_order=True):
        rank = rank_evaluate(
            group,
            pred_col="p_raw",
            realized_col=config.realized_column,
            date_col=config.date_col,
        )
        for pred_col in pred_cols:
            if group.get_column(pred_col).null_count() == group.height:
                continue
            report = _classification(group, pred_col, config.cal_target_column, k)
            rows.append(
                {
                    "bucket": bucket,
                    "pred_col": pred_col,
                    "n_obs": report.n_obs,
                    "log_loss": report.log_loss,
                    "brier": report.brier,
                    "ece": report.ece,
                    "auc_daily_mean": report.auc_daily_mean,
                    "rank_ic_mean": rank.rank_ic_mean,
                    "base_rate": report.base_rate,
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None)


def _reliability(result: TrainResult, *, n_bins: int) -> pl.DataFrame:
    """Pooled and per-fold calibration tables (`03` §6 ``reliability.parquet``)."""
    config = result.config
    frames: list[pl.DataFrame] = []
    for pred_col in _probability_columns(result):
        pooled = result.predictions
        if pooled.is_empty() or pooled.get_column(pred_col).null_count() == pooled.height:
            continue
        table = reliability_table(
            pooled, pred_col=pred_col, y_col=config.cal_target_column, n_bins=n_bins
        )
        frames.append(
            table.with_columns(pl.lit("pooled").alias("scope"), pl.lit(pred_col).alias("pred_col"))
        )
        for fold in result.folds:
            if fold.predictions is None:
                continue
            per_fold = reliability_table(
                fold.predictions,
                pred_col=pred_col,
                y_col=config.cal_target_column,
                n_bins=n_bins,
            )
            frames.append(
                per_fold.with_columns(
                    pl.lit(f"fold{fold.fold_id}").alias("scope"),
                    pl.lit(pred_col).alias("pred_col"),
                )
            )
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()


def evaluate_run(
    result: TrainResult,
    *,
    k: int = 100,
    cost_bps: float = 60.0,
    tau: float = 0.6,
    n_bins: int = 10,
) -> RunEvaluation:
    """Full report set for one run.

    ``k``/``cost_bps``/``tau`` default to D-4's assumptions and are passed
    through from the spec by the runner, so a run's economics can be read
    against model 01's gate without re-deriving anything.
    """
    config = result.config
    rows = fold_rows(result, k=k, cost_bps=cost_bps, tau=tau)
    fold_metrics = pl.DataFrame(rows, infer_schema_length=None)

    primary_pred = "p_raw" if config.is_classifier else "p_cal"
    scored = (
        fold_metrics.filter(pl.col("pred_col") == primary_pred)
        if "pred_col" in fold_metrics.columns
        else pl.DataFrame()
    )
    economic_metric = PRIMARY_ECONOMIC[config.horizon]
    probability_value = _mean(scored, PRIMARY_PROBABILITY)
    economic_value = _mean(scored, economic_metric)

    summary = {
        **result.summary(),
        "primary_pred_col": primary_pred,
        "probability_metric": PRIMARY_PROBABILITY,
        "probability_value": probability_value,
        "probability_fold_std": _std(scored, PRIMARY_PROBABILITY),
        "economic_metric": economic_metric,
        "economic_value": economic_value,
        "economic_fold_std": _std(scored, economic_metric),
        "ece": _mean(scored, "ece"),
        "ece_ceiling": ECE_CEILING,
        "ece_over_ceiling": bool(_mean(scored, "ece") > ECE_CEILING),
        "rank_ic_mean": _mean(scored, "rank_ic_mean"),
        "brier": _mean(scored, "brier"),
        "auc_daily_mean": _mean(scored, "auc_daily_mean"),
        "precision_at_k": _mean(scored, "precision_at_k"),
        "lift_at_k": _mean(scored, "lift_at_k"),
        "topk_turnover": _mean(scored, "topk_turnover"),
        "base_rate": _mean(scored, "base_rate"),
        "k": k,
        "cost_bps_roundtrip": cost_bps,
        "tau": tau,
        "n_folds_scored": int(scored.height),
    }
    return RunEvaluation(
        fold_metrics=fold_metrics,
        reliability=_reliability(result, n_bins=n_bins),
        economics=_economics(fold_metrics),
        rebalance_returns=_rebalance_returns(result, k=k, cost_bps=cost_bps),
        yearly=_breakdown(result, "year", k=k),
        by_market=_breakdown(result, "market", k=k),
        summary=summary,
    )


def _economics(fold_metrics: pl.DataFrame) -> pl.DataFrame:
    """The three portfolios' columns, split out of the fold table (`03` §6)."""
    if fold_metrics.is_empty() or "pred_col" not in fold_metrics.columns:
        return pl.DataFrame()
    keep = ["fold_id", "pred_col"]
    prefixes = ("topk_", "tau_", "decile_")
    columns = [c for c in fold_metrics.columns if c.startswith(prefixes)]
    return fold_metrics.filter(pl.col("pred_col").is_not_null()).select([*keep, *columns])


def _mean(frame: pl.DataFrame, column: str) -> float:
    if frame.is_empty() or column not in frame.columns:
        return float("nan")
    values = frame.get_column(column).drop_nulls().drop_nans().to_numpy()
    return float(np.mean(values)) if values.size else float("nan")


def _std(frame: pl.DataFrame, column: str) -> float:
    """Fold standard deviation — `03` §3.3 reads improvements against this."""
    if frame.is_empty() or column not in frame.columns:
        return float("nan")
    values = frame.get_column(column).drop_nulls().drop_nans().to_numpy()
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")
