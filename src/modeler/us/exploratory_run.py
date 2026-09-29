"""US exploratory completion run: fixed models, OOF backtest, and saved scorers.

This deliberately does not use single-feature grades as an admission gate. It is
an exploratory replay of the development period, never a new holdout result.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

from modeler.etl.config import DataRoot
from modeler.etl.metrics import drawdown_stats, newey_west_tstat
from modeler.us import benchmark, cost, metrics
from modeler.us.lake import UsLake
from modeler.us.m4_splits import build_wf_folds
from modeler.us.m4_transform import rank_transform, to_design_arrays
from modeler.us.scan import DEV_END, DEV_START, FEATURE_COLUMNS, monthly_rank_ic
from modeler.us.scan_long2 import NEW_FEATURE_REGISTRY

FEATURES = tuple(c for c in FEATURE_COLUMNS if c != "turnover_rank") + tuple(
    s.feature for s in NEW_FEATURE_REGISTRY
)
DATASETS = ("us_features_v2", "us_features_flow_v1", "us_labels_v2")
MODEL_PARAMS = {
    "ridge": {"alpha": 100.0},
    "lightgbm": {
        "objective": "regression",
        "num_leaves": 15,
        "learning_rate": 0.03,
        "n_estimators": 200,
        "min_child_samples": 500,
        "reg_lambda": 1.0,
        "random_state": 0,
        "n_jobs": 2,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    },
}


@dataclass(frozen=True)
class PinnedLake(UsLake):
    snapshots: dict[str, str]

    def latest_snapshot(self, table: str) -> date:
        return date.fromisoformat(self.snapshots[table])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False) + "\n"
    )


def prepare_features(frame: pl.DataFrame, features=FEATURES) -> pl.DataFrame:
    """Rank each contemporaneous eligible cross-section without using labels."""
    frame = frame.filter(pl.col("price_ge_5")).sort("date", "symbol")
    frame = frame.with_columns(
        [
            pl.when(pl.col(c).cast(pl.Float64).is_finite())
            .then(pl.col(c).cast(pl.Float64))
            .otherwise(None)
            .alias(c)
            for c in features
        ]
    )
    frame = frame.with_columns([pl.col(c).is_null().alias(f"{c}_isna") for c in features])
    return rank_transform(frame, features)


def load_inputs(root: DataRoot) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    manifests = {
        name: json.loads((root.datasets / name / "manifest.json").read_text()) for name in DATASETS
    }
    base = pl.read_parquet(root.datasets / DATASETS[0] / "part.parquet")
    flow = pl.read_parquet(root.datasets / DATASETS[1] / "part.parquet")
    base_cols = [c for c in FEATURES if c in base.columns]
    flow_cols = [c for c in FEATURES if c not in base.columns]
    raw = base.select("date", "symbol", "price_ge_5", "close", "adv_20d", *base_cols).join(
        flow.select("date", "symbol", *flow_cols),
        on=["date", "symbol"],
        how="left",
        validate="1:1",
    )
    if raw.height != flow.height:
        raise ValueError("Feature dataset keys differ")
    # Push the temporal boundary into the label scan; later labels are never collected.
    labels = (
        pl.scan_parquet(root.datasets / DATASETS[2] / "part.parquet")
        .filter(pl.col("date").is_between(DEV_START, DEV_END))
        .select("date", "symbol", "terminal_date", "L0", "L2", "y_rank")
        .collect()
    )
    # Drop an entire rebalance if any label still ends beyond the development wall.
    eligible_dates = (
        labels.group_by("date")
        .agg(pl.col("terminal_date").max())
        .filter(pl.col("terminal_date") <= DEV_END)["date"]
    )
    labels = labels.filter(pl.col("date").is_in(eligible_dates.implode()))
    dev_features = prepare_features(raw.filter(pl.col("date").is_between(DEV_START, DEV_END)))
    core = dev_features.join(labels, on=["date", "symbol"], how="inner", validate="1:1")
    if core.select(pl.any_horizontal(pl.col("y_rank", "L0", "L2").is_null()).any()).item():
        raise ValueError("Missing training labels")
    if core["terminal_date"].max() > DEV_END:
        raise ValueError("Training label crosses development wall")
    latest = prepare_features(raw.filter(pl.col("date") == raw["date"].max()))
    return core.sort("date", "symbol"), latest, manifests


def training_rows(core: pl.DataFrame, train_dates, valid_start: date) -> pl.DataFrame:
    out = core.filter(pl.col("date").is_in(list(train_dates))).filter(
        pl.col("terminal_date") < valid_start
    )
    if not out.height or out["date"].max() >= valid_start:
        raise ValueError("Invalid temporal split")
    return out


def make_model(name: str):
    return (
        Ridge(**MODEL_PARAMS[name]) if name == "ridge" else lgb.LGBMRegressor(**MODEL_PARAMS[name])
    )


def score_model(model, frame: pl.DataFrame, features=FEATURES) -> pl.DataFrame:
    x, _ = to_design_arrays(frame, features)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite design matrix")
    preds = model.predict(x)
    if not np.isfinite(preds).all():
        raise ValueError("Non-finite predictions")
    return frame.with_columns(pl.Series("pred", preds))


def rankings(scored: pl.DataFrame) -> pl.DataFrame:
    return (
        scored.sort(["date", "pred", "symbol"], descending=[False, True, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("date").alias("rank"))
        .select("date", "rank", "symbol", "pred", "close", "adv_20d")
    )


def evaluate(oof: pl.DataFrame, spy: pl.DataFrame) -> tuple[dict, pl.DataFrame, list[dict]]:
    # rv_20 is frozen trailing daily return std, with the ticker-reuse gap mask.
    invalid_sigma = ~pl.col("rv_20").is_finite() | (pl.col("rv_20") <= 0) | (pl.col("rv_20") > 0.2)
    frame = oof.sort("date", "symbol").with_columns(
        pl.when(invalid_sigma.fill_null(True))
        .then(0.20)
        .otherwise(pl.col("rv_20"))
        .alias("sigma_daily")
    )
    ew = frame.group_by("date").agg(pl.col("L0").mean().alias("ew_return"))
    ic = monthly_rank_ic(frame, x_col="pred", y_col="L2", group_col="date")
    if not np.isfinite(ic["ic"].to_numpy()).all():
        raise ValueError("Non-finite monthly IC")
    sensitivity = []
    tracks = {}
    for q in cost.Q_GRID:
        track = (
            metrics.portfolio_track(frame, q_dollar=q)
            .join(ew, on="date", how="left", validate="1:1")
            .join(spy, on="date", how="left", validate="1:1")
            .sort("date")
        )
        for c in ["cost_drag", "net_return", "ew_return"]:
            if not track[c].is_finite().all() or track[c].null_count():
                raise ValueError(f"Missing/non-finite {c}")
        sensitivity.append(
            {
                "q_dollar": q,
                "net_mean_monthly": track["net_return"].mean(),
                "excess_ew_mean_monthly": (track["net_return"] - track["ew_return"]).mean(),
                "excess_spy_mean_monthly": (track["net_return"] - track["spy_h21_return"]).mean(),
                "cost_mean_monthly": track["cost_drag"].mean(),
            }
        )
        tracks[q] = track
    track = tracks[cost.DEFAULT_Q_DOLLAR].join(ic.select("date", "ic"), on="date")
    values = track["net_return"].to_numpy()
    summary = {
        **next(row for row in sensitivity if row["q_dollar"] == cost.DEFAULT_Q_DOLLAR),
        "n_months": track.height,
        "spy_comparison_months": track["spy_h21_return"].count(),
        "spy_missing_dates": [
            str(d) for d in track.filter(pl.col("spy_h21_return").is_null())["date"]
        ],
        "start": str(track["date"].min()),
        "end": str(track["date"].max()),
        "rank_ic": float(ic["ic"].mean()),
        "rank_ic_t_hac": float(newey_west_tstat(ic["ic"].to_numpy(), np.arange(ic.height), 3)),
        "gross_mean_monthly": track["gross_return"].mean(),
        "turnover_mean": track["turnover"].mean(),
        "max_drawdown": drawdown_stats(values.tolist()).max_drawdown,
        "compounded_h21_return": float(np.prod(1 + values) - 1),
        "sigma_fallback_rows": frame.filter(invalid_sigma.fill_null(True)).height,
        "oof_rows": frame.height,
    }
    return summary, track, sensitivity


def report(out: Path, results: dict, config: dict) -> None:
    lines = [
        "# US exploratory model results",
        "",
        "Completed training, walk-forward backtest, saved models, and latest-date rankings.",
        "Exploratory replay of previously inspected development data; not a fresh holdout result.",
        "",
        "| Model | Monthly net | Net - EW | Net - SPY | Rank IC | Max drawdown |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in results.items():
        lines.append(
            f"| {name} | {row['net_mean_monthly']:.2%} | "
            f"{row['excess_ew_mean_monthly']:.2%} | {row['excess_spy_mean_monthly']:.2%} | "
            f"{row['rank_ic']:.4f} | {row['max_drawdown']:.2%} |"
        )
    lines += [
        "",
        f"- Latest ranking date: {config['score_date']} (not today's quote).",
        "- Primary cost: existing per-name Q=$10M, k=0.1; see sensitivity.csv for other Q values.",
        "- 53 features; turnover_rank excluded for its known MIDAS publication-lag issue.",
        "- Existing financial freshness, identifier, and corporate-action limitations remain.",
        "- The 21-trading-day labels approximate a monthly portfolio; "
        "compounded results are not an execution simulation.",
        "- All training/evaluation labels end by 2025-06-30. "
        "Later feature rows are scored without their labels.",
        "- Both fixed models are retained; no performance-based retraining or model promotion.",
        "- SPY comparisons use only dates with observed returns; "
        "missing months stay blank in monthly.csv.",
        "",
        "Artifacts: model.joblib, oof.parquet, monthly.csv, sensitivity.csv, "
        "latest_rankings.csv, top100.csv.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_tag):
        parser.error("run-tag must contain only letters, digits, underscore, or hyphen")
    root = DataRoot.resolve("us")
    out = root.output / "exploratory_models" / args.run_tag
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    print("Loading frozen inputs", flush=True)
    core, latest, manifests = load_inputs(root)
    folds = build_wf_folds(sorted(core["date"].unique().to_list()), dev_end=DEV_END)
    source_dir = Path(__file__).parent
    config = {
        "status": "running",
        "kind": "exploratory_replay",
        "started_at": datetime.now(UTC),
        "features": FEATURES,
        "excluded_features": {"turnover_rank": "MIDAS publication lag"},
        "params": MODEL_PARAMS,
        "label_end_exclusive": "2025-07-01",
        "target": "y_rank",
        "universe": "price_ge_5",
        "rank_tie_break": "symbol ascending",
        "top_k": 100,
        "train_rows": core.height,
        "train_start": core["date"].min(),
        "train_end": core["date"].max(),
        "max_training_terminal": core["terminal_date"].max(),
        "score_date": latest["date"].max(),
        "score_rows": latest.height,
        "input_sha256": {n: sha256(root.datasets / n / "part.parquet") for n in DATASETS},
        "input_manifests": manifests,
        "modeler_head": subprocess.check_output(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {p.name: sha256(p) for p in source_dir.glob("*.py")},
        "library_versions": {
            n: importlib.metadata.version(n)
            for n in ["polars", "numpy", "scikit-learn", "lightgbm", "joblib"]
        },
        "folds": [],
    }
    for f in folds:
        train = training_rows(core, f.train_dates, f.valid_start)
        config["folds"].append(
            {
                "fold_id": f.fold_id,
                "train_start": train["date"].min(),
                "train_end": train["date"].max(),
                "max_train_terminal": train["terminal_date"].max(),
                "valid_start": f.valid_start,
                "valid_end": f.valid_end,
                "train_rows": train.height,
            }
        )
    write_json(out / "config.json", config)
    lake = PinnedLake(root, manifests["us_labels_v2"]["input_table_snapshots"])
    valid_dates = sorted({d for fold in folds for d in fold.valid_dates})
    print("Building pinned SPY benchmark", flush=True)
    spy = benchmark.spy_monthly_return(lake, valid_dates, base_date=DEV_END)
    results = {}
    for name in MODEL_PARAMS:
        print(f"Training {name}: 5 folds, {len(FEATURES)} features", flush=True)
        frames = []
        for f in folds:
            train = training_rows(core, f.train_dates, f.valid_start)
            valid = core.filter(pl.col("date").is_in(list(f.valid_dates)))
            x_train, _ = to_design_arrays(train, FEATURES)
            if not np.isfinite(x_train).all():
                raise ValueError("Non-finite training matrix")
            model = make_model(name).fit(x_train, train["y_rank"].to_numpy())
            keep = [
                "date",
                "symbol",
                "close",
                "adv_20d",
                "rv_20",
                "L0",
                "L2",
                "y_rank",
                "pred",
            ]
            frames.append(
                score_model(model, valid)
                .select(keep)
                .with_columns(pl.lit(f.fold_id).alias("fold_id"))
            )
            print(f"  {name} fold {f.fold_id} complete", flush=True)
        oof = pl.concat(frames).sort("date", "symbol")
        if oof.select(pl.struct("date", "symbol").n_unique()).item() != oof.height:
            raise ValueError("Duplicate OOF keys")
        summary, track, sensitivity = evaluate(oof, spy)
        model_out = out / name
        model_out.mkdir()
        oof.write_parquet(model_out / "oof.parquet")
        track.write_csv(model_out / "monthly.csv")
        pl.DataFrame(sensitivity).write_csv(model_out / "sensitivity.csv")
        write_json(model_out / "metrics.json", summary)
        x_train, _ = to_design_arrays(core, FEATURES)
        final = make_model(name).fit(x_train, core["y_rank"].to_numpy())
        bundle = {
            "model": final,
            "features": FEATURES,
            "train_end": config["train_end"],
            "max_training_terminal": config["max_training_terminal"],
            "kind": "exploratory",
        }
        joblib.dump(bundle, model_out / "model.joblib")
        scores = score_model(final, latest)
        restored = joblib.load(model_out / "model.joblib")
        check = score_model(restored["model"], latest, restored["features"])
        np.testing.assert_allclose(
            scores["pred"].to_numpy(), check["pred"].to_numpy(), rtol=0, atol=1e-12
        )
        ranks = rankings(scores)
        ranks.write_csv(model_out / "latest_rankings.csv")
        ranks.head(100).write_csv(model_out / "top100.csv")
        results[name] = summary
        print(json.dumps({"model": name, **summary}), flush=True)
    config.update(
        status="complete",
        elapsed_seconds=time.monotonic() - t0,
        completed_at=datetime.now(UTC),
        saved_model_reload_verified=True,
    )
    write_json(out / "config.json", config)
    write_json(out / "summary.json", results)
    report(out, results, config)
    print(f"Complete: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
