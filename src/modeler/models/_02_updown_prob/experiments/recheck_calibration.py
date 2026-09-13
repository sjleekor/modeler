"""recheck_calibration — `03` §3.3's conditional ECE recheck.

The rule: an adopted candidate must have ECE <= 0.03; if it is over, it is
re-judged **on its isotonic-calibrated column**, and dropped if it is still
over. The stage matrix has no run for this because the recheck only exists when
a candidate breaches the ceiling — ``selection.json`` says which ones did, in
``pending_calibration_recheck``.

Why this is a new fit rather than a re-read of the adopted run: turning
calibration on holds the tail of each fold's train slice out of the model fit
(`04` §4), so a calibrated run is not the adopted run with an extra column. Its
``p_raw`` is therefore *not* comparable to the adopted numbers, and only the
question the recheck asks — is the calibrated probability inside the ceiling —
is answered here. The record lands in ``results/recheck/<stage>/`` so the
stage's own records stay exactly the twelve (or eight, or sixteen) runs the
preregistration names.

    uv run --extra analysis python -m \
        modeler.models._02_updown_prob.experiments.recheck_calibration \
        --stage E2 --h 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from modeler.models._02_updown_prob import evaluate as ev
from modeler.models._02_updown_prob.experiments import registry as rg
from modeler.models._02_updown_prob.experiments import run_matrix as rm
from modeler.models._02_updown_prob.spec import lake_config

CALIBRATED_COLUMN = "p_cal"
RECHECK_VARIANT = "recal-isotonic"


def recheck_run(stage: str, horizon: int, *, smoke: bool = False) -> rg.Run:
    """The adopted config of ``stage`` at ``horizon``, with isotonic on top."""
    selection = rm.load_selection(stage, smoke=smoke)
    if selection is None:
        raise SystemExit(f"{stage} has no selection.json — run --stage {stage} first (`06` §4).")
    entry = selection.get("by_horizon", {}).get(str(horizon))
    if entry is None:
        raise SystemExit(f"{stage}'s selection has no h={horizon} entry")
    return rg.Run(
        stage=stage,
        variant=RECHECK_VARIANT,
        horizon=horizon,
        label=entry["label"],
        model=entry["model"],
        feature_set=entry["feature_set"],
        flow_variant=entry["flow_variant"],
        preprocess_profile=entry["preprocess_profile"],
        seed=entry["seed"],
        grid=entry["grid"],
        calibrate="isotonic",
    )


def calibrated_verdict(fold_metrics: pl.DataFrame) -> dict:
    """Mean ECE and log-loss of the calibrated column, and the `03` §3.3 call."""
    if "pred_col" not in fold_metrics.columns:
        raise ValueError("fold_metrics has no pred_col — nothing to re-judge")
    rows = fold_metrics.filter(pl.col("pred_col") == CALIBRATED_COLUMN)
    if rows.is_empty():
        raise ValueError(f"the run recorded no {CALIBRATED_COLUMN} rows — did calibration run?")
    ece = float(rows["ece"].mean())
    return {
        "pred_col": CALIBRATED_COLUMN,
        "n_folds": rows.height,
        "ece": ece,
        "ece_ceiling": ev.ECE_CEILING,
        "log_loss": float(rows["log_loss"].mean()),
        "ece_fold_max": float(rows["ece"].max()),
        "clears_ceiling": ece <= ev.ECE_CEILING,
        "rule": "`03` §3.3: over the ceiling after isotonic -> drop from the candidates",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=rg.STAGES)
    parser.add_argument("--h", type=int, required=True, choices=rg.HORIZONS)
    parser.add_argument("--force-run", action="store_true", help="re-fit even if recorded")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved run only")
    args = parser.parse_args(argv)

    run = recheck_run(args.stage, args.h)
    out_root = rm.RESULTS_ROOT / rm.RECHECK_DIR / args.stage
    print(
        f"{run.run_id}: {run.label}/{run.model} {run.feature_set} "
        f"grid={run.grid} profile={run.preprocess_profile} flow={run.flow_variant} "
        f"calibrate={run.calibrate} -> {out_root}"
    )
    if args.dry_run:
        return 0

    record = rm.execute_run(
        run,
        config=lake_config(),
        force_run=args.force_run,
        out_root=out_root,
    )
    out_dir = out_root / run.run_id
    verdict = calibrated_verdict(pl.read_parquet(out_dir / "fold_metrics.parquet"))
    verdict["run_id"] = run.run_id
    adopted = rm.load_selection(args.stage, smoke=False)["by_horizon"][str(args.h)]
    verdict["adopted_config"] = {"run_id": adopted["run_id"], "ece": adopted.get("ece")}
    # this run's own uncalibrated column, kept only to show what holding the cal
    # slice out of the fit costs. It is not the adopted run's number (`04` §4).
    verdict["this_run_p_raw_ece"] = record.summary.get("ece")
    Path(out_dir / "recheck.json").write_text(
        json.dumps(verdict, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(
        f"  {CALIBRATED_COLUMN}: ECE {verdict['ece']:.4f} "
        f"(fold max {verdict['ece_fold_max']:.4f}, ceiling {verdict['ece_ceiling']}) "
        f"log-loss {verdict['log_loss']:.5f}"
    )
    print(
        "  clears the ceiling — the adopted config stands"
        if verdict["clears_ceiling"]
        else "  still over the ceiling — `03` §3.3 drops it from the candidates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
