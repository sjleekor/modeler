"""run_matrix — run one stage of the preregistered matrix, then decide it.

Plan: ``docs/dev/20260907_model_experiment/06_implementation_plan.md`` §2.2, §4
(the commands), `03` §6 (what a run records), `05` §2 (the selection rules,
which live in ``selection.py``).

    uv run --extra analysis python -m modeler.models._02_updown_prob.experiments.run_matrix \\
        --stage E0 [--h 20] [--dry-run] [--smoke] [--summarize]

Four properties this runner is built to have.

**A stage cannot start without the previous stage's answer.** ``--stage E2``
resolves its label and model from ``results/E1/selection.json`` and refuses to
run if it is missing, so an inherited choice can never be made by default.

**Every run records what produced it.** ``run_spec.json`` carries the git sha, a
hash of the model code, the resolved run, the spec, and the contract hash of
every mart the panel was read from — the mitigation `06` §8 asks for against a
mart being rebuilt underneath the experiment.

**The panel is built once per config, not once per run.** The dataset key
excludes the seed and the model family, so E4a's extra seeds reuse the panel
their adopted config built.

**Predictions are written outside the repository.** ``results/`` keeps the small
frames that get committed; ``predictions_valid.parquet`` is hundreds of MB and
goes under ``data/``, with its path recorded in the summary (`03` §6).

**A stage is resumable, and can be run a horizon at a time.** A finished run is
reused when its recorded ``run_spec.json`` matches the run about to execute —
same resolved run, same spec, same ``code_hash`` — so ``--stage E2 --h 5`` today
and ``--stage E2`` tomorrow costs one stage, not two. A run interrupted midway
leaves ``run_spec.json`` without ``summary.json`` and is redone, because the
match requires both. ``--force-runs`` ignores the cache.

That matters beyond convenience: an eight-hour stage on a laptop that has to
move should not have to start over, and re-running a finished fit to satisfy the
runner would be a different (identical-seeded, but wasted) two hours.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import polars as pl

from modeler.etl.config import REPO_ROOT, LakeConfig
from modeler.etl.manifest import current_git_sha
from modeler.models._02_updown_prob import build_dataset as bd
from modeler.models._02_updown_prob import evaluate as ev
from modeler.models._02_updown_prob import train as tr
from modeler.models._02_updown_prob.experiments import registry as rg
from modeler.models._02_updown_prob.experiments import selection as sel
from modeler.models._02_updown_prob.spec import (
    ModelSpec,
    lake_config,
)

RESULTS_ROOT = REPO_ROOT / "docs" / "dev" / "20260907_model_experiment" / "results"
SMOKE_DIR = "smoke"
# `03` §3.3's conditional ECE recheck writes here rather than into the stage
# directory: it is not one of the stage's preregistered runs, and a stage's
# records are read by globbing that directory.
RECHECK_DIR = "recheck"

GRIDS: dict[str, tuple[dict, ...]] = {
    "HGB_REG_GRID": tr.HGB_REG_GRID,
    "HGB_CLF_GRID": tr.HGB_CLF_GRID,
    "HGB_CLF_GRID_E1": tr.HGB_CLF_GRID_E1,
    "LOGIT_GRID": tr.LOGIT_GRID,
}

# Variants whose input does not exist yet (`05` §2 E3-b, E4e). Both are optional
# in the registry, so the default plan never reaches them; asking for one
# explicitly should say why it cannot run rather than quietly running the
# adopted config again under a new name.
UNIMPLEMENTED_VARIANTS: dict[str, str] = {
    "fin_pit-strict": "needs the parallel stream's F-9.1 strict-vintage fin_pit rewrite (FS0')",
    "E4e-no_isna": "needs a preprocess switch that drops the *_isna design columns",
}

# Measured 2026-09-10 on this host, at the heaviest point of the 16-grid
# (max_iter 400, max_leaf_nodes 31): an FS2 fold with 4.57M train rows and 136
# design columns took 157s. Per million train rows that is ~34s at FS2's width;
# FS0 (80 columns) measured ~25s. Used only to print an estimate in --dry-run.
SECONDS_PER_FIT_PER_MILLION_TRAIN_ROWS = 34.0

# The smoke config: one grid point, two folds, and a period long enough for
# h120 to exist. Measured 2026-09-09: ending 2016-12-31 left an h120 fold with 8
# train sessions, so the calibration slice plus its 120-session embargo could
# not fit and both h120 runs failed. Five years is the shortest window where
# every horizon in the matrix produces folds.
SMOKE_PERIOD_END = "2019-12-31"
SMOKE_FOLDS = 2


# The files whose content decides a run's *numbers*: the panel, the labels, the
# preprocessing, the split, the model, the metrics. Deliberately not here:
# ``experiments/`` — the registry says which runs exist and the runner drives
# them, and both are already compared directly (the resolved run, spec and train
# config are stored per run). Keeping orchestration out of this hash is what
# lets a stage resume after an unrelated edit to the runner instead of re-fitting
# hours of identical models.
MODEL_CODE_FILES: tuple[str, ...] = (
    "src/modeler/models/_02_updown_prob/build_dataset.py",
    "src/modeler/models/_02_updown_prob/calibrate.py",
    "src/modeler/models/_02_updown_prob/evaluate.py",
    "src/modeler/models/_02_updown_prob/features.py",
    "src/modeler/models/_02_updown_prob/spec.py",
    "src/modeler/models/_02_updown_prob/train.py",
    "src/modeler/etl/features/regime.py",
    "src/modeler/etl/labels.py",
    "src/modeler/etl/metrics.py",
    "src/modeler/etl/preprocess.py",
    "src/modeler/etl/splits.py",
    "src/modeler/etl/universe.py",
)


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(path.relative_to(REPO_ROOT).as_posix().encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def model_code_hash() -> str:
    """Hash of :data:`MODEL_CODE_FILES` — what a cached result is keyed on."""
    return _hash_files([REPO_ROOT / name for name in MODEL_CODE_FILES])


def code_hash() -> str:
    """Hash of everything a run touches, orchestration included — provenance.

    Recorded, never compared: the git sha alone does not identify the code when
    the tree is dirty, and this says exactly what ran.
    """
    return _hash_files(
        [
            *sorted(
                (REPO_ROOT / "src" / "modeler" / "models" / "_02_updown_prob").rglob("*.py")
            ),
            *[REPO_ROOT / name for name in MODEL_CODE_FILES],
        ]
    )


def horizon_scan_running() -> bool:
    """`06` §4: the two must not share this host's 14 cores and 36GB."""
    try:
        out = subprocess.run(
            ["pgrep", "-f", "horizon_scan"], capture_output=True, text=True, timeout=5
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return out.returncode == 0 and bool(out.stdout.strip())


def stage_dir(stage: str, *, smoke: bool) -> Path:
    return RESULTS_ROOT / (SMOKE_DIR if smoke else stage)


def spec_for(run: rg.Run, *, smoke: bool) -> ModelSpec:
    """The :class:`ModelSpec` a run needs — D-2's boundary stays untouched."""
    kwargs: dict = {
        "feature_set": run.feature_set,
        "flow_variant": run.flow_variant,
        "preprocess_profile": run.preprocess_profile,
        "seed": run.seed,
    }
    if smoke:
        kwargs["period_end"] = SMOKE_PERIOD_END
        kwargs["n_folds"] = SMOKE_FOLDS
    return ModelSpec(**kwargs)


def train_config_for(run: rg.Run, *, smoke: bool) -> tr.TrainConfig:
    grid = GRIDS[run.grid]
    return tr.TrainConfig(
        model=run.model,
        target=run.label,
        horizon=run.horizon,
        grid=grid[:1] if smoke else grid,
        seed=run.seed,
        calibrate=run.calibrate,
        cal_target=run.cal_target,
        monotonic=run.monotonic,
    )


def ensure_dataset(
    spec: ModelSpec, config: LakeConfig, horizon: int, *, force: bool = False
) -> tuple[Path, dict]:
    """Build the panel for this config, or reuse the cached one.

    Reuse is keyed on the manifest: same feature set, flow variant, profile,
    horizon and period. Anything else (seed, model) does not change the panel.
    """
    dataset_dir = config.dataset_dir(spec.model_id) / bd.dataset_key(spec, horizon)
    manifest_path = dataset_dir / "dataset_manifest.json"
    std_dir = dataset_dir / bd.STD_DIR
    built = std_dir.is_dir() and any(std_dir.glob("*.parquet"))
    if not force and built and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        extra = manifest.get("extra", {})
        same = (
            extra.get("feature_set") == spec.feature_set
            and extra.get("flow_variant") == spec.flow_variant
            and extra.get("preprocess_profile") == spec.preprocess_profile
            and extra.get("horizon") == horizon
            and manifest.get("period") == {"start": spec.period_start, "end": spec.period_end}
            # a dataset written before the design matrix became a directory has
            # no layout recorded, and its single .parquet is not readable by
            # DatasetFolds — rebuild rather than guess.
            and extra.get("std_layout") is not None
        )
        if same:
            return dataset_dir, manifest
    result = bd.build_dataset(spec, config, horizon=horizon, write=True)
    return result.dataset_dir, json.loads(result.manifest_path.read_text(encoding="utf-8"))


def cached_record(
    run: rg.Run,
    spec: ModelSpec,
    train_cfg: tr.TrainConfig,
    *,
    smoke: bool,
    out_root: Path | None = None,
) -> sel.RunRecord | None:
    """The recorded result of this exact run, if it already finished.

    "This exact run" is checked, not assumed: the stored ``run_spec.json`` must
    carry the same resolved run, the same spec, the same train config, and the
    same :func:`model_code_hash`. ``summary.json`` must be there too — a run
    killed between the two files is not finished.

    Comparing the resolved spec and train config rather than hashing the runner
    is the point: those objects are *what the runner decided*, so a change that
    would alter the run is caught, while a change that would not — a flag, a
    message, an estimate — does not throw away hours of finished fits.
    """
    out_dir = (out_root or stage_dir(run.stage, smoke=smoke)) / run.run_id
    spec_path, summary_path = out_dir / "run_spec.json", out_dir / "summary.json"
    if not (spec_path.is_file() and summary_path.is_file()):
        return None
    try:
        stored = json.loads(spec_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # compare through JSON on both sides: a tuple field (``inherits``) comes back
    # from the file as a list, and an unnormalized == would never match.
    if (
        stored.get("run") != _as_json(asdict(run))
        or stored.get("spec") != _as_json(_spec_payload(spec))
        or stored.get("train_config") != _as_json(asdict(train_cfg))
        or stored.get("model_code_hash") != model_code_hash()
        or bool(stored.get("smoke")) != smoke
    ):
        return None
    fold_metrics = out_dir / "fold_metrics.parquet"
    return sel.RunRecord(
        run=run,
        summary=summary,
        fold_log_loss=(
            _fold_log_loss(pl.read_parquet(fold_metrics)) if fold_metrics.is_file() else {}
        ),
    )


def execute_run(
    run: rg.Run,
    *,
    config: LakeConfig,
    smoke: bool = False,
    force_build: bool = False,
    force_run: bool = False,
    out_root: Path | None = None,
) -> sel.RunRecord:
    """Build (or reuse) the panel, fit, evaluate, and write the run's record.

    ``out_root`` overrides where the record lands. Only `03` §3.3's conditional
    ECE recheck uses it (:mod:`recheck_calibration`) — a stage's own runs always
    write to their stage directory, which is where the selection reads them.
    """
    if run.variant in UNIMPLEMENTED_VARIANTS:
        raise NotImplementedError(f"{run.run_id}: {UNIMPLEMENTED_VARIANTS[run.variant]}")
    if not run.is_resolved():
        raise ValueError(f"{run.run_id} still carries unresolved fields; resolve it first")

    spec = spec_for(run, smoke=smoke)
    train_cfg = train_config_for(run, smoke=smoke)
    if not force_run:
        cached = cached_record(run, spec, train_cfg, smoke=smoke, out_root=out_root)
        if cached is not None:
            print(
                f"  {run.run_id}: cached "
                f"({cached.summary['probability_metric']}={cached.probability:.5f})"
            )
            return cached
    started = time.time()
    dataset_dir, manifest = ensure_dataset(spec, config, run.horizon, force=force_build)

    # one fold at a time, not the whole design matrix (see train.DatasetFolds)
    result = tr.walk_forward(tr.DatasetFolds(dataset_dir), train_cfg)
    report = ev.evaluate_run(result, k=spec.topk, cost_bps=spec.cost_bps_roundtrip, tau=spec.tau)

    out_dir = (out_root or stage_dir(run.stage, smoke=smoke)) / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    # the panel directory is shared with the official build of the same config,
    # so the file name has to say which run wrote it — and whether it counts.
    suffix = "__smoke" if smoke else ""
    predictions_path = dataset_dir / f"predictions_valid__{run.run_id}{suffix}.parquet"
    result.predictions.write_parquet(predictions_path)

    elapsed = time.time() - started
    summary = {
        **report.summary,
        "run_id": run.run_id,
        "stage": run.stage,
        "variant": run.variant,
        "elapsed_seconds": round(elapsed, 1),
        "predictions_path": str(predictions_path),
        "dataset_dir": str(dataset_dir),
        "panel_rows": manifest.get("row_count"),
        "smoke": smoke,
    }
    (out_dir / "run_spec.json").write_text(
        json.dumps(
            {
                "run": asdict(run),
                "spec": _spec_payload(spec),
                "train_config": asdict(train_cfg),
                "snapshot": {"snapshot_date": config.snapshot_date, "source": config.source},
                "git_sha": current_git_sha(),
                "code_hash": code_hash(),
                "model_code_hash": model_code_hash(),
                "mart_contracts": manifest.get("extra", {}).get("mart_contracts", {}),
                "dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
                "grid_name": run.grid,
                "grid_size": len(train_cfg.grid),
                "smoke": smoke,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )
    _write_frame(report.fold_metrics, out_dir / "fold_metrics.parquet")
    _write_frame(report.reliability, out_dir / "reliability.parquet")
    _write_frame(report.economics, out_dir / "economics.parquet")
    _write_frame(report.yearly, out_dir / "yearly.parquet")
    _write_frame(report.by_market, out_dir / "by_market.parquet")
    (out_dir / "summary.md").write_text(
        _render_summary(run, spec, summary, report), encoding="utf-8"
    )

    print(
        f"  {run.run_id}: {summary['probability_metric']}={summary['probability_value']:.5f} "
        f"{summary['economic_metric']}={summary['economic_value']:+.5f} "
        f"ece={summary['ece']:.4f} ({elapsed:.0f}s)"
    )
    return sel.RunRecord(
        run=run, summary=summary, fold_log_loss=_fold_log_loss(report.fold_metrics)
    )


def _as_json(value: dict) -> dict:
    """The value as it would look after a round trip through ``run_spec.json``."""
    return json.loads(json.dumps(value, default=str))


def _spec_payload(spec: ModelSpec) -> dict:
    return {
        "model_id": spec.model_id,
        "horizons": list(spec.horizons),
        "feature_set": spec.feature_set,
        "flow_variant": spec.flow_variant,
        "preprocess_profile": spec.preprocess_profile,
        "period_start": spec.period_start,
        "period_end": spec.period_end,
        "n_folds": spec.n_folds,
        "seed": spec.seed,
        "topk": spec.topk,
        "cost_bps_roundtrip": spec.cost_bps_roundtrip,
        "tau": spec.tau,
        "label": {
            "horizons": list(spec.label.horizons),
            "kind": spec.label.kind,
            "bench": spec.label.bench,
            "outputs": list(spec.label.outputs),
        },
    }


def _write_frame(frame: pl.DataFrame, path: Path) -> None:
    if frame.is_empty():
        return
    frame.write_parquet(path)


def _fold_log_loss(fold_metrics: pl.DataFrame) -> dict[int, float]:
    """Per-fold log-loss on the primary column — the fold-majority rule reads it."""
    if fold_metrics.is_empty() or "pred_col" not in fold_metrics.columns:
        return {}
    primary = "p_raw" if "p_raw" in fold_metrics["pred_col"].to_list() else "p_cal"
    rows = fold_metrics.filter(pl.col("pred_col") == primary)
    if rows.is_empty() or "log_loss" not in rows.columns:
        return {}
    return {
        int(fold): float(value)
        for fold, value in zip(rows["fold_id"].to_list(), rows["log_loss"].to_list(), strict=True)
        if value is not None
    }


def _render_summary(run: rg.Run, spec: ModelSpec, summary: dict, report: ev.RunEvaluation) -> str:
    lines = [
        f"# {run.run_id}",
        "",
        f"- stage: {run.stage} / variant: {run.variant}",
        f"- horizon: {run.horizon}, label: {run.label}, model: {run.model}, seed: {run.seed}",
        f"- feature set: {spec.feature_set} ({len(spec.feature_columns(run.horizon))} columns), "
        f"flow: {spec.flow_variant}, preprocess: {spec.preprocess_profile}",
        f"- period: {spec.period_start} .. {spec.period_end}, folds: {spec.n_folds}",
        f"- grid: {run.grid} ({summary['grid_size']} "
        f"point{'s' if summary['grid_size'] != 1 else ''}), best: {summary['best_params']}",
        f"- calibration: {run.calibrate}" + (f" -> {run.cal_target}" if run.cal_target else ""),
        "",
        "## Primary metrics (`03` §3.2)",
        "",
        "| metric | value | fold std |",
        "|---|---|---|",
        f"| {summary['probability_metric']} | {summary['probability_value']:.5f} | "
        f"{summary['probability_fold_std']:.5f} |",
        f"| {summary['economic_metric']} | {summary['economic_value']:+.5f} | "
        f"{summary['economic_fold_std']:.5f} |",
        "",
        "## Secondary",
        "",
        "| metric | value |",
        "|---|---|",
    ]
    for key in (
        "ece",
        "brier",
        "auc_daily_mean",
        "rank_ic_mean",
        "precision_at_k",
        "lift_at_k",
        "topk_turnover",
        "base_rate",
    ):
        value = summary.get(key)
        lines.append(
            f"| {key} | {value:.5f} |" if isinstance(value, float) else f"| {key} | {value} |"
        )
    lines += [
        "",
        f"ECE ceiling {ev.ECE_CEILING}: "
        + (
            "**over** — recheck after calibration (`03` §3.3)"
            if summary["ece_over_ceiling"]
            else "under"
        ),
        "",
        "## Folds",
        "",
        "```",
        str(report.fold_metrics),
        "```",
        "",
        f"- predictions: `{summary['predictions_path']}`",
        f"- panel rows: {summary['panel_rows']}",
        f"- elapsed: {summary['elapsed_seconds']}s",
    ]
    if summary.get("skipped_folds"):
        lines.append(f"- skipped folds: {summary['skipped_folds']}")
    return "\n".join(lines) + "\n"


def load_selection(stage: str, *, smoke: bool) -> dict | None:
    path = stage_dir(stage, smoke=smoke) / "selection.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_rechecks(stage: str) -> dict[int, dict]:
    """`03` §3.3's recheck verdicts for a stage, keyed by horizon.

    Read from ``results/recheck/<stage>/``, which is where
    :mod:`recheck_calibration` writes them — so re-running a stage's selection
    reproduces the same drops instead of forgetting them.
    """
    directory = RESULTS_ROOT / RECHECK_DIR / stage
    verdicts: dict[int, dict] = {}
    for path in sorted(directory.glob("*/recheck.json")):
        spec_path = path.parent / "run_spec.json"
        if not spec_path.is_file():
            continue
        run = rg.Run(**json.loads(spec_path.read_text(encoding="utf-8"))["run"])
        verdicts[run.horizon] = json.loads(path.read_text(encoding="utf-8"))
    return verdicts


def load_records(stage: str, *, smoke: bool) -> list[sel.RunRecord]:
    """Rebuild a finished stage's records from what it wrote."""
    directory = stage_dir(stage, smoke=smoke)
    records: list[sel.RunRecord] = []
    for run_dir in sorted(p for p in directory.glob("*") if p.is_dir()):
        spec_path, summary_path = run_dir / "run_spec.json", run_dir / "summary.json"
        if not (spec_path.is_file() and summary_path.is_file()):
            continue  # a failed run leaves only failed.json, and is not a record
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        run = rg.Run(**payload["run"])
        if run.stage != stage:
            continue
        fold_metrics_path = run_dir / "fold_metrics.parquet"
        fold_log_loss = (
            _fold_log_loss(pl.read_parquet(fold_metrics_path))
            if fold_metrics_path.is_file()
            else {}
        )
        records.append(
            sel.RunRecord(
                run=run,
                summary=json.loads(summary_path.read_text(encoding="utf-8")),
                fold_log_loss=fold_log_loss,
            )
        )
    return records


def resolve_stage(
    stage: str, *, horizon: int | None, smoke: bool, variant: str | None = None
) -> list[rg.Run]:
    """The stage's runs with every inherited field filled in.

    A horizon an earlier stage dropped (`03` §3.3's calibration condition) has
    no config to inherit, so its runs are skipped rather than resolved against
    a config that was rejected.
    """
    runs = rg.stage_runs(stage, horizon=horizon, variant=variant)
    needed = sorted({s for run in runs for s in run.inherits})
    selections: dict[str, dict] = {}
    dropped: dict[int, tuple[str, str]] = {}
    for prior in needed:
        payload = load_selection(prior, smoke=smoke)
        if payload is None:
            raise SystemExit(
                f"{stage} inherits from {prior}, whose selection.json is missing. "
                f"Run --stage {prior} first (`06` §4)."
            )
        selections[prior] = payload
        for key, entry in payload.get("dropped_horizons", {}).items():
            dropped.setdefault(int(key), (prior, entry.get("reason", "dropped")))
    for h, (prior, reason) in sorted(dropped.items()):
        if any(run.horizon == h for run in runs):
            print(f"  h{h}: skipped — {prior} dropped it ({reason})")
    return [rg.resolve(run, selections) for run in runs if run.horizon not in dropped]


def write_selection(stage: str, records: list[sel.RunRecord], *, smoke: bool) -> Path:
    """Apply the stage's rule and write ``selection.json`` (`05` §2)."""
    prior: dict[str, dict] = {}
    prior_records: dict[str, list[sel.RunRecord]] = {}
    for other in ("E0", "E1", "E2", "E4"):
        payload = load_selection(other, smoke=smoke)
        if payload is not None:
            prior[other] = payload
            prior_records[other] = load_records(other, smoke=smoke)
    payload = sel.build_selection(
        stage,
        records,
        prior=prior,
        prior_records=prior_records,
        rechecks={} if smoke else load_rechecks(stage),
    )
    path = stage_dir(stage, smoke=smoke) / "selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"selection -> {path}")
    for note in payload.get("notes", []):
        print(f"  {note}")
    for pending in payload.get("pending_calibration_recheck", []):
        print(f"  ECE recheck pending: {pending}")
    for horizon, entry in payload.get("dropped_horizons", {}).items():
        print(f"  h{horizon} dropped: {entry['reason']}")
    return path


def summarize_stage(stage: str, *, smoke: bool) -> int:
    """Render a stage's runs and selection as one markdown table (`06` §4)."""
    records = load_records(stage, smoke=smoke)
    if not records:
        print(f"no runs recorded for {stage}")
        return 1
    lines = [
        f"# {stage} — runs",
        "",
        "| run | h | label | model | FS | seed | prob | econ | ece |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for record in sorted(records, key=lambda r: (r.horizon, r.run.variant, r.run.seed)):
        run = record.run
        lines.append(
            f"| {run.run_id} | {run.horizon} | {run.label} | {run.model} | {run.feature_set} | "
            f"{run.seed} | {record.probability:.5f} | {record.economic:+.5f} | {record.ece:.4f} |"
        )
    payload = load_selection(stage, smoke=smoke)
    if payload:
        lines += ["", "## Selection", "", f"rule: {payload['rule']}", ""]
        lines += [f"- {note}" for note in payload.get("notes", [])]
    path = stage_dir(stage, smoke=smoke) / "runs.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten: {path}")
    return 0


def _dry_run(stage: str, runs: list[rg.Run], *, smoke: bool) -> int:
    print(f"{stage}: {len(runs)} runs ({'smoke' if smoke else 'official'})")
    total_fits = 0
    for run in runs:
        spec = spec_for(run, smoke=smoke)
        train_cfg = train_config_for(run, smoke=smoke)
        # the sweep fits every grid point on every fold, then the winner is refit
        # once per fold — not twice over the whole grid
        fits = len(train_cfg.grid) * spec.n_folds + spec.n_folds
        total_fits += fits
        print(
            f"  {run.run_id}: {run.label}/{run.model} {run.feature_set} "
            f"grid={run.grid}({len(train_cfg.grid)}) profile={spec.preprocess_profile} "
            f"flow={spec.flow_variant} seed={run.seed} -> {fits} fits"
        )
        if run.variant in UNIMPLEMENTED_VARIANTS:
            print(f"    NOT RUNNABLE: {UNIMPLEMENTED_VARIANTS[run.variant]}")
    # The average fold is smaller than the last one, and the cheaper grid points
    # are faster than the point the constant was measured at, so this is an upper
    # bound rather than a forecast.
    hours = total_fits * SECONDS_PER_FIT_PER_MILLION_TRAIN_ROWS * 3.0 / 3600
    print(
        f"total {total_fits} fits. At the measured "
        f"{SECONDS_PER_FIT_PER_MILLION_TRAIN_ROWS:.0f}s per fit per million train rows, "
        f"a full-period fold (~3M train rows) is about "
        f"{SECONDS_PER_FIT_PER_MILLION_TRAIN_ROWS * 3 / 60:.1f} minutes per fit "
        f"-> up to ~{hours:.0f}h for this stage. Check `05` §1's budget before starting."
    )
    return 0


def run_stage(
    stage: str,
    *,
    horizon: int | None = None,
    variant: str | None = None,
    smoke: bool = False,
    dry_run: bool = False,
    force_build: bool = False,
    force_runs: bool = False,
    allow_concurrent: bool = False,
) -> int:
    runs = resolve_stage(stage, horizon=horizon, variant=variant, smoke=smoke)
    if dry_run:
        return _dry_run(stage, runs, smoke=smoke)
    if not allow_concurrent and horizon_scan_running():
        raise SystemExit(
            "a horizon_scan process is running; `06` §4 says not to share this host. "
            "Pass --allow-concurrent to override."
        )

    config = lake_config()
    print(f"{stage}: {len(runs)} runs on snapshot {config.snapshot_date}/{config.source}")
    records: list[sel.RunRecord] = []
    failures: list[str] = []
    for run in runs:
        try:
            records.append(
                execute_run(
                    run,
                    config=config,
                    smoke=smoke,
                    force_build=force_build,
                    force_run=force_runs,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one run's failure is not the stage's
            # A stage is a set of independent runs, so losing one must not
            # discard the rest — but it must also not be rounded away: the
            # selection is refused below until every run has a number.
            failures.append(run.run_id)
            _record_failure(run, exc, smoke=smoke)
            print(f"  {run.run_id}: FAILED — {type(exc).__name__}: {exc}")

    if failures:
        print(f"{len(failures)} of {len(runs)} runs failed: {', '.join(failures)}")
        print("selection not written — a stage is decided on a complete set of runs")
        return 1
    if horizon is not None or variant is not None:
        remaining = [
            r.run_id
            for r in resolve_stage(stage, horizon=None, smoke=smoke)
            if cached_record(
                r, spec_for(r, smoke=smoke), train_config_for(r, smoke=smoke), smoke=smoke
            )
            is None
        ]
        narrowed = ", ".join(
            part
            for part in (
                f"--h {horizon}" if horizon is not None else "",
                f"--variant {variant}" if variant is not None else "",
            )
            if part
        )
        print(f"partial stage ({narrowed}): selection not written")
        if remaining:
            print(f"  {len(remaining)} runs left in {stage}: {', '.join(remaining)}")
        else:
            print(f"  every {stage} run is recorded — run without --h to write the selection")
        return 0
    write_selection(stage, records, smoke=smoke)
    return 0


def _record_failure(run: rg.Run, exc: BaseException, *, smoke: bool) -> None:
    """Leave the failure on disk beside the runs that worked."""
    out_dir = stage_dir(run.stage, smoke=smoke) / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "failed.json").write_text(
        json.dumps(
            {
                "run": asdict(run),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "git_sha": current_git_sha(),
                "code_hash": code_hash(),
                "model_code_hash": model_code_hash(),
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=rg.STAGES)
    parser.add_argument("--h", type=int, default=None, choices=rg.HORIZONS)
    parser.add_argument(
        "--variant",
        default=None,
        help="run only this column of the stage (e.g. FS2, MA-rank, E4a-seed); "
        "combine with --h to run exactly one run",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan only")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"one grid point, {SMOKE_FOLDS} folds, period to {SMOKE_PERIOD_END}; "
        f"writes to results/{SMOKE_DIR}/ and is not an official result",
    )
    parser.add_argument("--summarize", action="store_true", help="render a finished stage")
    parser.add_argument("--force-build", action="store_true", help="rebuild the panel")
    parser.add_argument(
        "--force-runs", action="store_true", help="re-run runs that already have a result"
    )
    parser.add_argument(
        "--allow-concurrent", action="store_true", help="run even if a horizon scan is running"
    )
    parser.add_argument(
        "--clean-smoke", action="store_true", help=f"delete results/{SMOKE_DIR}/ and exit"
    )
    args = parser.parse_args(argv)

    if args.clean_smoke:
        directory = RESULTS_ROOT / SMOKE_DIR
        if directory.exists():
            shutil.rmtree(directory)
            print(f"removed {directory}")
        return 0
    if args.summarize:
        return summarize_stage(args.stage, smoke=args.smoke)
    return run_stage(
        args.stage,
        horizon=args.h,
        variant=args.variant,
        smoke=args.smoke,
        dry_run=args.dry_run,
        force_build=args.force_build,
        force_runs=args.force_runs,
        allow_concurrent=args.allow_concurrent,
    )


if __name__ == "__main__":
    raise SystemExit(main())
