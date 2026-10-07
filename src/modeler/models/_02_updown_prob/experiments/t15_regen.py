"""T1-5 DSR·PBO 계측용 단축 재생성 — 기록된 ``best_params`` 로 예측만 다시 만든다.

이번에는 ``--check`` 모드(fold 하나 복원 후 ``RES`` 의 ``fold_metrics`` 와 대조)만 있다.
아무 파일도 쓰지 않는다. 전체 재생성(``--all --out``)은 나중에 같은 함수 위에 붙인다.

설계: ``my/milestones/kr/modeling/assessment/20261005_dsr_pbo.md`` §1.4.
``MODEL_CODE_FILES`` 밖이라 ``run_matrix.model_code_hash`` 에 들어가지 않는다.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from modeler.models._02_updown_prob import build_dataset as bd
from modeler.models._02_updown_prob import evaluate as ev
from modeler.models._02_updown_prob import train as tr
from modeler.models._02_updown_prob.experiments.run_matrix import (
    RESULTS_ROOT,
    model_code_hash,
)
from modeler.models._02_updown_prob.spec import ModelSpec, lake_config

# 준비 문서 §1.4 의 기준. float 열은 이 두 묶음 중 하나에 들어가야 한다.
BITWISE_COLUMNS: tuple[str, ...] = (
    "log_loss", "brier", "ece", "auc_pooled", "auc_daily_mean",
    "rank_ic_mean", "rank_ic_std", "icir", "rank_ic_tstat",
    "topk_turnover", "topk_mean_names_held", "topk_mean_names_scored",
    "decile_turnover", "tau_mean_names_held", "tau_tau",
    "topk_cost_bps_roundtrip", "tau_cost_bps_roundtrip", "decile_cost_bps_roundtrip",
)  # fmt: skip
CLOSE_COLUMNS: tuple[str, ...] = (
    "topk_grid_topk_mean_return", "topk_cost_adjusted_return",
    "tau_grid_mean_return", "tau_cost_adjusted_return", "tau_turnover",
    "decile_grid_top_decile_spread", "decile_cost_adjusted_spread",
    "top_decile_spread", "top_minus_bottom", "hit_ratio_top",
    "precision_at_k", "lift_at_k", "base_rate",
)  # fmt: skip
REL_TOL = 1e-12
# 식별 열. 비교하지 않고 행을 고르는 데 쓴다.
KEY_COLUMNS: tuple[str, ...] = ("fold_id", "pred_col")


@dataclass
class RecordedRun:
    stage: str
    run_id: str
    run_dir: Path
    run_spec: dict
    summary: dict

    @property
    def best_params(self) -> dict:
        return dict(self.summary["best_params"])

    @property
    def primary_pred_col(self) -> str:
        return str(self.summary["primary_pred_col"])


def find_run_dir(run_id: str, results_root: Path = RESULTS_ROOT) -> Path:
    hits = sorted(p for p in results_root.glob(f"*/{run_id}") if (p / "run_spec.json").is_file())
    if len(hits) != 1:
        raise FileNotFoundError(f"{run_id}: RES 아래 run_spec.json 이 있는 폴더 {len(hits)}개 {hits}")
    return hits[0]


def load_recorded(run_id: str, results_root: Path = RESULTS_ROOT) -> RecordedRun:
    run_dir = find_run_dir(run_id, results_root)
    run_spec = json.loads((run_dir / "run_spec.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    return RecordedRun(run_dir.parent.name, run_id, run_dir, run_spec, summary)


def restore_train_config(rec: RecordedRun) -> tr.TrainConfig:
    """기록된 ``TrainConfig`` 에서 그리드만 ``(best_params,)`` 한 점으로 바꾼다."""
    cfg = rec.run_spec["train_config"]
    return tr.TrainConfig(
        model=cfg["model"],
        target=cfg["target"],
        horizon=int(cfg["horizon"]),
        grid=(rec.best_params,),
        seed=int(cfg["seed"]),
        calibrate=cfg["calibrate"],
        cal_target=cfg.get("cal_target"),
        cal_frac=float(cfg["cal_frac"]),
        monotonic=bool(cfg["monotonic"]),
        date_col=cfg["date_col"],
        id_cols=tuple(cfg["id_cols"]),
    )


def restore_spec(rec: RecordedRun) -> ModelSpec:
    """``run_matrix.spec_for`` 와 같은 네 인자."""
    run = rec.run_spec["run"]
    return ModelSpec(
        feature_set=run["feature_set"],
        flow_variant=run["flow_variant"],
        preprocess_profile=run["preprocess_profile"],
        seed=int(run["seed"]),
    )


def dataset_path(rec: RecordedRun, lake=None) -> Path:
    """기록된 ``summary.dataset_dir`` 은 쓰지 않는다 — 지워진 구 경로다."""
    lake = lake or lake_config()
    spec = restore_spec(rec)
    horizon = int(rec.run_spec["train_config"]["horizon"])
    return lake.dataset_dir(spec.model_id) / bd.dataset_key(spec, horizon)


def regenerate_fold(rec: RecordedRun, fold_id: int, dataset_dir: Path) -> tuple[pl.DataFrame, dict]:
    """fold 하나를 ``_fit_fold`` 로 다시 적합하고 ``fold_rows`` 의 ``primary_pred_col`` 행을 돌려준다."""
    config = restore_train_config(rec)
    source = tr.DatasetFolds(dataset_dir)
    design = source.design_columns(config.target_column)
    train, valid = source.slices(fold_id)
    n_train, n_valid = train.height, valid.height
    predictions, calibrated, empty = tr._fit_fold(train, valid, design, config, rec.best_params)
    fit = tr.FoldFit(
        fold_id=fold_id,
        params=rec.best_params,
        n_train=n_train,
        n_valid=n_valid,
        metric=tr._fold_metric(predictions, config),
        calibrated=calibrated,
        empty_design_columns=list(empty),
        predictions=predictions,
    )
    result = tr.TrainResult(
        config=config,
        best_params=rec.best_params,
        best_metric=fit.metric,
        grid_metrics=[],
        folds=[fit],
        design_columns=design,
    )
    rows = ev.fold_rows(
        result,
        k=int(rec.summary["k"]),
        cost_bps=float(rec.summary["cost_bps_roundtrip"]),
        tau=float(rec.summary["tau"]),
    )
    frame = pl.DataFrame(rows, infer_schema_length=None)
    info = {"n_train": n_train, "n_valid": n_valid, "empty_design_columns": sorted(empty)}
    return frame, info


@dataclass
class Comparison:
    table: pl.DataFrame
    unclassified: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.unclassified and not self.failures


def _is_float(dtype: pl.DataType) -> bool:
    return dtype.is_float()


def _same_bits(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    return a == b


def _rel_diff(a, b) -> float:
    if a is None or b is None:
        return 0.0 if (a is None and b is None) else float("inf")
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return float("inf")
    if a == b:
        return 0.0
    return abs(a - b) / max(abs(a), abs(b))


def compare_rows(recorded: dict, fresh: dict, schema: dict[str, pl.DataType]) -> Comparison:
    """한 행씩 대조한다. 분류: 정수·문자·불리언(같음) / (가) 비트 / (나) 상대차 / 분류 밖 float.

    ``schema`` 는 기록된 ``fold_metrics`` 의 열 종류다. 재현 쪽에만 있는 열이나 기록 쪽에만
    있는 열은 ``failures`` 에 이름을 적는다.
    """
    out: list[dict] = []
    unclassified: list[str] = []
    failures: list[str] = []
    for col in sorted(set(recorded) | set(fresh), key=lambda c: list(recorded).index(c) if c in recorded else 10**6):
        if col not in recorded or col not in fresh:
            failures.append(f"{col}: 한쪽에만 있음")
            out.append({"column": col, "kind": "missing", "recorded": recorded.get(col), "fresh": fresh.get(col), "diff": None, "ok": False})
            continue
        a, b = recorded[col], fresh[col]
        dtype = schema.get(col)
        if col in BITWISE_COLUMNS:
            kind, ok, diff = "bitwise", _same_bits(a, b), (None if a is None or b is None else (b - a))
        elif col in CLOSE_COLUMNS:
            rel = _rel_diff(a, b)
            kind, ok, diff = "rel<=1e-12", rel <= REL_TOL, rel
        elif dtype is not None and _is_float(dtype):
            kind, ok, diff = "UNCLASSIFIED", False, None
            unclassified.append(col)
        else:
            kind, ok, diff = "exact", a == b, None
        if not ok and kind != "UNCLASSIFIED":
            failures.append(col)
        out.append({"column": col, "kind": kind, "recorded": a, "fresh": b, "diff": diff, "ok": ok})
    table = pl.DataFrame(
        out,
        schema={"column": pl.String, "kind": pl.String, "recorded": pl.Object, "fresh": pl.Object, "diff": pl.Object, "ok": pl.Boolean},
    )
    return Comparison(table, unclassified, failures)


def recorded_row(rec: RecordedRun, fold_id: int) -> tuple[dict, dict]:
    metrics = pl.read_parquet(rec.run_dir / "fold_metrics.parquet")
    row = metrics.filter((pl.col("fold_id") == fold_id) & (pl.col("pred_col") == rec.primary_pred_col))
    if row.height != 1:
        raise ValueError(f"fold_metrics 에 fold {fold_id} · {rec.primary_pred_col} 행이 {row.height}개")
    return row.row(0, named=True), dict(metrics.schema)


def print_comparison(cmp: Comparison) -> None:
    print(f"{'column':36s} {'kind':12s} {'recorded':>24s} {'fresh':>24s} {'diff':>12s}  ok")
    for r in cmp.table.iter_rows(named=True):
        def fmt(v):
            return f"{v:.17g}" if isinstance(v, float) else str(v)
        diff = "" if r["diff"] is None else f"{r['diff']:.3e}" if isinstance(r["diff"], float) else str(r["diff"])
        print(f"{r['column']:36s} {r['kind']:12s} {fmt(r['recorded']):>24s} {fmt(r['fresh']):>24s} {diff:>12s}  {'ok' if r['ok'] else 'XX'}")


def check(run_id: str, fold_id: int) -> int:
    started = time.time()
    rec = load_recorded(run_id)
    config = restore_train_config(rec)
    dataset_dir = dataset_path(rec)
    print(f"run {run_id} (stage {rec.stage}) · fold {fold_id} · pred_col {rec.primary_pred_col}")
    print(f"best_params {rec.best_params}")
    print(f"데이터셋 {dataset_dir}")
    print(f"model_code_hash 지금 {model_code_hash()} · run 기록 {rec.run_spec.get('model_code_hash')}")
    print(f"run 기록 git_sha {rec.run_spec.get('git_sha')} · code_hash {rec.run_spec.get('code_hash')}")
    print(f"config {config.model}/{config.target}/h{config.horizon} seed {config.seed} calibrate {config.calibrate}")

    recorded, schema = recorded_row(rec, fold_id)
    frame, info = regenerate_fold(rec, fold_id, dataset_dir)
    fresh_rows = frame.filter(pl.col("pred_col") == rec.primary_pred_col)
    if fresh_rows.height != 1:
        raise ValueError(f"재현 fold_rows 에 {rec.primary_pred_col} 행이 {fresh_rows.height}개")
    print(f"학습 {info['n_train']:,}행 · 검증 {info['n_valid']:,}행 · 학습 구간에 값 없는 설계 열 {len(info['empty_design_columns'])}개")
    cmp = compare_rows(recorded, fresh_rows.row(0, named=True), schema)
    print_comparison(cmp)
    if cmp.unclassified:
        print(f"분류에서 빠진 float 열 {len(cmp.unclassified)}개: {cmp.unclassified}")
    if cmp.failures:
        print(f"어긋난 열 {len(cmp.failures)}개: {cmp.failures}")
    elapsed = time.time() - started
    print(f"{'PASS' if cmp.passed else 'FAIL'}  {elapsed:.1f}s")
    return 0 if cmp.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--check", action="store_true", help="fold 하나 복원 후 RES 와 대조. 아무것도 쓰지 않는다")
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("지금은 --check 모드만 있다")
    return check(args.run_id, args.fold)


if __name__ == "__main__":
    sys.exit(main())
