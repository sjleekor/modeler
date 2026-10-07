"""T1-5 DSR·PBO 계측용 단축 재생성 — 기록된 ``best_params`` 로 예측만 다시 만든다.

모드 셋.

* ``--run-id X --fold N --check`` — fold 하나 복원 후 ``RES`` 의 ``fold_metrics`` 와 대조. 아무것도 쓰지 않는다.
* ``--run-id X`` / ``--all`` + ``--out DIR`` — 15개 run(채택 제외)의 fold 1~5를 적합해
  ``DIR/runs/<run_id>/`` 에 예측·``fold_metrics``·``rebalance_returns``·``regen_spec.json`` 을 쓴다.
  ``regen_spec.json`` 의 ``status: done`` 이 있는 run 은 건너뛴다(이어 돌리기).
* ``--adopted --out DIR`` — 채택 run 은 적합하지 않는다. 저장된 예측에서 시계열만 다시 만들어
  ``RES`` 의 ``rebalance_returns`` 와 비트 단위로 대조한다.

``RES`` 와 ``stock_data/kr/datasets`` 에는 쓰지 않는다. 데이터셋 빌드도 하지 않는다.

설계: ``my/milestones/kr/modeling/assessment/20261005_dsr_pbo.md`` §1.4.
``MODEL_CODE_FILES`` 밖이라 ``run_matrix.model_code_hash`` 에 들어가지 않는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import polars as pl

from modeler.models._02_updown_prob import build_dataset as bd
from modeler.models._02_updown_prob import evaluate as ev
from modeler.models._02_updown_prob import train as tr
from modeler.etl.config import REPO_ROOT
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

ADOPTED_RUN_ID = "E2_h20_FS1h_seed0"
# 준비 문서 §1.1 의 h20 run 16개 중 채택을 뺀 15개. (stage, run_id)
REGEN_RUNS: tuple[tuple[str, str], ...] = (
    ("E0", "E0_h20_MA-rank_seed0"),
    ("E0", "E0_h20_MA-tree_seed0"),
    ("E1", "E1_h20_y_top-hgb_clf_seed0"),
    ("E1", "E1_h20_y_top-logit_seed0"),
    ("E1", "E1_h20_y_up-hgb_clf_seed0"),
    ("E1", "E1_h20_y_up-logit_seed0"),
    ("E2", "E2_h20_FS1_seed0"),
    ("E2", "E2_h20_FS2_seed0"),
    ("E3", "E3_h20_flow-native_t_seed0"),
    ("E4", "E4_h20_E4a-seed_seed1"),
    ("E4", "E4_h20_E4a-seed_seed2"),
    ("E4", "E4_h20_E4b-monotonic_seed0"),
    ("E5", "E5_h20_FS3_seed0"),
    ("E5", "E5_h20_FS3_seed1"),
    ("E5", "E5_h20_FS3_seed2"),
)
REGEN_RUN_IDS: tuple[str, ...] = tuple(r for _, r in REGEN_RUNS)
E5_STAGE = "E5"
FOLD_IDS: tuple[int, ...] = (1, 2, 3, 4, 5)
ADOPTED_DATASET_KEY = "FS1h_h20_lag1_rank"
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
    hits = sorted(p for p in results_root.glob(f"E[0-9]/{run_id}") if (p / "run_spec.json").is_file())
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


def e5_lake():
    """E5 는 ``kr/derived/_e5`` 루트의 데이터셋을 쓴다(공유 레이크와 마트가 다르다, 준비 문서 §1.2)."""
    from modeler.models._02_updown_prob.experiments import isolated_lake

    return replace(lake_config(), root=isolated_lake._isolated_root())


def lake_for(rec: RecordedRun, shared=None, e5=None):
    """stage E5 면 ``_e5`` 루트, 아니면 공유 레이크. 인자는 시험용 주입."""
    if rec.stage == E5_STAGE:
        return e5 if e5 is not None else e5_lake()
    return shared if shared is not None else lake_config()


def dataset_path(rec: RecordedRun, lake=None) -> Path:
    """기록된 ``summary.dataset_dir`` 은 쓰지 않는다 — 지워진 구 경로다."""
    lake = lake or lake_for(rec)
    spec = restore_spec(rec)
    horizon = int(rec.run_spec["train_config"]["horizon"])
    return lake.dataset_dir(spec.model_id) / bd.dataset_key(spec, horizon)


def dataset_ready(dataset_dir: Path) -> bool:
    std = dataset_dir / bd.STD_DIR
    return (
        (dataset_dir / "dataset_manifest.json").is_file()
        and std.is_dir()
        and any(std.glob("*.parquet"))
    )


def regenerate_fold(rec: RecordedRun, fold_id: int, dataset_dir: Path) -> tuple[pl.DataFrame, dict]:
    """fold 하나를 ``_fit_fold`` 로 다시 적합하고 ``fold_rows`` 의 ``primary_pred_col`` 행을 돌려준다."""
    config = restore_train_config(rec)
    source = tr.DatasetFolds(dataset_dir)
    design = source.design_columns(config.target_column)
    fit = fit_fold(rec, fold_id, source, config, design)
    result = _result_of(config, rec, [fit], design)
    rows = ev.fold_rows(
        result,
        k=int(rec.summary["k"]),
        cost_bps=float(rec.summary["cost_bps_roundtrip"]),
        tau=float(rec.summary["tau"]),
    )
    frame = pl.DataFrame(rows, infer_schema_length=None)
    info = {"n_train": fit.n_train, "n_valid": fit.n_valid, "empty_design_columns": sorted(fit.empty_design_columns)}
    return frame, info


def fit_fold(rec: RecordedRun, fold_id: int, source, config: tr.TrainConfig, design: list[str]) -> tr.FoldFit:
    train, valid = source.slices(fold_id)
    n_train, n_valid = train.height, valid.height
    predictions, calibrated, empty = tr._fit_fold(train, valid, design, config, rec.best_params)
    return tr.FoldFit(
        fold_id=fold_id,
        params=rec.best_params,
        n_train=n_train,
        n_valid=n_valid,
        metric=tr._fold_metric(predictions, config),
        calibrated=calibrated,
        empty_design_columns=list(empty),
        predictions=predictions,
    )


def _result_of(config: tr.TrainConfig, rec: RecordedRun, folds: list[tr.FoldFit], design: list[str]) -> tr.TrainResult:
    return tr.TrainResult(
        config=config,
        best_params=rec.best_params,
        best_metric=folds[0].metric if folds else float("nan"),
        grid_metrics=[],
        folds=folds,
        design_columns=design,
    )


@dataclass
class Comparison:
    table: pl.DataFrame
    unclassified: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    new_columns: list[str] = field(default_factory=list)  # 재현 쪽에만 있음 — 대조 안 함, 판정 제외

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

    ``schema`` 는 기록된 ``fold_metrics`` 의 열 종류다. 판정 규칙(2026-10-07 12:56 사용자 결정):
    재현 쪽에만 있는 열은 kind ``new`` 로 따로 적고 판정에서 뺀다("기록에 없는 새 열 — 대조 안 함").
    기록에만 있는 열은 ``failures`` 에 넣는다.
    """
    out: list[dict] = []
    unclassified: list[str] = []
    failures: list[str] = []
    new_columns: list[str] = []
    for col in sorted(set(recorded) | set(fresh), key=lambda c: list(recorded).index(c) if c in recorded else 10**6):
        if col not in recorded:
            new_columns.append(col)
            out.append({"column": col, "kind": "new", "recorded": None, "fresh": fresh[col], "diff": None, "ok": True})
            continue
        if col not in fresh:
            failures.append(f"{col}: 기록에만 있음")
            out.append({"column": col, "kind": "missing", "recorded": recorded[col], "fresh": None, "diff": None, "ok": False})
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
    return Comparison(table, unclassified, failures, new_columns)


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
        mark = "새 열 - 대조 안 함" if r["kind"] == "new" else ("ok" if r["ok"] else "XX")
        print(f"{r['column']:36s} {r['kind']:12s} {fmt(r['recorded']):>24s} {fmt(r['fresh']):>24s} {diff:>12s}  {mark}")


# ---------------------------------------------------------------- 재생성 모드

STATUS_DONE = "done"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def run_out_dir(out_root: Path, run_id: str) -> Path:
    return Path(out_root) / "runs" / run_id


def is_done(out_root: Path, run_id: str) -> bool:
    """``regen_spec.json`` 에 ``status: done`` 이 있으면 끝난 run 이다."""
    path = run_out_dir(out_root, run_id) / "regen_spec.json"
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text()).get("status") == STATUS_DONE
    except (OSError, json.JSONDecodeError):
        return False


def _check_out_root(out_root: Path) -> None:
    """RES 와 stock_data/kr/datasets 아래에는 쓰지 않는다."""
    from modeler.models._02_updown_prob.spec import lake_config as _lc

    out = Path(out_root).resolve()
    forbidden = [RESULTS_ROOT.resolve()]
    try:
        forbidden.append(Path(_lc().root.datasets).resolve())
    except Exception:  # noqa: BLE001 — 경로를 못 구하면 RES 만 막는다
        pass
    for bad in forbidden:
        if out == bad or bad in out.parents:
            raise SystemExit(f"--out {out} 은 쓰면 안 되는 곳({bad}) 아래다")


def compare_fold_metrics(rec: RecordedRun, fresh: pl.DataFrame) -> dict[int, Comparison]:
    """fold 1~5 를 ``RES`` 의 ``fold_metrics`` 와 대조한다. 기록에 행이 없으면 failures 에 적는다."""
    out: dict[int, Comparison] = {}
    for fold_id in FOLD_IDS:
        rows = fresh.filter((pl.col("fold_id") == fold_id) & (pl.col("pred_col") == rec.primary_pred_col))
        try:
            recorded, schema = recorded_row(rec, fold_id)
        except ValueError as exc:
            out[fold_id] = Comparison(pl.DataFrame(), failures=[str(exc)])
            continue
        if rows.height != 1:
            out[fold_id] = Comparison(pl.DataFrame(), failures=[f"재현 fold_rows 에 fold {fold_id} 행이 {rows.height}개"])
            continue
        out[fold_id] = compare_rows(recorded, rows.row(0, named=True), schema)
    return out


def _fold_summary(cmps: dict[int, Comparison]) -> dict:
    return {
        str(f): {
            "pass": c.passed,
            "failures": c.failures,
            "unclassified": c.unclassified,
            "new_columns": c.new_columns,
            "max_rel_diff_close_cols": max(
                (r["diff"] for r in c.table.iter_rows(named=True) if r["kind"] == "rel<=1e-12" and isinstance(r["diff"], float)),
                default=None,
            ) if c.table.height else None,
        }
        for f, c in cmps.items()
    }


def regenerate_run(run_id: str, out_root: Path, *, lake=None, log=print) -> dict:
    """한 run 을 fold 1~5 적합 → 평가 → 출력. FAIL 이어도 쓰고 ``status: done`` 을 남긴다."""
    started = time.time()
    rec = load_recorded(run_id)
    dataset_dir = dataset_path(rec, lake=lake_for(rec, shared=lake, e5=lake))
    if not dataset_ready(dataset_dir):
        return {"run_id": run_id, "status": "no_dataset", "dataset_dir": str(dataset_dir)}
    config = restore_train_config(rec)
    source = tr.DatasetFolds(dataset_dir)
    design = source.design_columns(config.target_column)
    log(f"[{run_id}] 데이터셋 {dataset_dir} · pred_col {rec.primary_pred_col} · best_params {rec.best_params}")
    folds: list[tr.FoldFit] = []
    for fold_id in FOLD_IDS:
        t0 = time.time()
        folds.append(fit_fold(rec, fold_id, source, config, design))
        log(f"[{run_id}] fold {fold_id} 적합 {time.time() - t0:.1f}s · 학습 {folds[-1].n_train:,} · 검증 {folds[-1].n_valid:,}")
    result = _result_of(config, rec, folds, design)
    evaluation = ev.evaluate_run(
        result,
        k=int(rec.summary["k"]),
        cost_bps=float(rec.summary["cost_bps_roundtrip"]),
        tau=float(rec.summary["tau"]),
    )
    cmps = compare_fold_metrics(rec, evaluation.fold_metrics)
    for fold_id, cmp in cmps.items():
        log(
            f"[{run_id}] fold {fold_id} {'PASS' if cmp.passed else 'FAIL'}"
            f" 어긋남 {cmp.failures} 분류밖 {cmp.unclassified} 새 열 {len(cmp.new_columns)}개(대조 안 함)"
        )
    out_dir = run_out_dir(out_root, run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.predictions.write_parquet(out_dir / "predictions_valid.parquet")
    evaluation.fold_metrics.write_parquet(out_dir / "fold_metrics.parquet")
    evaluation.rebalance_returns.write_parquet(out_dir / "rebalance_returns.parquet")
    all_pass = all(c.passed for c in cmps.values())
    spec = {
        "status": STATUS_DONE,
        "run_id": run_id,
        "stage": rec.stage,
        "all_folds_pass": all_pass,
        "primary_pred_col": rec.primary_pred_col,
        "best_params": rec.best_params,
        "model_code_hash": model_code_hash(),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--short")),
        "dataset_dir": str(dataset_dir),
        "dataset_manifest_sha256": _sha256(dataset_dir / "dataset_manifest.json"),
        "original_run_spec_sha256": _sha256(rec.run_dir / "run_spec.json"),
        "original_summary_sha256": _sha256(rec.run_dir / "summary.json"),
        "original_model_code_hash": rec.run_spec.get("model_code_hash"),
        "original_git_sha": rec.run_spec.get("git_sha"),
        "folds": _fold_summary(cmps),
        "elapsed_seconds": round(time.time() - started, 1),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / "regen_spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False, default=str))
    return {"run_id": run_id, "status": "pass" if all_pass else "fail", "elapsed": spec["elapsed_seconds"],
            "bad_folds": [f for f, c in cmps.items() if not c.passed]}


def regen_many(run_ids: list[str], out_root: Path, *, lake=None, log=print) -> list[dict]:
    """run 을 차례로 돈다. 끝난 run 은 건너뛰고, 한 run 이 예외로 죽어도 다음으로 간다."""
    _check_out_root(out_root)
    rows: list[dict] = []
    for run_id in run_ids:
        if is_done(out_root, run_id):
            spec = json.loads((run_out_dir(out_root, run_id) / "regen_spec.json").read_text())
            log(f"[{run_id}] 이미 끝남 - 건너뜀")
            rows.append({"run_id": run_id, "status": "skip_done", "prev_pass": spec.get("all_folds_pass")})
            continue
        try:
            rows.append(regenerate_run(run_id, out_root, lake=lake, log=log))
        except Exception as exc:  # noqa: BLE001 — 기록하고 다음 run 으로
            log(f"[{run_id}] 오류: {type(exc).__name__}: {exc}")
            rows.append({"run_id": run_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return rows


def print_summary(rows: list[dict]) -> None:
    print("\n== 요약 ==")
    print(f"{'run_id':34s} {'결과':14s} {'초':>8s}  비고")
    for r in rows:
        status = {"pass": "PASS", "fail": "FAIL", "no_dataset": "데이터셋 없음", "skip_done": "건너뜀(완료)", "error": "오류"}.get(r["status"], r["status"])
        note = ""
        if r["status"] == "fail":
            note = f"어긋난 fold {r['bad_folds']}"
        elif r["status"] == "no_dataset":
            note = r["dataset_dir"]
        elif r["status"] == "skip_done":
            note = f"이전 결과 {'PASS' if r.get('prev_pass') else 'FAIL'}"
        elif r["status"] == "error":
            note = r["error"]
        print(f"{r['run_id']:34s} {status:14s} {r.get('elapsed', ''):>8}  {note}")


# ---------------------------------------------------------------- 채택 run

def _float_bits_equal(a: pl.Series, b: pl.Series) -> bool:
    if a.null_count() != b.null_count() or a.is_null().to_list() != b.is_null().to_list():
        return False
    x = a.fill_null(0.0).cast(pl.Float64).to_numpy().view(np.int64)
    y = b.fill_null(0.0).cast(pl.Float64).to_numpy().view(np.int64)
    return bool(np.array_equal(x, y))


def frames_bit_equal(a: pl.DataFrame, b: pl.DataFrame) -> tuple[bool, list[str]]:
    """열 이름·순서·dtype·행 수, float 은 비트, 나머지는 값이 같은지. 어긋난 것을 적어 돌려준다."""
    problems: list[str] = []
    if a.columns != b.columns:
        problems.append(f"열 다름 {a.columns} vs {b.columns}")
        return False, problems
    if a.height != b.height:
        problems.append(f"행 수 {a.height} vs {b.height}")
        return False, problems
    for col in a.columns:
        if a.schema[col] != b.schema[col]:
            problems.append(f"{col}: dtype {a.schema[col]} vs {b.schema[col]}")
            continue
        ok = _float_bits_equal(a[col], b[col]) if a.schema[col].is_float() else a[col].equals(b[col])
        if not ok:
            problems.append(f"{col}: 값 다름")
    return not problems, problems


def adopted(out_root: Path, *, res_root: Path = RESULTS_ROOT, dataset_dir: Path | None = None, log=print) -> int:
    """저장된 예측에서 ``rebalance_returns`` 를 다시 만들어 ``RES`` 의 65행과 비트 대조한다. 적합하지 않는다."""
    _check_out_root(out_root)
    started = time.time()
    rec = load_recorded(ADOPTED_RUN_ID, res_root)
    config = restore_train_config(rec)
    if dataset_dir is None:
        dataset_dir = dataset_path(rec)
    pred_path = dataset_dir / f"predictions_valid__{ADOPTED_RUN_ID}.parquet"
    predictions = pl.read_parquet(pred_path)
    # 저장된 예측에는 fold_id 열이 없다 — 데이터셋의 split_folds 검증 구간으로 나눈다.
    split = pl.read_parquet(dataset_dir / "split_folds.parquet")
    n_valid_rec = pl.read_parquet(rec.run_dir / "fold_metrics.parquet").filter(
        pl.col("pred_col") == rec.primary_pred_col
    )
    folds = []
    for row in split.filter(pl.col("role") == "fold").sort("fold_id").iter_rows(named=True):
        part = predictions.filter(
            (pl.col(config.date_col) >= row["valid_start"]) & (pl.col(config.date_col) <= row["valid_end"])
        )
        want = n_valid_rec.filter(pl.col("fold_id") == row["fold_id"]).get_column("n_valid").item()
        if part.height != want:
            raise ValueError(f"fold {row['fold_id']}: 예측 {part.height}행, 기록 n_valid {want}")
        folds.append(
            tr.FoldFit(
                fold_id=int(row["fold_id"]), params=rec.best_params, n_train=0, n_valid=part.height,
                metric=float("nan"),
                calibrated="p_cal" in part.columns and part.get_column("p_cal").null_count() < part.height,
                predictions=part,
            )
        )
    if sum(f.n_valid for f in folds) != predictions.height:
        raise ValueError("fold 로 나눈 행 합이 예측 전체와 다르다")
    folds.sort(key=lambda f: f.fold_id)
    result = _result_of(config, rec, folds, [])
    fresh = ev._rebalance_returns(result, k=int(rec.summary["k"]), cost_bps=float(rec.summary["cost_bps_roundtrip"]))
    recorded = pl.read_parquet(rec.run_dir / "rebalance_returns.parquet")
    ok, problems = frames_bit_equal(recorded, fresh)
    log(f"채택 run {ADOPTED_RUN_ID}: 예측 {pred_path.name} {predictions.height:,}행 · fold {[f.fold_id for f in folds]}")
    log(f"재계산 {fresh.height}행 x {fresh.width}열 · RES 기록 {recorded.height}행 x {recorded.width}열")
    log(f"비트 대조: {'PASS (65행 전 열 비트 일치)' if ok and fresh.height == 65 else 'FAIL ' + str(problems)}")
    out_dir = run_out_dir(out_root, ADOPTED_RUN_ID)
    out_dir.mkdir(parents=True, exist_ok=True)
    fresh.write_parquet(out_dir / "rebalance_returns.parquet")
    pl.read_parquet(rec.run_dir / "fold_metrics.parquet").write_parquet(out_dir / "fold_metrics.parquet")
    (out_dir / "regen_spec.json").write_text(
        json.dumps(
            {
                "status": STATUS_DONE,
                "run_id": ADOPTED_RUN_ID,
                "mode": "adopted_recompute_from_saved_predictions",
                "note": "적합하지 않음. fold_metrics 는 RES 기록 복사. 예측 파일은 복사하지 않고 경로만 적는다",
                "bitwise_equal_to_res": ok,
                "problems": problems,
                "n_rows": fresh.height,
                "predictions_path": str(pred_path),
                "predictions_sha256": _sha256(pred_path),
                "res_rebalance_returns_sha256": _sha256(rec.run_dir / "rebalance_returns.parquet"),
                "model_code_hash": model_code_hash(),
                "git_sha": _git("rev-parse", "HEAD"),
                "primary_pred_col": rec.primary_pred_col,
                "elapsed_seconds": round(time.time() - started, 1),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if ok and fresh.height == 65 else 1


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
    if cmp.new_columns:
        print(f"기록에 없는 새 열 {len(cmp.new_columns)}개 - 대조 안 함, 판정에서 뺌: {cmp.new_columns}")
    if cmp.failures:
        print(f"어긋난 열 {len(cmp.failures)}개: {cmp.failures}")
    elapsed = time.time() - started
    print(f"{'PASS' if cmp.passed else 'FAIL'}  {elapsed:.1f}s")
    return 0 if cmp.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id")
    parser.add_argument("--all", action="store_true", help=f"{len(REGEN_RUN_IDS)}개 run(채택 제외) 모두")
    parser.add_argument("--adopted", action="store_true", help="채택 run 시계열을 저장된 예측에서 다시 만들어 RES 와 비트 대조")
    parser.add_argument("--out", type=Path, help="출력 폴더. 재생성·--adopted 에 필수")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--check", action="store_true", help="fold 하나 복원 후 RES 와 대조. 아무것도 쓰지 않는다")
    args = parser.parse_args(argv)
    if args.check:
        if not (args.run_id and args.fold):
            parser.error("--check 에는 --run-id 와 --fold 가 필요하다")
        return check(args.run_id, args.fold)
    if args.adopted:
        if not args.out:
            parser.error("--adopted 에는 --out 이 필요하다")
        return adopted(args.out)
    if not args.out:
        parser.error("재생성에는 --out 이 필요하다")
    if bool(args.run_id) == args.all:
        parser.error("--run-id 와 --all 중 하나만")
    if args.run_id:
        if args.run_id == ADOPTED_RUN_ID:
            parser.error("채택 run 은 --adopted 로 한다")
        if args.run_id not in REGEN_RUN_IDS:
            parser.error(f"{args.run_id} 는 재생성 대상 15개에 없다")
    run_ids = list(REGEN_RUN_IDS) if args.all else [args.run_id]
    rows = regen_many(run_ids, args.out, log=lambda m: print(m, flush=True))
    print_summary(rows)
    return 1 if any(r["status"] in {"fail", "error"} for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
