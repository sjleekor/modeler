"""M4 — walk-forward 5 fold, 다섯 모델 (``03_model_candidates.md`` §4).

    uv run python -m modeler.us.m4_run

M-L Ridge · M-L ElasticNet · M-L OLS-3 · M-G(LightGBM, 타깃 ``y_rank``) ·
M-G on L1(대조군, 타깃 L1의 횡단면 순위) 다섯을 ``05_validation_protocol.md``
§2 walk-forward 5 fold로 돌려 rank IC 비교표를 낸다.

**개발 구간만 읽는다.** ``modeler.us.scan.DEV_END``(2025-06-30) 뒤 날짜는
``load_dev_frame``이 잘라내고, 모든 프레임이 ``assert_dev_window``를 거친다
— holdout을 열 경로가 이 파일 안에 없다.

**피쳐를 더하거나 정의를 바꾸지 않는다** (R6 동결). 모델 입력 여섯은 손으로
박지 않고 M3 산출물(``feature_scan`` manifest의 ``model_input_features["all"]``)
에서 읽는다. OLS-3의 세 컬럼(``OLS3_FEATURES``)은 예외다 — 등급과 무관하게
``03`` §4가 사전등록한 GKX null model이라 M3 목록과 별개다.
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from modeler.etl.config import DataRoot
from modeler.us.m4_models import (
    ENET_MAX_ITER,
    LGBM_EARLY_STOPPING_ROUNDS,
    LGBM_MAX_ESTIMATORS,
    LGBM_SEEDS,
    LgbmParams,
    elasticnet_grid,
    fit_predict_elasticnet,
    fit_predict_lgbm,
    fit_predict_ols,
    fit_predict_ridge,
    lgbm_grid,
    ridge_grid,
)
from modeler.us.m4_splits import (
    ES_VALID_MONTHS,
    FOLD_SPLIT_MONTHS,
    PURGE_EMBARGO_TRADING_DAYS,
    VALID_MONTHS,
    WfFold,
    build_wf_folds,
)
from modeler.us.m4_transform import cross_sectional_percentile, rank_transform, to_design_arrays
from modeler.us.scan import (
    DEV_END,
    DEV_START,
    HAC_LAG,
    MIN_NAMES,
    assert_dev_window,
    ic_and_t,
    load_dev_frame,
    monthly_rank_ic,
)

# --- 0. 상수 ------------------------------------------------------------------

#: OLS-3(GKX null model)의 사전등록 셋 — ``03`` §4. 등급·M3 목록과 무관하다.
#: size = ``mcap_rank``(F4 유동성·규모) · B/M = ``bm``(F5 밸류) ·
#: momentum = ``mom_12_1``(F1 모멘텀, 12-1개월 표준 정의) — GKX(2020) 원 정의와
#: 이름·의미가 가장 가까운 컬럼 셋이다.
OLS3_FEATURES: tuple[str, ...] = ("mcap_rank", "bm", "mom_12_1")

#: M3 산출물 디렉터리 이름 — ``scan.py``의 ``main()``이 쓰는 것과 같다.
FEATURE_SCAN_DIR_NAME = "feature_scan"

#: rank IC 평가는 항상 L2(중립화 잔차)와 예측의 스피어만 상관이다
#: (``05`` §4 "정의: 월별 spearman(예측, L2) 평균") — 학습 타깃이 ``y_rank``든
#: L1의 순위든, 평가축은 하나로 고정해야 M-G(L2)와 M-G on L1을 나란히 비교할
#: 수 있다("알파가 사이즈·업종에 얼마나 있었나", ``03`` §4).
EVAL_Y_COL = "L2"


# --- 1. M3 모델 입력 읽기 ------------------------------------------------------


def load_m3_model_input_features(root: DataRoot) -> tuple[list[str], Path]:
    """M3 산출물(``feature_scan``) 최신 snapshot manifest에서
    ``model_input_features["all"]``을 읽는다. 손으로 박지 않는다(``06`` 지시)."""
    base = root.output / FEATURE_SCAN_DIR_NAME
    if not base.is_dir():
        raise FileNotFoundError(f"M3 산출물이 없습니다: {base}")
    snapshot_dirs = sorted(
        p for p in base.iterdir() if p.is_dir() and p.name.startswith("snapshot_date=")
    )
    if not snapshot_dirs:
        raise FileNotFoundError(f"{base} 아래 snapshot_date= 디렉터리가 없습니다")
    latest = snapshot_dirs[-1]
    manifest_path = latest / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    features = manifest["model_input_features"]["all"]
    if not features:
        raise ValueError(f"{manifest_path}의 model_input_features['all']이 비어 있습니다")
    return list(features), manifest_path


# --- 2. 입력 조립 --------------------------------------------------------------


@dataclass
class M4Inputs:
    core: (
        pl.DataFrame
    )  # date, symbol, month_idx, <feature>_rank/_isna..., L0,L1,L2,y_rank,y_up,l1_rank
    dates: list[date]
    model_features: list[str]
    m3_manifest_path: Path
    features_content_hash: str | None
    labels_content_hash: str | None
    features_row_count: int
    labels_row_count: int


def _dataset_manifest_field(root: DataRoot, name: str, field_name: str):
    manifest_path = root.datasets / name / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return manifest.get(field_name)


def build_m4_inputs(root: DataRoot) -> M4Inputs:
    model_features, m3_manifest_path = load_m3_model_input_features(root)

    features_dev = load_dev_frame(root, "us_features_v1")
    labels_dev = load_dev_frame(root, "us_labels_v1")
    assert_dev_window(features_dev)
    assert_dev_window(labels_dev)

    needed = sorted(set(model_features) | set(OLS3_FEATURES))
    isna_cols = [f"{c}_isna" for c in needed]
    base_cols = ["date", "symbol", *needed, *isna_cols]
    core = features_dev.select(base_cols).join(
        labels_dev.select("date", "symbol", "L0", "L1", "L2", "y_rank", "y_up"),
        on=["date", "symbol"],
        how="inner",
    )
    assert_dev_window(core)

    core = rank_transform(core, needed)
    core = cross_sectional_percentile(core, "L1", out_col="l1_rank")
    core = core.with_columns(pl.col("date").rank(method="dense").cast(pl.Int64).alias("month_idx"))

    dates = sorted(core["date"].unique().to_list())

    return M4Inputs(
        core=core,
        dates=dates,
        model_features=model_features,
        m3_manifest_path=m3_manifest_path,
        features_content_hash=_dataset_manifest_field(root, "us_features_v1", "content_hash"),
        labels_content_hash=_dataset_manifest_field(root, "us_labels_v1", "content_hash"),
        features_row_count=features_dev.height,
        labels_row_count=labels_dev.height,
    )


# --- 3. walk-forward 실행 ------------------------------------------------------

FitPredictFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

_OOF_KEEP_COLS = ("date", "symbol", "month_idx", "L2", "y_rank")


def evaluate_oof(oof: pl.DataFrame, *, group_col: str = "date") -> tuple[float, int]:
    """fold(또는 전체) OOF 예측의 평균 rank IC — ``group_col``별 spearman(pred, L2) 평균."""
    ic_table = monthly_rank_ic(
        oof, x_col="pred", y_col=EVAL_Y_COL, group_col=group_col, min_names=MIN_NAMES
    )
    values = ic_table["ic"].drop_nulls()
    return (float(values.mean()) if values.len() else float("nan")), ic_table.height


def run_walk_forward_linear(
    core: pl.DataFrame,
    folds: list[WfFold],
    feature_cols: list[str],
    *,
    target_col: str,
    fit_predict_fn: FitPredictFn,
) -> tuple[pl.DataFrame, list[float]]:
    """Ridge/ElasticNet/OLS-3 공통 walk-forward 실행.

    fold마다 학습 전체(``fold.train_dates``)에 맞추고 검증(``fold.valid_dates``)에
    예측한다 — early stopping이 없는 모델이라 ES 분할을 쓰지 않는다.
    """
    frames = []
    fold_ics = []
    for fold in folds:
        train_df = core.filter(pl.col("date").is_in(fold.train_dates))
        valid_df = core.filter(pl.col("date").is_in(fold.valid_dates))
        x_train, _ = to_design_arrays(train_df, feature_cols)
        y_train = train_df[target_col].to_numpy()
        x_valid, _ = to_design_arrays(valid_df, feature_cols)
        preds = fit_predict_fn(x_train, y_train, x_valid)
        pred_df = valid_df.select(*_OOF_KEEP_COLS).with_columns(
            pl.Series("pred", preds), pl.lit(fold.fold_id).alias("fold_id")
        )
        fold_ic, _ = evaluate_oof(pred_df)
        fold_ics.append(fold_ic)
        frames.append(pred_df)
    return pl.concat(frames), fold_ics


def run_walk_forward_lgbm(
    core: pl.DataFrame,
    folds: list[WfFold],
    feature_cols: list[str],
    *,
    target_col: str,
    params: LgbmParams,
    seed: int,
) -> tuple[pl.DataFrame, list[float], list[int]]:
    """LightGBM walk-forward. early stopping은 fold 학습 구간의 마지막
    ``ES_VALID_MONTHS``개월(``fold.es_valid_dates``)로 하고, 테스트 fold
    (``fold.valid_dates``)는 보지 않는다(``05`` §2)."""
    frames = []
    fold_ics = []
    best_iterations = []
    for fold in folds:
        es_train_df = core.filter(pl.col("date").is_in(fold.es_train_dates))
        es_valid_df = core.filter(pl.col("date").is_in(fold.es_valid_dates))
        valid_df = core.filter(pl.col("date").is_in(fold.valid_dates))
        x_es_train, _ = to_design_arrays(es_train_df, feature_cols)
        y_es_train = es_train_df[target_col].to_numpy()
        x_es_valid, _ = to_design_arrays(es_valid_df, feature_cols)
        y_es_valid = es_valid_df[target_col].to_numpy()
        x_valid, _ = to_design_arrays(valid_df, feature_cols)
        preds, best_iteration = fit_predict_lgbm(
            params,
            x_es_train=x_es_train,
            y_es_train=y_es_train,
            x_es_valid=x_es_valid,
            y_es_valid=y_es_valid,
            x_valid=x_valid,
            seed=seed,
        )
        pred_df = valid_df.select(*_OOF_KEEP_COLS).with_columns(
            pl.Series("pred", preds), pl.lit(fold.fold_id).alias("fold_id")
        )
        fold_ic, _ = evaluate_oof(pred_df)
        fold_ics.append(fold_ic)
        best_iterations.append(best_iteration)
        frames.append(pred_df)
    return pl.concat(frames), fold_ics, best_iterations


def _safe_mean_ic(fold_ics: list[float]) -> float:
    """유한값만으로 평균 IC — 그리드 선택 기준.

    ``np.mean``을 NaN 섞인 리스트에 바로 쓰면 결과가 NaN이 되고, ``나중
    후보 mean_ic > best_mean_ic`` 비교에서 ``real > nan``이 항상 False라
    그 뒤로 어떤 후보가 와도 최초 NaN 후보에서 멈춘다(실측 — ElasticNet
    alpha=0.1이 5 fold 다 예측 상수라 IC가 정의 안 될 때 이 버그가 그리드
    나머지 23개를 다 가렸다). 유한값이 하나도 없으면 ``-inf``로 둬 "고를 수
    없음"을 나타낸다(한국 ``_01`` ``train.py``의 ``_isnan`` 필터와 같은 관례).
    """
    finite = [v for v in fold_ics if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("-inf")


# --- 4. 결과 묶음 --------------------------------------------------------------


@dataclass
class ModelRunResult:
    model_id: str
    target_col: str
    feature_cols: list[str]
    selected_params: dict
    fold_ics: list[float]
    rank_ic_mean: float
    rank_ic_t_hac: float
    n_months_pooled: int
    fold_ic_sd: float
    oof: pl.DataFrame
    train_seconds: float
    extra: dict = field(default_factory=dict)

    def metrics_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "target_col": self.target_col,
            "feature_cols": self.feature_cols,
            "selected_params": self.selected_params,
            "fold_ic": self.fold_ics,
            "fold_ic_sd": self.fold_ic_sd,
            "rank_ic_mean": self.rank_ic_mean,
            "rank_ic_t_hac": self.rank_ic_t_hac,
            "n_months_pooled": self.n_months_pooled,
            "train_seconds": self.train_seconds,
            **self.extra,
        }


def _pooled_stats(oof: pl.DataFrame) -> tuple[float, float, int]:
    ic_table = monthly_rank_ic(
        oof, x_col="pred", y_col=EVAL_Y_COL, group_col="month_idx", min_names=MIN_NAMES
    )
    ic_mean, t_nw, n_dates = ic_and_t(ic_table, group_col="month_idx", lag=HAC_LAG)
    return ic_mean, t_nw, n_dates


def _finalize(
    *,
    model_id: str,
    target_col: str,
    feature_cols: list[str],
    selected_params: dict,
    oof: pl.DataFrame,
    fold_ics: list[float],
    train_seconds: float,
    extra: dict | None = None,
) -> ModelRunResult:
    rank_ic_mean, rank_ic_t_hac, n_months_pooled = _pooled_stats(oof)
    finite_fold_ics = [v for v in fold_ics if np.isfinite(v)]
    fold_ic_sd = (
        float(np.std(finite_fold_ics, ddof=1)) if len(finite_fold_ics) > 1 else float("nan")
    )
    return ModelRunResult(
        model_id=model_id,
        target_col=target_col,
        feature_cols=feature_cols,
        selected_params=selected_params,
        fold_ics=fold_ics,
        rank_ic_mean=rank_ic_mean,
        rank_ic_t_hac=rank_ic_t_hac,
        n_months_pooled=n_months_pooled,
        fold_ic_sd=fold_ic_sd,
        oof=oof,
        train_seconds=train_seconds,
        extra=extra or {},
    )


# --- 5. 모델별 실행 ------------------------------------------------------------


def run_ridge(inputs: M4Inputs, folds: list[WfFold]) -> ModelRunResult:
    t0 = time.monotonic()
    grid = ridge_grid()
    grid_table = []
    best = None
    for params in grid:
        fn: FitPredictFn = functools.partial(fit_predict_ridge, params)
        oof, fold_ics = run_walk_forward_linear(
            inputs.core, folds, inputs.model_features, target_col="y_rank", fit_predict_fn=fn
        )
        mean_ic = _safe_mean_ic(fold_ics)
        grid_table.append({"params": params, "fold_ic": fold_ics, "mean_ic": mean_ic})
        if best is None or mean_ic > best[1]:
            best = (params, mean_ic, fold_ics, oof)
    params, _mean_ic, fold_ics, oof = best
    return _finalize(
        model_id="m4_ridge",
        target_col="y_rank",
        feature_cols=inputs.model_features,
        selected_params=params,
        oof=oof,
        fold_ics=fold_ics,
        train_seconds=time.monotonic() - t0,
        extra={"grid_search": grid_table},
    )


def run_elasticnet(inputs: M4Inputs, folds: list[WfFold]) -> ModelRunResult:
    t0 = time.monotonic()
    grid = elasticnet_grid()
    grid_table = []
    best = None
    for params in grid:
        fn: FitPredictFn = functools.partial(fit_predict_elasticnet, params)
        oof, fold_ics = run_walk_forward_linear(
            inputs.core, folds, inputs.model_features, target_col="y_rank", fit_predict_fn=fn
        )
        mean_ic = _safe_mean_ic(fold_ics)
        grid_table.append({"params": params, "fold_ic": fold_ics, "mean_ic": mean_ic})
        if best is None or mean_ic > best[1]:
            best = (params, mean_ic, fold_ics, oof)
    params, _mean_ic, fold_ics, oof = best
    return _finalize(
        model_id="m4_enet",
        target_col="y_rank",
        feature_cols=inputs.model_features,
        selected_params=params,
        oof=oof,
        fold_ics=fold_ics,
        train_seconds=time.monotonic() - t0,
        extra={"grid_search": grid_table, "max_iter": ENET_MAX_ITER},
    )


def run_ols3(inputs: M4Inputs, folds: list[WfFold]) -> ModelRunResult:
    t0 = time.monotonic()
    oof, fold_ics = run_walk_forward_linear(
        inputs.core, folds, list(OLS3_FEATURES), target_col="y_rank", fit_predict_fn=fit_predict_ols
    )
    return _finalize(
        model_id="m4_ols3",
        target_col="y_rank",
        feature_cols=list(OLS3_FEATURES),
        selected_params={},
        oof=oof,
        fold_ics=fold_ics,
        train_seconds=time.monotonic() - t0,
        extra={
            "ols3_column_choice": {
                "size": "mcap_rank",
                "book_to_market": "bm",
                "momentum": "mom_12_1",
            }
        },
    )


def run_lgbm(
    inputs: M4Inputs,
    folds: list[WfFold],
    *,
    model_id: str,
    target_col: str,
    grid_search_seed: int = 0,
    seeds_for_variance: tuple[int, ...] = LGBM_SEEDS,
) -> ModelRunResult:
    t0 = time.monotonic()
    grid = lgbm_grid()
    grid_table = []
    best = None
    for params in grid:
        oof, fold_ics, best_iters = run_walk_forward_lgbm(
            inputs.core,
            folds,
            inputs.model_features,
            target_col=target_col,
            params=params,
            seed=grid_search_seed,
        )
        mean_ic = _safe_mean_ic(fold_ics)
        grid_table.append(
            {
                "params": params.as_dict(),
                "fold_ic": fold_ics,
                "mean_ic": mean_ic,
                "best_iteration": best_iters,
            }
        )
        if best is None or mean_ic > best[1]:
            best = (params, mean_ic, fold_ics, oof, best_iters)
    params, _mean_ic, fold_ics, oof, best_iters = best

    seed_ic_means: dict[int, float] = {}
    for seed in seeds_for_variance:
        if seed == grid_search_seed:
            seed_oof, seed_fold_ics = oof, fold_ics
        else:
            seed_oof, seed_fold_ics, _ = run_walk_forward_lgbm(
                inputs.core,
                folds,
                inputs.model_features,
                target_col=target_col,
                params=params,
                seed=seed,
            )
        seed_ic_mean, _t, _n = _pooled_stats(seed_oof)
        seed_ic_means[seed] = seed_ic_mean

    seed_values = list(seed_ic_means.values())
    seed_ic_sd = float(np.std(seed_values, ddof=1)) if len(seed_values) > 1 else float("nan")
    seed_ic_mean_of_means = float(np.mean(seed_values))

    return _finalize(
        model_id=model_id,
        target_col=target_col,
        feature_cols=inputs.model_features,
        selected_params=params.as_dict(),
        oof=oof,
        fold_ics=fold_ics,
        train_seconds=time.monotonic() - t0,
        extra={
            "grid_search": grid_table,
            "grid_search_seed": grid_search_seed,
            "best_iteration_per_fold": best_iters,
            "seed_ic_mean": seed_ic_means,
            "seed_ic_mean_of_means": seed_ic_mean_of_means,
            "seed_ic_sd": seed_ic_sd,
            "seed_ic_sd_over_mean_pct": (
                100.0 * seed_ic_sd / abs(seed_ic_mean_of_means)
                if seed_ic_mean_of_means and np.isfinite(seed_ic_sd)
                else float("nan")
            ),
            "early_stopping_rounds": LGBM_EARLY_STOPPING_ROUNDS,
            "max_estimators": LGBM_MAX_ESTIMATORS,
        },
    )


def run_lgbm_on_l1(
    inputs: M4Inputs, folds: list[WfFold], *, m_g_result: ModelRunResult
) -> ModelRunResult:
    """대조군 — M-G와 **같은 설정**, 타깃만 L1의 횡단면 순위(``03`` §4).

    별도 그리드서치를 하지 않는다 — ``m_g_result.selected_params``를 그대로
    받는다. seed는 M-G의 grid-search seed(=0)와 같은 하나만 쓴다(시드 분산은
    M-G(L2)에서만 본다, M4 지시 §6-5의 범위).
    """
    t0 = time.monotonic()
    params = LgbmParams(**m_g_result.selected_params)
    oof, fold_ics, best_iters = run_walk_forward_lgbm(
        inputs.core, folds, inputs.model_features, target_col="l1_rank", params=params, seed=0
    )
    return _finalize(
        model_id="m4_lgbm_on_l1",
        target_col="l1_rank",
        feature_cols=inputs.model_features,
        selected_params=params.as_dict(),
        oof=oof,
        fold_ics=fold_ics,
        train_seconds=time.monotonic() - t0,
        extra={
            "note": "M-G(L2)와 같은 하이퍼파라미터, 타깃만 L1 순위. 그리드서치 재실행 없음.",
            "best_iteration_per_fold": best_iters,
            "reused_from": m_g_result.model_id,
        },
    )


# --- 6. 출력 -------------------------------------------------------------------


def _git_commit(repo: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}-dirty" if dirty else head


def _fold_records(folds: list[WfFold], core: pl.DataFrame) -> list[dict]:
    records = []
    for fold in folds:
        train_n = core.filter(pl.col("date").is_in(fold.train_dates)).height
        valid_n = core.filter(pl.col("date").is_in(fold.valid_dates)).height
        es_train_n = core.filter(pl.col("date").is_in(fold.es_train_dates)).height
        es_valid_n = core.filter(pl.col("date").is_in(fold.es_valid_dates)).height
        records.append(
            {
                "fold_id": fold.fold_id,
                "split_k": fold.split_k.isoformat(),
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "train_months": len(fold.train_dates),
                "train_rows": train_n,
                "valid_start": fold.valid_start.isoformat(),
                "valid_end": fold.valid_end.isoformat(),
                "valid_months": len(fold.valid_dates),
                "valid_rows": valid_n,
                "es_train_start": fold.es_train_dates[0].isoformat(),
                "es_train_end": fold.es_train_dates[-1].isoformat(),
                "es_train_months": len(fold.es_train_dates),
                "es_train_rows": es_train_n,
                "es_valid_start": fold.es_valid_dates[0].isoformat(),
                "es_valid_end": fold.es_valid_dates[-1].isoformat(),
                "es_valid_months": len(fold.es_valid_dates),
                "es_valid_rows": es_valid_n,
            }
        )
    return records


def write_run(
    root: DataRoot,
    result: ModelRunResult,
    *,
    inputs: M4Inputs,
    folds: list[WfFold],
    snapshot_date: str,
    modeler_git_commit: str,
) -> Path:
    run_id = f"{result.model_id}_{snapshot_date.replace('-', '')}"
    out_dir = root.output / "model_runs" / run_id
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for fold_id in sorted(result.oof["fold_id"].unique().to_list()):
        fold_df = result.oof.filter(pl.col("fold_id") == fold_id)
        fold_df.write_parquet(pred_dir / f"fold_{fold_id}.parquet")

    config = {
        "run_id": run_id,
        "model_id": result.model_id,
        "target_col": result.target_col,
        "feature_cols": result.feature_cols,
        "design_matrix": "rank([0,1], 결측 0.5) + _isna 플래그, 06 §2",
        "selected_params": result.selected_params,
        "eval_y_col": EVAL_Y_COL,
        "hac_lag": HAC_LAG,
        "min_names": MIN_NAMES,
        "purge_embargo_trading_days": PURGE_EMBARGO_TRADING_DAYS,
        "valid_months": VALID_MONTHS,
        "es_valid_months": ES_VALID_MONTHS,
        "fold_split_months": [d.isoformat() for d in FOLD_SPLIT_MONTHS],
        "folds": _fold_records(folds, inputs.core),
        "dev_start": DEV_START.isoformat(),
        "dev_end": DEV_END.isoformat(),
        "m3_manifest_path": str(inputs.m3_manifest_path),
        "model_input_features_source": "us_features_v1 manifest.model_input_features.all (M3)",
        "input_datasets": {
            "us_features_v1": {
                "content_hash": inputs.features_content_hash,
                "row_count": inputs.features_row_count,
            },
            "us_labels_v1": {
                "content_hash": inputs.labels_content_hash,
                "row_count": inputs.labels_row_count,
            },
        },
        "modeler_git_commit": modeler_git_commit,
        "created_at": datetime.now(UTC).isoformat(),
    }
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    )
    (out_dir / "metrics.json").write_text(
        json.dumps(result.metrics_dict(), indent=2, ensure_ascii=False, sort_keys=True, default=str)
        + "\n"
    )
    return out_dir


# --- 7. CLI -------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="run_id 접미사로 쓸 날짜 (기본: 오늘 날짜)",
    )
    args = parser.parse_args(argv)
    snapshot_date = args.snapshot_date or date.today().isoformat()

    t_start = time.monotonic()
    root = DataRoot.resolve(market="us")
    modeler_repo = Path(__file__).resolve().parents[3]
    modeler_git_commit = _git_commit(modeler_repo)

    inputs = build_m4_inputs(root)
    folds = build_wf_folds(inputs.dates, dev_end=DEV_END)

    print(f"모델 입력: {inputs.model_features}  (M3: {inputs.m3_manifest_path})")
    for record in _fold_records(folds, inputs.core):
        print(
            f"  fold{record['fold_id']} split_k={record['split_k']} "
            f"train=({record['train_start']}~{record['train_end']}, {record['train_months']}개월, "
            f"{record['train_rows']}행) valid=({record['valid_start']}~{record['valid_end']}, "
            f"{record['valid_months']}개월, {record['valid_rows']}행)"
        )

    results: list[ModelRunResult] = []

    print("M-L Ridge ...")
    ridge_result = run_ridge(inputs, folds)
    results.append(ridge_result)

    print("M-L ElasticNet ...")
    enet_result = run_elasticnet(inputs, folds)
    results.append(enet_result)

    print("M-L OLS-3 ...")
    ols3_result = run_ols3(inputs, folds)
    results.append(ols3_result)

    print("M-G LightGBM (그리드 24점 x 5 fold, 시드 5개 분산) ...")
    lgbm_result = run_lgbm(inputs, folds, model_id="m4_lgbm", target_col="y_rank")
    results.append(lgbm_result)

    print("M-G on L1 (대조군, M-G와 같은 설정) ...")
    lgbm_l1_result = run_lgbm_on_l1(inputs, folds, m_g_result=lgbm_result)
    results.append(lgbm_l1_result)

    for result in results:
        out_dir = write_run(
            root,
            result,
            inputs=inputs,
            folds=folds,
            snapshot_date=snapshot_date,
            modeler_git_commit=modeler_git_commit,
        )
        fold_ic_str = [f"{v:.4f}" for v in result.fold_ics]
        print(
            f"{result.model_id}: rank_ic_mean={result.rank_ic_mean:.4f} "
            f"t_hac={result.rank_ic_t_hac:.2f} fold_ic={fold_ic_str} "
            f"sd={result.fold_ic_sd:.4f} -> {out_dir}"
        )

    total_seconds = time.monotonic() - t_start
    print(f"총 소요 시간: {total_seconds:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
