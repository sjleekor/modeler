"""M5 — 앙상블(M-E)과 채택 판정 (``03_model_candidates.md`` §4 M-E · §5 채택 규칙).

    uv run python -m modeler.us.m5_ensemble

**M4 결과가 M5 범위를 정했다.** M-G(LightGBM)가 M-L(Ridge·ElasticNet)을 못
이겼다 — 그래서 M-R(lambdarank)·M-C(분류)는 돌리지 않는다(``03`` §4·§5,
``06_execution_steps.md`` M4 완료 판정). M-E는 **M-G 시드 5개 예측의 순위
평균**뿐이다.

**M4가 저장한 것은 시드별 예측이 아니다.** ``m4_run.run_lgbm``은 그리드서치
seed(=0, 최선 설정)의 OOF만 ``write_run``으로 저장하고, 나머지 시드
1\\~4는 ``seed_ic_mean``(fold 평균 rank IC 스칼라 하나)만 ``metrics.json``에
남긴다 — fold별 예측 parquet가 없다. 그래서 이 모듈이 **M4가 고른 설정
그대로**(``MG_SELECTED_PARAMS`` — ``m4_lgbm_*/metrics.json``
``selected_params``와 같다) 시드 0\\~4를 다시 돌려 fold별 예측을 저장한다.
**그리드서치는 하지 않는다** — 재사용하는 함수(``run_walk_forward_lgbm``)
자체가 그리드를 모른다.

**개발 구간만 읽는다.** ``build_m4_inputs``·``build_wf_folds``를 M4와
그대로 재사용하므로, holdout 날짜 벽(``modeler.us.scan.DEV_END``·
``assert_dev_window``)도 그대로 적용된다 — 이 파일 안에 holdout을 열 경로가
없다.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from modeler.etl.config import DataRoot
from modeler.us.m4_models import LgbmParams
from modeler.us.m4_run import (
    EVAL_Y_COL,
    M4Inputs,
    _fold_records,
    _git_commit,
    _pooled_stats,
    build_m4_inputs,
    evaluate_oof,
    run_walk_forward_lgbm,
)
from modeler.us.m4_splits import WfFold, build_wf_folds
from modeler.us.m4_transform import cross_sectional_percentile
from modeler.us.scan import DEV_END, HAC_LAG, MIN_NAMES

# --- 0. 상수 ------------------------------------------------------------------

#: M4가 그리드서치로 고른 M-G 설정 — 재실행하지 않는다(M5 지시 §2).
#: ``m4_lgbm_<날짜>/metrics.json``의 ``selected_params``와 같은 값이다
#: (2026-09-21 M4 결과 확인).
MG_SELECTED_PARAMS = LgbmParams(
    num_leaves=15, learning_rate=0.1, min_data_in_leaf=200, feature_fraction=1.0
)

#: 앙상블 재료 시드 5개 — M4의 ``LGBM_SEEDS``와 같다(``03`` §4 "시드 5개 —
#: 검증용이 아니라 앙상블 재료다").
ENSEMBLE_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

#: M4 산출물에서 M-L(Ridge·ElasticNet) 비교 숫자를 읽어올 model_id — 재실행하지
#: 않는다(M5는 M-E만 만든다).
M4_COMPARISON_MODEL_IDS: tuple[str, ...] = ("m4_ridge", "m4_enet", "m4_lgbm")


# --- 1. 시드별 M-G 실행 --------------------------------------------------------


def run_mg_seed(
    inputs: M4Inputs,
    folds: list[WfFold],
    *,
    seed: int,
    params: LgbmParams = MG_SELECTED_PARAMS,
) -> pl.DataFrame:
    """M-G를 시드 하나로 5 fold walk-forward 실행.

    ``m4_run.run_walk_forward_lgbm``을 그대로 재사용한다 — early stopping
    분할(``fold.es_train_dates``/``es_valid_dates``)·purge/embargo(fold 경계
    자체에 이미 반영됨) 전부 M4와 동일하다. 그리드서치가 없다 — ``params``는
    고정값 하나다.
    """
    oof, _fold_ics, _best_iters = run_walk_forward_lgbm(
        inputs.core,
        folds,
        inputs.model_features,
        target_col="y_rank",
        params=params,
        seed=seed,
    )
    return oof


# --- 2. 앙상블 — 순위 평균 -----------------------------------------------------


def build_ensemble_oof(seed_oofs: dict[int, pl.DataFrame]) -> pl.DataFrame:
    """M-E 정의(``03`` §4, M5 지시 §2):

        각 시드 예측을 그날 횡단면 백분위로 바꾼 뒤 평균하고, 그 평균을
        다시 순위로 본다.

    시드마다 예측 스케일이 다를 수 있어 원값(``pred``)을 직접 평균하지
    않는다 — 항상 ``cross_sectional_percentile``로 백분위 공간으로 옮긴
    뒤에만 평균한다. 반환 프레임은 시드별 백분위(``pctile_seed<n>``)·그
    평균(``ensemble_pctile_mean``)·평균을 다시 순위로 본 최종값(``pred``)을
    다 담는다 — 최종 열만 트림하지 않는 것은 산출물에서 앙상블이 실제로
    어떻게 만들어졌는지 추적하기 위해서다.
    """
    seeds = sorted(seed_oofs)
    if len(seeds) < 2:
        raise ValueError("앙상블에는 시드가 2개 이상 필요합니다")

    base: pl.DataFrame | None = None
    pctile_cols: list[str] = []
    n_rows: int | None = None
    for seed in seeds:
        oof = seed_oofs[seed]
        if n_rows is None:
            n_rows = oof.height
        elif oof.height != n_rows:
            raise ValueError(
                f"시드 {seed}의 OOF 행수({oof.height})가 다른 시드와 다릅니다({n_rows}) "
                "— 시드마다 같은 fold·데이터를 써야 합니다"
            )
        pct_col = f"pctile_seed{seed}"
        pct = cross_sectional_percentile(oof, "pred", date_col="date", out_col=pct_col)
        pctile_cols.append(pct_col)
        sub = pct.select("date", "symbol", "month_idx", "fold_id", "L2", "y_rank", pct_col)
        base = (
            sub
            if base is None
            else base.join(
                sub.select("date", "symbol", pct_col), on=["date", "symbol"], how="inner"
            )
        )

    assert base is not None
    if base.height != n_rows:
        raise ValueError(
            "시드 간 조인 후 행수가 줄었습니다 — 시드마다 (date, symbol) 집합이 달랐습니다"
        )

    base = base.with_columns(
        pl.mean_horizontal([pl.col(c) for c in pctile_cols]).alias("ensemble_pctile_mean")
    )
    base = cross_sectional_percentile(base, "ensemble_pctile_mean", date_col="date", out_col="pred")
    return base


# --- 3. 결과 묶음 --------------------------------------------------------------


@dataclass
class EnsembleResult:
    model_id: str
    seeds: list[int]
    rank_ic_mean: float
    rank_ic_t_hac: float
    n_months_pooled: int
    fold_ics: list[float]
    fold_ic_sd: float
    seed_rank_ic_mean: dict[int, float]
    best_single_seed: int
    best_single_seed_ic: float
    oof: pl.DataFrame
    train_seconds: float

    def metrics_dict(self, *, comparison: dict) -> dict:
        return {
            "model_id": self.model_id,
            "seeds": self.seeds,
            "rank_ic_mean": self.rank_ic_mean,
            "rank_ic_t_hac": self.rank_ic_t_hac,
            "n_months_pooled": self.n_months_pooled,
            "fold_ic": self.fold_ics,
            "fold_ic_sd": self.fold_ic_sd,
            "seed_rank_ic_mean": {str(k): v for k, v in self.seed_rank_ic_mean.items()},
            "best_single_seed": self.best_single_seed,
            "best_single_seed_ic": self.best_single_seed_ic,
            "ensemble_minus_best_single_seed": self.rank_ic_mean - self.best_single_seed_ic,
            "ensemble_minus_seed_mean_ic": (
                self.rank_ic_mean - float(np.mean(list(self.seed_rank_ic_mean.values())))
            ),
            "train_seconds": self.train_seconds,
            "comparison": comparison,
        }


def fold_rank_ics(oof: pl.DataFrame, *, fold_col: str = "fold_id") -> list[float]:
    """``fold_col``별 rank IC — ``evaluate_oof``(``date``별 spearman(pred, L2)
    평균)를 fold마다 적용한다. ``m4_run.run_walk_forward_lgbm``이 fold 루프
    안에서 하던 것과 같은 계산을 OOF 프레임 하나에서 사후에 한다(``fold_id``
    컬럼이 이미 있으므로)."""
    fold_ids = sorted(oof[fold_col].unique().to_list())
    fold_ics: list[float] = []
    for fold_id in fold_ids:
        fold_df = oof.filter(pl.col(fold_col) == fold_id)
        fold_ic, _n = evaluate_oof(fold_df, group_col="date")
        fold_ics.append(fold_ic)
    return fold_ics


def run_ensemble(
    inputs: M4Inputs, folds: list[WfFold], *, seeds: tuple[int, ...] = ENSEMBLE_SEEDS
) -> tuple[EnsembleResult, dict[int, pl.DataFrame]]:
    t0 = time.monotonic()
    seed_oofs: dict[int, pl.DataFrame] = {}
    seed_ic_means: dict[int, float] = {}
    for seed in seeds:
        oof = run_mg_seed(inputs, folds, seed=seed)
        seed_oofs[seed] = oof
        ic_mean, _t, _n = _pooled_stats(oof)
        seed_ic_means[seed] = ic_mean

    ensemble_oof = build_ensemble_oof(seed_oofs)
    rank_ic_mean, rank_ic_t_hac, n_months_pooled = _pooled_stats(ensemble_oof)

    fold_ics = fold_rank_ics(ensemble_oof)
    finite_fold_ics = [v for v in fold_ics if np.isfinite(v)]
    fold_ic_sd = (
        float(np.std(finite_fold_ics, ddof=1)) if len(finite_fold_ics) > 1 else float("nan")
    )

    best_seed = max(seed_ic_means, key=lambda s: seed_ic_means[s])

    result = EnsembleResult(
        model_id="m5_ensemble",
        seeds=list(seeds),
        rank_ic_mean=rank_ic_mean,
        rank_ic_t_hac=rank_ic_t_hac,
        n_months_pooled=n_months_pooled,
        fold_ics=fold_ics,
        fold_ic_sd=fold_ic_sd,
        seed_rank_ic_mean=seed_ic_means,
        best_single_seed=best_seed,
        best_single_seed_ic=seed_ic_means[best_seed],
        oof=ensemble_oof,
        train_seconds=time.monotonic() - t0,
    )
    return result, seed_oofs


# --- 4. M4 산출물에서 M-L 비교 숫자 읽기 ----------------------------------------


def _run_dir_snapshot_suffix(dir_name: str, model_id: str) -> str | None:
    """``dir_name``이 정확히 ``<model_id>_<snapshot>``이면 ``<snapshot>``을,
    아니면 ``None``을 돌려준다.

    ``str.startswith(f"{model_id}_")``만 쓰면 ``model_id="m4_lgbm"``이
    ``m4_lgbm_on_l1_20260921``(대조군 run)에도 걸린다 — 접두어 충돌이다
    (2026-09-21 M5 첫 실행에서 실측: 정렬상 ``m4_lgbm_on_l1_...``이
    ``m4_lgbm_...``보다 뒤라 대조군 숫자가 M-G로 잘못 뽑혔다). snapshot이
    ``date.today().isoformat().replace('-', '')`` 형태(숫자만)라는 전제로
    나머지가 전부 숫자인지 확인해 걸러낸다.
    """
    prefix = f"{model_id}_"
    if not dir_name.startswith(prefix):
        return None
    suffix = dir_name[len(prefix) :]
    return suffix if suffix.isdigit() else None


def _latest_run_dir(root: DataRoot, model_id: str) -> Path | None:
    base = root.output / "model_runs"
    if not base.is_dir():
        return None
    candidates = sorted(
        (p for p in base.iterdir() if p.is_dir() and _run_dir_snapshot_suffix(p.name, model_id)),
        key=lambda p: _run_dir_snapshot_suffix(p.name, model_id),
    )
    return candidates[-1] if candidates else None


def load_m4_comparison(root: DataRoot) -> dict[str, dict]:
    """M4가 이미 만든 M-L(Ridge·ElasticNet)·M-G 단일 결과를 나란히 읽는다.

    M5는 M-L·M-G 그리드서치를 다시 돌리지 않는다(M5 지시 §7) — M4
    산출물(``m4_ridge_*``·``m4_enet_*``·``m4_lgbm_*``의 ``metrics.json``)을
    그대로 인용한다.
    """
    comparison: dict[str, dict] = {}
    for model_id in M4_COMPARISON_MODEL_IDS:
        run_dir = _latest_run_dir(root, model_id)
        if run_dir is None:
            comparison[model_id] = {"error": f"{model_id}_* run 디렉터리가 없습니다"}
            continue
        metrics = json.loads((run_dir / "metrics.json").read_text())
        comparison[model_id] = {
            "run_dir": str(run_dir),
            "selected_params": metrics.get("selected_params"),
            "rank_ic_mean": metrics.get("rank_ic_mean"),
            "rank_ic_t_hac": metrics.get("rank_ic_t_hac"),
            "fold_ic_sd": metrics.get("fold_ic_sd"),
        }
    return comparison


def adoption_verdict(ensemble_rank_ic: float, comparison: dict[str, dict]) -> dict:
    """``03`` §5 채택 규칙: "검증 fold에서 M-E가 M-L을 rank IC로 이기지
    못하면 M-L을 채택한다."

    M-L이 Ridge·ElasticNet 둘이라 계획에 둘 사이 우선순위가 없다(M5 지시
    §3) — 여기서는 어느 쪽도 임의로 고르지 않고, **M-E가 둘 다 이기는지·
    둘 다 지는지·엇갈리는지**만 기계적으로 판정한다. 최종 채택(Ridge vs
    ElasticNet)은 상위 세션의 몫이다.
    """
    beats: dict[str, bool | None] = {}
    for model_id in ("m4_ridge", "m4_enet"):
        entry = comparison.get(model_id, {})
        other_ic = entry.get("rank_ic_mean")
        beats[model_id] = (ensemble_rank_ic > other_ic) if other_ic is not None else None
    known = [v for v in beats.values() if v is not None]
    if not known:
        # all()/any()가 빈 리스트에서 각각 True/False라 값 없음을 "둘 다 이겼다"로
        # 잘못 읽을 수 있다 — 데이터가 없으면 그렇다고 명시한다.
        verdict = "판정 불가 — M4 비교 산출물(m4_ridge_*/m4_enet_*)을 읽지 못했습니다"
    elif all(known):
        verdict = "M-E 채택 — M-L(Ridge·ElasticNet) 둘 다 이겼다"
    elif not any(known):
        verdict = "M-L 채택 — M-E가 M-L(Ridge·ElasticNet) 둘 다 못 이겼다"
    else:
        verdict = "엇갈림 — M-E가 한쪽만 이겼다. 어느 M-L을 기준으로 볼지 상위 세션이 정한다"
    return {
        "beats_m4_ridge": beats["m4_ridge"],
        "beats_m4_enet": beats["m4_enet"],
        "verdict": verdict,
    }


# --- 5. 출력 -------------------------------------------------------------------


def write_run(
    root: DataRoot,
    result: EnsembleResult,
    *,
    inputs: M4Inputs,
    folds: list[WfFold],
    comparison: dict,
    adoption: dict,
    snapshot_date: str,
    modeler_git_commit: str,
) -> Path:
    run_id = f"m5_ensemble_{snapshot_date.replace('-', '')}"
    out_dir = root.output / "model_runs" / run_id
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    for fold_id in sorted(result.oof["fold_id"].unique().to_list()):
        fold_df = result.oof.filter(pl.col("fold_id") == fold_id)
        fold_df.write_parquet(pred_dir / f"fold_{fold_id}.parquet")

    config = {
        "run_id": run_id,
        "model_id": result.model_id,
        "method": "순위 단순 평균 — 시드별 그날 횡단면 백분위 평균 후 재순위(메타모델 없음)",
        "seeds": result.seeds,
        "mg_selected_params": MG_SELECTED_PARAMS.as_dict(),
        "mg_selected_params_source": (
            "M4 그리드서치 결과 재사용 — 재실행 없음(m4_lgbm_*/metrics.json selected_params)"
        ),
        "feature_cols": inputs.model_features,
        "target_col_per_seed_model": "y_rank",
        "eval_y_col": EVAL_Y_COL,
        "hac_lag": HAC_LAG,
        "min_names": MIN_NAMES,
        "folds": _fold_records(folds, inputs.core),
        "dev_start": inputs.core["date"].min().isoformat() if inputs.core.height else None,
        "dev_end": DEV_END.isoformat(),
        "m3_manifest_path": str(inputs.m3_manifest_path),
        "adoption_rule": (
            "03_model_candidates.md §5 — M-E가 검증 fold rank IC로 M-L을 못 이기면 M-L 채택"
        ),
        "adoption_verdict": adoption,
        "m4_comparison": comparison,
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
        json.dumps(
            result.metrics_dict(comparison=comparison),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    return out_dir


# --- 6. CLI -------------------------------------------------------------------


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

    print(f"M-E 재료: M-G 시드 {list(ENSEMBLE_SEEDS)}, 설정 {MG_SELECTED_PARAMS.as_dict()}")
    print("M-R·M-C는 돌리지 않는다 (M4 결과 — 03 §5).")

    result, _seed_oofs = run_ensemble(inputs, folds, seeds=ENSEMBLE_SEEDS)
    comparison = load_m4_comparison(root)
    adoption = adoption_verdict(result.rank_ic_mean, comparison)

    out_dir = write_run(
        root,
        result,
        inputs=inputs,
        folds=folds,
        comparison=comparison,
        adoption=adoption,
        snapshot_date=snapshot_date,
        modeler_git_commit=modeler_git_commit,
    )

    fold_ic_str = [f"{v:.4f}" for v in result.fold_ics]
    print(
        f"m5_ensemble: rank_ic_mean={result.rank_ic_mean:.4f} "
        f"t_hac={result.rank_ic_t_hac:.2f} fold_ic={fold_ic_str} "
        f"sd={result.fold_ic_sd:.4f} -> {out_dir}"
    )
    print(f"시드별 IC: {result.seed_rank_ic_mean}")
    print(
        f"단일 시드 최고: seed={result.best_single_seed} ic={result.best_single_seed_ic:.4f} "
        f"(앙상블 - 최고시드 = {result.rank_ic_mean - result.best_single_seed_ic:+.4f})"
    )
    print(f"채택 판정: {adoption['verdict']}")

    total_seconds = time.monotonic() - t_start
    print(f"총 소요 시간: {total_seconds:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
