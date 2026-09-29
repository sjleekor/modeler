"""시장·섹터 레이어 2 실행기: 피쳐 -> walk-forward 학습 -> OOF -> 점수 -> 평가 -> 산출물.

    python -m modeler.scores.market_sector.run --market us --panel-version ms_panel_v1 \\
        [--run-id ms_us_<yyyymmddHHMM>] [--smoke] [--build-features-only]

* ``--smoke``: 실제 데이터·모든 fold를 돌리되 ``output/market_sector/smoke_<ts>/``에만 쓴다.
* 산출물: ``stock_data/us/output/market_sector/<run_id>/{oof_predictions.parquet,
  scores.parquet, metrics.json, report.md, latest_scores.{json,csv}, models/, manifest.json}``.
* ``--market kr``: 입력은 ``build_panel --market kr``이 만든 KR 패널
  (``stock_data/kr/datasets/market_sector/<panel-version>/``)과 KR raw parquet
  (``USD/KRW``·외국인 순매수·거래대금)다. 거시는 US 일별 계열을 그대로 쓴다(US 레이크의 최신
  ``macro_series``, 08:30 KST 결정에 ``available_at``이 tz-aware라 그대로 맞는다).
  패널이나 KR 표가 없으면 ``KrNotSyncedError``로 멈춘다.
* ``raw/``·``derived/``에는 쓰지 않는다. 같은 입력·설정·시드는 같은 parquet 바이트를 낸다.

``run_pipeline``은 IO 없는 순수 함수라 합성 데이터로 테스트한다.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import lightgbm
import numpy as np
import polars as pl
import scipy
import sklearn

from modeler.etl.config import REPO_ROOT, DataRoot
from modeler.scores.common.cash import CASH_BASIS
from modeler.scores.common.inputs import PinnedScopedLake, sha256_file
from modeler.scores.common.kr_inputs import KrLake, KrNotSyncedError, load_kr_macro
from modeler.scores.market_sector import baselines as bl
from modeler.scores.market_sector.config import MsConfig
from modeler.scores.market_sector.evaluate import evaluate_all, render_report
from modeler.scores.market_sector.features import (
    build_features,
    feature_manifest,
    load_us_macro,
    model_feature_columns,
    write_features,
)
from modeler.scores.market_sector.models import (
    date_market_weights,
    predict,
    save_model,
    train_task,
    verify_reload,
)
from modeler.scores.market_sector.scoring import (
    OPPORTUNITY_REFERENCE_VERSION,
    STABILITY_CALIBRATION_METHOD,
    STABILITY_VALIDATION_STATUS,
    STATUS_NO_PREDICTION,
    opportunity_scores,
    percentile_against,
    stability_scores,
)
from modeler.scores.market_sector.splits import (
    Fold,
    expanding_annual_folds,
    live_fold,
)
from modeler.us.dataset import content_hash, git_commit

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_VERSION = "ms_feat_v1"
OPP_PRIMARY = "excess_return_60d_vs_cash"
OPP_FALLBACK = "total_return_60d"
OPP_FALLBACK_LABEL = "absolute_return_exploratory"

#: 예측 열 <- (타깃 열, 모델 종류)
TASKS: dict[str, tuple[str, str]] = {
    "p_opp_ridge": ("y_opp", "ridge"),
    "p_opp_lgbm": ("y_opp", "lgbm_reg"),
    "p_mkt_ridge": ("y_mkt", "ridge"),
    "p_mkt_lgbm": ("y_mkt", "lgbm_reg"),
    "p_stab_logit": ("y_loss", "logit"),
    "p_stab_lgbm": ("y_loss", "lgbm_clf"),
}
BASELINE_B = "b_stab_logit_rvol"


@dataclass
class RunResult:
    oof: pl.DataFrame
    scores: pl.DataFrame
    latest: pl.DataFrame
    metrics: dict[str, Any]
    fold_table: list[dict[str, Any]]
    opportunity_target: str
    opportunity_target_column: str
    checks: list[str] = field(default_factory=list)
    reload_verified: dict[str, bool] = field(default_factory=dict)
    models: dict[str, Any] = field(default_factory=dict)


def choose_opportunity_target(frame: pl.DataFrame) -> tuple[str, str]:
    """(라벨 열, 표시 이름). 현금 대비 라벨이 패널 전체에서 null이면 절대수익 탐색으로 표시."""
    if frame[OPP_PRIMARY].null_count() == frame.height:
        return OPP_FALLBACK, OPP_FALLBACK_LABEL
    return OPP_PRIMARY, OPP_PRIMARY


def _frame(cols: dict[str, Any]) -> pl.DataFrame:
    """numpy NaN을 null로 바꿔 DataFrame을 만든다(평가는 null로 결측을 판단한다)."""
    out = []
    for k, v in cols.items():
        if isinstance(v, pl.Series):
            out.append(v.alias(k))
        elif isinstance(v, np.ndarray) and v.dtype.kind == "f":
            out.append(pl.Series(k, v, dtype=pl.Float64, nan_to_null=True))
        elif isinstance(v, np.ndarray) and v.dtype == object:
            out.append(pl.Series(k, v.tolist(), dtype=pl.String))
        else:
            out.append(pl.Series(k, v))
    return pl.DataFrame(out)


def _x_matrix(frame: pl.DataFrame, cols: list[str]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float64) for c in cols]).to_numpy()


def _fold_row(fold: Fold, market: str, n_train_by: dict[str, int]) -> dict[str, Any]:
    return {
        "market": market.upper(),
        "year": fold.year,
        "fold_start": fold.fold_start,
        "n_train": fold.n_train,
        "n_test": fold.n_test,
        "n_test_matured": fold.n_test_matured,
        "first_decision": fold.first_decision,
        "last_decision": fold.last_decision,
        "skipped_reason": fold.skipped_reason,
        "n_train_by_task": n_train_by,
    }


def _fit_fold(
    fold: Fold,
    X: np.ndarray,
    Xb: np.ndarray,
    ys: dict[str, np.ndarray],
    arr: dict[str, np.ndarray],
    cfg: MsConfig,
    has_parent: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, int], dict[str, str | None]]:
    """fold 하나: 예측(test 행 길이), 모델, 태스크별 학습 행 수, 자산별 Stability 사유."""
    preds: dict[str, np.ndarray] = {}
    models: dict[str, Any] = {}
    n_used: dict[str, int] = {}
    test = fold.test_idx
    status = bl.train_event_status(arr["asset"], ys["y_loss"], fold.train_idx, cfg)
    for name, (ycol, kind) in TASKS.items():
        y = ys[ycol]
        use = fold.train_idx[~np.isnan(y[fold.train_idx])]
        if len(use) < cfg.min_train_rows:
            preds[name] = np.full(len(test), np.nan)
            n_used[name] = len(use)
            continue
        model, n = train_task(kind, X, y, fold.train_idx, arr["session"], arr["market"], cfg)
        p = predict(model, X[test], kind)
        if ycol == "y_mkt":
            p = np.where(has_parent[test] == 1, p, np.nan)
        preds[name], models[name], n_used[name] = p, (model, kind), n
    # baseline b: 자산 절편 + rvol_20
    y = ys["y_loss"]
    use = fold.train_idx[~np.isnan(y[fold.train_idx])]
    if len(use) >= cfg.min_train_rows:
        model, n = train_task("logit", Xb, y, fold.train_idx, arr["session"], arr["market"], cfg)
        preds[BASELINE_B] = predict(model, Xb[test], "logit")
        models[BASELINE_B] = (model, "logit")
        n_used[BASELINE_B] = n
    else:
        preds[BASELINE_B] = np.full(len(test), np.nan)
        n_used[BASELINE_B] = len(use)
    # Stability null 규칙: 사건 30개 미만·한 클래스인 자산은 모델 확률 null
    reasons = np.array([status.get(str(a)) for a in arr["asset"][test]], dtype=object)
    for name in ("p_stab_logit", "p_stab_lgbm", BASELINE_B):
        preds[name] = np.where(reasons == None, preds[name], np.nan)  # noqa: E711
    return preds, models, n_used, status


def run_pipeline(
    frame: pl.DataFrame,
    cfg: MsConfig,
    market: str,
    *,
    models_dir: Path | None = None,
    return_basis: str = "total_return",
    cash_basis: str = CASH_BASIS,
) -> RunResult:
    """``frame``: 피쳐 + 라벨을 (asset_id, session)으로 합친 프레임.

    필요 열: ``asset_id, session, decision_at, label_end_at, label_matured, feature_ready,
    has_parent``, 라벨 넷(``total_return_60d, excess_return_60d_vs_cash,
    excess_return_60d_vs_market, loss_event_60d_8pct``), ``model_feature_columns`` 전부.
    """
    market = market.upper()
    frame = frame.sort(["asset_id", "session"])
    asset_ids = sorted(frame["asset_id"].unique().to_list())
    opp_col, opp_name = choose_opportunity_target(frame)
    cols = model_feature_columns(asset_ids)
    X = _x_matrix(frame, cols)
    bcols = ["rvol_20", *[c for c in cols if c.startswith("asset_")]]
    Xb = _x_matrix(frame, bcols)
    ys = {
        "y_opp": frame[opp_col].cast(pl.Float64).to_numpy(),
        "y_mkt": frame["excess_return_60d_vs_market"].cast(pl.Float64).to_numpy(),
        "y_loss": frame["loss_event_60d_8pct"].cast(pl.Float64).to_numpy(),
    }
    n = frame.height
    arr = {
        "asset": frame["asset_id"].to_numpy(),
        "session": frame["session"].to_numpy(),
        "market": np.full(n, market, dtype=object),
    }
    has_parent = frame["has_parent"].cast(pl.Int8).to_numpy()
    dec_us, end_us = bl.epoch_us(frame["decision_at"]), bl.epoch_us(frame["label_end_at"])

    result_checks = [
        "pit: feature_available_at <= decision_at < entry_at (features.build_features)",
    ]
    folds = expanding_annual_folds(frame, market, cfg)
    preds_all = {k: np.full(n, np.nan) for k in (*TASKS, BASELINE_B)}
    fold_year = np.full(n, -1, dtype=np.int64)
    stab_reason = np.full(n, None, dtype=object)
    fold_table: list[dict[str, Any]] = []
    reload_verified: dict[str, bool] = {}
    kept_models: dict[str, Any] = {}
    for fold in folds:
        if fold.skipped_reason:
            fold_table.append(_fold_row(fold, market, {}))
            continue
        preds, models, n_used, status = _fit_fold(fold, X, Xb, ys, arr, cfg, has_parent)
        for k, v in preds.items():
            preds_all[k][fold.test_idx] = v
        fold_year[fold.test_idx] = int(fold.year)
        stab_reason[fold.test_idx] = [status.get(str(a)) for a in arr["asset"][fold.test_idx]]
        fold_table.append(_fold_row(fold, market, n_used))
        if models_dir is not None:
            reload_verified.update(
                _save_models(models, models_dir / f"fold_{fold.year}", X, Xb, fold.test_idx)
            )
        else:
            kept_models.update({f"fold_{fold.year}/{k}": v[0] for k, v in models.items()})
    result_checks.append(
        f"split: label_end_at < fold_start for every train row in {len(folds)} folds "
        "(splits.assert_train_boundary)"
    )

    # ---- baselines (PIT) -------------------------------------------------
    pool = (
        frame["label_matured"].fill_null(False)
        & frame["feature_ready"].fill_null(False)
        & (frame["session"] >= cfg.train_start[market])
    ).to_numpy()
    b_opp = bl.pit_expanding_mean(arr["asset"], dec_us, end_us, ys["y_opp"], pool)
    b_mkt = bl.pit_expanding_mean(arr["asset"], dec_us, end_us, ys["y_mkt"], pool)
    sm = bl.pit_smoothed_rate(
        arr["asset"], dec_us, end_us, ys["y_loss"], pool, cfg.baseline_smoothing_k
    )
    oof_mask = fold_year >= 0
    oi = np.flatnonzero(oof_mask)
    w = date_market_weights(arr["session"], arr["market"], oof_mask)
    oof = _frame(
        {
            "asset_id": arr["asset"][oi],
            "session": frame["session"].gather(oi.tolist()),
            "decision_at": frame["decision_at"].gather(oi.tolist()),
            "market": arr["market"][oi],
            "fold_year": fold_year[oi],
            "label_end_at": frame["label_end_at"].gather(oi.tolist()),
            "label_matured": frame["label_matured"].fill_null(False).gather(oi.tolist()),
            "has_parent": has_parent[oi],
            "weight_date_market": w[oi],
            "y_opp": ys["y_opp"][oi],
            "y_mkt": ys["y_mkt"][oi],
            "y_loss": ys["y_loss"][oi],
            **{k: v[oi] for k, v in preds_all.items()},
            "b_opp_mean": b_opp[oi],
            "b_mkt_mean": b_mkt[oi],
            "b_stab_asset_rate": sm["asset_rate"][oi],
            "b_stab_pooled": sm["pooled_rate"][oi],
            "stab_null_reason": stab_reason[oi],
        },
    ).sort(["session", "asset_id"])

    scores = build_scores(oof, cfg, opp_name, return_basis, cash_basis)
    metrics = evaluate_all(oof, cfg, opportunity_target=opp_name)
    result_checks.append("evaluate: block bootstrap resamples whole date blocks (all assets)")

    # ---- live: 가장 최근 결정 ------------------------------------------------
    latest = _live_scores(
        frame,
        X,
        Xb,
        ys,
        arr,
        cfg,
        market,
        has_parent,
        oof,
        opp_name,
        return_basis,
        cash_basis,
        models_dir,
        reload_verified,
        kept_models,
    )
    return RunResult(
        oof=oof,
        scores=scores,
        latest=latest,
        metrics=metrics,
        fold_table=fold_table,
        opportunity_target=opp_name,
        opportunity_target_column=opp_col,
        checks=result_checks,
        reload_verified=reload_verified,
        models=kept_models,
    )


def _save_models(
    models: dict[str, Any], out: Path, X: np.ndarray, Xb: np.ndarray, test_idx: np.ndarray
) -> dict[str, bool]:
    ver = {}
    probe = test_idx[:300] if len(test_idx) else np.arange(min(300, len(X)))
    for name, (model, kind) in models.items():
        path = out / f"{name}.joblib"
        save_model(model, path)
        ver[f"{out.name}/{name}"] = verify_reload(
            model, path, (Xb if name == BASELINE_B else X)[probe], kind
        )
    return ver


def build_scores(
    oof: pl.DataFrame, cfg: MsConfig, opp_name: str, return_basis: str, cash_basis: str
) -> pl.DataFrame:
    parts = []
    base = oof.select("asset_id", "session", "decision_at", "fold_year", "market")
    fy, mk = oof["fold_year"].to_numpy(), oof["market"].to_numpy()
    for model in ("ridge", "lgbm"):
        p = oof[f"p_opp_{model}"].to_numpy()
        sc, st, rn = opportunity_scores(p, fy, mk, cfg.opportunity_reference_min_oof)
        parts.append(
            base.with_columns(
                pl.lit("opportunity").alias("score_type"),
                pl.lit(f"opp_{model}").alias("model"),
                pl.Series("raw_prediction", p, nan_to_null=True),
                pl.Series("score", sc, nan_to_null=True),
                pl.Series("score_status", st, dtype=pl.String),
                pl.lit(None, dtype=pl.String).alias("null_reason"),
                pl.lit(OPPORTUNITY_REFERENCE_VERSION).alias("reference_version"),
                pl.Series("reference_n", rn),
                pl.lit("none").alias("calibration_method"),
                pl.lit("research").alias("validation_status"),
                pl.lit("percentile_of_expanding_oof_previous_folds").alias("score_basis"),
            )
        )
    for model in ("logit", "lgbm"):
        p = oof[f"p_stab_{model}"].to_numpy()
        sc = stability_scores(p)
        st = np.where(np.isnan(p), STATUS_NO_PREDICTION, "ok").astype(object)
        parts.append(
            base.with_columns(
                pl.lit("stability").alias("score_type"),
                pl.lit(f"stab_{model}").alias("model"),
                pl.Series("raw_prediction", p, nan_to_null=True),
                pl.Series("score", sc, nan_to_null=True),
                pl.Series("score_status", st, dtype=pl.String),
                oof["stab_null_reason"].alias("null_reason"),
                pl.lit(None, dtype=pl.String).alias("reference_version"),
                pl.lit(None, dtype=pl.Int64).alias("reference_n"),
                pl.lit(STABILITY_CALIBRATION_METHOD).alias("calibration_method"),
                pl.lit(STABILITY_VALIDATION_STATUS).alias("validation_status"),
                pl.lit("100*(1-p_hat_raw)").alias("score_basis"),
            )
        )
    return (
        pl.concat(parts)
        .with_columns(
            pl.lit(opp_name).alias("opportunity_target"),
            pl.lit(return_basis).alias("return_basis"),
            pl.lit(cash_basis).alias("cash_basis"),
        )
        .sort(["score_type", "model", "session", "asset_id"])
    )


def _live_scores(
    frame,
    X,
    Xb,
    ys,
    arr,
    cfg,
    market,
    has_parent,
    oof,
    opp_name,
    return_basis,
    cash_basis,
    models_dir,
    reload_verified,
    kept_models,
) -> pl.DataFrame:
    """가장 최근 결정 시각 경계로 학습해 자산별 마지막 행을 예측한다."""
    fold = live_fold(frame, market, cfg)
    if fold.skipped_reason or fold.n_test == 0:
        return pl.DataFrame()
    preds, models, n_used, status = _fit_fold(fold, X, Xb, ys, arr, cfg, has_parent)
    if models_dir is not None:
        reload_verified.update(_save_models(models, models_dir / "live", X, Xb, fold.test_idx))
    else:
        kept_models.update({f"live/{k}": v[0] for k, v in models.items()})
    t = fold.test_idx
    rows = frame.select("asset_id", "session", "decision_at").gather(t.tolist())
    rows = rows.with_columns(
        pl.lit(market).alias("market"),
        pl.lit(fold.fold_start).alias("model_train_boundary_label_end_before"),
        pl.lit(fold.n_train).alias("n_train"),
    )
    cols: dict[str, Any] = {}
    for model in ("ridge", "lgbm"):
        p = preds[f"p_opp_{model}"]
        ref = oof.filter(pl.col("market") == market)[f"p_opp_{model}"].to_numpy()
        sc = percentile_against(ref, p, cfg.opportunity_reference_min_oof)
        cols[f"opp_{model}_raw"] = p
        cols[f"opp_{model}_score"] = sc
        cols[f"opp_{model}_status"] = np.where(
            np.isnan(p), STATUS_NO_PREDICTION, np.where(np.isnan(sc), "warmup", "ok")
        ).astype(object)
    for model in ("logit", "lgbm"):
        p = preds[f"p_stab_{model}"]
        cols[f"stab_{model}_p_hat"] = p
        cols[f"stab_{model}_score"] = stability_scores(p)
    for model in ("ridge", "lgbm"):
        cols[f"mkt_{model}_raw"] = preds[f"p_mkt_{model}"]
    reasons = [status.get(str(a)) for a in arr["asset"][t]]
    out = rows.with_columns(
        [
            pl.Series(k, v, dtype=pl.String if v.dtype == object else pl.Float64)
            for k, v in cols.items()
        ]
    ).with_columns(
        pl.Series("stab_null_reason", reasons, dtype=pl.String),
        pl.lit(OPPORTUNITY_REFERENCE_VERSION).alias("opportunity_reference_version"),
        pl.lit(int(oof.filter(pl.col("market") == market).height)).alias("opportunity_reference_n"),
        pl.lit(STABILITY_CALIBRATION_METHOD).alias("calibration_method"),
        pl.lit(STABILITY_VALIDATION_STATUS).alias("validation_status"),
        pl.lit(opp_name).alias("opportunity_target"),
        pl.lit(return_basis).alias("return_basis"),
        pl.lit(cash_basis).alias("cash_basis"),
    )
    return out.sort("asset_id")


# --------------------------------------------------------------------------- IO
def _sha_dir(files: list[Path]) -> dict[str, str]:
    return {f.name: sha256_file(f) for f in files}


def _json(o: Any) -> str:
    def d(x: Any) -> Any:
        if hasattr(x, "isoformat"):
            return x.isoformat()
        if isinstance(x, np.generic):
            return x.item()
        raise TypeError(type(x))

    return json.dumps(o, indent=2, ensure_ascii=False, sort_keys=True, default=d) + "\n"


def write_outputs(res: RunResult, out: Path) -> dict[str, str]:
    """결정적 parquet + json/csv/md. parquet sha256을 돌려준다."""
    out.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, df in (("oof_predictions", res.oof), ("scores", res.scores)):
        p = out / f"{name}.parquet"
        df.write_parquet(p, compression="zstd", statistics=True)
        hashes[f"{name}.parquet"] = sha256_file(p)
    (out / "metrics.json").write_text(_json(res.metrics))
    if res.latest.height:
        res.latest.write_csv(out / "latest_scores.csv")
        (out / "latest_scores.json").write_text(_json(res.latest.to_dicts()))
    else:
        (out / "latest_scores.json").write_text("[]\n")
        (out / "latest_scores.csv").write_text("")
    return hashes


def _versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "scipy": scipy.__version__,
        "joblib": joblib.__version__,
    }


def _load_or_build_features(
    root: DataRoot,
    panel: pl.DataFrame,
    panel_manifest: dict[str, Any],
    cfg: MsConfig,
    feature_version: str,
    *,
    run_dir: Path | None,
    allow_dirty_commit: str,
    build_only: bool,
    smoke: bool = False,
    market: str = "US",
) -> tuple[pl.DataFrame, dict[str, Any], Path]:
    fdir = root.datasets / "market_sector" / feature_version
    if fdir.is_dir() and (fdir / "features.parquet").is_file():
        man = json.loads((fdir / "manifest.json").read_text())
        if man["feature_config_hash"] != cfg.feature_config_hash():
            raise RuntimeError(
                f"{fdir} 의 피쳐 설정이 현재 설정과 다릅니다. 새 버전 이름을 쓰십시오."
            )
        if man["panel_manifest_sha256"] != panel_manifest["_sha256"]:
            raise RuntimeError(f"{fdir} 는 다른 패널로 만든 것입니다. 새 버전 이름을 쓰십시오.")
        return pl.read_parquet(fdir / "features.parquet"), man, fdir
    market = market.upper()
    extra_inputs: dict[str, Any] = {}
    kr_macro = None
    if market == "KR":
        kr_snap = panel_manifest["inputs"]["common_feature_observation_raw"]["snapshot_date"]
        kr_lake = KrLake.resolve(root, snapshot_date=kr_snap)
        kr_macro = load_kr_macro(kr_lake, sorted(panel["session"].unique().to_list()))
        us_lake = PinnedScopedLake(root=DataRoot.resolve("us"), snapshots={}, symbols=())
        snap = us_lake.latest_snapshot("macro_series").isoformat()
        us_lake = PinnedScopedLake(root=us_lake.root, snapshots={"macro_series": snap}, symbols=())
        macro = load_us_macro(us_lake)
        extra_inputs = {
            "kr_raw_snapshot_date": kr_snap,
            "kr_raw_files": panel_manifest["inputs"]["common_feature_observation_raw"]["files"],
            "kr_macro_series": ["fx_usdkrw_ecos", "foreign_net_kospi_ecos", "trdval_kospi_ecos"],
            "kr_available_at_basis": "available_from_date_0830_kst",
            "price_available_at_basis": "session_close_plus_60min",
        }
        macro_files = {
            f.name: sha256_file(f) for f in us_lake.input_files(("macro_series",))["macro_series"]
        }
    else:
        snap = panel_manifest["inputs"]["macro_series"]["snapshot_date"]
        lake = PinnedScopedLake(root=root, snapshots={"macro_series": snap}, symbols=())
        macro = load_us_macro(lake)
        macro_files = panel_manifest["inputs"]["macro_series"]["files"]
    f = build_features(panel, macro, cfg, market=market, kr_macro=kr_macro)
    asset_ids = sorted(f["asset_id"].unique().to_list())
    man = feature_manifest(
        f,
        cfg,
        asset_ids=asset_ids,
        extra={
            "dataset": feature_version,
            "market": market,
            "panel_version": panel_manifest["dataset"],
            "panel_manifest_sha256": panel_manifest["_sha256"],
            "panel_content_hash": panel_manifest["outputs"]["panel_content_hash"],
            "macro_series_snapshot_date": snap,
            "macro_series_files": macro_files,
            "macro_series_ids": sorted(macro["series_id"].unique().to_list()),
            "modeler_git_commit": allow_dirty_commit,
            "available_at_rule": "feature_available_at <= decision_at < entry_at asserted",
            **extra_inputs,
        },
    )
    # smoke는 데이터셋 디렉터리를 만들지 않고 run 안에 둔다
    target = run_dir / "features" if (smoke and not build_only and run_dir) else fdir
    write_features(f, target, man)
    return f, man, target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--market", choices=["us", "kr"], required=True)
    ap.add_argument("--panel-version", default="ms_panel_v1")
    ap.add_argument("--feature-version", default=DEFAULT_FEATURE_VERSION)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--build-features-only", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    market = args.market.upper()
    try:
        root = DataRoot.resolve(args.market)
    except RuntimeError as exc:
        if market == "KR":
            raise KrNotSyncedError(str(exc)) from exc
        raise
    cfg = MsConfig()
    pdir = root.datasets / "market_sector" / args.panel_version
    if market == "KR" and not (pdir / "manifest.json").is_file():
        raise KrNotSyncedError(
            f"KR 패널이 없습니다 ({pdir}). 먼저 "
            "`python -m modeler.scores.market_sector.build_panel --market kr ...`를 돌리십시오 "
            "(KR raw parquet가 sync돼 있어야 합니다)."
        )
    panel_manifest = json.loads((pdir / "manifest.json").read_text())
    panel_manifest["_sha256"] = sha256_file(pdir / "manifest.json")
    for fn in ("panel.parquet", "labels.parquet"):
        if sha256_file(pdir / fn) != panel_manifest["outputs"][fn]:
            raise RuntimeError(f"{pdir / fn} 가 manifest 해시와 다릅니다")
    panel = pl.read_parquet(pdir / "panel.parquet")
    labels = pl.read_parquet(pdir / "labels.parquet")
    stamp = datetime.now().strftime("%Y%m%d%H%M")
    run_id = args.run_id or (f"smoke_{stamp}" if args.smoke else f"ms_{args.market}_{stamp}")
    if args.smoke and not run_id.startswith("smoke_"):
        run_id = f"smoke_{run_id}"
    out_dir = root.output / "market_sector" / run_id
    commit = git_commit(REPO_ROOT, allow_dirty=True)

    if args.build_features_only:
        _, fman, fdir = _load_or_build_features(
            root,
            panel,
            panel_manifest,
            cfg,
            args.feature_version,
            run_dir=None,
            allow_dirty_commit=commit,
            build_only=True,
            market=market,
        )
        print(f"features: {fdir}  rows={fman['rows']} ready={fman['feature_ready_rows']}")
        return 0
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} 가 이미 있습니다. --run-id를 바꾸십시오.")
    out_dir.mkdir(parents=True)

    feats, fman, fdir = _load_or_build_features(
        root,
        panel,
        panel_manifest,
        cfg,
        args.feature_version,
        run_dir=out_dir,
        allow_dirty_commit=commit,
        build_only=False,
        smoke=args.smoke,
        market=market,
    )
    t_feat = time.time()
    lab_cols = [
        "asset_id",
        "session",
        "label_end_at",
        "label_matured",
        "total_return_60d",
        "excess_return_60d_vs_cash",
        "excess_return_60d_vs_market",
        "loss_event_60d_8pct",
    ]
    frame = feats.join(labels.select(lab_cols), on=["asset_id", "session"], how="left")
    if frame.height != feats.height:
        raise AssertionError("피쳐-라벨 조인에서 행 수가 달라졌습니다")
    return_basis = "+".join(sorted(panel["return_basis"].unique().to_list()))
    cash_present = bool(panel_manifest["cash"]["series_present_in_lake"])
    res = run_pipeline(
        frame,
        cfg,
        market,
        models_dir=out_dir / "models",
        return_basis=return_basis,
        cash_basis=CASH_BASIS if cash_present else "none_cash_series_missing",
    )
    t_fit = time.time()
    hashes = write_outputs(res, out_dir)
    header = {
        "run_id": run_id,
        "market": market,
        "panel_version": args.panel_version,
        "smoke": args.smoke,
        "return_basis": return_basis,
        "cash_basis": CASH_BASIS if cash_present else "none_cash_series_missing",
    }
    (out_dir / "report.md").write_text(render_report(res.metrics, header, _fold_table_json(res)))
    fold_json = _fold_table_json(res)
    manifest = {
        "run_id": run_id,
        "smoke": args.smoke,
        "market": market,
        "layer": "market_sector_layer2_ms1",
        "created_at": datetime.now().astimezone().isoformat(),
        "config": cfg.as_dict(),
        "config_hash": cfg.config_hash(),
        "modeler_git_commit": commit.removesuffix("-dirty"),
        "git_dirty": commit.endswith("-dirty"),
        "seeds": {"bootstrap": cfg.bootstrap_seed, "lightgbm": cfg.lgbm_random_state},
        "package_versions": _versions(),
        "opportunity_target": res.opportunity_target,
        "opportunity_target_column": res.opportunity_target_column,
        "return_basis": return_basis,
        "cash_basis": header["cash_basis"],
        "panel": {
            "version": args.panel_version,
            "manifest_sha256": panel_manifest["_sha256"],
            "panel_parquet_sha256": panel_manifest["outputs"]["panel.parquet"],
            "labels_parquet_sha256": panel_manifest["outputs"]["labels.parquet"],
            "panel_content_hash": panel_manifest["outputs"]["panel_content_hash"],
            "labels_content_hash": panel_manifest["outputs"]["labels_content_hash"],
        },
        "features": {
            "dir": str(fdir),
            "manifest_sha256": sha256_file(fdir / "manifest.json"),
            "features_parquet_sha256": sha256_file(fdir / "features.parquet"),
            "features_content_hash": fman["outputs"]["features_content_hash"],
        },
        "outputs": hashes,
        "content_hashes": {
            "oof": content_hash(res.oof),
            "scores": content_hash(res.scores),
        },
        "folds": fold_json,
        "checks_passed": res.checks,
        "model_reload_equal": all(res.reload_verified.values()) if res.reload_verified else None,
        "model_reload_checked": len(res.reload_verified),
        "runtime_seconds": {
            "features_load_or_build": round(t_feat - t0, 2),
            "fit_score_evaluate": round(t_fit - t_feat, 2),
            "total": round(time.time() - t0, 2),
        },
    }
    (out_dir / "manifest.json").write_text(_json(manifest))
    print(f"run_id={run_id} out={out_dir}")
    print("folds:")
    for r in fold_json:
        print(
            f"  {r['year']}: n_train={r['n_train']} n_test={r['n_test']} "
            f"matured={r['n_test_matured']} {str(r['first_decision'])[:10]}.."
            f"{str(r['last_decision'])[:10]} {r.get('skipped_reason') or ''}"
        )
    print("hashes:", json.dumps(hashes))
    print("reload_equal:", manifest["model_reload_equal"], manifest["model_reload_checked"])
    print("runtime:", manifest["runtime_seconds"])
    return 0


def _fold_table_json(res: RunResult) -> list[dict[str, Any]]:
    return [{k: (v if k != "fold_start" else v) for k, v in r.items()} for r in res.fold_table]


if __name__ == "__main__":
    raise SystemExit(main())
