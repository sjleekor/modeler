"""모델: Ridge(Opportunity) · L2 Logistic(Stability) · 고정 LightGBM 한 후보.

* 전처리(중앙값 대치 -> 표준화)는 **학습 행으로만** 맞춘다(``Pipeline.fit``). 대치는
  ``StandardScaler`` 앞이라 학습 창 중앙값이다. null이 많은 거시 피쳐(초기 vintage 없음)를
  조용히 0으로 두지 않는다.
* 표본 가중치: 날짜마다 시장별 총 가중치 1, 시장 안 자산 균등(``date_market_weights``).
  학습에서는 평균 1로 정규화한다(규제 세기가 표본 수에 안 끌리게).
* LightGBM: ``MsConfig``의 고정 후보 하나. early stopping·탐색 없음.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeler.scores.market_sector.config import MsConfig

KINDS = ("ridge", "logit", "lgbm_reg", "lgbm_clf")


def date_market_weights(
    session: np.ndarray, market: np.ndarray, mask: np.ndarray, *, normalize_mean: bool = False
) -> np.ndarray:
    """``mask`` 행의 가중치: 날짜·시장별 ``1/(그날 그 시장의 자산 수)``. 나머지 행은 0.

    날짜별·시장별 합이 1이다(``normalize_mean=True``이면 마스크 행 평균이 1이 되게 곱한다).
    """
    n = len(mask)
    df = pl.DataFrame(
        {"i": np.arange(n), "s": session, "m": market, "u": mask.astype(bool)}
    ).filter(pl.col("u"))
    w = np.zeros(n)
    if df.height:
        df = df.with_columns((1.0 / pl.len().over(["s", "m"])).alias("w"))
        w[df["i"].to_numpy()] = df["w"].to_numpy()
        if normalize_mean:
            w[mask] = w[mask] / w[mask].mean()
    return w


def build_model(kind: str, cfg: MsConfig):
    if kind == "ridge":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=cfg.ridge_alpha)),
            ]
        )
    if kind == "logit":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=cfg.logit_C,
                        l1_ratio=0.0,
                        solver="lbfgs",
                        class_weight=None,
                        max_iter=cfg.logit_max_iter,
                    ),
                ),
            ]
        )
    if kind == "lgbm_reg":
        return LGBMRegressor(**cfg.lgbm_params())
    if kind == "lgbm_clf":
        return LGBMClassifier(**cfg.lgbm_params())
    raise KeyError(kind)


def fit_model(kind: str, X: np.ndarray, y: np.ndarray, w: np.ndarray, cfg: MsConfig):
    m = build_model(kind, cfg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(m, Pipeline):
            m.fit(X, y, model__sample_weight=w)
        else:
            m.fit(X, y, sample_weight=w)
    return m


def predict(model, X: np.ndarray, kind: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind in ("logit", "lgbm_clf"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X)


def train_task(
    kind: str,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    session: np.ndarray,
    market: np.ndarray,
    cfg: MsConfig,
):
    """``train_idx`` 중 ``y``가 있는 행으로 학습. 반환: (모델, 사용한 행 수)."""
    use = train_idx[~np.isnan(y[train_idx])]
    mask = np.zeros(len(y), dtype=bool)
    mask[use] = True
    if cfg.market_weight_equal:
        w = date_market_weights(session, market, mask, normalize_mean=True)[use]
    else:
        w = np.ones(len(use))
    return fit_model(kind, X[use], y[use], w, cfg), len(use)


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)


def verify_reload(model: Any, path: Path, X: np.ndarray, kind: str) -> bool:
    """저장 -> 재로딩한 모델이 같은 예측을 내는지(정확히 같아야 한다)."""
    loaded = joblib.load(path)
    return bool(np.array_equal(predict(model, X, kind), predict(loaded, X, kind), equal_nan=True))
