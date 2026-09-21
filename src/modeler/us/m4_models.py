"""M4 모델 — ``03_model_candidates.md`` §4.

다섯 모델 다 같은 인터페이스로 다룬다: 후보 하이퍼파라미터 하나를 받아
``(x_train, y_train)``에 학습하고 ``x_valid``에 예측값을 낸다. 순위·IC
계산은 이 모듈이 하지 않는다 — ``m4_run.py``가 예측값을 받아 한다(``scan.py``의
``monthly_rank_ic``를 그대로 재사용한다).

**LightGBM 하이퍼파라미터 이름.** ``min_data_in_leaf``는 LightGBM 코어
파라미터고, sklearn 래퍼(``LGBMRegressor``)는 ``**kwargs``로 받은 별칭을
그대로 Booster에 전달한다 — ``min_child_samples``로 바꿔 쓸 필요가 없다
(2026-09-21 로컬 lightgbm 4.7.0에서 확인).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge

# --- 1. 선형 바닥선(M-L) ------------------------------------------------------

#: Ridge alpha 그리드 — ``03`` §4.
RIDGE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

#: ElasticNet l1_ratio 그리드 — ``03`` §4.
ENET_L1_RATIOS: tuple[float, ...] = (0.1, 0.5, 0.9)

#: ElasticNet alpha 그리드. ``03``·M4 지시 어디에도 ENet alpha 숫자가 없다
#: (``l1_ratio``만 사전등록됐다) — 그래서 Ridge와 같은 척도를 쓸 수 없는 이유를
#: 확인하고 여기서 정한다.
#:
#: sklearn ``Ridge``는 ``||y-Xw||^2 + alpha*||w||^2``(표본 크기로 나누지
#: 않음)를, ``ElasticNet``은 ``(1/2n)*||y-Xw||^2 + alpha*l1_ratio*||w||_1 +
#: 0.5*alpha*(1-l1_ratio)*||w||^2``(표본 평균 손실)를 최소화한다 — **같은
#: ``alpha`` 숫자가 두 모델에서 전혀 다른 세기가 된다.** 2026-09-21 fold 1
#: 학습 데이터(``n≈66,645``)로 직접 확인: Ridge는 ``alpha=100``에서도 12개
#: 계수가 전부 살아 있는데, ElasticNet은 ``alpha=0.1``(``03``이 Ridge에 준
#: 그리드의 최솟값)에서 **이미 계수 12개가 전부 0**이 된다(``l1_ratio``와
#: 무관하게) — 예측이 fold 전체에서 상수가 되어 그 fold의 rank IC가 정의되지
#: 않는다(NaN). ``alpha`` 그리드를 그대로 재사용하면 ElasticNet 12개 후보가
#: 전부 이 상태였다(``m4_run.select_best`` 로그로 확인).
#:
#: 대신 같은 데이터에서 계수가 아직 죽지 않는 구간(``alpha<=1e-2``, Ridge의
#: 유효 계수와 거의 같아지는 ``alpha<=1e-4``까지)을 span하는 네 값을 쓴다 —
#: "완전 축소"부터 "거의 무규제"까지 실제로 다른 결과를 내는 그리드다.
ENET_ALPHAS: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2)

#: ElasticNet이 수렴하도록 ``max_iter``를 넉넉히 둔다(한국 ``_01`` train.py와
#: 같은 값, 5000) — 그리드 자체(alpha·l1_ratio)와 무관한 구현 상수라 ``03``에
#: 없다.
ENET_MAX_ITER = 5000


def ridge_grid(alphas: tuple[float, ...] = RIDGE_ALPHAS) -> list[dict]:
    return [{"alpha": a} for a in alphas]


def elasticnet_grid(
    alphas: tuple[float, ...] = ENET_ALPHAS, l1_ratios: tuple[float, ...] = ENET_L1_RATIOS
) -> list[dict]:
    """alpha × l1_ratio 전체 조합 — 그리드 구조는 한국 ``_01`` ``train.py``의
    ``_param_grid``와 같다("Ridge/ENet 설정 그대로다", ``03`` §4). ``alpha``
    값 자체는 ``ENET_ALPHAS`` 참고(Ridge와 다른 척도가 필요한 이유가 거기 있다)."""
    return [{"alpha": a, "l1_ratio": r} for a in alphas for r in l1_ratios]


def fit_predict_ridge(
    params: dict, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray
) -> np.ndarray:
    model = Ridge(alpha=params["alpha"])
    model.fit(x_train, y_train)
    return model.predict(x_valid)


def fit_predict_elasticnet(
    params: dict, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray
) -> np.ndarray:
    model = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=ENET_MAX_ITER)
    model.fit(x_train, y_train)
    return model.predict(x_valid)


def fit_predict_ols(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray) -> np.ndarray:
    """OLS-3(GKX null model) — 정규화 없는 순수 OLS. 하이퍼파라미터가 없다."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model.predict(x_valid)


# --- 2. LightGBM 회귀(M-G) ----------------------------------------------------

#: 그리드 24점 — ``03`` §4: num_leaves × learning_rate × min_data_in_leaf × feature_fraction.
LGBM_NUM_LEAVES: tuple[int, ...] = (15, 31, 63)
LGBM_LEARNING_RATES: tuple[float, ...] = (0.03, 0.1)
LGBM_MIN_DATA_IN_LEAF: tuple[int, ...] = (200, 500)
LGBM_FEATURE_FRACTIONS: tuple[float, ...] = (0.7, 1.0)

#: ``n_estimators`` 상한과 early stopping — ``03`` §4 "최대 1,000, early stopping".
#: stopping_rounds 자체는 ``03``에 숫자가 없어 여기서 정한다(문서에 없는 값은
#: 정할 때 근거를 남긴다는 관례, ``04`` §7.8과 같은 자리) — 최대 라운드의 5%인
#: 50으로 둔다. 너무 짧으면(예: 10) 학습률 0.03 그리드에서 곡선이 아직 개선
#: 중인데 멈출 위험이 있고, 너무 길면(예: 200) early stopping의 의미가
#: 옅어진다.
LGBM_MAX_ESTIMATORS = 1000
LGBM_EARLY_STOPPING_ROUNDS = 50

#: 시드 5개 — 검증용이 아니라 앙상블 재료(``03`` §4), M4는 이걸로 시드 분산만 본다.
LGBM_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)


@dataclass(frozen=True)
class LgbmParams:
    num_leaves: int
    learning_rate: float
    min_data_in_leaf: int
    feature_fraction: float

    def as_dict(self) -> dict:
        return {
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
        }


def lgbm_grid() -> list[LgbmParams]:
    """24점 그리드. 순서는 ``03`` §4가 나열한 순서(num_leaves가 가장 바깥)."""
    combos = itertools.product(
        LGBM_NUM_LEAVES, LGBM_LEARNING_RATES, LGBM_MIN_DATA_IN_LEAF, LGBM_FEATURE_FRACTIONS
    )
    return [
        LgbmParams(num_leaves=nl, learning_rate=lr, min_data_in_leaf=mdl, feature_fraction=ff)
        for nl, lr, mdl, ff in combos
    ]


assert len(lgbm_grid()) == 24, "LightGBM 그리드는 24점이어야 한다"


def fit_predict_lgbm(
    params: LgbmParams,
    *,
    x_es_train: np.ndarray,
    y_es_train: np.ndarray,
    x_es_valid: np.ndarray,
    y_es_valid: np.ndarray,
    x_valid: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, int]:
    """early stopping용 내부 분할(``x_es_train``/``x_es_valid``, fold 학습 구간의
    마지막 12개월)로 라운드를 고른 뒤, ``x_valid``(테스트 fold)에 예측한다.

    반환: (예측값, 고른 부스팅 라운드 수 ``best_iteration_``).
    """
    model = lgb.LGBMRegressor(
        objective="regression",
        num_leaves=params.num_leaves,
        learning_rate=params.learning_rate,
        min_data_in_leaf=params.min_data_in_leaf,
        feature_fraction=params.feature_fraction,
        n_estimators=LGBM_MAX_ESTIMATORS,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(
        x_es_train,
        y_es_train,
        eval_X=x_es_valid,
        eval_y=y_es_valid,
        callbacks=[lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    preds = model.predict(x_valid)
    best_iteration = int(model.best_iteration_) if model.best_iteration_ else LGBM_MAX_ESTIMATORS
    return preds, best_iteration
