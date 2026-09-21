"""``modeler.us.m4_models`` 단위 테스트 — 전부 합성 데이터다.

그리드 크기·값과 LightGBM early stopping이 테스트 fold를 보지 않는지를
검사한다(``06_execution_steps.md`` M4 §7 지시).
"""

from __future__ import annotations

import numpy as np
import pytest

from modeler.us.m4_models import (
    ENET_ALPHAS,
    ENET_L1_RATIOS,
    LGBM_FEATURE_FRACTIONS,
    LGBM_LEARNING_RATES,
    LGBM_MIN_DATA_IN_LEAF,
    LGBM_NUM_LEAVES,
    RIDGE_ALPHAS,
    LgbmParams,
    elasticnet_grid,
    fit_predict_lgbm,
    fit_predict_ols,
    fit_predict_ridge,
    lgbm_grid,
    ridge_grid,
)

# --- 1. 그리드 크기·값 ---------------------------------------------------------


def test_ridge_grid_matches_registered_alphas() -> None:
    grid = ridge_grid()
    assert [g["alpha"] for g in grid] == list(RIDGE_ALPHAS)
    assert len(grid) == 4


def test_elasticnet_grid_is_full_cross_product() -> None:
    grid = elasticnet_grid()
    assert len(grid) == len(ENET_ALPHAS) * len(ENET_L1_RATIOS)
    pairs = {(g["alpha"], g["l1_ratio"]) for g in grid}
    assert pairs == {(a, r) for a in ENET_ALPHAS for r in ENET_L1_RATIOS}


def test_lgbm_grid_has_24_points() -> None:
    grid = lgbm_grid()
    assert len(grid) == 24
    expected = len(LGBM_NUM_LEAVES) * len(LGBM_LEARNING_RATES)
    expected *= len(LGBM_MIN_DATA_IN_LEAF) * len(LGBM_FEATURE_FRACTIONS)
    assert expected == 24


def test_lgbm_grid_values_are_from_registered_sets() -> None:
    for params in lgbm_grid():
        assert params.num_leaves in LGBM_NUM_LEAVES
        assert params.learning_rate in LGBM_LEARNING_RATES
        assert params.min_data_in_leaf in LGBM_MIN_DATA_IN_LEAF
        assert params.feature_fraction in LGBM_FEATURE_FRACTIONS


def test_lgbm_grid_has_no_duplicates() -> None:
    grid = lgbm_grid()
    as_tuples = {
        (p.num_leaves, p.learning_rate, p.min_data_in_leaf, p.feature_fraction) for p in grid
    }
    assert len(as_tuples) == 24


# --- 2. 선형 모델 fit/predict 형태 -----------------------------------------------


def _toy_xy(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    y = 0.5 * x[:, 0] + rng.normal(scale=0.01, size=n)
    return x, y


def test_fit_predict_ridge_returns_correct_shape() -> None:
    x, y = _toy_xy()
    x_valid, _ = _toy_xy(n=10, seed=1)
    preds = fit_predict_ridge({"alpha": 1.0}, x, y, x_valid)
    assert preds.shape == (10,)


def test_fit_predict_ols_recovers_strong_signal() -> None:
    x, y = _toy_xy(n=2000)
    preds = fit_predict_ols(x, y, x)
    corr = np.corrcoef(preds, y)[0, 1]
    assert corr > 0.9  # 강한 신호(x[:,0]이 y를 거의 결정)를 OLS가 잡아야 한다


# --- 3. LightGBM early stopping이 테스트 fold를 보지 않는다 ---------------------


def test_lgbm_early_stopping_ignores_x_valid_content() -> None:
    """``best_iteration_``은 ``x_es_valid``/``y_es_valid``만으로 정해져야 한다.

    ``x_valid``(테스트 fold)를 바꿔도 고른 라운드 수가 같아야 "테스트 fold를
    안 본다"(``05_validation_protocol.md`` §2)가 지켜진다."""
    rng = np.random.default_rng(0)
    x_es_train = rng.normal(size=(300, 4))
    y_es_train = 0.4 * x_es_train[:, 0] + rng.normal(scale=0.05, size=300)
    x_es_valid = rng.normal(size=(60, 4))
    y_es_valid = 0.4 * x_es_valid[:, 0] + rng.normal(scale=0.05, size=60)

    params = LgbmParams(num_leaves=15, learning_rate=0.1, min_data_in_leaf=20, feature_fraction=1.0)

    x_valid_a = rng.normal(size=(20, 4))
    x_valid_b = rng.normal(size=(20, 4)) * 1000.0 + 500.0  # 완전히 다른 분포

    _preds_a, best_iter_a = fit_predict_lgbm(
        params,
        x_es_train=x_es_train,
        y_es_train=y_es_train,
        x_es_valid=x_es_valid,
        y_es_valid=y_es_valid,
        x_valid=x_valid_a,
        seed=0,
    )
    _preds_b, best_iter_b = fit_predict_lgbm(
        params,
        x_es_train=x_es_train,
        y_es_train=y_es_train,
        x_es_valid=x_es_valid,
        y_es_valid=y_es_valid,
        x_valid=x_valid_b,
        seed=0,
    )
    assert best_iter_a == best_iter_b


def test_lgbm_seed_changes_predictions() -> None:
    """시드가 다르면 예측이 달라져야 한다 — 시드 분산을 재는 전제 조건."""
    rng = np.random.default_rng(1)
    x_es_train = rng.normal(size=(300, 4))
    y_es_train = 0.4 * x_es_train[:, 0] + rng.normal(scale=0.2, size=300)
    x_es_valid = rng.normal(size=(60, 4))
    y_es_valid = 0.4 * x_es_valid[:, 0] + rng.normal(scale=0.2, size=60)
    x_valid = rng.normal(size=(20, 4))
    params = LgbmParams(num_leaves=31, learning_rate=0.1, min_data_in_leaf=20, feature_fraction=0.7)

    preds_seed0, _ = fit_predict_lgbm(
        params,
        x_es_train=x_es_train,
        y_es_train=y_es_train,
        x_es_valid=x_es_valid,
        y_es_valid=y_es_valid,
        x_valid=x_valid,
        seed=0,
    )
    preds_seed1, _ = fit_predict_lgbm(
        params,
        x_es_train=x_es_train,
        y_es_train=y_es_train,
        x_es_valid=x_es_valid,
        y_es_valid=y_es_valid,
        x_valid=x_valid,
        seed=1,
    )
    assert not np.allclose(preds_seed0, preds_seed1)


def test_lgbm_respects_max_estimators_upper_bound() -> None:
    """early stopping 없이도 무한정 돌지 않는다 — ``n_estimators`` 상한이 있다."""
    rng = np.random.default_rng(2)
    x_es_train = rng.normal(size=(50, 2))
    y_es_train = rng.normal(size=50)  # 신호 없음 — 계속 개선되지 않아 빨리 멈춰야 한다
    x_es_valid = rng.normal(size=(20, 2))
    y_es_valid = rng.normal(size=20)
    x_valid = rng.normal(size=(5, 2))
    params = LgbmParams(num_leaves=15, learning_rate=0.1, min_data_in_leaf=5, feature_fraction=1.0)

    _preds, best_iter = fit_predict_lgbm(
        params,
        x_es_train=x_es_train,
        y_es_train=y_es_train,
        x_es_valid=x_es_valid,
        y_es_valid=y_es_valid,
        x_valid=x_valid,
        seed=0,
    )
    assert 0 <= best_iter <= 1000


def test_lgbm_grid_helper_and_ridge_grid_are_independent_of_run_order() -> None:
    assert lgbm_grid() == lgbm_grid()
    assert ridge_grid() == ridge_grid()


def test_elasticnet_alphas_are_distinct_from_ridge_alphas() -> None:
    """ENet과 Ridge는 sklearn 손실 척도가 달라(모듈 docstring 참고) 같은
    alpha 그리드를 재사용하면 안 된다 — 재사용 회귀를 막는 잠금 테스트."""
    assert set(ENET_ALPHAS).isdisjoint(RIDGE_ALPHAS)


@pytest.mark.parametrize("alpha", RIDGE_ALPHAS)
def test_ridge_alpha_grid_values_are_ridge_scale(alpha: float) -> None:
    assert alpha >= 0.1  # Ridge 척도(비정규화 손실)에서 쓰는 값
