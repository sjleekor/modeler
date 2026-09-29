from __future__ import annotations

from dataclasses import fields
from datetime import date

import numpy as np
import polars as pl
import pytest

from modeler.scores.market_sector.config import MsConfig
from modeler.scores.market_sector.models import (
    build_model,
    date_market_weights,
    fit_model,
    predict,
    save_model,
    verify_reload,
)
from modeler.scores.market_sector.splits import (
    SplitBoundaryError,
    assert_train_boundary,
    expanding_annual_folds,
    live_fold,
)

from ._helpers import make_layer2_frame, small_cfg


@pytest.fixture(scope="module")
def frame():
    return make_layer2_frame()[0].sort(["asset_id", "session"])


def test_train_rows_never_reach_fold_start(frame):
    cfg = small_cfg()
    folds = expanding_annual_folds(frame, "US", cfg)
    assert len(folds) >= 2
    end = frame["label_end_at"]
    for f in folds:
        sub = end.gather(f.train_idx.tolist())
        assert sub.null_count() == 0 and (sub < f.fold_start).all()
        # 연말까지 자르면 새는 행이 실제로 있다: 경계 검사가 의미 있다
        y_end = date(int(f.year) - 1, 12, 31)
        naive = np.flatnonzero(
            (frame["session"] <= y_end).to_numpy() & frame["label_matured"].to_numpy()
        )
        leaked = end.gather(naive.tolist()).filter(end.gather(naive.tolist()) >= f.fold_start)
        assert leaked.len() > 0
        assert set(f.train_idx).isdisjoint(f.test_idx)
    live = live_fold(frame, "US", cfg)
    assert (end.gather(live.train_idx.tolist()) < live.fold_start).all()


def test_boundary_rejects_row_with_label_end_at_fold_start(frame):
    cfg = small_cfg()
    fold = expanding_annual_folds(frame, "US", cfg)[0]
    # 학습 행 하나를 label_end_at >= fold_start 인 test 행으로 바꿔 넣는다
    bad_idx = np.append(fold.train_idx, fold.test_idx[0])
    with pytest.raises(SplitBoundaryError):
        assert_train_boundary(frame["label_end_at"], bad_idx, fold.fold_start)
    # 경계와 정확히 같은 시각도 거부한다 (엄격한 <)
    tie = frame.with_columns(pl.lit(fold.fold_start).alias("label_end_at"))
    with pytest.raises(SplitBoundaryError):
        assert_train_boundary(tie["label_end_at"], np.array([0]), fold.fold_start)


def test_scaler_is_fit_on_train_only(frame):
    rng = np.random.default_rng(0)
    Xtr = rng.normal(0, 1, (300, 4))
    Xte = rng.normal(50, 1, (100, 4))  # 다른 분포의 test
    y = Xtr[:, 0] + rng.normal(0, 0.1, 300)
    cfg = MsConfig()
    m = fit_model("ridge", Xtr, y, np.ones(300), cfg)
    sc = m.named_steps["scaler"]
    assert np.allclose(sc.mean_, Xtr.mean(axis=0))
    assert not np.allclose(sc.mean_, np.vstack([Xtr, Xte]).mean(axis=0))
    imp = m.named_steps["imputer"]
    assert imp.statistics_.shape == (4,)


def test_weights_sum_to_one_per_date_and_market():
    session = np.array(["2024-01-02"] * 5 + ["2024-01-03"] * 3, dtype="datetime64[D]")
    market = np.array(["US", "US", "US", "KR", "KR", "US", "US", "KR"], dtype=object)
    mask = np.ones(8, dtype=bool)
    mask[6] = False
    w = date_market_weights(session, market, mask)
    df = pl.DataFrame({"s": session, "m": market, "w": w, "u": mask}).filter(pl.col("u"))
    sums = df.group_by(["s", "m"]).agg(pl.col("w").sum())
    assert np.allclose(sums["w"].to_numpy(), 1.0)
    assert w[6] == 0.0
    wn = date_market_weights(session, market, mask, normalize_mean=True)
    assert wn[mask].mean() == pytest.approx(1.0)


def test_lgbm_config_is_fixed_single_candidate():
    cfg = MsConfig()
    p = cfg.lgbm_params()
    expected = dict(
        num_leaves=7,
        max_depth=3,
        n_estimators=300,
        learning_rate=0.03,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        random_state=0,
    )
    for k, v in expected.items():
        assert p[k] == v
    assert not any(f.name.startswith(("lgbm_grid", "lgbm_search")) for f in fields(cfg))
    for kind in ("lgbm_reg", "lgbm_clf"):
        m = build_model(kind, cfg)
        got = m.get_params()
        assert all(got[k] == v for k, v in expected.items())
        assert not hasattr(m, "best_iteration_") or m.best_iteration_ in (None, -1, 0)
    # fit 경로에는 eval_set/early stopping이 없다
    import inspect

    from modeler.scores.market_sector import models

    src = inspect.getsource(models.fit_model)
    assert "eval_set" not in src and "early_stopping" not in src


def test_model_reload_gives_identical_predictions(tmp_path):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(400, 5))
    X[::17, 2] = np.nan
    yc = X[:, 0] + rng.normal(size=400)
    yb = (yc > 0.5).astype(float)
    cfg = small_cfg()
    for kind, y in (("ridge", yc), ("logit", yb), ("lgbm_reg", yc), ("lgbm_clf", yb)):
        m = fit_model(
            kind, np.nan_to_num(X, nan=0.0) if kind == "lgbm_clf" else X, y, np.ones(400), cfg
        )
        path = tmp_path / f"{kind}.joblib"
        save_model(m, path)
        Xp = np.nan_to_num(X, nan=0.0) if kind == "lgbm_clf" else X
        assert verify_reload(m, path, Xp, kind)
        assert np.array_equal(predict(m, Xp, kind), predict(m, Xp, kind))
