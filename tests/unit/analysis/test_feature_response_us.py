"""US 층 3 진단 — 합성 데이터 시험 (실데이터·라벨 없음)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
from sklearn.linear_model import Ridge

from modeler.analysis.feature_response import us
from modeler.us.m4_transform import design_matrix_columns, to_design_arrays


def test_design_column_order_matches_model_features():
    feats = ["a", "b", "c"]
    df = pl.DataFrame(
        {f"{f}_rank": [0.1, 0.5] for f in feats} | {f"{f}_isna": [0, 1] for f in feats}
    )
    x, cols = to_design_arrays(df, feats)
    assert cols == ["a_rank", "b_rank", "c_rank", "a_isna", "b_isna", "c_isna"]
    assert cols == design_matrix_columns(feats)
    assert x.shape == (2, 6)


def test_no_label_reference_in_source():
    src = Path(us.__file__).read_text()
    assert "us_labels" not in src
    assert all("label" not in n for n in us.FEATURE_DATASETS)


def test_label_paths_refused(tmp_path):
    with pytest.raises(RuntimeError):
        us._forbid_label_path(tmp_path / "us_labels_v2" / "part.parquet")
    us._forbid_label_path(tmp_path / "us_features_v2" / "part.parquet")


def test_load_feature_rows_never_opens_label_paths(tmp_path, monkeypatch):
    """합성 두 데이터셋만 있는 디렉터리 — 'labels'가 든 경로를 열면 실패하도록 막고 돌린다."""
    feats = ["f1", "f2"]
    dates = [date(2025, m, 1) for m in (7, 8, 9, 10, 11, 12)] + [
        date(2026, 1, 2),
        *[date(2026, m, 1) for m in (2, 3, 4, 5, 6)],
    ]
    rng = np.random.default_rng(0)
    rows = [(d, f"S{i}") for d in dates for i in range(30)]
    n = len(rows)
    base = pl.DataFrame(
        {
            "date": [r[0] for r in rows],
            "symbol": [r[1] for r in rows],
            "price_ge_5": [True] * n,
            "f1": rng.normal(size=n),
        }
    )
    flow = pl.DataFrame({"date": base["date"], "symbol": base["symbol"], "f2": rng.normal(size=n)})
    for name, df in (("us_features_v2", base), ("us_features_flow_v1", flow)):
        (tmp_path / name).mkdir()
        df.write_parquet(tmp_path / name / "part.parquet")
        (tmp_path / name / "manifest.json").write_text("{}")
    (tmp_path / "us_labels_v2").mkdir()
    (tmp_path / "us_labels_v2" / "part.parquet").write_bytes(b"x")

    real_scan, real_read, real_open = pl.scan_parquet, pl.read_parquet, Path.open

    def guard(fn):
        def inner(p, *a, **k):
            if "labels" in str(p):
                raise AssertionError(f"label path opened: {p}")
            return fn(p, *a, **k)

        return inner

    monkeypatch.setattr(pl, "scan_parquet", guard(real_scan))
    monkeypatch.setattr(pl, "read_parquet", guard(real_read))
    monkeypatch.setattr(Path, "open", lambda self, *a, **k: guard(real_open)(self, *a, **k))
    frame, got_dates, info = us.load_feature_rows(tmp_path, feats, None)
    assert len(got_dates) == 12
    assert set(info["input_part_sha256"]) == set(us.FEATURE_DATASETS)
    x, cols = to_design_arrays(frame, feats)
    assert cols == ["f1_rank", "f2_rank", "f1_isna", "f2_isna"]
    assert x.shape == (360, 4)


def test_lgbm_pred_contrib_additivity():
    rng = np.random.default_rng(1)
    x = rng.uniform(size=(600, 6))
    y = 2 * x[:, 0] - x[:, 1] + 0.1 * rng.normal(size=600)
    model = lgb.LGBMRegressor(n_estimators=30, num_leaves=7, verbosity=-1, n_jobs=1).fit(x, y)
    contrib, _, err = us.lgbm_contrib(model, x, threads=1)
    assert contrib.shape == (600, 6)
    assert err < 1e-9
    top = np.abs(contrib).mean(axis=0).argsort()[::-1][:2]
    assert set(top) == {0, 1}


def test_ridge_contrib_exact_and_components():
    rng = np.random.default_rng(2)
    x = rng.uniform(size=(200, 6))  # 3 rank + 3 isna
    y = x[:, 0] - 2 * x[:, 4] + 0.05 * rng.normal(size=200)
    model = Ridge(alpha=1.0).fit(x, y)
    contrib, base, err = us.ridge_contrib(model, x)
    assert err < 1e-10
    comps = us.split_components(contrib, 3)
    assert np.allclose(comps["combined"], comps["rank"] + comps["isna"])
    assert np.allclose(comps["isna"][:, 1], contrib[:, 4])


def test_pdp_sweep_restores_matrix_and_sets_isna_zero():
    x = np.full((4, 4), 0.3)
    x[:, 2:] = 1.0
    before = x.copy()
    seen = []

    def predict(a):
        seen.append((a[:, 0].copy(), a[:, 2].copy()))
        return a[:, 0]

    out = us.pdp_sweep(predict, x, 0, 2)
    assert np.array_equal(x, before)
    assert out.shape == (us.N_GRID, 4)
    assert np.allclose(out.mean(axis=1), us.GRID)
    assert all((s[1] == 0).all() for s in seen)


@pytest.mark.parametrize(
    "curve,expected",
    [
        (np.linspace(0, 1, 21), us.SHAPE_UP),
        (np.linspace(1, 0, 21), us.SHAPE_DOWN),
        ((np.linspace(-1, 1, 21)) ** 2, us.SHAPE_U),
        (-((np.linspace(-1, 1, 21)) ** 2), us.SHAPE_INV_U),
        (np.linspace(0, 0.01, 21), us.SHAPE_FLAT),
        (np.array([0.0] + [0.9, 0.1] * 10 + [1.0]), us.SHAPE_OTHER),
    ],
)
def test_classify_shape(curve, expected):
    assert us.classify_shape(curve, median_range=1.0) == expected


def test_classify_shape_flat_threshold_is_25_percent_of_median():
    curve = np.linspace(0, 0.24, 21)
    assert us.classify_shape(curve, 1.0) == us.SHAPE_FLAT
    assert us.classify_shape(np.linspace(0, 0.26, 21), 1.0) == us.SHAPE_UP


def test_stability_rule():
    ok, _ = us.is_stable(1, 8, 53, 0.4, 0.2)
    assert ok
    ok, _ = us.is_stable(40, 30, 53, -0.1, -0.3)  # 하위 절반 두 번
    assert ok
    ok, why = us.is_stable(1, 20, 53, 0.4, 0.2)
    assert not ok and "순위" in why
    ok, why = us.is_stable(1, 2, 53, 0.4, -0.2)
    assert not ok and "부호" in why
    ok, _ = us.is_stable(1, 2, 53, float("nan"), 0.2)
    assert not ok
    ok, _ = us.is_stable(26, 27, 53, 0.4, 0.2)  # 중간 — 해당 없음
    assert not ok


def test_union_find_groups_transitive_and_threshold():
    names = ["a", "b", "c", "d", "e"]
    rho = np.eye(5)
    rho[0, 1] = rho[1, 0] = 0.8
    rho[1, 2] = rho[2, 1] = -0.75  # 부호 무관, a-b-c 연결
    rho[3, 4] = rho[4, 3] = 0.69  # 문턱 미만
    rho[0, 3] = rho[3, 0] = np.nan
    groups = us.union_find_groups(names, rho, 0.7)
    assert sorted(map(sorted, groups)) == [["a", "b", "c"], ["d"], ["e"]]


def test_daily_mean_spearman_detects_duplicate_columns():
    rng = np.random.default_rng(3)
    base = rng.uniform(size=(100, 1))
    x = np.hstack([base, base**2, rng.uniform(size=(100, 1))])
    dates = np.repeat(np.array([1, 2]), 50)
    rho = us.daily_mean_spearman(x, dates)
    assert rho[0, 1] > 0.99 and abs(rho[0, 2]) < 0.5


def test_bootstrap_interval_contains_mean():
    rng = np.random.default_rng(4)
    sums = rng.uniform(1, 2, size=(12, 3)) * 100
    n = np.full(12, 100.0)
    lo, hi = us.bootstrap_mean_abs(sums, n, np.random.default_rng(0))
    mean = sums.sum(axis=0) / n.sum()
    assert (lo <= mean).all() and (mean <= hi).all()
