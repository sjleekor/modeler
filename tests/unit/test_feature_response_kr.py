from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from modeler.analysis.feature_response import kr


def _sample(n_dates: int = 6, per: int = 40, n_feat: int = 3, labeled: bool = True) -> kr.Sample:
    rng = np.random.default_rng(0)
    feats = [f"f{i}" for i in range(n_feat)]
    design = [*feats, *(f"{f}_isna" for f in feats)]
    dates = (np.repeat(np.arange(n_dates) * 60, per)).astype("timedelta64[D]") + np.datetime64("2025-08-01")
    X = np.hstack([rng.random((n_dates * per, n_feat)), np.zeros((n_dates * per, n_feat))])
    X[:, n_feat] = (rng.random(n_dates * per) < 0.2)  # f0_isna
    keys = pl.DataFrame({"trade_date": dates, "ticker": [str(i) for i in range(len(dates))], "market": "KOSPI"})
    y = (X[:, 0] + 0.3 * rng.standard_normal(len(dates)) > 0.5).astype(np.int8) if labeled else None
    ret = X[:, 0] + rng.standard_normal(len(dates)) if labeled else None
    return kr.Sample(X, dates, keys, design, y, ret)


def test_permuted_block_stays_in_date_and_moves_isna_together() -> None:
    s = _sample()
    cols = [0, 3]
    blk = kr.permuted_block(s.X, cols, s.offsets, np.random.default_rng(1))
    changed = False
    for i in range(s.n_dates):
        a, b = s.offsets[i], s.offsets[i + 1]
        for k, c in enumerate(cols):
            assert np.array_equal(np.sort(blk[a:b, k]), np.sort(s.X[a:b, c]))
        # same row permutation: (value, isna) pairs preserved as a set
        pairs_new = sorted(map(tuple, blk[a:b].tolist()))
        pairs_old = sorted(map(tuple, s.X[a:b][:, cols].tolist()))
        assert pairs_new == pairs_old
        changed |= not np.array_equal(blk[a:b], s.X[a:b][:, cols])
    assert changed


def test_permutation_deterministic_by_seed() -> None:
    s = _sample()
    a = kr.permuted_block(s.X, [0], s.offsets, np.random.default_rng(5))
    b = kr.permuted_block(s.X, [0], s.offsets, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_load_design_never_loads_excluded_dates_and_checks_order(tmp_path: Path) -> None:
    from modeler.serving import kr_model as km  # noqa: F401

    feats = ["px_ret_1d", "px_vol_20d"]
    days = [dt.date(2025, 8, 1), dt.date(2026, 8, 28), dt.date(2026, 9, 17), dt.date(2026, 9, 18)]
    rows = []
    for d in days:
        for t in range(5):
            rows.append({"trade_date": d, "ticker": f"T{t}", "market": "KOSPI",
                         "px_ret_1d": float(t), "px_vol_20d": float(5 - t) if t else None})
    p = tmp_path / "panel.parquet"
    pl.DataFrame(rows).write_parquet(p)
    design = [*feats, *(f"{f}_isna" for f in feats)]
    out, got = kr.load_design(p, feats, design, log=lambda m: None)
    assert out["trade_date"].max() == dt.date(2026, 8, 28)
    assert out.height == 10
    assert got == design
    # ranked: percentile in [0,1], null filled 0.5 + isna flag
    assert out["px_ret_1d"].min() == 0.0 and out["px_ret_1d"].max() == 1.0
    assert out.filter(pl.col("px_vol_20d_isna") == 1)["px_vol_20d"].to_list() == [0.5, 0.5]
    with pytest.raises(AssertionError, match="design column order"):
        kr.load_design(p, feats, design[::-1], log=lambda m: None)


def test_union_find_and_corr_groups() -> None:
    names = ["a", "b", "c", "d", "e"]
    rho = np.eye(5)
    rho[0, 1] = rho[1, 0] = 0.8
    rho[1, 2] = rho[2, 1] = -0.75  # chains a-b-c
    rho[3, 4] = rho[4, 3] = 0.69
    assert kr.correlation_groups(rho, names) == [["a", "b", "c"], ["d"], ["e"]]
    rho[3, 4] = rho[4, 3] = np.nan
    assert len(kr.correlation_groups(rho, names)) == 3


def test_shape_classes() -> None:
    x = np.linspace(0, 1, 21)
    thr = 0.01
    assert kr.classify_shape(x * 0.001, thr) == "평평"
    assert kr.classify_shape(x, thr) == "단조 증가"
    assert kr.classify_shape(-x, thr) == "단조 감소"
    assert kr.classify_shape((x - 0.5) ** 2, thr) == "U자"
    assert kr.classify_shape(-((x - 0.5) ** 2), thr) == "역U자"
    zig = np.full(21, 0.5)
    zig[5], zig[15] = 0.0, 1.0  # min and max both interior; ends above min -> U (us rule)
    assert kr.classify_shape(zig, thr) == "U자"
    flat_end = np.full(21, 0.5)
    flat_end[0], flat_end[10] = 0.0, 1.0  # min at an end, max interior, last end 0.5 < 1
    assert kr.classify_shape(flat_end, thr) == "역U자"
    assert kr.flat_threshold(np.array([0.1, 0.2, 0.4])) == pytest.approx(0.05)


def test_stability_rule() -> None:
    names = [f"x{i}" for i in range(20)]
    imp1 = {n: float(20 - i) for i, n in enumerate(names)}
    imp2 = dict(imp1)
    s1 = {n: 1.0 for n in names}
    s2 = dict(s1)
    s2["x0"] = -1.0  # top in both but sign flips
    imp2["x5"] = 0.0  # top10 in half1, bottom in half2 -> unstable
    imp2["x15"] = 100.0  # bottom half 1, top half 2 -> unstable
    out = kr.stable_flags(imp1, imp2, s1, s2, top_k=3)
    assert out["x0"] is False and out["x1"] is True and out["x5"] is False
    assert out["x19"] is True and out["x15"] is False
    assert out["x10"] is True and out["x7"] is False  # x7: rank 8 -> middle band


def test_daily_scorer_known_ic_and_halves() -> None:
    s = _sample()
    sc = kr.DailyScorer(s)
    ic, ll = sc.score(s.ret.copy() / 10 + 0.5)  # monotone in ret -> IC 1
    assert np.allclose(ic, 1.0)
    masks = kr.half_masks(s.unique_dates)
    assert masks["full"].all()
    with pytest.raises(ValueError):
        kr.half_masks(np.array(["2026-02-01"], dtype="datetime64[D]"))  # gap day belongs to neither half


def test_permutation_and_shap_on_tiny_hgb() -> None:
    pytest.importorskip("shap")
    s = _sample(n_dates=8, per=60)
    model = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(s.X, s.y)
    groups = {"f:f0": ["f0"], "f:f1": ["f1"], "c01": ["f0", "f1"]}
    kind = {"f:f0": "feature", "f:f1": "feature", "c01": "corr_group"}
    res = kr.run_permutation(model, s, groups, s.design, repeats=2, log=lambda m: None)
    # the signal column f0 must matter more than the pure-noise f1
    assert res["base_ic"].mean() - res["perm_ic"][0].mean() > res["base_ic"].mean() - res["perm_ic"][1].mean()
    long, summ = kr.importance_tables(res, s.unique_dates, groups, kind)
    assert long.height == 3 * 3 * 2 and summ.height == 3
    assert summ["delta_rank_ic_first"].null_count() == 0 and summ["delta_rank_ic_second"].null_count() == 0
    sh = kr.run_shap(model, s, log=lambda m: None)
    assert sh["additivity_max_abs_error"] < 1e-8
    df, _ = kr.shap_summary(sh["values"], s, ["f0", "f1", "f2"])
    assert df.height == 3 and df["mean_abs_shap_combined"][0] > df["mean_abs_shap_combined"][1]
    pdp = kr.run_pdp(model, s, ["f0"], log=lambda m: None)
    full = pdp.filter(pl.col("subset") == "full").sort("grid_idx")
    assert full.height == kr.GRID_POINTS and full["pdp_p"][-1] > full["pdp_p"][0]
    ice = kr.run_ice(model, s, ["f0"])
    assert ice.height == min(kr.ICE_ROWS, s.X.shape[0]) * kr.GRID_POINTS


def test_run_end_to_end_on_synthetic_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("shap")
    from types import SimpleNamespace

    from modeler.serving import kr_model as km

    rng = np.random.default_rng(3)
    feats = ["px_ret_1d", "px_vol_20d", "fin_roa"]
    design = [*feats, *(f"{f}_isna" for f in feats)]
    days = [dt.date(2025, 8, 1) + dt.timedelta(days=45 * k) for k in range(12)]
    prow, lrow = [], []
    for d in days:
        for t in range(80):
            v = rng.standard_normal(3)
            prow.append({"trade_date": d, "ticker": f"T{t:03d}", "market": "KOSPI",
                         "px_ret_1d": v[0], "px_vol_20d": v[1], "fin_roa": None if t % 9 == 0 else v[2]})
            lrow.append({"trade_date": d, "ticker": f"T{t:03d}", "market": "KOSPI",
                         "y_up_20d": int(v[0] + rng.standard_normal() * 0.5 > 0),
                         "raw_label_20d": float(v[0] + rng.standard_normal())})
    root = tmp_path / "sd"
    ds = root / kr.SNAPSHOT_DIR
    ds.mkdir(parents=True)
    pl.DataFrame(prow).write_parquet(ds / "feat_panel.parquet")
    pl.DataFrame(lrow).write_parquet(ds / "label_daily.parquet")
    (ds / "dataset_manifest.json").write_text("{}")
    monkeypatch.setenv("STOCK_DATA_ROOT", str(root))
    tr, _ = kr.load_design(ds / "feat_panel.parquet", feats, design, log=lambda m: None)
    joined, _ = kr.attach_labels(tr, ds / "label_daily.parquet", design)
    s0 = kr.to_sample(joined, design, labeled=True)
    model = HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(s0.X, s0.y)
    bundle = SimpleNamespace(model=model, feature_columns=tuple(feats), design_columns=tuple(design),
                             golden={"symbols": []}, manifest={"files": {}, "manifest_sha256": "x",
                                                                 "model_id": "t", "model_version": "0"})
    monkeypatch.setattr(kr, "_load_bundle_checked", lambda d: (bundle, {"golden_verify": "passed"}))
    monkeypatch.setattr(km, "_model_input", km._model_input)
    args = kr.argparse.Namespace(bundle_dir="x", out_root=str(tmp_path / "out"), dates_limit=1000,
                                 repeats=2, resume=None)
    assert kr.run(args) == 0
    run_dir = next((tmp_path / "out").iterdir())
    for name in ("manifest.json", "importance.parquet", "importance_summary.parquet", "shap_summary.parquet",
                 "pdp.parquet", "ice_sample.parquet", "groups.json", "report.md", ".done"):
        assert (run_dir / name).exists(), name
    manifest = kr.json.loads((run_dir / "manifest.json").read_text())
    assert manifest["labels_read"] is True and manifest["compute"]["sampling_switch_every_2nd_date"] is False
    assert manifest["method"]["shap_additivity_max_abs_error"] < 1e-8
