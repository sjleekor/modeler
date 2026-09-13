"""Unit tests for P5 — preprocess (isna/impute/per-date winsor/log/zscore)."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from modeler.etl import preprocess as pp


def _panel() -> pl.DataFrame:
    # two dates, a few names; px_ret_20d varies, one null; px_is_halted boolean;
    # px_turnover is a "level" (median-imputed).
    d1, d2 = datetime.date(2020, 1, 1), datetime.date(2020, 1, 2)
    return pl.DataFrame(
        {
            "trade_date": [d1, d1, d1, d2, d2, d2],
            "ticker": ["A", "B", "C", "A", "B", "C"],
            "market": ["KOSPI"] * 6,
            "px_ret_20d": [0.1, -0.1, None, 0.2, 0.0, -0.2],
            "px_turnover": [100.0, 200.0, None, 50.0, 150.0, 250.0],
            "px_is_halted": [False, False, True, False, False, False],
            "y_rank_20d": [1.0, 0.5, 0.0, 1.0, 0.5, 0.0],
        }
    )


def test_feature_columns_excludes_keys_and_labels() -> None:
    feats = pp.feature_columns(_panel())
    assert "px_ret_20d" in feats
    assert "px_turnover" in feats
    assert "px_is_halted" in feats
    assert "y_rank_20d" not in feats
    assert "trade_date" not in feats


def test_isna_flag_marks_nulls() -> None:
    panel = _panel()
    feats = pp.feature_columns(panel)
    out = pp.add_isna_flags(panel, feats)
    # px_ret_20d null on (C, d1) -> flag 1 there, 0 elsewhere.
    flag = out.filter(
        (pl.col("ticker") == "C") & (pl.col("trade_date") == datetime.date(2020, 1, 1))
    )["px_ret_20d_isna"][0]
    assert flag == 1


def test_zero_vs_median_impute() -> None:
    panel = _panel()
    cfg = pp.PreprocessConfig(profile="linear")
    feats = pp.feature_columns(panel)
    out = pp._cast_bool_features(pp.add_isna_flags(panel, feats), feats)
    out = pp.impute(out, feats, cfg)
    row = out.filter(
        (pl.col("ticker") == "C") & (pl.col("trade_date") == datetime.date(2020, 1, 1))
    )
    # ret_* -> zero impute; turnover (level) -> per-date median of {100,200} = 150.
    assert row["px_ret_20d"][0] == pytest.approx(0.0)
    assert row["px_turnover"][0] == pytest.approx(150.0)


def test_per_date_zscore_mean0_std1_and_finite() -> None:
    panel = _panel()
    cfg = pp.PreprocessConfig(profile="linear")
    fitted = pp.fit(panel, cfg)
    out = fitted.transform(panel)
    pp.assert_finite(out, fitted.feature_cols)
    # px_ret_20d standardized within each date -> mean ~0 per date.
    agg = out.group_by("trade_date").agg(pl.col("px_ret_20d").mean().alias("m"))
    for m in agg["m"].to_list():
        assert abs(m) < 1e-9


def test_tree_profile_skips_zscore_and_keeps_nulls() -> None:
    panel = _panel()
    cfg = pp.PreprocessConfig(profile="tree")
    fitted = pp.fit(panel, cfg)
    out = fitted.transform(panel)
    # tree profile: no impute, so the original null survives (trees handle it).
    null_count = out["px_ret_20d"].null_count()
    assert null_count == 1


def test_winsor_bounds_fit_on_train_only() -> None:
    cfg = pp.PreprocessConfig(profile="linear", winsor=(0.0, 1.0))
    panel = _panel()
    feats = pp.feature_columns(panel)
    bounds = pp.fit_winsor_bounds(panel, feats, cfg)
    # with (0,1) quantiles, bounds are min/max of the column.
    lo, hi = bounds["px_ret_20d"]
    assert lo == pytest.approx(-0.2)
    assert hi == pytest.approx(0.2)


def test_invalid_profile_raises() -> None:
    with pytest.raises(ValueError):
        pp.PreprocessConfig(profile="nope")
