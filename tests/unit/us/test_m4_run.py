"""``modeler.us.m4_run`` 단위 테스트 — 전부 합성 데이터다. 실제 레이크를 읽지
않는다(``06_execution_steps.md`` M4 §7 지시).

M3 입력 목록 읽기, ``build_m4_inputs``의 holdout 날짜 벽, 그리드 선택 통계량
(``_safe_mean_ic``), rank IC 집계(``evaluate_oof``/``_pooled_stats``)를
검사한다.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.dataset import write_dataset
from modeler.us.m4_run import (
    OLS3_FEATURES,
    _pooled_stats,
    _safe_mean_ic,
    build_m4_inputs,
    evaluate_oof,
    load_m3_model_input_features,
)
from modeler.us.scan import DEV_END, assert_dev_window

D1 = date(2020, 1, 2)
D2 = date(2020, 2, 3)


# --- 1. load_m3_model_input_features --------------------------------------------


def _write_feature_scan_manifest(root: DataRoot, snapshot_date: str, features: list[str]) -> None:
    d = root.output / "feature_scan" / f"snapshot_date={snapshot_date}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps({"model_input_features": {"all": features}}, ensure_ascii=False)
    )


def test_load_m3_model_input_features_reads_latest_snapshot(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    _write_feature_scan_manifest(root, "2026-09-01", ["old_feature"])
    _write_feature_scan_manifest(root, "2026-09-21", ["sv_share_20", "turnover_rank"])

    features, manifest_path = load_m3_model_input_features(root)

    assert features == ["sv_share_20", "turnover_rank"]
    assert "2026-09-21" in str(manifest_path)


def test_load_m3_model_input_features_missing_dir_raises(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_m3_model_input_features(root)


def test_load_m3_model_input_features_empty_list_raises(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    _write_feature_scan_manifest(root, "2026-09-21", [])
    with pytest.raises(ValueError, match="비어"):
        load_m3_model_input_features(root)


# --- 2. build_m4_inputs — holdout 날짜 벽 ----------------------------------------


def test_build_m4_inputs_drops_holdout_and_adds_rank_columns(tmp_path: Path) -> None:
    """M3 목록에 없는 OLS-3 셋(``mcap_rank``·``bm``·``mom_12_1``)도 같이
    읽혀 순위 변환이 붙어야 하고, holdout 날짜(2025-08-01)는 전부 걸러져야
    한다."""
    root = DataRoot(base=tmp_path)
    n_symbols = 25
    dates_in_dev = [D1, D2]
    dates_after_dev = [date(2025, 8, 1)]
    model_features = ["feat_a", "feat_b"]
    all_feature_cols = [*model_features, *OLS3_FEATURES]

    rows = []
    for d in [*dates_in_dev, *dates_after_dev]:
        for s in range(n_symbols):
            row = {"date": d, "symbol": f"S{s:03d}"}
            for name in all_feature_cols:
                row[name] = float(s)
                row[f"{name}_isna"] = False
            rows.append(row)
    write_dataset(pl.DataFrame(rows), root, "us_features_v1", manifest={})

    labels_rows = []
    for d in [*dates_in_dev, *dates_after_dev]:
        for s in range(n_symbols):
            labels_rows.append(
                {
                    "date": d,
                    "symbol": f"S{s:03d}",
                    "L0": float(s) / n_symbols,
                    "L1": float(s) / n_symbols,
                    "L2": float(s),
                    "y_rank": float(s) / (n_symbols - 1),
                    "y_up": s % 2 == 0,
                }
            )
    write_dataset(pl.DataFrame(labels_rows), root, "us_labels_v1", manifest={})

    _write_feature_scan_manifest(root, "2026-09-21", model_features)

    inputs = build_m4_inputs(root)

    assert inputs.model_features == model_features
    assert inputs.core.height > 0
    assert inputs.core["date"].max() <= DEV_END
    assert_dev_window(inputs.core)
    assert set(inputs.dates) == set(dates_in_dev)
    for col in (
        "feat_a_rank",
        "feat_b_rank",
        "mcap_rank_rank",
        "bm_rank",
        "mom_12_1_rank",
        "l1_rank",
        "month_idx",
    ):
        assert col in inputs.core.columns


def test_build_m4_inputs_reads_dataset_content_hash(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    df = pl.DataFrame(
        {
            "date": [D1],
            "symbol": ["S000"],
            "feat_a": [1.0],
            "feat_a_isna": [False],
            **{c: [1.0] for c in OLS3_FEATURES},
            **{f"{c}_isna": [False] for c in OLS3_FEATURES},
        }
    )
    write_dataset(df, root, "us_features_v1", manifest={})
    labels_df = pl.DataFrame(
        {
            "date": [D1],
            "symbol": ["S000"],
            "L0": [0.0],
            "L1": [0.0],
            "L2": [0.0],
            "y_rank": [0.5],
            "y_up": [True],
        }
    )
    write_dataset(labels_df, root, "us_labels_v1", manifest={})
    _write_feature_scan_manifest(root, "2026-09-21", ["feat_a"])

    inputs = build_m4_inputs(root)

    assert inputs.features_content_hash is not None
    assert inputs.labels_content_hash is not None


# --- 3. _safe_mean_ic — 그리드 선택이 NaN 후보에서 멈추지 않아야 한다 -------------


def test_safe_mean_ic_ignores_nan_values() -> None:
    assert _safe_mean_ic([0.1, float("nan"), 0.3]) == pytest.approx(0.2)


def test_safe_mean_ic_all_nan_returns_negative_infinity() -> None:
    assert _safe_mean_ic([float("nan"), float("nan")]) == float("-inf")


def test_safe_mean_ic_does_not_get_stuck_on_first_nan_candidate() -> None:
    """회귀 테스트 — 2026-09-21에 발견한 버그: ``np.mean``이 NaN 섞인 리스트를
    그대로 받으면 NaN이 되고, ``real_mean_ic > nan``이 항상 False라 그리드의
    첫 후보가 NaN이면 이후 실제로 더 나은 후보가 와도 못 골랐다."""
    candidate_means = [
        _safe_mean_ic([float("nan")] * 5),  # 그리드의 첫 후보라고 가정
        _safe_mean_ic([0.05] * 5),
        _safe_mean_ic([0.2] * 5),  # 가장 좋은 후보
    ]
    best_idx = max(range(len(candidate_means)), key=lambda i: candidate_means[i])
    assert best_idx == 2


# --- 4. evaluate_oof / _pooled_stats --------------------------------------------


def _perfect_positive_corr_df(dates: list[date], n: int = 20) -> pl.DataFrame:
    rows = []
    for i, d in enumerate(dates):
        for s in range(n):
            rows.append(
                {
                    "date": d,
                    "symbol": f"S{s:03d}",
                    "pred": float(s),
                    "L2": float(s),
                    "y_rank": float(s) / (n - 1),
                    "month_idx": i + 1,
                }
            )
    return pl.DataFrame(rows)


def test_evaluate_oof_detects_perfect_positive_correlation() -> None:
    df = _perfect_positive_corr_df([D1, D2])
    ic_mean, n_dates = evaluate_oof(df, group_col="date")
    assert ic_mean == pytest.approx(1.0)
    assert n_dates == 2


def test_evaluate_oof_detects_perfect_negative_correlation() -> None:
    df = _perfect_positive_corr_df([D1]).with_columns((-pl.col("pred")).alias("pred"))
    ic_mean, n_dates = evaluate_oof(df, group_col="date")
    assert ic_mean == pytest.approx(-1.0)
    assert n_dates == 1


def test_evaluate_oof_below_min_names_is_excluded() -> None:
    """``MIN_NAMES``(20) 미만인 날짜는 그 달 IC 계산에서 빠진다."""
    df = _perfect_positive_corr_df([D1], n=5)
    ic_mean, n_dates = evaluate_oof(df, group_col="date")
    assert n_dates == 0
    assert math.isnan(ic_mean)


def test_pooled_stats_groups_by_month_idx() -> None:
    df = _perfect_positive_corr_df([D1, D2])
    ic_mean, t_hac, n_months = _pooled_stats(df)
    assert ic_mean == pytest.approx(1.0)
    assert n_months == 2
    assert math.isfinite(t_hac) or math.isnan(t_hac)  # 표본 2개면 HAC이 불안정할 수 있다
