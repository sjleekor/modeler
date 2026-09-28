"""``modeler.us.m4_run`` 단위 테스트 — 전부 합성 데이터다. 실제 레이크를 읽지
않는다(``06_execution_steps.md`` M4 §7 지시).

M3 입력 목록 읽기, ``build_m4_inputs``의 holdout 날짜 벽, 그리드 선택 통계량
(``_safe_mean_ic``), rank IC 집계(``evaluate_oof``/``_pooled_stats``)를
검사한다. us4 입력 선택(``--features-dataset``·``--labels-version``·
``--model-input-from``·``--run-tag``)의 순수 함수(``select_scan_ab_features``·
``dedup_correlated_candidates``·``resolve_model_input_features``·
``model_runs_dir``)와, 그 옵션을 지정하지 않으면 지금과 완전히 같은지도
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
    DEDUP_RHO_THRESHOLD,
    DEFAULT_FEATURES_DATASET,
    OLS3_FEATURES,
    ScanFeatureCandidate,
    _pooled_stats,
    _safe_mean_ic,
    build_m4_inputs,
    dedup_correlated_candidates,
    evaluate_oof,
    labels_dataset_name,
    load_m3_model_input_features,
    model_runs_dir,
    resolve_model_input_features,
    select_scan_ab_features,
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


# --- 5. select_scan_ab_features — --model-input-from의 A·B 필터(순수 함수) ------


def _scan_row(
    feature: str,
    *,
    universe: str = "price_ge_5",
    direction: str = "registered",
    grade: str = "A",
    t_long: float | None = 3.5,
) -> dict:
    return {
        "feature": feature,
        "direction": direction,
        "universe": universe,
        "grade": grade,
        "t_LONG": t_long,
    }


def test_select_scan_ab_features_filters_universe_and_grade() -> None:
    table = pl.DataFrame(
        [
            _scan_row("f_a", universe="price_ge_5", grade="A", t_long=3.0),
            _scan_row("f_b", universe="price_ge_5", grade="C", t_long=5.0),  # 등급 탈락
            _scan_row("f_c", universe="all", grade="A", t_long=4.0),  # 유니버스 탈락
            _scan_row("f_d", universe="price_ge_5", grade="B", t_long=2.5),
        ]
    )
    candidates = select_scan_ab_features(table)
    features = [c.feature for c in candidates]
    assert features == ["f_a", "f_d"]  # |t_LONG| 내림차순


def test_select_scan_ab_features_collapses_bidirectional_by_abs_t() -> None:
    """both_a/both_b 둘 다 A·B 등급이면 |t_LONG|이 큰 쪽만 남고, 피쳐는
    하나로 합쳐진다."""
    table = pl.DataFrame(
        [
            _scan_row("f_a", direction="both_a", grade="B", t_long=-2.5),
            _scan_row("f_a", direction="both_b", grade="A", t_long=4.0),
        ]
    )
    candidates = select_scan_ab_features(table)
    assert len(candidates) == 1
    assert candidates[0].feature == "f_a"
    assert candidates[0].t_long == pytest.approx(4.0)
    assert candidates[0].direction == "both_b"


def test_select_scan_ab_features_ignores_non_finite_t() -> None:
    table = pl.DataFrame(
        [
            _scan_row("f_a", grade="A", t_long=None),
            _scan_row("f_b", grade="A", t_long=float("nan")),
            _scan_row("f_c", grade="A", t_long=1.5),
        ]
    )
    candidates = select_scan_ab_features(table)
    assert [c.feature for c in candidates] == ["f_c"]


def test_select_scan_ab_features_missing_columns_raises() -> None:
    table = pl.DataFrame({"feature": ["f_a"]})
    with pytest.raises(ValueError, match="칸이 빠졌습니다"):
        select_scan_ab_features(table)


def test_select_scan_ab_features_empty_when_nothing_qualifies() -> None:
    table = pl.DataFrame([_scan_row("f_a", universe="all", grade="A")])
    assert select_scan_ab_features(table) == []


# --- 6. dedup_correlated_candidates — |rho|>0.8 규칙(scan.py 그대로 재사용) ------


def _corr_core(dates: list[date], n: int = 25) -> pl.DataFrame:
    """``feat_x``·``feat_y``(``feat_x``와 완전 상관) · ``feat_z``(독립)를 담은
    ``month_idx`` 프레임. ``pairwise_avg_rank_corr``가 유한값을 내려면
    ``MIN_NAMES``(20) 이상 이름이 필요해 ``n=25``로 둔다."""
    rows = []
    for i, d in enumerate(dates):
        for s in range(n):
            rows.append(
                {
                    "date": d,
                    "month_idx": i + 1,
                    "symbol": f"S{s:03d}",
                    "feat_x": float(s),
                    "feat_y": float(s) * 2.0 + 1.0,  # feat_x와 완전 상관(순위 기준)
                    "feat_z": float((s * 37 + 5) % n),  # feat_x와 무관한 순열
                }
            )
    return pl.DataFrame(rows)


def test_dedup_correlated_candidates_drops_high_corr_keeps_higher_priority() -> None:
    core = _corr_core([D1, D2])
    candidates = [
        ScanFeatureCandidate(feature="feat_x", t_long=4.0, grade="A", direction="registered"),
        ScanFeatureCandidate(feature="feat_y", t_long=3.0, grade="A", direction="registered"),
    ]
    kept, dropped = dedup_correlated_candidates(candidates, core)
    assert kept == ["feat_x"]
    assert len(dropped) == 1
    assert dropped[0]["feature"] == "feat_y"
    assert dropped[0]["collides_with"] == "feat_x"
    assert abs(dropped[0]["rho"]) > DEDUP_RHO_THRESHOLD


def test_dedup_correlated_candidates_keeps_uncorrelated() -> None:
    core = _corr_core([D1, D2])
    candidates = [
        ScanFeatureCandidate(feature="feat_x", t_long=4.0, grade="A", direction="registered"),
        ScanFeatureCandidate(feature="feat_z", t_long=3.0, grade="A", direction="registered"),
    ]
    kept, dropped = dedup_correlated_candidates(candidates, core)
    assert kept == ["feat_x", "feat_z"]
    assert dropped == []


def test_dedup_correlated_candidates_empty_input_returns_empty() -> None:
    core = _corr_core([D1])
    assert dedup_correlated_candidates([], core) == ([], [])


# --- 7. resolve_model_input_features -------------------------------------------


def test_resolve_model_input_features_default_reads_m3_manifest(tmp_path: Path) -> None:
    """``model_input_from``이 ``None``이면 지금과 완전히 같다 — M3 manifest."""
    root = DataRoot(base=tmp_path)
    _write_feature_scan_manifest(root, "2026-09-21", ["feat_a", "feat_b"])
    features_dev = pl.DataFrame({"date": [D1], "symbol": ["S000"]})
    labels_dev = pl.DataFrame({"date": [D1], "symbol": ["S000"]})

    features, path, source, dropped = resolve_model_input_features(
        root, features_dev=features_dev, labels_dev=labels_dev, model_input_from=None
    )

    assert features == ["feat_a", "feat_b"]
    assert "2026-09-21" in str(path)
    assert "M3" in source
    assert dropped == []


def _write_scan_long2_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def test_resolve_model_input_features_from_scan_dedups_by_correlation(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    scan_path = tmp_path / "feature_scan_long2.parquet"
    _write_scan_long2_parquet(
        scan_path,
        [
            _scan_row("feat_x", grade="A", t_long=4.0),
            _scan_row("feat_y", grade="A", t_long=3.0),  # feat_x와 상관 1.0 — 제거 대상
            _scan_row("feat_z", grade="B", t_long=2.0),  # 독립 — 남는다
            _scan_row("feat_c_grade", grade="C", t_long=9.0),  # 등급 탈락
        ],
    )
    dates = [D1, D2]
    core = _corr_core(dates)
    features_dev = core.select("date", "symbol", "feat_x", "feat_y", "feat_z")
    labels_dev = features_dev.select("date", "symbol").with_columns(
        pl.lit(True).alias("price_ge_5")
    )

    features, path, source, dropped = resolve_model_input_features(
        root, features_dev=features_dev, labels_dev=labels_dev, model_input_from=scan_path
    )

    assert features == ["feat_x", "feat_z"]
    assert path == scan_path
    assert "scan_long2" in source
    assert len(dropped) == 1
    assert dropped[0]["feature"] == "feat_y"


def test_resolve_model_input_features_no_candidates_raises(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    scan_path = tmp_path / "feature_scan_long2.parquet"
    _write_scan_long2_parquet(scan_path, [_scan_row("feat_x", grade="C", t_long=9.0)])
    features_dev = pl.DataFrame({"date": [D1], "symbol": ["S000"], "feat_x": [1.0]})
    labels_dev = pl.DataFrame({"date": [D1], "symbol": ["S000"], "price_ge_5": [True]})

    with pytest.raises(ValueError, match="후보가 없습니다"):
        resolve_model_input_features(
            root, features_dev=features_dev, labels_dev=labels_dev, model_input_from=scan_path
        )


def test_resolve_model_input_features_missing_price_ge_5_raises(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    scan_path = tmp_path / "feature_scan_long2.parquet"
    _write_scan_long2_parquet(scan_path, [_scan_row("feat_x", grade="A", t_long=3.0)])
    features_dev = pl.DataFrame({"date": [D1], "symbol": ["S000"], "feat_x": [1.0]})
    labels_dev = pl.DataFrame({"date": [D1], "symbol": ["S000"]})  # price_ge_5 없음

    with pytest.raises(KeyError, match="price_ge_5"):
        resolve_model_input_features(
            root, features_dev=features_dev, labels_dev=labels_dev, model_input_from=scan_path
        )


def test_resolve_model_input_features_missing_candidate_column_raises(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    scan_path = tmp_path / "feature_scan_long2.parquet"
    _write_scan_long2_parquet(scan_path, [_scan_row("feat_missing", grade="A", t_long=3.0)])
    features_dev = pl.DataFrame({"date": [D1], "symbol": ["S000"]})  # feat_missing 없음
    labels_dev = pl.DataFrame({"date": [D1], "symbol": ["S000"], "price_ge_5": [True]})

    with pytest.raises(KeyError, match="feat_missing"):
        resolve_model_input_features(
            root, features_dev=features_dev, labels_dev=labels_dev, model_input_from=scan_path
        )


# --- 8. model_runs_dir — output/model_runs[/<run_tag>] --------------------------


def test_model_runs_dir_without_tag_matches_current_behavior(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    assert model_runs_dir(root) == root.output / "model_runs"
    assert model_runs_dir(root, run_tag=None) == root.output / "model_runs"


def test_model_runs_dir_with_tag_nests_under_tag(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    assert model_runs_dir(root, run_tag="us4_flow") == root.output / "model_runs" / "us4_flow"


# --- 9. labels_dataset_name ------------------------------------------------------


def test_labels_dataset_name_v1_matches_current_default() -> None:
    assert labels_dataset_name("v1") == "us_labels_v1"


def test_labels_dataset_name_v2() -> None:
    assert labels_dataset_name("v2") == "us_labels_v2"


# --- 10. build_m4_inputs — 입력 선택 옵션이 지금 동작을 완전히 보존하는가 --------


def _write_features_and_labels(
    root: DataRoot,
    *,
    dataset_name: str,
    feature_cols: list[str],
    dates: list[date],
    n_symbols: int = 25,
    include_ols3: bool = True,
) -> None:
    cols = list(feature_cols)
    if include_ols3:
        cols += [c for c in OLS3_FEATURES if c not in cols]
    rows = []
    for d in dates:
        for s in range(n_symbols):
            row: dict = {"date": d, "symbol": f"S{s:03d}"}
            for name in cols:
                row[name] = float(s)
                row[f"{name}_isna"] = False
            rows.append(row)
    write_dataset(pl.DataFrame(rows), root, dataset_name, manifest={})


def _write_labels(
    root: DataRoot, *, dataset_name: str, dates: list[date], n_symbols: int = 25
) -> None:
    rows = []
    for d in dates:
        for s in range(n_symbols):
            rows.append(
                {
                    "date": d,
                    "symbol": f"S{s:03d}",
                    "L0": float(s) / n_symbols,
                    "L1": float(s) / n_symbols,
                    "L2": float(s),
                    "y_rank": float(s) / (n_symbols - 1),
                    "y_up": s % 2 == 0,
                    "price_ge_5": True,
                }
            )
    write_dataset(pl.DataFrame(rows), root, dataset_name, manifest={})


def test_build_m4_inputs_defaults_match_current_behavior(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    dates = [D1, D2]
    model_features = ["feat_a", "feat_b"]
    _write_features_and_labels(
        root, dataset_name="us_features_v1", feature_cols=model_features, dates=dates
    )
    _write_labels(root, dataset_name="us_labels_v1", dates=dates)
    _write_feature_scan_manifest(root, "2026-09-21", model_features)

    inputs = build_m4_inputs(root)

    assert inputs.model_features == model_features
    assert inputs.features_dataset == DEFAULT_FEATURES_DATASET
    assert inputs.labels_dataset == "us_labels_v1"
    assert inputs.dedup_dropped == []
    assert "M3" in inputs.model_input_source


def test_build_m4_inputs_respects_labels_version(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    dates = [D1, D2]
    model_features = ["feat_a"]
    _write_features_and_labels(
        root, dataset_name="us_features_v1", feature_cols=model_features, dates=dates
    )
    _write_labels(root, dataset_name="us_labels_v2", dates=dates)
    _write_feature_scan_manifest(root, "2026-09-21", model_features)

    inputs = build_m4_inputs(root, labels_dataset=labels_dataset_name("v2"))

    assert inputs.labels_dataset == "us_labels_v2"
    assert inputs.core.height > 0


def test_build_m4_inputs_joins_ols3_from_baseline_when_features_dataset_differs(
    tmp_path: Path,
) -> None:
    """``--features-dataset``이 OLS3 컬럼이 없는 다른 family 데이터셋(예:
    ``us_features_flow_v1``)이어도, OLS3_FEATURES는 항상
    ``DEFAULT_FEATURES_DATASET``에서 붙어야 한다."""
    root = DataRoot(base=tmp_path)
    dates = [D1, D2]
    flow_features = ["flow_a"]
    # us_features_v1 (baseline) — OLS3 컬럼을 포함해 항상 존재한다.
    _write_features_and_labels(
        root, dataset_name="us_features_v1", feature_cols=[], dates=dates, include_ols3=True
    )
    # us_features_flow_v1 — OLS3 컬럼이 전혀 없다(F17~F19 계열만 있다고 가정).
    _write_features_and_labels(
        root,
        dataset_name="us_features_flow_v1",
        feature_cols=flow_features,
        dates=dates,
        include_ols3=False,
    )
    _write_labels(root, dataset_name="us_labels_v1", dates=dates)
    _write_feature_scan_manifest(root, "2026-09-21", flow_features)

    inputs = build_m4_inputs(root, features_dataset="us_features_flow_v1")

    assert inputs.features_dataset == "us_features_flow_v1"
    assert inputs.model_features == flow_features
    for col in ("mcap_rank_rank", "bm_rank", "mom_12_1_rank", "flow_a_rank"):
        assert col in inputs.core.columns


def test_build_m4_inputs_model_input_from_scan_end_to_end(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    dates = [D1, D2]
    # feat_x·feat_y가 완전 상관 — dedup으로 feat_y가 빠져야 한다. feat_z는 등급 C라 애초에 탈락.
    _write_features_and_labels(
        root,
        dataset_name="us_features_v1",
        feature_cols=["feat_x", "feat_y", "feat_z"],
        dates=dates,
        include_ols3=True,
    )
    _write_labels(root, dataset_name="us_labels_v1", dates=dates)
    scan_path = tmp_path / "feature_scan_long2.parquet"
    _write_scan_long2_parquet(
        scan_path,
        [
            _scan_row("feat_x", grade="A", t_long=4.0),
            _scan_row("feat_y", grade="A", t_long=3.0),
            _scan_row("feat_z", grade="C", t_long=9.0),
        ],
    )

    inputs = build_m4_inputs(root, model_input_from=scan_path)

    assert inputs.model_features == ["feat_x"]
    assert len(inputs.dedup_dropped) == 1
    assert inputs.dedup_dropped[0]["feature"] == "feat_y"
    assert "scan_long2" in inputs.model_input_source
    assert "feat_x_rank" in inputs.core.columns
    assert "feat_y_rank" not in inputs.core.columns  # 모델 입력에서 빠졌다
