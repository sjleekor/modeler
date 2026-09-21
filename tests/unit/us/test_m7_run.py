"""``modeler.us.m7_run`` 단위 테스트 — 전부 합성 데이터다. 실제 레이크를 읽지
않는다(``06_execution_steps.md`` M7 §7 지시. 실제 레이크를 읽는 pytest를 만들지
않는다).

**M7의 핵심 위험 셋을 검사한다** (``06`` M7 §7-4):

1. 학습 프레임에 holdout 날짜가 한 행도 섞이지 않는가(``assert_holdout_window``·
   ``assert_no_window_overlap``)
2. 재학습이 walk-forward가 아니라 **정확히 한 번**인가(``fit_predict_holdout``이
   ``fit_predict_ridge``를 한 번만 부르고, 그 학습 입력 행 수가 holdout을
   포함하지 않는가)
3. 갈래 판정(``00`` §3.2)이 부호만 보는가

부호 뒤섞기(permutation) 자체의 성질은 ``test_metrics.py``가 검사한다 —
``m7_run.monthly_series_frame``이 그 재료(월별 시계열)를 조립하는 조인 로직만
여기서 확인한다.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.dataset import write_dataset
from modeler.us.m4_run import OLS3_FEATURES, build_m4_inputs
from modeler.us.m7_run import (
    DEV_END,
    HOLDOUT_END,
    HOLDOUT_START,
    assert_holdout_window,
    assert_no_window_overlap,
    build_holdout_core,
    closed_by_reason_counts,
    decide_branch,
    enforce_holdout_window,
    fit_predict_holdout,
    load_holdout_frame,
    monthly_series_frame,
)

D_DEV_1 = date(2020, 1, 2)
D_DEV_2 = date(2020, 2, 3)
D_HOLD_1 = date(2025, 8, 1)
D_HOLD_2 = date(2025, 9, 2)
MODEL_FEATURES = ["feat_a", "feat_b"]


# --- 1. enforce_holdout_window / load_holdout_frame -----------------------------


def test_enforce_holdout_window_keeps_only_start_end_inclusive() -> None:
    df = pl.DataFrame(
        {"date": [DEV_END, HOLDOUT_START, D_HOLD_1, HOLDOUT_END, date(2026, 7, 1)], "v": range(5)}
    )
    kept = enforce_holdout_window(df)
    assert kept["date"].to_list() == [HOLDOUT_START, D_HOLD_1, HOLDOUT_END]


def test_enforce_holdout_window_missing_date_column_raises() -> None:
    with pytest.raises(ValueError, match="date"):
        enforce_holdout_window(pl.DataFrame({"v": [1, 2]}))


def _write_features_and_labels(root: DataRoot, dates: list[date], *, n_symbols: int = 25) -> None:
    all_cols = [*MODEL_FEATURES, *OLS3_FEATURES]
    rows = []
    for d in dates:
        for s in range(n_symbols):
            row = {"date": d, "symbol": f"S{s:03d}"}
            for name in all_cols:
                row[name] = float(s)
                row[f"{name}_isna"] = False
            rows.append(row)
    write_dataset(pl.DataFrame(rows), root, "us_features_v1", manifest={})

    labels_rows = []
    for d in dates:
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
                    "close_reason": "other" if s == 0 else None,
                }
            )
    write_dataset(pl.DataFrame(labels_rows), root, "us_labels_v1", manifest={})


def test_load_holdout_frame_drops_dev_dates(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    _write_features_and_labels(root, [D_DEV_1, D_DEV_2, D_HOLD_1, D_HOLD_2])

    holdout = load_holdout_frame(root, "us_features_v1")

    assert set(holdout["date"].unique().to_list()) == {D_HOLD_1, D_HOLD_2}


def test_load_holdout_frame_raises_if_only_dev_dates_present(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    _write_features_and_labels(root, [D_DEV_1, D_DEV_2])
    with pytest.raises(ValueError, match="비어"):
        load_holdout_frame(root, "us_features_v1")


# --- 2. assert_holdout_window — holdout 벽 -----------------------------------


def test_assert_holdout_window_raises_when_dev_date_present() -> None:
    df = pl.DataFrame({"date": [D_DEV_1, D_HOLD_1], "v": [1, 2]})
    with pytest.raises(ValueError, match="개발 구간 날짜가 섞였습니다"):
        assert_holdout_window(df)


def test_assert_holdout_window_raises_when_date_after_end() -> None:
    df = pl.DataFrame({"date": [D_HOLD_1, date(2026, 7, 15)], "v": [1, 2]})
    with pytest.raises(ValueError, match=r"\[.*\] 밖입니다"):
        assert_holdout_window(df)


def test_assert_holdout_window_raises_when_empty() -> None:
    df = pl.DataFrame(schema={"date": pl.Date, "v": pl.Int64})
    with pytest.raises(ValueError, match="비어"):
        assert_holdout_window(df)


def test_assert_holdout_window_passes_for_valid_frame() -> None:
    df = pl.DataFrame({"date": [D_HOLD_1, D_HOLD_2], "v": [1, 2]})
    assert_holdout_window(df)  # 죽지 않으면 통과


# --- 3. assert_no_window_overlap — 마지막 방어선 --------------------------------


def test_assert_no_window_overlap_raises_on_common_date() -> None:
    with pytest.raises(ValueError, match="겹치는 날짜"):
        assert_no_window_overlap([D_DEV_1, D_HOLD_1], [D_HOLD_1, D_HOLD_2])


def test_assert_no_window_overlap_raises_when_train_max_not_before_holdout_min() -> None:
    with pytest.raises(ValueError, match="앞서지 않습니다"):
        assert_no_window_overlap([D_DEV_1, D_HOLD_2], [D_HOLD_1])


def test_assert_no_window_overlap_passes_for_disjoint_ordered_windows() -> None:
    assert_no_window_overlap([D_DEV_1, D_DEV_2], [D_HOLD_1, D_HOLD_2])  # 죽지 않으면 통과


def test_assert_no_window_overlap_empty_lists_do_not_raise() -> None:
    assert_no_window_overlap([], [])


# --- 4. build_holdout_core — 순위 변환 · month_idx -------------------------------


def test_build_holdout_core_adds_rank_and_isna_columns_and_month_idx() -> None:
    n = 25
    features = pl.DataFrame(
        {
            "date": [D_HOLD_1] * n + [D_HOLD_2] * n,
            "symbol": [f"S{s:03d}" for s in range(n)] * 2,
            "feat_a": [float(s) for s in range(n)] * 2,
            "feat_a_isna": [False] * (n * 2),
            "feat_b": [float(s) for s in range(n)] * 2,
            "feat_b_isna": [False] * (n * 2),
        }
    )
    labels = pl.DataFrame(
        {
            "date": [D_HOLD_1] * n + [D_HOLD_2] * n,
            "symbol": [f"S{s:03d}" for s in range(n)] * 2,
            "L0": [0.0] * (n * 2),
            "L1": [0.0] * (n * 2),
            "L2": [0.0] * (n * 2),
            "y_rank": [0.5] * (n * 2),
            "y_up": [True] * (n * 2),
        }
    )

    core = build_holdout_core(features, labels, MODEL_FEATURES)

    for col in ("feat_a_rank", "feat_b_rank", "feat_a_isna", "feat_b_isna", "month_idx"):
        assert col in core.columns
    assert set(core["date"].unique().to_list()) == {D_HOLD_1, D_HOLD_2}
    assert core["month_idx"].n_unique() == 2


def test_build_holdout_core_raises_when_dev_date_leaks_in() -> None:
    """``build_holdout_core``에 넘긴 프레임 자체에 개발 구간 날짜가 있으면
    ``assert_holdout_window``가 죽는다 — 호출부가 ``load_holdout_frame``을
    건너뛰어도 마지막에 걸린다."""
    features = pl.DataFrame(
        {
            "date": [D_DEV_1],
            "symbol": ["S000"],
            "feat_a": [1.0],
            "feat_a_isna": [False],
            "feat_b": [1.0],
            "feat_b_isna": [False],
        }
    )
    labels = pl.DataFrame(
        {
            "date": [D_DEV_1],
            "symbol": ["S000"],
            "L0": [0.0],
            "L1": [0.0],
            "L2": [0.0],
            "y_rank": [0.5],
            "y_up": [True],
        }
    )
    with pytest.raises(ValueError, match="개발 구간 날짜가 섞였습니다"):
        build_holdout_core(features, labels, MODEL_FEATURES)


# --- 5. fit_predict_holdout — 재학습이 정확히 한 번 -------------------------------


def test_fit_predict_holdout_trains_once_on_train_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    n_symbols = 25
    train_dates = [D_DEV_1, D_DEV_2]
    holdout_dates = [D_HOLD_1]

    def _core(dates: list[date]) -> pl.DataFrame:
        rows = []
        for d in dates:
            for s in range(n_symbols):
                rows.append(
                    {
                        "date": d,
                        "symbol": f"S{s:03d}",
                        "feat_a_rank": float(s) / (n_symbols - 1),
                        "feat_a_isna": 0.0,
                        "feat_b_rank": float(s) / (n_symbols - 1),
                        "feat_b_isna": 0.0,
                        "y_rank": float(s) / (n_symbols - 1),
                    }
                )
        return pl.DataFrame(rows)

    train_core = _core(train_dates)
    holdout_core = _core(holdout_dates)

    call_count = 0
    captured: dict[str, np.ndarray] = {}

    def _counting_fit_predict_ridge(params, x_train, y_train, x_valid):
        nonlocal call_count
        call_count += 1
        captured["x_train"] = x_train
        captured["x_valid"] = x_valid
        return np.zeros(x_valid.shape[0])

    monkeypatch.setattr("modeler.us.m7_run.fit_predict_ridge", _counting_fit_predict_ridge)

    preds = fit_predict_holdout(train_core, holdout_core, MODEL_FEATURES, alpha=100.0)

    assert call_count == 1  # walk-forward가 아니다 — 재학습은 한 번뿐이다
    assert captured["x_train"].shape[0] == train_core.height  # holdout 행이 안 섞였다
    assert captured["x_valid"].shape[0] == holdout_core.height
    assert preds.shape[0] == holdout_core.height


# --- 6. build_m4_inputs(개발) + build_holdout_core(holdout) 통합 -----------------


def test_train_and_holdout_windows_never_overlap_end_to_end(tmp_path: Path) -> None:
    """``build_m4_inputs``(개발 전체)와 ``build_holdout_core``(holdout)를 같은
    합성 레이크에서 조립해도 두 프레임의 날짜 집합이 겹치지 않는지 확인한다."""
    root = DataRoot(base=tmp_path)
    all_dates = [D_DEV_1, D_DEV_2, D_HOLD_1, D_HOLD_2]
    _write_features_and_labels(root, all_dates)
    d = root.output / "feature_scan" / "snapshot_date=2026-09-21"
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(
        json.dumps({"model_input_features": {"all": MODEL_FEATURES}}, ensure_ascii=False)
    )

    train_inputs = build_m4_inputs(root)
    features_holdout = load_holdout_frame(root, "us_features_v1")
    labels_holdout = load_holdout_frame(root, "us_labels_v1")
    holdout_core = build_holdout_core(features_holdout, labels_holdout, MODEL_FEATURES)
    holdout_dates = sorted(holdout_core["date"].unique().to_list())

    assert set(train_inputs.dates) == {D_DEV_1, D_DEV_2}
    assert set(holdout_dates) == {D_HOLD_1, D_HOLD_2}
    assert_no_window_overlap(train_inputs.dates, holdout_dates)  # 죽지 않으면 통과

    preds = fit_predict_holdout(train_inputs.core, holdout_core, MODEL_FEATURES, alpha=100.0)
    assert preds.shape[0] == holdout_core.height


# --- 7. decide_branch — 부호만 (``00`` §3.2) -------------------------------------


def _fake_metrics(*, e: float, e_ew: float, i: float, s: float) -> SimpleNamespace:
    return SimpleNamespace(E=e, E_ew=e_ew, I=i, S=s)


def test_decide_branch_c_when_e_ew_nonpositive() -> None:
    result = decide_branch(_fake_metrics(e=1.0, e_ew=0.0, i=1.0, s=1.0))
    assert result.branch == "C"


def test_decide_branch_a_when_all_four_positive() -> None:
    result = decide_branch(_fake_metrics(e=0.1, e_ew=0.1, i=0.1, s=0.1))
    assert result.branch == "A"


def test_decide_branch_b_when_e_ew_positive_but_e_nonpositive() -> None:
    result = decide_branch(_fake_metrics(e=-0.1, e_ew=0.1, i=0.1, s=0.1))
    assert result.branch == "B"


def test_decide_branch_b_when_s_nonpositive() -> None:
    result = decide_branch(_fake_metrics(e=0.1, e_ew=0.1, i=0.1, s=0.0))
    assert result.branch == "B"


def test_decide_branch_to_dict_carries_values() -> None:
    result = decide_branch(_fake_metrics(e=0.1, e_ew=0.2, i=0.3, s=0.4))
    assert result.to_dict() == {"branch": "A", "E": 0.1, "E_ew": 0.2, "I": 0.3, "S": 0.4}


# --- 8. monthly_series_frame — E·E_ew·I·S 조인 -----------------------------------


def test_monthly_series_frame_joins_four_series_on_date() -> None:
    dates = [D_HOLD_1, D_HOLD_2]
    n = 25
    joined = pl.DataFrame(
        {
            "date": dates * n,
            "symbol": [f"S{s:03d}" for s in range(n) for _ in dates],
            "pred": [float(s) for s in range(n) for _ in dates],
            "L0": [0.01 * s for s in range(n) for _ in dates],
            "price_ge_5": [True] * (n * len(dates)),
            "close": [10.0] * (n * len(dates)),
            "sigma_daily": [0.01] * (n * len(dates)),
            "adv_20d": [1_000_000.0] * (n * len(dates)),
        }
    )
    universe_ew = pl.DataFrame({"date": dates, "ew_l0_h21_return": [0.01, 0.02]})
    spy = pl.DataFrame({"date": dates, "spy_h21_return": [0.005, 0.015]})

    monthly = monthly_series_frame(joined, universe_ew=universe_ew, spy=spy)

    assert monthly.columns == ["date", "E", "E_ew", "I", "S"]
    assert monthly["date"].to_list() == dates


# --- 9. closed_by_reason_counts --------------------------------------------------


def test_closed_by_reason_counts_ignores_nulls() -> None:
    df = pl.DataFrame({"close_reason": ["distress_delisted", "other", "other", None, None]})
    counts = closed_by_reason_counts(df)
    assert counts == {"distress_delisted": 1, "other": 2}


def test_closed_by_reason_counts_empty_when_all_null() -> None:
    df = pl.DataFrame({"close_reason": [None, None]}, schema={"close_reason": pl.Utf8})
    assert closed_by_reason_counts(df) == {}
