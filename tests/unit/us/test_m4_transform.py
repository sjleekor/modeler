"""``modeler.us.m4_transform`` 단위 테스트 — 전부 합성 데이터다.

``03_model_candidates.md`` §4 입력 처리("피쳐를 그날 횡단면 순위 [0,1]로 →
결측은 0.5 + ``_isna`` 플래그")를 검사한다(``06_execution_steps.md`` M4 §7
지시 "순위 변환 · 결측 0.5 처리").
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from modeler.us.m4_transform import (
    MISSING_FILL,
    cross_sectional_percentile,
    design_matrix_columns,
    rank_transform,
    to_design_arrays,
)

D1 = date(2020, 1, 2)
D2 = date(2020, 2, 3)


# --- 1. rank_transform ---------------------------------------------------------


def test_rank_transform_percentile_formula_no_missing() -> None:
    """4개 이름, 결측 없음: (rank(min)-1)/(n-1) — SQL PERCENT_RANK()와 같다."""
    df = pl.DataFrame(
        {
            "date": [D1] * 4,
            "symbol": ["A", "B", "C", "D"],
            "x": [10.0, 40.0, 20.0, 30.0],
            "x_isna": [False, False, False, False],
        }
    )
    out = rank_transform(df, ["x"])
    got = dict(zip(out["symbol"].to_list(), out["x_rank"].to_list(), strict=True))
    # 순위 1,4,2,3 -> (0,3,1,2)/3
    assert got == pytest.approx({"A": 0.0, "B": 1.0, "C": 1 / 3, "D": 2 / 3})


def test_rank_transform_missing_filled_with_missing_fill() -> None:
    df = pl.DataFrame(
        {
            "date": [D1] * 3,
            "symbol": ["A", "B", "C"],
            "x": [10.0, None, 30.0],
            "x_isna": [False, True, False],
        }
    )
    out = rank_transform(df, ["x"])
    got = dict(zip(out["symbol"].to_list(), out["x_rank"].to_list(), strict=True))
    assert got["B"] == MISSING_FILL == 0.5
    # 비결측 둘끼리는 percentile [0,1] 그대로 (n_present=2 -> denom=1)
    assert got["A"] == 0.0
    assert got["C"] == 1.0


def test_rank_transform_ranks_within_date_group_only() -> None:
    """다른 날짜의 값이 그 날의 순위에 섞이면 안 된다."""
    df = pl.DataFrame(
        {
            "date": [D1, D1, D2, D2],
            "symbol": ["A", "B", "A", "B"],
            "x": [1.0, 2.0, 100.0, 200.0],
            "x_isna": [False, False, False, False],
        }
    )
    out = rank_transform(df, ["x"])
    # 두 날짜 모두 "그날 최솟값"이 0.0, "그날 최댓값"이 1.0이어야 한다 — 날짜를
    # 안 갈랐다면 D1의 값이 D2보다 작아 전부 0 근처로 뭉친다.
    day1 = out.filter(pl.col("date") == D1).sort("symbol")["x_rank"].to_list()
    day2 = out.filter(pl.col("date") == D2).sort("symbol")["x_rank"].to_list()
    assert day1 == [0.0, 1.0]
    assert day2 == [0.0, 1.0]


def test_rank_transform_all_missing_in_group_still_fills() -> None:
    """그날 전부 결측이면 denom=n-1(0 방지 clip)로도 값이 전부 MISSING_FILL이다."""
    df = pl.DataFrame(
        {
            "date": [D1, D1],
            "symbol": ["A", "B"],
            "x": [None, None],
            "x_isna": [True, True],
        },
        schema={"date": pl.Date, "symbol": pl.Utf8, "x": pl.Float64, "x_isna": pl.Boolean},
    )
    out = rank_transform(df, ["x"])
    assert out["x_rank"].to_list() == [MISSING_FILL, MISSING_FILL]


def test_rank_transform_original_and_isna_columns_survive() -> None:
    df = pl.DataFrame(
        {
            "date": [D1, D1],
            "symbol": ["A", "B"],
            "x": [1.0, None],
            "x_isna": [False, True],
        }
    )
    out = rank_transform(df, ["x"])
    assert "x" in out.columns
    assert "x_isna" in out.columns
    assert out["x"].to_list() == [1.0, None]


def test_rank_transform_missing_isna_column_raises() -> None:
    df = pl.DataFrame({"date": [D1], "symbol": ["A"], "x": [1.0]})
    with pytest.raises(KeyError, match="x_isna"):
        rank_transform(df, ["x"])


def test_rank_transform_handles_multiple_features_independently() -> None:
    df = pl.DataFrame(
        {
            "date": [D1, D1, D1],
            "symbol": ["A", "B", "C"],
            "x": [1.0, 2.0, 3.0],
            "x_isna": [False, False, False],
            "y": [30.0, 20.0, 10.0],
            "y_isna": [False, False, False],
        }
    )
    out = rank_transform(df, ["x", "y"])
    assert out.sort("symbol")["x_rank"].to_list() == [0.0, 0.5, 1.0]
    assert out.sort("symbol")["y_rank"].to_list() == [1.0, 0.5, 0.0]  # 반대 순서


def test_rank_transform_ties_use_min_method() -> None:
    df = pl.DataFrame(
        {
            "date": [D1] * 4,
            "symbol": ["A", "B", "C", "D"],
            "x": [1.0, 1.0, 3.0, 4.0],
            "x_isna": [False] * 4,
        }
    )
    out = rank_transform(df, ["x"]).sort("symbol")
    # rank(min): A,B 둘 다 1등 -> (1-1)/3=0.0, C: 3등 -> (3-1)/3=2/3, D: 4등 -> 1.0
    assert out["x_rank"].to_list() == pytest.approx([0.0, 0.0, 2 / 3, 1.0])


def test_rank_transform_boolean_feature() -> None:
    """``sp500_member``처럼 Boolean 피쳐도 순위를 매길 수 있어야 한다."""
    df = pl.DataFrame(
        {
            "date": [D1, D1, D1],
            "symbol": ["A", "B", "C"],
            "x": [True, False, True],
            "x_isna": [False, False, False],
        }
    )
    out = rank_transform(df, ["x"])
    assert out["x_rank"].null_count() == 0


# --- 2. design_matrix_columns / to_design_arrays --------------------------------


def test_design_matrix_columns_order() -> None:
    cols = design_matrix_columns(["a", "b"])
    assert cols == ["a_rank", "b_rank", "a_isna", "b_isna"]


def test_to_design_arrays_shape_and_isna_is_numeric() -> None:
    df = pl.DataFrame(
        {
            "date": [D1, D1],
            "symbol": ["A", "B"],
            "a": [1.0, None],
            "a_isna": [False, True],
        }
    )
    df = rank_transform(df, ["a"])
    x, cols = to_design_arrays(df, ["a"])
    assert cols == ["a_rank", "a_isna"]
    assert x.shape == (2, 2)
    assert x.dtype == np.float64
    assert list(x[:, 1]) == [0.0, 1.0]  # False/True -> 0.0/1.0


# --- 3. cross_sectional_percentile ---------------------------------------------


def test_cross_sectional_percentile_matches_manual_computation() -> None:
    df = pl.DataFrame(
        {"date": [D1] * 4, "symbol": ["A", "B", "C", "D"], "L1": [0.05, -0.02, 0.10, 0.0]}
    )
    out = cross_sectional_percentile(df, "L1")
    got = dict(zip(out["symbol"].to_list(), out["L1_rank"].to_list(), strict=True))
    # 오름차순: B(-0.02)=0, D(0.0)=1, A(0.05)=2, C(0.10)=3 -> /3
    assert got == pytest.approx({"B": 0.0, "D": 1 / 3, "A": 2 / 3, "C": 1.0})


def test_cross_sectional_percentile_groups_by_date() -> None:
    df = pl.DataFrame(
        {
            "date": [D1, D1, D2, D2],
            "symbol": ["A", "B", "A", "B"],
            "L1": [1.0, 2.0, -5.0, -1.0],
        }
    )
    out = cross_sectional_percentile(df, "L1")
    day1 = out.filter(pl.col("date") == D1).sort("symbol")["L1_rank"].to_list()
    day2 = out.filter(pl.col("date") == D2).sort("symbol")["L1_rank"].to_list()
    assert day1 == [0.0, 1.0]
    assert day2 == [0.0, 1.0]


def test_cross_sectional_percentile_custom_out_col() -> None:
    df = pl.DataFrame({"date": [D1, D1], "symbol": ["A", "B"], "L1": [1.0, 2.0]})
    out = cross_sectional_percentile(df, "L1", out_col="l1_rank")
    assert "l1_rank" in out.columns
    assert "L1_rank" not in out.columns
