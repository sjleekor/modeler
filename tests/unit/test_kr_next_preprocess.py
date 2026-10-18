"""Rank transform and Ridge fill of the next KR model (spec C-4, G-17)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from modeler.models._03_kr_next.preprocess import rank_transform, ridge_fill

D1, D2 = date(2024, 1, 2), date(2024, 1, 3)


def _frame(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["trade_date", "market", "x"], orient="row")


def test_ties_use_average_rank_and_range_is_unit_interval():
    df = _frame([(D1, "KOSPI", 1.0), (D1, "KOSDAQ", 2.0), (D1, "KOSPI", 2.0), (D1, "KOSDAQ", 3.0)])
    out = rank_transform(df, ["x"])
    assert out["x"].to_list() == [0.0, 0.5, 0.5, 1.0]  # (rank-1)/(n-1), both markets one section


def test_null_stays_null_and_n_counts_non_null_only():
    df = _frame(
        [(D1, "KOSPI", 10.0), (D1, "KOSPI", None), (D1, "KOSDAQ", 30.0), (D1, "KOSDAQ", 20.0)]
    )
    out = rank_transform(df, ["x"])
    assert out["x"].to_list() == [0.0, None, 1.0, 0.5]


def test_single_non_null_row_is_half_and_all_null_date_stays_null():
    df = _frame([(D1, "KOSPI", 7.0), (D1, "KOSDAQ", None), (D2, "KOSPI", None)])
    out = rank_transform(df, ["x"])
    assert out["x"].to_list() == [0.5, None, None]


def test_dates_are_independent_and_row_order_kept():
    df = _frame([(D2, "KOSPI", 5.0), (D1, "KOSPI", 1.0), (D2, "KOSDAQ", 9.0), (D1, "KOSDAQ", 0.0)])
    out = rank_transform(df, ["x"])
    assert out["x"].to_list() == [0.0, 1.0, 1.0, 0.0]
    assert out["trade_date"].to_list() == df["trade_date"].to_list()


def test_boolean_is_cast_to_float_first():
    df = pl.DataFrame(
        {"trade_date": [D1] * 4, "b": [True, False, None, True]}, schema_overrides={"b": pl.Boolean}
    )
    out = rank_transform(df, ["b"])
    assert out["b"].dtype == pl.Float64
    # non-null values: True, False, True -> ranks 2.5, 1, 2.5 -> 0.75, 0, 0.75
    assert out["b"].to_list() == [0.75, 0.0, None, 0.75]


def test_missing_column_is_an_error():
    with pytest.raises(ValueError):
        rank_transform(_frame([(D1, "KOSPI", 1.0)]), ["nope"])


def test_ridge_fill_replaces_null_and_nan_with_half():
    df = pl.DataFrame({"a": [0.1, None, float("nan")], "b": [None, 1.0, 0.0]})
    out = ridge_fill(df, ["a", "b"])
    assert out["a"].to_list() == [0.1, 0.5, 0.5]
    assert out["b"].to_list() == [0.5, 1.0, 0.0]
