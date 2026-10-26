"""Preprocessing of the next KR model (spec C-4, G-17). Pure polars.

``rank_transform`` — per ``trade_date`` percentile rank over both markets in one cross-section
(the same cross-section as the label ``y``), average rank for ties, ``(rank - 1) / (n - 1)`` in
[0, 1] with ``n`` the non-null count of the column on that date (``n == 1`` -> 0.5). NULL stays
NULL (HistGradientBoosting reads it as missing). Booleans are cast to float first.

``ridge_fill`` — NULL -> 0.5 on already ranked columns (Ridge only).

This is *not* ``modeler.etl.preprocess`` (a model02 code-hash file); nothing is imported from it.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

DATE_COL = "trade_date"
RIDGE_FILL_VALUE = 0.5


def _rank_expr(col: str, date_col: str) -> pl.Expr:
    x = pl.col(col).cast(pl.Float64)
    n = x.count().over(date_col)  # non-null count per date
    rank = x.rank(method="average").over(date_col)  # NULL stays NULL
    return (
        pl.when(n == 1)
        .then(pl.when(x.is_not_null()).then(pl.lit(0.5)).otherwise(None))
        .otherwise((rank - 1.0) / (n - 1).cast(pl.Float64))
        .alias(col)
    )


def rank_transform(
    df: pl.DataFrame, cols: Sequence[str], *, date_col: str = DATE_COL
) -> pl.DataFrame:
    """Replace ``cols`` by their per-date percentile rank (see module docstring)."""
    missing = [c for c in (*cols, date_col) if c not in df.columns]
    if missing:
        raise ValueError(f"columns missing from frame: {missing}")
    if not cols:
        return df
    return df.with_columns([_rank_expr(c, date_col) for c in cols])


def ridge_fill(df: pl.DataFrame, cols: Sequence[str]) -> pl.DataFrame:
    """NULL (and NaN) -> 0.5 on ``cols``; expects rank-transformed columns."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"columns missing from frame: {missing}")
    return df.with_columns(
        [pl.col(c).cast(pl.Float64).fill_nan(None).fill_null(RIDGE_FILL_VALUE) for c in cols]
    )
