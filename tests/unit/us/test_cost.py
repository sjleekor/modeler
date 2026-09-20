"""``modeler.us.cost`` 단위 테스트."""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.cost import (
    DEFAULT_K,
    DEFAULT_Q_DOLLAR,
    K_GRID,
    MIN_SPREAD,
    Q_GRID,
    TICK_SIZE,
    cost_grid,
    cost_roundtrip,
    daily_volatility,
    impact,
    spread,
)
from modeler.us.lake import UsLake


def _write_snapshot(root: Path, table: str, snapshot_date: str, frame: pl.DataFrame) -> None:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")


@pytest.fixture()
def lake(tmp_path: Path) -> UsLake:
    return UsLake(root=DataRoot(base=tmp_path))


def _write_corp_actions_empty(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        schema={
            "symbol": pl.String,
            "ex_date": pl.Date,
            "kind": pl.String,
            "to_factor": pl.Float64,
            "for_factor": pl.Float64,
        }
    )
    _write_snapshot(tmp_path, "corp_actions", "2026-09-18", frame)


def _write_prices(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )
    _write_snapshot(tmp_path, "prices_daily", "2026-09-18", frame)


# --- spread ------------------------------------------------------------------------


def test_spread_is_tick_over_price_not_one_over_price() -> None:
    """``0.01/price``다 — ``1/price``가 아니다 (계획 검토 V11)."""
    df = pl.DataFrame({"price": [5.0]})
    value = df.select(spread(pl.col("price")).alias("s")).item()
    assert value == pytest.approx(TICK_SIZE / 5.0)
    assert value == pytest.approx(0.002)  # $5 종목 -> 20bp
    assert value != pytest.approx(1.0 / 5.0)  # 1/price였다면 20%가 나왔을 것


def test_spread_has_a_2bp_floor_for_high_priced_stocks() -> None:
    df = pl.DataFrame({"price": [1000.0]})  # 0.01/1000 = 0.1bp < 2bp 하한
    value = df.select(spread(pl.col("price")).alias("s")).item()
    assert value == pytest.approx(MIN_SPREAD)
    assert value == pytest.approx(0.0002)


# --- impact --------------------------------------------------------------------


def test_impact_is_square_root_of_participation() -> None:
    df = pl.DataFrame({"sigma": [0.02], "adv": [100_000_000.0]})
    value = df.select(
        impact(pl.col("sigma"), pl.col("adv"), q_dollar=10_000_000.0, k=0.1).alias("i")
    ).item()
    expected = 0.1 * 0.02 * math.sqrt(10_000_000.0 / 100_000_000.0)
    assert value == pytest.approx(expected)


def test_impact_scales_with_sqrt_not_linear_in_q() -> None:
    """Q를 4배로 늘리면 충격은 2배(제곱근)여야 한다 — 4배가 아니다."""
    df = pl.DataFrame({"sigma": [0.02], "adv": [100_000_000.0]})
    small = df.select(impact(pl.col("sigma"), pl.col("adv"), q_dollar=1_000_000.0, k=0.1)).item()
    large = df.select(impact(pl.col("sigma"), pl.col("adv"), q_dollar=4_000_000.0, k=0.1)).item()
    assert large == pytest.approx(small * 2.0)


# --- cost_roundtrip --------------------------------------------------------------


def test_cost_roundtrip_is_spread_plus_twice_impact() -> None:
    df = pl.DataFrame({"price": [50.0], "sigma": [0.02], "adv": [100_000_000.0]})
    value = df.select(
        cost_roundtrip(
            pl.col("price"), pl.col("sigma"), pl.col("adv"), q_dollar=10_000_000.0, k=0.1
        ).alias("c")
    ).item()
    s = TICK_SIZE / 50.0
    i = 0.1 * 0.02 * math.sqrt(10_000_000.0 / 100_000_000.0)
    assert value == pytest.approx(s + 2 * i)


def test_cost_roundtrip_default_params_match_judgement_values() -> None:
    assert DEFAULT_Q_DOLLAR == 10_000_000.0
    assert DEFAULT_K == 0.1


# --- daily_volatility --------------------------------------------------------------


def test_daily_volatility_matches_manual_rolling_std(tmp_path: Path, lake: UsLake) -> None:
    _write_corp_actions_empty(tmp_path)
    start = date(2020, 1, 1)
    closes = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0, 96.0, 105.0, 100.0]
    rows = [
        {
            "date": start + timedelta(days=i),
            "symbol": "AAA",
            "open": c,
            "high": c,
            "low": c,
            "close": c,
            "volume": 1_000.0,
        }
        for i, c in enumerate(closes)
    ]
    _write_prices(tmp_path, rows)

    sigma = daily_volatility(lake, window=5).collect().sort("date")

    returns = [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes))]
    manual = pl.Series(returns).rolling_std(window_size=5, min_samples=5)
    # sigma_daily의 첫 값(윈도우 준비 전)은 결측이어야 한다.
    assert sigma["sigma_daily"].null_count() > 0
    non_null = sigma.drop_nulls()
    assert non_null.height == manual.drop_nulls().len()
    assert non_null["sigma_daily"].to_list() == pytest.approx(manual.drop_nulls().to_list())


# --- cost_grid -----------------------------------------------------------------


def test_cost_grid_covers_full_q_by_k_grid() -> None:
    df = pl.DataFrame(
        {
            "close": [10.0, 50.0, 100.0],
            "sigma_daily": [0.03, 0.02, 0.01],
            "adv_20d": [5_000_000.0, 50_000_000.0, 500_000_000.0],
        }
    )

    grid = cost_grid(df)

    assert grid.height == len(Q_GRID) * len(K_GRID)
    assert set(grid["q_dollar"].unique().to_list()) == set(Q_GRID)
    assert set(grid["k"].unique().to_list()) == set(K_GRID)
    assert (grid["n"] == 3).all()


def test_cost_grid_ignores_rows_with_missing_or_nonfinite_sigma() -> None:
    df = pl.DataFrame(
        {
            "close": [10.0, 50.0, 100.0],
            "sigma_daily": [0.03, None, math.nan],
            "adv_20d": [5_000_000.0, 50_000_000.0, 500_000_000.0],
        }
    )

    grid = cost_grid(df)

    assert (grid["n"] == 1).all()
    assert grid["mean_cost_roundtrip"].is_not_nan().all()


def test_cost_grid_higher_k_and_q_raise_mean_cost() -> None:
    df = pl.DataFrame(
        {"close": [10.0] * 5, "sigma_daily": [0.02] * 5, "adv_20d": [10_000_000.0] * 5}
    )

    grid = cost_grid(df)

    at_default_q = grid.filter(pl.col("q_dollar") == DEFAULT_Q_DOLLAR).sort("k")
    means = at_default_q["mean_cost_roundtrip"].to_list()
    assert means == sorted(means)  # k가 커질수록 비용도 커진다

    at_default_k = grid.filter(pl.col("k") == DEFAULT_K).sort("q_dollar")
    means_q = at_default_k["mean_cost_roundtrip"].to_list()
    assert means_q == sorted(means_q)  # Q가 커질수록 비용도 커진다
