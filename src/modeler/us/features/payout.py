"""F8 배당·자사주 — ``04_feature_test_plan.md`` §3.

``div_yield``는 재무제표 표가 아니라 ``corp_actions``의 배당 이벤트(``ex_date``,
``amount`` = 주당 현금배당)를 쓴다. PIT는 ``ex_date <= t``만이면 된다 —
배당락일은 공개 즉시 알려지는 정보라 ``01_data_readiness.md``의 다른 표처럼
별도 공표 지연을 더할 필요가 없다(F8 PIT 낮음, ``04`` §3).

**무배당은 결측이 아니다.** 지난 12개월 배당이 0건이면 ``div_yield`` 는 0이고
``div_yield_isna``는 False다 — "배당을 안 준다"는 그 자체로 정보이지, 값을
몰라서 못 채운 게 아니다. 종가가 없어 분모를 못 만드는 경우만 결측으로 본다.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from modeler.us.features.fundamentals_ttm import flow_ttm, market_cap, safe_ratio
from modeler.us.lake import UsLake

#: 지난 12개월 창. 달력 365일 — corp_actions의 ex_date는 공표일 자체라
#: fundamentals처럼 회계연도별 유연한 창(300~430일)이 필요 없다.
_TRAILING_DAYS = 365


def _dividend_ttm(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """(date, symbol)별 지난 12개월(``ex_date <= t`` ∧ ``ex_date > t - 365일``) 배당 합."""
    dividends = (
        lake.scan("corp_actions")
        .filter(pl.col("kind") == "dividend")
        .select("symbol", "ex_date", pl.col("amount").cast(pl.Float64))
        .collect()
    )
    symbols = panel.select("symbol").unique()
    dividends = dividends.join(symbols, on="symbol", how="inner")

    candidates = panel.select("date", "symbol").join(dividends, on="symbol", how="inner")
    candidates = candidates.filter(
        (pl.col("ex_date") <= pl.col("date"))
        & (pl.col("ex_date") > pl.col("date") - timedelta(days=_TRAILING_DAYS))
    )
    summed = candidates.group_by(["date", "symbol"]).agg(pl.col("amount").sum().alias("div_sum"))

    out = panel.select("date", "symbol").join(summed, on=["date", "symbol"], how="left")
    # 배당 이벤트가 창 안에 하나도 없으면 0 — 결측이 아니라 "무배당"이다.
    return out.with_columns(pl.col("div_sum").fill_null(0.0))


def add_payout(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """F8 피쳐 2개(``div_yield · buyback_yield``) + ``_isna`` 2개를 붙인다."""
    div_sum = _dividend_ttm(panel, lake)
    mcap = market_cap(panel, lake).select("date", "symbol", "mcap")
    buyback_ttm = flow_ttm(panel, lake, ["PaymentsForRepurchaseOfCommonStock"]).select(
        "date", "symbol", pl.col("value").alias("buyback_ttm")
    )

    out = (
        panel.select("date", "symbol", "close")
        .join(div_sum, on=["date", "symbol"], how="left")
        .join(mcap, on=["date", "symbol"], how="left")
        .join(buyback_ttm, on=["date", "symbol"], how="left")
    )

    out = out.with_columns(
        safe_ratio(pl.col("div_sum"), pl.col("close")).alias("div_yield"),
        safe_ratio(pl.col("buyback_ttm"), pl.col("mcap")).alias("buyback_yield"),
    ).with_columns(
        # div_yield는 close가 없을 때만 결측이다 — div_sum은 fill_null(0.0)이라
        # 절대 null이 아니다.
        pl.col("close").is_null().alias("div_yield_isna"),
        pl.col("buyback_yield").is_null().alias("buyback_yield_isna"),
    )

    feature_cols = ["div_yield", "div_yield_isna", "buyback_yield", "buyback_yield_isna"]
    return panel.join(
        out.select("date", "symbol", *feature_cols), on=["date", "symbol"], how="left"
    )
