"""F5 밸류 — ``04_feature_test_plan.md`` §3.

분모 시총은 ``fundamentals_ttm.market_cap``(``EntityCommonStockSharesOutstanding``
최신 ``filed <= t`` × 그날 원시 종가)이다. **분기 계단이 있으므로 순위로만
쓴다** — 여기서는 절대값(원 단위 배수)만 만들고, 횡단면 순위화는 검정
단계(``scan.py``)의 몫이다(``07_risks.md`` Y6).

외국발행사(20-F)는 ``fundamentals``에서 이미 빠져 있다(``lake.py``
``_clean_fundamentals``) — 그 종목은 여기 네 피쳐가 전부 결측이 되고
``_isna``가 곧 "외국발행사" 플래그다(``04`` §5).
"""

from __future__ import annotations

import polars as pl

from modeler.us.features.fundamentals_ttm import (
    REVENUE_TAGS,
    flow_ttm,
    instant_latest,
    market_cap,
    safe_ratio,
)
from modeler.us.lake import UsLake


def add_valuation(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """``panel``의 (date, symbol)을 그대로 두고 F5 피쳐 4개 + ``_isna`` 4개를 붙인다.

    ``bm · ep_ttm · cfp_ttm · sp_ttm``.
    """
    mcap = market_cap(panel, lake).select("date", "symbol", "mcap")

    equity = instant_latest(panel, lake, "StockholdersEquity").select(
        "date", "symbol", pl.col("value").alias("equity")
    )
    ni_ttm = flow_ttm(panel, lake, ["NetIncomeLoss"]).select(
        "date", "symbol", pl.col("value").alias("ni_ttm")
    )
    ocf_ttm = flow_ttm(panel, lake, ["NetCashProvidedByUsedInOperatingActivities"]).select(
        "date", "symbol", pl.col("value").alias("ocf_ttm")
    )
    revenue_ttm = flow_ttm(panel, lake, list(REVENUE_TAGS)).select(
        "date", "symbol", pl.col("value").alias("revenue_ttm")
    )

    out = (
        panel.select("date", "symbol")
        .join(mcap, on=["date", "symbol"], how="left")
        .join(equity, on=["date", "symbol"], how="left")
        .join(ni_ttm, on=["date", "symbol"], how="left")
        .join(ocf_ttm, on=["date", "symbol"], how="left")
        .join(revenue_ttm, on=["date", "symbol"], how="left")
    )

    out = out.with_columns(
        safe_ratio(pl.col("equity"), pl.col("mcap")).alias("bm"),
        safe_ratio(pl.col("ni_ttm"), pl.col("mcap")).alias("ep_ttm"),
        safe_ratio(pl.col("ocf_ttm"), pl.col("mcap")).alias("cfp_ttm"),
        safe_ratio(pl.col("revenue_ttm"), pl.col("mcap")).alias("sp_ttm"),
    ).with_columns(
        pl.col("bm").is_null().alias("bm_isna"),
        pl.col("ep_ttm").is_null().alias("ep_ttm_isna"),
        pl.col("cfp_ttm").is_null().alias("cfp_ttm_isna"),
        pl.col("sp_ttm").is_null().alias("sp_ttm_isna"),
    )

    feature_cols = [
        "bm",
        "bm_isna",
        "ep_ttm",
        "ep_ttm_isna",
        "cfp_ttm",
        "cfp_ttm_isna",
        "sp_ttm",
        "sp_ttm_isna",
    ]
    return panel.join(
        out.select("date", "symbol", *feature_cols), on=["date", "symbol"], how="left"
    )
