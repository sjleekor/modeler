"""F7 투자·발생액 — ``04_feature_test_plan.md`` §3.

``asset_growth``·``net_issuance``는 "최신 값 / 1년 전 값 − 1"이다. 여기서
1년은 달력 365일이 아니라 최신 보고기간(``end``)보다 300\\~430일 이른 보고기간을
찾는 것이다(``fundamentals_ttm.instant_yoy_pair``) — 회사마다 회계연도 끝이
다르다.
"""

from __future__ import annotations

import polars as pl

from modeler.us.features.fundamentals_ttm import (
    flow_ttm,
    instant_latest,
    instant_yoy_pair,
    safe_ratio,
)
from modeler.us.lake import UsLake


def add_investment(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """F7 피쳐 3개(``asset_growth · accruals · net_issuance``) + ``_isna`` 3개를 붙인다."""
    assets_pair = instant_yoy_pair(panel, lake, "Assets").select(
        "date",
        "symbol",
        pl.col("cur_val").alias("assets_cur"),
        pl.col("prior_val").alias("assets_prior"),
    )
    assets_latest = instant_latest(panel, lake, "Assets").select(
        "date", "symbol", pl.col("value").alias("assets")
    )
    shares_pair = instant_yoy_pair(panel, lake, "EntityCommonStockSharesOutstanding").select(
        "date",
        "symbol",
        pl.col("cur_val").alias("shares_cur"),
        pl.col("prior_val").alias("shares_prior"),
    )
    ni_ttm = flow_ttm(panel, lake, ["NetIncomeLoss"]).select(
        "date", "symbol", pl.col("value").alias("ni_ttm")
    )
    ocf_ttm = flow_ttm(panel, lake, ["NetCashProvidedByUsedInOperatingActivities"]).select(
        "date", "symbol", pl.col("value").alias("ocf_ttm")
    )

    out = (
        panel.select("date", "symbol")
        .join(assets_pair, on=["date", "symbol"], how="left")
        .join(assets_latest, on=["date", "symbol"], how="left")
        .join(shares_pair, on=["date", "symbol"], how="left")
        .join(ni_ttm, on=["date", "symbol"], how="left")
        .join(ocf_ttm, on=["date", "symbol"], how="left")
    )

    out = out.with_columns(
        (safe_ratio(pl.col("assets_cur"), pl.col("assets_prior")) - 1).alias("asset_growth"),
        safe_ratio(pl.col("ni_ttm") - pl.col("ocf_ttm"), pl.col("assets")).alias("accruals"),
        (safe_ratio(pl.col("shares_cur"), pl.col("shares_prior")) - 1).alias("net_issuance"),
    ).with_columns(
        pl.col("asset_growth").is_null().alias("asset_growth_isna"),
        pl.col("accruals").is_null().alias("accruals_isna"),
        pl.col("net_issuance").is_null().alias("net_issuance_isna"),
    )

    feature_cols = [
        "asset_growth",
        "asset_growth_isna",
        "accruals",
        "accruals_isna",
        "net_issuance",
        "net_issuance_isna",
    ]
    return panel.join(
        out.select("date", "symbol", *feature_cols), on=["date", "symbol"], how="left"
    )
