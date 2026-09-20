"""F6 수익성 — ``04_feature_test_plan.md`` §3.

``GrossProfit`` 태그는 유니버스 행 기준 커버리지가 31.2%뿐이다(M0 실측,
``01_data_readiness.md`` §3.1) — ``04`` §5의 "결측률 > 50%면 모델 입력에서
뺀다"에는 안 걸리지만 낮다. ``Revenues − CostOfRevenue`` 폴백을 TTM 레벨에서
합친다(``coalesce``) — 회사가 어느 쪽을 쓰는지는 filer마다 고정이므로 TTM
레벨 결합이 분기 레벨과 결과가 같다. 폴백 전후 커버리지는 이 모듈이 아니라
보고서 §4에 실측치를 적는다.
"""

from __future__ import annotations

import polars as pl

from modeler.us.features.fundamentals_ttm import (
    REVENUE_TAGS,
    flow_ttm,
    instant_latest,
    safe_ratio,
)
from modeler.us.lake import UsLake


def add_profitability(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """F6 피쳐 4개(``roa_ttm · roe_ttm · gpa · opm_ttm``) + ``_isna`` 4개를 붙인다."""
    assets = instant_latest(panel, lake, "Assets").select(
        "date", "symbol", pl.col("value").alias("assets")
    )
    equity = instant_latest(panel, lake, "StockholdersEquity").select(
        "date", "symbol", pl.col("value").alias("equity")
    )
    ni_ttm = flow_ttm(panel, lake, ["NetIncomeLoss"]).select(
        "date", "symbol", pl.col("value").alias("ni_ttm")
    )
    revenue_ttm = flow_ttm(panel, lake, list(REVENUE_TAGS)).select(
        "date", "symbol", pl.col("value").alias("revenue_ttm")
    )
    gross_profit_ttm = flow_ttm(panel, lake, ["GrossProfit"]).select(
        "date", "symbol", pl.col("value").alias("gross_profit_ttm")
    )
    cost_of_revenue_ttm = flow_ttm(panel, lake, ["CostOfRevenue"]).select(
        "date", "symbol", pl.col("value").alias("cost_of_revenue_ttm")
    )
    operating_income_ttm = flow_ttm(panel, lake, ["OperatingIncomeLoss"]).select(
        "date", "symbol", pl.col("value").alias("operating_income_ttm")
    )

    out = (
        panel.select("date", "symbol")
        .join(assets, on=["date", "symbol"], how="left")
        .join(equity, on=["date", "symbol"], how="left")
        .join(ni_ttm, on=["date", "symbol"], how="left")
        .join(revenue_ttm, on=["date", "symbol"], how="left")
        .join(gross_profit_ttm, on=["date", "symbol"], how="left")
        .join(cost_of_revenue_ttm, on=["date", "symbol"], how="left")
        .join(operating_income_ttm, on=["date", "symbol"], how="left")
    )

    out = out.with_columns(
        # GrossProfit이 있으면 그것을, 없으면 매출 - 매출원가(둘 다 TTM)로
        # 대신한다(``01`` §6). coalesce는 첫 인자가 null일 때만 다음으로 간다.
        pl.coalesce(
            pl.col("gross_profit_ttm"),
            pl.col("revenue_ttm") - pl.col("cost_of_revenue_ttm"),
        ).alias("gp_ttm")
    )

    out = out.with_columns(
        safe_ratio(pl.col("ni_ttm"), pl.col("assets")).alias("roa_ttm"),
        safe_ratio(pl.col("ni_ttm"), pl.col("equity")).alias("roe_ttm"),
        safe_ratio(pl.col("gp_ttm"), pl.col("assets")).alias("gpa"),
        safe_ratio(pl.col("operating_income_ttm"), pl.col("revenue_ttm")).alias("opm_ttm"),
    ).with_columns(
        pl.col("roa_ttm").is_null().alias("roa_ttm_isna"),
        pl.col("roe_ttm").is_null().alias("roe_ttm_isna"),
        pl.col("gpa").is_null().alias("gpa_isna"),
        pl.col("opm_ttm").is_null().alias("opm_ttm_isna"),
    )

    feature_cols = [
        "roa_ttm",
        "roa_ttm_isna",
        "roe_ttm",
        "roe_ttm_isna",
        "gpa",
        "gpa_isna",
        "opm_ttm",
        "opm_ttm_isna",
    ]
    return panel.join(
        out.select("date", "symbol", *feature_cols), on=["date", "symbol"], how="left"
    )
