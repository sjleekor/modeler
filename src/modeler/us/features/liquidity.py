"""F4 유동성·규모 — ``log_dvol_20`` · ``amihud_20`` · ``turnover_rank`` · ``mcap_rank``.

거래일 창(``log_dvol_20``·``amihud_20``)은 ``_daily.py``의 관례(종목별 가격 행
순서)를 쓴다::

    log_dvol_20 = log( mean_20(adj_dollar_volume) )
    amihud_20   = mean_20( |ret| / adj_dollar_volume )        # Amihud(2002) 비유동성

``adj_dollar_volume``이 0 이하면(거래 정지 등) 그 날의 항을 null로 두고 넘어간다
— 0으로 나누거나 ``log(0)``이 ``-inf``가 되는 것을 막는다.

``turnover_rank``는 ``midas_security_daily.turn_rank``(``lake.scan``이 이미
``security_type = 'Stock'``으로 걸러 준다 — ``lake.py`` ``_clean_midas_security_daily``)를
``date``로 그대로 맞춘 값이다. 원천 컬럼명이 ``ticker``라 ``symbol``로 바꿔 조인한다.

**``mcap_rank``는 새로 계산하지 않는다.** ``panel.build_panel``이 이미
``universe_daily.mcap_rank``를 패널에 붙여 준다 — 여기서는 ``_isna`` 플래그만
더한다. **절대 시총 피쳐는 만들지 않는다**(Y6) — 상장주식수가 분기 계단이라
분모로 쓰면 급락처럼 보인다.

``log_dvol_20``·``amihud_20``도 창이 티커 재사용 공백을 건너뛰면 무의미해진다
(``momentum.py`` 참고) — ``_daily.mask_ticker_reuse_gap``으로 창의 가장 오래된
행까지 무효화한다.
"""

from __future__ import annotations

import polars as pl

from modeler.us.features._daily import (
    daily_prices,
    mask_ticker_reuse_gap,
    panel_symbols,
)
from modeler.us.lake import UsLake

_FEATURES = ("log_dvol_20", "amihud_20", "turnover_rank", "mcap_rank")


def add_liquidity(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """``panel``의 ``(date, symbol)``에 F4 유동성·규모 피쳐 + ``_isna``를 붙인다.

    ``panel``에 ``mcap_rank`` 컬럼이 이미 있어야 한다(``panel.build_panel``의
    산출물이면 있다).
    """
    daily = daily_prices(lake, symbols=panel_symbols(panel))
    date_col = pl.col("date")
    dvol = pl.col("adj_dollar_volume")
    illiq_daily = pl.when(dvol > 0).then(pl.col("ret").abs() / dvol).otherwise(None)
    positive_dvol = pl.when(dvol > 0).then(dvol).otherwise(None)

    price_features = (
        daily.with_columns(
            positive_dvol.rolling_mean(window_size=20).over("symbol").alias("_dvol_mean_20"),
            illiq_daily.rolling_mean(window_size=20).over("symbol").alias("_amihud_20"),
        )
        .with_columns(
            pl.when(pl.col("_dvol_mean_20") > 0)
            .then(pl.col("_dvol_mean_20").log())
            .otherwise(None)
            .alias("_log_dvol_20")
        )
        .with_columns(
            mask_ticker_reuse_gap(pl.col("_log_dvol_20"), date_col, 19).alias("log_dvol_20"),
            mask_ticker_reuse_gap(pl.col("_amihud_20"), date_col, 19).alias("amihud_20"),
        )
        .select("date", "symbol", "log_dvol_20", "amihud_20")
    )

    midas = lake.scan("midas_security_daily").select(
        "date", pl.col("ticker").alias("symbol"), pl.col("turn_rank").alias("turnover_rank")
    )

    joined = (
        panel.lazy()
        .join(price_features, on=["date", "symbol"], how="left")
        .join(midas, on=["date", "symbol"], how="left")
    )
    isna_flags = [pl.col(c).is_null().alias(f"{c}_isna") for c in _FEATURES]
    return joined.with_columns(isna_flags).collect()
