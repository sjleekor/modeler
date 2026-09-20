"""F15 시장 수준 — ``macro_series``(``realtime_start``) · SPY(``prices_daily``).

``04_feature_test_plan.md`` §3 F15. 네 피쳐: ``mkt_vix`` · ``mkt_term`` ·
``mkt_hy`` · ``mkt_ret_1m``.

**횡단면이 아니다.** 같은 날짜의 모든 종목에 같은 값이 붙는다 — 단독 IC는
정의상 0이라 단독 검정을 하지 않는다(``04`` §3). 그래도 계획대로 만들어
둔다(상호작용·국면별 분해용).

``macro_series``의 as-of 축은 **``realtime_start``**(vintage)다. 같은
``(series_id, date)``에 값이 여럿 있는 것이 정상이고, 기준일 T의 값은
``realtime_start <= T`` 중 최신이다(재무의 ``filed``와 같은 규칙,
``01_data_readiness.md`` §2). ``SP500`` 계열은 ``lake.scan()``이 이미
정제 규칙(``_clean_macro_series``)으로 걸러 준다 — 지수는 SPY 가격으로
대신한다.
"""

from __future__ import annotations

import polars as pl

from modeler.us.features._trading_days import shift_trading_days
from modeler.us.lake import UsLake
from modeler.us.prices import adjusted_daily

_VIX_SERIES = "VIXCLS"
_DGS10_SERIES = "DGS10"
_DGS2_SERIES = "DGS2"
_HY_SERIES = "BAMLH0A0HYM2"
_SPY_SYMBOL = "SPY"

#: ``mkt_ret_1m``의 창(거래일). F1 ``mom_1m``이 쓰는 "21거래일 = 1개월" 관례를
#: 그대로 따른다(``04`` §3 F1).
_RET_1M_WINDOW_TRADING_DAYS = 21


def _series_asof(lake: UsLake, series_id: str, value_alias: str) -> pl.LazyFrame:
    """(realtime_start, value) — 한 계열의 개정 이력. asof backward로 t 시점 값을 고른다.

    같은 ``realtime_start``에 값이 둘 이상이면(같은 날 여러 번 개정) 마지막
    것을 남긴다 — 원천에 그런 사례가 있다는 보고는 없지만 asof join이 키
    중복에 민감해 방어적으로 정리한다.
    """
    return (
        lake.scan("macro_series")
        .filter(pl.col("series_id") == series_id)
        .select("realtime_start", pl.col("value").alias(value_alias))
        .sort("realtime_start")
        .unique(subset=["realtime_start"], keep="last")
    )


def _market_table(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """패널의 고유 날짜마다 F15 네 값을 한 번씩 계산한다 (횡단면이 아니므로 심볼 무관)."""
    dates = panel.select("date").unique().sort("date").lazy()

    vix = _series_asof(lake, _VIX_SERIES, "mkt_vix")
    dgs10 = _series_asof(lake, _DGS10_SERIES, "_dgs10")
    dgs2 = _series_asof(lake, _DGS2_SERIES, "_dgs2")
    hy = _series_asof(lake, _HY_SERIES, "mkt_hy")

    with_vix = dates.join_asof(vix, left_on="date", right_on="realtime_start", strategy="backward")
    with_dgs10 = with_vix.join_asof(
        dgs10, left_on="date", right_on="realtime_start", strategy="backward", suffix="_dgs10"
    )
    with_dgs2 = with_dgs10.join_asof(
        dgs2, left_on="date", right_on="realtime_start", strategy="backward", suffix="_dgs2"
    )
    with_hy = with_dgs2.join_asof(
        hy, left_on="date", right_on="realtime_start", strategy="backward", suffix="_hy"
    )
    with_term = with_hy.with_columns((pl.col("_dgs10") - pl.col("_dgs2")).alias("mkt_term"))

    spy = adjusted_daily(lake).filter(pl.col("symbol") == _SPY_SYMBOL).select("date", "adj_close")
    spy_shifted = shift_trading_days(
        lake, spy, "date", -_RET_1M_WINDOW_TRADING_DAYS, out_col="_date_1m_ago"
    ).rename({"adj_close": "_adj_close_now"})
    spy_past = spy.rename({"date": "_date_1m_ago", "adj_close": "_adj_close_past"})
    spy_ret = (
        spy_shifted.join(spy_past, on="_date_1m_ago", how="left")
        .with_columns(
            (pl.col("_adj_close_now") / pl.col("_adj_close_past") - 1.0).alias("mkt_ret_1m")
        )
        .select("date", "mkt_ret_1m")
    )

    result = with_term.join(spy_ret, on="date", how="left").select(
        "date", "mkt_vix", "mkt_term", "mkt_hy", "mkt_ret_1m"
    )
    return result.collect()


def add_market(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """F15(``mkt_vix`` · ``mkt_term`` · ``mkt_hy`` · ``mkt_ret_1m``)를 붙인다.

    **단독 검정을 하지 않는다** — ``04`` §3, 이 계획 지시문 §1. 붙이기만 하고
    횡단면 랭크는 만들지 않는다(호출자가 상호작용·국면 분해에 쓸 때 필요한
    형태로 다룬다).
    """
    market = _market_table(panel, lake)
    result = (
        panel.lazy()
        .join(market.lazy(), on="date", how="left")
        .with_columns(
            pl.col("mkt_vix").is_null().alias("mkt_vix_isna"),
            pl.col("mkt_term").is_null().alias("mkt_term_isna"),
            pl.col("mkt_hy").is_null().alias("mkt_hy_isna"),
            pl.col("mkt_ret_1m").is_null().alias("mkt_ret_1m_isna"),
        )
        .collect()
    )
    return result.sort(["date", "symbol"])
