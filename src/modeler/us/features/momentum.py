"""F1 모멘텀 — ``mom_12_1`` · ``mom_6_1`` · ``mom_1m``.

``04_feature_test_plan.md`` §3 정의 그대로다. 거래일 창은 ``_daily.py``의 관례대로
**종목별 가격 행 순서**로 센다 — 그 근거는 ``_daily.py`` 모듈 docstring에 있다.

세 피쳐 다 조정 종가(``adj_close``)의 비율이다. t−21을 경계로 두는 것은
Jegadeesh-Titman 표준 구성 그대로 — 최근 한 달(t−21~t)의 단기 반전 효과가
장기 모멘텀 신호에 섞이지 않게 뺀다::

    mom_12_1 = adj_close[t-21] / adj_close[t-252] - 1
    mom_6_1  = adj_close[t-21] / adj_close[t-126] - 1
    mom_1m   = adj_close[t]    / adj_close[t-21]  - 1   # 뺀 구간 자체

창을 못 채우면(``shift``가 종목 시계열 시작 밖을 가리키면) null이 되고
``_isna``가 켜진다.

**티커 재사용 경고.** 상장폐지된 티커가 몇 년 뒤 무관한 회사에 재사용되면,
"n 거래일 전"이 그 무관한 회사의 가격을 가리켜 말이 안 되는 값(수백 배 수익률)이
나온다 — 검산 5번에서 ``CBK``(2019\\~2025년 사이 2,360일 공백)로 실제 발견했다.
``_daily.mask_ticker_reuse_gap``으로 세 shift 각각을 무효화한다.
"""

from __future__ import annotations

import polars as pl

from modeler.us.features._daily import (
    daily_prices,
    join_features,
    mask_ticker_reuse_gap,
    panel_symbols,
)
from modeler.us.lake import UsLake

_FEATURES = ("mom_12_1", "mom_6_1", "mom_1m")


def add_momentum(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """``panel``의 ``(date, symbol)``에 F1 모멘텀 피쳐 + ``_isna``를 붙인다."""
    daily = daily_prices(lake, symbols=panel_symbols(panel))
    close = pl.col("adj_close")
    date_col = pl.col("date")
    close_t21 = close.shift(21).over("symbol")

    mom_12_1 = close_t21 / close.shift(252).over("symbol") - 1.0
    mom_6_1 = close_t21 / close.shift(126).over("symbol") - 1.0
    mom_1m = close / close_t21 - 1.0

    features = daily.with_columns(
        mask_ticker_reuse_gap(mom_12_1, date_col, 252).alias("mom_12_1"),
        mask_ticker_reuse_gap(mom_6_1, date_col, 126).alias("mom_6_1"),
        mask_ticker_reuse_gap(mom_1m, date_col, 21).alias("mom_1m"),
    )
    return join_features(panel, features, _FEATURES)
