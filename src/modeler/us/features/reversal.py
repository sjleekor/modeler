"""F2 단기 반전 — ``rev_1w`` · ``max_ret_1m``.

``04_feature_test_plan.md`` §3 정의::

    rev_1w      = adj_close[t] / adj_close[t-5] - 1        # t-5 ~ t 누적수익률
    max_ret_1m  = max(ret[t-20], ..., ret[t])              # 지난 21일 최대 일수익률 (Bali 2011 MAX)

거래일 창은 ``_daily.py``의 관례(종목별 가격 행 순서)를 그대로 쓴다. ``ret``은
``_daily.daily_prices``가 조정 종가로 만든 당일 단순수익률이라 분할 낀 날에도
점프가 없다.

``max_ret_1m``의 롤링 창은 ``window_size=21``로 **당일을 포함한 21개 관측치**를
본다(polars ``rolling_max``의 기본 동작 — "이 행 자신과 그 앞 20개"). 창이 21개를
못 채우면(``min_samples`` 기본값 = ``window_size``) null이고 ``_isna``가 켜진다.

``rev_1w``는 티커 재사용으로 몇 년 전 무관한 회사 가격을 가리킬 수 있어
``_daily.mask_ticker_reuse_gap``으로 무효화한다(``momentum.py`` 참고 — 검산
5번에서 실제로 발견한 문제). ``max_ret_1m``은 창이 5일보다 훨씬 짧아(21거래일 ≈
30일) 같은 문제가 나타날 가능성이 낮지만, 창의 가장 오래된 행(20행 전)까지 같은
기준으로 무효화해 일관성을 둔다.
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

_FEATURES = ("rev_1w", "max_ret_1m")


def add_reversal(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """``panel``의 ``(date, symbol)``에 F2 단기 반전 피쳐 + ``_isna``를 붙인다."""
    daily = daily_prices(lake, symbols=panel_symbols(panel))
    date_col = pl.col("date")

    rev_1w = pl.col("adj_close") / pl.col("adj_close").shift(5).over("symbol") - 1.0
    max_ret_1m = pl.col("ret").rolling_max(window_size=21).over("symbol")

    features = daily.with_columns(
        mask_ticker_reuse_gap(rev_1w, date_col, 5).alias("rev_1w"),
        mask_ticker_reuse_gap(max_ret_1m, date_col, 20).alias("max_ret_1m"),
    )
    return join_features(panel, features, _FEATURES)
