"""F1·F2·F3·F4·F13이 같이 쓰는 일별 수익률·롤링 헬퍼.

``04_feature_test_plan.md`` §3의 창(예: ``mom_12_1``의 t−252)은 **거래일** 수다.
이 모듈은 그 거래일을 **trading_calendar가 아니라 종목별 가격 행 순서**로 센다:

``prices_daily``는 그 종목이 실제로 거래된 날에만 행이 있다(수집이 원천의 거래일
데이터를 그대로 담는다 — 수집 계획 ``03_schema_and_pit.md``). 그래서 ``(symbol,
date)`` 오름차순으로 정렬한 뒤 ``.shift(n).over("symbol")``로 얻는 값이 곧
"그 종목 기준 n 거래일 전" 값이다. ``trading_calendar``(XNYS)를 따로 조인해 달력
거래일과 맞추는 방식도 가능하지만, 그러려면 "종목이 시장 전체 거래일에 결측 없이
낀다"는 전제가 필요하고 상장 초기·폐지 직전 구간에서 그 전제가 깨진다. 회귀 계열
피쳐(``idio_vol_60``·``beta_252``)에서는 이 방식이 오히려 이점이다 — 종목이 실제로
거래된 날짜에서만 SPY와 짝을 지어 회귀하게 된다. ``04``도 두 방식 중 어느 것을
쓰라고 못박지 않았다.

각 피쳐 모듈은 ``daily_prices()``로 조정 가격 + 당일 단순수익률을 얻고, 필요하면
``spy_daily_returns()``로 SPY 수익률을 날짜로 붙인 뒤, 롤링 피쳐를 계산해
``join_features()``로 패널에 붙인다.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl

from modeler.us.lake import UsLake
from modeler.us.prices import adjusted_daily

#: SPY를 시장 수익률 대용으로 쓴다 — ``01_data_readiness.md`` §6 (``SP500``
#: 거시 계열은 vintage가 없어 쓰지 않는다).
SPY_SYMBOL = "SPY"

#: 252 거래일 ≈ 365 달력일이 자연스러운 비율이다(주말·공휴일 포함). 그 비율의
#: 1.5배를 상한으로 둔다 — 공휴일이 몰린 정상적인 구간까지 넉넉히 봐주면서도,
#: 몇 달\~몇 년째 티커가 재사용된 구간(아래 참고)은 걸러낸다.
_CALENDAR_DAYS_PER_TRADING_YEAR = 365
_TRADING_DAYS_PER_YEAR = 252
_GAP_SAFETY_FACTOR = 1.5


def _max_calendar_days(n_trading_days: int) -> int:
    return math.ceil(
        n_trading_days
        * _CALENDAR_DAYS_PER_TRADING_YEAR
        / _TRADING_DAYS_PER_YEAR
        * _GAP_SAFETY_FACTOR
    )


def panel_symbols(panel: pl.DataFrame) -> list[str]:
    """``panel``에 나오는 고유 종목 목록. 가격 스캔을 이 종목만으로 줄이는 데 쓴다."""
    return panel["symbol"].unique().to_list()


def daily_prices(lake: UsLake, *, symbols: Sequence[str] | None = None) -> pl.LazyFrame:
    """종목별 (date 오름차순) 정렬된 조정 가격 + 당일 단순수익률.

    반환 컬럼: ``date, symbol, adj_open, adj_high, adj_low, adj_close, adj_volume,
    adj_dollar_volume, ret``. ``ret``은 ``adj_close``의 전일 대비 단순수익률이다 —
    조정 가격을 쓰므로 분할 낀 날에도 점프가 없다(``prices.py`` 경계 검산 참고).
    각 종목의 첫 행은 전일이 없어 ``ret``이 null이다.

    ``symbols``를 주면 그 종목만 남긴다 — 패널에 없는 종목의 29M행 전체를 스캔할
    필요가 없다. ``None``이면 전체 종목이다.
    """
    prices = adjusted_daily(lake)
    if symbols is not None:
        prices = prices.filter(pl.col("symbol").is_in(list(symbols)))
    prices = prices.sort(["symbol", "date"])
    return prices.with_columns(
        (pl.col("adj_close") / pl.col("adj_close").shift(1).over("symbol") - 1.0).alias("ret")
    )


def spy_daily_returns(lake: UsLake) -> pl.LazyFrame:
    """SPY 일별 단순수익률. 반환 컬럼: ``date, ret_spy``."""
    return daily_prices(lake, symbols=[SPY_SYMBOL]).select("date", pl.col("ret").alias("ret_spy"))


def with_market_return(daily: pl.LazyFrame, lake: UsLake) -> pl.LazyFrame:
    """``daily``(date, symbol, ...)에 그날 SPY 수익률(``ret_spy``)을 날짜로 붙인다.

    **``ret_spy``의 null을 0.0으로 채운다.** 처음에는 "왼쪽 조인이라 SPY가 없는
    날엔 null이 되고, 그 창은 ``rolling_*``의 ``min_samples`` 기본값이 알아서
    부족한 것으로 처리하겠지"라고 생각했는데 **틀렸다.** polars의
    ``rolling_std``/``rolling_cov``는 창 안에 null이 **하나만** 있어도 그 창
    전체가 null이 된다(``min_samples`` 기본값이 "``window_size``개 전부
    non-null"이지 "``window_size``개 중 최소 개수"가 아니다) — 즉 하루짜리 결측이
    그 뒤 **``window_size`` 거래일 전체**를 null로 밀어버린다.

    2026-09-20 실측으로 ``SPY``가 다른 종목은 거래한 날에 자기 행이 없는 날이
    **5번** 있다(2011-10-18·2015-06-30·2019-10-02·2022-08-01·2022-09-19) — 이
    5번 때문에 ``beta_252``가 **252 거래일씩** 통째로 결측이 되는 게 실제로 검산
    5번(결측률 표)에서 걸렸다(2019-11\\~2020-09, 2022-08\\~2023-09 두 구간이 거의
    100% 결측). 시장이 그날 거래를 안 한 게 아니라(다른 종목은 다 거래했다) SPY
    행 하나가 원천에 빠진 것뿐이라, 그날의 시장 수익률을 0%로 보는 것이
    "그 날짜를 통째로 버려 뒤 1년을 다 결측 처리하는" 것보다 낫다.
    """
    return daily.join(spy_daily_returns(lake), on="date", how="left").with_columns(
        pl.col("ret_spy").fill_null(0.0)
    )


def mask_ticker_reuse_gap(value: pl.Expr, date_col: pl.Expr, n_trading_days: int) -> pl.Expr:
    """``n_trading_days``개 행 전을 참조하는 계산에서, 그 구간의 달력일 폭이
    비정상적으로 크면(같은 티커가 몇 년 뒤 다른 회사에 재사용된 경우) ``value``를
    null로 무효화한다.

    **실제로 걸린 사례다.** 2026-09-20 실측으로 ``mom_12_1`` 상위 종목을 눈으로
    보는 검산(``06`` M2 완료 판정)에서 ``CBK``가 269배가 아니라 **2019-03-25(종가
    $0.55)에서 2025-10-02(종가 $24.00)로 2,360일 공백을 건너뛴 티커 재사용**임을
    발견했다 — ``AKTS``·``JAN``도 각각 755일·986일 공백을 건너뛰어 같은 문제였다.
    ``_daily.py``의 관례(종목별 가격 행 순서로 거래일을 센다)는 이런 공백이 있는
    종목에서 "n 거래일 전"이 실제로는 몇 년 전의 무관한 회사 가격을 가리키게
    만든다 — trading_calendar로 세도 똑같이 걸릴 문제다(공백 자체가 원천 데이터에
    있다).

    ``n_trading_days``개 행 전 날짜와의 달력일 차이가 ``_max_calendar_days``를
    넘으면 무효화한다. warm-up으로 ``shift``가 애초에 null이면(``date_col`` 뺄셈도
    null) 비교가 null이 되어 ``when``이 ``otherwise``(=None)로 빠진다 — 원래도
    null이었을 값이라 동작이 바뀌지 않는다.
    """
    shifted_date = date_col.shift(n_trading_days).over("symbol")
    gap_days = (date_col - shifted_date).dt.total_days()
    return pl.when(gap_days <= _max_calendar_days(n_trading_days)).then(value).otherwise(None)


def join_features(
    panel: pl.DataFrame,
    daily: pl.LazyFrame,
    feature_cols: Sequence[str],
    *,
    on: Sequence[str] = ("date", "symbol"),
) -> pl.DataFrame:
    """``panel``의 키에 ``daily``의 ``feature_cols``를 왼쪽 조인으로 붙이고,
    피쳐마다 ``<이름>_isna`` 플래그를 만든다.

    ``on``이 ``("date",)``처럼 종목을 뺀 키면(``calendar.py``의 시장 전체 값)
    모든 종목에 같은 값이 방송(broadcast)된다.
    """
    keys = list(on)
    joined = panel.lazy().join(daily.select([*keys, *feature_cols]), on=keys, how="left")
    isna_flags = [pl.col(c).is_null().alias(f"{c}_isna") for c in feature_cols]
    return joined.with_columns(isna_flags).collect()
