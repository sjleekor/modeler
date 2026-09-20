"""F13 옵션 IV — ``iv_rank`` · ``iv_hv_spread`` · ``iv_isna``.

``volatility_daily``는 커버리지가 유니버스의 절반이 안 된다(약 1,600종목,
``01_data_readiness.md`` §1) — 대형·유동 종목에 쏠려 있다(Y11). **``iv_isna``
자체가 정보다**(유동성 프록시)라 ``04`` §3이 이것을 개별 피쳐로 사전등록했다 —
다른 피쳐의 일반 ``_isna`` 관례와 별개로 이름 그대로 낸다.

**``date``가 매 거래일 찍히는 표가 아니다.** 2026-09-20 실측으로 보고 간격이
2019년엔 거의 매주(7일), 2020\\~2023년엔 이틀\\~사흘, 2024년 이후엔 하루\\~이틀로
좁아진다(``p99`` 간격 **7일**, 중앙값 2일). 그래서 패널의 리밸런스일과 정확히
같은 날짜에 값이 있는 경우가 드물어 **정확히 일치하는 날짜로 조인하면 실제보다
훨씬 결측이 많아진다** — 실제로 이 방식으로 처음 짰을 때 표본 구간(2018-09\\~
2019-03) 전체가 100% 결측으로 나왔다. 그래서 ``short_interest``의 "다음 보고
전까지 forward-fill" 관례(``01`` §6)를 그대로 따라 **가장 최근 보고(``date <=
t``)를 그 다음 보고가 나올 때까지 쓴다** — ``join_asof(strategy="backward")``.

**단, 무제한으로 끌어오지 않는다.** 실측 최댓값이 **2,125일**(종목이 몇 년째
끊겼다가 다시 보고되는 경우)이라 그대로 두면 몇 년 전 값을 최신인 것처럼 쓰게
된다. `tolerance=30일`을 건다 — p99(7일)의 4배가 넘어 정상적인 보고 간격은 다
덮고, 연 단위로 끊긴 종목만 걸러 null(그리고 ``iv_isna``)로 남긴다. **이
30일이라는 값은 04 문서에 없다 — 이번에 실측(p99=7일)을 보고 정한 이 모듈만의
판단이다.** 다르게 잡아야 한다면 여기 상수(``_TOLERANCE``) 하나만 바꾸면 된다.

``iv_rank = (iv_current - iv_year_low) / (iv_year_high - iv_year_low)``. 이
계산은 ``volatility_daily``의 각 보고 시점 자기 행 값만 쓴다 —
``iv_year_high_date``가 ``date``보다 미래인 행이 0건임을 2026-09-20에 확인했다
(검토 V17, ``01`` §2). forward-fill은 "이 보고가 그 다음 보고 전까지 최선의
추정값"이라는 뜻이지, 미래 보고를 앞당겨 쓰는 것이 아니다 — PIT 단위 테스트로
박아 둔다.

``iv_hv_spread = iv_current - hv_20``. **``hv_20``은 ``volatility_daily``의
``hv_current``가 아니라 이 모듈이 직접 계산한다** — 소스 문서(수집 계획 ``03``
§4.6)가 ``hv_current``의 계산 창을 명시하지 않아 ``iv_current``(연율화된 decimal)와
단위가 맞는다는 보장이 없다. 대신 ``_daily.py``의 조정 종가 수익률로 20일 실현
변동성을 만들고 연율화(``sqrt(252)``)해 ``iv_current``와 같은 눈금으로 맞춘다 —
F3의 ``rv_20``(연율화 안 함, 순위 검정이라 상수배가 무의미)과는 별도로 계산한다.
``hv_20``은 매 거래일 값이 있으므로(가격은 결측이 거의 없다) 이쪽은 forward-fill
없이 패널 날짜에 정확히 일치하는 값을 쓴다 — 오래된 쪽(``iv_current``)만
forward-fill한다. 그 대신 ``hv_20``도 다른 F1\\~F4 피쳐와 같은 티커 재사용 위험이
있어(``momentum.py`` 참고) ``_daily.mask_ticker_reuse_gap``으로 무효화한다.
"""

from __future__ import annotations

import math
from datetime import timedelta

import polars as pl

from modeler.us.features._daily import daily_prices, mask_ticker_reuse_gap, panel_symbols
from modeler.us.lake import UsLake

_TRADING_DAYS_PER_YEAR = 252
#: 04에 없는 이 모듈만의 판단이다 — 위 docstring의 실측(p99=7일, 최대 2,125일) 근거.
_TOLERANCE = timedelta(days=30)
_FEATURES = ("iv_rank", "iv_hv_spread")


def add_options_iv(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """``panel``의 ``(date, symbol)``에 F13 옵션 IV 피쳐 + ``_isna``를 붙인다."""
    symbols = panel_symbols(panel)

    vol_reports = (
        lake.scan("volatility_daily")
        .filter(pl.col("symbol").is_in(symbols))
        .select(
            "date",
            "symbol",
            pl.col("iv_current").cast(pl.Float64),
            pl.col("iv_year_high").cast(pl.Float64),
            pl.col("iv_year_low").cast(pl.Float64),
        )
        .with_columns(
            pl.when(pl.col("iv_year_high") > pl.col("iv_year_low"))
            .then(
                (pl.col("iv_current") - pl.col("iv_year_low"))
                / (pl.col("iv_year_high") - pl.col("iv_year_low"))
            )
            .otherwise(None)
            .alias("iv_rank")
        )
        .select("date", "symbol", "iv_current", "iv_rank")
        .sort(["symbol", "date"])
    )

    panel_keys = panel.lazy().select("date", "symbol").sort(["symbol", "date"])

    # 가장 최근 보고(date <= t, tolerance 안)를 forward-fill한다 — 04에 없는 판단.
    forward_filled = panel_keys.join_asof(
        vol_reports,
        on="date",
        by="symbol",
        strategy="backward",
        tolerance=_TOLERANCE,
    )

    hv_20_raw = pl.col("ret").rolling_std(window_size=20).over("symbol") * math.sqrt(
        _TRADING_DAYS_PER_YEAR
    )
    hv_20 = (
        daily_prices(lake, symbols=symbols)
        .with_columns(mask_ticker_reuse_gap(hv_20_raw, pl.col("date"), 19).alias("hv_20"))
        .select("date", "symbol", "hv_20")
    )

    # hv_20은 forward-fill하지 않는다 — 가격은 거의 매 거래일 있으니 t와 정확히
    # 맞는 값을 그대로 쓴다.
    features = forward_filled.join(hv_20, on=["date", "symbol"], how="left").with_columns(
        (pl.col("iv_current") - pl.col("hv_20")).alias("iv_hv_spread")
    )

    joined = panel.lazy().join(
        features.select("date", "symbol", "iv_current", "iv_rank", "iv_hv_spread"),
        on=["date", "symbol"],
        how="left",
    )
    isna_flags = [pl.col(c).is_null().alias(f"{c}_isna") for c in _FEATURES]
    joined = joined.with_columns(
        pl.col("iv_current").is_null().alias("iv_isna"),
        *isna_flags,
    ).drop("iv_current")
    return joined.collect()
