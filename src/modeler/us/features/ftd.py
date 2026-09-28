"""F17 결제실패(FTD) [``ftd_fails``] — ``ftd_share_20`` · ``ftd_days_20`` · ``ftd_chg``.

``us4_flow_features/00_draft.md`` §5.1. 정의::

    ftd_share_20 = 최근 사용 가능한 20결제일 quantity 합 / 같은 20결제일의 거래량(주식 수) 합
    ftd_days_20  = 그 20결제일 중 quantity > 0 인 날 수
    ftd_chg      = ftd_share_20 - 21거래일 전 ftd_share_20

("20 × 20일 평균 거래량"과 "20일 합"은 20이 약분되므로 같은 값이다 — 창 길이가
가변(§"창 유효 일수" 참고)이어도 분자·분모가 같은 창을 쓰는 한 성립한다.)

**PIT — 사용 가능일.** ``settlement_date``가 속한 반월 구간의 끝(1\\~15일이면
15일, 16일\\~말일이면 그 달 말일) + ``LAG_FTD_DAYS``(달력일)부터 그 결제일의
값을 알 수 있다. ``LAG_FTD_DAYS``는 ``short.py``의 ``ASOF_LAG_TRADING_DAYS``
(발행 지연 상수를 코드에 박아 두는 선례)와 같은 자리다 — 다만 FTD는 반월
파일 단위 발행이라 **거래일이 아니라 달력일**로 잰다
(``02_lag_constants.md`` §1·§2: 2021\\~2026년 138개 파일의 중앙값 15일·p90
18일, 20일을 넘긴 것은 3개(2.2%)뿐이라 20일로 정했다).

``liquidity.py``의 ``turnover_rank``(MIDAS 분기 값을 ``date``로 그대로
조인해 발행 지연을 아예 안 본 것)를 **따라 하지 않는다** — 이 모듈은
``short.py``처럼 사용 가능일을 먼저 계산하고 ``join_asof``로 시점을 지킨다.

**"20결제일"은 이 종목이 파일에 나온 날이 아니라 전역 결제일 축이다.** FTD
파일은 결제실패 수량이 **1만 주 이상**인 (날짜, 심볼)만 싣는다(초안 §5.1) —
즉 어떤 심볼이 특정 결제일에 파일에 없다고 해서 그 결제일 자체가 없는
게 아니라 "그 심볼은 그날 대량 실패가 없었다"(=quantity 0)는 뜻이다.
그래서 심볼별 20행이 아니라, ``ftd_fails`` 전체(모든 심볼)에서 뽑은
전역 결제일 축(``_global_settlement_calendar``) 위에서 심볼마다 결측 칸을
0으로 채운 밀집 격자(``add_ftd``의 ``dense``)를 만든 뒤 20결제일 창을 센다
— 그렇게 하지 않으면 결제실패가 드문 종목의 창이 몇 달 전 결제일까지
늘어나 버린다.

**거래량 분모는 ``prices_daily``의 조정 전(raw) 주식 수 거래량이다** —
``quantity``도 조정되지 않은 그 시점 주식 수라 단위를 맞춘다(초안 §5.1).
분할이 창 안에 있을 때 창 전체를 같은 기준으로 맞추는 ``corp_actions``
보정은 **이번 구현에 넣지 않았다** — 결제일 창(20영업일 안팎)에 분할이
겹치는 경우 자체가 드물고, 겹치더라도 영향은 ``ftd_share_20`` 한 창 값의
왜곡에 그쳐(전체 계열을 오염시키지 않는다) 우선순위가 낮다고 판단했다.
보고에 이 결정을 적는다 — 나중에 문제가 되면 ``prices.split_factors``를
가져와 창 안 결제일마다 같은 기준으로 재조정하면 된다.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from modeler.us.features._daily import panel_symbols
from modeler.us.lake import UsLake

#: 창(결제일 수). 창 안 유효 일수가 이 값 미만이면 null — §5 인트로 일반 규칙
#: ("창 안 유효 일수가 10 미만이면 피쳐를 null로 둔다")을 그대로 따른다.
_WINDOW = 20
_MIN_VALID_DAYS = 10

#: ``ftd_chg``가 비교하는 과거 시차(거래일). 초안 §5.1.
_CHG_LAG_TRADING_DAYS = 21

#: 반월 구간 끝 + 이 값(달력일)부터 그 반월의 결제 데이터를 알 수 있다.
#: ``02_lag_constants.md`` §1·§2 (2026-09-28 실측, sj2-server `curl -I` 412개) —
#: 임시값(20)과 같은 값으로 확정됐다.
LAG_FTD_DAYS = 20

#: 밀집 격자·거래량 스캔을 패널 시작일 이전 이 만큼(달력일)부터로 줄인다.
#: 20결제일 창(~28일) + 지연(20일) + chg 이동(21거래일 ~29일)을 넉넉히
#: 덮는 여유값이다(성능 최적화일 뿐 정의에는 영향이 없다 — 패널 최초 시점보다
#: 훨씬 전의 FTD 이력은 그 시점 계산에 쓰이지 않는다).
_LOOKBACK_BUFFER_DAYS = 200

_FEATURES = ("ftd_share_20", "ftd_days_20", "ftd_chg")


def _half_month_end(date_col: pl.Expr) -> pl.Expr:
    """``date_col``이 속한 반월 구간의 끝 — 1~15일이면 15일, 16일~말일이면 그 달 말일."""
    return (
        pl.when(date_col.dt.day() <= 15)
        .then(pl.date(date_col.dt.year(), date_col.dt.month(), 15))
        .otherwise(date_col.dt.month_end())
    )


def _global_settlement_calendar(lake: UsLake, buffer_start: date) -> pl.LazyFrame:
    """``ftd_fails`` 전체(심볼 무관)의 고유 ``settlement_date`` + 사용 가능일.

    심볼로 먼저 필터링한 뒤 이 축을 뽑으면 유니버스 구성에 따라 "20결제일"의
    뜻이 달라진다 — 그래서 전체 표에서 날짜 컬럼만 스캔한다.
    """
    return (
        lake.scan("ftd_fails")
        .select("settlement_date")
        .filter(pl.col("settlement_date") >= buffer_start)
        .unique()
        .sort("settlement_date")
        .with_columns(
            (_half_month_end(pl.col("settlement_date")) + pl.duration(days=LAG_FTD_DAYS)).alias(
                "available_date"
            )
        )
    )


def add_ftd(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """``panel``의 ``(date, symbol)``에 F17 결제실패 피쳐 + ``_isna``를 붙인다."""
    symbols = panel_symbols(panel)
    panel_min_date = panel["date"].min()
    buffer_start = panel_min_date - timedelta(days=_LOOKBACK_BUFFER_DAYS)

    calendar = _global_settlement_calendar(lake, buffer_start).collect()

    ftd = (
        lake.scan("ftd_fails")
        .filter(pl.col("symbol").is_in(symbols) & (pl.col("settlement_date") >= buffer_start))
        .select("settlement_date", "symbol", pl.col("quantity").cast(pl.Float64))
        .collect()
    )

    # 밀집 격자: (심볼 × 전역 결제일) 전부를 채운다. 없는 칸은 quantity 0
    # ("0은 명단에 없음", 초안 §5.1).
    dense = (
        pl.DataFrame({"symbol": symbols})
        .lazy()
        .join(calendar.lazy(), how="cross")
        .join(ftd.lazy(), on=["settlement_date", "symbol"], how="left")
        .with_columns(pl.col("quantity").fill_null(0.0))
    )

    raw_volume = (
        lake.scan("prices_daily")
        .filter(pl.col("symbol").is_in(symbols) & (pl.col("date") >= buffer_start))
        .select(
            pl.col("date").alias("settlement_date"),
            "symbol",
            pl.col("volume").cast(pl.Float64),
        )
    )

    daily = (
        dense.join(raw_volume, on=["settlement_date", "symbol"], how="left")
        .sort(["symbol", "settlement_date"])
        .with_columns(
            pl.col("quantity")
            .rolling_sum(window_size=_WINDOW, min_samples=_MIN_VALID_DAYS)
            .over("symbol")
            .alias("_qty_sum_20"),
            (pl.col("quantity") > 0)
            .cast(pl.Int32)
            .rolling_sum(window_size=_WINDOW, min_samples=_MIN_VALID_DAYS)
            .over("symbol")
            .alias("ftd_days_20"),
            pl.col("volume")
            .rolling_sum(window_size=_WINDOW, min_samples=_MIN_VALID_DAYS)
            .over("symbol")
            .alias("_vol_sum_20"),
        )
        .with_columns(
            pl.when(pl.col("_vol_sum_20") > 0)
            .then(pl.col("_qty_sum_20") / pl.col("_vol_sum_20"))
            .otherwise(None)
            .alias("ftd_share_20")
        )
        .with_columns(
            (
                pl.col("ftd_share_20")
                - pl.col("ftd_share_20").shift(_CHG_LAG_TRADING_DAYS).over("symbol")
            ).alias("ftd_chg")
        )
        .select("symbol", "settlement_date", "available_date", *_FEATURES)
    )

    panel_lf = panel.lazy().sort(["symbol", "date"])
    joined = panel_lf.join_asof(
        daily.sort(["symbol", "available_date"]),
        left_on="date",
        right_on="available_date",
        by="symbol",
        strategy="backward",
    )

    isna_flags = [pl.col(c).is_null().alias(f"{c}_isna") for c in _FEATURES]
    result = joined.with_columns(isna_flags)

    keep = [*panel.columns]
    for c in _FEATURES:
        keep.extend([c, f"{c}_isna"])
    return result.select(keep).sort(["date", "symbol"]).collect()
