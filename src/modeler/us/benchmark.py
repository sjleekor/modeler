"""SPY 총수익(전략 벤치마크)과 유니버스 동일가중(라벨 벤치마크) 비교 — ``02`` §3.

라벨 벤치마크(그날 유니버스 동일가중, ``labels.py``의 ``L1`` 계산에 이미
들어있다)와 전략 벤치마크(SPY 총수익)는 **다른 것**이다. 이 모듈은 SPY
총수익과, 둘의 월별 차이(``ew_minus_spy``)를 낸다.

배당 재투자 식(``tr_adj``)은 수집 계획 ``03_schema_and_pit.md`` §2가 정본이고
그 문서 자체가 "이 식은 아직 검산하지 않았다"고 적어 뒀다 — 이 모듈이 처음
검산한다 (``build_labels.py`` 실행 보고에 배당락일 사례를 남긴다).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import polars as pl

from modeler.us.labels import HORIZON_TRADING_DAYS, trading_day_offsets
from modeler.us.lake import UsLake
from modeler.us.prices import adjusted_daily

_SPY = "SPY"
_DIVIDEND_KIND = "dividend"

_EMPTY_DIVIDEND_FACTOR_SCHEMA = {"date": pl.Date, "dividend_factor": pl.Float64}


def _default_base_date(lake: UsLake) -> date:
    row = lake.scan("prices_daily").select(pl.col("date").max().alias("d")).collect()
    value = row.item()
    if value is None:
        raise ValueError("prices_daily가 비어 있어 base_date를 정할 수 없습니다.")
    return value


def _dividend_factors(lake: UsLake, *, base_date: date) -> pl.LazyFrame:
    """(date, dividend_factor) — SPY 배당 재투자 누적 계수.

    ``prices.split_factors``와 같은 backward 누적곱 구조를 분할 대신 배당에
    적용한다::

        tr_adj(t;T) = price_adj(t;T) × Π (1 − amount(d)/close(prev_trading_day(d)))
                                        d
           단 d는 t < ex_date(d) <= T인 모든 배당 (``03_schema_and_pit.md`` §2)

    ``close(prev_trading_day(d))``는 배당 ex_date 바로 앞 거래일의 **원시** 종가다
    — ``prices_daily``를 ``ex_date``로 정렬해 한 칸 당긴 값(``shift(1)``)으로
    구한다. SPY는 분할이 없어(2026-09-20 실측 ``corp_actions`` 0건) 이 값이
    분할조정 여부와 무관하게 정확하다.
    """
    raw_close = (
        lake.scan("prices_daily")
        .filter(pl.col("symbol") == _SPY)
        .select("date", pl.col("close").cast(pl.Float64))
        .sort("date")
        .collect()
    )
    raw_close = raw_close.with_columns(pl.col("close").shift(1).alias("prev_close"))

    dividends = (
        lake.scan("corp_actions")
        .filter(
            (pl.col("symbol") == _SPY)
            & (pl.col("kind") == _DIVIDEND_KIND)
            & (pl.col("ex_date") <= base_date)
        )
        .select("ex_date", pl.col("amount").cast(pl.Float64))
        .collect()
    )
    if dividends.height == 0:
        return pl.DataFrame(schema=_EMPTY_DIVIDEND_FACTOR_SCHEMA).lazy()

    price_start = raw_close["date"].min()
    joined = dividends.join(raw_close, left_on="ex_date", right_on="date", how="left")

    # price_start 이전 배당은 가격 이력 자체가 없어 prev_close를 못 구한다 —
    # 안전하다: 그 배당들은 관측 구간 전체에 걸리는 상수 배율일 뿐이라 구간
    # 안의 수익률(비율)에는 영향이 없다. price_start 이후인데도 prev_close가
    # 없으면(SPY처럼 유동성 높은 종목에서는 이례적이다) 진짜 문제이므로 던진다.
    joined = joined.filter(pl.col("ex_date") >= price_start)
    missing = joined.filter(pl.col("prev_close").is_null())
    if missing.height:
        raise ValueError(
            "SPY 배당 ex_date에 직전 거래일 종가가 없습니다(가격 이력 범위 안인데도): "
            f"{missing['ex_date'].to_list()}"
        )

    with_yield = joined.with_columns(
        (1 - pl.col("amount") / pl.col("prev_close")).alias("_own_factor")
    ).select("ex_date", "_own_factor")

    cumulative = (
        with_yield.sort("ex_date", descending=True)
        .with_columns(pl.col("_own_factor").cum_prod().alias("dividend_factor"))
        .select(pl.col("ex_date").alias("date"), "dividend_factor")
        .sort("date")
    )
    return cumulative.lazy()


def spy_total_return_daily(lake: UsLake, *, base_date: date | None = None) -> pl.DataFrame:
    """SPY 일별 총수익 계열.

    반환 컬럼: ``date, close(원시), adj_close(분할조정 — SPY는 분할이 없어
    close와 같다), tr_adj(배당 재투자 포함 총수익)``.
    """
    if base_date is None:
        base_date = _default_base_date(lake)

    price = (
        adjusted_daily(lake, base_date=base_date)
        .filter(pl.col("symbol") == _SPY)
        .select("date", "close", "adj_close")
    )
    factors = _dividend_factors(lake, base_date=base_date)

    # "t < ex_date"를 join_asof(strategy="forward")의 ">="로 바꾸려고 경계를
    # 하루 당긴다 — prices.py의 adjusted_daily와 같은 관례.
    breakpoints = (
        factors.with_columns((pl.col("date") - timedelta(days=1)).alias("_boundary"))
        .select("_boundary", "dividend_factor")
        .sort("_boundary")
    )

    joined = price.sort("date").join_asof(
        breakpoints, left_on="date", right_on="_boundary", strategy="forward"
    )
    joined = joined.with_columns(pl.col("dividend_factor").fill_null(1.0))
    return (
        joined.with_columns((pl.col("adj_close") * pl.col("dividend_factor")).alias("tr_adj"))
        .select("date", "close", "adj_close", "tr_adj")
        .collect()
    )


def spy_monthly_return(
    lake: UsLake, dates: Sequence[date], *, base_date: date | None = None
) -> pl.DataFrame:
    """리밸런스일 ``t`` 각각에서 SPY 총수익 h21 수익률 (``t -> t+21`` 거래일).

    ``t+21``이 데이터 밖이면 그 ``t``는 빠진다 — ``labels.build_labels``의
    ``dropped_rebalance_dates``와 같은 기준이다.
    """
    tr = spy_total_return_daily(lake, base_date=base_date)
    tr_map = dict(zip(tr["date"].to_list(), tr["tr_adj"].to_list()))
    offsets = trading_day_offsets(lake, list(dates), HORIZON_TRADING_DAYS)

    rows: list[dict[str, object]] = []
    for t in dates:
        t21 = offsets.get(t)
        if t21 is None or t not in tr_map or t21 not in tr_map:
            continue
        rows.append({"date": t, "spy_h21_return": tr_map[t21] / tr_map[t] - 1})
    return pl.DataFrame(rows, schema={"date": pl.Date, "spy_h21_return": pl.Float64})


def universe_equal_weight_monthly(labels_df: pl.DataFrame) -> pl.DataFrame:
    """라벨 벤치마크 — 그날 유니버스 동일가중 평균 ``L0`` (``02`` §3)."""
    return (
        labels_df.group_by("date").agg(pl.col("L0").mean().alias("ew_l0_h21_return")).sort("date")
    )


def ew_minus_spy_monthly(
    lake: UsLake, labels_df: pl.DataFrame, *, base_date: date | None = None
) -> pl.DataFrame:
    """``ew_minus_spy`` — 유니버스 동일가중이 SPY 총수익을 이긴 만큼, 월별 (``02`` §3).

    한국은 라벨·전략 벤치마크를 구분하지 않아 "시장 대비"가 석 달간 어긋났다
    — 그 차이를 매달 같이 낸다.
    """
    ew = universe_equal_weight_monthly(labels_df)
    spy = spy_monthly_return(lake, ew["date"].to_list(), base_date=base_date)
    return ew.join(spy, on="date", how="inner").with_columns(
        (pl.col("ew_l0_h21_return") - pl.col("spy_h21_return")).alias("ew_minus_spy")
    )
