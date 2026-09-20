"""일별 조정 가격 — ``corp_actions``로 분할 계수를 계산해 읽을 때 적용한다.

계산식은 수집 계획 ``03_schema_and_pit.md`` §2가 정본이다::

    price_adj(t; T)  = close(t)  × Π ( for_factor / to_factor )   단 s 는 t < ex_date(s) <= T
    volume_adj(t; T) = volume(t) × Π ( to_factor / for_factor )   (가격과 반대 방향)

**OHLC 네 컬럼에 같은 계수를 걸고, 거래량에는 역수를 곱한다.** 같은 계수를 둘 다에
걸면 거래대금이 분할계수 제곱만큼 틀어진다 — 계획이 명시적으로 경고한 자리다.

**총수익(``tr_adj``, 배당 재투자)은 여기 넣지 않는다.** ``03`` §2가 "이 식은 아직
검산하지 않았다"고 적어 뒀고, M0 검산 목록(``06`` M0)에도 배당 사례가 없다 — 검산
없이 넣지 않는다는 원칙을 지키면 M0에서는 뺀다. SPY 총수익은 ``02`` §3에 따라 M1이
배당 재투자로 따로 만든다.

이 모듈은 ``panel.py``와 일부러 분리했다 — M2(피쳐)가 모멘텀·변동성 계산에 조정
가격을 그대로 재사용하므로, 패널 조립 로직 안에 묻으면 재사용이 어려워진다.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from modeler.us.lake import UsLake


def _base_date_default(lake: UsLake) -> date:
    """``base_date``를 안 주면 ``prices_daily``의 최대 ``date``를 T로 쓴다."""
    row = lake.scan("prices_daily").select(pl.col("date").max().alias("max_date")).collect()
    value = row.item()
    if value is None:
        raise ValueError("prices_daily가 비어 있어 base_date를 정할 수 없습니다.")
    return value


def split_factors(lake: UsLake, *, base_date: date | None = None) -> pl.LazyFrame:
    """(symbol, date) -> 누적 분할 계수.

    ``date``는 분할 이벤트의 ``ex_date``다. 이 행의 ``split_factor``는 "이 ex_date
    바로 이전(``t < ex_date``)의 가격에 곱해야 하는 누적 계수" — 즉 이 ex_date부터
    ``base_date``(T)까지 일어난 모든 분할(자기 자신 포함)의 ``for_factor/to_factor``
    곱이다. ``base_date``보다 뒤에 일어난 분할은 T 시점엔 알 수 없으므로 애초에
    제외한다.

    ``adjusted_daily``가 이 표를 실제 가격 날짜에 asof 조인해 붙인다 — 각 가격
    날짜 ``t``에는 "``t`` 다음에 오는 가장 이른 ex_date"의 ``split_factor``가
    적용된다(그 값이 이미 그 시점부터 T까지의 전체 곱이기 때문이다).
    """
    if base_date is None:
        base_date = _base_date_default(lake)

    splits = (
        lake.scan("corp_actions")
        .filter((pl.col("kind") == "split") & (pl.col("ex_date") <= base_date))
        .with_columns(
            (pl.col("for_factor").cast(pl.Float64) / pl.col("to_factor").cast(pl.Float64)).alias(
                "_own_factor"
            )
        )
        # 같은 (symbol, ex_date)에 분할이 둘 겹치면(2026-09-20 실측으로는 없었지만
        # 방어적으로) 곱해서 하나로 합친다.
        .group_by(["symbol", "ex_date"])
        .agg(pl.col("_own_factor").product().alias("_own_factor"))
    )

    # symbol별로 ex_date를 **내림차순**으로 두고 누적곱하면, 각 행에는 "자신과
    # 자신보다 늦은 모든 분할"의 곱이 쌓인다 — 그게 바로 그 ex_date 이전 가격에
    # 필요한 계수다 (t < ex_date_i <= ... <= T인 모든 분할이 적용되는 구간).
    cumulative = (
        splits.sort(["symbol", "ex_date"], descending=[False, True])
        .with_columns(pl.col("_own_factor").cum_prod().over("symbol").alias("split_factor"))
        .select(["symbol", "ex_date", "split_factor"])
        .rename({"ex_date": "date"})
        .sort(["symbol", "date"])
    )
    return cumulative


def adjusted_daily(lake: UsLake, *, base_date: date | None = None) -> pl.LazyFrame:
    """일별 조정 가격.

    반환 컬럼: ``date, symbol, close(원시), adj_open, adj_high, adj_low,
    adj_close, adj_volume, adj_dollar_volume``.

    ``adj_open/high/low``는 스펙이 요구한 최소 컬럼(``close, adj_close,
    adj_volume, adj_dollar_volume``)은 아니지만 같은 계수를 곱하는 것뿐이라
    비용이 없고, M2가 고가·저가 기반 피쳐(예: 변동성)를 만들 때 다시 만들지
    않아도 되게 같이 낸다.
    """
    if base_date is None:
        base_date = _base_date_default(lake)

    prices = lake.scan("prices_daily").select(
        "date",
        "symbol",
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64),
    )

    # "t < ex_date"의 엄격 부등호를 join_asof(strategy="forward")의 기본 동작인
    # ">="로 바꾸려고 경계를 하루 당긴다: ex_date - 1일 >= t  <=>  ex_date > t.
    # 둘 다 날짜(하루 단위) 컬럼이라 이 변환이 정확하다 — 경계 검산은
    # tests/unit/us/test_prices.py에 있다.
    breakpoints = (
        split_factors(lake, base_date=base_date)
        .with_columns((pl.col("date") - timedelta(days=1)).alias("_boundary"))
        .select(["symbol", "_boundary", "split_factor"])
        .sort(["symbol", "_boundary"])
    )

    joined = prices.sort(["symbol", "date"]).join_asof(
        breakpoints,
        left_on="date",
        right_on="_boundary",
        by="symbol",
        strategy="forward",
    )
    # 매칭되는 미래 분할이 없으면(마지막 분할 이후, 또는 분할이 아예 없는 종목)
    # 계수는 1 — 조정할 것이 없다.
    joined = joined.with_columns(pl.col("split_factor").fill_null(1.0))

    adjusted = joined.with_columns(
        (pl.col("open") * pl.col("split_factor")).alias("adj_open"),
        (pl.col("high") * pl.col("split_factor")).alias("adj_high"),
        (pl.col("low") * pl.col("split_factor")).alias("adj_low"),
        (pl.col("close") * pl.col("split_factor")).alias("adj_close"),
        # 거래량은 반대 방향이다 — 같은 계수를 걸면 거래대금이 분할계수 제곱만큼
        # 틀어진다.
        (pl.col("volume") / pl.col("split_factor")).alias("adj_volume"),
    ).with_columns((pl.col("adj_close") * pl.col("adj_volume")).alias("adj_dollar_volume"))

    return adjusted.select(
        "date",
        "symbol",
        "close",
        "adj_open",
        "adj_high",
        "adj_low",
        "adj_close",
        "adj_volume",
        "adj_dollar_volume",
    )
