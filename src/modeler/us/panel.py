"""(리밸런스일, 종목) 월간 패널 조립.

패널 1행 = (리밸런스일, 종목). ``universe_daily.in_universe = 1``인 것만 담는다.
조정 가격은 ``prices.py``에서 계산한 것을 그대로 붙인다.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from modeler.us.lake import UsLake
from modeler.us.prices import adjusted_daily

#: ``trading_calendar``에 실제로 있는 거래소는 하나뿐이다 — 2026-09-20 실측
#: (수집 계획 ``03_schema_and_pit.md`` §4.15: "실측 — XNYS · 2011-01-03~2026-12-31
#: · 4,023행"). 여러 거래소가 있었다면 어느 것을 쓸지 정해야 했겠지만, 지금은
#: 고를 필요가 없다. 나중에 표에 다른 거래소가 추가되면 이 상수를 다시 본다.
_EXCHANGE = "XNYS"


def month_first_trading_days(lake: UsLake, start: date, end: date) -> list[date]:
    """``[start, end]`` 구간에서 달마다 첫 거래일.

    ``trading_calendar``를 ``_EXCHANGE``(XNYS)로 필터링하고, 연-월로 묶어 그 안의
    최소 ``date``를 고른다 — 최소값을 쓰기 때문에 신정처럼 월초에 낀 휴장일이
    있어도 자동으로 그다음 거래일이 뽑힌다.
    """
    calendar = lake.scan("trading_calendar").filter(
        (pl.col("exchange") == _EXCHANGE) & (pl.col("date") >= start) & (pl.col("date") <= end)
    )
    result = (
        calendar.with_columns(pl.col("date").dt.strftime("%Y-%m").alias("_year_month"))
        .group_by("_year_month")
        .agg(pl.col("date").min().alias("date"))
        .sort("date")
        .collect()
    )
    return result["date"].to_list()


def build_panel(
    lake: UsLake, *, start: date = date(2018, 9, 7), end: date | None = None
) -> pl.DataFrame:
    """월간 패널을 만든다.

    ``end``를 안 주면 스냅샷이 주는 ``universe_daily``의 최대 ``date``까지다.

    컬럼: ``date, symbol``(키) · ``cik, sic, sic2, mcap_rank, adv_20d,
    exchange``(``universe_daily``) · ``close``(``prices_daily`` 원시 종가) ·
    ``adj_close, adj_volume``(``prices.py``) · ``price_ge_5``(조정 전 종가 >= $5).

    ``sic2``는 ``sic`` 앞 두 자리다. ``sic``이 없으면 ``sic2``도 null로 둔다 —
    "미분류"로 바꾸는 것은 M1(라벨) 몫이다 (``02`` §2 유니버스 SIC 결측 처리 참고).

    조정 가격의 기준 시점 T는 ``adjusted_daily``의 기본값(``prices_daily`` 최대
    ``date``)을 그대로 쓴다 — ``end``와 다르게 둬도 된다. 분할 조정으로 얻는 것은
    "같은 T로 정규화된 계열의 두 시점 사이 비율(수익률)이 T 선택과 무관하다"는
    성질이라, 패널의 ``end``와 T를 굳이 맞출 필요가 없다.
    """
    if end is None:
        end_row = lake.scan("universe_daily").select(pl.col("date").max().alias("d")).collect()
        end = end_row.item()
        if end is None:
            raise ValueError("universe_daily가 비어 있어 end를 정할 수 없습니다.")

    rebalance_dates = month_first_trading_days(lake, start, end)

    universe = (
        lake.scan("universe_daily")
        .filter(pl.col("in_universe") & pl.col("date").is_in(rebalance_dates))
        .select(["date", "symbol", "cik", "sic", "mcap_rank", "adv_20d", "exchange"])
        .with_columns(
            pl.when(pl.col("sic").is_not_null())
            .then(pl.col("sic").str.slice(0, 2))
            .otherwise(None)
            .alias("sic2")
        )
    )

    raw_close = lake.scan("prices_daily").select(
        "date", "symbol", pl.col("close").cast(pl.Float64).alias("close")
    )
    adjusted = adjusted_daily(lake).select("date", "symbol", "adj_close", "adj_volume")

    panel = (
        universe.join(raw_close, on=["date", "symbol"], how="left")
        .join(adjusted, on=["date", "symbol"], how="left")
        # price_ge_5는 조정 전(원시) 종가를 본다 — 02 §4 지표 I의 거래가능
        # 유니버스 정의 그대로다. raw_close가 없는(join 실패) 행은 null로 남는다.
        .with_columns((pl.col("close") >= 5).alias("price_ge_5"))
        .select(
            "date",
            "symbol",
            "cik",
            "sic",
            "sic2",
            "mcap_rank",
            "adv_20d",
            "exchange",
            "close",
            "adj_close",
            "adj_volume",
            "price_ge_5",
        )
        .sort(["date", "symbol"])
    )
    return panel.collect()
