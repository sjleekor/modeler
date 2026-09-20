"""F11 공매도 — ``short_interest`` · ``short_volume``.

``04_feature_test_plan.md`` §3 F11. 네 피쳐: ``si_ratio`` · ``dtc`` ·
``si_chg`` · ``sv_share_20``.

**PIT의 핵심(``07_risks.md`` Y10).** ``short_interest``는 ``settlement_date``로
자르면 안 된다. FINRA가 결제일 + 7영업일에 공표하고 나스닥이 그날 16:00 ET
뒤 배포한다 — 표에 공표일 컬럼이 없어 ``lake.ASOF_LAG_TRADING_DAYS
["short_interest"]``(10거래일)를 **결제일에 더해서** 자른다. 그냥
``settlement_date <= t``로 자르면 아직 공표 전인 값을 쓰게 된다.

``si_ratio``의 분모는 ``adv_20d``(패널의 20일 평균 **거래대금**, 달러 기준)가
아니다 — ``04`` §3이 "(주수 기준)"이라고 못 박았다. ``short_interest`` 표
자신이 이미 주수 기준 평균거래량(``avg_daily_volume_qty``)을 갖고 있어
그것을 쓴다 — 같은 행 안의 값이라 시점이 어긋날 일도 없다. ``dtc``·``si_chg``도
마찬가지로 FINRA가 이미 계산해 준 ``days_to_cover``·``change_percent``를
그대로 쓴다(``03_schema_and_pit.md`` §4.4) — 재계산하지 않는다.
"""

from __future__ import annotations

import polars as pl

from modeler.us.features._trading_days import shift_trading_days
from modeler.us.lake import ASOF_LAG_TRADING_DAYS, UsLake

_SETTLEMENT_LAG = ASOF_LAG_TRADING_DAYS["short_interest"]

#: ``sv_share_20``의 롤링 창(거래일). 20 미만이면 결측으로 둔다.
_SV_WINDOW = 20


def _eligible_short_interest(lake: UsLake) -> pl.LazyFrame:
    """``short_interest``에 "이 값을 언제부터 알 수 있는가"(``_eligible_date``)를 붙인다.

    ``settlement_date``는 원천이 이미 거래일로 당겨 둔 값이라
    (수집 계획 ``03`` §4.4 "거래일이 아니면 앞으로 당긴다") ``shift_trading_days``의
    전제(입력이 거래일)를 만족한다.
    """
    src = lake.scan("short_interest").select(
        "settlement_date",
        "symbol",
        "current_short_qty",
        "avg_daily_volume_qty",
        "days_to_cover",
        "change_percent",
    )
    return shift_trading_days(
        lake, src, "settlement_date", _SETTLEMENT_LAG, out_col="_eligible_date"
    ).filter(pl.col("_eligible_date").is_not_null())


def _sv_share_daily(lake: UsLake) -> pl.LazyFrame:
    """(date, symbol) -> off-exchange 공매도 비율의 20거래일 평균.

    "off-exchange 공매도량"은 ``short_volume + short_exempt_volume``이다 —
    둘 다 공매도 카테고리고, 후자만 특정 Reg SHO 예외를 받은 것뿐이다
    (수집 계획 ``03`` §4.16). ``short_exempt_volume``이 null이면(2018년 이전
    원문에 칸이 아예 없었던 경우) 0으로 채운다 — 결측이 아니라 그 카테고리가
    당시 원천에 없었다는 뜻이다.

    **분모(``total_volume``)는 이 표 자신의 off-exchange 총량이다.** 시장
    전체 거래량이 아니다 — ``sv_share_20``이 "시장 전체 공매도 비율"이 아닌
    이유가 여기 있다(전체의 중앙값 39.2%, ``01_data_readiness.md`` §2).
    """
    daily = (
        lake.scan("short_volume")
        .with_columns(pl.col("short_exempt_volume").fill_null(0.0))
        .filter(pl.col("total_volume") > 0)
        .with_columns(
            (
                (pl.col("short_volume") + pl.col("short_exempt_volume")) / pl.col("total_volume")
            ).alias("_sv_ratio")
        )
        .sort(["symbol", "date"])
        .with_columns(
            pl.col("_sv_ratio")
            .rolling_mean(window_size=_SV_WINDOW, min_samples=_SV_WINDOW)
            .over("symbol")
            .alias("sv_share_20")
        )
        .select("date", "symbol", "sv_share_20")
    )
    return daily


def add_short(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """F11(``si_ratio`` · ``dtc`` · ``si_chg`` · ``sv_share_20``)를 붙인다."""
    eligible = _eligible_short_interest(lake).sort(["symbol", "_eligible_date"])
    sv_daily = _sv_share_daily(lake).sort(["symbol", "date"])

    panel_lf = panel.lazy().sort(["symbol", "date"])

    with_si = panel_lf.join_asof(
        eligible,
        left_on="date",
        right_on="_eligible_date",
        by="symbol",
        strategy="backward",
    ).with_columns(
        pl.when(pl.col("avg_daily_volume_qty") > 0)
        .then(pl.col("current_short_qty") / pl.col("avg_daily_volume_qty"))
        .otherwise(None)
        .alias("si_ratio"),
        pl.col("days_to_cover").alias("dtc"),
        pl.col("change_percent").alias("si_chg"),
    )

    with_sv = with_si.join_asof(
        sv_daily,
        left_on="date",
        right_on="date",
        by="symbol",
        strategy="backward",
    )

    result = with_sv.with_columns(
        pl.col("si_ratio").is_null().alias("si_ratio_isna"),
        pl.col("dtc").is_null().alias("dtc_isna"),
        pl.col("si_chg").is_null().alias("si_chg_isna"),
        pl.col("sv_share_20").is_null().alias("sv_share_20_isna"),
    )

    keep = [
        *panel.columns,
        "si_ratio",
        "si_ratio_isna",
        "dtc",
        "dtc_isna",
        "si_chg",
        "si_chg_isna",
        "sv_share_20",
        "sv_share_20_isna",
    ]
    return result.select(keep).sort(["date", "symbol"]).collect()
