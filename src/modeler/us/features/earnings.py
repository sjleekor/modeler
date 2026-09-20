"""F9 실적 이벤트 — ``earnings_calendar`` · ``filings_index``.

``04_feature_test_plan.md`` §3 F9. 네 피쳐: ``sue_last`` · ``days_since_earn`` ·
``days_to_earn`` · ``n_estimates``.

**PIT의 핵심.** ``earnings_calendar.date``는 그 자체로 PIT 축이지만
(``01_data_readiness.md`` §2), ``time_code``로는 장전·장후를 못 가른다
(72.0%가 ``surprise_pct``는 있어도 ``time_code``는 126,538행이
``time-not-supplied``다 — 수집 계획 ``03`` §4.17). 그래서 실적 발표의 실제
접수 시각은 같은 ``cik``의 8-K item 2.02(실적 발표) 공시를
``filings_index.acceptance_datetime``에서 찾아 ET 16:00 규칙(``07_risks.md``
Y4)을 적용한 "유효 거래일"로 삼는다. 대응하는 8-K를 못 찾으면(공시
메타가 안 붙거나 cik가 없는 종목) ``earnings_calendar.date``를 그대로 쓴다 —
그 값도 PIT 축이므로 안전한 쪽으로 물러나는 것이다.

``days_to_earn``은 다르다. **다음 예정 발표일 자체는 사전 공지라 PIT다**
(``04`` §3 F9) — 8-K 보정 없이 ``earnings_calendar.date`` 원값을 그대로 쓴다.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from modeler.us.features._trading_days import filings_effective_date, trading_days_between
from modeler.us.lake import UsLake, acceptance_datetime_to_et

#: 8-K item 2.02 = 실적 발표(Results of Operations). ``items``는 쉼표로 이은
#: 문자열이라(예: ``"1.01,2.02,2.03"``) 부분 문자열로 찾는다.
_EARNINGS_8K_ITEM = "2.02"

#: 실적 캘린더 행을 8-K로 보정할 때 허용하는 거리. 컨퍼런스콜 발표와 8-K 접수가
#: 하루이틀 어긋나는 경우가 있어(전날 장마감 후 발표 → 다음날 오전 8-K 등) 3일로
#: 잡는다. 이보다 멀면 보정을 포기하고 ``date`` 원값으로 물러난다.
_MATCH_TOLERANCE_DAYS = 3


def _earnings_with_cik(lake: UsLake) -> pl.LazyFrame:
    """``earnings_calendar``에 그날 그 심볼의 ``cik``를 붙인다 (있으면).

    ``universe_daily``의 (date, symbol) -> cik는 그 시점 값이라 PIT다.
    ``earnings_calendar``의 심볼-날짜가 ``universe_daily``에 없으면(유니버스
    밖 종목·상장폐지 등) cik는 null로 남고, 8-K 보정 없이 원값으로 물러난다.
    """
    earnings = lake.scan("earnings_calendar").select(
        "date", "symbol", "eps", "eps_forecast", "surprise_pct", "n_estimates"
    )
    cik_map = (
        lake.scan("universe_daily")
        .select("date", "symbol", "cik")
        .unique(subset=["date", "symbol"])
    )
    return earnings.join(cik_map, on=["date", "symbol"], how="left")


def _earnings_8k_effective_dates(lake: UsLake) -> pl.LazyFrame:
    """(cik, 8-K 접수 ET 날짜, 유효 거래일) — item 2.02 공시만."""
    filings = (
        lake.scan("filings_index")
        .filter(
            (pl.col("form") == "8-K")
            & pl.col("items").is_not_null()
            & pl.col("items").str.contains(_EARNINGS_8K_ITEM, literal=True)
        )
        .select("cik", "acceptance_datetime")
    )
    with_effective = filings_effective_date(lake, filings, "acceptance_datetime")

    return with_effective.with_columns(
        acceptance_datetime_to_et(pl.col("acceptance_datetime")).dt.date().alias("_et_date")
    ).select("cik", "_et_date", "acceptance_datetime_effective_date")


def _earnings_history(lake: UsLake) -> pl.LazyFrame:
    """(symbol, effective_date, raw_date, surprise_pct, n_estimates) — 발표별 한 행.

    ``effective_date``는 8-K로 보정된 값(있으면), 없으면 ``date`` 그대로다.
    """
    earnings = _earnings_with_cik(lake)
    eightk = _earnings_8k_effective_dates(lake)

    # cik가 있는 것만 8-K asof 매칭 대상이다. join_asof는 두 쪽 다 정렬 + by가
    # 같은 dtype이어야 한다 (cik: Int64).
    with_cik = earnings.filter(pl.col("cik").is_not_null()).sort(["cik", "date"])
    matched = with_cik.join_asof(
        eightk.sort(["cik", "_et_date"]),
        left_on="date",
        right_on="_et_date",
        by="cik",
        strategy="nearest",
        tolerance=timedelta(days=_MATCH_TOLERANCE_DAYS),
    )
    without_cik = earnings.filter(pl.col("cik").is_null()).with_columns(
        pl.lit(None, dtype=pl.Date).alias("_et_date"),
        pl.lit(None, dtype=pl.Date).alias("acceptance_datetime_effective_date"),
    )

    combined = pl.concat([matched, without_cik], how="diagonal_relaxed")
    return combined.with_columns(
        pl.coalesce(["acceptance_datetime_effective_date", "date"]).alias("effective_date")
    ).select(
        "symbol", "effective_date", pl.col("date").alias("raw_date"), "surprise_pct", "n_estimates"
    )


def add_earnings(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """F9(``sue_last`` · ``days_since_earn`` · ``days_to_earn`` · ``n_estimates``)를 붙인다."""
    history = _earnings_history(lake).sort(["symbol", "effective_date"]).collect()

    panel_lf = panel.lazy().sort(["symbol", "date"])

    # --- 최근 발표(effective_date <= t): sue_last · n_estimates · days_since_earn
    past = panel_lf.join_asof(
        history.lazy().select("symbol", "effective_date", "surprise_pct", "n_estimates"),
        left_on="date",
        right_on="effective_date",
        by="symbol",
        strategy="backward",
    )
    past = trading_days_between(
        lake, past, "effective_date", "date", out_col="days_since_earn"
    ).rename({"surprise_pct": "sue_last"})

    # --- 다음 예정 발표(raw_date > t, 8-K 보정 없음): days_to_earn
    future_src = lake.scan("earnings_calendar").select("date", "symbol").sort(["symbol", "date"])
    future = panel_lf.with_columns(
        (pl.col("date") + timedelta(days=1)).alias("_search_date")
    ).join_asof(
        future_src,
        left_on="_search_date",
        right_on="date",
        by="symbol",
        strategy="forward",
        suffix="_next_earn",
    )
    future = trading_days_between(
        lake, future, "date", "date_next_earn", out_col="days_to_earn"
    ).select("symbol", "date", "days_to_earn")

    result = past.join(future, on=["symbol", "date"], how="left").collect()

    result = result.with_columns(
        pl.col("sue_last").is_null().alias("sue_last_isna"),
        pl.col("days_since_earn").is_null().alias("days_since_earn_isna"),
        pl.col("days_to_earn").is_null().alias("days_to_earn_isna"),
        pl.col("n_estimates").is_null().alias("n_estimates_isna"),
    )

    keep = [c for c in result.columns if c not in ("effective_date",)]
    return result.select(keep).sort(["date", "symbol"])
