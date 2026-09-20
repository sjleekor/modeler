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

import logging
import math
from datetime import date, timedelta

import polars as pl

from modeler.us.lake import UsLake

logger = logging.getLogger(__name__)

#: 가격 검증 허용 배수 오차 — ``실행 보고`` 참고. 분할이 진짜면
#: ``close(ex_date) / close(직전 거래일)``이 ``for_factor/to_factor``에서
#: 이 배수 이상 벗어나지 않는다.
SPLIT_PRICE_TOLERANCE = 1.5

_SPLIT_LOG_TOLERANCE = math.log(SPLIT_PRICE_TOLERANCE)


def _base_date_default(lake: UsLake) -> date:
    """``base_date``를 안 주면 ``prices_daily``의 최대 ``date``를 T로 쓴다."""
    row = lake.scan("prices_daily").select(pl.col("date").max().alias("max_date")).collect()
    value = row.item()
    if value is None:
        raise ValueError("prices_daily가 비어 있어 base_date를 정할 수 없습니다.")
    return value


def _validate_splits_against_price(
    lake: UsLake, candidates: pl.LazyFrame
) -> tuple[pl.LazyFrame, dict[str, int]]:
    """분할 후보 (symbol, ex_date, ``_own_factor``)를 원시 가격으로 검증한다.

    레이크 ``corp_actions``에 실측으로 확인된 결함 둘 — ``to_factor=0`` 행(0
    나눗셈으로 ``adj_close``가 NaN/Inf가 된다)과 같은 종목·비슷한 배율이 60일
    안에 중복되는 행(가짜 ex_date가 그 사이 구간을 배수만큼 어긋나게 한다) —
    을 가격으로 걸러낸다. ``to_factor<=0``·``for_factor<=0`` 행은 호출부가
    이미 걸러 여기 들어오지 않는다.

    **판정 규칙**: 분할이 진짜면 ``ex_date`` 당일 원시 종가가 그 직전 거래일
    종가 대비 배수 ``f = for_factor/to_factor``만큼 뛴다(액면병합은 배수가
    1보다 커 가격이 오른다). ``|log(price_ratio / f)| < log(SPLIT_PRICE_TOLERANCE)``
    (기본 1.5배 오차)면 확인된 분할로 본다.

    ``ex_date``나 그 직전 거래일에 이 심볼의 원시 가격이 없어 검증 자체가
    안 되면 **버리지 않고 살린다** — 모르는 것을 버리는 쪽이 더 위험하다.

    반환: (살아남은(확인됨 + 검증불가) 후보 ``LazyFrame``,
    ``{"confirmed", "rejected_price_mismatch", "unverifiable"}`` 세 부류의 수).
    """
    candidates_df = candidates.collect()
    if candidates_df.height == 0:
        return candidates_df.lazy(), {
            "confirmed": 0,
            "rejected_price_mismatch": 0,
            "unverifiable": 0,
        }

    symbols = candidates_df["symbol"].unique().to_list()
    try:
        raw_close = (
            lake.scan("prices_daily")
            .filter(pl.col("symbol").is_in(symbols))
            .select("date", "symbol", pl.col("close").cast(pl.Float64))
            .collect()
        )
    except FileNotFoundError:
        # prices_daily 스냅샷 자체가 없다 — 검증할 방법이 없으니 전부 검증
        # 불가로 살린다(모르는 것을 버리는 쪽이 더 위험하다는 원칙 그대로).
        logger.warning(
            "prices_daily 스냅샷이 없어 분할 %d건을 가격으로 검증하지 못했다 — 전부 살린다.",
            candidates_df.height,
        )
        raw_close = pl.DataFrame(schema={"date": pl.Date, "symbol": pl.String, "close": pl.Float64})

    # ex_date 당일의 원시 종가는 정확히 일치하는 날짜로만 찾는다(asof가 아니다) —
    # 그날 가격이 없으면 검증 불가로 살려야지, 엉뚱한 다른 날 가격에 맞추면 안 된다.
    ex_close = raw_close.rename({"date": "ex_date", "close": "_ex_close"})
    prev_source = raw_close.rename({"date": "_prev_date", "close": "_prev_close"}).sort(
        ["symbol", "_prev_date"]
    )

    # "직전 거래일"은 이 심볼의 원시 가격이 실제로 있는 가장 최근 이전 날짜다
    # (거래소 달력이 아니라 이 심볼 자신의 관측 기준) — ex_date - 1일을 경계로
    # backward asof 조인하면 ex_date 미만인 가장 늦은 관측을 찾는다.
    joined = (
        candidates_df.with_columns((pl.col("ex_date") - timedelta(days=1)).alias("_boundary"))
        .sort(["symbol", "_boundary"])
        .join_asof(
            prev_source,
            left_on="_boundary",
            right_on="_prev_date",
            by="symbol",
            strategy="backward",
        )
        .join(ex_close, on=["symbol", "ex_date"], how="left")
    )

    verifiable = (
        pl.col("_ex_close").is_not_null()
        & pl.col("_prev_close").is_not_null()
        & (pl.col("_prev_close") > 0)
        & (pl.col("_ex_close") > 0)
    )
    price_ratio = pl.col("_ex_close") / pl.col("_prev_close")
    log_diff = (price_ratio / pl.col("_own_factor")).log().abs()

    flagged = joined.with_columns(
        verifiable.alias("_verifiable"),
        (verifiable & (log_diff < _SPLIT_LOG_TOLERANCE)).alias("_confirmed"),
    )

    counts = {
        "confirmed": flagged.filter(pl.col("_verifiable") & pl.col("_confirmed")).height,
        "rejected_price_mismatch": flagged.filter(
            pl.col("_verifiable") & ~pl.col("_confirmed")
        ).height,
        "unverifiable": flagged.filter(~pl.col("_verifiable")).height,
    }

    survivors = flagged.filter(~pl.col("_verifiable") | pl.col("_confirmed")).select(
        "symbol", "ex_date", "_own_factor"
    )
    return survivors.lazy(), counts


def split_factors(
    lake: UsLake,
    *,
    base_date: date | None = None,
    diagnostics: dict[str, int] | None = None,
) -> pl.LazyFrame:
    """(symbol, date) -> 누적 분할 계수.

    ``date``는 분할 이벤트의 ``ex_date``다. 이 행의 ``split_factor``는 "이 ex_date
    바로 이전(``t < ex_date``)의 가격에 곱해야 하는 누적 계수" — 즉 이 ex_date부터
    ``base_date``(T)까지 일어난 모든 분할(자기 자신 포함)의 ``for_factor/to_factor``
    곱이다. ``base_date``보다 뒤에 일어난 분할은 T 시점엔 알 수 없으므로 애초에
    제외한다.

    ``adjusted_daily``가 이 표를 실제 가격 날짜에 asof 조인해 붙인다 — 각 가격
    날짜 ``t``에는 "``t`` 다음에 오는 가장 이른 ex_date"의 ``split_factor``가
    적용된다(그 값이 이미 그 시점부터 T까지의 전체 곱이기 때문이다).

    **레이크 결함 방어 (2026-09-20 실측).** ``corp_actions``에 ``to_factor=0``인
    분할 행(0 나눗셈)과, 같은 종목·비슷한 배율이 60일 안에 중복되는 행(가짜
    ex_date)이 있다 — 둘 다 ``adj_close``를 오염시킨다. 여기서 ``to_factor``·
    ``for_factor``가 0 이하인 행은 무조건 버리고, 나머지는 원시 가격으로 검증한다
    (``_validate_splits_against_price``). 검증 결과 세 부류(``confirmed``·
    ``rejected_price_mismatch``·``unverifiable``)와 무조건 버린 수(``invalid_factor``)를
    로그로 남기고, ``diagnostics``(dict)를 주면 그 안에도 채운다.
    """
    if base_date is None:
        base_date = _base_date_default(lake)

    raw_splits = (
        lake.scan("corp_actions")
        .filter((pl.col("kind") == "split") & (pl.col("ex_date") <= base_date))
        .select(
            "symbol",
            "ex_date",
            pl.col("to_factor").cast(pl.Float64).alias("_to_factor"),
            pl.col("for_factor").cast(pl.Float64).alias("_for_factor"),
        )
    )

    # 경계 처리 ①: to_factor<=0 또는 for_factor<=0인 행은 검증 이전 문제라
    # 가격을 보지 않고 무조건 버린다(0 나눗셈이 NaN/Inf를 낸다).
    invalid_mask = (pl.col("_to_factor") <= 0) | (pl.col("_for_factor") <= 0)
    valid_splits = raw_splits.filter(~invalid_mask)
    invalid_count = raw_splits.filter(invalid_mask).select(pl.len()).collect().item()

    candidates = (
        valid_splits.with_columns(
            (pl.col("_for_factor") / pl.col("_to_factor")).alias("_own_factor")
        )
        # 같은 (symbol, ex_date)에 분할이 둘 겹치면(2026-09-20 실측으로는 없었지만
        # 방어적으로) 곱해서 하나로 합친다. 가격 검증은 합쳐진 뒤의 배수를 본다.
        .group_by(["symbol", "ex_date"]).agg(pl.col("_own_factor").product().alias("_own_factor"))
    )

    survivors, price_counts = _validate_splits_against_price(lake, candidates)

    counts: dict[str, int] = {"invalid_factor": invalid_count, **price_counts}
    logger.info(
        "corp_actions 분할 검증 — to/for_factor<=0로 버림 %d · 가격으로 확인 %d · "
        "가격 불일치로 버림 %d · 검증 불가(살림) %d",
        counts["invalid_factor"],
        counts["confirmed"],
        counts["rejected_price_mismatch"],
        counts["unverifiable"],
    )
    if diagnostics is not None:
        diagnostics.update(counts)

    # symbol별로 ex_date를 **내림차순**으로 두고 누적곱하면, 각 행에는 "자신과
    # 자신보다 늦은 모든 분할"의 곱이 쌓인다 — 그게 바로 그 ex_date 이전 가격에
    # 필요한 계수다 (t < ex_date_i <= ... <= T인 모든 분할이 적용되는 구간).
    cumulative = (
        survivors.sort(["symbol", "ex_date"], descending=[False, True])
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
