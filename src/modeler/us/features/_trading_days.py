"""거래일 계산 공유 헬퍼 — ``trading_calendar``만 본다.

F9(실적)·F11(공매도)·F12(공시 활동)이 같이 쓴다. 다른 family가 더 필요하면
여기 추가한다 — **거래일 계산 로직을 family 파일마다 따로 만들지 않는다.**

``prices.py``나 ``panel.py``의 것을 고치지 않는다 — 이 파일은 새로 만드는
공유 자리다 (``panel.py``의 ``_EXCHANGE`` 상수를 그대로 흉내내되, 그쪽을
import하지 않는다. ``panel.py``가 이 모듈을 참조하게 되면 그때 합친다).
"""

from __future__ import annotations

import polars as pl

from modeler.us.lake import UsLake, acceptance_datetime_to_et

#: ``trading_calendar``에 실제로 있는 거래소는 하나뿐이다 (``panel.py``와 같은
#: 실측 근거 — 수집 계획 ``03_schema_and_pit.md`` §4.15).
DEFAULT_EXCHANGE = "XNYS"


def trading_day_index(lake: UsLake, *, exchange: str = DEFAULT_EXCHANGE) -> pl.LazyFrame:
    """(date, trading_day_number) — ``trading_calendar`` 전체 구간에 0부터 순번을 매긴다.

    두 거래일 사이의 "며칠 거래일"을 구할 때(``trading_days_between``) 이
    순번의 차를 쓴다. 순번은 이 함수를 부를 때마다 같은 정의로 다시 매겨지므로
    호출 시점이 달라도 같은 (date, 순번) 관계가 나온다.
    """
    return (
        lake.scan("trading_calendar")
        .filter(pl.col("exchange") == exchange)
        .select("date")
        .unique()
        .sort("date")
        .with_row_index("trading_day_number")
    )


def ceil_to_trading_day(
    lake: UsLake,
    lf: pl.LazyFrame,
    date_col: str,
    *,
    exchange: str = DEFAULT_EXCHANGE,
    out_col: str | None = None,
) -> pl.LazyFrame:
    """``date_col``(달력일, 거래일이 아닐 수 있다)을 그날 또는 그 뒤 첫 거래일로 올린다.

    ``join_asof(strategy="forward")``는 "왼쪽 키 이상인 값 중 가장 작은 것"을
    찾아 준다 — 정확히 올림(ceiling)이다. ``date_col``이 이미 거래일이면
    자기 자신이 나온다. 캘린더 범위(2011\\~2026-12-31) 밖으로 넘어가면 매칭이
    없어 결과는 null이다.
    """
    out_col = out_col or f"{date_col}_trading"
    calendar = (
        lake.scan("trading_calendar")
        .filter(pl.col("exchange") == exchange)
        .select(pl.col("date").alias("_trading_date"))
        .unique()
        .sort("_trading_date")
    )
    result = lf.sort(date_col).join_asof(
        calendar, left_on=date_col, right_on="_trading_date", strategy="forward"
    )
    return result.rename({"_trading_date": out_col})


def shift_trading_days(
    lake: UsLake,
    lf: pl.LazyFrame,
    date_col: str,
    n: int,
    *,
    exchange: str = DEFAULT_EXCHANGE,
    out_col: str | None = None,
) -> pl.LazyFrame:
    """``date_col``(이미 거래일)을 거래일 기준 ``n``칸 옮긴 컬럼을 붙인다.

    ``n``이 양수면 미래로, 음수면 과거로 옮긴다. **``date_col`` 값 자체가
    ``trading_calendar``에 있는 거래일이어야 한다** — 아니면 그 행은 결과가
    null이다(캘린더에 없는 날짜라 순번을 못 찾는다). ``short_interest.
    settlement_date``는 원천이 이미 거래일로 당겨 두므로 이 조건을 만족한다.
    """
    out_col = out_col or f"{date_col}_shift{n:+d}"
    # UInt32 순번에 음수 n을 더하면(과거로 이동) 0 밑에서 랩어라운드한다 —
    # Int64로 캐스팅한 뒤 더한다.
    calendar = (
        trading_day_index(lake, exchange=exchange)
        .collect()
        .with_columns(pl.col("trading_day_number").cast(pl.Int64))
    )
    left_idx = calendar.rename({"date": date_col, "trading_day_number": "_tdn"}).lazy()
    shifted_idx = calendar.rename({"trading_day_number": "_tdn_shifted", "date": out_col}).lazy()
    return (
        lf.join(left_idx, on=date_col, how="left")
        .with_columns((pl.col("_tdn") + n).alias("_tdn_shifted"))
        .join(shifted_idx, on="_tdn_shifted", how="left")
        .drop(["_tdn", "_tdn_shifted"])
    )


def trading_days_between(
    lake: UsLake,
    lf: pl.LazyFrame,
    from_col: str,
    to_col: str,
    *,
    exchange: str = DEFAULT_EXCHANGE,
    out_col: str = "trading_days",
) -> pl.LazyFrame:
    """``to_col`` 순번 − ``from_col`` 순번. 둘 다 거래일이어야 정확하다.

    ``t``(패널의 리밸런스일)와 이 모듈이 만든 유효일(``effective_date``,
    ``ceil_to_trading_day``의 결과)은 둘 다 거래일이라 이 함수로 정확한
    거래일 수 차를 구할 수 있다.
    """
    # UInt32 순번을 그대로 빼면 to < from일 때(음수가 나와야 할 자리) 부호 없는
    # 정수라 랩어라운드로 거대한 양수가 된다 — Int64로 캐스팅한 뒤 뺀다.
    calendar = (
        trading_day_index(lake, exchange=exchange)
        .collect()
        .with_columns(pl.col("trading_day_number").cast(pl.Int64))
    )
    from_idx = calendar.rename({"date": from_col, "trading_day_number": "_from_tdn"}).lazy()
    to_idx = calendar.rename({"date": to_col, "trading_day_number": "_to_tdn"}).lazy()
    return (
        lf.join(from_idx, on=from_col, how="left")
        .join(to_idx, on=to_col, how="left")
        .with_columns((pl.col("_to_tdn") - pl.col("_from_tdn")).alias(out_col))
        .drop(["_from_tdn", "_to_tdn"])
    )


def filings_effective_date(
    lake: UsLake,
    lf: pl.LazyFrame,
    acceptance_col: str,
    *,
    exchange: str = DEFAULT_EXCHANGE,
    out_col: str | None = None,
) -> pl.LazyFrame:
    """``filings_index.acceptance_datetime``(UTC) → PIT 유효 거래일.

    ``01_data_readiness.md`` §6 · ``07_risks.md`` Y4의 규칙 그대로다:

    1. ``acceptance_datetime_to_et()``로 ET로 바꾼다.
    2. ET 기준 **16:00 시(時) 이상이면** 다음 날(달력일)로 민다. ``hour == 16,
       minute == 0``(정각)도 포함한다 — 마감 이후 정보를 당겨 쓰는 쪽보다
       하루 늦게 보는 쪽이 안전하다는 판단이다(경계값이라 어느 쪽으로 정해도
       근거 문서가 초 단위까지 규정하지 않는다 — 이 자리가 이번 구현의 판단
       하나다. 보고에 적는다).
    3. 그 결과(주말·휴장일일 수 있다)를 ``ceil_to_trading_day``로 다음 거래일로
       올린다. 16:00 규칙과 달력→거래일 올림을 한 번에 처리하므로, 접수
       시각이 애초에 비거래일(주말 등)이었던 경우도 자동으로 다음 거래일로
       밀린다.
    """
    out_col = out_col or f"{acceptance_col}_effective_date"
    et = acceptance_datetime_to_et(pl.col(acceptance_col))
    cutoff_date = (
        pl.when(et.dt.hour() >= 16).then(et.dt.date() + pl.duration(days=1)).otherwise(et.dt.date())
    )
    staged = lf.with_columns(cutoff_date.alias("_cutoff_date"))
    result = ceil_to_trading_day(lake, staged, "_cutoff_date", exchange=exchange, out_col=out_col)
    return result.drop("_cutoff_date")
