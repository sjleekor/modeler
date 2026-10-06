"""종목 구간(``security_id``) 붙이기 — 유니버스 v2 설계 §3 (``20261006_universe_v2/02_design.md``).

같은 심볼이라도 가격 공백·CUSIP 교체 같은 **재사용 사건**을 지나면 다른 증권이다
(``security_segments``). 롤링 창·분할 조정·라벨이 그 경계를 건너 앞 구간 행을 읽지
않게 하려면 가격 행마다 구간 id를 붙여 ``symbol`` 대신 그것으로 묶어야 한다.

**PIT 보기(``view = 'pit'``)만 읽는다** (설계 T11). 사후 정정 보기(``post``)는 진단용이고
판정·계산 경로에 쓰지 않는다 — ``pit_segments``가 그 필터를 한 곳에서 건다.

구간은 심볼마다 이어 붙어 있다(구간 ``k``의 끝 = 구간 ``k+1``의 시작 − 1일). 그래서
``(symbol, date)``가 속한 구간은 ``seg_start <= date``인 마지막 구간이다. 첫 구간은
``seg_start``가 null(열려 있다)이라 맨 처음으로 둔다.

``attach_security_id``는 날짜가 구간을 정한다 — 수급·외부 표(FTD·공매도·MIDAS 등)도
``(symbol, date)``로 이 도우미를 거치면 같은 id를 얻는다.
"""

from __future__ import annotations

from datetime import date

import polars as pl

from modeler.us.lake import UsLake

SEGMENTS_TABLE = "security_segments"

#: 첫 구간(``seg_start`` null)을 asof 조인 키로 쓸 때의 시작값. 어떤 가격 날짜보다 앞선다.
_OPEN_START = date(1900, 1, 1)

#: ``security_id`` 열 이름.
SECURITY_ID = "security_id"


def pit_segments(lake: UsLake) -> pl.DataFrame:
    """PIT 구간 표: ``symbol, segment_no, security_id, seg_start, seg_end``.

    ``seg_start``는 첫 구간이 null이다(열림). ``view`` 열이 있으면 ``pit`` 행만 남기고,
    ``view`` 열이 없는 표는 거부한다 — PIT 전용인지 확인할 길이 없는 표를 판정에 쓰지 않는다.
    """
    lf = lake.scan(SEGMENTS_TABLE)
    if "view" not in lf.collect_schema().names():
        raise ValueError(f"{SEGMENTS_TABLE}에 view 열이 없습니다 — PIT 보기를 가려낼 수 없습니다.")
    return (
        lf.filter(pl.col("view") == "pit")
        .select("symbol", "segment_no", SECURITY_ID, "seg_start", "seg_end")
        .sort(["symbol", "segment_no"])
        .collect()
    )


def _asof_table(lake: UsLake) -> pl.LazyFrame:
    return (
        pit_segments(lake)
        .lazy()
        .with_columns(pl.col("seg_start").fill_null(_OPEN_START).alias("_seg_key"))
        .select("symbol", "_seg_key", SECURITY_ID, "seg_end")
        .sort(["symbol", "_seg_key"])
    )


def attach_security_id(
    lf: pl.LazyFrame,
    lake: UsLake,
    *,
    date_col: str = "date",
    symbol_col: str = "symbol",
    with_seg_end: bool = False,
) -> pl.LazyFrame:
    """``lf``의 ``(symbol_col, date_col)`` 행마다 ``security_id``를 붙인다.

    구간 표에 없는 심볼은 ``<symbol>#1``로 둔다 — 알려진 분리가 없다는 뜻이라 첫 구간과 같다.
    ``with_seg_end=True``면 그 구간의 끝(``seg_end``, 열려 있으면 null)도 붙인다(라벨이 쓴다).
    행 순서는 ``(symbol_col, date_col)``으로 정렬된다.
    """
    seg = _asof_table(lake)
    if symbol_col != "symbol":
        seg = seg.rename({"symbol": symbol_col})
    joined = lf.sort([symbol_col, date_col]).join_asof(
        seg,
        left_on=date_col,
        right_on="_seg_key",
        by=symbol_col,
        strategy="backward",
    )
    exprs = [pl.coalesce(pl.col(SECURITY_ID), pl.col(symbol_col) + "#1").alias(SECURITY_ID)]
    drop = ["_seg_key"]
    if with_seg_end:
        exprs.append(pl.col("seg_end"))
    else:
        drop.append("seg_end")
    return joined.with_columns(exprs).drop(drop)


def group_key(lake: UsLake) -> str:
    """롤링·시프트를 묶는 열 이름. 꺼짐이면 ``symbol``(지금과 같다), 켜짐이면 ``security_id``."""
    return SECURITY_ID if lake.security_boundaries else "symbol"


def mask_warmup(value: pl.Expr, by: str, window: int) -> pl.Expr:
    """그룹(``by``) 안에서 0부터 센 행 위치가 ``window-1`` 미만이면 ``value``를 null로 둔다.

    종목 구간 모드의 예열 규칙("구간 첫 (창−1)행 비움")이다. 위치만 보므로 창 안에 빈 값이
    있어도 위치가 ``window-1`` 이상이면 값이 나온다.
    호출 전에 ``(by, 날짜)`` 순으로 정렬돼 있어야 한다.
    """
    position = pl.int_range(pl.len()).over(by)
    return pl.when(position >= window - 1).then(value).otherwise(None)
