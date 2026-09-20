"""F16 캘린더 — ``early_close_in_window``.

**단독 검정을 하지 않는다**(``04_feature_test_plan.md`` §3 F16) — 거래량 계열
피쳐(F4의 ``log_dvol_20``·``amihud_20`` 등)를 조기종료일 영향에서 정규화하는 데
쓰는 보조 피쳐다. 그래도 F1\\~F4·F13과 같은 파이프라인에 태우려고 만들어 둔다.

``early_close_in_window``는 XNYS ``trading_calendar``의 ``is_early_close``를
**20 거래일 창(당일 포함, 그 앞 19일)**에서 센 합이다 — F3·F4의 20일 창과 창
길이·포함 경계를 맞췄다. 종목이 아니라 **거래소 캘린더 값**이라 종목별 가격 행이
아니라 ``trading_calendar`` 자체의 날짜 순서로 롤링한다. 그 값을 같은 날짜의
모든 종목에 그대로 방송(broadcast)한다.

**PIT 위험이 없다**(``04`` F16) — 거래소 휴장·조기종료 일정은 연초에 미리
공표되는 값이라 그 시점에 이미 안다. 그래도 창 자체가 미래를 보지 않는지는
단위 테스트로 확인한다.
"""

from __future__ import annotations

import polars as pl

from modeler.us.features._daily import join_features
from modeler.us.lake import UsLake

_EXCHANGE = "XNYS"
_FEATURES = ("early_close_in_window",)


def add_calendar(panel: pl.DataFrame, lake: UsLake) -> pl.DataFrame:
    """``panel``의 ``date``에 F16 캘린더 피쳐 + ``_isna``를 붙인다(종목 무관, 방송)."""
    calendar = (
        lake.scan("trading_calendar")
        .filter(pl.col("exchange") == _EXCHANGE)
        .sort("date")
        .with_columns(
            pl.col("is_early_close")
            .cast(pl.Int32)
            .rolling_sum(window_size=20)
            .alias("early_close_in_window")
        )
        .select("date", "early_close_in_window")
    )
    return join_features(panel, calendar, _FEATURES, on=("date",))
