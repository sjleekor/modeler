"""F17(``ftd.py``)·F18(``order_flow.py``) 단위 테스트 — us4 flow features.

``test_feature_event.py``와 같은 관례로 ``tmp_path``에 합성 parquet을 쓴다.
**실제 레이크는 읽지 않는다.**

PIT가 중심이다:

- F17: 결제일이 속한 반월 구간 끝 + ``LAG_FTD_DAYS``(달력일) 전에는 그 반월
  전체가 안 보인다(배치 단위 공개). 창 안 유효 결제일이 10 미만이면 null.
- F18: ``date``가 속한 분기 끝을 표(``MIDAS_AVAILABLE_FROM``) 또는 폴백
  (``MIDAS_FALLBACK_LAG_DAYS``)으로 늦춰야 보인다. ``security_type != 'Stock'``
  (ETF)은 애초에 ``lake.scan()``이 걸러 늘 결측이다.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.features.ftd import LAG_FTD_DAYS, add_ftd
from modeler.us.features.order_flow import (
    MIDAS_AVAILABLE_FROM,
    MIDAS_FALLBACK_LAG_DAYS,
    add_order_flow,
)
from modeler.us.lake import UsLake

# --- 공용 픽스처 -----------------------------------------------------------


def _write_snapshot(root: Path, table: str, snapshot_date: str, frame: pl.DataFrame) -> None:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")


@pytest.fixture()
def lake(tmp_path: Path) -> UsLake:
    return UsLake(root=DataRoot(base=tmp_path))


def _weekday_trading_days(start: date, end: date) -> list[date]:
    """토·일만 뺀 평일 목록 — 공휴일 없는 단순 달력."""
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _panel(rows: list[dict]) -> pl.DataFrame:
    """``build_flow_features``가 실제로 넘기는 최소 패널 — date·symbol만."""
    return pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.String})


def _half_month_end(d: date) -> date:
    """``ftd.py._half_month_end``와 같은 규칙을 파이썬으로 다시 쓴 것 — 기대값 계산용."""
    if d.day <= 15:
        return date(d.year, d.month, 15)
    if d.month == 12:
        return date(d.year, 12, 31)
    return date(d.year, d.month + 1, 1) - timedelta(days=1)


# --- F17 ftd.py --------------------------------------------------------------


def _write_ftd_fails(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={"settlement_date": pl.Date, "symbol": pl.String, "quantity": pl.Int64},
    )
    _write_snapshot(tmp_path, "ftd_fails", "2026-09-24", frame)


def _write_prices_daily_volume(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.String, "volume": pl.Float64})
    _write_snapshot(tmp_path, "prices_daily", "2026-09-21", frame)


def test_add_ftd_share_and_days_pit_boundary_with_absent_day(tmp_path: Path, lake: UsLake) -> None:
    """반월(1~15일) 전체가 배치로 공개된다 — 구간 끝 + LAG_FTD_DAYS 전날은 통째로 null.

    9일은 quantity가 있고 2일(1/4·1/11)은 명단에 없다(=0) — "0은 명단에 없음"
    (초안 §5.1)을 그대로 반영한다. 명단에 없는 날도 원시 거래량은 있다.
    """
    days = _weekday_trading_days(date(2024, 1, 1), date(2024, 1, 15))  # 11 영업일
    assert len(days) == 11
    absent = {date(2024, 1, 4), date(2024, 1, 11)}
    present_days = [d for d in days if d not in absent]
    assert len(present_days) == 9

    ftd_rows = [{"settlement_date": d, "symbol": "AAA", "quantity": 100} for d in present_days]
    # ZZZ가 결측일에도 결제일 축 자체는 살아 있게 한다(전역 결제일 축은 심볼
    # 무관 — ftd.py 모듈독스트링 참고).
    ftd_rows += [{"settlement_date": d, "symbol": "ZZZ", "quantity": 50} for d in days]
    _write_ftd_fails(tmp_path, ftd_rows)
    _write_prices_daily_volume(
        tmp_path, [{"date": d, "symbol": "AAA", "volume": 1000.0} for d in days]
    )

    available = _half_month_end(date(2024, 1, 15)) + timedelta(days=LAG_FTD_DAYS)
    assert available == date(2024, 2, 4)
    panel = _panel(
        [
            {"date": available - timedelta(days=1), "symbol": "AAA"},
            {"date": available, "symbol": "AAA"},
        ]
    )

    result = add_ftd(panel, lake).sort("date")
    before, after = result.row(0, named=True), result.row(1, named=True)

    assert before["ftd_share_20_isna"] is True
    assert before["ftd_days_20_isna"] is True
    assert before["ftd_chg_isna"] is True

    assert after["ftd_share_20_isna"] is False
    assert after["ftd_days_20"] == 9
    assert after["ftd_share_20"] == pytest.approx(900.0 / 11000.0)


def test_add_ftd_second_half_month_unlocks_only_at_its_own_availability(
    tmp_path: Path, lake: UsLake
) -> None:
    """반월 둘(1~15일·16~31일)이 서로 다른 사용 가능일을 갖는다 — 15일/말일 경계.

    또한 창 안 유효 결제일이 10 미만이면(첫 반월 5일뿐) null이 된다는 것도
    같이 확인한다.
    """
    first_half = _weekday_trading_days(date(2024, 1, 1), date(2024, 1, 5))  # 5일
    second_half = _weekday_trading_days(date(2024, 1, 16), date(2024, 1, 22))  # 5일
    assert len(first_half) == 5
    assert len(second_half) == 5
    all_days = first_half + second_half

    ftd_rows = [{"settlement_date": d, "symbol": "AAA", "quantity": 100} for d in first_half]
    ftd_rows += [{"settlement_date": d, "symbol": "AAA", "quantity": 200} for d in second_half]
    _write_ftd_fails(tmp_path, ftd_rows)
    _write_prices_daily_volume(
        tmp_path, [{"date": d, "symbol": "AAA", "volume": 1000.0} for d in all_days]
    )

    avail_first = _half_month_end(date(2024, 1, 5)) + timedelta(days=LAG_FTD_DAYS)
    avail_second = _half_month_end(date(2024, 1, 22)) + timedelta(days=LAG_FTD_DAYS)
    assert avail_first == date(2024, 2, 4)  # 1~15일 반월 끝(1/15) + 20일
    assert avail_second == date(2024, 2, 20)  # 16~31일 반월 끝(1/31) + 20일

    panel = _panel(
        [
            {"date": avail_first, "symbol": "AAA"},  # 둘째 반월 아직 안 보임 · 5일뿐(<10)
            {"date": avail_second, "symbol": "AAA"},  # 둘 다 보임 · 10일
        ]
    )

    result = add_ftd(panel, lake).sort("date")
    only_first, both = result.row(0, named=True), result.row(1, named=True)

    assert only_first["ftd_share_20_isna"] is True  # 유효 결제일 5 < 10
    assert both["ftd_share_20_isna"] is False
    assert both["ftd_days_20"] == 10
    assert both["ftd_share_20"] == pytest.approx((5 * 100 + 5 * 200) / (10 * 1000.0))


def test_add_ftd_chg_is_current_minus_21_trading_days_ago(tmp_path: Path, lake: UsLake) -> None:
    days = _weekday_trading_days(date(2024, 1, 1), date(2024, 3, 15))[:45]
    assert len(days) == 45
    quantities = [100] * 25 + [500] * 20
    ftd_rows = [
        {"settlement_date": d, "symbol": "AAA", "quantity": q} for d, q in zip(days, quantities)
    ]
    _write_ftd_fails(tmp_path, ftd_rows)
    _write_prices_daily_volume(
        tmp_path, [{"date": d, "symbol": "AAA", "volume": 1000.0} for d in days]
    )

    last_day = days[-1]
    available = _half_month_end(last_day) + timedelta(days=LAG_FTD_DAYS)
    t = available + timedelta(days=5)  # 마지막 날이 확실히 컷오프가 되도록 여유를 둔다
    panel = _panel([{"date": t, "symbol": "AAA"}])

    result = add_ftd(panel, lake).row(0, named=True)

    now_share = (20 * 500) / (20 * 1000.0)  # 최근 20결제일(전부 quantity=500)
    then_share = (20 * 100) / (20 * 1000.0)  # 21거래일 전 시점의 20결제일 창(전부 quantity=100)
    assert result["ftd_share_20"] == pytest.approx(now_share)
    assert result["ftd_chg"] == pytest.approx(now_share - then_share)


# --- F18 order_flow.py --------------------------------------------------------


def _write_midas(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "ticker": pl.String,
            "security_type": pl.String,
            "cancels": pl.Int64,
            "lit_trades": pl.Int64,
            "hidden_vol_k": pl.Float64,
            "trade_vol_for_hidden_k": pl.Float64,
            "odd_lot_vol_k": pl.Float64,
            "trade_vol_for_odd_lots_k": pl.Float64,
            "lit_vol_k": pl.Float64,
            "order_vol_k": pl.Float64,
        },
    )
    _write_snapshot(tmp_path, "midas_security_daily", "2026-09-23", frame)


def _constant_midas_rows(days: list[date], ticker: str, security_type: str) -> list[dict]:
    return [
        {
            "date": d,
            "ticker": ticker,
            "security_type": security_type,
            "cancels": 5,
            "lit_trades": 50,
            "hidden_vol_k": 2.0,
            "trade_vol_for_hidden_k": 10.0,
            "odd_lot_vol_k": 1.0,
            "trade_vol_for_odd_lots_k": 20.0,
            "lit_vol_k": 80.0,
            "order_vol_k": 100.0,
        }
        for d in days
    ]


def test_add_order_flow_matches_definition_and_pit_quarter_in_table(
    tmp_path: Path, lake: UsLake
) -> None:
    """2024q1은 ``MIDAS_AVAILABLE_FROM`` 표에 있다(2024-05-02) — 그 전날엔 안 보인다."""
    days = _weekday_trading_days(date(2024, 1, 2), date(2024, 1, 29))[:20]  # Q1 안, 20거래일
    assert len(days) == 20
    _write_midas(tmp_path, _constant_midas_rows(days, "AAA", "Stock"))

    available = MIDAS_AVAILABLE_FROM[(2024, 1)]
    assert available == date(2024, 5, 2)
    panel = _panel(
        [
            {"date": available - timedelta(days=1), "symbol": "AAA"},
            {"date": available, "symbol": "AAA"},
        ]
    )

    result = add_order_flow(panel, lake).sort("date")
    before, after = result.row(0, named=True), result.row(1, named=True)

    assert before["cancel_ratio_20_isna"] is True

    assert after["cancel_ratio_20"] == pytest.approx(5.0 / 50.0)
    assert after["hidden_share_20"] == pytest.approx(2.0 / 10.0)
    assert after["oddlot_share_20"] == pytest.approx(1.0 / 20.0)
    assert after["fill_ratio_20"] == pytest.approx(80.0 / 100.0)
    for c in ("cancel_ratio_20", "hidden_share_20", "oddlot_share_20", "fill_ratio_20"):
        assert after[f"{c}_isna"] is False


def test_add_order_flow_pit_quarter_missing_from_table_uses_fallback(
    tmp_path: Path, lake: UsLake
) -> None:
    """2019q2는 표에 없다(사이트 이전으로 미측정) — 분기 끝 + 60일 폴백을 쓴다."""
    days = _weekday_trading_days(date(2019, 4, 1), date(2019, 4, 26))[:20]  # 2019q2 안
    assert len(days) == 20
    _write_midas(tmp_path, _constant_midas_rows(days, "BBB", "Stock"))

    assert (2019, 2) not in MIDAS_AVAILABLE_FROM
    quarter_end = date(2019, 6, 30)
    available = quarter_end + timedelta(days=MIDAS_FALLBACK_LAG_DAYS)
    panel = _panel(
        [
            {"date": available - timedelta(days=1), "symbol": "BBB"},
            {"date": available, "symbol": "BBB"},
        ]
    )

    result = add_order_flow(panel, lake).sort("date")
    before, after = result.row(0, named=True), result.row(1, named=True)

    assert before["fill_ratio_20_isna"] is True
    assert after["fill_ratio_20_isna"] is False
    assert after["fill_ratio_20"] == pytest.approx(80.0 / 100.0)


def test_add_order_flow_etf_is_always_isna(tmp_path: Path, lake: UsLake) -> None:
    """``security_type == 'ETF'``는 ``lake.scan()``이 걸러 늘 결측이다."""
    days = _weekday_trading_days(date(2024, 1, 2), date(2024, 1, 29))[:20]
    _write_midas(tmp_path, _constant_midas_rows(days, "SPY", "ETF"))

    panel = _panel([{"date": date(2024, 12, 31), "symbol": "SPY"}])

    result = add_order_flow(panel, lake).row(0, named=True)

    for c in ("cancel_ratio_20", "hidden_share_20", "oddlot_share_20", "fill_ratio_20"):
        assert result[f"{c}_isna"] is True


def test_add_order_flow_needs_at_least_10_valid_days_in_window(
    tmp_path: Path, lake: UsLake
) -> None:
    days = _weekday_trading_days(date(2024, 1, 2), date(2024, 1, 8))[:5]  # 5거래일뿐
    assert len(days) == 5
    _write_midas(tmp_path, _constant_midas_rows(days, "CCC", "Stock"))

    available = MIDAS_AVAILABLE_FROM[(2024, 1)]
    panel = _panel([{"date": available + timedelta(days=30), "symbol": "CCC"}])

    result = add_order_flow(panel, lake).row(0, named=True)

    assert result["fill_ratio_20_isna"] is True
