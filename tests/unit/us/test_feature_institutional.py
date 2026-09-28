"""F19(``institutional.py``) 단위 테스트 — us4 조건부 family.

``test_feature_flow.py``와 같은 관례로 ``tmp_path``에 합성 parquet을 쓴다.
**실제 레이크는 읽지 않는다.**

다루는 것:

- 정의 일치(``inst_n_log``·``inst_breadth_chg``·``inst_shares_chg``)
- PIT — ``period_of_report + LAG_13F_DAYS`` 전에는 그 분기가 안 보인다
- CUSIP 다대다 — 후보 심볼이 여럿이면 ``n_settlement_dates``가 큰 쪽을 고른다
- "직전 분기"는 CUSIP 기준이다 — 첫 관측이거나 분기를 하나 건너뛰면 chg가 null
- ``inst_holdings_q``(또는 ``cusip_symbol_pit``) 표가 레이크에 없으면 조용히
  건너뛰지 않고 ``FileNotFoundError``로 멈춘다(``build_flow_features.py``가
  그대로 물려받는 동작)
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.features.institutional import LAG_13F_DAYS, add_institutional
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
    days = []
    d = start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _panel(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.String})


def _write_inst_holdings_q(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "cusip": pl.String,
            "period_of_report": pl.Date,
            "n_holders": pl.Int32,
            "shares_total": pl.Int64,
            "n_filers_total_that_period": pl.Int32,
        },
    )
    _write_snapshot(tmp_path, "inst_holdings_q", "2026-09-27", frame)


def _write_cusip_symbol_pit(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "cusip": pl.String,
            "symbol": pl.String,
            "first_seen": pl.Date,
            "last_seen": pl.Date,
            "n_settlement_dates": pl.Int32,
        },
    )
    _write_snapshot(tmp_path, "cusip_symbol_pit", "2026-09-24", frame)


def _write_prices_daily_volume(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(rows, schema={"date": pl.Date, "symbol": pl.String, "volume": pl.Float64})
    _write_snapshot(tmp_path, "prices_daily", "2026-09-21", frame)


def _constant_volume_rows(days: list[date], symbol: str, volume: float) -> list[dict]:
    return [{"date": d, "symbol": symbol, "volume": volume} for d in days]


# --- 정의 일치 + PIT 경계 ------------------------------------------------------


def test_add_institutional_matches_definition_and_pit_boundary(
    tmp_path: Path, lake: UsLake
) -> None:
    """두 분기(Q1·Q2) 연속 — Q1은 아직 아무 분기도 안 보일 때, Q2는 정의 그대로."""
    q1 = date(2024, 3, 31)
    q2 = date(2024, 6, 30)
    _write_inst_holdings_q(
        tmp_path,
        [
            {
                "cusip": "111111111",
                "period_of_report": q1,
                "n_holders": 100,
                "shares_total": 1_000_000,
                "n_filers_total_that_period": 5000,
            },
            {
                "cusip": "111111111",
                "period_of_report": q2,
                "n_holders": 120,
                "shares_total": 1_100_000,
                "n_filers_total_that_period": 5100,
            },
        ],
    )
    _write_cusip_symbol_pit(
        tmp_path,
        [
            {
                "cusip": "111111111",
                "symbol": "AAA",
                "first_seen": date(2020, 1, 1),
                "last_seen": date(2026, 12, 31),
                "n_settlement_dates": 500,
            }
        ],
    )
    price_days = _weekday_trading_days(date(2024, 5, 1), date(2024, 6, 28))
    _write_prices_daily_volume(tmp_path, _constant_volume_rows(price_days, "AAA", 1000.0))

    available_q1 = q1 + timedelta(days=LAG_13F_DAYS)
    available_q2 = q2 + timedelta(days=LAG_13F_DAYS)
    panel = _panel(
        [
            {"date": available_q1 - timedelta(days=1), "symbol": "AAA"},  # 아무 분기도 안 보임
            {"date": available_q2, "symbol": "AAA"},  # Q2까지 보임
        ]
    )

    result = add_institutional(panel, lake).sort("date")
    nothing_visible, q2_visible = result.row(0, named=True), result.row(1, named=True)

    assert nothing_visible["inst_n_log_isna"] is True
    assert nothing_visible["inst_breadth_chg_isna"] is True
    assert nothing_visible["inst_shares_chg_isna"] is True

    assert q2_visible["inst_n_log_isna"] is False
    assert q2_visible["inst_n_log"] == pytest.approx(math.log1p(120))
    assert q2_visible["inst_breadth_chg"] == pytest.approx((120 - 100) / 5100)
    assert q2_visible["inst_shares_chg"] == pytest.approx((1_100_000 - 1_000_000) / 1000.0)


# --- CUSIP 다대다 -------------------------------------------------------------


def test_add_institutional_cusip_many_to_many_picks_symbol_with_more_settlement_dates(
    tmp_path: Path, lake: UsLake
) -> None:
    """CUSIP 하나가 두 심볼 후보(BBB·CCC)와 같은 기간에 겹친다 —
    ``n_settlement_dates``가 큰 CCC만 값을 받고, BBB는 이 CUSIP에서 못 붙어
    (다른 원천이 없으니) 계속 결측이다.
    """
    q1 = date(2024, 3, 31)
    _write_inst_holdings_q(
        tmp_path,
        [
            {
                "cusip": "222222222",
                "period_of_report": q1,
                "n_holders": 50,
                "shares_total": 500_000,
                "n_filers_total_that_period": 4000,
            }
        ],
    )
    _write_cusip_symbol_pit(
        tmp_path,
        [
            {
                "cusip": "222222222",
                "symbol": "BBB",
                "first_seen": date(2020, 1, 1),
                "last_seen": date(2026, 12, 31),
                "n_settlement_dates": 10,
            },
            {
                "cusip": "222222222",
                "symbol": "CCC",
                "first_seen": date(2020, 1, 1),
                "last_seen": date(2026, 12, 31),
                "n_settlement_dates": 50,
            },
        ],
    )
    price_days = _weekday_trading_days(date(2024, 2, 1), date(2024, 3, 28))
    _write_prices_daily_volume(
        tmp_path,
        _constant_volume_rows(price_days, "BBB", 2000.0)
        + _constant_volume_rows(price_days, "CCC", 2000.0),
    )

    available = q1 + timedelta(days=LAG_13F_DAYS)
    panel = _panel([{"date": available, "symbol": "BBB"}, {"date": available, "symbol": "CCC"}])

    result = add_institutional(panel, lake).sort("symbol")
    bbb, ccc = result.row(0, named=True), result.row(1, named=True)

    assert bbb["symbol"] == "BBB"
    assert bbb["inst_n_log_isna"] is True  # n_settlement_dates가 작아 못 붙는다

    assert ccc["symbol"] == "CCC"
    assert ccc["inst_n_log_isna"] is False
    assert ccc["inst_n_log"] == pytest.approx(math.log1p(50))


# --- 직전 분기 없음 / 분기를 건너뜀 -> chg null ---------------------------------


def test_add_institutional_missing_or_skipped_previous_quarter_makes_chg_null(
    tmp_path: Path, lake: UsLake
) -> None:
    """DDD: 딱 한 분기뿐(첫 관측) — chg null. EEE: Q1·Q3만 있고 Q2를 건너뜀 —
    바로 앞 행이 있어도 분기가 연속이 아니면 chg null. 둘 다 ``inst_n_log``은 채워진다.
    """
    q1 = date(2024, 3, 31)
    q3 = date(2024, 9, 30)
    _write_inst_holdings_q(
        tmp_path,
        [
            {
                "cusip": "333333333",
                "period_of_report": q1,
                "n_holders": 10,
                "shares_total": 100_000,
                "n_filers_total_that_period": 3000,
            },
            {
                "cusip": "444444444",
                "period_of_report": q1,
                "n_holders": 20,
                "shares_total": 200_000,
                "n_filers_total_that_period": 3000,
            },
            {
                "cusip": "444444444",
                "period_of_report": q3,
                "n_holders": 30,
                "shares_total": 300_000,
                "n_filers_total_that_period": 3200,
            },
        ],
    )
    _write_cusip_symbol_pit(
        tmp_path,
        [
            {
                "cusip": "333333333",
                "symbol": "DDD",
                "first_seen": date(2020, 1, 1),
                "last_seen": date(2026, 12, 31),
                "n_settlement_dates": 100,
            },
            {
                "cusip": "444444444",
                "symbol": "EEE",
                "first_seen": date(2020, 1, 1),
                "last_seen": date(2026, 12, 31),
                "n_settlement_dates": 100,
            },
        ],
    )
    price_days = _weekday_trading_days(date(2024, 8, 1), date(2024, 9, 27))
    _write_prices_daily_volume(
        tmp_path,
        _constant_volume_rows(price_days, "DDD", 500.0)
        + _constant_volume_rows(price_days, "EEE", 500.0),
    )

    available_q1 = q1 + timedelta(days=LAG_13F_DAYS)
    available_q3 = q3 + timedelta(days=LAG_13F_DAYS)
    panel = _panel(
        [
            {"date": available_q1, "symbol": "DDD"},  # 첫(유일한) 관측
            {"date": available_q3, "symbol": "EEE"},  # Q1->Q3, Q2를 건너뜀
        ]
    )

    result = add_institutional(panel, lake).sort("symbol")
    ddd, eee = result.row(0, named=True), result.row(1, named=True)

    assert ddd["inst_n_log_isna"] is False
    assert ddd["inst_breadth_chg_isna"] is True
    assert ddd["inst_shares_chg_isna"] is True

    assert eee["inst_n_log_isna"] is False
    assert eee["inst_n_log"] == pytest.approx(math.log1p(30))
    assert eee["inst_breadth_chg_isna"] is True
    assert eee["inst_shares_chg_isna"] is True


# --- 표가 레이크에 없으면 조용히 안 넘어가고 멈춘다 -----------------------------


def test_add_institutional_raises_when_inst_holdings_q_table_is_missing(
    tmp_path: Path, lake: UsLake
) -> None:
    """``inst_holdings_q`` 스냅샷 디렉터리 자체가 없으면(맥에 아직 안 옴)
    ``build_flow_features.py``가 조용히 F19를 빼지 않고 그대로 멈춰야 한다 —
    이 함수 수준에서 ``FileNotFoundError``가 나는 것으로 그 동작을 보장한다.
    """
    _write_cusip_symbol_pit(
        tmp_path,
        [
            {
                "cusip": "555555555",
                "symbol": "FFF",
                "first_seen": date(2020, 1, 1),
                "last_seen": date(2026, 12, 31),
                "n_settlement_dates": 10,
            }
        ],
    )
    panel = _panel([{"date": date(2024, 1, 1), "symbol": "FFF"}])

    with pytest.raises(FileNotFoundError):
        add_institutional(panel, lake)


def test_add_institutional_raises_when_cusip_symbol_pit_table_is_missing(
    tmp_path: Path, lake: UsLake
) -> None:
    """``cusip_symbol_pit``이 없어도 마찬가지로 멈춘다."""
    _write_inst_holdings_q(
        tmp_path,
        [
            {
                "cusip": "666666666",
                "period_of_report": date(2024, 3, 31),
                "n_holders": 1,
                "shares_total": 1,
                "n_filers_total_that_period": 1,
            }
        ],
    )
    panel = _panel([{"date": date(2024, 1, 1), "symbol": "GGG"}])

    with pytest.raises(FileNotFoundError):
        add_institutional(panel, lake)
