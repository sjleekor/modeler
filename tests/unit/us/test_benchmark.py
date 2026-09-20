"""``modeler.us.benchmark`` 단위 테스트. ``tmp_path``에 합성 parquet을 쓴다."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.benchmark import (
    ew_minus_spy_monthly,
    spy_monthly_return,
    spy_total_return_daily,
    universe_equal_weight_monthly,
)
from modeler.us.lake import UsLake


def _write_snapshot(root: Path, table: str, snapshot_date: str, frame: pl.DataFrame) -> None:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")


@pytest.fixture()
def lake(tmp_path: Path) -> UsLake:
    return UsLake(root=DataRoot(base=tmp_path))


def _write_trading_calendar(tmp_path: Path, dates: list[date], exchange: str = "XNYS") -> None:
    frame = pl.DataFrame({"date": dates, "exchange": [exchange] * len(dates)})
    _write_snapshot(tmp_path, "trading_calendar", "2026-09-19", frame)


def _write_prices(tmp_path: Path, rows: list[dict]) -> None:
    """``rows``는 ``date, symbol, close``만 준다 — 나머지는 ``close``로 채운다.

    ``prices.adjusted_daily``가 ``open/high/low/volume`` 컬럼 존재를 요구하므로
    (분할조정 계산에 필요), 이 테스트가 쓰지 않는 값이라도 채워 넣는다.
    """
    filled = [
        {
            "date": row["date"],
            "symbol": row["symbol"],
            "open": row["close"],
            "high": row["close"],
            "low": row["close"],
            "close": row["close"],
            "volume": 1_000.0,
        }
        for row in rows
    ]
    frame = pl.DataFrame(
        filled,
        schema={
            "date": pl.Date,
            "symbol": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
        },
    )
    _write_snapshot(tmp_path, "prices_daily", "2026-09-18", frame)


def _write_corp_actions(tmp_path: Path, rows: list[dict] | None = None) -> None:
    """``rows``는 배당 행(symbol, ex_date, kind, amount)만 준다.

    ``to_factor``·``for_factor``는 ``prices.split_factors``(``adjusted_daily``가
    내부에서 부른다)가 스키마 존재를 요구해서 채운다 — 이 테스트는 분할이
    없으니 값 자체는 안 쓰인다.
    """
    filled = [{**row, "to_factor": None, "for_factor": None} for row in (rows or [])]
    frame = pl.DataFrame(
        filled,
        schema={
            "symbol": pl.String,
            "ex_date": pl.Date,
            "kind": pl.String,
            "amount": pl.Float64,
            "to_factor": pl.Float64,
            "for_factor": pl.Float64,
        },
    )
    _write_snapshot(tmp_path, "corp_actions", "2026-09-18", frame)


_D0 = date(2021, 1, 1)
_D1 = _D0 + timedelta(days=1)
_D2 = _D0 + timedelta(days=2)


# --- spy_total_return_daily -------------------------------------------------------


def test_spy_total_return_daily_without_dividends_equals_adj_close(
    tmp_path: Path, lake: UsLake
) -> None:
    _write_corp_actions(tmp_path)
    _write_prices(
        tmp_path,
        [
            {"date": _D0, "symbol": "SPY", "close": 100.0},
            {"date": _D1, "symbol": "SPY", "close": 101.0},
            {"date": _D2, "symbol": "SPY", "close": 102.0},
        ],
    )

    tr = spy_total_return_daily(lake).sort("date")

    assert tr["tr_adj"].to_list() == tr["adj_close"].to_list()
    assert tr["tr_adj"].to_list() == [100.0, 101.0, 102.0]


def test_spy_total_return_daily_reinvests_dividend_before_ex_date(
    tmp_path: Path, lake: UsLake
) -> None:
    """배당락일(``_D1``) 앞은 배당수익률만큼 할인되고, 그 뒤는 원래 가격 그대로다.

    ``03_schema_and_pit.md`` §2 ``tr_adj`` 식의 검산 — ``build_labels.py``
    실행 보고에 실제 레이크 사례 검산도 같이 남긴다.
    """
    _write_corp_actions(
        tmp_path,
        [{"symbol": "SPY", "ex_date": _D1, "kind": "dividend", "amount": 1.0}],
    )
    _write_prices(
        tmp_path,
        [
            {"date": _D0, "symbol": "SPY", "close": 100.0},
            {"date": _D1, "symbol": "SPY", "close": 99.0},
            {"date": _D2, "symbol": "SPY", "close": 100.0},
        ],
    )

    tr = spy_total_return_daily(lake).sort("date")

    own_factor = 1 - 1.0 / 100.0  # amount / close(직전 거래일=_D0)
    assert tr["tr_adj"].to_list() == pytest.approx([100.0 * own_factor, 99.0, 100.0])


def test_spy_total_return_daily_dividend_before_price_history_is_ignored(
    tmp_path: Path, lake: UsLake
) -> None:
    """가격 이력 시작 이전 배당은 prev_close가 없어도 죽지 않는다(상수 배율이라 무시해도 안전)."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "SPY",
                "ex_date": _D0 - timedelta(days=365),
                "kind": "dividend",
                "amount": 1.0,
            }
        ],
    )
    _write_prices(
        tmp_path,
        [
            {"date": _D0, "symbol": "SPY", "close": 100.0},
            {"date": _D1, "symbol": "SPY", "close": 101.0},
        ],
    )

    tr = spy_total_return_daily(lake).sort("date")

    assert tr["tr_adj"].to_list() == pytest.approx([100.0, 101.0])


# --- spy_monthly_return ------------------------------------------------------------


def test_spy_monthly_return_computes_h21_total_return(tmp_path: Path, lake: UsLake) -> None:
    calendar = [_D0 + timedelta(days=i) for i in range(25)]
    t21 = calendar[21]
    _write_trading_calendar(tmp_path, calendar)
    _write_corp_actions(tmp_path)
    _write_prices(
        tmp_path,
        [
            {"date": _D0, "symbol": "SPY", "close": 100.0},
            {"date": t21, "symbol": "SPY", "close": 110.0},
        ],
    )

    result = spy_monthly_return(lake, [_D0])

    assert result["date"].to_list() == [_D0]
    assert result["spy_h21_return"][0] == pytest.approx(0.10)


def test_spy_monthly_return_drops_date_beyond_price_data(tmp_path: Path, lake: UsLake) -> None:
    calendar = [_D0 + timedelta(days=i) for i in range(25)]
    _write_trading_calendar(tmp_path, calendar)
    _write_corp_actions(tmp_path)
    _write_prices(tmp_path, [{"date": _D0, "symbol": "SPY", "close": 100.0}])

    result = spy_monthly_return(lake, [_D0])

    assert result.height == 0


# --- universe_equal_weight_monthly · ew_minus_spy_monthly --------------------------


def test_universe_equal_weight_monthly_averages_l0_per_date() -> None:
    labels_df = pl.DataFrame(
        {
            "date": [_D0, _D0, _D1],
            "symbol": ["AAA", "BBB", "AAA"],
            "L0": [0.10, 0.20, -0.05],
        }
    )

    result = universe_equal_weight_monthly(labels_df)

    result = result.sort("date")
    assert result["ew_l0_h21_return"].to_list() == pytest.approx([0.15, -0.05])


def test_ew_minus_spy_monthly_is_the_difference(tmp_path: Path, lake: UsLake) -> None:
    calendar = [_D0 + timedelta(days=i) for i in range(25)]
    t21 = calendar[21]
    _write_trading_calendar(tmp_path, calendar)
    _write_corp_actions(tmp_path)
    _write_prices(
        tmp_path,
        [
            {"date": _D0, "symbol": "SPY", "close": 100.0},
            {"date": t21, "symbol": "SPY", "close": 105.0},  # SPY h21 수익률 5%
        ],
    )
    labels_df = pl.DataFrame({"date": [_D0, _D0], "symbol": ["AAA", "BBB"], "L0": [0.10, 0.20]})

    result = ew_minus_spy_monthly(lake, labels_df)

    assert result.height == 1
    assert result["ew_l0_h21_return"][0] == pytest.approx(0.15)
    assert result["spy_h21_return"][0] == pytest.approx(0.05)
    assert result["ew_minus_spy"][0] == pytest.approx(0.10)
