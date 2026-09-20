"""``modeler.us.prices`` 단위 테스트.

``test_lake.py``와 같은 관례로 ``tmp_path``에 합성 parquet을 쓴다. 실제 레이크는
읽지 않는다.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.lake import UsLake
from modeler.us.prices import adjusted_daily, split_factors


def _write_snapshot(root: Path, table: str, snapshot_date: str, frame: pl.DataFrame) -> None:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")


@pytest.fixture()
def lake(tmp_path: Path) -> UsLake:
    return UsLake(root=DataRoot(base=tmp_path))


def _write_corp_actions(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "symbol": pl.String,
            "ex_date": pl.Date,
            "kind": pl.String,
            "to_factor": pl.Float64,
            "for_factor": pl.Float64,
        },
    )
    _write_snapshot(tmp_path, "corp_actions", "2026-09-18", frame)


def _write_prices(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
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


# --- split_factors ------------------------------------------------------------


def test_split_factors_cumulates_two_overlapping_splits(tmp_path: Path, lake: UsLake) -> None:
    """AAPL 7:1(2014-06-09) · 4:1(2020-08-31) 예시. 겹치는 구간에서 곱해져야 한다."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "AAPL",
                "ex_date": date(2014, 6, 9),
                "kind": "split",
                "to_factor": 7.0,
                "for_factor": 1.0,
            },
            {
                "symbol": "AAPL",
                "ex_date": date(2020, 8, 31),
                "kind": "split",
                "to_factor": 4.0,
                "for_factor": 1.0,
            },
        ],
    )

    result = split_factors(lake, base_date=date(2026, 1, 1)).collect().sort("date")

    factors = dict(zip(result["date"].to_list(), result["split_factor"].to_list()))
    # 2014-06-09 행: 자신(1/7)과 그 뒤 2020-08-31 분할(1/4)이 겹쳐 곱해진다.
    assert factors[date(2014, 6, 9)] == pytest.approx(1 / 7 * 1 / 4)
    # 2020-08-31 행: 그 뒤 분할이 없으니 자신만.
    assert factors[date(2020, 8, 31)] == pytest.approx(1 / 4)


def test_split_factors_excludes_splits_after_base_date(tmp_path: Path, lake: UsLake) -> None:
    """base_date(T) 뒤에 일어난 분할은 T 시점엔 알 수 없으므로 제외한다."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "AAPL",
                "ex_date": date(2014, 6, 9),
                "kind": "split",
                "to_factor": 7.0,
                "for_factor": 1.0,
            },
            {
                "symbol": "AAPL",
                "ex_date": date(2020, 8, 31),
                "kind": "split",
                "to_factor": 4.0,
                "for_factor": 1.0,
            },
        ],
    )

    # T를 2020-08-31 이전으로 두면 그 분할은 안 보인다.
    result = split_factors(lake, base_date=date(2015, 1, 1)).collect()

    assert result["date"].to_list() == [date(2014, 6, 9)]
    assert result["split_factor"].item() == pytest.approx(1 / 7)


def test_split_factors_ignores_dividend_rows(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame(
        {
            "symbol": ["AAPL", "AAPL"],
            "ex_date": [date(2020, 8, 31), date(2020, 9, 1)],
            "kind": ["split", "dividend"],
            "to_factor": [4.0, None],
            "for_factor": [1.0, None],
            "amount": [None, 0.5],
        }
    )
    _write_snapshot(tmp_path, "corp_actions", "2026-09-18", frame)

    result = split_factors(lake, base_date=date(2026, 1, 1)).collect()

    assert result["date"].to_list() == [date(2020, 8, 31)]


# --- adjusted_daily -------------------------------------------------------------


def test_adjusted_daily_is_identity_when_no_splits(tmp_path: Path, lake: UsLake) -> None:
    _write_corp_actions(tmp_path, [])
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2020, 1, 2),
                "symbol": "MSFT",
                "open": 10.0,
                "high": 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000.0,
            }
        ],
    )

    result = adjusted_daily(lake, base_date=date(2026, 1, 1)).collect()

    row = result.row(0, named=True)
    assert row["adj_close"] == pytest.approx(10.5)
    assert row["adj_volume"] == pytest.approx(1000.0)
    assert row["adj_dollar_volume"] == pytest.approx(10.5 * 1000.0)


def test_adjusted_daily_ohlc_share_the_same_factor(tmp_path: Path, lake: UsLake) -> None:
    """OHLC 네 컬럼에 같은 계수가 걸리는가."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "NVDA",
                "ex_date": date(2024, 6, 10),
                "kind": "split",
                "to_factor": 10.0,
                "for_factor": 1.0,
            }
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2024, 6, 7),
                "symbol": "NVDA",
                "open": 1200.0,
                "high": 1250.0,
                "low": 1180.0,
                "close": 1210.0,
                "volume": 500.0,
            }
        ],
    )

    result = adjusted_daily(lake, base_date=date(2026, 1, 1)).collect().row(0, named=True)

    assert result["adj_open"] == pytest.approx(1200.0 / 10)
    assert result["adj_high"] == pytest.approx(1250.0 / 10)
    assert result["adj_low"] == pytest.approx(1180.0 / 10)
    assert result["adj_close"] == pytest.approx(1210.0 / 10)


def test_adjusted_daily_ex_date_itself_is_not_adjusted(tmp_path: Path, lake: UsLake) -> None:
    """``t < ex_date <= T`` 경계 — ex_date 당일 가격은 이미 조정된 가격이라
    그날 자신의 분할 계수가 다시 걸리면 안 된다."""
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "AAPL",
                "ex_date": date(2020, 8, 31),
                "kind": "split",
                "to_factor": 4.0,
                "for_factor": 1.0,
            }
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2020, 8, 28),  # 분할 직전 거래일
                "symbol": "AAPL",
                "open": 500.0,
                "high": 500.0,
                "low": 500.0,
                "close": 500.0,
                "volume": 100.0,
            },
            {
                "date": date(2020, 8, 31),  # 분할 당일 — 이미 조정된 가격
                "symbol": "AAPL",
                "open": 125.0,
                "high": 125.0,
                "low": 125.0,
                "close": 125.0,
                "volume": 400.0,
            },
        ],
    )

    result = adjusted_daily(lake, base_date=date(2026, 1, 1)).collect().sort("date")

    before, on_ex_date = result.row(0, named=True), result.row(1, named=True)
    # 직전 거래일은 4:1로 나뉜다.
    assert before["adj_close"] == pytest.approx(500.0 / 4)
    # 당일은 원시값 그대로 — 계수가 다시 걸리면 125/4가 되어 이 값과 어긋난다.
    assert on_ex_date["adj_close"] == pytest.approx(125.0)
    # 조정 뒤 두 값이 이어진다(연속) — 점프가 없다.
    assert before["adj_close"] == pytest.approx(on_ex_date["adj_close"], rel=1e-6)


def test_adjusted_daily_volume_factor_is_inverse_of_price_factor(
    tmp_path: Path, lake: UsLake
) -> None:
    """거래량 계수가 가격과 반대 방향인가 — 틀리면 거래대금이 조용히 망가진다.

    2:1 분할 전후로 주가는 반토막, 주식 수(거래량)는 두 배가 되는 것이 정상이므로
    원시 달러거래량은 분할 전후로 거의 그대로다. 조정 뒤에도 이 성질이 이어져야
    한다 — 같은 계수를 가격·거래량에 다 걸면 여기서 제곱만큼 틀어진다.
    """
    _write_corp_actions(
        tmp_path,
        [
            {
                "symbol": "T",
                "ex_date": date(2021, 6, 1),
                "kind": "split",
                "to_factor": 2.0,
                "for_factor": 1.0,
            }
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2021, 5, 28),  # 분할 전: 종가 100, 거래량 1000
                "symbol": "T",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000.0,
            },
            {
                "date": date(2021, 6, 1),  # 분할 후: 종가 50, 거래량 2000 (달러거래량 그대로)
                "symbol": "T",
                "open": 50.0,
                "high": 50.0,
                "low": 50.0,
                "close": 50.0,
                "volume": 2000.0,
            },
        ],
    )

    result = adjusted_daily(lake, base_date=date(2026, 1, 1)).collect().sort("date")
    before, after = result.row(0, named=True), result.row(1, named=True)

    assert before["adj_close"] == pytest.approx(50.0)
    assert before["adj_volume"] == pytest.approx(2000.0)
    assert before["adj_dollar_volume"] == pytest.approx(after["adj_dollar_volume"], rel=1e-9)
    assert before["adj_dollar_volume"] == pytest.approx(100_000.0)
