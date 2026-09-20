"""``modeler.us.panel`` 단위 테스트. ``tmp_path``에 합성 parquet을 쓴다."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.lake import UsLake
from modeler.us.panel import build_panel, month_first_trading_days


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


def _write_corp_actions_empty(tmp_path: Path) -> None:
    frame = pl.DataFrame(
        schema={
            "symbol": pl.String,
            "ex_date": pl.Date,
            "kind": pl.String,
            "to_factor": pl.Float64,
            "for_factor": pl.Float64,
        }
    )
    _write_snapshot(tmp_path, "corp_actions", "2026-09-18", frame)


def _write_universe(tmp_path: Path, rows: list[dict]) -> None:
    frame = pl.DataFrame(
        rows,
        schema={
            "date": pl.Date,
            "symbol": pl.String,
            "cik": pl.Int64,
            "sic": pl.String,
            "mcap_rank": pl.Int32,
            "adv_20d": pl.Float64,
            "exchange": pl.String,
            "in_universe": pl.Boolean,
        },
    )
    _write_snapshot(tmp_path, "universe_daily", "2026-09-18", frame)


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


# --- month_first_trading_days --------------------------------------------------


def test_month_first_trading_days_picks_min_date_per_month(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(
        tmp_path,
        [
            date(2019, 1, 2),
            date(2019, 1, 3),
            date(2019, 2, 1),
            date(2019, 2, 4),
        ],
    )

    result = month_first_trading_days(lake, date(2019, 1, 1), date(2019, 2, 28))

    assert result == [date(2019, 1, 2), date(2019, 2, 1)]


def test_month_first_trading_days_skips_new_year_holiday(tmp_path: Path, lake: UsLake) -> None:
    """신정(1/1)이 껴서 거래일이 1/2부터 시작하는 달을 포함한다."""
    _write_trading_calendar(
        tmp_path,
        [date(2019, 1, 2), date(2019, 1, 3), date(2019, 1, 4)],  # 1/1 없음(휴장)
    )

    result = month_first_trading_days(lake, date(2019, 1, 1), date(2019, 1, 31))

    assert result == [date(2019, 1, 2)]


def test_month_first_trading_days_ignores_other_exchange(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2019, 1, 2), date(2019, 1, 1)],
            "exchange": ["XNYS", "SOME_OTHER"],
        }
    )
    _write_snapshot(tmp_path, "trading_calendar", "2026-09-19", frame)

    result = month_first_trading_days(lake, date(2019, 1, 1), date(2019, 1, 31))

    # XNYS만 본다 — SOME_OTHER의 1/1이 더 이르지만 고르면 안 된다.
    assert result == [date(2019, 1, 2)]


def test_month_first_trading_days_respects_date_range(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(
        tmp_path,
        [date(2018, 12, 31), date(2019, 1, 2), date(2019, 2, 1)],
    )

    result = month_first_trading_days(lake, date(2019, 1, 1), date(2019, 1, 31))

    assert result == [date(2019, 1, 2)]


# --- build_panel ----------------------------------------------------------------


def test_build_panel_filters_to_in_universe_rows(tmp_path: Path, lake: UsLake) -> None:
    _write_trading_calendar(tmp_path, [date(2019, 1, 2)])
    _write_corp_actions_empty(tmp_path)
    _write_universe(
        tmp_path,
        [
            {
                "date": date(2019, 1, 2),
                "symbol": "AAPL",
                "cik": 320193,
                "sic": "3571",
                "mcap_rank": 1,
                "adv_20d": 1_000_000.0,
                "exchange": "XNYS",
                "in_universe": True,
            },
            {
                "date": date(2019, 1, 2),
                "symbol": "PENNY",
                "cik": 999999,
                "sic": None,
                "mcap_rank": 9,
                "adv_20d": 500.0,
                "exchange": "XNYS",
                "in_universe": False,  # 유니버스 밖 — 패널에 없어야 한다
            },
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2019, 1, 2),
                "symbol": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            },
            {
                "date": date(2019, 1, 2),
                "symbol": "PENNY",
                "open": 2.0,
                "high": 2.0,
                "low": 2.0,
                "close": 2.0,
                "volume": 1000.0,
            },
        ],
    )

    panel = build_panel(lake, start=date(2019, 1, 1), end=date(2019, 1, 31))

    assert panel["symbol"].to_list() == ["AAPL"]


def test_build_panel_sic2_is_first_two_digits_and_null_when_sic_missing(
    tmp_path: Path, lake: UsLake
) -> None:
    _write_trading_calendar(tmp_path, [date(2019, 1, 2)])
    _write_corp_actions_empty(tmp_path)
    _write_universe(
        tmp_path,
        [
            {
                "date": date(2019, 1, 2),
                "symbol": "AAPL",
                "cik": 320193,
                "sic": "3571",
                "mcap_rank": 1,
                "adv_20d": 1_000_000.0,
                "exchange": "XNYS",
                "in_universe": True,
            },
            {
                "date": date(2019, 1, 2),
                "symbol": "NOSIC",
                "cik": None,
                "sic": None,
                "mcap_rank": None,
                "adv_20d": 2_000_000.0,
                "exchange": "XNYS",
                "in_universe": True,
            },
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2019, 1, 2),
                "symbol": "AAPL",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            },
            {
                "date": date(2019, 1, 2),
                "symbol": "NOSIC",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 1000.0,
            },
        ],
    )

    panel = build_panel(lake, start=date(2019, 1, 1), end=date(2019, 1, 31)).sort("symbol")

    sic2 = dict(zip(panel["symbol"].to_list(), panel["sic2"].to_list()))
    assert sic2["AAPL"] == "35"
    assert sic2["NOSIC"] is None


def test_build_panel_price_ge_5_uses_raw_close_not_adjusted(tmp_path: Path, lake: UsLake) -> None:
    """price_ge_5가 **조정 전** 종가를 보는가.

    패널이 담는 리밸런스일(2019-01-02)보다 뒤에 2:1 분할이 있어, 조정 종가는
    원시 종가의 절반이 된다 — 원시 ``6``(>=5)이 조정하면 ``3``(<5)이 된다.
    ``price_ge_5``는 조정값이 아니라 원시값을 봐야 하므로 여전히 ``True``다.
    """
    _write_trading_calendar(tmp_path, [date(2019, 1, 2)])
    frame = pl.DataFrame(
        {
            "symbol": ["HIGHRAW"],
            "ex_date": [date(2019, 3, 1)],  # 패널 날짜보다 뒤 — 이 행은 조정 대상이다
            "kind": ["split"],
            "to_factor": [2.0],
            "for_factor": [1.0],
        }
    )
    _write_snapshot(tmp_path, "corp_actions", "2026-09-18", frame)
    _write_universe(
        tmp_path,
        [
            {
                "date": date(2019, 1, 2),
                "symbol": "HIGHRAW",
                "cik": 1,
                "sic": "1000",
                "mcap_rank": 1,
                "adv_20d": 1_000_000.0,
                "exchange": "XNYS",
                "in_universe": True,
            },
        ],
    )
    _write_prices(
        tmp_path,
        [
            {
                "date": date(2019, 1, 2),
                "symbol": "HIGHRAW",
                "open": 6.0,
                "high": 6.0,
                "low": 6.0,
                "close": 6.0,
                "volume": 1000.0,
            },
            {
                # base_date(T)를 분할 뒤로 밀어 두는 용도 — adjusted_daily는
                # base_date를 안 받으면 prices_daily 최대 date를 T로 쓴다.
                "date": date(2019, 6, 1),
                "symbol": "HIGHRAW",
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000.0,
            },
        ],
    )

    panel = build_panel(lake, start=date(2019, 1, 1), end=date(2019, 1, 31))

    row = panel.row(0, named=True)
    assert row["close"] == pytest.approx(6.0)
    assert row["adj_close"] == pytest.approx(3.0)  # 조정하면 5 밑으로 떨어진다
    assert row["price_ge_5"] is True  # 그래도 원시 종가 기준으로는 True
