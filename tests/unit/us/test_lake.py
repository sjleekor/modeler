"""``modeler.us.lake`` 단위 테스트.

전부 ``tmp_path``에 작은 합성 parquet을 써서 실제 레이크 구조
(``$STOCK_DATA_ROOT/us/derived/snapshots/<table>/snapshot_date=YYYY-MM-DD/*.parquet``)를
흉내 내 검사한다. 이 저장소 ``tests/unit`` 관례대로 진짜 2GB 레이크는 읽지 않는다.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.lake import (
    ASOF_AXIS,
    ASOF_LAG_TRADING_DAYS,
    US_TABLES,
    UsLake,
    acceptance_datetime_to_et,
)


def _write_snapshot(root: Path, table: str, snapshot_date: str, frame: pl.DataFrame) -> None:
    directory = root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"
    directory.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(directory / "part.parquet")


@pytest.fixture()
def lake(tmp_path: Path) -> UsLake:
    return UsLake(root=DataRoot(base=tmp_path))


# --- US_TABLES / ASOF_AXIS ---------------------------------------------------


def test_us_tables_has_18_unique_entries() -> None:
    assert len(US_TABLES) == 18
    assert len(set(US_TABLES)) == 18


def test_asof_axis_covers_every_table_exactly() -> None:
    assert set(ASOF_AXIS) == set(US_TABLES)


def test_asof_lag_keys_are_known_tables_with_an_axis() -> None:
    """공표 지연은 as-of 축이 있는 표에만 붙는다.

    ``short_interest``는 결제일과 공표일이 다르다. 축만 보고 자르면 공표 전
    값을 쓴다 (``07_risks.md`` Y10).
    """
    assert set(ASOF_LAG_TRADING_DAYS) <= set(US_TABLES)
    assert all(ASOF_AXIS[t] is not None for t in ASOF_LAG_TRADING_DAYS)
    assert ASOF_LAG_TRADING_DAYS["short_interest"] == 10


def test_asof_axis_none_only_for_tables_without_own_date_column() -> None:
    # company_meta(현재값 스냅샷) · insider_owners(join으로 빌려 씀) ·
    # trading_calendar(참조표)만 축이 없다. 01_data_readiness.md §2 그대로.
    assert {t for t, axis in ASOF_AXIS.items() if axis is None} == {
        "company_meta",
        "insider_owners",
        "trading_calendar",
    }


# --- latest_snapshot ---------------------------------------------------------


def test_latest_snapshot_picks_newest_among_several(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame({"date": [date(2020, 1, 1)]})
    _write_snapshot(tmp_path, "prices_daily", "2026-09-01", frame)
    _write_snapshot(tmp_path, "prices_daily", "2026-09-18", frame)
    _write_snapshot(tmp_path, "prices_daily", "2026-09-10", frame)

    assert lake.latest_snapshot("prices_daily") == date(2026, 9, 18)


def test_latest_snapshot_raises_when_table_dir_missing(lake: UsLake) -> None:
    with pytest.raises(FileNotFoundError):
        lake.latest_snapshot("prices_daily")


def test_latest_snapshot_raises_when_no_snapshot_subdirs(tmp_path: Path, lake: UsLake) -> None:
    (tmp_path / "derived" / "snapshots" / "prices_daily").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        lake.latest_snapshot("prices_daily")


def test_latest_snapshot_raises_for_unknown_table(lake: UsLake) -> None:
    with pytest.raises(KeyError):
        lake.latest_snapshot("not_a_real_table")


# --- snapshot_dir / scan_raw --------------------------------------------------


def test_snapshot_dir_uses_explicit_date_without_touching_disk(lake: UsLake) -> None:
    result = lake.snapshot_dir("prices_daily", date(2020, 1, 1))
    assert result.name == "snapshot_date=2020-01-01"


def test_scan_raw_reads_the_written_frame(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame({"date": [date(2020, 1, 2)], "symbol": ["AAPL"]})
    _write_snapshot(tmp_path, "prices_daily", "2026-09-18", frame)

    result = lake.scan_raw("prices_daily").collect()
    assert result.to_dicts() == frame.to_dicts()


def test_scan_raw_raises_when_no_parquet_for_date(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame({"date": [date(2020, 1, 2)]})
    _write_snapshot(tmp_path, "prices_daily", "2026-09-18", frame)

    with pytest.raises(FileNotFoundError):
        lake.scan_raw("prices_daily", date(2099, 1, 1))


# --- scan() 정제 규칙 다섯 ----------------------------------------------------


def test_scan_fundamentals_drops_end_after_2030(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame(
        {
            "end": [date(2025, 12, 31), date(2031, 1, 1)],
            "form": ["10-K", "10-K"],
            "val": [1.0, 2.0],
        }
    )
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", frame)

    result = lake.scan("fundamentals").collect()
    assert result["end"].to_list() == [date(2025, 12, 31)]


def test_scan_fundamentals_keeps_periodic_forms_and_amendments(
    tmp_path: Path, lake: UsLake
) -> None:
    """정정본(/A)은 남기고 외국발행사 form은 버린다.

    정정본을 버리면 ``filed <= t`` 최신 규칙이 덮을 행 자체가 없어진다.
    2026-09-20 실측으로 (cik, end, tag) 조합 527,329개가 정정본에만 있었다.
    """
    frame = pl.DataFrame(
        {
            "end": [date(2025, 12, 31)] * 6,
            "form": ["10-K", "10-Q", "10-K/A", "10-Q/A", "20-F", "6-K"],
            "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    _write_snapshot(tmp_path, "fundamentals", "2026-09-19", frame)

    result = lake.scan("fundamentals").collect()
    assert result["form"].to_list() == ["10-K", "10-Q", "10-K/A", "10-Q/A"]


def test_scan_midas_keeps_only_stock(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame(
        {
            "date": [date(2026, 1, 1), date(2026, 1, 1)],
            "ticker": ["AAPL", "SPY"],
            "security_type": ["Stock", "ETF"],
        }
    )
    _write_snapshot(tmp_path, "midas_security_daily", "2026-09-18", frame)

    result = lake.scan("midas_security_daily").collect()
    assert result["ticker"].to_list() == ["AAPL"]


def test_scan_macro_series_drops_sp500(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame(
        {
            "series_id": ["SP500", "FEDFUNDS"],
            "realtime_start": [date(2020, 1, 1), date(2020, 1, 1)],
            "value": [3000.0, 0.25],
        }
    )
    _write_snapshot(tmp_path, "macro_series", "2026-09-19", frame)

    result = lake.scan("macro_series").collect()
    assert result["series_id"].to_list() == ["FEDFUNDS"]


def test_scan_short_interest_prefers_revision_row(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame(
        {
            "settlement_date": [date(2026, 9, 15), date(2026, 9, 15), date(2026, 8, 31)],
            "symbol": ["AAPL", "AAPL", "AAPL"],
            "current_short_qty": [100, 999, 50],
            "revision_flag": [False, True, False],
        }
    )
    _write_snapshot(tmp_path, "short_interest", "2026-09-19", frame)

    result = lake.scan("short_interest").collect().sort("settlement_date")
    # 8/31 행은 정정이 없어 그대로, 9/15는 revision_flag=True(999)만 남는다.
    assert result["current_short_qty"].to_list() == [50, 999]
    assert result["revision_flag"].to_list() == [False, True]


def test_scan_passthrough_for_table_without_cleaner(tmp_path: Path, lake: UsLake) -> None:
    frame = pl.DataFrame({"date": [date(2020, 1, 1)], "symbol": ["AAPL"]})
    _write_snapshot(tmp_path, "universe_daily", "2026-09-18", frame)

    result = lake.scan("universe_daily").collect()
    assert result.to_dicts() == frame.to_dicts()


# --- snapshot_manifest --------------------------------------------------------


def test_snapshot_manifest_returns_all_18_tables_with_their_own_dates(
    tmp_path: Path, lake: UsLake
) -> None:
    frame = pl.DataFrame({"date": [date(2020, 1, 1)]})
    dates = {"prices_daily": "2026-09-18", "fundamentals": "2026-09-19"}
    for table in US_TABLES:
        _write_snapshot(tmp_path, table, dates.get(table, "2026-09-19"), frame)

    manifest = lake.snapshot_manifest()

    assert set(manifest) == set(US_TABLES)
    assert manifest["prices_daily"] == "2026-09-18"
    assert manifest["fundamentals"] == "2026-09-19"


# --- acceptance_datetime_to_et -----------------------------------------------


def test_acceptance_datetime_to_et_16_00_boundary() -> None:
    # 2026-01-05는 EST(UTC-5)다. ET 15:59:00 / 16:00:00 / 16:01:00은
    # 각각 UTC 20:59:00 / 21:00:00 / 21:01:00.
    utc_times = [
        datetime(2026, 1, 5, 20, 59, 0, tzinfo=UTC),
        datetime(2026, 1, 5, 21, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 5, 21, 1, 0, tzinfo=UTC),
    ]
    frame = pl.DataFrame({"acceptance_datetime": utc_times})

    result = frame.select(acceptance_datetime_to_et(pl.col("acceptance_datetime")))
    et_times = result["acceptance_datetime"].to_list()

    assert [(t.hour, t.minute) for t in et_times] == [(15, 59), (16, 0), (16, 1)]


def test_acceptance_datetime_to_et_handles_dst_boundary() -> None:
    # 2026년 미국 서머타임은 3/8 시작(EST->EDT), 11/1 종료(EDT->EST)다(3월 둘째
    # 일요일 · 11월 첫째 일요일, 2026-09-20 확인). 전환 다음 날인 3/9 16:00 ET는
    # EDT(UTC-4)라 UTC 20:00이다 — 고정 -5시간 오프셋을 쓰면 15:00으로 틀린다.
    edt_instant = datetime(2026, 3, 9, 20, 0, 0, tzinfo=UTC)
    frame = pl.DataFrame({"acceptance_datetime": [edt_instant]})

    result = frame.select(acceptance_datetime_to_et(pl.col("acceptance_datetime")))
    (et_time,) = result["acceptance_datetime"].to_list()

    assert (et_time.hour, et_time.minute) == (16, 0)
