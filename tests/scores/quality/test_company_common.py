"""company_common 시험. 레이크가 없으면 데이터 시험은 건너뛴다."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from modeler.scores.quality import company_common as cc

LAKE_ROOT = cc.stock_data_root()
needs_lake = pytest.mark.skipif(
    not (LAKE_ROOT / "kr/derived/feature/snapshot_date=2026-09-29").is_dir()
    or not (LAKE_ROOT / "kr/raw/raw_postgres/snapshot_date=2026-09-30").is_dir(),
    reason="레이크 없음",
)


def test_base_date_and_constants():
    assert cc.base_date(2024) == date(2025, 6, 30)
    assert cc.base_date(2019) == date(2020, 6, 30)
    assert cc.JUDGMENT_YEARS == (2019, 2020, 2021, 2022, 2023, 2024)
    assert cc.DEV_YEARS == (2017, 2018) and cc.FIRST_FIN_YEAR == 2015


def test_guard_years(monkeypatch):
    monkeypatch.delenv(cc.JUDGMENT_ENV, raising=False)
    cc.guard_years(cc.JUDGMENT_YEARS, "inputs")
    cc.guard_years(cc.DEV_YEARS, "outcome")
    cc.guard_years(cc.DEV_YEARS, "link")
    for p in ("outcome", "link"):
        with pytest.raises(PermissionError):
            cc.guard_years((2018, 2019), p)
    with pytest.raises(ValueError):
        cc.guard_years((2017,), "etc")
    monkeypatch.setenv(cc.JUDGMENT_ENV, "yes")
    cc.guard_years(cc.JUDGMENT_YEARS, "outcome")


def test_lake_paths(tmp_path, monkeypatch):
    lake = cc.Lake(tmp_path, raw_snapshot="2026-10-18", derived_snapshot="2026-09-29")
    assert lake.raw_dir("t") == (
        tmp_path / "kr/raw/raw_postgres/snapshot_date=2026-10-18/source=sj2_remote/t"
    )
    assert "snapshot_date=2026-09-29" in str(lake.derived_dir("t"))
    assert lake.output_dir("x") == tmp_path / "kr/output/x"
    with pytest.raises(FileNotFoundError, match="레이크 경로가 없습니다"):
        lake.raw_glob("t")
    d = lake.derived_dir("cal")
    d.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        lake.derived_glob("cal")
    pl.DataFrame({"a": [1]}).write_parquet(d / "p.parquet")
    assert lake.derived_glob("cal").endswith("**/*.parquet")
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    assert cc.stock_data_root() == tmp_path


def test_calendar_synthetic():
    days = pl.Series([date(2025, 6, 26), date(2025, 6, 27), date(2025, 6, 30)])
    cal = cc.TradingCalendar(days.to_numpy())
    assert cal.next_trading_day(date(2025, 6, 27)) == date(2025, 6, 30)
    assert cal.next_trading_day(date(2025, 6, 28)) == date(2025, 6, 30)
    assert cal.next_trading_day(date(2025, 6, 30)) is None
    s = pl.Series("d", [date(2025, 6, 26), None, date(2025, 6, 30)])
    assert cal.next_trading_days(s).to_list() == [date(2025, 6, 27), None, None]


def test_available_on_or_before_synthetic():
    avail = pl.DataFrame(
        {"rcept_no": ["a", "b", "c"], "avail_date": [date(2025, 6, 30), date(2025, 7, 1), None]}
    )
    r = cc.available_on_or_before(pl.Series("rcept_no", ["a", "b", "c", "z"]), 2024, avail)
    assert r["status"].to_list() == [
        cc.STATUS_OK,
        cc.STATUS_AFTER_BASE,
        cc.STATUS_NO_AVAIL,
        cc.STATUS_MISSING_RECEIPT,
    ]
    assert r["available"].to_list() == [True, False, False, False]
    assert cc.is_available(date(2025, 6, 30), 2024) and not cc.is_available(None, 2024)


def test_hash_and_git(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("abc")
    assert cc.sha256_file(f) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert len(cc.code_sha256(cc.__file__)) == 64
    assert cc.git_head(tmp_path) is None


@pytest.fixture(scope="module")
def lake():
    return cc.Lake(LAKE_ROOT)


@pytest.fixture(scope="module")
def cal(lake):
    return cc.TradingCalendar.from_lake(lake)


@needs_lake
def test_real_calendar(cal):
    assert cal.next_trading_day(date(2025, 6, 27)) == date(2025, 6, 30)
    assert cal.next_trading_day(date(2025, 3, 21)) == date(2025, 3, 24)


@needs_lake
def test_filing_availability(lake, cal):
    av = cc.filing_availability(lake, cal)
    assert av["rcept_no"].is_unique().all()
    got = dict(zip(av["rcept_no"], av["avail_date"]))
    assert got["20250321001375"] == date(2025, 3, 24)
    assert got["20260506000539"] == date(2026, 5, 7)
    if "20190329004479" in got:
        assert got["20190329004479"] == date(2019, 4, 2)
    r = cc.available_on_or_before(pl.Series("rcept_no", ["20250321001375", "nope"]), 2024, av)
    assert r["status"].to_list() == [cc.STATUS_OK, cc.STATUS_MISSING_RECEIPT]


@needs_lake
def test_mismatch_counts(lake, cal):
    c = cc.availability_mismatch_counts(lake, cal)
    assert c["receipt_rows"] > 1_000_000
    assert 0 <= c["vintage_mismatch"] <= c["vintage_rcept"]
    print(c)
