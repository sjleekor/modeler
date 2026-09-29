"""KR 입력 로더 테스트 — 합성 parquet만 쓴다(실데이터·실제 stock_data 접근 없음)."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from modeler.scores.common import assets as assets_mod
from modeler.scores.common.assets import (
    ASSET_REGISTRY_VERSION,
    ASSETS,
    KR_SECTOR_CANDIDATES,
    activate_kr_sectors,
    active_kr_sector_ids,
    assets_for_market,
    deactivate_kr_sectors,
    get_asset,
    kr_index_key,
    registry_hash,
    registry_version,
)
from modeler.scores.common.calendar import UTC_TS, SessionCalendar
from modeler.scores.common.cash import build_cash_account
from modeler.scores.common.kr_inputs import (
    KRX_AVAILABLE_AT_BASIS,
    KrLake,
    KrNotSyncedError,
    kr_session_calendar,
    krx_index_available_at,
    load_kr_index_paths,
    load_kr_macro,
    load_kr_rates,
)
from modeler.scores.common.panel import assert_pit
from modeler.scores.common.total_return import RETURN_BASIS_PRICE
from modeler.scores.market_sector import build_panel as bp
from modeler.scores.market_sector.config import MsConfig
from modeler.scores.market_sector.features import (
    MACRO_SERIES_IDS,
    build_features,
    fred_available_at,
)

SNAP = "2026-09-29"
SOURCE = "sj2_remote"
N_SESSIONS = 330
FRI = date(2024, 1, 5)


def _sessions(n: int = N_SESSIONS) -> list[date]:
    cal = SessionCalendar.from_exchange_calendars("XKRX", date(2024, 1, 2), date(2025, 12, 31))
    assert cal is not None
    return list(cal.sessions[:n])


def _write(root: Path, table: str, df: pl.DataFrame, *, snap: str = SNAP) -> None:
    d = root / "kr" / "raw" / "raw_postgres" / f"snapshot_date={snap}" / f"source={SOURCE}" / table
    d.mkdir(parents=True, exist_ok=True)
    df.write_parquet(d / "part-0.parquet")


def _marker(root: Path, tables: list[str], *, snap: str = SNAP) -> None:
    m = root / "kr" / "raw" / "raw_postgres" / f"snapshot_date={snap}" / f"source={SOURCE}"
    (m / "_manifests").mkdir(parents=True, exist_ok=True)
    (m / "_manifests" / "_SUCCESS.json").write_text(
        json.dumps({"tables": {t: {"rows": 1} for t in tables}})
    )


def _index_rows(sessions: list[date]) -> pl.DataFrame:
    rows = []
    specs = [
        ("kospi", "KOSPI", "코스피", 2000.0, 1.0),
        ("kosdaq", "KOSDAQ", "코스닥", 800.0, 0.5),
        ("krx", "KRX", "KRX 300 금융", 500.0, 0.3),
        # 같은 이름이 두 그룹에 있다 — 그룹으로 구분해야 한다
        ("kospi", "KOSPI", "건설", 100.0, 9.0),
        ("kosdaq", "KOSDAQ", "건설", 200.0, 7.0),
    ]
    for g, c, n, base, step in specs:
        for i, d in enumerate(sessions):
            rows.append((d, g, c, n, Decimal(f"{base + step * i:.2f}")))
    for d in sessions:  # 쓸 수 없는 지수: close 전부 null
        rows.append((d, "kospi", "KOSPI", "코스피 (외국주포함)", None))
    return pl.DataFrame(
        rows,
        schema={
            "bas_dd": pl.Date,
            "index_group": pl.String,
            "idx_clss": pl.String,
            "idx_nm": pl.String,
            "close_idx": pl.Decimal(18, 2),
        },
        orient="row",
    )


def _obs_rows(
    sessions: list[date], *, with_avail: bool = True, extra_dupe: bool = False
) -> pl.DataFrame:
    rows = []
    series = {
        "rate_kr_cd91": (3.0, 0.0001),
        "fx_usdkrw_ecos": (1300.0, 0.5),
        "foreign_net_kospi_ecos": (100.0, 1.0),
        "trdval_kospi_ecos": (10000.0, 1.0),
    }
    for sid, (base, step) in series.items():
        for i, d in enumerate(sessions):
            nxt = sessions[i + 1] if i + 1 < len(sessions) else d + timedelta(days=1)
            rows.append(
                (
                    sid,
                    d,
                    Decimal(f"{base + step * i:.8f}"),
                    nxt if with_avail else None,
                    datetime(2026, 9, 29, tzinfo=UTC),
                )
            )
    if extra_dupe:  # 같은 관측일 두 행 -> 나중에 받은 행이 이긴다
        rows.append(
            (
                "fx_usdkrw_ecos",
                sessions[0],
                Decimal("9999.00000000"),
                sessions[1],
                datetime(2026, 9, 30, tzinfo=UTC),
            )
        )
    return pl.DataFrame(
        rows,
        schema={
            "series_id": pl.String,
            "observation_date": pl.Date,
            "value_numeric": pl.Decimal(20, 8),
            "available_from_date": pl.Date,
            "fetched_at": pl.Datetime("us", "UTC"),
        },
        orient="row",
    )


@pytest.fixture
def kr_root(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    ss = _sessions()
    _write(tmp_path, "krx_index_daily", _index_rows(ss))
    _write(tmp_path, "common_feature_observation_raw", _obs_rows(ss))
    _marker(tmp_path, ["krx_index_daily", "common_feature_observation_raw"])
    yield tmp_path
    deactivate_kr_sectors()


def _lake(root: Path) -> KrLake:
    from modeler.etl.config import DataRoot

    return KrLake.resolve(DataRoot(root / "kr"))


# --------------------------------------------------------------------------- 스냅샷·오류
def test_missing_snapshot_raises(tmp_path):
    from modeler.etl.config import DataRoot

    (tmp_path / "kr" / "raw").mkdir(parents=True)
    with pytest.raises(KrNotSyncedError):
        KrLake.resolve(DataRoot(tmp_path / "kr"))


def test_missing_table_in_marker_raises(tmp_path):
    from modeler.etl.config import DataRoot

    ss = _sessions(10)
    _write(tmp_path, "common_feature_observation_raw", _obs_rows(ss))
    _marker(tmp_path, ["common_feature_observation_raw"])  # krx_index_daily 없음
    with pytest.raises(KrNotSyncedError, match="krx_index_daily"):
        KrLake.resolve(DataRoot(tmp_path / "kr"))


def test_marker_without_parquet_raises(tmp_path):
    from modeler.etl.config import DataRoot

    ss = _sessions(10)
    _write(tmp_path, "krx_index_daily", _index_rows(ss))
    _marker(tmp_path, ["krx_index_daily", "common_feature_observation_raw"])  # 파일은 한 표뿐
    with pytest.raises(KrNotSyncedError, match="common_feature_observation_raw"):
        KrLake.resolve(DataRoot(tmp_path / "kr"))


def test_falls_back_to_older_complete_snapshot(tmp_path):
    from modeler.etl.config import DataRoot

    ss = _sessions(10)
    for snap, tables in (("2026-09-20", None), ("2026-09-29", ["krx_index_daily"])):
        _write(tmp_path, "krx_index_daily", _index_rows(ss), snap=snap)
        if tables is None:
            _write(tmp_path, "common_feature_observation_raw", _obs_rows(ss), snap=snap)
            tables = ["krx_index_daily", "common_feature_observation_raw"]
        _marker(tmp_path, tables, snap=snap)
    assert KrLake.resolve(DataRoot(tmp_path / "kr")).snapshot_date == "2026-09-20"


def test_build_kr_and_run_kr_stop_without_data(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    (tmp_path / "kr" / "raw").mkdir(parents=True)
    with pytest.raises(KrNotSyncedError):
        bp.main(["--market", "kr", "--version", "v_none", "--allow-dirty"])
    from modeler.scores.market_sector.run import main as run_main

    with pytest.raises(KrNotSyncedError):
        run_main(["--market", "kr"])
    assert not (tmp_path / "kr" / "datasets").exists()


# --------------------------------------------------------------------------- 지수 경로
def test_index_paths_schema_and_price_only(kr_root):
    lake = _lake(kr_root)
    keys = {"kr_kospi": ("kospi", "코스피"), "kr_kosdaq": ("kosdaq", "코스닥")}
    paths, diags = load_kr_index_paths(lake, keys)
    for aid, p in paths.items():
        assert p.columns == ["session", "px_raw", "px_adj", "div_adj", "tr_index", "return_basis"]
        assert set(p["return_basis"]) == {RETURN_BASIS_PRICE}
        assert (p["px_raw"] == p["px_adj"]).all()
        assert (p["div_adj"] == 0).all()  # 배당을 더하지 않는다
        assert p["session"].is_sorted() and p["session"].n_unique() == p.height
        assert diags[aid]["dividend_rows"] == 0
    k = paths["kr_kospi"]
    assert k["px_raw"][0] == pytest.approx(2000.0)
    assert k["tr_index"][1] == pytest.approx(2001.0 / 2000.0)
    assert paths["kr_kosdaq"]["px_raw"][0] == pytest.approx(800.0)


def test_same_name_in_two_groups_is_not_mixed(kr_root):
    lake = _lake(kr_root)
    paths, _ = load_kr_index_paths(lake, {"a": ("kospi", "건설"), "b": ("kosdaq", "건설")})
    assert paths["a"]["px_raw"][0] == pytest.approx(100.0)
    assert paths["b"]["px_raw"][0] == pytest.approx(200.0)
    assert paths["a"].height == paths["b"].height == N_SESSIONS


def test_missing_or_all_null_index_raises(kr_root):
    lake = _lake(kr_root)
    with pytest.raises(ValueError, match="행이 없습니다"):
        load_kr_index_paths(lake, {"x": ("krx", "KRX 없는지수")})
    with pytest.raises(ValueError, match="종가가 없습니다"):
        load_kr_index_paths(lake, {"x": ("kospi", "코스피 (외국주포함)")})


def test_off_calendar_rows_are_dropped_and_counted(kr_root):
    lake = _lake(kr_root)
    ss = _sessions()
    keep = frozenset(ss[:-3])
    paths, diags = load_kr_index_paths(lake, {"kr_kospi": ("kospi", "코스피")}, sessions=keep)
    assert paths["kr_kospi"].height == N_SESSIONS - 3
    assert diags["kr_kospi"]["prices_off_calendar"] == 3


# --------------------------------------------------------------------------- available_at·PIT
def test_krx_available_at_is_next_calendar_day_0830_kst():
    a = krx_index_available_at(date(2024, 1, 3))  # 수
    assert a == datetime(2024, 1, 3, 23, 30, tzinfo=UTC)  # 목 08:30 KST
    assert krx_index_available_at(FRI) == datetime(2024, 1, 5, 23, 30, tzinfo=UTC)  # 토 08:30 KST


def test_panel_pit_holds_on_xkrx_sessions_with_equality(kr_root):
    lake = _lake(kr_root)
    keys = {"kr_kospi": ("kospi", "코스피"), "kr_kosdaq": ("kosdaq", "코스닥")}
    cal = kr_session_calendar(_sessions())
    assert cal.calendar_id == "XKRX"
    paths, _ = load_kr_index_paths(lake, keys, sessions=frozenset(cal.sessions))
    panel, labels = bp.assemble(
        [get_asset("kr_kospi"), get_asset("kr_kosdaq")],
        paths,
        cal,
        None,
        available_at_fn=krx_index_available_at,
        available_at_basis=KRX_AVAILABLE_AT_BASIS,
    )
    assert panel.height > 0 and labels.height == panel.height
    assert (panel["price_available_at"] <= panel["decision_at"]).all()
    # 연속한 평일 세션에서는 등호가 성립한다(08:30 KST == 08:30 KST)
    assert (panel["price_available_at"] == panel["decision_at"]).sum() > 0
    assert set(panel["available_at_basis"]) == {KRX_AVAILABLE_AT_BASIS}
    assert set(panel["return_basis"]) == {RETURN_BASIS_PRICE}
    assert panel["price_available_at"].dtype == UTC_TS
    assert_pit(panel, ["price_available_at"])
    # 한 시간 늦추면 위반이다 — 검사가 실제로 작동한다
    late = panel.with_columns(pl.col("price_available_at") + pl.duration(hours=1))
    with pytest.raises(AssertionError):
        assert_pit(late, ["price_available_at"])


def test_default_panel_basis_unchanged():
    # KR 인자를 안 주면 기존 US 규칙(폐장 + 60분)이 그대로다
    from modeler.scores.common.panel import AVAILABLE_AT_BASIS

    assert AVAILABLE_AT_BASIS == "session_close_plus_60min"


# --------------------------------------------------------------------------- 현금·거시
def test_cash_series_mapping(kr_root):
    lake = _lake(kr_root)
    ss = _sessions()
    rates = load_kr_rates(lake, "rate_kr_cd91", ss)
    assert rates is not None
    assert rates.columns == ["date", "realtime_start", "value"]
    assert rates["value"].dtype == pl.Float64
    assert rates["date"][0] == ss[0] and rates["realtime_start"][0] == ss[1]
    assert rates["value"][0] == pytest.approx(3.0)
    acct = build_cash_account(rates, ss, series_id="rate_kr_cd91")
    f = acct.frame
    # ss[1] 세션에서 ss[0] 관측이 처음 쓰인다 (T+1)
    assert f["rate_obs_date"][0] is None and f["step_status"][0] == "no_rate_yet"
    assert f["rate_obs_date"][1] == ss[0] and f["step_status"][1] == "ok"
    assert load_kr_rates(lake, "rate_kr_no_such", ss) is None


def test_rates_fallback_next_session_when_available_from_date_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    ss = _sessions(20)
    _write(tmp_path, "krx_index_daily", _index_rows(ss))
    _write(tmp_path, "common_feature_observation_raw", _obs_rows(ss, with_avail=False))
    _marker(tmp_path, ["krx_index_daily", "common_feature_observation_raw"])
    rates = load_kr_rates(_lake(tmp_path), "rate_kr_cd91", ss)
    assert rates is not None
    assert rates["realtime_start"].to_list()[:-1] == ss[1:]  # 다음 XKRX 세션


def test_available_from_date_not_after_observation_is_pushed_forward(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    ss = _sessions(10)
    obs = _obs_rows(ss).with_columns(pl.col("observation_date").alias("available_from_date"))
    _write(tmp_path, "krx_index_daily", _index_rows(ss))
    _write(tmp_path, "common_feature_observation_raw", obs)
    _marker(tmp_path, ["krx_index_daily", "common_feature_observation_raw"])
    rates = load_kr_rates(_lake(tmp_path), "rate_kr_cd91", ss)
    assert rates is not None
    assert (rates["realtime_start"] > rates["date"]).all()


def test_macro_frames_and_available_at(kr_root):
    lake = _lake(kr_root)
    ss = _sessions()
    m = load_kr_macro(lake, ss)
    for df in (m.fx, m.foreign_net, m.trdval):
        assert df.columns == ["date", "value", "available_at"]
        assert df["available_at"].dtype == UTC_TS
        assert df["available_at"].is_sorted()
        # 관측일 다음 세션 08:30 KST 이하로 당겨지지 않는다
        assert (df["available_at"] > df["date"].cast(pl.Datetime("us", "UTC"))).all()
    # ss[0] 관측 -> ss[1]일 08:30 KST
    assert m.fx["available_at"][0] == datetime(
        ss[1].year, ss[1].month, ss[1].day, 8, 30, tzinfo=UTC
    ) - timedelta(hours=9)


def test_macro_dedupes_same_observation_date_keeping_latest_fetch(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    ss = _sessions(80)
    _write(tmp_path, "krx_index_daily", _index_rows(ss))
    _write(tmp_path, "common_feature_observation_raw", _obs_rows(ss, extra_dupe=True))
    _marker(tmp_path, ["krx_index_daily", "common_feature_observation_raw"])
    m = load_kr_macro(_lake(tmp_path), ss)
    assert m.fx.height == len(ss)
    assert m.fx["value"][0] == pytest.approx(9999.0)


def test_macro_missing_series_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    ss = _sessions(20)
    obs = _obs_rows(ss).filter(pl.col("series_id") != "fx_usdkrw_ecos")
    _write(tmp_path, "krx_index_daily", _index_rows(ss))
    _write(tmp_path, "common_feature_observation_raw", obs)
    _marker(tmp_path, ["krx_index_daily", "common_feature_observation_raw"])
    with pytest.raises(KrNotSyncedError, match="fx_usdkrw_ecos"):
        load_kr_macro(_lake(tmp_path), ss)


def test_non_monotone_availability_is_raised_to_running_max(tmp_path, monkeypatch):
    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    ss = _sessions(80)
    obs = _obs_rows(ss).with_columns(
        pl.when(pl.col("observation_date") == ss[5])
        .then(pl.lit(ss[20]))  # 옛 관측이 늦게 백필됨
        .otherwise(pl.col("available_from_date"))
        .alias("available_from_date")
    )
    _write(tmp_path, "krx_index_daily", _index_rows(ss))
    _write(tmp_path, "common_feature_observation_raw", obs)
    _marker(tmp_path, ["krx_index_daily", "common_feature_observation_raw"])
    m = load_kr_macro(_lake(tmp_path), ss)
    assert m.fx["available_at"].is_sorted()  # kr_foreign_net_20_over_trdval이 요구하는 단조


# --------------------------------------------------------------------------- 등록부
def test_kr_sector_candidates_are_not_active():
    assert len(KR_SECTOR_CANDIDATES) == 5
    active_ids = {a.asset_id for a in ASSETS}
    assert not active_ids & {c.asset_id for c in KR_SECTOR_CANDIDATES}
    assert ASSET_REGISTRY_VERSION == "ms_assets_v1"
    assert registry_version() == ASSET_REGISTRY_VERSION
    assert {a.asset_id for a in assets_for_market("KR")} == {"kr_kospi", "kr_kosdaq"}
    assert kr_index_key("kr_kospi") == ("kospi", "코스피")
    assert kr_index_key("kr_fin") == ("krx", "KRX 300 금융")
    with pytest.raises(KeyError):
        get_asset("kr_fin")
    with pytest.raises(KeyError):
        kr_index_key("us_spx")


def test_activate_kr_sectors_changes_version_and_hash():
    base = registry_hash()
    try:
        got = activate_kr_sectors(["kr_fin", "kr_tech"])
        assert [a.asset_id for a in got] == ["kr_fin", "kr_tech"]
        assert get_asset("kr_fin").parent_benchmark == "kr_kospi"
        assert get_asset("kr_fin").asset_type == "sector"
        assert registry_version() == "ms_assets_v1+kr_sectors"
        assert registry_hash() != base
        assert active_kr_sector_ids() == ("kr_fin", "kr_tech")
        assert {a.asset_id for a in assets_for_market("KR")} >= {"kr_fin", "kr_tech"}
        activate_kr_sectors(["kr_fin"])  # 멱등
        assert active_kr_sector_ids() == ("kr_fin", "kr_tech")
        with pytest.raises(KeyError):
            activate_kr_sectors(["kr_bank"])
    finally:
        deactivate_kr_sectors()
    assert registry_hash() == base and registry_version() == "ms_assets_v1"
    assert assets_mod.active_assets() == ASSETS


# --------------------------------------------------------------------------- 엔드투엔드
def test_build_kr_end_to_end_and_features(kr_root, monkeypatch):
    monkeypatch.setattr(bp, "git_commit", lambda *a, **k: "test-commit")
    rc = bp.main(
        ["--market", "kr", "--version", "ms_kr_test", "--kr-sectors", "kr_fin", "--allow-dirty"]
    )
    assert rc == 0
    out = kr_root / "kr" / "datasets" / "market_sector" / "ms_kr_test"
    assert {p.name for p in out.iterdir()} == {
        "panel.parquet",
        "labels.parquet",
        "manifest.json",
        "readiness.json",
    }
    man = json.loads((out / "manifest.json").read_text())
    assert man["market"] == "KR"
    assert man["kr_sectors_activated"] == ["kr_fin"]
    assert man["asset_registry_version"] == "ms_assets_v1+kr_sectors"
    assert man["time_contract"]["price_available_at_basis"] == KRX_AVAILABLE_AT_BASIS
    assert man["time_contract"]["return_basis"] == "price_only"
    assert man["cash"]["series_present_in_lake"] is True
    assert set(man["inputs"]) == {"krx_index_daily", "common_feature_observation_raw"}
    assert man["inputs"]["krx_index_daily"]["snapshot_date"] == SNAP
    panel = pl.read_parquet(out / "panel.parquet")
    labels = pl.read_parquet(out / "labels.parquet")
    assert set(panel["asset_id"]) == {"kr_kospi", "kr_kosdaq", "kr_fin"}
    assert (panel["price_available_at"] <= panel["decision_at"]).all()
    assert set(panel["calendar_basis"]).pop().startswith("exchange_calendars")
    # 현금 라벨이 KR 금리로 채워진다
    assert labels.filter(pl.col("asset_id") == "kr_kospi")[
        "excess_return_60d_vs_cash"
    ].null_count() < (labels.filter(pl.col("asset_id") == "kr_kospi").height)
    # 이미 있는 버전은 덮지 않는다
    with pytest.raises(FileExistsError):
        bp.main(["--market", "kr", "--version", "ms_kr_test", "--kr-sectors", "kr_fin"])

    # 피쳐: KR 전용 둘이 채워지고 PIT가 지켜진다
    lake = _lake(kr_root)
    kr_macro = load_kr_macro(lake, sorted(panel["session"].unique().to_list()))
    rows = []
    for sid in MACRO_SERIES_IDS:
        for i, d in enumerate(sorted(panel["session"].unique().to_list())):
            rows.append((sid, d, d, 3.0 + 0.001 * i + (1.0 if sid == "VIXCLS" else 0.0)))
    macro = fred_available_at(
        pl.DataFrame(rows, schema=["series_id", "date", "realtime_start", "value"], orient="row")
    )
    f = build_features(panel, macro, MsConfig(), market="KR", kr_macro=kr_macro)
    assert (f["market_is_kr"] == 1).all()
    assert f["usdkrw_ret_60"].null_count() < f.height
    assert f["kr_foreign_net_20_over_trdval"].null_count() < f.height
    assert (f["feature_available_at"] <= f["decision_at"]).all()
    assert "asset_kr_fin" in f.columns
    # 합성값: 순매수 100+i, 거래대금 10,000+i -> 20일 합 비율은 1%에서 4% 사이
    ratio = f["kr_foreign_net_20_over_trdval"].drop_nulls()
    assert ratio.min() > 0.01 and ratio.max() < 0.05


def test_us_market_features_unchanged_columns():
    """US 경로: KR 전용 둘은 0으로 채워지고 market_is_kr=0 (기존 동작)."""
    from ._helpers import make_layer2_frame

    frame, _macro, _panel = make_layer2_frame(n_sessions=300)
    assert (frame["market_is_kr"] == 0).all()
    assert (frame["usdkrw_ret_60"] == 0.0).all()
