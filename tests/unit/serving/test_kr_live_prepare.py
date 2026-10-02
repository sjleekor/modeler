from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from collector.lake import DataRoot
from modeler.etl.config import CONFIG_TABLES, RAW_TABLES, LakeConfig, REMOTE_SOURCE
from modeler.serving import kr_live_prepare as live


def _holiday_csv(path: Path, days: list[str]) -> Path:
    path.write_text("date,name\n" + "".join(f"{day},closure\n" for day in days), encoding="utf-8")
    return path


def test_holiday_calendar_fails_closed(tmp_path: Path) -> None:
    upper = date(2026, 9, 30)
    with pytest.raises(FileNotFoundError, match="holiday calendar CSV is missing"):
        live.load_holiday_calendar(upper=upper, feature_asof_date="2026-09-29", path=tmp_path / "none.csv")
    empty = _holiday_csv(tmp_path / "empty.csv", [])
    with pytest.raises(ValueError, match="no rows"):
        live.load_holiday_calendar(upper=upper, feature_asof_date="2026-09-29", path=empty)
    old = _holiday_csv(tmp_path / "old.csv", ["2025-12-31"])
    with pytest.raises(ValueError, match="coverage through year 2026"):
        live.load_holiday_calendar(upper=upper, feature_asof_date="2026-09-29", path=old)
    # K in a later year than upper is impossible in practice, but the larger year governs.
    ok = _holiday_csv(tmp_path / "ok.csv", ["2026-12-31"])
    with pytest.raises(ValueError, match="coverage through year 2027"):
        live.load_holiday_calendar(upper=upper, feature_asof_date="2027-01-04", path=ok)
    holidays, record = live.load_holiday_calendar(upper=upper, feature_asof_date="2026-09-29", path=ok)
    assert holidays == {date(2026, 12, 31)} and record["sha256"] == live._sha256(ok)


def test_packaged_holiday_calendar_is_read_when_present() -> None:
    _, record = live.load_holiday_calendar(upper=date(2026, 9, 30), feature_asof_date="2026-09-29")
    assert record["name"] == live.HOLIDAY_CALENDAR_NAME and record["row_count"] > 0


def _raw_marker(
    config: LakeConfig, *, tables: set[str] | None = None, route: str = "remote",
    pg_snapshot_id: str | None = None,
) -> Path:
    path = config.raw_root / "_manifests" / "_SUCCESS.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    selected = tables if tables is not None else set(RAW_TABLES) | set(CONFIG_TABLES)
    entries = {}
    source = {"name": REMOTE_SOURCE, "snapshot_date": config.snapshot_date}
    if pg_snapshot_id is not None:
        source |= {"snapshot_policy": live.RAW_POLICY_EXPORTED_SNAPSHOT,
                   "pg_snapshot_id": pg_snapshot_id}
    for name in selected:
        detail_path = path.parent / "table_manifests" / f"{name}.json"
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        detail_path.write_text(json.dumps({
            "source": source,
            "table": {"name": name, "rows_exported": 1, "schema": {"hash": "fixture"}},
        }))
        entries[name] = {"manifest_path": str(detail_path), "rows_exported": 1, "schema_hash": "fixture"}
    body = {
        "route": route, "finished_at": "2026-09-30T00:30:00+09:00",
        "snapshot_policy": live.RAW_POLICY_PER_CHUNK, "tables": entries,
    }
    if pg_snapshot_id is not None:
        body |= {"snapshot_policy": live.RAW_POLICY_EXPORTED_SNAPSHOT,
                 "pg_snapshot_id": pg_snapshot_id}
    path.write_text(json.dumps(body))
    return path


def test_raw_capture_accepts_one_exported_snapshot_for_every_table(tmp_path: Path) -> None:
    from datetime import datetime

    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    cutoff = datetime.fromisoformat("2026-10-01T09:30:00+09:00")
    marker = _raw_marker(config, pg_snapshot_id="00000003-0000002A-1")
    digest = live.verify_raw(config, cutoff=cutoff, feature_asof_date="2026-09-30")
    assert digest == live._sha256(marker)
    detail = config.raw_root / "_manifests" / "table_manifests" / "daily_ohlcv.json"
    changed = json.loads(detail.read_text())
    changed["source"]["pg_snapshot_id"] = "00000003-0000002B-1"
    detail.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="marker's snapshot"):
        live.verify_raw(config, cutoff=cutoff, feature_asof_date="2026-09-30")
    body = json.loads(marker.read_text())
    body.pop("pg_snapshot_id")
    marker.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="no pg_snapshot_id"):
        live.verify_raw(config, cutoff=cutoff, feature_asof_date="2026-09-30")
    body["snapshot_policy"] = "something_else"
    marker.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="snapshot policy"):
        live.verify_raw(config, cutoff=cutoff, feature_asof_date="2026-09-30")


def test_raw_capture_requires_full_remote_completed_table_set(tmp_path: Path) -> None:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    from datetime import datetime

    cutoff = datetime.fromisoformat("2026-10-01T09:30:00+09:00")
    marker = _raw_marker(config)
    assert live.verify_raw(config, cutoff=cutoff, feature_asof_date="2026-09-30") == live._sha256(marker)
    _raw_marker(config, tables={"daily_ohlcv"})
    with pytest.raises(ValueError, match="table set"):
        live.verify_raw(config, cutoff=cutoff, feature_asof_date="2026-09-30")
    _raw_marker(config, route="local")
    with pytest.raises(ValueError, match="sj2 direct route"):
        live.verify_raw(config, cutoff=cutoff, feature_asof_date="2026-09-30")
    _raw_marker(config)
    detail = config.raw_root / "_manifests" / "table_manifests" / "daily_ohlcv.json"
    changed = json.loads(detail.read_text())
    changed["source"]["snapshot_date"] = "2026-09-29"
    detail.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="snapshot/source"):
        live.verify_raw(config, cutoff=cutoff, feature_asof_date="2026-09-30")


_STEP_PATCHES = (
    "materialize_stock_pit", "materialize_price_quality", "materialize_price",
    "materialize_flow", "materialize_universe", "materialize_fin_pit",
    "materialize_filing_activity", "materialize_stock_metric_vintage_fact",
    "materialize_fin_quarterly_metric_vintage", "materialize_fin_scan_daily",
)


def _stub_build(monkeypatch, calls: list[str], seen: dict) -> None:
    """Replace every heavy step with a recorder; the connection answers only count/min-max."""

    class _Result:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class _Connection:
        def execute(self, sql):
            if "receipt_beyond_calendar" in sql:
                return _Result((0, 0))
            if "count(*)" in sql:
                return _Result((7,))
            assert "min(trade_date)" in sql
            return _Result((date(2020, 1, 2), date(2026, 9, 30)))

        def close(self):
            calls.append("close")

    def _connect(config):
        seen["engine"] = config.engine
        return _Connection()

    monkeypatch.setattr(live, "connect", _connect)

    def _days(lower, upper, holidays=None):
        seen["holidays"] = holidays
        # Calendar days stand in for sessions: the builder needs some after K.
        return [date(2026, 9, 30) + timedelta(days=offset) for offset in range(12)]

    monkeypatch.setattr(live, "get_trading_days", _days)
    monkeypatch.setattr(live, "register_views", lambda _con, _config, *, tables: calls.append("raw"))
    for attr in _STEP_PATCHES:
        monkeypatch.setattr(live, attr, lambda *_args, _name=attr, **_kwargs: calls.append(_name))
    monkeypatch.setattr(live, "materialize", lambda *_args, **_kwargs: calls.append("broad_universe"))
    monkeypatch.setattr(live, "register_mart_view", lambda *_args, **_kwargs: calls.append("broad_registered"))
    monkeypatch.setattr(live, "register_derived_marts", lambda *_args, **_kwargs: calls.append("stock_metric_fact"))
    monkeypatch.setattr(live, "_mart_record", lambda _con, _config, name, k: {
        "view": name, "row_count": 1, "max_trade_date": k, "sql_hash": "fixture"
    })
    monkeypatch.setattr(live, "_intermediate_versions", lambda _config, _profile=None: {})


def test_serving_build_seals_only_required_label_free_marts(tmp_path: Path, monkeypatch) -> None:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    _raw_marker(config)
    calls: list[str] = []
    seen: dict = {}
    _stub_build(monkeypatch, calls, seen)
    csv_path = _holiday_csv(tmp_path / "h.csv", ["2026-01-01", "2026-12-31"])
    body = live.build_live_marts(
        snapshot_date="2026-09-30", feature_asof_date="2026-09-30",
        input_cutoff="2026-10-01T09:30:00+09:00", stock_data_root=tmp_path, holidays_csv=csv_path,
    )
    assert seen["holidays"] == {date(2026, 1, 1), date(2026, 12, 31)}
    assert body["holiday_calendar"] == {
        "name": "h.csv", "sha256": live._sha256(csv_path), "row_count": 2,
        "min_date": "2026-01-01", "max_date": "2026-12-31"}
    assert [row["view"] for row in body["marts"]] == list(live.SERVING_MARTS)
    assert "materialize_fin_scan_daily" in calls
    assert calls.index("stock_metric_fact") < calls.index("materialize_fin_pit")
    assert calls.index("materialize_fin_quarterly_metric_vintage") < calls.index("materialize_fin_scan_daily")
    assert "label_scan" not in json.dumps(body)
    marker = config.feature_mart_root / "_manifests" / "_SUCCESS.json"
    assert marker.is_file()
    with pytest.raises(FileExistsError, match="already sealed"):
        live.build_live_marts(
            snapshot_date="2026-09-30", feature_asof_date="2026-09-30",
            input_cutoff="2026-10-01T09:30:00+09:00", stock_data_root=tmp_path, holidays_csv=csv_path,
        )


def test_late_mart_build_is_sealed_but_D_cutoff_rejects_it(tmp_path: Path, monkeypatch) -> None:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    _raw_marker(config)

    class _Connection:
        def close(self):
            pass

    class _LateDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromisoformat("2026-10-01T00:31:00+00:00").astimezone(tz)

    monkeypatch.setattr(live, "connect", lambda _: _Connection())
    monkeypatch.setattr(live, "_build_marts", lambda *_, **__: {"name": "fixture"})
    monkeypatch.setattr(live, "_mart_record", lambda _con, _config, name, k: {
        "view": name, "row_count": 1, "max_trade_date": k, "sql_hash": "fixture"
    })
    monkeypatch.setattr(live, "_intermediate_versions", lambda _config, _profile=None: {})
    monkeypatch.setattr(live, "datetime", _LateDatetime)
    body = live.build_live_marts(
        snapshot_date="2026-09-30", feature_asof_date="2026-09-30",
        input_cutoff="2026-10-01T09:30:00+09:00", stock_data_root=tmp_path,
    )
    assert body["created_at"] == "2026-10-01T00:31:00+00:00"
    from modeler.serving.kr_prepare import _check_snapshot_markers

    with pytest.raises(ValueError, match="after the 09:30 input cutoff"):
        _check_snapshot_markers(
            config, input_cutoff="2026-10-01T09:30:00+09:00",
            feature_asof_date="2026-09-30",
        )


_BUILD_ARGS = dict(
    snapshot_date="2026-09-30", feature_asof_date="2026-09-30",
    input_cutoff="2026-10-01T09:30:00+09:00",
)
_STEP_ORDER = [
    "raw_views", "calendar", "stock_pit", "price_quality", "feat_price", "feat_flow",
    "dim_universe_broad_daily", "dim_universe_daily", "stock_metric_fact", "feat_fin_pit",
    "feat_filing_activity", "stock_metric_vintage_fact", "fin_quarterly_metric_vintage",
    "feat_fin_scan_daily", "verify_marts",
]


def test_build_writes_profile_beside_marker_and_removes_temp(tmp_path: Path, monkeypatch) -> None:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    _raw_marker(config)
    seen: dict = {}
    _stub_build(monkeypatch, [], seen)
    body = live.build_live_marts(**_BUILD_ARGS, stock_data_root=tmp_path)

    manifests = config.feature_mart_root / "_manifests"
    profile = json.loads((manifests / "build_profile.json").read_text(encoding="utf-8"))
    assert profile["status"] == "success" and profile["schema_version"] == "kr-build-profile.v1"
    assert [step["name"] for step in profile["steps"]] == _STEP_ORDER
    assert [step["order"] for step in profile["steps"]] == list(range(1, len(_STEP_ORDER) + 1))
    for step in profile["steps"]:
        assert step["status"] == "success" and step["elapsed_seconds"] >= 0
        assert step["started_at"].endswith("+00:00") and step["finished_at"].endswith("+00:00")
        assert step["peak_temp_bytes"] == 0 and step["row_count"] == (7 if step["output"] else None)
    assert profile["engine"]["max_temp_directory_size"] == "30GB"
    assert profile["engine"]["threads"] == "2" and profile["engine"]["memory_limit"] == "4GB"
    assert {"duckdb_version", "python_version", "platform", "cpu_count"} <= set(
        profile["environment"])
    # The sealed marker is untouched by profiling; kr_prepare validates it as before.
    marker = json.loads((manifests / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert marker == json.loads(json.dumps(body, sort_keys=True))
    # Timing stays out of the marker; the build profile *name* (full / serving) is in it.
    assert not any("profile" in key and key != "profile" for key in marker)
    assert marker["profile"] == "full" and profile["profile"] == "full"
    parent = tmp_path / "kr" / "derived" / "_duckdb_tmp"
    assert Path(seen["engine"].temp_directory).parent == parent
    assert list(parent.iterdir()) == []


def test_failed_build_keeps_temp_dir_with_partial_profile(tmp_path: Path, monkeypatch) -> None:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    _raw_marker(config)
    seen: dict = {}
    _stub_build(monkeypatch, [], seen)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("spill failed")

    monkeypatch.setattr(live, "materialize_fin_pit", _boom)
    custom = tmp_path / "scratch"
    with pytest.raises(RuntimeError, match="spill failed"):
        live.build_live_marts(**_BUILD_ARGS, stock_data_root=tmp_path, temp_dir=custom,
                              max_temp_size="5GB")
    run_dir = Path(seen["engine"].temp_directory)
    assert run_dir.parent == custom and seen["engine"].max_temp_directory_size == "5GB"
    partial = json.loads((run_dir / "build_profile.partial.json").read_text(encoding="utf-8"))
    assert partial["status"] == "failed" and "spill failed" in partial["error"]
    last = partial["steps"][-1]
    assert last["name"] == "feat_fin_pit" and last["status"] == "failed"
    assert not (config.feature_mart_root / "_manifests" / "_SUCCESS.json").exists()
    assert not (config.feature_mart_root / "_manifests" / "build_profile.json").exists()


def test_temp_dir_must_be_absolute_and_limits_are_guarded(tmp_path: Path) -> None:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    _raw_marker(config)
    with pytest.raises(ValueError, match="absolute"):
        live.build_live_marts(**_BUILD_ARGS, stock_data_root=tmp_path, temp_dir=Path("rel"))
    with pytest.raises(ValueError, match="two threads and 4GB"):
        live.build_live_marts(**_BUILD_ARGS, stock_data_root=tmp_path, threads=3)
    with pytest.raises(ValueError, match="max_temp_size"):
        live.build_live_marts(**_BUILD_ARGS, stock_data_root=tmp_path, max_temp_size=" ")
    assert not (tmp_path / "kr" / "derived" / "_duckdb_tmp").exists()


def test_temp_cap_reaches_duckdb_and_stops_a_spilling_query(tmp_path: Path) -> None:
    import duckdb

    from modeler.etl.config import EngineOptions
    from modeler.etl.lake import connect

    engine = EngineOptions(
        threads=1, memory_limit="30MB", temp_directory=str(tmp_path / "new" / "spill"),
        max_temp_directory_size="1MB")
    con = connect(LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE, engine=engine))
    try:
        assert con.execute("SELECT current_setting('max_temp_directory_size')").fetchone()[0]
        with pytest.raises(duckdb.OutOfMemoryException, match="max_temp_directory_size"):
            con.execute("SELECT count(*) FROM (SELECT * FROM range(3000000) t(i) ORDER BY hash(i))")
    finally:
        con.close()
    # Without the option nothing is set: existing callers keep DuckDB's own default.
    assert EngineOptions(threads=2).as_pragmas() == {"threads": "2"}


# --------------------------------------------------------------------------
# E1 — versions the serving builder selects, calendar extension
# --------------------------------------------------------------------------


def test_serving_calendar_extends_past_k_and_stops_at_csv_coverage() -> None:
    holidays = {date(2026, 10, 5), date(2026, 10, 9), date(2026, 12, 25), date(2026, 12, 31)}
    days, record = live.serving_trading_days(
        date(2026, 9, 28), date(2026, 9, 30), feature_asof_date="2026-09-29", holidays=holidays)
    after = [day for day in days if day > date(2026, 9, 30)]
    assert len(after) == live.CALENDAR_EXTENSION_SESSIONS == 10
    assert date(2026, 10, 5) not in days and date(2026, 10, 9) not in days
    assert after[0] == date(2026, 10, 1) and days[0] == date(2026, 9, 28)
    assert record["last_session"] == after[-1].isoformat() == days[-1].isoformat()
    assert record["sessions_after_anchor"] == 10 and record["session_count"] == len(days)
    assert record["sessions_sha256"] == live.hashlib.sha256(
        ",".join(day.isoformat() for day in days).encode()).hexdigest()
    # K later than the last price date anchors the extension at K.
    _, later = live.serving_trading_days(
        date(2026, 9, 28), date(2026, 9, 29), feature_asof_date="2026-09-30", holidays=holidays)
    assert later["anchor_date"] == "2026-09-30"
    # Near the end of the CSV's last year the extension is clipped, not guessed.
    clipped, record = live.serving_trading_days(
        date(2026, 12, 20), date(2026, 12, 24), feature_asof_date="2026-12-24", holidays=holidays)
    assert [d.isoformat() for d in clipped if d > date(2026, 12, 24)] == [
        "2026-12-28", "2026-12-29", "2026-12-30"]
    assert record["sessions_after_anchor"] == 3


def test_serving_calendar_fails_closed_without_a_known_session_after_k() -> None:
    holidays = {date(2026, 12, 31)}
    with pytest.raises(ValueError, match="no session after 2026-12-30"):
        live.serving_trading_days(
            date(2026, 12, 1), date(2026, 12, 30), feature_asof_date="2026-12-30",
            holidays=holidays)


def _vintage_table(*rows: tuple[str, str]):
    import duckdb

    con = duckdb.connect()
    con.execute(
        "CREATE TABLE stock_metric_vintage_fact AS SELECT * FROM (VALUES "
        + ", ".join(f"('{source}', '{rcept}')" for source, rcept in rows)
        + ") t(availability_source, rcept_no)")
    return con


def test_receipts_beyond_the_calendar_are_counted_and_warned_not_fatal(capsys) -> None:
    con = _vintage_table(
        ("rcept_no", "20260101000001"), ("receipt_beyond_calendar", "20991231000001"),
        ("receipt_beyond_calendar", "20991231000001"),
        ("receipt_beyond_calendar", "20991231000002"))
    record = live._check_beyond_calendar(con, max_rows=3)
    assert record == {
        "rows": 3, "filings": 2, "sample_rcept_no": ["20991231000001", "20991231000002"],
        "max_rows": 3}
    assert "3 stock_metric_vintage_fact rows (2 filings)" in capsys.readouterr().err
    con.execute("DELETE FROM stock_metric_vintage_fact WHERE availability_source <> 'rcept_no'")
    assert live._check_beyond_calendar(con, max_rows=0)["rows"] == 0
    assert capsys.readouterr().err == ""


def test_more_beyond_calendar_rows_than_allowed_still_fail_the_build() -> None:
    con = _vintage_table(("receipt_beyond_calendar", "20991231000001"),
                         ("receipt_beyond_calendar", "20991231000002"))
    with pytest.raises(ValueError, match="2 stock_metric_vintage_fact rows .* 1 allowed"):
        live._check_beyond_calendar(con, max_rows=1)
    # The default is not zero: one data error must not stop the nightly build.
    assert live.DEFAULT_MAX_BEYOND_CALENDAR_ROWS > 0
    with pytest.raises(ValueError, match="must not be negative"):
        live._check_beyond_calendar(con, max_rows=-1)


def test_the_marker_records_the_beyond_calendar_count(tmp_path: Path, monkeypatch) -> None:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    _raw_marker(config)
    _stub_build(monkeypatch, [], {})
    record = {"rows": 4, "filings": 1, "sample_rcept_no": ["x"], "max_rows": 500}
    monkeypatch.setattr(live, "_check_beyond_calendar", lambda _con, *, max_rows: record)
    body = live.build_live_marts(**_BUILD_ARGS, stock_data_root=tmp_path)
    sealed = json.loads(
        (config.feature_mart_root / "_manifests" / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert body["beyond_calendar"] == sealed["beyond_calendar"] == record
    assert sealed["profile"] == "full"


def test_serving_selects_the_new_versions_and_records_them(tmp_path: Path, monkeypatch) -> None:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    _raw_marker(config)
    seen: dict = {}
    _stub_build(monkeypatch, [], seen)
    kwargs: dict = {}
    for attr in ("materialize_stock_metric_vintage_fact", "materialize_fin_pit",
                 "materialize_fin_scan_daily", "materialize_fin_quarterly_metric_vintage"):
        monkeypatch.setattr(
            live, attr, lambda *_args, _name=attr, **kw: kwargs.setdefault(_name, kw))
    def _record(_con, _config, name, k):
        v2 = name == "feat_fin_pit"
        return {
            "view": name, "row_count": 1, "max_trade_date": k, "sql_hash": "fixture",
            "semantics_version": "v2" if v2 else "v1", "plan": "single", "plan_hash": None,
        }

    monkeypatch.setattr(live, "_mart_record", _record)
    monkeypatch.setattr(live, "_intermediate_versions", lambda _config, _profile=None: {
        "stock_metric_vintage_fact": {
            "semantics_version": "v2", "plan": "staged", "plan_hash": "h"},
        "fin_quarterly_metric_vintage": {
            "semantics_version": "v1", "plan": "single", "plan_hash": None},
    })
    csv_path = _holiday_csv(tmp_path / "h.csv", ["2026-01-01", "2026-12-31"])
    body = live.build_live_marts(
        **_BUILD_ARGS, stock_data_root=tmp_path, holidays_csv=csv_path)

    smvf = kwargs["materialize_stock_metric_vintage_fact"]
    assert (smvf["semantics"], smvf["plan"]) == ("v2", "staged")
    run_dir = Path(seen["engine"].temp_directory)
    assert smvf["stage_dir"] == run_dir / "stages"  # inside the per-run dir, never shared
    assert smvf["trading_days"][-1] > date(2026, 9, 30)
    assert kwargs["materialize_fin_pit"]["semantics"] == "v2"
    assert kwargs["materialize_fin_scan_daily"]["join_plan"] == "coalesce_join"
    assert kwargs["materialize_fin_scan_daily"]["semantics"] == "v2"
    assert kwargs["materialize_fin_quarterly_metric_vintage"]["semantics"] == "v2"

    assert body["mart_versions"]["stock_metric_vintage_fact"] == {
        "semantics_version": "v2", "plan": "staged", "plan_hash": "h"}
    assert body["mart_versions"]["feat_fin_pit"]["semantics_version"] == "v2"
    assert body["mart_versions"]["feat_price"] == {
        "semantics_version": "v1", "plan": "single", "plan_hash": None}
    assert set(body["mart_versions"]) == set(live.SERVING_MARTS) | set(
        live.INTERMEDIATE_VERSIONED_MARTS)
    calendar = body["trading_calendar"]
    assert calendar["anchor_date"] == "2026-09-30" and calendar["sessions_after_anchor"] == 10
    environment = body["execution_environment"]
    assert (environment["threads"], environment["memory_limit"]) == (2, "4GB")
    assert environment["duckdb"] and environment["python"] and environment["platform"]


def _recorded_build(tmp_path: Path, monkeypatch) -> dict:
    """Run the builder with every step recording its keyword arguments."""
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    _raw_marker(config)
    _stub_build(monkeypatch, [], {})
    kwargs: dict = {}
    for attr in ("materialize_stock_pit", "materialize_price", "materialize_flow"):
        monkeypatch.setattr(
            live, attr, lambda *_args, _name=attr, **kw: kwargs.setdefault(_name, kw))
    monkeypatch.setattr(
        live, "register_derived_marts",
        lambda *_args, **kw: kwargs.setdefault("register_derived_marts", kw))
    csv_path = _holiday_csv(tmp_path / "h.csv", ["2026-01-01", "2026-12-31"])
    live.build_live_marts(
        snapshot_date="2026-09-30", feature_asof_date="2026-09-30",
        input_cutoff="2026-10-01T09:30:00+09:00", stock_data_root=tmp_path, holidays_csv=csv_path)
    return kwargs


def test_serving_selects_the_round_two_plans(tmp_path: Path, monkeypatch) -> None:
    kwargs = _recorded_build(tmp_path, monkeypatch)
    assert kwargs["materialize_stock_pit"]["plan"] == "asof_intervals"
    assert kwargs["materialize_price"]["semantics"] == "v2"
    # argmin_pivot moved flow_*_netbuy_z_20d in the last bits: available, not selected.
    assert kwargs["materialize_flow"]["pivot_plan"] == "window_dedup"
    assert kwargs["register_derived_marts"]["metric_fact_plan"] == "split_argmin"
    # The model trained on the 2026-08-23 mart: serving reads that rule set, not the current one.
    assert kwargs["register_derived_marts"]["metric_rules_version"] == "mrv1_20260818"
    assert kwargs["register_derived_marts"]["which"] == ("stock_metric_fact",)


def test_serving_feat_flow_gets_the_price_view_like_the_research_builder(
    tmp_path: Path, monkeypatch
) -> None:
    """Regression: serving once called ``materialize_flow`` without ``price_view`` and took the
    degraded path, so every ``flow_*_netbuy_to_volume_*`` was NULL at K."""
    import hashlib

    from modeler.etl.features.flow import build_flow_sql

    flow_kwargs = _recorded_build(tmp_path, monkeypatch)["materialize_flow"]
    assert flow_kwargs["price_view"] == "daily_ohlcv"
    assert {k: flow_kwargs[k] for k in ("price_view", "pit_view", "quality_view")} == (
        live.SERVING_FLOW_VIEWS)
    # The same views as the research builder, so the contract SQL is the research SQL.
    views = {k: v for k, v in flow_kwargs.items() if k.endswith("_view")}
    text = build_flow_sql(**views)
    # sha256 of the research builder's SQL (pinned the same way in tests/unit/test_mart_plans.py).
    assert hashlib.sha256(text.encode()).hexdigest() == (
        "1bb97c537cd0656bf49df38e52c9a444bf7484aad68fc1ec6999f0e6b43aad4b")
    # The serving plan is the window plan, so its text is the research text. The argmin plan
    # (not selected) would change the text of the build, never the cache contract.
    assert build_flow_sql(**views, pivot_plan=flow_kwargs["pivot_plan"]) == text
    assert build_flow_sql(**views, pivot_plan="argmin_pivot") != text


def test_serving_flow_views_match_the_research_call_in_compute_all() -> None:
    """Read the research builder's ``materialize_flow`` call, so the two cannot drift apart."""
    import ast
    import inspect

    from modeler.etl import compute_all
    from modeler.etl.quality import QUALITY_TABLE
    from modeler.etl.stock_pit import PIT_TABLE

    tree = ast.parse(inspect.getsource(compute_all._build_features))
    constants = {"PIT_TABLE": PIT_TABLE, "QUALITY_TABLE": QUALITY_TABLE}
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "materialize_flow"
    ]
    assert len(calls) == 1
    research = {}
    for keyword in calls[0].keywords:
        if keyword.arg in live.SERVING_FLOW_VIEWS:
            node = keyword.value
            research[keyword.arg] = (
                node.value if isinstance(node, ast.Constant) else constants[node.id])
    assert research == live.SERVING_FLOW_VIEWS


def test_marker_records_the_derived_metric_fact_plan(tmp_path: Path) -> None:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    for name in live.INTERMEDIATE_VERSIONED_MARTS:
        directory = config.feature_mart_root / name
        directory.mkdir(parents=True)
        (directory / "_cache_metadata.json").write_text(json.dumps({"sql_hash": "h"}))
    assert live._intermediate_versions(config)["stock_metric_fact"] == {
        "semantics_version": "v1", "plan": "single", "plan_hash": None}
    sidecar = config.derived_mart_root / "stock_metric_fact"
    sidecar.mkdir(parents=True)
    record = {"semantics_version": "v1", "plan": "split_argmin", "plan_hash": "abc"}
    (sidecar / "_plan.json").write_text(json.dumps(record))
    assert live._intermediate_versions(config)["stock_metric_fact"] == record


def test_marker_records_the_metric_rules_version_and_hash(tmp_path: Path) -> None:
    from modeler.etl.marts.metrics_normalize import PLAN_SPLIT_ARGMIN, stock_metric_fact_plan_record

    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    for name in live.INTERMEDIATE_VERSIONED_MARTS:
        directory = config.feature_mart_root / name
        directory.mkdir(parents=True)
        (directory / "_cache_metadata.json").write_text(json.dumps({"sql_hash": "h"}))
    record = stock_metric_fact_plan_record(
        live.SERVING_METRIC_FACT_PLAN, rules_version=live.SERVING_METRIC_RULES_VERSION)
    assert live.SERVING_METRIC_RULES_VERSION == "mrv1_20260818"
    assert live.SERVING_METRIC_FACT_PLAN == PLAN_SPLIT_ARGMIN
    assert record["rules_version"] == "mrv1_20260818" and len(record["rules_hash"]) == 64
    sidecar = config.derived_mart_root / "stock_metric_fact"
    sidecar.mkdir(parents=True)
    (sidecar / "_plan.json").write_text(json.dumps(record))
    assert live._intermediate_versions(config)["stock_metric_fact"] == record


# --------------------------------------------------------------------------
# A0: the serving build profile
# --------------------------------------------------------------------------


def _serving_build(tmp_path: Path, monkeypatch, **extra) -> tuple[dict, dict, Path]:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    _raw_marker(config)
    calls: list[str] = []
    _stub_build(monkeypatch, calls, {})
    kwargs: dict = {}
    for attr in ("materialize_fin_scan_daily", "materialize_stock_metric_vintage_fact",
                 "materialize_fin_quarterly_metric_vintage"):
        monkeypatch.setattr(
            live, attr, lambda *_args, _name=attr, **kw: kwargs.setdefault(_name, kw))
    body = live.build_live_marts(
        **_BUILD_ARGS, stock_data_root=tmp_path, profile="serving", **extra)
    return body, kwargs, config.feature_mart_root / "_manifests"


def test_the_default_profile_is_full() -> None:
    import inspect

    assert live.DEFAULT_BUILD_PROFILE == "full" and live.BUILD_PROFILES == ("full", "serving")
    assert inspect.signature(live.build_live_marts).parameters["profile"].default == "full"
    parser_default = live.argparse.ArgumentParser  # the CLI shares DEFAULT_BUILD_PROFILE
    assert parser_default


def test_an_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown KR build profile"):
        live.build_live_marts(**_BUILD_ARGS, profile="lite")


def test_serving_profile_skips_the_vintage_marts_and_projects_fin_scan(
    tmp_path: Path, monkeypatch
) -> None:
    body, kwargs, manifests = _serving_build(tmp_path, monkeypatch)
    assert "materialize_stock_metric_vintage_fact" not in kwargs
    assert "materialize_fin_quarterly_metric_vintage" not in kwargs
    # Derived from the requested features, not typed in.
    assert kwargs["materialize_fin_scan_daily"]["columns"] == ("fin_log_mcap",)
    assert "join_plan" not in kwargs["materialize_fin_scan_daily"]
    profile = json.loads((manifests / "build_profile.json").read_text(encoding="utf-8"))
    names = [step["name"] for step in profile["steps"]]
    assert names == [
        name for name in _STEP_ORDER
        if name not in ("calendar", "stock_metric_vintage_fact", "fin_quarterly_metric_vintage")]
    assert profile["profile"] == "serving"
    marker = json.loads((manifests / "_SUCCESS.json").read_text(encoding="utf-8"))
    assert marker["profile"] == body["profile"] == "serving"
    assert [row["view"] for row in marker["marts"]] == list(live.SERVING_MARTS)
    # No calendar was needed, so none is recorded; neither is a beyond-calendar count.
    assert marker["holiday_calendar"] is None and marker["trading_calendar"] is None
    assert marker["beyond_calendar"] is None
    assert set(marker["mart_versions"]) == set(live.SERVING_MARTS)  # intermediate stubbed out


def test_serving_profile_marker_describes_only_the_marts_it_built(tmp_path: Path) -> None:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    # No vintage mart exists on disk: the serving profile must not ask for them.
    assert set(live._intermediate_versions(config, "serving")) == {"stock_metric_fact"}
    with pytest.raises(ValueError, match="no SQL cache contract"):
        live._intermediate_versions(config, "full")


def test_serving_profile_marker_names_the_projected_columns(tmp_path: Path, monkeypatch) -> None:
    config = LakeConfig(DataRoot(tmp_path / "kr"), "2026-09-30", REMOTE_SOURCE)
    directory = config.feature_mart_root / "feat_fin_scan_daily"
    directory.mkdir(parents=True)
    metadata = {"sql_hash": "h", "plan": "projection", "semantics_version": "v1",
                "plan_hash": "p", "projected_columns": ["fin_log_mcap"]}
    (directory / "_cache_metadata.json").write_text(json.dumps(metadata))
    (directory / "part.parquet").write_bytes(b"")

    class _Connection:
        def execute(self, sql):
            class _R:
                def fetchone(self_inner):
                    return (3, date(2026, 9, 30))
            return _R()

    record = live._mart_record(_Connection(), config, "feat_fin_scan_daily", "2026-09-30")
    assert record["projected_columns"] == ["fin_log_mcap"] and record["plan"] == "projection"
    other = live._version_entry({"sql_hash": "h"})
    assert "projected_columns" not in other


def test_serving_profile_fails_loudly_for_a_requested_column_it_cannot_project(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        live, "requested_mart_columns",
        lambda: {"feat_fin_scan_daily": ["fin_log_mcap", "fin_value_z"]})
    with pytest.raises(ValueError, match=r"does not support \['fin_value_z'\]"):
        _serving_build(tmp_path, monkeypatch)
    # The full profile never projects, so the same feature list builds there.
    config = LakeConfig(DataRoot(tmp_path / "full" / "kr"), "2026-09-30", REMOTE_SOURCE)
    _raw_marker(config)
    _stub_build(monkeypatch, [], {})
    live.build_live_marts(**_BUILD_ARGS, stock_data_root=tmp_path / "full", profile="full")


def test_the_requested_scan_columns_come_from_the_model_features() -> None:
    assert live._scan_projection_columns() == ("fin_log_mcap",)
    assert live.requested_mart_columns()["feat_fin_scan_daily"] == ["fin_log_mcap"]
