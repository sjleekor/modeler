from __future__ import annotations

import json
from datetime import date, datetime, timedelta, time
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import polars as pl
import pytest

from collector.lake import DataRoot
from modeler.etl.config import LakeConfig, REMOTE_SOURCE
from modeler.models._02_updown_prob import features as fx
from modeler.serving.kr_prepare import (
    KrBriefingSpec,
    REQUIRED_PREP_MARTS,
    UNIVERSE_VIEW,
    _check_snapshot_markers,
    _check_completion_marker,
    _flow_sql,
    _query_sql,
    _write_completion_marker,
    prepare_cross_section,
)


def test_prepare_flow_timing_shifts_native_flows_once_and_uses_lag1_twins() -> None:
    spec = KrBriefingSpec()
    columns = list(spec.feature_columns(20))
    sql = _flow_sql(spec, [name for name in columns if fx.COLUMN_MART[name] == fx.GROUP_VIEW["flow"]])
    assert "LAG(flow_foreign_netbuy_sum_5d) OVER w AS flow_foreign_netbuy_sum_5d" in sql
    assert "flow_individual_netbuy_to_volume_5d_lag1 AS flow_individual_netbuy_to_volume_5d" in sql
    assert "LAG(flow_individual_netbuy_to_volume_5d_lag1)" not in sql
    assert "flow_short_balance_qty" not in sql
    assert "flow_short_balance_chg_20d" not in sql


def test_feature_query_does_not_select_labels_or_balance_features() -> None:
    sql, columns, _ = _query_sql(KrBriefingSpec(), "2026-09-28")
    assert not set(columns) & {"flow_short_balance_qty", "flow_short_balance_chg_20d"}
    assert "y_up_20d" not in sql
    assert "label_end_date" not in sql
    assert "flow_short_balance_qty" not in sql


def test_feature_query_joins_K_price_quality_at_exact_exchange_grain() -> None:
    sql, _, needed = _query_sql(KrBriefingSpec(), "2026-09-28")
    con = duckdb.connect()
    con.execute("CREATE TABLE dim_universe_daily (trade_date DATE, ticker VARCHAR, market VARCHAR, in_universe BOOLEAN)")
    con.execute("INSERT INTO dim_universe_daily VALUES (DATE '2026-09-28', '000001', 'KOSPI', TRUE)")
    con.execute("""CREATE TABLE dim_price_quality_daily (
        trade_date DATE, ticker VARCHAR, market VARCHAR, is_halted BOOLEAN,
        ca_price_jump_suspect BOOLEAN, ca_share_change_confirmed BOOLEAN,
        ca_rule_applicability_unknown BOOLEAN, simple_ret DOUBLE
    )""")
    con.execute("INSERT INTO dim_price_quality_daily VALUES (DATE '2026-09-28', '000001', 'KOSPI', FALSE, TRUE, FALSE, FALSE, 0.35)")
    con.execute("INSERT INTO dim_price_quality_daily VALUES (DATE '2026-09-28', '000001', 'KOSDAQ', TRUE, FALSE, FALSE, FALSE, 0.0)")
    for mart, mapped in needed.items():
        columns = set(mapped)
        if mart == fx.GROUP_VIEW["flow"]:
            columns.update(fx.FLOW_MART_LAG1_SOURCE.get(name, name) for name in mapped)
        extra = sorted(columns - {"trade_date", "ticker", "market"})
        con.execute(f"CREATE TABLE {mart} (trade_date DATE, ticker VARCHAR, market VARCHAR, "
                    + ", ".join(f"{name} DOUBLE" for name in extra) + ")")
        con.execute(f"INSERT INTO {mart} VALUES (DATE '2026-09-28', '000001', 'KOSPI', "
                    + ", ".join("0.1" for _ in extra) + ")")
    result = con.execute(sql)
    row = dict(zip([item[0] for item in result.description], result.fetchone(), strict=True))
    assert row["ticker"] == "000001"
    assert row["market"] == "KOSPI"
    assert row["ca_price_jump_suspect"] is True
    assert row["simple_ret"] == 0.35
    assert result.fetchone() is None
    con.close()


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_prepare_requires_completed_remote_raw_and_feature_snapshots(tmp_path: Path) -> None:
    config = LakeConfig(
        root=DataRoot(tmp_path / "lake" / "kr"),
        snapshot_date="2026-09-29",
        source=REMOTE_SOURCE,
    )
    _write_json(
        config.raw_root / "_manifests" / "_SUCCESS.json",
        {
            "route": "remote",
            "finished_at": "2026-09-29T09:15:00+09:00",
            "tables": {"daily_ohlcv": {}, "krx_security_flow_raw": {}},
        },
    )
    _write_json(
        config.feature_mart_root / "_manifests" / "_SUCCESS.json",
        {
            "schema_version": "kr-serving-feature-marts.v1",
            "status": "success",
            "snapshot_date": "2026-09-29",
            "source": REMOTE_SOURCE,
            "feature_asof_date": "2026-09-28",
            "raw_marker_sha256": __import__("hashlib").sha256(
                (config.raw_root / "_manifests" / "_SUCCESS.json").read_bytes()
            ).hexdigest(),
            "created_at": "2026-09-29T09:20:00+09:00",
            "marts": [
                {"view": view, "max_trade_date": "2026-09-28", "sql_hash": "fixture"}
                for view in REQUIRED_PREP_MARTS
            ],
        },
    )
    for view in REQUIRED_PREP_MARTS:
        _write_json(config.feature_mart_root / view / "_cache_metadata.json", {"sql_hash": "fixture"})
    hashes, times = _check_snapshot_markers(
        config,
        input_cutoff="2026-09-29T09:30:00+09:00",
        feature_asof_date="2026-09-28",
    )
    assert set(hashes) == {"raw", "feature"}
    assert times["raw_snapshot_completed_at"] == "2026-09-29T09:15:00+09:00"
    assert times["feature_marts_completed_at"] == "2026-09-29T09:20:00+09:00"
    raw_marker = config.raw_root / "_manifests" / "_SUCCESS.json"
    raw_body = json.loads(raw_marker.read_text(encoding="utf-8"))
    raw_body["finished_at"] = "2026-09-29T09:31:00+09:00"
    _write_json(raw_marker, raw_body)
    with pytest.raises(ValueError, match="after the 09:30"):
        _check_snapshot_markers(
            config,
            input_cutoff="2026-09-29T09:30:00+09:00",
            feature_asof_date="2026-09-28",
        )
    raw_body["finished_at"] = "2026-09-29T09:15:00+09:00"
    _write_json(raw_marker, raw_body)
    raw_body["note"] = "raw marker replaced after mart completion"
    _write_json(raw_marker, raw_body)
    with pytest.raises(ValueError, match="does not match"):
        _check_snapshot_markers(
            config, input_cutoff="2026-09-29T09:30:00+09:00",
            feature_asof_date="2026-09-28",
        )
    raw_body.pop("note")
    _write_json(raw_marker, raw_body)
    feature_marker = config.feature_mart_root / "_manifests" / "_SUCCESS.json"
    original_feature = json.loads(feature_marker.read_text(encoding="utf-8"))
    duplicate_feature = json.loads(feature_marker.read_text(encoding="utf-8"))
    duplicate_feature["marts"].append(dict(duplicate_feature["marts"][0]))
    _write_json(feature_marker, duplicate_feature)
    with pytest.raises(ValueError, match="incomplete mart inventory"):
        _check_snapshot_markers(
            config, input_cutoff="2026-09-29T09:30:00+09:00",
            feature_asof_date="2026-09-28",
        )
    wrong_contract = json.loads(feature_marker.read_text(encoding="utf-8"))
    wrong_contract["marts"] = list(original_feature["marts"])
    wrong_contract["marts"][0]["sql_hash"] = "changed"
    _write_json(feature_marker, wrong_contract)
    with pytest.raises(ValueError, match="cache contract"):
        _check_snapshot_markers(
            config, input_cutoff="2026-09-29T09:30:00+09:00",
            feature_asof_date="2026-09-28",
        )
    _write_json(feature_marker, original_feature)
    for invalid_cutoff in (
        "2026-09-29T09:10:00+09:00",
        "2026-09-29T09:30:01+09:00",
        "2026-09-29T09:30:59+09:00",
    ):
        with pytest.raises(ValueError, match="fixed at 09:30"):
            _check_snapshot_markers(
                config,
                input_cutoff=invalid_cutoff,
                feature_asof_date="2026-09-28",
            )


def test_prepare_completion_marker_is_written_after_publish_and_pins_hashes(tmp_path: Path) -> None:
    published = tmp_path / "prepared"
    published.mkdir()
    feature = published / "feature_panel.parquet"
    feature.write_bytes(b"fixture feature parquet bytes")
    manifest = published / "prepare_manifest.json"
    from modeler.serving.kr_prepare import _sha256

    digest = _sha256(feature)
    manifest.write_text(
        json.dumps({"features_sha256": digest, "input_sha256": digest}), encoding="utf-8"
    )
    assert not (published / "completion.json").exists()
    created = _write_completion_marker(published)
    checked = _check_completion_marker(published)
    assert created == checked
    assert created["schema_version"] == "prepared-features-completion.v1"
    assert created["availability_evidence_type"] == "prepared_features_completion"
    assert created["features_sha256"] == digest
    assert created["native_prepare_manifest_sha256"] == _sha256(manifest)
    assert created["verified_available_by"].endswith("+00:00")

    feature.write_bytes(b"changed feature bytes")
    with pytest.raises(ValueError, match="does not match"):
        _check_completion_marker(published)


def _synthetic_native_world(tmp_path: Path, *, stock_master: bool = False) -> dict:
    """Raw prices, the seven marts and both markers of one snapshot (K = today in Seoul)."""
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    k = now.date()
    previous = k - timedelta(days=1)
    d = k + timedelta(days=1)
    k_text = k.isoformat()
    snapshot = k_text
    cutoff = datetime.combine(d, time(9, 30), tzinfo=ZoneInfo("Asia/Seoul")).isoformat()
    config = LakeConfig(DataRoot(tmp_path / "kr"), snapshot, REMOTE_SOURCE)
    raw = config.raw_root / "_manifests" / "_SUCCESS.json"
    _write_json(raw, {
        "route": "remote", "finished_at": (now - timedelta(minutes=2)).isoformat(),
        "tables": {"daily_ohlcv": {}, "krx_security_flow_raw": {}},
    })
    query, columns, needed = _query_sql(KrBriefingSpec(), k_text)
    _ = query, columns
    keys = {"trade_date": [previous, k], "ticker": ["000001"] * 2, "market": ["KOSPI"] * 2}
    raw_prices = config.raw_root / "daily_ohlcv"
    raw_prices.mkdir(parents=True)
    pl.DataFrame(keys).write_parquet(raw_prices / "part.parquet")
    marts = {"dim_universe_daily": pl.DataFrame({**keys, "in_universe": [True, True]})}
    marts["dim_price_quality_daily"] = pl.DataFrame({
        **keys, "is_halted": [False, False], "ca_price_jump_suspect": [False, True],
        "ca_share_change_confirmed": [False, False],
        "ca_rule_applicability_unknown": [False, False], "simple_ret": [0.01, 0.35],
    })
    for mart, mapped in needed.items():
        columns = set(mapped)
        if mart == fx.GROUP_VIEW["flow"]:
            columns.update(fx.FLOW_MART_LAG1_SOURCE.get(name, name) for name in mapped)
        marts[mart] = pl.DataFrame({**keys, **{name: [0.1, 0.2] for name in columns}})
    for name, frame in marts.items():
        path = config.feature_mart_root / name
        path.mkdir(parents=True)
        frame.write_parquet(path / "part.parquet")
        metadata = {"sql_hash": "fixture"}
        if name == "dim_universe_daily":
            from modeler.etl.mart import sql_contract_hash
            from modeler.etl.universe import build_universe_sql
            metadata["sql_hash"] = sql_contract_hash(build_universe_sql(KrBriefingSpec().universe))
        _write_json(path / "_cache_metadata.json", metadata)
    import hashlib

    _write_json(config.feature_mart_root / "_manifests" / "_SUCCESS.json", {
        "schema_version": "kr-serving-feature-marts.v1", "status": "success",
        "snapshot_date": snapshot, "source": REMOTE_SOURCE,
        "feature_asof_date": k_text,
        "raw_marker_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "created_at": (now - timedelta(minutes=1)).isoformat(),
        "marts": [{"view": name, "max_trade_date": k_text,
                   "sql_hash": json.loads((config.feature_mart_root / name / "_cache_metadata.json").read_text())["sql_hash"]}
                  for name in marts],
    })
    if stock_master:
        names = config.raw_root / "stock_master"
        names.mkdir(parents=True)
        pl.DataFrame({
            "ticker": ["000001", "000001", "999999"], "market": ["KOSPI", "KOSDAQ", "KOSPI"],
            "name": ["Alpha Corp", "Other Market Alpha", "Not In Panel"],
        }).write_parquet(names / "part.parquet")
    return {"config": config, "k_text": k_text, "snapshot": snapshot, "cutoff": cutoff,
            "needed": needed, "root": tmp_path}


def test_native_prepare_from_synthetic_label_free_marts_seals_provenance(tmp_path: Path) -> None:
    world = _synthetic_native_world(tmp_path)
    k_text, needed = world["k_text"], world["needed"]
    output = tmp_path / "prepared" / k_text
    manifest = prepare_cross_section(
        snapshot_date=world["snapshot"], feature_asof_date=k_text,
        input_cutoff=world["cutoff"], output_dir=output, stock_data_root=tmp_path,
    )
    assert manifest["quality"]["eligible_rows"] == 1
    assert manifest["required_marts"] == sorted(needed.keys() | {"dim_price_quality_daily"})
    row = pl.read_parquet(output / "feature_panel.parquet").row(0, named=True)
    assert row["ca_price_jump_suspect"] is True
    assert row["simple_ret"] == 0.35
    assert "label_scan" not in manifest["required_marts"]
    assert _check_completion_marker(output)["features_sha256"] == manifest["features_sha256"]


def test_marker_tolerates_version_and_calendar_fields(tmp_path: Path) -> None:
    """E1 adds ``mart_versions`` / ``trading_calendar`` to the marker and plan keys to the cache
    metadata. kr_prepare checks only the keys it knows, so these must not break it, and the
    versions reach ``prepare_manifest.json`` through ``mart_contracts`` (the cache metadata)."""
    config = LakeConfig(
        root=DataRoot(tmp_path / "lake" / "kr"), snapshot_date="2026-09-29", source=REMOTE_SOURCE)
    _write_json(
        config.raw_root / "_manifests" / "_SUCCESS.json",
        {"route": "remote", "finished_at": "2026-09-29T09:15:00+09:00",
         "tables": {"daily_ohlcv": {}, "krx_security_flow_raw": {}}},
    )
    versions = {"semantics_version": "v2", "plan": "staged", "plan_hash": "abc"}
    _write_json(
        config.feature_mart_root / "_manifests" / "_SUCCESS.json",
        {
            "schema_version": "kr-serving-feature-marts.v1", "status": "success",
            "snapshot_date": "2026-09-29", "source": REMOTE_SOURCE,
            "feature_asof_date": "2026-09-28",
            "raw_marker_sha256": __import__("hashlib").sha256(
                (config.raw_root / "_manifests" / "_SUCCESS.json").read_bytes()).hexdigest(),
            "created_at": "2026-09-29T09:20:00+09:00",
            "mart_versions": {view: versions for view in REQUIRED_PREP_MARTS},
            "trading_calendar": {"last_session": "2026-10-15", "sessions_sha256": "x"},
            "marts": [
                {"view": view, "max_trade_date": "2026-09-28", "sql_hash": "fixture", **versions}
                for view in REQUIRED_PREP_MARTS
            ],
        },
    )
    for view in REQUIRED_PREP_MARTS:
        _write_json(config.feature_mart_root / view / "_cache_metadata.json",
                    {"sql_hash": "fixture", **versions})
    hashes, _ = _check_snapshot_markers(
        config, input_cutoff="2026-09-29T09:30:00+09:00", feature_asof_date="2026-09-28")
    assert set(hashes) == {"raw", "feature"}


def test_restricted_flow_source_gives_the_same_k_rows_as_lagging_the_whole_mart() -> None:
    """A5: each ticker's predecessor is its own last row before K, which is not the previous
    calendar session for a halted ticker. Keeping exactly that row must not change any K row."""
    from modeler.serving.kr_prepare import _flow_rows_cte

    spec = KrBriefingSpec()
    columns = [
        name for name in spec.feature_columns(20)
        if fx.COLUMN_MART[name] == fx.GROUP_VIEW["flow"]
    ]
    stored = sorted({fx.FLOW_MART_LAG1_SOURCE.get(name, name) for name in columns} | set(columns))
    k = "2026-09-29"
    days = ["2026-09-23", "2026-09-24", "2026-09-25", "2026-09-28", k]
    # ticker -> sessions it has a row on
    sessions = {
        ("A", "KOSPI"): days,  # normal
        ("B", "KOSPI"): days[:2] + [k],  # halted for two sessions before K
        ("C", "KOSPI"): days[:3],  # no K row (delisted / not yet loaded)
        ("D", "KOSPI"): [k],  # first listed on K: no predecessor at all
        ("A", "KOSDAQ"): [days[0], k],  # same ticker code on the other market
    }
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE feat_flow (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        + ", ".join(f"{name} DOUBLE" for name in stored) + ")"
    )
    value = 0.0
    for (ticker, market), rows in sessions.items():
        for day in rows:
            value += 1.0
            con.execute(
                "INSERT INTO feat_flow VALUES (?, ?, ?, " + ", ".join(
                    "NULL" if i % 7 == 3 else str(value + i) for i, _ in enumerate(stored)) + ")",
                [day, ticker, market],
            )
    full = f"WITH flow_src AS ({_flow_sql(spec, columns)}) SELECT * FROM flow_src"
    restricted = (
        f"WITH {_flow_rows_cte('feat_flow', k)}, "
        f"flow_src AS ({_flow_sql(spec, columns, source='flow_rows')}) SELECT * FROM flow_src"
    )
    left = con.execute(f"{full} WHERE trade_date = DATE '{k}' ORDER BY ticker, market").fetchall()
    right = con.execute(
        f"{restricted} WHERE trade_date = DATE '{k}' ORDER BY ticker, market").fetchall()
    assert left == right and len(left) == 4
    # B's predecessor is two sessions back (09-24): its shifted values are present, not NULL.
    b_row = con.execute(
        f"{restricted} WHERE trade_date = DATE '{k}' AND ticker = 'B'").fetchone()
    assert any(item is not None for item in b_row[3:])
    # D has no predecessor, so only the two lag1-twin passthrough columns can be non-NULL.
    d_row = con.execute(
        f"{restricted} WHERE trade_date = DATE '{k}' AND ticker = 'D'").fetchone()
    assert sum(item is not None for item in d_row[3:]) <= 2
    # The restricted source never carries more than K plus one predecessor per ticker.
    kept = con.execute(
        f"WITH {_flow_rows_cte('feat_flow', k)} SELECT count(*) FROM flow_rows").fetchone()[0]
    assert kept == 4 + 4


def test_query_sql_restricts_the_flow_source_by_default_and_can_opt_out() -> None:
    restricted, _, _ = _query_sql(KrBriefingSpec(), "2026-09-28")
    original, _, _ = _query_sql(KrBriefingSpec(), "2026-09-28", restrict_flow=False)
    assert "flow_rows AS" in restricted and "FROM flow_rows" in restricted
    assert "flow_rows" not in original and f"FROM {fx.GROUP_VIEW['flow']} WINDOW" in original


def _profile_marker_config(tmp_path: Path, *, profile: str | None, scan_meta: dict,
                           marker_projected: list | None = None) -> LakeConfig:
    config = LakeConfig(
        root=DataRoot(tmp_path / "lake" / "kr"), snapshot_date="2026-09-29", source=REMOTE_SOURCE)
    _write_json(
        config.raw_root / "_manifests" / "_SUCCESS.json",
        {"route": "remote", "finished_at": "2026-09-29T09:15:00+09:00",
         "tables": {"daily_ohlcv": {}, "krx_security_flow_raw": {}}},
    )
    marts = []
    for view in REQUIRED_PREP_MARTS:
        entry = {"view": view, "max_trade_date": "2026-09-28", "sql_hash": "fixture"}
        meta = {"sql_hash": "fixture"}
        if view == "feat_fin_scan_daily":
            meta = {"sql_hash": "fixture", **scan_meta}
            if marker_projected is not None:
                entry["projected_columns"] = marker_projected
        marts.append(entry)
        _write_json(config.feature_mart_root / view / "_cache_metadata.json", meta)
    body = {
        "schema_version": "kr-serving-feature-marts.v1", "status": "success",
        "snapshot_date": "2026-09-29", "source": REMOTE_SOURCE, "feature_asof_date": "2026-09-28",
        "raw_marker_sha256": __import__("hashlib").sha256(
            (config.raw_root / "_manifests" / "_SUCCESS.json").read_bytes()).hexdigest(),
        "created_at": "2026-09-29T09:20:00+09:00", "marts": marts,
    }
    if profile is not None:
        body["profile"] = profile
    _write_json(config.feature_mart_root / "_manifests" / "_SUCCESS.json", body)
    return config


def _check(config: LakeConfig):
    return _check_snapshot_markers(
        config, input_cutoff="2026-09-29T09:30:00+09:00", feature_asof_date="2026-09-28")


_PROJECTION = {"plan": "projection", "projected_columns": ["fin_log_mcap"]}


def test_a_marker_without_profile_is_full_and_a_plain_scan_mart_passes(tmp_path: Path) -> None:
    assert _check(_profile_marker_config(tmp_path, profile=None, scan_meta={}))
    assert _check(_profile_marker_config(tmp_path / "b", profile="full", scan_meta={}))


def test_serving_profile_needs_a_projection_that_covers_the_requested_columns(
    tmp_path: Path,
) -> None:
    ok = _profile_marker_config(
        tmp_path, profile="serving", scan_meta=_PROJECTION, marker_projected=["fin_log_mcap"])
    assert _check(ok)
    from modeler.serving.kr_prepare import requested_mart_columns

    assert requested_mart_columns()["feat_fin_scan_daily"] == ["fin_log_mcap"]
    # Marker says serving but the mart on disk is the full one.
    with pytest.raises(ValueError, match="serving profile but .* is not a projection"):
        _check(_profile_marker_config(
            tmp_path / "a", profile="serving", scan_meta={}, marker_projected=["fin_log_mcap"]))
    # Marker says full but the mart on disk is a projection: it must not be read as the full mart.
    with pytest.raises(ValueError, match="full profile but feat_fin_scan_daily is a projection"):
        _check(_profile_marker_config(tmp_path / "b", profile="full", scan_meta=_PROJECTION))
    with pytest.raises(ValueError, match="full profile but feat_fin_scan_daily is a projection"):
        _check(_profile_marker_config(tmp_path / "c", profile=None, scan_meta=_PROJECTION))
    # A projection that lacks a column the model asks for would feed NULLs.
    with pytest.raises(ValueError, match=r"lacks requested columns: \['fin_log_mcap'\]"):
        _check(_profile_marker_config(
            tmp_path / "d", profile="serving",
            scan_meta={"plan": "projection", "projected_columns": ["fin_value_z"]},
            marker_projected=["fin_value_z"]))
    # The marker and the mart must agree on what was projected.
    with pytest.raises(ValueError, match="projected columns differ"):
        _check(_profile_marker_config(
            tmp_path / "e", profile="serving", scan_meta=_PROJECTION, marker_projected=[]))
    with pytest.raises(ValueError, match="unknown build profile"):
        _check(_profile_marker_config(tmp_path / "f", profile="lite", scan_meta={}))


def test_prepared_panel_carries_display_names_from_raw_stock_master(tmp_path: Path) -> None:
    """F3: names come from the snapshot's raw stock_master, joined at (ticker, market)."""
    world = _synthetic_native_world(tmp_path, stock_master=True)
    output = tmp_path / "prepared" / world["k_text"]
    manifest = prepare_cross_section(
        snapshot_date=world["snapshot"], feature_asof_date=world["k_text"],
        input_cutoff=world["cutoff"], output_dir=output, stock_data_root=tmp_path,
    )
    panel = pl.read_parquet(output / "feature_panel.parquet")
    assert panel.row(0, named=True)["name"] == "Alpha Corp"  # not the KOSDAQ namesake
    # Display only: not a model feature, not in the feature list the manifest pins.
    assert "name" not in manifest["model_config"]["feature_columns"]
    assert manifest["quality"]["display_columns"] == ["name"]
    assert manifest["quality"]["display_name_source"] == {
        "table": "stock_master", "snapshot_date": world["snapshot"], "rows_named": 1,
        "rows_unnamed": 0, "model_input": False}
    assert _check_completion_marker(output)["features_sha256"] == manifest["features_sha256"]


def test_prepared_panel_without_stock_master_has_null_names_and_says_so(tmp_path: Path) -> None:
    world = _synthetic_native_world(tmp_path)
    output = tmp_path / "prepared" / world["k_text"]
    manifest = prepare_cross_section(
        snapshot_date=world["snapshot"], feature_asof_date=world["k_text"],
        input_cutoff=world["cutoff"], output_dir=output, stock_data_root=tmp_path,
    )
    assert pl.read_parquet(output / "feature_panel.parquet")["name"].to_list() == [None]
    assert manifest["quality"]["display_name_source"]["rows_named"] == 0
    assert manifest["quality"]["display_name_source"]["rows_unnamed"] == 1


def _evidence(path: Path, *, reference: str, verdict: str = "fallback_K_prime") -> Path:
    _write_json(path, {"schema": "kr-reference-selection.v1", "verdict": verdict,
                       "reference_date": reference, "k": "2099-01-01", "lag_sessions": 1})
    return path


def test_native_manifest_embeds_the_reference_evidence_and_checks_its_date(tmp_path: Path) -> None:
    world = _synthetic_native_world(tmp_path)
    k_text = world["k_text"]
    evidence = _evidence(tmp_path / "evidence.json", reference=k_text)
    manifest = prepare_cross_section(
        snapshot_date=world["snapshot"], feature_asof_date=k_text, input_cutoff=world["cutoff"],
        output_dir=tmp_path / "prepared" / k_text, stock_data_root=tmp_path,
        reference_evidence=evidence)
    assert manifest["reference_selection"]["verdict"] == "fallback_K_prime"
    assert manifest["reference_selection"]["reference_date"] == k_text
    wrong = _evidence(tmp_path / "wrong.json", reference="2020-01-02")
    with pytest.raises(ValueError, match="does not name the requested feature_asof_date"):
        prepare_cross_section(
            snapshot_date=world["snapshot"], feature_asof_date=k_text, input_cutoff=world["cutoff"],
            output_dir=tmp_path / "prepared" / "other", stock_data_root=tmp_path,
            reference_evidence=wrong)
    none = _evidence(tmp_path / "none.json", reference=k_text, verdict="none")
    with pytest.raises(ValueError, match="does not name the requested feature_asof_date"):
        prepare_cross_section(
            snapshot_date=world["snapshot"], feature_asof_date=k_text, input_cutoff=world["cutoff"],
            output_dir=tmp_path / "prepared" / "third", stock_data_root=tmp_path,
            reference_evidence=none)


def test_feature_marker_whose_raw_cut_ends_elsewhere_is_rejected(tmp_path: Path) -> None:
    """A mart cut at one session must not be read as the marts of another."""
    world = _synthetic_native_world(tmp_path)
    config, k_text = world["config"], world["k_text"]
    marker = config.feature_mart_root / "_manifests" / "_SUCCESS.json"
    body = json.loads(marker.read_text())
    body["raw_cut"] = {"cut_asof": k_text, "predicates": {"daily_ohlcv": f"trade_date <= DATE '{k_text}'"}}
    _write_json(marker, body)
    cutoff = world["cutoff"]
    _check_snapshot_markers(config, input_cutoff=cutoff, feature_asof_date=k_text)  # consistent: fine
    body["raw_cut"]["cut_asof"] = "2020-01-02"
    _write_json(marker, body)
    with pytest.raises(ValueError, match="raw cut does not end at feature_asof_date"):
        _check_snapshot_markers(config, input_cutoff=cutoff, feature_asof_date=k_text)
