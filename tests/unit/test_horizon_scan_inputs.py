from __future__ import annotations

import json
from pathlib import Path

import duckdb

from modeler.etl.config import DataRoot
from modeler.etl.horizon_scan_inputs import REQUIRED_RAW_INPUTS, build_a0_inputs


def _write_table(root: Path, table: str, query: str) -> None:
    path = root / "raw" / "raw_postgres" / "snapshot_date=2024-03-01" / "source=sj2_remote" / table
    path.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"COPY ({query}) TO '{(path / 'part.parquet').as_posix()}' (FORMAT PARQUET)")
    con.close()


def test_a0_builder_writes_success_manifest_after_all_marts(tmp_path: Path) -> None:
    raw = (
        tmp_path
        / "raw"
        / "raw_postgres"
        / "snapshot_date=2024-03-01"
        / "source=sj2_remote"
        / "_manifests"
    )
    raw.mkdir(parents=True)
    (raw / "_SUCCESS.json").write_text(
        json.dumps({"tables": {table: {} for table in REQUIRED_RAW_INPUTS}}), encoding="utf-8"
    )
    prices = ",".join(
        f"(DATE '2024-02-{i:02d}', 'A', 'KOSPI', 100,100,100,100,{1000+i})"
        for i in range(1, 21)
    )
    _write_table(
        tmp_path,
        "daily_ohlcv",
        "SELECT * FROM (VALUES "
        + prices
        + ") t(trade_date,ticker,market,open,high,low,close,volume)",
    )
    flows = []
    for i in range(1, 21):
        d = f"2024-02-{i:02d}"
        for code, value in (
            ("foreign_net_buy_volume", 10.0),
            ("institution_net_buy_volume", 5.0),
            ("individual_net_buy_volume", -2.0),
            ("foreign_holding_shares", 100.0),
            ("short_selling_volume", 2.0),
            ("short_selling_value", 200.0),
            ("short_selling_balance_quantity", 20.0),
        ):
            flows.append(f"(DATE '{d}','A','KOSPI','{code}',{value},'KRX')")
    _write_table(
        tmp_path,
        "krx_security_flow_raw",
        "SELECT * FROM (VALUES "
        + ",".join(flows)
        + ") t(trade_date,ticker,market,metric_code,value,source)",
    )
    _write_table(
        tmp_path,
        "dart_share_count_raw",
        "SELECT * FROM (VALUES "
        "('A',2023,'11011','20240102000001','합계',10000,1000,9000,DATE '2023-12-31')"
        ") t(ticker,bsns_year,reprt_code,rcept_no,se,istc_totqy,tesstk_co,distb_stock_co,stlm_dt)",
    )
    for table in ("stock_master", "stock_master_snapshot", "stock_master_snapshot_items"):
        _write_table(tmp_path, table, "SELECT 'A' AS ticker")

    root = DataRoot(tmp_path)
    manifest = build_a0_inputs(
        root=root,
        source="sj2_remote",
        output_root=tmp_path / "manifest",
        research_output_root=tmp_path / "research_output",
        force=True,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["snapshot_date"] == "2024-03-01"
    assert {item["view"] for item in payload["marts"]} == {
        "dim_stock_pit_daily",
        "dim_price_quality_daily",
        "dim_universe_broad_daily",
        "dim_universe_tradable_daily",
        "feat_price",
        "feat_flow",
        "label_scan",
    }
