"""Topological materialization of the Phase A0 shared inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

from modeler.analysis.horizon_scan_config import CONFIG_PATH, load_config
from modeler.etl.config import REMOTE_SOURCE, DataRoot, EngineOptions
from modeler.etl.features.flow import materialize_flow
from modeler.etl.features.price import materialize_price
from modeler.etl.labels import materialize_label_scan
from modeler.etl.lake import connect, register_views
from modeler.etl.mart import mart_root
from modeler.etl.quality import (
    QUALITY_TABLE,
    materialize_price_quality,
    write_publication_lag_evidence,
)
from modeler.etl.snapshot import resolve_config
from modeler.etl.stock_pit import PIT_TABLE, materialize_stock_pit
from modeler.etl.universe import materialize_named_universes

REQUIRED_RAW_INPUTS = (
    "daily_ohlcv",
    "krx_security_flow_raw",
    "dart_share_count_raw",
    "stock_master",
    "stock_master_snapshot",
    "stock_master_snapshot_items",
)


def _json_default(value: Any) -> str:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _schema_hash(con: duckdb.DuckDBPyConnection, view: str) -> str:
    rows = con.execute(f"DESCRIBE {view}").fetchall()
    encoded = json.dumps(rows, default=_json_default, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _mart_stats(con: duckdb.DuckDBPyConnection, view: str) -> dict[str, Any]:
    count, min_date, max_date = con.execute(
        f"SELECT count(*), min(trade_date), max(trade_date) FROM {view}"
    ).fetchone()
    return {
        "view": view,
        "row_count": count,
        "min_trade_date": min_date,
        "max_trade_date": max_date,
        "schema_hash": _schema_hash(con, view),
    }


def build_a0_inputs(
    *,
    root: DataRoot | None = None,
    source: str = REMOTE_SOURCE,
    engine: EngineOptions | None = None,
    snapshot_date: str | None = None,
    force: bool = False,
    output_root: Path | None = None,
    research_output_root: Path | None = None,
    config_path: Path | str = CONFIG_PATH,
) -> Path:
    """Build A0 marts and atomically write the success manifest last."""
    # Deferred import: horizon_scan_readiness imports REQUIRED_RAW_INPUTS from
    # this module, so a top-level import here would be circular.
    from modeler.analysis.horizon_scan_readiness import (
        build_readiness_rows,
        write_readiness_artifacts,
    )

    root = root or DataRoot.resolve(market="kr")
    horizon_config = load_config(config_path)
    pinned, resolution = resolve_config(
        root,
        source,
        required_inputs=REQUIRED_RAW_INPUTS,
        snapshot_date=snapshot_date,
        engine=engine,
    )
    pinned = replace(pinned, analysis_config_hash=horizon_config.config_hash)
    con = connect(pinned)
    register_views(con, pinned, tables=list(REQUIRED_RAW_INPUTS))

    research_root = research_output_root or (pinned.root.output / "scan" / "horizon_scan")
    # The evidence artifact is unresolved until the source-specific diagnostic
    # is completed. That state intentionally blocks only balance families.
    evidence_path = research_root / "short_balance_publication_lag.json"
    write_publication_lag_evidence(evidence_path, con)
    publication_lag_verified = (
        json.loads(evidence_path.read_text(encoding="utf-8")).get("status") == "verified"
    )

    materialize_stock_pit(con, pinned, force=force)
    quality = horizon_config.raw["quality"]
    materialize_price_quality(
        con,
        pinned,
        pit_view=PIT_TABLE,
        price_limit_multiplier=float(quality["price_limit_multiplier"]),
        share_change_threshold=float(quality["ca_abs_share_change"]),
        share_ratio_low=float(quality["ca_ratio_product_range"][0]),
        share_ratio_high=float(quality["ca_ratio_product_range"][1]),
        short_balance_lag_sessions=int(quality["short_balance_lag_sessions"]),
        force=force,
    )
    materialize_named_universes(con, pinned, force=force)
    materialize_price(con, pinned, quality_view=QUALITY_TABLE, force=force)
    materialize_flow(
        con,
        pinned,
        price_view="daily_ohlcv",
        pit_view=PIT_TABLE,
        quality_view=QUALITY_TABLE,
        force=force,
    )
    materialize_label_scan(
        con,
        pinned,
        quality_view=QUALITY_TABLE,
        holdout_start="2025-08-01",
        force=force,
    )

    marts = [
        PIT_TABLE,
        QUALITY_TABLE,
        "dim_universe_broad_daily",
        "dim_universe_tradable_daily",
        "feat_price",
        "feat_flow",
        "label_scan",
    ]
    columns_by_mart = {
        view: {row[0] for row in con.execute(f"DESCRIBE {view}").fetchall()} for view in marts
    }
    readiness_rows = build_readiness_rows(
        horizon_config,
        columns_by_mart=columns_by_mart,
        publication_lag_verified=publication_lag_verified,
    )
    readiness_parquet, readiness_markdown = write_readiness_artifacts(
        horizon_config,
        research_root,
        columns_by_mart=columns_by_mart,
        publication_lag_verified=publication_lag_verified,
    )

    # Only the 12 global-BH families gate smoke_only; short-selling families
    # blocked on publication-lag evidence do not (A0 plan §1.2).
    primary_rows = [r for r in readiness_rows if r["phase"] == "A" and r["fdr_include"]]
    primary_not_ready = [r["family"] for r in primary_rows if r["status"] != "ready"]
    ca_mask_applied = "ca_mask" in columns_by_mart.get(QUALITY_TABLE, set())
    is_sj2_remote = pinned.source == REMOTE_SOURCE

    smoke_only_reasons = []
    if not ca_mask_applied:
        smoke_only_reasons.append("ca_mask_missing")
    if not is_sj2_remote:
        smoke_only_reasons.append("source_not_sj2_remote")
    if not resolution.auto_selected:
        smoke_only_reasons.append("snapshot_manually_overridden")
    if primary_not_ready:
        smoke_only_reasons.append("primary_family_not_ready:" + ",".join(primary_not_ready))
    smoke_only = bool(smoke_only_reasons)

    payload = {
        "schema_version": 2,
        "status": "success",
        "snapshot_date": pinned.snapshot_date,
        "source": pinned.source,
        "auto_selected": resolution.auto_selected,
        "official": resolution.auto_selected,
        "smoke_only": smoke_only,
        "smoke_only_reasons": smoke_only_reasons,
        "raw_marker": str(resolution.raw_marker),
        "config_hash": horizon_config.config_hash,
        "publication_lag_evidence": str(evidence_path),
        "publication_lag_verified": publication_lag_verified,
        "readiness_matrix_parquet": str(readiness_parquet),
        "readiness_matrix_markdown": str(readiness_markdown),
        "primary_family_count": len(primary_rows),
        "primary_family_not_ready": primary_not_ready,
        "marts": [_mart_stats(con, view) for view in marts],
        "created_at": datetime.now().astimezone().isoformat(),
    }
    target_root = output_root or mart_root(pinned)
    manifest_dir = target_root / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    temp = manifest_dir / "_SUCCESS.json.tmp"
    target = manifest_dir / "_SUCCESS.json"
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    temp.replace(target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--source", default=REMOTE_SOURCE)
    parser.add_argument("--data-lake-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    root = DataRoot(args.data_lake_root) if args.data_lake_root else DataRoot.resolve(market="kr")
    print(
        build_a0_inputs(
            root=root,
            source=args.source,
            engine=EngineOptions(threads=4, memory_limit="4GB"),
            snapshot_date=args.snapshot_date,
            force=args.force,
            config_path=args.config,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
