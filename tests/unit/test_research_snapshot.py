from __future__ import annotations

import json
from pathlib import Path

import pytest
from modeler.etl.config import DataRoot
from modeler.etl.snapshot import resolve_config, resolve_snapshot


def _complete(base: Path, snapshot: str, source: str, tables: list[str]) -> None:
    path = (
        base / "raw" / "raw_postgres" / f"snapshot_date={snapshot}" / f"source={source}"
        / "_manifests"
    )
    path.mkdir(parents=True)
    (path / "_SUCCESS.json").write_text(json.dumps({"tables": {t: {} for t in tables}}))


def _derived(base: Path, snapshot: str, source: str, table: str) -> None:
    path = base / "derived" / "metric" / f"snapshot_date={snapshot}" / f"source={source}" / table
    path.mkdir(parents=True)
    (path / "part.parquet").write_bytes(b"fixture")


def test_a0_selects_latest_complete_raw_snapshot(tmp_path: Path) -> None:
    tables = ["daily_ohlcv", "krx_security_flow_raw"]
    _complete(tmp_path, "2026-07-30", "sj2_remote", tables)
    _complete(tmp_path, "2026-07-31", "sj2_remote", tables[:-1])
    root = DataRoot(tmp_path)

    result = resolve_snapshot(root, "sj2_remote", required_inputs=tables)

    assert result.snapshot_date == "2026-07-30"
    assert result.auto_selected is True


def test_phase_b_requires_derived_intersection(tmp_path: Path) -> None:
    tables = ["daily_ohlcv"]
    for snapshot in ("2026-07-30", "2026-07-31"):
        _complete(tmp_path, snapshot, "sj2_remote", tables)
    _derived(tmp_path, "2026-07-30", "sj2_remote", "stock_metric_fact")
    _derived(tmp_path, "2026-07-30", "sj2_remote", "common_feature_daily_fact")
    root = DataRoot(tmp_path)

    pinned, result = resolve_config(root, "sj2_remote", required_inputs=tables, require_derived=True)

    assert pinned.snapshot_date == "2026-07-30"
    assert set(result.derived_tables) == {"stock_metric_fact", "common_feature_daily_fact"}


def test_explicit_incomplete_snapshot_fails(tmp_path: Path) -> None:
    _complete(tmp_path, "2026-07-30", "sj2_remote", ["daily_ohlcv"])
    root = DataRoot(tmp_path)
    with pytest.raises(FileNotFoundError, match="no complete"):
        resolve_snapshot(
            root, "sj2_remote", required_inputs=["krx_security_flow_raw"], snapshot_date="2026-07-30"
        )
