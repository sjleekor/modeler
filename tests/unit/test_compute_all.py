"""Unit tests for the compute_all CLI arg parsing -> run() wiring (P1, P4)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from modeler.etl import compute_all
from modeler.etl.config import CONFIG_TABLES, RAW_TABLES


def test_main_passes_source_through_to_run(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(compute_all, "run", fake_run)

    exit_code = compute_all.main(["--source", "sj2_remote", "--snapshot-date", "2026-07-30"])

    assert exit_code == 0
    assert captured["source"] == "sj2_remote"
    assert captured["snapshot_date"] == "2026-07-30"


def test_main_defaults_source_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(compute_all, "run", fake_run)

    compute_all.main([])

    assert captured["source"] is None


def _make_fake_lake_config(tmp_path: Path, captured_cfg: dict[str, object]):
    class _FakeLakeConfig:
        def __init__(self, **kwargs: object) -> None:
            captured_cfg.update(kwargs)
            self.snapshot_date = kwargs.get("snapshot_date", "2026-06-19")
            self.source = kwargs.get("source", "local_mydb")
            self.raw_root = tmp_path / "raw_postgres" / self.source

    return _FakeLakeConfig


def test_run_resolves_none_source_to_local_mydb(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_cfg: dict[str, object] = {}
    monkeypatch.setattr(compute_all, "LakeConfig", _make_fake_lake_config(tmp_path, captured_cfg))

    exit_code = compute_all.run(snapshot_date="2026-07-30", source=None, end=date(2026, 7, 30))

    # no _SUCCESS.json under tmp_path -> gate fails -> exit 1, but source resolution
    # (the thing this test targets) still happened correctly before the gate ran.
    assert exit_code == 1
    assert captured_cfg["source"] == "local_mydb"


def _write_success_marker(raw_root: Path, tables: set[str]) -> None:
    manifests_dir = raw_root / "_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    payload = {"route": "local", "tables": {t: {} for t in tables}}
    (manifests_dir / "_SUCCESS.json").write_text(json.dumps(payload), encoding="utf-8")


def test_check_lake_complete_missing_marker(tmp_path: Path) -> None:
    from modeler.etl.config import DataRoot, LakeConfig

    cfg = LakeConfig(root=DataRoot(tmp_path), snapshot_date="2026-07-30")

    error = compute_all._check_lake_complete(cfg, allow_incomplete_lake=False)

    assert error is not None
    assert "_SUCCESS.json" in error
    assert "missing" in error


def test_check_lake_complete_passes_with_full_table_set(tmp_path: Path) -> None:
    from modeler.etl.config import DataRoot, LakeConfig

    cfg = LakeConfig(root=DataRoot(tmp_path), snapshot_date="2026-07-30")
    _write_success_marker(cfg.raw_root, set(RAW_TABLES) | set(CONFIG_TABLES))

    assert compute_all._check_lake_complete(cfg, allow_incomplete_lake=False) is None


def test_check_lake_complete_rejects_incomplete_table_set(tmp_path: Path) -> None:
    from modeler.etl.config import DataRoot, LakeConfig

    cfg = LakeConfig(root=DataRoot(tmp_path), snapshot_date="2026-07-30")
    _write_success_marker(cfg.raw_root, {RAW_TABLES[0]})

    error = compute_all._check_lake_complete(cfg, allow_incomplete_lake=False)

    assert error is not None
    assert "table set mismatch" in error


def test_main_allow_incomplete_lake_flag_defaults_false(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_run(**kwargs: object) -> int:
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(compute_all, "run", fake_run)

    compute_all.main([])

    assert captured["allow_incomplete_lake"] is False

    compute_all.main(["--allow-incomplete-lake"])

    assert captured["allow_incomplete_lake"] is True
