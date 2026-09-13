"""Unit tests for P5 — dataset_manifest (00_shared §4)."""

from __future__ import annotations

import json
from pathlib import Path

from modeler.etl.config import DataRoot, LakeConfig
from modeler.etl.manifest import build_manifest


def _cfg(tmp: Path) -> LakeConfig:
    return LakeConfig(root=DataRoot(tmp), snapshot_date="2026-06-19", source="local_mydb")


def test_build_manifest_fields(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    man = build_manifest(
        model_id="01_20_access_return_rank",
        config=cfg,
        feature_groups=["px", "flow"],
        label_spec={"horizons": [20, 5, 60], "primary_target": "y_rank_20d"},
        universe_filter={"min_liquidity_krw": 1e8},
        period={"start": "2015-01-02", "end": "2026-06-10"},
        row_count=123,
        code_rev="deadbee",
        created_at="2026-06-20T00:00:00Z",
    )
    assert man.model_id == "01_20_access_return_rank"
    assert man.snapshot_date == "2026-06-19"
    assert man.feature_groups == ["px", "flow"]
    assert man.row_count == 123
    assert man.code_rev == "deadbee"
    assert "raw_postgres" in man.lake["raw"]
    assert "canonical" not in man.lake


def test_manifest_roundtrips_json(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    man = build_manifest(
        model_id="m",
        config=cfg,
        feature_groups=["px"],
        label_spec={"horizons": [20]},
        universe_filter={},
        period={"start": "2015-01-02", "end": "2026-06-10"},
        row_count=1,
        code_rev="abc1234",
    )
    out = tmp_path / "dataset_manifest.json"
    man.write(out)
    loaded = json.loads(out.read_text())
    assert loaded["model_id"] == "m"
    assert loaded["label_spec"]["horizons"] == [20]
    assert loaded["code_rev"] == "abc1234"
