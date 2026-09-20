"""``modeler.us.dataset`` 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from modeler.etl.config import DataRoot
from modeler.us.dataset import content_hash, write_dataset

# --- content_hash ---------------------------------------------------------------


def test_content_hash_is_stable_across_row_order(tmp_path: Path) -> None:
    df1 = pl.DataFrame({"symbol": ["AAPL", "MSFT"], "value": [1.0, 2.0]})
    df2 = pl.DataFrame({"symbol": ["MSFT", "AAPL"], "value": [2.0, 1.0]})

    assert content_hash(df1) == content_hash(df2)


def test_content_hash_changes_when_a_value_changes() -> None:
    df1 = pl.DataFrame({"symbol": ["AAPL"], "value": [1.0]})
    df2 = pl.DataFrame({"symbol": ["AAPL"], "value": [1.5]})

    assert content_hash(df1) != content_hash(df2)


def test_content_hash_reproduces_after_write_and_reread(tmp_path: Path) -> None:
    """parquet에 쓰고 다시 읽어도(바이트가 달라져도) 내용 해시는 같다."""
    df = pl.DataFrame({"symbol": ["AAPL", "MSFT", "GOOGL"], "value": [1.0, 2.0, 3.0]})
    path = tmp_path / "roundtrip.parquet"

    df.write_parquet(path)
    reread = pl.read_parquet(path)

    assert content_hash(df) == content_hash(reread)


# --- write_dataset ----------------------------------------------------------------


def test_write_dataset_writes_parquet_and_manifest(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    df = pl.DataFrame({"symbol": ["AAPL", "MSFT"], "value": [1.0, 2.0]})

    dataset_dir = write_dataset(df, root, "us_panel_v1", manifest={"note": "test"})

    assert dataset_dir == tmp_path / "datasets" / "us_panel_v1"
    assert (dataset_dir / "part.parquet").exists()
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    assert manifest["note"] == "test"
    assert manifest["row_count"] == 2
    assert manifest["content_hash"] == content_hash(df)
    assert "created_at" in manifest

    written = pl.read_parquet(dataset_dir / "part.parquet")
    assert written.sort("symbol").to_dicts() == df.sort("symbol").to_dicts()


def test_write_dataset_delete_and_rebuild_reproduces_content_hash(tmp_path: Path) -> None:
    """데이터셋을 지우고 다시 만들어도 content_hash가 같아야 한다 (M0 완료 판정)."""
    root = DataRoot(base=tmp_path)
    df = pl.DataFrame({"symbol": ["AAPL", "MSFT", "GOOGL"], "value": [1.0, 2.0, 3.0]})

    first_dir = write_dataset(df, root, "us_panel_v1", manifest={})
    first_manifest = json.loads((first_dir / "manifest.json").read_text())

    for child in first_dir.iterdir():
        child.unlink()
    first_dir.rmdir()

    second_dir = write_dataset(df, root, "us_panel_v1", manifest={})
    second_manifest = json.loads((second_dir / "manifest.json").read_text())

    assert first_manifest["content_hash"] == second_manifest["content_hash"]


def test_write_dataset_overwrites_existing_files(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    df1 = pl.DataFrame({"symbol": ["AAPL"], "value": [1.0]})
    df2 = pl.DataFrame({"symbol": ["AAPL", "MSFT"], "value": [1.0, 2.0]})

    write_dataset(df1, root, "us_panel_v1", manifest={})
    dataset_dir = write_dataset(df2, root, "us_panel_v1", manifest={})

    written = pl.read_parquet(dataset_dir / "part.parquet")
    assert written.height == 2
