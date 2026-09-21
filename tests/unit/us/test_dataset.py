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


# --- 더러운 트리로는 데이터셋을 안 만든다 (2026-09-22) -------------------------


def _git(tmp_path):
    """작은 git 저장소 하나. 커밋이 하나 있고 트리는 깨끗하다."""
    import subprocess

    repo = tmp_path / "repo"
    repo.mkdir()
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", "-C", str(repo), *a], check=True, capture_output=True
    )
    run("init", "-q")
    run("config", "user.email", "t@t")
    run("config", "user.name", "t")
    (repo / "a.txt").write_text("1")
    run("add", "-A")
    run("commit", "-qm", "first")
    return repo


def test_git_commit_returns_head_when_clean(tmp_path):
    from modeler.us.dataset import git_commit

    head = git_commit(_git(tmp_path))
    assert len(head) == 40 and "-dirty" not in head


def test_git_commit_refuses_a_dirty_tree(tmp_path):
    """**manifest 의 커밋으로 다시 만들 수 없게 되는 것**을 막는다.

    `us_features_v1` 이 실제로 `d38d1d44…-dirty` 로 만들어졌고, 지금 코드로
    `sp_ttm` 을 다시 계산하면 168,161행 중 17행이 다르다 (2026-09-21).
    """
    import pytest

    from modeler.us.dataset import DirtyWorktreeError, git_commit

    repo = _git(tmp_path)
    (repo / "a.txt").write_text("2")
    with pytest.raises(DirtyWorktreeError, match="--allow-dirty"):
        git_commit(repo)


def test_git_commit_allows_dirty_when_asked(tmp_path):
    """버리는 실험용 탈출구. 대신 `-dirty` 가 manifest 에 남는다."""
    from modeler.us.dataset import git_commit

    repo = _git(tmp_path)
    (repo / "a.txt").write_text("2")
    assert git_commit(repo, allow_dirty=True).endswith("-dirty")


def test_untracked_file_also_counts_as_dirty(tmp_path):
    """새 파일을 안 세면 '피쳐 하나 더 만들고 커밋 안 함' 이 통과한다."""
    import pytest

    from modeler.us.dataset import DirtyWorktreeError, git_commit

    repo = _git(tmp_path)
    (repo / "new.py").write_text("x")
    with pytest.raises(DirtyWorktreeError):
        git_commit(repo)


def test_all_three_builders_share_one_git_commit():
    """세 빌더가 각자 복사본을 두지 않는다 — 하나만 고치면 벌어진다."""
    from modeler.us import build_features, build_labels, build_panel
    from modeler.us.dataset import git_commit

    for mod in (build_panel, build_features, build_labels):
        assert mod.git_commit is git_commit
        assert not hasattr(mod, "_git_commit")
