"""``modeler.us.dataset`` 단위 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.dataset import (
    FROZEN_DATASET_NAMES,
    DatasetExistsError,
    DatasetNamingError,
    assert_dataset_absent,
    check_dataset_name,
    content_hash,
    manifest_universe_version,
    read_panel_dataset,
    write_dataset,
)

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


def test_write_dataset_refuses_an_existing_name_and_keeps_the_old_files(tmp_path: Path) -> None:
    """같은 이름이 이미 있으면 쓰기를 거부한다 — 옛 파일은 그대로다 (설계 T9)."""
    root = DataRoot(base=tmp_path)
    df1 = pl.DataFrame({"symbol": ["AAPL"], "value": [1.0]})
    df2 = pl.DataFrame({"symbol": ["AAPL", "MSFT"], "value": [1.0, 2.0]})

    dataset_dir = write_dataset(df1, root, "us_panel_v1", manifest={})
    before = (dataset_dir / "part.parquet").read_bytes()

    with pytest.raises(DatasetExistsError, match="이미 있습니다"):
        write_dataset(df2, root, "us_panel_v1", manifest={})

    assert (dataset_dir / "part.parquet").read_bytes() == before
    assert pl.read_parquet(dataset_dir / "part.parquet").height == 1


@pytest.mark.parametrize("name", sorted(FROZEN_DATASET_NAMES))
def test_write_dataset_refuses_every_frozen_name_when_present(tmp_path: Path, name: str) -> None:
    """동결 이름 전부 — 이미 있는 루트에서 쓰기를 거부하고 메시지에 동결본이라고 적는다."""
    root = DataRoot(base=tmp_path)
    existing = root.datasets / name
    existing.mkdir(parents=True)
    (existing / "part.parquet").write_bytes(b"frozen")

    with pytest.raises(DatasetExistsError, match="동결 데이터셋"):
        write_dataset(pl.DataFrame({"a": [1]}), root, name, manifest={})

    assert (existing / "part.parquet").read_bytes() == b"frozen"
    assert not (existing / "manifest.json").exists()


def test_frozen_names_cover_the_documented_set() -> None:
    assert FROZEN_DATASET_NAMES == {
        "us_panel_v1",
        "us_panel_v2",
        "us_features_v1",
        "us_features_v2",
        "us_features_flow_v1",
        "us_labels_v1",
        "us_labels_v2",
        "us_labels_h5_v1",
        "us_labels_h63_v1",
        "us_labels_h63_v2",
    }


def test_write_dataset_has_no_bypass_flag() -> None:
    import inspect

    assert list(inspect.signature(write_dataset).parameters) == ["df", "root", "name", "manifest"]


# --- 유니버스 버전과 이름 규칙 (설계 §3) ----------------------------------------------------


def test_v2_name_needs_u2_token_and_v1_name_must_not_have_it() -> None:
    check_dataset_name("us_panel_v3_u2", universe_version="v2")
    check_dataset_name("us_features_flow_v1_fwd_u2", universe_version="v2")
    check_dataset_name("us_panel_v1", universe_version="v1")
    with pytest.raises(DatasetNamingError, match="_u2"):
        check_dataset_name("us_panel_v3", universe_version="v2")
    with pytest.raises(DatasetNamingError, match="_u2"):
        check_dataset_name("us_panel_v1_u2", universe_version="v1")
    with pytest.raises(DatasetNamingError, match="모르는"):
        check_dataset_name("us_panel_v1", universe_version="v3")


def test_u2_must_be_a_whole_token() -> None:
    """``u2``가 이름 일부로 박힌 것(``us2``·``plu2x``)은 v2 표시로 치지 않는다."""
    with pytest.raises(DatasetNamingError):
        check_dataset_name("us2_panel", universe_version="v2")
    check_dataset_name("us2_panel", universe_version="v1")


def test_security_boundaries_needs_v2_and_rejects_fwd_names() -> None:
    check_dataset_name("us_features_v3_u2", universe_version="v2", security_boundaries=True)
    with pytest.raises(DatasetNamingError, match="v2에서만"):
        check_dataset_name("us_features_v1", universe_version="v1", security_boundaries=True)
    with pytest.raises(DatasetNamingError, match="_fwd"):
        check_dataset_name("us_features_v2_fwd_u2", universe_version="v2", security_boundaries=True)
    # _fwd 이름은 꺼짐(동결 코드)이면 허용된다 — 전진 등록 §10 보조 판정 데이터셋.
    check_dataset_name("us_features_v2_fwd_u2", universe_version="v2")


def test_assert_dataset_absent_and_panel_manifest_default_is_v1(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    assert_dataset_absent(root, "us_panel_new")  # 없으면 통과
    write_dataset(pl.DataFrame({"a": [1]}), root, "us_panel_new", manifest={})
    with pytest.raises(DatasetExistsError):
        assert_dataset_absent(root, "us_panel_new")
    _, manifest = read_panel_dataset(root, "us_panel_new")
    assert manifest_universe_version(manifest) == "v1"  # 동결 패널은 universe_version 키가 없다
    assert manifest_universe_version({"universe_version": "v2"}) == "v2"
    with pytest.raises(FileNotFoundError):
        read_panel_dataset(root, "no_such_panel")


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
