"""Guarded panel builder: tag guard, write-once, manifest, serving without labels."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import duckdb
import polars as pl
import pytest

from modeler.models._03_kr_next import build_panel as bp
from modeler.models._03_kr_next import dataset as ds

from .test_kr_next_dataset import MART_VIEWS, SMALL_UNIVERSE, _con


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True, env=env
    ).stdout.strip()


def _repo(path: Path, tag: bool) -> Path:
    path.mkdir()
    _git(path, "init", "-q")
    (path / "doc.md").write_text("prereg")
    _git(path, "add", "doc.md")
    _git(path, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "init")
    if tag:
        _git(path, "-c", "tag.gpgsign=false", "tag", "kr-next-frozen")
    return path


def _build(con: duckdb.DuckDBPyConnection, out: Path, mode: str, **kw) -> dict:
    return bp.build_panel_file(
        con,
        mode=mode,
        snapshot="2020-02-01",
        start="2019-01-01",
        end=None,
        out_dir=out,
        feature_set="noflow",
        mart_overrides=MART_VIEWS,
        universe_filter=SMALL_UNIVERSE,
        **kw,
    )


def test_labeled_without_tag_exits_2_before_touching_the_lake(tmp_path, capsys) -> None:
    repo = _repo(tmp_path / "prereg", tag=False)
    out = tmp_path / "out"
    code = bp.main(
        [
            "--mode", "labeled",
            "--lake", str(tmp_path / "no_such_lake"),
            "--snapshot", "2020-02-01",
            "--end", "2025-07-31",
            "--out-dir", str(out),
            "--prereg-repo", str(repo),
        ]
    )  # fmt: skip
    assert code == 2
    assert "kr-next-frozen" in capsys.readouterr().err
    assert not out.exists()
    assert not list(tmp_path.glob(".*duckdb_tmp"))  # no connection was even prepared


def test_labeled_guard_uses_a_missing_repo_too(tmp_path) -> None:
    code = bp.main(
        [
            "--mode", "labeled", "--lake", str(tmp_path), "--snapshot", "s", "--end", "2020-01-01",
            "--out-dir", str(tmp_path / "o"), "--prereg-repo", str(tmp_path / "nope"),
        ]
    )  # fmt: skip
    assert code == 2


def test_serving_needs_no_tag_but_stops_on_a_missing_view_file(tmp_path, capsys) -> None:
    code = bp.main(
        [
            "--mode", "serving", "--lake", str(tmp_path), "--snapshot", "s", "--end", "2020-01-31",
            "--out-dir", str(tmp_path / "o"), "--prereg-repo", str(tmp_path / "nope"),
        ]
    )  # fmt: skip
    err = capsys.readouterr().err
    assert code == 2 and "kr-next-frozen" not in err  # refused for the lake, not for a tag


def test_write_once_refuses_and_leaves_the_files_alone(tmp_path, capsys) -> None:
    out = tmp_path / "out"
    out.mkdir()
    panel, manifest = bp.output_paths(out, "serving")
    panel.write_bytes(b"keep")
    code = bp.main(
        [
            "--mode", "serving", "--lake", str(tmp_path), "--snapshot", "s", "--end", "2020-01-31",
            "--out-dir", str(out), "--prereg-repo", str(tmp_path),
        ]
    )  # fmt: skip
    assert code == 2 and "not overwriting" in capsys.readouterr().err
    assert panel.read_bytes() == b"keep" and not manifest.exists()
    con = _con()
    with pytest.raises(bp.BuildRefused):
        _build(con, out, "serving")
    assert panel.read_bytes() == b"keep"


def test_serving_output_has_no_label_columns_and_a_manifest(tmp_path) -> None:
    con = _con()
    out = tmp_path / "serving"
    m = _build(con, out, "serving")
    frame = pl.read_parquet(out / "panel_serving.parquet")
    assert frame.columns == m["columns"]
    assert not [c for c in frame.columns if bp._BANNED_SERVING.search(c)]
    assert {"liq_notice_flag", "liq_notice_rcept_no", "fin_ttm_conflict_v2"} <= set(frame.columns)
    assert frame["liq_notice_flag"].null_count() == 0
    disk = json.loads((out / "manifest_serving.json").read_text())
    assert disk["mode"] == "serving" and disk["snapshot"] == "2020-02-01"
    assert disk["row_count"] == frame.height and disk["end"] is None
    assert disk["start"] == "2019-01-01" and disk["columns"] == frame.columns
    assert "modeler_commit" in disk and "modeler_dirty" in disk
    assert (
        disk["panel_sha256"]
        == hashlib.sha256((out / "panel_serving.parquet").read_bytes()).hexdigest()
    )
    shas = disk["view_sql_sha256"]
    assert {"kn_feat_fin_v2", "kn_tail", "kn_uni", "kn_liq"} <= set(shas)
    assert "kn_label" not in shas
    assert all(len(v) == 64 for v in shas.values())
    # the hash is of the SQL text the panel was built from
    parts: dict[str, str] = {}
    ds.build_kr_next_panel_sql(
        "noflow",
        mart_overrides=MART_VIEWS,
        universe_filter=SMALL_UNIVERSE,
        start_date="2019-01-01",
        view_sql_out=parts,
    )
    assert shas["kn_liq"] == hashlib.sha256(parts["kn_liq"].encode()).hexdigest()


def test_labeled_build_has_labels_and_a_label_view_hash(tmp_path) -> None:
    con = _con()
    out = tmp_path / "labeled"
    m = _build(con, out, "labeled", extra_manifest={"prereg_commit": "abc"})
    frame = pl.read_parquet(out / "panel_labeled.parquet")
    assert {"y", "trainable", "label_exit_kind_c1_20d", "label_end_date_c1_20d"} <= set(
        frame.columns
    )
    assert "fwd_ret_c1_20d" not in frame.columns  # not emitted unless asked
    assert "kn_label" in m["view_sql_sha256"] and m["prereg_commit"] == "abc"
    assert {"liq_notice_flag", "liq_notice_rcept_no"} <= set(frame.columns)


def test_serving_refuses_raw_label_and_bad_mode(tmp_path) -> None:
    con = _con()
    with pytest.raises(bp.BuildRefused):
        _build(con, tmp_path / "a", "serving", keep_raw_label=True)
    with pytest.raises(bp.BuildRefused):
        _build(con, tmp_path / "b", "training")
    assert not (tmp_path / "a").exists()
