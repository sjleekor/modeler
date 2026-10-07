"""t15_build_datasets — 계획 판정, smoke 이름 바꾸기, 동결본 해시 가드. 빌드는 하지 않는다."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from modeler.models._02_updown_prob.experiments import t15_build_datasets as b


def _lake(tmp_path):
    return SimpleNamespace(dataset_dir=lambda m: tmp_path / "shared" / m)


def _e5(tmp_path):
    return SimpleNamespace(dataset_dir=lambda m: tmp_path / "_e5" / m)


def _write_manifest(path, **over):
    path.mkdir(parents=True, exist_ok=True)
    m = {
        "period": {"start": "2015-01-02", "end": "2025-07-31"},
        "row_count": 5223425,
        "extra": {"feature_set": "FS0", "flow_variant": "lag1", "preprocess_profile": "tree",
                  "horizon": 20, "n_folds": 5, "std_layout": "per_fold"},
    }
    m.update(over)
    (path / "dataset_manifest.json").write_text(json.dumps(m))


def test_six_targets_and_paths(tmp_path):
    assert b.TARGET_NAMES == ("fs0_rank", "fs0_tree", "fs1", "fs2", "native_t", "fs1h_fs3")
    keys = {t.name: t.key for t in b.TARGETS}
    assert keys["fs0_tree"] == "FS0_h20_lag1_tree"
    assert keys["native_t"] == "FS1h_h20_native_t_rank"
    assert keys["fs1h_fs3"] == "FS1h_FS3_h20_lag1_rank"
    assert b.FROZEN_KEY not in keys.values()
    paths = {t.name: b.target_dir(t, _lake(tmp_path), _e5(tmp_path)) for t in b.TARGETS}
    assert "_e5" in str(paths["fs1h_fs3"])
    assert all("_e5" not in str(p) for n, p in paths.items() if n != "fs1h_fs3")


def test_plan_build_when_absent(tmp_path):
    t = b.TARGETS[1]
    plan = b.plan_target(t, _lake(tmp_path), _e5(tmp_path))
    assert plan.action == "build"


def test_smoke_is_renamed_not_deleted(tmp_path):
    t = b.TARGETS[1]  # fs0_tree
    path = b.target_dir(t, _lake(tmp_path), _e5(tmp_path))
    _write_manifest(path, period={"start": "2015-01-02", "end": "2019-12-31"},
                    extra={"feature_set": "FS0", "n_folds": 2})
    (path / "marker.txt").write_text("keep")
    plan = b.plan_target(t, _lake(tmp_path), _e5(tmp_path))
    assert plan.action == "rename_then_build"
    assert plan.rename_to.name == "_SMOKE__FS0_h20_lag1_tree"
    b.apply_rename(plan)
    assert not path.exists()
    assert (plan.rename_to / "marker.txt").read_text() == "keep"
    assert (plan.rename_to / "dataset_manifest.json").is_file()


def test_full_is_reused_not_renamed(tmp_path):
    t = b.TARGETS[1]
    path = b.target_dir(t, _lake(tmp_path), _e5(tmp_path))
    _write_manifest(path)
    plan = b.plan_target(t, _lake(tmp_path), _e5(tmp_path))
    assert plan.action == "reuse" and plan.rename_to is None
    assert path.exists()


def test_unknown_existing_dir_aborts(tmp_path):
    t = b.TARGETS[1]
    path = b.target_dir(t, _lake(tmp_path), _e5(tmp_path))
    path.mkdir(parents=True)  # manifest 없음
    assert b.plan_target(t, _lake(tmp_path), _e5(tmp_path)).action == "abort"
    _write_manifest(path, row_count=1, period={"start": "2016-01-01", "end": "2024-01-01"})
    assert b.plan_target(t, _lake(tmp_path), _e5(tmp_path)).action == "abort"


def test_smoke_rename_target_exists_aborts(tmp_path):
    t = b.TARGETS[1]
    path = b.target_dir(t, _lake(tmp_path), _e5(tmp_path))
    _write_manifest(path, period={"start": "2015-01-02", "end": "2019-12-31"}, extra={"n_folds": 2})
    path.with_name("_SMOKE__" + path.name).mkdir(parents=True)
    assert b.plan_target(t, _lake(tmp_path), _e5(tmp_path)).action == "abort"


def test_frozen_hash_guard(tmp_path):
    d = tmp_path / "FS1h_h20_lag1_rank"
    _write_manifest(d)
    before = b.manifest_hash(d)
    b.assert_frozen_unchanged(before, b.manifest_hash(d))
    (d / "dataset_manifest.json").write_text("{}")
    with pytest.raises(b.FrozenDatasetChanged):
        b.assert_frozen_unchanged(before, b.manifest_hash(d))
    assert b.manifest_hash(tmp_path / "nope") is None


def test_dry_run_changes_nothing(tmp_path, capsys):
    shared, e5 = _lake(tmp_path), _e5(tmp_path)
    smoke = b.target_dir(b.TARGETS[1], shared, e5)
    _write_manifest(smoke, period={"start": "2015-01-02", "end": "2019-12-31"}, extra={"n_folds": 2})
    _write_manifest(shared.dataset_dir("02_updown_prob") / b.FROZEN_KEY)
    before = sorted(p.as_posix() for p in tmp_path.rglob("*"))
    assert b.run(None, True, shared=shared, e5=e5) == 0
    assert sorted(p.as_posix() for p in tmp_path.rglob("*")) == before
    assert "rename_then_build" in capsys.readouterr().out


def test_compare_manifests():
    pre = {"row_count": 5, "period": {"s": 1}, "extra": {"design_columns": ["a", "b"], "null_ratios": {"a": 0.1}}}
    assert all(ok for _, ok, _ in b.compare_manifests(pre, pre))
    new = {"row_count": 6, "period": {"s": 1}, "extra": {"design_columns": ["b", "a"], "null_ratios": {"a": 0.2}}}
    res = {k: ok for k, ok, _ in b.compare_manifests(pre, new)}
    assert res == {"row_count": False, "design_columns": False, "period": True, "null_ratios": False}
