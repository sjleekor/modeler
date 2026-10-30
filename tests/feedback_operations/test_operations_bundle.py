"""Fake code projections only: no reports, keys, prices, runtime or network."""

import io
import json
import subprocess
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from modeler.feedback.core import FeedbackError, canonical_bytes, sha256_bytes
from modeler.feedback_operations import bundle

REPO = Path(__file__).resolve().parents[2]


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


@pytest.fixture
def projection():
    # Use /private/tmp spelling: /tmp itself is a macOS symlink and is rejected.
    with TemporaryDirectory(prefix="feedback-code-projection-", dir="/private/tmp") as tmp:
        root = Path(tmp)
        files = set(bundle._FIXED_FILES) | bundle._REQUIRED
        for package in bundle._PACKAGES:
            files.add(f"src/modeler/{package}/fake_projection.py")
        for name in sorted(files):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(".json") or name.endswith(".sh"):
                # Only reviewed inert templates/wrapper; every Python body is fake.
                path.write_bytes((REPO / name).read_bytes())
            else:
                path.write_text('"""Fake source projection, no runtime imports."""\n')
        git(root, "init", "-q")
        git(root, "add", "src", "deploy")
        git(
            root,
            "-c",
            "user.name=Fake Projection",
            "-c",
            "user.email=fake@example.invalid",
            "commit",
            "-qm",
            "fake source projection",
        )
        yield root, git(root, "rev-parse", "HEAD")


def test_inventory_is_narrow_and_complete(projection):
    root, commit = projection
    for name in (".envrc", "secrets/key", "data/prices.parquet", "tests/fake.py", ".venv/x.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("excluded fake projection")
    manifest = bundle.build_manifest(root, expected_commit=commit)
    assert manifest["source_state"] == "commit"
    assert manifest["changed_or_untracked_files"] == []
    assert set(bundle._FIXED_FILES) <= manifest["files"].keys()
    assert all("excluded" not in row for row in manifest["files"])
    assert "src/modeler/feedback_operations/fake_projection.py" in manifest["files"]
    assert "src/modeler/feedback_outcomes/fake_projection.py" in manifest["files"]
    assert manifest["source_inventory_sha256"] == sha256_bytes(canonical_bytes(manifest["files"]))
    assert manifest["dependency_notes"]["venv_copied_or_dependencies_installed"] is False


def test_dirty_and_untracked_source_is_honest(projection):
    root, commit = projection
    changed = "src/modeler/feedback_outcomes/fake_projection.py"
    untracked = "src/modeler/feedback_operations/fresh.py"
    (root / changed).write_text("VALUE = 'changed fake bytes'\n")
    (root / untracked).write_text("VALUE = 'new fake bytes'\n")
    manifest = bundle.build_manifest(root, expected_commit=commit)
    assert manifest["source_state"] == "working_tree"
    assert manifest["source_git_commit"] == commit
    assert manifest["changed_or_untracked_files"] == sorted([changed, untracked])
    assert manifest["files"][changed]["committed_sha256"] is not None
    assert manifest["files"][untracked]["committed_sha256"] is None
    assert not manifest["files"][untracked]["matches_commit"]
    with pytest.raises(FeedbackError, match="Git HEAD"):
        bundle.build_manifest(root, expected_commit="a" * 40)


def test_archive_is_deterministic_and_hashes_exact_payload(projection):
    root, commit = projection
    first, second = root / "first", root / "second"
    receipts = [bundle.build_bundle(root, dest, expected_commit=commit) for dest in (first, second)]
    assert receipts[0]["archive_sha256"] == receipts[1]["archive_sha256"]
    assert (first / "bundle.tar.gz").read_bytes() == (second / "bundle.tar.gz").read_bytes()
    manifest_raw = (first / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    assert receipts[0]["manifest_sha256"] == sha256_bytes(manifest_raw)
    with tarfile.open(
        fileobj=io.BytesIO((first / "bundle.tar.gz").read_bytes()), mode="r:gz"
    ) as tar:
        assert tar.getnames() == sorted(
            ["manifest.json", *[f"code/{p}" for p in manifest["files"]]]
        )
        for member in tar:
            assert member.isfile() and member.mtime == 0 and member.uid == member.gid == 0
            raw = tar.extractfile(member).read()
            if member.name == "manifest.json":
                assert raw == manifest_raw
            else:
                name = member.name.removeprefix("code/")
                assert sha256_bytes(raw) == manifest["files"][name]["sha256"]
                assert raw == (first / member.name).read_bytes()
    assert not receipts[0]["activation_performed"]
    with pytest.raises(FeedbackError, match="already exist"):
        bundle.build_bundle(root, first, expected_commit=commit)


@pytest.mark.parametrize(
    "bad", ["../outside.py", "src/modeler/feedback/../secret.py", ".envrc", "src/other.py"]
)
def test_unknown_paths_rejected_before_open(projection, bad, monkeypatch):
    root, _ = projection
    with monkeypatch.context() as patch:
        patch.setattr(bundle.os, "open", lambda *a, **k: pytest.fail("opened invalid source"))
        with pytest.raises(FeedbackError, match="allowlist"):
            bundle._read_source(root, bad)


@pytest.mark.parametrize("kind", ["leaf", "package", "root", "hardlink"])
def test_symlinks_and_hardlinks_cannot_escape(projection, kind):
    root, commit = projection
    source = root / "src/modeler/feedback_operations/fake_projection.py"
    if kind == "leaf":
        source.unlink()
        source.symlink_to(root / "src/modeler/__init__.py")
    elif kind == "package":
        package = source.parent
        moved = root / "moved-package"
        package.rename(moved)
        package.symlink_to(moved, target_is_directory=True)
    elif kind == "root":
        linked = root / "linked-root"
        linked.symlink_to(root, target_is_directory=True)
        root = linked
    else:
        source.unlink()
        source.hardlink_to(root / "src/modeler/__init__.py")
    with pytest.raises((FeedbackError, OSError)):
        bundle.build_manifest(root, expected_commit=commit)


@pytest.mark.parametrize(
    "name,mutate",
    [
        ("runtime.disabled.json", lambda value: value.update(enabled=True)),
        ("runtime.disabled.json", lambda value: value.update(timeout_seconds=151)),
        ("runtime.disabled.json", lambda value: value.update(extra_credential="FAKE-ONLY")),
        ("capture.disabled.json", lambda value: value["producer_scope"].update(reviewed=True)),
        ("operations-event.disabled.json", lambda value: value["events"][0].update(enabled=1)),
    ],
)
def test_changed_templates_rejected(projection, name, mutate):
    root, commit = projection
    path = root / "deploy/feedback" / name
    value = json.loads(path.read_bytes())
    mutate(value)
    path.write_bytes(canonical_bytes(value))
    with pytest.raises(FeedbackError, match="template pin"):
        bundle.build_manifest(root, expected_commit=commit)


def test_wrapper_pin_and_inventory_subsets_rejected(projection):
    root, commit = projection
    with pytest.raises(FeedbackError, match="complete narrow"):
        bundle.build_manifest(root, expected_commit=commit, files=["src/modeler/__init__.py"])
    (root / "deploy/feedback/feedback-stage.sh").write_text("#!/bin/bash\nexit 0\n")
    with pytest.raises(FeedbackError, match="reviewed wrapper"):
        bundle.build_manifest(root, expected_commit=commit)


def test_cli_and_destination_limits(projection, capsys):
    root, commit = projection
    assert bundle.main(["--code-root", str(root), "--expected-commit", commit]) == 0
    assert json.loads(capsys.readouterr().out)["source_git_commit"] == commit
    assert bundle.main(["--code-root", str(root), "--expected-commit", "invalid"]) == 2
    assert capsys.readouterr().err == "feedback-operations-bundle: verification failed\n"
    for dest in (root / ".." / "unsafe", Path("relative"), Path("/private/outside")):
        with pytest.raises(FeedbackError):
            bundle.build_bundle(root, dest, expected_commit=commit)


def test_missing_entrypoint_and_oversized_file(projection):
    root, commit = projection
    source = root / "src/modeler/feedback_operations/credential_launcher.py"
    source.unlink()
    with pytest.raises(FeedbackError, match="required entrypoints"):
        bundle.build_manifest(root, expected_commit=commit)
    source.write_bytes(b"#" * (bundle.MAX_FILE_BYTES + 1))
    with pytest.raises(FeedbackError, match="bounded regular"):
        bundle.build_manifest(root, expected_commit=commit)


def test_proposal_keeps_inert_flags_and_explicit_launcher():
    proposal = json.loads((REPO / "deploy/feedback/operations-event.disabled.json").read_bytes())
    event = proposal["events"][0]
    assert proposal["enabled"] is False and event["enabled"] == 0
    assert not proposal["is_cronicle_create_payload"]
    assert proposal["requires_fresh_server_review"]
    assert proposal["requires_live_category_target_plugin_review"]
    assert event["id"] == "prediction_feedback_kr_incremental"
    assert (event["category"], event["target"], event["plugin"]) == (
        "general",
        "maingrp",
        "shellplug",
    )
    assert event["argv"][1:5] == [
        "-s",
        "-P",
        "-m",
        "modeler.feedback_operations.credential_launcher",
    ]
    assert "UNPROVISIONED" in event["argv"][6]
    assert "UNINSTALLED" in event["argv"][8]
    assert event["environment"]["PYTHONPATH"] == event["environment"]["FEEDBACK_CODE_ROOT"] + "/src"
    assert not any("KEY" in name or "FD" in name for name in event["environment"])
