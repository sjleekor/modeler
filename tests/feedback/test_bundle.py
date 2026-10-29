"""Pinned synthetic calendar metadata only; no network or production data reads."""

from __future__ import annotations

import copy
import json
import sys

import pytest

from modeler.feedback import bundle
from modeler.feedback.core import FeedbackError, sha256_bytes


def calendar():
    return {
        "market": "KR",
        "timezone": "Asia/Seoul",
        "coverage_start": "2026-11-01",
        "coverage_end": "2026-11-30",
        "sessions": ["2026-11-05", "2026-11-03", "2026-11-02", "2026-11-04"],
        "default_open_at": "09:00",
        "default_close_at": "15:30",
        "unconfirmed_dates": ["2026-11-04"],
        "overrides": [{"date": "2026-11-05", "open_at": "10:00", "close_at": "16:30"}],
    }


def write_calendar(tmp_path, value):
    path = tmp_path / "calendar.json"
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return path, sha256_bytes(raw)


def test_only_confirmed_unique_sessions_are_returned_sorted(tmp_path):
    path, digest = write_calendar(tmp_path, calendar())
    assert bundle.pinned_sessions(path, digest) == ["2026-11-02", "2026-11-03"]


def test_confirmed_override_and_ordinary_sessions_are_included(tmp_path):
    value = calendar()
    value["overrides"][0]["confirmed"] = True
    path, digest = write_calendar(tmp_path, value)
    assert bundle.pinned_sessions(path, digest) == ["2026-11-02", "2026-11-03", "2026-11-05"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("market", "US"),
        ("timezone", "UTC"),
        ("coverage_start", "2026-12-01"),
        ("coverage_end", "2026-11-03"),
        ("coverage_start", "20261101"),
        ("coverage_end", "2026-W48-1"),
        ("sessions", ["20261102"]),
        ("sessions", ["2026-11-02", "2026-11-02"]),
        ("sessions", ["2026-11-31"]),
        ("sessions", [20261102]),
        ("sessions", {}),
        ("default_open_at", "15:30"),
        ("default_open_at", "09:00+09:00"),
        ("default_close_at", "25:00"),
        ("default_close_at", None),
        ("unconfirmed_dates", ["2026-11-04", "2026-11-04"]),
        ("unconfirmed_dates", ["2026-11-06"]),
        ("unconfirmed_dates", ["20261104"]),
        ("unconfirmed_dates", {}),
        ("overrides", {}),
        ("overrides", [None]),
    ],
)
def test_invalid_calendar_contract_rejected(tmp_path, field, value):
    document = calendar()
    document[field] = value
    path, digest = write_calendar(tmp_path, document)
    with pytest.raises(FeedbackError):
        bundle.pinned_sessions(path, digest)


@pytest.mark.parametrize("change", ["duplicate", "undeclared", "conflict", "bool", "hours"])
def test_override_contract_rejected(tmp_path, change):
    value = calendar()
    row = value["overrides"][0]
    if change == "duplicate":
        value["overrides"].append(copy.deepcopy(row))
    elif change == "undeclared":
        row["date"] = "2026-11-06"
    elif change == "conflict":
        row.update(date="2026-11-04", confirmed=True)
    elif change == "bool":
        row["confirmed"] = 1
    else:
        row["close_at"] = row["open_at"]
    path, digest = write_calendar(tmp_path, value)
    with pytest.raises(FeedbackError):
        bundle.pinned_sessions(path, digest)


def test_calendar_byte_pin_required(tmp_path):
    path, _ = write_calendar(tmp_path, calendar())
    with pytest.raises(FeedbackError, match="pin mismatch"):
        bundle.pinned_sessions(path, "f" * 64)
    with pytest.raises(FeedbackError, match="digest"):
        bundle.pinned_sessions(path, "unknown")


def test_symlink_and_oversized_metadata_rejected(tmp_path):
    path, digest = write_calendar(tmp_path, calendar())
    alias = tmp_path / "alias.json"
    alias.symlink_to(path)
    with pytest.raises(FeedbackError):
        bundle.pinned_sessions(alias, digest)
    path.write_bytes(b"x" * (bundle.MAX_BUNDLE_METADATA_BYTES + 1))
    with pytest.raises(FeedbackError, match="oversized"):
        bundle.pinned_sessions(path, sha256_bytes(path.read_bytes()))


def test_duplicate_json_fields_are_not_accepted(tmp_path):
    path = tmp_path / "calendar.json"
    raw = b'{"market":"KR","market":"US"}'
    path.write_bytes(raw)
    with pytest.raises(FeedbackError, match="duplicate fields"):
        bundle.pinned_sessions(path, sha256_bytes(raw))


def test_calendar_helper_does_not_import_serving_or_data_dependencies(tmp_path):
    before = set(sys.modules)
    path, digest = write_calendar(tmp_path, calendar())
    bundle.pinned_sessions(path, digest)
    imported = set(sys.modules) - before
    assert not any(name.startswith(("modeler.serving", "polars", "pandas")) for name in imported)


def test_cli_outputs_confirmed_sessions_or_generic_error(tmp_path, capsys):
    path, digest = write_calendar(tmp_path, calendar())
    assert bundle.main(["--calendar-file", str(path), "--calendar-sha256", digest]) == 0
    assert json.loads(capsys.readouterr().out) == {"sessions": ["2026-11-02", "2026-11-03"]}
    assert bundle.main(["--calendar-file", str(path), "--calendar-sha256", "f" * 64]) == 2
    assert capsys.readouterr().err == "feedback-bundle: metadata verification failed\n"


def source_repo(tmp_path):
    import subprocess

    root = tmp_path / "source"
    root.mkdir()
    names = [*bundle._INTEGRATION_FILES, "src/modeler/feedback/runtime.py"]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Synthetic fixture."""\n' if name.endswith(".py") else "#!/bin/sh\n")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *args],
            stderr=subprocess.DEVNULL,
        )

    git("init", "-q")
    git("add", "--", "src", "deploy")
    git(
        "-c",
        "user.name=Synthetic Fixture",
        "-c",
        "user.email=synthetic@example.invalid",
        "commit",
        "-qm",
        "synthetic sources",
    )
    return root, git("rev-parse", "HEAD").decode().strip()


def test_inventory_is_explicit_and_excludes_git_venv_secret_and_extra_modules(tmp_path):
    root, commit = source_repo(tmp_path)
    for name in (".env", ".venv/secret.py", "src/modeler/serving/not-in-closure.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic sentinel never copied")
    manifest = bundle.build_manifest(root, expected_commit=commit)
    assert manifest["modeler_commit"] == commit
    assert manifest["source_state"] == "commit"
    assert manifest["changed_or_untracked_files"] == []
    assert set(manifest["files"]) == set(bundle.source_inventory(root))
    assert "synthetic sentinel never copied" not in json.dumps(manifest)
    assert all(row["matches_commit"] for row in manifest["files"].values())
    assert "src/modeler/reporting/security_names.py" in manifest["files"]
    assert "deploy/reports/validate_reports.py" in manifest["files"]


def test_dirty_and_untracked_bytes_are_honest_and_inventory_is_deterministic(tmp_path):
    root, commit = source_repo(tmp_path)
    modified = "src/modeler/feedback/runtime.py"
    untracked = "src/modeler/feedback/new_helper.py"
    (root / modified).write_text("import json\nfrom pathlib import Path\n")
    (root / untracked).write_text("from .core import FeedbackError\n")
    manifest = bundle.build_manifest(root, expected_commit=commit)
    assert manifest == bundle.build_manifest(root, expected_commit=commit)
    assert manifest["source_state"] == "working_tree"
    assert manifest["changed_or_untracked_files"] == [untracked, modified]
    assert manifest["files"][modified]["imports"] == ["json", "pathlib"]
    assert manifest["files"][untracked]["imports"] == [".core"]
    assert manifest["files"][untracked]["committed_sha256"] is None
    assert manifest["files"][modified]["committed_sha256"] is not None
    rows = manifest["files"]
    assert manifest["source_inventory_sha256"] == sha256_bytes(bundle.canonical_bytes(rows))


def test_source_commit_must_be_verifiable_instead_of_unknown_or_caller_claim(tmp_path):
    root, commit = source_repo(tmp_path)
    with pytest.raises(FeedbackError):
        bundle.build_manifest(root, expected_commit="unknown")
    with pytest.raises(FeedbackError, match="cannot be verified"):
        bundle.build_manifest(root, expected_commit="f" * 40)
    with pytest.raises(FeedbackError, match="cannot be verified"):
        bundle.build_manifest(tmp_path, expected_commit=commit)


@pytest.mark.parametrize(
    "files", [[".env"], [".git/config"], [".venv/example.py"], ["../escape.py"], []]
)
def test_manifest_file_allowlist_rejects_non_code_inputs(tmp_path, files):
    root, commit = source_repo(tmp_path)
    with pytest.raises(FeedbackError):
        bundle.build_manifest(root, expected_commit=commit, files=files)


def test_manifest_rejects_source_symlink_and_duplicate_inventory(tmp_path):
    root, commit = source_repo(tmp_path)
    name = "src/modeler/feedback/runtime.py"
    with pytest.raises(FeedbackError, match="unique"):
        bundle.build_manifest(root, expected_commit=commit, files=[name, name])
    path = root / name
    path.unlink()
    path.symlink_to(root / "src/modeler/__init__.py")
    with pytest.raises(FeedbackError):
        bundle.build_manifest(root, expected_commit=commit)


def test_inventory_declares_optional_nonstdlib_paths_instead_of_claiming_full_closure(tmp_path):
    root, commit = source_repo(tmp_path)
    notes = bundle.build_manifest(root, expected_commit=commit)["dependency_notes"]
    assert notes["optional_external_modules"]["feedback.shadow"] == [
        "polars",
        "modeler.models._03_kr_next.shadow",
    ]
    assert notes["optional_external_modules"][
        "reporting.security_names (name input generation)"
    ] == ["polars"]


def test_manifest_cli_uses_only_local_source_and_verified_commit(tmp_path, capsys):
    root, commit = source_repo(tmp_path)
    assert bundle.main(["--code-root", str(root), "--expected-commit", commit]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["modeler_commit"] == commit
