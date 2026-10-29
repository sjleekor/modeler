"""CLI isolation, pinned document membership, and explicit source integration."""

from __future__ import annotations

import hashlib
import subprocess

from test_briefing import build_case
from test_store import DOC

from modeler.feedback.__main__ import main
from modeler.feedback.store import load_index


def pinned_document(tmp_path):
    repo = tmp_path / "my"
    repo.mkdir()
    doc = repo / "issues.md"
    doc.write_bytes(DOC)

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init")
    git("add", "issues.md")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-m",
        "synthetic issue document",
    )
    return doc, git("rev-parse", "HEAD"), git("hash-object", "issues.md")


def argv(tmp_path):
    day, checkout = build_case(tmp_path)
    doc, commit, blob = pinned_document(tmp_path)
    return [
        "build",
        "--source-root",
        str(day),
        "--checkout",
        str(checkout),
        "--output-root",
        str(tmp_path / "out"),
        "--issues-document",
        str(doc),
        "--my-commit",
        commit,
        "--issues-blob",
        blob,
    ]


def test_cli_roundtrip_idempotent_and_source_unchanged(tmp_path):
    args = argv(tmp_path)
    day = tmp_path / "2026-10-29"
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in day.rglob("*") if p.is_file()
    }
    assert main(args) == main(args) == 0
    index = load_index(tmp_path / "out")
    assert len(index["predictions"]) == len(index["publications"]) == 1
    assert index["predictions"][0]["evaluation_status"] == "pending"
    assert index["publications"][0]["verified_by"] is None
    assert not (tmp_path / "out/evidence" / (hashlib.sha256(DOC).hexdigest() + ".bin")).exists()
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in day.rglob("*") if p.is_file()
    }


def test_unbound_document_commit_stops_before_opening_source(tmp_path):
    args = argv(tmp_path)
    args[args.index("--my-commit") + 1] = "f" * 40
    report = tmp_path / "2026-10-29/report-2026-10-29.json"
    report.unlink()
    assert main(args) == 2
    assert not (tmp_path / "out").exists()


def test_missing_source_is_recorded_and_never_fabricated(tmp_path):
    args = argv(tmp_path)
    assert main(args) == 0
    (tmp_path / "2026-10-29/report-2026-10-29.json").unlink()
    assert main(args) == 2
    index = load_index(tmp_path / "out")
    assert len(index["predictions"]) == 1
    assert index["predictions"][0]["source_status"] == "source_incomplete"


def test_no_implicit_output_or_data_root(tmp_path, monkeypatch):
    args = argv(tmp_path)
    start = args.index("--output-root")
    del args[start : start + 2]
    monkeypatch.delenv("PREDICTION_FEEDBACK_ROOT", raising=False)
    assert main(args) == 2
    assert not (tmp_path / "out").exists()


def test_server_uses_pinned_mapping_without_human_document_or_git(tmp_path):
    args = argv(tmp_path)
    commit = args[args.index("--my-commit") + 1]
    blob = args[args.index("--issues-blob") + 1]
    doc = tmp_path / "my/issues.md"
    exported = tmp_path / "mapping-export"
    assert (
        main(
            [
                "mapping",
                "--issues-document",
                str(doc),
                "--my-commit",
                commit,
                "--issues-blob",
                blob,
                "--output-root",
                str(exported),
            ]
        )
        == 0
    )
    mapping_file = exported / "mapping.json"
    digest = hashlib.sha256(mapping_file.read_bytes()).hexdigest()
    (tmp_path / "my").rename(tmp_path / "offline-my")
    start = args.index("--issues-document")
    args[start : start + 2] = ["--mapping-file", str(mapping_file), "--mapping-sha256", digest]
    assert main(args) == 0
    assert len(load_index(tmp_path / "out")["predictions"]) == 1
    mapping_file.write_bytes(mapping_file.read_bytes() + b" ")
    assert main(args) == 2
