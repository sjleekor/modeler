"""Synthetic end-to-end publisher -> subprocess helper -> immutable capture."""

from __future__ import annotations

import json

from reports_world import PROJECT, World, pr, sha

from modeler.feedback import capture
from modeler.feedback.core import canonical_bytes, sha256_bytes

DAY = "2026-10-07"
REVIEWED = {
    "reviewed": True,
    "registry_complete": True,
    "controlled_epoch": True,
    "registry_sha256": "e" * 64,
    "producers": ["reports-publisher"],
}


def make_run(world, *, score=None):
    day = world.make_run(DAY, score=score)
    (day / "coordinator-inference.json").write_text(
        json.dumps(
            {
                "report_date": DAY,
                "report_sha256": sha(day / f"report-{DAY}.json"),
                "invocation_id": f"inv-{DAY}",
                "status": "inferred",
            }
        )
    )
    return day


def enable(tmp_path, monkeypatch, *, scope=None):
    path = tmp_path / "capture-settings.json"
    settings = {
        "schema": "prediction-feedback-capture-settings.v1",
        "enabled": True,
        "epoch_id": "synthetic-publisher-epoch",
        "output_root": str(tmp_path / "capture-output"),
    }
    if scope is not None:
        settings["producer_scope"] = scope
    path.write_text(json.dumps(settings))
    monkeypatch.setenv("FEEDBACK_CAPTURE_SETTINGS", str(path))
    monkeypatch.setenv("PYTHONPATH", str(PROJECT / "src"))
    return path, settings


def capture_config(world, settings):
    session = sha256_bytes(canonical_bytes({"epoch_id": settings["epoch_id"], "unit": DAY}))
    return {
        "checkout_dir": str(world.checkout),
        "report_date": DAY,
        "run_dir": str(world.runs / DAY),
        "feedback_capture": {"output_root": settings["output_root"], "session_id": session},
    }


def publications(config):
    return [
        r for r in capture.read_capture(config)["records"] if r["kind"] == "publication_captured"
    ]


def test_real_hook_archives_before_publisher_removes_journal(tmp_path, monkeypatch):
    world = World(tmp_path)
    day = make_run(world)
    _, settings = enable(tmp_path, monkeypatch)
    events = []
    real_observer = pr._feedback_event
    real_push = pr.push

    def observer(mode, config, entry, unit, **evidence):
        assert unit in world.journal()["units"]
        result = real_observer(mode, config, entry, unit, **evidence)
        assert result is not None
        events.append((mode, result["status"]))
        return result

    def push(*args):
        events.append(("push", "local-bare"))
        return real_push(*args)

    monkeypatch.setattr(pr, "_feedback_event", observer)
    monkeypatch.setattr(pr, "push", push)
    assert pr.run_step(world.config(DAY))["status"] == "published"
    assert events == [
        ("prepare", "intent_recorded"),
        ("push", "local-bare"),
        ("published", "captured"),
    ]
    assert world.journal() is None
    config = capture_config(world, settings)
    result = publications(config)[0]
    saved = capture.read_capture(config, result["capture_ref"])
    assert saved["artifacts"][saved["report_sha256"]] == (day / f"report-{DAY}.json").read_bytes()
    assert result["producer_event"]["remote_identity"] == "synthetic-local"
    assert capture.coverage_summary(config)["coverage_status"] == "awaiting_coverage"


def test_real_correction_keeps_first_report_and_hashes_actual_revision_two(tmp_path, monkeypatch):
    world = World(tmp_path)
    day = make_run(world)
    _, settings = enable(tmp_path, monkeypatch, scope=REVIEWED)
    assert pr.run_step(world.config(DAY))["status"] == "published"
    config = capture_config(world, settings)
    first = publications(config)[0]
    original = capture.read_capture(config, first["capture_ref"])
    make_run(world, score=0.123456)
    assert pr.run_step(world.config(DAY))["status"] == "correction_required"
    assert (
        pr.sync_step(world.config(DAY), correct=DAY, reason="synthetic report correction")["status"]
        == "published"
    )
    rows = publications(config)
    assert len(rows) == 2
    corrected = rows[1]
    assert corrected["revision"] == 2
    assert corrected["report_sha256"] != first["report_sha256"]
    assert corrected["unit_sha256"] != corrected["original_render_unit_sha256"]
    assert original == capture.read_capture(config, first["capture_ref"])
    assert (
        original["artifacts"][first["report_sha256"]] != (day / f"report-{DAY}.json").read_bytes()
    )
    assert capture.coverage_summary(config)["first_receipt_ref"] == first["capture_ref"]
    assert world.journal() is None


def test_real_unchanged_hook_skips_new_receipt(tmp_path, monkeypatch):
    world = World(tmp_path)
    make_run(world)
    _, settings = enable(tmp_path, monkeypatch)
    assert pr.run_step(world.config(DAY))["status"] == "published"
    config = capture_config(world, settings)
    before = capture.read_capture(config)
    assert pr.run_step(world.config(DAY))["status"] == "unchanged"
    assert capture.read_capture(config) == before
    assert len(publications(config)) == 1


def test_observer_failure_after_intent_does_not_change_publication_result(tmp_path, monkeypatch):
    world = World(tmp_path)
    make_run(world)
    settings_path, settings = enable(tmp_path, monkeypatch)
    real_push = pr.push

    def push(*args):
        settings_path.write_text("synthetic broken settings after durable intent")
        return real_push(*args)

    monkeypatch.setattr(pr, "push", push)
    assert pr.run_step(world.config(DAY))["status"] == "published"
    assert world.journal() is None
    assert world.remote_head()
    config = capture_config(world, settings)
    assert publications(config) == []
    summary = capture.coverage_summary(config)
    assert summary["coverage_status"] == "awaiting_coverage"
    assert "unfinished_publication_intent" in summary["reasons"]
