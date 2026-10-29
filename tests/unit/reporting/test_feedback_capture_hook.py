"""Publisher boundary order, opt-in isolation and observer failure regression."""

from __future__ import annotations

import json

from reports_world import World, pr

DAY = "2026-10-07"


def test_absent_settings_never_starts_observer(monkeypatch):
    monkeypatch.delenv("FEEDBACK_CAPTURE_SETTINGS", raising=False)
    # Even malformed metadata is untouched when the observer is off.
    monkeypatch.setattr(pr, "rev", lambda *a: (_ for _ in ()).throw(AssertionError()))
    assert pr._feedback_event("prepare", {}, {}, DAY) is None


def test_intent_precedes_push_and_capture_precedes_journal_removal(tmp_path, monkeypatch):
    world = World(tmp_path)
    world.make_run(DAY)
    events = []
    real_push = pr.push

    def observer(mode, config, entry, unit, **evidence):
        assert unit in world.journal()["units"]
        assert entry["commit"] == world.journal()["units"][unit]["commit"]
        events.append(mode)
        if mode == "prepare":
            return {"status": "intent_recorded", "intent_ref": "a" * 64}
        assert evidence["intent"]["intent_ref"] == "a" * 64
        assert world.remote_head() == entry["commit"]
        return {"status": "captured"}

    def push(*args):
        events.append("push")
        real_push(*args)

    monkeypatch.setattr(pr, "_feedback_event", observer)
    monkeypatch.setattr(pr, "push", push)
    assert pr.run_step(world.config(DAY))["status"] == "published"
    assert events == ["prepare", "push", "published"]
    assert world.journal() is None


def test_bad_observer_settings_preserve_publish_and_correction(tmp_path, monkeypatch):
    world = World(tmp_path)
    world.make_run(DAY)
    settings = tmp_path / "broken.json"
    settings.write_text("not json")
    monkeypatch.setenv("FEEDBACK_CAPTURE_SETTINGS", str(settings))
    assert pr.run_step(world.config(DAY))["status"] == "published"
    world.make_run(DAY, score=0.123456)
    assert pr.run_step(world.config(DAY))["status"] == "correction_required"
    assert (
        pr.sync_step(world.config(DAY), correct=DAY, reason="synthetic correction")["status"]
        == "published"
    )
    assert world.journal() is None


def test_unchanged_rerun_and_push_recovery_pass_through_observer(tmp_path, monkeypatch):
    world = World(tmp_path)
    world.make_run(DAY)
    events = []

    def observer(mode, *args, **kwargs):
        events.append(mode)

    monkeypatch.setattr(pr, "_feedback_event", observer)
    assert pr.run_step(world.config(DAY))["status"] == "published"
    assert pr.run_step(world.config(DAY))["status"] == "unchanged"
    assert events == ["prepare", "published", "reobserve"]
    # Simulate crash after successful push but before deleting the recovery entry.
    pr.local_step(world.config(DAY))
    journal = world.journal()
    journal["units"][DAY].update(
        state=pr.PUSH_PENDING,
        commit=world.remote_head(),
        committed_source_sha256=journal["units"][DAY]["unit_sha256"],
    )
    pr.write_journal(world.checkout, journal)
    assert pr.sync_step(world.config(DAY))["status"] == "published"
    assert events[-1] == "reobserve"
    assert world.journal() is None


def test_successful_observer_replies_are_bounded_and_failures_sanitized(
    tmp_path, monkeypatch, capsys
):
    world = World(tmp_path)
    monkeypatch.setenv("FEEDBACK_CAPTURE_SETTINGS", str(tmp_path / "settings.json"))
    world.make_run(DAY)
    secret_marker = "synthetic-sensitive-marker"

    class Failed:
        pid = 999999999
        returncode = 1
        stdin = stdout = None

        def communicate(self, raw, timeout):
            request = json.loads(raw)
            assert set(request["config"]) == {
                "checkout_dir",
                "expected_remote_url",
                "remote_name",
                "branch",
            }
            assert secret_marker not in raw.decode()
            return secret_marker.encode(), None

        def wait(self, **kwargs):
            return 1

    monkeypatch.setattr(pr.subprocess, "Popen", lambda *a, **kw: Failed())
    monkeypatch.setattr(pr, "rev", lambda *a: "a" * 40)
    assert (
        pr._feedback_event(
            "prepare", world.config(DAY, report_sha256="a" * 64), {"irrelevant": secret_marker}, DAY
        )
        is None
    )
    output = capsys.readouterr()
    assert secret_marker not in output.err + output.out
    assert "coverage awaits review" in output.err
