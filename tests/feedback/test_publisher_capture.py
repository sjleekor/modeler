"""Optional hook uses synthetic files and git, without any network call."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_briefing import DAY, git, write
from test_capture import INTENT, SCOPE, VERIFIED, setup

from modeler.feedback import capture
from modeler.feedback import publisher_capture as hook


def request_case(tmp_path, *, scope=SCOPE):
    config, entry, report, baseline = setup(tmp_path)
    path = tmp_path / "capture-settings.json"
    settings = {
        "schema": hook.SETTINGS_SCHEMA,
        "enabled": True,
        "output_root": str(tmp_path / "hook-output"),
        "epoch_id": "synthetic-epoch",
    }
    if scope is not None:
        settings["producer_scope"] = scope
    write(path, settings)
    request = {
        "mode": "prepare",
        "settings_path": str(path),
        "config": {
            "checkout_dir": config["checkout_dir"],
            "expected_remote_url": capture.REMOTE,
            "branch": "main",
            "remote_name": "origin",
        },
        "entry": entry,
        "unit": DAY,
        "previous_remote_head": baseline,
        "observation_started_at": INTENT,
    }
    return request, report


def config_for(request):
    return hook._request(request)[1]


def published(request, intent=None, *, mode="published", **extra):
    return hook.handle(
        {
            **request,
            "mode": mode,
            "intent": intent,
            "remote_head": request["entry"]["commit"],
            "verified_by": VERIFIED,
            **extra,
        }
    )


def test_prepare_publish_records_closed_intent_and_covered_epoch(tmp_path):
    request, _ = request_case(tmp_path)
    response = hook.handle(request)
    assert response["status"] == "intent_recorded"
    result = published(request, response)
    assert result["status"] == "captured"
    summary = capture.coverage_summary(config_for(request))
    assert summary["coverage_status"] == "covered"
    assert summary["captured_count"] == 1


def test_disabled_and_default_settings_create_no_output(tmp_path):
    path = tmp_path / "disabled.json"
    for fields in ({"enabled": False}, {}):
        write(path, {"schema": hook.SETTINGS_SCHEMA, **fields})
        assert hook.handle({"settings_path": str(path)}) == {"status": "disabled"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["disabled.json"]


def test_unknown_registry_is_never_coverage(tmp_path):
    request, _ = request_case(tmp_path, scope=None)
    response = hook.handle(request)
    assert published(request, response)["status"] == "captured"
    assert capture.coverage_summary(config_for(request))["coverage_status"] == "awaiting_coverage"


def test_retry_prepare_reuses_identical_open_intent(tmp_path):
    request, _ = request_case(tmp_path)
    first = hook.handle(request)
    second = hook.handle({**request, "observation_started_at": VERIFIED})
    assert first == second
    assert len(capture.read_capture(config_for(request))["records"]) == 2


def test_reconcile_recovers_unique_durable_intent(tmp_path):
    request, _ = request_case(tmp_path)
    response = hook.handle(request)
    result = published(request, mode="reobserve", previous_remote_head=request["entry"]["commit"])
    assert result["status"] == "captured"
    saved = capture.read_capture(config_for(request), result["capture_ref"])
    assert saved["remote_evidence"]["intent_ref"] == response["intent_ref"]
    assert saved["remote_evidence"]["method"] == "publisher-reconciled-fetch.v1"
    assert capture.coverage_summary(config_for(request))["coverage_status"] == "covered"


def test_known_exact_reobservation_preserves_covered_chain_and_earliest_receipt(tmp_path):
    request, _ = request_case(tmp_path)
    response = hook.handle(request)
    result = published(request, response)
    before = capture.read_capture(config_for(request))
    repeated = published(
        request,
        mode="reobserve",
        previous_remote_head=request["entry"]["commit"],
        verified_by=DAY + "T11:00:00+09:00",
    )
    assert repeated == {"status": "already_captured", "capture_ref": result["capture_ref"]}
    assert capture.read_capture(config_for(request)) == before
    assert capture.coverage_summary(config_for(request))["first_verified_by"] == VERIFIED


def test_post_push_first_hook_is_snapshot_and_no_intent_is_invented(tmp_path):
    request, _ = request_case(tmp_path)
    request.pop("observation_started_at")
    result = published(request, mode="reobserve")
    assert result["status"] == "captured"
    records = capture.read_capture(config_for(request))["records"]
    assert [r["kind"] for r in records] == ["session_started", "publication_captured"]
    assert (
        "missing_or_reused_publication_intent"
        in capture.coverage_summary(config_for(request))["reasons"]
    )


def test_changed_source_after_intent_records_failure_and_keeps_intent(tmp_path):
    request, _ = request_case(tmp_path)
    response = hook.handle(request)
    write(
        Path(request["entry"]["run_dir"]) / "coordinator-render.json",
        {"invocation_id": request["entry"]["invocation_id"], "new": True},
    )
    assert published(request, response) == {"status": "failed", "reason": "capture_error"}
    assert capture.read_capture(config_for(request))["records"][-1]["kind"] == "capture_failed"
    assert (
        "unfinished_publication_intent" in capture.coverage_summary(config_for(request))["reasons"]
    )


def test_existing_day_baseline_is_explicitly_pending(tmp_path):
    request, _ = request_case(tmp_path)
    request["previous_remote_head"] = request["entry"]["commit"]
    response = hook.handle(request)
    assert published(request, response)["status"] == "captured"
    assert (
        "baseline_existing_or_prior_history"
        in capture.coverage_summary(config_for(request))["reasons"]
    )


def test_helper_never_invokes_network_git_and_suppresses_partial_clone_fetch(tmp_path, monkeypatch):
    request, _ = request_case(tmp_path)
    original = subprocess.run
    calls = []

    def local_only(command, **kwargs):
        assert command[0] == "git"
        assert command[3] in {"ls-tree", "log", "show", "merge-base"}
        assert kwargs["env"]["GIT_NO_LAZY_FETCH"] == "1"
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0"
        calls.append(command[3])
        return original(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", local_only)
    response = hook.handle(request)
    assert published(request, response)["status"] == "captured"
    assert "show" in calls and "log" in calls


def test_entry_and_request_extra_fields_never_reach_output(tmp_path):
    request, _ = request_case(tmp_path)
    sentinel = "SYNTHETIC_MUST_NOT_SERIALIZE"
    request["entry"]["unrecognized_secret"] = sentinel
    request["config"]["ignored_credential"] = sentinel
    response = hook.handle(request)
    assert published(request, response)["status"] == "captured"
    root = Path(config_for(request)["feedback_capture"]["output_root"])
    assert all(sentinel.encode() not in p.read_bytes() for p in root.rglob("*") if p.is_file())


def test_cli_disabled_is_small_json_and_does_not_write_output(tmp_path):
    path = tmp_path / "disabled.json"
    write(path, {"schema": hook.SETTINGS_SCHEMA, "enabled": False})
    result = subprocess.run(
        [sys.executable, "-m", "modeler.feedback.publisher_capture"],
        input=json.dumps({"settings_path": str(path)}),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src")},
        check=False,
    )
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"status": "disabled"}
    assert not result.stderr


@pytest.mark.parametrize(
    "invalid", [None, [], {"settings_path": "/nonexistent/synthetic-settings.json"}]
)
def test_invalid_request_returns_no_exception_details(invalid):
    assert hook.handle(invalid) == {"status": "failed", "reason": "capture_error"}


def test_settings_symlink_is_rejected_without_opening_target(tmp_path):
    target = tmp_path / "target.json"
    write(target, {"schema": hook.SETTINGS_SCHEMA, "enabled": False})
    link = tmp_path / "settings.json"
    link.symlink_to(target)
    assert hook.handle({"settings_path": str(link)})["status"] == "failed"


def test_optional_explain_bytes_are_preserved_without_claiming_validity(tmp_path):
    request, _ = request_case(tmp_path)
    day = Path(request["entry"]["run_dir"])
    write(
        day / f"explain-{DAY}.json", {"schema": "synthetic-invalid-explain", "report_sha256": "bad"}
    )
    (day / "coordinator-explain.json").write_bytes(b"synthetic invalid JSON; raw only")
    response = hook.handle(request)
    result = published(request, response)
    assert result["status"] == "captured"
    saved = capture.read_capture(config_for(request), result["capture_ref"])
    paths = {s["path"] for s in saved["sources"]}
    assert f"explain-{DAY}.json" in paths
    assert "coordinator-explain.json" in paths


def test_other_day_head_advance_reobservation_preserves_original_coverage(tmp_path):
    request, _ = request_case(tmp_path)
    response = hook.handle(request)
    result = published(request, response)
    before = capture.read_capture(config_for(request))
    checkout = Path(request["config"]["checkout_dir"])
    (checkout / "synthetic-other-day-metadata.txt").write_text("another unit changes HEAD")
    git(checkout, "add", "synthetic-other-day-metadata.txt")
    git(
        checkout,
        "-c",
        "user.name=Synthetic",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "Synthetic other unit",
    )
    later_head = git(checkout, "rev-parse", "HEAD")
    request["entry"]["commit"] = later_head
    repeated = published(request, mode="reobserve", previous_remote_head=later_head)
    assert repeated == {"status": "already_captured", "capture_ref": result["capture_ref"]}
    assert capture.read_capture(config_for(request)) == before
    assert capture.coverage_summary(config_for(request))["coverage_status"] == "covered"


def test_settings_enabled_requires_boolean(tmp_path):
    request, _ = request_case(tmp_path)
    path = Path(request["settings_path"])
    value = json.loads(path.read_bytes())
    value["enabled"] = "false"
    write(path, value)
    assert hook.handle(request)["status"] == "failed"
    assert not Path(value["output_root"]).exists()


def test_subprocess_errors_are_sanitized(tmp_path, monkeypatch):
    request, _ = request_case(tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("synthetic", 1)

    monkeypatch.setattr(capture, "_git", timeout)
    assert hook.handle(request) == {"status": "failed", "reason": "capture_error"}


def test_unpinned_remote_is_masked_and_cannot_claim_reviewed_coverage(tmp_path):
    request, _ = request_case(tmp_path)
    sensitive_url = "file:///synthetic/private/must-not-persist"
    request["config"]["expected_remote_url"] = sensitive_url
    pending = hook.handle(request)
    result = published(request, pending)
    assert result["status"] == "captured"
    saved = capture.read_capture(config_for(request), result["capture_ref"])
    assert saved["producer_event"]["remote_identity"] == "synthetic-local"
    assert saved["producer_event"]["remote"] is None
    assert sensitive_url not in json.dumps({k: v for k, v in saved.items() if k != "artifacts"})
    assert saved["collector_sha256"]
    assert "capture.py" in saved["collector_files_sha256"]
    assert capture.coverage_summary(config_for(request))["coverage_status"] == "awaiting_coverage"
