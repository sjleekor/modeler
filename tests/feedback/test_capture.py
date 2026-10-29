"""Synthetic local git fixtures; capture never contacts any remote or data source."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from test_briefing import DAY, INVOCATION, fixture, git, publication, update_report, write

from modeler.feedback import capture
from modeler.feedback.core import FeedbackError, sha256_bytes

START = DAY + "T10:01:00+09:00"
INTENT = DAY + "T10:02:00+09:00"
VERIFIED = DAY + "T10:03:00+09:00"
SCOPE = {
    "reviewed": True,
    "registry_complete": True,
    "controlled_epoch": True,
    "registry_sha256": "e" * 64,
    "producers": ["reports-publisher"],
}


def setup(tmp_path, *, scope=SCOPE):
    day, report, digest = fixture(tmp_path)
    checkout = tmp_path / "reports"
    checkout.mkdir()
    git(checkout, "init")
    git(
        checkout,
        "-c",
        "user.name=Synthetic",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "Synthetic baseline",
    )
    baseline = git(checkout, "rev-parse", "HEAD")
    config = {
        "checkout_dir": str(checkout),
        "run_dir": str(day),
        "report_date": DAY,
        "feedback_capture": {
            "output_root": str(tmp_path / "capture"),
            "session_id": "synthetic-session",
        },
    }
    capture.begin_session(
        config,
        started_at=START,
        baseline_remote_head=baseline,
        day_unit_absent=True,
        no_prior_unit_history=True,
        producer_scope=scope,
    )
    publication(day, tmp_path, digest)
    entry = entry_for(day, checkout, digest)
    return config, entry, report, baseline


def entry_for(day, checkout, digest):
    manifest = json.loads((day / "markdown/manifest.json").read_bytes())
    return {
        "unit": DAY,
        "run_dir": str(day),
        "markdown_dir": str(day / "markdown"),
        "report_sha256": digest,
        "invocation_id": INVOCATION,
        "release": "synthetic-release",
        "unit_sha256": manifest["unit_sha256"],
        "commit": git(checkout, "rev-parse", "HEAD"),
    }


def intent(config, entry, baseline, *, started=INTENT):
    return capture.record_intent(
        config, entry, observation_started_at=started, previous_remote_head=baseline
    )


def publish(config, entry, baseline, *, pending=None, verified=VERIFIED, scope=SCOPE, **overrides):
    evidence = {
        "method": "publisher-verified-push.v1",
        "remote_head": entry["commit"],
        "previous_remote_head": baseline,
        "observation_started_at": INTENT,
        "intent_ref": pending["intent_ref"] if pending else None,
        **overrides,
    }
    return capture.capture_published(
        config, entry, entry["commit"], verified, baseline_evidence=evidence, producer_scope=scope
    )


def fingerprint(root):
    return {
        p.relative_to(root).as_posix(): sha256_bytes(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file()
    }


def test_controlled_capture_preserves_exact_sources_and_actual_git_unit(tmp_path):
    config, entry, _, baseline = setup(tmp_path)
    before = fingerprint(Path(entry["run_dir"]))
    head = git(Path(config["checkout_dir"]), "rev-parse", "HEAD")
    pending = intent(config, entry, baseline)
    result = publish(config, entry, baseline, pending=pending)
    saved = capture.read_capture(config, result["capture_ref"])
    report_source = next(s for s in saved["sources"] if s["path"] == f"report-{DAY}.json")
    assert (
        saved["artifacts"][report_source["sha256"]]
        == (Path(entry["run_dir"]) / f"report-{DAY}.json").read_bytes()
    )
    assert len([s for s in saved["sources"] if s["path"].startswith("git:")]) == 5
    assert saved["producer_event"]["status"] == "published"
    assert before == fingerprint(Path(entry["run_dir"]))
    assert head == git(Path(config["checkout_dir"]), "rev-parse", "HEAD")
    assert capture.coverage_summary(config)["coverage_status"] == "covered"


@pytest.mark.parametrize("basis", [None, "unknown", "snapshot_only"])
def test_unknown_basis_and_missing_intent_fail_closed(tmp_path, basis):
    config, entry, _, baseline = setup(tmp_path)
    result = publish(config, entry, baseline, method=basis)
    assert result["kind"] == "publication_captured"
    summary = capture.coverage_summary(config)
    assert summary["coverage_status"] == "awaiting_coverage"
    assert "missing_or_reused_publication_intent" in summary["reasons"]
    assert "snapshot_only_or_unknown_observation_basis" in summary["reasons"]


def test_default_registry_is_unreviewed(tmp_path):
    config, entry, _, baseline = setup(tmp_path, scope=None)
    pending = intent(config, entry, baseline)
    publish(config, entry, baseline, pending=pending, scope=None)
    assert (
        "producer_registry_unreviewed_or_incomplete" in capture.coverage_summary(config)["reasons"]
    )


def test_unfinished_intent_is_a_durable_gap(tmp_path):
    config, entry, _, baseline = setup(tmp_path)
    intent(config, entry, baseline)
    assert "unfinished_publication_intent" in capture.coverage_summary(config)["reasons"]
    capture.record_gap(config, "publisher_callback_unavailable", observed_at=VERIFIED)
    assert "collection_gap_or_failure" in capture.coverage_summary(config)["reasons"]


def test_first_report_retained_after_different_report_correction(tmp_path):
    config, entry, report, baseline = setup(tmp_path)
    pending = intent(config, entry, baseline)
    first = publish(config, entry, baseline, pending=pending)
    original = capture.read_capture(config, first["capture_ref"])
    report["markets"][0]["rankings"][0]["score"] = 0.75
    day = Path(entry["run_dir"])
    digest = update_report(day, report)
    publication(day, tmp_path, digest, revision=2)
    corrected = entry_for(day, Path(config["checkout_dir"]), digest)
    pending2 = intent(config, corrected, entry["commit"], started=DAY + "T10:04:00+09:00")
    publish(
        config,
        corrected,
        entry["commit"],
        pending=pending2,
        verified=DAY + "T10:06:00+09:00",
        observation_started_at=DAY + "T10:04:00+09:00",
    )
    summary = capture.coverage_summary(config)
    assert summary["first_receipt_ref"] == first["capture_ref"]
    assert summary["first_report_sha256"] == entry["report_sha256"]
    assert original == capture.read_capture(config, first["capture_ref"])
    assert summary["captured_count"] == 2
    assert summary["coverage_status"] == "covered"


def test_first_receipt_uses_datetime_not_append_order_or_timestamp_text(tmp_path):
    config, entry, _, baseline = setup(tmp_path)
    later = publish(config, entry, baseline, verified=DAY + "T11:10:00+09:00")
    earlier = publish(config, entry, entry["commit"], verified=DAY + "T02:05:00+00:00")
    summary = capture.coverage_summary(config)
    assert summary["receipt_refs_chronological"] == [earlier["capture_ref"], later["capture_ref"]]
    assert summary["first_verified_by"] == DAY + "T02:05:00+00:00"
    assert summary["coverage_status"] == "awaiting_coverage"


def test_revision_hash_is_actual_git_not_original_local_render_hash(tmp_path):
    config, entry, _, baseline = setup(tmp_path)
    day = Path(entry["run_dir"])
    publication(day, tmp_path, entry["report_sha256"], revision=2)
    corrected = entry_for(day, Path(config["checkout_dir"]), entry["report_sha256"])
    # Publisher's prepared manifest may retain the original rendered unit hash.
    write(
        day / "markdown/manifest.json",
        {
            **json.loads((day / "markdown/manifest.json").read_text()),
            "unit_sha256": entry["unit_sha256"],
        },
    )
    corrected["unit_sha256"] = entry["unit_sha256"]
    pending = intent(config, corrected, baseline)
    result = publish(config, corrected, baseline, pending=pending)
    assert result["revision"] == 2
    assert result["unit_sha256"] != result["original_render_unit_sha256"]
    assert result["original_render_unit_sha256"] == entry["unit_sha256"]


@pytest.mark.parametrize("change", ["report", "invocation", "missing"])
def test_bad_sources_record_failure_and_never_success(tmp_path, change):
    config, entry, report, baseline = setup(tmp_path)
    day = Path(entry["run_dir"])
    if change == "report":
        report["extra"] = "changed"
        write(day / f"report-{DAY}.json", report)
    elif change == "invocation":
        write(day / "run-state.json", {"invocation_id": "other"})
    else:
        (day / "coordinator-inference.json").unlink()
    with pytest.raises(FeedbackError, match="failure recorded"):
        publish(config, entry, baseline)
    assert capture.read_capture(config)["records"][-1]["kind"] == "capture_failed"
    assert capture.coverage_summary(config)["coverage_status"] == "awaiting_coverage"


def test_mutation_during_capture_is_detected(tmp_path, monkeypatch):
    config, entry, _, baseline = setup(tmp_path)
    original = capture._bound_unit

    def mutate(*args):
        result = original(*args)
        write(Path(entry["run_dir"]) / "run-state.json", {"invocation_id": "changed-during-read"})
        return result

    monkeypatch.setattr(capture, "_bound_unit", mutate)
    with pytest.raises(FeedbackError):
        publish(config, entry, baseline)
    assert capture.read_capture(config)["records"][-1]["kind"] == "capture_failed"


def test_mutation_after_intent_preserves_intended_bytes_and_refuses_receipt(tmp_path):
    config, entry, _, baseline = setup(tmp_path)
    pending = intent(config, entry, baseline)
    # Even an unbound optional sidecar change invalidates the before-push bundle.
    write(
        Path(entry["run_dir"]) / "coordinator-render.json",
        {"invocation_id": INVOCATION, "extra": "new"},
    )
    with pytest.raises(FeedbackError):
        publish(config, entry, baseline, pending=pending)
    assert capture.read_capture(config, pending["intent_ref"])["artifacts"]
    assert "unfinished_publication_intent" in capture.coverage_summary(config)["reasons"]


def test_crash_before_state_replace_exposes_no_partial_capture(tmp_path, monkeypatch):
    config, entry, _, baseline = setup(tmp_path)
    state = copy.deepcopy(capture.read_capture(config))
    original = capture._atomic

    def crash(path, raw):
        if path.name == "state.json":
            raise OSError("synthetic crash")
        return original(path, raw)

    monkeypatch.setattr(capture, "_atomic", crash)
    with pytest.raises(OSError):
        publish(config, entry, baseline)
    assert capture.read_capture(config) == state
    objects = Path(config["feedback_capture"]["output_root"]) / "objects"
    orphan = next(p.stem for p in objects.glob("*.json") if p.stem != state["state"]["head_ref"])
    with pytest.raises(FeedbackError, match="not in the committed session"):
        capture.read_capture(config, orphan)


def test_existing_unit_or_deleted_history_cannot_claim_absent_baseline(tmp_path):
    config, entry, _, _ = setup(tmp_path)
    other = copy.deepcopy(config)
    other["feedback_capture"]["session_id"] = "existing"
    with pytest.raises(FeedbackError, match="baseline assertion"):
        capture.begin_session(
            other,
            started_at=START,
            baseline_remote_head=entry["commit"],
            day_unit_absent=True,
            no_prior_unit_history=True,
        )
    capture.begin_session(
        other,
        started_at=START,
        baseline_remote_head=entry["commit"],
        day_unit_absent=False,
        no_prior_unit_history=False,
        producer_scope=SCOPE,
    )
    assert "baseline_existing_or_prior_history" in capture.coverage_summary(other)["reasons"]
    checkout = Path(config["checkout_dir"])
    git(checkout, "rm", "-r", f"reports/daily-briefing/2026/10/{DAY}")
    git(
        checkout,
        "-c",
        "user.name=Synthetic",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "Synthetic deletion",
    )
    deleted = git(checkout, "rev-parse", "HEAD")
    other["feedback_capture"]["session_id"] = "deleted"
    capture.begin_session(
        other,
        started_at=START,
        baseline_remote_head=deleted,
        day_unit_absent=True,
        no_prior_unit_history=False,
        producer_scope=SCOPE,
    )
    assert "baseline_existing_or_prior_history" in capture.coverage_summary(other)["reasons"]


def test_fixed_allowlist_does_not_open_unrelated_inputs(tmp_path, monkeypatch):
    config, entry, _, baseline = setup(tmp_path)
    day = Path(entry["run_dir"])
    for name in ("raw/private.json", "labels/secret.json", "prices.parquet", ".envrc"):
        path = day / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("do not read")
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name in {"private.json", "secret.json", "prices.parquet", ".envrc"}:
            raise AssertionError("forbidden source opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    publish(config, entry, baseline)
