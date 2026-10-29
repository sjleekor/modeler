"""Captured bytes survive daily overwrite, preserving receipt/revision lineage."""

import json
from pathlib import Path

import pytest
from test_briefing import DAY, NOW, publication
from test_capture import VERIFIED, intent, publish, setup

from modeler.feedback import capture
from modeler.feedback.captured import read_captured, session_config
from modeler.feedback.core import FeedbackError


def test_archive_adapter_never_reopens_overwritten_day_or_git_checkout(tmp_path):
    config, entry, report, baseline = setup(tmp_path)
    pending = intent(config, entry, baseline)
    result = publish(config, entry, baseline, pending=pending)
    before = read_captured(config, observed_at=NOW)
    assert not before.errors
    assert before.publications[0]["coverage_status"] == "covered"
    assert before.publications[0]["eligibility_status"] == "not_evaluated"
    assert before.publications[0]["receipt_ref"] == result["capture_ref"]
    assert before.predictions[0]["publication_status"] == "published_verified"
    Path(entry["run_dir"]).rename(tmp_path / "old-day")
    Path(config["checkout_dir"]).rename(tmp_path / "old-checkout")
    after = read_captured(config, observed_at=NOW)
    assert after.predictions == before.predictions
    assert after.publications == before.publications
    from modeler.feedback.runtime import _capture_completion

    _capture_completion(config, DAY)
    with pytest.raises(FeedbackError, match="missing"):
        _capture_completion(config, "2026-10-30")


def test_old_publication_sidecar_is_raw_evidence_not_new_producer_event(tmp_path):
    config, entry, _, baseline = setup(tmp_path)
    path = Path(entry["run_dir"]) / "coordinator-publication.json"
    path.write_text(json.dumps({"status": "failed", "report_sha256": "a" * 64}))
    pending = intent(config, entry, baseline)
    publish(config, entry, baseline, pending=pending)
    batch = read_captured(config, observed_at=NOW)
    assert not batch.errors
    assert batch.publications[0]["producer_event"]["status"] == "published"
    assert (
        json.loads(
            batch.artifacts[next(s["sha256"] for s in batch.sources if s["path"] == path.name)]
        )["status"]
        == "failed"
    )
    assert path.read_text() == json.dumps({"status": "failed", "report_sha256": "a" * 64})


def test_malformed_optional_explain_preserves_predictions_with_explicit_errors(tmp_path):
    config, entry, _, baseline = setup(tmp_path)
    day = Path(entry["run_dir"])
    (day / "coordinator-explain.json").write_bytes(b"invalid optional JSON")
    (day / f"explain-{DAY}.json").write_bytes(b"invalid optional explanation")
    pending = intent(config, entry, baseline)
    publish(config, entry, baseline, pending=pending)
    batch = read_captured(config, observed_at=NOW)
    assert len(batch.predictions) == len(batch.publications) == 1
    assert batch.predictions[0]["explain_sha256"] is None
    assert {e["source"] for e in batch.errors} == {
        "coordinator-explain.json",
        f"explain-{DAY}.json",
    }
    assert all(p["publication_status"] == "published_verified" for p in batch.predictions)


def test_corrections_keep_original_receipts_and_use_actual_revision_hash(tmp_path):
    config, entry, _, baseline = setup(tmp_path)
    pending = intent(config, entry, baseline)
    first = publish(config, entry, baseline, pending=pending)
    publication(Path(entry["run_dir"]), tmp_path, entry["report_sha256"], revision=2)
    from test_capture import entry_for

    corrected = entry_for(
        Path(entry["run_dir"]), Path(config["checkout_dir"]), entry["report_sha256"]
    )
    second_intent = intent(config, corrected, entry["commit"], started=DAY + "T10:04:00+09:00")
    second = publish(
        config,
        corrected,
        entry["commit"],
        pending=second_intent,
        observation_started_at=DAY + "T10:04:00+09:00",
        verified=DAY + "T10:05:00+09:00",
    )
    batch = read_captured(config, observed_at=NOW)
    assert not batch.errors
    assert [p["revision"] for p in batch.publications] == [1, 2]
    assert len({p["prediction_ref"] for p in batch.predictions}) == 1
    summary = capture.coverage_summary(config)
    assert summary["first_receipt_ref"] == first["capture_ref"]
    assert summary["receipt_refs_chronological"] == [first["capture_ref"], second["capture_ref"]]
    assert {p["verified_by"] for p in batch.publications} == {VERIFIED, DAY + "T10:05:00+09:00"}


def test_later_collection_gap_does_not_claim_eligible_first_report(tmp_path):
    config, entry, _, baseline = setup(tmp_path)
    pending = intent(config, entry, baseline)
    publish(config, entry, baseline, pending=pending)
    capture.record_gap(config, "synthetic interruption", observed_at=NOW)
    batch = read_captured(config, observed_at=NOW)
    assert batch.publications[0]["coverage_status"] == "awaiting_coverage"
    assert batch.publications[0]["eligibility_status"] == "not_evaluated"


def test_adapter_rejects_tampered_archived_bytes_before_index(tmp_path):
    config, entry, _, baseline = setup(tmp_path)
    pending = intent(config, entry, baseline)
    result = publish(config, entry, baseline, pending=pending)
    saved = capture.read_capture(config, result["capture_ref"])
    digest = next(s["sha256"] for s in saved["sources"] if s["path"] == f"report-{DAY}.json")
    (Path(config["feedback_capture"]["output_root"]) / "evidence" / f"{digest}.bin").write_bytes(
        b"{}"
    )
    with pytest.raises(FeedbackError, match="hash mismatch"):
        read_captured(config, observed_at=NOW)


def test_settings_route_same_epoch_and_day_to_same_session(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "schema": "prediction-feedback-capture-settings.v1",
                "enabled": True,
                "epoch_id": "synthetic",
                "output_root": str(tmp_path / "capture"),
            }
        )
    )
    assert session_config(path, DAY) == session_config(path, DAY)
    original = session_config(path, DAY)
    assert session_config(path, "2026-10-30") != original
