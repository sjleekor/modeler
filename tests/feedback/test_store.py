from __future__ import annotations

import fcntl
import hashlib
from pathlib import Path

import pytest

from modeler.feedback import store
from modeler.feedback.core import Batch, FeedbackError, identity, read_json, sha256_bytes
from modeler.feedback.issues import compile_mapping

NOW = "2026-10-30T01:00:00+00:00"
LATER = "2026-10-31T01:00:00+00:00"
DOC = (
    "| issue_id | 문제·분류 / 대상 | reason_code |\n|---|---|---|\n"
    "| KR-FB-0001 | kr_daily_h20_v1 | W20_halt_in_window |\n"
).encode()


def mapping():
    blob = hashlib.sha1(b"blob " + str(len(DOC)).encode() + b"\0" + DOC).hexdigest()
    return compile_mapping(DOC, my_commit="a" * 40, expected_blob_sha1=blob)


def prediction(report="a", at=NOW):
    return {
        "prediction_ref": identity("test", {"report": report}),
        "kind": "briefing",
        "market": "KR",
        "security_market": "KOSPI",
        "model_id": "kr_daily_h20_v1",
        "symbol": "000001",
        "rank": 1,
        "score": 0.5,
        "report_date": "2026-10-30",
        "feature_asof_date": "2026-10-29",
        "report_sha256": report * 64,
        "quality_reasons": ["W20_halt_in_window"],
        "checker_version": "test.v1",
        "source_evidence_sha256": report * 64,
        "observed_at": at,
        "evaluation_status": "pending",
        "publication_status": "published_evidence",
    }


def commit(tmp_path, batch, *, at=NOW):
    return store.commit_batch(
        tmp_path / "out",
        batch,
        mapping=mapping(),
        source_key="day-1",
        source_roots=[tmp_path / "source"],
        observed_at=at,
        issue_document_ref="a" * 40,
    )


def publication(prediction_row, revision):
    return {
        "publication_ref": identity("publication", {"revision": revision}),
        "unit_binding_verified": True,
        "revision": revision,
        "prediction_refs": [prediction_row["prediction_ref"]],
        "status": "published_evidence",
        "observed_at": NOW,
    }


def test_rerun_does_not_duplicate_and_retains_first_observation(tmp_path):
    first = commit(tmp_path, Batch(predictions=[prediction()]))
    second = commit(tmp_path, Batch(predictions=[prediction(at=LATER)]), at=LATER)
    index = store.load_index(tmp_path / "out")
    assert len(index["predictions"]) == len(index["occurrences"]) == 1
    assert index["predictions"][0]["first_observed_at"] == NOW
    assert (tmp_path / "out/runs" / first["run_id"] / "manifest.json").exists()
    assert (tmp_path / "out/runs" / second["run_id"] / "manifest.json").exists()


def test_correction_lineage_survives_old_source_replay(tmp_path):
    first, second = prediction("a"), prediction("b")
    commit(tmp_path, Batch(predictions=[first], publications=[publication(first, 1)]))
    commit(tmp_path, Batch(predictions=[second], publications=[publication(second, 2)]))
    commit(tmp_path, Batch(predictions=[prediction("a")]))
    rows = {r["report_sha256"]: r for r in store.load_index(tmp_path / "out")["predictions"]}
    assert rows["b" * 64]["supersedes"] == [rows["a" * 64]["prediction_ref"]]
    assert not rows["a" * 64].get("supersedes")


def test_reverse_backfill_uses_revision_not_connection_order(tmp_path):
    first, second = prediction("a"), prediction("b")
    commit(tmp_path, Batch(predictions=[second], publications=[publication(second, 2)]))
    commit(tmp_path, Batch(predictions=[first], publications=[publication(first, 1)]))
    rows = {r["report_sha256"]: r for r in store.load_index(tmp_path / "out")["predictions"]}
    assert rows["b" * 64]["supersedes"] == [rows["a" * 64]["prediction_ref"]]
    assert not rows["a" * 64].get("supersedes")


def test_no_revision_evidence_never_invents_order(tmp_path):
    commit(tmp_path, Batch(predictions=[prediction("b")]))
    commit(tmp_path, Batch(predictions=[prediction("a")]))
    rows = store.load_index(tmp_path / "out")["predictions"]
    assert all(not r.get("supersedes") for r in rows)
    assert all(len(r["related_predictions"]) == 1 for r in rows)


def test_prior_publication_fact_survives_missing_new_evidence(tmp_path):
    row = prediction()
    commit(tmp_path, Batch(predictions=[row], publications=[publication(row, 1)]))
    missing = {**row, "publication_status": "publication_unverified"}
    missing_pub = {**publication(row, 1), "status": "publication_unverified"}
    commit(
        tmp_path,
        Batch(
            predictions=[missing], publications=[missing_pub], errors=[{"reason": "source_missing"}]
        ),
    )
    index = store.load_index(tmp_path / "out")
    assert index["publications"][0]["status"] == "published_evidence"
    assert index["publications"][0]["last_link_status"] == "publication_unverified"
    assert index["predictions"][0]["source_status"] == "source_incomplete"


def test_failure_before_pointer_keeps_previous_generation(tmp_path, monkeypatch):
    commit(tmp_path, Batch(predictions=[prediction()]))
    before = store.load_index(tmp_path / "out")
    atomic = store._atomic

    def fail_pointer(path, raw):
        if path.name == "current.json":
            raise OSError("interrupted")
        return atomic(path, raw)

    monkeypatch.setattr(store, "_atomic", fail_pointer)
    with pytest.raises(OSError, match="interrupted"):
        commit(tmp_path, Batch(predictions=[prediction("b")]))
    assert store.load_index(tmp_path / "out") == before


def test_two_writers_do_not_publish_partial_index(tmp_path):
    commit(tmp_path, Batch(predictions=[prediction()]))
    with (tmp_path / "out/.write.lock").open("a+b") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(FeedbackError, match="locked"):
            commit(tmp_path, Batch(predictions=[prediction("b")]))
    assert len(store.load_index(tmp_path / "out")["predictions"]) == 1


def test_missing_source_keeps_original_reference_and_reason(tmp_path):
    commit(tmp_path, Batch(predictions=[prediction()]))
    commit(tmp_path, Batch(errors=[{"source": "report.json", "reason": "source_missing"}]))
    index = store.load_index(tmp_path / "out")
    assert len(index["predictions"]) == 1
    assert index["source_states"]["day-1"]["errors"][0]["reason"] == "source_missing"
    assert index["predictions"][0]["source_status"] == "source_incomplete"


def test_source_output_overlap_rejected_before_writing(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(FeedbackError, match="disjoint"):
        store.commit_batch(
            source / "output",
            Batch(),
            mapping=mapping(),
            source_key="day",
            source_roots=[source],
            observed_at=NOW,
            issue_document_ref="doc",
        )
    assert not (source / "output").exists()


def test_evidence_corruption_does_not_replace_current_index(tmp_path):
    raw = b'{"evidence":1}'
    digest = sha256_bytes(raw)
    commit(tmp_path, Batch(artifacts={digest: raw}))
    evidence = tmp_path / f"out/evidence/{digest}.bin"
    evidence.write_bytes(b"changed")
    with pytest.raises(FeedbackError, match="conflict"):
        commit(tmp_path, Batch(artifacts={digest: raw}))


@pytest.mark.parametrize("name", ["forward/check.json", "checks/day.json", "labels/data.json"])
def test_forbidden_sources_rejected_before_open(tmp_path, name, monkeypatch):
    source = tmp_path / name
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path == source:
            pytest.fail("forbidden file opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(FeedbackError, match="allowlist"):
        read_json(source, root=tmp_path, batch=Batch())


def test_symlink_input_and_output_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "report.json").write_text("{}")
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    with pytest.raises(FeedbackError, match="symlink"):
        read_json(linked / "report.json", root=tmp_path, batch=Batch())
    with pytest.raises(FeedbackError, match="symlink"):
        store.check_output_root(linked / "out", [])


def test_forbidden_root_itself_cannot_bypass_guard(tmp_path):
    blocked = tmp_path / "forward/checks"
    blocked.mkdir(parents=True)
    source = blocked / "2026-10-30.json"
    source.write_text('{"sealed_return":123}')
    with pytest.raises(FeedbackError, match="allowlist"):
        read_json(source, root=blocked, batch=Batch())
