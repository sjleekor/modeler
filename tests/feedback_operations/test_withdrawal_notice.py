"""Generated metadata only; no reports, prices, credentials or remote reads."""

import json
import os
import shutil
import uuid
from copy import deepcopy
from pathlib import Path

import pytest

from modeler.feedback.core import FeedbackError, canonical_bytes, sha256_bytes
from modeler.feedback_operations import withdrawal_notice as notices
from modeler.feedback_outcomes.contracts import BUNDLE_SHA256, MODEL_ID, MODEL_VERSION

DECISION = "2026-11-04T13:00:00+09:00"
CREATED = "2026-11-04T13:01:00+09:00"
PREPARED = "2026-11-04T13:02:00+09:00"
CODE_REF = sha256_bytes(b"invented-test-code-commit")[:40]


@pytest.fixture
def binding():
    return {
        "publication_ref": sha256_bytes(b"invented-original-publication"),
        "report_sha256": sha256_bytes(b"invented-original-report"),
        "invocation_id": "generated-test-invocation",
        "market": "KR",
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "bundle_hash": BUNDLE_SHA256,
        "input_sha256": sha256_bytes(b"invented-original-input"),
        "feature_asof_date": "2026-11-03",
        "report_date": "2026-11-04",
        "decision_at": "2026-11-04T10:00:00+09:00",
    }


@pytest.fixture
def root():
    path = Path("/private/tmp") / ("withdrawal-preparation-test-" + uuid.uuid4().hex)
    result = notices.create_preparation_root(path)
    try:
        yield result
    finally:
        shutil.rmtree(path)


def produce(binding, **kwargs):
    return notices.produce_notice(
        binding,
        notice_id=kwargs.pop("notice_id", "generated-notice-1"),
        decision_at=kwargs.pop("decision_at", DECISION),
        created_at=kwargs.pop("created_at", CREATED),
        synthetic_fixture=kwargs.pop("synthetic_fixture", True),
        **kwargs,
    )


def stage(root, raw, binding, **kwargs):
    return notices.stage_notice(
        root,
        raw,
        original_binding=binding,
        prepared_at=kwargs.pop("prepared_at", PREPARED),
        producer_code_ref=kwargs.pop("producer_code_ref", CODE_REF),
        **kwargs,
    )


def reseal(value):
    body = {key: item for key, item in value.items() if key != "notice_ref"}
    return canonical_bytes({**body, "notice_ref": sha256_bytes(canonical_bytes(body))})


def assert_unpublished(result):
    assert result["protocol_status"] == "proposal"
    assert result["publication_status"] == "unpublished"
    for field in ("published", "captured", "authorized", "real_ready"):
        assert result[field] is False
    assert result["absence_claim"] == "none"
    assert "source_bound_v1" not in repr(result)
    assert not {"verified_by", "remote_verified", "receipt", "coverage"} & result.keys()


def test_deterministic_pure_roundtrip_binding_and_two_distinct_hashes(binding, monkeypatch):
    original = deepcopy(binding)
    monkeypatch.setattr(Path, "read_bytes", lambda *_: pytest.fail("producer must not read input"))
    raw = produce(binding)
    assert produce(binding) == raw
    parsed = notices.parse_notice(raw, original_binding=binding)
    assert parsed["original_binding"] == original == binding
    assert parsed["action"] == "withdraw"
    assert parsed["scope"] == "complete_original_prediction"
    assert parsed["notice_ref"] != sha256_bytes(raw)
    assert parsed["synthetic_fixture"] is True
    assert "verified_by" not in parsed
    whitespace_raw = b" \n" + raw + b"\n"
    assert notices.parse_notice(whitespace_raw, original_binding=binding) == parsed
    assert sha256_bytes(whitespace_raw) != sha256_bytes(raw)


@pytest.mark.parametrize("field", list(notices.BINDING_KEYS))
def test_exact_original_binding_is_required_for_every_field(binding, field):
    raw = produce(binding)
    expected = deepcopy(binding)
    expected[field] = "different"
    with pytest.raises(FeedbackError):
        notices.parse_notice(raw, original_binding=expected)


@pytest.mark.parametrize(
    "field,value",
    [
        ("action", "correct"),
        ("scope", "selected_slots"),
        ("scope", "no_withdrawal"),
        ("schema", "future.v2"),
        ("protocol_status", "approved"),
        ("synthetic_fixture", 0),
        ("synthetic_fixture", "false"),
        ("notice_id", "../escape"),
        ("notice_id", "a/b"),
        ("notice_id", ""),
        ("notice_id", "a" * 81),
        ("decision_at", "2026-11-04T13:00:00"),
        ("decision_at", "20261104T130000+0900"),
        ("decision_at", "2026-11-04T13:00:00+99:00"),
        ("decision_at", "2026-11-04T09:59:59+09:00"),
        ("created_at", "2026-11-04T12:59:59+09:00"),
        ("created_at", "2026-11-31T13:01:00+09:00"),
    ],
)
def test_strict_declaration_and_time_scope(binding, field, value):
    payload = json.loads(produce(binding))
    payload[field] = value
    with pytest.raises(FeedbackError):
        notices.parse_notice(reseal(payload), original_binding=binding)


@pytest.mark.parametrize(
    "field,value",
    [
        ("report_date", "2026-11-03"),
        ("report_date", "2027-01-30"),
        ("feature_asof_date", "2026-11-04"),
        ("feature_asof_date", "2026-99-01"),
        ("feature_asof_date", "20261103"),
        ("decision_at", "2026-11-03T23:59:59+09:00"),
        ("market", "US"),
        ("invocation_id", ""),
        ("invocation_id", "has\ncontrol"),
        ("invocation_id", " outer space "),
        ("invocation_id", "a" * 257),
    ],
)
def test_producer_refuses_out_of_scope_or_incomplete_original(binding, field, value):
    binding[field] = value
    with pytest.raises(FeedbackError):
        produce(binding)


@pytest.mark.parametrize("where", ["top", "binding"])
def test_unknown_or_missing_fields_rejected_even_with_new_self_hash(binding, where):
    for missing in (True, False):
        payload = json.loads(produce(binding))
        target = payload if where == "top" else payload["original_binding"]
        if missing:
            target.pop("scope" if where == "top" else "invocation_id")
        else:
            target["unknown"] = "x"
        with pytest.raises(FeedbackError):
            notices.parse_notice(reseal(payload), original_binding=binding)


@pytest.mark.parametrize(
    "bad", [b"{}", b"[]", b"null", b"\xff", b"{", b" " * (notices.MAX_BYTES + 1), "not bytes"]
)
def test_bounded_strict_json(binding, bad):
    with pytest.raises(FeedbackError):
        notices.parse_notice(bad, original_binding=binding)


@pytest.mark.parametrize(
    "fragment",
    [b'"extra":NaN,', b'"extra":Infinity,', b'"extra":-Infinity,', b'"action":"withdraw",'],
)
def test_duplicate_keys_and_nonfinite_values_rejected(binding, fragment):
    raw = produce(binding)
    with pytest.raises(FeedbackError):
        notices.parse_notice(b"{" + fragment + raw[1:], original_binding=binding)


def test_nested_duplicate_key_and_tamper_rejected(binding):
    raw = produce(binding)
    with pytest.raises(FeedbackError):
        notices.parse_notice(
            raw.replace(b'"market":"KR"', b'"market":"KR","market":"KR"'), original_binding=binding
        )
    with pytest.raises(FeedbackError, match="hash mismatch"):
        notices.parse_notice(
            raw.replace(b"generated-notice-1", b"generated-notice-2"), original_binding=binding
        )


@pytest.mark.parametrize("fixture_flag", [True, False])
def test_staging_and_export_are_immutable_idempotent_unpublished(root, binding, fixture_flag):
    # False exercises future producer metadata using generated bytes only.
    raw = b" \n" + produce(binding, synthetic_fixture=fixture_flag) + b"\n"
    result = stage(root, raw, binding)
    assert result["status"] == "prepared"
    assert result["source_sha256"] == sha256_bytes(raw)
    assert result["synthetic_fixture"] is fixture_flag
    assert_unpublished(result)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino) for path in files
    }
    second = stage(root, raw, binding)
    assert second == {**result, "status": "unchanged"}
    exported = notices.export_staged_notice(root, "generated-notice-1", original_binding=binding)
    assert exported["notice_bytes"] == raw
    assert_unpublished(exported)
    assert {
        path: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ino) for path in files
    } == before
    assert {path.name for path in root.iterdir()} == {"proposal.json", "generated-notice-1"}


@pytest.mark.parametrize("change", ["bytes", "timestamp", "code"])
def test_collision_does_not_overwrite(root, binding, change):
    raw = produce(binding)
    stage(root, raw, binding)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    altered_raw = raw + b"\n" if change == "bytes" else raw
    kwargs = {"prepared_at": "2026-11-04T13:03:00+09:00"} if change == "timestamp" else {}
    if change == "code":
        kwargs["producer_code_ref"] = sha256_bytes(b"another-invented-code-commit")[:40]
    with pytest.raises(FeedbackError, match="collision"):
        stage(root, altered_raw, binding, **kwargs)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("filename", ["notice.json", "preparation.json", "proposal.json"])
def test_tamper_refuses_export_and_repeat_without_overwrite(root, binding, filename):
    raw = produce(binding)
    stage(root, raw, binding)
    path = (
        root / filename if filename == "proposal.json" else root / "generated-notice-1" / filename
    )
    path.write_bytes(path.read_bytes().replace(b"proposal", b"tampered"))
    changed = path.read_bytes()
    with pytest.raises((FeedbackError, OSError)):
        notices.export_staged_notice(root, "generated-notice-1", original_binding=binding)
    with pytest.raises((FeedbackError, OSError)):
        stage(root, raw, binding)
    assert path.read_bytes() == changed


def test_partial_or_extra_inventory_refuses_reuse(root, binding):
    (root / "generated-notice-1").mkdir(mode=0o700)
    with pytest.raises(FeedbackError, match="inventory"):
        stage(root, produce(binding), binding)
    assert list((root / "generated-notice-1").iterdir()) == []


@pytest.mark.parametrize(
    "path",
    [
        "/tmp/withdrawal-preparation-a",
        "/private/tmp/withdrawal-preparation-a/../b",
        "/private/tmp/withdrawal-preparation-a/sub",
        "relative",
        "/private/tmp/other",
    ],
)
def test_no_traversal_or_other_output_root(path):
    with pytest.raises(FeedbackError):
        notices.create_preparation_root(path)


def test_root_must_be_fresh_and_private(root, binding):
    with pytest.raises(FileExistsError):
        notices.create_preparation_root(root)
    root.chmod(0o770)
    try:
        with pytest.raises((FeedbackError, OSError)):
            stage(root, produce(binding), binding)
    finally:
        root.chmod(0o700)


@pytest.mark.parametrize("target", ["entry", "notice", "manifest", "marker"])
def test_symlink_cannot_redirect_export_or_stage(root, binding, target):
    raw = produce(binding)
    stage(root, raw, binding)
    entry = root / "generated-notice-1"
    paths = {
        "entry": entry,
        "notice": entry / "notice.json",
        "manifest": entry / "preparation.json",
        "marker": root / "proposal.json",
    }
    path = paths[target]
    saved = root / "saved-for-symlink-test"
    path.rename(saved)
    path.symlink_to(saved, target_is_directory=target == "entry")
    with pytest.raises((FeedbackError, OSError)):
        notices.export_staged_notice(root, "generated-notice-1", original_binding=binding)
    with pytest.raises((FeedbackError, OSError)):
        stage(root, raw, binding)


def test_hardlink_and_unsafe_id_refused(root, binding):
    stage(root, produce(binding), binding)
    os.link(root / "generated-notice-1" / "notice.json", root / "hardlink-copy")
    with pytest.raises((FeedbackError, OSError)):
        notices.export_staged_notice(root, "generated-notice-1", original_binding=binding)
    with pytest.raises(FeedbackError):
        notices.export_staged_notice(root, "../escape", original_binding=binding)


def test_invalid_preparation_metadata_writes_nothing(root, binding):
    raw = produce(binding)
    with pytest.raises(FeedbackError):
        stage(root, raw, binding, prepared_at="2026-11-04T13:00:00+09:00")
    with pytest.raises(FeedbackError):
        stage(root, raw, binding, producer_code_ref="unverified")
    assert [path.name for path in root.iterdir()] == ["proposal.json"]


@pytest.mark.parametrize("field", ["published", "captured", "authorized", "real_ready"])
def test_even_rehashed_manifest_cannot_claim_publication_or_authority(root, binding, field):
    stage(root, produce(binding), binding)
    path = root / "generated-notice-1" / "preparation.json"
    manifest = json.loads(path.read_bytes())
    manifest[field] = True
    body = {key: value for key, value in manifest.items() if key != "preparation_ref"}
    manifest["preparation_ref"] = sha256_bytes(canonical_bytes(body))
    path.write_bytes(canonical_bytes(manifest))
    with pytest.raises(FeedbackError, match="integrity"):
        notices.export_staged_notice(root, "generated-notice-1", original_binding=binding)


def test_root_symlink_cannot_redirect_stage(root, binding):
    alias = root.with_name(root.name + "-alias")
    alias.symlink_to(root, target_is_directory=True)
    try:
        with pytest.raises((FeedbackError, OSError)):
            stage(alias, produce(binding), binding)
        assert [path.name for path in root.iterdir()] == ["proposal.json"]
    finally:
        alias.unlink()


def test_failed_write_leaves_closed_entry_and_never_overwrites(root, binding, monkeypatch):
    create = notices._create_file

    def fail_manifest(directory, name, raw):
        if name == "preparation.json":
            raise OSError("injected local preparation write failure")
        return create(directory, name, raw)

    raw = produce(binding)
    monkeypatch.setattr(notices, "_create_file", fail_manifest)
    with pytest.raises(OSError, match="injected"):
        stage(root, raw, binding)
    monkeypatch.setattr(notices, "_create_file", create)
    with pytest.raises(FeedbackError, match="inventory"):
        stage(root, raw, binding)
    assert (root / "generated-notice-1" / "notice.json").read_bytes() == raw
    assert not (root / "generated-notice-1" / "preparation.json").exists()
