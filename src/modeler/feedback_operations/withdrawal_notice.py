"""PROPOSED explicit public withdrawal bytes and local unpublished preparation.

This module does not publish, capture, verify remotely, collect coverage, produce
outcome events, or authorize real reads. Caller metadata and bytes stay local.
Only fresh dedicated /private/tmp directories may hold preparation artifacts.
"""

from __future__ import annotations

import json
import os
import re
import stat
from copy import deepcopy
from datetime import date
from pathlib import Path

from modeler.feedback.core import FeedbackError, canonical_bytes, require_hash, sha256_bytes
from modeler.feedback_operations.credential_launcher import (
    LauncherError,
)
from modeler.feedback_operations.credential_launcher import (
    _open_directory as _secure_directory,
)
from modeler.feedback_operations.credential_launcher import (
    _read_regular as _secure_read,
)
from modeler.feedback_outcomes.contracts import (
    BUNDLE_SHA256,
    MODEL_ID,
    MODEL_VERSION,
    REPORT_END,
    REPORT_START,
    SEOUL,
    stamp,
)
from modeler.feedback_outcomes.withdrawals import BINDING_KEYS

NOTICE_SCHEMA = "kr-v1-public-withdrawal-notice-proposal.v1"
PREPARATION_SCHEMA = "kr-v1-unpublished-withdrawal-preparation.v1"
MAX_BYTES = 64 * 1024
_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})\Z")
_STATE = {
    "protocol_status": "proposal",
    "publication_status": "unpublished",
    "published": False,
    "captured": False,
    "authorized": False,
    "real_ready": False,
    "absence_claim": "none",
}
_ROOT_MARKER = canonical_bytes({"schema": PREPARATION_SCHEMA, **_STATE})


def _shape(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise FeedbackError(f"invalid {name} schema")
    return value


def _timestamp(value):
    if not isinstance(value, str) or not _TIME.fullmatch(value):
        raise FeedbackError("timestamp must use RFC3339 seconds and timezone")
    return stamp(value)


def _notice_id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise FeedbackError("invalid withdrawal notice id")
    return value


def _binding(value):
    value = _shape(value, BINDING_KEYS, "original binding")
    for key in ("publication_ref", "report_sha256", "bundle_hash", "input_sha256"):
        require_hash(value[key])
    if (value["market"], value["model_id"], value["model_version"], value["bundle_hash"]) != (
        "KR",
        MODEL_ID,
        MODEL_VERSION,
        BUNDLE_SHA256,
    ):
        raise FeedbackError("original binding is outside the frozen KR model")
    invocation = value["invocation_id"]
    if (
        not isinstance(invocation, str)
        or not 1 <= len(invocation) <= 256
        or invocation.strip() != invocation
        or any(ord(char) < 32 or ord(char) == 127 for char in invocation)
    ):
        raise FeedbackError("invalid original invocation")
    for key in ("feature_asof_date", "report_date"):
        try:
            valid = (
                isinstance(value[key], str)
                and date.fromisoformat(value[key]).isoformat() == value[key]
            )
        except ValueError:
            valid = False
        if not valid:
            raise FeedbackError("invalid original date")
    if not (
        REPORT_START <= value["report_date"] <= REPORT_END
        and value["feature_asof_date"] < value["report_date"]
        and _timestamp(value["decision_at"]).astimezone(SEOUL).date().isoformat()
        == value["report_date"]
    ):
        raise FeedbackError("original binding is outside prospective date scope")
    return value


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FeedbackError("duplicate notice metadata key")
        result[key] = value
    return result


def _reject_constant(value):
    raise FeedbackError(f"nonfinite JSON constant: {value}")


def _json(raw):
    if type(raw) is not bytes or not 0 < len(raw) <= MAX_BYTES:
        raise FeedbackError("bounded in-memory bytes required")
    try:
        return json.loads(
            raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_reject_constant
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise FeedbackError("invalid strict withdrawal JSON") from exc


def produce_notice(
    original_binding: dict,
    *,
    notice_id: str,
    decision_at: str,
    created_at: str,
    synthetic_fixture: bool,
) -> bytes:
    """Build deterministic proposal bytes solely from exact caller metadata.

    ``synthetic_fixture=False`` is allowed for future reviewed preparation. It
    proves neither the supplied binding nor publication, capture or readiness.
    No input source is opened. This is a complete-original withdrawal only.
    """
    _binding(original_binding)
    _notice_id(notice_id)
    if type(synthetic_fixture) is not bool:
        raise FeedbackError("explicit fixture flag required")
    if (
        not _timestamp(original_binding["decision_at"])
        <= _timestamp(decision_at)
        <= _timestamp(created_at)
    ):
        raise FeedbackError("withdrawal decision/creation time is out of order")
    body = {
        "schema": NOTICE_SCHEMA,
        "protocol_status": "proposal",
        "action": "withdraw",
        "scope": "complete_original_prediction",
        "notice_id": notice_id,
        "original_binding": deepcopy(original_binding),
        "decision_at": decision_at,
        "created_at": created_at,
        "synthetic_fixture": synthetic_fixture,
    }
    raw = canonical_bytes({**body, "notice_ref": sha256_bytes(canonical_bytes(body))})
    parse_notice(raw, original_binding=original_binding)
    return raw


def parse_notice(raw: bytes, *, original_binding: dict) -> dict:
    """Strictly check bytes, self-hash and equality to caller's original binding.

    Equality and hashes are local consistency checks, never authentication or a
    remote receipt. Whitespace is preserved for the separate source byte hash.
    """
    expected = _binding(original_binding)
    notice = _shape(
        _json(raw),
        {
            "schema",
            "protocol_status",
            "action",
            "scope",
            "notice_id",
            "original_binding",
            "decision_at",
            "created_at",
            "synthetic_fixture",
            "notice_ref",
        },
        "withdrawal notice",
    )
    if (
        notice["schema"] != NOTICE_SCHEMA
        or notice["protocol_status"] != "proposal"
        or notice["action"] != "withdraw"
        or notice["scope"] != "complete_original_prediction"
        or type(notice["synthetic_fixture"]) is not bool
    ):
        raise FeedbackError("explicit complete-original withdrawal proposal required")
    _notice_id(notice["notice_id"])
    if _binding(notice["original_binding"]) != expected:
        raise FeedbackError("withdrawal original binding mismatch")
    if (
        not _timestamp(expected["decision_at"])
        <= _timestamp(notice["decision_at"])
        <= _timestamp(notice["created_at"])
    ):
        raise FeedbackError("withdrawal decision/creation time is out of order")
    body = {key: value for key, value in notice.items() if key != "notice_ref"}
    if require_hash(notice["notice_ref"]) != sha256_bytes(canonical_bytes(body)):
        raise FeedbackError("withdrawal notice hash mismatch")
    return deepcopy(notice)


def _root_path(value):
    raw = os.fspath(value)
    if not isinstance(raw, str) or not re.fullmatch(
        r"/private/tmp/withdrawal-preparation-[a-zA-Z0-9_-]+", raw
    ):
        raise FeedbackError("fresh /private/tmp/withdrawal-preparation-* root required")
    return Path(raw)


def _open_directory(*args, **kwargs):
    try:
        return _secure_directory(*args, **kwargs)
    except LauncherError as exc:
        raise FeedbackError("unsafe withdrawal preparation directory") from exc


def _read_regular(path, maximum, **kwargs):
    try:
        return _secure_read(path, maximum, **kwargs)
    except LauncherError as exc:
        raise FeedbackError("unsafe withdrawal preparation file") from exc


def _create_file(directory, name, raw):
    descriptor = os.open(
        name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=directory
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def create_preparation_root(path) -> Path:
    """Exclusively create a fresh private preparation directory; never reuse it."""
    root = _root_path(path)
    parent = _open_directory(root.parent, trusted=True)
    try:
        os.mkdir(root.name, mode=0o700, dir_fd=parent)
        directory = os.open(root.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        try:
            _create_file(directory, "proposal.json", _ROOT_MARKER)
            os.fsync(directory)
        finally:
            os.close(directory)
        os.fsync(parent)
    finally:
        os.close(parent)
    return root


def _open_root(root):
    root = _root_path(root)
    directory = _open_directory(root, trusted=True, trust_root=root)
    try:
        info = os.fstat(directory)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
            raise FeedbackError("private preparation root protection required")
        if (
            _read_regular(root / "proposal.json", MAX_BYTES, protected=True, trusted=True)
            != _ROOT_MARKER
        ):
            raise FeedbackError("invalid unpublished proposal root marker")
    except BaseException:
        os.close(directory)
        raise
    return root, directory


def _entry(root, notice_id, original_binding):
    path = root / _notice_id(notice_id)
    directory = _open_directory(path, trusted=True, trust_root=root)
    try:
        if set(os.listdir(directory)) != {"notice.json", "preparation.json"}:
            raise FeedbackError("incomplete or unexpected preparation inventory")
        raw = _read_regular(path / "notice.json", MAX_BYTES, protected=True, trusted=True)
        notice = parse_notice(raw, original_binding=original_binding)
        manifest_raw = _read_regular(
            path / "preparation.json", MAX_BYTES, protected=True, trusted=True
        )
        manifest = _shape(
            _json(manifest_raw),
            {
                "schema",
                *_STATE,
                "notice_id",
                "notice_ref",
                "source_sha256",
                "original_binding_ref",
                "synthetic_fixture",
                "prepared_at",
                "producer_code_ref",
                "preparation_ref",
            },
            "preparation",
        )
        body = {key: value for key, value in manifest.items() if key != "preparation_ref"}
        if (
            manifest_raw != canonical_bytes(manifest)
            or manifest["schema"] != PREPARATION_SCHEMA
            or any(
                manifest[key] != value or type(manifest[key]) is not type(value)
                for key, value in _STATE.items()
            )
            or manifest["notice_id"] != notice_id
            or notice["notice_id"] != notice_id
            or manifest["notice_ref"] != notice["notice_ref"]
            or manifest["source_sha256"] != sha256_bytes(raw)
            or manifest["original_binding_ref"] != sha256_bytes(canonical_bytes(original_binding))
            or type(manifest["synthetic_fixture"]) is not bool
            or manifest["synthetic_fixture"] != notice["synthetic_fixture"]
            or require_hash(manifest["preparation_ref"]) != sha256_bytes(canonical_bytes(body))
        ):
            raise FeedbackError("unpublished preparation integrity mismatch")
        require_hash(manifest["producer_code_ref"], length=40)
        if _timestamp(manifest["prepared_at"]) < _timestamp(notice["created_at"]):
            raise FeedbackError("preparation precedes notice creation")
        return raw, manifest
    finally:
        os.close(directory)


def stage_notice(
    root, raw: bytes, *, original_binding: dict, prepared_at: str, producer_code_ref: str
) -> dict:
    """Write one immutable journal entry. Same bytes/metadata are idempotent.

    The code ref is a caller assertion for later review, not a verified pin.
    Collision or interrupted/partial writes fail closed; no repair/overwrite is
    attempted. Successful entries contain only two bounded regular files.
    """
    notice = parse_notice(raw, original_binding=original_binding)
    if _timestamp(prepared_at) < _timestamp(notice["created_at"]):
        raise FeedbackError("preparation precedes notice creation")
    require_hash(producer_code_ref, length=40)
    body = {
        "schema": PREPARATION_SCHEMA,
        **_STATE,
        "notice_id": notice["notice_id"],
        "notice_ref": notice["notice_ref"],
        "source_sha256": sha256_bytes(raw),
        "original_binding_ref": sha256_bytes(canonical_bytes(original_binding)),
        "synthetic_fixture": notice["synthetic_fixture"],
        "prepared_at": prepared_at,
        "producer_code_ref": producer_code_ref,
    }
    manifest = {**body, "preparation_ref": sha256_bytes(canonical_bytes(body))}
    root, directory = _open_root(root)
    try:
        try:
            os.mkdir(notice["notice_id"], mode=0o700, dir_fd=directory)
        except FileExistsError:
            existing_raw, existing = _entry(root, notice["notice_id"], original_binding)
            if existing_raw != raw or existing != manifest:
                raise FeedbackError("immutable withdrawal preparation collision")
            return {**deepcopy(existing), "status": "unchanged"}
        child = os.open(
            notice["notice_id"], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
        )
        try:
            _create_file(child, "notice.json", raw)
            _create_file(child, "preparation.json", canonical_bytes(manifest))
            os.fsync(child)
        finally:
            os.close(child)
        os.fsync(directory)
        return {**deepcopy(manifest), "status": "prepared"}
    finally:
        os.close(directory)


def export_staged_notice(root, notice_id: str, *, original_binding: dict) -> dict:
    """Return exact locally staged bytes for review, with no publication claim."""
    root, directory = _open_root(root)
    try:
        raw, manifest = _entry(root, notice_id, original_binding)
        return {**deepcopy(manifest), "status": "exported_for_review", "notice_bytes": raw}
    finally:
        os.close(directory)
