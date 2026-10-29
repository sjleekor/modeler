"""Explicit real-read plans; clock/scope checks precede every file open.

No queries, globbing, lake discovery, US/P6 fallback, or credential loading.
Synthetic validation uses in-memory fixtures and never calls this reader.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from modeler.feedback.core import canonical_bytes, require_hash, sha256_bytes

from .contracts import (
    BUNDLE_SHA256,
    MODEL_ID,
    MODEL_VERSION,
    REPORT_END,
    REPORT_START,
    RULE_ID,
    RULE_SHA256,
    OutcomeError,
    require_real_review,
    stamp,
)

FORBIDDEN = frozenset(
    {
        "us",
        "usa",
        "p6",
        "p6prime",
        "kr-next",
        "kr_next",
        "_03_kr_next",
        "forward",
        "checks",
        "labels",
        "holdout",
        "raw",
        "derived",
        "datasets",
        "points.parquet",
    }
)
READ_PLAN_SCHEMA = "kr-v1-outcome-read-plan.v1"
MAX_BYTES = 32 * 1024 * 1024


def explicit_path(value, *, root=None):
    if not isinstance(value, (str, Path)):
        raise OutcomeError("an explicit absolute input path is required")
    path = Path(value)
    if (
        not path.is_absolute()
        or path == Path(path.anchor)
        or ".." in path.parts
        or any(c in str(path) for c in "*?[]{}\n\r")
        or any(p.lower() in FORBIDDEN or p.startswith(".env") for p in path.parts)
    ):
        raise OutcomeError("input path is outside the registered KR scope")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise OutcomeError("input paths cannot traverse symlinks")
    if root is not None and path.parent != Path(root):
        raise OutcomeError("read plans accept direct explicitly named extracts only")
    return path


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise OutcomeError("duplicate read-plan or input fields")
        value[key] = item
    return value


def _read(path, expected_sha256=None, *, maximum=MAX_BYTES):
    # Walk every component with openat/O_NOFOLLOW. Path checks alone leave a
    # symlink replacement window before open and can expose unregistered bytes.
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    descriptor = None
    try:
        for component in path.parts[1:-1]:
            next_directory = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = next_directory
        before = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum
        ):
            raise OutcomeError("explicit input is missing, linked, or oversized")
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        opened = os.fstat(descriptor)

        def fields(info):
            return (
                info.st_dev,
                info.st_ino,
                info.st_mode,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )

        if fields(before) != fields(opened):
            raise OutcomeError("explicit input changed before open")
        raw = bytearray()
        while len(raw) <= maximum:
            chunk = os.read(descriptor, min(maximum + 1 - len(raw), 65536))
            if not chunk:
                break
            raw.extend(chunk)
        current = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if (
            len(raw) != before.st_size
            or fields(opened) != fields(os.fstat(descriptor))
            or fields(current) != fields(opened)
        ):
            raise OutcomeError("explicit input changed during read")
        raw = bytes(raw)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)
    if expected_sha256 is not None and sha256_bytes(raw) != require_hash(expected_sha256):
        raise OutcomeError("explicit input byte hash mismatch")
    value = json.loads(raw, object_pairs_hook=_object)
    if not isinstance(value, dict):
        raise OutcomeError("explicit inputs must be JSON objects")
    return value, sha256_bytes(raw)


@dataclass(frozen=True)
class ReadPermit:
    payload: bytes
    request_sha256: str
    authorized_at: str

    @property
    def request(self):
        return json.loads(self.payload)


def load_real_plan(path, *, now):
    """Reject before opening even the request file when the window is closed."""
    local = require_real_review(now)
    path = explicit_path(path)
    if not path.name.endswith("read-plan.json"):
        raise OutcomeError("a named read-plan.json is required")
    plan, digest = _read(path, maximum=1024 * 1024)
    _validate_plan(plan, local)
    return ReadPermit(canonical_bytes(plan), digest, local.isoformat())


def _validate_plan(plan, local):
    expected = {
        "schema": READ_PLAN_SCHEMA,
        "rule_id": RULE_ID,
        "rule_sha256": RULE_SHA256,
        "market": "KR",
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "bundle_sha256": BUNDLE_SHA256,
        "report_start": REPORT_START,
        "report_end": REPORT_END,
        "synthetic_fixture": False,
        "source_scope": "explicit-kr-v1-endpoints",
    }
    if not isinstance(plan, dict) or set(plan) != set(expected) | {
        "cutoff",
        "max_session",
        "source_root",
        "inputs",
        "adapter_code_sha256",
    }:
        raise OutcomeError("read plan has unregistered fields")
    if type(plan.get("synthetic_fixture")) is not bool or any(
        plan.get(k) != v for k, v in expected.items()
    ):
        raise OutcomeError("real read plan differs from the frozen scope")
    cutoff = stamp(plan.get("cutoff"))
    if cutoff > local or cutoff < stamp("2026-12-04T13:00:00+09:00"):
        raise OutcomeError("cutoff is outside the registered observation window")
    max_session = plan.get("max_session")
    if (
        not isinstance(max_session, str)
        or date.fromisoformat(max_session).isoformat() != max_session
    ):
        raise OutcomeError("max_session must be a canonical session date")
    if max_session > cutoff.astimezone(local.tzinfo).date().isoformat():
        raise OutcomeError("read plan includes future sessions")
    source_root = explicit_path(plan.get("source_root"))
    inputs = plan.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"calendar", "publications", "extract"}:
        raise OutcomeError("read plan must name exactly three registered inputs")
    paths = set()
    for role, entry in inputs.items():
        if not isinstance(entry, dict) or entry.get("role") != role:
            raise OutcomeError("read plan role mismatch")
        keys = {"role", "path", "sha256"} | (
            {"min_session", "max_session"} if role == "extract" else set()
        )
        if set(entry) != keys:
            raise OutcomeError("read plan input has unregistered fields")
        candidate = explicit_path(entry.get("path"), root=source_root)
        if candidate.suffix != ".json":
            raise OutcomeError("only registered prepared JSON extracts are accepted")
        require_hash(entry.get("sha256"))
        paths.add(candidate)
    if len(paths) != 3:
        raise OutcomeError("read plan roles must name distinct files")
    extract = inputs["extract"]
    if extract.get("max_session") != max_session:
        raise OutcomeError("extract scope differs from the read plan")
    if (
        not isinstance(extract.get("min_session"), str)
        or not REPORT_START <= extract["min_session"] <= max_session
        or date.fromisoformat(extract["min_session"]).isoformat() != extract["min_session"]
    ):
        raise OutcomeError("extract predates the prospective sample window")
    require_hash(plan.get("adapter_code_sha256"))
    if plan["adapter_code_sha256"] != sha256_bytes(
        Path(__file__).with_name("prices.py").read_bytes()
    ):
        raise OutcomeError("price adapter code pin mismatch")


def load_bound_input(permit, role, *, now):
    """Recheck the live clock and fixed scope before each bound input is opened."""
    local = require_real_review(now)
    if not isinstance(permit, ReadPermit) or role not in {"calendar", "publications", "extract"}:
        raise OutcomeError("registered read permit required")
    require_hash(permit.request_sha256)
    authorized = require_real_review(stamp(permit.authorized_at))
    if authorized > local:
        raise OutcomeError("read permit authorization has not arrived")
    request = permit.request
    _validate_plan(request, local)
    entry = request["inputs"][role]
    path = explicit_path(entry["path"], root=explicit_path(request["source_root"]))
    value, digest = _read(path, entry["sha256"])
    if role in {"publications", "extract"} and value.get("synthetic_fixture") is not False:
        raise OutcomeError("real reads require explicit non-fixture inputs")
    return value, {"role": role, "sha256": digest, "request_sha256": permit.request_sha256}


def permit_identity(permit):
    return sha256_bytes(canonical_bytes(permit.request))
