"""Prepare metadata for a future read; never authorize or inspect source paths.

``prepared_valid`` means only that supplied metadata matches the frozen plan
shape. It proves neither that files exist nor that their byte hashes are true.
Use the real access guard on the actual review day; this module issues no permit.
The CLI accepts JSON from stdin only and writes JSON to stdout only.
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from datetime import date
from pathlib import PurePosixPath
from types import MappingProxyType

from modeler.feedback.core import require_hash

from .access import FORBIDDEN, READ_PLAN_SCHEMA
from .contracts import (
    BUNDLE_SHA256,
    FIRST_REVIEW,
    MODEL_ID,
    MODEL_VERSION,
    REPORT_END,
    REPORT_START,
    RULE_ID,
    RULE_SHA256,
    SEOUL,
    OutcomeError,
    stamp,
)

PINS = MappingProxyType(
    {
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
)
VARIABLE_FIELDS = frozenset(
    {"cutoff", "max_session", "source_root", "inputs", "adapter_code_sha256"}
)
ROLES = frozenset({"calendar", "publications", "extract"})
REMAINING = (
    "actual Friday review clock gate (preparation cannot authorize access)",
    "source withdrawal/schema readiness and credentials, if required by the source owner",
    "actual source existence, symlink safety and source byte hashes at guarded read time",
    "current price adapter byte pin at guarded read time",
)


def _path(value, *, root=None):
    """Lexical counterpart of access.explicit_path; no filesystem calls."""
    if not isinstance(value, str):
        raise OutcomeError("an explicit absolute input path string is required")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or path == PurePosixPath(path.anchor)
        or ".." in path.parts
        or any(c in str(path) for c in "*?[]{}\n\r")
        or any(p.lower() in FORBIDDEN or p.startswith(".env") for p in path.parts)
    ):
        raise OutcomeError("input path is outside the registered KR scope")
    if root is not None and path.parent != root:
        raise OutcomeError("read plans accept direct explicitly named extracts only")
    return path


def _session(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise OutcomeError("session must be a canonical session date")
    return value


def _check(plan):
    # Keep this metadata-only mirror aligned with access._validate_plan. That
    # validator also checks a live clock, filesystem symlinks and local code
    # bytes, so calling it here would cross the preparation boundary.
    if not isinstance(plan, dict) or set(plan) != set(PINS) | VARIABLE_FIELDS:
        raise OutcomeError("read plan has unregistered or missing fields")
    if type(plan.get("synthetic_fixture")) is not bool or any(
        plan.get(k) != v for k, v in PINS.items()
    ):
        raise OutcomeError("real read plan differs from the frozen scope")
    cutoff = stamp(plan["cutoff"])
    if cutoff < FIRST_REVIEW:
        raise OutcomeError("cutoff is outside the registered observation window")
    max_session = _session(plan["max_session"])
    if max_session > cutoff.astimezone(SEOUL).date().isoformat():
        raise OutcomeError("read plan includes future sessions")
    source_root = _path(plan["source_root"])
    inputs = plan["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != ROLES:
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
        candidate = _path(entry["path"], root=source_root)
        if candidate.suffix != ".json":
            raise OutcomeError("only registered prepared JSON extracts are accepted")
        require_hash(entry["sha256"])
        paths.add(candidate)
    if len(paths) != 3:
        raise OutcomeError("read plan roles must name distinct files")
    extract = inputs["extract"]
    if extract["max_session"] != max_session:
        raise OutcomeError("extract scope differs from the read plan")
    if not REPORT_START <= _session(extract["min_session"]) <= max_session:
        raise OutcomeError("extract predates the prospective sample window")
    require_hash(plan["adapter_code_sha256"])


def _missing(plan):
    """List placeholders without implying that present values are verified."""
    if not isinstance(plan, dict):
        return ["plan"]
    missing = [key for key in sorted(set(PINS) | VARIABLE_FIELDS) if plan.get(key) is None]
    inputs = plan.get("inputs")
    if isinstance(inputs, dict):
        for role in sorted(ROLES):
            entry = inputs.get(role)
            if not isinstance(entry, dict):
                missing.append(f"inputs.{role}")
                continue
            keys = {"role", "path", "sha256"} | (
                {"min_session", "max_session"} if role == "extract" else set()
            )
            missing.extend(f"inputs.{role}.{key}" for key in sorted(keys) if entry.get(key) is None)
    return missing


def validate_preparation(plan):
    """Return metadata diagnostics without reading, probing or authorizing files.

    Future cutoff metadata is allowed. There is intentionally no ``now`` or
    ``observed_at`` argument: a preparation result cannot supply a review clock.
    Supplied digest strings are format checked only, never treated as verified
    source bytes or as proof of the current adapter implementation.
    """
    errors = []
    try:
        _check(plan)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        errors.append(str(exc))
    return {
        "prepared_valid": not errors,
        "authorized": False,
        "real_ready": False,
        "validation_scope": "metadata-only; supplied byte hashes remain unverified",
        "errors": errors,
        "missing": _missing(plan),
        "remaining": list(REMAINING),
    }


def template():
    """Return an explicitly incomplete preparation envelope, never a permit."""
    plan = {
        **PINS,
        "cutoff": None,
        "max_session": None,
        "source_root": None,
        "adapter_code_sha256": None,
        "inputs": {role: {"role": role, "path": None, "sha256": None} for role in sorted(ROLES)},
    }
    plan["inputs"]["extract"].update(min_session=None, max_session=None)
    return {"plan": plan, **validate_preparation(plan)}


def prepare_read_plan(*, source_root, bindings, cutoff, max_session, adapter_code_sha256):
    """Build the exact read schema from caller-supplied metadata, with no I/O.

    The caller must supply source byte hashes and the current adapter code hash.
    No paths are discovered, opened, stat'ed or resolved. A returned plan still
    requires the existing real-read guard and its actual review clock.
    """
    plan = {
        **PINS,
        "source_root": source_root,
        "inputs": deepcopy(bindings),
        "cutoff": cutoff,
        "max_session": max_session,
        "adapter_code_sha256": adapter_code_sha256,
    }
    diagnostics = validate_preparation(plan)
    if not diagnostics["prepared_valid"]:
        raise OutcomeError("; ".join(diagnostics["errors"]))
    return plan


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise OutcomeError("duplicate metadata fields")
        result[key] = value
    return result


def main(argv=None):
    """Read only explicitly piped JSON metadata; never accept input file paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("template", "validate", "create"))
    args = parser.parse_args(argv)
    try:
        if args.command == "template":
            result = template()
        else:
            raw = sys.stdin.read(1024 * 1024 + 1)
            if len(raw) > 1024 * 1024:
                raise OutcomeError("metadata exceeds the 1 MiB limit")
            value = json.loads(raw, object_pairs_hook=_object)
            if args.command == "validate":
                result = validate_preparation(value)
            else:
                expected = {
                    "source_root",
                    "bindings",
                    "cutoff",
                    "max_session",
                    "adapter_code_sha256",
                }
                if not isinstance(value, dict) or set(value) != expected:
                    raise OutcomeError("create requires exactly the preparation metadata fields")
                plan = prepare_read_plan(**value)
                result = {"plan": plan, **validate_preparation(plan)}
    except (ValueError, TypeError, KeyError) as exc:
        result = {**validate_preparation(None), "errors": [str(exc)]}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if result["prepared_valid"] or args.command == "template" else 2


if __name__ == "__main__":
    raise SystemExit(main())
