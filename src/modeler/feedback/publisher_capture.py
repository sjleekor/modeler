"""Optional stdlib-only publisher hook. JSON stdin/stdout, local objects only.

An enabled settings file is deliberately separate from the existing publisher
config and journal. Every error is a small result; the publisher remains free to
finish its own successful publication. Durable intents expose interrupted hooks
as awaiting coverage when the collector can next read its output.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from . import capture
from .core import Batch, FeedbackError, canonical_bytes, read_json, require_hash, sha256_bytes

SETTINGS_SCHEMA = "prediction-feedback-capture-settings.v1"
MAX_REQUEST_BYTES = 1024 * 1024
_ENTRY_FIELDS = (
    "run_dir",
    "markdown_dir",
    "unit_sha256",
    "report_sha256",
    "invocation_id",
    "release",
    "commit",
    "committed_source_sha256",
    "synthetic_fixture",
)


def _request(request: dict):
    if not isinstance(request, dict):
        raise FeedbackError("hook request must be an object")
    path = Path(request["settings_path"])
    if not path.is_absolute():
        raise FeedbackError("hook settings path must be absolute")
    settings = read_json(path, root=path.parent, batch=Batch())
    if settings.get("schema") != SETTINGS_SCHEMA:
        raise FeedbackError("invalid publisher capture settings schema")
    if type(settings.get("enabled", False)) is not bool:
        raise FeedbackError("capture enabled must be a boolean")
    if settings.get("enabled") is not True:
        return None
    epoch = settings.get("epoch_id")
    if not isinstance(epoch, str) or not epoch or len(epoch) > 128:
        raise FeedbackError("invalid capture epoch")
    unit = capture._unit(request.get("unit"))
    entry = request["entry"]
    if not isinstance(entry, dict):
        raise FeedbackError("invalid publisher entry")
    entry = {key: entry[key] for key in _ENTRY_FIELDS if key in entry}
    entry["unit"] = unit
    require_hash(entry.get("commit"), length=40)
    checkout = request["config"]["checkout_dir"]
    if not isinstance(checkout, str) or not Path(checkout).is_absolute():
        raise FeedbackError("capture checkout must be absolute")
    session = sha256_bytes(canonical_bytes({"epoch_id": epoch, "unit": unit}))
    config = {
        "checkout_dir": checkout,
        "expected_remote_url": request["config"].get("expected_remote_url"),
        "branch": request["config"].get("branch"),
        "report_date": unit,
        "run_dir": entry["run_dir"],
        "feedback_capture": {"output_root": settings["output_root"], "session_id": session},
    }
    scope = capture._scope(settings.get("producer_scope"))
    if config["expected_remote_url"] != capture.REMOTE or config["branch"] != "main":
        scope["reviewed"] = False
    return settings, config, entry, unit, scope


def _existing(config: dict) -> list[dict] | None:
    root, session = capture._settings(config)
    path = capture._state_path(root, session)
    if path.exists() or path.is_symlink():
        return capture.read_capture(config)["records"]
    return None


def _matching(record: dict, entry: dict, *, intent=False, commit=True) -> bool:
    return all(
        record.get(key) == entry.get(key)
        for key in ("unit", "report_sha256", "invocation_id", "release")
    ) and (
        not commit
        or record.get("intended_commit" if intent else "reports_commit") == entry.get("commit")
    )


def _open_intent(records: list[dict], entry: dict, *, previous_head=None) -> dict | None:
    closed = {
        r.get("remote_evidence", {}).get("intent_ref")
        for r in records
        if r["kind"] == "publication_captured"
    }
    matches = [
        r
        for r in records
        if r["kind"] == "publication_intent"
        and r["capture_ref"] not in closed
        and _matching(r, entry, intent=True)
        and (previous_head is None or r.get("previous_remote_head") == previous_head)
    ]
    return matches[0] if len(matches) == 1 else None


def _intent_response(record: dict) -> dict:
    return {
        "status": "intent_recorded",
        "intent_ref": record["capture_ref"],
        "observation_started_at": record["observed_at"],
        "previous_remote_head": record["previous_remote_head"],
    }


def _begin(config: dict, unit: str, head: str, scope: dict, started: str):
    checkout = Path(config["checkout_dir"])
    path = capture._unit_path(unit)
    absent = not bool(capture._git(checkout, "ls-tree", "-r", head, "--", path))
    no_history = not bool(
        capture._git(checkout, "log", "--full-history", "-1", "--format=%H", head, "--", path)
    )
    capture.begin_session(
        config,
        started_at=started,
        baseline_remote_head=head,
        day_unit_absent=absent,
        no_prior_unit_history=no_history,
        producer_scope=scope,
    )


def _handle(request: dict) -> dict:
    parsed = _request(request)
    if parsed is None:
        return {"status": "disabled"}
    _, config, entry, unit, scope = parsed
    mode = request.get("mode")
    if mode not in {"prepare", "published", "reobserve"}:
        raise FeedbackError("unknown publisher hook mode")
    previous = require_hash(request.get("previous_remote_head"), length=40)
    started = (
        request.get("observation_started_at")
        or (request.get("verified_by") if mode != "prepare" else None)
        or capture._now()
    )
    capture.require_timestamp(started)
    records = _existing(config)
    if records is None:
        _begin(config, unit, previous, scope, started)
        records = capture.read_capture(config)["records"]
    if mode == "prepare":
        pending = _open_intent(records, entry, previous_head=previous)
        if pending:
            # A retry reuses only exactly the same currently present source bytes.
            batch = Batch()
            capture._snapshot(entry, unit, batch)
            expected = [s for s in pending["sources"] if not s["path"].startswith("git:")]
            if batch.sources == expected:
                return _intent_response(pending)
        pending = capture.record_intent(
            config, entry, observation_started_at=started, previous_remote_head=previous
        )
        return _intent_response(pending)
    remote = require_hash(request.get("remote_head"), length=40)
    verified = capture.require_timestamp(request.get("verified_by"))
    # Reobservations of a committed exact capture preserve its original evidence.
    known = [
        r
        for r in records
        if r["kind"] == "publication_captured" and _matching(r, entry, commit=False)
    ]
    if known:
        checkout = Path(config["checkout_dir"])
        digest, bound = capture._bound_unit(checkout, remote, unit, entry, Batch())
        matching = []
        for record in known:
            if record["unit_sha256"] != digest or record["revision"] != bound[0]:
                continue
            try:
                capture._git(
                    checkout, "merge-base", "--is-ancestor", record["reports_commit"], remote
                )
            except FeedbackError:
                continue
            matching.append(record)
        if matching:
            earliest = min(matching, key=lambda r: (capture._time(r["verified_by"]), r["sequence"]))
            return {"status": "already_captured", "capture_ref": earliest["capture_ref"]}
    supplied = request.get("intent")
    intent_ref = None
    intent_previous = previous
    if supplied is not None:
        if not isinstance(supplied, dict):
            raise FeedbackError("invalid publisher hook intent")
        intent_ref = require_hash(supplied.get("intent_ref"))
        pending = capture.read_capture(config, intent_ref)
        if pending.get("kind") != "publication_intent" or not _matching(
            pending, entry, intent=True
        ):
            raise FeedbackError("publisher hook intent binding mismatch")
        started = pending["observed_at"]
        intent_previous = pending["previous_remote_head"]
    elif mode == "reobserve":
        pending = _open_intent(records, entry)
        if pending:
            intent_ref = pending["capture_ref"]
            started = pending["observed_at"]
            intent_previous = pending["previous_remote_head"]
    # A hook first seen after push remains snapshot-only: no intent is invented.
    result = capture.capture_published(
        config,
        entry,
        entry["commit"],
        verified,
        baseline_evidence={
            "remote_head": remote,
            "previous_remote_head": intent_previous,
            "observation_started_at": started,
            "method": (
                "publisher-reconciled-fetch.v1"
                if mode == "reobserve"
                else "publisher-verified-push.v1"
            ),
            "intent_ref": intent_ref,
        },
        producer_scope=scope,
    )
    return {"status": "captured", "capture_ref": result["capture_ref"]}


def handle(request: dict) -> dict:
    """Return a small sanitized result; never print configs, inputs, or exceptions."""
    try:
        return _handle(request)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError, RecursionError):
        return {"status": "failed", "reason": "capture_error"}


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise FeedbackError("hook request exceeds size limit")
        result = handle(json.loads(raw))
    except (OSError, ValueError, TypeError, RecursionError):
        result = {"status": "failed", "reason": "invalid_request"}
    sys.stdout.write(canonical_bytes(result).decode("utf-8") + "\n")
    return 1 if result["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
