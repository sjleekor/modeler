"""Bounded publisher-boundary captures; no remote, index, prices, or source writes.

The publisher holds its own lock when calling :func:`capture_published`. This
module takes only a lock in its separate output root. A producer registry is an
explicit reviewed assertion, never inferred from observing one writer. Evidence
and chain records are immutable; state.json is the sole last-step commit point.
Unreferenced objects left by a crash are not observations.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, date, datetime
from pathlib import Path

from modeler.reporting.markdown import parse_front_matter

from .briefing import _UNIT_FILES
from .core import (
    MAX_METADATA_BYTES,
    Batch,
    FeedbackError,
    canonical_bytes,
    read_bytes,
    require_hash,
    require_timestamp,
    sha256_bytes,
    source_path,
)
from .receipt import REMOTE
from .store import _atomic, _immutable, _writer

SCHEMA = "prediction-feedback-producer-capture.v1"
METHODS = frozenset({"publisher-verified-push.v1", "publisher-reconciled-fetch.v1"})
_SIDES = (
    "run-state.json",
    "coordinator-inference.json",
    "coordinator-render.json",
    "reports-publisher-config.json",
    "coordinator-publication.json",
    "coordinator-explain.json",
)


def _time(value: str) -> datetime:
    return datetime.fromisoformat(require_timestamp(value).replace("Z", "+00:00"))


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _unit(value: str) -> str:
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise FeedbackError("capture unit must be an ISO date")
    return value


def _unit_path(unit: str) -> str:
    return f"reports/daily-briefing/{unit[:4]}/{unit[5:7]}/{unit}"


def _settings(config: dict, entry: dict | None = None) -> tuple[Path, str]:
    settings = config.get("feedback_capture")
    if not isinstance(settings, dict):
        raise FeedbackError("feedback_capture configuration is required")
    session = settings.get("session_id")
    if not isinstance(session, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", session):
        raise FeedbackError("invalid capture session id")
    raw = settings.get("output_root")
    if not isinstance(raw, str) or not Path(raw).is_absolute():
        raise FeedbackError("capture output_root must be absolute")
    root = Path(raw)
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise FeedbackError("capture output cannot traverse symlinks")
    for owner in (config, entry or {}):
        for key in ("run_dir", "markdown_dir", "checkout_dir"):
            if owner.get(key):
                source = Path(owner[key]).resolve()
                if root.resolve().is_relative_to(source) or source.is_relative_to(root.resolve()):
                    raise FeedbackError("capture output and source roots must be disjoint")
    # Inspect only fixed output paths, never scan source directories or a full index.
    for relative in ("objects", "evidence", "sessions", f"sessions/{session}", ".write.lock"):
        if (root / relative).is_symlink():
            raise FeedbackError("capture output symlink is not allowed")
    return root, session


def _state_path(root: Path, session: str) -> Path:
    return root / "sessions" / session / "state.json"


def _load(root: Path, session: str) -> dict:
    path = _state_path(root, session)
    if path.is_symlink():
        raise FeedbackError("capture state symlink is not allowed")
    try:
        state = json.loads(path.read_bytes())
    except FileNotFoundError as exc:
        raise FeedbackError("capture session has not started") from exc
    if state.get("schema") != SCHEMA or state.get("session_id") != session:
        raise FeedbackError("invalid capture state")
    require_hash(state.get("head_ref"))
    if type(state.get("sequence")) is not int or state["sequence"] < 0:
        raise FeedbackError("invalid capture sequence")
    return state


def _object(root: Path, ref: str) -> dict:
    ref = require_hash(ref)
    path = root / "objects" / f"{ref}.json"
    if path.is_symlink():
        raise FeedbackError("capture record symlink is not allowed")
    raw = path.read_bytes()
    if sha256_bytes(raw) != ref:
        raise FeedbackError("capture record hash mismatch")
    result = json.loads(raw)
    if result.get("schema") != SCHEMA:
        raise FeedbackError("invalid capture record schema")
    return result


def _collector() -> dict:
    files = {
        path.name: sha256_bytes(path.read_bytes())
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }
    return {
        "collector_sha256": sha256_bytes(canonical_bytes(files)),
        "collector_files_sha256": files,
    }


def _producer_identity(config: dict) -> dict:
    expected = config.get("expected_remote_url")
    if expected == REMOTE and config.get("branch", "main") == "main":
        return {"remote_identity": "pinned-reports", "remote": REMOTE, "branch": "main"}
    return {
        "remote_identity": "synthetic-local" if expected is not None else "unknown",
        "remote": None,
        "branch": None,
    }


def _append(root: Path, session: str, previous: dict | None, record: dict, artifacts=None) -> dict:
    record = {
        "schema": SCHEMA,
        "session_id": session,
        "sequence": previous["sequence"] + 1 if previous else 0,
        "previous_ref": previous["head_ref"] if previous else None,
        **_collector(),
        **record,
    }
    for digest, raw in (artifacts or {}).items():
        if sha256_bytes(raw) != require_hash(digest):
            raise FeedbackError("capture evidence hash mismatch")
        _immutable(root / "evidence" / f"{digest}.bin", raw)
    raw = canonical_bytes(record)
    ref = sha256_bytes(raw)
    _immutable(root / "objects" / f"{ref}.json", raw)
    state = {
        "schema": SCHEMA,
        "session_id": session,
        "sequence": record["sequence"],
        "head_ref": ref,
    }
    _atomic(_state_path(root, session), canonical_bytes(state))
    return {**record, "capture_ref": ref}


def _git(checkout: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), *args],
            capture_output=True,
            check=True,
            timeout=10,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise FeedbackError("capture local git verification failed") from exc
    return result.stdout


def _scope(value: dict | None) -> dict:
    """Copy a small explicit registry assertion, not arbitrary config/secrets."""
    value = value or {}
    if not isinstance(value, dict):
        raise FeedbackError("invalid capture producer scope")
    producers = value.get("producers", [])
    if not isinstance(producers, list) or not all(isinstance(p, str) and p for p in producers):
        raise FeedbackError("invalid capture producer registry")
    reviewed = value.get("reviewed") is True
    digest = value.get("registry_sha256")
    if digest is not None:
        require_hash(digest)
    return {
        "reviewed": reviewed,
        "registry_complete": value.get("registry_complete") is True,
        "controlled_epoch": value.get("controlled_epoch") is True,
        "registry_sha256": digest,
        "producers": sorted(set(producers)),
    }


def begin_session(
    config: dict,
    *,
    started_at: str,
    baseline_remote_head: str,
    day_unit_absent: bool,
    no_prior_unit_history: bool,
    producer_scope: dict | None = None,
    unit: str | None = None,
) -> dict:
    """Record a dated collection boundary, locally checking the supplied remote head.

    The caller must have freshly observed baseline_remote_head remotely. The local
    object checks prove absence/history at that head, not freshness of the remote
    assertion. ``producer_scope`` defaults to unreviewed, hence awaiting_coverage.
    """
    root, session = _settings(config)
    unit = _unit(unit or config.get("report_date"))
    require_timestamp(started_at)
    require_hash(baseline_remote_head, length=40)
    scope = _scope(producer_scope)
    checkout = Path(config["checkout_dir"])
    path = _unit_path(unit)
    absent = not bool(_git(checkout, "ls-tree", "-r", baseline_remote_head, "--", path))
    no_history = not bool(
        _git(
            checkout, "log", "--full-history", "-1", "--format=%H", baseline_remote_head, "--", path
        )
    )
    if type(day_unit_absent) is not bool or type(no_prior_unit_history) is not bool:
        raise FeedbackError("capture baseline absence assertions must be booleans")
    if day_unit_absent != absent or no_prior_unit_history != no_history:
        raise FeedbackError("capture baseline assertion does not match local git evidence")
    with _writer(root):
        if _state_path(root, session).exists():
            raise FeedbackError("capture session already exists")
        return _append(
            root,
            session,
            None,
            {
                "kind": "session_started",
                "observed_at": started_at,
                "started_at": started_at,
                "unit": unit,
                "baseline_remote_head": baseline_remote_head,
                "day_unit_absent": absent,
                "no_prior_unit_history": no_history,
                "producer_scope": scope,
                "baseline_basis": "caller_remote_observation_and_local_git_history",
            },
        )


def record_gap(config: dict, reason: str, *, observed_at: str, unit: str | None = None) -> dict:
    """Preserve a known unobserved/failing interval; a later success cannot erase it."""
    root, session = _settings(config)
    require_timestamp(observed_at)
    if not isinstance(reason, str) or not reason or len(reason) > 256:
        raise FeedbackError("capture gap requires a bounded reason code")
    if unit is not None:
        _unit(unit)
    with _writer(root):
        previous = _load(root, session)
        return _append(
            root,
            session,
            previous,
            {"kind": "gap", "reason": reason, "observed_at": observed_at, "unit": unit},
        )


def _snapshot(entry: dict, unit: str, batch: Batch) -> tuple[dict, dict[str, dict]]:
    day = Path(entry["run_dir"])
    raw = read_bytes(day / f"report-{unit}.json", root=day, batch=batch)
    report = json.loads(raw)
    if not isinstance(report, dict) or report.get("report_date") != unit:
        raise FeedbackError("capture report date mismatch")
    if sha256_bytes(raw) != require_hash(entry.get("report_sha256")):
        raise FeedbackError("capture report hash mismatch")
    sides = {}
    for name in _SIDES:
        path = source_path(day / name, root=day)
        if path.exists():
            side_raw = read_bytes(path, root=day, batch=batch)
            if name != "coordinator-explain.json":
                sides[name] = json.loads(side_raw)
        elif name in {"run-state.json", "coordinator-inference.json"}:
            raise FeedbackError("capture required producer metadata missing")
    explain_path = source_path(day / f"explain-{unit}.json", root=day)
    if explain_path.exists():
        read_bytes(explain_path, root=day, batch=batch)
    markdown = Path(entry.get("markdown_dir", day / "markdown"))
    raw = read_bytes(markdown / "manifest.json", root=markdown, batch=batch)
    manifest = json.loads(raw)
    # Prefix the separate fixed manifest path to distinguish it from day sidecars.
    batch.sources[-1]["path"] = "markdown/manifest.json"
    invocation = entry.get("invocation_id")
    if not isinstance(invocation, str) or not invocation:
        raise FeedbackError("capture invocation missing")
    if report.get("invocation_id", invocation) != invocation:
        raise FeedbackError("capture report invocation mismatch")
    for name in ("run-state.json", "coordinator-inference.json", "coordinator-render.json"):
        if name not in sides:
            continue
        item = sides[name]
        if not isinstance(item, dict) or item.get("invocation_id") != invocation:
            raise FeedbackError("capture producer invocation mismatch")
        if item.get("report_date", unit) != unit:
            raise FeedbackError("capture producer date mismatch")
        if item.get("report_sha256", entry["report_sha256"]) != entry["report_sha256"]:
            raise FeedbackError("capture producer report hash mismatch")
    inference = sides["coordinator-inference.json"]
    if inference.get("report_sha256") != entry["report_sha256"]:
        raise FeedbackError("capture inference does not bind report bytes")
    for key, expected in (
        ("unit", unit),
        ("report_sha256", entry["report_sha256"]),
        ("invocation_id", invocation),
        ("release", entry.get("release")),
        ("unit_sha256", entry.get("unit_sha256")),
    ):
        if not isinstance(manifest, dict) or manifest.get(key) != expected:
            raise FeedbackError("capture publisher manifest binding mismatch")
    return report, sides


def _evidence_metadata(value: dict) -> dict:
    if not isinstance(value, dict):
        raise FeedbackError("capture remote evidence must be an object")
    result = {
        key: value.get(key)
        for key in (
            "remote_head",
            "previous_remote_head",
            "observation_started_at",
            "method",
            "intent_ref",
        )
    }
    require_hash(result["remote_head"], length=40)
    if result["previous_remote_head"] is not None:
        require_hash(result["previous_remote_head"], length=40)
    if result["observation_started_at"] is not None:
        require_timestamp(result["observation_started_at"])
    if result["intent_ref"] is not None:
        require_hash(result["intent_ref"])
    return result


def _bound_unit(checkout: Path, commit: str, unit: str, entry: dict, batch: Batch):
    """Hash the actual immutable unit, including revision-dependent front matter."""
    path = _unit_path(unit)
    tree = _git(checkout, "ls-tree", "-rlz", commit, "--", path + "/")
    entries = []
    for item in tree.split(b"\0"):
        if not item:
            continue
        metadata, raw_path = item.split(b"\t", 1)
        mode, kind, _, size = metadata.split()
        name = raw_path.decode("utf-8")[len(path) + 1 :]
        if (
            mode != b"100644"
            or kind != b"blob"
            or name not in _UNIT_FILES
            or int(size) > MAX_METADATA_BYTES
        ):
            raise FeedbackError("capture git unit contains nonregular or nested files")
        entries.append((name, raw_path.decode("utf-8")))
    if {name for name, _ in entries} != _UNIT_FILES:
        raise FeedbackError("capture git unit file allowlist mismatch")
    files = {name: _git(checkout, "show", f"{commit}:{full}") for name, full in entries}
    digest = sha256_bytes(
        b"".join(f"{name}\0{sha256_bytes(files[name])}\n".encode() for name in sorted(files))
    )
    revisions = set()
    for name, raw in files.items():
        meta, _ = parse_front_matter(raw.decode("utf-8"))
        if (
            not meta
            or meta.get("schema") != "stock-reports.v1"
            or meta.get("unit") != unit
            or not isinstance(meta.get("source"), dict)
            or meta["source"].get("report_sha256") != entry["report_sha256"]
            or meta["source"].get("release") != entry["release"]
        ):
            raise FeedbackError("capture git unit source binding mismatch")
        revision = meta.get("revision")
        if type(revision) is not int or revision < 1:
            raise FeedbackError("capture git unit revision invalid")
        revisions.add(revision)
        artifact_hash = sha256_bytes(raw)
        batch.artifacts[artifact_hash] = raw
        batch.sources.append(
            {
                "path": f"git:{commit}:{path}/{name}",
                "sha256": artifact_hash,
                "hash_kind": "file_bytes_sha256",
                "producer": "stock-reports.v1",
            }
        )
    if len(revisions) != 1:
        raise FeedbackError("capture git unit revision mismatch")
    return digest, (next(iter(revisions)), digest, files)


def record_intent(
    config: dict,
    entry: dict,
    *,
    observation_started_at: str,
    previous_remote_head: str,
) -> dict:
    """Archive the intended bytes before network push, opening a durable interval."""
    root, session = _settings(config, entry)
    require_timestamp(observation_started_at)
    require_hash(previous_remote_head, length=40)
    with _writer(root):
        previous = _load(root, session)
        try:
            unit = _unit(entry.get("unit") or Path(entry["run_dir"]).name)
            commit = require_hash(entry.get("commit"), length=40)
            before = Batch()
            _snapshot(entry, unit, before)
            digest, bound = _bound_unit(Path(config["checkout_dir"]), commit, unit, entry, before)
            after = Batch()
            _snapshot(entry, unit, after)
            sources = [s for s in before.sources if not s["path"].startswith("git:")]
            if sources != after.sources:
                raise FeedbackError("capture sources changed during intent")
            fields = {
                "kind": "publication_intent",
                "observed_at": observation_started_at,
                "unit": unit,
                "report_sha256": entry["report_sha256"],
                "invocation_id": entry["invocation_id"],
                "release": entry["release"],
                "unit_sha256": digest,
                "revision": bound[0],
                "original_render_unit_sha256": entry.get("unit_sha256"),
                "intended_commit": commit,
                "previous_remote_head": previous_remote_head,
                "sources": before.sources,
            }
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
            _append(
                root,
                session,
                previous,
                {
                    "kind": "intent_failed",
                    "observed_at": _now(),
                    "reason": type(exc).__name__,
                    "unit": Path(entry.get("run_dir", "unknown")).name,
                },
            )
            raise FeedbackError("producer intent failed; failure recorded") from exc
        record = _append(root, session, previous, fields, before.artifacts)
        return {**record, "intent_ref": record["capture_ref"]}


def _capture(config, entry, commit, verified_by, baseline_evidence, producer_scope):
    require_hash(commit, length=40)
    stamp = _time(verified_by)
    unit = _unit(entry.get("unit") or Path(entry["run_dir"]).name)
    evidence = _evidence_metadata(baseline_evidence)
    if evidence["observation_started_at"] and _time(evidence["observation_started_at"]) > stamp:
        raise FeedbackError("capture remote observation clock moved backwards")
    before = Batch()
    report, _ = _snapshot(entry, unit, before)
    decision = report.get("decision_at")
    if decision is not None and stamp < _time(decision):
        raise FeedbackError("capture remote observation predates decision")
    checkout = Path(config["checkout_dir"])
    unit_hash, bound = _bound_unit(checkout, commit, unit, entry, before)
    if bound is None:
        raise FeedbackError("capture git unit binding failed")
    if evidence["remote_head"] != commit:
        _git(checkout, "merge-base", "--is-ancestor", commit, evidence["remote_head"])
        remote_hash, remote = _bound_unit(checkout, evidence["remote_head"], unit, entry, Batch())
        if remote_hash != unit_hash or remote[0] != bound[0]:
            raise FeedbackError("capture observed remote head changed the published unit")
    if evidence["previous_remote_head"]:
        _git(
            checkout,
            "merge-base",
            "--is-ancestor",
            evidence["previous_remote_head"],
            evidence["remote_head"],
        )
    after = Batch()
    _snapshot(entry, unit, after)
    source_before = [source for source in before.sources if not source["path"].startswith("git:")]
    if source_before != after.sources or any(
        before.artifacts[h] != after.artifacts[h] for h in after.artifacts
    ):
        raise FeedbackError("capture sources changed during capture")
    record = {
        "kind": "publication_captured",
        "observed_at": _now(),
        "verified_by": verified_by,
        "unit": unit,
        "report_date": unit,
        "report_sha256": entry["report_sha256"],
        "invocation_id": entry["invocation_id"],
        "release": entry["release"],
        "reports_commit": commit,
        "unit_sha256": unit_hash,
        "original_render_unit_sha256": entry.get("unit_sha256"),
        "revision": bound[0],
        "sources": before.sources,
        "source_stability": "same_bytes_before_after",
        "producer_scope": _scope(producer_scope),
        "remote_evidence": evidence,
        "synthetic_fixture": report.get("synthetic_fixture") is True
        or entry.get("synthetic_fixture") is True,
        "historical_replay": report.get("historical_replay") is True,
        # This is new producer evidence, not a fabricated coordinator source file.
        "producer_event": {
            "producer": "reports-publisher",
            "status": "published",
            "verified_by": verified_by,
            **_producer_identity(config),
        },
    }
    return record, before.artifacts


def capture_published(
    config: dict,
    entry: dict,
    commit: str,
    verified_by: str,
    *,
    baseline_evidence: dict,
    producer_scope: dict | None = None,
) -> dict:
    """Archive exact bytes at the locked publisher success boundary.

    A failed capture appends a failure record, then raises FeedbackError. The
    caller decides how to report the optional hook failure. If output itself fails, the exception
    propagates and no partial record becomes visible through state.json.
    """
    root, session = _settings(config, entry)
    with _writer(root):
        previous = _load(root, session)
        try:
            record, artifacts = _capture(
                config, dict(entry), commit, verified_by, baseline_evidence, producer_scope
            )
            intent_ref = record["remote_evidence"].get("intent_ref")
            if intent_ref:
                intent = read_capture(config, intent_ref)
                if (
                    intent.get("kind") != "publication_intent"
                    or any(
                        intent.get(key) != record.get(key)
                        for key in (
                            "unit",
                            "report_sha256",
                            "invocation_id",
                            "release",
                            "unit_sha256",
                        )
                    )
                    or intent.get("intended_commit") != commit
                ):
                    raise FeedbackError("capture does not match its publication intent")
                if intent["sources"] != record["sources"]:
                    raise FeedbackError("capture sources changed after publication intent")
                if _time(intent["observed_at"]) > _time(verified_by):
                    raise FeedbackError("capture precedes publication intent")
        except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as exc:
            _append(
                root,
                session,
                previous,
                {
                    "kind": "capture_failed",
                    "observed_at": _now(),
                    "reason": type(exc).__name__,
                    "unit": Path(entry.get("run_dir", "unknown")).name,
                },
            )
            raise FeedbackError("producer capture failed; failure recorded") from exc
        return _append(root, session, previous, record, artifacts)


def read_capture(config: dict, capture_ref: str | None = None) -> dict:
    """Read only committed chain records; reject orphan/uncommitted object refs."""
    root, session = _settings(config)
    state = _load(root, session)
    ref = state["head_ref"]
    sequence = state["sequence"]
    records = []
    while ref:
        record = _object(root, ref)
        if record.get("sequence") != sequence or record.get("session_id") != session:
            raise FeedbackError("capture chain discontinuity")
        records.append({**record, "capture_ref": ref})
        ref = record.get("previous_ref")
        sequence -= 1
    if sequence != -1 or records[-1].get("kind") != "session_started":
        raise FeedbackError("capture chain has no session baseline")
    if capture_ref is None:
        return {"state": state, "records": list(reversed(records))}
    require_hash(capture_ref)
    matches = [r for r in records if r["capture_ref"] == capture_ref]
    if not matches:
        raise FeedbackError("capture reference is not in the committed session")
    result = matches[0]
    artifacts = {}
    for source in result.get("sources", []):
        digest = require_hash(source.get("sha256"))
        path = root / "evidence" / f"{digest}.bin"
        if path.is_symlink():
            raise FeedbackError("capture evidence symlink is not allowed")
        raw = path.read_bytes()
        if sha256_bytes(raw) != digest:
            raise FeedbackError("capture evidence hash mismatch")
        artifacts[digest] = raw
    return {**result, "artifacts": artifacts}


def _reviewed(scope: dict) -> bool:
    return bool(
        scope.get("reviewed") is True
        and scope.get("registry_complete") is True
        and scope.get("controlled_epoch") is True
        and scope.get("registry_sha256")
        and "reports-publisher" in scope.get("producers", [])
    )


def coverage_summary(config: dict, unit: str | None = None) -> dict:
    """Conservative collection coverage; does not decide model/report eligibility."""
    records = read_capture(config)["records"]
    baseline = records[0]
    unit = _unit(unit or baseline["unit"])
    reasons = set()
    if baseline["unit"] != unit:
        reasons.add("unit_not_in_session_baseline")
    if not baseline.get("day_unit_absent") or not baseline.get("no_prior_unit_history"):
        reasons.add("baseline_existing_or_prior_history")
    scope = baseline.get("producer_scope", {})
    if not _reviewed(scope):
        reasons.add("producer_registry_unreviewed_or_incomplete")
    head = baseline["baseline_remote_head"]
    captures = []
    intents = {}
    closed_intents = set()
    last_time = _time(baseline["started_at"])
    for record in records[1:]:
        if record["kind"] == "publication_intent":
            intents[record["capture_ref"]] = record
            if record.get("previous_remote_head") != head:
                reasons.add("intent_remote_head_observation_gap")
            if _time(record["observed_at"]) < last_time:
                reasons.add("intent_time_out_of_order")
            continue
        if record["kind"] != "publication_captured":
            reasons.add("collection_gap_or_failure")
            continue
        evidence = record["remote_evidence"]
        intent_ref = evidence.get("intent_ref")
        if not intent_ref or intent_ref not in intents or intent_ref in closed_intents:
            reasons.add("missing_or_reused_publication_intent")
        else:
            closed_intents.add(intent_ref)
            if evidence.get("previous_remote_head") != intents[intent_ref].get(
                "previous_remote_head"
            ):
                reasons.add("intent_remote_head_mismatch")
            if evidence.get("observation_started_at") and _time(
                evidence["observation_started_at"]
            ) < _time(intents[intent_ref]["observed_at"]):
                reasons.add("remote_observation_predates_intent")
        if record["unit"] != unit:
            reasons.add("unexpected_unit_in_session")
        if record.get("producer_scope") != scope:
            reasons.add("producer_registry_changed_or_unknown")
        if evidence.get("method") not in METHODS or not evidence.get("observation_started_at"):
            reasons.add("snapshot_only_or_unknown_observation_basis")
        if evidence.get("previous_remote_head") != head:
            reasons.add("remote_head_observation_gap")
        head = evidence["remote_head"]
        current_time = _time(record["verified_by"])
        if current_time < last_time:
            reasons.add("observation_time_out_of_order")
        last_time = current_time
        if record["unit"] == unit:
            captures.append(record)
    if set(intents) != closed_intents:
        reasons.add("unfinished_publication_intent")
    if not captures:
        reasons.add("no_publication_captured")
    ordered = sorted(captures, key=lambda r: (_time(r["verified_by"]), r["sequence"]))
    first = ordered[0] if ordered else None
    return {
        "unit": unit,
        "coverage_status": "awaiting_coverage" if reasons else "covered",
        "reasons": sorted(reasons),
        "first_receipt_ref": first["capture_ref"] if first else None,
        "first_verified_by": first["verified_by"] if first else None,
        "first_report_sha256": first["report_sha256"] if first else None,
        "receipt_refs_chronological": [r["capture_ref"] for r in ordered],
        "captured_count": len(captures),
        "eligibility": "requires_separate_valid_report_check",
    }
