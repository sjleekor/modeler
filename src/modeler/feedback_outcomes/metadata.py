"""Bounded F3 publication metadata export and immutable archive adaptation.

Normalization makes an ephemeral *replica* of original content-addressed capture
records, state and evidence, then reuses F3's reader and coverage semantics. It
never creates/reconstructs producer day files, reports checkouts or receipts and
never reads a price, label, evaluation or unrelated filesystem source.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from copy import deepcopy
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from modeler.feedback import capture, captured
from modeler.feedback.core import (
    MAX_METADATA_BYTES,
    FeedbackError,
    canonical_bytes,
    require_hash,
    require_timestamp,
    sha256_bytes,
)
from modeler.feedback_outcomes.contracts import stamp

SCHEMA = "kr-v1-outcome-publications.v1"
MAX_BUNDLE_BYTES = MAX_METADATA_BYTES
_KINDS = {
    "session_started",
    "publication_intent",
    "publication_captured",
    "gap",
    "capture_failed",
    "intent_failed",
}
_SESSION = re.compile(r"[A-Za-z0-9_-]{1,96}\Z")


def _object(value: Any, name: str) -> dict:
    if not isinstance(value, dict):
        raise FeedbackError(f"{name} must be an object")
    return value


def _day(value: Any) -> str:
    try:
        valid = isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        valid = False
    if not valid:
        raise FeedbackError("capture unit must be a canonical date")
    return value


def _sources(record: dict) -> dict[str, dict]:
    """Validate every source name before any artifact is opened or decoded."""
    sources = record.get("sources", [])
    if not isinstance(sources, list):
        raise FeedbackError("capture sources must be a list")
    if record["kind"] not in {"publication_intent", "publication_captured"}:
        if sources:
            raise FeedbackError("unexpected sources on a non-publication record")
        return {}
    day = _day(record.get("unit"))
    commit = require_hash(record.get("reports_commit", record.get("intended_commit")), length=40)
    prefix = f"git:{commit}:{capture._unit_path(day)}/"
    ordinary = {
        f"report-{day}.json",
        f"explain-{day}.json",
        "markdown/manifest.json",
        *capture._SIDES,
    }
    required = {
        f"report-{day}.json",
        "run-state.json",
        "coordinator-inference.json",
        "markdown/manifest.json",
        *(prefix + name for name in capture._UNIT_FILES),
    }
    output = {}
    for source in sources:
        source = _object(source, "capture source")
        path = source.get("path")
        git_source = isinstance(path, str) and path.startswith(prefix)
        allowed = (isinstance(path, str) and path in ordinary) or (
            git_source and path[len(prefix) :] in capture._UNIT_FILES
        )
        if not allowed:
            raise FeedbackError("capture source path is outside the fixed publication allowlist")
        if path in output:
            raise FeedbackError("duplicate captured source path")
        require_hash(source.get("sha256"))
        expected_kind = "file_bytes_sha256" if git_source else "bytes_sha256"
        if source.get("hash_kind") != expected_kind:
            raise FeedbackError("invalid captured source hash kind")
        if "bytes" in source and (
            type(source["bytes"]) is not int or not 0 <= source["bytes"] <= MAX_BUNDLE_BYTES
        ):
            raise FeedbackError("captured source exceeds metadata size limit")
        output[path] = source
    if not required <= output.keys():
        raise FeedbackError("captured publication source inventory is incomplete")
    report = output[f"report-{day}.json"]
    if report["sha256"] != require_hash(record.get("report_sha256")):
        raise FeedbackError("capture report hash binding mismatch")
    return output


def _chain(session: dict, seen_refs: set[str]) -> dict[str, list[dict]]:
    if set(session) != {"session_id", "state", "records", "artifacts"}:
        raise FeedbackError("invalid capture session wrapper schema")
    session_id = session.get("session_id")
    if not isinstance(session_id, str) or not _SESSION.fullmatch(session_id):
        raise FeedbackError("invalid capture session id")
    state = _object(session.get("state"), "capture state")
    if set(state) != {"schema", "session_id", "sequence", "head_ref"}:
        raise FeedbackError("invalid capture state schema")
    if state.get("schema") != capture.SCHEMA or state.get("session_id") != session_id:
        raise FeedbackError("capture state identity mismatch")
    records = session.get("records")
    if not isinstance(records, list) or not records:
        raise FeedbackError("capture session requires its original baseline chain")
    previous = None
    evidence: dict[str, list[dict]] = {}
    for sequence, record in enumerate(records):
        record = _object(record, "capture record")
        ref = require_hash(record.get("capture_ref"))
        if ref in seen_refs:
            raise FeedbackError("duplicate capture reference")
        seen_refs.add(ref)
        original = {key: value for key, value in record.items() if key != "capture_ref"}
        if sha256_bytes(canonical_bytes(original)) != ref:
            raise FeedbackError("capture record byte hash mismatch")
        if (
            record.get("schema") != capture.SCHEMA
            or record.get("session_id") != session_id
            or type(record.get("sequence")) is not int
            or record["sequence"] != sequence
            or record.get("previous_ref") != previous
        ):
            raise FeedbackError("capture chain discontinuity")
        if record.get("kind") not in _KINDS:
            raise FeedbackError("unknown capture record kind")
        if (sequence == 0) != (record["kind"] == "session_started"):
            raise FeedbackError("capture chain has invalid baseline")
        require_timestamp(record.get("observed_at"))
        if sequence == 0:
            _day(record.get("unit"))
            require_timestamp(record.get("started_at"))
        if record["kind"] == "publication_captured":
            require_timestamp(record.get("verified_by"))
            if (
                type(record.get("synthetic_fixture")) is not bool
                or type(record.get("historical_replay")) is not bool
            ):
                raise FeedbackError("capture fixture flags must be explicit booleans")
        for source in _sources(record).values():
            evidence.setdefault(source["sha256"], []).append(source)
        previous = ref
    if (
        type(state.get("sequence")) is not int
        or state["sequence"] != len(records) - 1
        or state.get("head_ref") != previous
    ):
        raise FeedbackError("capture state head does not match original chain")
    return evidence


def _validate(
    bundle: dict, *, synthetic: bool | None = None
) -> list[tuple[dict, dict[str, bytes]]]:
    bundle = _object(bundle, "publication bundle")
    if set(bundle) != {"schema", "synthetic_fixture", "sessions"} or bundle.get("schema") != SCHEMA:
        raise FeedbackError("invalid publication metadata bundle schema")
    if type(bundle.get("synthetic_fixture")) is not bool:
        raise FeedbackError("publication bundle requires an explicit fixture flag")
    if synthetic is not None and bundle["synthetic_fixture"] is not synthetic:
        raise FeedbackError("fixture and real publication bundles cannot be mixed")
    if len(canonical_bytes(bundle)) > MAX_BUNDLE_BYTES:
        raise FeedbackError("publication bundle exceeds the 32 MiB metadata cap")
    sessions = bundle.get("sessions")
    if not isinstance(sessions, list):
        raise FeedbackError("publication sessions must be a list")
    checked, session_ids, refs = [], set(), set()
    # Complete all path/chain/inventory checks before decoding even one base64 value.
    for session in sessions:
        session = _object(session, "capture session")
        evidence = _chain(session, refs)
        if session["session_id"] in session_ids:
            raise FeedbackError("duplicate capture session")
        session_ids.add(session["session_id"])
        artifacts = _object(session.get("artifacts"), "capture artifacts")
        if set(artifacts) != set(evidence):
            raise FeedbackError("capture evidence inventory mismatch")
        for record in session["records"]:
            if (
                record["kind"] == "publication_captured"
                and record["synthetic_fixture"] is not bundle["synthetic_fixture"]
            ):
                raise FeedbackError("capture and bundle fixture flags disagree")
        checked.append((session, evidence))
    output, total = [], 0
    for session, evidence in checked:
        decoded = {}
        for digest, encoded in session["artifacts"].items():
            require_hash(digest)
            if not isinstance(encoded, str):
                raise FeedbackError("capture artifact must be base64 text")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise FeedbackError("invalid capture evidence base64") from exc
            total += len(raw)
            if total > MAX_BUNDLE_BYTES:
                raise FeedbackError("publication evidence exceeds the 32 MiB metadata cap")
            if sha256_bytes(raw) != digest:
                raise FeedbackError("capture evidence byte hash mismatch")
            if any(
                "bytes" in source and source["bytes"] != len(raw) for source in evidence[digest]
            ):
                raise FeedbackError("capture evidence length mismatch")
            decoded[digest] = raw
        output.append((session, decoded))
    return output


def export_capture_metadata(config: dict) -> dict:
    """Read only one explicit committed F3 session and its fixed metadata evidence."""
    saved = capture.read_capture(config)
    session = {
        "session_id": saved["state"]["session_id"],
        "state": saved["state"],
        "records": saved["records"],
        "artifacts": {},
    }
    _chain(session, set())  # Reject unexpected source paths before F3 opens evidence.
    flags = {
        record["synthetic_fixture"]
        for record in session["records"]
        if record["kind"] == "publication_captured"
    }
    if len(flags) > 1:
        raise FeedbackError("fixture and real captures cannot share an outcome bundle")
    total = len(canonical_bytes(session))
    for record in session["records"]:
        if not record.get("sources"):
            continue
        evidence = capture.read_capture(config, record["capture_ref"])["artifacts"]
        for digest, raw in evidence.items():
            if digest in session["artifacts"]:
                continue
            total += len(raw)
            if total > MAX_BUNDLE_BYTES:
                raise FeedbackError("publication export exceeds the 32 MiB metadata cap")
            session["artifacts"][digest] = base64.b64encode(raw).decode("ascii")
    bundle = {
        "schema": SCHEMA,
        "synthetic_fixture": next(iter(flags), False),
        "sessions": [session],
    }
    _validate(bundle)
    return deepcopy(bundle)


def _freshness_lag(model: dict, provenance: dict) -> tuple[Any, str | None, list[dict]]:
    """Map explicit KR producer lag fields, without deriving lag from a status.

    serving/adapters.py preserves context.freshness under provenance.freshness.
    serving/freshness.py's KR branch writes the session distance as delivery_lag
    (and market_lag=0 for fresh KR input, None for stale KR input). market_lag
    alone and quality.reference_lag_sessions never identify this session distance.
    """
    explicit = {}
    for prefix, value in (
        ("model.freshness", model.get("freshness")),
        ("provenance.freshness", provenance.get("freshness")),
    ):
        if isinstance(value, dict):
            for key in ("lag_sessions", "delivery_lag", "market_lag"):
                if key in value and value[key] is not None:
                    explicit[f"{prefix}.{key}"] = value[key]
    errors = []
    if any(type(value) is not int or value < 0 for value in explicit.values()):
        errors.append(
            {"source": "freshness", "reason": "freshness_lag_invalid", "fields": deepcopy(explicit)}
        )
        return None, None, errors
    if len(set(explicit.values())) > 1:
        errors.append(
            {
                "source": "freshness",
                "reason": "freshness_lag_conflict",
                "fields": deepcopy(explicit),
            }
        )
        return None, None, errors
    for path in (
        "model.freshness.lag_sessions",
        "provenance.freshness.lag_sessions",
        "provenance.freshness.delivery_lag",
    ):
        if path in explicit:
            return explicit[path], path, errors
    return None, None, errors


def _event(
    publication: dict, model: dict, report: dict, batch, baseline: dict, record: dict
) -> dict:
    provenance = model.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    freshness = model.get("freshness", provenance.get("freshness"))
    lag, lag_source, lag_errors = _freshness_lag(model, provenance)
    candidates = {
        prediction["prediction_ref"]: prediction
        for prediction in batch.predictions
        if prediction["report_sha256"] == publication["report_sha256"]
        and prediction["model_id"] == model.get("model_id")
        and prediction["market"] == "KR"
    }
    rows = []
    for ref in publication["prediction_refs"]:
        prediction = candidates.get(ref)
        if prediction is None or not 1 <= prediction["rank"] <= 100:
            continue
        raw = json.loads(batch.artifacts[prediction["source_evidence_sha256"]])["row"]
        rows.append(
            {
                key: prediction[key]
                for key in (
                    "prediction_ref",
                    "symbol",
                    "security_market",
                    "rank",
                    "score",
                    "quality_validation",
                )
            }
        )
        rows[-1]["quality_reasons"] = raw.get("quality_reasons", [])
    rows.sort(key=lambda row: row["rank"])
    return {
        "kind": "publication",
        "market": "KR",
        "model_id": model.get("model_id"),
        "model_version": model.get("model_version"),
        "bundle_hash": provenance.get("bundle_manifest_sha256"),
        "input_sha256": provenance.get("input_sha256"),
        "feature_asof_date": model.get("feature_asof_date"),
        "freshness_lag_sessions": lag,
        "freshness_lag_source": lag_source,
        "model_status": model.get("status"),
        "report_date": model.get("report_date"),
        "decision_at": model.get("decision_at"),
        "verified_by": publication["verified_by"],
        **{
            key: publication[key]
            for key in (
                "publication_ref",
                "receipt_ref",
                "report_sha256",
                "invocation_id",
                "reports_commit",
                "unit_sha256",
                "revision",
                "coverage_status",
            )
        },
        "collection_started_at": baseline["started_at"],
        "collection_coverage": deepcopy(publication["collection_coverage"]),
        "synthetic_fixture": record["synthetic_fixture"] or model.get("synthetic_fixture") is True,
        "historical_replay": record["historical_replay"]
        or report.get("historical_replay") is True
        or model.get("historical_replay") is True,
        "rows": rows,
        "original_provenance": deepcopy(provenance),
        "original_freshness": deepcopy(freshness),
        "adapter_errors": [*deepcopy(batch.errors), *lag_errors],
        "explicit_withdrawal_adapter": "not_implemented_no_original_schema",
    }


def normalize_publications(
    bundle: dict, *, observed_at: str, synthetic: bool = False
) -> list[dict]:
    """Validate metadata then reuse F3 on an exact temporary immutable archive replica.

    This is not producer evidence synthesis: original byte-addressed record bodies,
    chain state and evidence are copied unchanged. No mutable producer source is
    read or generated. Missing/changed publications never imply a withdrawal.
    """
    require_timestamp(observed_at)
    validated = _validate(bundle, synthetic=synthetic)
    cutoff = stamp(observed_at)
    for session, _ in validated:
        for record in session["records"]:
            for key in ("observed_at", "started_at", "verified_by"):
                if key in record and stamp(record[key]) > cutoff:
                    raise FeedbackError(
                        "capture observation is after the metadata observation cutoff"
                    )
    events = []
    with TemporaryDirectory(prefix="outcome-capture-replica-") as temporary:
        root = Path(temporary).resolve()
        for session, artifacts in validated:
            config = {
                "feedback_capture": {"output_root": str(root), "session_id": session["session_id"]}
            }
            for record in session["records"]:
                raw = canonical_bytes(
                    {key: value for key, value in record.items() if key != "capture_ref"}
                )
                path = root / "objects" / f"{record['capture_ref']}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            state_path = root / "sessions" / session["session_id"] / "state.json"
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_bytes(canonical_bytes(session["state"]))
            for digest, raw in artifacts.items():
                path = root / "evidence" / f"{digest}.bin"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            if not any(record["kind"] == "publication_captured" for record in session["records"]):
                continue
            batch = captured.read_captured(config, observed_at=observed_at)
            by_ref = {record["capture_ref"]: record for record in session["records"]}
            baseline = session["records"][0]
            for publication in batch.publications:
                record = by_ref[publication["receipt_ref"]]
                sources = _sources(record)
                report = json.loads(artifacts[sources[f"report-{record['unit']}.json"]["sha256"]])
                for key in ("synthetic_fixture", "historical_replay"):
                    if key in report and type(report[key]) is not bool:
                        raise FeedbackError("original report fixture flag is invalid")
                seen_models = set()
                for model in report["markets"]:
                    if not isinstance(model, dict) or model.get("market") != "KR":
                        continue
                    model_key = canonical_bytes([model.get("market"), model.get("model_id")])
                    if model_key in seen_models:
                        raise FeedbackError("duplicate original KR model reference")
                    seen_models.add(model_key)
                    for key in ("synthetic_fixture", "historical_replay"):
                        if key in model and type(model[key]) is not bool:
                            raise FeedbackError("original model fixture flag is invalid")
                    event = _event(publication, model, report, batch, baseline, record)
                    if not synthetic and (event["synthetic_fixture"] or event["historical_replay"]):
                        continue
                    events.append(event)
    return events
