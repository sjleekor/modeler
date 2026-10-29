"""Adapt immutable publisher captures without reopening mutable daily sources.

Producer events are explicit new evidence. Old coordinator-publication sidecars
remain archived bytes and are never rewritten to impersonate this observer.
Coverage is a prerequisite for F1b, not a decision that a report is eligible.
"""

from __future__ import annotations

import json
from pathlib import Path

from modeler.reporting.markdown import parse_front_matter

from . import capture
from .briefing import (
    ADAPTER_VERSION,
    _adapt_predictions,
    _day,
    _decision,
    _exposed_predictions,
    _invocation,
    _validated_explain,
)
from .core import (
    SOURCE_IDENTITY_VERSION,
    Batch,
    FeedbackError,
    canonical_bytes,
    identity,
    read_json,
    require_timestamp,
    sha256_bytes,
)
from .publisher_capture import SETTINGS_SCHEMA


def session_config(settings_path: Path, day: str) -> dict:
    if not _day(day):
        raise FeedbackError("invalid capture day")
    path = Path(settings_path)
    if not path.is_absolute():
        raise FeedbackError("capture settings must be absolute")
    settings = read_json(path, root=path.parent, batch=Batch())
    if settings.get("schema") != SETTINGS_SCHEMA or settings.get("enabled") is not True:
        raise FeedbackError("enabled capture settings are required")
    epoch = settings.get("epoch_id")
    if not isinstance(epoch, str) or not epoch or len(epoch) > 128:
        raise FeedbackError("invalid capture epoch")
    config = {
        "feedback_capture": {
            "output_root": settings["output_root"],
            "session_id": sha256_bytes(canonical_bytes({"epoch_id": epoch, "unit": day})),
        }
    }
    capture._settings(config)
    return config


def _adapt(config, record, coverage, observed_at):
    saved = capture.read_capture(config, record["capture_ref"])
    day = record["unit"]
    batch = Batch(artifacts=saved["artifacts"].copy(), sources=saved["sources"].copy())
    sources = {source["path"]: source["sha256"] for source in batch.sources}
    if len(sources) != len(batch.sources):
        raise FeedbackError("duplicate captured source paths")

    def raw(name):
        return batch.artifacts[sources[name]]

    def optional_json(name):
        try:
            value = json.loads(raw(name))
            if not isinstance(value, dict):
                raise ValueError("object required")
            return value
        except (ValueError, UnicodeError):
            batch.errors.append({"source": name, "reason": "captured_metadata_invalid"})
            return {}

    report_hash = sources[f"report-{day}.json"]
    report = json.loads(raw(f"report-{day}.json"))
    if (
        report_hash != record["report_sha256"]
        or report.get("schema_version") != "1.0"
        or not isinstance(report.get("markets"), list)
        or report.get("report_date") != day
        or not _decision(report.get("decision_at"), day)
    ):
        raise FeedbackError("captured daily envelope binding failed")
    # A publication callback can run before the coordinator updates this sidecar.
    # Its original bytes stay in sources; only the explicit event binds publication.
    sides = {
        name: optional_json(name)
        for name in sources
        if not name.startswith("git:")
        and name.endswith(".json")
        and name
        not in {f"report-{day}.json", f"explain-{day}.json", "coordinator-publication.json"}
    }
    invocation = _invocation(report, sides, report_hash, day, batch)
    if invocation != record["invocation_id"]:
        raise FeedbackError("capture invocation binding failed")
    explain, explain_hash = {}, None
    if f"explain-{day}.json" in sources:
        digest = sources[f"explain-{day}.json"]
        explain, explain_hash = _validated_explain(
            optional_json(f"explain-{day}.json"), digest, sides, report_hash, invocation, day, batch
        )
    _adapt_predictions(
        report,
        day=day,
        report_hash=report_hash,
        invocation=invocation,
        explain=explain,
        explain_hash=explain_hash,
        observed_at=observed_at,
        historical_link=False,
        batch=batch,
    )
    prefix = f"git:{record['reports_commit']}:{capture._unit_path(day)}/"
    files = {name[len(prefix) :]: raw(name) for name in sources if name.startswith(prefix)}
    if set(files) != capture._UNIT_FILES:
        raise FeedbackError("captured git file inventory mismatch")
    unit_hash = sha256_bytes(
        b"".join(f"{name}\0{sha256_bytes(files[name])}\n".encode() for name in sorted(files))
    )
    if unit_hash != record["unit_sha256"]:
        raise FeedbackError("captured git unit hash mismatch")
    for content in files.values():
        meta, _ = parse_front_matter(content.decode("utf-8"))
        if (
            meta.get("unit") != day
            or meta.get("revision") != record["revision"]
            or meta.get("source", {}).get("report_sha256") != report_hash
            or meta.get("source", {}).get("release") != record["release"]
        ):
            raise FeedbackError("captured git unit source mismatch")
    exposed = _exposed_predictions(files, batch)
    receipt_ref = record["capture_ref"]
    record_raw = canonical_bytes({k: v for k, v in record.items() if k != "capture_ref"})
    if sha256_bytes(record_raw) != receipt_ref:
        raise FeedbackError("captured record byte hash mismatch")
    batch.artifacts[receipt_ref] = record_raw
    batch.sources.append(
        {"path": "producer-capture", "sha256": receipt_ref, "hash_kind": "bytes_sha256"}
    )
    observation = {
        "receipt_ref": receipt_ref,
        "receipt_sha256": receipt_ref,
        "verified_by": record["verified_by"],
        "remote_head": record["remote_evidence"]["remote_head"],
    }
    binding = {
        key: record[key]
        for key in (
            "report_sha256",
            "invocation_id",
            "unit",
            "revision",
            "unit_sha256",
            "reports_commit",
        )
    }
    publication = {
        **binding,
        "publication_ref": identity("publication", {"binding": binding}),
        "report_date": day,
        "release": record["release"],
        "unit_path": capture._unit_path(day),
        "unit_binding_verified": True,
        "prediction_refs": sorted(exposed),
        "report_prediction_refs": [p["prediction_ref"] for p in batch.predictions],
        "status": "published_verified",
        "verified_by": record["verified_by"],
        "observed_at": observed_at,
        "historical_publication_evidence": False,
        "receipt_ref": receipt_ref,
        "receipt_sha256": receipt_ref,
        "publication_observation_ref": receipt_ref,
        "verification_observations": [observation],
        "coverage_status": coverage["coverage_status"],
        "collection_coverage": coverage,
        "eligibility_status": "not_evaluated",
        "producer_event": record["producer_event"],
        "adapter_version": ADAPTER_VERSION,
        "source_identity_version": SOURCE_IDENTITY_VERSION,
        "evidence_sha256": sorted(batch.artifacts),
    }
    batch.publications.append(publication)
    for prediction in batch.predictions:
        prediction["publication_status"] = (
            "published_verified" if prediction["prediction_ref"] in exposed else "not_published"
        )
        if prediction["prediction_ref"] in exposed:
            prediction.update(
                verified_by=record["verified_by"],
                receipt_ref=receipt_ref,
                receipt_sha256=receipt_ref,
                verification_observations=[observation],
                coverage_status=coverage["coverage_status"],
                synthetic_fixture=record["synthetic_fixture"],
                historical_replay=record["historical_replay"],
            )
    return batch


def read_captured(config: dict, *, observed_at: str) -> Batch:
    """Read one committed session in receipt-time order; never infer first publication.

    The full receipt sequence stays available for F4a's first valid report rule.
    Collection coverage alone never sets an eligibility flag or replaces a sample.
    """
    require_timestamp(observed_at)
    records = capture.read_capture(config)["records"]
    coverage = capture.coverage_summary(config)
    published = sorted(
        (r for r in records if r["kind"] == "publication_captured"),
        key=lambda r: (capture._time(r["verified_by"]), r["sequence"]),
    )
    batch = Batch()
    if not published:
        raise FeedbackError("no committed publication capture")
    for record in published:
        batch.extend(_adapt(config, record, coverage, observed_at))
    return batch
