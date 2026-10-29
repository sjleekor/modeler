"""Single-writer generations with a last-step atomic pointer; sources stay read-only."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path

from .core import (
    ADAPTER_VERSION,
    SOURCE_IDENTITY_VERSION,
    Batch,
    FeedbackError,
    canonical_bytes,
    require_hash,
    require_timestamp,
    sha256_bytes,
)


def _json(path: Path) -> dict:
    if path.is_symlink():
        raise FeedbackError("output symlink is not allowed")
    return json.loads(path.read_bytes())


def _atomic(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def _immutable(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise FeedbackError("immutable evidence conflict")
        return
    _atomic(path, raw)


def check_output_root(root: Path, source_roots: list[Path]) -> Path:
    root = Path(root).absolute()
    if any(p.is_symlink() for p in [root, *root.parents]):
        raise FeedbackError("output path cannot traverse a symlink")
    resolved = root.resolve()
    for source in source_roots:
        source = Path(source).resolve()
        if resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise FeedbackError("output and source roots must be disjoint")
    if root.exists() and any(p.is_symlink() for p in root.rglob("*")):
        raise FeedbackError("output contains symlinks")
    return root


@contextmanager
def _writer(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".write.lock").open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise FeedbackError("feedback writer is locked") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def load_index(root: Path) -> dict:
    root = Path(root)
    pointer = root / "index/current.json"
    if not pointer.exists():
        return {"predictions": [], "publications": [], "occurrences": [], "source_states": {}}
    current = _json(pointer)
    generation = require_hash(current.get("generation"), length=32)
    path = root / "index/generations" / generation / "index.json"
    raw = path.read_bytes()
    if sha256_bytes(raw) != require_hash(current.get("index_sha256")):
        raise FeedbackError("current index hash mismatch")
    return json.loads(raw)


def _implementation() -> dict:
    package = Path(__file__).parent
    files = {p.name: sha256_bytes(p.read_bytes()) for p in sorted(package.glob("*.py"))}
    result = subprocess.run(
        ["git", "-C", str(package), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return {
        "files": files,
        "code_sha256": sha256_bytes(canonical_bytes(files)),
        "modeler_commit": result.stdout.strip() if result.returncode == 0 else "unknown",
    }


def _merge(old: list[dict], new: list[dict], key: str) -> list[dict]:
    rows = {row[key]: dict(row) for row in old}
    for row in new:
        ref = require_hash(row.get(key))
        if ref in rows:
            previous = dict(rows[ref])
            first = previous.get("first_observed_at", previous.get("observed_at"))
            rows[ref].update(row)
            rows[ref]["first_observed_at"] = first
            observations = {
                r["receipt_ref"]: r for r in previous.get("verification_observations", [])
            }
            for observation in row.get("verification_observations", []):
                observations[observation["receipt_ref"]] = observation
            if observations:
                rows[ref]["verification_observations"] = [
                    observations[k] for k in sorted(observations)
                ]
            published = {"published_verified", "published_evidence"}
            state_field = "status" if key == "publication_ref" else "publication_status"
            if (
                key in {"publication_ref", "prediction_ref"}
                and previous.get(state_field) in published
            ):
                rows[ref]["ever_published"] = True
                if row.get(state_field) not in published:
                    rows[ref]["last_link_status"] = row.get(state_field)
                    rows[ref][state_field] = previous[state_field]
            if previous.get("receipt_ref") and not row.get("receipt_ref"):
                for field in (
                    "verified_by",
                    "receipt_ref",
                    "receipt_sha256",
                    "coverage_status",
                    "publication_observation_ref",
                ):
                    if field in previous:
                        rows[ref][field] = previous[field]
                if key == "publication_ref":
                    rows[ref]["last_link_status"] = row.get("status")
                    rows[ref]["status"] = previous["status"]
                elif key == "prediction_ref":
                    rows[ref]["publication_status"] = previous.get("publication_status")
        else:
            rows[ref] = {**row, "first_observed_at": row.get("observed_at")}
    return [rows[k] for k in sorted(rows)]


def _lineage(rows: list[dict], publications: list[dict]) -> list[dict]:
    revisions = {}
    for publication in publications:
        if not publication.get("unit_binding_verified"):
            continue
        revision = publication.get("revision")
        if type(revision) is int and revision > 0:
            for ref in publication.get("prediction_refs", []):
                revisions.setdefault(ref, set()).add(revision)
    groups = {}
    keys = ("market", "security_market", "model_id", "symbol", "report_date")
    for row in rows:
        if row.get("kind") == "briefing":
            groups.setdefault(tuple(row.get(k) for k in keys), []).append(row)
    for row in rows:
        if row.get("kind") != "briefing":
            continue
        row.pop("supersedes", None)
        related, predecessors = [], []
        mine = revisions.get(row["prediction_ref"], set())
        for other in groups[tuple(row.get(k) for k in keys)]:
            if other["prediction_ref"] == row["prediction_ref"] or other.get("kind") != "briefing":
                continue
            if all(
                other.get(k) == row.get(k)
                for k in ("market", "security_market", "model_id", "symbol", "report_date")
            ) and other.get("report_sha256") != row.get("report_sha256"):
                related.append(other["prediction_ref"])
                theirs = revisions.get(other["prediction_ref"], set())
                if mine and theirs and min(mine) > max(theirs):
                    predecessors.append(other["prediction_ref"])
        if related:
            row["related_predictions"] = sorted(set(related))
        if predecessors:
            row["supersedes"] = sorted(set(predecessors))
    return rows


def commit_batch(
    root: Path,
    batch: Batch,
    *,
    mapping: dict,
    source_key: str,
    source_roots: list[Path],
    observed_at: str,
    issue_document_ref: str,
) -> dict:
    """Merge a batch, retain evidence, then expose one complete generation."""
    from .issues import build_occurrences, render_diagnostic

    observed_at = require_timestamp(observed_at)
    root = check_output_root(root, source_roots)
    if not source_key:
        raise FeedbackError("source key is required")
    with _writer(root):
        previous = load_index(root)
        incoming = [dict(row) for row in batch.predictions]
        for row in incoming:
            row["source_key"] = source_key
        predictions = _merge(previous["predictions"], incoming, "prediction_ref")
        for prediction in predictions:
            if prediction.get("source_key") == source_key:
                prediction["source_status"] = "source_incomplete" if batch.errors else "available"
                prediction["source_errors"] = batch.errors
        publications = _merge(previous["publications"], batch.publications, "publication_ref")
        predictions = _lineage(predictions, publications)
        # Re-map all preserved occurrences from immutable prediction evidence, not run timestamps.
        occurrences = _merge(
            previous["occurrences"], build_occurrences(predictions, mapping), "occurrence_ref"
        )
        source_states = dict(previous.get("source_states", {}))
        source_states[source_key] = {
            "observed_at": observed_at,
            "errors": batch.errors,
            "status": "source_incomplete" if batch.errors else "available",
        }
        generation = uuid.uuid4().hex
        index = {
            "schema": "prediction-feedback-index.v1",
            "source_identity_version": SOURCE_IDENTITY_VERSION,
            "predictions": predictions,
            "publications": publications,
            "occurrences": occurrences,
            "source_states": source_states,
        }
        encoded = canonical_bytes(index)
        manifest = {
            "schema": "prediction-feedback-run.v1",
            "run_id": generation,
            "adapter_version": ADAPTER_VERSION,
            "implementation": _implementation(),
            "source_identity_version": SOURCE_IDENTITY_VERSION,
            "observed_at": observed_at,
            "source_key": source_key,
            "mapping": mapping,
            "issue_document_ref": issue_document_ref,
            "sources": batch.sources,
            "errors": batch.errors,
            "index_sha256": sha256_bytes(encoded),
            "counts": {
                "predictions": len(predictions),
                "publications": len(publications),
                "occurrences": len(occurrences),
                "errors": len(batch.errors),
            },
        }
        for digest, raw in batch.artifacts.items():
            if sha256_bytes(raw) != require_hash(digest):
                raise FeedbackError("captured evidence hash mismatch")
            _immutable(root / "evidence" / f"{digest}.bin", raw)
        target = root / "index/generations" / generation
        _immutable(target / "index.json", encoded)
        _immutable(
            target / "diagnostic.md",
            render_diagnostic(
                predictions, occurrences, issue_document_ref=issue_document_ref
            ).encode("utf-8"),
        )
        _immutable(root / "runs" / generation / "manifest.json", canonical_bytes(manifest))
        _atomic(
            root / "index/current.json",
            canonical_bytes(
                {
                    "generation": generation,
                    "index_sha256": sha256_bytes(encoded),
                    "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
                }
            ),
        )
        return manifest
