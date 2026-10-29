"""Day-bounded store commits with a small atomic catalog.

Readers use catalog generation references, never a shard's current pointer: a
crash after shard commit leaves the previous catalog snapshot intact. Replaying
that batch repairs the catalog and retains all immutable generations/evidence.
Undated shadow rows require explicit day plus recomputable source identity;
undated publications must bind only to dated/validated rows in this batch.
No unpartitioned index is migrated implicitly.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from .core import (
    SOURCE_IDENTITY_VERSION,
    Batch,
    FeedbackError,
    canonical_bytes,
    identity,
    require_hash,
    sha256_bytes,
)
from .store import _atomic, _writer, check_output_root, commit_batch

CATALOG_SCHEMA = "prediction-feedback-day-catalog.v1"


def _day(value: str) -> str:
    try:
        if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
            raise ValueError
    except ValueError as exc:
        raise FeedbackError("partition day must be YYYY-MM-DD") from exc
    return value


def load_catalog(root: Path) -> dict:
    """Load only small catalog metadata; do not aggregate prediction rows."""
    path = Path(root) / "catalog.json"
    if path.is_symlink():
        raise FeedbackError("catalog symlink is not allowed")
    if not path.exists():
        return {"schema": CATALOG_SCHEMA, "days": {}}
    catalog = json.loads(path.read_bytes())
    if (
        not isinstance(catalog, dict)
        or catalog.get("schema") != CATALOG_SCHEMA
        or not isinstance(catalog.get("days"), dict)
    ):
        raise FeedbackError("invalid partition catalog")
    for day, entry in catalog["days"].items():
        _day(day)
        if not isinstance(entry, dict):
            raise FeedbackError("invalid partition catalog entry")
        require_hash(entry.get("generation"), length=32)
        require_hash(entry.get("index_sha256"))
        require_hash(entry.get("manifest_sha256"))
        if entry.get("root") != f"days/{day}":
            raise FeedbackError("invalid partition catalog root")
    return catalog


def _validate_batch_day(day: str, batch: Batch) -> None:
    refs = set()
    for prediction in batch.predictions:
        ref = require_hash(prediction.get("prediction_ref"))
        if prediction.get("report_date") is not None:
            if prediction["report_date"] != day:
                raise FeedbackError("prediction report_date differs from partition day")
        else:
            if (
                prediction.get("kind") != "shadow"
                or prediction.get("source_identity_version") != SOURCE_IDENTITY_VERSION
            ):
                raise FeedbackError("undated prediction requires validated shadow source identity")
            fields = {
                "bundle_hash": require_hash(prediction.get("bundle_hash")),
                "record_sha256": require_hash(prediction.get("record_sha256")),
                "ticker": prediction.get("symbol"),
                "market": prediction.get("security_market"),
            }
            if identity("shadow", fields) != ref:
                raise FeedbackError("undated shadow source identity mismatch")
        refs.add(ref)
    for publication in batch.publications:
        if publication.get("report_date") is not None:
            if publication["report_date"] != day:
                raise FeedbackError("publication report_date differs from partition day")
        else:
            links = publication.get("prediction_refs")
            if not isinstance(links, list) or not links or not set(links).issubset(refs):
                raise FeedbackError("undated publication must bind to same-batch day predictions")


def commit_day_batch(
    root: Path,
    batch: Batch,
    *,
    day: str,
    mapping: dict,
    source_key: str,
    source_roots: list[Path],
    observed_at: str,
    issue_document_ref: str,
) -> dict:
    """Commit one day and atomically expose its immutable generation in catalog.

    Returns the original day manifest; its run_id locates that day's generation.
    The explicit day routes source-error-only batches without inventing rows.
    """
    day = _day(day)
    _validate_batch_day(day, batch)
    root = check_output_root(root, source_roots)
    with _writer(root):
        if (root / "index").exists():
            raise FeedbackError("unpartitioned index requires explicit migration; refusing")
        catalog = load_catalog(root)
        shard = root / "days" / day
        manifest = commit_batch(
            shard,
            batch,
            mapping=mapping,
            source_key=source_key,
            source_roots=source_roots,
            observed_at=observed_at,
            issue_document_ref=issue_document_ref,
        )
        entry = {
            "root": f"days/{day}",
            "generation": manifest["run_id"],
            "index_sha256": manifest["index_sha256"],
            "manifest_sha256": sha256_bytes(canonical_bytes(manifest)),
            "counts": manifest["counts"],
            "source_key": source_key,
            "observed_at": observed_at,
        }
        catalog["days"][day] = entry
        _atomic(root / "catalog.json", canonical_bytes(catalog))
        return manifest
