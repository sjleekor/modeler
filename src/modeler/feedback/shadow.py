"""Read verified P6 prediction metadata without opening evaluation artifacts.

The frozen verifier accepts arbitrary record ``files`` keys. This adapter instead
allows only the producer's three output names, hashes opaque panels/baskets, and
projects only four prediction columns. It never follows record-provided paths.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from modeler.feedback.core import (
    ADAPTER_VERSION,
    SOURCE_IDENTITY_VERSION,
    Batch,
    identity,
    read_bytes,
    require_timestamp,
    sha256_bytes,
)
from modeler.models._03_kr_next.shadow import counting_records

CHECKER_VERSION = "kr-next-shadow-metadata.v1"
_HEX = re.compile(r"^[0-9a-f]{64}$")
_OUTPUTS = {"panel.parquet", "predictions.parquet", "basket.parquet"}
_COLUMNS = ["ticker", "market", "rank", "score"]


class _Invalid(ValueError):
    pass


def _path(root: Path, relative: str) -> Path:
    """Reject escaping paths and symlinks before any file content is opened."""
    rel = Path(relative)
    if rel.is_absolute() or ".." in rel.parts:
        raise _Invalid(f"unsafe source path: {relative}")
    candidate = root / rel
    for part in (candidate, *candidate.parents):
        if part == root.parent:
            break
        if part.is_symlink():
            raise _Invalid(f"source symlink is not allowed: {relative}")
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise _Invalid(f"source path escapes root: {relative}")
    return candidate


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise _Invalid(f"{name} is not a SHA-256")
    return value


def _day(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise _Invalid(f"{name} is not an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise _Invalid(f"{name} is not an ISO date") from exc
    if parsed.isoformat() != value:
        raise _Invalid(f"{name} is not an ISO date")
    return value


def _error(batch: Batch, path: str, exc: Exception, *, line: dict | None = None) -> None:
    missing = isinstance(exc, FileNotFoundError) or str(exc) == "source_missing"
    item = {
        "kind": "shadow",
        "path": path,
        "source": path,
        "reason": "source_missing" if missing else "shadow_integrity_error",
        "status": "source_missing" if missing else "source_incomplete",
        "reason_code": "source_missing" if missing else "shadow_integrity_error",
        "message": str(exc),
        "checker_version": CHECKER_VERSION,
    }
    if line:
        item.update({key: line.get(key) for key in ("D", "attempt", "record_sha256", "valid")})
    batch.errors.append(item)


def _ledger(root: Path, batch: Batch) -> list[dict]:
    path = _path(root, "ledger.jsonl")
    raw = read_bytes(path, root=root, batch=batch)
    if raw and not raw.endswith(b"\n"):
        raise _Invalid("ledger does not end with a newline (torn write?)")
    prev = "0" * 64
    attempts: dict[str, int] = {}
    last_d = ""
    lines = []
    for number, text in enumerate(raw[:-1].split(b"\n") if raw else [], 1):
        obj = json.loads(text)
        if not isinstance(obj, dict):
            raise _Invalid(f"ledger line {number}: expected an object")
        if obj.get("prev_ledger_sha256") != prev:
            raise _Invalid(f"ledger line {number}: prev_ledger_sha256 mismatch")
        prev = sha256_bytes(text)
        if obj.get("schema") != "kr-next-shadow-ledger.v1":
            raise _Invalid(f"ledger line {number}: unknown schema")
        d = _day(obj.get("D"), "D")
        attempt = obj.get("attempt")
        if type(attempt) is not int or attempt != attempts.get(d, 0) + 1:
            raise _Invalid(f"ledger line {number}: attempt is out of sequence")
        if d < last_d:
            raise _Invalid(f"ledger line {number}: D is earlier than the previous line")
        attempts[d], last_d = attempt, d
        rel = f"D={d}" + (f"/attempt-{attempt}" if attempt > 1 else "")
        if obj.get("dir") != rel:
            raise _Invalid(f"ledger line {number}: dir differs from D/attempt")
        if type(obj.get("valid")) is not bool or type(obj.get("has_basket")) is not bool:
            raise _Invalid(f"ledger line {number}: valid/has_basket must be boolean")
        _digest(obj.get("record_sha256"), "record_sha256")
        require_timestamp(obj.get("written_at"))
        lines.append(obj)
    # Only inspect producer record directory names, never forward/check directories.
    seen = {line["dir"] for line in lines}
    for dd in sorted(root.glob("D=*")):
        _path(root, dd.name)
        if not dd.is_dir():
            continue
        if dd.name not in seen:
            raise _Invalid(f"orphan record directory: {dd.name}")
        for ad in sorted(dd.glob("attempt-*")):
            rel = f"{dd.name}/{ad.name}"
            _path(root, rel)
            if rel not in seen:
                raise _Invalid(f"orphan record directory: {rel}")
    return lines


def _record(root: Path, line: dict, batch: Batch) -> dict:
    relative = f"{line['dir']}/record.json"
    raw = read_bytes(_path(root, relative), root=root, batch=batch)
    if sha256_bytes(raw) != line["record_sha256"]:
        raise _Invalid(f"{relative}: record_sha256 differs from ledger")
    rec = json.loads(raw)
    if not isinstance(rec, dict) or rec.get("schema_version") != "kr-next-shadow-record.v1":
        raise _Invalid(f"{relative}: unknown record schema")
    if any(
        rec.get(key) != line.get(key) or type(rec.get(key)) is not type(line.get(key))
        for key in ("D", "attempt", "valid", "written_at")
    ):
        raise _Invalid(f"{relative}: D/attempt/valid/written_at differ from ledger")
    _day(rec.get("t"), "t")
    if rec["t"] >= rec["D"]:
        raise _Invalid(f"{relative}: t must precede D")
    if rec.get("d0") is not None:
        _day(rec["d0"], "d0")
        if rec["d0"] > rec["D"]:
            raise _Invalid(f"{relative}: d0 is later than D")
    for key in ("decision_at", "deadline_at", "written_at"):
        require_timestamp(rec.get(key))
    bundle = rec.get("bundle")
    if not isinstance(bundle, dict):
        raise _Invalid(f"{relative}: bundle metadata is missing")
    _digest(bundle.get("manifest_sha256"), "bundle.manifest_sha256")
    files = rec.get("files")
    if not isinstance(files, dict) or not {"panel.parquet", "predictions.parquet"} <= files.keys():
        raise _Invalid(f"{relative}: required output signatures are missing")
    # Reject extra paths before hashing anything named by the record.
    if not files.keys() <= _OUTPUTS:
        raise _Invalid(f"{relative}: undeclared output name in files")
    if line["has_basket"] != ("basket.parquet" in files):
        raise _Invalid(f"{relative}: has_basket differs from record")
    for name, want in sorted(files.items()):
        _digest(want, f"files.{name}")
        path = _path(root, f"{line['dir']}/{name}")
        if _hash_file(path) != want:
            raise _Invalid(f"{line['dir']}/{name}: sha256 differs from record")
        batch.sources.append(
            {
                "path": f"{line['dir']}/{name}",
                "sha256": want,
                "hash_kind": "bytes_sha256",
                "bytes": path.stat().st_size,
                "kind": "shadow_output",
                "status": "verified",
            }
        )
    record_source = next(source for source in reversed(batch.sources) if source["path"] == relative)
    record_source.update(
        {
            "kind": "shadow_record",
            "status": "verified",
            "producer_schema": rec["schema_version"],
            "D": rec["D"],
            "t": rec["t"],
            "bundle_hash": bundle["manifest_sha256"],
            "bundle": bundle,
            "code": rec.get("code"),
            "panel_manifest": rec.get("panel_manifest"),
            "panel_manifest_sha256": rec.get("panel_manifest_sha256"),
            "files": files,
        }
    )
    return rec


def _predictions(root: Path, line: dict, record: dict) -> list[dict]:
    # Column projection is a privacy boundary, not just a performance optimization.
    path = _path(root, f"{line['dir']}/predictions.parquet")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise _Invalid("prediction metadata file exceeds 64 MiB")
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024:
        raise _Invalid("prediction metadata file exceeds 64 MiB")
    if sha256_bytes(raw) != record["files"]["predictions.parquet"]:
        raise _Invalid("prediction file changed after integrity verification")
    rows = pl.read_parquet(io.BytesIO(raw), columns=_COLUMNS).to_dicts()
    keys = set()
    for row in rows:
        ticker, market, rank, score = (row[key] for key in _COLUMNS)
        if not isinstance(ticker, str) or not ticker or market not in {"KOSPI", "KOSDAQ"}:
            raise _Invalid("prediction ticker/market is missing or invalid")
        if type(rank) is not int or rank < 1:
            raise _Invalid("prediction rank must be a positive integer")
        if type(score) not in (float, int) or not math.isfinite(score):
            raise _Invalid("prediction score must be finite")
        if (ticker, market) in keys:
            raise _Invalid("duplicate prediction ticker/market")
        keys.add((ticker, market))
    return rows


def read_shadow(records_root: Path, *, observed_at: str, historical_link: bool = True) -> Batch:
    """Link all intact attempts; only P6's first valid attempt per D counts.

    Ledger corruption rejects the whole connection. Missing/corrupt record files
    remain explicit source errors and cannot promote a subsequent attempt.
    """
    require_timestamp(observed_at)
    batch = Batch()
    root = Path(records_root).absolute()
    try:
        if root.is_symlink():
            raise _Invalid("records root must not be a symlink")
        lines = _ledger(root, batch)
    except (OSError, ValueError, TypeError) as exc:
        _error(batch, "ledger.jsonl", exc)
        return batch
    counting = counting_records(lines)
    verified = []
    for line in lines:
        try:
            rec = _record(root, line, batch)
            verified.append((line, rec))
        except (OSError, ValueError, TypeError) as exc:
            _error(batch, f"{line['dir']}/record.json", exc, line=line)
    # A record byte-hash mismatch means the ledger no longer authenticates its
    # contents. Reject the connection just as the frozen ledger verifier does.
    if any("record_sha256 differs from ledger" in item["message"] for item in batch.errors):
        return batch
    for line, rec in verified:
        try:
            rows = _predictions(root, line, rec)
        except (OSError, ValueError, TypeError, pl.exceptions.PolarsError) as exc:
            _error(batch, f"{line['dir']}/predictions.parquet", exc, line=line)
            continue
        for row in rows:
            fields = {
                "bundle_hash": rec["bundle"]["manifest_sha256"],
                "record_sha256": line["record_sha256"],
                "ticker": row["ticker"],
                "market": row["market"],
            }
            batch.predictions.append(
                {
                    "prediction_ref": identity("shadow", fields),
                    "kind": "shadow",
                    "market": "KR",
                    "security_market": row["market"],
                    "model_id": "kr-next",
                    "bundle_hash": fields["bundle_hash"],
                    "record_sha256": line["record_sha256"],
                    "symbol": row["ticker"],
                    "rank": row["rank"],
                    "score": row["score"],
                    "report_date": rec["D"],
                    "feature_asof_date": rec["t"],
                    "observed_at": observed_at,
                    "historical_link": historical_link,
                    "valid": rec["valid"],
                    "counting": counting.get(rec["D"]) is line,
                    "attempt": rec["attempt"],
                    "d0": rec.get("d0"),
                    "is_grid_date": rec.get("is_grid_date"),
                    "grid_k": rec.get("grid_k"),
                    "decision_at": rec["decision_at"],
                    "deadline_at": rec["deadline_at"],
                    "written_at": rec["written_at"],
                    "quality_reasons": [],
                    "checker_version": CHECKER_VERSION,
                    "source_evidence_sha256": line["record_sha256"],
                    "evaluation_status": "awaiting_check",
                    "source_status": "linked",
                    "producer_schema": rec["schema_version"],
                    "source_identity_version": SOURCE_IDENTITY_VERSION,
                    "adapter_version": ADAPTER_VERSION,
                    "bundle_hash_kind": "file_bytes_sha256",
                }
            )
    return batch
