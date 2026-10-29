"""Shared identity and bounded, content-addressed source capture."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

SOURCE_IDENTITY_VERSION = "feedback-source.v1"
ADAPTER_VERSION = "feedback-adapter.v1"
MAX_METADATA_BYTES = 32 * 1024 * 1024
FORBIDDEN_COMPONENTS = frozenset({"forward", "checks", "labels", "raw", "derived", "datasets"})


class FeedbackError(ValueError):
    """An explicit source contract or integrity check failed."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def identity(kind: str, fields: dict) -> str:
    if not kind or not fields or any(v is None or v == "" for v in fields.values()):
        raise FeedbackError("identity fields are incomplete")
    return sha256_bytes(
        canonical_bytes(
            {"source_identity_version": SOURCE_IDENTITY_VERSION, "kind": kind, "fields": fields}
        )
    )


def require_timestamp(value: str) -> str:
    if not isinstance(value, str):
        raise FeedbackError("timestamp must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise FeedbackError("invalid timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FeedbackError("timestamp must include a timezone")
    return value


def require_hash(value: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise FeedbackError("invalid digest")
    return value


def finite_score(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


@dataclass
class Batch:
    predictions: list[dict] = field(default_factory=list)
    publications: list[dict] = field(default_factory=list)
    findings: list[dict] = field(default_factory=list)
    artifacts: dict[str, bytes] = field(default_factory=dict)
    sources: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def extend(self, other: Batch) -> None:
        for name in ("predictions", "publications", "findings", "sources", "errors"):
            getattr(self, name).extend(getattr(other, name))
        self.artifacts.update(other.artifacts)


def source_path(path: Path, *, root: Path) -> Path:
    """Reject source escapes, symlinks, and evaluation/lake paths before opening."""
    path, root = Path(path).absolute(), Path(root).absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise FeedbackError("source is outside its declared root") from exc
    if ".." in path.parts or FORBIDDEN_COMPONENTS.intersection(path.parts):
        raise FeedbackError("source path is outside the metadata allowlist")
    if path.name.startswith(".env") or path.name in {"points.parquet", "ops.json", "pins.json"}:
        raise FeedbackError("source path is not an input artifact")
    current = root
    if root.is_symlink():
        raise FeedbackError("source root cannot be a symlink")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise FeedbackError("source symlinks are not allowed")
    if not path.resolve().is_relative_to(root.resolve()):
        raise FeedbackError("resolved source escapes its root")
    return path


def read_bytes(path: Path, *, root: Path, batch: Batch) -> bytes:
    path = source_path(path, root=root)
    if not path.is_file():
        raise FeedbackError("source_missing")
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise FeedbackError("source metadata exceeds the size limit")
    with path.open("rb") as stream:
        raw = stream.read(MAX_METADATA_BYTES + 1)
    if len(raw) > MAX_METADATA_BYTES:
        raise FeedbackError("source metadata exceeds the size limit")
    digest = sha256_bytes(raw)
    batch.artifacts[digest] = raw
    batch.sources.append(
        {
            "path": path.relative_to(Path(root).absolute()).as_posix(),
            "sha256": digest,
            "hash_kind": "bytes_sha256",
            "bytes": len(raw),
        }
    )
    return raw


def read_json(path: Path, *, root: Path, batch: Batch) -> dict:
    try:
        value = json.loads(read_bytes(path, root=root, batch=batch))
    except (ValueError, UnicodeError) as exc:
        if isinstance(exc, FeedbackError):
            raise
        raise FeedbackError("invalid source JSON") from exc
    if not isinstance(value, dict):
        raise FeedbackError("source JSON must be an object")
    return value
