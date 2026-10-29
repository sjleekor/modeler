"""Pure, metadata-only links to the human-owned issue document.

This module never opens a document, prediction artifact, evaluation, or URL.
Callers provide the bytes/metadata that their input contract permits reading.
Human issue/action states deliberately have no representation in these records.
"""

from __future__ import annotations

import hashlib
import html
import re
from collections import defaultdict
from datetime import datetime
from typing import Any

from .core import canonical_bytes, identity, require_timestamp

_MAPPING_SCHEMA = "prediction_feedback.issue_mapping"
_MAPPING_VERSION = "1"
_HEADERS = ("issue_id", "문제·분류 / 대상", "reason_code")
_PUBLISHED = frozenset({"published_verified", "published_evidence"})
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_REASON = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*\Z")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value.strip()


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _hex(value: Any, pattern: re.Pattern[str], field: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"invalid {field}")
    return value


def _cells(line: str) -> list[str]:
    # Escaped table separators are text, rather than additional columns.
    return [
        cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))
    ]


def _literal(cell: str) -> str:
    return cell[1:-1].strip() if cell.startswith("`") and cell.endswith("`") else cell


def compile_mapping(document: bytes, *, my_commit: str, expected_blob_sha1: str) -> dict:
    """Compile only the fixed issue ID/target/reason columns of pinned bytes.

    A Git blob digest includes its header; a bare content SHA-1 is not accepted.
    The mapping checksum protects an in-memory/exported mapping against accidental
    mutation. It does not claim authenticity beyond the caller's pinned document.
    """
    if not isinstance(document, bytes):
        raise ValueError("document must be bytes")
    _hex(my_commit, _HEX40, "my_commit")
    _hex(expected_blob_sha1, _HEX40, "expected_blob_sha1")
    actual = hashlib.sha1(
        b"blob " + str(len(document)).encode("ascii") + b"\0" + document
    ).hexdigest()
    if actual != expected_blob_sha1:
        raise ValueError("issue document Git blob hash mismatch")
    try:
        lines = document.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("issue document must be UTF-8") from exc

    headers: list[str] | None = None
    rules: list[dict[str, str]] = []
    seen_issues: set[str] = set()
    seen_scopes: set[tuple[str, str]] = set()
    table_count = 0
    for line in lines:
        if not line.lstrip().startswith("|"):
            headers = None
            continue
        cells = _cells(line)
        if all(header in cells for header in _HEADERS):
            if any(cells.count(header) != 1 for header in _HEADERS):
                raise ValueError("duplicate issue document header")
            table_count += 1
            if table_count != 1:
                raise ValueError("ambiguous issue document tables")
            headers = cells
            continue
        if headers is None:
            continue
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        if len(cells) != len(headers):
            raise ValueError("malformed issue table row")
        row = dict(zip(headers, cells, strict=True))
        issue_id = _text(_literal(row["issue_id"]), "issue_id")
        if issue_id in seen_issues:
            raise ValueError("duplicate issue_id")
        seen_issues.add(issue_id)
        target_cell = _literal(row["문제·분류 / 대상"])
        # The left side is a problem description, never a model-matching rule.
        target = _text(target_cell.rsplit(" / ", 1)[-1], "target")
        reason = _literal(row["reason_code"])
        if reason in {"-", "—"}:
            continue
        if not _REASON.fullmatch(reason):
            raise ValueError("reason_code must be one unambiguous code")
        scope = (target, reason)
        if scope in seen_scopes:
            raise ValueError("ambiguous target/reason_code mapping")
        seen_scopes.add(scope)
        rules.append({"issue_id": issue_id, "target": target, "reason_code": reason})
    if table_count != 1:
        raise ValueError("fixed issue mapping table is missing")
    mapping = {
        "schema": _MAPPING_SCHEMA,
        "version": _MAPPING_VERSION,
        "my_commit": my_commit,
        "document_blob_sha1": actual,
        "document_sha256": hashlib.sha256(document).hexdigest(),
        "rules": sorted(
            rules, key=lambda rule: (rule["target"], rule["reason_code"], rule["issue_id"])
        ),
    }
    mapping["mapping_sha256"] = _hash(mapping)
    return mapping


def _validate_mapping(mapping: dict) -> None:
    if not isinstance(mapping, dict):
        raise ValueError("mapping must be a dictionary")
    expected_keys = {
        "schema",
        "version",
        "my_commit",
        "document_blob_sha1",
        "document_sha256",
        "rules",
        "mapping_sha256",
    }
    if set(mapping) != expected_keys:
        raise ValueError("invalid mapping fields")
    if mapping["schema"] != _MAPPING_SCHEMA or mapping["version"] != _MAPPING_VERSION:
        raise ValueError("unsupported issue mapping schema/version")
    _hex(mapping["my_commit"], _HEX40, "mapping my_commit")
    _hex(mapping["document_blob_sha1"], _HEX40, "mapping document_blob_sha1")
    _hex(mapping["document_sha256"], _HEX64, "mapping document_sha256")
    _hex(mapping["mapping_sha256"], _HEX64, "mapping_sha256")
    unsigned = {key: value for key, value in mapping.items() if key != "mapping_sha256"}
    if _hash(unsigned) != mapping["mapping_sha256"]:
        raise ValueError("issue mapping digest mismatch")
    if not isinstance(mapping["rules"], list):
        raise ValueError("mapping rules must be a list")
    scopes = set()
    issues = set()
    for rule in mapping["rules"]:
        if not isinstance(rule, dict) or set(rule) != {"issue_id", "target", "reason_code"}:
            raise ValueError("invalid mapping rule fields")
        for key, value in rule.items():
            _text(value, f"rule {key}")
        if not _REASON.fullmatch(rule["reason_code"]):
            raise ValueError("invalid mapping reason_code")
        scope = (rule["target"], rule["reason_code"])
        if scope in scopes or rule["issue_id"] in issues:
            raise ValueError("ambiguous mapping rules")
        scopes.add(scope)
        issues.add(rule["issue_id"])


def _matches_target(prediction: dict, target: str) -> bool:
    """Fail closed: literal scopes or the single documented operating-v1 alias."""
    if target in {"운영 v1 가격 피쳐", "운영 v1 행동 관찰"}:
        return prediction.get("market") == "KR" and prediction.get("model_id") == "kr_daily_h20_v1"
    # Arbitrary caller-supplied target labels cannot override model identity.
    return target == prediction.get("model_id")


def build_occurrences(predictions: list[dict], mapping: dict) -> list[dict]:
    """Index publication-linked warnings, preserving unknown warnings as unmapped.

    Generated/local/unverified rows remain in the diagnostic only. Publication
    verification is the reference adapter's responsibility, before this function.
    """
    _validate_mapping(mapping)
    records: dict[str, dict] = {}
    for prediction in predictions:
        if prediction.get("publication_status") not in _PUBLISHED:
            continue
        reasons = prediction.get("quality_reasons", [])
        if not isinstance(reasons, list):
            raise ValueError("quality_reasons must be a list")
        if not reasons:
            continue
        ref = _text(prediction.get("prediction_ref"), "prediction_ref")
        checker = _text(prediction.get("checker_version"), "checker_version")
        evidence = _hex(prediction.get("source_evidence_sha256"), _HEX64, "source_evidence_sha256")
        for reason in reasons:
            reason = _text(reason, "quality reason")
            if not _REASON.fullmatch(reason):
                raise ValueError("invalid quality reason")
            key = {
                "prediction_ref": ref,
                "reason_code": reason,
                "checker_version": checker,
                "source_evidence_sha256": evidence,
            }
            occurrence_ref = identity("occurrence", key)
            matches = [
                rule["issue_id"]
                for rule in mapping["rules"]
                if rule["reason_code"] == reason and _matches_target(prediction, rule["target"])
            ]
            if len(matches) > 1:
                raise ValueError("prediction matches multiple issue scopes")
            record = {
                "schema": "prediction_feedback.occurrence",
                "version": "1",
                "occurrence_ref": occurrence_ref,
                **key,
                "mapping_status": "mapped" if matches else "unmapped",
                "issue_id": matches[0] if matches else None,
                "mapping_sha256": mapping["mapping_sha256"],
                "my_commit": mapping["my_commit"],
                "document_blob_sha1": mapping["document_blob_sha1"],
            }
            # Observation times are metadata, and never part of occurrence identity.
            if prediction.get("observed_at") is not None:
                record["observed_at"] = require_timestamp(prediction["observed_at"])
            previous = records.get(occurrence_ref)
            if previous:
                if previous["issue_id"] != record["issue_id"]:
                    raise ValueError("conflicting prediction scopes for one occurrence")
                old_time = previous.get("observed_at")
                new_time = record.get("observed_at")
                if old_time is not None and (
                    new_time is None
                    or datetime.fromisoformat(old_time.replace("Z", "+00:00"))
                    <= datetime.fromisoformat(new_time.replace("Z", "+00:00"))
                ):
                    continue
            records[occurrence_ref] = record
    return [records[key] for key in sorted(records)]


def _markdown(value: Any) -> str:
    if value is None or value == "":
        return "unknown"
    if isinstance(value, (list, tuple)):
        return ", ".join(_markdown(item) for item in value) or "none"
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError("diagnostic cells must contain scalar metadata")
    # Text-only cells: no injected links, raw HTML, table rows, or Markdown markup.
    text = html.escape(str(value), quote=True).replace("\r", " ").replace("\n", " ")
    return re.sub(r"([\\`*_{}\[\]()#+.!|~>])", r"\\\1", text)


def render_diagnostic(
    predictions: list[dict], occurrences: list[dict], *, issue_document_ref: str
) -> str:
    """Render allowlisted metadata without copying human states or outcome values."""
    _text(issue_document_ref, "issue_document_ref")
    linked: dict[str, list[str]] = defaultdict(list)
    for occurrence in occurrences:
        ref = occurrence.get("prediction_ref")
        if not isinstance(ref, str):
            raise ValueError("occurrence prediction_ref is missing")
        label = occurrence.get("issue_id") or "unmapped"
        if label not in linked[ref]:
            linked[ref].append(label)
    lines = [
        "# 내부 예측 진단",
        "",
        "사람의 문제·조치 상태 정본: " + _markdown(issue_document_ref),
        "",
        "| 모델 | 종목 | 기준일 | 당시 순위 | 당시 점수 | 설명 참조 | 게시·유효성 "
        "| 품질 경고 | 연결 문제 | 사람 상태 문서 | 평가 접근 상태 | 기존 점검 참조 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for prediction in predictions:
        quality = prediction.get("quality_reasons")
        if prediction.get("quality_review") == "not_checked":
            quality = "not_checked"
        elif quality is None:
            quality = prediction.get("quality_status", "not_checked")
        if not isinstance(quality, (str, list)):
            raise ValueError("invalid diagnostic quality metadata")
        publication = prediction.get("publication_status", "unknown")
        validity = [publication]
        if prediction.get("source_status"):
            validity.append(prediction["source_status"])
        for key in ("valid", "counting"):
            if key in prediction:
                validity.append(f"{key}={prediction[key]}")
        fields = [
            prediction.get("model_id"),
            prediction.get("symbol", prediction.get("ticker")),
            prediction.get(
                "asof", prediction.get("feature_asof_date", prediction.get("report_date"))
            ),
            prediction.get("rank"),
            prediction.get("score"),
            prediction.get("explain_ref", prediction.get("explain_sha256")),
            validity,
            quality,
            sorted(linked.get(prediction.get("prediction_ref"), [])),
            issue_document_ref,
            prediction.get("evaluation_status", "not_defined"),
            prediction.get("check_ref"),
        ]
        lines.append("| " + " | ".join(_markdown(value) for value in fields) + " |")
    return "\n".join(lines) + "\n"
