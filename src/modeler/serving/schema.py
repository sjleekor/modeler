"""Shared, dependency-free contracts for daily market reports."""
from __future__ import annotations

import math
from datetime import date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

SCHEMA_VERSION = "1.0"
SEOUL = ZoneInfo("Asia/Seoul")
PUBLICATION_STATES = {"unresolved", "allowed", "withheld"}
REPORT_STATES = {"ok", "partial", "stale", "withheld", "failed", "unavailable"}


def validate_report(report: dict[str, Any]) -> None:
    required = {"schema_version", "market", "report_date", "decision_at", "feature_asof_date",
                "status", "model_id", "model_version", "rankings", "quality", "provenance",
                "publication"}
    missing = required - report.keys()
    if missing:
        raise ValueError(f"missing report fields: {', '.join(sorted(missing))}")
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported schema_version")
    if report["market"] not in {"KR", "US"}:
        raise ValueError("market must be KR or US")
    day = date.fromisoformat(report["report_date"])
    if day.isoformat() != report["report_date"]:
        raise ValueError("report_date must use YYYY-MM-DD")
    instant = datetime.fromisoformat(report["decision_at"])
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("decision_at must include a timezone")
    instant = instant.astimezone(SEOUL)
    if instant.date() != day or instant.time().replace(tzinfo=None) != time(10, 0):
        raise ValueError("decision_at must be report_date at 10:00 Asia/Seoul")
    if report["feature_asof_date"] is not None:
        asof = date.fromisoformat(report["feature_asof_date"])
        if asof.isoformat() != report["feature_asof_date"]:
            raise ValueError("feature_asof_date must use YYYY-MM-DD")
    if report["status"] not in REPORT_STATES:
        raise ValueError("unknown report status")
    for field in ("model_id", "model_version"):
        if not isinstance(report[field], str) or not report[field]:
            raise ValueError(f"{field} is required")
    if not isinstance(report["rankings"], list):
        raise ValueError("rankings must be a list")
    last_rank = 0
    symbols: set[str] = set()
    for row in report["rankings"]:
        if not isinstance(row, dict) or not {"rank", "symbol", "name", "score"} <= row.keys():
            raise ValueError("each ranking requires rank, symbol, name, and score")
        if isinstance(row["rank"], bool) or not isinstance(row["rank"], int) or row["rank"] <= last_rank:
            raise ValueError("ranking ranks must be positive and strictly increasing")
        if not isinstance(row["symbol"], str) or not row["symbol"]:
            raise ValueError("ranking symbol is required")
        if not isinstance(row["name"], str):
            raise ValueError("ranking name must be a string")
        if row["symbol"] in symbols:
            raise ValueError("ranking symbols must be unique")
        if isinstance(row["score"], bool) or not isinstance(row["score"], (int, float)) or not math.isfinite(row["score"]):
            raise ValueError("ranking score must be a finite number")
        symbols.add(row["symbol"])
        last_rank = row["rank"]
    for field in ("quality", "provenance"):
        if not isinstance(report[field], dict):
            raise ValueError(f"{field} must be an object")
    publication = report["publication"]
    if not isinstance(publication, dict) or publication.get("status") not in PUBLICATION_STATES:
        raise ValueError("publication.status must be unresolved, allowed, or withheld")
    if not isinstance(publication.get("evidence"), list):
        raise ValueError("publication.evidence must be a list")
    inference_started_at = report.get("inference_started_at")
    if inference_started_at is not None:
        if not isinstance(inference_started_at, str):
            raise ValueError("inference_started_at must be an ISO timestamp")
        started = datetime.fromisoformat(inference_started_at)
        if started.tzinfo is None or started.utcoffset() is None:
            raise ValueError("inference_started_at must include a timezone")


def report_template(*, market: str, report_date: str, decision_at: datetime,
                    feature_asof_date: str | None, model_id: str,
                    model_version: str) -> dict[str, Any]:
    """Return a valid, empty report with publication blocked pending evidence."""
    report = {"schema_version": SCHEMA_VERSION, "market": market, "report_date": report_date,
              "decision_at": decision_at.isoformat(), "feature_asof_date": feature_asof_date,
              "status": "unavailable", "model_id": model_id, "model_version": model_version,
              "rankings": [], "quality": {}, "provenance": {},
              "publication": {"status": "unresolved", "evidence": []}}
    validate_report(report)
    return report
