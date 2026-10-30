"""Pure publication selection and KR h20 scheduling; no source or price reads."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any

from modeler.feedback.core import identity as source_identity
from modeler.feedback_outcomes.contracts import (
    BUNDLE_SHA256,
    HORIZON,
    MODEL_ID,
    MODEL_VERSION,
    REPORT_END,
    REPORT_START,
    RULE_ID,
    RULE_SHA256,
    SEOUL,
)
from modeler.serving.calendars import SessionCalendar

MODEL, VERSION, BUNDLE, RULE = MODEL_ID, MODEL_VERSION, BUNDLE_SHA256, RULE_ID
START, END = date.fromisoformat(REPORT_START), date.fromisoformat(REPORT_END)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")


def _day(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError("date must be a canonical ISO string")
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError("date must be a canonical ISO string")
    return parsed


def _instant(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must include a timezone")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(SEOUL)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Calendar:
    """Validated explicit operating-calendar metadata, never a calendar file loader."""

    calendar: SessionCalendar
    calendar_sha256: str | None = None
    calendar_code_version: str | None = None
    holiday_source_status: str | None = None
    confirmed_through: date | None = None
    synthetic_fixture: bool = False

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> Calendar:
        if not isinstance(manifest, dict) or manifest.get("market") != "KR":
            raise ValueError("only the KR operating calendar is accepted")
        _day(manifest.get("coverage_start"))
        _day(manifest.get("coverage_end"))
        for key in ("sessions", "unconfirmed_dates"):
            values = manifest.get(key, [])
            if not isinstance(values, list):
                raise ValueError(f"{key} must be a list")
            for value in values:
                _day(value)
        overrides = manifest.get("overrides", [])
        if not isinstance(overrides, list):
            raise ValueError("overrides must be a list")
        for override in overrides:
            if not isinstance(override, dict):
                raise ValueError("override must be an object")
            _day(override.get("date"))
        sha = manifest.get("calendar_sha256")
        if sha is not None and (not isinstance(sha, str) or not _HASH.fullmatch(sha)):
            raise ValueError("calendar_sha256 must be a lowercase SHA-256")
        code = manifest.get("calendar_code_version")
        if code is not None and (not isinstance(code, str) or not code.strip()):
            raise ValueError("calendar_code_version must be nonempty")
        source = manifest.get("source", {})
        if not isinstance(source, dict):
            raise ValueError("calendar source must be an object")
        status = source.get("holiday_source_status")
        if status is not None and not isinstance(status, str):
            raise ValueError("holiday_source_status must be a string")
        confirmed_through = (
            _day(source["confirmed_through"]) if "confirmed_through" in source else None
        )
        synthetic = manifest.get("synthetic_fixture", False)
        if not isinstance(synthetic, bool):
            raise ValueError("calendar synthetic_fixture must be boolean")
        return cls(
            SessionCalendar.from_manifest(deepcopy(manifest)),
            sha,
            code,
            status,
            confirmed_through,
            synthetic,
        )

    def _confirmation_reasons_between(self, start: date, end: date) -> list[str]:
        reasons = []
        if not self.synthetic_fixture:
            if "2027_provisional" in (self.holiday_source_status or "") and end.year >= 2027:
                reasons.append("calendar_holiday_provisional")
            elif end > (self.confirmed_through or date(2026, 12, 31)):
                reasons.append("calendar_holiday_coverage_unconfirmed")
        if any(
            not self.calendar.session(day).confirmed
            for day in self.calendar.sessions
            if start <= day <= end
        ):
            reasons.append("calendar_hours_unconfirmed")
        return reasons

    def _confirmed_between(self, start: date, end: date) -> bool:
        return not self._confirmation_reasons_between(start, end)


def entry_exit(report_date: str, verified_by: str, calendar: Calendar) -> dict[str, Any]:
    """Schedule E and E+20 from metadata; uncertainty never invents sessions."""
    result: dict[str, Any] = {
        "status": "pending",
        "entry_session": None,
        "exit_session": None,
        "reasons": [],
    }
    try:
        day, observed = _day(report_date), _instant(verified_by)
    except (TypeError, ValueError):
        result["reasons"] = ["invalid_schedule_metadata"]
        return result
    cal = calendar.calendar
    observed_day = observed.date()
    if observed_day < day:
        result["reasons"] = ["verified_before_report_date"]
        return result
    if not cal.knows(day) or not cal.knows(observed_day):
        result["reasons"] = ["calendar_coverage_unknown"]
        return result
    if observed_day == day and observed.time() <= time(14, 30) and cal.is_session(day):
        entry = day
    else:
        entry = next((s for s in sorted(cal.sessions) if s > observed_day), None)
    if entry is None:
        result["reasons"] = ["entry_session_unknown"]
        return result
    if not calendar._confirmed_between(day, entry):
        result["reasons"] = calendar._confirmation_reasons_between(day, entry)
        return result
    result["entry_session"] = entry.isoformat()
    following = sorted(s for s in cal.sessions if s > entry)
    if len(following) < HORIZON:
        result["reasons"] = ["exit_session_unknown"]
        return result
    exit_day = following[HORIZON - 1]
    if not calendar._confirmed_between(entry, exit_day):
        result["reasons"] = calendar._confirmation_reasons_between(entry, exit_day)
        return result
    result.update(status="scheduled", exit_session=exit_day.isoformat())
    return result


def _evidence_reasons(event: dict[str, Any]) -> list[str]:
    reasons = []
    for key in ("publication_ref", "receipt_ref", "unit_sha256"):
        if not isinstance(event.get(key), str) or not _HASH.fullmatch(event[key]):
            reasons.append(f"invalid_{key}")
    revision = event.get("revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision <= 0:
        reasons.append("invalid_revision")
    commit = event.get("reports_commit", event.get("commit"))
    if not isinstance(commit, str) or not _COMMIT.fullmatch(commit):
        reasons.append("invalid_reports_commit")
    try:
        observed = _instant(event.get("verified_by"))
        started = _instant(event.get("collection_started_at"))
        if started > observed:
            reasons.append("collection_started_after_verification")
    except (ValueError, TypeError):
        reasons.append("invalid_observation_timestamp")
    if event.get("coverage_status") != "covered":
        reasons.append("publication_unverified")
    return reasons


def _publication_reasons(event: dict[str, Any], calendar: Calendar, synthetic: bool) -> list[str]:
    reasons = _evidence_reasons(event)
    if event.get("historical_replay") is True:
        reasons.append("historical_replay")
    if event.get("synthetic_fixture") is True and not synthetic:
        reasons.append("synthetic_fixture")
    if synthetic and event.get("synthetic_fixture") is not True:
        reasons.append("synthetic_fixture_required")
    if any(
        not isinstance(event.get(k, False), bool)
        for k in ("synthetic_fixture", "historical_replay")
    ):
        reasons.append("invalid_fixture_flag")
    if (
        event.get("market"),
        event.get("model_id"),
        event.get("model_version"),
        event.get("bundle_hash"),
    ) != ("KR", MODEL, VERSION, BUNDLE):
        reasons.append("sealed_model_or_bundle")
    for key in ("report_sha256", "input_sha256"):
        if not isinstance(event.get(key), str) or not _HASH.fullmatch(event[key]):
            reasons.append(f"invalid_{key}")
    if not isinstance(event.get("invocation_id"), str) or not event["invocation_id"].strip():
        reasons.append("invalid_invocation_id")
    if event.get("model_status") not in {"ok", "success", "partial"}:
        reasons.append("model_not_publishable")
    try:
        day = _day(event.get("report_date"))
        signal = _day(event.get("feature_asof_date"))
        decision = _instant(event.get("decision_at"))
        observed = _instant(event.get("verified_by"))
        if not START <= day <= END:
            reasons.append("outside_report_window")
        if decision != datetime.combine(day, time(10), SEOUL):
            reasons.append("invalid_decision_at")
        if observed < decision:
            reasons.append("verified_before_decision")
        previous = calendar.calendar.previous_session(day)
        if previous is None or not calendar._confirmed_between(previous, day):
            reasons.append("signal_session_unknown")
            if previous is not None:
                reasons.extend(calendar._confirmation_reasons_between(previous, day))
        elif signal != previous:
            reasons.append("stale_signal")
    except (ValueError, TypeError):
        reasons.append("invalid_prediction_timestamp")
    lag = event.get("freshness_lag_sessions")
    if isinstance(lag, bool) or not isinstance(lag, int) or lag != 0:
        reasons.append("stale_signal")
    rows = event.get("rows")
    if not isinstance(rows, list) or not rows:
        reasons.append("rankings_not_published")
    else:
        ranks, securities, refs = set(), set(), set()
        for row in rows:
            if not isinstance(row, dict):
                reasons.append("invalid_published_row")
                continue
            rank, symbol, market = row.get("rank"), row.get("symbol"), row.get("security_market")
            ref, score = row.get("prediction_ref"), row.get("score")
            valid = (
                isinstance(rank, int)
                and not isinstance(rank, bool)
                and 1 <= rank <= 100
                and isinstance(symbol, str)
                and re.fullmatch(r"[0-9]{6}", symbol)
                and market in {"KOSPI", "KOSDAQ"}
                and isinstance(ref, str)
                and _HASH.fullmatch(ref)
                and isinstance(score, (float, int))
                and not isinstance(score, bool)
                and math.isfinite(score)
                and isinstance(row.get("quality_reasons"), list)
                and all(isinstance(item, str) for item in row["quality_reasons"])
            )
            if not valid:
                reasons.append("invalid_published_row")
                continue
            identity_fields = {
                "market": "KR",
                "security_market": market,
                "model_id": event.get("model_id"),
                "bundle_hash": event.get("bundle_hash"),
                "report_sha256": event.get("report_sha256"),
                "invocation_id": event.get("invocation_id"),
                "symbol": symbol,
            }
            if any(
                value is None or value == "" for value in identity_fields.values()
            ) or ref != source_identity("prediction", identity_fields):
                reasons.append("prediction_identity_mismatch")
            if rank in ranks or (symbol, market.lower()) in securities or ref in refs:
                reasons.append("duplicate_published_row")
            ranks.add(rank)
            securities.add((symbol, market.lower()))
            refs.add(ref)
    return sorted(set(reasons))


def select_cohorts(
    events: list[dict[str, Any]], calendar: Calendar, *, synthetic: bool = False
) -> dict[str, list[dict[str, Any]]]:
    """Select first valid observed reports without changing original published ranks."""
    cohorts: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    candidates: list[tuple[datetime, dict[str, Any], dict[str, Any]]] = []
    uncertain_publications: dict[str, list[datetime]] = defaultdict(list)
    withdrawals = []
    for event in deepcopy(events):
        decision = {
            "publication_ref": event.get("publication_ref"),
            "report_sha256": event.get("report_sha256"),
            "report_date": event.get("report_date"),
            "status": "excluded",
            "reasons": [],
            "cohort_refs": [],
            "synthetic_fixture": synthetic,
        }
        decisions.append(decision)
        if event.get("kind") == "withdrawal":
            withdrawals.append((event, decision))
            continue
        if event.get("kind") != "publication":
            decision["reasons"] = ["unsupported_event_kind"]
            continue
        reasons = _publication_reasons(event, calendar, synthetic)
        if reasons:
            decision["reasons"] = reasons
            if any(r in reasons for r in ("publication_unverified", "signal_session_unknown")):
                decision["status"] = "pending"
            # An otherwise eligible earlier report with unverified collection
            # history cannot silently disappear from the first-report decision.
            # Explicit failures/stale/other models remain ordinary exclusions.
            if reasons == ["publication_unverified"]:
                uncertain_publications[event["report_date"]].append(_instant(event["verified_by"]))
            continue
        candidates.append((_instant(event["verified_by"]), event, decision))

    grouped: dict[str, list[tuple[datetime, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    bindings: dict[str, set[str]] = defaultdict(set)
    for _, event, _ in candidates:
        bindings[event["report_sha256"]].add(
            _digest(
                {
                    k: event[k]
                    for k in (
                        "market",
                        "model_id",
                        "model_version",
                        "bundle_hash",
                        "report_date",
                        "feature_asof_date",
                        "input_sha256",
                        "invocation_id",
                        "decision_at",
                        "rows",
                    )
                }
            )
        )
    accepted = []
    for candidate in candidates:
        _, event, decision = candidate
        if len(bindings[event["report_sha256"]]) > 1:
            decision.update(status="excluded", reasons=["report_binding_conflict"])
            continue
        grouped[candidate[1]["report_date"]].append(candidate)
        accepted.append(candidate)
    selected_dates: dict[str, str] = {}
    selected_keys: set[tuple[Any, ...]] = set()
    selected_hashes: dict[str, list[dict[str, Any]]] = {}
    for observed, event, decision in sorted(
        accepted,
        key=lambda item: (
            item[0],
            item[1]["report_date"],
            item[1]["report_sha256"],
            item[1]["publication_ref"],
        ),
    ):
        day, report_hash = event["report_date"], event["report_sha256"]
        earliest = min(item[0] for item in grouped[day])
        if any(stamp <= earliest for stamp in uncertain_publications.get(day, [])):
            decision.update(
                status="pending",
                reasons=["publication_unverified", "earlier_publication_unverified"],
            )
            continue
        tied = {item[1]["report_sha256"] for item in grouped[day] if item[0] == earliest}
        if len(tied) > 1 and day not in selected_dates:
            decision.update(status="pending", reasons=["ambiguous_first_publication"])
            continue
        if report_hash in selected_hashes:
            decision.update(status="duplicate", reasons=["same_report_observation"])
            continue
        if day in selected_dates:
            decision.update(
                status="correction",
                reasons=["first_valid_report_preserved"],
                supersedes=selected_dates[day],
            )
            continue
        key = tuple(
            event[k] for k in ("model_id", "bundle_hash", "feature_asof_date", "input_sha256")
        )
        if key in selected_keys:
            decision.update(status="duplicate", reasons=["same_prediction_input"])
            continue
        schedule = entry_exit(day, event["verified_by"], calendar)
        selected_dates[day] = report_hash
        selected_keys.add(key)
        selected_hashes[report_hash] = []
        decision.update(status="selected", reasons=[])
        for size in (20, 100):
            slots = [deepcopy(row) for row in event["rows"] if row["rank"] <= size]
            slots.sort(key=lambda row: row["rank"])
            missing = sorted(set(range(1, size + 1)) - {row["rank"] for row in slots})
            reasons = list(schedule["reasons"])
            status = "selected" if schedule["status"] == "scheduled" else "pending"
            if missing:
                reasons.append("source_incomplete")
                status = "source_incomplete"
            identity = {
                "rule_id": RULE,
                "rule_sha256": RULE_SHA256,
                "report_date": day,
                "report_sha256": report_hash,
                "model_id": MODEL,
                "model_version": VERSION,
                "bundle_hash": BUNDLE,
                "feature_asof_date": event["feature_asof_date"],
                "input_sha256": event["input_sha256"],
                "top_n": size,
                "synthetic_fixture": synthetic,
            }
            cohort = {
                **identity,
                "cohort_ref": _digest(identity),
                "cohort_size": size,
                "size": size,
                "n": size,
                "entry_session": schedule["entry_session"],
                "exit_session": schedule["exit_session"],
                "verified_by": event["verified_by"],
                "decision_at": event["decision_at"],
                "publication_ref": event["publication_ref"],
                "receipt_ref": event["receipt_ref"],
                "invocation_id": event["invocation_id"],
                "reports_commit": event["reports_commit"],
                "unit_sha256": event["unit_sha256"],
                "revision": event["revision"],
                "status": status,
                "reasons": reasons,
                "slots": slots,
                "missing_ranks": missing,
                "withdrawn_before_entry": False,
            }
            for key in ("entry", "exit"):
                session_day = schedule[f"{key}_session"]
                session = calendar.calendar.session(_day(session_day)) if session_day else None
                cohort[f"{key}_close_at"] = (
                    datetime.combine(session.day, session.close_at, SEOUL).isoformat()
                    if session is not None and session.confirmed
                    else None
                )
            cohorts.append(cohort)
            selected_hashes[report_hash].append(cohort)
            decision["cohort_refs"].append(cohort["cohort_ref"])

    for event, decision in withdrawals:
        reasons = _evidence_reasons(event)
        if (
            event.get("market"),
            event.get("model_id"),
            event.get("model_version"),
            event.get("bundle_hash"),
        ) != ("KR", MODEL, VERSION, BUNDLE):
            reasons.append("sealed_model_or_bundle")
        if any(
            not isinstance(event.get(k, False), bool)
            for k in ("synthetic_fixture", "historical_replay")
        ):
            reasons.append("invalid_fixture_flag")
        if event.get("historical_replay") is True or (
            event.get("synthetic_fixture") is True and not synthetic
        ):
            reasons.append("non_prospective_withdrawal")
        if synthetic and event.get("synthetic_fixture") is not True:
            reasons.append("synthetic_fixture_required")
        original = event.get("original_report_sha256")
        if not isinstance(original, str) or not _HASH.fullmatch(original):
            reasons.append("invalid_original_report_sha256")
        linked = selected_hashes.get(original, []) if isinstance(original, str) else []
        if not linked:
            reasons.append("original_report_not_selected")
        else:
            first = linked[0]
            if event.get("report_date") != first["report_date"]:
                reasons.append("withdrawal_report_date_mismatch")
            for field in (
                "report_sha256",
                "invocation_id",
                "input_sha256",
                "feature_asof_date",
                "decision_at",
            ):
                if field in event and event[field] != first[field]:
                    reasons.append("withdrawal_report_binding_mismatch")
        if reasons:
            decision.update(reasons=sorted(set(reasons)))
            continue
        observed = _instant(event["verified_by"])
        first = linked[0]
        if observed < _instant(first["verified_by"]):
            decision.update(reasons=["withdrawal_before_original_observation"])
            continue
        if first["entry_session"] is None:
            decision.update(status="pending", reasons=["entry_session_unknown"])
            continue
        entry_day = _day(first["entry_session"])
        session = calendar.calendar.session(entry_day)
        if session is None or not session.confirmed:
            decision.update(status="pending", reasons=["calendar_hours_unconfirmed"])
            continue
        close = datetime.combine(entry_day, session.close_at, SEOUL)
        if observed >= close:
            decision.update(status="withdrawal_after_entry", reasons=["original_cohort_preserved"])
            continue
        decision.update(status="withdrawn_before_entry", reasons=[])
        for cohort in linked:
            cohort["withdrawn_before_entry"] = True
            cohort["reasons"] = sorted(set(cohort["reasons"] + ["withdrawn_before_entry"]))
            if not cohort["missing_ranks"]:
                cohort["status"] = "withdrawn_before_entry"
            cohort["withdrawal_ref"] = event["publication_ref"]
            decision["cohort_refs"].append(cohort["cohort_ref"])
    return {"cohorts": cohorts, "decisions": decisions}
