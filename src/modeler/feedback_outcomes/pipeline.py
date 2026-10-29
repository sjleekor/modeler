"""Compose metadata selection and guarded endpoint diagnostics without training."""

from __future__ import annotations

from datetime import UTC, datetime

from .calculation import aggregate_cohort, calculate_row, summarize
from .contracts import OutcomeError, require_real_review, rule_pin, stamp
from .prices import EndpointExtract
from .selection import Calendar, select_cohorts


def selected_sessions(cohorts, cutoff):
    result = set()
    for cohort in cohorts:
        if cohort.get("entry_close_at") and cohort.get("exit_close_at"):
            if stamp(cohort["exit_close_at"]) < stamp(cutoff):
                result.update((cohort["entry_session"], cohort["exit_session"]))
    return sorted(result)


def evaluate(calendar_manifest, events, extract, *, cutoff, synthetic=False, now=None):
    """Synthetic calls are pure and must contain explicit fixture markers.

    Real entrypoints additionally need an authorized read plan before loading
    these values; the wall-clock gate here prevents accidental direct runs.
    """
    if type(synthetic) is not bool:
        raise OutcomeError("synthetic must be an explicit boolean")
    if not synthetic:
        clock = require_real_review(now or datetime.now(UTC))
        if stamp(cutoff) > clock:
            raise OutcomeError("outcome cutoff has not arrived")
        if calendar_manifest.get("synthetic_fixture") is True:
            raise OutcomeError("fixture calendar cannot be used for actual outcomes")
        if any(event.get("explicit_withdrawal_adapter") != "source_bound_v1" for event in events):
            raise OutcomeError("source-bound withdrawal evidence adapter is not ready")
    elif calendar_manifest.get("synthetic_fixture") is not True:
        raise OutcomeError("synthetic evaluation requires a fixture calendar")
    calendar = Calendar.from_manifest(calendar_manifest)
    selected = select_cohorts(events, calendar, synthetic=synthetic)
    sessions = selected_sessions(selected["cohorts"], cutoff)
    adapter = None
    if sessions and extract is not None:
        adapter = EndpointExtract(extract, cutoff=cutoff, synthetic=synthetic, sessions=sessions)
    rows = {}
    evaluated = []
    for cohort in selected["cohorts"]:
        cohort_rows = []
        for original in cohort["slots"]:
            ref = original["prediction_ref"]
            if ref not in rows:
                slot = {
                    **original,
                    **{
                        k: cohort.get(k)
                        for k in (
                            "rule_id",
                            "rule_sha256",
                            "report_date",
                            "report_sha256",
                            "model_id",
                            "model_version",
                            "bundle_hash",
                            "input_sha256",
                            "feature_asof_date",
                            "decision_at",
                            "verified_by",
                            "publication_ref",
                            "receipt_ref",
                            "invocation_id",
                            "reports_commit",
                            "unit_sha256",
                            "revision",
                            "synthetic_fixture",
                        )
                    },
                    "market": "KR",
                    "cohort_refs": [],
                }
                if not cohort.get("entry_session") or not cohort.get("exit_session"):
                    row = {
                        **slot,
                        "status": "pending",
                        "reason": "calendar_session_unknown",
                        "entry_session": cohort.get("entry_session"),
                        "exit_session": cohort.get("exit_session"),
                        "mature": False,
                        "entry_executed": False,
                        "exit_executed": False,
                        **dict.fromkeys(("gross", "net", "benchmark", "excess", "cost")),
                    }
                else:
                    observations = {
                        "status": "source_missing",
                        "reason": "prepared_endpoint_contract_unavailable",
                    }
                    if adapter is not None and stamp(cohort["exit_close_at"]) < stamp(cutoff):
                        observations = adapter.observations(
                            slot, cohort["entry_session"], cohort["exit_session"]
                        )
                    row = calculate_row(
                        slot,
                        entry_session={
                            "session": cohort["entry_session"],
                            "close_at": cohort["entry_close_at"],
                        },
                        exit_session={
                            "session": cohort["exit_session"],
                            "close_at": cohort["exit_close_at"],
                        },
                        observations=observations,
                        cutoff=cutoff,
                        withdrawn_before_entry=cohort["withdrawn_before_entry"],
                    )
                rows[ref] = row
            rows[ref]["cohort_refs"].append(cohort["cohort_ref"])
            cohort_rows.append(rows[ref])
        evaluated.append(aggregate_cohort(cohort, cohort_rows))
    output_rows = sorted(rows.values(), key=lambda r: (r["report_date"], r["rank"]))
    return {
        "schema": "kr-v1-outcome-evaluation.v1",
        "synthetic_fixture": synthetic,
        "rule": rule_pin(),
        "cutoff": cutoff,
        "calendar_sha256": calendar.calendar_sha256,
        "rows": output_rows,
        "cohorts": evaluated,
        "decisions": selected["decisions"],
        "summary": summarize(evaluated, output_rows),
        "real_price_reads": 0 if synthetic else None,
    }
