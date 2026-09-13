"""§7.1 ``family_summary.parquet`` / ``family_cards.md`` — B-10 Stage 4.

One row and one card per Phase B family (8 today), collapsing everything the
run already computed — the readiness freeze, the assembled primary table, the
Stage 2 coverage diagnostics, the rank-correlation pairs — into the per-family
conclusion §6 B-10 specifies.

Two rules shape the whole module:

**A blocked family is the normal case, not an error path.** Every Phase B cell
is ``blocked_exploratory`` until the new raw lands, so the card's job right now
is to name the missing dependency and nothing else. Statistics fields stay
``None`` rather than 0/False — "not evaluated" and "evaluated, found nothing"
are different facts and a card that conflates them is worse than no card.

**Nothing is recomputed here.** Every number is read off a row some earlier
stage already produced, so a card can never disagree with the artifact it
summarizes. The only derived values are aggregations over a family's own cells
(best grade, min q, union of failed gates) and the ``next_step`` sentence.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from modeler.analysis.horizon_scan_config import HorizonScanConfig
from modeler.analysis.horizon_scan_phase_b import phase_b_families

FAMILY_SUMMARY_TABLE = "family_summary"

# Best-first. A family's grade is the best its cells reached, mirroring how
# "the family is Grade A" is read in 06_grade_a_deep_dive — one qualifying cell
# is what puts a family on the list.
_GRADE_ORDER: tuple[str, ...] = ("A", "B", "C", "D", "NE")

# fin_sue lives on an event grain, so its coverage comes from event_coverage
# rather than feature_coverage — a different table with different columns.
_EVENT_GRAIN_FAMILY = "fin_sue"


def _best_grade(grades: Iterable[str]) -> str:
    present = {g for g in grades if g}
    for grade in _GRADE_ORDER:
        if grade in present:
            return grade
    return "NE"


def _min_or_none(values: Iterable[Any]) -> Any:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def _feature_coverage_for(rows: Sequence[dict[str, Any]], feature: str) -> dict[str, Any]:
    """Roll ``feature_coverage``'s (market, year) rows up to one family line."""
    matching = [r for r in rows if r.get("feature") == feature and r.get("variant") == "native_t"]
    if not matching:
        return {
            "coverage_source": None,
            "effective_start": None,
            "coverage_ratio": None,
            "min_names_per_date": None,
            "observations": None,
        }
    panel_rows = sum(int(r.get("panel_rows") or 0) for r in matching)
    nonnull_rows = sum(int(r.get("nonnull_rows") or 0) for r in matching)
    return {
        "coverage_source": "feature_coverage",
        "effective_start": _min_or_none(r.get("first_value_date") for r in matching),
        "coverage_ratio": (nonnull_rows / panel_rows) if panel_rows else None,
        "min_names_per_date": _min_or_none(r.get("min_names_per_date") for r in matching),
        "observations": nonnull_rows,
    }


def _event_coverage_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "coverage_source": None,
            "effective_start": None,
            "coverage_ratio": None,
            "min_names_per_date": None,
            "observations": None,
        }
    events = sum(int(r.get("events") or 0) for r in rows)
    with_sue = sum(int(r.get("events_with_sue") or 0) for r in rows)
    return {
        "coverage_source": "event_coverage",
        "effective_start": _min_or_none(r.get("first_formation_date") for r in rows),
        "coverage_ratio": (with_sue / events) if events else None,
        # Names-per-date is a continuous-panel notion; an event family has no
        # equivalent, so it stays empty instead of borrowing a lookalike number.
        "min_names_per_date": None,
        "observations": with_sue,
    }


def _next_step(
    *, readiness: str, blocker: str, evaluated: int, grade: str, discoveries: int
) -> str:
    if readiness != "ready":
        return f"collect {blocker}" if blocker else "resolve missing dependency"
    if evaluated == 0:
        return "rerun --phase B; family is ready but no cell was scanned"
    if discoveries == 0:
        return "no primary discovery; keep as exploratory"
    if grade == "A":
        return "candidate for adoption; confirm on the next holdout window"
    if grade in ("B", "C"):
        return f"discovery at grade {grade}; resolve the failed gates before adopting"
    return "screen_pass failed; not a candidate"


def build_family_summary_rows(
    config: HorizonScanConfig,
    *,
    readiness_rows: Sequence[dict[str, Any]],
    assembled_rows: Sequence[dict[str, Any]] = (),
    feature_coverage_rows: Sequence[dict[str, Any]] = (),
    event_coverage_rows: Sequence[dict[str, Any]] = (),
    rank_correlation_rows: Sequence[dict[str, Any]] = (),
    formula_versions: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """One row per Phase B family, in config order.

    ``formula_versions`` is supplied by the caller rather than looked up: which
    feature module a family came from is a run-level fact, and a family whose
    module has no fingerprint must show ``—`` rather than a borrowed one. The
    caller routes fin/event/payout versions in ``_feature_formula_versions``.
    """
    versions = formula_versions or {}
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in readiness_rows:
        by_family.setdefault(row["family"], []).append(row)
    assembled_by_family: dict[str, list[dict[str, Any]]] = {}
    for row in assembled_rows:
        assembled_by_family.setdefault(row.get("family", ""), []).append(row)

    summary: list[dict[str, Any]] = []
    for family in phase_b_families(config):
        name = family["family"]
        cells = by_family.get(name, [])
        primary_feature = next(f["column"] for f in family["features"] if f["role"] == "primary")
        secondary = [f["column"] for f in family["features"] if f["role"] != "primary"]

        ready_cells = [c for c in cells if c["role"] == "ready_primary"]
        blocked_cells = [c for c in cells if c["role"] != "ready_primary"]
        missing = sorted(
            {
                dep
                for cell in blocked_cells
                for dep in (cell.get("missing_dependencies") or "").split(",")
                if dep
            }
        )
        if not cells:
            readiness = "unknown"
        elif not blocked_cells:
            readiness = "ready"
        elif ready_cells:
            readiness = "partial"
        else:
            readiness = "blocked"

        if name == _EVENT_GRAIN_FAMILY:
            coverage = _event_coverage_summary(event_coverage_rows)
        else:
            coverage = _feature_coverage_for(feature_coverage_rows, primary_feature)

        evaluated = [
            r for r in assembled_by_family.get(name, []) if r.get("status") != "not_evaluated"
        ]
        grades = [r.get("evidence_grade") for r in evaluated if r.get("evidence_grade")]
        grade = _best_grade(grades) if evaluated else "NE"
        discoveries = sum(1 for r in evaluated if r.get("primary_discovery_ab"))
        failed_gates = sorted({gate for r in evaluated for gate in (r.get("failed_gates") or [])})
        peak = max(
            (r for r in evaluated if r.get("ic_mean") is not None),
            key=lambda r: abs(float(r["ic_mean"])),
            default=None,
        )

        # compute_phase_b_rank_correlation puts the Phase A side in family_a /
        # feature_a and the Phase B side in family_b.
        pairs = [r for r in rank_correlation_rows if r.get("family_b") == name]
        top_pair = max(
            (p for p in pairs if p.get("mean_rank_corr") is not None),
            key=lambda p: abs(float(p["mean_rank_corr"])),
            default=None,
        )

        summary.append(
            {
                "family": name,
                "fdr_family": family.get("fdr_family"),
                "primary_feature": primary_feature,
                "secondary_features": ",".join(secondary),
                "expected_sign": family.get("expected_sign"),
                "official_feature_variant": family.get("official_feature_variant"),
                "formula_version": versions.get(name),
                "candidate_cells": len(cells),
                "ready_cells": len(ready_cells),
                "blocked_cells": len(blocked_cells),
                "readiness": readiness,
                "blocker": ",".join(missing),
                "readiness_dependencies": ",".join(family["readiness_dependencies"]),
                **coverage,
                "evaluated_cells": len(evaluated),
                "q_fdr_phase_b_min": _min_or_none(r.get("q_fdr_phase_b") for r in evaluated),
                "q_fdr_global_ab_min": _min_or_none(r.get("q_fdr_global_ab") for r in evaluated),
                "primary_discovery_cells": discoveries,
                "screen_pass_cells": sum(1 for r in evaluated if r.get("screen_pass")),
                "peak_cell": peak.get("hypothesis_id") if peak else None,
                "peak_ic_mean": float(peak["ic_mean"]) if peak else None,
                "evidence_grade": grade,
                "failed_gates": ",".join(failed_gates),
                "top_rank_correlation_pair": top_pair.get("feature_a") if top_pair else None,
                "top_rank_correlation": (float(top_pair["mean_rank_corr"]) if top_pair else None),
                "next_step": _next_step(
                    readiness=readiness,
                    blocker=",".join(missing),
                    evaluated=len(evaluated),
                    grade=grade,
                    discoveries=discoveries,
                ),
            }
        )
    return summary


def _cell(value: Any) -> str:
    """Empty means "not computed here" — never 0, never False."""
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_family_cards_md(rows: Sequence[dict[str, Any]], *, run_id: str | None = None) -> str:
    """§6 B-10's card layout, one section per family."""
    header = ["# Phase B family cards"]
    if run_id:
        header.append(f"\nrun_id: `{run_id}`")
    header.append(
        "\n`—` means the value was not computed in this run, which for a blocked "
        "family is every statistic. It is not zero.\n"
    )
    parts = ["\n".join(header)]

    for row in rows:
        evaluated = row["evaluated_cells"]
        # A count over zero evaluated cells says nothing, so it reads as absent
        # too — otherwise "0 discoveries" looks like a measured result.
        discoveries = row["primary_discovery_cells"] if evaluated else None
        screen_pass = row["screen_pass_cells"] if evaluated else None
        lines = [f"## {row['family']}", ""]
        lines += [
            "| | |",
            "|---|---|",
            f"| primary feature | `{row['primary_feature']}`"
            f" ({_cell(row['official_feature_variant'])}) |",
            f"| secondary | {_cell(row['secondary_features'])} |",
            # Backticked: a "-" expected sign is otherwise easy to misread as
            # the "—" used for absent values.
            f"| expected sign | `{_cell(row['expected_sign'])}` |",
            f"| formula version | {_cell(row['formula_version'])} |",
            f"| fdr family | {_cell(row['fdr_family'])} |",
            "",
            "**readiness**",
            "",
            f"- {row['readiness']} — {row['ready_cells']}/{row['candidate_cells']} cells ready",
            f"- blocker: {_cell(row['blocker'])}",
            f"- dependencies: `{row['readiness_dependencies']}`",
            "",
            "**coverage**",
            "",
            f"- source: {_cell(row['coverage_source'])}",
            f"- effective start: {_cell(row['effective_start'])}",
            f"- coverage ratio: {_cell(row['coverage_ratio'])}",
            f"- thinnest date: {_cell(row['min_names_per_date'])} names",
            f"- observations: {_cell(row['observations'])}",
            "",
            "**result**",
            "",
            f"- evaluated cells: {evaluated}",
            f"- q_fdr_phase_b (min): {_cell(row['q_fdr_phase_b_min'])}",
            f"- q_fdr_global_ab (min): {_cell(row['q_fdr_global_ab_min'])}",
            f"- primary discoveries: {_cell(discoveries)}" f", screen_pass: {_cell(screen_pass)}",
            f"- peak cell: {_cell(row['peak_cell'])}" f" (ic_mean {_cell(row['peak_ic_mean'])})",
            f"- evidence grade: **{row['evidence_grade']}**",
            f"- failed gates: {_cell(row['failed_gates'])}",
            f"- top A/B rank correlation: {_cell(row['top_rank_correlation_pair'])}"
            f" ({_cell(row['top_rank_correlation'])})",
            "",
            f"**next step** — {row['next_step']}",
        ]
        parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"
