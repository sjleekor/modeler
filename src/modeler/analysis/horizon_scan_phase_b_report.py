"""§7.1 ``03b_horizon_scan_results.md`` and the phase=AB summary — B-10 Stage 5.

The markdown counterpart of ``horizon_scan_report.py``'s ``03a`` for Phase B,
following the same three rules that file established and today's diagnostics
kept:

*Every context key is required.* An empty list for a section with nothing to
say, never a missing key — a caller that forgot a section gets an error rather
than a silently blank heading.

*Nothing is recomputed.* Every number comes from a row an earlier stage already
produced, so the report cannot disagree with the parquet beside it. The report
renders; it does not analyse.

*Absent is not zero.* A statistic that was never computed renders as ``—``. For
a run where all 38 cells are blocked that is nearly the whole report, and the
header says so rather than letting a page of zeros imply measurement.

Two files, two audiences. ``03b`` describes Phase B's own run: what was ready,
what the source layer looked like, what the scan found. ``03ab`` describes what
changed when Phase A's 75 hypotheses and Phase B's ready cells went through one
BH pass together — the only question the phase=AB run exists to answer.

``03ab`` is not in §7.1's artifact list, which names only parquet plus the
manifest for phase=AB. It is added deliberately as a *rendering* of artifacts
that contract already requires — it introduces no new statistic — and it is
kept out of the publish-required set so nothing depends on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PHASE_B_REPORT_SECTION_TITLES: tuple[str, ...] = (
    "1. Run identity and readiness freeze",
    "2. Source-layer quality",
    "3. Readiness matrix by family",
    "4. Phase B-only BH summary",
    "5. Robustness diagnostics",
    "6. Family cards",
    "7. Blocked candidates",
    "8. Limitations",
)

PHASE_B_REPORT_CONTEXT_KEYS = frozenset(
    {
        "run_identity",
        "readiness_summary",
        "source_quality",
        "family_rows",
        "bh_summary",
        "robustness",
        "diagnostics_written",
        "limitations",
    }
)

COMBINED_AB_REPORT_SECTION_TITLES: tuple[str, ...] = (
    "1. Run identity and the two source runs",
    "2. Combined A+B BH summary",
    "3. What the combination changed",
    "4. Phase B screen_pass and evidence grade",
    "5. Combined cross-sectional permutation",
)

COMBINED_AB_REPORT_CONTEXT_KEYS = frozenset(
    {"run_identity", "bh_summary", "discovery_changes", "phase_b_verdicts", "permutation"}
)


def _cell(value: Any) -> str:
    """``—`` for absent. Never 0, never False — those are measurements."""
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _bullets(lines: list[str]) -> list[str]:
    return [f"- {line}" for line in lines] if lines else ["- (none)"]


def build_phase_b_report_context(
    *,
    run_spec: dict[str, Any],
    readiness_rows: list[dict[str, Any]],
    family_rows: list[dict[str, Any]],
    assembled_rows: list[dict[str, Any]],
    robustness: dict[str, list[dict[str, Any]]],
    diagnostics_written: list[str],
    q_threshold: float,
    source_quality: dict[str, dict[str, Any]] | None = None,
    timings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the ``03b`` context out of rows ``run_phase_b_core`` already has."""
    ready = [r for r in readiness_rows if r["role"] == "ready_primary"]
    evaluated = [r for r in assembled_rows if r.get("status") != "not_evaluated"]
    return {
        "run_identity": {
            "run_id": run_spec.get("run_id"),
            "snapshot_date": run_spec.get("snapshot_date"),
            "source": run_spec.get("source"),
            "config_hash": run_spec.get("config_hash"),
            "event_feature_formula_version": run_spec.get("event_feature_formula_version"),
            "payout_feature_formula_version": run_spec.get("payout_feature_formula_version"),
            "fin_feature_formula_version": run_spec.get("fin_feature_formula_version"),
            "started_at": run_spec.get("started_at"),
        },
        "readiness_summary": {
            "candidates": len(readiness_rows),
            "ready": len(ready),
            "blocked": len(readiness_rows) - len(ready),
        },
        "source_quality": source_quality or {},
        "family_rows": family_rows,
        "bh_summary": {
            "q_threshold": q_threshold,
            "evaluated_cells": len(evaluated),
            "bh_pass": sum(1 for r in evaluated if r.get("bh_pass_phase_b")),
            "primary_discovery": sum(1 for r in evaluated if r.get("primary_discovery_phase_b")),
        },
        "robustness": {name: len(rows) for name, rows in robustness.items()},
        "diagnostics_written": diagnostics_written,
        "timings": timings or {},
        "limitations": _phase_b_limitations(readiness_rows, source_quality or {}),
    }


def _phase_b_limitations(
    readiness_rows: list[dict[str, Any]], source_quality: dict[str, dict[str, Any]]
) -> list[str]:
    """Limitations that follow from this run's own state, not a fixed blurb."""
    limitations: list[str] = []
    blocked = [r for r in readiness_rows if r["role"] != "ready_primary"]
    if blocked:
        deps = sorted(
            {
                dep
                for row in blocked
                for dep in (row.get("missing_dependencies") or "").split(",")
                if dep
            }
        )
        limitations.append(
            f"{len(blocked)} of {len(readiness_rows)} candidate cells are blocked; "
            f"missing: {', '.join(deps) or 'unknown'}. Their statistics were never computed."
        )
    unmeasured = sorted(
        family
        for family, verdict in source_quality.items()
        if verdict.get("source_quality_status") == "unmeasured"
    )
    if unmeasured:
        limitations.append(
            "source quality could not be established for "
            f"{', '.join(unmeasured)} — grade A is capped for them, and the cap will "
            "lift on its own once the missing diagnostic input exists."
        )
    limitations.append(
        "no PIT industry classification exists in this lake, so value / profitability / "
        "accrual families carry a structural grade-B cap regardless of their statistics."
    )
    return limitations


def render_phase_b_report(context: dict[str, Any]) -> str:
    missing = PHASE_B_REPORT_CONTEXT_KEYS - context.keys()
    if missing:
        raise ValueError(f"phase B report context missing keys: {sorted(missing)}")

    ri = context["run_identity"]
    rs = context["readiness_summary"]
    bh = context["bh_summary"]
    families = context["family_rows"]
    quality = context["source_quality"]
    timing = context.get("timings") or {}
    timing_lines = [
        f"- {name}: {_cell(record.get('wall_seconds'))} s"
        for name, record in sorted((timing.get("stages") or {}).items())
        if isinstance(record, dict)
    ]
    if timing.get("total_wall_seconds") is not None:
        timing_lines.append(f"- total wall time: {_cell(timing['total_wall_seconds'])} s")

    header = [
        "# Phase B horizon scan results",
        "",
        "`—` means the value was not computed in this run. It is not zero.",
    ]

    sections = [
        "\n".join(
            [
                f"## {PHASE_B_REPORT_SECTION_TITLES[0]}",
                "",
                f"- run_id: `{_cell(ri.get('run_id'))}`",
                f"- snapshot_date: {_cell(ri.get('snapshot_date'))} / "
                f"source: {_cell(ri.get('source'))}",
                f"- config_hash: `{_cell(ri.get('config_hash'))}`",
                f"- feature formula: issuance {_cell(ri.get('event_feature_formula_version'))}"
                f" / payout {_cell(ri.get('payout_feature_formula_version'))}"
                f" / fin {_cell(ri.get('fin_feature_formula_version'))}",
                f"- started: {_cell(ri.get('started_at'))}",
                f"- readiness freeze: {rs.get('ready')} ready / {rs.get('blocked')} blocked "
                f"of {rs.get('candidates')} candidate cells",
                *(["", "Stage timing:", *timing_lines] if timing_lines else []),
            ]
        ),
        "\n".join(
            [
                f"## {PHASE_B_REPORT_SECTION_TITLES[1]}",
                "",
                "Diagnostics written this run: "
                + (", ".join(f"`{n}`" for n in context["diagnostics_written"]) or "(none)"),
                "",
                "| family | status | mapping fallback | revision | pairing mismatch |",
                "|---|---|---|---|---|",
            ]
            + [
                f"| {family} | {_cell(v.get('source_quality_status'))} "
                f"| {_cell(v.get('mapping_fallback_ratio'))} "
                f"({_cell(v.get('mapping_fallback_worst_metric'))}) "
                f"| {_cell(v.get('revision_ratio'))} "
                f"| {_cell(v.get('pairing_mismatch_ratio'))} |"
                for family, v in sorted(quality.items())
            ]
            or ["| (no source-quality diagnostic in this run) | — | — | — | — |"]
        ),
        "\n".join(
            [
                f"## {PHASE_B_REPORT_SECTION_TITLES[2]}",
                "",
                "| family | ready / cells | blocker | effective start | coverage |",
                "|---|---|---|---|---|",
            ]
            + [
                f"| {row['family']} | {row['ready_cells']} / {row['candidate_cells']} "
                f"| {_cell(row.get('blocker'))} | {_cell(row.get('effective_start'))} "
                f"| {_cell(row.get('coverage_ratio'))} |"
                for row in families
            ]
        ),
        "\n".join(
            [
                f"## {PHASE_B_REPORT_SECTION_TITLES[3]}",
                "",
                f"- evaluated cells: {bh.get('evaluated_cells')} "
                f"(q < {_cell(bh.get('q_threshold'))})",
                f"- bh_pass: {_cell(bh.get('bh_pass') if bh.get('evaluated_cells') else None)}",
                "- primary_discovery: "
                + _cell(bh.get("primary_discovery") if bh.get("evaluated_cells") else None),
                "",
                "This is the Phase B-only diagnostic pass. The decision-grade q value is "
                "`q_fdr_global_ab`, produced by the phase=AB run over all 75 + M_B_ready "
                "hypotheses together.",
            ]
        ),
        "\n".join(
            [f"## {PHASE_B_REPORT_SECTION_TITLES[4]}", ""]
            + _bullets(
                [f"{name}: {count} rows" for name, count in sorted(context["robustness"].items())]
            )
        ),
        "\n".join(
            [
                f"## {PHASE_B_REPORT_SECTION_TITLES[5]}",
                "",
                "| family | grade | q_fdr_phase_b | discoveries | next step |",
                "|---|---|---|---|---|",
            ]
            + [
                f"| {row['family']} | {row['evidence_grade']} "
                f"| {_cell(row.get('q_fdr_phase_b_min'))} "
                f"| {_cell(row['primary_discovery_cells'] if row['evaluated_cells'] else None)} "
                f"| {row['next_step']} |"
                for row in families
            ]
            + ["", "Full cards with every §7.1 field: `family_cards.md`."]
        ),
        "\n".join(
            [f"## {PHASE_B_REPORT_SECTION_TITLES[6]}", ""]
            + _bullets(
                [
                    f"{row['family']}: {row['blocked_cells']} cells blocked on "
                    f"`{row['blocker']}`"
                    for row in families
                    if row["blocked_cells"]
                ]
            )
        ),
        "\n".join(
            [f"## {PHASE_B_REPORT_SECTION_TITLES[7]}", ""] + _bullets(context["limitations"])
        ),
    ]
    return "\n".join(header) + "\n\n" + "\n\n".join(sections) + "\n"


def write_phase_b_report(output_path: Path, context: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_phase_b_report(context), encoding="utf-8")
    return output_path


def build_combined_ab_report_context(
    *,
    manifest: dict[str, Any],
    combined_rows: list[dict[str, Any]],
    phase_a_overlay: list[dict[str, Any]],
    phase_b_ready_ids: set[str],
) -> dict[str, Any]:
    changed = [row for row in phase_a_overlay if row.get("discovery_changed_vs_phase_a_only")]
    return {
        "run_identity": {
            "run_id": manifest.get("run_id"),
            "generated_at": manifest.get("generated_at"),
            "config_hash": manifest.get("config_hash"),
            "phase_a_run_id": manifest.get("phase_a_run_id"),
            "phase_a_content_hash": manifest.get("phase_a_content_hash"),
            "phase_b_run_id": manifest.get("phase_b_run_id"),
            "phase_b_content_hash": manifest.get("phase_b_content_hash"),
        },
        "bh_summary": {
            "m_ab": manifest.get("m_ab"),
            "q_threshold": manifest.get("q_threshold"),
            "phase_b_ready": len(phase_b_ready_ids),
            "discoveries": sum(1 for row in combined_rows if row.get("primary_discovery_ab")),
        },
        "discovery_changes": [
            f"`{row['hypothesis_id']}` ({row.get('family')}): "
            f"phase A alone {bool(row.get('primary_discovery_phase_a'))} "
            f"-> combined {bool(row.get('primary_discovery_ab'))}"
            for row in changed
        ],
        "phase_b_verdicts": {
            "screen_pass": manifest.get("phase_b_screen_pass_count"),
            "grades": manifest.get("phase_b_evidence_grade_counts", {}),
        },
        "permutation": manifest.get("combined_cross_sectional_permutation"),
    }


def render_combined_ab_report(context: dict[str, Any]) -> str:
    missing = COMBINED_AB_REPORT_CONTEXT_KEYS - context.keys()
    if missing:
        raise ValueError(f"combined AB report context missing keys: {sorted(missing)}")

    ri = context["run_identity"]
    bh = context["bh_summary"]
    verdicts = context["phase_b_verdicts"]
    perm = context["permutation"]

    sections = [
        "\n".join(
            [
                f"## {COMBINED_AB_REPORT_SECTION_TITLES[0]}",
                "",
                f"- run_id: `{_cell(ri.get('run_id'))}` (generated "
                f"{_cell(ri.get('generated_at'))})",
                f"- config_hash: `{_cell(ri.get('config_hash'))}`",
                f"- phase A run: `{_cell(ri.get('phase_a_run_id'))}` "
                f"(content hash `{_cell(ri.get('phase_a_content_hash'))}`)",
                f"- phase B run: `{_cell(ri.get('phase_b_run_id'))}` "
                f"(content hash `{_cell(ri.get('phase_b_content_hash'))}`)",
                "",
                "Both source runs were integrity-verified against those hashes before "
                "this combination ran (§2.3 rule 5).",
            ]
        ),
        "\n".join(
            [
                f"## {COMBINED_AB_REPORT_SECTION_TITLES[1]}",
                "",
                f"- hypotheses in the combined family: {_cell(bh.get('m_ab'))} "
                f"(75 Phase A + {_cell(bh.get('phase_b_ready'))} Phase B ready)",
                f"- q threshold: {_cell(bh.get('q_threshold'))}",
                f"- primary discoveries: {_cell(bh.get('discoveries'))}",
            ]
        ),
        "\n".join(
            [
                f"## {COMBINED_AB_REPORT_SECTION_TITLES[2]}",
                "",
                "Phase A hypotheses whose discovery status differs from the Phase A-only "
                "pass — the whole reason the combined pass exists, since adding Phase B "
                "hypotheses widens the BH family and can only make the bar stricter.",
                "",
            ]
            + _bullets(context["discovery_changes"])
        ),
        "\n".join(
            [
                f"## {COMBINED_AB_REPORT_SECTION_TITLES[3]}",
                "",
                f"- screen_pass cells: {_cell(verdicts.get('screen_pass'))}",
                "- evidence grades: "
                + (
                    ", ".join(
                        f"{grade}={count}"
                        for grade, count in sorted((verdicts.get("grades") or {}).items())
                    )
                    or "—"
                ),
            ]
        ),
        "\n".join(
            [f"## {COMBINED_AB_REPORT_SECTION_TITLES[4]}", ""]
            + (
                [
                    f"- real discovery count: {_cell(perm.get('real_discovery_count'))}",
                    f"- p_empirical_count: {_cell(perm.get('p_empirical_count'))} "
                    f"({_cell(perm.get('n_replicates'))} replicates)",
                ]
                if perm
                else ["- (no null distribution was published by the Phase B run)"]
            )
        ),
    ]
    header = [
        "# Combined Phase A + B results",
        "",
        "`—` means the value was not computed in this run. It is not zero.",
    ]
    return "\n".join(header) + "\n\n" + "\n\n".join(sections) + "\n"


def write_combined_ab_report(output_path: Path, context: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_combined_ab_report(context), encoding="utf-8")
    return output_path
