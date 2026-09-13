"""B-10 Stage 5 — 03b_horizon_scan_results.md and the combined AB summary."""

from __future__ import annotations

from datetime import date

import pytest

from modeler.analysis.horizon_scan_phase_b_report import (
    COMBINED_AB_REPORT_CONTEXT_KEYS,
    PHASE_B_REPORT_CONTEXT_KEYS,
    build_combined_ab_report_context,
    build_phase_b_report_context,
    render_combined_ab_report,
    render_phase_b_report,
)

_RUN_SPEC = {
    "run_id": "20260812T120000-abcd1234",
    "snapshot_date": "2026-08-09",
    "source": "sj2_remote",
    "config_hash": "e55c3046",
    "event_feature_formula_version": "issuance_v2",
    "started_at": "2026-08-12T12:00:00+09:00",
}


def _readiness_rows(*, ready: int = 0, blocked: int = 4) -> list[dict]:
    rows = [
        {
            "family": "fin_value_z",
            "hypothesis_id": f"fin_value_z|cum|{i}",
            "role": "ready_primary",
            "missing_dependencies": "",
        }
        for i in range(ready)
    ]
    rows += [
        {
            "family": "fin_value_z",
            "hypothesis_id": f"fin_value_z|blocked|{i}",
            "role": "blocked_exploratory",
            "missing_dependencies": "feat_fin_scan_daily",
        }
        for i in range(blocked)
    ]
    return rows


def _family_rows(**overrides) -> list[dict]:
    row = {
        "family": "fin_value_z",
        "candidate_cells": 4,
        "ready_cells": 0,
        "blocked_cells": 4,
        "blocker": "feat_fin_scan_daily",
        "effective_start": None,
        "coverage_ratio": None,
        "evaluated_cells": 0,
        "q_fdr_phase_b_min": None,
        "primary_discovery_cells": 0,
        "evidence_grade": "NE",
        "next_step": "collect feat_fin_scan_daily",
    }
    row.update(overrides)
    return [row]


def _context(**overrides):
    context = build_phase_b_report_context(
        run_spec=_RUN_SPEC,
        readiness_rows=_readiness_rows(),
        family_rows=_family_rows(),
        assembled_rows=[],
        robustness={},
        diagnostics_written=["filing_receipt_quality"],
        q_threshold=0.10,
        source_quality={
            "fin_value_z": {
                "source_quality_status": "unmeasured",
                "mapping_fallback_ratio": 0.3555,
                "mapping_fallback_worst_metric": "controlling_net_income",
                "revision_ratio": None,
                "pairing_mismatch_ratio": 0.000117,
            }
        },
    )
    context.update(overrides)
    return context


def test_every_context_key_is_required() -> None:
    context = _context()
    for key in sorted(PHASE_B_REPORT_CONTEXT_KEYS):
        partial = {k: v for k, v in context.items() if k != key}
        # An omitted section is a caller bug, not a silently blank heading.
        with pytest.raises(ValueError, match=key):
            render_phase_b_report(partial)


def test_blocked_run_reports_the_blocker_and_no_fabricated_statistics() -> None:
    markdown = render_phase_b_report(_context())

    assert "0 ready / 4 blocked of 4 candidate cells" in markdown
    assert "feat_fin_scan_daily" in markdown
    # Nothing was evaluated, so the BH counts read as absent rather than zero.
    assert "- bh_pass: —" in markdown
    assert "- primary_discovery: —" in markdown
    assert "It is not zero." in markdown


def test_phase_b_report_renders_stage_timing_when_supplied() -> None:
    context = _context(
        timings={
            "stages": {"phase_b_scan": {"wall_seconds": 12.5}},
            "total_wall_seconds": 12.5,
        }
    )

    markdown = render_phase_b_report(context)

    assert "Stage timing:" in markdown
    assert "phase_b_scan: 12.5000 s" in markdown
    assert "total wall time: 12.5000 s" in markdown


def test_bh_counts_are_real_numbers_once_cells_were_evaluated() -> None:
    assembled = [
        {
            "family": "fin_value_z",
            "status": "ok",
            "bh_pass_phase_b": True,
            "primary_discovery_phase_b": False,
        },
        {
            "family": "fin_value_z",
            "status": "ok",
            "bh_pass_phase_b": False,
            "primary_discovery_phase_b": False,
        },
    ]
    context = build_phase_b_report_context(
        run_spec=_RUN_SPEC,
        readiness_rows=_readiness_rows(ready=2, blocked=2),
        family_rows=_family_rows(evaluated_cells=2, ready_cells=2, blocked_cells=2),
        assembled_rows=assembled,
        robustness={"nonoverlap_rows": [{"a": 1}]},
        diagnostics_written=[],
        q_threshold=0.10,
    )
    markdown = render_phase_b_report(context)

    assert context["bh_summary"]["evaluated_cells"] == 2
    assert "- bh_pass: 1" in markdown
    # Zero discoveries out of two evaluated cells is a measurement, so it prints.
    assert "- primary_discovery: 0" in markdown
    assert "nonoverlap_rows: 1 rows" in markdown


def test_source_quality_table_names_the_worst_metric() -> None:
    markdown = render_phase_b_report(_context())

    assert "| fin_value_z | unmeasured | 0.3555 (controlling_net_income) | — | 0.0001 |" in markdown


def test_limitations_are_derived_from_this_run_not_boilerplate() -> None:
    blocked = _context()["limitations"]
    assert any("4 of 4 candidate cells are blocked" in line for line in blocked)
    assert any("grade A is capped" in line for line in blocked)

    clean = build_phase_b_report_context(
        run_spec=_RUN_SPEC,
        readiness_rows=_readiness_rows(ready=4, blocked=0),
        family_rows=_family_rows(ready_cells=4, blocked_cells=0, blocker=""),
        assembled_rows=[],
        robustness={},
        diagnostics_written=[],
        q_threshold=0.10,
        source_quality={"fin_value_z": {"source_quality_status": "ok"}},
    )["limitations"]
    assert not any("blocked" in line for line in clean)
    # The PIT-industry cap is structural, so it survives even a clean run.
    assert any("PIT industry" in line for line in clean)


def test_coverage_row_renders_absent_values_as_a_dash() -> None:
    markdown = render_phase_b_report(_context())

    assert "| fin_value_z | 0 / 4 | feat_fin_scan_daily | — | — |" in markdown


# --- combined AB ---

_AB_MANIFEST = {
    "run_id": "20260812T130000-deadbeef",
    "generated_at": "2026-08-12T13:00:00+09:00",
    "config_hash": "e55c3046",
    "q_threshold": 0.10,
    "m_ab": 79,
    "phase_a_run_id": "20260810T141014-7212fe82",
    "phase_a_content_hash": "aaaa",
    "phase_b_run_id": "20260812T120000-abcd1234",
    "phase_b_content_hash": "bbbb",
    "phase_b_screen_pass_count": 1,
    "phase_b_evidence_grade_counts": {"A": 0, "B": 1, "C": 0, "D": 3},
}


def test_combined_report_requires_every_context_key() -> None:
    context = build_combined_ab_report_context(
        manifest=_AB_MANIFEST,
        combined_rows=[],
        phase_a_overlay=[],
        phase_b_ready_ids=set(),
    )
    for key in sorted(COMBINED_AB_REPORT_CONTEXT_KEYS):
        partial = {k: v for k, v in context.items() if k != key}
        with pytest.raises(ValueError, match=key):
            render_combined_ab_report(partial)


def test_combined_report_lists_only_hypotheses_whose_status_changed() -> None:
    overlay = [
        {
            "hypothesis_id": "px_maxret_20d|cum|0|20",
            "family": "px_maxret_20d",
            "primary_discovery_phase_a": True,
            "primary_discovery_ab": False,
            "discovery_changed_vs_phase_a_only": True,
        },
        {
            "hypothesis_id": "px_reversal_5d|cum|0|5",
            "family": "px_reversal_5d",
            "primary_discovery_phase_a": True,
            "primary_discovery_ab": True,
            "discovery_changed_vs_phase_a_only": False,
        },
    ]
    context = build_combined_ab_report_context(
        manifest=_AB_MANIFEST,
        combined_rows=[{"primary_discovery_ab": True}],
        phase_a_overlay=overlay,
        phase_b_ready_ids={"fin_value_z|cum|0|60"},
    )
    markdown = render_combined_ab_report(context)

    assert "px_maxret_20d|cum|0|20" in markdown
    # A hypothesis that survived the wider family unchanged is not news.
    assert "px_reversal_5d" not in markdown
    assert "75 Phase A + 1 Phase B ready" in markdown
    assert "B=1, C=0, D=3" in markdown


def test_combined_report_says_so_when_no_null_distribution_was_published() -> None:
    context = build_combined_ab_report_context(
        manifest=_AB_MANIFEST, combined_rows=[], phase_a_overlay=[], phase_b_ready_ids=set()
    )
    markdown = render_combined_ab_report(context)

    assert "no null distribution was published" in markdown
    assert "- (none)" in markdown  # no discovery changes


def test_combined_report_renders_the_permutation_result_when_present() -> None:
    manifest = {
        **_AB_MANIFEST,
        "combined_cross_sectional_permutation": {
            "real_discovery_count": 3,
            "n_replicates": 100,
            "p_empirical_count": 0.0396,
        },
    }
    context = build_combined_ab_report_context(
        manifest=manifest, combined_rows=[], phase_a_overlay=[], phase_b_ready_ids=set()
    )
    markdown = render_combined_ab_report(context)

    assert "real discovery count: 3" in markdown
    assert "p_empirical_count: 0.0396 (100 replicates)" in markdown


def test_report_dates_render_without_crashing() -> None:
    markdown = render_phase_b_report(
        _context(family_rows=_family_rows(effective_start=date(2016, 3, 2), coverage_ratio=0.75))
    )

    assert "2016-03-02" in markdown
    assert "0.7500" in markdown
