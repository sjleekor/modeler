"""B-10 Stage 4 — family_summary / family_cards."""

from __future__ import annotations

from datetime import date

import pytest

from modeler.analysis.horizon_scan_config import CONFIG_PATH, load_config
from modeler.analysis.horizon_scan_phase_b import (
    build_phase_b_readiness_rows,
    phase_b_families,
)
from modeler.analysis.horizon_scan_phase_b_cards import (
    build_family_summary_rows,
    render_family_cards_md,
)


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG_PATH)


def _blocked_rows(config) -> list[dict]:
    """Today's real state: no Phase B raw, so every cell is blocked."""
    return build_phase_b_readiness_rows(config, available_assets=set())


def _rows_by_family(rows: list[dict]) -> dict[str, dict]:
    return {row["family"]: row for row in rows}


def test_one_row_per_phase_b_family_in_config_order(config) -> None:
    summary = build_family_summary_rows(config, readiness_rows=_blocked_rows(config))

    assert [r["family"] for r in summary] == [f["family"] for f in phase_b_families(config)]
    assert len(summary) == 8


def test_a_blocked_family_names_its_blocker_and_leaves_stats_empty(config) -> None:
    summary = _rows_by_family(
        build_family_summary_rows(config, readiness_rows=_blocked_rows(config))
    )
    sue = summary["fin_sue"]

    assert sue["readiness"] == "blocked"
    assert sue["ready_cells"] == 0
    assert sue["blocked_cells"] == sue["candidate_cells"] == 6
    assert sue["blocker"] == "dart_filing_receipt_raw,dart_xbrl_fact_raw,fin_sue_event"
    # Not evaluated is not the same as evaluated-and-zero.
    assert sue["q_fdr_phase_b_min"] is None
    assert sue["q_fdr_global_ab_min"] is None
    assert sue["peak_ic_mean"] is None
    assert sue["evidence_grade"] == "NE"
    assert sue["next_step"].startswith("collect ")


def test_formula_version_is_only_set_for_the_family_that_has_one(config) -> None:
    summary = _rows_by_family(
        build_family_summary_rows(
            config,
            readiness_rows=_blocked_rows(config),
            formula_versions={"ev_net_share_issuance_yoy": "issuance_v2"},
        )
    )

    assert summary["ev_net_share_issuance_yoy"]["formula_version"] == "issuance_v2"
    # No fin family is fingerprinted yet — an invented version on the card
    # would be worse than an empty one.
    assert summary["fin_value_z"]["formula_version"] is None


def test_coverage_rolls_up_native_t_rows_only(config) -> None:
    coverage = [
        {
            "feature": "fin_value_z",
            "variant": "native_t",
            "market": "KOSPI",
            "panel_rows": 100,
            "nonnull_rows": 60,
            "first_value_date": date(2016, 3, 2),
            "min_names_per_date": 12,
        },
        {
            "feature": "fin_value_z",
            "variant": "native_t",
            "market": "KOSDAQ",
            "panel_rows": 100,
            "nonnull_rows": 40,
            "first_value_date": date(2015, 5, 4),
            "min_names_per_date": 5,
        },
        # The lag1 variant is the same feature one session later; folding it in
        # would double every count.
        {
            "feature": "fin_value_z_lag1",
            "variant": "lag1",
            "market": "KOSPI",
            "panel_rows": 100,
            "nonnull_rows": 60,
            "first_value_date": date(2016, 3, 3),
            "min_names_per_date": 12,
        },
    ]
    summary = _rows_by_family(
        build_family_summary_rows(
            config, readiness_rows=_blocked_rows(config), feature_coverage_rows=coverage
        )
    )
    row = summary["fin_value_z"]

    assert row["coverage_source"] == "feature_coverage"
    assert row["coverage_ratio"] == 0.5
    assert row["effective_start"] == date(2015, 5, 4)
    assert row["min_names_per_date"] == 5
    assert row["observations"] == 100


def test_sue_coverage_comes_from_the_event_grain_table(config) -> None:
    summary = _rows_by_family(
        build_family_summary_rows(
            config,
            readiness_rows=_blocked_rows(config),
            event_coverage_rows=[
                {
                    "events": 200,
                    "events_with_sue": 150,
                    "first_formation_date": date(2017, 4, 3),
                }
            ],
            feature_coverage_rows=[
                {
                    "feature": "fin_sue",
                    "variant": "native_t",
                    "panel_rows": 1,
                    "nonnull_rows": 1,
                    "first_value_date": date(2020, 1, 1),
                    "min_names_per_date": 1,
                }
            ],
        )
    )
    row = summary["fin_sue"]

    assert row["coverage_source"] == "event_coverage"
    assert row["coverage_ratio"] == 0.75
    assert row["effective_start"] == date(2017, 4, 3)
    # Names-per-date has no event-grain meaning; it must not borrow the
    # continuous-panel lookalike above.
    assert row["min_names_per_date"] is None


def test_family_grade_is_the_best_its_cells_reached(config) -> None:
    readiness = _blocked_rows(config)
    cells = [r for r in readiness if r["family"] == "fin_log_mcap"]
    assembled = [
        {
            **cells[0],
            "status": "ok",
            "evidence_grade": "D",
            "ic_mean": 0.01,
            "q_fdr_phase_b": 0.4,
            "failed_gates": ["tradable_pass"],
        },
        {
            **cells[1],
            "status": "ok",
            "evidence_grade": "B",
            "ic_mean": -0.05,
            "q_fdr_phase_b": 0.02,
            "q_fdr_global_ab": 0.03,
            "primary_discovery_ab": True,
            "screen_pass": True,
            "failed_gates": [],
        },
    ]
    summary = _rows_by_family(
        build_family_summary_rows(config, readiness_rows=readiness, assembled_rows=assembled)
    )
    row = summary["fin_log_mcap"]

    assert row["evaluated_cells"] == 2
    assert row["evidence_grade"] == "B"
    assert row["q_fdr_phase_b_min"] == 0.02
    assert row["primary_discovery_cells"] == 1
    assert row["screen_pass_cells"] == 1
    # Peak is by magnitude, so the negative one wins.
    assert row["peak_ic_mean"] == -0.05
    assert row["failed_gates"] == "tradable_pass"


def test_not_evaluated_rows_never_count_as_results(config) -> None:
    readiness = _blocked_rows(config)
    assembled = [
        {**cell, "status": "not_evaluated", "status_reason": "blocked_exploratory"}
        for cell in readiness
    ]
    summary = build_family_summary_rows(config, readiness_rows=readiness, assembled_rows=assembled)

    assert all(row["evaluated_cells"] == 0 for row in summary)
    assert all(row["evidence_grade"] == "NE" for row in summary)


def test_rank_correlation_matches_on_the_phase_b_side(config) -> None:
    summary = _rows_by_family(
        build_family_summary_rows(
            config,
            readiness_rows=_blocked_rows(config),
            rank_correlation_rows=[
                {
                    "family_a": "px_maxret_20d",
                    "feature_a": "px_maxret_20d",
                    "family_b": "fin_value_z",
                    "feature_b": "fin_value_z",
                    "mean_rank_corr": -0.12,
                },
                {
                    "family_a": "px_reversal_5d",
                    "feature_a": "px_reversal_5d",
                    "family_b": "fin_value_z",
                    "feature_b": "fin_value_z",
                    "mean_rank_corr": 0.31,
                },
            ],
        )
    )

    assert summary["fin_value_z"]["top_rank_correlation_pair"] == "px_reversal_5d"
    assert summary["fin_value_z"]["top_rank_correlation"] == 0.31
    assert summary["fin_log_mcap"]["top_rank_correlation_pair"] is None


def test_cards_render_absent_values_as_a_dash_not_zero(config) -> None:
    rows = build_family_summary_rows(config, readiness_rows=_blocked_rows(config))
    markdown = render_family_cards_md(rows, run_id="20260812T090000-deadbeef")

    assert "run_id: `20260812T090000-deadbeef`" in markdown
    for family in phase_b_families(config):
        assert f"## {family['family']}" in markdown
    assert "q_fdr_phase_b (min): —" in markdown
    assert "It is not zero." in markdown
    # A blocked family has no grade, and "NE" must be what is shown.
    assert "evidence grade: **NE**" in markdown
    # Counts over zero evaluated cells are absent too — "0 discoveries" would
    # read as a measured result.
    assert "primary discoveries: —, screen_pass: —" in markdown
    # A "-" expected sign must not be confusable with the "—" placeholder.
    assert "| expected sign | `-` |" in markdown


def test_cards_show_real_counts_once_cells_are_evaluated(config) -> None:
    readiness = _blocked_rows(config)
    cell = next(r for r in readiness if r["family"] == "fin_log_mcap")
    rows = build_family_summary_rows(
        config,
        readiness_rows=readiness,
        assembled_rows=[
            {
                **cell,
                "status": "ok",
                "evidence_grade": "D",
                "ic_mean": 0.01,
                "primary_discovery_ab": False,
                "screen_pass": False,
            }
        ],
    )
    markdown = render_family_cards_md(rows)

    # Now 0 is a real measurement, so it prints as 0.
    assert "primary discoveries: 0, screen_pass: 0" in markdown
