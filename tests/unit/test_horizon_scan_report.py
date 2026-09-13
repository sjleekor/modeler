from __future__ import annotations

import pytest

from modeler.analysis.horizon_scan_report import (
    PATTERN_DELAYED,
    PATTERN_EXPLORATORY_ONLY,
    PATTERN_IMMEDIATE,
    PATTERN_NO_SIGNAL,
    PATTERN_SEGMENT_LIMITED,
    PATTERN_SIGN_REVERSAL,
    assign_evidence_grade,
    build_family_card,
    classify_pattern_auto,
    compute_decay_summary,
    compute_screen_pass,
    plot_bucket_ic_bar,
    plot_coverage_curve,
    plot_cumulative_ic_curve,
    plot_native_vs_lag1,
    plot_offset_distribution,
    plot_segment_dot,
    plot_subperiod_heatmap,
    render_family_plots,
    render_markdown_report,
)


def _cum(h_end, ic_mean, q=None):
    return {"h_end": h_end, "ic_mean": ic_mean, "q_fdr_global": q}


def _bucket(h_start, h_end, ic_mean, q=None):
    return {"h_start": h_start, "h_end": h_end, "ic_mean": ic_mean, "q_fdr_global": q}


def test_peak_h_cum_picks_largest_absolute_aligned_ic() -> None:
    cum_rows = [_cum(5, 0.01), _cum(10, -0.06), _cum(20, 0.04)]
    summary = compute_decay_summary(cum_rows, [], expected_sign="+")
    # |−0.06| = 0.06 is the largest magnitude even though its sign is "wrong"
    assert summary["peak_h_cum"] == 10


def test_peak_bucket_picks_largest_aligned_ic_not_absolute() -> None:
    bucket_rows = [_bucket(0, 5, 0.02), _bucket(5, 10, -0.08), _bucket(10, 20, 0.05)]
    summary = compute_decay_summary([], bucket_rows, expected_sign="+")
    # -0.08 has the largest magnitude but peak_bucket uses signed max, not abs
    assert summary["peak_bucket"] == (10, 20)


def test_onset_h_is_first_grid_order_cell_passing_bh_with_right_sign() -> None:
    cum_rows = [
        _cum(5, 0.01, q=0.5),  # not significant
        _cum(10, -0.05, q=0.01),  # significant but wrong sign
        _cum(20, 0.04, q=0.05),  # first qualifying cell
        _cum(40, 0.06, q=0.01),
    ]
    summary = compute_decay_summary(cum_rows, [], expected_sign="+")
    assert summary["onset_h"] == 20


def test_onset_h_is_none_when_nothing_clears_bh() -> None:
    cum_rows = [_cum(5, 0.01, q=0.5), _cum(10, 0.02, q=0.3)]
    summary = compute_decay_summary(cum_rows, [], expected_sign="+")
    assert summary["onset_h"] is None
    assert summary["onset_h_reason"] == "no_cell_passed_bh"


def test_half_life_bucket_found_after_peak() -> None:
    # peak at (5,10)=0.08; next bucket (10,20)=0.03 is below half (0.04) -> half-life there
    bucket_rows = [_bucket(0, 5, 0.05), _bucket(5, 10, 0.08), _bucket(10, 20, 0.03)]
    summary = compute_decay_summary([], bucket_rows, expected_sign="+")
    assert summary["peak_bucket"] == (5, 10)
    assert summary["half_life_bucket"] == (10, 20)


def test_half_life_bucket_not_forced_when_no_subsequent_bucket() -> None:
    bucket_rows = [_bucket(0, 5, 0.05), _bucket(5, 10, 0.08)]  # peak is the last bucket
    summary = compute_decay_summary([], bucket_rows, expected_sign="+")
    assert summary["half_life_bucket"] is None
    assert summary["half_life_bucket_reason"] == "no_subsequent_bucket"


def test_half_life_bucket_not_forced_when_peak_is_not_positive() -> None:
    bucket_rows = [_bucket(0, 5, -0.02), _bucket(5, 10, -0.01)]
    summary = compute_decay_summary([], bucket_rows, expected_sign="+")
    assert summary["half_life_bucket"] is None
    assert summary["half_life_bucket_reason"] == "peak_not_positive"


def test_half_life_bucket_reason_when_decay_never_reaches_half() -> None:
    bucket_rows = [_bucket(0, 5, 0.08), _bucket(5, 10, 0.07), _bucket(10, 20, 0.06)]
    summary = compute_decay_summary([], bucket_rows, expected_sign="+")
    assert summary["half_life_bucket"] is None
    assert summary["half_life_bucket_reason"] == "decay_never_reaches_half_life"


def test_sign_flip_bucket_detects_first_reversal_from_the_first_valid_bucket() -> None:
    bucket_rows = [_bucket(0, 5, 0.05), _bucket(5, 10, 0.03), _bucket(10, 20, -0.02)]
    summary = compute_decay_summary([], bucket_rows, expected_sign="+")
    assert summary["sign_flip_bucket"] == (10, 20)


def test_sign_flip_bucket_none_when_all_buckets_agree() -> None:
    bucket_rows = [_bucket(0, 5, 0.05), _bucket(5, 10, 0.03), _bucket(10, 20, 0.01)]
    summary = compute_decay_summary([], bucket_rows, expected_sign="+")
    assert summary["sign_flip_bucket"] is None


def test_negative_expected_sign_flips_alignment_for_peak_and_onset() -> None:
    cum_rows = [_cum(5, -0.01, q=0.5), _cum(20, -0.05, q=0.01)]
    summary = compute_decay_summary(cum_rows, [], expected_sign="-")
    assert summary["peak_h_cum"] == 20
    assert summary["onset_h"] == 20


def test_two_sided_family_uses_observed_sign_directly() -> None:
    cum_rows = [_cum(5, 0.02, q=0.2), _cum(20, -0.06, q=0.01)]
    summary = compute_decay_summary(cum_rows, [], expected_sign=None)
    # no sign flip applied: -0.06's magnitude (0.06) still wins the peak,
    # but its raw (negative) sign fails the onset "aligned > 0" requirement
    assert summary["peak_h_cum"] == 20
    assert summary["onset_h"] is None


def test_classify_pattern_no_signal_and_exploratory_only() -> None:
    assert (
        classify_pattern_auto(
            has_primary_discovery=False,
            has_exploratory_significant=False,
            peak_bucket=None,
            sign_flip_bucket=None,
            segment_gates_all_pass=None,
        )
        == PATTERN_NO_SIGNAL
    )
    assert (
        classify_pattern_auto(
            has_primary_discovery=False,
            has_exploratory_significant=True,
            peak_bucket=None,
            sign_flip_bucket=None,
            segment_gates_all_pass=None,
        )
        == PATTERN_EXPLORATORY_ONLY
    )


def test_classify_pattern_sign_reversal_takes_priority() -> None:
    result = classify_pattern_auto(
        has_primary_discovery=True,
        has_exploratory_significant=False,
        peak_bucket=(0, 5),
        sign_flip_bucket=(10, 20),
        segment_gates_all_pass=False,
    )
    assert result == PATTERN_SIGN_REVERSAL


def test_classify_pattern_segment_limited_when_gates_fail() -> None:
    result = classify_pattern_auto(
        has_primary_discovery=True,
        has_exploratory_significant=False,
        peak_bucket=(5, 10),
        sign_flip_bucket=None,
        segment_gates_all_pass=False,
    )
    assert result == PATTERN_SEGMENT_LIMITED


def test_classify_pattern_immediate_vs_delayed() -> None:
    immediate = classify_pattern_auto(
        has_primary_discovery=True,
        has_exploratory_significant=False,
        peak_bucket=(0, 5),
        sign_flip_bucket=None,
        segment_gates_all_pass=True,
    )
    assert immediate == PATTERN_IMMEDIATE

    delayed = classify_pattern_auto(
        has_primary_discovery=True,
        has_exploratory_significant=False,
        peak_bucket=(10, 20),
        sign_flip_bucket=None,
        segment_gates_all_pass=True,
    )
    assert delayed == PATTERN_DELAYED


def _all_pass_kwargs(**overrides):
    kwargs = dict(
        role="ready",
        primary_discovery=True,
        tradable_pass=True,
        period_sign_pass=True,
        isolated_spike=False,
        available_direction_pass=True,
        delay_required=False,
        delay_pass=None,
        temporal_null_required=False,
        temporal_null_pass=None,
    )
    kwargs.update(overrides)
    return kwargs


def test_screen_pass_true_when_every_applicable_gate_passes() -> None:
    result = compute_screen_pass(**_all_pass_kwargs())
    assert result["screen_pass"] is True
    assert result["failed_gates"] == []


def test_screen_pass_non_ready_role_is_never_a_candidate() -> None:
    result = compute_screen_pass(**_all_pass_kwargs(role="exploratory_short_regime"))
    assert result["screen_pass"] is False
    assert result["not_applicable_role"] is True


def test_screen_pass_fails_on_any_single_failed_gate() -> None:
    result = compute_screen_pass(**_all_pass_kwargs(tradable_pass=False))
    assert result["screen_pass"] is False
    assert "tradable_pass" in result["failed_gates"]

    result = compute_screen_pass(**_all_pass_kwargs(isolated_spike=True))
    assert result["screen_pass"] is False
    assert "isolated_spike_clear" in result["failed_gates"]


def test_screen_pass_available_direction_none_does_not_block() -> None:
    result = compute_screen_pass(**_all_pass_kwargs(available_direction_pass=None))
    assert result["screen_pass"] is True
    assert "available_direction_pass" not in result["failed_gates"]


def test_screen_pass_delay_gate_only_checked_when_required() -> None:
    not_required = compute_screen_pass(
        **_all_pass_kwargs(delay_required=False, delay_pass=False)
    )
    assert not_required["screen_pass"] is True  # delay_pass=False is irrelevant here

    required = compute_screen_pass(**_all_pass_kwargs(delay_required=True, delay_pass=False))
    assert required["screen_pass"] is False
    assert "delay_pass" in required["failed_gates"]


def test_screen_pass_temporal_null_gate_only_checked_when_required() -> None:
    not_required = compute_screen_pass(
        **_all_pass_kwargs(temporal_null_required=False, temporal_null_pass=False)
    )
    assert not_required["screen_pass"] is True

    required = compute_screen_pass(
        **_all_pass_kwargs(temporal_null_required=True, temporal_null_pass=False)
    )
    assert required["screen_pass"] is False
    assert "temporal_null_pass" in required["failed_gates"]


def test_evidence_grade_reference_role_is_always_r() -> None:
    assert assign_evidence_grade(role="reference", screen_pass=True) == "R"
    assert assign_evidence_grade(role="reference", screen_pass=False) == "R"


def test_evidence_grade_exploratory_and_secondary_roles_are_c() -> None:
    for role in ("exploratory_short_regime", "exploratory_horizon", "secondary_feature"):
        assert assign_evidence_grade(role=role, screen_pass=True) == "C"


def test_evidence_grade_available_sign_flip_caps_at_c_even_if_screen_pass() -> None:
    assert assign_evidence_grade(role="ready", screen_pass=True, available_sign_flip=True) == "C"


def test_evidence_grade_clean_screen_pass_is_a() -> None:
    result = assign_evidence_grade(
        role="ready", screen_pass=True, has_nonfatal_warning=False, all_offsets_evaluable=True
    )
    assert result == "A"


def test_evidence_grade_nonfatal_warning_caps_at_b() -> None:
    result = assign_evidence_grade(
        role="ready", screen_pass=True, has_nonfatal_warning=True, all_offsets_evaluable=True
    )
    assert result == "B"


def test_evidence_grade_insufficient_offset_caps_at_b() -> None:
    result = assign_evidence_grade(
        role="ready", screen_pass=True, has_nonfatal_warning=False, all_offsets_evaluable=False
    )
    assert result == "B"


def test_evidence_grade_no_screen_pass_is_d() -> None:
    assert assign_evidence_grade(role="ready", screen_pass=False) == "D"


# --- A-9 plots ---


def test_plot_cumulative_ic_curve_writes_a_png_with_data(tmp_path) -> None:
    curves = {
        "broad_common_survivor": [{"h_end": 5, "ic_mean": 0.02}, {"h_end": 10, "ic_mean": 0.04}],
        "tradable_available": [],
    }
    out = plot_cumulative_ic_curve(curves, family="fam", output_path=tmp_path / "cum.png")
    assert out.is_file() and out.stat().st_size > 0


def test_plot_cumulative_ic_curve_handles_all_empty_curves(tmp_path) -> None:
    out = plot_cumulative_ic_curve({"a": []}, family="fam", output_path=tmp_path / "cum_empty.png")
    assert out.is_file() and out.stat().st_size > 0


def test_plot_bucket_ic_bar_writes_a_png(tmp_path) -> None:
    rows = [
        {"h_start": 0, "h_end": 5, "ic_mean": 0.02, "t_nw": 2.5, "q_fdr_global": 0.05},
        {"h_start": 5, "h_end": 10, "ic_mean": -0.01, "t_nw": -0.5, "q_fdr_global": 0.4},
    ]
    out = plot_bucket_ic_bar(
        rows, family="fam", expected_sign="+", output_path=tmp_path / "bucket.png"
    )
    assert out.is_file() and out.stat().st_size > 0


def test_plot_bucket_ic_bar_handles_empty_rows(tmp_path) -> None:
    out = plot_bucket_ic_bar(
        [], family="fam", expected_sign="+", output_path=tmp_path / "bucket_empty.png"
    )
    assert out.is_file()


def test_plot_native_vs_lag1_writes_a_png(tmp_path) -> None:
    native = [{"h_end": 5, "ic_mean": 0.02}]
    lag1 = [{"h_end": 5, "ic_mean": 0.015}]
    out = plot_native_vs_lag1(native, lag1, family="fam", output_path=tmp_path / "nvl.png")
    assert out.is_file()


def test_plot_native_vs_lag1_handles_both_empty(tmp_path) -> None:
    out = plot_native_vs_lag1([], [], family="fam", output_path=tmp_path / "nvl_empty.png")
    assert out.is_file()


def test_plot_subperiod_heatmap_writes_a_png(tmp_path) -> None:
    rows = [
        {"period_id": "2020_2021", "h_end": 5, "ic_mean": 0.02},
        {"period_id": "2020_2021", "h_end": 20, "ic_mean": -0.01},
        {"period_id": "2022_2023", "h_end": 5, "ic_mean": 0.03},
    ]
    out = plot_subperiod_heatmap(rows, family="fam", output_path=tmp_path / "heatmap.png")
    assert out.is_file()


def test_plot_subperiod_heatmap_handles_empty_rows(tmp_path) -> None:
    out = plot_subperiod_heatmap([], family="fam", output_path=tmp_path / "heatmap_empty.png")
    assert out.is_file()


def test_plot_segment_dot_writes_a_png(tmp_path) -> None:
    rows = [
        {"segment_axis": "size", "segment": "small", "ic_mean": 0.03},
        {"segment_axis": "size", "segment": "large", "ic_mean": 0.01},
    ]
    out = plot_segment_dot(rows, family="fam", output_path=tmp_path / "segment.png")
    assert out.is_file()


def test_plot_coverage_curve_writes_a_png(tmp_path) -> None:
    rows = [
        {"h_end": 5, "feature_coverage": 0.9, "label_coverage": 0.95, "survival_to_h": 0.99},
        {"h_end": 20, "feature_coverage": 0.85, "label_coverage": 0.9, "survival_to_h": 0.95},
    ]
    out = plot_coverage_curve(rows, family="fam", output_path=tmp_path / "coverage.png")
    assert out.is_file()


def test_plot_offset_distribution_writes_a_png(tmp_path) -> None:
    offset_summary = {
        "offsets": [
            {"status": "valid", "ic_mean": 0.02},
            {"status": "valid", "ic_mean": 0.03},
            {"status": "insufficient", "ic_mean": None},
        ]
    }
    out = plot_offset_distribution(
        offset_summary, family="fam", output_path=tmp_path / "offsets.png"
    )
    assert out.is_file()


def test_plot_offset_distribution_handles_no_valid_offsets(tmp_path) -> None:
    out = plot_offset_distribution(
        {"offsets": []}, family="fam", output_path=tmp_path / "offsets_empty.png"
    )
    assert out.is_file()


def test_render_family_plots_produces_all_seven_kinds(tmp_path) -> None:
    result = render_family_plots(
        family="fam",
        output_dir=tmp_path,
        cumulative_curves={"broad_common_survivor": [{"h_end": 5, "ic_mean": 0.02}]},
        bucket_rows=[
            {"h_start": 0, "h_end": 5, "ic_mean": 0.02, "t_nw": 2.0, "q_fdr_global": 0.05}
        ],
        expected_sign="+",
        native_rows=[{"h_end": 5, "ic_mean": 0.02}],
        lag1_rows=[{"h_end": 5, "ic_mean": 0.01}],
        period_rows=[{"period_id": "p1", "h_end": 5, "ic_mean": 0.02}],
        segment_rows=[{"segment_axis": "size", "segment": "small", "ic_mean": 0.02}],
        coverage_rows=[
            {"h_end": 5, "feature_coverage": 0.9, "label_coverage": 0.9, "survival_to_h": 0.9}
        ],
        offset_summary={"offsets": [{"status": "valid", "ic_mean": 0.02}]},
    )
    assert set(result) == {
        "cumulative_ic_curve",
        "bucket_ic_bar",
        "native_vs_lag1",
        "subperiod_heatmap",
        "segment_dot",
        "coverage_curve",
        "offset_distribution",
    }
    for path in result.values():
        assert path.is_file() and path.stat().st_size > 0


# --- A-9 markdown report ---


def _sample_card(family: str = "px_reversal_5d", evidence_grade: str = "A") -> dict:
    return build_family_card(
        family=family,
        domain="price",
        primary_feature=f"{family}_feat",
        expected_sign="+",
        observed_sign="+",
        decay_summary=compute_decay_summary([], [], expected_sign="+"),
        pattern_auto=PATTERN_IMMEDIATE,
        primary_discoveries=["cum|0|5"],
        candidate_horizon_band=(0, 5),
        broad_ic=0.05,
        tradable_ic=0.03,
        tradable_retention=0.6,
        valid_subperiods=4,
        sign_consistent_subperiods=3,
        native_ic=0.05,
        lag1_ic=0.04,
        delay_pass=True,
        common_survivor_ic=0.05,
        available_ic=0.04,
        attrition_warning=False,
        nonoverlap_offset_summary=None,
        kospi_weight_mean=0.5,
        kosdaq_weight_mean=0.5,
        p_temporal_nw=None,
        temporal_null_pass=None,
        q_fdr_global=0.02,
        evidence_grade=evidence_grade,
        screen_pass=True,
    )


def _report_context(**overrides) -> dict:
    context = {
        "run_identity": {
            "run_id": "20260802-abcdef01",
            "snapshot_date": "2026-08-01",
            "source": "sj2_remote",
            "config_hash": "cfg-hash",
            "official": True,
            "started_at": "2026-08-02T00:00:00+09:00",
            "finished_at": "2026-08-02T03:00:00+09:00",
        },
        "preflight": {"status": "ok"},
        "sample_coverage": {
            "holdout_start": "2025-08-01",
            "effective_sample_start": "2014-06-01",
            "effective_sample_end": "2025-07-31",
            "common_formation_end": "2023-11-01",
        },
        "bh_summary": {
            "n_hypotheses": 75,
            "n_valid": 75,
            "n_bh_pass": 31,
            "n_primary_discovery": 31,
            "q_threshold": 0.10,
        },
        "short_exploratory_summary": {"n_cells": 28, "n_valid": 28},
        "permutation_summary": {
            "real_discovery_count": 31,
            "p_empirical_count": 0.01,
            "n_replicates": 100,
        },
        "temporal_summary": {
            "n_replicates": 100,
            "per_cell": {
                "famA|feat|cum|0|60": {"p_temporal_nw": 0.05, "temporal_null_pass": True}
            },
        },
        "price_cards": [_sample_card("px_reversal_5d")],
        "flow_cards": [_sample_card("flow_foreign_netbuy_to_volume", evidence_grade="B")],
        "warnings": ["px_amihud_20d: sparse primary grid"],
        "acceptance_gate": ["px_reversal_5d: (0,5] -> 5d label candidate"],
        "deferred_candidates": ["flow_short_turnover: exploratory only"],
        "limitations": ["survivorship bias unresolved for all 60/120d cards"],
    }
    context.update(overrides)
    return context


def test_render_markdown_report_has_all_eleven_sections_in_order() -> None:
    text = render_markdown_report(_report_context())
    expected_order = [
        "## 1. Run identity",
        "## 2. Sample, coverage",
        "## 3. Global BH summary",
        "## 4. Real discovery count",
        "## 5. Long-cell NW t",
        "## 6. Price overview",
        "## 7. Flow overview",
        "## 8. Segment, delay",
        "## 9. Acceptance-gate",
        "## 10. No-signal",
        "## 11. Limitations",
    ]
    positions = [text.index(h) for h in expected_order]
    assert positions == sorted(positions)


def test_render_markdown_report_includes_key_values() -> None:
    text = render_markdown_report(_report_context())
    assert "cfg-hash" in text
    assert "31" in text
    assert "px_reversal_5d" in text
    assert "flow_foreign_netbuy_to_volume" in text


def test_render_markdown_report_rejects_missing_context_key() -> None:
    context = _report_context()
    del context["limitations"]
    with pytest.raises(ValueError, match="missing keys"):
        render_markdown_report(context)
