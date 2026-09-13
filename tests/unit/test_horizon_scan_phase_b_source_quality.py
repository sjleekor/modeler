"""§9 B-9 source 비치명 경고 — the grade-A cap from B-10 Stage 2's diagnostics."""

from __future__ import annotations

from modeler.analysis.horizon_scan_phase_b_source_quality import (
    FAMILY_METRIC_DEPENDENCIES,
    MAPPING_FALLBACK_WARN,
    PAIRING_MISMATCH_WARN,
    REVISION_WARN,
    compute_family_source_quality,
    source_quality_allows_grade_a,
)


def _vintage_row(
    *,
    metric_code: str,
    rows: int = 1000,
    mapping_fallback_rows: int = 0,
    revision_known_rows: int | None = 1000,
    revision_rows: int = 0,
) -> dict:
    return {
        "metric_code": metric_code,
        "bsns_year": 2024,
        "reprt_code": "11011",
        "rows": rows,
        "mapping_fallback_rows": mapping_fallback_rows,
        "revision_known_rows": revision_known_rows,
        "revision_rows": revision_rows,
    }


def _pairing_row(*, applicable_rows: int = 1000, value_mismatch_rows: int = 0) -> dict:
    return {
        "bsns_year": 2024,
        "reprt_code": "11011",
        "applicable_rows": applicable_rows,
        "value_mismatch_rows": value_mismatch_rows,
    }


def _clean_vintage_rows() -> list[dict]:
    metrics = {m for metrics in FAMILY_METRIC_DEPENDENCIES.values() for m in metrics}
    return [_vintage_row(metric_code=m) for m in sorted(metrics)]


def test_every_phase_b_family_has_a_declared_dependency_set() -> None:
    # Absent would silently mean "no verdict"; empty means "nothing to check".
    assert set(FAMILY_METRIC_DEPENDENCIES) == {
        "fin_log_mcap",
        "fin_value_z",
        "fin_gross_profitability",
        "fin_asset_growth_yoy",
        "fin_accruals_to_assets",
        "fin_sue",
        "ev_net_share_issuance_yoy",
        "ev_payout_yield",
        "mcap_krx_log",
        "ev_filing_activity",
        "ev_amendment_ratio",
        "own_insider_filing_activity",
        "own_major_filing_activity",
        "own_amendment_ratio",
        "hc_employee_growth",
        "hc_productivity",
        "own_major_stake_level",
        "own_major_stake_change",
        "rel_peer_mom",
        "rel_own_minus_peer",
        "rel_peer_bigcap_lag",
        "rel_peer_dispersion",
        "fin_debt_to_assets",
        "fin_net_debt_to_mcap",
        "fin_interest_coverage",
        "fin_ext_finance",
        "fin_lifecycle_stage",
        "fin_lifecycle_transition",
        "fin_profit_turn",
        "fin_dividend_initiation",
        "fin_negative_equity_exit",
    }


def test_clean_source_layer_allows_grade_a() -> None:
    verdicts = compute_family_source_quality(
        vintage_quality_rows=_clean_vintage_rows(),
        pairing_quality_rows=[_pairing_row()],
    )

    assert verdicts["fin_value_z"]["source_quality_status"] == "ok"
    assert verdicts["fin_value_z"]["mapping_fallback_ratio"] == 0.0
    assert verdicts["fin_value_z"]["revision_ratio"] == 0.0
    assert source_quality_allows_grade_a(verdicts["fin_value_z"]["source_quality_status"])


def test_families_that_never_read_the_metric_layer_are_not_applicable() -> None:
    verdicts = compute_family_source_quality(
        vintage_quality_rows=_clean_vintage_rows(), pairing_quality_rows=[_pairing_row()]
    )

    # Market cap is priced, and issuance comes straight off raw share counts.
    for family in ("fin_log_mcap", "ev_net_share_issuance_yoy"):
        assert verdicts[family]["source_quality_status"] == "not_applicable"
        assert source_quality_allows_grade_a(verdicts[family]["source_quality_status"])


def test_mapping_fallback_over_the_threshold_warns_only_the_families_that_read_it() -> None:
    rows = _clean_vintage_rows()
    # total_assets is read by asset growth, accruals and profitability, but not
    # by value — the warning must not spread to a family that never touches it.
    rows = [
        (
            _vintage_row(
                metric_code="total_assets",
                mapping_fallback_rows=int(1000 * MAPPING_FALLBACK_WARN) + 1,
            )
            if r["metric_code"] == "total_assets"
            else r
        )
        for r in rows
    ]
    verdicts = compute_family_source_quality(
        vintage_quality_rows=rows, pairing_quality_rows=[_pairing_row()]
    )

    assert verdicts["fin_asset_growth_yoy"]["source_quality_status"] == "warn"
    assert "mapping_fallback" in verdicts["fin_asset_growth_yoy"]["source_quality_reasons"]
    assert verdicts["fin_asset_growth_yoy"]["mapping_fallback_worst_metric"] == "total_assets"
    assert verdicts["fin_value_z"]["source_quality_status"] == "ok"


def test_the_worst_input_metric_decides_not_the_pooled_average() -> None:
    rows = [r for r in _clean_vintage_rows() if r["metric_code"] != "controlling_net_income"]
    # fin_sue reads controlling_net_income and weighted_avg_shares. Pooling the
    # two would read 0.05 and pass; the feature is an EPS ratio, so a
    # half-revised numerator compromises it however clean the denominator is.
    rows.append(_vintage_row(metric_code="controlling_net_income", revision_rows=500))
    verdicts = compute_family_source_quality(
        vintage_quality_rows=rows, pairing_quality_rows=[_pairing_row()]
    )

    assert verdicts["fin_sue"]["source_quality_status"] == "warn"
    assert verdicts["fin_sue"]["revision_ratio"] == 0.5
    assert verdicts["fin_sue"]["revision_worst_metric"] == "controlling_net_income"


def test_the_ratio_is_row_weighted_not_a_mean_of_per_year_ratios() -> None:
    rows = _clean_vintage_rows()
    rows = [r for r in rows if r["metric_code"] != "total_assets"]
    # One tiny fully-fallback year and one large clean year. A mean of ratios
    # would read 0.5 and trip the threshold; the row-weighted ratio is 0.004.
    rows.append(_vintage_row(metric_code="total_assets", rows=40, mapping_fallback_rows=40))
    rows.append(_vintage_row(metric_code="total_assets", rows=10_000, mapping_fallback_rows=0))
    verdicts = compute_family_source_quality(
        vintage_quality_rows=rows, pairing_quality_rows=[_pairing_row()]
    )

    assert verdicts["fin_asset_growth_yoy"]["mapping_fallback_ratio"] < MAPPING_FALLBACK_WARN
    assert verdicts["fin_asset_growth_yoy"]["source_quality_status"] == "ok"


def test_revision_ratio_over_the_threshold_warns() -> None:
    rows = [
        (
            _vintage_row(metric_code=r["metric_code"], revision_rows=int(1000 * REVISION_WARN) + 1)
            if r["metric_code"] == "controlling_net_income"
            else r
        )
        for r in _clean_vintage_rows()
    ]
    verdicts = compute_family_source_quality(
        vintage_quality_rows=rows, pairing_quality_rows=[_pairing_row()]
    )

    assert verdicts["fin_sue"]["source_quality_status"] == "warn"
    assert "revision" in verdicts["fin_sue"]["source_quality_reasons"]


def test_unmeasurable_revision_ratio_caps_just_like_a_breach() -> None:
    # Today's real state: no receipt history, so is_revision is unknown for
    # every row and revision_known_rows is 0.
    rows = [
        _vintage_row(metric_code=r["metric_code"], revision_known_rows=0)
        for r in _clean_vintage_rows()
    ]
    verdicts = compute_family_source_quality(
        vintage_quality_rows=rows, pairing_quality_rows=[_pairing_row()]
    )

    assert verdicts["fin_value_z"]["source_quality_status"] == "unmeasured"
    assert verdicts["fin_value_z"]["revision_ratio"] is None
    assert not source_quality_allows_grade_a(verdicts["fin_value_z"]["source_quality_status"])


def test_pairing_mismatch_is_lake_wide_and_hits_every_metric_reading_family() -> None:
    verdicts = compute_family_source_quality(
        vintage_quality_rows=_clean_vintage_rows(),
        pairing_quality_rows=[
            _pairing_row(
                applicable_rows=100_000,
                value_mismatch_rows=int(100_000 * PAIRING_MISMATCH_WARN) + 1,
            )
        ],
    )

    for family in ("fin_value_z", "fin_sue", "ev_payout_yield"):
        assert verdicts[family]["source_quality_status"] == "warn"
        assert "pairing_mismatch" in verdicts[family]["source_quality_reasons"]
    # Still not applicable to the families that never read the metric layer.
    assert verdicts["fin_log_mcap"]["source_quality_status"] == "not_applicable"


def test_a_pairing_rate_below_the_threshold_does_not_warn() -> None:
    # The measured rate after the 2026-08-12 B-2 fix: 157 of 1,346,588.
    verdicts = compute_family_source_quality(
        vintage_quality_rows=_clean_vintage_rows(),
        pairing_quality_rows=[_pairing_row(applicable_rows=1_346_588, value_mismatch_rows=157)],
    )

    assert verdicts["fin_value_z"]["source_quality_status"] == "ok"
    assert verdicts["fin_value_z"]["pairing_mismatch_ratio"] < PAIRING_MISMATCH_WARN


def test_no_diagnostics_at_all_is_unmeasured_not_clean() -> None:
    verdicts = compute_family_source_quality()

    assert verdicts["fin_value_z"]["source_quality_status"] == "unmeasured"
    assert "no_vintage_quality_rows" in verdicts["fin_value_z"]["source_quality_reasons"]
    assert not source_quality_allows_grade_a(None)


def test_unmeasured_wins_over_warn_when_both_are_present() -> None:
    rows = [
        _vintage_row(metric_code=r["metric_code"], revision_known_rows=0)
        for r in _clean_vintage_rows()
    ]
    verdicts = compute_family_source_quality(
        vintage_quality_rows=rows,
        pairing_quality_rows=[
            _pairing_row(applicable_rows=1000, value_mismatch_rows=500),
        ],
    )

    # "Could not check" is the weaker claim, so it is what the card reports —
    # both reasons are still listed.
    assert verdicts["fin_value_z"]["source_quality_status"] == "unmeasured"
    reasons = verdicts["fin_value_z"]["source_quality_reasons"]
    assert "revision_unmeasured" in reasons
    assert "pairing_mismatch" in reasons
