from __future__ import annotations

from copy import deepcopy

import pytest
from modeler.analysis.horizon_scan_config import (
    CONFIG_PATH,
    _canonical_hash,
    bucket_primary_cells,
    load_config,
    validate_config,
)
from modeler.analysis.horizon_scan_readiness import build_readiness_rows


def test_preregistered_registry_and_hypothesis_count() -> None:
    config = load_config()
    assert len(config.families) == 25
    assert sum(f["phase"] == "A" for f in config.families) == 17
    assert config.primary_hypothesis_count == 75
    assert sum(f["role"] == "exploratory_short_regime" for f in config.families) == 4


def test_expansion_config_extends_base_without_mutating_it() -> None:
    expansion_path = CONFIG_PATH.with_name("horizon_scan_expansion_20260827.yaml")
    base = load_config()
    expansion = load_config(expansion_path)

    assert base.raw["schema_version"] == 4
    assert len(base.families) == 25
    assert expansion.raw["schema_version"] == 5
    assert len(expansion.families) == 35
    assert sum(f["phase"] == "A" for f in expansion.families) == 17
    assert sum(f["phase"] == "B" for f in expansion.families) == 18
    assert expansion.raw["phase_b"]["primary_candidate_count_max"] == 78
    assert expansion.primary_hypothesis_count == 75
    assert expansion.config_hash != base.config_hash

    new_names = {f["family"] for f in expansion.families} - {f["family"] for f in base.families}
    assert new_names == {
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
    }
    regimes = expansion.raw["phase_c"]["regime_candidates"]
    assert {row["feature_code"] for row in regimes} == {
        "macro_unemployment_rate_level",
        "macro_employment_rate_level",
    }


def test_bucket_cells_are_grid_intersection_not_cumulative_count() -> None:
    """px_reversal_5d's cumulative {1,2,3,5,10} only meets the bucket grid at
    {5,10} — a naive len(primary_horizon_set) bucket count would wrongly give 5
    (04_specific_plan_A.md §2.1 table: 5 cumulative + 2 bucket = 7 BH cells)."""
    config = load_config()
    buckets = config.buckets
    by_family = {f["family"]: f for f in config.families}

    reversal = by_family["px_reversal_5d"]
    assert bucket_primary_cells(reversal, buckets) == [(0, 5), (5, 10)]
    assert len(reversal["primary_horizon_set"]) + len(bucket_primary_cells(reversal, buckets)) == 7

    holding_chg = by_family["flow_foreign_holding_ratio_chg"]
    assert bucket_primary_cells(holding_chg, buckets) == [(10, 20), (20, 40), (40, 60)]

    short_turnover = by_family["flow_short_turnover"]
    assert (
        len(short_turnover["primary_horizon_set"])
        + len(bucket_primary_cells(short_turnover, buckets))
        == 10
    )


def test_missing_marts_and_unresolved_lag_are_explicit() -> None:
    config = load_config()
    rows = build_readiness_rows(
        config,
        columns_by_mart={"feat_price": {"px_reversal_5d"}},
        publication_lag_verified=False,
    )
    by_family = {row["family"]: row for row in rows}
    assert by_family["px_reversal_5d"]["status"] == "blocked_missing_dependency"
    assert by_family["flow_short_interest"]["status"] == "exploratory_blocked_publication_lag"
    assert by_family["fin_sue"]["status"] == "phase_b_blocked"


def test_validator_rejects_overlapping_horizons_and_missing_sign() -> None:
    raw = deepcopy(load_config().raw)
    raw["families"][0]["exploratory_horizon_set"].append(
        raw["families"][0]["primary_horizon_set"][0]
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_config(raw)

    raw = deepcopy(load_config().raw)
    del raw["families"][0]["expected_sign"]
    with pytest.raises(ValueError, match="expected_sign"):
        validate_config(raw)


def test_phase_a_discovery_population_is_fixed() -> None:
    config = load_config()
    assert config.raw["discovery"]["universe"] == "broad"
    assert config.raw["discovery"]["sample_kind"] == "common_survivor"

    raw = deepcopy(config.raw)
    raw["discovery"]["universe"] = "tradable"
    with pytest.raises(ValueError, match="discovery.universe"):
        validate_config(raw)


def test_phase_a_missing_section_is_rejected() -> None:
    raw = deepcopy(load_config().raw)
    del raw["evidence_grade"]
    with pytest.raises(ValueError, match="missing keys"):
        validate_config(raw)


def test_evidence_grade_rubric_and_evaluation_order() -> None:
    config = load_config()
    assert config.raw["evidence_grade"]["evaluation_order"] == ["R", "C", "A", "B", "D"]

    raw = deepcopy(config.raw)
    del raw["evidence_grade"]["B"]
    with pytest.raises(ValueError, match="evidence_grade missing keys"):
        validate_config(raw)

    raw = deepcopy(config.raw)
    raw["evidence_grade"]["evaluation_order"] = ["A", "B", "C", "D", "R"]
    with pytest.raises(ValueError, match="R/C before"):
        validate_config(raw)


def test_sparse_primary_grid_families_must_be_registered() -> None:
    raw = deepcopy(load_config().raw)
    raw["stats"]["sparse_primary_grid_families"] = ["does_not_exist"]
    with pytest.raises(ValueError, match="unknown families"):
        validate_config(raw)


def test_bh_missing_p_value_is_fixed_at_one() -> None:
    raw = deepcopy(load_config().raw)
    raw["stats"]["bh_missing_p_value"] = 0.5
    with pytest.raises(ValueError, match="bh_missing_p_value"):
        validate_config(raw)


def test_scan_engine_is_runtime_only_for_config_hash() -> None:
    raw = {
        "execution": {"scan_engine": "legacy", "mapping_contract_version": "joint_cs_v2"},
        "sample": {"start": "2014-06-01"},
    }
    native = {
        **raw,
        "execution": {**raw["execution"], "scan_engine": "polars_native_v1"},
    }
    changed_contract = {
        **raw,
        "execution": {**raw["execution"], "mapping_contract_version": "v1"},
    }

    assert _canonical_hash(raw) == _canonical_hash(native)
    assert _canonical_hash(raw) != _canonical_hash(changed_contract)


# --- macro overlay + Phase C contract (01_design/03 §3.2, 04 §1) ---

MACRO_PATH = CONFIG_PATH.with_name("horizon_scan_macro_20260829.yaml")


def _macro_raw() -> dict:
    from modeler.analysis.horizon_scan_config import _load_raw

    return _load_raw(MACRO_PATH)


def test_macro_overlay_extends_the_expansion_without_moving_either_base_hash() -> None:
    """Both earlier layers' hashes are their preregistration identity — every
    published Phase A/B/AB run is keyed on one of them. A third layer that
    disturbed either would detach those runs from their own contract."""
    base = load_config()
    expansion = load_config(CONFIG_PATH.with_name("horizon_scan_expansion_20260827.yaml"))
    macro = load_config(MACRO_PATH)

    assert base.config_hash == "ab0de63411c40ca3b59c1c7e6f8653a8e16d980108bee42f5f8cea8e7fcb6588"
    assert (
        expansion.config_hash == "889c3e8377c2f400907611f7402651eee6a23c2765c051e4eb2a4a59ca36cbea"
    )
    assert macro.config_hash not in {base.config_hash, expansion.config_hash}
    assert macro.raw["preregistration"]["base_config_hash_prefix"] == expansion.config_hash[:8]


def test_macro_overlay_appends_six_families_and_twentyfour_cells() -> None:
    expansion = load_config(CONFIG_PATH.with_name("horizon_scan_expansion_20260827.yaml"))
    macro = load_config(MACRO_PATH)

    new = {f["family"] for f in macro.families} - {f["family"] for f in expansion.families}
    assert new == {
        "macro_beta_usdkrw",
        "macro_beta_wti",
        "macro_beta_kr10y",
        "macro_beta_sp500_lag",
        "macro_beta_vix",
        "px_market_beta",
    }
    assert len(macro.families) == 41
    assert sum(f["phase"] == "A" for f in macro.families) == 17
    assert sum(f["phase"] == "B" for f in macro.families) == 24
    # 78 + 6 families x (2 cumulative + 2 bucket) = 102.
    assert macro.raw["phase_b"]["primary_candidate_count_max"] == 102
    # Phase A's own population is untouched: the append is Phase-B-only.
    assert macro.primary_hypothesis_count == 75


def test_macro_families_are_blocked_and_outside_fdr() -> None:
    macro = load_config(MACRO_PATH)
    for family in macro.families:
        if family.get("fdr_family") != "macro_exposure":
            continue
        assert family["role"] == "phase_b_blocked"
        assert family["fdr_include"] is False
        assert family["primary_horizon_set"] == [20, 60]
        assert "feat_macro_exposure" in family["readiness_dependencies"]
        assert "common_feature_daily_fact" in family["readiness_dependencies"]


def test_phase_c_registers_fifteen_primary_pairs_and_two_references() -> None:
    """``m`` for Phase C is exactly these 15. A pair dropped by a typo, or one
    added later, moves every q-value in the run."""
    phase_c = load_config(MACRO_PATH).raw["phase_c"]
    assert sum(1 for p in phase_c["pairs"] if p["role"] == "primary") == 15
    assert sum(1 for p in phase_c["pairs"] if p["role"] == "reference") == 2
    assert [p["id"] for p in phase_c["pairs"] if p["role"] == "reference"] == ["X1", "X2"]
    assert phase_c["contract"] == "conditional_ic_v1"
    assert phase_c["grid"] == "krx_sessions"


def test_phase_c_cells_sit_inside_their_familys_preregistered_horizons() -> None:
    macro = load_config(MACRO_PATH)
    horizons = {f["family"]: set(f["primary_horizon_set"]) for f in macro.families}
    regime_ids = {r["id"] for r in macro.raw["phase_c"]["regimes"]}
    for pair in macro.raw["phase_c"]["pairs"]:
        assert pair["cell"]["scan_type"] == "cum"
        assert pair["cell"]["h_end"] in horizons[pair["family"]], pair["id"]
        assert pair["regime"] in regime_ids, pair["id"]
        assert pair["direction"] in {"+", "-", None}


def test_phase_c_exploratory_extras_reference_registered_families_and_regimes() -> None:
    macro = load_config(MACRO_PATH)
    horizons = {f["family"]: set(f["primary_horizon_set"]) for f in macro.families}
    regime_ids = {r["id"] for r in macro.raw["phase_c"]["regimes"]}
    for entry in macro.raw["phase_c"]["exploratory_grid"]["extra"]:
        assert entry["family"] in horizons
        assert entry["cell"]["h_end"] in horizons[entry["family"]]
        assert entry["regime"] in regime_ids


def test_n8_employment_regime_candidates_stay_dormant() -> None:
    """The expansion round registered these as *candidates* and deliberately
    left Phase C closed. Inheriting them must not open them (00_overview §1.4)."""
    phase_c = load_config(MACRO_PATH).raw["phase_c"]
    candidates = {c["feature_code"] for c in phase_c["regime_candidates"]}
    assert candidates == {"macro_unemployment_rate_level", "macro_employment_rate_level"}
    assert candidates & {r["id"] for r in phase_c["regimes"]} == set()
    assert candidates & {p["regime"] for p in phase_c["pairs"]} == set()


def test_phase_c_validator_rejects_a_pair_on_an_unregistered_horizon() -> None:
    raw = _macro_raw()
    # px_idio_vol_60d's primary horizons are [20, 40, 60]; 120 is not one.
    raw["phase_c"]["pairs"][0]["cell"]["h_end"] = 120
    with pytest.raises(ValueError, match="primary_horizon_set"):
        validate_config(raw)


def test_phase_c_validator_rejects_a_pair_on_an_unknown_regime() -> None:
    raw = _macro_raw()
    raw["phase_c"]["pairs"][0]["regime"] = "vix_upp"
    with pytest.raises(ValueError, match="unknown regime"):
        validate_config(raw)


def test_phase_c_validator_rejects_a_pair_on_an_unknown_family() -> None:
    raw = _macro_raw()
    raw["phase_c"]["exploratory_grid"]["extra"][0]["family"] = "px_not_a_family"
    with pytest.raises(ValueError, match="unknown family"):
        validate_config(raw)


def test_phase_c_validator_rejects_a_changed_primary_pair_count() -> None:
    raw = _macro_raw()
    raw["phase_c"]["pairs"].append({**raw["phase_c"]["pairs"][0], "id": "P16"})
    with pytest.raises(ValueError, match="exactly 15 primary pairs"):
        validate_config(raw)


def test_phase_c_validator_rejects_duplicate_regime_ids() -> None:
    raw = _macro_raw()
    raw["phase_c"]["regimes"][1]["id"] = "vix_up"
    with pytest.raises(ValueError, match="regimes ids must be unique"):
        validate_config(raw)


def test_phase_c_validator_requires_a_date_sample_start_and_integer_seed() -> None:
    raw = _macro_raw()
    raw["phase_c"]["sample_start"] = "2015-06-16"  # a string YAML never parsed
    with pytest.raises(ValueError, match="sample_start must be a date"):
        validate_config(raw)

    raw = _macro_raw()
    raw["phase_c"]["placebo"]["seed"] = "20260829"
    with pytest.raises(ValueError, match="seed must be an integer"):
        validate_config(raw)


def test_preregistration_registered_at_must_not_be_a_placeholder() -> None:
    """``registered_at`` is inside the config hash, so a layer committed with a
    placeholder would freeze it into the contract's identity (review M7)."""
    raw = _macro_raw()
    raw["preregistration"]["registered_at"] = "TBD"
    with pytest.raises(ValueError, match="registered_at must be a date"):
        validate_config(raw)


def test_layers_without_registered_pairs_skip_the_phase_c_checks() -> None:
    """The base config and the expansion overlay both carry a ``phase_c`` block
    with only an open policy and dormant candidates — not a contract."""
    base = load_config()
    expansion = load_config(CONFIG_PATH.with_name("horizon_scan_expansion_20260827.yaml"))
    assert "pairs" not in base.raw.get("phase_c", {})
    assert "pairs" not in expansion.raw["phase_c"]


# --------------------------------------------------------------------------
# F-HS-1 — the fourth layer: F-2 relation + F-4 financial risk/lifecycle
# --------------------------------------------------------------------------

F_HS1_PATH = CONFIG_PATH.with_name("horizon_scan_expansion_202609.yaml")

_F_HS1_FAMILIES = {
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


def test_f_hs1_overlay_moves_none_of_the_three_earlier_layer_hashes() -> None:
    """Every published Phase A/B/AB/C run is keyed on one of those three."""
    base = load_config()
    expansion = load_config(CONFIG_PATH.with_name("horizon_scan_expansion_20260827.yaml"))
    macro = load_config(MACRO_PATH)
    layer = load_config(F_HS1_PATH)

    assert base.config_hash == "ab0de63411c40ca3b59c1c7e6f8653a8e16d980108bee42f5f8cea8e7fcb6588"
    assert (
        expansion.config_hash == "889c3e8377c2f400907611f7402651eee6a23c2765c051e4eb2a4a59ca36cbea"
    )
    assert macro.config_hash == "236d0d3515043e44e280f0c2c2707ca2cc486aa44b638eb893a7095ddac1110f"
    assert layer.config_hash not in {base.config_hash, expansion.config_hash, macro.config_hash}
    assert layer.raw["preregistration"]["base_config_hash_prefix"] == macro.config_hash[:8]


def test_f_hs1_appends_thirteen_families_and_sixtysix_cells() -> None:
    macro = load_config(MACRO_PATH)
    layer = load_config(F_HS1_PATH)

    assert {f["family"] for f in layer.families} - {f["family"] for f in macro.families} == (
        _F_HS1_FAMILIES
    )
    assert len(layer.families) == 54
    assert sum(f["phase"] == "A" for f in layer.families) == 17
    assert sum(f["phase"] == "B" for f in layer.families) == 37
    # 102 (macro) + 4 relation x 6 + 42 fin_risk = 168.
    assert layer.raw["phase_b"]["primary_candidate_count_max"] == 168
    # Phase A's own BH population is untouched: the append is Phase-B-only.
    assert layer.primary_hypothesis_count == 75


def test_f_hs1_families_are_blocked_and_outside_fdr() -> None:
    layer = load_config(F_HS1_PATH)
    for family in layer.families:
        if family["family"] not in _F_HS1_FAMILIES:
            continue
        assert family["phase"] == "B"
        assert family["role"] == "phase_b_blocked"
        assert family["fdr_include"] is False
        assert family["official_feature_variant"] == "native_t"
        assert family["fdr_family"] in {"relation", "financial_risk", "lifecycle", "transition"}


def test_f_hs1_families_point_at_columns_their_marts_carry() -> None:
    """The lag1 mapping is the one the validator cannot check: it only requires
    the key to exist. feat_relation_stat had no lag1 columns until
    relation_stat_v2, so naming them here would have frozen a contract on
    columns that do not exist."""
    from modeler.etl.features import fin_risk, relation_stat

    available = {
        *relation_stat.FEATURE_COLUMNS,
        *(f"{c}_lag1" for c in relation_stat.PRIMARY_COLUMNS),
        *fin_risk.PRIMARY_COLUMNS,
        *(f"{c}_lag1" for c in fin_risk.PRIMARY_COLUMNS),
    }
    for family in load_config(F_HS1_PATH).families:
        if family["family"] not in _F_HS1_FAMILIES:
            continue
        for feature in family["features"]:
            assert feature["column"] in available, family["family"]
        for variant in ("native_t", "lag1"):
            assert family["variant_columns"][variant] in available, family["family"]


def test_f_hs1_event_cohort_families_use_the_frozen_event_grid() -> None:
    """The 5%-of-names rule put these two on the event path, so their cells are
    the frozen event grid rather than a horizon set — and they name an event
    mart that does not exist, which is what keeps them out of the SUE-shaped
    cohort scan instead of being handed to it."""
    layer = load_config(F_HS1_PATH)
    event_families = {"fin_dividend_initiation", "fin_negative_equity_exit"}
    for family in layer.families:
        if family["family"] not in event_families:
            continue
        assert family["primary_horizon_set"] == []
        assert family["exploratory_horizon_set"] == []
        assert family["include_bucket_primary"] is False
        assert family["event_buckets"] == [[0, 3], [3, 5], [5, 10], [10, 20], [20, 40], [40, 60]]
        assert "fin_risk_event" in family["readiness_dependencies"]


def test_f_hs1_bidirectional_families_are_exactly_the_reports_seven() -> None:
    """A sign silently filled in after the fact is the failure preregistration
    exists to prevent, so the count is pinned to the two reports' own tables."""
    layer = load_config(F_HS1_PATH)
    bidirectional = {
        f["family"]
        for f in layer.families
        if f["family"] in _F_HS1_FAMILIES and f["expected_sign"] is None
    }
    assert bidirectional == {
        "rel_peer_mom",
        "rel_peer_dispersion",
        "fin_debt_to_assets",
        "fin_net_debt_to_mcap",
        "fin_lifecycle_stage",
        "fin_lifecycle_transition",
    }
    signed = {
        f["family"]: f["expected_sign"]
        for f in layer.families
        if f["family"] in _F_HS1_FAMILIES and f["expected_sign"] is not None
    }
    assert signed == {
        "rel_own_minus_peer": "-",
        "rel_peer_bigcap_lag": "+",
        "fin_interest_coverage": "+",
        "fin_ext_finance": "-",
        "fin_profit_turn": "+",
        "fin_dividend_initiation": "+",
        "fin_negative_equity_exit": "+",
    }


def test_f_hs1_families_all_have_a_source_quality_verdict() -> None:
    """A family missing from FAMILY_METRIC_DEPENDENCIES gets no verdict at all,
    and ``source_quality_allows_grade_a(None)`` caps it at B — a grade cap by
    omission rather than by measurement."""
    from modeler.analysis.horizon_scan_phase_b_source_quality import (
        FAMILY_METRIC_DEPENDENCIES,
    )

    assert _F_HS1_FAMILIES <= set(FAMILY_METRIC_DEPENDENCIES)
