"""Unit tests for model 02's feature sets and spec (20260907_model_experiment).

Two jobs. The feature sets are a preregistration, so these tests are what stops
one from drifting: FS0 must stay byte-identical to model 01's baseline, each set
must contain the one before it, and the expected-sign table must cover exactly
the screened columns. And the spec must refuse to reach past the holdout
boundary — `06` §8 names an accidental ``period_end`` edit as the risk that
would quietly spend the one holdout this experiment has.
"""

from __future__ import annotations

import pytest

from modeler.etl.features.regime import MODEL_REGIME_COLUMNS
from modeler.models._01_20_access_return_rank.experiments.run_grade_a_acceptance_gate import (
    BASELINE_COLS,
    NEW_CANDIDATE_COLS,
)
from modeler.models._02_updown_prob import features as fx
from modeler.models._02_updown_prob.spec import (
    HOLDOUT_START,
    PERIOD_END,
    SPEC_HORIZONS,
    ModelSpec,
)

ALL_SETS = ("FS0", "FS1", "FS1h", "FS2", "FS2h", "FS3", "FS3h")


# --- FS0 is model 01's baseline, unchanged ----------------------------------


def test_fs0_is_byte_identical_to_the_existing_baseline() -> None:
    assert fx.FS0_COLS == BASELINE_COLS
    assert len(fx.FS0_COLS) == 40


def test_fs1_adds_exactly_the_screened_candidates() -> None:
    # the T1 columns are the same five the Grade A gate tested
    assert fx.FS1_T1_COLS == NEW_CANDIDATE_COLS
    assert len(fx.FS1_T2_COLS) == 11
    assert len(fx.FS1_COLS) == 56


# --- the containment chain ---------------------------------------------------


def test_each_feature_set_contains_the_one_before_it() -> None:
    assert set(fx.FS0_COLS) < set(fx.FS1_COLS)
    assert set(fx.FS1_COLS) < set(fx.FS2_COLS)
    assert set(fx.FS2_COLS) < set(fx.FS3_COLS)
    assert [len(fx.FEATURE_SETS[k]) for k in ("FS0", "FS1", "FS2", "FS3")] == [40, 56, 68, 79]


def test_no_feature_set_repeats_a_column() -> None:
    for name in ALL_SETS:
        for horizon in fx.HORIZONS:
            columns = fx.feature_columns(name, horizon)
            assert len(columns) == len(set(columns)), f"{name} h{horizon} repeats a column"


def test_horizon_subsets_only_ever_drop_fs1_additions() -> None:
    for horizon in fx.HORIZONS:
        subset = fx.feature_columns("FS1h", horizon)
        assert set(subset) <= set(fx.FS1_COLS)
        # FS0 stays whole at every h: the baseline must be the same 40 columns
        # everywhere or the per-h comparisons stop being comparable.
        assert set(fx.FS0_COLS) <= set(subset)
        dropped = set(fx.FS1_COLS) - set(subset)
        assert dropped == set(fx.FS1H_DROPPED[horizon])


def test_horizon_subset_sizes_match_the_preregistered_lists() -> None:
    sizes = {h: len(fx.feature_columns("FS1h", h)) for h in fx.HORIZONS}
    # From `02` §1.3's dropped-column lists. h120 is 50: the six listed columns
    # leave ten of FS1's sixteen, and the table's original "40 + 9 = 49" was the
    # typo (settled 2026-09-09, `02` §1.3 note).
    assert sizes == {5: 46, 20: 49, 60: 54, 120: 50}


def test_fs2_and_fs3_stack_on_whichever_fs1_variant_is_in_play() -> None:
    for horizon in fx.HORIZONS:
        assert set(fx.feature_columns("FS2", horizon)) == set(fx.FS1_COLS) | set(fx.FS2_ADDED)
        assert set(fx.feature_columns("FS2h", horizon)) == set(
            fx.feature_columns("FS1h", horizon)
        ) | set(fx.FS2_ADDED)
        assert set(fx.feature_columns("FS3", horizon)) == set(fx.FS2_COLS) | set(fx.FS3_ADDED)


def test_fs3h_keeps_only_the_horizons_each_column_was_tested_at() -> None:
    assert set(fx.FS3_TESTED_HORIZONS) == set(fx.FS3_ADDED)
    for horizon in fx.HORIZONS:
        added = set(fx.feature_columns("FS3h", horizon)) - set(fx.feature_columns("FS2h", horizon))
        assert added == {c for c in fx.FS3_ADDED if horizon in fx.FS3_TESTED_HORIZONS[c]}
    # h120's only tested FS3 column is the net-debt one (`02` §1.5).
    assert {c for c in fx.FS3_ADDED if 120 in fx.FS3_TESTED_HORIZONS[c]} == {"fin_net_debt_to_mcap"}


def test_unknown_set_or_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="feature set"):
        fx.feature_columns("FS9", 20)
    with pytest.raises(ValueError, match="horizon"):
        fx.feature_columns("FS1", 10)


# --- interactions (`02` §3.2) ------------------------------------------------


def test_four_interactions_are_defined_over_the_model_regimes() -> None:
    assert len(fx.INTERACTIONS) == 4
    assert fx.FS2_INTERACTION_COLS == (
        "ix_reversal_vixhigh",
        "ix_tshock_liqhigh",
        "ix_foreign_liqhigh",
        "ix_mktbeta_mktup",
    )
    for _name, _feature, regime in fx.INTERACTIONS:
        assert regime in MODEL_REGIME_COLUMNS


def test_grade_d_materials_enter_only_through_their_interaction() -> None:
    """px_turnover_shock and the foreign flow ratio are stable-*opposite*-sign.

    Putting them in FS2 as plain columns would be a different hypothesis (E4c,
    sign flipped), so FS2 carries the interaction and not the material.
    """
    assert "px_turnover_shock" in fx.INTERACTION_MATERIAL_COLS
    assert "flow_foreign_netbuy_to_volume_20d_lag1" in fx.INTERACTION_MATERIAL_COLS
    assert "px_turnover_shock" not in fx.FS3_COLS
    assert "flow_foreign_netbuy_to_volume_20d_lag1" not in fx.FS3_COLS
    # the two that are already model features stay features.
    assert "px_reversal_5d" in fx.FS1_COLS
    assert "px_market_beta" in fx.FS2_COLS


def test_fs2_adds_four_betas_four_regimes_four_interactions() -> None:
    assert len(fx.FS2_MACRO_BETA_COLS) == 4
    assert fx.FS2_REGIME_COLS == MODEL_REGIME_COLUMNS
    assert len(fx.FS2_ADDED) == 12


# --- mart mapping (`02` §5) --------------------------------------------------


def test_every_feature_column_maps_to_a_mart() -> None:
    built = set(fx.FS2_INTERACTION_COLS)
    for name in ALL_SETS:
        for horizon in fx.HORIZONS:
            for column in fx.feature_columns(name, horizon):
                if column in built:
                    continue  # built by the panel builder, not read from a mart
                assert column in fx.COLUMN_MART, f"{column} has no mart"


def test_interaction_materials_map_to_a_mart_too() -> None:
    for column in fx.INTERACTION_MATERIAL_COLS:
        assert column in fx.COLUMN_MART


def test_mart_names_are_registered_group_views() -> None:
    assert set(fx.COLUMN_MART.values()) <= set(fx.GROUP_VIEW.values())
    # the three filing-count families are the easy ones to mis-map: their
    # ev_/own_ prefixes point at the scan marts, but they live in filing_activity.
    assert fx.COLUMN_MART["ev_filing_burst_60d"] == "feat_filing_activity"
    assert fx.COLUMN_MART["own_major_filing_60d"] == "feat_filing_activity"
    assert fx.COLUMN_MART["ev_payout_yield"] == "feat_event_scan_daily"
    assert fx.COLUMN_MART["px_market_beta"] == "feat_macro_exposure"
    assert fx.COLUMN_MART["rg_vix_high"] == "dim_regime_daily"


def test_group_view_extends_model_01s_mapping_without_changing_it() -> None:
    from modeler.models._01_20_access_return_rank.build_dataset import _GROUP_VIEW

    for group, view in _GROUP_VIEW.items():
        assert fx.GROUP_VIEW[group] == view


# --- flow timing (`02` §2, D-6) ----------------------------------------------


def test_fs0_flow_columns_are_the_ones_the_builder_has_to_lag() -> None:
    assert fx.FLOW_BUILDER_LAG_COLUMNS == fx.FS0_FLOW_COLS
    # ...and none of them has a _lag1 twin to read instead.
    assert not set(fx.FLOW_BUILDER_LAG_COLUMNS) & set(fx.FLOW_MART_LAG1_SOURCE)


def test_mart_lag1_sources_are_fs1_flow_columns_named_with_the_suffix() -> None:
    for column, source in fx.FLOW_MART_LAG1_SOURCE.items():
        assert column in fx.FS1_COLS
        assert source == f"{column}_lag1"


# --- expected signs (E4b) ----------------------------------------------------


def test_expected_signs_cover_exactly_the_screened_columns() -> None:
    assert set(fx.EXPECTED_SIGN) == set(fx.FS1_ADDED) | set(fx.FS3_ADDED)
    assert set(fx.EXPECTED_SIGN.values()) == {1, -1}
    # FS0 was never screened and FS2's conditional effects have no prior sign.
    assert not set(fx.EXPECTED_SIGN) & set(fx.FS0_COLS)
    assert not set(fx.EXPECTED_SIGN) & set(fx.FS2_ADDED)


def test_monotonic_constraints_are_zero_where_no_sign_was_preregistered() -> None:
    columns = fx.feature_columns("FS2", 20)
    constraints = fx.monotonic_constraints(columns)
    assert len(constraints) == len(columns)
    by_column = dict(zip(columns, constraints, strict=True))
    assert by_column["px_reversal_5d"] == 1
    assert by_column["fin_log_mcap"] == -1
    assert by_column["px_ret_1d"] == 0  # FS0
    assert by_column["rg_vix_high"] == 0  # FS2


# --- the spec ----------------------------------------------------------------


def test_default_spec_is_the_m0_decision_set() -> None:
    spec = ModelSpec()
    assert spec.horizons == SPEC_HORIZONS == (20, 5, 60, 120)
    assert tuple(spec.label.outputs) == ("rank", "up", "top")
    assert spec.feature_set == "FS0"
    assert spec.flow_variant == "lag1"
    assert spec.preprocess_profile == "rank"
    assert spec.period_end == PERIOD_END == "2025-07-31"
    assert HOLDOUT_START == "2025-08-01"
    assert (spec.topk, spec.cost_bps_roundtrip, spec.tau) == (100, 60.0, 0.6)
    assert spec.universe.min_liquidity_krw == 1e8  # D-5: the default filter


def test_a_period_end_past_the_holdout_boundary_is_refused() -> None:
    with pytest.raises(ValueError, match="holdout boundary"):
        ModelSpec(period_end="2025-08-01")
    with pytest.raises(ValueError, match="holdout boundary"):
        ModelSpec(period_end="2026-06-10")  # model 01's period_end
    # the boundary itself, and anything earlier, is fine.
    assert ModelSpec(period_end="2025-07-31").period_end == "2025-07-31"
    assert ModelSpec(period_end="2024-12-31").period_end == "2024-12-31"


def test_spec_rejects_unknown_knobs() -> None:
    with pytest.raises(ValueError, match="preregistered set"):
        ModelSpec(horizons=(10,))
    with pytest.raises(ValueError, match="feature_set"):
        ModelSpec(feature_set="FS9")
    with pytest.raises(ValueError, match="flow_variant"):
        ModelSpec(flow_variant="native")
    with pytest.raises(ValueError, match="preprocess_profile"):
        ModelSpec(preprocess_profile="linear")  # a probability model is not Ridge
    with pytest.raises(ValueError, match="n_folds"):
        ModelSpec(n_folds=1)
    with pytest.raises(ValueError, match="tau"):
        ModelSpec(tau=1.0)


def test_label_horizons_cannot_drift_from_spec_horizons() -> None:
    from modeler.etl.labels import LabelSpec

    with pytest.raises(ValueError, match="label horizons"):
        ModelSpec(label=LabelSpec(horizons=(20, 5), outputs=("rank", "up", "top")))


def test_for_horizon_promotes_one_horizon_and_keeps_the_rest() -> None:
    spec = ModelSpec().for_horizon(120)
    assert spec.primary_horizon == 120
    assert set(spec.horizons) == set(SPEC_HORIZONS)
    assert tuple(spec.label.horizons) == spec.horizons
    assert spec.label_column("y_up") == "y_up_120d"
    assert spec.realized_column() == "raw_label_120d"


def test_fold_kwargs_purge_and_embargo_equal_the_horizon() -> None:
    spec = ModelSpec()
    for horizon in spec.horizons:
        kwargs = spec.fold_kwargs(horizon)
        assert kwargs["embargo"] == kwargs["purge"] == horizon
        # holdout_len 0: the trailing holdout is outside period_end already, so
        # reserving another slice would only shrink the selection window.
        assert kwargs["holdout_len"] == 0
        assert kwargs["n_folds"] == 5


def test_spec_feature_columns_follow_the_feature_set() -> None:
    assert ModelSpec().feature_columns(20) == fx.FS0_COLS
    assert len(ModelSpec(feature_set="FS1h").feature_columns(5)) == 46


def test_label_and_realized_columns_reject_unknown_targets_and_horizons() -> None:
    spec = ModelSpec()
    with pytest.raises(ValueError, match="target"):
        spec.label_column("y_absup")
    with pytest.raises(ValueError, match="horizon"):
        spec.label_column("y_up", 10)
    with pytest.raises(ValueError, match="horizon"):
        spec.realized_column(10)


# --- E5's composite feature sets (`05` §2 E5, `02` §1.5) ---------------------


def test_fs3_is_added_to_the_adopted_base_not_to_fs2() -> None:
    """`05` §2 E5 compares "the adopted config -> + FS3".

    At h5 and h60 the adopted config is FS0, so plain "FS3" (FS2 + 11) would
    bring back the 28 columns E2 rejected and confound the comparison.
    """
    for horizon in fx.HORIZONS:
        composite = fx.feature_columns("FS0_FS3", horizon)
        assert set(composite) == set(fx.FS0_COLS) | set(fx.FS3_ADDED)
        assert len(composite) == len(fx.FS0_COLS) + len(fx.FS3_ADDED)


def test_the_horizon_subset_follows_the_base() -> None:
    """`02` §1.5: subset FS3 when the adopted config is itself horizon-scoped."""
    for horizon in fx.HORIZONS:
        scoped = fx.feature_columns("FS1h_FS3", horizon)
        added = set(scoped) - set(fx.feature_columns("FS1h", horizon))
        assert added == {
            c for c in fx.FS3_ADDED if horizon in fx.FS3_TESTED_HORIZONS[c]
        }
        # an unscoped base keeps all eleven at every horizon
        whole = fx.feature_columns("FS1_FS3", horizon)
        assert set(whole) - set(fx.feature_columns("FS1", horizon)) == set(fx.FS3_ADDED)


def test_the_composite_subsumes_the_fixed_fs3_ids() -> None:
    """FS3 was always "FS2 + 11", so the general form has to agree with it."""
    for horizon in fx.HORIZONS:
        assert fx.feature_columns("FS2_FS3", horizon) == fx.feature_columns("FS3", horizon)
        assert fx.feature_columns("FS2h_FS3", horizon) == fx.feature_columns("FS3h", horizon)


def test_an_unknown_composite_base_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown feature set"):
        fx.feature_columns("FS9_FS3", 20)
    with pytest.raises(ValueError, match="unknown feature set"):
        fx.feature_columns("FS3_FS3", 20)
