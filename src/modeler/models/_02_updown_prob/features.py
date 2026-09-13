"""Feature sets FS0-FS3 for model 02, fixed as column lists in code.

Plan: ``docs/dev/20260907_model_experiment/02_features_and_preprocessing.md``
§1 (the sets), §2 (PIT), §3.2 (interactions), §5 (mart mapping).

The lists live in code rather than YAML on purpose (`02` §0): a preregistered
feature set that can be edited between runs is not preregistered. Every set is
a superset of the one before it — FS0 (40) -> FS1 (56) -> FS2 (68) -> FS3 (79)
— so an E-stage comparison is always "the same columns plus these".

Two things this module does *not* do. It does not screen features: FS1's
additions are there because the univariate horizon scan passed them (T1/T2),
FS3's because F-HS-1 did, and a model is not allowed to re-decide that. And it
does not read a mart — ``COLUMN_MART`` only records where each column comes
from, so the builder (M-PR5) can register the views it needs and a test can
check the mapping is complete.
"""

from __future__ import annotations

from modeler.etl.features.regime import MODEL_REGIME_COLUMNS, REGIME_TABLE

HORIZONS: tuple[int, ...] = (5, 20, 60, 120)

# --- FS0: the existing model's baseline 40 (`02` §1.1) ----------------------
# Byte-identical to run_grade_a_acceptance_gate.BASELINE_COLS — pinned by
# test_updown_features. These 40 were never univariately screened and
# feat_fin_pit's +90/+45 day PIT lag differs from the validation track's strict
# vintage; both stay as they are, because changing FS0 would break the chain
# back to model 01's gate numbers. FS0's job is to be the comparison baseline.

FS0_PX_COLS: tuple[str, ...] = (
    "px_ret_1d",
    "px_ret_5d",
    "px_ret_20d",
    "px_ret_60d",
    "px_mom_20_60",
    "px_vol_20d",
    "px_vol_60d",
    "px_high_low_range_20d",
    "px_turnover",
    "px_turnover_ma20",
    "px_amihud_20d",
    "px_gap_vs_ma20",
    "px_dist_52w_high",
    "px_is_halted",
    "px_halt_ratio_20d",
)
FS0_FLOW_COLS: tuple[str, ...] = (
    "flow_foreign_netbuy_sum_5d",
    "flow_foreign_netbuy_sum_20d",
    "flow_inst_netbuy_sum_5d",
    "flow_inst_netbuy_sum_20d",
    "flow_indiv_netbuy_sum_5d",
    "flow_indiv_netbuy_sum_20d",
    "flow_foreign_holding_chg_5d",
    "flow_foreign_holding_chg_20d",
    "flow_short_balance_chg_20d",
    "flow_foreign_netbuy_z_20d",
    "flow_inst_netbuy_z_20d",
    "flow_short_avg_price",
    "flow_short_selling_volume",
    "flow_short_selling_value",
    "flow_short_balance_qty",
)
FS0_FIN_COLS: tuple[str, ...] = (
    "fin_roa",
    "fin_roe",
    "fin_operating_margin",
    "fin_debt_to_equity",
    "fin_equity_ratio",
    "fin_ocf_to_assets",
    "fin_cash_ratio",
    "fin_asset_turnover",
    "fin_is_negative_equity",
    "fin_has_fs",
)
FS0_COLS: tuple[str, ...] = (*FS0_PX_COLS, *FS0_FLOW_COLS, *FS0_FIN_COLS)

# --- FS1: + the screened candidates (`02` §1.2) ------------------------------
# T1 Grade A (5). px_amihud_20d / px_near_52w_high are already in FS0 (the
# latter as px_dist_52w_high, same formula), so only the net-new ones appear.
FS1_T1_COLS: tuple[str, ...] = (
    "px_reversal_5d",
    "px_maxret_20d",
    "px_idio_vol_60d",
    "flow_individual_netbuy_to_volume_5d",
    "flow_individual_netbuy_to_volume_20d",
)
# T2 screen_pass, minus 3 duplicates: mcap_krx_log (fin_log_mcap is the one that
# passed the time placebo), own_amendment_ratio_1y (subset of ev_), own_major_stake
# (own_major_stake_chg is better on every metric).
FS1_T2_COLS: tuple[str, ...] = (
    "fin_log_mcap",
    "fin_value_z",
    "fin_gross_profitability",
    "ev_amendment_ratio_1y",
    "ev_filing_burst_60d",
    "ev_net_share_issuance_yoy",
    "ev_payout_yield",
    "own_major_filing_60d",
    "own_major_stake_chg",
    "hc_employee_growth_yoy",
    "hc_revenue_per_employee",
)
FS1_ADDED: tuple[str, ...] = (*FS1_T1_COLS, *FS1_T2_COLS)
FS1_COLS: tuple[str, ...] = (*FS0_COLS, *FS1_ADDED)

# --- FS2: + macro exposure, regime, interactions (`02` §1.4) -----------------
# usdkrw / kr10y / rawbeta / semibeta are left out: no signal, or a market beta
# in disguise.
FS2_MACRO_BETA_COLS: tuple[str, ...] = (
    "macro_beta_vix",
    "macro_beta_wti",
    "macro_beta_sp500_lag",
    "px_market_beta",
)
# The four binaries from dim_regime_daily. Date-constant, so they carry zero
# information on their own under a cross-sectional label — they earn their place
# as split conditions a tree can put an interaction under.
FS2_REGIME_COLS: tuple[str, ...] = MODEL_REGIME_COLUMNS

# (output, feature, regime). Value = pct_rank(feature) within (trade_date,
# market) x regime, so a session where the regime is 0 contributes 0 and a NULL
# feature stays NULL. These are Phase C's four passing pairs (`02` §3.2).
#
# The two materials px_turnover_shock and flow_foreign_netbuy_to_volume_20d_lag1
# are Grade D on their own (stable *opposite* sign), so they enter only through
# the interaction. Adding them as plain columns is a separate, sign-flipped
# variant (E4c), not part of FS2.
INTERACTIONS: tuple[tuple[str, str, str], ...] = (
    ("ix_reversal_vixhigh", "px_reversal_5d", "rg_vix_high"),
    ("ix_tshock_liqhigh", "px_turnover_shock", "rg_liq_high"),
    ("ix_foreign_liqhigh", "flow_foreign_netbuy_to_volume_20d_lag1", "rg_liq_high"),
    ("ix_mktbeta_mktup", "px_market_beta", "rg_market_up"),
)
FS2_INTERACTION_COLS: tuple[str, ...] = tuple(name for name, _f, _r in INTERACTIONS)
INTERACTION_MATERIAL_COLS: tuple[str, ...] = tuple(
    dict.fromkeys(feature for _n, feature, _r in INTERACTIONS)
)
FS2_ADDED: tuple[str, ...] = (
    *FS2_MACRO_BETA_COLS,
    *FS2_REGIME_COLS,
    *FS2_INTERACTION_COLS,
)
FS2_COLS: tuple[str, ...] = (*FS1_COLS, *FS2_ADDED)

# --- FS3: the parallel stream's 11 (`02` §1.5, confirmed 2026-09-09) --------
# From F-HS-1 config 3ca949e6..., snapshot 2026-09-08. Relation 4 are Grade A;
# the fin_risk 7 are capped at Grade B (revision_ratio ~0.1015, measured, not
# missing). E5 only — and E5 first needs the four 08-23 marts rebuilt with the
# v2 formulas (`02` §1.5 last item).
FS3_RELATION_COLS: tuple[str, ...] = (
    "rel_peer_dispersion_20d",
    "rel_own_minus_peer_20d",
    "rel_peer_mom_20d",
    "rel_peer_bigcap_lag_ret_5d",
)
FS3_FIN_RISK_COLS: tuple[str, ...] = (
    "fin_net_debt_to_mcap",
    "fin_ext_finance_to_assets",
    "fin_interest_coverage",
    "fin_lifecycle_stage",
    "fin_debt_to_assets",
    "fin_profit_turn",
    "fin_lifecycle_transition",
)
FS3_ADDED: tuple[str, ...] = (*FS3_RELATION_COLS, *FS3_FIN_RISK_COLS)
FS3_COLS: tuple[str, ...] = (*FS2_COLS, *FS3_ADDED)

# --- horizon subsets (`02` §1.3) ---------------------------------------------
# What FS1h drops at each h, from `08_quality_summary_gaps.md` §4's placement.
# Only the FS1 additions are ever dropped: FS0 stays whole at every h so the
# baseline is the same 40 columns everywhere.
#
# h=120 is 50 columns, not the "40 + 9 = 49" `02` §1.3 first printed: the six
# dropped columns leave ten of FS1's sixteen. Settled 2026-09-09 — the column
# list is the spec and the total was the typo, because reaching 49 would mean
# choosing a tenth column to drop to fit a number.
FS1H_DROPPED: dict[int, tuple[str, ...]] = {
    5: (
        "fin_log_mcap",
        "fin_value_z",
        "fin_gross_profitability",
        "ev_amendment_ratio_1y",
        "ev_net_share_issuance_yoy",
        "ev_payout_yield",
        "own_major_filing_60d",
        "own_major_stake_chg",
        "hc_employee_growth_yoy",
        "hc_revenue_per_employee",
    ),
    20: (
        "fin_value_z",
        "fin_gross_profitability",
        "ev_net_share_issuance_yoy",
        "ev_payout_yield",
        "own_major_stake_chg",
        "hc_employee_growth_yoy",
        "hc_revenue_per_employee",
    ),
    60: (
        "px_reversal_5d",
        "flow_individual_netbuy_to_volume_5d",
    ),
    120: (
        "px_reversal_5d",
        "px_maxret_20d",
        "flow_individual_netbuy_to_volume_5d",
        "flow_individual_netbuy_to_volume_20d",
        "ev_filing_burst_60d",
        "own_major_filing_60d",
    ),
}

# The horizons each FS3 column was actually tested at (`02` §1.5 table). Used
# only by the ``h`` variants: outside these, the column is dropped, the same
# rule §1.3 applies to FS1h. 10 and 40 are scan horizons the model does not use;
# they are kept here so the table matches the record it came from.
FS3_TESTED_HORIZONS: dict[str, tuple[int, ...]] = {
    "rel_peer_dispersion_20d": (20, 40, 60),
    "rel_own_minus_peer_20d": (5, 10, 20),
    "rel_peer_mom_20d": (20, 40, 60),
    "rel_peer_bigcap_lag_ret_5d": (10, 20),
    "fin_net_debt_to_mcap": (60, 120),
    "fin_ext_finance_to_assets": (60,),
    "fin_interest_coverage": (60,),
    "fin_lifecycle_stage": (60,),
    "fin_debt_to_assets": (60,),
    "fin_profit_turn": (20,),
    "fin_lifecycle_transition": (20, 60),
}

# E5's suffix. `05` §2 E5 compares "the adopted config -> + FS3", and at h5 and
# h60 the adopted config is FS0 — so plain "FS3" (which is FS2 + 11) would drag
# back the 28 columns E2 rejected and confound the 11 the parallel stream is
# actually asking about. ``<base>_FS3`` names "that base, plus FS3's additions".
# It subsumes the two fixed ids: FS2_FS3 == FS3 and FS2h_FS3 == FS3h.
FS3_SUFFIX = "_FS3"
FEATURE_SET_BASES: tuple[str, ...] = ("FS0", "FS1", "FS1h", "FS2", "FS2h")
FEATURE_SET_IDS: tuple[str, ...] = (
    "FS0",
    "FS1",
    "FS1h",
    "FS2",
    "FS2h",
    "FS3",
    "FS3h",
    *(f"{base}{FS3_SUFFIX}" for base in FEATURE_SET_BASES),
)


def feature_columns(feature_set: str, horizon: int) -> tuple[str, ...]:
    """Columns for one ``(feature_set, horizon)``.

    The ``h`` suffix means "restricted to the horizons this column was tested
    at": ``FS1h`` drops per §1.3, ``FS3h`` additionally drops FS3 columns
    outside §1.5's tested horizons. ``FS2h``/``FS3h`` exist because E2-c stacks
    FS2 on whichever of FS1/FS1h won E2-a/b (`05` §2) — the stack is mechanical,
    not a new hypothesis.
    """
    if feature_set not in FEATURE_SET_IDS:
        raise ValueError(f"unknown feature set {feature_set!r}; valid: {FEATURE_SET_IDS}")
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon}; valid: {HORIZONS}")

    if feature_set.endswith(FS3_SUFFIX):
        base_id = feature_set[: -len(FS3_SUFFIX)]
        base_cols = feature_columns(base_id, horizon)
        # The horizon rule follows the *base*, which is `02` §1.5's wording:
        # subset FS3 when the adopted config is itself horizon-scoped, keep all
        # eleven when it is not.
        additions = FS3_ADDED
        if base_id.endswith("h"):
            additions = tuple(c for c in FS3_ADDED if horizon in FS3_TESTED_HORIZONS[c])
        return (*base_cols, *additions)

    if feature_set == "FS0":
        return FS0_COLS
    horizon_scoped = feature_set.endswith("h")
    dropped = set(FS1H_DROPPED[horizon]) if horizon_scoped else set()
    base = tuple(c for c in FS1_COLS if c not in dropped)
    if feature_set in ("FS1", "FS1h"):
        return base
    base = (*base, *FS2_ADDED)
    if feature_set in ("FS2", "FS2h"):
        return base
    fs3 = FS3_ADDED
    if horizon_scoped:
        fs3 = tuple(c for c in FS3_ADDED if horizon in FS3_TESTED_HORIZONS[c])
    return (*base, *fs3)


FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "FS0": FS0_COLS,
    "FS1": FS1_COLS,
    "FS2": FS2_COLS,
    "FS3": FS3_COLS,
}
HORIZON_SUBSETS: dict[int, tuple[str, ...]] = {h: feature_columns("FS1h", h) for h in HORIZONS}

# --- where the columns come from (`02` §5) -----------------------------------
# _01's _GROUP_VIEW, extended. "cf" (feat_common) is mapped but no feature set
# uses it: macro series enter as exposure betas and regimes, not as raw levels
# broadcast to every name (`07` §1.2 — that is a question for an absolute-return
# label).
GROUP_VIEW: dict[str, str] = {
    "px": "feat_price",
    "flow": "feat_flow",
    "fin": "feat_fin_pit",
    "cf": "feat_common",
    "ev": "feat_event",
    "fin_scan": "feat_fin_scan_daily",
    "mcap": "feat_market_cap",
    "evs": "feat_event_scan_daily",
    "filing": "feat_filing_activity",
    "pex": "feat_periodic_extras",
    "mx": "feat_macro_exposure",
    "rg": REGIME_TABLE,
    "rel": "feat_relation_stat",
    "fin_risk": "feat_fin_risk",
}


def _mart_map(columns: tuple[str, ...], group: str) -> dict[str, str]:
    return dict.fromkeys(columns, GROUP_VIEW[group])


COLUMN_MART: dict[str, str] = {
    **_mart_map(FS0_PX_COLS, "px"),
    **_mart_map(FS0_FLOW_COLS, "flow"),
    **_mart_map(FS0_FIN_COLS, "fin"),
    **_mart_map(("px_reversal_5d", "px_maxret_20d", "px_idio_vol_60d"), "px"),
    **_mart_map(
        (
            "flow_individual_netbuy_to_volume_5d",
            "flow_individual_netbuy_to_volume_20d",
        ),
        "flow",
    ),
    **_mart_map(("fin_log_mcap", "fin_value_z", "fin_gross_profitability"), "fin_scan"),
    # the three filing-count families live in feat_filing_activity, not in
    # feat_event_scan_daily where their ev_/own_ prefixes suggest.
    **_mart_map(
        ("ev_amendment_ratio_1y", "ev_filing_burst_60d", "own_major_filing_60d"),
        "filing",
    ),
    **_mart_map(("ev_net_share_issuance_yoy", "ev_payout_yield"), "evs"),
    **_mart_map(
        ("own_major_stake_chg", "hc_employee_growth_yoy", "hc_revenue_per_employee"),
        "pex",
    ),
    **_mart_map(FS2_MACRO_BETA_COLS, "mx"),
    **_mart_map(FS2_REGIME_COLS, "rg"),
    **_mart_map(("px_turnover_shock",), "px"),
    **_mart_map(("flow_foreign_netbuy_to_volume_20d_lag1",), "flow"),
    **_mart_map(FS3_RELATION_COLS, "rel"),
    **_mart_map(FS3_FIN_RISK_COLS, "fin_risk"),
}

# --- flow timing (`02` §2, D-6) ----------------------------------------------
# The validated canon is lag1, and the builder is the only place in this model
# that moves a mart value in time.
#
# FS0's 15 flow columns have no _lag1 twin in feat_flow, so the builder has to
# LAG them itself — over *valid sessions*, so a halt is stepped over rather than
# counted as a day.
FLOW_BUILDER_LAG_COLUMNS: tuple[str, ...] = FS0_FLOW_COLS
# These do have a twin: read the _lag1 column and expose it under the plain
# name, so a feature set never has to spell the variant.
FLOW_MART_LAG1_SOURCE: dict[str, str] = {
    "flow_individual_netbuy_to_volume_5d": "flow_individual_netbuy_to_volume_5d_lag1",
    "flow_individual_netbuy_to_volume_20d": "flow_individual_netbuy_to_volume_20d_lag1",
}
FLOW_VARIANTS: tuple[str, ...] = ("lag1", "native_t")

# --- expected signs (E4b monotonic constraints) ------------------------------
# The label is "went up", so these are the IC signs. FS1's are `02` §1.2's, FS3's
# are `02` §1.5's (fixed by F-HS-1 after its first run, including the six
# registered in both directions — none flipped). FS0 has none: those 40 were
# never screened, so there is no preregistered direction to constrain, and FS2's
# betas/regimes/interactions are conditional effects whose sign the model reads.
# Marked "(관측)" in the docs where the direction came from observation rather
# than a prior — that distinction lives in the docs, not in this table.
EXPECTED_SIGN: dict[str, int] = {
    "px_reversal_5d": +1,
    "px_maxret_20d": -1,
    "px_idio_vol_60d": -1,
    "flow_individual_netbuy_to_volume_5d": +1,
    "flow_individual_netbuy_to_volume_20d": +1,
    "fin_log_mcap": -1,
    "fin_value_z": +1,
    "fin_gross_profitability": +1,
    "ev_amendment_ratio_1y": -1,
    "ev_filing_burst_60d": -1,
    "ev_net_share_issuance_yoy": -1,
    "ev_payout_yield": +1,
    "own_major_filing_60d": -1,
    "own_major_stake_chg": +1,
    "hc_employee_growth_yoy": +1,
    "hc_revenue_per_employee": +1,
    "rel_peer_dispersion_20d": -1,
    "rel_own_minus_peer_20d": -1,
    "rel_peer_mom_20d": -1,
    "rel_peer_bigcap_lag_ret_5d": +1,
    "fin_net_debt_to_mcap": +1,
    "fin_ext_finance_to_assets": -1,
    "fin_interest_coverage": +1,
    "fin_lifecycle_stage": +1,
    "fin_debt_to_assets": -1,
    "fin_profit_turn": +1,
    "fin_lifecycle_transition": -1,
}


def monotonic_constraints(columns: tuple[str, ...]) -> tuple[int, ...]:
    """Per-column constraint for E4b: the expected sign, 0 where none is preregistered."""
    return tuple(EXPECTED_SIGN.get(c, 0) for c in columns)
