"""C stage 2 — the next KR model's feature versions, fixed as column lists in code.

Plan: ``my/milestones/common/20261012_feature_explainer/17_c2_feature_set.md`` §2 (the 47-column
serving map), §3 (the two sets), §4 (the missingness rule).

Same reason as ``models/_02_updown_prob/features.py``: a preregistered column list that can be
edited between runs is not preregistered, so the lists live in code and the tests pin them.

The rule (17 §1, ``13`` §4) is "the same economic set as the 47 serving columns, definitions
fixed, no new feature". Nothing here was chosen by looking at development-period scores:

* ``SERVING_MAP`` — each serving column (FS1h at h=20 minus ``serving.kr_model.EXCLUDED_FEATURES``)
  to its replacement, or ``None`` when it is dropped (``DROPPED`` says why).
* ``MAIN_COLS`` (43) — the replacements. ``NOFLOW_COLS`` (30) — the same without the 13 flow
  columns, the main model if the delisted-flow backfill (15 §10.6 F1/F2) slips (15 §10.4 E3).
* ``COLUMN_MART`` — which mart each column comes from, so a builder registers exactly the
  views it needs.
* ``HYPOTHESIS_COLS`` — the columns the "avoid the tail" hypothesis (``14`` §6.3) is written on.
  All of them are inside ``MAIN_COLS``; the hypothesis adds no feature.

Missingness rule (17 §4, applied by C stage 3, not here): a column whose missing rate differs by
more than ``MISSING_GAP_PP`` between later-delisted tickers and the rest gets no ``_isna`` and no
0.5 fill.
"""

from __future__ import annotations

# --- the 43 columns of the main model (17 §3) --------------------------------

PRICE_COLS: tuple[str, ...] = (
    "px_ret_1d_v2",
    "px_ret_5d_v2",
    "px_ret_20d_v2",
    "px_ret_60d_v2",
    "px_ret_60_20_v2",
    "px_vol_20d_v2",
    "px_vol_60d_v2",
    "px_high_low_range_20d_v2",
    "px_turnover",
    "px_turnover_ma20",
    "px_amihud_20d_v2",
    "px_gap_vs_ma20_v2",
    "px_dist_52w_high_v2",
    "px_halt_ratio_20d",
    "px_reversal_5d",
    "px_maxret_20d",
    "px_idio_vol_60d",
)
FLOW_COLS: tuple[str, ...] = (
    "flow_foreign_netbuy_to_float_5d_v2",
    "flow_foreign_netbuy_to_float_20d_v2",
    "flow_inst_netbuy_to_float_5d_v2",
    "flow_inst_netbuy_to_float_20d_v2",
    "flow_indiv_netbuy_to_float_5d_v2",
    "flow_indiv_netbuy_to_float_20d_v2",
    "flow_foreign_netbuy_z_20d_v2",
    "flow_inst_netbuy_z_20d_v2",
    "flow_short_avg_price_v2",
    "flow_short_selling_volume_v2",
    "flow_short_selling_value_v2",
    "flow_individual_netbuy_to_volume_5d",
    "flow_individual_netbuy_to_volume_20d",
)
FIN_COLS: tuple[str, ...] = (
    "fin_roa_v2",
    "fin_roe_v2",
    "fin_operating_margin_v2",
    "fin_debt_to_equity_v2",
    "fin_equity_ratio_v2",
    "fin_ocf_to_assets_v2",
    "fin_cash_ratio_v2",
    "fin_asset_turnover_v2",
    "fin_capital_impaired_v2",
    "mcap_krx_log",
)
FILING_COLS: tuple[str, ...] = (
    "ev_amendment_ratio_1y",
    "ev_filing_burst_60d",
    "own_major_filing_60d",
)

MAIN_COLS: tuple[str, ...] = (*PRICE_COLS, *FLOW_COLS, *FIN_COLS, *FILING_COLS)
#: The main set without flow. Becomes the main model if delisted-flow backfill is late (15 §10.4).
NOFLOW_COLS: tuple[str, ...] = (*PRICE_COLS, *FIN_COLS, *FILING_COLS)

# --- where each column comes from (17 §3 "입력 마트") -------------------------

#: Mart -> the columns of ``MAIN_COLS`` it produces. Explicit so a typo shows up in review.
MART_COLUMNS: dict[str, tuple[str, ...]] = {
    "feat_price_v2": (
        "px_ret_1d_v2",
        "px_ret_5d_v2",
        "px_ret_20d_v2",
        "px_ret_60d_v2",
        "px_ret_60_20_v2",
        "px_vol_20d_v2",
        "px_vol_60d_v2",
        "px_high_low_range_20d_v2",
        "px_amihud_20d_v2",
        "px_gap_vs_ma20_v2",
        "px_dist_52w_high_v2",
    ),
    # unchanged definitions: no v2 column exists or is needed (17 §2)
    "feat_price": (
        "px_turnover",
        "px_turnover_ma20",
        "px_halt_ratio_20d",
        "px_reversal_5d",
        "px_maxret_20d",
        "px_idio_vol_60d",
    ),
    "feat_flow_v2": (
        "flow_foreign_netbuy_to_float_5d_v2",
        "flow_foreign_netbuy_to_float_20d_v2",
        "flow_inst_netbuy_to_float_5d_v2",
        "flow_inst_netbuy_to_float_20d_v2",
        "flow_indiv_netbuy_to_float_5d_v2",
        "flow_indiv_netbuy_to_float_20d_v2",
        "flow_foreign_netbuy_z_20d_v2",
        "flow_inst_netbuy_z_20d_v2",
        "flow_short_avg_price_v2",
        "flow_short_selling_volume_v2",
        "flow_short_selling_value_v2",
    ),
    "feat_flow": (
        "flow_individual_netbuy_to_volume_5d",
        "flow_individual_netbuy_to_volume_20d",
    ),
    "feat_fin_v2": (
        "fin_roa_v2",
        "fin_roe_v2",
        "fin_operating_margin_v2",
        "fin_debt_to_equity_v2",
        "fin_equity_ratio_v2",
        "fin_ocf_to_assets_v2",
        "fin_cash_ratio_v2",
        "fin_asset_turnover_v2",
        "fin_capital_impaired_v2",
    ),
    "feat_market_cap": ("mcap_krx_log",),
    "feat_filing_activity": (
        "ev_amendment_ratio_1y",
        "ev_filing_burst_60d",
        "own_major_filing_60d",
    ),
}
COLUMN_MART: dict[str, str] = {col: mart for mart, cols in MART_COLUMNS.items() for col in cols}

# --- the 47 serving columns -> next model (17 §2) ----------------------------
# Key = a serving column (models/_02_updown_prob FS1h at h=20, minus the two short-balance
# columns serving blocks). Value = its replacement column(s), or None when dropped.

SERVING_MAP: dict[str, tuple[str, ...] | None] = {
    # price
    "px_ret_1d": ("px_ret_1d_v2",),
    "px_ret_5d": ("px_ret_5d_v2",),
    "px_ret_20d": ("px_ret_20d_v2",),
    "px_ret_60d": ("px_ret_60d_v2",),
    "px_mom_20_60": ("px_ret_60_20_v2",),
    "px_vol_20d": ("px_vol_20d_v2",),
    "px_vol_60d": ("px_vol_60d_v2",),
    "px_high_low_range_20d": ("px_high_low_range_20d_v2",),
    "px_turnover": ("px_turnover",),
    "px_turnover_ma20": ("px_turnover_ma20",),
    "px_amihud_20d": ("px_amihud_20d_v2",),
    "px_gap_vs_ma20": ("px_gap_vs_ma20_v2",),
    "px_dist_52w_high": ("px_dist_52w_high_v2",),
    "px_is_halted": None,
    "px_halt_ratio_20d": ("px_halt_ratio_20d",),
    "px_reversal_5d": ("px_reversal_5d",),
    "px_maxret_20d": ("px_maxret_20d",),
    "px_idio_vol_60d": ("px_idio_vol_60d",),
    # flow
    "flow_foreign_netbuy_sum_5d": ("flow_foreign_netbuy_to_float_5d_v2",),
    "flow_foreign_netbuy_sum_20d": ("flow_foreign_netbuy_to_float_20d_v2",),
    "flow_inst_netbuy_sum_5d": ("flow_inst_netbuy_to_float_5d_v2",),
    "flow_inst_netbuy_sum_20d": ("flow_inst_netbuy_to_float_20d_v2",),
    "flow_indiv_netbuy_sum_5d": ("flow_indiv_netbuy_to_float_5d_v2",),
    "flow_indiv_netbuy_sum_20d": ("flow_indiv_netbuy_to_float_20d_v2",),
    "flow_foreign_holding_chg_5d": None,
    "flow_foreign_holding_chg_20d": None,
    "flow_foreign_netbuy_z_20d": ("flow_foreign_netbuy_z_20d_v2",),
    "flow_inst_netbuy_z_20d": ("flow_inst_netbuy_z_20d_v2",),
    "flow_short_avg_price": ("flow_short_avg_price_v2",),
    "flow_short_selling_volume": ("flow_short_selling_volume_v2",),
    "flow_short_selling_value": ("flow_short_selling_value_v2",),
    "flow_individual_netbuy_to_volume_5d": ("flow_individual_netbuy_to_volume_5d",),
    "flow_individual_netbuy_to_volume_20d": ("flow_individual_netbuy_to_volume_20d",),
    # financials and size
    "fin_roa": ("fin_roa_v2",),
    "fin_roe": ("fin_roe_v2",),
    "fin_operating_margin": ("fin_operating_margin_v2",),
    "fin_debt_to_equity": ("fin_debt_to_equity_v2",),
    "fin_equity_ratio": ("fin_equity_ratio_v2",),
    "fin_ocf_to_assets": ("fin_ocf_to_assets_v2",),
    "fin_cash_ratio": ("fin_cash_ratio_v2",),
    "fin_asset_turnover": ("fin_asset_turnover_v2",),
    "fin_is_negative_equity": ("fin_capital_impaired_v2",),
    "fin_has_fs": None,
    "fin_log_mcap": ("mcap_krx_log",),
    # filings
    "ev_amendment_ratio_1y": ("ev_amendment_ratio_1y",),
    "ev_filing_burst_60d": ("ev_filing_burst_60d",),
    "own_major_filing_60d": ("own_major_filing_60d",),
}

#: The four serving columns that are not carried over, and why (17 §2).
DROPPED: dict[str, str] = {
    "px_is_halted": "universe drops halted rows, so it is always 0 in the training sample",
    "flow_foreign_holding_chg_5d": "KIS gives no history, delisted tickers cannot be filled",
    "flow_foreign_holding_chg_20d": "KIS gives no history, delisted tickers cannot be filled",
    "fin_has_fs": "marks whether financials exist, i.e. a missingness flag itself",
}

# --- hypothesis columns (14 §6.3) --------------------------------------------
# "Avoid the tail" is defined on these. Both sets are inside MAIN_COLS.

HYPOTHESIS_COLS: dict[str, tuple[str, ...]] = {
    "tail_vol": ("px_idio_vol_60d", "px_vol_20d_v2"),
    "tail_loss": ("fin_roa_v2", "fin_capital_impaired_v2"),
}

#: Fixed benchmark G (16 §7): (short key, ``krx_index_daily.idx_nm``).
FIXED_BENCHMARK: tuple[str, str] = ("k200", "코스피 200")

#: Missingness rule threshold in percentage points (17 §4, set before the result was seen).
MISSING_GAP_PP = 10.0
