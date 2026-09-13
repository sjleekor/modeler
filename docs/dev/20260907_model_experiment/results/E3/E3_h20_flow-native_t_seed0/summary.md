# E3_h20_flow-native_t_seed0

- stage: E3 / variant: flow-native_t
- horizon: 20, label: y_up, model: hgb_clf, seed: 0
- feature set: FS1h (49 columns), flow: native_t, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66855 | 0.00516 |
| topk_cost_adjusted_return | +0.01013 | 0.00988 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01261 |
| brier | 0.23787 |
| auc_daily_mean | 0.54711 |
| rank_ic_mean | 0.13811 |
| precision_at_k | 0.46378 |
| lift_at_k | 1.16549 |
| topk_turnover | 0.70150 |
| base_rate | 0.39741 |

ECE ceiling 0.03: under

## Folds

```
shape: (5, 52)
┌─────────┬──────────┬─────────┬─────────┬───┬─────────────┬─────────────┬────────────┬────────────┐
│ fold_id ┆ pred_col ┆ skipped ┆ n_train ┆ … ┆ decile_grid ┆ decile_turn ┆ decile_cos ┆ decile_cos │
│ ---     ┆ ---      ┆ ---     ┆ ---     ┆   ┆ _top_decile ┆ over        ┆ t_bps_roun ┆ t_adjusted │
│ i64     ┆ str      ┆ null    ┆ i64     ┆   ┆ _spread     ┆ ---         ┆ dtrip      ┆ _spread    │
│         ┆          ┆         ┆         ┆   ┆ ---         ┆ f64         ┆ ---        ┆ ---        │
│         ┆          ┆         ┆         ┆   ┆ f64         ┆             ┆ f64        ┆ f64        │
╞═════════╪══════════╪═════════╪═════════╪═══╪═════════════╪═════════════╪════════════╪════════════╡
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ 0.003699    ┆ 0.631129    ┆ 60.0       ┆ -0.000088  │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ 0.019407    ┆ 0.561998    ┆ 60.0       ┆ 0.016035   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.016528    ┆ 0.547272    ┆ 60.0       ┆ 0.013244   │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.009937    ┆ 0.591267    ┆ 60.0       ┆ 0.006389   │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.021378    ┆ 0.688838    ┆ 60.0       ┆ 0.017245   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1h_h20_native_t_rank/predictions_valid__E3_h20_flow-native_t_seed0.parquet`
- panel rows: 5223425
- elapsed: 5003.7s
