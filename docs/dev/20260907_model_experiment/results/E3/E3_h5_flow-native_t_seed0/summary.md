# E3_h5_flow-native_t_seed0

- stage: E3 / variant: flow-native_t
- horizon: 5, label: y_up, model: hgb_clf, seed: 0
- feature set: FS0 (40 columns), flow: native_t, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67798 | 0.00238 |
| topk_cost_adjusted_return | +0.00148 | 0.00085 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01326 |
| brier | 0.24248 |
| auc_daily_mean | 0.53940 |
| rank_ic_mean | 0.09794 |
| precision_at_k | 0.49122 |
| lift_at_k | 1.17319 |
| topk_turnover | 0.83533 |
| base_rate | 0.41854 |

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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ 0.003069    ┆ 0.753812    ┆ 60.0       ┆ -0.001454  │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ 0.005932    ┆ 0.760838    ┆ 60.0       ┆ 0.001367   │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.005043    ┆ 0.759014    ┆ 60.0       ┆ 0.000489   │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.005608    ┆ 0.755947    ┆ 60.0       ┆ 0.001073   │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ 0.004846    ┆ 0.755657    ┆ 60.0       ┆ 0.000312   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h5_native_t_rank/predictions_valid__E3_h5_flow-native_t_seed0.parquet`
- panel rows: 5223425
- elapsed: 4943.3s
