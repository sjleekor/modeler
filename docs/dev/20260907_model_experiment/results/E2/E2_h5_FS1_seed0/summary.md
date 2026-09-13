# E2_h5_FS1_seed0

- stage: E2 / variant: FS1
- horizon: 5, label: y_up, model: hgb_clf, seed: 0
- feature set: FS1 (56 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67821 | 0.00242 |
| topk_cost_adjusted_return | +0.00068 | 0.00215 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01319 |
| brier | 0.24258 |
| auc_daily_mean | 0.53750 |
| rank_ic_mean | 0.09879 |
| precision_at_k | 0.48549 |
| lift_at_k | 1.15943 |
| topk_turnover | 0.74753 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ 0.00206     ┆ 0.664366    ┆ 60.0       ┆ -0.001927  │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ 0.004915    ┆ 0.648049    ┆ 60.0       ┆ 0.001026   │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.004814    ┆ 0.650811    ┆ 60.0       ┆ 0.000909   │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.004835    ┆ 0.671918    ┆ 60.0       ┆ 0.000804   │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ 0.003202    ┆ 0.660471    ┆ 60.0       ┆ -0.000761  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1_h5_lag1_rank/predictions_valid__E2_h5_FS1_seed0.parquet`
- panel rows: 5223425
- elapsed: 6219.9s
