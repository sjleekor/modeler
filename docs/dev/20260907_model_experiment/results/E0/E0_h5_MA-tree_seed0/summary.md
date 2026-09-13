# E0_h5_MA-tree_seed0

- stage: E0 / variant: MA-tree
- horizon: 5, label: y_rank, model: hgb_reg, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: tree
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_REG_GRID (4 points), best: {'max_iter': 100, 'learning_rate': 0.1}
- calibration: isotonic -> y_up

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67831 | 0.00393 |
| topk_cost_adjusted_return | +0.00086 | 0.00128 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01183 |
| brier | 0.24263 |
| auc_daily_mean | 0.53239 |
| rank_ic_mean | 0.11422 |
| precision_at_k | 0.47659 |
| lift_at_k | 1.13899 |
| topk_turnover | 0.72231 |
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
│ 1       ┆ p_cal    ┆ null    ┆ 2351735 ┆ … ┆ 0.004346    ┆ 0.652792    ┆ 60.0       ┆ 0.000429   │
│ 2       ┆ p_cal    ┆ null    ┆ 2900087 ┆ … ┆ 0.004865    ┆ 0.691193    ┆ 60.0       ┆ 0.000718   │
│ 3       ┆ p_cal    ┆ null    ┆ 3478717 ┆ … ┆ 0.00332     ┆ 0.706434    ┆ 60.0       ┆ -0.000919  │
│ 4       ┆ p_cal    ┆ null    ┆ 4056713 ┆ … ┆ 0.004505    ┆ 0.623742    ┆ 60.0       ┆ 0.000763   │
│ 5       ┆ p_cal    ┆ null    ┆ 4636599 ┆ … ┆ 0.003083    ┆ 0.578031    ┆ 60.0       ┆ -0.000385  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h5_lag1_tree/predictions_valid__E0_h5_MA-tree_seed0.parquet`
- panel rows: 5223425
- elapsed: 737.8s
