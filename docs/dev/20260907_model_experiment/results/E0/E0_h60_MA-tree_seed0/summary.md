# E0_h60_MA-tree_seed0

- stage: E0 / variant: MA-tree
- horizon: 60, label: y_rank, model: hgb_reg, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: tree
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_REG_GRID (4 points), best: {'max_iter': 200, 'learning_rate': 0.01}
- calibration: isotonic -> y_up

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66272 | 0.00839 |
| topk_cost_adjusted_return | +0.02477 | 0.01780 |

## Secondary

| metric | value |
|---|---|
| ece | 0.02492 |
| brier | 0.23493 |
| auc_daily_mean | 0.55380 |
| rank_ic_mean | 0.17663 |
| precision_at_k | 0.44985 |
| lift_at_k | 1.17516 |
| topk_turnover | 0.77750 |
| base_rate | 0.38268 |

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
│ 1       ┆ p_cal    ┆ null    ┆ 2136878 ┆ … ┆ 0.048411    ┆ 0.677533    ┆ 60.0       ┆ 0.044346   │
│ 2       ┆ p_cal    ┆ null    ┆ 2661837 ┆ … ┆ 0.055155    ┆ 0.738784    ┆ 60.0       ┆ 0.050722   │
│ 3       ┆ p_cal    ┆ null    ┆ 3233289 ┆ … ┆ 0.020741    ┆ 0.794213    ┆ 60.0       ┆ 0.015976   │
│ 4       ┆ p_cal    ┆ null    ┆ 3809273 ┆ … ┆ 0.018057    ┆ 0.685085    ┆ 60.0       ┆ 0.013946   │
│ 5       ┆ p_cal    ┆ null    ┆ 4391913 ┆ … ┆ 0.005202    ┆ 0.73631     ┆ 60.0       ┆ 0.000784   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h60_lag1_tree/predictions_valid__E0_h60_MA-tree_seed0.parquet`
- panel rows: 5223425
- elapsed: 715.4s
