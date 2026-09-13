# E0_h5_MA-rank_seed0

- stage: E0 / variant: MA-rank
- horizon: 5, label: y_rank, model: hgb_reg, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_REG_GRID (4 points), best: {'max_iter': 200, 'learning_rate': 0.01}
- calibration: isotonic -> y_up

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67937 | 0.00326 |
| topk_cost_adjusted_return | -0.00076 | 0.00141 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01526 |
| brier | 0.24315 |
| auc_daily_mean | 0.52685 |
| rank_ic_mean | 0.11149 |
| precision_at_k | 0.47110 |
| lift_at_k | 1.12530 |
| topk_turnover | 0.79208 |
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
│ 1       ┆ p_cal    ┆ null    ┆ 2351735 ┆ … ┆ 0.002762    ┆ 0.758074    ┆ 60.0       ┆ -0.001787  │
│ 2       ┆ p_cal    ┆ null    ┆ 2900087 ┆ … ┆ 0.004701    ┆ 0.764388    ┆ 60.0       ┆ 0.000115   │
│ 3       ┆ p_cal    ┆ null    ┆ 3478717 ┆ … ┆ 0.003016    ┆ 0.711153    ┆ 60.0       ┆ -0.001251  │
│ 4       ┆ p_cal    ┆ null    ┆ 4056713 ┆ … ┆ 0.004722    ┆ 0.668859    ┆ 60.0       ┆ 0.000709   │
│ 5       ┆ p_cal    ┆ null    ┆ 4636599 ┆ … ┆ 0.001958    ┆ 0.601972    ┆ 60.0       ┆ -0.001654  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h5_lag1_rank/predictions_valid__E0_h5_MA-rank_seed0.parquet`
- panel rows: 5223425
- elapsed: 944.0s
