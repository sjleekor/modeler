# E5_h5_FS3_seed1

- stage: E5 / variant: FS3
- horizon: 5, label: y_up, model: hgb_clf, seed: 1
- feature set: FS0_FS3 (51 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 1.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67835 | 0.00263 |
| topk_cost_adjusted_return | +0.00064 | 0.00270 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01428 |
| brier | 0.24265 |
| auc_daily_mean | 0.53580 |
| rank_ic_mean | 0.09560 |
| precision_at_k | 0.48519 |
| lift_at_k | 1.15857 |
| topk_turnover | 0.76898 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ 0.000297    ┆ 0.693733    ┆ 60.0       ┆ -0.003865  │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ 0.005725    ┆ 0.70351     ┆ 60.0       ┆ 0.001504   │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.004498    ┆ 0.683023    ┆ 60.0       ┆ 0.0004     │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.005224    ┆ 0.641217    ┆ 60.0       ┆ 0.001376   │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ 0.003916    ┆ 0.673259    ┆ 60.0       ┆ -0.000124  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_FS3_h5_lag1_rank/predictions_valid__E5_h5_FS3_seed1.parquet`
- panel rows: 5223425
- elapsed: 5673.7s
