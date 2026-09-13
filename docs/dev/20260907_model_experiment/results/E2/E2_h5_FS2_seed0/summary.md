# E2_h5_FS2_seed0

- stage: E2 / variant: FS2
- horizon: 5, label: y_up, model: hgb_clf, seed: 0
- feature set: FS2 (68 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 1.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67811 | 0.00317 |
| topk_cost_adjusted_return | +0.00068 | 0.00162 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01416 |
| brier | 0.24254 |
| auc_daily_mean | 0.53663 |
| rank_ic_mean | 0.09908 |
| precision_at_k | 0.48450 |
| lift_at_k | 1.15754 |
| topk_turnover | 0.71871 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ 0.002931    ┆ 0.682156    ┆ 60.0       ┆ -0.001162  │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ 0.005836    ┆ 0.59238     ┆ 60.0       ┆ 0.002282   │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.004045    ┆ 0.632809    ┆ 60.0       ┆ 0.000248   │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.005198    ┆ 0.614904    ┆ 60.0       ┆ 0.001508   │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ 0.00315     ┆ 0.667859    ┆ 60.0       ┆ -0.000857  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS2_h5_lag1_rank/predictions_valid__E2_h5_FS2_seed0.parquet`
- panel rows: 5223425
- elapsed: 7216.6s
