# E1_h5_y_up-hgb_clf_seed0

- stage: E1 / variant: y_up-hgb_clf
- horizon: 5, label: y_up, model: hgb_clf, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID_E1 (4 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67829 | 0.00253 |
| topk_cost_adjusted_return | +0.00016 | 0.00246 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01358 |
| brier | 0.24263 |
| auc_daily_mean | 0.53583 |
| rank_ic_mean | 0.09543 |
| precision_at_k | 0.47949 |
| lift_at_k | 1.14498 |
| topk_turnover | 0.76824 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ -0.000072   ┆ 0.676473    ┆ 60.0       ┆ -0.004131  │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ 0.003755    ┆ 0.666988    ┆ 60.0       ┆ -0.000247  │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.004252    ┆ 0.698274    ┆ 60.0       ┆ 0.000063   │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.006184    ┆ 0.700073    ┆ 60.0       ┆ 0.001984   │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ 0.003213    ┆ 0.696427    ┆ 60.0       ┆ -0.000965  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h5_lag1_rank/predictions_valid__E1_h5_y_up-hgb_clf_seed0.parquet`
- panel rows: 5223425
- elapsed: 1786.2s
