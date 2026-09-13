# E5_h5_FS3_seed0

- stage: E5 / variant: FS3
- horizon: 5, label: y_up, model: hgb_clf, seed: 0
- feature set: FS0_FS3 (51 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 1.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67835 | 0.00275 |
| topk_cost_adjusted_return | +0.00042 | 0.00242 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01423 |
| brier | 0.24265 |
| auc_daily_mean | 0.53575 |
| rank_ic_mean | 0.09522 |
| precision_at_k | 0.48438 |
| lift_at_k | 1.15665 |
| topk_turnover | 0.76463 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ 0.000584    ┆ 0.709978    ┆ 60.0       ┆ -0.003676  │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ 0.005757    ┆ 0.695828    ┆ 60.0       ┆ 0.001582   │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.004558    ┆ 0.679911    ┆ 60.0       ┆ 0.000479   │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.005162    ┆ 0.640066    ┆ 60.0       ┆ 0.001322   │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ 0.00373     ┆ 0.674112    ┆ 60.0       ┆ -0.000315  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_FS3_h5_lag1_rank/predictions_valid__E5_h5_FS3_seed0.parquet`
- panel rows: 5223425
- elapsed: 5570.7s
