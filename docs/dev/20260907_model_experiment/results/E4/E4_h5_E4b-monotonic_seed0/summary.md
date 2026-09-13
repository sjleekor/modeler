# E4_h5_E4b-monotonic_seed0

- stage: E4 / variant: E4b-monotonic
- horizon: 5, label: y_up, model: hgb_clf, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 1.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67827 | 0.00240 |
| topk_cost_adjusted_return | +0.00021 | 0.00250 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01297 |
| brier | 0.24262 |
| auc_daily_mean | 0.53638 |
| rank_ic_mean | 0.09486 |
| precision_at_k | 0.48037 |
| lift_at_k | 1.14709 |
| topk_turnover | 0.76596 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ -0.000206   ┆ 0.657439    ┆ 60.0       ┆ -0.00415   │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ 0.0038      ┆ 0.684393    ┆ 60.0       ┆ -0.000307  │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.005018    ┆ 0.691777    ┆ 60.0       ┆ 0.000868   │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.005959    ┆ 0.703027    ┆ 60.0       ┆ 0.001741   │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ 0.003459    ┆ 0.697108    ┆ 60.0       ┆ -0.000723  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h5_lag1_rank/predictions_valid__E4_h5_E4b-monotonic_seed0.parquet`
- panel rows: 5223425
- elapsed: 4887.4s
