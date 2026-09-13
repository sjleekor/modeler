# E1_h60_y_top-hgb_clf_seed0

- stage: E1 / variant: y_top-hgb_clf
- horizon: 60, label: y_top, model: hgb_clf, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID_E1 (4 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.50190 | 0.00591 |
| topk_cost_adjusted_return | +0.03184 | 0.02231 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01557 |
| brier | 0.16053 |
| auc_daily_mean | 0.54313 |
| rank_ic_mean | 0.00273 |
| precision_at_k | 0.24312 |
| lift_at_k | 1.21036 |
| topk_turnover | 0.76950 |
| base_rate | 0.20069 |

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
│ 1       ┆ p_raw    ┆ null    ┆ 2136878 ┆ … ┆ 0.013453    ┆ 0.755334    ┆ 60.0       ┆ 0.008921   │
│ 2       ┆ p_raw    ┆ null    ┆ 2661837 ┆ … ┆ 0.005612    ┆ 0.609739    ┆ 60.0       ┆ 0.001954   │
│ 3       ┆ p_raw    ┆ null    ┆ 3233289 ┆ … ┆ 0.046718    ┆ 0.646824    ┆ 60.0       ┆ 0.042837   │
│ 4       ┆ p_raw    ┆ null    ┆ 3809273 ┆ … ┆ 0.022632    ┆ 0.689021    ┆ 60.0       ┆ 0.018498   │
│ 5       ┆ p_raw    ┆ null    ┆ 4391913 ┆ … ┆ 0.037784    ┆ 0.696138    ┆ 60.0       ┆ 0.033607   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h60_lag1_rank/predictions_valid__E1_h60_y_top-hgb_clf_seed0.parquet`
- panel rows: 5223425
- elapsed: 1444.0s
