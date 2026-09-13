# E2_h20_FS1_seed0

- stage: E2 / variant: FS1
- horizon: 20, label: y_up, model: hgb_clf, seed: 0
- feature set: FS1 (56 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66874 | 0.00467 |
| topk_cost_adjusted_return | +0.01128 | 0.01111 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01383 |
| brier | 0.23796 |
| auc_daily_mean | 0.54611 |
| rank_ic_mean | 0.13638 |
| precision_at_k | 0.46518 |
| lift_at_k | 1.16853 |
| topk_turnover | 0.70267 |
| base_rate | 0.39741 |

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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ -0.000049   ┆ 0.578548    ┆ 60.0       ┆ -0.00352   │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ 0.018722    ┆ 0.593533    ┆ 60.0       ┆ 0.015161   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.018804    ┆ 0.600839    ┆ 60.0       ┆ 0.015199   │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.011487    ┆ 0.599252    ┆ 60.0       ┆ 0.007891   │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.013094    ┆ 0.647512    ┆ 60.0       ┆ 0.009209   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1_h20_lag1_rank/predictions_valid__E2_h20_FS1_seed0.parquet`
- panel rows: 5223425
- elapsed: 5583.1s
