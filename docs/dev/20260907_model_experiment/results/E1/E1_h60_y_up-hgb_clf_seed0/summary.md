# E1_h60_y_up-hgb_clf_seed0

- stage: E1 / variant: y_up-hgb_clf
- horizon: 60, label: y_up, model: hgb_clf, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID_E1 (4 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66312 | 0.00545 |
| topk_cost_adjusted_return | +0.02392 | 0.01099 |

## Secondary

| metric | value |
|---|---|
| ece | 0.02352 |
| brier | 0.23519 |
| auc_daily_mean | 0.55117 |
| rank_ic_mean | 0.15471 |
| precision_at_k | 0.43154 |
| lift_at_k | 1.12887 |
| topk_turnover | 0.81100 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2136878 ┆ … ┆ 0.041542    ┆ 0.75365     ┆ 60.0       ┆ 0.03702    │
│ 2       ┆ p_raw    ┆ null    ┆ 2661837 ┆ … ┆ 0.010927    ┆ 0.700639    ┆ 60.0       ┆ 0.006723   │
│ 3       ┆ p_raw    ┆ null    ┆ 3233289 ┆ … ┆ 0.023802    ┆ 0.697744    ┆ 60.0       ┆ 0.019615   │
│ 4       ┆ p_raw    ┆ null    ┆ 3809273 ┆ … ┆ 0.016152    ┆ 0.682281    ┆ 60.0       ┆ 0.012058   │
│ 5       ┆ p_raw    ┆ null    ┆ 4391913 ┆ … ┆ 0.021753    ┆ 0.782403    ┆ 60.0       ┆ 0.017059   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h60_lag1_rank/predictions_valid__E1_h60_y_up-hgb_clf_seed0.parquet`
- panel rows: 5223425
- elapsed: 1524.9s
