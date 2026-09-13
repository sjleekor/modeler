# E4_h60_E4b-monotonic_seed0

- stage: E4 / variant: E4b-monotonic
- horizon: 60, label: y_up, model: hgb_clf, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66218 | 0.00592 |
| topk_cost_adjusted_return | +0.02190 | 0.01817 |

## Secondary

| metric | value |
|---|---|
| ece | 0.02140 |
| brier | 0.23477 |
| auc_daily_mean | 0.55276 |
| rank_ic_mean | 0.15998 |
| precision_at_k | 0.43415 |
| lift_at_k | 1.13642 |
| topk_turnover | 0.79850 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2136878 ┆ … ┆ 0.042       ┆ 0.741873    ┆ 60.0       ┆ 0.037549   │
│ 2       ┆ p_raw    ┆ null    ┆ 2661837 ┆ … ┆ 0.017935    ┆ 0.670703    ┆ 60.0       ┆ 0.01391    │
│ 3       ┆ p_raw    ┆ null    ┆ 3233289 ┆ … ┆ 0.025151    ┆ 0.701286    ┆ 60.0       ┆ 0.020944   │
│ 4       ┆ p_raw    ┆ null    ┆ 3809273 ┆ … ┆ 0.019285    ┆ 0.683465    ┆ 60.0       ┆ 0.015185   │
│ 5       ┆ p_raw    ┆ null    ┆ 4391913 ┆ … ┆ 0.02763     ┆ 0.777806    ┆ 60.0       ┆ 0.022963   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h60_lag1_rank/predictions_valid__E4_h60_E4b-monotonic_seed0.parquet`
- panel rows: 5223425
- elapsed: 3992.8s
