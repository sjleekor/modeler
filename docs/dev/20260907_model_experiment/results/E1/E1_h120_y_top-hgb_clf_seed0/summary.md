# E1_h120_y_top-hgb_clf_seed0

- stage: E1 / variant: y_top-hgb_clf
- horizon: 120, label: y_top, model: hgb_clf, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID_E1 (4 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.49836 | 0.00559 |
| rank_ic_mean | +0.03525 | 0.04626 |

## Secondary

| metric | value |
|---|---|
| ece | 0.02129 |
| brier | 0.15882 |
| auc_daily_mean | 0.53042 |
| rank_ic_mean | 0.03525 |
| precision_at_k | 0.22317 |
| lift_at_k | 1.13268 |
| topk_turnover | 0.80000 |
| base_rate | 0.19710 |

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
│ 1       ┆ p_raw    ┆ null    ┆ 1904784 ┆ … ┆ 0.070022    ┆ 0.751787    ┆ 60.0       ┆ 0.065511   │
│ 2       ┆ p_raw    ┆ null    ┆ 2410430 ┆ … ┆ 0.024421    ┆ 0.675619    ┆ 60.0       ┆ 0.020367   │
│ 3       ┆ p_raw    ┆ null    ┆ 2964442 ┆ … ┆ 0.042049    ┆ 0.684269    ┆ 60.0       ┆ 0.037944   │
│ 4       ┆ p_raw    ┆ null    ┆ 3543893 ┆ … ┆ -0.009147   ┆ 0.710757    ┆ 60.0       ┆ -0.013412  │
│ 5       ┆ p_raw    ┆ null    ┆ 4123129 ┆ … ┆ 0.032363    ┆ 0.751883    ┆ 60.0       ┆ 0.027852   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h120_lag1_rank/predictions_valid__E1_h120_y_top-hgb_clf_seed0.parquet`
- panel rows: 5223425
- elapsed: 1328.2s
