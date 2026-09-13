# E1_h5_y_top-hgb_clf_seed0

- stage: E1 / variant: y_top-hgb_clf
- horizon: 5, label: y_top, model: hgb_clf, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID_E1 (4 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.49731 | 0.00337 |
| topk_cost_adjusted_return | -0.00115 | 0.00305 |

## Secondary

| metric | value |
|---|---|
| ece | 0.00634 |
| brier | 0.15960 |
| auc_daily_mean | 0.58931 |
| rank_ic_mean | -0.07370 |
| precision_at_k | 0.30755 |
| lift_at_k | 1.50940 |
| topk_turnover | 0.55475 |
| base_rate | 0.20369 |

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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ 0.002177    ┆ 0.493666    ┆ 60.0       ┆ -0.000785  │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ -0.000943   ┆ 0.459423    ┆ 60.0       ┆ -0.0037    │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.00217     ┆ 0.44068     ┆ 60.0       ┆ -0.000474  │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.000806    ┆ 0.439187    ┆ 60.0       ┆ -0.001829  │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ -0.002111   ┆ 0.422268    ┆ 60.0       ┆ -0.004645  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h5_lag1_rank/predictions_valid__E1_h5_y_top-hgb_clf_seed0.parquet`
- panel rows: 5223425
- elapsed: 1935.1s
