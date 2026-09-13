# E4_h5_E4a-seed_seed2

- stage: E4 / variant: E4a-seed
- horizon: 5, label: y_up, model: hgb_clf, seed: 2
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67828 | 0.00260 |
| topk_cost_adjusted_return | +0.00029 | 0.00293 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01344 |
| brier | 0.24263 |
| auc_daily_mean | 0.53579 |
| rank_ic_mean | 0.09558 |
| precision_at_k | 0.47873 |
| lift_at_k | 1.14319 |
| topk_turnover | 0.76725 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ -0.000088   ┆ 0.68458     ┆ 60.0       ┆ -0.004195  │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ 0.003553    ┆ 0.663659    ┆ 60.0       ┆ -0.000429  │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.004713    ┆ 0.699283    ┆ 60.0       ┆ 0.000517   │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.005862    ┆ 0.700567    ┆ 60.0       ┆ 0.001659   │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ 0.003311    ┆ 0.693367    ┆ 60.0       ┆ -0.000849  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h5_lag1_rank/predictions_valid__E4_h5_E4a-seed_seed2.parquet`
- panel rows: 5223425
- elapsed: 4784.1s
