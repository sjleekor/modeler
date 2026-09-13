# E4_h5_E4a-seed_seed1

- stage: E4 / variant: E4a-seed
- horizon: 5, label: y_up, model: hgb_clf, seed: 1
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67829 | 0.00257 |
| topk_cost_adjusted_return | +0.00038 | 0.00275 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01357 |
| brier | 0.24263 |
| auc_daily_mean | 0.53575 |
| rank_ic_mean | 0.09543 |
| precision_at_k | 0.47759 |
| lift_at_k | 1.14048 |
| topk_turnover | 0.76835 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ 0.000755    ┆ 0.677715    ┆ 60.0       ┆ -0.003311  │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ 0.003731    ┆ 0.660696    ┆ 60.0       ┆ -0.000233  │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.00603     ┆ 0.706683    ┆ 60.0       ┆ 0.00179    │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.005843    ┆ 0.696261    ┆ 60.0       ┆ 0.001666   │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ 0.003321    ┆ 0.69871     ┆ 60.0       ┆ -0.000871  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h5_lag1_rank/predictions_valid__E4_h5_E4a-seed_seed1.parquet`
- panel rows: 5223425
- elapsed: 4798.7s
