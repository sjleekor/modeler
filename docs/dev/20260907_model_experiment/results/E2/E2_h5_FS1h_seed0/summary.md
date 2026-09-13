# E2_h5_FS1h_seed0

- stage: E2 / variant: FS1h
- horizon: 5, label: y_up, model: hgb_clf, seed: 0
- feature set: FS1h (46 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 1.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67819 | 0.00236 |
| topk_cost_adjusted_return | +0.00015 | 0.00181 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01325 |
| brier | 0.24258 |
| auc_daily_mean | 0.53719 |
| rank_ic_mean | 0.09744 |
| precision_at_k | 0.48367 |
| lift_at_k | 1.15484 |
| topk_turnover | 0.76086 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ 0.000524    ┆ 0.662346    ┆ 60.0       ┆ -0.00345   │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ 0.004256    ┆ 0.670409    ┆ 60.0       ┆ 0.000234   │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.004761    ┆ 0.686169    ┆ 60.0       ┆ 0.000644   │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.004847    ┆ 0.69303     ┆ 60.0       ┆ 0.000689   │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ 0.003443    ┆ 0.681948    ┆ 60.0       ┆ -0.000649  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1h_h5_lag1_rank/predictions_valid__E2_h5_FS1h_seed0.parquet`
- panel rows: 5223425
- elapsed: 5502.5s
