# E1_h5_y_top-logit_seed0

- stage: E1 / variant: y_top-logit
- horizon: 5, label: y_top, model: logit, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 0.1}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.49857 | 0.00369 |
| topk_cost_adjusted_return | -0.00705 | 0.00451 |

## Secondary

| metric | value |
|---|---|
| ece | 0.00514 |
| brier | 0.16000 |
| auc_daily_mean | 0.58261 |
| rank_ic_mean | -0.08607 |
| precision_at_k | 0.29436 |
| lift_at_k | 1.44491 |
| topk_turnover | 0.57584 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ 0.000033    ┆ 0.453443    ┆ 60.0       ┆ -0.002688  │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ -0.005791   ┆ 0.454788    ┆ 60.0       ┆ -0.008519  │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.000079    ┆ 0.454353    ┆ 60.0       ┆ -0.002647  │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ -0.004382   ┆ 0.432445    ┆ 60.0       ┆ -0.006977  │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ -0.007099   ┆ 0.413035    ┆ 60.0       ┆ -0.009577  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h5_lag1_rank/predictions_valid__E1_h5_y_top-logit_seed0.parquet`
- panel rows: 5223425
- elapsed: 166.8s
