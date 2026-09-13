# E1_h5_y_up-logit_seed0

- stage: E1 / variant: y_up-logit
- horizon: 5, label: y_up, model: logit, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 1.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67919 | 0.00266 |
| topk_cost_adjusted_return | -0.00159 | 0.00262 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01528 |
| brier | 0.24306 |
| auc_daily_mean | 0.52933 |
| rank_ic_mean | 0.08263 |
| precision_at_k | 0.47223 |
| lift_at_k | 1.12795 |
| topk_turnover | 0.79635 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2351735 ┆ … ┆ -0.000452   ┆ 0.699113    ┆ 60.0       ┆ -0.004647  │
│ 2       ┆ p_raw    ┆ null    ┆ 2900087 ┆ … ┆ 0.002334    ┆ 0.71321     ┆ 60.0       ┆ -0.001945  │
│ 3       ┆ p_raw    ┆ null    ┆ 3478717 ┆ … ┆ 0.002195    ┆ 0.720374    ┆ 60.0       ┆ -0.002127  │
│ 4       ┆ p_raw    ┆ null    ┆ 4056713 ┆ … ┆ 0.00545     ┆ 0.739627    ┆ 60.0       ┆ 0.001013   │
│ 5       ┆ p_raw    ┆ null    ┆ 4636599 ┆ … ┆ 0.003039    ┆ 0.705752    ┆ 60.0       ┆ -0.001196  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h5_lag1_rank/predictions_valid__E1_h5_y_up-logit_seed0.parquet`
- panel rows: 5223425
- elapsed: 155.5s
