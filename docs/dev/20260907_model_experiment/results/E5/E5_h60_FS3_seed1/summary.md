# E5_h60_FS3_seed1

- stage: E5 / variant: FS3
- horizon: 60, label: y_up, model: logit, seed: 1
- feature set: FS0_FS3 (51 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 0.1}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66171 | 0.00635 |
| topk_cost_adjusted_return | +0.02268 | 0.02561 |

## Secondary

| metric | value |
|---|---|
| ece | 0.02112 |
| brier | 0.23453 |
| auc_daily_mean | 0.55281 |
| rank_ic_mean | 0.16525 |
| precision_at_k | 0.44561 |
| lift_at_k | 1.16253 |
| topk_turnover | 0.73000 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2136878 ┆ … ┆ 0.043942    ┆ 0.536696    ┆ 60.0       ┆ 0.040722   │
│ 2       ┆ p_raw    ┆ null    ┆ 2661837 ┆ … ┆ 0.029081    ┆ 0.641981    ┆ 60.0       ┆ 0.025229   │
│ 3       ┆ p_raw    ┆ null    ┆ 3233289 ┆ … ┆ 0.004569    ┆ 0.666841    ┆ 60.0       ┆ 0.000568   │
│ 4       ┆ p_raw    ┆ null    ┆ 3809273 ┆ … ┆ 0.025839    ┆ 0.67036     ┆ 60.0       ┆ 0.021817   │
│ 5       ┆ p_raw    ┆ null    ┆ 4391913 ┆ … ┆ 0.018785    ┆ 0.733015    ┆ 60.0       ┆ 0.014387   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_FS3_h60_lag1_rank/predictions_valid__E5_h60_FS3_seed1.parquet`
- panel rows: 5223425
- elapsed: 255.4s
