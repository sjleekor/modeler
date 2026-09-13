# E2_h60_FS1h_seed0

- stage: E2 / variant: FS1h
- horizon: 60, label: y_up, model: logit, seed: 0
- feature set: FS1h (54 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 0.1}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66203 | 0.00651 |
| topk_cost_adjusted_return | +0.03549 | 0.01560 |

## Secondary

| metric | value |
|---|---|
| ece | 0.02251 |
| brier | 0.23467 |
| auc_daily_mean | 0.55106 |
| rank_ic_mean | 0.16233 |
| precision_at_k | 0.45825 |
| lift_at_k | 1.19634 |
| topk_turnover | 0.75550 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2136878 ┆ … ┆ 0.053122    ┆ 0.622603    ┆ 60.0       ┆ 0.049386   │
│ 2       ┆ p_raw    ┆ null    ┆ 2661837 ┆ … ┆ 0.042929    ┆ 0.671942    ┆ 60.0       ┆ 0.038897   │
│ 3       ┆ p_raw    ┆ null    ┆ 3233289 ┆ … ┆ 0.020533    ┆ 0.685911    ┆ 60.0       ┆ 0.016417   │
│ 4       ┆ p_raw    ┆ null    ┆ 3809273 ┆ … ┆ 0.024793    ┆ 0.684586    ┆ 60.0       ┆ 0.020686   │
│ 5       ┆ p_raw    ┆ null    ┆ 4391913 ┆ … ┆ 0.015597    ┆ 0.743416    ┆ 60.0       ┆ 0.011137   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1h_h60_lag1_rank/predictions_valid__E2_h60_FS1h_seed0.parquet`
- panel rows: 5223425
- elapsed: 345.4s
