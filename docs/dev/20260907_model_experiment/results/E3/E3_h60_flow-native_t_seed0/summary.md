# E3_h60_flow-native_t_seed0

- stage: E3 / variant: flow-native_t
- horizon: 60, label: y_up, model: logit, seed: 0
- feature set: FS0 (40 columns), flow: native_t, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 1.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66186 | 0.00707 |
| topk_cost_adjusted_return | +0.02500 | 0.02451 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01964 |
| brier | 0.23461 |
| auc_daily_mean | 0.55026 |
| rank_ic_mean | 0.15704 |
| precision_at_k | 0.43909 |
| lift_at_k | 1.14588 |
| topk_turnover | 0.76050 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2136878 ┆ … ┆ 0.063302    ┆ 0.6036      ┆ 60.0       ┆ 0.059681   │
│ 2       ┆ p_raw    ┆ null    ┆ 2661837 ┆ … ┆ 0.030697    ┆ 0.675206    ┆ 60.0       ┆ 0.026646   │
│ 3       ┆ p_raw    ┆ null    ┆ 3233289 ┆ … ┆ 0.009406    ┆ 0.702789    ┆ 60.0       ┆ 0.00519    │
│ 4       ┆ p_raw    ┆ null    ┆ 3809273 ┆ … ┆ 0.023971    ┆ 0.683661    ┆ 60.0       ┆ 0.019869   │
│ 5       ┆ p_raw    ┆ null    ┆ 4391913 ┆ … ┆ 0.009472    ┆ 0.76291     ┆ 60.0       ┆ 0.004894   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h60_native_t_rank/predictions_valid__E3_h60_flow-native_t_seed0.parquet`
- panel rows: 5223425
- elapsed: 179.0s
