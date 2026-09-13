# E2_h60_FS1_seed0

- stage: E2 / variant: FS1
- horizon: 60, label: y_up, model: logit, seed: 0
- feature set: FS1 (56 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 10.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66202 | 0.00651 |
| topk_cost_adjusted_return | +0.03074 | 0.01827 |

## Secondary

| metric | value |
|---|---|
| ece | 0.02239 |
| brier | 0.23467 |
| auc_daily_mean | 0.55124 |
| rank_ic_mean | 0.16252 |
| precision_at_k | 0.45766 |
| lift_at_k | 1.19470 |
| topk_turnover | 0.76450 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2136878 ┆ … ┆ 0.054708    ┆ 0.624929    ┆ 60.0       ┆ 0.050958   │
│ 2       ┆ p_raw    ┆ null    ┆ 2661837 ┆ … ┆ 0.038443    ┆ 0.683019    ┆ 60.0       ┆ 0.034345   │
│ 3       ┆ p_raw    ┆ null    ┆ 3233289 ┆ … ┆ 0.020451    ┆ 0.693675    ┆ 60.0       ┆ 0.016289   │
│ 4       ┆ p_raw    ┆ null    ┆ 3809273 ┆ … ┆ 0.024518    ┆ 0.681202    ┆ 60.0       ┆ 0.020431   │
│ 5       ┆ p_raw    ┆ null    ┆ 4391913 ┆ … ┆ 0.014439    ┆ 0.743341    ┆ 60.0       ┆ 0.009979   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1_h60_lag1_rank/predictions_valid__E2_h60_FS1_seed0.parquet`
- panel rows: 5223425
- elapsed: 270.5s
