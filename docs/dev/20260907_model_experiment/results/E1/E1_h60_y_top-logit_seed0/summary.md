# E1_h60_y_top-logit_seed0

- stage: E1 / variant: y_top-logit
- horizon: 60, label: y_top, model: logit, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 0.1}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.50099 | 0.00622 |
| topk_cost_adjusted_return | +0.01268 | 0.03267 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01202 |
| brier | 0.16025 |
| auc_daily_mean | 0.53660 |
| rank_ic_mean | -0.03780 |
| precision_at_k | 0.24808 |
| lift_at_k | 1.23595 |
| topk_turnover | 0.71550 |
| base_rate | 0.20069 |

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
│ 1       ┆ p_raw    ┆ null    ┆ 2136878 ┆ … ┆ 0.046579    ┆ 0.551166    ┆ 60.0       ┆ 0.043272   │
│ 2       ┆ p_raw    ┆ null    ┆ 2661837 ┆ … ┆ -0.00711    ┆ 0.629803    ┆ 60.0       ┆ -0.010888  │
│ 3       ┆ p_raw    ┆ null    ┆ 3233289 ┆ … ┆ 0.03032     ┆ 0.642658    ┆ 60.0       ┆ 0.026464   │
│ 4       ┆ p_raw    ┆ null    ┆ 3809273 ┆ … ┆ 0.00018     ┆ 0.669298    ┆ 60.0       ┆ -0.003836  │
│ 5       ┆ p_raw    ┆ null    ┆ 4391913 ┆ … ┆ -0.004146   ┆ 0.760876    ┆ 60.0       ┆ -0.008711  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h60_lag1_rank/predictions_valid__E1_h60_y_top-logit_seed0.parquet`
- panel rows: 5223425
- elapsed: 142.3s
