# E4_h60_E4a-seed_seed2

- stage: E4 / variant: E4a-seed
- horizon: 60, label: y_up, model: logit, seed: 2
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 10.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66195 | 0.00685 |
| topk_cost_adjusted_return | +0.02157 | 0.02742 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01884 |
| brier | 0.23466 |
| auc_daily_mean | 0.54953 |
| rank_ic_mean | 0.15580 |
| precision_at_k | 0.43972 |
| lift_at_k | 1.14780 |
| topk_turnover | 0.77150 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2136878 ┆ … ┆ 0.063405    ┆ 0.646923    ┆ 60.0       ┆ 0.059524   │
│ 2       ┆ p_raw    ┆ null    ┆ 2661837 ┆ … ┆ 0.033979    ┆ 0.669671    ┆ 60.0       ┆ 0.029961   │
│ 3       ┆ p_raw    ┆ null    ┆ 3233289 ┆ … ┆ 0.01409     ┆ 0.703646    ┆ 60.0       ┆ 0.009868   │
│ 4       ┆ p_raw    ┆ null    ┆ 3809273 ┆ … ┆ 0.021731    ┆ 0.689286    ┆ 60.0       ┆ 0.017595   │
│ 5       ┆ p_raw    ┆ null    ┆ 4391913 ┆ … ┆ 0.012346    ┆ 0.7697      ┆ 60.0       ┆ 0.007727   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h60_lag1_rank/predictions_valid__E4_h60_E4a-seed_seed2.parquet`
- panel rows: 5223425
- elapsed: 169.0s
