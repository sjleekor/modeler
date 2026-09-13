# E1_h120_y_up-logit_seed0

- stage: E1 / variant: y_up-logit
- horizon: 120, label: y_up, model: logit, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 0.1}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.65703 | 0.00699 |
| rank_ic_mean | +0.16348 | 0.03938 |

## Secondary

| metric | value |
|---|---|
| ece | 0.03213 |
| brier | 0.23225 |
| auc_daily_mean | 0.55185 |
| rank_ic_mean | 0.16348 |
| precision_at_k | 0.42482 |
| lift_at_k | 1.14441 |
| topk_turnover | 0.80200 |
| base_rate | 0.37084 |

ECE ceiling 0.03: **over** — recheck after calibration (`03` §3.3)

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
│ 1       ┆ p_raw    ┆ null    ┆ 1904784 ┆ … ┆ 0.102738    ┆ 0.663662    ┆ 60.0       ┆ 0.098756   │
│ 2       ┆ p_raw    ┆ null    ┆ 2410430 ┆ … ┆ 0.070755    ┆ 0.650975    ┆ 60.0       ┆ 0.066849   │
│ 3       ┆ p_raw    ┆ null    ┆ 2964442 ┆ … ┆ 0.028       ┆ 0.723889    ┆ 60.0       ┆ 0.023656   │
│ 4       ┆ p_raw    ┆ null    ┆ 3543893 ┆ … ┆ 0.030984    ┆ 0.710952    ┆ 60.0       ┆ 0.026718   │
│ 5       ┆ p_raw    ┆ null    ┆ 4123129 ┆ … ┆ 0.006535    ┆ 0.746368    ┆ 60.0       ┆ 0.002057   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h120_lag1_rank/predictions_valid__E1_h120_y_up-logit_seed0.parquet`
- panel rows: 5223425
- elapsed: 171.1s
