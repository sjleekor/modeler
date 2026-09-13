# E2_h120_FS1h_seed0

- stage: E2 / variant: FS1h
- horizon: 120, label: y_up, model: logit, seed: 0
- feature set: FS1h (50 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 10.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.65932 | 0.00607 |
| rank_ic_mean | +0.16369 | 0.02391 |

## Secondary

| metric | value |
|---|---|
| ece | 0.03929 |
| brier | 0.23317 |
| auc_daily_mean | 0.54963 |
| rank_ic_mean | 0.16369 |
| precision_at_k | 0.44134 |
| lift_at_k | 1.18913 |
| topk_turnover | 0.74500 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 1904784 ┆ … ┆ 0.086427    ┆ 0.612235    ┆ 60.0       ┆ 0.082753   │
│ 2       ┆ p_raw    ┆ null    ┆ 2410430 ┆ … ┆ 0.062222    ┆ 0.561479    ┆ 60.0       ┆ 0.058853   │
│ 3       ┆ p_raw    ┆ null    ┆ 2964442 ┆ … ┆ 0.024806    ┆ 0.633655    ┆ 60.0       ┆ 0.021004   │
│ 4       ┆ p_raw    ┆ null    ┆ 3543893 ┆ … ┆ 0.025212    ┆ 0.673347    ┆ 60.0       ┆ 0.021172   │
│ 5       ┆ p_raw    ┆ null    ┆ 4123129 ┆ … ┆ 0.027814    ┆ 0.690713    ┆ 60.0       ┆ 0.02367    │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1h_h120_lag1_rank/predictions_valid__E2_h120_FS1h_seed0.parquet`
- panel rows: 5223425
- elapsed: 248.3s
