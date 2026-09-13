# E2_h120_recal-isotonic_seed0

- stage: E2 / variant: recal-isotonic
- horizon: 120, label: y_up, model: logit, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 0.1}
- calibration: isotonic

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66564 | 0.00729 |
| rank_ic_mean | +0.13021 | 0.03750 |

## Secondary

| metric | value |
|---|---|
| ece | 0.05872 |
| brier | 0.23583 |
| auc_daily_mean | 0.54190 |
| rank_ic_mean | 0.13021 |
| precision_at_k | 0.41137 |
| lift_at_k | 1.10685 |
| topk_turnover | 0.70700 |
| base_rate | 0.37084 |

ECE ceiling 0.03: **over** — recheck after calibration (`03` §3.3)

## Folds

```
shape: (10, 52)
┌─────────┬──────────┬─────────┬─────────┬───┬─────────────┬─────────────┬────────────┬────────────┐
│ fold_id ┆ pred_col ┆ skipped ┆ n_train ┆ … ┆ decile_grid ┆ decile_turn ┆ decile_cos ┆ decile_cos │
│ ---     ┆ ---      ┆ ---     ┆ ---     ┆   ┆ _top_decile ┆ over        ┆ t_bps_roun ┆ t_adjusted │
│ i64     ┆ str      ┆ null    ┆ i64     ┆   ┆ _spread     ┆ ---         ┆ dtrip      ┆ _spread    │
│         ┆          ┆         ┆         ┆   ┆ ---         ┆ f64         ┆ ---        ┆ ---        │
│         ┆          ┆         ┆         ┆   ┆ f64         ┆             ┆ f64        ┆ f64        │
╞═════════╪══════════╪═════════╪═════════╪═══╪═════════════╪═════════════╪════════════╪════════════╡
│ 1       ┆ p_raw    ┆ null    ┆ 1904784 ┆ … ┆ 0.055827    ┆ 0.54038     ┆ 60.0       ┆ 0.052585   │
│ 1       ┆ p_cal    ┆ null    ┆ 1904784 ┆ … ┆ 0.05608     ┆ 0.535917    ┆ 60.0       ┆ 0.052865   │
│ 2       ┆ p_raw    ┆ null    ┆ 2410430 ┆ … ┆ 0.051219    ┆ 0.610696    ┆ 60.0       ┆ 0.047555   │
│ 2       ┆ p_cal    ┆ null    ┆ 2410430 ┆ … ┆ 0.049623    ┆ 0.597403    ┆ 60.0       ┆ 0.046038   │
│ 3       ┆ p_raw    ┆ null    ┆ 2964442 ┆ … ┆ -0.001588   ┆ 0.582632    ┆ 60.0       ┆ -0.005084  │
│ 3       ┆ p_cal    ┆ null    ┆ 2964442 ┆ … ┆ -0.002081   ┆ 0.587398    ┆ 60.0       ┆ -0.005605  │
│ 4       ┆ p_raw    ┆ null    ┆ 3543893 ┆ … ┆ 0.011176    ┆ 0.706488    ┆ 60.0       ┆ 0.006937   │
│ 4       ┆ p_cal    ┆ null    ┆ 3543893 ┆ … ┆ 0.009777    ┆ 0.712762    ┆ 60.0       ┆ 0.005501   │
│ 5       ┆ p_raw    ┆ null    ┆ 4123129 ┆ … ┆ -0.005483   ┆ 0.757783    ┆ 60.0       ┆ -0.01003   │
│ 5       ┆ p_cal    ┆ null    ┆ 4123129 ┆ … ┆ -0.006409   ┆ 0.773857    ┆ 60.0       ┆ -0.011052  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h120_lag1_rank/predictions_valid__E2_h120_recal-isotonic_seed0.parquet`
- panel rows: 5223425
- elapsed: 146.1s
