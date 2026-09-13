# E0_h120_MA-rank_seed0

- stage: E0 / variant: MA-rank
- horizon: 120, label: y_rank, model: hgb_reg, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_REG_GRID (4 points), best: {'max_iter': 200, 'learning_rate': 0.01}
- calibration: isotonic -> y_up

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.65595 | 0.00792 |
| rank_ic_mean | +0.18703 | 0.05242 |

## Secondary

| metric | value |
|---|---|
| ece | 0.03027 |
| brier | 0.23181 |
| auc_daily_mean | 0.55656 |
| rank_ic_mean | 0.18703 |
| precision_at_k | 0.42411 |
| lift_at_k | 1.13972 |
| topk_turnover | 0.74400 |
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
│ 1       ┆ p_cal    ┆ null    ┆ 1904784 ┆ … ┆ 0.031765    ┆ 0.66106     ┆ 60.0       ┆ 0.027799   │
│ 2       ┆ p_cal    ┆ null    ┆ 2410430 ┆ … ┆ 0.109781    ┆ 0.697248    ┆ 60.0       ┆ 0.105597   │
│ 3       ┆ p_cal    ┆ null    ┆ 2964442 ┆ … ┆ 0.01478     ┆ 0.653474    ┆ 60.0       ┆ 0.010859   │
│ 4       ┆ p_cal    ┆ null    ┆ 3543893 ┆ … ┆ 0.020508    ┆ 0.690465    ┆ 60.0       ┆ 0.016365   │
│ 5       ┆ p_cal    ┆ null    ┆ 4123129 ┆ … ┆ -0.003835   ┆ 0.807592    ┆ 60.0       ┆ -0.008681  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h120_lag1_rank/predictions_valid__E0_h120_MA-rank_seed0.parquet`
- panel rows: 5223425
- elapsed: 631.1s
