# E2_h120_FS2_seed0

- stage: E2 / variant: FS2
- horizon: 120, label: y_up, model: logit, seed: 0
- feature set: FS2 (68 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 10.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66102 | 0.00612 |
| rank_ic_mean | +0.15811 | 0.01667 |

## Secondary

| metric | value |
|---|---|
| ece | 0.03900 |
| brier | 0.23382 |
| auc_daily_mean | 0.54890 |
| rank_ic_mean | 0.15811 |
| precision_at_k | 0.43297 |
| lift_at_k | 1.16647 |
| topk_turnover | 0.76500 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 1904784 ┆ … ┆ 0.065264    ┆ 0.605677    ┆ 60.0       ┆ 0.061629   │
│ 2       ┆ p_raw    ┆ null    ┆ 2410430 ┆ … ┆ 0.064276    ┆ 0.592699    ┆ 60.0       ┆ 0.06072    │
│ 3       ┆ p_raw    ┆ null    ┆ 2964442 ┆ … ┆ 0.019119    ┆ 0.635877    ┆ 60.0       ┆ 0.015303   │
│ 4       ┆ p_raw    ┆ null    ┆ 3543893 ┆ … ┆ 0.041927    ┆ 0.704353    ┆ 60.0       ┆ 0.0377     │
│ 5       ┆ p_raw    ┆ null    ┆ 4123129 ┆ … ┆ 0.015231    ┆ 0.733144    ┆ 60.0       ┆ 0.010832   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS2_h120_lag1_rank/predictions_valid__E2_h120_FS2_seed0.parquet`
- panel rows: 5223425
- elapsed: 441.2s
