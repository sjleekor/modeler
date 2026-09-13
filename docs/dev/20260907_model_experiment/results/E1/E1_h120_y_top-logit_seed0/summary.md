# E1_h120_y_top-logit_seed0

- stage: E1 / variant: y_top-logit
- horizon: 120, label: y_top, model: logit, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 0.1}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.49777 | 0.00978 |
| rank_ic_mean | +0.03609 | 0.06288 |

## Secondary

| metric | value |
|---|---|
| ece | 0.02183 |
| brier | 0.15865 |
| auc_daily_mean | 0.52379 |
| rank_ic_mean | 0.03609 |
| precision_at_k | 0.22527 |
| lift_at_k | 1.14588 |
| topk_turnover | 0.74600 |
| base_rate | 0.19710 |

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
│ 1       ┆ p_raw    ┆ null    ┆ 1904784 ┆ … ┆ 0.093737    ┆ 0.619492    ┆ 60.0       ┆ 0.09002    │
│ 2       ┆ p_raw    ┆ null    ┆ 2410430 ┆ … ┆ -0.004819   ┆ 0.633098    ┆ 60.0       ┆ -0.008617  │
│ 3       ┆ p_raw    ┆ null    ┆ 2964442 ┆ … ┆ 0.044907    ┆ 0.635789    ┆ 60.0       ┆ 0.041093   │
│ 4       ┆ p_raw    ┆ null    ┆ 3543893 ┆ … ┆ 0.027467    ┆ 0.719393    ┆ 60.0       ┆ 0.023151   │
│ 5       ┆ p_raw    ┆ null    ┆ 4123129 ┆ … ┆ -0.005676   ┆ 0.804424    ┆ 60.0       ┆ -0.010502  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h120_lag1_rank/predictions_valid__E1_h120_y_top-logit_seed0.parquet`
- panel rows: 5223425
- elapsed: 145.7s
