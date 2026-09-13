# E0_h60_MA-rank_seed0

- stage: E0 / variant: MA-rank
- horizon: 60, label: y_rank, model: hgb_reg, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_REG_GRID (4 points), best: {'max_iter': 200, 'learning_rate': 0.01}
- calibration: isotonic -> y_up

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66273 | 0.00747 |
| topk_cost_adjusted_return | +0.01523 | 0.02626 |

## Secondary

| metric | value |
|---|---|
| ece | 0.02705 |
| brier | 0.23504 |
| auc_daily_mean | 0.55312 |
| rank_ic_mean | 0.18015 |
| precision_at_k | 0.44093 |
| lift_at_k | 1.15273 |
| topk_turnover | 0.76100 |
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
│ 1       ┆ p_cal    ┆ null    ┆ 2136878 ┆ … ┆ 0.055819    ┆ 0.696994    ┆ 60.0       ┆ 0.051637   │
│ 2       ┆ p_cal    ┆ null    ┆ 2661837 ┆ … ┆ 0.028539    ┆ 0.697963    ┆ 60.0       ┆ 0.024352   │
│ 3       ┆ p_cal    ┆ null    ┆ 3233289 ┆ … ┆ 0.009698    ┆ 0.694086    ┆ 60.0       ┆ 0.005534   │
│ 4       ┆ p_cal    ┆ null    ┆ 3809273 ┆ … ┆ 0.024562    ┆ 0.680718    ┆ 60.0       ┆ 0.020478   │
│ 5       ┆ p_cal    ┆ null    ┆ 4391913 ┆ … ┆ 0.018133    ┆ 0.770275    ┆ 60.0       ┆ 0.013511   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h60_lag1_rank/predictions_valid__E0_h60_MA-rank_seed0.parquet`
- panel rows: 5223425
- elapsed: 736.2s
