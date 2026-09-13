# E2_h120_FS1_seed0

- stage: E2 / variant: FS1
- horizon: 120, label: y_up, model: logit, seed: 0
- feature set: FS1 (56 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 0.1}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.65794 | 0.00756 |
| rank_ic_mean | +0.16984 | 0.03070 |

## Secondary

| metric | value |
|---|---|
| ece | 0.03565 |
| brier | 0.23259 |
| auc_daily_mean | 0.55198 |
| rank_ic_mean | 0.16984 |
| precision_at_k | 0.44992 |
| lift_at_k | 1.21225 |
| topk_turnover | 0.76200 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 1904784 ┆ … ┆ 0.095727    ┆ 0.675346    ┆ 60.0       ┆ 0.091675   │
│ 2       ┆ p_raw    ┆ null    ┆ 2410430 ┆ … ┆ 0.072184    ┆ 0.57263     ┆ 60.0       ┆ 0.068748   │
│ 3       ┆ p_raw    ┆ null    ┆ 2964442 ┆ … ┆ 0.027962    ┆ 0.64462     ┆ 60.0       ┆ 0.024094   │
│ 4       ┆ p_raw    ┆ null    ┆ 3543893 ┆ … ┆ 0.031249    ┆ 0.675823    ┆ 60.0       ┆ 0.027194   │
│ 5       ┆ p_raw    ┆ null    ┆ 4123129 ┆ … ┆ 0.029225    ┆ 0.708622    ┆ 60.0       ┆ 0.024973   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1_h120_lag1_rank/predictions_valid__E2_h120_FS1_seed0.parquet`
- panel rows: 5223425
- elapsed: 257.4s
