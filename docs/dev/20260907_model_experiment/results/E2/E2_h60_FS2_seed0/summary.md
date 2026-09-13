# E2_h60_FS2_seed0

- stage: E2 / variant: FS2
- horizon: 60, label: y_up, model: logit, seed: 0
- feature set: FS2 (68 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 0.1}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66230 | 0.00460 |
| topk_cost_adjusted_return | +0.03411 | 0.02832 |

## Secondary

| metric | value |
|---|---|
| ece | 0.02116 |
| brier | 0.23476 |
| auc_daily_mean | 0.55147 |
| rank_ic_mean | 0.16034 |
| precision_at_k | 0.45740 |
| lift_at_k | 1.19364 |
| topk_turnover | 0.75600 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2136878 ┆ … ┆ 0.050568    ┆ 0.575635    ┆ 60.0       ┆ 0.047114   │
│ 2       ┆ p_raw    ┆ null    ┆ 2661837 ┆ … ┆ 0.042286    ┆ 0.68968     ┆ 60.0       ┆ 0.038148   │
│ 3       ┆ p_raw    ┆ null    ┆ 3233289 ┆ … ┆ 0.021387    ┆ 0.717169    ┆ 60.0       ┆ 0.017084   │
│ 4       ┆ p_raw    ┆ null    ┆ 3809273 ┆ … ┆ 0.031219    ┆ 0.702404    ┆ 60.0       ┆ 0.027004   │
│ 5       ┆ p_raw    ┆ null    ┆ 4391913 ┆ … ┆ 0.020518    ┆ 0.760342    ┆ 60.0       ┆ 0.015955   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS2_h60_lag1_rank/predictions_valid__E2_h60_FS2_seed0.parquet`
- panel rows: 5223425
- elapsed: 417.5s
