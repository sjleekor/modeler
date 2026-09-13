# E1_h120_y_up-hgb_clf_seed0

- stage: E1 / variant: y_up-hgb_clf
- horizon: 120, label: y_up, model: hgb_clf, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID_E1 (4 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66148 | 0.01059 |
| rank_ic_mean | +0.13956 | 0.05250 |

## Secondary

| metric | value |
|---|---|
| ece | 0.03724 |
| brier | 0.23398 |
| auc_daily_mean | 0.54359 |
| rank_ic_mean | 0.13956 |
| precision_at_k | 0.40747 |
| lift_at_k | 1.10089 |
| topk_turnover | 0.80500 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 1904784 ┆ … ┆ 0.072292    ┆ 0.710072    ┆ 60.0       ┆ 0.068032   │
│ 2       ┆ p_raw    ┆ null    ┆ 2410430 ┆ … ┆ 0.021227    ┆ 0.601748    ┆ 60.0       ┆ 0.017617   │
│ 3       ┆ p_raw    ┆ null    ┆ 2964442 ┆ … ┆ 0.0438      ┆ 0.719708    ┆ 60.0       ┆ 0.039482   │
│ 4       ┆ p_raw    ┆ null    ┆ 3543893 ┆ … ┆ 0.031102    ┆ 0.706439    ┆ 60.0       ┆ 0.026863   │
│ 5       ┆ p_raw    ┆ null    ┆ 4123129 ┆ … ┆ 0.014084    ┆ 0.764633    ┆ 60.0       ┆ 0.009496   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h120_lag1_rank/predictions_valid__E1_h120_y_up-hgb_clf_seed0.parquet`
- panel rows: 5223425
- elapsed: 1399.2s
