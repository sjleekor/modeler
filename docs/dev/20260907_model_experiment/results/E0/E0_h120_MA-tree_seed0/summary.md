# E0_h120_MA-tree_seed0

- stage: E0 / variant: MA-tree
- horizon: 120, label: y_rank, model: hgb_reg, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: tree
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_REG_GRID (4 points), best: {'max_iter': 200, 'learning_rate': 0.01}
- calibration: isotonic -> y_up

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.65675 | 0.00804 |
| rank_ic_mean | +0.17920 | 0.06540 |

## Secondary

| metric | value |
|---|---|
| ece | 0.03058 |
| brier | 0.23208 |
| auc_daily_mean | 0.55272 |
| rank_ic_mean | 0.17920 |
| precision_at_k | 0.41829 |
| lift_at_k | 1.12336 |
| topk_turnover | 0.69800 |
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
│ 1       ┆ p_cal    ┆ null    ┆ 1904784 ┆ … ┆ 0.050901    ┆ 0.720307    ┆ 60.0       ┆ 0.046579   │
│ 2       ┆ p_cal    ┆ null    ┆ 2410430 ┆ … ┆ 0.125832    ┆ 0.719799    ┆ 60.0       ┆ 0.121513   │
│ 3       ┆ p_cal    ┆ null    ┆ 2964442 ┆ … ┆ 0.010492    ┆ 0.652913    ┆ 60.0       ┆ 0.006575   │
│ 4       ┆ p_cal    ┆ null    ┆ 3543893 ┆ … ┆ 0.039341    ┆ 0.710797    ┆ 60.0       ┆ 0.035076   │
│ 5       ┆ p_cal    ┆ null    ┆ 4123129 ┆ … ┆ -0.001737   ┆ 0.769716    ┆ 60.0       ┆ -0.006355  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h120_lag1_tree/predictions_valid__E0_h120_MA-tree_seed0.parquet`
- panel rows: 5223425
- elapsed: 617.2s
