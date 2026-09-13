# E0_h20_MA-tree_seed0

- stage: E0 / variant: MA-tree
- horizon: 20, label: y_rank, model: hgb_reg, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: tree
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_REG_GRID (4 points), best: {'max_iter': 200, 'learning_rate': 0.01}
- calibration: isotonic -> y_up

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66976 | 0.00647 |
| topk_cost_adjusted_return | +0.01359 | 0.00953 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01657 |
| brier | 0.23839 |
| auc_daily_mean | 0.54322 |
| rank_ic_mean | 0.15150 |
| precision_at_k | 0.46721 |
| lift_at_k | 1.17517 |
| topk_turnover | 0.71683 |
| base_rate | 0.39741 |

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
│ 1       ┆ p_cal    ┆ null    ┆ 2291444 ┆ … ┆ 0.011915    ┆ 0.644421    ┆ 60.0       ┆ 0.008048   │
│ 2       ┆ p_cal    ┆ null    ┆ 2834673 ┆ … ┆ 0.016247    ┆ 0.651215    ┆ 60.0       ┆ 0.01234    │
│ 3       ┆ p_cal    ┆ null    ┆ 3411936 ┆ … ┆ 0.016912    ┆ 0.74104     ┆ 60.0       ┆ 0.012466   │
│ 4       ┆ p_cal    ┆ null    ┆ 3988663 ┆ … ┆ 0.013486    ┆ 0.56099     ┆ 60.0       ┆ 0.01012    │
│ 5       ┆ p_cal    ┆ null    ┆ 4569840 ┆ … ┆ 0.019047    ┆ 0.630231    ┆ 60.0       ┆ 0.015265   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h20_lag1_tree/predictions_valid__E0_h20_MA-tree_seed0.parquet`
- panel rows: 5223425
- elapsed: 835.8s
