# E4_h20_E4a-seed_seed1

- stage: E4 / variant: E4a-seed
- horizon: 20, label: y_up, model: hgb_clf, seed: 1
- feature set: FS1h (49 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 1.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66875 | 0.00505 |
| topk_cost_adjusted_return | +0.01180 | 0.00916 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01311 |
| brier | 0.23797 |
| auc_daily_mean | 0.54620 |
| rank_ic_mean | 0.13612 |
| precision_at_k | 0.46053 |
| lift_at_k | 1.15710 |
| topk_turnover | 0.68383 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ 0.003265    ┆ 0.615902    ┆ 60.0       ┆ -0.00043   │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ 0.018888    ┆ 0.547515    ┆ 60.0       ┆ 0.015603   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.017047    ┆ 0.543112    ┆ 60.0       ┆ 0.013789   │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.010309    ┆ 0.592148    ┆ 60.0       ┆ 0.006756   │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.013718    ┆ 0.673467    ┆ 60.0       ┆ 0.009677   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1h_h20_lag1_rank/predictions_valid__E4_h20_E4a-seed_seed1.parquet`
- panel rows: 5223425
- elapsed: 5048.4s
