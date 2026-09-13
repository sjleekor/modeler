# E5_h20_FS3_seed2

- stage: E5 / variant: FS3
- horizon: 20, label: y_up, model: hgb_clf, seed: 2
- feature set: FS1h_FS3 (55 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66886 | 0.00517 |
| topk_cost_adjusted_return | +0.01101 | 0.00926 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01102 |
| brier | 0.23802 |
| auc_daily_mean | 0.54561 |
| rank_ic_mean | 0.13551 |
| precision_at_k | 0.46451 |
| lift_at_k | 1.16725 |
| topk_turnover | 0.70200 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ 0.00416     ┆ 0.627927    ┆ 60.0       ┆ 0.000392   │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ 0.021329    ┆ 0.587585    ┆ 60.0       ┆ 0.017804   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.018176    ┆ 0.564312    ┆ 60.0       ┆ 0.01479    │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.01        ┆ 0.590214    ┆ 60.0       ┆ 0.006459   │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.01352     ┆ 0.680736    ┆ 60.0       ┆ 0.009435   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1h_FS3_h20_lag1_rank/predictions_valid__E5_h20_FS3_seed2.parquet`
- panel rows: 5223425
- elapsed: 5804.2s
