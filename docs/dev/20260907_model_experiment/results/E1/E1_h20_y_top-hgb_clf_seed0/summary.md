# E1_h20_y_top-hgb_clf_seed0

- stage: E1 / variant: y_top-hgb_clf
- horizon: 20, label: y_top, model: hgb_clf, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID_E1 (4 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.50091 | 0.00401 |
| topk_cost_adjusted_return | +0.00942 | 0.01309 |

## Secondary

| metric | value |
|---|---|
| ece | 0.00683 |
| brier | 0.16073 |
| auc_daily_mean | 0.56519 |
| rank_ic_mean | -0.06696 |
| precision_at_k | 0.28539 |
| lift_at_k | 1.40027 |
| topk_turnover | 0.66467 |
| base_rate | 0.20361 |

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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ 0.000074    ┆ 0.594038    ┆ 60.0       ┆ -0.003491  │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ -0.00095    ┆ 0.570442    ┆ 60.0       ┆ -0.004373  │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.019351    ┆ 0.579893    ┆ 60.0       ┆ 0.015871   │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.018611    ┆ 0.58353     ┆ 60.0       ┆ 0.01511    │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.014787    ┆ 0.598184    ┆ 60.0       ┆ 0.011198   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h20_lag1_rank/predictions_valid__E1_h20_y_top-hgb_clf_seed0.parquet`
- panel rows: 5223425
- elapsed: 1585.6s
