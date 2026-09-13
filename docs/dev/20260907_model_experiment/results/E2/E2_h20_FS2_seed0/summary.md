# E2_h20_FS2_seed0

- stage: E2 / variant: FS2
- horizon: 20, label: y_up, model: hgb_clf, seed: 0
- feature set: FS2 (68 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66870 | 0.00514 |
| topk_cost_adjusted_return | +0.00830 | 0.00999 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01185 |
| brier | 0.23794 |
| auc_daily_mean | 0.54407 |
| rank_ic_mean | 0.13708 |
| precision_at_k | 0.45395 |
| lift_at_k | 1.14020 |
| topk_turnover | 0.72883 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ -0.001483   ┆ 0.580233    ┆ 60.0       ┆ -0.004965  │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ 0.017807    ┆ 0.678403    ┆ 60.0       ┆ 0.013737   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.017927    ┆ 0.674495    ┆ 60.0       ┆ 0.01388    │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.00974     ┆ 0.583477    ┆ 60.0       ┆ 0.006239   │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.01057     ┆ 0.695275    ┆ 60.0       ┆ 0.006399   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS2_h20_lag1_rank/predictions_valid__E2_h20_FS2_seed0.parquet`
- panel rows: 5223425
- elapsed: 6477.9s
