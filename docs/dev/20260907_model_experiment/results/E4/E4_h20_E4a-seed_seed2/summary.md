# E4_h20_E4a-seed_seed2

- stage: E4 / variant: E4a-seed
- horizon: 20, label: y_up, model: hgb_clf, seed: 2
- feature set: FS1h (49 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 1.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66878 | 0.00492 |
| topk_cost_adjusted_return | +0.00865 | 0.01038 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01321 |
| brier | 0.23798 |
| auc_daily_mean | 0.54606 |
| rank_ic_mean | 0.13634 |
| precision_at_k | 0.46036 |
| lift_at_k | 1.15666 |
| topk_turnover | 0.68500 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ 0.001988    ┆ 0.596489    ┆ 60.0       ┆ -0.001591  │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ 0.019622    ┆ 0.545637    ┆ 60.0       ┆ 0.016348   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.019054    ┆ 0.54531     ┆ 60.0       ┆ 0.015782   │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.008171    ┆ 0.575013    ┆ 60.0       ┆ 0.004721   │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.014397    ┆ 0.674274    ┆ 60.0       ┆ 0.010352   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1h_h20_lag1_rank/predictions_valid__E4_h20_E4a-seed_seed2.parquet`
- panel rows: 5223425
- elapsed: 5057.5s
