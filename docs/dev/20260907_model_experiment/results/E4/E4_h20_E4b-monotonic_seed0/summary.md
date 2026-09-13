# E4_h20_E4b-monotonic_seed0

- stage: E4 / variant: E4b-monotonic
- horizon: 20, label: y_up, model: hgb_clf, seed: 0
- feature set: FS1h (49 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66887 | 0.00513 |
| topk_cost_adjusted_return | +0.01252 | 0.00822 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01380 |
| brier | 0.23803 |
| auc_daily_mean | 0.54615 |
| rank_ic_mean | 0.13836 |
| precision_at_k | 0.45958 |
| lift_at_k | 1.15504 |
| topk_turnover | 0.67333 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ 0.005132    ┆ 0.596135    ┆ 60.0       ┆ 0.001555   │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ 0.018412    ┆ 0.546003    ┆ 60.0       ┆ 0.015136   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.017352    ┆ 0.532601    ┆ 60.0       ┆ 0.014157   │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.009328    ┆ 0.563507    ┆ 60.0       ┆ 0.005946   │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.017189    ┆ 0.638571    ┆ 60.0       ┆ 0.013358   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1h_h20_lag1_rank/predictions_valid__E4_h20_E4b-monotonic_seed0.parquet`
- panel rows: 5223425
- elapsed: 5138.8s
