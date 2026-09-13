# E5_h20_FS3_seed0

- stage: E5 / variant: FS3
- horizon: 20, label: y_up, model: hgb_clf, seed: 0
- feature set: FS1h_FS3 (55 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 1.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66889 | 0.00524 |
| topk_cost_adjusted_return | +0.00829 | 0.00929 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01003 |
| brier | 0.23803 |
| auc_daily_mean | 0.54506 |
| rank_ic_mean | 0.13755 |
| precision_at_k | 0.45940 |
| lift_at_k | 1.15428 |
| topk_turnover | 0.68817 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ 0.003221    ┆ 0.640895    ┆ 60.0       ┆ -0.000624  │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ 0.017364    ┆ 0.560874    ┆ 60.0       ┆ 0.013999   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.016039    ┆ 0.517097    ┆ 60.0       ┆ 0.012936   │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.01922     ┆ 0.575072    ┆ 60.0       ┆ 0.01577    │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.010811    ┆ 0.656387    ┆ 60.0       ┆ 0.006873   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1h_FS3_h20_lag1_rank/predictions_valid__E5_h20_FS3_seed0.parquet`
- panel rows: 5223425
- elapsed: 5520.6s
