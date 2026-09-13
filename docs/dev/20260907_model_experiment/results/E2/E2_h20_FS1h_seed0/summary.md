# E2_h20_FS1h_seed0

- stage: E2 / variant: FS1h
- horizon: 20, label: y_up, model: hgb_clf, seed: 0
- feature set: FS1h (49 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 31, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66872 | 0.00516 |
| topk_cost_adjusted_return | +0.01218 | 0.00884 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01269 |
| brier | 0.23795 |
| auc_daily_mean | 0.54629 |
| rank_ic_mean | 0.13673 |
| precision_at_k | 0.46124 |
| lift_at_k | 1.15908 |
| topk_turnover | 0.68633 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ 0.003293    ┆ 0.606728    ┆ 60.0       ┆ -0.000347  │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ 0.019218    ┆ 0.548284    ┆ 60.0       ┆ 0.015929   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.017837    ┆ 0.543484    ┆ 60.0       ┆ 0.014576   │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.007019    ┆ 0.583896    ┆ 60.0       ┆ 0.003515   │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.014341    ┆ 0.670847    ┆ 60.0       ┆ 0.010316   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1h_h20_lag1_rank/predictions_valid__E2_h20_FS1h_seed0.parquet`
- panel rows: 5223425
- elapsed: 5158.6s
