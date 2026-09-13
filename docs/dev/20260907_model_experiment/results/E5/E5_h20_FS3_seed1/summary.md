# E5_h20_FS3_seed1

- stage: E5 / variant: FS3
- horizon: 20, label: y_up, model: hgb_clf, seed: 1
- feature set: FS1h_FS3 (55 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID (16 points), best: {'max_iter': 200, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66889 | 0.00520 |
| topk_cost_adjusted_return | +0.00975 | 0.00905 |

## Secondary

| metric | value |
|---|---|
| ece | 0.00994 |
| brier | 0.23803 |
| auc_daily_mean | 0.54519 |
| rank_ic_mean | 0.13770 |
| precision_at_k | 0.45992 |
| lift_at_k | 1.15568 |
| topk_turnover | 0.68900 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ 0.003861    ┆ 0.625413    ┆ 60.0       ┆ 0.000109   │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ 0.018612    ┆ 0.556771    ┆ 60.0       ┆ 0.015272   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.016184    ┆ 0.526773    ┆ 60.0       ┆ 0.013023   │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.009305    ┆ 0.578727    ┆ 60.0       ┆ 0.005833   │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.011758    ┆ 0.639004    ┆ 60.0       ┆ 0.007924   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS1h_FS3_h20_lag1_rank/predictions_valid__E5_h20_FS3_seed1.parquet`
- panel rows: 5223425
- elapsed: 5508.7s
