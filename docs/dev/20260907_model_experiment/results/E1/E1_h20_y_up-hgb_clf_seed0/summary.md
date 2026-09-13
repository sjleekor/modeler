# E1_h20_y_up-hgb_clf_seed0

- stage: E1 / variant: y_up-hgb_clf
- horizon: 20, label: y_up, model: hgb_clf, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_CLF_GRID_E1 (4 points), best: {'max_iter': 400, 'learning_rate': 0.03, 'max_leaf_nodes': 15, 'l2_regularization': 0.0}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.66912 | 0.00494 |
| topk_cost_adjusted_return | +0.00939 | 0.00922 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01279 |
| brier | 0.23815 |
| auc_daily_mean | 0.54388 |
| rank_ic_mean | 0.12916 |
| precision_at_k | 0.45514 |
| lift_at_k | 1.14376 |
| topk_turnover | 0.74450 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ -0.001436   ┆ 0.578805    ┆ 60.0       ┆ -0.004909  │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ 0.015467    ┆ 0.661839    ┆ 60.0       ┆ 0.011496   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.019009    ┆ 0.630783    ┆ 60.0       ┆ 0.015225   │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ 0.011727    ┆ 0.659964    ┆ 60.0       ┆ 0.007767   │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ 0.014655    ┆ 0.697422    ┆ 60.0       ┆ 0.01047    │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h20_lag1_rank/predictions_valid__E1_h20_y_up-hgb_clf_seed0.parquet`
- panel rows: 5223425
- elapsed: 1654.3s
