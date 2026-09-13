# E0_h20_MA-rank_seed0

- stage: E0 / variant: MA-rank
- horizon: 20, label: y_rank, model: hgb_reg, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: HGB_REG_GRID (4 points), best: {'max_iter': 200, 'learning_rate': 0.01}
- calibration: isotonic -> y_up

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.67045 | 0.00578 |
| topk_cost_adjusted_return | +0.00665 | 0.00583 |

## Secondary

| metric | value |
|---|---|
| ece | 0.01943 |
| brier | 0.23873 |
| auc_daily_mean | 0.53884 |
| rank_ic_mean | 0.14961 |
| precision_at_k | 0.44977 |
| lift_at_k | 1.13154 |
| topk_turnover | 0.70983 |
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
│ 1       ┆ p_cal    ┆ null    ┆ 2291444 ┆ … ┆ 0.01613     ┆ 0.697012    ┆ 60.0       ┆ 0.011948   │
│ 2       ┆ p_cal    ┆ null    ┆ 2834673 ┆ … ┆ 0.017134    ┆ 0.685427    ┆ 60.0       ┆ 0.013021   │
│ 3       ┆ p_cal    ┆ null    ┆ 3411936 ┆ … ┆ 0.009844    ┆ 0.650508    ┆ 60.0       ┆ 0.005941   │
│ 4       ┆ p_cal    ┆ null    ┆ 3988663 ┆ … ┆ 0.014763    ┆ 0.547824    ┆ 60.0       ┆ 0.011477   │
│ 5       ┆ p_cal    ┆ null    ┆ 4569840 ┆ … ┆ 0.011584    ┆ 0.583781    ┆ 60.0       ┆ 0.008081   │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h20_lag1_rank/predictions_valid__E0_h20_MA-rank_seed0.parquet`
- panel rows: 5223425
- elapsed: 789.6s
