# E1_h20_y_top-logit_seed0

- stage: E1 / variant: y_top-logit
- horizon: 20, label: y_top, model: logit, seed: 0
- feature set: FS0 (40 columns), flow: lag1, preprocess: rank
- period: 2015-01-02 .. 2025-07-31, folds: 5
- grid: LOGIT_GRID (3 points), best: {'C': 0.1}
- calibration: none

## Primary metrics (`03` §3.2)

| metric | value | fold std |
|---|---|---|
| log_loss | 0.50221 | 0.00440 |
| topk_cost_adjusted_return | -0.00708 | 0.01344 |

## Secondary

| metric | value |
|---|---|
| ece | 0.00693 |
| brier | 0.16115 |
| auc_daily_mean | 0.55621 |
| rank_ic_mean | -0.10249 |
| precision_at_k | 0.26762 |
| lift_at_k | 1.31347 |
| topk_turnover | 0.69600 |
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
│ 1       ┆ p_raw    ┆ null    ┆ 2291444 ┆ … ┆ -0.007373   ┆ 0.590056    ┆ 60.0       ┆ -0.010913  │
│ 2       ┆ p_raw    ┆ null    ┆ 2834673 ┆ … ┆ -0.016784   ┆ 0.572764    ┆ 60.0       ┆ -0.02022   │
│ 3       ┆ p_raw    ┆ null    ┆ 3411936 ┆ … ┆ 0.006516    ┆ 0.61787     ┆ 60.0       ┆ 0.002809   │
│ 4       ┆ p_raw    ┆ null    ┆ 3988663 ┆ … ┆ -0.004857   ┆ 0.597555    ┆ 60.0       ┆ -0.008443  │
│ 5       ┆ p_raw    ┆ null    ┆ 4569840 ┆ … ┆ -0.001909   ┆ 0.595811    ┆ 60.0       ┆ -0.005484  │
└─────────┴──────────┴─────────┴─────────┴───┴─────────────┴─────────────┴────────────┴────────────┘
```

- predictions: `/Users/whishaw/wss_p/stock_data_collector/data/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/FS0_h20_lag1_rank/predictions_valid__E1_h20_y_top-logit_seed0.parquet`
- panel rows: 5223425
- elapsed: 164.2s
