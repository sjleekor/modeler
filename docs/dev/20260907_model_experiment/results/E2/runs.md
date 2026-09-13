# E2 — runs

| run | h | label | model | FS | seed | prob | econ | ece |
|---|---|---|---|---|---|---|---|---|
| E2_h5_FS1_seed0 | 5 | y_up | hgb_clf | FS1 | 0 | 0.67821 | +0.00068 | 0.0132 |
| E2_h5_FS1h_seed0 | 5 | y_up | hgb_clf | FS1h | 0 | 0.67819 | +0.00015 | 0.0132 |
| E2_h5_FS2_seed0 | 5 | y_up | hgb_clf | FS2 | 0 | 0.67811 | +0.00068 | 0.0142 |
| E2_h20_FS1_seed0 | 20 | y_up | hgb_clf | FS1 | 0 | 0.66874 | +0.01128 | 0.0138 |
| E2_h20_FS1h_seed0 | 20 | y_up | hgb_clf | FS1h | 0 | 0.66872 | +0.01218 | 0.0127 |
| E2_h20_FS2_seed0 | 20 | y_up | hgb_clf | FS2 | 0 | 0.66870 | +0.00830 | 0.0118 |
| E2_h60_FS1_seed0 | 60 | y_up | logit | FS1 | 0 | 0.66202 | +0.03074 | 0.0224 |
| E2_h60_FS1h_seed0 | 60 | y_up | logit | FS1h | 0 | 0.66203 | +0.03549 | 0.0225 |
| E2_h60_FS2_seed0 | 60 | y_up | logit | FS2 | 0 | 0.66230 | +0.03411 | 0.0212 |
| E2_h120_FS1_seed0 | 120 | y_up | logit | FS1 | 0 | 0.65794 | +0.16984 | 0.0357 |
| E2_h120_FS1h_seed0 | 120 | y_up | logit | FS1h | 0 | 0.65932 | +0.16369 | 0.0393 |
| E2_h120_FS2_seed0 | 120 | y_up | logit | FS2 | 0 | 0.66102 | +0.15811 | 0.0390 |

## Selection

rule: `05` §2 E2

- h5: FS0
- h20: FS1h
- h60: FS0
- h120: dropped (no calibrated probability inside the ceiling)
