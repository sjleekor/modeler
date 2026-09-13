# E1 — runs

| run | h | label | model | FS | seed | prob | econ | ece |
|---|---|---|---|---|---|---|---|---|
| E1_h5_y_top-hgb_clf_seed0 | 5 | y_top | hgb_clf | FS0 | 0 | 0.49731 | -0.00115 | 0.0063 |
| E1_h5_y_top-logit_seed0 | 5 | y_top | logit | FS0 | 0 | 0.49857 | -0.00705 | 0.0051 |
| E1_h5_y_up-hgb_clf_seed0 | 5 | y_up | hgb_clf | FS0 | 0 | 0.67829 | +0.00016 | 0.0136 |
| E1_h5_y_up-logit_seed0 | 5 | y_up | logit | FS0 | 0 | 0.67919 | -0.00159 | 0.0153 |
| E1_h20_y_top-hgb_clf_seed0 | 20 | y_top | hgb_clf | FS0 | 0 | 0.50091 | +0.00942 | 0.0068 |
| E1_h20_y_top-logit_seed0 | 20 | y_top | logit | FS0 | 0 | 0.50221 | -0.00708 | 0.0069 |
| E1_h20_y_up-hgb_clf_seed0 | 20 | y_up | hgb_clf | FS0 | 0 | 0.66912 | +0.00939 | 0.0128 |
| E1_h20_y_up-logit_seed0 | 20 | y_up | logit | FS0 | 0 | 0.67014 | +0.00671 | 0.0161 |
| E1_h60_y_top-hgb_clf_seed0 | 60 | y_top | hgb_clf | FS0 | 0 | 0.50190 | +0.03184 | 0.0156 |
| E1_h60_y_top-logit_seed0 | 60 | y_top | logit | FS0 | 0 | 0.50099 | +0.01268 | 0.0120 |
| E1_h60_y_up-hgb_clf_seed0 | 60 | y_up | hgb_clf | FS0 | 0 | 0.66312 | +0.02392 | 0.0235 |
| E1_h60_y_up-logit_seed0 | 60 | y_up | logit | FS0 | 0 | 0.66195 | +0.02157 | 0.0188 |
| E1_h120_y_top-hgb_clf_seed0 | 120 | y_top | hgb_clf | FS0 | 0 | 0.49836 | +0.03525 | 0.0213 |
| E1_h120_y_top-logit_seed0 | 120 | y_top | logit | FS0 | 0 | 0.49777 | +0.03609 | 0.0218 |
| E1_h120_y_up-hgb_clf_seed0 | 120 | y_up | hgb_clf | FS0 | 0 | 0.66148 | +0.13956 | 0.0372 |
| E1_h120_y_up-logit_seed0 | 120 | y_up | logit | FS0 | 0 | 0.65703 | +0.16348 | 0.0321 |

## Selection

rule: `05` §2 E1

- h5: y_up + hgb_clf
- h20: y_up + hgb_clf
- h60: y_up + logit
- h120: y_up + logit
