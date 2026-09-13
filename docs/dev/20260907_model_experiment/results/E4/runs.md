# E4 — runs

| run | h | label | model | FS | seed | prob | econ | ece |
|---|---|---|---|---|---|---|---|---|
| E4_h5_E4a-seed_seed1 | 5 | y_up | hgb_clf | FS0 | 1 | 0.67829 | +0.00038 | 0.0136 |
| E4_h5_E4a-seed_seed2 | 5 | y_up | hgb_clf | FS0 | 2 | 0.67828 | +0.00029 | 0.0134 |
| E4_h5_E4b-monotonic_seed0 | 5 | y_up | hgb_clf | FS0 | 0 | 0.67827 | +0.00021 | 0.0130 |
| E4_h20_E4a-seed_seed1 | 20 | y_up | hgb_clf | FS1h | 1 | 0.66875 | +0.01180 | 0.0131 |
| E4_h20_E4a-seed_seed2 | 20 | y_up | hgb_clf | FS1h | 2 | 0.66878 | +0.00865 | 0.0132 |
| E4_h20_E4b-monotonic_seed0 | 20 | y_up | hgb_clf | FS1h | 0 | 0.66887 | +0.01252 | 0.0138 |
| E4_h60_E4a-seed_seed1 | 60 | y_up | logit | FS0 | 1 | 0.66195 | +0.02157 | 0.0188 |
| E4_h60_E4a-seed_seed2 | 60 | y_up | logit | FS0 | 2 | 0.66195 | +0.02157 | 0.0188 |
| E4_h60_E4b-monotonic_seed0 | 60 | y_up | hgb_clf | FS0 | 0 | 0.66218 | +0.02190 | 0.0214 |

## Selection

rule: `05` §2 E4

- h5: seed std 0.00000, adopted config kept
- h20: seed std 0.00003, adopted config kept
- h60: seed std 0.00000, adopted config kept
