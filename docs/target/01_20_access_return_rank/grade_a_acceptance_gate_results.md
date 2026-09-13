# Grade A candidates — Phase 1 acceptance-gate walk-forward 결과

> 실행 모드: **FULL** / 모델: hgb
> baseline vs candidate(= baseline + px_reversal_5d/px_maxret_20d/px_idio_vol_60d/
> flow_individual_netbuy_to_volume_5d/_20d) / holdout 미사용(walk-forward valid만)

## target = y_rank_5d

| config | 피쳐(raw/design) | mean Rank IC | 경제성(grid top-decile spread) | turnover | cost-adj spread |
|---|---|---|---|---|---|
| baseline | 40/80 | **0.1155** | 0.0040 | 0.696 | -0.0002 |
| candidate | 45/90 | **0.1202** | 0.0048 | 0.682 | 0.0007 |

fold별 Rank IC:
- `baseline`: 0.111, 0.125, 0.118, 0.125, 0.099
- `candidate`: 0.119, 0.128, 0.122, 0.127, 0.106

- Δ mean Rank IC = 0.0048, Δ cost-adjusted spread = 0.0009

## target = y_rank_20d

| config | 피쳐(raw/design) | mean Rank IC | 경제성(grid top-decile spread) | turnover | cost-adj spread |
|---|---|---|---|---|---|
| baseline | 40/80 | **0.1436** | 0.0162 | 0.597 | 0.0126 |
| candidate | 45/90 | **0.1521** | 0.0135 | 0.571 | 0.0101 |

fold별 Rank IC:
- `baseline`: 0.126, 0.186, 0.143, 0.157, 0.105
- `candidate`: 0.153, 0.196, 0.146, 0.152, 0.114

- Δ mean Rank IC = 0.0084, Δ cost-adjusted spread = -0.0025

## target = y_rank_60d

| config | 피쳐(raw/design) | mean Rank IC | 경제성(grid top-decile spread) | turnover | cost-adj spread |
|---|---|---|---|---|---|
| baseline | 40/80 | **0.1753** | 0.0242 | 0.638 | 0.0204 |
| candidate | 45/90 | **0.1840** | 0.0247 | 0.656 | 0.0207 |

fold별 Rank IC:
- `baseline`: 0.179, 0.229, 0.168, 0.186, 0.115
- `candidate`: 0.194, 0.234, 0.170, 0.189, 0.132

- Δ mean Rank IC = 0.0087, Δ cost-adjusted spread = 0.0004

## 메타

- `baseline`/h=5d: design_feat=80, train=1252.3s, n_rebalances=280
- `baseline`/h=20d: design_feat=80, train=1139.8s, n_rebalances=70
- `baseline`/h=60d: design_feat=80, train=1149.9s, n_rebalances=24
- `candidate`/h=5d: design_feat=90, train=1274.4s, n_rebalances=280
- `candidate`/h=20d: design_feat=90, train=1249.2s, n_rebalances=70
- `candidate`/h=60d: design_feat=90, train=1238.8s, n_rebalances=24
