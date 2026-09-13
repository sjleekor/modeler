# Phase B candidates — validation-only acceptance gate

> AB run: `20260828T165038-4e0ae8b0` / config `889c3e8377c2f400907611f7402651eee6a23c2765c051e4eb2a4a59ca36cbea`
> 후보 선택: AB `screen_pass`가 하나 이상인 family의 primary feature를 전부 함께 추가
> holdout: 미사용. h60 라벨이 성숙하는 2026년 10~11월 이후 한 번만 평가

후보 family 14개, feature 14개:

`ev_amendment_ratio_1y`, `ev_filing_burst_60d`, `ev_net_share_issuance_yoy`, `ev_payout_yield`, `fin_gross_profitability`, `fin_log_mcap`, `fin_value_z`, `hc_employee_growth_yoy`, `hc_revenue_per_employee`, `mcap_krx_log`, `own_amendment_ratio_1y`, `own_major_filing_60d`, `own_major_stake`, `own_major_stake_chg`

## h=5

| config | raw/design | mean Rank IC | cost-adjusted spread |
|---|---:|---:|---:|
| baseline | 40/80 | 0.1155 | -0.0002 |
| phase_b_candidate | 54/108 | 0.1186 | 0.0015 |

- candidate − baseline: Rank IC **0.0031**, cost-adjusted spread **0.0017**

## h=20

| config | raw/design | mean Rank IC | cost-adjusted spread |
|---|---:|---:|---:|
| baseline | 40/80 | 0.1436 | 0.0126 |
| phase_b_candidate | 54/108 | 0.1447 | 0.0155 |

- candidate − baseline: Rank IC **0.0011**, cost-adjusted spread **0.0030**

## h=60

| config | raw/design | mean Rank IC | cost-adjusted spread |
|---|---:|---:|---:|
| baseline | 40/80 | 0.1753 | 0.0204 |
| phase_b_candidate | 54/108 | 0.1755 | 0.0283 |

- candidate − baseline: Rank IC **0.0003**, cost-adjusted spread **0.0080**

## 판정 범위

valid 구간에서는 세 horizon의 Rank IC와 비용 반영 spread가 모두 개선됐습니다. 따라서
validation 단계 결과는 `improved_all_horizons`입니다.

이 결과는 purged walk-forward valid 구간의 증분성과 경제성만 판정합니다. 최종 채택은 h60까지 값이 있는 새 holdout을 한 번 평가한 뒤 정합니다.
