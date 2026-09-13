# 02. 종목 간 관계 피쳐 — F-2 (통계적 peer, 지금) · F-3 (업종 기반, F-1 뒤)

- 작성일: 2026-09-07
- 근거: `11_feature_taxonomy.md` §7(관계 축 C5 = 0, 두 갈래: 상대 위치·리드래그), §7.2 후보표. 문헌: Moskowitz-Grinblatt 1999(산업 모멘텀), Hou 2007(대형주→소형주 리드래그), Menzly-Ozbas 2010, Cohen-Frazzini 2008.
- 원천: `daily_ohlcv`(수익률), `daily_market_cap`(시총·거래대금), `krx_security_flow_raw`(수급), `dim_industry_pit_daily`(F-1). **F-2는 새 수집이 없다.**

---

## 1. F-2 — 통계적 peer 피쳐 (`feat_relation_stat`)

### 1.1 peer 집합 정의 — 월 1회 재추정, 사용은 다음 달

```
t0 = 각 달의 마지막 KRX 세션
입력: t0 이전 252 valid session의 시장모형 잔차 수익률 resid_ret (price.py의 residuals CTE와 같은 정의)
      — 종목 i는 그 252세션 중 유효 잔차가 189(75%) 이상일 때만 후보
상관: ρ_ij = corr(resid_i, resid_j), 두 종목이 함께 유효한 세션 ≥ 189
peer(i, t0) = ρ_ij 상위 K=20 종목 (j ≠ i, 같은 시장 제한 없음)
사용 구간: t0 다음 세션부터 다음 달 t0'까지 (peer 집합 고정)
```

- **잔차**로 재는 이유: 원수익률 상관은 시장 공통 요인에 지배돼 모든 종목이 서로 peer가 된다. 잔차 상관이 업종·스타일 공통 움직임을 본다.
- **K=20 고정.** 상관 임계값 방식(ρ ≥ c)은 시장 국면에 따라 집합 크기가 널뛰어 피쳐 분포가 흔들린다. K는 사전등록값이고 바꾸지 않는다(다른 K는 진단 컬럼으로만, §1.4).
- **PIT**: t0까지의 수익률만 쓰고 t0 이후에만 적용한다. 자연히 look-ahead가 없다.
- 규모: 종목 약 2,800 → 상관 행렬 2,800² ≈ 7.8M/월, 2015~2025 약 130개월. numpy로 월당 수 초. peer 테이블 `(month_end, ticker, market, peer_ticker, peer_market, rho, rank)` 약 2,800 × 20 × 130 ≈ 7.3M행.

### 1.2 피쳐 정의 (일별)

| 컬럼 | 정의 | 축 | 기대 부호 | 사전등록 horizon |
|---|---|---|---|---|
| `rel_peer_mom_20d` | peer 20개의 최근 20세션 로그수익률 합의 **동일가중 평균**(자기 제외) | 리드래그·집계 모멘텀 | **양방향**(개별 모멘텀이 한국에서 반대. 집계는 문헌상 `+`이나 재현 근거 없음) | [20, 40, 60] |
| `rel_peer_mom_60d` | 같은 것, 60세션 | | 양방향 | [20, 40, 60] |
| `rel_own_minus_peer_20d` | 자기 20세션 수익률 − `rel_peer_mom_20d` | 상대 위치(peer 대비 과열·소외) | **`−`**(peer 대비 앞선 종목은 되돌아온다 — 반전 가설) | [5, 10, 20] |
| `rel_peer_bigcap_lag_ret_5d` | peer 20개 중 시총 상위 5개의 **직전 5세션** 수익률 평균(`daily_market_cap` 기준) | 대형주→소형주 리드래그(Hou 2007) | **`+`** | [5, 10, 20] |
| `rel_peer_dispersion_20d` | peer 20개의 20세션 수익률 표준편차 | 집단 내 이견·불확실성 | 양방향 | [20, 40, 60] |
| `rel_peer_corr_mean` | 자기와 peer 20개의 ρ 평균(연결 강도) | 진단·조건 변수 | — (조건화용) | 검정 안 함 |

- 값은 마트에서 표준화하지 않는다(모델이 `rank`로 변환). rank IC는 척도 불변이다.
- 자기 제외·동일가중은 고정. 시총가중 variant는 만들지 않는다(정의 늘리기 금지).
- NULL: peer 집합이 없는 종목(상장 1년 미만, 유효 세션 부족), peer 중 그 날 유효 세션이 15개 미만이면 NULL.

### 1.3 계산 설계

- `research/etl/features/relation_stat.py`. 두 단계: (a) `dim_peer_monthly`(peer 테이블, numpy 상관 → parquet), (b) `feat_relation_stat`(DuckDB join: 날짜 × 자기 × peer 20 → 집계). (b)는 행 수 7M × 20 = 140M 중간 조인이라 월별 파티션으로 나눠 materialize한다.
- 입력 `resid_ret`은 `price.py`의 `residuals` CTE를 함수로 꺼내 공유한다(`macro_exposure.py`가 이미 같은 방식으로 재사용한다). **`feat_price`는 건드리지 않는다**(A0 계약 해시).
- `FORMULA_VERSION = "relation_stat_v1"`. K·창·유효 비율·잔차 정의를 버전에 묶는다.

### 1.4 진단 컬럼 (검정하지 않음)

- `rel_peer_mom_20d_k10`, `_k40`: K 민감도. 결과 문서에 상관만 적는다.
- `rel_peer_ksic_agree`: peer 20개 중 자기와 KSIC 2자리가 같은 비율(`dim_industry_pit_daily`의 소급값 사용, 진단이라 허용). 통계적 peer가 업종을 얼마나 재현하는지.

### 1.5 검증할 것 (마트 단위, 검정 전)

- PIT: peer 집합 결정에 t0 이후 수익률이 들어가지 않음(테스트: t0 이후 행을 바꿔도 peer 불변).
- 커버리지: 날짜당 값 있는 종목 비율(2016년 이후 0.8 이상 기대).
- 분포: `rel_own_minus_peer_20d`의 날짜별 중앙값 ≈ 0(정의상), 꼬리.
- 자기 상관: `rel_peer_mom_20d`의 lag1 자기상관(회전율 감각).
- 규모 축 중복: `px_amihud_20d`·`fin_log_mcap`과의 일별 순위상관(|ρ| ≥ 0.5면 카드에 경고).

---

## 2. F-3 — 업종 기반 관계 피쳐 (`feat_relation_ind`, `_ind` variant)

F-1의 D-F1 판정(소급 업종 허용 여부) 뒤에 만든다. 정의는 지금 고정한다.

### 2.1 피쳐

| 컬럼 | 정의 | 기대 부호 | horizon |
|---|---|---|---|
| `rel_ind_mom_20d` | 같은 `ind_group`(자기 제외) 20세션 수익률의 **시총가중** 평균 | 양방향 | [20, 40, 60] |
| `rel_mom_ex_ind_12_1` | `px_mom_12_1` − 같은 그룹의 12-1개월 수익률 평균(업종 중립 모멘텀) | 양방향(개별 모멘텀 반대 부호를 업종 효과가 설명하는지) | [20, 40, 60, 120] |
| `rel_ind_bigcap_lag_ret_5d` | 그룹 시총 상위 5개의 직전 5세션 수익률 평균 | `+` | [5, 10, 20] |
| **`rel_ind_flow_foreign_20d`** | 그룹 단위 외국인 20세션 순매수 합 / 그룹 20세션 거래량 합 | 양방향(개별은 반대 부호) | [5, 10, 20] |
| `rel_ind_flow_inst_20d` | 같은 것, 기관 | 양방향 | [5, 10, 20] |
| `rel_ind_dispersion_20d` | 그룹 내 20세션 수익률 표준편차 | 양방향 | [20, 40, 60] |
| `ind_group_id` | 그룹 코드(범주형) | 모델 더미용, 검정 안 함 | — |

- **`rel_ind_flow_*`가 실용적으로 가장 기대된다**(`11` §7.2). 개별 수급 8개 중 개인만 정방향이었는데, 외국인·기관의 업종 단위 쏠림은 개별 순매수보다 선행할 수 있다. 원천 `krx_security_flow_raw`는 있다.
- 업종 중립 variant(`fin_value_z_ind`, `fin_gross_profitability_ind`, `fin_accruals_to_assets_ind` 등)는 `feat_fin_scan_daily_ind`가 이미 만든다. D-F1이 허용하면 **같은 컬럼을 `ind_is_backcast` 표지와 함께 family로 등록**한다. 정의는 `fin_scan.py`의 `industry_view` 경로 그대로.
- 그룹은 `dim_industry_pit_daily.ind_group`(`industry_groups.py` fold-up, 20 미만 병합). 날짜별 유효 구성원 20 미만 그룹은 NULL(N2 V6에서 그런 그룹이 69.4%였다 — 커버리지가 크게 줄 수 있다. 그것도 결과다).

### 2.2 PIT 표지

- `ind_is_backcast=true` 구간의 값은 마트에 있되, Horizon Scan family 정의에서 `require_pit: true`이면 NULL 처리한다. D-F1이 "허용"이면 `require_pit: false`로 등록하고 카드에 누수 크기를 적는다.
- 모델 FS3에는 `ind_is_backcast`를 이진 피쳐로 같이 넘긴다.

---

## 3. 사전등록 초안 (F-HS에 합칠 것)

| family | primary | secondary | fdr_family | 부호 | primary horizon | 국면 쌍 후보(Phase C) |
|---|---|---|---|---|---|---|
| `rel_peer_mom` | `rel_peer_mom_20d` | `rel_peer_mom_60d` | `relation` | 양방향 | [20, 40, 60] | × `liq_high`, × `market_up` |
| `rel_own_minus_peer` | `rel_own_minus_peer_20d` | — | `relation` | `−` | [5, 10, 20] | × `vix_high` |
| `rel_peer_bigcap_lag` | `rel_peer_bigcap_lag_ret_5d` | — | `relation` | `+` | [5, 10, 20] | × `liq_high` |
| `rel_peer_dispersion` | `rel_peer_dispersion_20d` | — | `relation` | 양방향 | [20, 40, 60] | — |
| (F-3) `rel_ind_mom` | `rel_ind_mom_20d` | — | `relation_ind` | 양방향 | [20, 40, 60] | × `market_up` |
| (F-3) `rel_mom_ex_ind` | `rel_mom_ex_ind_12_1` | — | `relation_ind` | 양방향 | [20, 40, 60, 120] | — |
| (F-3) `rel_ind_bigcap_lag` | `rel_ind_bigcap_lag_ret_5d` | — | `relation_ind` | `+` | [5, 10, 20] | — |
| (F-3) `rel_ind_flow_foreign` | `rel_ind_flow_foreign_20d` | `rel_ind_flow_inst_20d` | `relation_ind` | 양방향 | [5, 10, 20] | × `liq_high` |
| (F-3) `rel_ind_dispersion` | `rel_ind_dispersion_20d` | — | `relation_ind` | 양방향 | [20, 40, 60] | — |

- `fdr_family`를 `relation`으로 따로 둔다. 결합 BH 모집단이 커지므로(153/177 → +16~36) 기존 discovery의 q가 움직인다 — 지금까지 판정 변화는 0이었지만 매번 확인한다.
- 양방향 등록이 많다. `12`의 규칙대로 **관측 부호를 첫 run 뒤 고정**하고 이후 바꾸지 않는다.
- 국면 쌍은 Phase C 2라운드(`05` §5)에 넣는다. 유동성 국면이 수급·회전율에서 통했으므로 `rel_ind_flow_* × liq_high`, `rel_peer_bigcap_lag × liq_high`가 첫 후보다.

---

## 4. 모델에 넘기는 형태 (FS3)

- F-HS에서 `screen_pass`한 primary 컬럼 + `rel_peer_corr_mean`(조건 변수, 검정 없이 넘김) + `ind_group_id`(더미, D-F1 허용 시) + `ind_is_backcast`.
- 마트 두 개(`feat_relation_stat`, `feat_relation_ind`)를 모델 `GROUP_VIEW`에 `rel`·`ind`로 추가하는 것으로 끝난다.
- 관계 피쳐는 **같은 날 다른 종목의 값**을 쓴다. 학습 패널에서 종목 행 사이의 독립성이 더 약해진다. 모델 스트림 `03` §1의 fold 경계(날짜 기준 purge)는 이것을 이미 다룬다 — 종목을 섞는 K-fold를 쓰지 않는 이유가 여기서 하나 더 생긴다.

---

## 5. 하지 않는 것

- 공급망(customer-supplier) 관계: 사업보고서 텍스트 파싱 필요(`09_w4_filing_text.md` §6). N9 착수 조건 미충족.
- 지분·계열 관계 그래프: DART 지분공시 원문(DS004 `majorstock`) 미수집. `dart_filing_receipt_raw`의 5% 공시 건수만 있다.
- 애널리스트 공유 네트워크: 원천 없음.
- peer 정의 여러 개(K·창·상관 종류)를 만들어 고르기.
