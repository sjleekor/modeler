# access_return_rank 모델 — 사용 피쳐 / 학습 성능 / 추가 활용 후보 정리

> 대상 모델: `research/models/_01_20_access_return_rank` (`model_id = 01_20_access_return_rank`)
> 예측 타깃: **20거래일 초과수익률(시장 등가중 대비)의 일자별 횡단면 순위 rank** (`y_rank_20d`)
> 기준 시점: 2026-06-22 / 검증 스냅샷 `snapshot_date=2026-06-19`
> 관련 문서: `prediction_target_20d_excess_return_rank.md`, `etl_00_*`, `etl_01_*`

본 문서는 (1) 현재 모델이 실제로 학습에 사용하는 피쳐 종류, (2) 실데이터로 3→4→5단계를
실행해 얻은 학습 성능 결과, (3) 이미 수집·적재되어 있으나 아직 학습에 쓰지 않는 추가 활용
후보 피쳐를 한 곳에 정리한다.

---

## 1. 현재 사용 중인 피쳐 (milestone A)

`ModelSpec.feature_groups = ("px", "flow")` — 가격(px)과 자금흐름(flow) 두 그룹만 사용한다.
패널 조립 단계에서 `dim_universe_daily`(유니버스 게이트)를 기준으로 `feat_price`, `feat_flow`,
`label_daily`를 as-of LEFT JOIN 한 뒤, fold별 train slice로 표준화(winsor/log/z-score)를 적합·적용해
`feat_panel_std.parquet`를 만든다.

- **원시 피쳐 컬럼 수: 30개** (`px_*` 15 + `flow_*` 15, 아래 표 기준)
- **학습 입력 컬럼 수: 60개** — 표준화 피쳐 30개 + 결측 플래그 `*_isna` 30개
  (`train.design_columns`가 키/라벨/fold 메타를 제외한 numeric 컬럼을 선택)

### 1.1 가격 그룹 `px_*` — 원천 `daily_ohlcv` (`research/etl/features/price.py`)

| 컬럼 | 의미 |
|------|------|
| `px_ret_1d`, `px_ret_5d`, `px_ret_20d`, `px_ret_60d` | 1/5/20/60거래일 로그수익률 |
| `px_mom_20_60` | 모멘텀 스프레드 (`ret_20d - ret_60d`) |
| `px_vol_20d`, `px_vol_60d` | 일간수익률의 20/60일 표준편차 (실현변동성) |
| `px_high_low_range_20d` | 20일 고저폭 / 종가 |
| `px_turnover`, `px_turnover_ma20` | 거래대금(종가×거래량) 및 20일 이동평균 |
| `px_amihud_20d` | Amihud 비유동성 (`mean(|ret_1d| / turnover)` 20일) |
| `px_gap_vs_ma20` | 종가 / 20일 이평 − 1 (이격도) |
| `px_dist_52w_high` | 종가 / 52주 최고가 − 1 |
| `px_is_halted` | 거래정지일 플래그 (`open=high=low=0`) |
| `px_halt_ratio_20d` | 최근 20일 거래정지 비율 |

가격은 레벨이 아니라 수익률/비율로 변환해 사용하며, 거래정지일의 종가 왜곡은
`px_is_halted` / `px_halt_ratio_20d`로 표시해 전처리 단계에서 마스킹 가능하다.

### 1.2 자금흐름 그룹 `flow_*` — 원천 `krx_security_flow_raw` (`research/etl/features/flow.py`)

원천 테이블(약 76M행)을 KRX 우선 dedup → 7개 metric pivot → 파생.

| 컬럼 | 의미 |
|------|------|
| `flow_foreign_netbuy_sum_5d`, `_20d` | 외국인 순매수량 5/20일 누적 |
| `flow_inst_netbuy_sum_5d`, `_20d` | 기관 순매수량 5/20일 누적 |
| `flow_indiv_netbuy_sum_5d`, `_20d` | 개인 순매수량 5/20일 누적 |
| `flow_foreign_holding_chg_5d`, `_20d` | 외국인 보유주식수 5/20일 변화(레벨 차분) |
| `flow_short_balance_chg_20d` | 공매도 잔고수량 20일 변화 (2016-06-30 이전 NULL) |
| `flow_foreign_netbuy_z_20d`, `flow_inst_netbuy_z_20d` | 외국인/기관 순매수량 20일 z-score |
| `flow_short_avg_price` | 공매도 평균가 (`short_value / short_volume`) |
| `flow_short_selling_volume`, `flow_short_selling_value`, `flow_short_balance_qty` | 패널 단계 비율 계산용 passthrough 레벨 |

순매수는 원시 주식수가 아니라 누적/z-score로 표준화해 사용하고, 공매도 잔고는
커버리지 비대칭(2016-06-30 시작)으로 인해 초기 구간 NULL → `*_isna`로 처리된다.

### 1.3 라벨 (`research/etl/labels.py`)

- 주 타깃: `y_rank_20d` — `(trade_date, market)` 내 20일 초과수익률의 백분위 순위 [0,1]
- 부수 출력: `y_reg_{h}d`(winsor 회귀값), `y_cls_{h}d`(0.2/0.8 임계 3-class), horizon 5/60일
- 실현값(평가용): `raw_label_20d` = `fwd_ret_20d - bench_ret_20d` (시장 등가중 벤치)
- 거래정지일을 제외한 per-ticker 거래일 인덱스로 t+H를 계산해 halt/휴일 갭을 흡수

---

## 2. 학습 성능 결과 (실데이터 실행)

스냅샷 `2026-06-19` 기준으로 3(parquet export)→4(ETL `build_dataset`)→5(`train_from_dataset`)
단계를 실제 실행해 얻은 결과.

| 항목 | 값 |
|------|-----|
| 패널 행 수 | 5,395,167 |
| fold 수 | 5 (walk-forward, embargo=20, purge=20, holdout_len=0) |
| 학습 기간 | 2015-01-02 ~ 2026-06-10 |
| 모델 | Ridge (alphas={0.1,1.0,10.0} 그리드) |
| 선택 하이퍼파라미터 | `alpha = 10.0` |
| 학습 입력 피쳐 수 | 60 (px/flow 30 + `*_isna` 30) |
| **평균 Rank IC** | **0.128** (5-fold) |
| fold별 Rank IC | 0.136 / 0.165 / 0.127 / 0.134 / 0.079 |

- 평가 지표(`research/etl/metrics.py`): `rank_ic_mean`, `rank_ic_std`, `icir`,
  `rank_ic_tstat`, `top_decile_spread`(상위 10% 예측의 실현 초과수익), `top_minus_bottom`,
  `hit_ratio_top`. 모델 선택은 평균 Rank IC 최대 기준.
- 해석: 평균 Rank IC ≈ 0.13은 선형 베이스라인(Ridge) + px/flow 30피쳐만으로도 일관된
  횡단면 예측력이 있음을 의미한다(fold 전반 양의 IC). 마지막 fold(0.079)에서 다소 하락.
- `holdout`은 `holdout_len=0`이라 None(정상). ElasticNet은 `TrainConfig(model="elasticnet")`로 선택 가능.

---

## 3. 수집되어 있으나 아직 학습에 쓰지 않는 추가 활용 후보

현재 spec은 px/flow만 사용하지만, ETL 피쳐 빌더와 원천 테이블은 아래 그룹을 이미 갖추고 있어
`feature_groups` 확장만으로 투입 가능하다. (각 그룹의 export 대상 테이블 export 여부를 사전 확인 필요)

### 3.1 fin 그룹 `fin_*` — point-in-time 재무 (`research/etl/features/fin_pit.py`)
- 원천: `stock_metric_fact` (canonical lake). 공시일 부재 → 보수적 PIT 래그(연 +90d / 분기 +45d).
- 피쳐: `fin_roa`, `fin_roe`, `fin_operating_margin`, `fin_debt_to_equity`, `fin_equity_ratio`,
  `fin_ocf_to_assets`, `fin_cash_ratio`, `fin_asset_turnover`, `fin_is_negative_equity`, `fin_has_fs`
- 특징: 레벨 금액이 아닌 비율/성장률, 자본잠식(`total_equity<=0`) 플래그·클리핑 처리.
- 활용 가치: 밸류/퀄리티 팩터 축을 추가 → px/flow의 모멘텀·수급 축과 상호보완 기대.

### 3.2 ev 그룹 `ev_*` — 기업행위 플래그 (`research/etl/features/event.py`)
- 원천: `dart_share_count_raw` (발행/자기주식). 1차 패스는 `se='합계'` 기반 저빈도 플래그만.
- 피쳐: `ev_treasury_ratio`(자사주/발행주식 = 자사주 매입 강도), `ev_has_treasury`,
  `ev_shares_chg_yoy`(발행주식수 YoY = 희석/매입).
- 활용 가치: 자사주·증자 신호는 수익률에 비대칭 영향. 단, 배당 피쳐
  (`dart_shareholder_return_raw`)는 se/stock_knd canonical 매핑 후로 보류 상태.

### 3.3 cf 그룹 `cf_*` — 시장/매크로 공통 (`research/etl/features/common.py`)
- 원천: `common_feature_daily_fact` (지수/시장폭/금리/환율/원자재/매크로). 날짜 단위 broadcast.
- 한계: 일별 히스토리가 **2025-12-15부터** 시작 → 2015+ 학습 패널에서는 대부분 NULL.
  현재로서는 "시장 레짐" 보조 피쳐 수준이며, 장기 백필(3차 milestone) 이후 본격 활용 가능.

### 3.4 기타 수집 중이나 미연결 원천
스냅샷에 적재되어 있으나 위 빌더에 아직 연결되지 않은 원천(추후 정규화·매핑 필요):
- `dart_shareholder_return_raw` (배당/주주환원) — se/stock_knd 정규화 후 배당수익률·배당성향 파생 가능
- `dart_financial_statement_raw`, `dart_xbrl_fact_raw` — 추가 재무 항목 확장 여지
- `common_feature_observation_raw` — cf 그룹의 원시 관측치(백필 소스)

---

## 4. 확장 시 점검 사항

- **그룹 추가 절차**: `ModelSpec.feature_groups`에 `fin`/`ev`/`cf` 추가 → `build_dataset`가 해당
  canonical/raw 테이블(`stock_metric_fact`, `dart_share_count_raw`, `common_feature_daily_fact`)을
  register & materialize. export 단계(`export_tables.toml`)에 해당 테이블 포함 여부 사전 확인.
- **결측 처리**: 신규 그룹은 초기 구간/커버리지 비대칭으로 NULL이 많으므로 `*_isna` 플래그가
  자동 추가됨(전처리 L1). cf는 NULL 비중이 매우 커서 단독 투입 시 효과 제한적.
- **검증 권장**: 그룹 확장 후 평균 Rank IC가 px/flow 베이스라인(≈0.128) 대비 개선되는지
  walk-forward로 비교. fold별 IC 안정성(특히 최근 fold)도 함께 확인.

## 5. 20260731 raw-feature 후보 검증 (px_reversal_5d 등 4종)

`px_amihud_20d`/`px_dist_52w_high`는 §1.1에 이미 있던 것과 별개로, 2026-07-31
raw-feature 연구 트랙(Phase A horizon-scan)이 screening한 6개 Grade A 후보 중
나머지 4개(`px_reversal_5d`, `px_maxret_20d`, `px_idio_vol_60d`,
`flow_individual_netbuy_to_volume_{5,20}d`)를 이 모델에 추가하는 acceptance
gate를 실행했다 — 결과와 판정은
[`docs/dev/20260731_raw_features/01_feature_candidate/07_phase1_acceptance_gate.md`](../../dev/20260731_raw_features/01_feature_candidate/07_phase1_acceptance_gate.md)
참고. 요지: walk-forward 증분성은 뚜렷하나(20일 Rank IC +0.0081, 대부분 fold
개선) 경제성(turnover 차감 top-decile spread)은 불명확 — 조건부 채택.
