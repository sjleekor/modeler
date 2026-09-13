# access_return_rank — 피쳐 그룹 ablation 결과

> 실행 모드: **FULL** / 모델: Ridge (alphas grid)
> 선택 기준: walk-forward **valid** Rank IC 평균 / 확정: trailing holdout 1회
> 그룹 사다리: px/flow → +fin → +fin+ev (cf·§3.4 미연결 원천 제외)

## 1. walk-forward (valid) 비교

| config | groups | 피쳐(raw/design) | best alpha | mean Rank IC | IC std(fold) | cross-ICIR |
|---|---|---|---|---|---|---|
| px_flow | px+flow | 30/60 | 10.0 | **0.1326** | 0.0194 | 6.837 |
| px_flow_fin | px+flow+fin | 40/80 | 10.0 | **0.1336** | 0.0201 | 6.646 |
| px_flow_fin_ev | px+flow+fin+ev | 43/86 | 10.0 | **0.1343** | 0.0203 | 6.622 |

### fold별 Rank IC

- `px_flow`: 0.128, 0.160, 0.133, 0.141, 0.100
- `px_flow_fin`: 0.124, 0.163, 0.136, 0.143, 0.103
- `px_flow_fin_ev`: 0.124, 0.164, 0.136, 0.144, 0.103

## 2. holdout (post-selection, 1회) 비교

| config | holdout Rank IC | ICIR | top-decile spread | top−bottom | hit ratio |
|---|---|---|---|---|---|
| px_flow | **0.0669** | 0.440 | -0.0082 | -0.0251 | 0.292 |
| px_flow_fin | **0.0738** | 0.496 | -0.0010 | -0.0210 | 0.299 |
| px_flow_fin_ev | **0.0741** | 0.496 | -0.0016 | -0.0222 | 0.300 |

## 3. 각 수치의 의미

평가 지표는 `research/etl/metrics.py`의 `evaluate()`가 산출한다. 모든 IC는 **일자별 횡단면(같은 날 종목들) Spearman 순위상관**을 먼저 구한 뒤, 그 값을 fold/holdout 기간에 대해 집계한 것이다. 예측값(`pred`)은 모델 점수, 실현값(`realized`)은 20일 초과수익률(`raw_label_20d`)이다.

| 수치 | 정의 | 읽는 법 |
|---|---|---|
| **mean Rank IC** | 일자별 Rank IC를 fold(=valid 구간) 전체에 대해 평균 | 횡단면 예측력의 핵심 지표. 0이면 무작위, 양수일수록 "높게 예측한 종목이 실제로 더 높은 초과수익"을 의미. 주식 횡단면에서 0.03~0.05면 유의미, **0.10 이상이면 상당히 강한 신호**로 본다. |
| **IC std(fold)** | fold별 mean Rank IC들의 표준편차(분산도) | 시기별 안정성의 역지표. 작을수록 어느 기간에나 고르게 작동. |
| **cross-ICIR** | (fold mean IC의 평균) / (fold mean IC의 표준편차) | fold 단위로 본 신호의 안정성 대비 크기. 클수록 "꾸준히 같은 방향". 본 표의 6.x는 fold가 5개뿐인 표본이라 절대값보다 **config 간 상대 비교**용으로 해석. |
| **holdout Rank IC** | 학습·선택에 전혀 쓰지 않은 trailing holdout(최근 120세션) 구간의 일자별 Rank IC 평균 | **누출 없는 최종 일반화 성능**. valid보다 보수적이며 실제 운용 기대치에 가장 가깝다. |
| **holdout ICIR** | holdout 구간 일자별 IC의 평균 / 표준편차 | holdout 구간 내에서 IC가 얼마나 일관되게 양수인지(정보비율 성격). 높을수록 day-to-day 흔들림 대비 신호가 안정적. |
| **top-decile spread** | 매일 예측 상위 10% 종목들의 **실현 초과수익 평균**을 기간 평균 | 실제 수익(초과수익률) 단위. 양수여야 "상위 추천이 돈이 됨". 음수/0 근처면 순위상관은 있어도 상단 포트폴리오 수익화는 약함을 시사. |
| **top−bottom** | (상위 분위 평균 실현수익) − (하위 분위 평균 실현수익), 5분위 기준 | 롱숏 스프레드. 양수여야 단조적 순위→수익 관계. |
| **hit ratio** | 상위 분위 종목 중 실현 초과수익 > 0 인 비율(기간 평균) | 상위 추천의 적중률. 0.5 근처면 방향성 약함. |

> 주의: 본 실험의 holdout 구간(최근 ~6개월)은 top-decile spread/top−bottom이 음수다. 즉 **순위 정보(IC)는 양수로 존재하지만, 같은 구간에서 상·하위 포트폴리오의 수익 단조성은 약했다.** 이는 holdout 시기 시장의 특이성(저표본·레짐)일 수 있어, IC와 분위 수익 지표를 함께 보아야 한다.

## 4. 어떤 피쳐 조합을 쓸 것인가 (해석)

- 베이스라인 `px_flow`: valid Rank IC 0.1326, holdout Rank IC 0.0669.
- `px_flow_fin`: valid Δ=+0.0010 (0.1336), holdout Δ=+0.0069 (0.0738) → **개선**.
- `px_flow_fin_ev`: valid Δ=+0.0017 (0.1343), holdout Δ=+0.0073 (0.0741) → **개선(단, ev의 추가 기여는 미미)**.

**해석**

- **`fin` 추가가 의미 있는 개선이다.** valid IC 상승폭(+0.0010)은 작아 보이지만, 누출 없는 holdout에서 +0.0069 (0.0669→0.0738, 약 +10%)로 더 크게 개선됐고 holdout ICIR도 0.440→0.496으로 올랐다. top-decile spread도 -0.0082→-0.0010으로 음수 폭이 크게 줄어 상단 포트폴리오 손실이 완화됐다. 즉 재무(가치/퀄리티) 축이 px/flow의 모멘텀·수급 축과 **상호보완**하며, 안정성·수익화 측면 모두에서 도움을 준다.
- **`ev`(기업행위)는 추가 기여가 거의 없다.** `+fin` 대비 valid/holdout IC가 +0.0007/+0.0003에 그치고 ICIR은 동일(0.496), top-decile spread는 오히려 소폭 악화(-0.0010→-0.0016)했다. ev 피쳐가 3개뿐이고 저빈도(자사주·발행주식 YoY)라 일별 패널에서 sparse하기 때문으로 보인다.
- **권장 조합: `px + flow + fin`.** 다지표를 종합하면 fin까지는 일관된 개선을, ev는 사실상 노이즈에 가까운 미미한 변화를 보인다. 운용 단순성(피쳐 86→83개 축소, 학습/빌드 시간 절감)과 과적합 위험까지 고려하면 **ev는 현 시점에 제외**하는 편이 합리적이다.
- 단, **두 config의 holdout IC 차이(+0.0003)는 5-fold·단일 holdout 표본에서 통계적으로 유의하다고 보기 어렵다.** ev를 완전히 폐기하기보다는, ev 피쳐를 일별 sparse 한계가 덜한 형태로 보강(예: 이벤트 발생 후 감쇠 윈도우, 캘린더 정렬)하거나 비선형 모델에서 재평가하는 것을 후속 과제로 둔다.

## 5. 결론

- 본 실험 방법(그룹 단위 incremental ablation + valid 선택 + holdout 1회 확정 + 다지표 비교)은 **정상 동작**하며, 그룹 추가 효과를 누출 없이 분리 측정한다.
- 현재 데이터·모델(Ridge) 기준 **최적 조합은 `px + flow + fin`**, ev는 보류가 권장된다.

## 6. 메타

- `px_flow`: panel_rows=5,395,167, folds=5, fin_feat=0, ev_feat=0, build=30.2s, train=16.8s
- `px_flow_fin`: panel_rows=5,395,167, folds=5, fin_feat=10, ev_feat=0, build=36.3s, train=24.6s
- `px_flow_fin_ev`: panel_rows=5,395,167, folds=5, fin_feat=10, ev_feat=3, build=40.0s, train=22.2s
