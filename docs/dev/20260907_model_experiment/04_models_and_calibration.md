# 04. 모델과 확률 보정

- 작성일: 2026-09-07
- 의존성: `scikit-learn>=1.4`(`pyproject.toml` `analysis` extra). **LightGBM·XGBoost는 넣지 않는다** — sklearn HGB가 같은 계열이고, 새 의존성은 결과의 재현 조건을 늘린다. 필요하면 E5 뒤 별도 결정.
- 기존 코드: `research/models/_01_20_access_return_rank/train.py`(`TrainConfig`, `_make_model`, `walk_forward`)

---

## 1. 모델 세 가지

| id | 모델 | 타깃 | 출력 | 역할 |
|---|---|---|---|---|
| **M-A** | `HistGradientBoostingRegressor` (기존 `_01` `hgb`) + isotonic | `y_rank_{h}d` | 순위 점수 → isotonic으로 p | **연결 기준선.** 기존 모델을 확률로 읽으면 어디까지 가는가 |
| **M-B** | `HistGradientBoostingClassifier(loss="log_loss")` | `y_up_{h}d` / `y_top_{h}d` | `predict_proba[:,1]` | **주 모델** |
| **M-C** | `LogisticRegression(penalty="l2")` | 같음 | `predict_proba[:,1]` | 대조. `rank` 피쳐 위의 선형 확률 모델. 해석 가능, 보정이 구조적으로 좋다 |

세 모델 모두 입력은 `rank` 프로파일(`02` §4)의 design 행렬(피쳐 + `*_isna`)이다. M-A만 기존 `tree` 프로파일과 `rank` 프로파일 둘로 돌려 기존 결과와의 연결을 확인한다(E0).

---

## 2. 하이퍼파라미터 — 고정 grid

grid는 아래에 적힌 것만 돈다. 선택 기준은 fold 평균 **valid log-loss**(M-A는 Rank IC, 기존과 같음).

### 2.1 M-B HGB 분류기

| 파라미터 | 값 | 비고 |
|---|---|---|
| `max_iter` | {200, 400} | early stopping 없음(결정성) |
| `learning_rate` | {0.03, 0.1} | |
| `max_leaf_nodes` | {15, 31} | |
| `l2_regularization` | {0.0, 1.0} | |
| `min_samples_leaf` | 200 | 고정. 날짜당 약 1,000종목 × 2,600일 패널에서 잎이 종목 몇 개에 매달리지 않게 |
| `max_bins` | 255 | 기본 |
| `class_weight` | None | **보정을 깨므로 쓰지 않는다.** L-B의 불균형은 지표(precision@k)로 읽는다 |
| `categorical_features` | None | 국면 이진은 수치로 둔다 |
| `random_state` | {0, 1, 2} | E4에서만 3개, 그 전은 0 |

16개 조합. 5-fold × 16 = 80 fit/run. `_01`의 h20 게이트가 fold 5개 × grid 4개에 약 20분이었으므로 grid 16이면 약 1시간이다. **E1에서는 grid를 절반(`max_iter` 400, `l2` 0 고정 → 4개)으로 줄이고, E2 이후 채택 후보에만 16개 grid를 쓴다.**

### 2.2 M-C 로지스틱

| 파라미터 | 값 |
|---|---|
| `C` | {0.1, 1.0, 10.0} |
| `solver` | `lbfgs`, `max_iter` 500 |
| 입력 | `rank` 피쳐(0~1)라 추가 표준화 없음. `*_isna`는 0/1 |

3개 조합. 수 분이면 끝난다.

### 2.3 M-A 기준선

`_01.train.TrainConfig(model="hgb", max_iters=(100, 200), learning_rates=(0.01, 0.1))` 그대로. 순위 점수를 §3의 isotonic으로 p에 매핑한다.

### 2.4 단조 제약 변형 (E4b)

HGB는 `monotonic_cst`로 피쳐별 단조 방향을 강제할 수 있다. `02` §1.2의 기대 부호를 그대로 제약으로 넣은 변형을 E4b에서 하나 돈다. 사전등록 부호를 모델 구조에 넣는 것이라 과적합이 줄고 해석이 쉬워지지만, 반대 부호가 진짜라면(모멘텀 계열) 성능이 떨어진다. 부호가 미고정인 피쳐(`hc_*`, `own_*` 관측 부호)는 제약하지 않는다.

---

## 3. 확률 보정

### 3.1 왜 필요한가

- M-A는 순위 점수라 확률이 아니다. 매핑이 필수다.
- M-B는 log-loss로 학습해 대체로 보정되지만, 부스팅 반복이 많으면 과신(overconfident)으로 기운다.
- M-C는 로지스틱이라 구조적으로 보정에 가깝다.

### 3.2 방법

| 방법 | 어디에 | 적합 데이터 |
|---|---|---|
| **isotonic regression** (sklearn `IsotonicRegression(out_of_bounds="clip")`) | M-A 필수, M-B·M-C 선택 | fold train 슬라이스의 **뒤쪽 20% 세션**(embargo h 두고)에서 fit. valid에는 적용만 |
| Platt (로지스틱 1변수) | 비교용 | 같음 |
| 없음 | M-B·M-C 기본 | — |

- **valid 데이터로 보정을 fit하지 않는다.** valid는 평가에만 쓴다.
- 보정은 h·모델·fold마다 따로 fit한다.
- L-B(양성 0.2)에서 isotonic은 상위 구간 표본이 적어 계단이 거칠다. 구간별 `n`을 reliability 표에 남긴다.

### 3.3 판정에서의 자리

- M-B·M-C는 **보정 없는 p를 주 결과**로 하고, 보정 후 p를 보조로 기록한다. ECE > 0.03이면 보정 후 값으로 다시 판정한다(`03` §3.3).
- M-A는 보정 후 p만 있다.

---

## 4. 학습 절차 (fold 하나)

```
train slice (dates ≤ T_k)
  ├─ 전처리 fit (rank 프로파일은 무상태) → design 행렬
  ├─ [보정 fit용] 뒤쪽 20% 세션을 떼어 embargo h 뒤 → cal slice
  ├─ 모델 fit (grid마다) on train − cal
  ├─ cal slice에 predict → isotonic fit
valid slice (T_k + h + 1 ~)
  ├─ 전처리 transform → predict → p_raw, p_cal
  └─ 지표 (03 §2) 기록
grid 선택: fold 평균 valid log-loss 최소 (M-A는 Rank IC 최대)
```

- 학습 표본은 날짜×종목 행을 pooled로 쓴다. 날짜 가중은 하지 않는다(날짜당 종목 수가 900~1,100으로 비슷하다).
- `y`가 NULL인 행(라벨 미성숙)은 학습·평가에서 뺀다. purge가 이미 대부분 처리한다.
- 예측 p는 [1e-6, 1−1e-6]로 클립해 log-loss가 발산하지 않게 한다.

---

## 5. h 사이의 관계

- **h마다 독립 모델.** 멀티태스크나 공유 표현은 하지 않는다. h별 피쳐 집합(FS1h)이 다를 수 있어서다.
- 진단으로 h 간 p의 상관(같은 날 같은 종목의 p5·p20·p60·p120)을 기록한다. 상관이 0.9 이상이면 h를 나눈 의미가 적다는 신호다.
- 운용 규칙(어느 h의 p를 쓸지)은 이 실험 범위 밖이다. 결과 문서가 h별 후보를 내고, 조합은 사용자가 정한다.

---

## 6. 시드와 결정성

- 전처리·분할·라벨은 결정적이다. HGB만 `random_state`에 의존한다(`max_bins` 구간 표본 추출).
- E0~E3는 시드 0 하나. E4에서 채택 후보를 시드 {0,1,2}로 다시 돌려 평균·표준편차를 기록한다.
- 결과의 byte-level 재현은 요구하지 않는다. 지표의 소수 4자리 재현을 요구한다.

---

## 7. 하지 않는 것

- 신경망·시퀀스 모델. 표본 10.5년·피쳐 68개에서 근거가 없다.
- 스태킹·앙상블(시드 평균은 예외).
- 손실 함수 커스텀(top-k 직접 최적화). `improvement_proposals.md` §3.2가 제안했으나 이번 범위 밖이다. L-B 라벨이 그 목적을 부분적으로 대신한다.
- 온라인 재학습·롤링 재적합 빈도 최적화. walk-forward fold 5개가 곧 재적합 시점이다.
