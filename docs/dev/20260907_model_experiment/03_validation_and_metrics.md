# 03. 검증 설계와 지표

- 작성일: 2026-09-07
- 정본 코드: `research/etl/splits.py`(`walk_forward_splits`), `research/etl/metrics.py`(`evaluate`, `economic_report`, `topk_economic_report`, `rebalance_grid`)
- 기존 게이트: `research/models/_01_20_access_return_rank/experiments/run_grade_a_acceptance_gate.py`, `run_phase_b_acceptance_gate.py`, `run_topk_cost_check.py`

---

## 1. 분할 — purged walk-forward

```
walk_forward_splits(dates, horizon=h, embargo=h, purge=h, n_folds=5, holdout_len=0)
dates = 선택 구간의 KRX 세션 (formation 2015-01-02 ~ 2025-07-31)
```

- **expanding window 5-fold.** fold k는 시작부터 `T_k`까지 학습, `T_k + h + 1`부터 valid. train 끝 h세션의 라벨은 valid를 침범하므로 purge.
- **h마다 fold 경계가 다르다.** `embargo = purge = h`라 h=120은 h=5보다 valid 시작이 늦다. h 간 비교는 하지 않는다 — h마다 "baseline 대비 개선"만 본다.
- **holdout_len = 0.** `_01`의 spec은 `holdout_len=120`으로 뒤쪽 120세션을 남겼는데, 이번에는 `period_end = 2025-07-31`로 잘라 formation 2025-08-01 이후 전체가 holdout이다. 코드에서 `period_end`를 넘기지 않으면 실패하도록 spec 기본값을 박는다.
- fold별 train 슬라이스에서 전처리 통계량을 fit한다. `rank` 프로파일은 fit할 것이 없다.

**확인할 것 (E0 전).** `run_phase_b_acceptance_gate.py`·`run_grade_a_acceptance_gate.py`의 walk-forward `period_end`가 2025-08-01 이후를 포함했는지 확인한다. 포함했다면 T1·T2 valid 결과는 Horizon Scan holdout 구간을 일부 읽은 것이고, 이번 모델은 그 구간을 쓰지 않으므로 baseline 숫자가 그 문서들과 다르게 나온다. E0가 그 차이를 기록한다.

---

## 2. 지표

### 2.1 확률 지표 (신규, `metrics.classification_report`)

| 지표 | 정의 | 읽기 |
|---|---|---|
| **log-loss** | `−mean(y·ln p + (1−y)·ln(1−p))`, p는 [1e-6, 1−1e-6]로 클립 | **주 지표.** L-A 기준선(p=날짜별 양성 비율)은 약 0.693 |
| Brier | `mean((p − y)²)` | 기준선 L-A 약 0.25, L-B 0.16 |
| AUC | 날짜별 AUC의 평균(날짜×시장 pooled AUC도 병기) | 순위 능력. 0.5 기준 |
| **ECE** | 10 등분 확률 구간에서 `Σ (n_b/N)·|mean(p_b) − mean(y_b)|` | 보정. 0.02 이하면 실용적으로 보정됨 |
| reliability 표 | 구간별 `mean(p)`, `mean(y)`, `n` | 결과 문서에 그림으로 |
| precision@k | 날짜별 상위 k(=100)의 양성 비율 평균 | L-B에서 특히 |
| lift@k | precision@k ÷ 날짜별 양성 비율 | 1.0이 무정보 |

모두 fold별 valid에서 계산하고, 5-fold 평균과 표준편차를 기록한다.

### 2.2 순위 지표 (기존 `evaluate`)

`pred_col = p`, `realized_col = raw_label_{h}d`로 `RankICReport`를 그대로 낸다: `rank_ic_mean`, `rank_ic_std`, `icir`, `rank_ic_tstat`, `top_decile_spread`, `top_minus_bottom`, `hit_ratio_top`. 확률 모델도 순위로 읽을 수 있어야 기존 모델과 이어진다.

### 2.3 경제성 지표 (기존 + 임계 포트폴리오)

| 포트폴리오 | 규칙 | 코드 |
|---|---|---|
| top-decile | 비중첩 격자(`rebalance_grid(dates, h)`)마다 상위 10% 매수, h일 보유 | `economic_report` |
| **top-k** | 격자마다 p 상위 k=100 매수, h일 보유, 왕복 60bp | `topk_economic_report(k=100, cost_bps_roundtrip=60)` |
| **임계 τ**(신규) | 격자마다 `p ≥ τ`(τ=0.6) 전부 매수, 동일가중. 0개면 현금 | `threshold_economic_report` |

- 기록: `grid_topk_mean_return`(격자 평균 초과수익), `turnover`, `cost_adjusted_return`, 보유 종목 수 평균·최소, 격자 수(`n_rebalances`).
- **비용 반영 수익 = 격자 평균 초과수익 − turnover × 60bp.** 기존 게이트와 같은 정의다.
- 절대 수익(`fwd_ret`) 기준 성과와 MDD는 진단으로만 낸다.

### 2.4 안정성

- fold별 값과 마지막 fold(최근) 값. 기존 모델은 마지막 fold가 가장 낮았다(h20 0.105).
- 연도별 log-loss·Rank IC.
- 시장별(KOSPI/KOSDAQ).

---

## 3. h별 주 지표와 게이트 — 사전등록

### 3.1 원칙

h마다 **주 지표 둘**(확률 하나 + 경제성 하나)을 고정하고, 둘 다 baseline보다 나아야 "개선"이다. 나머지는 보조다. 여러 지표 중 좋은 것을 고르지 않는다.

### 3.2 표

| h | 확률 주 지표 | 경제성 주 지표 | 보조 | 이유 |
|---|---|---|---|---|
| 5 | valid log-loss | **top-k 비용 반영 수익**(k=100, 60bp) | turnover, precision@k, Rank IC | 연 50회 회전이라 비용이 판정을 뒤집는다(T1 교훈) |
| 20 | valid log-loss | top-k 비용 반영 수익 | ECE, Rank IC, top-decile spread | 기존 배포 대상. 기존 게이트와 같은 잣대 |
| 60 | valid log-loss | top-k 비용 반영 수익 | ECE, Rank IC, 기간 일관성 | 격자 24개라 표본이 적다. fold 표준편차를 같이 본다 |
| 120 | valid log-loss | **Rank IC**(격자 약 22개로 top-k 수익의 분산이 너무 크다) | ECE, top-decile spread, 연도별 부호 | 비중첩 격자가 적어 경제성은 진단 |

### 3.3 "개선" 판정 규칙

비교는 항상 **같은 h, 같은 fold 경계, 같은 유니버스**의 두 config 사이에서만 한다.

- **개선**: 확률 주 지표가 좋아지고(log-loss Δ < 0) **그리고** 경제성 주 지표가 나쁘지 않다(Δ ≥ 0). 5-fold 중 4 이상에서 log-loss가 좋아야 한다.
- **보정 조건**: 채택 후보는 ECE ≤ 0.03. 넘으면 isotonic 보정 뒤 다시 본다(`04` §3). 보정 뒤에도 넘으면 후보에서 뺀다.
- **무개선**: 둘 중 하나가 나빠지면 무개선. 이유를 적고 다음 단계로 넘어간다. 후보 조합을 바꿔 다시 돌리지 않는다.
- Δ의 크기 문턱은 두지 않는다. 대신 fold 표준편차를 병기해 "개선폭 < fold 표준편차"이면 **약한 개선**으로 표기한다.

### 3.4 E 단계 간 선택 규칙

`05` §2에 있다. 요지: E1에서 h마다 라벨(L-A/L-B)과 모델(HGB/logit)을 **valid log-loss 최소**로 고르고, 그 뒤 단계는 고른 조합만 돈다.

---

## 4. 다중 비교

E0~E4 합계 약 52 run, h당 약 13 run이다. 비교 수를 줄이는 장치는 셋이다.

1. **단계별 선택.** 매트릭스 전체를 돌려 최고를 고르는 대신, 단계마다 하나의 질문(라벨? 모델? 피쳐 집합? 시점?)만 답하고 다음 단계는 답을 고정한다.
2. **주 지표 둘 고정.** 지표 선택의 자유도가 없다.
3. **시드 3개.** HGB의 최종 후보는 시드 3개 평균으로 보고하고, 시드 간 표준편차보다 작은 개선은 약한 개선이다.

BH 같은 형식적 보정은 하지 않는다. 대신 **선택 구간에서 고른 뒤 holdout 한 번**이 최종 보정이다.

---

## 5. holdout — 한 번

| 항목 | 값 |
|---|---|
| 경계 | formation `2025-08-01` 이후 (Horizon Scan `sample.holdout_start`와 같다) |
| 열 때 | **2026-10~11**, T1·T2 h60 one-shot holdout과 같은 날. 새 snapshot으로 마트를 다시 만든다 |
| 대상 | E5까지 끝난 뒤 h마다 채택 후보 **하나**(라벨·모델·피쳐 집합·보정 방법이 전부 고정된 것) + baseline FS0 |
| 라벨 성숙 | h5·h20·h60은 2026-10에 약 12개월치 formation이 성숙. h120은 약 9개월치 |
| 판정 | 선택 구간과 같은 주 지표 둘. 기준은 **미리 적은 방향**(개선 여부)만이고 크기 문턱은 없다 |
| 재사용 금지 | 이 구간을 연 뒤에는 어떤 선택도 다시 하지 않는다. 다음 holdout은 그 뒤 새로 쌓인 구간이다 |

기존 `_01` 모델은 뒤쪽 holdout을 두 번(fin/ev 선택, Ridge→HGB) 열고 새 창(2026-06-11~07-31)을 T1에 한 번 더 열었다. 그 세 번의 경험이 이 규칙의 근거다.

---

## 6. 기록 항목 (run마다)

`results/<E>/<run_id>/` 아래.

| 파일 | 내용 |
|---|---|
| `run_spec.json` | model_id, h, 라벨, FS id, 모델·grid, 시드, 전처리 프로파일, flow variant, period, snapshot, git sha, 코드 hash |
| `dataset_manifest.json` | `_01`과 같은 형식(패널 행 수, 컬럼, fold 경계, 마트 해시) |
| `fold_metrics.parquet` | fold × 지표 |
| `predictions_valid.parquet` | `(trade_date, ticker, market, p, y, raw_label)` — valid만. holdout 예측은 만들지 않는다 |
| `reliability.parquet` | 구간별 보정표 |
| `economics.parquet` | top-decile·top-k·임계 포트폴리오 격자별 수익·turnover |
| `summary.md` | 위를 사람이 읽는 한 장 |

`predictions_valid.parquet`을 남기는 이유는 E 단계가 끝난 뒤 **사후 진단**(어느 종목·언제 틀렸나, 업종 편중)을 다시 학습 없이 할 수 있게 하려는 것이다. Horizon Scan이 `daily_ic.parquet`을 저장하기 전까지 겪은 공백을 반복하지 않는다.
