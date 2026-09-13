# 01. 예측 대상과 라벨

- 작성일: 2026-09-07
- 정본 코드: `research/etl/labels.py` (`LabelSpec`, `build_label_sql`, `build_label_scan_sql`)
- 기존 설계: `docs/target/01_20_access_return_rank/prediction_target_20d_excess_return_rank.md`, `docs/dev/20260830_feature_summary/07_labels_sample_validation.md` §1

---

## 1. 확률 라벨 세 가지

### 1.1 정의

기존 라벨 라이브러리의 `raw_label_{h}d`(h거래일 초과수익률)와 `y_rank_{h}d`(시장 내 백분위) 위에 이진 라벨을 얹는다.

| id | 이름 | 정의 | 양성 비율 | 기존 컬럼과의 관계 |
|---|---|---|---|---|
| **L-A** | 시장 대비 상승 | `y_up_{h}d = 1[raw_label_{h}d > 0]` | 날짜×시장마다 약 0.5 (동일가중 평균이 0이므로 중앙값 근처) | `raw_label` 부호 |
| **L-B** | 상위 20% | `y_top_{h}d = 1[y_rank_{h}d >= 0.8]` | 정확히 0.2 | `y_cls_{h}d == 1` |
| L-C | 절대 상승 | `y_absup_{h}d = 1[fwd_ret_{h}d > 0]` (`LabelSpec(kind="abs")`) | 시장 국면에 따라 0.3~0.7로 흔들림 | `kind="abs"`의 `raw_label` 부호 |

`raw_label_{h}d = fwd_ret_{h}d − bench_ret_{h}d`, 벤치는 같은 `(trade_date, market)`의 동일가중 평균 forward 수익률(`bench="eqw_market"`)이다. 지수 벤치(`bench="index"`)는 예약만 돼 있고 `NotImplementedError`다.

### 1.2 왜 L-A가 주 라벨인가

- **기존 증거가 그 위에 있다.** 41 family의 IC·spread·국면 의존은 전부 시장 중립 라벨에서 재졌다. L-A는 그 라벨의 부호다.
- **양성 비율이 시간에 따라 안정적이다.** 날짜마다 약 절반이 양성이라 보정 기준선이 명확하고, 국면이 바뀌어도 학습 목표가 흔들리지 않는다.
- **운용 목적 둘을 다 덮는다.** "여유 자금으로 길게"도 "단기 수익"도 같은 시장 안에서 상대적으로 강한 종목을 고르는 문제로 환원된다. 시장 자체를 살지 말지는 별도 판단이다.

### 1.3 L-B와 L-C의 자리

- **L-B**는 실매수 리스트(top-k)와 직결된다. p가 "상위 20%에 들 확률"이라 k개 선택의 정밀도(precision@k)로 바로 읽힌다. 양성 비율이 0.2로 고정돼 불균형 분류 문제가 된다. L-A와 같은 피쳐·모델로 돌려 **h마다 어느 라벨이 나은지**를 E1에서 정한다(`05` §2).
- **L-C**는 사용자가 말한 "주가가 상승할지"에 가장 가깝지만 시장 방향이 라벨에 들어온다. 매크로 `cf_*`가 직접 예측 변수가 되는데, 2014~2025년에 금리·물가 사이클이 두세 번뿐이라 외삽 위험이 크다(`06_macro_regime.md` §3.3). 그리고 기존 피쳐 검증 결과를 그대로 옮길 수 없다. **이번 E0~E5 범위에서 뺀다.** 후속으로 열 때는 `P(절대 상승) ≈ P(시장 상승) × P(시장 대비 상승 | …)`처럼 시장 방향 모델을 따로 두는 합성이 해석에 낫다.

### 1.4 확률의 뜻 — 운용에서 어떻게 읽나

| 라벨 | p = 0.7의 뜻 | 포트폴리오 규칙(`03` §3.3) |
|---|---|---|
| L-A | h일 뒤 같은 시장의 평균보다 높을 확률 70% | 상위 k개 매수(순위 사용) 또는 p ≥ τ 전부 매수(확률 사용). 시장 헤지를 붙이면 순수 상대수익 |
| L-B | h일 뒤 상위 20%에 들 확률 70% | 상위 k개 매수. p ≥ τ는 k가 매일 변한다 |

**보정이 안 된 p는 순위 점수와 다르지 않다.** 그래서 이 모델의 판정에 log-loss·Brier·ECE가 들어간다(`03` §2).

---

## 2. horizon

| h | 라벨 컬럼 | 라벨 성숙(formation → 확정) | 연 리밸런싱(비중첩) | 사전등록 밴드가 있는 피쳐 |
|---|---|---|---|---|
| 5 | `*_5d` | 5세션 후 | 약 50회 | 반전·회전율 충격·수급 5/20d·filing burst |
| 20 | `*_20d` | 20세션 후 | 약 12회 | 변동성·MAX·개인 수급·filing·매크로 베타·시장 베타 |
| 60 | `*_60d` | 60세션 후 | 약 4회 | 재무·밸류·규모·주주환원·지분·매크로 베타 |
| 120 | `*_120d` | 120세션 후 | 약 2회 | 규모·비유동성·발행·밸류 |

- 기존 `LabelSpec.horizons` 기본값은 `(20, 5, 60)`이다. 이번 모델은 `(20, 5, 60, 120)`으로 넓힌다. `build_label_sql`은 임의 h를 받으므로 코드 변경은 없다. 다만 **h=120은 `common_survivor` 표본 끝(2025-02-05)을 정하는 값**이라 선택 구간 끝에서 120세션이 라벨 없이 잘린다 — `walk_forward_splits`의 purge가 이것을 처리한다.
- 라벨 겹침: h가 길수록 인접 날짜 라벨이 겹친다. 통계는 Newey-West, 분할은 `purge = embargo = h`로 다룬다(`03` §1).

---

## 3. 라벨 라이브러리 확장 (M-1)

`LabelSpec.outputs`에 `"up"`·`"top"`을 추가한다. `_build_select_cols`에서 기존 `cls`와 같은 null guard 패턴으로 만든다.

```python
# labels.py — _VALID_OUTPUTS = ("reg", "rank", "cls", "up", "top")
if "up" in spec.outputs:
    cols.append(f"CASE WHEN {null_guard} THEN CAST({raw} > 0 AS TINYINT) END AS y_up_{h}d")
if "top" in spec.outputs:
    r = f"PERCENT_RANK() OVER ({win} ORDER BY {raw})"
    cols.append(f"CASE WHEN {null_guard} THEN CAST({r} >= {spec.cls_top} AS TINYINT) END AS y_top_{h}d")
```

- `raw = 0`인 행은 `y_up = 0`이다(초과수익 0은 "오르지 않았다"). 실측상 거의 없다.
- `y_top`은 `y_cls == 1`과 같은 값이다. 별도 컬럼으로 두는 이유는 모델 코드가 `{-1,0,1}` 3-class를 이진으로 다시 자르지 않게 하려는 것이다.
- forward 수익률이 없는 행(패널 끝 h세션)은 `NULL`이 유지돼야 한다. `cls`의 null guard 회귀 테스트를 `up`·`top`에도 복제한다.
- `kind="abs"`일 때 `y_up`은 L-C가 된다. 이번 실험은 `kind="excess"`만 쓴다.

**기존 `_01` 모델의 라벨 출력은 바뀌지 않는다.** `outputs` 기본값 `("reg","rank","cls")`를 유지한다. 새 모델 spec만 `("rank","up","top")`을 요청한다(`reg`는 이 모델에 필요 없다. `raw_label`은 항상 나온다).

---

## 4. 양성 비율과 표본

| 라벨 | 날짜별 양성 비율 | 학습에 주는 영향 |
|---|---|---|
| L-A | 중앙값 근처. 초과수익 분포가 오른쪽으로 치우쳐 평균 0이면 중앙값은 음수라 **양성이 절반보다 약간 적다**(대략 0.45~0.48, 실측은 E0에서 기록) | 균형에 가깝다. class weight 불필요 |
| L-B | 0.20 고정 | 불균형. log-loss는 그대로 쓰고, 지표는 precision@k·lift@k로 읽는다. class weight는 쓰지 않는다(보정이 깨진다) |

- 패널: 기존 모델 기준 2015-01-02 ~ 2026-06-10에서 5,395,167행(`dim_universe_daily` 필터 통과). 이번 선택 구간은 formation ≤ 2025-07-31이라 약 10.5년, 행 수는 E0 build에서 기록한다.
- h=120 라벨은 formation 2025-07-31 기준 2026-01 말에 성숙했다. holdout(formation 2025-08-01~)의 h120 라벨은 2026-10 시점에 약 9개월 분량이 성숙해 있다.

---

## 5. 실현값과 경제성 계산에 쓰는 컬럼

| 용도 | 컬럼 | 비고 |
|---|---|---|
| 학습 타깃 | `y_up_{h}d` / `y_top_{h}d` | 이진 |
| 순위 지표(Rank IC·spread) | `raw_label_{h}d` | 수익률 단위. `research/etl/metrics.evaluate(realized_col=...)` |
| top-k·임계 포트폴리오 수익 | `raw_label_{h}d` | 비중첩 리밸런싱 격자 `rebalance_grid(dates, h)` |
| 보정 지표 | `y_up_{h}d` / `y_top_{h}d` | 예측 p 대비 실현 비율 |

절대 수익률(`fwd_ret_{h}d`)도 라벨 테이블에 있으므로, 시장 헤지 없는 포트폴리오의 절대 성과는 진단으로 같이 낸다. 판정에는 쓰지 않는다.

---

## 6. 정해진 것

- **확정(2026-09-07, D-1·D-3)**: L-A 주 라벨, L-B 보조 라벨, h {5, 20, 60, 120}, `kind="excess"`, `bench="eqw_market"`, `cls_top=0.8`. 코드는 L-A·L-B를 둘 다 만들고, E1의 라벨 선택에서 동률이면 L-A다.
- 하지 않는 것: 지수 벤치 전환(`bench="index"`), L-C, 3-class 다항 분류, 회귀 타깃.
