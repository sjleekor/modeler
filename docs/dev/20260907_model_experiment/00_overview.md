# 00. 상승 확률 모델 실험 — 개요와 작업 계획

- 작성일: 2026-09-07
- 목표: **h거래일 뒤 주가가 시장 대비 오를 확률**을 내는 모델을 여러 h(5·20·60·120)에서 실험하고, 피쳐 조합·국면 interaction을 함께 평가한다.
- 입력 자료: `docs/dev/20260830_feature_summary/`(피쳐 카탈로그), `docs/target/01_20_access_return_rank/`(기존 모델·acceptance gate), `research/models/_01_20_access_return_rank/`(기존 코드).
- 이 디렉터리는 **계획**이다. 실행 결과는 `results/`에 run별로 쌓고, 판정 기준은 결과를 보기 전에 이 문서들에 고정한다.
- 병행 스트림: 종목 간 관계·추가 수집은 `docs/dev/20260907_additional_feature/`에서 따로 진행한다. 두 스트림의 접점은 §5.

---

## 1. 한 장 요약

| 항목 | 결정 |
|---|---|
| 예측 대상 | 주 라벨 **L-A** `P(h일 초과수익률 > 0)`, 보조 라벨 **L-B** `P(상위 20%)`. 절대 상승 L-C는 별도 후속 (`01`) |
| horizon | **5 · 20 · 60 · 120** 거래일. h마다 모델을 따로 만든다 |
| 피쳐 | FS0 baseline 40 → FS1 검증 통과 피쳐 추가(56) → FS2 매크로 베타·국면·interaction 추가(68). 관계·재무위험 피쳐(FS3, 11 컬럼 79)는 병행 스트림에서 왔다 — **2026-09-09 확정** (`02` §1.5) |
| 모델 | HGB 분류기(주), 로지스틱(대조), 기존 rank-HGB + isotonic(연결 기준선). LightGBM 등 새 의존성은 넣지 않는다 (`04`) |
| 검증 | purged walk-forward 5-fold, `embargo = purge = h`. 주 지표는 **valid log-loss 개선 ∧ 비용 반영 top-k 수익 ≥ baseline** (`03`) |
| holdout | **formation 2025-08-01 이후는 열지 않는다.** Horizon Scan과 같은 경계. 2026-10~11 T1·T2 h60 판정과 같은 날 한 번만 연다 |
| 실험 규모 | E0~E4 약 52 run, run당 약 20분 → 약 17시간 (`05`) |
| 코드 | 새 패키지 `research/models/_02_updown_prob/`. 기존 `_01`은 건드리지 않고 import만 한다 (`06`) |

---

## 2. 왜 새 모델인가 — 기존 모델과의 차이

기존 `01_20_access_return_rank`는 `y_rank_20d`(20일 초과수익률의 시장 내 백분위)를 HGB 회귀로 맞추는 **순위 모델**이다. walk-forward 평균 Rank IC h20 0.1436, 비용 반영 top-decile spread 0.0126이다. 이번 모델은 세 가지가 다르다.

1. **출력이 확률이다.** 순위가 아니라 "오를 확률 p"를 낸다. 그래서 순위 지표(Rank IC·spread) 외에 **보정(calibration)** 지표(log-loss·Brier·ECE)가 판정에 들어간다. p=0.7이면 실제로 10번 중 7번 올라야 한다.
2. **h가 여럿이다.** 기존 모델은 h20이 배포 대상이고 h5·h60은 게이트용이었다. 이번에는 h마다 피쳐 집합·모델·게이트를 따로 둔다. 여유 자금 장기 운용(h60·h120)과 단기(h5)는 다른 모델이다.
3. **거시를 넣는다.** Phase C에서 검증된 interaction 4개와 매크로 exposure 베타 4개, 국면 변수 4개를 명시적으로 넣는다.

바꾸지 않는 것: 라벨의 시장 중립 정의(`(거래일, 시장)` 동일가중 벤치), purged walk-forward, 비용 60bp·k=100 경제성 확인, 사전등록 규율.

---

## 3. 작업 패키지

**상태 (2026-09-12): M-0~M-8 완료. 남은 것은 M-9 holdout 하나뿐이고, 라벨이 성숙하는
2026-10~11까지 열지 않는다.** 실행은 58 run·장비 시간 38.2시간으로 끝났고 결과는
[`results/README.md`](results/README.md)에 있다. 단계별 기록은 `06` §3.2~§3.9다.

| id | 작업 | 산출물 | 선행 | 문서 |
|---|---|---|---|---|
| **M-0** | 결정 고정 — 라벨·h·경계·k·비용·유니버스·수급 variant (§4) | **완료 2026-09-07** — 권고안 채택 | — | `00` |
| **M-1** | 라벨 확장 — `LabelSpec.outputs`에 `up`·`top` 추가 (`y_up_{h}d`, `y_top_{h}d`), h=120 지원 | **완료 2026-09-09** — `06` §3.1 M-PR1 | M-0 | `01` |
| **M-2** | 피쳐 집합 registry — FS0/FS1/FS1h/FS2 컬럼 목록을 코드로 고정, 마트 뷰 매핑 확장, 국면 마트 `dim_regime_daily`, interaction 빌더 | **완료 2026-09-09** — `06` §3.1 M-PR0·PR4 | M-0 | `02` |
| **M-3** | 전처리 `rank` 프로파일 — `(date, market)` 내 백분위 변환 + `*_isna` | **완료 2026-09-09** — `06` §3.1 M-PR2 | — | `02` |
| **M-4** | 지표 — `classification_report`(AUC·log-loss·Brier·ECE·precision@k), 확률 임계 포트폴리오, 기존 `topk_economic_report` 재사용 | **완료 2026-09-09** — `06` §3.1 M-PR3 | — | `03` |
| **M-5** | 모델 — HGB 분류기·로지스틱·isotonic 보정, 고정 grid, 시드 | **완료 2026-09-09** — `06` §3.1 M-PR6 | M-3 | `04` |
| **M-6** | 실험 러너 — 매트릭스 E0~E4를 manifest와 함께 실행, 결과 parquet·markdown | **완료 2026-09-09** — `06` §3.1 M-PR5·PR7 | M-1~M-5 | `05`·`06` |
| **M-7** | 실행 E0 → E1 → E2 → E3 → E4 (순서대로, 앞 단계 판정 뒤) | **완료 2026-09-12** — 49 run. `06` §3.4~§3.8. h120은 E2에서 탈락(§3.6) | M-6 | `05` |
| **M-8** | 결과 문서 — h별 채택 후보, 보정 곡선, 비용 반영 성과, 한계 | **완료 2026-09-12** — [`results/README.md`](results/README.md) | M-7 | `05` §6 |
| M-9 | holdout 1회 (2026-10~11, T1·T2와 같은 날) | **대기** — 열 config 6줄은 `results/README.md` §7에 사전등록 완료 | M-8 + 라벨 성숙 | `03` §5 |

M-1~M-5는 서로 독립이라 병행할 수 있다. M-6이 합류점이다.

**E5(FS3)는 이 표에 없다.** 병행 스트림 F-HS-1의 11 컬럼이 2026-09-09에 확정되면서 뒤에 붙은
단계다. 2026-09-12에 9 run으로 끝났고 h60에서 채택됐다(`06` §3.9).

---

## 4. 결과를 보기 전에 정한 것 — 확정 (2026-09-07)

**2026-09-07 사용자가 아래 여섯 항목을 권고안 그대로 채택했다.** 이 표가 M-0의 산출물이다. 결과를 본 뒤 바꾸지 않는다.

| id | 결정 | 선택지 | **확정** | 이유 |
|---|---|---|---|---|
| **D-1** | 주 라벨 | L-A 시장 대비 상승 / L-B 상위 20% / L-C 절대 상승 | **L-A 주, L-B 보조. L-C는 후속** | 기존 검증 41 family가 시장 중립 라벨 위에 있다. L-C는 시장 방향이 라벨에 들어와 매크로 외삽 위험이 크다 (`01` §1.3) |
| **D-2** | 선택·holdout 경계 | (a) formation ≤ 2025-07-31 / (b) 기존 모델처럼 2026-06까지 | **(a)** | Horizon Scan holdout(2025-08-01~)과 같은 경계여야 T1·T2와 한 번에 판정할 수 있다. (b)는 최근 1년을 선택에 써서 holdout이 없어진다 |
| **D-3** | horizon 집합 | {5, 20, 60, 120} / +10 | **{5, 20, 60, 120}** | 사전등록 밴드가 있는 h. 10은 5와 20 사이 보간이라 정보 추가가 적다 |
| **D-4** | 경제성 가정 | k=100, 왕복 60bp / 다른 k·비용 | **k=100, 60bp 유지 + 확률 임계 τ=0.6 포트폴리오 병행** | 기존 게이트와 비교 가능. 운용 자금 규모가 정해지면 k를 바꾼다 |
| **D-5** | 학습 유니버스 | 기본 필터(60일 평균 거래대금 1억 이상, warm-up) / broad | **기본 필터** | 실제 매수 가능 종목이 대상이다. broad는 Horizon Scan discovery 좌표이고 모델 대상이 아니다 |
| **D-6** | 수급 피쳐 시점 | `lag1`(검증 정본) / 당일 값(baseline 관행) | **`lag1` 기본, 당일 값은 E3 변형** | 당일 집계 사용 가능 여부가 미검증이다. 검증된 쪽을 기본으로 둔다 |

확정에 따라 코드 기본값이 정해진다: `ModelSpec.period_end = "2025-07-31"`, `horizons = (20, 5, 60, 120)`, `label.outputs = ("rank", "up", "top")`, `flow_variant = "lag1"`, 유니버스 기본 `UniverseFilter`, 경제성 `k=100`·`cost_bps_roundtrip=60`·`tau=0.6`. E1의 라벨 선택 규칙에서 동률이면 L-A다(`05` §2).

---

## 5. 병행 스트림과의 접점

`20260907_additional_feature/`는 종목 간 관계·업종 PIT·레버리지·생애주기·SUE 백필·매크로 2단계를 만든다. 이 스트림에 넘겨받는 방식은 하나다.

- **마트 계약.** 새 피쳐는 `research/etl/features/<name>.py`가 `feat_<name>` 마트로 만들고, grain은 `(trade_date, ticker, market)`, PIT 규칙과 유효 시작일을 docstring에 적는다. 이 스트림은 `_02/features.py`의 FS3에 컬럼 목록을 추가하는 것으로만 받는다.
- **시점.** FS3는 E2가 끝난 뒤 별도 단계 **E5**로 검정한다. E0~E4의 사전등록은 FS3 없이 닫는다. FS3가 늦어도 E0~E4는 끝난다.
- **단변량 검증 선행.** FS3 후보는 Horizon Scan 새 config로 먼저 검정하고(병행 스트림 담당), `screen_pass`한 것만 E5에 넣는다. 모델이 피쳐 검증을 대신하지 않는다.
- **PIT 통일.** 병행 스트림의 F-9(`fin_pit` strict vintage 재작성)가 끝나면 FS0의 `fin_*` 10개를 그것으로 바꾼 **FS0'**를 E3에서 비교한다. 끝나지 않으면 E3는 수급 variant만 본다.

---

## 6. 규율

1. **holdout은 한 번.** formation 2025-08-01 이후 라벨을 E0~E5 어디서도 읽지 않는다. 코드 레벨에서 `period_end = 2025-07-31`을 spec 기본값으로 박는다.
2. **주 지표를 h마다 하나로 고정한다**(`03` §4). 여러 지표 중 좋은 것을 고르지 않는다.
3. **grid는 고정.** 하이퍼파라미터 격자·시드·피쳐 집합·임계값은 `05`에 적힌 것만 돈다.
4. **E 단계 사이의 선택은 앞 단계의 사전등록 규칙으로만** 한다. E1에서 h별 라벨·모델을 고르는 규칙은 `05` §2.
5. **snapshot을 고정한다.** 모든 run은 `snapshot_date=2026-08-23`, `source=sj2_remote` 마트를 쓴다(Horizon Scan canonical과 같다). holdout 때만 새 snapshot을 만든다.
6. **기존 `_01` 코드·결과를 바꾸지 않는다.** 필요한 함수는 import한다. 공통화 리팩터는 이 실험 뒤에 한다.
7. **Horizon Scan과 동시에 돌리지 않는다.** 14코어/36GB에서 Phase B가 peak 20GB를 쓴다. 실험 run은 5.4M행 × 100컬럼 패널이라 겹치면 서로 느려진다.

---

## 7. 문서 지도

| 파일 | 내용 |
|---|---|
| **`00_overview.md`** | 이 문서. 목표·작업 패키지·결정·규율 |
| [`01_target_and_labels.md`](01_target_and_labels.md) | 확률 라벨 3안 정의, 라벨 라이브러리 확장, 양성 비율, 성숙 시점 |
| [`02_features_and_preprocessing.md`](02_features_and_preprocessing.md) | 피쳐 집합 FS0~FS3 컬럼 목록, PIT·시점 규칙, `rank` 전처리, interaction·국면 |
| [`03_validation_and_metrics.md`](03_validation_and_metrics.md) | walk-forward 설계, 지표 정의, h별 주 지표와 게이트, holdout 규칙 |
| [`04_models_and_calibration.md`](04_models_and_calibration.md) | 모델 3종, 고정 grid, 보정 방법, 시드, 단조 제약 변형 |
| [`05_experiment_matrix.md`](05_experiment_matrix.md) | 사전등록 실험 매트릭스 E0~E5, 단계 간 선택 규칙, 예산, 기록 항목 |
| [`06_implementation_plan.md`](06_implementation_plan.md) | 패키지 구조, 변경 파일, PR 분할, 테스트, 실행 명령, 산출물 경로 |
| `results/` | run별 결과(실행 뒤 생성) |
