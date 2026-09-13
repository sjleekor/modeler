# ETL 구현 계획 — 단계·순서·산출물·완료기준

- 작성 일시: 2026-06-20
- 위치: `docs/target/01_20_access_return_rank/`
- 상위 설계: [`../00_shared_etl_platform.md`](../00_shared_etl_platform.md) (멀티모델 플랫폼·mart/dataset 분리), [`prediction_target_20d_excess_return_rank.md`](./prediction_target_20d_excess_return_rank.md) (예측 대상)
- 구현 명세 출처: [`etl_00_ridge_elasticnet.md`](./etl_00_ridge_elasticnet.md) (라벨·피처·전처리·분할·평가), [`etl_01_parquet_data_flow_plan.md`](./etl_01_parquet_data_flow_plan.md) (Parquet lake 접근·canonical export), [`etl_02_engine_comparison.md`](./etl_02_engine_comparison.md) (엔진: DuckDB 메인 + Polars 보조)
- 본 문서의 목적: 위 설계들을 **실제 코드로 옮기기 위한 작업 분해**다. *무엇을* 할지가 아니라 *어떤 순서로, 어떤 크기로, 무엇을 산출하고, 무엇을 통과하면 끝인지*를 정한다. 설계 결정(라벨식·dedup·PIT·전처리)은 전부 위 문서가 authoritative이며 여기서 다시 정하지 않는다.

---

## 0. 결론 (TL;DR)

1. **코드 위치 결정(고정)**: 헥사고날 코어(`src/krx_collector`)와 분리된 **`research/` 최상위 패키지**에 구현한다([`../00_shared_etl_platform.md`](../00_shared_etl_platform.md) §6의 기본값 채택, "1줄 결정" 확정). `research/etl/`(모델 무관 공통)·`research/models/01_20_access_return_rank/`(모델별). DB·`krx_collector`를 import하지 않고 `data_lake/` parquet만 의존. 의존성은 `pyproject.toml`의 `research` extra(이미 존재).
2. **9개 단계(P0~P8)**, 3개 마일스톤(A=1차 시세+수급 / B=2차 재무 / C=3차 공통·거시·PIT정밀화)으로 분해. **마일스톤 A만으로 Ridge/ElasticNet end-to-end가 돈다**(exporter 수정 불필요, lake 이미 준비됨).
3. 각 단계는 **독립 산출물 + 검증(test) + 완료기준(DoD)**을 갖는다. mart(L2a)는 snapshot당 1회·모델 무관, dataset(L2b)는 모델별 가벼운 조립 — 플랫폼 계층 규약을 코드 구조로 강제한다.
4. **세로 슬라이스 우선**: P0~P6(A)를 먼저 끝내 "끝에서 끝까지 도는 가장 얇은 파이프라인"을 만든 뒤, P7(B)·P8(C)에서 피처를 넓힌다. 넓이(피처)보다 깊이(end-to-end)를 먼저.

---

## 1. 패키지 레이아웃 (구현 위치)

```text
research/                                   # 최상위 연구 패키지 (src/krx_collector 와 분리)
  __init__.py
  etl/                                      # ── 모델 무관 공통 계층 (00_shared §3, §6) ──
    __init__.py
    config.py                # snapshot_date, lake 루트 경로, 엔진 옵션(threads/memory_limit)
    lake.py                  # L1 reader: DuckDB 뷰 팩토리 (hive=false 강제, 타입 캐스팅) — P1
    calendar.py              # dim_trading_calendar (d_idx) — P2
    universe.py              # dim_universe_daily (필터 플래그) — P2
    features/
      price.py               # feat_price  (etl_00 §3.1) — P3
      flow.py                # feat_flow   (KRX dedup → pivot → 파생, etl_00 §3.2) — P3
      fin_pit.py             # feat_fin_pit (PIT as-of, etl_00 §3.3) — P7
      common.py              # feat_common (broadcast, etl_00 §3.5) — P8
      event.py               # feat_event  (배당·자사주 플래그, etl_00 §3.4) — P8
    labels.py                # make_label(H, kind, bench, outputs) — P4
    preprocess.py            # per-date winsor/log/zscore, isna 플래그, profile={linear|tree} — P5
    splits.py                # walk_forward_splits(horizon, embargo, scheme) — P5
    metrics.py               # rank_ic / top_decile_spread / icir — P6
    registry.py              # feature_registry (피처 사전, 00_shared §3.4) — P5/P7
    manifest.py              # dataset_manifest.json 생성 (재현성, 00_shared §4) — P5
  models/
    __init__.py
    _01_20_access_return_rank/   # 모델별 조립·학습 (디렉토리 slug 그대로, import용 _ prefix)
      __init__.py
      build_dataset.py       # mart 피처 그룹 선택 → 패널 조립 → 전처리 → split — P5
      train.py               # Ridge/ElasticNet walk-forward 학습·평가 — P6
      spec.py                # 이 모델의 라벨 spec·유니버스 필터·피처 그룹 토글
tests/
  unit/
    test_research_lake.py        # hive=false·캐스팅 단위 (작은 fixture parquet)
    test_research_labels.py      # d_idx forward·excess·rank 정확성
    test_research_preprocess.py  # per-date zscore·isna·fold-aware fit
    test_research_splits.py      # embargo/purge 경계
  integration/
    test_research_etl_pipeline.py  # 실제 lake 있으면 end-to-end, 없으면 self-skip
```

> **import 경로**: `research`는 wheel(`src/krx_collector`)에 포함되지 않는다(분석 전용, 운영 이미지 제외). 실행은 repo 루트에서 `uv run python -m research.models._01_20_access_return_rank.build_dataset ...`. 테스트가 import하도록 `pyproject.toml [tool.pytest.ini_options].pythonpath`에 `"."`(또는 `"research"` 상위)를 추가한다 — **P0의 1줄 작업**.

---

## 2. 단계 의존 순서도

```text
                         ┌──────────────────────────────────────────────┐
                         │  P0  스캐폴딩·결정 고정 (패키지/엔진/경로)      │
                         └───────────────────┬──────────────────────────┘
                                             ▼
                         ┌──────────────────────────────────────────────┐
                         │  P1  Lake 접근 계층 (DuckDB 뷰, hive=false)    │  ← 모든 단계의 토대
                         └───────────────────┬──────────────────────────┘
                                             ▼
            ┌────────────────────────────────┴───────────────────────────┐
            ▼                                                              ▼
┌─────────────────────────┐                              ┌───────────────────────────────┐
│ P2 dim 계층              │                              │ P4 라벨 라이브러리             │
│  calendar(d_idx)        │                              │  make_label(H,kind,bench,out)  │
│  universe(필터)          │                              │  (P2 calendar 의존)            │
└───────────┬─────────────┘                              └───────────────┬───────────────┘
            ▼                                                             │
┌─────────────────────────┐                                              │
│ P3 mart 핵심 피처        │                                              │
│  feat_price             │                                              │
│  feat_flow (KRX dedup)  │                                              │
└───────────┬─────────────┘                                              │
            └───────────────────────────┬────────────────────────────────┘
                                         ▼
                         ┌──────────────────────────────────────────────┐
                         │  P5  dataset 조립 (L2b) + 전처리 + split      │
                         │      + dataset_manifest (재현성)              │
                         └───────────────────┬──────────────────────────┘
                                             ▼
                         ┌──────────────────────────────────────────────┐
                         │  P6  Ridge/ElasticNet 학습·walk-forward·RankIC │  ◀── 마일스톤 A 완료 (1차 end-to-end)
                         └───────────────────┬──────────────────────────┘
                                             ▼
                 ┌───────────────────────────┴──────────────────────────┐
                 ▼                                                        ▼
┌──────────────────────────────────┐                  ┌──────────────────────────────────────┐
│ P7 feat_fin_pit (재무 PIT as-of)  │                  │ P8 feat_common + feat_event           │
│    F_fin ablation 결합            │                  │    (+ §6 rcept_dt PIT 정밀화, 선택)   │
│    ◀── 마일스톤 B                 │                  │    ◀── 마일스톤 C                     │
└──────────────────────────────────┘                  └──────────────────────────────────────┘
```

핵심: **P1이 단일 토대**, P2·P4는 P1 위에서 병행 가능, **P3+P4+P5+P6이 1차 세로 슬라이스**. P7/P8은 P5의 조립 파이프라인에 피처 그룹을 추가로 끼우는 작업(구조 변경 없음 — `etl_00` §4.5 토글).

---

## 3. 마일스톤 ↔ 단계 매핑

| 마일스톤 | 범위 | 단계 | exporter | 산출 | 완료 의미 |
|---|---|---|:--:|---|---|
| **A (1차)** | 시세 + 수급 | P0~P6 | 불필요 | Ridge/ElasticNet 학습·Rank IC 리포트 | "모델이 종목을 잘 고르는가"를 처음 측정 |
| **B (2차)** | + 재무(`stock_metric_fact`) | P7 | 완료(canonical lake) | F_fin 포함 ablation | 재무 피처 유효성 검증 |
| **C (3차)** | + 공통/거시 + 이벤트 + PIT정밀화 | P8 | 완료(canonical lake) | F_common/F_event, rcept_dt PIT(선택) | 멀티모달 풀세트 |

A는 [`etl_01`](./etl_01_parquet_data_flow_plan.md) §4("exporter 수정 없이 즉시 실행")에 정확히 대응. B/C에 필요한 canonical lake는 이미 export·검증 완료([`etl_01`](./etl_01_parquet_data_flow_plan.md) §5.5).

---

## 4. 단계별 상세 (P0~P8)

각 단계: **목적 / 입력 / 산출물 / 검증 / 완료기준(DoD) / 의존 / 크기**.

### P0 — 스캐폴딩·결정 고정

- **목적**: 패키지 골격과 실행/테스트 경로를 만들고, 미결 결정(코드 위치)을 코드로 못박는다.
- **입력**: 없음(기존 repo).
- **산출물**: `research/` 패키지 트리(빈 모듈 + `__init__.py`), `research/etl/config.py`(snapshot_date·lake 루트·DuckDB 옵션을 한 곳에서), `pyproject.toml` pytest `pythonpath`에 루트 추가, `uv sync --extra research` 동작 확인.
- **검증**: `uv run python -c "import research.etl.config"` 성공, `uv run pytest tests/unit -k research`(빈 통과).
- **DoD**: 빈 파이프라인이 import되고 `research` extra가 설치된다.
- **의존**: —. **크기**: XS (반나절).

### P1 — Lake 접근 계층 (L1 reader)

- **목적**: `etl_01` §3의 DuckDB-over-Parquet 접근을 단일 함수로 캡슐화. **모든 SQL이 이 위에서 돈다.**
- **입력**: `data_lake/raw_postgres/...`, `data_lake/canonical_postgres/...`.
- **산출물**: `research/etl/lake.py`
  - `connect(threads, memory_limit, temp_dir)` — DuckDB 커넥션([`etl_02`](./etl_02_engine_comparison.md) §6 옵션).
  - `register_views(con, snapshot_date)` — raw 13종 + canonical 5종을 뷰로 등록. **`hive_partitioning=false` 강제**([`etl_01`](./etl_01_parquet_data_flow_plan.md) §4.2 치명 버그 — `source` 컬럼 오염 방지).
  - `cast_helpers` — `numeric→DOUBLE`, `jsonb/uuid` 문자열 인지(`etl_01` §3 타입 주의).
- **검증**: `test_research_lake.py` — 작은 fixture parquet로 (1) hive=false일 때 `krx_security_flow_raw.source`가 `KRX`/`PYKRX`로 보존되는가, (2) Decimal128 컬럼이 `::DOUBLE` 캐스팅되는가. + 실제 lake가 있으면 `krx_flow`에서 `SELECT source, count(*) GROUP BY source`가 KRX/PYKRX 2값인지 스모크.
- **DoD**: 두 lake 루트의 18개 뷰가 등록되고 `source` 오염이 없다.
- **의존**: P0. **크기**: S.

### P2 — dim 계층 (calendar, universe)

- **목적**: forward/embargo의 기준 `d_idx`와 PIT 유니버스. mart의 차원 테이블.
- **입력**: `daily_ohlcv` 뷰(P1).
- **산출물**:
  - `calendar.py` → `dim_trading_calendar`(거래일 → `d_idx`, [`etl_00`](./etl_00_ridge_elasticnet.md) 체크리스트의 "캘린더 아닌 거래일 인덱스" 단일 소스).
  - `universe.py` → `dim_universe_daily`([`etl_00`](./etl_00_ridge_elasticnet.md) §1.2 필터 4종: t행 존재·`is_halted=false`, warm-up 40/60, 60일 평균거래대금 하한, t+20 존재[학습셋 한정]). parquet으로 `data_lake/feature_mart/snapshot_date=<…>/`에 물질화.
- **검증**: `is_halted := (open=high=low=0)` 규약·warm-up 경계 단위 테스트. 유니버스 행수가 기간·시장별로 합리적(2015+ ~2,500종목, [`etl_00`](./etl_00_ridge_elasticnet.md) §1.1)인지 스모크.
- **DoD**: `dim_universe_daily` parquet 산출, 필터 플래그 컬럼 존재.
- **의존**: P1. **크기**: S.

### P3 — mart 핵심 피처 (price, flow)

- **목적**: 1차의 두 피처 그룹. **가장 무거운 연산(Q1 수급 dedup, [`etl_02`](./etl_02_engine_comparison.md))을 mart에서 1회**.
- **입력**: `daily_ohlcv`, `krx_security_flow_raw` 뷰.
- **산출물**:
  - `features/price.py` → `feat_price`([`etl_00`](./etl_00_ridge_elasticnet.md) §3.1: ret/mom/vol/turnover/amihud/`is_halted`…). `px_` prefix.
  - `features/flow.py` → `feat_flow`([`etl_00`](./etl_00_ridge_elasticnet.md) §3.2: **KRX 우선 dedup** `QUALIFY ROW_NUMBER()` → metric 7종 wide pivot → 누적·비율·z-score). `flow_` prefix. 공매도 잔고(2016-06-30~) 결측 `*_isna`.
  - 둘 다 `feature_mart/.../feat_price/`, `feat_flow/`에 parquet 물질화.
- **검증**: dedup 후 row count가 **55,918,702 distinct**([`etl_01`](./etl_01_parquet_data_flow_plan.md) §4.2 / [`etl_02`](./etl_02_engine_comparison.md) §3 Q1 실측)와 일치하는지 — 회귀 가드. 순매수 3주체 합 항등식 미사용([`etl_00`](./etl_00_ridge_elasticnet.md) §3.2).
- **DoD**: `feat_price`·`feat_flow` parquet 산출, dedup row count 검증 통과.
- **의존**: P1(P2와 병행 가능). **크기**: M (flow가 핵심 복잡도).

### P4 — 라벨 라이브러리

- **목적**: `make_label(...)` 단일 생성기([`../00_shared_etl_platform.md`](../00_shared_etl_platform.md) §3.1). 모델 수 = 라벨 수이므로 가장 재사용성 높음.
- **입력**: `daily_ohlcv`, `dim_trading_calendar`(d_idx).
- **산출물**: `labels.py` → `make_label(prices, horizon, kind={excess|abs}, bench={eqw_market|index}, outputs={reg,rank,cls}, winsor=[.005,.995])`. [`etl_00`](./etl_00_ridge_elasticnet.md) §2의 d_idx forward join → 시장별 동일가중 벤치(`eqw_market`) 차감 → per-date `PERCENT_RANK`. 출력 컬럼 `y_reg_20d`/`y_rank_20d`/`y_cls_20d` + 보조 `y_*_5d/60d`, `y_vol_20d`, `y_mdd_20d`.
- **검증**: `test_research_labels.py` — 소형 합성 시세로 (1) d_idx+20이 정확히 20거래일 후인가(정지일·휴일 흡수), (2) `excess = fwd - mkt`가 시장 평균 0 합 만족, (3) `y_cls` 임계(0.8/0.2) 경계. **벤치는 1차 `eqw_market`**(지수는 1줄 교체, [`prediction_target`](./prediction_target_20d_excess_return_rank.md) §1 주석).
- **DoD**: `label_daily` parquet, 3종 라벨 + 보조 horizon 산출.
- **의존**: P1, P2(calendar). **크기**: M.

### P5 — dataset 조립 (L2b) + 전처리 + split + manifest

- **목적**: mart 피처를 모델별로 조립하고 선형모델 전처리·분할·재현 메타를 붙인다.
- **입력**: `dim_universe_daily`, `feat_price`, `feat_flow`, `label_daily`(+ P7/P8에서 `feat_fin_pit` 등).
- **산출물**:
  - `models/_01_20_access_return_rank/build_dataset.py` — [`etl_00`](./etl_00_ridge_elasticnet.md) §4.1 as-of join 파이프라인(universe ⟕ 피처 그룹 ⟕ label).
  - `preprocess.py` — [`etl_00`](./etl_00_ridge_elasticnet.md) §4.2/§4.3: `*_isna` 플래그 → impute → **per-date winsorize→log→zscore**, `profile={linear|tree}` 토글. **통계량은 fold train에서만 fit**.
  - `splits.py` — [`etl_00`](./etl_00_ridge_elasticnet.md) §5 purged walk-forward, embargo/purge=20거래일.
  - `manifest.py` — [`../00_shared_etl_platform.md`](../00_shared_etl_platform.md) §4 `dataset_manifest.json`(snapshot·피처그룹·label_spec·기간·code_rev·row_count).
  - 산출 위치: `data/datasets/01_20_access_return_rank/snapshot_date=<…>/`(`feat_panel(_std)`, `label_daily`, `split_folds`, manifest). `.gitignore` 대상.
- **검증**: `test_research_preprocess.py`(per-date zscore 평균0/표준편차1, isna 정합, NaN/Inf 0 assert), `test_research_splits.py`(embargo 공백·purge 경계, 룩어헤드 0).
- **DoD**: 모델 입력 행렬에 **NaN/Inf 0** 보장 + manifest 생성. ablation 위해 그룹 prefix(`px_`,`flow_`,…) 유지.
- **의존**: P3, P4. **크기**: M~L (전처리 정확성이 핵심).

### P6 — 모델 학습·검증 (마일스톤 A 완료)

- **목적**: Ridge/ElasticNet end-to-end, 랭킹 중심 평가.
- **입력**: `feat_panel_std`, `label_daily`, `split_folds`.
- **산출물**: `models/_01_20_access_return_rank/train.py` — sklearn Ridge/ElasticNet, walk-forward 루프, `alpha`/`l1_ratio` 선택. Polars→numpy zero-copy 핸드오프([`etl_02`](./etl_02_engine_comparison.md) §6). `metrics.py` — Rank IC(일자별→평균)·Top-decile·Top-minus-bottom·ICIR([`etl_00`](./etl_00_ridge_elasticnet.md) §6).
- **검증**: holdout(2026-01~06-10) 1회 평가 경로. Rank IC 분포가 fold별로 산출되는지. 회귀 sanity로 per-date 표준화 타깃 RMSE/R².
- **DoD**: **walk-forward Rank IC 리포트가 나온다**(= 1차 목표 달성). 하이퍼파라미터 선택 로그.
- **의존**: P5. **크기**: M.

### P7 — 재무 PIT 결합 (마일스톤 B)

- **목적**: `feat_fin_pit`를 mart에 추가하고 F_fin ablation으로 결합.
- **입력**: `stock_metric_fact`(canonical 뷰), `dim_universe_daily`.
- **산출물**: `features/fin_pit.py` — [`etl_00`](./etl_00_ridge_elasticnet.md) §3.3 / [`etl_01`](./etl_01_parquet_data_flow_plan.md) §6: `available_from := period_end+90d`(분기 +45d), **metric_code를 partition에 포함한 JOIN + ROW_NUMBER()**(단순 ASOF 금지 — metric 차원 누락, [`etl_01`](./etl_01_parquet_data_flow_plan.md) §6 검증됨). 파생 비율(roa·debt_to_equity…), 자본잠식 `is_negative_equity` clip, `has_fs` 플래그. `fin_` prefix.
- **검증**: PIT as-of가 종목당 전 metric(~26종) 보존하는지(삼성전자 2024-06-03=26행, [`etl_01`](./etl_01_parquet_data_flow_plan.md) §6). `available_from <= t` 룩어헤드 0.
- **DoD**: `feat_fin_pit` parquet, F_fin on/off ablation 결과 비교.
- **의존**: P6(조립 파이프라인 재사용). **크기**: M.

### P8 — 공통/거시 + 이벤트 (마일스톤 C)

- **목적**: `feat_common`(국면 보정)·`feat_event`(플래그) 결합, 선택적 PIT 정밀화.
- **입력**: `common_feature_daily_fact`(canonical), `dart_shareholder_return_raw`·`dart_share_count_raw`(raw).
- **산출물**:
  - `features/common.py` — [`etl_00`](./etl_00_ridge_elasticnet.md) §3.5: `asof_available_date <= t` broadcast, 결측 구간 0+`cf_isna`. 일별 raw 2025-12-15~라 1차 패널 대부분 결측(보조 변수). `cf_` prefix.
  - `features/event.py` — [`etl_00`](./etl_00_ridge_elasticnet.md) §3.4: 배당/자사주/발행주식수 변화 저빈도 플래그. `ev_` prefix.
  - (선택) `rcept_dt` 기반 PIT 정밀화 — [`etl_01`](./etl_01_parquet_data_flow_plan.md) §6: raw lake에 `rcept_dt` 없음(`rcept_no`만) → payload 파싱/export 확장 **선행 과제**로 분리.
- **검증**: 공통 피처가 가용 구간 외 결측+플래그로만 들어가는지([`etl_00`](./etl_00_ridge_elasticnet.md) §9 체크리스트). feature_date↔거래일 정합.
- **DoD**: F_common/F_event ablation 결과. 풀 멀티모달 데이터셋 산출.
- **의존**: P6. **크기**: M (rcept_dt 정밀화는 별도 S~M, 선택).

---

## 5. 횡단 관심사 (전 단계 공통)

- **재현성**: 모든 산출물은 `snapshot_date=<…>` 디렉토리로 핀([`etl_01`](./etl_01_parquet_data_flow_plan.md) §0.5). mart는 디렉토리 있으면 skip, `--force`로 재빌드([`../00_shared_etl_platform.md`](../00_shared_etl_platform.md) §5).
- **DB 접근 금지**: `research/` 어디에서도 `psycopg`·`get_settings()` DB 접속 없음. lake parquet만([`etl_01`](./etl_01_parquet_data_flow_plan.md) §0.5 불변 규칙).
- **테스트 관례**: 단위는 소형 fixture parquet(`tests/fixtures/`)로 DB·lake 불요. 통합은 실제 lake 없으면 self-skip(기존 `tests/integration` 패턴 준수). 라이브 데이터 의존 테스트는 env gate.
- **엔진 경계**: 무거운 scan/join/window는 DuckDB SQL, per-date 전처리·sklearn 핸드오프는 Polars([`etl_02`](./etl_02_engine_comparison.md) §5.2). `con.execute(...).arrow()` → `pl.from_arrow(...)` zero-copy.
- **품질 게이트**: `uv run ruff check`·`uv run black`·`uv run pytest tests/unit` 통과를 각 PR 기준으로(기존 CLAUDE.md 관례).
- **피처 사전**: `feature_registry`(00_shared §3.4)를 P5에서 시드하고 P7/P8에서 확장. 모델별 `feature_dictionary.md`는 이 레지스트리의 뷰.

---

## 6. 권장 PR 분할 (리뷰 단위)

| PR | 단계 | 내용 | 의존 |
|---|---|---|---|
| PR1 | P0+P1 | 패키지 골격 + lake reader (hive=false) + 단위테스트 | — |
| PR2 | P2 | calendar + universe mart | PR1 |
| PR3 | P3 | feat_price + feat_flow (dedup row count 가드) | PR1 |
| PR4 | P4 | make_label 라이브러리 + 라벨 정확성 테스트 | PR2 |
| PR5 | P5 | dataset 조립 + 전처리 + split + manifest | PR3, PR4 |
| PR6 | P6 | Ridge/ElasticNet + Rank IC 리포트 **(마일스톤 A)** | PR5 |
| PR7 | P7 | feat_fin_pit + F_fin ablation **(마일스톤 B)** | PR6 |
| PR8 | P8 | feat_common + feat_event (+ rcept_dt 정밀화 선택) **(마일스톤 C)** | PR6 |

PR2·PR3은 PR1 후 병행 가능. PR7·PR8은 PR6 후 병행 가능.

---

## 7. 리스크 / 선결 확인 (설계 문서에서 이월)

- **hive_partitioning=false 강제**(P1) — 누락 시 `source` dedup 무력화([`etl_01`](./etl_01_parquet_data_flow_plan.md) §4.2). P1 테스트가 회귀 가드.
- **Decimal128 캐스팅**(P1/P3) — 비율·로그 전 `::DOUBLE`, 단 winsor/log 후 float화 순서 유지([`etl_01`](./etl_01_parquet_data_flow_plan.md) §3, [`etl_00`](./etl_00_ridge_elasticnet.md) §4.3).
- **PIT as-of metric 차원 보존**(P7) — 단순 ASOF JOIN 금지([`etl_01`](./etl_01_parquet_data_flow_plan.md) §6).
- **embargo/purge=20거래일**(P5) — 라벨이 t+20 미래를 보므로 누설 방지 필수([`etl_00`](./etl_00_ridge_elasticnet.md) §5).
- **per-date 표준화 fold-aware fit**(P5) — 글로벌 스케일러 금지([`etl_00`](./etl_00_ridge_elasticnet.md) §4.3).
- **공통 피처 단기 이력**(P8) — 2025-12-15~, 1차 비핵심. 장기 백필은 본 모델 범위 밖.
- **`research` 패키지 import 경로**(P0) — pytest `pythonpath`·실행 cwd 확정.

---

## 8. 진행 체크리스트

**마일스톤 A (1차, exporter 수정 없음)** — ✅ **완료 (2026-06-20)**
- [x] P0 패키지 골격·`research` extra·pytest 경로 — `research/etl`, `research/models`
- [x] P1 lake reader (hive=false, 캐스팅) + 단위테스트 — `research/etl/lake.py`
- [x] P2 calendar(d_idx) + universe 필터 mart — `research/etl/{calendar,universe,mart}.py`
- [x] P3 feat_price + feat_flow (dedup 55,918,702 가드 통과) — `research/etl/features/{price,flow}.py`
- [x] P4 make_label (eqw_market 벤치, y_rank_20d 메인; label 행수 6,408,188 가드) — `research/etl/labels.py`
- [x] P5 조립 + 전처리(linear) + split + manifest, NaN/Inf 0 — `research/etl/{splits,preprocess,manifest}.py`, `models/_01_20_access_return_rank/build_dataset.py`
- [x] P6 Ridge/ElasticNet walk-forward + Rank IC 리포트 — `research/etl/metrics.py`, `models/_01_20_access_return_rank/train.py`

> **검증**: 단위 418 + 통합(실 lake) 통과. 라벨 셔플 테스트로 구조적 누수 0 확인(real IC 0.158 → shuffled 0.04). dataset/manifest 산출, per-date z-score(mean≈0/std≈1), embargo/purge 갭 확인.

**마일스톤 B (2차, canonical lake 사용)** — ✅ **완료 (2026-06-20)**
- [x] P7 feat_fin_pit (period_end+90d/분기+45d PIT, interval-join으로 metric 전체 보존) + F_fin ablation — `research/etl/features/fin_pit.py`, build_dataset `fin` 그룹 토글

> **검증**: 단위 423 + 통합(실 lake) 통과. 삼성전자 2024-06-03 PIT as-of = 26 metric 보존(etl_01 §6 가드), 룩어헤드 0(available_from), 자본잠식 clip/flag, px+flow+fin ablation 빌드(40 feature) 동작.

**마일스톤 C (3차)** — ✅ **완료 (2026-06-20, rcept_dt 정밀화 제외)**
- [x] P8 feat_common + feat_event ablation — `research/etl/features/{common,event}.py`, build_dataset `cf`/`ev` 그룹 토글
- [ ] (선택, 미착수) rcept_dt 기반 PIT 정밀화 — raw lake에 `rcept_dt` 없음(`rcept_no`만), payload 파싱/export 확장 선행 필요(etl_01 §6). 2차 정밀화 과제로 보류.

> **검증**: 단위 431 + 통합(실 lake) 통과. feat_common broadcast(날짜당 1행, asof≤t PIT 위반 0, 2025-11~만 가용→1차 비핵심), feat_event(se='합계' totals, +90d PIT, treasury/shares 플래그). 풀 멀티모달 빌드 px+flow+fin+cf+ev = 55 feature 동작.

---

## 10. 코드 리뷰 반영 (2026-06-20)

구현 후 코드 리뷰에서 발견된 5건을 수정(단위 439 + 통합 통과). 회귀 테스트 추가.

| # | 심각도 | 문제 | 수정 |
|---|---|---|---|
| 1 | High | holdout fold의 eval 구간이 `fold_role="valid"`로 라벨돼 walk-forward selection에 누수, 최종 holdout 평가는 미실행 | build_dataset가 holdout fold eval을 `fold_role="holdout"`으로 태깅(선택 제외). `evaluate_holdout`는 같은 fold_id의 train으로 fit(표준화 정합). 회귀 테스트 2건 |
| 2 | Med/High | L2a mart를 우회하고 dataset 빌드마다 76M dedup/PIT 재실행 | `_materialize_source_marts`가 `materialize_*`로 mart를 snapshot당 1회 물질화(skip/force, 00_shared §5). 실측: 빌드#1 14s → #2 2.1s(7×). manifest에 `feature_mart` 경로 기록 |
| 3 | Med | NULL forward(보조 horizon)에 `PERCENT_RANK`가 가짜 rank/cls 부여 | rank/cls를 `WHEN raw IS NOT NULL` 가드로 감쌈. 30-session fixture에서 y_rank_60d 20→0 확인 |
| 4 | Med | `y_vol/y_mdd`가 label 산출물에 미포함 | `LabelSpec.include_risk`(기본 off) 시 risk SQL을 `build_label_sql`에 LEFT JOIN. risk SQL을 correlated subquery→window-frame+단일 self-join으로 재작성(동일값 검증) |
| 5 | Low/Med | metrics가 NaN/inf를 미제거(`drop_nulls`만) | `_finite_clean`(null+`is_finite`)로 rank IC·quantile·n_obs 공통 필터. NaN realized가 top-decile 오염 안 함 확인 |
