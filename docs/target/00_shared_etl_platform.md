# 멀티모델 ETL 플랫폼 설계 (공통 계층)

- 작성 일시: 2026-06-19
- 위치: `docs/target/00_shared_etl_platform.md` (모든 모델 디렉토리의 상위 공통 계획)
- 적용 대상: `docs/target/<NN>_<slug>/` 아래 모든 모델 ETL (현재 `01_20_access_return_rank`, 이후 추가될 모델 전부)
- 전제: 소스는 exporter가 출력한 Parquet lake([`01_20_access_return_rank/etl_01_parquet_data_flow_plan.md`](./01_20_access_return_rank/etl_01_parquet_data_flow_plan.md) §0.5), 엔진은 DuckDB 메인 + Polars 보조([`.../etl_02_engine_comparison.md`](./01_20_access_return_rank/etl_02_engine_comparison.md)).

---

## 0. 왜 이 문서가 필요한가 (문제)

지금까지의 계획(`01_20_access_return_rank/*`)은 **단일 모델**만 다룬다. 앞으로 모델이 여러 개 추가되는데(예: 단기 reversal, 변동성, 섹터 중립 롱숏, 분류 모델 …), 현재 구조를 그대로 복제하면:

| 문제 | 구체 위험 |
|---|---|
| **로직 중복** | 라벨(forward/excess)·유니버스 필터·PIT asof를 모델마다 재작성 → PIT 누설 버그가 모델 수만큼 복제 |
| **정의 불일치** | 모델 A의 `ret_20d`와 모델 B의 `ret_20d`가 미묘하게 달라짐 → 비교 불가, 디버깅 지옥 |
| **재계산 비용** | 모델마다 2.2GB 수급 dedup(Q1)·시세 self-join을 다시 스캔 → snapshot당 N배 낭비 |
| **재현 불가** | "이 모델은 어느 snapshot·어떤 피처·어떤 라벨로 학습됐나"가 기록되지 않음 |

**결론: Parquet lake와 per-model 데이터셋 사이에 "모델 무관 공통 피처 mart 계층"을 두고, 라벨·전처리·분할·레지스트리를 공유 컴포넌트로 만든다.** 그러면 새 모델 추가 = "어떤 피처 그룹을 쓸지 고르고, 라벨 1개 정의하고, 유니버스 필터·분할만 지정"으로 축소된다.

---

## 1. 계층 아키텍처 (mart 계층 추가)

```text
[L0] PostgreSQL mydb
      │ exporter
      ▼
[L1] Parquet lake                       (snapshot_date 핀, 모델 무관)
      ├─ raw_postgres/        (bronze: 원천 13종)
      └─ canonical_postgres/  (silver: 정규화 fact — stock_metric_fact, common_feature_daily_fact …)
      │
      │  ===== 공통 ETL (모델 무관, snapshot당 1회 계산·물질화) =====
      ▼
[L2a] feature marts          data_lake/feature_mart/snapshot_date=<…>/      ★신규 공통 계층
      ├─ dim_trading_calendar   (거래일 인덱스 d_idx — 모든 forward/embargo의 기준)
      ├─ dim_universe_daily     (t별 유니버스 + 필터 플래그, etl_01 §1.2)
      ├─ feat_price/            (시세 파생 — ret/mom/vol/turnover/amihud…)
      ├─ feat_flow/             (수급 dedup 후 파생 — 무거운 Q1을 여기서 1회)
      ├─ feat_fin_pit/          (재무 PIT asof 결과)
      ├─ feat_common/           (시장·거시 broadcast, 가용 구간)
      └─ feat_event/            (배당·자사주 플래그)
      │
      │  ===== per-model 조립 (모델별, mart 위 가벼운 join) =====
      ▼
[L2b] model datasets         data/datasets/<NN>_<slug>/snapshot_date=<…>/   (gold)
      ├─ label_daily/          (모델 고유 라벨 — label 라이브러리로 생성)
      ├─ feat_panel(_std)/     (선택한 mart 피처 그룹 join + per-date 표준화)
      ├─ split_folds/          (walk-forward + embargo)
      └─ dataset_manifest.json (재현 메타: snapshot, 피처 그룹, 라벨 spec, code rev)
```

핵심 원칙:

- **무거운 연산은 L2a에서 모델 무관으로 1회.** `etl_02` 벤치마크의 Q1(2.2GB dedup)·Q2(self-join)·Q4(PIT asof)는 mart 빌드에서 한 번 돌고, 모델들은 그 결과 Parquet을 가볍게 join한다.
- **mart는 snapshot_date로 핀.** 같은 snapshot이면 모델들이 동일 mart를 공유(캐시 재사용). snapshot 바뀌면 mart 재빌드.
- **L2a는 표준화 전(raw) 피처까지만.** per-date winsorize/zscore는 fold 누설 방지 위해 모델·fold별이므로 L2b에 남긴다(`etl_00` §4.3).

---

## 2. 디렉토리·네이밍 규약 (모든 모델 공통)

```text
docs/target/
  00_shared_etl_platform.md          # 본 문서
  README.md                          # 모델 인덱스(표)
  <NN>_<slug>/                       # 모델별 설계 (NN=2자리 순번, slug=영문 kebab)
    prediction_target_*.md           # 예측 대상
    etl_*.md                         # ETL 설계
data_lake/
  raw_postgres/ canonical_postgres/  # exporter 출력 (L1)
  feature_mart/snapshot_date=<…>/    # 공통 mart (L2a)  ★신규
data/datasets/<NN>_<slug>/snapshot_date=<…>/   # per-model (L2b)
```

- **모델 ID = `<NN>_<slug>`** (디렉토리·데이터셋·매니페스트에서 동일 키). 예: `01_20_access_return_rank`.
- **피처 컬럼 prefix 규약**(`etl_00` §4.5 확장, 전 모델 공통): `px_`(시세) `flow_`(수급) `fin_`(재무) `ev_`(이벤트) `cf_`(공통). mart 산출 단계부터 prefix를 박아 모델 간 ablation 토글이 그룹 단위로 가능.
- `data_lake/`·`data/datasets/`는 `.gitignore`(산출물). 설계 문서·코드만 git 추적.

---

## 3. 공통 재사용 컴포넌트 (라이브러리)

모델별로 베끼지 말고 함수/설정으로 공유한다. 위치는 §6.

### 3.1 Label 라이브러리 (가장 중요 — 모델 수 = 라벨 수)

라벨은 모델마다 다르지만 **생성기는 하나**여야 한다. 파라미터화:

```text
make_label(prices, horizon=H, kind={excess|abs}, bench={eqw_market|index},
           outputs={reg, rank, cls}, winsor=[0.005,0.995])
  → d_idx forward join (dim_trading_calendar) → 시장 벤치 차감 → per-date rank/cls
```

- `etl_00` §2의 20d excess return rank는 이 함수의 `H=20, kind=excess, outputs=all`인 한 인스턴스. 5d/60d·변동성·MDD도 같은 함수의 다른 파라미터.
- forward/embargo는 **반드시 `dim_trading_calendar.d_idx` 기준**(캘린더 결측·정지일 흡수, `etl_00` 체크리스트).

### 3.2 Feature builder (mart 빌더)

`feat_price/flow/fin_pit/common/event`를 만드는 빌더 함수 군. 각 빌더는 (L1 lake, snapshot_date) → mart parquet. **DuckDB SQL로 작성**(`etl_02` §5.1). 수급 KRX-우선 dedup·재무 `period_end+90d` PIT asof 등 까다로운 로직을 여기 단일 소스로 고정.

### 3.3 전처리·분할 헬퍼 (선형/트리 공용)

- `pit_asof_join(...)` — DuckDB `ASOF JOIN` 래퍼
- `per_date_winsorize / signed_log / per_date_zscore` — Polars, fold-aware(fit 범위 인자), `etl_00` §4.3
- `add_isna_flags(...)` — 선형모델 결측 처리(L1), `etl_00` §4.2
- `walk_forward_splits(horizon, embargo, scheme)` — purged WF, `etl_00` §5
- `rank_ic / top_decile_spread / icir` — 평가지표, `etl_00` §6

> 모델 패밀리별 차이(선형은 표준화·결측0 필수, 트리는 결측 native·표준화 불필요)는 **전처리 프로파일 토글**로 흡수: `preprocess(profile={linear|tree})`.

### 3.4 Feature registry (피처 사전 — 정의 1회)

전 모델 공통 `feature_registry.yaml`(또는 parquet): `feature_code, group, dtype, pit_rule, source_mart, transform, description`. `etl_00` §7의 모델별 `feature_dictionary.md`는 이 레지스트리의 부분집합 뷰로 생성. → "ret_20d가 정확히 뭔지"를 한 곳에서 정의.

---

## 4. 재현성 — dataset_manifest (모델마다 필수)

각 per-model 데이터셋은 빌드 시 매니페스트를 남긴다:

```json
{
  "model_id": "01_20_access_return_rank",
  "snapshot_date": "2026-06-19",
  "lake": {"raw": "data_lake/raw_postgres/...", "canonical": "...", "feature_mart": "..."},
  "feature_groups": ["px", "flow"],
  "label_spec": {"horizon": 20, "kind": "excess", "bench": "eqw_market", "outputs": ["reg","rank","cls"]},
  "universe_filter": {"min_liquidity_krw": 1e8, "warmup_days": 40},
  "period": {"start": "2015-01-02", "end": "2026-06-10"},
  "code_rev": "<git sha>",
  "row_count": 0
}
```

→ "이 모델 = 이 snapshot + 이 mart + 이 라벨 spec"이 1파일로 고정. 모델 비교·감사·재빌드의 단일 진실원.

---

## 5. 캐시·증분 정책

- **mart는 snapshot_date 단위 캐시.** 빌더는 대상 mart 디렉토리가 있으면 skip(`--force`로 재빌드). exporter 스크립트(§raw/canonical)와 동일한 멱등 패턴.
- 모델 N개가 같은 snapshot을 공유하면 **mart 빌드는 1회**, 이후 모델 추가는 L2b 조립만(가벼움).
- snapshot이 갱신되면(exporter 재실행) 새 `snapshot_date=` 디렉토리에 mart·데이터셋을 새로 쌓는다(과거 snapshot 불변 → 재현성).

---

## 6. 공통 코드 위치 (제안)

ETL은 **DB가 아니라 Parquet을 읽는 연구용 파이프라인**이라, 헥사고날 코어(`src/krx_collector`, `domain/service`가 `adapters/infra` 미import 규약)와 **분리**한다.

- 제안 기본값: 저장소 내 **별도 패키지** `research/etl/`(또는 `analysis/etl/`) — `krx_collector`를 import하지 않고 lake parquet만 의존. `pyproject.toml`에 `research` optional-dependency-group(duckdb, polars, scikit-learn)로 분리.
- 모델별 코드는 `research/models/<NN>_<slug>/`에서 위 공통 라이브러리를 호출.

> 이 위치는 합리적 기본값이며, 기존 `tools/` 패턴을 따르고 싶으면 `tools/research-etl/`도 가능. 확정 전이라면 1차 구현 착수 시 1줄 결정.

---

## 7. `01_20_access_return_rank`를 본 플랫폼에 매핑 (retrofit)

기존 단일모델 계획은 폐기가 아니라 **첫 인스턴스**로 재배치한다:

| `etl_00/01` 산출물 | 플랫폼 계층 | 변화 |
|---|---|---|
| price_features (§3.1) | `feat_price` mart (L2a) | 모델 무관으로 승격 |
| flow_features dedup (§3.2) | `feat_flow` mart (L2a) | 무거운 Q1을 mart에서 1회 |
| fin_features PIT (§3.3) | `feat_fin_pit` mart (L2a) | 〃 (Q4 asof) |
| 유니버스 (§1.2) | `dim_universe_daily` (L2a) | 공유 |
| label 20d excess rank (§2) | label 라이브러리 인스턴스 (L2b) | `make_label(H=20,…)` |
| 표준화·분할·평가 (§4.3,§5,§6) | 공통 헬퍼 (§3.3) | 함수화 |
| 산출 데이터셋 | `data/datasets/01_20_access_return_rank/` + manifest | 매니페스트 추가 |

→ `01`의 ETL을 구현할 때 **mart 빌더 + 라벨 라이브러리 형태로** 짜면, 그 자체가 플랫폼의 첫 구현이 된다(중복 작업 0).

---

## 8. 새 모델 추가 체크리스트 (이 플랫폼의 사용법)

1. [ ] `docs/target/<NN>_<slug>/`에 예측 대상 + ETL 설계 문서 작성, `README.md` 인덱스에 추가
2. [ ] 필요한 **피처 그룹** 선택 (이미 mart에 있으면 재계산 없음). 새 피처면 §3.2 빌더에 추가(전 모델에 재사용됨)
3. [ ] **라벨**을 `make_label(...)` 파라미터로 정의 (새 라벨 종류면 라이브러리 확장)
4. [ ] 유니버스 필터·기간·전처리 프로파일(linear/tree) 지정
5. [ ] `walk_forward_splits(...)`로 분할, 공통 평가지표로 검증
6. [ ] `dataset_manifest.json` 생성(재현성) → 학습

---

## 9. 후속 반영

- `etl_01` §7 산출물 트리에 `feature_mart` 계층과 `dataset_manifest` 추가(본 문서 §1·§4로 연결).
- 1차 구현(`01`) 착수 시 코드를 **mart 빌더 + 라벨/전처리/분할 라이브러리**로 구조화(§7). 단일 스크립트로 짜지 말 것.
