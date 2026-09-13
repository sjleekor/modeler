# ETL 실행 계획 (재정립) — Parquet 데이터 흐름 기준

- 작성 일시: 2026-06-19
- 위치: `docs/target/01_20_access_return_rank/`
- 상위 설계: [`prediction_target_20d_excess_return_rank.md`](./prediction_target_20d_excess_return_rank.md) (예측 대상), [`etl_00_ridge_elasticnet.md`](./etl_00_ridge_elasticnet.md) (선형모델 ETL 설계)
- 데이터 흐름: 로컬 PostgreSQL `mydb` → **Rust raw-parquet-exporter**([`tools/raw-parquet-exporter`](../../../tools/raw-parquet-exporter), [`bin/raw-parquet-export-all.sh`](../../../bin/raw-parquet-export-all.sh)) → **Parquet lake** `data_lake/raw_postgres/snapshot_date=2026-06-19/source=local_mydb/`
- 본 문서의 목적: `etl_00`은 **PostgreSQL 직접 접근**을 전제로 SQL을 작성했다. 실제 학습 ETL은 **Parquet lake**를 읽으므로, (1) 데이터 흐름·접근 방식, (2) 계획이 요구하는데 Parquet에 **없는** 테이블, (3) 그 갭을 메우기 위한 **exporter 수정**을 명시하여 계획을 재정립한다.

> 본 문서는 `etl_00`의 **데이터 접근 계층(§2~§4의 SQL 소스)** 만 갱신한다. 라벨 정의·피처 도메인·선형모델 전처리·walk-forward 분할·평가지표(`etl_00` §0,§2,§3,§4,§5,§6)는 그대로 유효하다.

---

## 0. 결론 (TL;DR)

1. **1차(시세+수급) 실험은 현재 Parquet lake만으로 즉시 실행 가능 — exporter 수정 불필요.**
   `daily_ohlcv`, `krx_security_flow_raw`가 이미 Parquet으로 export되어 있고, 라벨·유니버스·가격피처·수급피처 전부 이 두 테이블에서 나온다.
2. **2차(재무 결합)·3차(공통/거시)에 필요한 canonical 테이블은 raw lake엔 없지만 canonical lake로 export 완료(§5).**
   `etl_00` §3.3은 `stock_metric_fact`, §3.5는 `common_feature_daily_fact`에 의존하는데, 이 두 **canonical 테이블은 raw exporter가 의도적으로 제외**([`raw_parquet_exporter_rust_plan.md`](../../dev/20260619_rust_exporter/raw_parquet_exporter_rust_plan.md) §2.3, "Export 제외")했다. → 별도 **canonical lake**(`data_lake/canonical_postgres/`)로 export 완료(§5, DB와 row count 일치 검증). 즉 `raw_postgres/`엔 없고 `canonical_postgres/`엔 있다.
3. **그 export는 exporter Rust 코드 변경 없이 config 추가만으로 끝났다(완료).**
   타입 매핑(`schema.rs`)이 `numeric→Decimal128`, `jsonb/uuid→::text`, `timestamptz→UTC`를 이미 지원하고, `full_table` 전략이 임의 테이블 + 단순 컬럼 파티션을 지원한다. canonical 5종을 별도 lake 루트로 export해 raw lake의 "strict raw" 원칙도 보존했다(exporter 계획 §2.3 "별도 `derived` export"와 정합).

---

## 0.5 데이터 접근 원칙 (불변 규칙)

> **모든 ETL/학습 작업의 데이터 소스는 exporter가 출력한 Parquet lake(`data_lake/`)이며, PostgreSQL `mydb`를 직접 조회하지 않는다.**

- **PostgreSQL은 exporter(L0→L1) 단계에서만 읽는다.** 학습 ETL(L2) 코드/노트북/쿼리는 `data_lake/raw_postgres/...`(및 §5 적용 후 `data_lake/canonical_postgres/...`)의 Parquet만 읽는다. ETL 레이어에 DB DSN·`psycopg`·`get_settings()` DB 접속을 두지 않는다.
- **이유**: (1) 재현성 — 학습셋은 `snapshot_date=<…>` 디렉토리 핀으로 고정되어, 이후 DB가 갱신돼도 동일 입력이 보장된다. (2) DB 부하 격리 — 대형 scan(수급 76M·시세 6.5M 행)이 수집 DB를 방해하지 않는다(exporter 계획 §3.2와 동일 취지). (3) 이식성 — Parquet은 DB 연결 없이 DuckDB/Polars/Spark 어디서나 읽힌다.
- **계획서의 SQL은 "DB 쿼리"가 아니라 "Parquet 위에서 실행할 논리 명세"다.** `etl_00`/본 문서의 모든 SQL은 §3의 DuckDB 뷰(= Parquet glob) 위에서 실행한다. PostgreSQL 전용 문법(`DISTINCT ON`)은 DuckDB 등가물로 이식한다(§4.2, §6).
- **필요한 테이블이 lake에 없으면 ETL에서 DB로 우회하지 않고, exporter를 확장해 lake에 추가한 뒤 거기서 읽는다**(§2 갭 → §5 canonical export). 이것이 §2.3 캐논 테이블에 대한 유일한 해소 경로다.
- **예외**: exporter 출력 검증(manifest row count ↔ DB count, §5.5)에 한해 DB count 쿼리를 1회 대조용으로 쓴다. 이는 ETL이 아니라 export QA다.

---

## 1. 데이터 흐름 3계층

```text
[L0] 로컬 PostgreSQL  mydb  (sj2-server krx_data 의 미러, 23 public 테이블)
        │  bin/raw-parquet-export-all.sh  (Rust raw-parquet-exporter)
        ▼
[L1] Parquet lake  data_lake/.../snapshot_date=<YYYY-MM-DD>/source=local_mydb/
        │  raw_postgres/        — raw/reference 13개 테이블
        │  canonical_postgres/  — canonical 5개 (stock_metric_fact 등, §5)
        │  <table>/schema_version=1/<partition...>/part-*.parquet
        ▼
[L2] 학습 ETL  (DuckDB/Polars over Parquet → feature/label 패널 → Ridge/ElasticNet)
        →  data/datasets/01_20_access_return_rank/  (feat/label/split parquet)
```

- **L0 → L1 변환 규약**(exporter, 의미 변환 없음): `date→Date32`, `timestamptz→Timestamp(µs,UTC)`, `numeric(p,s)→Decimal128(p,s)` (float 변환 금지), `jsonb/uuid→::text`(LargeUtf8). 메타 컬럼(`__extract_*`)은 **주입하지 않음**(현재 구현 확인). 즉 Parquet 컬럼 = DB 컬럼 그대로.
- **L1 snapshot 의미**: `snapshot_date`는 export 실행일(=DB 미러 상태 고정 시점)이며 데이터 날짜가 아니다. 학습 재현성은 이 디렉토리 핀으로 보장.

---

## 2. 계획 의존 테이블 ↔ Parquet 가용성 매핑 (핵심)

`snapshot_date=2026-06-19/source=local_mydb` 기준 lake는 두 루트로 나뉜다.

- **raw lake** `data_lake/raw_postgres/` — 13개: `daily_ohlcv`, `krx_security_flow_raw`, `dart_xbrl_fact_raw`, `dart_financial_statement_raw`, `dart_shareholder_return_raw`, `dart_share_count_raw`, `dart_xbrl_document`, `dart_corp_master`, `stock_master`, `stock_master_snapshot`, `stock_master_snapshot_items`, `common_feature_observation_raw`, (`operating_source_document`=schema-only).
- **canonical lake** `data_lake/canonical_postgres/` — 5개(§5에서 export 완료): `stock_metric_fact`, `common_feature_daily_fact`, `metric_catalog`, `metric_mapping_rule`, `common_feature_catalog`.

아래 "lake" 열은 어느 루트에 있는지를 나타낸다(전부 가용).

| ETL 단계 (`etl_00`) | 필요 소스 테이블 | 성격 | lake | 조치 |
|---|---|---|:--:|---|
| §1.2 유니버스 (현행) | `daily_ohlcv` | raw | raw ✅ | 즉시 |
| §1.2 PIT 유니버스 (이상) | `stock_master_snapshot(_items)` | ref | raw ✅ | 즉시 (이력 2026-04~06 한정은 동일 제약) |
| §2 라벨 (forward/excess) | `daily_ohlcv` | raw | raw ✅ | 즉시 |
| §3.1 가격 피처 | `daily_ohlcv` | raw | raw ✅ | 즉시 |
| §3.2 수급/공매도 피처 | `krx_security_flow_raw` | raw | raw ✅ | 즉시 (dedupe는 ETL에서) |
| §3.4 이벤트 피처 | `dart_shareholder_return_raw`, `dart_share_count_raw` | raw | raw ✅ | 즉시 (raw 기반 플래그 피처) |
| **§3.3 재무 피처** | **`stock_metric_fact`** | **canonical** | **canonical ✅** | export 완료 (§5) |
| §3.3 (재무 매핑 해석) | `metric_catalog`, `metric_mapping_rule` | catalog | canonical ✅ | export 완료 (§5) — 선택 |
| **§3.5 공통/거시 피처** | **`common_feature_daily_fact`** | **canonical** | **canonical ✅** | export 완료 (§5) |
| §3.5 (공통 피처 해석) | `common_feature_catalog` | catalog | canonical ✅ | export 완료 (§5) — 선택 |
| §3.3 PIT 정확도 개선 | `dart_financial_statement_raw`(`rcept_no`) | raw | raw ⚠️ | raw에 `rcept_dt` 없음(`rcept_no`만) → payload 파싱/export 확장 선행 (§6) |

**결론**: §3.3·§3.5의 canonical 2종은 raw lake엔 없지만 **canonical lake로 export 완료**되어 모든 단계의 소스가 lake에 갖춰졌다(§5). raw에서 직접 재정규화(아래 대안)는 비용이 크고 불필요.

> 대안(비권장 v1): `stock_metric_fact`를 export하지 않고 `dart_financial_statement_raw`+`dart_xbrl_fact_raw`에 `metric_mapping_rule` 정규화 로직을 ETL에서 재구현. → DB 정규화 계층을 중복 구현하는 셈이라 비효율. canonical export가 단순·정확·이미 NULL 0% 정제 완료.

---

## 3. Parquet lake 접근 방식 (ETL 엔진)

> **엔진 결정**: DuckDB vs Polars vs DataFusion를 실제 lake 데이터로 벤치마크한 결과 **메인 엔진 = DuckDB, 보조 = Polars(모델 입력 단계)** 로 확정했다. 근거·실측치는 [`etl_02_engine_comparison.md`](./etl_02_engine_comparison.md). 요지: 가장 무거운 2.2GB 수급 dedup에서 DuckDB가 최속(805ms)·메모리 제한 하에서도 spill로 완주하고, `etl_00` SQL을 거의 그대로 이식 가능. per-date 표준화·sklearn 핸드오프는 Polars로 zero-copy 연결(§3 하단·`etl_02` §6).

`etl_00`의 PostgreSQL SQL을 **DuckDB-over-Parquet**로 옮긴다(SQL이 거의 그대로 이식되고, glob/파티션 pruning·ASOF JOIN을 native 지원).

```python
import duckdb
LAKE = "data_lake/raw_postgres/snapshot_date=2026-06-19/source=local_mydb"
con = duckdb.connect()
# 테이블별 뷰: hive_partitioning=false (원천 컬럼 보존)
con.execute(f"""
  CREATE VIEW daily_ohlcv AS
    SELECT * FROM read_parquet('{LAKE}/daily_ohlcv/**/*.parquet', hive_partitioning=false);
  CREATE VIEW krx_flow AS
    SELECT * FROM read_parquet('{LAKE}/krx_security_flow_raw/**/*.parquet', hive_partitioning=false);
""")
```

> ⚠️ **`hive_partitioning=true` 금지 (검증된 치명 버그)**: 경로의 **path partition 값이 동일 이름의 원천 데이터 컬럼을 덮어쓴다.** `krx_security_flow_raw`는 lake 경로가 `.../source=local_mydb/...`인데 raw 테이블에도 `source` 컬럼(`KRX`/`PYKRX`)이 있어서, `hive=true`로 읽으면 `source`가 전부 `local_mydb`가 된다(실측: hive=true → `source=local_mydb` 76,536,572행 단일값 / hive=false → `KRX` 55,908,238 + `PYKRX` 20,628,334). 이 상태면 §4.2의 `CASE source WHEN 'KRX'` dedupe가 **무력화**된다. → **항상 `hive_partitioning=false`로 읽고, 파티션 pruning이 필요하면 실제 데이터 컬럼(`trade_date` 등)으로 필터**한다(`year=/month=` 경로 자체가 `trade_date` 범위와 일치).

타입 주의(L1 규약 기인):

- `numeric` 컬럼(`daily_ohlcv` 가격은 BIGINT지만 `krx_security_flow_raw.value`, 재무 금액 등)은 **Decimal128**로 읽힌다 → 비율·로그 연산 전 `CAST(... AS DOUBLE)` 권장(오버플로/정밀도). 단 winsorize/로그 후 float화는 `etl_00` §4.3 순서 유지.
- `jsonb`/`uuid`는 **문자열**(`::text`)로 들어온다. 1차 피처엔 미사용이므로 무시.
- `trade_date`는 `Date32`. 필터는 `trade_date`로 건다(hive=false이므로 `year=/month=` 파티션 컬럼은 노출되지 않으며, DuckDB는 경로 기반 pruning을 자동 적용).

DuckDB 이식 시 PostgreSQL 문법 차이 1건: **`DISTINCT ON` 미지원** → `QUALIFY ROW_NUMBER() OVER(...) = 1` 또는 `ASOF JOIN`으로 대체(§4.2, §6).

---

## 4. 1차 실험 (시세 + 수급) — exporter 수정 없이 즉시 실행

대상 기간 2015-01-02 ~ 2026-06-10, KOSPI/KOSDAQ. 아래는 `etl_00`의 핵심 SQL을 Parquet/DuckDB로 이식한 것.

### 4.1 라벨 — 거래일 인덱스 forward + 시장 동일가중 초과수익률 (`etl_00` §2)

```sql
WITH px AS (
  SELECT trade_date, ticker, market, close,
         ROW_NUMBER() OVER (PARTITION BY ticker, market ORDER BY trade_date) AS d_idx
  FROM daily_ohlcv
  WHERE NOT (open=0 AND high=0 AND low=0)          -- 정지일 close 왜곡 방지
),
fwd AS (
  SELECT a.trade_date, a.ticker, a.market,
         a.close::DOUBLE AS close_t,
         f.close::DOUBLE / NULLIF(a.close,0) - 1 AS fwd_ret_20d
  FROM px a
  JOIN px f                                         -- 동일 ticker/market 의 정확히 20거래일 후
    ON f.ticker = a.ticker AND f.market = a.market
   AND f.d_idx  = a.d_idx + 20
),
mkt AS (
  SELECT trade_date, market, AVG(fwd_ret_20d) AS mkt_ret_20d
  FROM fwd GROUP BY 1,2
)
SELECT f.trade_date, f.ticker, f.market, f.fwd_ret_20d, m.mkt_ret_20d,
       f.fwd_ret_20d - m.mkt_ret_20d AS excess_ret_20d
FROM fwd f JOIN mkt m USING (trade_date, market);
```

보조 horizon은 `d_idx+5`, `d_idx+60`으로 동일 패턴. 랭킹/분류 라벨(`y_rank_20d`, `y_cls_20d`)은 `etl_00` §2.2 그대로 (`PERCENT_RANK() OVER (PARTITION BY trade_date, market ...)`).

### 4.2 수급 dedupe — KRX 우선 (`etl_00` §3.2, `DISTINCT ON` → `QUALIFY` 이식)

```sql
CREATE VIEW krx_flow_dedup AS
SELECT trade_date, ticker, market, metric_code, value::DOUBLE AS value
FROM krx_flow
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY trade_date, ticker, market, metric_code
  ORDER BY CASE source WHEN 'KRX' THEN 0 ELSE 1 END   -- KRX 우선 (값 충돌 0)
) = 1;
```

이후 metric_code 7종 wide pivot → 누적·비율·z-score 파생은 `etl_00` §3.2 그대로. 공매도 잔고(2016-06-30~) 결측은 `*_isna` 플래그(L1).

### 4.3 가격 피처 (`etl_00` §3.1)

`daily_ohlcv` 단일 소스. ret/mom/vol/turnover/amihud/`is_halted` 등 전부 Parquet에서 산출. `close*volume` turnover는 `close::DOUBLE * volume::DOUBLE`로 캐스팅 후 `log1p`.

### 4.4 1차 산출물

```text
universe_daily, label_daily(+5d/60d/vol), price_features, flow_features
→ feat_daily_panel → feat_daily_panel_std → split_folds   (etl_00 §7, §8 Step1~9)
```

1차는 위 4개 소스 뷰(`daily_ohlcv`, `krx_flow`)만으로 완결된다. **추가 export 불필요.**

---

## 5. Exporter 수정 (2차/3차에 필요) — config-only, Rust 변경 없음 ✅ 구현·검증 완료

> **상태(2026-06-19): 구현 및 검증 완료.** canonical 5종을 별도 lake 루트로 export했고, manifest row count가 DB와 정확히 일치함(§5.6). Rust 코드 변경 없음.

### 5.1 무엇이 왜 빠졌나

[`raw_parquet_exporter_rust_plan.md`](../../dev/20260619_rust_exporter/raw_parquet_exporter_rust_plan.md) §2.3는 `stock_metric_fact`·`common_feature_daily_fact`를 "raw를 정규화한 canonical"이라는 이유로 export 대상에서 명시 제외했다. 의도(raw lake의 순수성)는 타당하나, 학습 ETL은 이 canonical 값을 직접 쓰므로 **별도 canonical export 경로**가 필요하다(같은 문서 §2.3가 이미 "별도 `derived` export"를 예고).

### 5.2 Rust 코드 변경이 필요 없는 이유 (확인됨)

- 타입: `schema.rs::arrow_type_for_pg`가 `numeric→Decimal128(p,s)`, `jsonb|uuid→LargeUtf8`, `timestamptz→Timestamp(µs,UTC)`를 이미 매핑. `export.rs::select_expr`가 `jsonb|numeric|uuid`를 `::text`/그대로 SELECT. → 두 테이블의 모든 컬럼 타입(`numeric(30,4)`, `numeric(30,8)`, `jsonb`, `uuid`, `timestamptz`)이 이미 커버됨.
- 전략: `full_table`이 임의 테이블 + **단순 소스 컬럼 파티션**(`["bsns_year"]`, `["source"]` 등)을 지원(README, `dart_corp_master`·`common_feature_observation_raw`가 이미 사용). `stock_metric_fact.bsns_year`(INT NOT NULL)는 단순 파티션 OK. `common_feature_daily_fact`는 무파티션(5.5K행)으로 충분.

즉 **새 config 항목 + export 스크립트 목록 추가**만으로 끝난다.

### 5.3 추가할 테이블

| 테이블 | 행수(2026-06-15) | 전략 | 파티션 | 용도 |
|---|---:|---|---|---|
| `stock_metric_fact` | 765,966 | `full_table` | `["bsns_year"]` | §3.3 재무 피처 (canonical, NULL 0%) |
| `common_feature_daily_fact` | 5,550 | `full_table` | `[]` | §3.5 공통/거시 피처 (PIT-safe) |
| `metric_catalog` | 29 | `full_table` | `[]` | metric_code → 의미/단위 해석 (선택) |
| `metric_mapping_rule` | 59 | `full_table` | `[]` | 매핑 규칙 추적 (선택) |
| `common_feature_catalog` | 54 | `full_table` | `[]` | feature_code 해석/transform (선택) |

> `stock_metric_fact`는 `fact_id`(BIGSERIAL PK)가 있어 `raw_id_range`도 가능하나, 766K행이면 `full_table`이 단순·충분.

### 5.4 적용 방법 — raw lake 오염 없이 별도 canonical lake (구현됨)

raw lake의 "strict raw" 원칙을 보존하기 위해 **별도 config + 별도 runtime(출력 루트) + 별도 실행 스크립트**를 추가했다. (exporter의 출력 루트는 CLI 플래그가 아니라 runtime toml의 `[output].root`로만 정해지므로, 환경변수가 아닌 **전용 runtime toml**이 필요하다.)

추가된 파일:

| 파일 | 역할 |
|---|---|
| [`tools/raw-parquet-exporter/config/export_canonical_tables.toml`](../../../tools/raw-parquet-exporter/config/export_canonical_tables.toml) | canonical 5종 export 정의 (모두 `full_table`) |
| [`tools/raw-parquet-exporter/config/canonical.example.toml`](../../../tools/raw-parquet-exporter/config/canonical.example.toml) | runtime: `[output].root = data_lake/canonical_postgres`, tmp 분리 |
| [`bin/canonical-parquet-export-all.sh`](../../../bin/canonical-parquet-export-all.sh) | 5종을 도는 실행 스크립트 (`raw-parquet-export-all.sh` 패턴, `SDC_CANON_*` 환경변수) |

실행:

```bash
bin/canonical-parquet-export-all.sh --snapshot-date 2026-06-19 --force
# 빌드 생략(이미 release 빌드된 경우): --no-build
```

결과 경로:
`data_lake/canonical_postgres/snapshot_date=2026-06-19/source=local_mydb/stock_metric_fact/schema_version=1/bsns_year=2024/part-*.parquet` 등. `stock_metric_fact`는 `bsns_year=2015..2026` 파티션으로 분할됨(확인됨).

ETL은 raw + canonical 두 루트를 함께 glob:

```python
RAW = "data_lake/raw_postgres/snapshot_date=2026-06-19/source=local_mydb"
CAN = "data_lake/canonical_postgres/snapshot_date=2026-06-19/source=local_mydb"
con.execute(f"CREATE VIEW stock_metric_fact AS SELECT * FROM read_parquet('{CAN}/stock_metric_fact/**/*.parquet', hive_partitioning=false)")
con.execute(f"CREATE VIEW common_feature_daily_fact AS SELECT * FROM read_parquet('{CAN}/common_feature_daily_fact/**/*.parquet', hive_partitioning=false)")
```

> Rust 코드는 한 줄도 바뀌지 않았다(config + 스크립트만 추가). `cargo run -- plan`이 5종을 모두 정상 plan(`columns=18`/`columns=11` 등 introspection 성공)했다.

### 5.5 검증 (완료)

`canonical-parquet-export-all.sh`는 export 직후 각 테이블 manifest를 `validate`로 검증한다(`manifest_rows == parquet_rows`, 전부 `passed`). 추가로 DB `SELECT count(*)`와 대조한 결과 **5종 전부 정확 일치**:

| 테이블 | DB rows | manifest rows | 일치 |
|---|---:|---:|:--:|
| `stock_metric_fact` | 766,053 | 766,053 | ✅ |
| `common_feature_daily_fact` | 5,661 | 5,661 | ✅ |
| `metric_catalog` | 29 | 29 | ✅ |
| `metric_mapping_rule` | 59 | 59 | ✅ |
| `common_feature_catalog` | 54 | 54 | ✅ |

> 행수가 §5.3 표(2026-06-15 기준 765,966 / 5,550)보다 소폭 큰 것은 그 사이 수집/백필 증분이며 정상이다. Decimal128(`value_numeric`) round-trip spot check는 ETL §3.3 금액 비율 계산 시 함께 확인한다.

---

## 6. PIT 정확도 개선 — raw가 lake에 있으니 실제 공시일 사용 (선택)

`etl_00` §3.3은 공시일 컬럼 부재로 `period_end + 90d` 보수 래그를 썼다. `dart_financial_statement_raw`가 raw lake에 export되어 있으나 **컬럼은 `rcept_no`뿐이고 접수일 `rcept_dt`는 없다(검증됨).** 따라서 실 공시일 기반 PIT로 정밀화하려면 `rcept_no` → 접수일을 얻는 작업(payload 파싱 또는 export 확장)이 선행되어야 하며, 1차에서는 `period_end+90d`를 그대로 쓴다.

> ⚠️ **단순 `ASOF JOIN` 금지 — metric_code 차원 누락(검증됨)**: 재무는 종목당 여러 `metric_code`(`metric_catalog` 전체 29종 중 해당 종목이 보유한 ~26종, ROA·부채비율 등)를 동시에 가져와야 한다. DuckDB `ASOF JOIN`은 **왼쪽 1행당 오른쪽 1행만** 반환하므로, universe 1행에 metric 1개만 붙어 나머지 25개가 사라진다(실측: 삼성전자 2024-06-03 기준 단순 ASOF=1행 vs metric별 window-join=26행). 따라서 **`metric_code`를 partition에 포함한 일반 JOIN + `ROW_NUMBER()`**로 써야 한다(`etl_00` §3.3 `DISTINCT ON` 이식과 동일).

```sql
-- t 시점에 가용한, metric_code별 최신 보고서 값 (PIT as-of, 전 metric 보존)
SELECT trade_date, ticker, metric_code, value FROM (
  SELECT u.trade_date, u.ticker, f.metric_code, f.value_numeric::DOUBLE AS value,
         ROW_NUMBER() OVER (PARTITION BY u.trade_date, u.ticker, f.metric_code
                            ORDER BY f.available_from DESC, f.period_end DESC) AS rn
  FROM universe u
  JOIN stock_metric_fact f
    ON f.ticker = u.ticker
   AND f.available_from <= u.trade_date    -- available_from := period_end + 90d (분기 +45d)
) WHERE rn = 1;
```

1차는 `available_from := period_end+90d`(분기 +45d)로 시작한다.

> **rcept_dt 기반 정밀화는 "즉시 가능"이 아님(검증됨)**: 실제 lake의 `dart_financial_statement_raw`·`dart_xbrl_document` 컬럼에는 **`rcept_dt`가 없고 `rcept_no`만 있다.** 따라서 실 공시일(접수일) 기반 PIT로 가려면 (a) `raw_payload` JSON에서 접수일을 파싱하거나, (b) exporter에 `rcept_dt`를 포함하는 별도 컬럼/소스를 추가하는 작업이 **선행**되어야 한다. 2차 정밀화 과제로 둔다(`etl_00` §3.3 주석과 동일 방향).

---

## 7. 산출물 디렉토리

> **멀티모델 주의**: 본 모델은 단일 모델이 아니라 [`00_shared_etl_platform.md`](../00_shared_etl_platform.md)의 **첫 인스턴스**다. 아래 `feat_*`/`label_*`는 모델 무관 **feature mart(L2a)** 와 모델별 **dataset(L2b)** 로 분리된다(플랫폼 §1·§7). 무거운 수급 dedup·재무 PIT(§3.2·§3.3)는 mart에서 1회 계산해 후속 모델과 공유한다. 구현 시 단일 스크립트가 아니라 **mart 빌더 + 라벨/전처리 라이브러리**로 구조화한다.

```text
data_lake/
  raw_postgres/snapshot_date=2026-06-19/source=local_mydb/          # 기존 (raw 13종)
  canonical_postgres/snapshot_date=2026-06-19/source=local_mydb/    # 신규 (canonical 5종, §5)
  feature_mart/snapshot_date=2026-06-19/                            # 공통 mart (L2a, 플랫폼 §1) — 모델 무관
    dim_universe_daily/  feat_price/  feat_flow/  feat_fin_pit/ …
data/datasets/01_20_access_return_rank/snapshot_date=2026-06-19/    # 본 모델 dataset (L2b)
  feat_panel/  feat_panel_std/  label_daily/  split_folds/
  corr_report/ vif_report/  feature_dictionary.md  dataset_manifest.json  # 재현성(플랫폼 §4)
```

`data_lake/`는 `.gitignore` 대상(exporter 계획 §4). `data/datasets/`도 산출물이므로 ignore 권장.

---

## 8. 실행 순서 / 체크리스트

**1차 (지금 가능, exporter 수정 없음)**
- [ ] `snapshot_date=2026-06-19` raw lake 존재 확인 (`daily_ohlcv`, `krx_security_flow_raw`)
- [ ] DuckDB 뷰 생성 (§3) → 라벨/유니버스/가격/수급 피처 (§4)
- [ ] `feat_daily_panel(_std)`, `label_daily`, `split_folds` 산출 (`etl_00` Step1~9)
- [ ] Ridge/ElasticNet 학습·walk-forward 검증, Rank IC 리포트 (`etl_00` §5,§6)

**2차/3차 준비 (exporter 수정)**
- [ ] `export_canonical_tables.toml` 추가 (§5.4) — Rust 코드 변경 없음
- [ ] canonical export 실행 → `data_lake/canonical_postgres/...` 생성
- [ ] manifest row count ↔ DB count 대조 (§5.5)
- [ ] ETL에 `stock_metric_fact`(§3.3), `common_feature_daily_fact`(§3.5) 뷰 결합
- [ ] (선택) `rcept_dt` 기반 PIT as-of로 정밀화 (§6)

---

## 9. `etl_00` 대비 변경 요약

| 항목 | `etl_00` 전제 | 본 문서 (재정립) |
|---|---|---|
| **데이터 소스 (불변 규칙)** | PostgreSQL 직접 | **Parquet lake(`data_lake/`)만. DB 직접 조회 금지** (§0.5) |
| §3.3 재무 (`stock_metric_fact`) | DB에서 바로 조회 가능 가정 | **Parquet에 없음** → exporter canonical 확장 필요 (§5) |
| §3.5 공통 (`common_feature_daily_fact`) | DB에서 바로 조회 가능 가정 | **Parquet에 없음** → exporter canonical 확장 필요 (§5) |
| §3.2 dedupe / as-of join | `DISTINCT ON` | `QUALIFY ROW_NUMBER()` / `ASOF JOIN` (DuckDB) |
| 1차 실행 가능성 | 명시 안 됨 | **시세+수급은 현 lake로 즉시 가능** |
| exporter 수정 | 범위 밖 | canonical 5종 추가(config-only, Rust 불변) |
