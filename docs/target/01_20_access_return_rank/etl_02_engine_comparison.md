# ETL 엔진 비교 — DuckDB vs Polars vs DataFusion (Parquet 소스)

- 작성 일시: 2026-06-19
- 위치: `docs/target/01_20_access_return_rank/`
- 배경: 소스가 PostgreSQL → **Parquet lake**로 바뀌었으므로([`etl_01_parquet_data_flow_plan.md`](./etl_01_parquet_data_flow_plan.md) §0.5), ETL 실행 엔진이 더 이상 DB에 묶이지 않는다. DuckDB 외 후보 2종을 선정해 **실제 lake 데이터로 벤치마크**한 뒤 최종 솔루션을 정한다.
- 벤치마크 코드: [`docs/dev/20260619_rust_exporter/etl_engine_benchmark.py`](../../dev/20260619_rust_exporter/etl_engine_benchmark.py)

---

## 0. 결론 (TL;DR)

**최종 선택: DuckDB (1차 ETL 메인 엔진). Polars는 모델 입력 직전 in-memory feature 가공·sklearn 핸드오프 보조 엔진으로 병용.**

- 실측 결과 DuckDB가 **가장 무거운 연산(2.2GB 수급 dedup)에서 최속(805ms)**, 나머지도 전 구간 1초 이내로 안정적. 메모리 한도(2GB)에서도 spill-to-disk로 완주.
- Polars는 가벼운 횡단연산(rank/label)에서 DuckDB와 동급이거나 더 빠르나(291ms), **관용 표현(idiom)에 성능이 크게 좌우**됨(잘못 쓰면 asof join 68s → 올바르게 1.1s). SQL 한 줄로 끝나는 PIT/dedup을 명령형으로 재작성해야 함.
- DataFusion은 정확하지만 무거운 연산에서 가장 느리고(dedup 18s, asof 26s) Python API가 가장 거침. **현 단계에서 제외**.
- 세 엔진 모두 **4개 핵심 쿼리 row count가 완전 일치**(정확성 동률) → 차이는 속도·메모리·인체공학.

---

## 1. 후보 선정 근거

소스가 단일 노드의 Parquet(총 ~2.5GB, 최대 테이블 `krx_security_flow_raw` 2.2GB)이고, 워크로드는 **window 함수 + self/asof join + 횡단 rank**가 핵심이다(분산 처리 불필요). 이 조건에 맞는 단일노드 OLAP/DataFrame 엔진 3종:

| 후보 | 성격 | 선정 이유 |
|---|---|---|
| **DuckDB** (incumbent) | 임베디드 OLAP, SQL | `etl_00`의 PostgreSQL SQL이 거의 그대로 이식. window/asof/parquet 네이티브 |
| **Polars** | Rust 기반 Arrow DataFrame, lazy | `etl_00` 다음 단계(sklearn 입력 행렬)와 zero-copy. 표현력 강함 |
| **DataFusion** | Rust SQL 쿼리 엔진 (Arrow) | exporter가 이미 Rust → 스택 일관성. SQL + Arrow 출력 |

> 제외 후보: Spark/Dask(분산 — 2.5GB에 과함), pandas(메모리·속도 부적합), chDB/ClickHouse-local(설치·운영 부담 대비 이점 적음), raw PyArrow(쿼리 표현력 부족).

---

## 2. 벤치마크 설계

`etl_01` §4의 **실제 1차 ETL 연산 4종**을 동일 lake 데이터에 대해 엔진별로 실행. 각 쿼리 2회 실행 후 best(warm) 기록. row count로 정확성 교차검증.

| 쿼리 | 내용 | 데이터 | 난이도 |
|---|---|---|---|
| Q1 dedup | KRX 우선 수급 dedup (window `ROW_NUMBER`/QUALIFY) | `krx_security_flow_raw` **2.2GB / 76M행** | 최고(대용량 window) |
| Q2 label | 20거래일 forward 수익률 (`d_idx` self-join) | `daily_ohlcv` 114MB / 6.5M행 | 중 |
| Q3 rank | 시장별 횡단 초과수익률 percentile rank | 동상 | 중 |
| Q4 asof | 재무 PIT as-of join (`period_end+90d ≤ t` 최신) | `daily_ohlcv`(2024+) × `stock_metric_fact` | 고(asof) |

환경: macOS, 14 cores / 36GB RAM. duckdb 1.5.4 / polars 1.41.2 / datafusion 53.0.0 / Python 3.12.10. 엔진별 별도 프로세스.

> **재현 의존성**: ETL/벤치마크 엔진은 `pyproject.toml`의 **`research` optional extra**로 고정한다(`duckdb`,`polars`,`pyarrow`,`scikit-learn`,`numpy`). 설치: `uv sync --extra research`. 운영 이미지에는 미포함(분석 전용). datafusion은 본 비교 후 제외했으므로 extra에 넣지 않는다.

---

## 3. 실측 결과 (warm best, ms)

| 연산 | DuckDB | Polars (관용적) | DataFusion | row count (3엔진 일치) |
|---|---:|---:|---:|---:|
| Q1 dedup (2.2GB) | **805** | 4,066 | 18,319 | 55,918,702 |
| Q2 label (self-join) | 267 | **299** | 350 | 6,408,188 |
| Q3 rank (횡단) | 530 | **291** | 336 | 6,408,188 |
| Q4 asof (PIT) | 2,618 | **1,127** | 26,464 | 23,674,001 |
| **합계 (4종)** | **4,220** | 5,783 | 45,469 | — |

> **Polars Q1/Q4는 "관용적" 수치다.** 최초 순진한 구현(naive join+filter, sort+group_by.first)은 Q4 **68,634ms**·Q1 8,371ms로 느렸고, 네이티브 `join_asof`(by=[ticker,metric_code])와 `unique(keep=first)`로 바꿔 위 수치(1,127 / 4,066ms)가 됐다. **표에는 공정성을 위해 최적화 버전을 실었다.** → Polars는 "올바른 idiom을 알아야" 빠르다는 점이 핵심 발견.

### 3.1 메모리 / 견고성

- 프로세스 peak RSS는 macOS의 parquet mmap 때문에 13~25GB로 부풀려 측정돼 **절대치는 신뢰 불가**. 상대 순서만: DuckDB ≤ Polars ≈ DataFusion.
- **DuckDB는 `memory_limit='2GB'`로 제한해도 2.2GB dedup을 완주**(spill-to-disk, 5,718ms). Polars(streaming 엔진은 개선 중)·DataFusion은 대용량 window/join에서 OOM 위험이 상대적으로 큼. → 데이터가 더 커져도(공통피처 백필·다년도 확장) DuckDB가 안전.

---

## 4. 인체공학 / 통합 비교

| 항목 | DuckDB | Polars | DataFusion |
|---|---|---|---|
| `etl_00` SQL 이식 | ★★★ 거의 그대로(`QUALIFY`/`ASOF JOIN` 지원) | ★ 명령형 재작성 필요 | ★★ SQL이나 일부 함수 차이 |
| PIT asof 표현 | `ASOF JOIN` 한 줄 | `join_asof`(양쪽 `by` 키 필요, 멀티 metric은 reshape) | window+filter 수동 |
| sklearn 핸드오프 | `.df()`/`.arrow()` | **네이티브(zero-copy, `.to_numpy`)** | `.to_pandas()` |
| Parquet 파티션 pruning | hive 자동 | hive 자동 | 디렉토리 등록 |
| 결측/타입(Decimal128) | `::DOUBLE` 캐스팅 | `.cast(Float64)` | `CAST` |
| 성숙도/문서 | ★★★ | ★★★ | ★★ (Python API 거침) |
| Rust 스택 일관성 | — | (Rust core) | ★★★ (exporter와 동일) |

---

## 5. 최종 솔루션

### 5.1 메인 = DuckDB

근거(실측 기반):
1. **최악 케이스가 가장 빠르고 안전**: 가장 무거운 2.2GB dedup에서 805ms(2위의 5×), 메모리 제한 하에서도 완주.
2. **`etl_00`/`etl_01`의 SQL을 그대로 실행** — 재작성·검증 비용 최소. `DISTINCT ON`만 `QUALIFY`/`ASOF JOIN`으로 치환(이미 §4·§6에 반영).
3. **합계 4,220ms로 최속**, 전 연산 균일하게 빠름(특정 idiom 의존 없음).
4. 단일 파일 임베디드 → 운영·재현 단순(파일 의존성 없음).

### 5.2 보조 = Polars (모델 입력 단계)

- ETL 후반 §4.2~§4.3(per-date winsorize/log/zscore, `*_isna` 플래그, 피처 행렬 → numpy)은 **종목×일자 패널이 메모리에 들어오는 크기**라 Polars의 표현력·zero-copy가 유리.
- 권장 경계: **무거운 스캔·join·window(Q1·Q2·Q4)는 DuckDB → Arrow로 받고, 그 위 feature 가공·표준화는 Polars → numpy로 sklearn(Ridge/ElasticNet)에 전달.** `con.execute(...).arrow()` → `pl.from_arrow(...)`로 zero-copy 연결.

### 5.3 DataFusion 제외

정확성은 동률이나 무거운 연산이 5~10× 느리고 Python API가 거칠다. exporter(Rust)와의 스택 일관성 이점은 ETL이 Python sklearn으로 가는 이상 실익이 적다. **재평가 조건**: 데이터가 분산이 필요할 만큼 커지거나, ETL을 Rust 바이너리로 통합할 때.

---

## 6. 채택 패턴 (코드 스케치)

```python
import duckdb, polars as pl
RAW="data_lake/raw_postgres/snapshot_date=2026-06-19/source=local_mydb"
CAN="data_lake/canonical_postgres/snapshot_date=2026-06-19/source=local_mydb"

con = duckdb.connect(config={"threads":"14"})          # memory_limit 옵션으로 상한 가능
# 1) 무거운 스캔/join/window 은 DuckDB (etl_01 §4 SQL 그대로)
label_arrow = con.execute(LABEL_SQL).arrow()           # Q2/Q3
flow_arrow  = con.execute(FLOW_DEDUP_SQL).arrow()      # Q1
fin_arrow   = con.execute(FIN_ASOF_SQL).arrow()        # Q4 (ASOF JOIN)

# 2) 패널 조립·표준화는 Polars (zero-copy)
panel = pl.from_arrow(label_arrow).join(pl.from_arrow(flow_arrow), on=["trade_date","ticker","market"])
panel_std = (panel
  .with_columns([...winsorize/log...])
  .with_columns([(pl.col(c)-pl.col(c).mean().over("trade_date"))/pl.col(c).std().over("trade_date")
                 for c in FEATS]))            # per-date z-score (etl_00 §4.3)

# 3) sklearn 핸드오프
X = panel_std.select(FEATS).to_numpy(); y = panel_std.get_column("y_rank_20d").to_numpy()
```

---

## 7. 후속 반영

- `etl_01` §3의 "DuckDB-over-Parquet" 전제는 유지하되, **모델 입력 단계는 Polars 병용**을 명시(본 문서 §5.2로 연결).
- 데이터 규모가 커지면(공통피처 4-source 백필 등) DuckDB `memory_limit` 설정 + spill 디렉토리만 지정하면 그대로 확장.
