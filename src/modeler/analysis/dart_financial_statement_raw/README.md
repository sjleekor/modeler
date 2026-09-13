# `dart_financial_statement_raw` 분석

DuckDB 로 `dart_financial_statement_raw` 를 탐색하는 ad-hoc 분석 모음.

## 구조

- `queries/00_setup_views.sql` — parquet glob → `fs_raw` VIEW (상대경로). 터미널 `run.sh` 용.
- `queries/00b_setup_views_abs.sql` — 같은 VIEW의 **절대경로** 버전. IntelliJ/DataGrip 용.
- `queries/NN_*.sql` — 분석 쿼리 (커밋 대상, 재현 가능).
- `run.sh` — 전체 쿼리 실행 후 결과를 `reports/` 아래로 출력.

## IntelliJ Database 도구로 열기 (Ultimate 전용)

1. **Database** 패널 → `+` → **Data Source → DuckDB** (드라이버 없으면 IntelliJ가 다운로드 제안).
2. **File/Path**: `reports/analysis/dart_fs.duckdb` 를 **절대경로**로 지정 → Test Connection.
3. VIEW가 깨져 있으면(`No files found...`) 콘솔에서 `queries/00b_setup_views_abs.sql` 1회 실행.
   - 이유: VIEW에 박힌 parquet 경로가 상대경로면 IntelliJ의 working directory에서 못 찾는다. 절대경로 버전이 이를 해결.
4. 이후 `FROM fs_raw` 로 바로 쿼리. `.sql` 파일은 `queries/` 에 두고 콘솔에서 실행하면 결과 그리드/CSV export 사용 가능.

> ⚠️ DuckDB 파일은 **단일 writer**다. IntelliJ가 `dart_fs.duckdb` 를 잡고 있으면 같은 파일에 터미널 `duckdb`/`run.sh` 가 붙을 때 잠금 충돌이 난다. 한쪽을 닫고 쓰거나, IntelliJ 전용으로 별도 `.duckdb` 를 두고 거기서 `00b_...` 를 실행할 것.

## 데이터 소스

`data_lake/raw_postgres/snapshot_date=2026-06-19/source=local_mydb/dart_financial_statement_raw/**`
(2026-06-19 스냅샷, parquet 전수. `bsns_year`/`reprt_code` hive 파티셔닝.)

## 실행

```bash
bash research/analysis/dart_financial_statement_raw/run.sh
```

- 결과: `reports/analysis/dart_financial_statement_raw/<날짜>/NN_*.{md,csv}` (gitignore)
- 인터랙티브: `duckdb reports/analysis/dart_fs.duckdb` → `fs_raw` VIEW 가 이미 살아있음

## 쿼리 목록

| 파일 | 목적 |
| --- | --- |
| `01_nonstandard_account_nm.sql` | C9 후속 — `account_id = '-표준계정코드 미사용-'` 행들의 실제 `account_nm` 분포 |

## 발견 요약

<!-- 새 발견은 여기 누적 -->
- (작성 예정)
