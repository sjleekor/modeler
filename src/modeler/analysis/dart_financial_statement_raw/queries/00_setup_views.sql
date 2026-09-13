-- 00_setup_views.sql
-- dart_financial_statement_raw 분석용 VIEW 정의.
-- data_lake parquet 스냅샷을 가리키는 경로를 여기 한 곳에만 둔다.
-- 스냅샷 날짜를 바꾸려면 아래 SNAPSHOT 경로만 수정하면 된다.
--
-- 실행: repo 루트에서  duckdb reports/analysis/dart_fs.duckdb < .../00_setup_views.sql
-- (run.sh 가 자동으로 호출함)

CREATE OR REPLACE VIEW fs_raw AS
SELECT *
FROM read_parquet(
    'data_lake/raw_postgres/snapshot_date=2026-06-19/source=local_mydb/dart_financial_statement_raw/**/*.parquet',
    hive_partitioning = true   -- bsns_year, reprt_code 를 컬럼으로 노출
);
