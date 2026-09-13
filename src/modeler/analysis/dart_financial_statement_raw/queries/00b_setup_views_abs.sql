-- 00b_setup_views_abs.sql
-- IntelliJ Database 도구 / DataGrip 처럼 working directory 가 repo 루트가 아닌
-- 환경에서 fs_raw 를 쓰기 위한 절대경로 버전.
--
-- run.sh(터미널, cwd=repo 루트)는 상대경로인 00_setup_views.sql 을 쓴다.
-- IntelliJ 에서 .duckdb 를 처음 열었거나 fs_raw 가 "No files found" 로 깨지면,
-- 이 파일을 콘솔에 한 번 실행해 VIEW 를 절대경로로 재정의하면 된다.
--
-- 경로 끝부분(스냅샷 날짜)은 00_setup_views.sql 과 동일하게 유지할 것.

CREATE OR REPLACE VIEW fs_raw AS
SELECT *
FROM read_parquet(
    '/Users/whishaw/wss_p/stock_data_collector/data_lake/raw_postgres/snapshot_date=2026-06-19/source=local_mydb/dart_financial_statement_raw/**/*.parquet',
    hive_partitioning = true
);
