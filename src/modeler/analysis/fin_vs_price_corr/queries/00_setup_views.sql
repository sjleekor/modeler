-- DuckDB view setup for financials vs forward-return correlation EDA.
-- Snapshot pin: 2026-06-19 / source=local_mydb.
--
-- Run from the repository root. `label_daily` is created by
-- `run_fin_price_correlation.py` when it is missing, then this view works.

CREATE OR REPLACE VIEW daily_ohlcv AS
SELECT *
FROM read_parquet(
  'data_lake/raw_postgres/snapshot_date=2026-06-19/source=local_mydb/daily_ohlcv/**/*.parquet',
  hive_partitioning = false
);

CREATE OR REPLACE VIEW dart_financial_statement_raw AS
SELECT *
FROM read_parquet(
  'data_lake/raw_postgres/snapshot_date=2026-06-19/source=local_mydb/dart_financial_statement_raw/**/*.parquet',
  hive_partitioning = false
);

CREATE OR REPLACE VIEW stock_master AS
SELECT *
FROM read_parquet(
  'data_lake/raw_postgres/snapshot_date=2026-06-19/source=local_mydb/stock_master/**/*.parquet',
  hive_partitioning = false
);

CREATE OR REPLACE VIEW dim_universe_daily AS
SELECT *
FROM read_parquet(
  'data_lake/feature_mart/snapshot_date=2026-06-19/dim_universe_daily/**/*.parquet',
  hive_partitioning = false
);

CREATE OR REPLACE VIEW feat_fin_pit AS
SELECT *
FROM read_parquet(
  'data_lake/feature_mart/snapshot_date=2026-06-19/feat_fin_pit/**/*.parquet',
  hive_partitioning = false
);

CREATE OR REPLACE VIEW feat_price AS
SELECT *
FROM read_parquet(
  'data_lake/feature_mart/snapshot_date=2026-06-19/feat_price/**/*.parquet',
  hive_partitioning = false
);

CREATE OR REPLACE VIEW label_daily AS
SELECT *
FROM read_parquet(
  'data_lake/feature_mart/snapshot_date=2026-06-19/label_daily/**/*.parquet',
  hive_partitioning = false
);
