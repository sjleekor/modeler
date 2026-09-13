"""Model-agnostic ETL layer over the Parquet lake (L1 -> L2a).

Reads ``data_lake/{raw,canonical}_postgres`` parquet via DuckDB and never
connects to PostgreSQL. See ``docs/target/01_20_access_return_rank/etl_03_implementation_plan.md``.
"""
