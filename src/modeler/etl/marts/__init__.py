"""Derived marts that recompute canonical facts from the raw lake in DuckDB.

These replace the Postgres ``metrics normalize`` / ``common build-daily`` compute
steps (refactor plan §3). Each mart reads only the raw (+ config) lake views and
the pure code definitions in ``collector.kr.definitions``; nothing here reads a
canonical Postgres table.
"""
