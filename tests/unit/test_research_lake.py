"""Unit tests for the research ETL lake reader (P1).

These build a tiny synthetic lake on disk (no DB, no real ``data_lake/``) that
mimics the exporter layout, then assert the two invariants from etl_01:

1. hive_partitioning=false preserves a real ``source`` data column that
   collides with the path's ``source=`` partition (etl_01 §4.2).
2. Decimal columns are castable to DOUBLE via :func:`cast_double` (etl_01 §3).
"""

from __future__ import annotations

import decimal
from pathlib import Path

import duckdb
import pytest
from modeler.etl import lake
from modeler.etl.config import DataRoot, EngineOptions, LakeConfig


def _write_flow_fixture(lake_root: Path, snapshot_date: str, source: str) -> None:
    """Write a krx_security_flow_raw fixture under the canonical lake layout.

    The fixture deliberately gives rows a real ``source`` column value
    (``KRX``/``PYKRX``) that differs from the path partition (``source=...``),
    so a hive-partitioning regression would be caught.
    """
    table_dir = (
        lake_root
        / "raw"
        / "raw_postgres"
        / f"snapshot_date={snapshot_date}"
        / f"source={source}"
        / "krx_security_flow_raw"
        / "schema_version=1"
        / "year=2020"
        / "month=01"
    )
    table_dir.mkdir(parents=True, exist_ok=True)
    out = (table_dir / "part-000000.parquet").as_posix()

    d = "DATE '2020-01-02'"
    dec = "CAST(1234.5678 AS DECIMAL(30,4))"
    con = duckdb.connect()
    con.execute(f"""
        COPY (
          SELECT * FROM (VALUES
            ({d}, '005930', 'KOSPI', 'foreign_netbuy', {dec}, 'KRX'),
            ({d}, '005930', 'KOSPI', 'foreign_netbuy', {dec}, 'PYKRX'),
            ({d}, '000660', 'KOSPI', 'short_volume', CAST(10.0 AS DECIMAL(30,4)), 'KRX')
          ) AS t(trade_date, ticker, market, metric_code, value, source)
        ) TO '{out}' (FORMAT PARQUET)
        """)
    con.close()


@pytest.fixture()
def synthetic_lake(tmp_path: Path) -> LakeConfig:
    snapshot_date = "2020-01-01"
    source = "local_mydb"
    _write_flow_fixture(tmp_path, snapshot_date, source)
    return LakeConfig(
        root=DataRoot(tmp_path),
        snapshot_date=snapshot_date,
        source=source,
        engine=EngineOptions(threads=2),
    )


def test_register_views_creates_present_table(synthetic_lake: LakeConfig) -> None:
    con = lake.connect(synthetic_lake)
    created = lake.register_views(con, synthetic_lake)

    assert "krx_security_flow_raw" in created
    # Only the fixture table exists on disk; absent tables are skipped silently.
    assert created == ["krx_security_flow_raw"]


def test_hive_partitioning_false_preserves_source_column(synthetic_lake: LakeConfig) -> None:
    """The real KRX/PYKRX source column must survive, not be overwritten by the
    ``source=local_mydb`` path partition (etl_01 §4.2 dedup-killer bug)."""
    con = lake.connect(synthetic_lake)
    lake.register_views(con, synthetic_lake)

    rows = con.execute(
        "SELECT source, count(*) FROM krx_security_flow_raw GROUP BY source ORDER BY source"
    ).fetchall()

    assert rows == [("KRX", 2), ("PYKRX", 1)]
    # And crucially the path partition value never leaks into the data.
    assert "local_mydb" not in {r[0] for r in rows}


def test_value_column_is_decimal_and_casts_to_double(synthetic_lake: LakeConfig) -> None:
    con = lake.connect(synthetic_lake)
    lake.register_views(con, synthetic_lake)

    (raw_value,) = con.execute(
        "SELECT value FROM krx_security_flow_raw WHERE ticker = '005930' LIMIT 1"
    ).fetchone()
    assert isinstance(raw_value, decimal.Decimal)

    (as_double,) = con.execute(
        f"SELECT {lake.cast_double('value')} FROM krx_security_flow_raw "
        "WHERE ticker = '005930' LIMIT 1"
    ).fetchone()
    assert isinstance(as_double, float)
    assert as_double == pytest.approx(1234.5678)


def test_register_unknown_table_raises(synthetic_lake: LakeConfig) -> None:
    con = lake.connect(synthetic_lake)
    with pytest.raises(KeyError):
        lake.register_views(con, synthetic_lake, tables=["not_a_real_table"])


def test_register_requested_but_absent_table_raises(synthetic_lake: LakeConfig) -> None:
    con = lake.connect(synthetic_lake)
    # daily_ohlcv is a known raw table but absent in this fixture (only
    # krx_security_flow_raw is written); explicit request must fail loudly
    # rather than skip.
    with pytest.raises(FileNotFoundError):
        lake.register_views(con, synthetic_lake, tables=["daily_ohlcv"])


# --- mart cache contract (etl_01 §5) ----------------------------------------


def test_mart_cache_detects_a_formula_change_that_keeps_the_schema(
    synthetic_lake: LakeConfig,
) -> None:
    """A changed formula must invalidate the cache even at an identical schema.

    The cache key used to be (analysis_config_hash, output schema). A fixed NULL
    guard or a corrected mapping keeps every column name and type, so the stale
    mart was silently reused and a "rerun" reproduced the old numbers. See
    10_known_issues.md I10.
    """
    from modeler.etl import mart

    con = lake.connect(synthetic_lake)
    mart.materialize(con, synthetic_lake, "t", "SELECT 1 AS a")
    # Same schema (one INTEGER column named a), different formula.
    with pytest.raises(RuntimeError, match="contract mismatch"):
        mart.materialize(con, synthetic_lake, "t", "SELECT 2 AS a")
    # Unchanged formula still hits the cache.
    mart.materialize(con, synthetic_lake, "t", "SELECT 1 AS a")


def test_mart_cache_accepts_legacy_metadata_without_sql_hash(
    synthetic_lake: LakeConfig,
) -> None:
    """Entries written before ``sql_hash`` existed must not force a rebuild."""
    import json

    from modeler.etl import mart

    con = lake.connect(synthetic_lake)
    mart.materialize(con, synthetic_lake, "t", "SELECT 1 AS a")
    metadata_path = mart.mart_table_dir(synthetic_lake, "t") / "_cache_metadata.json"
    legacy = json.loads(metadata_path.read_text(encoding="utf-8"))
    legacy.pop("sql_hash")
    metadata_path.write_text(json.dumps(legacy), encoding="utf-8")

    mart.materialize(con, synthetic_lake, "t", "SELECT 1 AS a")


# --- persisted derived marts (Stage 1a §3) ---


def test_register_persisted_derived_mart_binds_the_snapshots_own_parquet(
    synthetic_lake: LakeConfig,
) -> None:
    """``feat_macro_exposure`` and the readiness gate that decides whether its
    families are ready must read the same ``common_feature_daily_fact``. This
    reads the parquet ``compute_all --from-step marts`` persisted rather than
    rebuilding the fact from raw, which is the only way to guarantee that."""
    table_dir = synthetic_lake.derived_mart_root / "common_feature_daily_fact"
    table_dir.mkdir(parents=True, exist_ok=True)
    writer = duckdb.connect()
    writer.execute(
        f"COPY (SELECT DATE '2024-01-02' AS feature_date, 'fx_usdkrw_level' AS feature_code, "
        f"1300.0 AS value_numeric) TO '{(table_dir / 'part-000000.parquet').as_posix()}' "
        "(FORMAT PARQUET)"
    )
    writer.close()

    con = duckdb.connect()
    name = lake.register_persisted_derived_mart(con, synthetic_lake, "common_feature_daily_fact")

    assert name == "common_feature_daily_fact"
    assert con.execute("SELECT feature_code FROM common_feature_daily_fact").fetchall() == [
        ("fx_usdkrw_level",)
    ]


def test_register_persisted_derived_mart_raises_when_the_snapshot_has_none(
    synthetic_lake: LakeConfig,
) -> None:
    """A snapshot whose marts step never ran must fail loudly here — the Phase B
    orchestrator turns that into ``blocked`` for the dependent families, which
    is not the same as quietly computing a different fact."""
    con = duckdb.connect()
    with pytest.raises(FileNotFoundError, match="common_feature_daily_fact"):
        lake.register_persisted_derived_mart(con, synthetic_lake, "common_feature_daily_fact")
