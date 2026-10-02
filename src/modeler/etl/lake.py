"""L1 lake reader — DuckDB connection + view registration over Parquet.

Every ETL query in the ``research`` package runs on top of the views created
here. Two invariants this module enforces (both verified bugs from etl_01):

1. ``hive_partitioning=false`` ALWAYS. The lake path contains ``source=...``
   and ``krx_security_flow_raw`` *also* has a real ``source`` column
   (``KRX``/``PYKRX``). With hive=true the path value silently overwrites the
   data column, neutralizing the KRX-first dedup (etl_01 §4.2). We never enable
   it; partition pruning is driven by real data columns (``trade_date`` etc.),
   which DuckDB still applies from the ``year=/month=`` path automatically.
2. ``numeric`` arrives as ``DECIMAL`` (Decimal128). Ratio/log math should cast
   to ``DOUBLE`` first (etl_01 §3); :func:`cast_double` builds that expression.

See ``docs/target/01_20_access_return_rank/etl_01_parquet_data_flow_plan.md`` §3, §4.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import duckdb

from modeler.etl.config import CONFIG_TABLES, RAW_TABLES, LakeConfig

DERIVED_PLAN_FILE = "_plan.json"


def connect(config: LakeConfig) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with the configured engine pragmas."""
    con = duckdb.connect()
    pragmas = config.engine.as_pragmas()
    if "temp_directory" in pragmas:
        # DuckDB creates the leaf lazily and fails on a missing parent.
        Path(pragmas["temp_directory"]).mkdir(parents=True, exist_ok=True)
    for key, value in pragmas.items():
        # Identifiers (key) are from a fixed allowlist; value is quoted.
        con.execute(f"SET {key} = '{value}'")
    return con


def _sql_str_literal(value: str) -> str:
    """Single-quote a string for inline SQL (escape embedded quotes)."""
    return "'" + value.replace("'", "''") + "'"


def _create_view(con: duckdb.DuckDBPyConnection, table: str, glob: str) -> None:
    # hive_partitioning=false is mandatory — see module docstring (etl_01 §4.2).
    # DuckDB rejects prepared parameters inside CREATE VIEW DDL, so the glob
    # (a trusted, config-derived path) is inlined as a quoted literal.
    con.execute(
        f"CREATE OR REPLACE VIEW {table} AS "
        f"SELECT * FROM read_parquet({_sql_str_literal(glob)}, hive_partitioning=false)"
    )


def register_views(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    tables: Iterable[str] | None = None,
) -> list[str]:
    """Register one DuckDB view per lake table, returning the names created.

    By default registers every raw + config table that has parquet files on
    disk. Missing tables are skipped so callers can register opportunistically.
    Pass ``tables`` to restrict to a specific set (a ``KeyError`` is raised for
    unknown names).
    """
    requested = tuple(tables) if tables is not None else RAW_TABLES + CONFIG_TABLES

    created: list[str] = []
    for table in requested:
        glob = config.table_glob(table)  # raises KeyError on unknown table
        # Probe: read_parquet errors if zero files match. Skip absent tables
        # unless they were explicitly requested.
        if not _glob_has_files(con, glob):
            if tables is not None:
                raise FileNotFoundError(f"no parquet files for table {table!r} at {glob}")
            continue
        _create_view(con, table, glob)
        created.append(table)
    return created


def _glob_has_files(con: duckdb.DuckDBPyConnection, glob: str) -> bool:
    """True if at least one parquet file matches the glob."""
    rows = con.execute("SELECT count(*) FROM glob(?)", [glob]).fetchone()
    return bool(rows and rows[0] > 0)


def cast_double(column: str) -> str:
    """SQL expression casting a (possibly Decimal128) column to DOUBLE.

    Use before ratio/log/z-score math to avoid Decimal overflow/precision
    surprises (etl_01 §3). Keep winsorize/log ordering per etl_00 §4.3.
    """
    return f"CAST({column} AS DOUBLE)"


def register_derived_marts(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    which: Iterable[str] = ("stock_metric_fact", "common_feature_daily_fact"),
    persist: bool = False,
    force: bool = False,
    metric_fact_plan: str = "single",
    metric_rules_version: str | None = None,
) -> list[str]:
    """Recompute the derived facts from the raw lake and register them as views.

    Single replacement for the old ``register_views(..., CANONICAL_TABLES)`` path
    (refactor §3.3 step 1): builds ``stock_metric_fact`` / ``common_feature_daily_fact``
    from the raw (+ ``common_feature_series``) views via the DuckDB marts, then
    registers each under its canonical view name so ``fin_pit.py`` / ``common.py``
    read them unchanged. Requires the needed raw views already registered on ``con``.

    With ``persist=True``, each derived view is written to
    ``data_lake/derived_mart/snapshot_date=.../source=.../<table>/`` as parquet and
    the canonical view name is rebound to that parquet output. The default remains
    in-memory for unit tests and smoke checks that should not write repository
    artifacts.

    ``metric_fact_plan`` picks the execution plan of ``stock_metric_fact`` (same rows, a
    different text; see ``metrics_normalize.PLANS``). A persisted mart built by a
    non-default plan carries a ``_plan.json`` and is only reused by the same plan.
    ``metric_rules_version`` picks the collector metric rule set (``None`` = current); a
    mart built under other rules records the version and rule hash in its ``_plan.json``
    and is only reused under the same rules.

    Returns the view names created.
    """
    # Imported here (not at module top) to avoid a circular import: the marts
    # import _sql_str_literal from this module.
    from modeler.etl.marts.common_build import (
        register_common_feature_daily_fact_view,
    )
    from modeler.etl.marts.metrics_normalize import (
        is_default_stock_metric_fact,
        register_stock_metric_fact_view,
        stock_metric_fact_plan_record,
    )

    requested = set(which)
    created: list[str] = []

    if "stock_metric_fact" in requested:
        view = register_stock_metric_fact_view(
            con, plan=metric_fact_plan, rules_version=metric_rules_version)
        if persist:
            _persist_derived_mart(
                con, config, view, force=force,
                plan_record=(
                    None
                    if is_default_stock_metric_fact(metric_fact_plan, metric_rules_version)
                    else stock_metric_fact_plan_record(
                        metric_fact_plan, rules_version=metric_rules_version)
                ),
            )
        created.append(view)

    if "common_feature_daily_fact" in requested:
        trading_days, feature_dates = _common_feature_calendars(con)
        view = register_common_feature_daily_fact_view(
            con, trading_days=trading_days, feature_dates=feature_dates
        )
        if persist:
            _persist_derived_mart(con, config, view, force=force)
        created.append(view)
    return created


def _persist_derived_mart(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    name: str,
    *,
    force: bool = False,
    plan_record: dict | None = None,
) -> None:
    """Write one derived canonical-compatible view to the derived mart lake.

    ``plan_record`` (a non-default execution plan) is stored beside the parquet as
    ``_plan.json``; reusing existing files requires the same record. A caller without one
    (the default definition) reuses legacy files without a sidecar, and refuses a mart whose
    sidecar says it was built by another plan or rule set.
    """
    from modeler.etl.mart import StaleMartContract

    table_dir = config.derived_mart_root / name
    if table_dir.exists() and force:
        shutil.rmtree(table_dir)
    glob_path = str(table_dir / "**" / "*.parquet")
    has_files = table_dir.exists() and _glob_has_files(con, glob_path)
    sidecar = table_dir / DERIVED_PLAN_FILE
    if has_files and plan_record is not None:
        try:
            stored = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            stored = None
        if stored != plan_record:
            raise StaleMartContract(
                f"persisted derived mart {name!r} was not written by plan "
                f"{plan_record.get('plan')!r}; use force=True to rebuild"
            )
    if has_files and plan_record is None and sidecar.is_file():
        raise StaleMartContract(
            f"persisted derived mart {name!r} was written by a non-default plan or rule set "
            f"({sidecar.read_text(encoding='utf-8').strip()}); use force=True to rebuild"
        )
    if not has_files:
        if table_dir.exists():
            shutil.rmtree(table_dir)
        table_dir.mkdir(parents=True, exist_ok=True)
        target = _sql_str_literal(str(table_dir / "part-000000.parquet"))
        con.execute(f"COPY (SELECT * FROM {name}) TO {target} (FORMAT PARQUET, COMPRESSION ZSTD)")
        if plan_record is not None:
            sidecar.write_text(json.dumps(plan_record, sort_keys=True) + "\n", encoding="utf-8")

    glob = _sql_str_literal(glob_path)
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_parquet({glob}, hive_partitioning=false)"
    )


def read_derived_plan_record(config: LakeConfig, name: str) -> dict:
    """semantics / plan / plan hash a persisted derived mart was written under.

    A mart persisted by the default plan has no ``_plan.json``; it reports the legacy
    statement (``v1`` / ``single`` / no plan hash), like a feature mart without plan keys.
    """
    sidecar = config.derived_mart_root / name / DERIVED_PLAN_FILE
    if not sidecar.is_file():
        return {"semantics_version": "v1", "plan": "single", "plan_hash": None}
    return json.loads(sidecar.read_text(encoding="utf-8"))


def register_persisted_derived_mart(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    name: str,
) -> str:
    """Bind a snapshot's already-persisted derived mart as a view. No recompute.

    ``register_derived_marts`` rebuilds the fact from the raw lake; this reads
    the parquet ``compute_all --from-step marts`` wrote and the coverage/
    readiness gates then passed. For a consumer that must agree with those
    gates — ``feat_macro_exposure`` and the readiness check that decides
    whether its families are ready — reading the same bytes is the only way to
    guarantee it, and it needs no raw views registered at all.

    Raises ``FileNotFoundError`` when the snapshot has no such mart, which is
    what leaves a dependent family ``blocked`` rather than silently computing a
    different fact.
    """
    glob_path = str(config.derived_mart_root / name / "**" / "*.parquet")
    if not _glob_has_files(con, glob_path):
        raise FileNotFoundError(f"no persisted derived mart for {name!r} at {glob_path}")
    glob = _sql_str_literal(glob_path)
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS "
        f"SELECT * FROM read_parquet({glob}, hive_partitioning=false)"
    )
    return name


def _common_feature_calendars(
    con: duckdb.DuckDBPyConnection,
    *,
    obs_view: str = "common_feature_observation_raw",
    ohlcv_view: str = "daily_ohlcv",
) -> tuple[list, list]:
    """KRX session calendars for the common build, derived from the raw lake.

    ``feature_dates`` = KRX sessions from the first observation availability
    through the last session the lake actually reaches; ``trading_days`` = the
    stale calendar (same span). Uses ``get_trading_days`` so it matches the
    Postgres build's KRX calendar exactly.

    **The end is clamped to the last priced session (F-9.9).** ``available_from_date``
    is an *announced* availability, so a monthly ECOS series legitimately carries
    a future one: measured on snapshot 2026-09-08, ``macro_cpi`` and
    ``macro_consumer_sentiment`` both say 2026-09-21 while the last KRX session
    with prices is 2026-09-07. Taking the raw maximum stretched the grid ten
    sessions past the end of the data, and every daily series was then charged
    with missing values for sessions that have not happened -- which is what made
    the readiness gate impossible to pass at its own default
    (``--required-coverage-ratio 1.0`` failed 33 of 38 features).

    A lake without ``ohlcv_view`` is left unclamped rather than failed: the
    clamp is a correction to an upper bound, not an input the calendar needs.
    """
    from collector.kr.shared import get_trading_days

    bounds = con.execute(
        f"SELECT min(available_from_date), max(available_from_date), max(observation_date) "
        f"FROM {obs_view}"
    ).fetchone()
    first_avail, last_avail, last_obs = bounds
    if first_avail is None:
        return [], []
    end = max(d for d in (last_avail, last_obs) if d is not None)
    try:
        last_session = con.execute(f"SELECT max(trade_date) FROM {ohlcv_view}").fetchone()[0]
    except duckdb.Error:
        last_session = None
    if last_session is not None and last_session < end:
        end = last_session
    sessions = list(get_trading_days(first_avail, end))
    # feature_dates and the stale calendar share the same KRX session span here;
    # the orchestrator can narrow feature_dates for incremental/backfill runs.
    return sessions, sessions
