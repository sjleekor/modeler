"""Feature mart (L2a) materialization helpers.

The mart is the model-agnostic, snapshot-pinned layer between the raw/canonical
lake and per-model datasets (00_shared §1). Heavy computations (flow dedup,
financial PIT as-of) run here once per snapshot and are written to parquet;
models then join the mart cheaply.

Caching is idempotent by directory: a mart that already exists is skipped unless
``force=True`` (00_shared §5), mirroring the exporter scripts' ``--force``.

See ``docs/target/00_shared_etl_platform.md`` §1, §5 and
``docs/target/01_20_access_return_rank/etl_03_implementation_plan.md`` §4 (P2).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from pathlib import Path

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.lake import _sql_str_literal


class StaleMartContract(RuntimeError):
    """A mart exists on disk but was not written under the current contract.

    Raised instead of reusing it — reuse is the failure the cache keys exist to
    stop. It says nothing about *why* the contract differs: a changed formula, a
    changed output schema, or a different (or absent) ``analysis_config_hash``
    all land here.

    Subclasses ``RuntimeError`` so callers that only ever wanted the message
    keep working. The type exists so a caller that owns the mart's build step
    can tell "this needs rebuilding" apart from a genuine failure without
    matching on message text — see ``register_phase_b_marts``, where dying on
    this meant a Phase B run could not start on a snapshot whose marts were
    pre-built by ``compute-all`` (which does not stamp ``analysis_config_hash``).
    """


def mart_root(config: LakeConfig) -> Path:
    """Root for the snapshot's feature mart (``<STOCK_DATA_ROOT>/kr/derived/feature/...``)."""
    return config.feature_mart_root


def mart_table_dir(config: LakeConfig, name: str) -> Path:
    """Directory holding one mart table's parquet part files."""
    return mart_root(config) / name


def mart_glob(config: LakeConfig, name: str) -> str:
    """Recursive parquet glob for a materialized mart table."""
    return str(mart_table_dir(config, name) / "**" / "*.parquet")


def is_materialized(config: LakeConfig, name: str) -> bool:
    """True if the mart table has at least one parquet part on disk."""
    directory = mart_table_dir(config, name)
    return directory.is_dir() and any(directory.rglob("*.parquet"))


def _metadata_path(config: LakeConfig, name: str) -> Path:
    return mart_table_dir(config, name) / "_cache_metadata.json"


def _schema_hash(con: duckdb.DuckDBPyConnection, select_sql: str) -> str:
    """Hash the DuckDB output schema, not merely the output file presence."""
    rows = con.execute(f"DESCRIBE SELECT * FROM ({select_sql}) AS _cache_query").fetchall()
    encoded = json.dumps(rows, default=str, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sql_hash(select_sql: str) -> str:
    """Hash the query text itself.

    ``_schema_hash`` only sees column names and types, so a formula change that
    keeps the same output shape — a fixed NULL guard, a corrected mapping — is
    invisible to it and the stale mart is silently reused. That is the failure
    this key exists to stop; see
    docs/dev/20260731_raw_features/01_feature_candidate/10_known_issues.md I10.
    """
    return hashlib.sha256(select_sql.encode()).hexdigest()


def sql_contract_hash(select_sql: str) -> str:
    """Public name for the SQL-text cache key (see :func:`_sql_hash`).

    A read-only consumer uses it to answer "is the mart on disk the one my own
    definition would have produced?" without being allowed to rebuild it.
    """
    return _sql_hash(select_sql)


def _expected_metadata(
    con: duckdb.DuckDBPyConnection, config: LakeConfig, select_sql: str
) -> dict[str, str | None]:
    return {
        "analysis_config_hash": config.analysis_config_hash,
        "schema_hash": _schema_hash(con, select_sql),
        "sql_hash": _sql_hash(select_sql),
    }


def _cache_contract_matches(actual: dict[str, object], expected: dict[str, str | None]) -> bool:
    """Compare a stored cache entry with the current contract.

    ``sql_hash`` was added after the marts in this lake were first written, so
    an entry without it is legacy: the keys it does carry must still match, and
    the missing one is simply unverifiable. Legacy entries are accepted rather
    than failed, so adding this key does not force a full lake rebuild — but a
    formula change after this point is caught.
    """
    if "sql_hash" not in actual:
        return all(actual.get(key) == value for key, value in expected.items() if key != "sql_hash")
    return actual == expected


def materialize(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    name: str,
    select_sql: str,
    *,
    force: bool = False,
    partition_by: list[str] | None = None,
) -> Path:
    """Write ``select_sql`` to ``feature_mart/.../<name>/`` as parquet.

    Idempotent: returns early (skips the write) when the table already exists
    and ``force`` is False. With ``force`` the existing directory is removed and
    rebuilt. Returns the table directory.
    """
    table_dir = mart_table_dir(config, name)
    expected = _expected_metadata(con, config, select_sql)
    if is_materialized(config, name) and not force:
        metadata_path = _metadata_path(config, name)
        if not metadata_path.is_file():
            raise StaleMartContract(
                f"mart cache metadata is missing for {name!r}; rerun with force=True"
            )
        try:
            actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StaleMartContract(
                f"invalid mart cache metadata for {name!r}; use force=True"
            ) from exc
        if not _cache_contract_matches(actual, expected):
            raise StaleMartContract(
                f"mart cache contract mismatch for {name!r}; use force=True to rebuild"
            )
        return table_dir

    if table_dir.exists():
        shutil.rmtree(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    out = _sql_str_literal(str(table_dir))
    copy_opts = ["FORMAT PARQUET", "COMPRESSION ZSTD"]
    if partition_by:
        cols = ", ".join(partition_by)
        copy_opts.append(f"PARTITION_BY ({cols})")
        # PARTITION_BY writes a directory tree; otherwise a single file.
        target = out
    else:
        target = _sql_str_literal(str(table_dir / "part-000000.parquet"))

    con.execute(f"COPY ({select_sql}) TO {target} ({', '.join(copy_opts)})")
    _metadata_path(config, name).write_text(
        json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8"
    )
    return table_dir


def materialize_in_parts(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    name: str,
    contract_sql: str,
    parts: Iterable[tuple[str, str]],
    *,
    force: bool = False,
) -> Path:
    """Write one mart table as several parquet parts, one query per part.

    Same cache contract as :func:`materialize` — ``contract_sql`` is the single
    unpartitioned statement the table *means*, and it is what ``schema_hash`` /
    ``sql_hash`` are taken from, so a formula change still invalidates the
    cache. ``parts`` supplies ``(part_name, part_sql)`` pairs whose union is
    that statement; each is written separately.

    This exists for marts whose one-shot query would materialize an
    intermediate far larger than the output — ``feat_relation_stat`` joins 40
    peer rows onto every ticker-session, 280M rows over the full history but
    only ~2M within one month. Splitting by month keeps the join working set
    bounded instead of relying on DuckDB spilling.

    ``contract_sql`` is only *described* (``DESCRIBE SELECT``), never run, so
    the unpartitioned form may be too heavy to execute.
    """
    table_dir = mart_table_dir(config, name)
    expected = _expected_metadata(con, config, contract_sql)
    if is_materialized(config, name) and not force:
        metadata_path = _metadata_path(config, name)
        if not metadata_path.is_file():
            raise StaleMartContract(
                f"mart cache metadata is missing for {name!r}; rerun with force=True"
            )
        try:
            actual = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StaleMartContract(
                f"invalid mart cache metadata for {name!r}; use force=True"
            ) from exc
        if not _cache_contract_matches(actual, expected):
            raise StaleMartContract(
                f"mart cache contract mismatch for {name!r}; use force=True to rebuild"
            )
        return table_dir

    if table_dir.exists():
        shutil.rmtree(table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for part_name, part_sql in parts:
        target = _sql_str_literal(str(table_dir / f"{part_name}.parquet"))
        con.execute(f"COPY ({part_sql}) TO {target} (FORMAT PARQUET, COMPRESSION ZSTD)")
        written += 1
    if written == 0:
        # No metadata is written, so the directory stays "not materialized" and
        # the next call retries rather than registering an empty view.
        raise RuntimeError(f"no parts produced for mart {name!r}")

    _metadata_path(config, name).write_text(
        json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8"
    )
    return table_dir


def mart_cache_metadata(config: LakeConfig, name: str) -> dict | None:
    """The contract a materialized mart was written under, or None if unstamped.

    Read-only consumers need this: a model that must *not* rebuild a mart (the
    A0 marts a horizon-scan run published, say) still has to record which build
    it read, and :func:`register_mart_view`'s all-or-nothing hash check cannot
    express that. Returns the stored dict as-is — ``analysis_config_hash`` may
    legitimately be None for a mart written by ``compute-all``.
    """
    path = _metadata_path(config, name)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaleMartContract(f"invalid mart cache metadata for {name!r}") from exc


def register_mart_view(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    name: str,
    *,
    view_name: str | None = None,
) -> str:
    """Register a DuckDB view over a materialized mart table (hive=false).

    Returns the created view name. Raises ``FileNotFoundError`` if the mart
    table has not been materialized yet.
    """
    if not is_materialized(config, name):
        raise FileNotFoundError(
            f"mart table {name!r} not materialized at {mart_table_dir(config, name)}"
        )
    metadata_path = _metadata_path(config, name)
    if not metadata_path.is_file():
        raise StaleMartContract(
            f"mart cache metadata is missing for {name!r}; rebuild with force=True"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaleMartContract(f"invalid mart cache metadata for {name!r}") from exc
    if metadata.get("analysis_config_hash") != config.analysis_config_hash:
        raise StaleMartContract(
            f"mart cache config hash mismatch for {name!r}; rebuild with force=True"
        )
    view = view_name or name
    glob = _sql_str_literal(mart_glob(config, name))
    con.execute(
        f"CREATE OR REPLACE VIEW {view} AS "
        f"SELECT * FROM read_parquet({glob}, hive_partitioning=false)"
    )
    return view
