"""Input and output locations for earnings forecast A, plus the DuckDB input views.

The dl_v4 label mart (r2) is never registered here. Only labels.py may read it.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import duckdb

from modeler.scores.earnings_a.spec import SNAPSHOT

DEFAULT_STOCK_DATA_ROOT = "../stock_data"

# Raw tables read by the universe and reasons steps. Each one sits at
# raw_root/<table>/schema_version=1/... with parquet files at varying depths.
RAW_TABLES: tuple[str, ...] = (
    "daily_ohlcv",
    "daily_market_cap",
    "stock_master",
    "dart_corp_master",
    "dart_filing_receipt_raw",
    "dart_financial_statement_raw",
)

EXPECTED_SHA256: dict[str, str] = {
    "v4p": "e277f5aeed7c48dfaf40b11e6a19c863cf2c53c45b187eac4150f12ce1c6b0c4",
    "r2": "a4212aa1fe668bef97a610244c10c36166aadee9520ae45df634fd60ed99e86f",
}

_OUTPUT_DIR_NAME = "earnings_forecast_a"
_RESEARCH_PREP = "kr/output/research_prep_20261027/a_financial_target_v4"
_FEATURE_TAIL = f"derived/feature/snapshot_date={SNAPSHOT}/source=sj2_remote"


@dataclass(frozen=True)
class DataPaths:
    stock_data_root: Path
    raw_root: Path
    v4p_parquet: Path
    r2_parquet: Path
    out_root: Path

    @classmethod
    def resolve(cls, env: Mapping[str, str] | None = None) -> DataPaths:
        """Build paths from ``STOCK_DATA_ROOT`` (default ``../stock_data``, resolved)."""
        source = os.environ if env is None else env
        root = Path(source.get("STOCK_DATA_ROOT", DEFAULT_STOCK_DATA_ROOT)).resolve()
        return cls(
            stock_data_root=root,
            raw_root=root
            / "kr/raw/raw_postgres"
            / f"snapshot_date={SNAPSHOT}"
            / "source=sj2_remote",
            v4p_parquet=root
            / _RESEARCH_PREP
            / "mart_p6prime_pit"
            / _FEATURE_TAIL
            / "fin_quarterly_metric_vintage_dlr_v4p"
            / "part-000000.parquet",
            r2_parquet=root
            / _RESEARCH_PREP
            / "mart_r2"
            / _FEATURE_TAIL
            / "fin_quarterly_metric_vintage_dl_v4"
            / "part-000000.parquet",
            out_root=root / "kr/output" / _OUTPUT_DIR_NAME,
        )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_input_hashes(paths: DataPaths) -> None:
    """Raise RuntimeError if an input file's sha256 is not the pre-registered one.

    The error names the file only. It does not print the hash values.
    """
    targets = {"v4p": paths.v4p_parquet, "r2": paths.r2_parquet}
    for key, path in targets.items():
        if sha256_file(path) != EXPECTED_SHA256[key]:
            raise RuntimeError(f"input hash mismatch: {path.name} ({key})")


def _quote(text: str) -> str:
    """Quote a string as a SQL literal for DuckDB DDL, which takes no bind parameters."""
    return "'" + text.replace("'", "''") + "'"


def connect_inputs(paths: DataPaths) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with the v4p view and the raw table views.

    Never creates a view over ``paths.r2_parquet``.
    """
    for table in RAW_TABLES:
        table_dir = paths.raw_root / table
        if not table_dir.is_dir() or not any(table_dir.rglob("*.parquet")):
            raise FileNotFoundError(f"raw table has no parquet files: {table}")

    con = duckdb.connect(":memory:")
    # Works around a statistics_propagation optimizer bug on hive-layout raw reads.
    con.execute("SET disabled_optimizers='statistics_propagation'")
    con.execute(f"CREATE VIEW v4p AS SELECT * FROM read_parquet({_quote(str(paths.v4p_parquet))})")
    for table in RAW_TABLES:
        glob = str(paths.raw_root / table / "**" / "*.parquet")
        con.execute(
            f"CREATE VIEW {table} AS SELECT * FROM read_parquet("
            f"{_quote(glob)}, hive_partitioning=false, union_by_name=true)"
        )
    return con
