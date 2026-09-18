"""ETL configuration — lake roots, snapshot pin, and DuckDB engine options.

Single source of truth for *where* the lake is and *how* the engine reads it.
All other ETL modules import paths/options from here rather than hard-coding.

See:
- ``docs/target/01_20_access_return_rank/etl_01_parquet_data_flow_plan.md`` §0.5, §1, §3
- ``docs/target/01_20_access_return_rank/etl_02_engine_comparison.md`` §6 (engine options)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# DataRoot는 collector/에 산다. modeler가 collector를 경로 의존하므로
# (반대로 두면 순환이고 prod 이미지에 모델링 의존성이 딸려 들어간다) 두
# 저장소가 공유하는 경로 계약은 그쪽 최상위 모듈에 둔다. 여기서 재노출해
# ``from modeler.etl.config import DataRoot`` 를 쓰는 39개 파일은 그대로 둔다.
from collector.lake import DataRoot as DataRoot

# --- repository / lake roots -------------------------------------------------

# src/modeler/etl/config.py -> repo root is three parents up (research/etl/config.py
# 시절엔 두 parents 였다 — S6에서 src/modeler/ 한 겹이 더 생겼다).
REPO_ROOT: Path = Path(__file__).resolve().parents[3]

RAW_LAKE_NAME = "raw_postgres"
# New home for DuckDB-produced derived facts (refactor §8.1 OQ2): the marts in
# research/etl/marts recompute stock_metric_fact / common_feature_daily_fact from
# raw, so the output is a *derived mart*, not a Postgres canonical export.
# Renamed from "derived_mart" when the lake moved to stock_data/ (S4, 2026-09-13).
DERIVED_METRIC_LAKE_NAME = "metric"
DERIVED_FEATURE_LAKE_NAME = "feature"

# Default export source (exporter writes ``source=<name>`` into the path).
# Override with SDC_LAKE_SOURCE for the sj2-direct capture route (dual-route
# raw export, docs/dev/20260730_refactor_dump/00_dual_route_raw_export_plan.md).
DEFAULT_SOURCE = os.environ.get("SDC_LAKE_SOURCE", "local_mydb")
REMOTE_SOURCE = "sj2_remote"


# --- table -> lake-root mapping (etl_01 §2) ---------------------------------
# raw lake: raw + reference tables exported from Postgres. The derived facts
# (stock_metric_fact / common_feature_daily_fact) are recomputed by the DuckDB
# marts in research/etl/marts, not exported from Postgres (canonical_postgres
# was decommissioned and removed at S4 2026-09-13, refactor §3.3).

RAW_TABLES: tuple[str, ...] = (
    "daily_ohlcv",
    "daily_market_cap",
    "krx_security_flow_raw",
    "dart_xbrl_fact_raw",
    "dart_financial_statement_raw",
    "dart_shareholder_return_raw",
    "dart_share_count_raw",
    "dart_capital_change_raw",
    "dart_filing_receipt_raw",
    "dart_employee_raw",
    "dart_governance_raw",
    "dart_xbrl_document",
    "dart_corp_master",
    "dart_corp_profile_history",
    "stock_master",
    "stock_master_snapshot",
    "stock_master_snapshot_items",
    "common_feature_observation_raw",
)

# Decision 7: common_feature_series is the one config table the collector reads at
# runtime AND the compute mart needs, so it is exported to the raw lake and read
# back as a view — the collector and compute see the same rows (no drift branch).
CONFIG_TABLES: tuple[str, ...] = ("common_feature_series",)


@dataclass(frozen=True)
class EngineOptions:
    """DuckDB connection knobs (etl_02 §6).

    ``threads`` defaults to DuckDB's own default when None. ``memory_limit``
    (e.g. ``"2GB"``) caps RAM; DuckDB spills the 2.2GB flow dedup to disk under
    a tight limit (etl_02 §3.1). ``temp_directory`` is where spill files land.
    """

    threads: int | None = None
    memory_limit: str | None = None
    temp_directory: str | None = None

    def as_pragmas(self) -> dict[str, str]:
        pragmas: dict[str, str] = {}
        if self.threads is not None:
            pragmas["threads"] = str(self.threads)
        if self.memory_limit is not None:
            pragmas["memory_limit"] = self.memory_limit
        if self.temp_directory is not None:
            pragmas["temp_directory"] = self.temp_directory
        return pragmas


@dataclass(frozen=True)
class LakeConfig:
    """Resolved lake location for one snapshot.

    A ``LakeConfig`` pins reproducibility: same ``snapshot_date`` => same input
    parquet regardless of later DB changes (etl_01 §0.5). ``root`` and
    ``snapshot_date`` have no default — every caller must say which snapshot
    it means (S4 2026-09-13: a silent default snapshot that drifted out of the
    lake was the point of removing it).
    """

    root: DataRoot
    snapshot_date: str
    source: str = DEFAULT_SOURCE
    engine: EngineOptions = field(default_factory=EngineOptions)
    # Optional analysis-contract hash.  A0 sets this to the canonical horizon
    # scan YAML hash so a changed preregistration cannot reuse old marts.
    analysis_config_hash: str | None = None
    # Rare: redirect dataset_dir() alone (e.g. a namespaced acceptance-gate
    # run whose baseline/candidate builds must not clobber each other), while
    # raw_root/derived_mart_root/feature_mart_root keep reading the real lake
    # under ``root``. Leave unset for the normal case.
    datasets_root_override: Path | None = None

    def _partition(self, base: Path, lake_name: str) -> Path:
        return base / lake_name / f"snapshot_date={self.snapshot_date}" / f"source={self.source}"

    @property
    def raw_root(self) -> Path:
        return self._partition(self.root.raw, RAW_LAKE_NAME)

    @property
    def derived_mart_root(self) -> Path:
        return self._partition(self.root.derived, DERIVED_METRIC_LAKE_NAME)

    @property
    def feature_mart_root(self) -> Path:
        return self._partition(self.root.derived, DERIVED_FEATURE_LAKE_NAME)

    def dataset_dir(self, model_id: str) -> Path:
        """Per-model dataset dir (L2b), isolated by source as well as snapshot."""
        datasets_root = (
            self.datasets_root_override
            if self.datasets_root_override is not None
            else self.root.datasets
        )
        return (
            datasets_root
            / model_id
            / f"snapshot_date={self.snapshot_date}"
            / f"source={self.source}"
        )

    def table_glob(self, table: str) -> str:
        """Recursive parquet glob for a table, across the lake roots.

        Raises ``KeyError`` if the table is not a known raw/config table.
        ``common_feature_series`` (CONFIG_TABLES) lives under the raw lake root
        (decision 7).
        """
        if table in RAW_TABLES or table in CONFIG_TABLES:
            root = self.raw_root
        else:
            known = RAW_TABLES + CONFIG_TABLES
            raise KeyError(f"unknown lake table {table!r}; expected one of {known}")
        return str(root / table / "**" / "*.parquet")
