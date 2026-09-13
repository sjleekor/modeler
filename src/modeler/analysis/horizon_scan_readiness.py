"""Generate machine-readable readiness for the preregistered families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import polars as pl

from modeler.analysis.horizon_scan_config import (
    CONFIG_PATH,
    HorizonScanConfig,
    bucket_primary_cells,
    load_config,
)
from modeler.etl.config import REMOTE_SOURCE, DataRoot
from modeler.etl.horizon_scan_inputs import REQUIRED_RAW_INPUTS
from modeler.etl.lake import connect
from modeler.etl.mart import mart_root, register_mart_view
from modeler.etl.snapshot import resolve_config

PRIMARY_HYPOTHESIS_COUNT = 75
SHORT_EXPLORATORY_CELL_COUNT = 28


def _hypothesis_rows(
    config: HorizonScanConfig,
    families: list[dict[str, Any]],
    *,
    hypothesis_role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family in families:
        primary_feature = next(
            feature["column"] for feature in family["features"] if feature["role"] == "primary"
        )
        official_variant = family["official_feature_variant"]
        # §1.1 condition 7 / §A-1: the scanned column is the family's frozen
        # *official* execution variant — for same-day-unverified flow this is
        # `lag1`, not the native_t column `primary_feature` happens to name.
        # `hypothesis_id` still embeds `primary_feature` (the stable native
        # name) purely as identity, not as the column actually scanned.
        scan_feature = family["variant_columns"][official_variant]
        common = {
            "family": family["family"],
            "fdr_family": family.get("fdr_family"),
            "feature": scan_feature,
            "feature_role": "primary",
            "feature_variant": official_variant,
            "hypothesis_role": hypothesis_role,
            "expected_sign": family.get("expected_sign"),
        }
        for h in family["primary_horizon_set"]:
            rows.append(
                {
                    **common,
                    "hypothesis_id": f"{family['family']}|{primary_feature}|cum|0|{h}",
                    "scan_type": "cum",
                    "h_start": 0,
                    "h_end": h,
                    "width": h,
                }
            )
        for b_start, b_end in bucket_primary_cells(family, config.buckets):
            hypothesis_id = f"{family['family']}|{primary_feature}|bucket|{b_start}|{b_end}"
            rows.append(
                {
                    **common,
                    "hypothesis_id": hypothesis_id,
                    "scan_type": "bucket",
                    "h_start": b_start,
                    "h_end": b_end,
                    "width": b_end - b_start,
                }
            )
    ids = [row["hypothesis_id"] for row in rows]
    if len(set(ids)) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"hypothesis registry has duplicate ids: {dupes}")
    return rows


def build_primary_hypothesis_registry(config: HorizonScanConfig) -> list[dict[str, Any]]:
    """One row per preregistered global-BH primary cell (§2.1, §2.3).

    Any drift from the preregistered 75 cells — a family readded/removed, a
    horizon added, a bucket miscounted — raises here rather than silently
    changing the multiple-testing correction's ``m``.
    """
    families = [f for f in config.families if f.get("fdr_include", False)]
    rows = _hypothesis_rows(config, families, hypothesis_role="primary")
    if len(rows) != PRIMARY_HYPOTHESIS_COUNT:
        raise ValueError(
            f"primary hypothesis registry must have {PRIMARY_HYPOTHESIS_COUNT} cells, "
            f"got {len(rows)}"
        )
    return rows


def build_short_exploratory_registry(config: HorizonScanConfig) -> list[dict[str, Any]]:
    """One row per preregistered short-regime exploratory cell (§2.1).

    These 28 cells are limited single-block-sample diagnostics and never enter
    global BH or ``screen_pass`` (§2.2).
    """
    families = [f for f in config.families if f.get("role") == "exploratory_short_regime"]
    rows = _hypothesis_rows(config, families, hypothesis_role="exploratory_short_regime")
    if len(rows) != SHORT_EXPLORATORY_CELL_COUNT:
        raise ValueError(
            f"short exploratory registry must have {SHORT_EXPLORATORY_CELL_COUNT} cells, "
            f"got {len(rows)}"
        )
    return rows


def _dependency_columns(
    con: duckdb.DuckDBPyConnection | None,
    dependencies: list[str],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    if con is None:
        return result
    for dependency in dependencies:
        try:
            result[dependency] = {
                row[0] for row in con.execute(f"DESCRIBE {dependency}").fetchall()
            }
        except duckdb.Error:
            result[dependency] = set()
    return result


def build_readiness_rows(
    config: HorizonScanConfig,
    *,
    columns_by_mart: dict[str, set[str]] | None = None,
    publication_lag_verified: bool = False,
    effective_sample_start: dict[str, str | None] | None = None,
) -> list[dict[str, Any]]:
    """Return one readiness row per registered family.

    A config-only run reports dependency names as missing; a connected run can
    pass DESCRIBE-derived columns for exact readiness.
    """
    columns_by_mart = columns_by_mart or {}
    effective_sample_start = effective_sample_start or {}
    rows: list[dict[str, Any]] = []
    for family in config.families:
        primary = next(
            feature["column"]
            for feature in family["features"]
            if feature["role"] == "primary"
        )
        dependencies = family["readiness_dependencies"]
        missing_dependencies = [
            d for d in dependencies
            if d not in columns_by_mart
            and not (d == "short_balance_publication_lag" and publication_lag_verified)
        ]
        missing_columns = [
            primary
            for dependency in dependencies
            if dependency.startswith("feat_") and dependency in columns_by_mart
            and primary not in columns_by_mart[dependency]
        ]
        role = family["role"]
        if family["phase"] == "B":
            readiness_class = "phase_b_blocked"
        elif "dim_stock_pit_daily" in dependencies:
            readiness_class = "pit_dependency"
        elif any(dep.startswith("feat_") for dep in dependencies):
            readiness_class = "mart_extension"
        else:
            readiness_class = "cross_mart_dependency"
        if family["phase"] == "B":
            status = "phase_b_blocked"
        elif role == "reference_only":
            status = "reference_only"
        elif family["family"] in {
            "flow_short_interest",
            "flow_days_to_cover",
            "flow_nat_proxy_20d",
        } and not publication_lag_verified:
            status = "exploratory_blocked_publication_lag"
        elif missing_dependencies or missing_columns:
            status = "blocked_missing_dependency"
        elif role == "exploratory_short_regime":
            status = "ready_exploratory"
        else:
            status = "ready"
        rows.append(
            {
                "family": family["family"],
                "phase": family["phase"],
                "primary_feature": primary,
                "role": role,
                "readiness_class": readiness_class,
                "status": status,
                "expected_sign": family.get("expected_sign"),
                "primary_horizon_set": ",".join(map(str, family["primary_horizon_set"])),
                "fdr_include": bool(family.get("fdr_include", False)),
                "missing_dependencies": ",".join(missing_dependencies),
                "missing_columns": ",".join(missing_columns),
                "effective_sample_start": effective_sample_start.get(family["family"]),
                "config_hash": config.config_hash,
            }
        )
    return rows


def write_readiness_artifacts(
    config: HorizonScanConfig,
    output_dir: Path,
    *,
    columns_by_mart: dict[str, set[str]] | None = None,
    publication_lag_verified: bool = False,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_readiness_rows(
        config,
        columns_by_mart=columns_by_mart,
        publication_lag_verified=publication_lag_verified,
    )
    frame = pl.DataFrame(rows)
    parquet = output_dir / "readiness_matrix.parquet"
    markdown = output_dir / "readiness_matrix.md"
    frame.write_parquet(parquet)
    lines = [
        "# Horizon scan readiness",
        "",
        f"config_hash: `{config.config_hash}`",
        "",
        "| family | phase | primary | status | missing |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        missing = ", ".join(filter(None, [row["missing_dependencies"], row["missing_columns"]]))
        lines.append(
            f"| {row['family']} | {row['phase']} | {row['primary_feature']} | "
            f"{row['status']} | {missing or '-'} |"
        )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return parquet, markdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--source", default=REMOTE_SOURCE)
    parser.add_argument("--data-lake-root", type=Path, default=None)
    parser.add_argument("--config-only", action="store_true")
    args = parser.parse_args(argv)
    root = DataRoot(args.data_lake_root) if args.data_lake_root else DataRoot.resolve(market="kr")
    output_dir = args.output_dir or (root.output / "scan" / "horizon_scan")
    config = load_config(args.config)
    if args.config_only:
        write_readiness_artifacts(config, output_dir)
        return 0

    lake, _resolution = resolve_config(
        root,
        args.source,
        required_inputs=REQUIRED_RAW_INPUTS,
        snapshot_date=args.snapshot_date,
    )
    manifest = mart_root(lake) / "_manifests" / "_SUCCESS.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"A0 manifest is required for connected readiness: {manifest}")
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    if manifest_payload.get("config_hash") != config.config_hash:
        raise RuntimeError("A0 manifest config hash does not match the readiness configuration")
    con = connect(lake)
    dependencies = sorted(
        {d for family in config.families for d in family["readiness_dependencies"]}
    )
    columns_by_mart: dict[str, set[str]] = {}
    for dependency in dependencies:
        if dependency == "short_balance_publication_lag":
            continue
        try:
            register_mart_view(con, lake, dependency)
        except FileNotFoundError:
            continue
        columns_by_mart[dependency] = {
            row[0] for row in con.execute(f"DESCRIBE {dependency}").fetchall()
        }
    evidence = Path(
        manifest_payload.get(
            "publication_lag_evidence", output_dir / "short_balance_publication_lag.json"
        )
    )
    publication_lag_verified = False
    if evidence.is_file():
        try:
            publication_lag_verified = json.loads(
                evidence.read_text(encoding="utf-8")
            ).get("status") == "verified"
        except (OSError, ValueError):
            publication_lag_verified = False
    write_readiness_artifacts(
        config,
        output_dir,
        columns_by_mart=columns_by_mart,
        publication_lag_verified=publication_lag_verified,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
