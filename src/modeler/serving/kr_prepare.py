"""Build one label-free KR h20 model-input cross-section from a frozen lake.

This module reads only raw prices needed to establish the session key and the
feature/universe marts. It never registers or builds a label view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from datetime import time as datetime_time
from pathlib import Path
from tempfile import mkdtemp
from zoneinfo import ZoneInfo

import duckdb
import polars as pl

from modeler.etl import preprocess as pp
from modeler.etl.config import DataRoot, LakeConfig, REMOTE_SOURCE
from modeler.etl.lake import connect, register_views
from modeler.etl.mart import is_materialized, mart_cache_metadata, mart_glob, sql_contract_hash
from modeler.etl.manifest import current_git_sha
from modeler.etl.universe import UniverseFilter, build_universe_sql
from modeler.models._02_updown_prob import features as fx

KEY_COLS = ("trade_date", "ticker", "market")
UNIVERSE_VIEW = "dim_universe_daily"
QUALITY_VIEW = "dim_price_quality_daily"
QUALITY_COLS = (
    "is_halted", "ca_price_jump_suspect", "ca_share_change_confirmed",
    "ca_rule_applicability_unknown", "simple_ret",
)
REQUIRED_PREP_MARTS = (
    UNIVERSE_VIEW, QUALITY_VIEW, "feat_price", "feat_flow", "feat_fin_pit",
    "feat_fin_scan_daily", "feat_filing_activity",
)
#: Which marts the feature builder wrote. ``full`` is every mart of the research chain;
#: ``serving`` builds only what the model's features and quality columns read, and writes
#: ``feat_fin_scan_daily`` as a projection. A marker without ``profile`` predates the choice
#: and is ``full``.
BUILD_PROFILE_FULL = "full"
BUILD_PROFILE_SERVING = "serving"
BUILD_PROFILES = (BUILD_PROFILE_FULL, BUILD_PROFILE_SERVING)
DEFAULT_BUILD_PROFILE = BUILD_PROFILE_FULL
DATE_GRAIN_MARTS = frozenset({"feat_common", fx.GROUP_VIEW["rg"]})
MART_ALIAS = {view: group for group, view in fx.GROUP_VIEW.items()}


def _required_marts(columns: list[str]) -> dict[str, list[str]]:
    needed: dict[str, list[str]] = {}
    interaction_outputs = set(fx.FS2_INTERACTION_COLS)
    for column in columns:
        if column in interaction_outputs:
            continue
        view = fx.COLUMN_MART.get(column)
        if view is None:
            raise KeyError(f"no mart is mapped for KR feature {column!r}")
        needed.setdefault(view, []).append(column)
    return needed


def _flow_rows_cte(flow_view: str, asof_date: str) -> str:
    """K's rows plus, per (ticker, market), the last observation before K.

    ``LAG`` over the whole mart reads each K row's predecessor *within its own ticker*,
    which is not the previous calendar session for a halted or delisted-and-relisted
    ticker. Keeping exactly each ticker's last row before K reproduces that predecessor and
    drops the other ~6.7M rows from the window sort.
    """
    return f"""flow_prev AS (
            SELECT ticker, market, max(trade_date) AS prev_date
            FROM {flow_view} WHERE trade_date < DATE '{asof_date}' GROUP BY ticker, market
        ),
        flow_rows AS (
            SELECT * FROM {flow_view} WHERE trade_date = DATE '{asof_date}'
            UNION ALL
            SELECT f.* FROM {flow_view} f
            JOIN flow_prev p
              ON f.ticker = p.ticker AND f.market = p.market AND f.trade_date = p.prev_date
        )"""


def _flow_sql(spec: KrBriefingSpec, columns: list[str], source: str | None = None) -> str:
    """The lag-shifted flow columns. ``source`` replaces the mart in ``FROM`` (see A5)."""
    selects = list(KEY_COLS)
    for column in columns:
        if column.endswith("_lag1") or spec.flow_variant == "native_t":
            selects.append(column)
        elif column in fx.FLOW_MART_LAG1_SOURCE:
            selects.append(f"{fx.FLOW_MART_LAG1_SOURCE[column]} AS {column}")
        elif column in fx.FLOW_BUILDER_LAG_COLUMNS:
            selects.append(f"LAG({column}) OVER w AS {column}")
        else:
            selects.append(column)
    window = " WINDOW w AS (PARTITION BY ticker, market ORDER BY trade_date)" if any(
        item.startswith("LAG(") for item in selects
    ) else ""
    return f"SELECT {', '.join(selects)} FROM {source or fx.GROUP_VIEW['flow']}{window}"


def requested_mart_columns() -> dict[str, list[str]]:
    """Mart -> the columns the fixed KR model reads from it (interaction inputs included)."""
    columns = list(KrBriefingSpec().feature_columns(20))
    materials = [
        material for output, material, _regime in fx.INTERACTIONS
        if output in columns and material not in columns
    ]
    return _required_marts([*columns, *materials])


def feature_mart_profile(marker: dict) -> str:
    """The build profile a feature marker declares (absent = ``full``); unknown fails."""
    profile = marker.get("profile", BUILD_PROFILE_FULL)
    if profile not in BUILD_PROFILES:
        raise ValueError(f"KR feature snapshot marker has an unknown build profile: {profile!r}")
    return profile


def _check_profile_contract(config: LakeConfig, marker: dict) -> None:
    """The marker's profile and the marts on disk must tell the same story.

    A serving-profile ``feat_fin_scan_daily`` holds the key columns and the projected ones
    only; reading it as the full mart (or a full mart as a projection) would be wrong, and
    a projection that lacks a column the model asks for would silently feed NULLs.
    """
    profile = feature_mart_profile(marker)
    metadata = mart_cache_metadata(config, "feat_fin_scan_daily") or {}
    projected = metadata.get("projected_columns")
    if profile == BUILD_PROFILE_FULL:
        if projected is not None or metadata.get("plan") == "projection":
            raise ValueError(
                "KR feature snapshot marker says full profile "
                "but feat_fin_scan_daily is a projection")
        return
    if not isinstance(projected, list) or metadata.get("plan") != "projection":
        raise ValueError(
            "KR feature snapshot marker says serving profile "
            "but feat_fin_scan_daily is not a projection")
    missing = sorted(set(requested_mart_columns().get("feat_fin_scan_daily", [])) - set(projected))
    if missing:
        raise ValueError(
            f"KR serving-profile feat_fin_scan_daily lacks requested columns: {missing}")
    recorded = next(
        (item for item in marker.get("marts", []) if item.get("view") == "feat_fin_scan_daily"), {})
    if recorded.get("projected_columns") != projected:
        raise ValueError("KR feat_fin_scan_daily projected columns differ from completion marker")


def _register_feature_marts(con: duckdb.DuckDBPyConnection, config: LakeConfig, names: list[str]) -> dict:
    contracts = {}
    for name in [UNIVERSE_VIEW, *names]:
        if not is_materialized(config, name):
            raise FileNotFoundError(f"required KR model mart is missing: {name}")
        glob = mart_glob(config, name).replace("'", "''")
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning=false)"
        )
        contracts[name] = mart_cache_metadata(config, name)
    return contracts


def _verify_universe_contract(config: LakeConfig, spec: KrBriefingSpec) -> str:
    stored = mart_cache_metadata(config, UNIVERSE_VIEW)
    expected = sql_contract_hash(build_universe_sql(spec.universe, price_view="daily_ohlcv"))
    if stored is None or stored.get("sql_hash") != expected:
        raise ValueError("dim_universe_daily SQL contract does not match the fixed KR universe")
    return expected


def _read_reference_evidence(path: Path | None, feature_asof_date: str) -> dict | None:
    """The reference-session evidence ``kr_reference`` wrote, checked against the requested date."""
    if path is None:
        return None
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(evidence, dict) or evidence.get("schema") != "kr-reference-selection.v1"
            or evidence.get("reference_date") != feature_asof_date
            or evidence.get("verdict") not in {"complete_K", "fallback_K_prime"}):
        raise ValueError("KR reference evidence does not name the requested feature_asof_date")
    return evidence


def _stock_names(con: duckdb.DuckDBPyConnection, config: LakeConfig) -> pl.DataFrame:
    """Display names from the raw ``stock_master`` of the same snapshot, one per (ticker, market).

    Names are display only: they are joined onto the prepared panel *after* the feature columns
    are fixed and the scorer never reads them as model input (``build_design_matrix`` selects the
    bundle's feature columns), so the design matrix stays the one the model was trained on.
    A snapshot without ``stock_master`` yields no names (the scorer then shows the code); the
    manifest records ``rows_named`` so this cannot go unnoticed.
    """
    try:
        register_views(con, config, tables=["stock_master"])
    except FileNotFoundError:
        return pl.DataFrame(schema={"ticker": pl.String, "market": pl.String, "name": pl.String})
    frame = pl.from_arrow(con.execute(
        "SELECT ticker, market, name FROM stock_master "
        "WHERE name IS NOT NULL AND trim(name) <> '' ORDER BY ticker, market").arrow())
    return frame.unique(subset=["ticker", "market"], keep="first", maintain_order=True)


def _check_snapshot_markers(
    config: LakeConfig, *, input_cutoff: str, feature_asof_date: str
) -> tuple[dict[str, str], dict[str, str]]:
    cutoff = datetime.fromisoformat(input_cutoff)
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("input_cutoff must include a timezone")
    cutoff_local = cutoff.astimezone(ZoneInfo("Asia/Seoul"))
    if cutoff_local.time().replace(tzinfo=None) != datetime_time(9, 30):
        raise ValueError("KR input cutoff is fixed at 09:30 Asia/Seoul")
    markers = {
        "raw": config.raw_root / "_manifests" / "_SUCCESS.json",
        "feature": config.feature_mart_root / "_manifests" / "_SUCCESS.json",
    }
    loaded: dict[str, str] = {}
    timestamps: dict[str, str] = {}
    for kind, path in markers.items():
        if not path.is_file():
            raise FileNotFoundError(f"KR {kind} snapshot has no completion marker: {path}")
        body = json.loads(path.read_text(encoding="utf-8"))
        stamp = body.get("finished_at") if kind == "raw" else body.get("created_at")
        if not stamp:
            raise ValueError(f"KR {kind} completion marker has no completion timestamp")
        completed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        if completed.tzinfo is None or completed > cutoff.astimezone(completed.tzinfo):
            raise ValueError(f"KR {kind} snapshot completed after the 09:30 input cutoff")
        if kind == "raw" and body.get("route") != "remote":
            raise ValueError("KR raw snapshot is not from the sj2 direct route")
        if kind == "raw":
            tables = body.get("tables", {})
            if not {"daily_ohlcv", "krx_security_flow_raw"} <= set(tables):
                raise ValueError("KR raw snapshot is missing price or flow input tables")
        if kind == "feature" and (
            body.get("status") != "success"
            or body.get("snapshot_date") != config.snapshot_date
            or body.get("source") != config.source
            or body.get("schema_version") != "kr-serving-feature-marts.v1"
            or body.get("feature_asof_date") != feature_asof_date
            or body.get("raw_marker_sha256") != loaded.get("raw")
        ):
            raise ValueError("KR feature snapshot marker does not match the requested source snapshot")
        if kind == "feature":
            entries = body.get("marts", [])
            if not isinstance(entries, list) or len(entries) != len(REQUIRED_PREP_MARTS):
                raise ValueError("KR feature snapshot marker has an incomplete mart inventory")
            marts = {item.get("view"): item for item in entries if isinstance(item, dict)}
            if set(marts) != set(REQUIRED_PREP_MARTS) or len(marts) != len(entries):
                raise ValueError("KR feature snapshot marker has duplicate or unexpected marts")
            for name in REQUIRED_PREP_MARTS:
                if marts[name].get("max_trade_date") != feature_asof_date:
                    raise ValueError(f"KR {name} is not complete through feature_asof_date={feature_asof_date}")
                metadata = mart_cache_metadata(config, name)
                if not metadata or not marts[name].get("sql_hash") or marts[name]["sql_hash"] != metadata.get("sql_hash"):
                    raise ValueError(f"KR {name} feature mart cache contract differs from completion marker")
            _check_profile_contract(config, body)
            cut = body.get("raw_cut")
            if cut is not None and (not isinstance(cut, dict) or cut.get("cut_asof") != feature_asof_date):
                raise ValueError("KR feature snapshot raw cut does not end at feature_asof_date")
        loaded[kind] = _sha256(path)
        timestamps[
            "raw_snapshot_completed_at" if kind == "raw" else "feature_marts_completed_at"
        ] = completed.isoformat()
    return loaded, timestamps


@dataclass(frozen=True)
class KrBriefingSpec:
    """Serving config pinned to E2 h20 FS1h and the balance-feature exclusion."""

    feature_set: str = "FS1h"
    flow_variant: str = "lag1"
    preprocess_profile: str = "rank"
    seed: int = 0
    universe: UniverseFilter = UniverseFilter(
        warmup_window=60,
        warmup_min_valid=40,
        liquidity_window=60,
        min_liquidity_krw=1e8,
        label_horizon=20,
        min_close_krw=0.0,
        apply_liquidity_filter=True,
        membership_reconstruction_available=False,
    )

    def feature_columns(self, horizon: int = 20) -> tuple[str, ...]:
        if horizon != 20:
            raise ValueError("KR briefing model is fixed at h20")
        return tuple(
            c for c in fx.feature_columns(self.feature_set, horizon)
            if c not in ("flow_short_balance_qty", "flow_short_balance_chg_20d")
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_completion_marker(published_dir: Path) -> dict:
    """Atomically attest that the renamed feature artifact is fully available."""
    feature_path = published_dir / "feature_panel.parquet"
    manifest_path = published_dir / "prepare_manifest.json"
    if not feature_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("published KR feature directory is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features_sha256 = _sha256(feature_path)
    manifest_sha256 = _sha256(manifest_path)
    if (
        manifest.get("features_sha256") != features_sha256
        or manifest.get("input_sha256") != features_sha256
    ):
        raise ValueError("published KR feature hash does not match native manifest")
    marker = {
        "schema_version": "prepared-features-completion.v1",
        "verified_available_by": datetime.now(UTC).isoformat(),
        "availability_evidence_type": "prepared_features_completion",
        "features_sha256": features_sha256,
        "native_prepare_manifest_sha256": manifest_sha256,
    }
    marker_path = published_dir / "completion.json"
    temporary = published_dir / ".completion.json.tmp"
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(marker, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, marker_path)
    return marker


def _check_completion_marker(published_dir: Path) -> dict:
    """Reject an incomplete or changed artifact, returning its pinned evidence."""
    marker_path = published_dir / "completion.json"
    if not marker_path.is_file():
        raise FileNotFoundError("KR feature output has no post-publish completion marker")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest_path = published_dir / "prepare_manifest.json"
    feature_path = published_dir / "feature_panel.parquet"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        marker.get("schema_version") != "prepared-features-completion.v1"
        or marker.get("availability_evidence_type") != "prepared_features_completion"
        or marker.get("features_sha256") != _sha256(feature_path)
        or marker.get("features_sha256") != manifest.get("features_sha256")
        or marker.get("native_prepare_manifest_sha256") != _sha256(manifest_path)
    ):
        raise ValueError("KR completion marker does not match the immutable prepared files")
    verified = datetime.fromisoformat(marker.get("verified_available_by", ""))
    if verified.tzinfo is None or verified.utcoffset() is None:
        raise ValueError("KR completion marker verified_available_by must include a timezone")
    return marker


def _query_sql(
    spec: KrBriefingSpec, asof_date: str, *, restrict_flow: bool = True,
) -> tuple[str, list[str], dict[str, list[str]]]:
    """The K cross-section query. ``restrict_flow=False`` is the original text, where the
    flow LAG runs over the whole mart; the rows are the same (E1 A5, 2.4 s -> 1.9 s)."""
    columns = list(spec.feature_columns(20))
    materials = [
        material for output, material, _regime in fx.INTERACTIONS
        if output in columns and material not in columns
    ]
    if set(columns) & {"flow_short_balance_qty", "flow_short_balance_chg_20d"}:
        raise AssertionError("short-balance feature leaked into KR serving contract")
    needed = _required_marts([*columns, *materials])
    flow_view = fx.GROUP_VIEW["flow"]
    ctes: list[str] = []
    joins: list[str] = []
    selects = [f"u.{c}" for c in KEY_COLS]
    selects.extend(f"q.{c}" for c in QUALITY_COLS)
    joins.append(f"LEFT JOIN {QUALITY_VIEW} AS q USING ({', '.join(KEY_COLS)})")
    for mart, mart_columns in needed.items():
        alias = MART_ALIAS[mart]
        source = mart
        if mart == flow_view:
            if restrict_flow:
                ctes.append(_flow_rows_cte(flow_view, asof_date))
                flow_sql = _flow_sql(spec, mart_columns, source="flow_rows")
            else:
                flow_sql = _flow_sql(spec, mart_columns)
            ctes.append(f"flow_src AS ({flow_sql})")
            source = "flow_src"
        join_keys = "trade_date" if mart in DATE_GRAIN_MARTS else "trade_date, ticker, market"
        joins.append(f"LEFT JOIN {source} AS {alias} USING ({join_keys})")
        selects.extend(f"{alias}.{c}" for c in mart_columns)
    with_clause = "WITH " + ",\n".join(ctes) + "\n" if ctes else ""
    sql = f"""
        {with_clause}
        SELECT {", ".join(selects)}
        FROM {UNIVERSE_VIEW} u
        {" ".join(joins)}
        WHERE u.in_universe AND u.trade_date = DATE '{asof_date}'
        ORDER BY u.trade_date, u.ticker, u.market
    """
    return sql, columns, needed


def prepare_cross_section(
    *, snapshot_date: str, feature_asof_date: str, output_dir: Path,
    input_cutoff: str, stock_data_root: Path | None = None, source: str = REMOTE_SOURCE,
    reference_evidence: Path | None = None,
) -> dict:
    """Read the specified ready snapshot and persist label-free features of the reference session.

    ``feature_asof_date`` is K, or an earlier K' when ``kr_reference`` found K incomplete.
    ``reference_evidence`` (that module's JSON) is embedded in the manifest.
    """
    if date.fromisoformat(snapshot_date).isoformat() != snapshot_date:
        raise ValueError("snapshot_date must use YYYY-MM-DD")
    if date.fromisoformat(feature_asof_date).isoformat() != feature_asof_date:
        raise ValueError("feature_asof_date must use YYYY-MM-DD")
    if source != REMOTE_SOURCE:
        raise ValueError("KR serving inputs must use the sj2 remote source")
    if stock_data_root is None:
        root = DataRoot.resolve(market="kr")
    else:
        root = DataRoot(base=stock_data_root / "kr")
    config = LakeConfig(root=root, snapshot_date=snapshot_date, source=source)
    reference = _read_reference_evidence(reference_evidence, feature_asof_date)
    marker_hashes, marker_times = _check_snapshot_markers(
        config, input_cutoff=input_cutoff, feature_asof_date=feature_asof_date
    )
    if output_dir.exists():
        prior_path = output_dir / "prepare_manifest.json"
        prior_panel = output_dir / "feature_panel.parquet"
        if prior_path.is_file() and prior_panel.is_file():
            prior = json.loads(prior_path.read_text(encoding="utf-8"))
            if (
                prior.get("snapshot_date") == snapshot_date
                and prior.get("feature_asof_date") == feature_asof_date
                and prior.get("source_marker_hashes") == marker_hashes
                and prior.get("files", {}).get("feature_panel.parquet") == _sha256(prior_panel)
            ):
                _check_completion_marker(output_dir)
                return prior
        raise FileExistsError(f"KR feature output already exists with a different input: {output_dir}")
    spec = KrBriefingSpec()
    columns = list(spec.feature_columns(20))
    materials = [
        material for output, material, _regime in fx.INTERACTIONS
        if output in columns and material not in columns
    ]
    marts = sorted(set(_required_marts([*columns, *materials])) | {QUALITY_VIEW})
    con = connect(config)
    try:
        # Only raw prices are registered for the universe contract check. No
        # label dataset or label builder is imported by this module.
        register_views(con, config, tables=["daily_ohlcv"])
        contracts = _register_feature_marts(con, config, marts)
        _verify_universe_contract(config, spec)
        query, columns, needed = _query_sql(spec, feature_asof_date)
        panel = pl.from_arrow(con.execute(query).arrow())
        names = _stock_names(con, config)
    finally:
        con.close()
    if panel.is_empty():
        raise ValueError(f"no eligible KR rows for feature_asof_date={feature_asof_date}")
    if set(panel.get_column("trade_date").cast(pl.String).unique().to_list()) != {feature_asof_date}:
        raise ValueError("prepared panel contains a feature date other than the requested K")
    interactions = [item for item in fx.INTERACTIONS if item[0] in columns]
    if interactions:
        exprs = []
        for output, material, regime in interactions:
            if material not in panel.columns or regime not in panel.columns:
                raise KeyError(f"interaction {output!r} misses {material!r} or {regime!r}")
            rank = pp.per_date_rank_expr(material, ["trade_date", "market"])
            exprs.append(
                pl.when(pl.col(material).is_null() | pl.col(regime).is_null())
                .then(None).otherwise(rank * pl.col(regime).cast(pl.Float64))
                .cast(pl.Float64).alias(output)
            )
        panel = panel.with_columns(exprs)
    panel = panel.select([*KEY_COLS, *columns, *QUALITY_COLS]).sort(["trade_date", "ticker", "market"])
    # Display names (F3). Added after the feature columns are fixed, never part of the model input.
    panel = panel.join(names, on=["ticker", "market"], how="left", validate="m:1").sort(
        ["trade_date", "ticker", "market"])
    named_rows = int(panel.get_column("name").is_not_null().sum())
    if any(name.startswith(("y_", "raw_label", "fwd_ret_", "bench_ret_")) for name in panel.columns):
        raise AssertionError("label columns reached the KR serving feature panel")
    if any(name in panel.columns for name in ("flow_short_balance_qty", "flow_short_balance_chg_20d")):
        raise AssertionError("excluded balance columns reached the KR serving feature panel")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    panel_path = temp_dir / "feature_panel.parquet"
    try:
        panel.write_parquet(panel_path, compression="zstd")
        null_counts = panel.select([pl.col(c).null_count().alias(c) for c in columns]).row(0)
        null_ratio = {c: round(n / panel.height, 6) for c, n in zip(columns, null_counts, strict=True)}
        manifest = {
            "schema_version": "kr-feature-panel.v1",
            "market": "KR",
            "snapshot_date": snapshot_date,
            "feature_asof_date": feature_asof_date,
            "input_cutoff": input_cutoff,
            # Export/mart completion proves snapshot completeness, not when the
            # source data first became available. Leave this null until the
            # collector supplies timestamped availability evidence.
            "source_available_at": None,
            "source_availability_evidence": None,
            "raw_snapshot_completed_at": marker_times["raw_snapshot_completed_at"],
            "feature_marts_completed_at": marker_times["feature_marts_completed_at"],
            "feature_build_completed_at": datetime.now(UTC).isoformat(),
            "generated_at": datetime.now(UTC).isoformat(),
            "availability_evidence_type": "prepared_features_completion",
            "source": source,
            "source_code_revision": current_git_sha(),
            "model_id": "kr_daily_h20_v1",
            "model_config": {
                "feature_set": "FS1h", "flow_variant": "lag1", "preprocess_profile": "rank",
                "horizon_sessions": 20, "feature_columns": columns,
                "excluded_features": ["flow_short_balance_qty", "flow_short_balance_chg_20d"],
                "feature_time_contract": {
                    "formation_date": feature_asof_date,
                    "price_features": "derived from completed OHLCV through K; no extra serving lag",
                    "flow_features": "model-02 lag1 mapping applied once at the per-ticker previous valid row",
                    "serving_applies_additional_lag": False,
                },
            },
            "quality": {
                "eligible_rows": panel.height,
                "null_ratio_by_feature": null_ratio,
                "management_filter_available": False,
                "management_state": "unverified",
                "trading_halt_check": "K feature mart only; does not establish D intraday tradability",
                "price_jump_review": "not_checked",
                "price_quality_fields": list(QUALITY_COLS),
                "display_columns": ["name"],
                "display_name_source": {
                    "table": "stock_master", "snapshot_date": snapshot_date,
                    "rows_named": named_rows, "rows_unnamed": panel.height - named_rows,
                    "model_input": False},
            },
            "reference_selection": reference,
            "mart_contracts": contracts,
            "feature_mart_profile": feature_mart_profile(json.loads(
                (config.feature_mart_root / "_manifests" / "_SUCCESS.json").read_text(
                    encoding="utf-8"))),
            "source_marker_hashes": marker_hashes,
            "required_marts": sorted(set(needed) | {QUALITY_VIEW}),
            "files": {"feature_panel.parquet": _sha256(panel_path)},
            "features_sha256": _sha256(panel_path),
            "input_sha256": _sha256(panel_path),
        }
        (temp_dir / "prepare_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
        )
        temp_dir.rename(output_dir)
        _write_completion_marker(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--feature-asof-date", required=True)
    parser.add_argument("--input-cutoff", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stock-data-root", type=Path)
    parser.add_argument("--source", default=REMOTE_SOURCE)
    parser.add_argument("--reference-evidence", type=Path,
                        help="kr_reference JSON naming the reference session; embedded in the manifest")
    args = parser.parse_args()
    manifest = prepare_cross_section(
        snapshot_date=args.snapshot_date,
        feature_asof_date=args.feature_asof_date,
        input_cutoff=args.input_cutoff,
        output_dir=args.output_dir,
        stock_data_root=args.stock_data_root,
        source=args.source,
        reference_evidence=args.reference_evidence,
    )
    print(json.dumps({"rows": manifest["quality"]["eligible_rows"], "feature_asof_date": manifest["feature_asof_date"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
