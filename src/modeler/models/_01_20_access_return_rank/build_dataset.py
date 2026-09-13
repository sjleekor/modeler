"""build_dataset — assemble model 01's per-model dataset (L2b, etl_00 §4.1, §7).

Pipeline (etl_03 P5):
  1. register lake views (P1) and MATERIALIZE the shared feature marts
     (universe/price/flow/fin/common/event) to ``data_lake/feature_mart/`` —
     snapshot-cached, skip-if-present (00_shared §1, §5). The label stays an
     in-memory view (model-specific, cheap).
  2. assemble the panel via DuckDB as-of join (etl_00 §4.1): universe (in_universe)
     LEFT JOIN mart features/label on (trade_date, ticker, market); cf broadcasts
     on trade_date.
  3. hand the panel to Polars (zero-copy via Arrow, etl_02 §6).
  4. build purged walk-forward folds over the trading dates (P5/splits); the
     holdout fold's eval slice is tagged fold_role="holdout" (excluded from
     selection).
  5. fit preprocessing on each fold's TRAIN slice, transform all slices, write
     feat_panel(_std) / label_daily / split_folds parquet + dataset_manifest.json.

The heavy DuckDB work (76M flow dedup, financial PIT join) is materialized ONCE
per snapshot and reused across builds; preprocessing/standardization is Polars
per the engine split (etl_02 §5).

Outputs land under ``config.dataset_dir(model_id)`` (``data/datasets/...``), which
is gitignored (etl_01 §7).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from modeler.etl import labels as labels_mod
from modeler.etl import preprocess as pp
from modeler.etl.config import LakeConfig
from modeler.etl.features.common import materialize_common
from modeler.etl.features.event import materialize_event
from modeler.etl.features.fin_pit import materialize_fin_pit
from modeler.etl.features.flow import materialize_flow
from modeler.etl.features.price import materialize_price
from modeler.etl.lake import connect, register_derived_marts, register_views
from modeler.etl.manifest import build_manifest
from modeler.etl.mart import is_materialized, mart_root, register_mart_view
from modeler.etl.splits import Fold, walk_forward_splits
from modeler.etl.universe import materialize_universe
from modeler.models._01_20_access_return_rank.spec import ModelSpec

# feature-group prefix -> the mart view it comes from (00_shared §2, etl_00 §4.5).
_GROUP_VIEW = {
    "px": "feat_price",
    "flow": "feat_flow",
    "fin": "feat_fin_pit",
    "cf": "feat_common",
    "ev": "feat_event",
}


@dataclass
class BuildResult:
    """Outcome of a dataset build (paths + counts)."""

    dataset_dir: Path
    panel_rows: int
    n_folds: int
    feature_cols: list[str]
    manifest_path: Path


def _materialize_source_marts(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    spec: ModelSpec,
    *,
    force: bool = False,
    reuse_existing: bool = False,
) -> None:
    """Materialize the shared feature marts (L2a) and register their views.

    The heavy DuckDB work (76M flow dedup, financial PIT join) runs ONCE per
    snapshot and is written to ``data_lake/feature_mart/`` (00_shared §1, §5);
    subsequent builds for the same snapshot skip the rebuild unless ``force``.
    The model then joins these light parquet marts (Fix #2 / etl_03 P5).

    The universe mart is materialized first because ``feat_fin_pit`` /
    ``feat_event`` read it for their PIT interval joins. The label is left as an
    in-memory view: it is model-specific (depends on ``LabelSpec``) and cheap.
    """
    materialize_universe(con, config, spec.universe, force=force)
    if "px" in spec.feature_groups:
        if reuse_existing and is_materialized(config, "feat_price"):
            register_mart_view(con, config, "feat_price")
        else:
            materialize_price(con, config, force=force)
    if "flow" in spec.feature_groups:
        if reuse_existing and is_materialized(config, "feat_flow"):
            register_mart_view(con, config, "feat_flow")
        else:
            materialize_flow(con, config, force=force)
    if "fin" in spec.feature_groups:
        materialize_fin_pit(con, config, force=force)
    if "cf" in spec.feature_groups:
        materialize_common(con, config, force=force)
    if "ev" in spec.feature_groups:
        materialize_event(con, config, force=force)
    con.execute(f"CREATE VIEW label_daily AS {labels_mod.build_label_sql(spec.label)}")


def _panel_sql(spec: ModelSpec) -> str:
    """As-of join panel SQL (etl_00 §4.1): in-universe rows + feature groups + labels."""
    joins = []
    selects = ["u.trade_date", "u.ticker", "u.market"]
    for group in spec.feature_groups:
        view = _GROUP_VIEW[group]
        alias = group
        if group == "cf":
            # common/macro features are stock-agnostic -> broadcast on date only.
            joins.append(f"LEFT JOIN {view} AS {alias} USING (trade_date)")
            selects.append(f"{alias}.* EXCLUDE (trade_date)")
        else:
            joins.append(f"LEFT JOIN {view} AS {alias} USING (trade_date, ticker, market)")
            # all non-key columns of the group view, by prefix
            selects.append(f"{alias}.* EXCLUDE (trade_date, ticker, market)")
    # label columns (y_* + raw_label + fwd/bench) from label_daily
    joins.append("LEFT JOIN label_daily AS l USING (trade_date, ticker, market)")
    selects.append("l.* EXCLUDE (trade_date, ticker, market)")

    # Without an explicit ORDER BY, DuckDB's parallel scan/join can return rows
    # in a different order on every execution; HGB's histogram splits are not
    # perfectly row-order-invariant under floating-point summation, so an
    # unordered panel makes walk-forward/holdout metrics irreproducible run to
    # run even with a fixed random_state (observed directly: repeated builds
    # of the same spec differed in row order and downstream Rank IC).
    return f"""
        SELECT {", ".join(selects)}
        FROM dim_universe_daily u
        {" ".join(joins)}
        WHERE u.in_universe
          AND u.trade_date BETWEEN DATE '{spec.period_start}' AND DATE '{spec.period_end}'
        ORDER BY u.trade_date, u.ticker, u.market
    """


def assemble_panel(con: duckdb.DuckDBPyConnection, spec: ModelSpec) -> pl.DataFrame:
    """Run the panel join and return a Polars DataFrame (Arrow zero-copy)."""
    arrow_tbl = con.execute(_panel_sql(spec)).arrow()
    return pl.from_arrow(arrow_tbl)


def _label_col(spec: ModelSpec) -> str:
    return f"y_rank_{spec.primary_horizon}d"


def _restrict_to_feature_cols(panel: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Drop raw feature columns not in ``feature_cols`` (keys/labels untouched.

    Used by ``feature_cols_override`` to pin an exact feature set (e.g. an
    acceptance-gate baseline vs candidate comparison) even though
    ``_panel_sql``'s ``{alias}.* EXCLUDE (...)`` pulls in every column a mart
    currently exposes. ``research/etl/preprocess.py::feature_columns`` scans
    whatever numeric columns survive here, so this is the only place a caller
    needs to intervene — standardization/``*_isna``/design columns downstream
    adapt automatically.
    """
    key_cols = {"trade_date", "ticker", "market"}
    label_prefixes = ("y_", "raw_label", "fwd_ret_", "bench_ret_")
    keep = set(feature_cols)
    cols = [c for c in panel.columns if c in key_cols or c.startswith(label_prefixes) or c in keep]
    return panel.select(cols)


def build_dataset(
    spec: ModelSpec | None = None,
    config: LakeConfig | None = None,
    *,
    created_at: str | None = None,
    write: bool = True,
    force_mart: bool = False,
    reuse_existing_marts: bool = False,
    feature_cols_override: list[str] | None = None,
) -> BuildResult:
    """Build (and optionally write) model 01's dataset. Returns a :class:`BuildResult`.

    ``write=False`` runs the full pipeline in-memory (used by tests/dry-runs) and
    skips parquet/manifest output. ``force_mart=True`` rebuilds the shared feature
    marts even if already materialized for this snapshot (00_shared §5).
    ``reuse_existing_marts=True`` registers already-materialized source marts
    without recomputing them. This is intended for consumers such as the
    acceptance gate that must read the official A0 quality-aware marts without
    clobbering them with a different SQL contract.
    ``feature_cols_override``, if given, restricts the assembled panel to exactly
    these raw feature columns (plus keys/labels) before standardization/folds —
    for pinning a baseline/candidate column set independent of whatever extra
    columns the current ``feat_price``/``feat_flow`` marts happen to expose.
    """
    spec = spec or ModelSpec()
    if config is None:
        raise ValueError("build_dataset() requires an explicit config (LakeConfig)")

    con = connect(config)
    # Raw lake views. The derived facts (stock_metric_fact / common_feature_daily_fact)
    # are no longer read from the canonical lake — they are recomputed from raw by the
    # DuckDB marts and registered under the same view names (refactor §3.3).
    lake_tables = ["daily_ohlcv", "krx_security_flow_raw"]
    if "fin" in spec.feature_groups:
        lake_tables.extend(
            [  # raw inputs for the stock_metric_fact mart
                "dart_financial_statement_raw",
                "dart_share_count_raw",
                "dart_shareholder_return_raw",
                "dart_xbrl_fact_raw",
                "dart_corp_master",
            ]
        )
    if "cf" in spec.feature_groups:
        lake_tables.append("common_feature_observation_raw")  # mart input
    if "ev" in spec.feature_groups:
        lake_tables.append("dart_share_count_raw")  # raw lake (P8)
    # common_feature_series is the shared collector config (decision 7). Add it
    # only if exported; until then the mart falls back to the identical code
    # definition, so registering it must not be mandatory.
    if "cf" in spec.feature_groups and config.table_glob("common_feature_series"):
        from modeler.etl.lake import _glob_has_files

        if _glob_has_files(con, config.table_glob("common_feature_series")):
            lake_tables.append("common_feature_series")
    register_views(con, config, tables=sorted(set(lake_tables)))

    derived: list[str] = []
    if "fin" in spec.feature_groups:
        derived.append("stock_metric_fact")
    if "cf" in spec.feature_groups:
        derived.append("common_feature_daily_fact")
    if derived:
        register_derived_marts(con, config, which=derived)

    _materialize_source_marts(
        con,
        config,
        spec,
        force=force_mart,
        reuse_existing=reuse_existing_marts,
    )

    panel = assemble_panel(con, spec)
    if feature_cols_override is not None:
        panel = _restrict_to_feature_cols(panel, feature_cols_override)
    panel_rows = panel.height

    # trading dates available in the panel (sorted unique) -> folds.
    dates = panel.get_column("trade_date").unique().sort().to_list()
    folds = walk_forward_splits(
        dates,
        horizon=spec.primary_horizon,
        embargo=spec.embargo,
        purge=spec.purge,
        n_folds=spec.n_folds,
        holdout_len=spec.holdout_len,
    )

    cfg_pp = pp.PreprocessConfig(profile=spec.preprocess_profile)
    label_col = _label_col(spec)

    # Standardize per fold: fit on train slice, transform train + eval slice
    # (etl_00 §4.3). The eval slice of a holdout fold is labeled "holdout" (NOT
    # "valid") so it is excluded from walk-forward selection and is only used by
    # the single post-selection evaluate_holdout() pass (etl_00 §5).
    std_frames: list[pl.DataFrame] = []
    feature_cols: list[str] = []
    for fold in folds:
        train = panel.filter(
            (pl.col("trade_date") >= fold.train_start) & (pl.col("trade_date") <= fold.train_end)
        )
        if train.height == 0:
            continue
        fitted = pp.fit(train, cfg_pp)
        feature_cols = fitted.feature_cols
        eval_role = "holdout" if fold.role == "holdout" else "valid"
        for role, start, end in (
            ("train", fold.train_start, fold.train_end),
            (eval_role, fold.valid_start, fold.valid_end),
        ):
            sl = panel.filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
            if sl.height == 0:
                continue
            out = fitted.transform(sl)
            out = out.with_columns(
                pl.lit(fold.fold_id).alias("fold_id"),
                pl.lit(role).alias("fold_role"),
            )
            std_frames.append(out)

    panel_std = pl.concat(std_frames, how="diagonal_relaxed") if std_frames else panel.clear()

    # L1 gate: model-input features must be finite for linear models.
    # Tree models (HGB) handle NaNs natively.
    if spec.preprocess_profile == "linear" and panel_std.height and feature_cols:
        pp.assert_finite(panel_std, feature_cols)

    result_dir = config.dataset_dir(spec.model_id)
    manifest_path = result_dir / "dataset_manifest.json"

    if write:
        _write_artifacts(result_dir, panel, panel_std, folds, label_col)
        manifest = build_manifest(
            model_id=spec.model_id,
            config=config,
            feature_groups=list(spec.feature_groups),
            label_spec={
                "horizons": list(spec.label.horizons),
                "kind": spec.label.kind,
                "bench": spec.label.bench,
                "outputs": list(spec.label.outputs),
                "primary_target": label_col,
            },
            universe_filter={
                "warmup_window": spec.universe.warmup_window,
                "warmup_min_valid": spec.universe.warmup_min_valid,
                "min_liquidity_krw": spec.universe.min_liquidity_krw,
                "label_horizon": spec.universe.label_horizon,
            },
            period={"start": spec.period_start, "end": spec.period_end},
            row_count=panel_rows,
            mart_root=str(mart_root(config)),
            created_at=created_at,
            extra={"n_folds": len([f for f in folds if f.role == "fold"])},
        )
        manifest.write(manifest_path)

    return BuildResult(
        dataset_dir=result_dir,
        panel_rows=panel_rows,
        n_folds=len([f for f in folds if f.role == "fold"]),
        feature_cols=feature_cols,
        manifest_path=manifest_path,
    )


def _write_artifacts(
    result_dir: Path,
    panel: pl.DataFrame,
    panel_std: pl.DataFrame,
    folds: list[Fold],
    label_col: str,
) -> None:
    """Write feat_panel / feat_panel_std / label_daily / split_folds parquet."""
    result_dir.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(result_dir / "feat_panel.parquet")
    panel_std.write_parquet(result_dir / "feat_panel_std.parquet")

    label_cols = [c for c in panel.columns if c.startswith("y_") or c.startswith("raw_label")]
    panel.select(["trade_date", "ticker", "market", *label_cols]).write_parquet(
        result_dir / "label_daily.parquet"
    )

    split_df = pl.DataFrame([f.as_record() for f in folds])
    split_df.write_parquet(result_dir / "split_folds.parquet")
