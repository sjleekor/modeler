"""build_dataset — assemble model 02's panel from marts that already exist.

Plan: ``docs/dev/20260907_model_experiment/06_implementation_plan.md`` §2.2,
`02` §2 (PIT), §3.2 (interactions), §5 (universe, mart contract).

The difference from model 01's builder is what it is *not allowed* to do. Every
mart this model reads was published by a horizon-scan A0 run on snapshot
2026-08-23, and rebuilding one under a different contract hash would break the
comparability the whole experiment rests on (`02` §5). So the marts are
registered **read-only**, their stored contracts are copied into the dataset
manifest, and nothing here calls ``materialize``.

Three consequences worth knowing before reading the code.

**The contract hashes on that snapshot are not uniform.** A0 stamped its marts
with its config hash (``236d0d35…``); ``dim_universe_daily`` carries a different
one because a different path wrote it. ``register_mart_view``'s check is
all-or-nothing, so it cannot register both from one ``LakeConfig`` —
:func:`register_read_only` registers each as it lies and records what it found.
For the universe mart, which this model *could* define itself, the check is
replaced by a stronger one: its stored ``sql_hash`` must equal the hash of the
SQL this spec's ``UniverseFilter`` would produce (D-5), so reading it is
verified rather than assumed.

**The flow lag is a plain ``LAG`` over the mart's own rows.** That is correct
only because A0's ``feat_flow`` holds no halt sessions — ``flow.py`` filters on
``dim_price_quality_daily.valid_session_idx`` before any window runs (measured
on 2026-08-23: 0 halt rows out of 6,572,831). So "previous row" is "previous
valid session", which is what `02` §2 asks for.

**Interactions are built here, from the same percentile the rank profile uses.**
``ix_* = pct_rank(material) x regime`` via
:func:`modeler.etl.preprocess.per_date_rank_expr`, and the materials are then
dropped: they are Grade D on their own and FS2 carries only the interaction.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import duckdb
import polars as pl

from modeler.etl import labels as labels_mod
from modeler.etl import preprocess as pp
from modeler.etl.config import LakeConfig
from modeler.etl.lake import connect, register_views
from modeler.etl.manifest import build_manifest
from modeler.etl.mart import (
    is_materialized,
    mart_cache_metadata,
    mart_glob,
    mart_root,
    sql_contract_hash,
)
from modeler.etl.splits import Fold, walk_forward_splits
from modeler.etl.universe import build_universe_sql
from modeler.models._02_updown_prob import features as fx
from modeler.models._02_updown_prob.spec import HOLDOUT_START, ModelSpec

KEY_COLS = ("trade_date", "ticker", "market")
UNIVERSE_VIEW = "dim_universe_daily"
LABEL_VIEW = "label_daily"
PRICE_TABLE = "daily_ohlcv"
INDEX_TABLE = "common_feature_observation_raw"  # only with label.index_bench (R5)
# The design matrix is a directory of parquet parts, not one file — see
# :func:`_write_preprocessed` for why.
STD_DIR = "feat_panel_std"

# Marts joined on ``trade_date`` alone — one row per session, broadcast across
# the cross-section at join time.
DATE_GRAIN_MARTS: frozenset[str] = frozenset({"feat_common", fx.GROUP_VIEW["rg"]})

# mart view -> the short alias the panel SQL uses for it.
_MART_ALIAS: dict[str, str] = {view: group for group, view in fx.GROUP_VIEW.items()}


@dataclass
class BuildResult:
    """Outcome of a build: where it landed and what it contains."""

    dataset_dir: Path
    horizon: int
    panel_rows: int
    n_dates: int
    n_folds: int
    feature_cols: list[str]
    mart_contracts: dict[str, dict | None]
    manifest_path: Path


def dataset_key(spec: ModelSpec, horizon: int) -> str:
    """Directory name for one panel: the knobs that change its contents.

    The seed and the model family are *not* in it — they change the fit, not the
    panel, so runs that differ only there share one build (`05` §3 "build 캐시").

    The label spec is not in it either, which is fine only while every build
    shares one. ``index_bench`` breaks that, so it gets a suffix: without one,
    the index-benchmark panel would land on the plain one's directory and
    ``ensure_dataset`` would hand the next run whichever was built first — it
    compares the feature knobs in the manifest, not the labels. The suffix is
    added only when the flag is on, so existing paths are untouched (R5,
    2026-09-18).
    """
    key = f"{spec.feature_set}_h{horizon}_{spec.flow_variant}_{spec.preprocess_profile}"
    return f"{key}_idxbench" if spec.label.index_bench else key


def panel_feature_columns(spec: ModelSpec, horizon: int) -> tuple[list[str], list[str]]:
    """``(feature_cols, material_cols)`` for one config.

    ``feature_cols`` is the feature set itself. ``material_cols`` are the
    interaction inputs that have to be pulled from a mart but must not survive
    into the design matrix.
    """
    columns = list(spec.feature_columns(horizon))
    interactions = [i for i in fx.INTERACTIONS if i[0] in columns]
    materials = [
        material
        for _name, material, _regime in interactions
        if material not in columns  # a material that is also a feature stays one
    ]
    return columns, list(dict.fromkeys(materials))


def required_mart_columns(columns: list[str]) -> dict[str, list[str]]:
    """mart view -> the columns to read from it, for a list of feature names.

    Interaction outputs are skipped: they do not exist in any mart.
    """
    needed: dict[str, list[str]] = {}
    built = set(fx.FS2_INTERACTION_COLS)
    for column in columns:
        if column in built:
            continue
        mart = fx.COLUMN_MART.get(column)
        if mart is None:
            raise KeyError(f"no mart mapped for feature column {column!r}")
        needed.setdefault(mart, []).append(column)
    return needed


def mart_source_columns(spec: ModelSpec, columns: list[str]) -> dict[str, list[str]]:
    """mart view -> the columns the panel SQL will actually read from it.

    Differs from :func:`required_mart_columns` for the flow mart under
    ``lag1``: the two ratio features are read from their ``_lag1`` twins, so
    those are the columns that have to exist. A pre-flight check (and a test
    fixture) wants this list, not the feature names.
    """
    needed = required_mart_columns(columns)
    flow_view = fx.GROUP_VIEW["flow"]
    if spec.flow_variant == "lag1" and flow_view in needed:
        needed[flow_view] = [fx.FLOW_MART_LAG1_SOURCE.get(c, c) for c in needed[flow_view]]
    return needed


def register_read_only(
    con: duckdb.DuckDBPyConnection, config: LakeConfig, marts: list[str]
) -> dict[str, dict | None]:
    """Register each mart exactly as it lies on disk; return the stored contracts.

    Deliberately *not* ``register_mart_view``: that refuses a mart whose
    ``analysis_config_hash`` differs from the caller's, which on this snapshot
    is the normal state (see the module docstring). Nothing is rebuilt, so the
    contract is recorded for the manifest rather than enforced.
    """
    contracts: dict[str, dict | None] = {}
    for name in marts:
        if not is_materialized(config, name):
            raise FileNotFoundError(
                f"mart {name!r} is not materialized under {mart_root(config)}. "
                "Model 02 never builds a mart — build it with its own module first."
            )
        glob = mart_glob(config, name).replace("'", "''")
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_parquet('{glob}', hive_partitioning=false)"
        )
        contracts[name] = mart_cache_metadata(config, name)
    return contracts


def verify_universe_contract(config: LakeConfig, spec: ModelSpec) -> str:
    """Assert the stored universe mart is the one ``spec.universe`` defines (D-5).

    Returns the verified ``sql_hash``. This is the check that makes reading
    someone else's ``dim_universe_daily`` safe: the thresholds are in the SQL
    text, so an equal hash means an equal filter — a broad or tradable universe
    would not match.
    """
    stored = mart_cache_metadata(config, UNIVERSE_VIEW)
    expected = sql_contract_hash(build_universe_sql(spec.universe, price_view=PRICE_TABLE))
    if stored is None or stored.get("sql_hash") != expected:
        found = None if stored is None else stored.get("sql_hash")
        raise ValueError(
            f"{UNIVERSE_VIEW} on snapshot {config.snapshot_date} was not built with this "
            f"spec's UniverseFilter (D-5): stored sql_hash {found!r} != {expected!r}"
        )
    return expected


def _flow_source_sql(spec: ModelSpec, columns: list[str], flow_view: str) -> str:
    """The flow CTE, timed by ``spec.flow_variant`` (`02` §2, D-6).

    ``lag1`` (the validated canon) reads yesterday's value under today's name:
    FS0's 15 columns have no ``_lag1`` twin in the mart so they are lagged here,
    the ratio pair does have one and is read from it. ``native_t`` reads
    everything as the mart has it — the E3-a variant whose whole point is to
    measure what the unverified same-day aggregate would look like.

    A column already spelled with ``_lag1`` (the interaction material) is read
    verbatim under either variant: that suffix is part of its definition.
    """
    selects: list[str] = [*KEY_COLS]
    for column in columns:
        if column.endswith("_lag1") or spec.flow_variant == "native_t":
            selects.append(column)
        elif column in fx.FLOW_MART_LAG1_SOURCE:
            selects.append(f"{fx.FLOW_MART_LAG1_SOURCE[column]} AS {column}")
        elif column in fx.FLOW_BUILDER_LAG_COLUMNS:
            selects.append(f"LAG({column}) OVER w AS {column}")
        else:
            selects.append(column)
    window = (
        "\n            WINDOW w AS (PARTITION BY ticker, market ORDER BY trade_date)"
        if any(s.startswith("LAG(") for s in selects)
        else ""
    )
    return f"""
        SELECT {", ".join(selects)}
        FROM {flow_view}{window}
    """


def build_panel_sql(spec: ModelSpec, horizon: int) -> str:
    """As-of join panel SQL: in-universe rows + the config's columns + labels.

    Columns are named explicitly rather than ``alias.* EXCLUDE (keys)``: the
    feature set is a fixed list, and pulling every column a 1.4GB mart happens
    to expose would be both slower and a silent way for an unregistered feature
    to reach the model.

    ``ORDER BY`` is not cosmetic. DuckDB's parallel joins return rows in a
    different order run to run, and HGB's histogram splits are not exactly
    row-order invariant, so an unordered panel makes a fixed seed reproduce
    nothing (model 01 measured this).
    """
    columns, materials = panel_feature_columns(spec, horizon)
    needed = required_mart_columns([*columns, *materials])

    flow_view = fx.GROUP_VIEW["flow"]
    ctes: list[str] = []
    joins: list[str] = []
    selects: list[str] = [f"u.{c}" for c in KEY_COLS]

    for mart, mart_columns in needed.items():
        alias = _MART_ALIAS[mart]
        source = mart
        if mart == flow_view:
            ctes.append(f"flow_src AS ({_flow_source_sql(spec, mart_columns, flow_view)})")
            source = "flow_src"
        if mart in DATE_GRAIN_MARTS:
            joins.append(f"LEFT JOIN {source} AS {alias} USING (trade_date)")
        else:
            joins.append(f"LEFT JOIN {source} AS {alias} USING (trade_date, ticker, market)")
        selects.extend(f"{alias}.{c}" for c in mart_columns)

    joins.append(f"LEFT JOIN {LABEL_VIEW} AS l USING (trade_date, ticker, market)")
    selects.append("l.* EXCLUDE (trade_date, ticker, market)")

    with_clause = "WITH " + ",\n        ".join(ctes) + "\n" if ctes else ""
    return f"""
        {with_clause}SELECT {", ".join(selects)}
        FROM {UNIVERSE_VIEW} u
        {" ".join(joins)}
        WHERE u.in_universe
          AND u.trade_date BETWEEN DATE '{spec.period_start}' AND DATE '{spec.period_end}'
        ORDER BY u.trade_date, u.ticker, u.market
    """


def add_interactions(panel: pl.DataFrame, interactions: list[tuple[str, str, str]]) -> pl.DataFrame:
    """Append ``ix_* = pct_rank(material) x regime`` (`02` §3.2).

    The percentile comes from :func:`modeler.etl.preprocess.per_date_rank_expr`
    — the same function the rank profile uses, so the two cannot drift.

    Three rules, all of them deliberate:

    * regime 0 -> 0. A session where the regime is off contributes nothing, and
      the tree reads ``rg_*`` beside ``ix_*`` to know which zero it is looking at.
    * material NULL -> NULL, so a missing input is missing rather than neutral.
    * regime NULL -> NULL. Before ``REGIME_VALID_FROM`` the regime is unknown,
      not off; calling it 0 would invent a partition for the first months of the
      panel, and the ``*_isna`` flag records the gap instead.
    """
    group = [c for c in ("trade_date", "market") if c in panel.columns]
    exprs = []
    for name, material, regime in interactions:
        for column in (material, regime):
            if column not in panel.columns:
                raise KeyError(f"interaction {name!r} needs column {column!r} in the panel")
        rank = pp.per_date_rank_expr(material, group)
        exprs.append(
            pl.when(pl.col(material).is_null() | pl.col(regime).is_null())
            .then(None)
            .otherwise(rank * pl.col(regime).cast(pl.Float64))
            .cast(pl.Float64)
            .alias(name)
        )
    return panel.with_columns(exprs) if exprs else panel


def _restrict_columns(panel: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    """Keep keys, labels and exactly ``feature_cols`` — drop interaction materials."""
    keys = set(KEY_COLS)
    label_prefixes = ("y_", "raw_label", "fwd_ret_", "bench_ret_")
    keep = set(feature_cols)
    return panel.select(
        [c for c in panel.columns if c in keys or c.startswith(label_prefixes) or c in keep]
    )


def null_ratios(panel: pl.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    """Per-feature share of NULL rows — recorded in the manifest, not enforced.

    Coverage is a property of the marts, and several FS1/FS2 columns are legitimately
    sparse (``hc_*`` and ``own_major_stake_chg`` start when their source does).
    Recording it is what lets a later run notice a *change*.
    """
    if panel.height == 0:
        return {}
    counts = panel.select([pl.col(c).null_count().alias(c) for c in feature_cols]).row(0)
    return {c: round(n / panel.height, 6) for c, n in zip(feature_cols, counts, strict=True)}


def build_dataset(
    spec: ModelSpec | None = None,
    config: LakeConfig | None = None,
    *,
    horizon: int | None = None,
    write: bool = True,
    created_at: str | None = None,
    strict_universe: bool = True,
) -> BuildResult:
    """Build one panel + folds + preprocessed slices for a single horizon.

    ``horizon`` defaults to the spec's primary. ``write=False`` keeps everything
    in memory (tests, dry runs). ``strict_universe=False`` skips the D-5 filter
    check — only for a synthetic lake whose universe mart was not built from the
    real SQL.
    """
    spec = spec or ModelSpec()
    if config is None:
        raise ValueError("build_dataset() requires an explicit config (LakeConfig)")
    h = spec.primary_horizon if horizon is None else horizon
    if h not in spec.horizons:
        raise ValueError(f"horizon {h} not in {spec.horizons}")

    columns, materials = panel_feature_columns(spec, h)
    marts = sorted(required_mart_columns([*columns, *materials]))

    con = connect(config)
    raw_tables = [PRICE_TABLE]
    if spec.label.index_bench:
        # R5: the index benchmark reads the raw observation table directly —
        # ``common_feature_daily_fact`` has no KOSDAQ close (labels._index_cte).
        raw_tables.append(INDEX_TABLE)
    register_views(con, config, tables=raw_tables)
    contracts = register_read_only(con, config, [UNIVERSE_VIEW, *marts])
    if strict_universe:
        verify_universe_contract(config, spec)
    con.execute(f"CREATE OR REPLACE VIEW {LABEL_VIEW} AS {labels_mod.build_label_sql(spec.label)}")

    panel = pl.from_arrow(con.execute(build_panel_sql(spec, h)).arrow())
    interactions = [i for i in fx.INTERACTIONS if i[0] in columns]
    panel = add_interactions(panel, interactions)
    panel = _restrict_columns(panel, columns)

    if panel.height and str(panel["trade_date"].max()) >= HOLDOUT_START:
        # The holdout guard, at the earliest point it can fire (`06` §8).
        raise AssertionError(
            f"panel reaches {panel['trade_date'].max()}, at or past the holdout boundary "
            f"{HOLDOUT_START}"
        )

    dates = panel.get_column("trade_date").unique().sort().to_list()
    folds = walk_forward_splits(dates, **spec.fold_kwargs(h))

    dataset_dir = config.dataset_dir(spec.model_id) / dataset_key(spec, h)
    manifest_path = dataset_dir / "dataset_manifest.json"
    cfg_pp = pp.PreprocessConfig(profile=spec.preprocess_profile)
    layout, feature_cols = _write_preprocessed(dataset_dir, panel, folds, cfg_pp, write=write)

    if write:
        _write_artifacts(dataset_dir, panel, folds)
        build_manifest(
            model_id=spec.model_id,
            config=config,
            feature_groups=[_MART_ALIAS[m] for m in marts],
            label_spec={
                "horizons": list(spec.label.horizons),
                "kind": spec.label.kind,
                "bench": spec.label.bench,
                "outputs": list(spec.label.outputs),
                "primary_horizon": h,
            },
            universe_filter={
                "warmup_window": spec.universe.warmup_window,
                "warmup_min_valid": spec.universe.warmup_min_valid,
                "min_liquidity_krw": spec.universe.min_liquidity_krw,
                "label_horizon": spec.universe.label_horizon,
            },
            period={"start": spec.period_start, "end": spec.period_end},
            row_count=panel.height,
            mart_root=str(mart_root(config)),
            created_at=created_at,
            extra={
                "feature_set": spec.feature_set,
                "flow_variant": spec.flow_variant,
                "preprocess_profile": spec.preprocess_profile,
                "horizon": h,
                "n_folds": len([f for f in folds if f.role == "fold"]),
                "n_dates": len(dates),
                "feature_columns": columns,
                "interaction_materials": materials,
                "interactions": [list(i) for i in interactions],
                "design_columns": feature_cols,
                "std_layout": layout,
                "mart_contracts": contracts,
                "null_ratios": null_ratios(panel, columns),
            },
        ).write(manifest_path)

    return BuildResult(
        dataset_dir=dataset_dir,
        horizon=h,
        panel_rows=panel.height,
        n_dates=len(dates),
        n_folds=len([f for f in folds if f.role == "fold"]),
        feature_cols=feature_cols,
        mart_contracts=contracts,
        manifest_path=manifest_path,
    )


def _write_preprocessed(
    dataset_dir: Path,
    panel: pl.DataFrame,
    folds: list[Fold],
    cfg: pp.PreprocessConfig,
    *,
    write: bool = True,
) -> tuple[str, list[str]]:
    """Write the design matrix under ``feat_panel_std/``; return (layout, columns).

    Two layouts, because two kinds of preprocessing:

    ``single`` — a stateless profile (``rank``). The transform depends only on a
    row's own ``(trade_date, market)`` cross-section, and a fold slice is whole
    dates, so every fold would transform a row to the same value. One
    transformed panel is written and the fold table says which dates belong to
    which fold. This is what keeps the design matrix at 1x the panel instead of
    3x: with five expanding folds the train slices alone re-materialize the
    early years five times over. Measured on E0's FS0 panel (5.2M rows): 4.15GB
    of per-fold copies against ~1.4GB for one, and the difference is worse for
    FS2 — 136 design columns instead of 80.

    ``per_fold`` — ``linear``/``tree``, whose winsorize bounds are fit on each
    fold's train range. Those transforms genuinely differ per fold, so each
    slice is written as its own part, one at a time. That is still a memory fix
    over concatenating them: peak is the panel plus one slice.
    """
    stateless = pp.is_stateless(cfg)
    std_dir = dataset_dir / STD_DIR
    if write:
        if std_dir.exists():
            shutil.rmtree(std_dir)
        # a dataset built before the layout change has a single 4GB file of the
        # same name; leaving it would double the directory's size for nothing.
        legacy = dataset_dir / f"{STD_DIR}.parquet"
        legacy.unlink(missing_ok=True)
        std_dir.mkdir(parents=True, exist_ok=True)

    if stateless:
        fitted = pp.fit(panel, cfg)
        out = fitted.transform(panel)
        if write:
            out.write_parquet(std_dir / "part-000000.parquet")
        return "single", fitted.feature_cols

    feature_cols: list[str] = []
    for fold in folds:
        train = panel.filter(
            (pl.col("trade_date") >= fold.train_start) & (pl.col("trade_date") <= fold.train_end)
        )
        if train.height == 0:
            continue
        fitted = pp.fit(train, cfg)
        feature_cols = fitted.feature_cols
        for role, start, end in (
            ("train", fold.train_start, fold.train_end),
            ("valid", fold.valid_start, fold.valid_end),
        ):
            sl = panel.filter((pl.col("trade_date") >= start) & (pl.col("trade_date") <= end))
            if sl.height == 0:
                continue
            part = fitted.transform(sl).with_columns(
                pl.lit(fold.fold_id).alias("fold_id"),
                pl.lit(role).alias("fold_role"),
            )
            if write:
                part.write_parquet(std_dir / f"fold{fold.fold_id}_{role}.parquet")
            del part
    return "per_fold", feature_cols


def _write_artifacts(
    dataset_dir: Path,
    panel: pl.DataFrame,
    folds: list[Fold],
) -> None:
    """Write panel / labels / folds as parquet (the design matrix has its own writer)."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(dataset_dir / "feat_panel.parquet")
    label_cols = [c for c in panel.columns if c.startswith(("y_", "raw_label", "fwd_ret_"))]
    panel.select([*KEY_COLS, *label_cols]).write_parquet(dataset_dir / "label_daily.parquet")
    pl.DataFrame([f.as_record() for f in folds]).write_parquet(dataset_dir / "split_folds.parquet")
