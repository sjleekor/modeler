"""dim_regime_daily — market regime binaries on the KRX session grid.

Design: ``docs/dev/20260907_model_experiment/02_features_and_preprocessing.md``
§1.4, §3.1. The definitions are Phase C's (``docs/dev/20260829_macro_features/
01_design/03_stage1b_conditional_ic_phase_c.md`` §2.3) and are **not** restated
here — this module imports
:mod:`modeler.analysis.horizon_scan_phase_c_regimes` and renames its output.
Two implementations of "was the VIX high that day" would be two regimes, and the
model's FS2 comparison only means something if it conditions on the same
partition Phase C measured the interactions on.

Grain ``(trade_date)``: one row per KRX session, broadcast across every name at
panel-join time. Each regime contributes two columns:

  ``rg_<id>``    the binary the model reads (``z > 0``)
  ``rg_<id>_z``  the continuous score behind it, for diagnostics

The four primary regimes (``vix_up``, ``vix_high``, ``market_up``, ``liq_high``)
are FS2 inputs; the three exploratory ones ride along in the mart and are not
put into a feature set (`02` §1.4).

Warm-up is a NULL, not a zero (`02` §3.1). Two different things have to be
full, and only one of them is Phase C's business:

  * the *session* window — Phase C emits NULL until ``session_idx`` reaches the
    regime's window, and this lake's session grid starts in 2007;
  * the *observation* window — ``common_feature_daily_fact``'s daily series
    start 2014-06-16, and a 252-session median taken over three observations is
    not a 252-session median. Phase C's own CLI handles this by trimming to
    ``--sample-start 2015-06-16``; a mart cannot trim, so it NULLs instead.

Hence ``REGIME_VALID_FROM``: every regime column is NULL before it, which is
what the panel's ``*_isna`` flags then carry. Without it the mart would hand the
model a ``rg_vix_high = false`` for 2014-07 that came from a window holding a
handful of values — a regime that flipped because its window was short, which is
the artefact Phase C's module docstring exists to prevent.

Phase C's alternative cut (``alt_s_*``) and its ``session_idx`` are deliberately
dropped: the alternative is a diagnostic for the scan, and a second session
index in the panel would shadow the one ``dim_price_quality_daily`` supplies.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import duckdb

from modeler.analysis.horizon_scan_phase_c_regimes import (
    REGIME_SPECS,
    build_regime_series_sql,
)
from modeler.etl.config import DEFAULT_SOURCE, DataRoot, LakeConfig
from modeler.etl.mart import mart_root, materialize, register_mart_view

REGIME_TABLE = "dim_regime_daily"

# 252 sessions after the fact's daily series begin (2014-06-16) — the same
# constant Phase C's CLI passes as ``--sample-start``. Before this date the long
# windows are not full of observations, so every regime is NULL.
REGIME_VALID_FROM = "2015-06-16"

# FS2's four (`02` §1.4). Phase C's `role="primary"` — the ones whose
# interactions passed, hence the ones the model is allowed to condition on.
MODEL_REGIME_IDS: tuple[str, ...] = tuple(
    spec.regime_id for spec in REGIME_SPECS if spec.role == "primary"
)
EXPLORATORY_REGIME_IDS: tuple[str, ...] = tuple(
    spec.regime_id for spec in REGIME_SPECS if spec.role != "primary"
)

MODEL_REGIME_COLUMNS: tuple[str, ...] = tuple(f"rg_{rid}" for rid in MODEL_REGIME_IDS)
EXPLORATORY_REGIME_COLUMNS: tuple[str, ...] = tuple(f"rg_{rid}" for rid in EXPLORATORY_REGIME_IDS)
REGIME_Z_COLUMNS: tuple[str, ...] = tuple(f"rg_{spec.regime_id}_z" for spec in REGIME_SPECS)
FEATURE_COLUMNS: tuple[str, ...] = (
    *MODEL_REGIME_COLUMNS,
    *EXPLORATORY_REGIME_COLUMNS,
    *REGIME_Z_COLUMNS,
)


def build_regime_mart_sql(
    *,
    session_view: str = "label_scan",
    fact_view: str = "common_feature_daily_fact",
    valid_from: str = REGIME_VALID_FROM,
) -> str:
    """SQL producing ``dim_regime_daily`` from Phase C's regime series.

    A pure projection of :func:`build_regime_series_sql`: rename ``s_*`` ->
    ``rg_*`` and ``z_*`` -> ``rg_*_z``, drop the alternative cut and the session
    index, and NULL everything before ``valid_from``. Nothing is recomputed, so
    the mart cannot drift from the partition Phase C judged.

    Every session keeps its row — the grain is one row per KRX session whether
    or not the regime is known — so a panel LEFT JOIN sees an explicit unknown
    rather than a missing key.
    """
    inner = build_regime_series_sql(session_view=session_view, fact_view=fact_view)
    warm = f"trade_date >= DATE '{valid_from}'"
    columns = ",\n            ".join(
        f"CASE WHEN {warm} THEN s_{spec.regime_id} END AS rg_{spec.regime_id}"
        for spec in REGIME_SPECS
    )
    zs = ",\n            ".join(
        f"CASE WHEN {warm} THEN z_{spec.regime_id} END AS rg_{spec.regime_id}_z"
        for spec in REGIME_SPECS
    )
    return f"""
        SELECT
            trade_date,
            {columns},
            {zs}
        FROM ({inner}) AS phase_c
    """


def materialize_regime(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    session_view: str = "label_scan",
    fact_view: str = "common_feature_daily_fact",
    valid_from: str = REGIME_VALID_FROM,
    force: bool = False,
) -> str:
    """Build + register the ``dim_regime_daily`` mart view. Returns its name.

    Requires ``session_view`` (the KRX session grid — ``label_scan``) and
    ``fact_view`` (``common_feature_daily_fact``) registered on ``con``. The
    session grid must come from the scan's own date axis: the fact carries every
    weekday for 2014-2023, so counting 252 rows on the fact's axis would measure
    a different window before and after 2024.
    """
    materialize(
        con,
        config,
        REGIME_TABLE,
        build_regime_mart_sql(
            session_view=session_view, fact_view=fact_view, valid_from=valid_from
        ),
        force=force,
    )
    return register_mart_view(con, config, REGIME_TABLE)


def _a0_config_hash(config: LakeConfig) -> str | None:
    """The snapshot's A0 contract hash, from its mart manifest (None if absent)."""
    manifest = mart_root(config) / "_manifests" / "_SUCCESS.json"
    if not manifest.is_file():
        return None
    return json.loads(manifest.read_text(encoding="utf-8")).get("config_hash")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--source", default=None)
    parser.add_argument("--force", action="store_true", help="rebuild an existing mart")
    parser.add_argument(
        "--stamp-config-hash",
        choices=("a0", "none"),
        default="a0",
        help=(
            "which analysis-contract hash to write onto the mart. 'a0' (default) "
            "matches the snapshot's other marts, so the model reads them all with "
            "one LakeConfig; 'none' writes an unstamped mart."
        ),
    )
    args = parser.parse_args(argv)

    from modeler.etl.lake import connect, register_persisted_derived_mart

    if args.snapshot_date is None:
        parser.error("--snapshot-date is required")
    base = LakeConfig(
        root=DataRoot.resolve(market="kr"),
        snapshot_date=args.snapshot_date,
        source=args.source or DEFAULT_SOURCE,
    )
    # The inputs were written under the A0 contract, so reading them needs it.
    input_config = replace(base, analysis_config_hash=_a0_config_hash(base))
    output_config = input_config if args.stamp_config_hash == "a0" else base

    con = connect(input_config)
    register_mart_view(con, input_config, "label_scan")
    register_persisted_derived_mart(con, input_config, "common_feature_daily_fact")

    materialize_regime(con, output_config, force=args.force)
    view = register_mart_view(con, output_config, REGIME_TABLE)

    rows, first, last = con.execute(
        f"SELECT count(*), min(trade_date), max(trade_date) FROM {view}"
    ).fetchone()
    print(f"{REGIME_TABLE}: {rows:,} sessions, {first} .. {last}")
    for column in (*MODEL_REGIME_COLUMNS, *EXPLORATORY_REGIME_COLUMNS):
        n, n_true, first_valid = con.execute(
            f"SELECT count({column}), sum(CASE WHEN {column} THEN 1 ELSE 0 END), "
            f"min(CASE WHEN {column} IS NOT NULL THEN trade_date END) FROM {view}"
        ).fetchone()
        share = (n_true / n) if n else float("nan")
        print(f"  {column:<20} valid={n:,} share_true={share:.3f} from={first_valid}")
    print(f"written: {Path(mart_root(output_config)) / REGIME_TABLE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
