"""On-demand compute orchestrator: raw lake -> derived marts -> gates (§3.4, OQ1).

Single entrypoint a human runs to refresh + recompute the derived layer from the
raw lake, replacing the sj2 compute Cronicle events. Steps (each gated):

    freshness -> marts (normalize + build-daily) -> coverage/readiness -> features

The raw mirror + parquet export run in the wrapping shell (db sync-remote +
raw-parquet-export-all.sh); this module requires that export wrapper's
``_manifests/_SUCCESS.json`` completion marker for ``--snapshot-date``/``--source``
before it will read the lake (bypass with ``--allow-incomplete-lake``; see
docs/dev/20260730_refactor_dump/00_dual_route_raw_export_plan.md §3.7). Gate
failures print a human-readable summary to stderr and exit non-zero (no notifier —
interactive run, OQ1). Nothing is written back to Postgres.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from collector.kr.shared import now_kst

from modeler.etl.config import (
    CONFIG_TABLES,
    DEFAULT_SOURCE,
    RAW_TABLES,
    DataRoot,
    EngineOptions,
    LakeConfig,
)
from modeler.etl.lake import (
    _common_feature_calendars,
    connect,
    register_derived_marts,
    register_views,
)
from modeler.etl.marts import reports

_SMF_RAW_INPUTS = [
    "daily_ohlcv",
    "dart_financial_statement_raw",
    "dart_share_count_raw",
    "dart_shareholder_return_raw",
    "dart_xbrl_fact_raw",
    "dart_corp_master",
]
_CF_RAW_INPUTS = ["common_feature_observation_raw"]

_STEPS = ("freshness", "marts", "reports", "features")


def _eprint(*args: object) -> None:
    print(*args, file=sys.stderr)


def _check_lake_complete(cfg: LakeConfig, *, allow_incomplete_lake: bool) -> str | None:
    """Verify the export wrapper's completion marker before trusting the raw lake.

    ``bin/raw-parquet-export-all.sh`` writes ``_manifests/_SUCCESS.json`` only after
    every configured table is confirmed complete (§3.7 of the dual-route plan) — a
    half-finished export (crash, killed job, still running) leaves no such marker.
    Returns an error message if the gate fails, or ``None`` if it passes.
    """
    success_path = cfg.raw_root / "_manifests" / "_SUCCESS.json"
    if not success_path.exists():
        return f"raw lake capture not marked complete: {success_path} missing"

    with open(success_path, encoding="utf-8") as f:
        success = json.load(f)
    expected_tables = set(RAW_TABLES) | set(CONFIG_TABLES)
    actual_tables = set(success.get("tables", {}))
    if actual_tables != expected_tables:
        missing = sorted(expected_tables - actual_tables)
        extra = sorted(actual_tables - expected_tables)
        return f"{success_path} table set mismatch (missing={missing}, extra={extra})"

    return None


def _register_series_if_present(con, cfg: LakeConfig) -> None:
    """Register common_feature_series opportunistically (decision 7 fallback)."""
    from modeler.etl.lake import _glob_has_files

    glob = cfg.table_glob("common_feature_series")
    if _glob_has_files(con, glob):
        register_views(con, cfg, tables=["common_feature_series"])


def run(
    *,
    snapshot_date: str | None,
    from_step: str = "freshness",
    end: date | None = None,
    required_coverage_ratio: float = 1.0,
    threads: int = 4,
    memory_limit: str = "4GB",
    with_features: bool = False,
    source: str | None = None,
    allow_incomplete_lake: bool = False,
) -> int:
    """Run the compute pipeline from ``from_step``. Returns a process exit code."""
    if from_step not in _STEPS:
        _eprint(f"unknown --from-step {from_step!r}; expected one of {_STEPS}")
        return 2
    start_idx = _STEPS.index(from_step)

    if snapshot_date is None:
        _eprint("--snapshot-date is required")
        return 2

    cfg = LakeConfig(
        root=DataRoot.resolve(market="kr"),
        snapshot_date=snapshot_date,
        source=source or DEFAULT_SOURCE,
        engine=EngineOptions(threads=threads, memory_limit=memory_limit),
    )
    lake_error = _check_lake_complete(cfg, allow_incomplete_lake=allow_incomplete_lake)
    if lake_error:
        if allow_incomplete_lake:
            _eprint(f"WARNING: {lake_error}; proceeding anyway (--allow-incomplete-lake)")
        else:
            _eprint(f"{lake_error}; run the export wrapper first (or pass --allow-incomplete-lake)")
            return 1

    con = connect(cfg)
    register_views(con, cfg, tables=_SMF_RAW_INPUTS + _CF_RAW_INPUTS)
    _register_series_if_present(con, cfg)
    _, feature_dates = _common_feature_calendars(con)
    gate_end = end or now_kst().date()

    # 1) freshness gate (raw inputs fresh enough?)
    if start_idx <= _STEPS.index("freshness"):
        fr = reports.freshness_violations(con, end=gate_end)
        if not fr.ok:
            _eprint(f"FRESHNESS GATE FAILED ({len(fr.violations)} violations):")
            for v in fr.violations:
                _eprint(f"  - [{v.series_id or v.check}] {v.message}")
            return 1
        print(f"freshness OK: {fr.checked_series} series fresh as of {gate_end}")

    # 2) derived marts (normalize + build-daily)
    if start_idx <= _STEPS.index("marts"):
        created = register_derived_marts(con, cfg, persist=True, force=True)
        smf_n = con.execute("SELECT count(*) FROM stock_metric_fact").fetchone()[0]
        cfdf_n = con.execute("SELECT count(*) FROM common_feature_daily_fact").fetchone()[0]
        print(f"marts built + persisted: {created} (smf={smf_n}, cfdf={cfdf_n})")
    else:
        register_derived_marts(con, cfg)  # need the views for later steps

    # 3) coverage + readiness gates
    if start_idx <= _STEPS.index("reports"):
        rdy = reports.readiness_report(
            con, feature_dates=feature_dates, required_coverage_ratio=required_coverage_ratio
        )
        not_ready = [r for r in rdy if not r.ready]
        cov = reports.coverage_report(con, feature_dates=feature_dates)
        ready_n = len(rdy) - len(not_ready)
        print(f"coverage: {len(cov)} features; readiness: {ready_n}/{len(rdy)} ready")
        if not_ready:
            _eprint(f"READINESS GATE: {len(not_ready)} features not ready:")
            for r in not_ready[:20]:
                _eprint(
                    f"  - {r.feature_code} (coverage={r.coverage_ratio}): {', '.join(r.blockers)}"
                )
            return 1

    # 4) optional downstream feature/label marts
    if with_features and start_idx <= _STEPS.index("features"):
        _build_features(con, cfg)
        print("feature/label marts built")

    print("compute pipeline OK")
    return 0


def _build_industry_pit_if_present(con, cfg: LakeConfig, materialize) -> None:
    """Build ``dim_industry_pit_daily``, or skip a lake that predates its raw input.

    ``dart_corp_profile_history`` arrived with F-1.1 (2026-09-08), so every
    snapshot exported before then simply has no such table and every snapshot
    after it does. A missing raw input is the one condition under which this
    mart is skipped rather than built; the empty-history case is a *failure*
    (``compute_industry_observations`` raises), because a lake that carries the
    table but no row means the monthly collection is broken, not absent.
    """
    import duckdb

    from modeler.etl.lake import register_views

    try:
        register_views(con, cfg, tables=["dart_corp_profile_history"])
        materialize(con, cfg, force=False)
    except (duckdb.Error, FileNotFoundError) as exc:
        _eprint(
            "WARNING: dim_industry_pit_daily skipped — dart_corp_profile_history "
            f"is not in this lake ({exc.__class__.__name__}). Snapshots before "
            "2026-09-08 predate F-1.1."
        )


def _build_features(con, cfg: LakeConfig) -> None:
    """Build the A0 price/flow/label inputs plus legacy model feature views."""
    from modeler.etl.calendar import materialize_calendar
    from modeler.etl.features import common as cf_feat
    from modeler.etl.features import fin_pit
    from modeler.etl.features.flow import materialize_flow
    from modeler.etl.features.industry_pit import materialize_industry_pit
    from modeler.etl.features.macro_exposure import materialize_macro_exposure
    from modeler.etl.features.market_cap import materialize_market_cap
    from modeler.etl.features.price import materialize_price
    from modeler.etl.labels import materialize_label_scan
    from modeler.etl.quality import QUALITY_TABLE, materialize_price_quality
    from modeler.etl.stock_pit import PIT_TABLE, materialize_stock_pit
    from modeler.etl.universe import (
        UniverseFilter,
        build_universe_sql,
        materialize_named_universes,
    )

    # daily_market_cap / dart_filing_receipt_raw are not among the smf/cf raw
    # inputs registered in run(), and feat_market_cap needs both — the first for
    # the exchange's listed share count, the second for the 권리락 dates its mask
    # is built from.
    register_views(
        con,
        cfg,
        tables=["krx_security_flow_raw", "daily_market_cap", "dart_filing_receipt_raw"],
    )
    materialize_stock_pit(con, cfg, force=False)
    _build_industry_pit_if_present(con, cfg, materialize_industry_pit)
    # feat_market_cap carries its own mask (권리락 window), so it registers the
    # distortion view itself rather than relying on build order here.
    materialize_market_cap(con, cfg, force=False)
    materialize_price_quality(con, cfg, pit_view=PIT_TABLE, force=False)
    materialize_named_universes(con, cfg, force=False)
    materialize_price(con, cfg, quality_view=QUALITY_TABLE, force=False)
    # feat_macro_exposure reads common_feature_daily_fact, which the marts step
    # has already registered (and persisted) on this connection — so the mart
    # and the readiness gate above are looking at the same fact.
    materialize_macro_exposure(con, cfg, force=False)
    materialize_flow(
        con,
        cfg,
        price_view="daily_ohlcv",
        pit_view=PIT_TABLE,
        quality_view=QUALITY_TABLE,
        force=False,
    )
    materialize_label_scan(con, cfg, quality_view=QUALITY_TABLE, force=False)
    materialize_calendar(con, cfg)
    con.execute(
        f"CREATE OR REPLACE VIEW dim_universe_daily AS {build_universe_sql(UniverseFilter())}"
    )
    con.execute(f"CREATE OR REPLACE VIEW feat_fin_pit AS {fin_pit.build_fin_pit_sql()}")
    con.execute(f"CREATE OR REPLACE VIEW feat_common AS {cf_feat.build_common_sql()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument(
        "--source", default=None, help="lake source= partition (default: local_mydb)"
    )
    parser.add_argument("--from-step", default="freshness", choices=_STEPS)
    parser.add_argument("--end", default=None, help="freshness gate end date YYYY-MM-DD")
    parser.add_argument("--required-coverage-ratio", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--features", action="store_true", help="also build feat_*/labels")
    parser.add_argument(
        "--allow-incomplete-lake",
        action="store_true",
        help="bypass the _SUCCESS.json completeness gate (prints a warning)",
    )
    args = parser.parse_args(argv)

    end = date.fromisoformat(args.end) if args.end else None
    return run(
        snapshot_date=args.snapshot_date,
        from_step=args.from_step,
        end=end,
        required_coverage_ratio=args.required_coverage_ratio,
        threads=args.threads,
        memory_limit=args.memory_limit,
        with_features=args.features,
        source=args.source,
        allow_incomplete_lake=args.allow_incomplete_lake,
    )


if __name__ == "__main__":
    raise SystemExit(main())
