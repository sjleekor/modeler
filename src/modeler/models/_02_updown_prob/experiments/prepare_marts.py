"""prepare_marts — build every mart the model 02 opening reads, without the scan.

Plan: ``my/milestones/kr/modeling/plan/20260922_remaining_work.md`` §1-6/§1-7
("FS3 마트 다섯 생성·계약 확인 … 경량 wrapper를 권한다"),
``20260922_preopen_recheck.md`` §4.1.

Model 02 never builds a mart — ``build_dataset.register_read_only`` reads them
as they lie and raises ``FileNotFoundError`` if one is absent. So a fresh
snapshot needs somebody else to bake them, and until now that somebody was two
different pipelines with a six-hour scan bolted to one of them:

===========================  ==================================================
``dim_universe_daily``       model 01's ``build_dataset._prepare_marts``
``feat_fin_pit``             model 01's ``build_dataset._prepare_marts``
``feat_filing_activity``     horizon scan Phase B's ``register_phase_b_marts``
``feat_fin_scan_daily``      horizon scan Phase B (via the quarterly-vintage root)
the FS3 five (h60)           horizon scan Phase B
===========================  ==================================================

``compute_all --features`` does *not* cover the first two. It creates them as
in-memory ``VIEW``s on its own connection and lets them die with it — which is
why a snapshot can pass every ``compute_all`` gate and still be unopenable.
Measured on snapshot 2026-09-22: ``compute_all`` left 11 marts on disk and
4 of the adopted config's 6 were missing.

This module is the wrapper that plan asked for. It runs the two materializers
model 01 owns and then ``register_phase_b_marts`` — the same function, in the
same dependency order — and stops. **It does not run the scan**, and it does
not import the scan: Phase B's marts are outcome-blind (they are features, not
labels), but its scan is a statistical run and has no business here.

    # what is missing, and nothing else
    uv run python -m modeler.models._02_updown_prob.experiments.prepare_marts \\
        --snapshot-date 2026-09-22 --verify-only

    # build it
    uv run python -m modeler.models._02_updown_prob.experiments.prepare_marts \\
        --snapshot-date 2026-09-22

The cost is the Phase B mart build, measured at **34 minutes for 13 marts** on
snapshot 2026-09-08, most of it ``feat_fin_scan_daily`` — against the ~6h17m
the full scan adds on top.
"""

from __future__ import annotations

import argparse
import time

import duckdb

from modeler.analysis.horizon_scan_phase_b_run import register_phase_b_marts
from modeler.analysis.horizon_scan_run_spec import REQUIRED_A0_MARTS
from modeler.etl.config import DataRoot, LakeConfig
from modeler.etl.features.fin_pit import materialize_fin_pit
from modeler.etl.lake import connect, register_persisted_derived_mart, register_views
from modeler.etl.mart import is_materialized, mart_cache_metadata, register_mart_view
from modeler.etl.universe import materialize_universe
from modeler.models._02_updown_prob import build_dataset as bd
from modeler.models._02_updown_prob.experiments.holdout_run import load_adopted
from modeler.models._02_updown_prob.spec import SNAPSHOT_DATE, SOURCE

#: ``feat_fin_pit`` joins this, and it is a *persisted* derived mart rather than
#: a feature mart — ``compute_all --from-step marts`` writes it under
#: ``derived/metric/``. Binding the persisted bytes (not recomputing) is what
#: keeps this build agreeing with the coverage gate that already passed on them.
PERSISTED_DERIVED = ("stock_metric_fact", "common_feature_daily_fact")


def required_marts(horizon: int | None = None) -> list[str]:
    """The marts the adopted config's panel actually reads, asked of the code.

    Not a hand-kept list: ``panel_feature_columns`` -> ``required_mart_columns``
    is the same resolution ``build_dataset`` performs, so this cannot drift from
    what the opening will demand.
    """
    adopted = load_adopted()
    spec = adopted.spec()
    h = horizon or adopted.horizon
    columns, materials = bd.panel_feature_columns(spec, h)
    return [bd.UNIVERSE_VIEW, *sorted(bd.required_mart_columns([*columns, *materials]))]


def missing_marts(config: LakeConfig, horizon: int | None = None) -> list[str]:
    return [name for name in required_marts(horizon) if not is_materialized(config, name)]


def verify(config: LakeConfig, *, horizon: int | None = None) -> bool:
    """Report every required mart and whether the universe contract still holds.

    The contract check is the one that matters beyond presence: ``build_dataset``
    refuses a ``dim_universe_daily`` whose ``sql_hash`` is not the one
    ``spec.universe`` defines (D-5), and a mart built with a different filter
    looks identical on disk.
    """
    adopted = load_adopted()
    names = required_marts(horizon)
    print(
        f"채택 config {adopted.run_id} 가 읽는 마트 {len(names)}개"
        f" — snapshot {config.snapshot_date}"
    )
    missing = []
    for name in names:
        present = is_materialized(config, name)
        meta = mart_cache_metadata(config, name) if present else None
        rows = (meta or {}).get("row_count")
        detail = f"{rows:,}행" if isinstance(rows, int) else ""
        print(f"  {'OK  ' if present else '없음'} {name:<32} {detail}")
        if not present:
            missing.append(name)
    if missing:
        print(f"\n없는 마트 {len(missing)}개: {missing}")
        return False

    try:
        sql_hash = bd.verify_universe_contract(config, adopted.spec())
    except ValueError as exc:
        print(f"\n유니버스 계약 불일치 (D-5): {exc}")
        return False
    print(f"\n유니버스 계약 OK — sql_hash {sql_hash}")
    return True


def prepare(config: LakeConfig, *, force: bool = False) -> set[str]:
    """Materialize the model-02 marts, then Phase B's, in dependency order."""
    adopted = load_adopted()
    con: duckdb.DuckDBPyConnection = connect(config)

    raw = set(register_views(con, config))
    print(f"raw 뷰 {len(raw)}개 등록")

    for name in REQUIRED_A0_MARTS:
        register_mart_view(con, config, name)
    print(f"A0 마트 {len(REQUIRED_A0_MARTS)}개 등록")

    for name in PERSISTED_DERIVED:
        try:
            register_persisted_derived_mart(con, config, name)
        except (duckdb.Error, FileNotFoundError) as exc:
            # feat_fin_pit needs stock_metric_fact; say so here rather than
            # letting it fail three calls later with a bare missing-view error.
            print(
                f"  ! {name} 없음 ({type(exc).__name__})"
                " — `compute_all --from-step marts` 를 먼저 돌려라"
            )
    print(f"persisted derived 마트 {len(PERSISTED_DERIVED)}개 등록")

    # Order is load-bearing: feat_fin_pit's PIT interval join reads the universe
    # (model 01's `_prepare_marts` says so and builds them in this order).
    # The filter comes from the adopted spec so the D-5 contract hash is the one
    # `build_dataset.verify_universe_contract` will demand.
    started = time.time()
    materialize_universe(con, config, adopted.spec().universe, force=force)
    print(f"  {bd.UNIVERSE_VIEW} {time.time() - started:.0f}s")

    started = time.time()
    materialize_fin_pit(con, config, force=force)
    print(f"  feat_fin_pit {time.time() - started:.0f}s")

    started = time.time()
    built = register_phase_b_marts(con, config, force=force)
    print(f"  Phase B 마트 {len(built)}개 {time.time() - started:.0f}s: {sorted(built)}")
    con.close()
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=SNAPSHOT_DATE)
    parser.add_argument("--source", default=SOURCE)
    parser.add_argument("--horizon", type=int, default=None, help="기본: 채택 run 의 horizon")
    parser.add_argument("--verify-only", action="store_true", help="굽지 않고 무엇이 없는지만 본다")
    parser.add_argument("--force", action="store_true", help="이미 있는 마트도 다시 굽는다")
    args = parser.parse_args(argv)

    config = LakeConfig(
        root=DataRoot.resolve(market="kr"),
        snapshot_date=args.snapshot_date,
        source=args.source,
    )
    if args.verify_only:
        return 0 if verify(config, horizon=args.horizon) else 1

    before = missing_marts(config, args.horizon)
    print(f"시작 전 없는 마트 {len(before)}개: {before}\n")
    if not before and not args.force:
        print("다 있다. --force 없이는 아무것도 하지 않는다")
        return 0

    started = time.time()
    prepare(config, force=args.force)
    print(f"\n전체 {time.time() - started:.0f}s\n")
    return 0 if verify(config, horizon=args.horizon) else 1


if __name__ == "__main__":
    raise SystemExit(main())
