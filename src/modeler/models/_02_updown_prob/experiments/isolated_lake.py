"""isolated_lake — E5's five marts, built without touching snapshot 2026-08-23.

`02` §1.5 says E5 needs five marts rebuilt on the 08-23 snapshot::

    stock_metric_vintage_fact -> fin_quarterly_metric_vintage -> feat_fin_risk
    dim_peer_monthly -> feat_relation_stat

and `02` §5 says the A0-built marts on that snapshot must not be rebuilt,
because A0 stamps its contract hash onto every mart it writes and a rebuild
under another contract breaks the comparability every earlier stage rests on.
Those two rules overlap: on 08-23, ``stock_metric_vintage_fact`` and
``fin_quarterly_metric_vintage`` carry A0's ``236d0d35…`` — the same hash as
the four marts §5 names.

So the five are built into a **separate lake root** (``kr/derived/_e5``, its
own tiny ``raw/`` + ``derived/`` pair mirroring the shared layout — S4
2026-09-13), and everything else is symlinked back, relative, into the shared
lake. The shared snapshot is read and never written: the symlinks point at
mart *directories*, and the only directories this module creates for real are
the five it rebuilds.

Nothing here can rebuild a symlinked mart, because nothing here calls
``materialize`` on one — and ``build_dataset`` registers marts read-only by
design (its module docstring), so the E5 panel cannot either.

Verifying the result takes two different checks, because the repo did not
stand still. The relation pair is unchanged since F-HS-1 validated it, so its
``sql_hash`` must equal snapshot 2026-09-08's exactly — same SQL, same hash,
any snapshot. The three fin marts cannot: F-5.1/F-5.2 and F-5.4 landed on
2026-09-09 at 22:13 and 22:41, *after* 09-08's marts were written at 05:53,
and they add metrics and derived families. Their hash is expected to differ
from both 09-08 and the shared snapshot's v1.

What matters for those three is not the hash but whether F-5.0's XBRL fallback
rules are in, and `02` §1.5 states the test: F-5.0 moved five fin_risk families'
first valid session 2-4 years earlier. ``--verify`` measures that directly
against the shared v1 mart. The extra F-5.4 columns are inert here — E5 selects
the seven `02` §1.5 names and nothing else.

Usage::

    uv run python -m modeler.models._02_updown_prob.experiments.isolated_lake --link
    uv run python -m modeler.models._02_updown_prob.experiments.isolated_lake --build
    uv run python -m modeler.models._02_updown_prob.experiments.isolated_lake --verify
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from modeler.etl.config import DataRoot

SHARED_ROOT = DataRoot.resolve(market="kr")
ISOLATED_ROOT = DataRoot(base=SHARED_ROOT.derived / "_e5")

SNAPSHOT = "2026-08-23"
SOURCE = "sj2_remote"
PARTITION = f"snapshot_date={SNAPSHOT}/source={SOURCE}"

# Lakes the marts read from, as (DataRoot attribute, lake dir name) pairs.
# Symlinked whole — nothing writes to them. canonical_postgres was
# decommissioned (S4 2026-09-13, empty since before that) and dropped here
# along with it.
LINKED_LAKES: tuple[tuple[str, str], ...] = (
    ("raw", "raw_postgres"),
    ("derived", "metric"),
)


def _symlink_relative(link_path: Path, target: Path) -> None:
    """Symlink ``link_path`` -> ``target``, relative (S4: absolute links break on the next move)."""
    link_path.symlink_to(os.path.relpath(target, link_path.parent))

# The five `02` §1.5 names, in dependency order.
REBUILD: tuple[str, ...] = (
    "stock_metric_vintage_fact",
    "fin_quarterly_metric_vintage",
    "feat_fin_risk",
    "dim_peer_monthly",
    "feat_relation_stat",
)

# Symlinked marts the five read from, registered read-only before the build.
LINKED_DEPENDENCIES: tuple[str, ...] = ("dim_stock_pit_daily", "dim_price_quality_daily")

# The relation pair has not changed since F-HS-1 validated it, so a rebuild has
# to reproduce snapshot 2026-09-08's hash exactly. The three fin marts are not
# listed on purpose — see the module docstring.
EXPECTED_SQL_HASH: dict[str, str] = {
    "dim_peer_monthly": "269413e44268",
    "feat_relation_stat": "3705f359b0bc",
}

# `02` §1.5: F-5.0's fallback rules move these five families' first valid
# session 2-4 years earlier. The other two fin_risk columns E5 uses are
# untouched by F-5.0, which is why they are not here.
FIN_RISK_EARLIER_FAMILIES: tuple[str, ...] = (
    "fin_net_debt_to_mcap",
    "fin_ext_finance_to_assets",
    "fin_interest_coverage",
    "fin_lifecycle_stage",
    "fin_lifecycle_transition",
)
MIN_YEARS_EARLIER = 2.0


def feature_dir(root: DataRoot) -> Path:
    return root.derived / "feature" / f"snapshot_date={SNAPSHOT}" / f"source={SOURCE}"


def link() -> int:
    """Build the isolated root: symlinks everywhere except the five."""
    shared_marts = feature_dir(SHARED_ROOT)
    if not shared_marts.is_dir():
        raise SystemExit(f"shared snapshot missing: {shared_marts}")

    for base_name, lake in LINKED_LAKES:
        shared_base = getattr(SHARED_ROOT, base_name)
        isolated_base = getattr(ISOLATED_ROOT, base_name)
        target = shared_base / lake / f"snapshot_date={SNAPSHOT}" / f"source={SOURCE}"
        if not target.is_dir():
            print(f"  {lake}: absent on the shared snapshot, skipped")
            continue
        dest = isolated_base / lake / f"snapshot_date={SNAPSHOT}"
        dest.mkdir(parents=True, exist_ok=True)
        source_link = dest / f"source={SOURCE}"
        if not source_link.exists():
            _symlink_relative(source_link, target)
        print(f"  {lake}: linked")

    marts = feature_dir(ISOLATED_ROOT)
    marts.mkdir(parents=True, exist_ok=True)
    linked = 0
    for entry in sorted(shared_marts.iterdir()):
        dest = marts / entry.name
        if entry.name in REBUILD:
            # Left absent on purpose: this is what the rebuild writes.
            if dest.is_symlink():
                dest.unlink()
            continue
        if not dest.exists():
            _symlink_relative(dest, entry)
            linked += 1
    print(f"  feature: {linked} marts linked, {len(REBUILD)} left to build")
    print(f"\nisolated lake: {ISOLATED_ROOT.base}")
    return 0


def _built(name: str) -> Path | None:
    path = feature_dir(ISOLATED_ROOT) / name / "_cache_metadata.json"
    return path if path.is_file() else None


def _fin_risk_first_valid(glob: str, columns: tuple[str, ...]) -> dict[str, object]:
    import duckdb  # noqa: PLC0415

    con = duckdb.connect()
    con.execute("SET memory_limit='6GB'")
    select = ", ".join(
        f"min(CASE WHEN {c} IS NOT NULL THEN trade_date END) AS {c}" for c in columns
    )
    row = con.execute(f"SELECT {select} FROM read_parquet('{glob}')").fetchone()
    con.close()
    return dict(zip(columns, row, strict=True))


def verify() -> int:
    """Two checks: the relation pair by hash, the fin chain by what F-5.0 did."""
    ok = True
    for name in REBUILD:
        meta = _built(name)
        if meta is None:
            print(f"  {name}: NOT BUILT")
            ok = False
            continue
        stored = json.loads(meta.read_text())["sql_hash"][:12]
        expected = EXPECTED_SQL_HASH.get(name)
        if expected is None:
            print(f"  {name}: {stored} (hash not pinned — see the module docstring)")
            continue
        matches = stored == expected
        ok = ok and matches
        print(f"  {name}: {stored} (09-08: {expected}) {'ok' if matches else 'MISMATCH'}")

    shared = feature_dir(SHARED_ROOT) / "feat_fin_risk" / "**" / "*.parquet"
    built = feature_dir(ISOLATED_ROOT) / "feat_fin_risk" / "**" / "*.parquet"
    if not _built("feat_fin_risk"):
        return 1
    v1 = _fin_risk_first_valid(str(shared), FIN_RISK_EARLIER_FAMILIES)
    v2 = _fin_risk_first_valid(str(built), FIN_RISK_EARLIER_FAMILIES)
    print("\n  F-5.0 coverage gain (`02` §1.5 expects 2-4 years on five families):")
    for name in FIN_RISK_EARLIER_FAMILIES:
        before, after = v1[name], v2[name]
        if before is None or after is None:
            print(f"    {name}: missing on one side — MISMATCH")
            ok = False
            continue
        years = (before - after).days / 365.25
        if years < MIN_YEARS_EARLIER:
            ok = False
        mark = "ok" if years >= MIN_YEARS_EARLIER else "TOO SMALL"
        print(f"    {name}: {before} -> {after}  ({years:+.1f}y) {mark}")
    return 0 if ok else 1


def build(force: bool) -> int:
    """Materialize the five into the isolated root, in dependency order."""
    from collector.kr.shared import get_trading_days  # noqa: PLC0415
    from modeler.etl.config import LakeConfig  # noqa: PLC0415
    from modeler.etl.features.fin_risk import materialize_fin_risk  # noqa: PLC0415
    from modeler.etl.features.relation_stat import (  # noqa: PLC0415
        materialize_peer_monthly,
        materialize_relation_stat,
    )
    from modeler.etl.lake import connect, register_views  # noqa: PLC0415
    from modeler.etl.marts.financial_quarters import (  # noqa: PLC0415
        materialize_fin_quarterly_metric_vintage,
    )
    from modeler.etl.marts.metric_vintages import (  # noqa: PLC0415
        materialize_stock_metric_vintage_fact,
    )
    from modeler.models._02_updown_prob.build_dataset import (  # noqa: PLC0415
        register_read_only,
    )

    config = LakeConfig(
        root=ISOLATED_ROOT,
        snapshot_date=SNAPSHOT,
        source=SOURCE,
        # Not an A0 run, so the contract field stays empty — which is what the
        # three F-HS marts already carry on 08-23. What separates v1 from v2 is
        # the sql_hash, and `verify` checks that.
        analysis_config_hash=None,
    )
    con = connect(config)
    register_views(con, config)

    bounds = con.execute("SELECT min(trade_date), max(trade_date) FROM daily_ohlcv").fetchone()
    trading_days = list(get_trading_days(bounds[0], bounds[1]))
    print(f"  calendar: {len(trading_days)} sessions, {bounds[0]} .. {bounds[1]}")

    # feat_fin_risk reads two symlinked marts. ``register_read_only`` is
    # build_dataset's: it registers a mart as it lies instead of enforcing a
    # contract hash, which is required here because the shared snapshot's marts
    # carry A0's hash and this run carries none. It also cannot rebuild.
    register_read_only(con, config, list(LINKED_DEPENDENCIES))
    print(f"  linked inputs registered: {', '.join(LINKED_DEPENDENCIES)}")

    materialize_stock_metric_vintage_fact(con, config, trading_days=trading_days, force=force)
    print("  stock_metric_vintage_fact: done")
    materialize_fin_quarterly_metric_vintage(con, config, force=force)
    print("  fin_quarterly_metric_vintage: done")
    materialize_fin_risk(con, config, trading_days=trading_days, force=force)
    print("  feat_fin_risk: done")
    materialize_peer_monthly(con, config, force=force)
    print("  dim_peer_monthly: done")
    materialize_relation_stat(con, config, force=force)
    print("  feat_relation_stat: done")
    con.close()
    return verify()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--link", action="store_true", help="create the symlinked root")
    parser.add_argument("--build", action="store_true", help="materialize the five marts")
    parser.add_argument("--verify", action="store_true", help="compare sql_hash with 09-08")
    parser.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = parser.parse_args(argv)
    if not (args.link or args.build or args.verify):
        parser.error("one of --link / --build / --verify is required")
    if args.link:
        link()
    if args.build:
        return build(args.force)
    if args.verify:
        return verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
