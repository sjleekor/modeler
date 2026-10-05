"""Pick the KR reference session of a briefing from one exported raw snapshot, without waiting.

The nightly path used to wait (until D 07:30) for the collection chains and then built the
marts for K, the KR session before D.  If K never completed there was no KR section at all.  Now
the wrapper exports whatever the database holds at that moment (one consistent PostgreSQL
snapshot) and this module decides **from that snapshot itself** which session it can serve:

  complete_K        K passes the completeness check -> K.
  fallback_K_prime  K fails, an earlier session K' passes -> K' (the marts and the native input
                    are cut to K', see ``kr_live_prepare``; the report says "N sessions before K").
  none              nothing within ``MAX_LOOKBACK_SESSIONS`` passes -> no prepared input is made
                    and the selector serves the newest older prepared one as ``stale``.

The check runs on the exported files, not on the live database, so no collection can slip in
between the check and the export.  It mirrors the collector's readiness gate
(``collector.kr.service.export_readiness``) for what a snapshot can show:

  * prices:  daily_ohlcv holds the session, and its ticker count is at least 97% of the previous
             session's (a half-written session loses far more than 3%).  Rows after the chosen
             session (K's partial rows when K' is chosen) are expected and are cut away
             downstream; ``cut_required`` in the evidence says so.
  * flows:   every active flow metric group (foreign holding, investor, shorting) reaches the
             session.  A group's date is the minimum over its active metrics of the newest row from
             the KRX or KIS source.

The collector's ingestion-run records (the DART chain) are not in the export.  The wrapper passes
the gate's verdict in as advisory evidence; a DART delay is not a look-ahead problem, it is a
difference between training and serving data (03_export_inputs 2.5), so it is recorded only.

CLI (used by ``deploy/prod/bin/kr-prepare.sh``)::

    python -m modeler.serving.kr_reference --snapshot-date SNAP --report-date D --k K \
        --calendar kr-calendar.json --stock-data-root ROOT --output evidence.json \
        [--gate-exit RC --gate-evidence file.json]

Exit codes: 0 complete_K or fallback_K_prime (stdout: the chosen session, then ``cut`` or
``nocut``), 32 none, 2 usage or input error.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = "kr-reference-selection.v1"
VERDICT_COMPLETE_K = "complete_K"
VERDICT_FALLBACK = "fallback_K_prime"
VERDICT_NONE = "none"
EXIT_NONE = 32

#: Same value as the collector gate (``DEFAULT_MIN_TICKER_RATIO``).
DEFAULT_MIN_TICKER_RATIO = 0.97
#: How far below K a fallback may reach.  Matches ``freshness.MAX_STALE_SESSIONS``: an older
#: session would be shown without a ranking anyway (Q5), so nothing is gained by building it.
MAX_LOOKBACK_SESSIONS = 5
#: Window the previous-session ticker count is read from (collector ``PRICE_COUNT_LOOKBACK_DAYS``).
PRICE_COUNT_LOOKBACK_DAYS = 30
FLOW_SOURCES = ("KRX", "KIS")
#: ``collector.kr.service.sync_krx_flows.FLOW_METRIC_GROUPS`` minus its discontinued metrics
#: (``short_selling_balance_quantity``).  Copied, not imported: the lint contract keeps modeler
#: out of ``collector.kr.service``.  If the collector adds a flow metric, add it here too.
FLOW_GROUPS: dict[str, tuple[str, ...]] = {
    "foreign_holding": ("foreign_holding_shares",),
    "investor": (
        "institution_net_buy_volume",
        "individual_net_buy_volume",
        "foreign_net_buy_volume",
    ),
    "shorting": ("short_selling_volume", "short_selling_value"),
}


@dataclass(frozen=True)
class SnapshotFacts:
    """What the exported snapshot shows, as plain values (no database handle)."""

    price_max: date | None
    price_counts: dict[date, int]
    flow_group_latest: dict[str, date | None]


def read_snapshot_facts(*, raw_root: Path, newest: date) -> SnapshotFacts:
    """Read price counts and flow dates from a raw snapshot's parquet files (read only)."""
    import duckdb

    def source(table: str) -> str:
        pattern = str(raw_root / table / "**" / "*.parquet").replace("'", "''")
        return f"read_parquet('{pattern}', hive_partitioning=false)"

    con = duckdb.connect()
    try:
        con.execute("SET threads = 2")
        price_max = con.execute(f"SELECT max(trade_date) FROM {source('daily_ohlcv')}").fetchone()[
            0
        ]
        lower = newest - timedelta(days=PRICE_COUNT_LOOKBACK_DAYS + MAX_LOOKBACK_SESSIONS * 4)
        counts = {
            row[0]: row[1]
            for row in con.execute(
                f"SELECT trade_date, count(*) FROM {source('daily_ohlcv')} "
                f"WHERE trade_date BETWEEN DATE '{lower}' AND DATE '{newest}' GROUP BY trade_date"
            ).fetchall()
        }
        sources = ", ".join(f"'{item}'" for item in FLOW_SOURCES)
        newest_by_metric = dict(
            con.execute(
                f"SELECT metric_code, max(trade_date) FROM {source('krx_security_flow_raw')} "
                f"WHERE source IN ({sources}) GROUP BY metric_code"
            ).fetchall()
        )
    except duckdb.IOException as exc:
        raise FileNotFoundError(f"KR raw snapshot tables cannot be read under {raw_root}") from exc
    finally:
        con.close()
    groups: dict[str, date | None] = {}
    for group, metrics in FLOW_GROUPS.items():
        found = [newest_by_metric.get(metric) for metric in metrics]
        present = [item for item in found if item is not None]
        # A metric with no row at all means the group cannot be judged complete.
        groups[group] = min(present) if present and len(present) == len(metrics) else None
    return SnapshotFacts(price_max=price_max, price_counts=counts, flow_group_latest=groups)


def assess_session(
    facts: SnapshotFacts,
    session: date,
    *,
    min_ticker_ratio: float = DEFAULT_MIN_TICKER_RATIO,
) -> dict[str, Any]:
    """The completeness check of one session, with every number the verdict rests on."""
    reasons: list[str] = []
    count = facts.price_counts.get(session)
    previous = max((day for day in facts.price_counts if day < session), default=None)
    previous_count = facts.price_counts.get(previous) if previous else None
    ratio = count / previous_count if count is not None and previous_count else None
    if facts.price_max is None or facts.price_max < session:
        reasons.append(f"daily_ohlcv newest date {facts.price_max} is older than {session}")
    if not count:
        reasons.append(f"no daily_ohlcv rows for {session}")
    elif previous_count is None:
        reasons.append("no previous session in the lookback; the ticker ratio cannot be judged")
    elif ratio is not None and ratio < min_ticker_ratio:
        reasons.append(
            f"ticker count {count} is {ratio:.4f} of the previous session {previous_count} "
            f"(< {min_ticker_ratio})"
        )
    flows: dict[str, dict[str, Any]] = {}
    for group, latest in sorted(facts.flow_group_latest.items()):
        ok = latest is not None and latest >= session
        flows[group] = {"latest": latest.isoformat() if latest else None, "reaches_session": ok}
        if not ok:
            reasons.append(f"flow group {group} newest date {latest} is older than {session}")
    return {
        "session": session.isoformat(),
        "complete": not reasons,
        "price": {
            "newest_trade_date": facts.price_max.isoformat() if facts.price_max else None,
            "rows_after_session": bool(facts.price_max and facts.price_max > session),
            "ticker_count": count,
            "previous_session": previous.isoformat() if previous else None,
            "previous_ticker_count": previous_count,
            "ticker_ratio": round(ratio, 6) if ratio is not None else None,
            "min_ticker_ratio": min_ticker_ratio,
        },
        "flows": flows,
        "reasons": reasons,
    }


def select_reference(
    facts: SnapshotFacts,
    *,
    k: date,
    sessions_before_k: list[date],
    min_ticker_ratio: float = DEFAULT_MIN_TICKER_RATIO,
) -> dict[str, Any]:
    """K when it is complete, else the newest complete earlier session, else none.

    ``sessions_before_k`` are the KR sessions below K, newest first.  Only the first
    ``MAX_LOOKBACK_SESSIONS`` are tried.
    """
    candidates = [assess_session(facts, k, min_ticker_ratio=min_ticker_ratio)]
    reference: date | None = k if candidates[0]["complete"] else None
    lag = 0
    if reference is None:
        for lag, session in enumerate(sessions_before_k[:MAX_LOOKBACK_SESSIONS], start=1):
            checked = assess_session(facts, session, min_ticker_ratio=min_ticker_ratio)
            candidates.append(checked)
            if checked["complete"]:
                reference = session
                break
    if reference == k:
        verdict = VERDICT_COMPLETE_K
    elif reference is not None:
        verdict = VERDICT_FALLBACK
    else:
        verdict, lag = VERDICT_NONE, None
    # The marts must end at the reference session.  A plain K needs no cut (the export holds
    # nothing later), which keeps the K path identical to the one before this change.
    cut_required = bool(reference and facts.price_max and facts.price_max > reference)
    return {
        "schema": SCHEMA,
        "verdict": verdict,
        "k": k.isoformat(),
        "reference_date": reference.isoformat() if reference else None,
        "lag_sessions": lag,
        "cut_required": cut_required,
        "max_lookback_sessions": MAX_LOOKBACK_SESSIONS,
        "candidates": candidates,
    }


def _gate_summary(exit_code: int | None, evidence_path: Path | None) -> dict[str, Any] | None:
    """The collector gate's verdict, kept as advisory evidence (never an input to the choice)."""
    if exit_code is None:
        return None
    verdicts = {0: "ready", 75: "not_yet", 1: "blocked"}
    summary: dict[str, Any] = {
        "exit_code": exit_code,
        "verdict": verdicts.get(exit_code, "error"),
        "advisory": True,
    }
    if evidence_path is not None and evidence_path.is_file():
        try:
            body = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary["evidence_error"] = "unreadable"
            return summary
        if isinstance(body, dict):
            summary["reasons"] = [str(item) for item in body.get("reasons", [])][:20]
            summary["dart_chain_ended_at"] = body.get("dart_chain_ended_at")
            summary["run_checks"] = [
                {
                    "run_type": item.get("run_type"),
                    "status": item.get("status"),
                    "run_status": item.get("run_status"),
                    "ended_at": item.get("ended_at"),
                }
                for item in body.get("checks", [])
                if isinstance(item, dict) and item.get("check") == "ingestion_run"
            ]
    return summary


def _sessions_before(calendar_path: Path, k: date) -> list[date]:
    from modeler.serving.calendars import SessionCalendar

    calendar = SessionCalendar.from_manifest(json.loads(calendar_path.read_text(encoding="utf-8")))
    if calendar.is_session(k) is not True:
        raise ValueError(f"K={k} is not a session of the KR calendar")
    found: list[date] = []
    cursor: date | None = k
    while len(found) < MAX_LOOKBACK_SESSIONS:
        cursor = calendar.previous_session(cursor)
        if cursor is None:
            break  # the calendar's coverage ends: do not guess older sessions
        found.append(cursor)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--snapshot-date", required=True)
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--k", required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--stock-data-root", type=Path)
    parser.add_argument("--min-ticker-ratio", type=float, default=DEFAULT_MIN_TICKER_RATIO)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gate-exit", type=int)
    parser.add_argument("--gate-evidence", type=Path)
    args = parser.parse_args(argv)
    try:
        from modeler.etl.config import REMOTE_SOURCE, DataRoot, LakeConfig

        k = date.fromisoformat(args.k)
        report_date = date.fromisoformat(args.report_date)
        if k >= report_date:
            raise ValueError("K must precede the report date")
        root = (
            DataRoot(base=args.stock_data_root / "kr")
            if args.stock_data_root
            else DataRoot.resolve(market="kr")
        )
        config = LakeConfig(root=root, snapshot_date=args.snapshot_date, source=REMOTE_SOURCE)
        facts = read_snapshot_facts(raw_root=config.raw_root, newest=k)
        evidence = select_reference(
            facts,
            k=k,
            sessions_before_k=_sessions_before(args.calendar, k),
            min_ticker_ratio=args.min_ticker_ratio,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"kr-reference: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    evidence.update(
        report_date=report_date.isoformat(),
        snapshot_date=args.snapshot_date,
        checked_at=datetime.now(UTC).isoformat(),
        gate=_gate_summary(args.gate_exit, args.gate_evidence),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(
        f"kr-reference: verdict={evidence['verdict']} K={args.k} "
        f"reference={evidence['reference_date']} lag_sessions={evidence['lag_sessions']}",
        file=sys.stderr,
    )
    for item in evidence["candidates"]:
        for reason in item["reasons"]:
            print(f"kr-reference:   {item['session']}: {reason}", file=sys.stderr)
    if evidence["verdict"] == VERDICT_NONE:
        return EXIT_NONE
    print(evidence["reference_date"])
    print("cut" if evidence["cut_required"] else "nocut")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
