"""§4.4.1 vintage distance probe — run the two metrics and apply the fixed verdict.

Reads the raw lake, compares every older annual capital-change vintage against
the newest one, and prints the numbers the pre-registered threshold table in
``04_specific_plan_B.md`` §4.4.1 is keyed on. The thresholds below were fixed
before any probe data existed; this script only reads them off.

    uv run python -m modeler.analysis.capital_change_vintage_probe \\
        --snapshot-date 2026-08-12

The probe needs at least two annual vintages per ticker in
``dart_capital_change_raw`` — see §4.1.1 of the implementation log for the
collection that produces them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb

from modeler.etl.config import REMOTE_SOURCE, DataRoot
from modeler.etl.lake import connect, register_views
from modeler.etl.snapshot import resolve_config
from modeler.etl.vintage_probe import (
    build_vintage_diff_summary_sql,
    build_vintage_row_diff_sql,
    measure_identity_pass_rate,
)

REQUIRED_TABLES = (
    "dart_corp_master",
    "dart_share_count_raw",
    "dart_capital_change_raw",
)
CALENDAR_SOURCE_TABLE = "daily_ohlcv"

# 04_specific_plan_B.md §4.4.1, fixed before the probe ran. Do not tune these
# against the numbers the probe returns.
RATE_ADOPT_LATEST = 0.01
RATE_ADOPT_LATEST_WITH_FLAG = 0.05
# Priority rule: a policy that never looks ahead but blanks the panel has not
# produced a usable candidate.
STRICT_COVERAGE_FLOOR = 0.5


def _rows(result: duckdb.DuckDBPyRelation | Any) -> list[dict[str, Any]]:
    columns = [d[0] for d in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def decide(
    *,
    distance_rows: list[dict[str, Any]],
    identity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the §4.4.1 threshold table to the measured numbers."""
    by_policy = {row["vintage_policy"]: row for row in identity_rows}
    latest_rate = by_policy.get("latest_vintage", {}).get("feature_available_rate")
    strict_rate = by_policy.get("strict_pit", {}).get("feature_available_rate")

    comparable = [row for row in distance_rows if row["compared_windows"]]
    if not comparable:
        return {
            "decision": "inconclusive",
            "rule": "no comparable windows — the probe needs a second annual vintage",
            "max_distance_years": None,
            "rate_at_max_distance": None,
            "latest_feature_available_rate": latest_rate,
            "strict_feature_available_rate": strict_rate,
        }

    furthest = max(comparable, key=lambda row: row["vintage_distance_years"])
    rate = furthest["feature_changing_rate"]
    verdict: dict[str, Any] = {
        "max_distance_years": furthest["vintage_distance_years"],
        "rate_at_max_distance": rate,
        "latest_feature_available_rate": latest_rate,
        "strict_feature_available_rate": strict_rate,
    }

    coverage_floor_breached = (
        latest_rate
        and strict_rate is not None
        and strict_rate <= STRICT_COVERAGE_FLOOR * latest_rate
    )
    if coverage_floor_breached:
        verdict["decision"] = "latest_vintage"
        verdict["rule"] = (
            "priority rule — strict_pit keeps at most half of latest_vintage's usable "
            "positions, so strict_pit stays a sensitivity analysis"
        )
        return verdict

    if rate < RATE_ADOPT_LATEST:
        verdict["decision"] = "latest_vintage"
        verdict["rule"] = f"feature-changing rate < {RATE_ADOPT_LATEST:.0%}"
    elif rate <= RATE_ADOPT_LATEST_WITH_FLAG:
        verdict["decision"] = "latest_vintage_with_quality_flag"
        verdict["rule"] = (
            f"{RATE_ADOPT_LATEST:.0%} <= rate <= {RATE_ADOPT_LATEST_WITH_FLAG:.0%} — adopt (a), "
            "carry vintage_lookahead_ratio and report the sensitivity"
        )
    else:
        verdict["decision"] = "strict_pit"
        verdict["rule"] = (
            f"rate > {RATE_ADOPT_LATEST_WITH_FLAG:.0%} — adopt (b) and collect the remaining "
            "annual vintages"
        )
    return verdict


def _format_report(payload: dict[str, Any]) -> str:
    lines = ["# capital-change vintage distance probe", ""]
    lines.append("## metric 1 — feature-changing disagreement by vintage distance")
    lines.append("")
    lines.append("| distance (y) | tickers | windows | changed | rate |")
    lines.append("|---|---|---|---|---|")
    for row in payload["distance_summary"]:
        rate = row["feature_changing_rate"]
        lines.append(
            f"| {row['vintage_distance_years']} | {row['tickers']} | "
            f"{row['compared_windows']} | {row['changed_windows']} | "
            f"{'-' if rate is None else f'{rate:.4f}'} |"
        )
    lines.append("")
    lines.append("## row-level colour — older events with no identical row in the newest")
    lines.append("")
    lines.append("| distance (y) | old events | absent from newest |")
    lines.append("|---|---|---|")
    for row in payload["row_summary"]:
        lines.append(
            f"| {row['vintage_distance_years']} | {row['old_events']} | "
            f"{row['old_events_absent_from_newest']} |"
        )
    lines.append("")
    lines.append("## metric 2 — identity pass rate by policy")
    lines.append("")
    lines.append("| policy | positions | with prior year | identity ok | feature | rate |")
    lines.append("|---|---|---|---|---|---|")
    for row in payload["identity_pass_rate"]:
        rate = row["feature_available_rate"]
        lines.append(
            f"| {row['vintage_policy']} | {row['positions']} | "
            f"{row['positions_with_prior_year']} | {row['identity_ok']} | "
            f"{row['feature_available']} | {'-' if rate is None else f'{rate:.4f}'} |"
        )
    lines.append("")
    verdict = payload["verdict"]
    lines.append("## verdict (§4.4.1 threshold table, fixed before the probe)")
    lines.append("")
    lines.append(f"- decision: **{verdict['decision']}**")
    lines.append(f"- rule: {verdict['rule']}")
    return "\n".join(lines) + "\n"


def run(
    con: duckdb.DuckDBPyConnection,
    *,
    trading_days: list[Any] | None = None,
) -> dict[str, Any]:
    """Run both metrics against already-registered views."""
    distance_summary = _rows(con.execute(build_vintage_diff_summary_sql()))
    row_summary = _rows(con.execute(build_vintage_row_diff_sql()))
    identity_rows = (
        measure_identity_pass_rate(con, trading_days=trading_days) if trading_days else []
    )
    payload = {
        "distance_summary": distance_summary,
        "row_summary": row_summary,
        "identity_pass_rate": identity_rows,
    }
    payload["verdict"] = decide(distance_rows=distance_summary, identity_rows=identity_rows)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None)
    parser.add_argument("--source", default=REMOTE_SOURCE)
    parser.add_argument("--data-lake-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-identity",
        action="store_true",
        help="metric 1 only — skips the daily_ohlcv session calendar metric 2 needs",
    )
    args = parser.parse_args(argv)

    root = DataRoot(args.data_lake_root) if args.data_lake_root else DataRoot.resolve(market="kr")
    output_dir = args.output_dir or (root.output / "scan" / "vintage_probe")
    tables = list(REQUIRED_TABLES) + ([] if args.skip_identity else [CALENDAR_SOURCE_TABLE])
    lake, _resolution = resolve_config(
        root, args.source, required_inputs=tuple(tables), snapshot_date=args.snapshot_date
    )
    con = connect(lake)
    register_views(con, lake, tables=tables)

    trading_days = None
    if not args.skip_identity:
        trading_days = [
            row[0]
            for row in con.execute(
                f"SELECT DISTINCT trade_date FROM {CALENDAR_SOURCE_TABLE} ORDER BY 1"
            ).fetchall()
        ]

    payload = run(con, trading_days=trading_days)
    report = _format_report(payload)
    print(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "vintage_probe.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    (output_dir / "vintage_probe.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
