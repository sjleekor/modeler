"""Compare two KR serving builds: feature marts, or prepared outputs plus scoring.

``marts``     two feature-mart roots, per mart: schema (names, types, order), row count,
              key duplicates, NULL-safe ``EXCEPT ALL`` in both directions on all columns,
              and, for float columns, NULL/NaN/inf position equality and a tolerance
              check with an absolute and a relative threshold (default exact).
``prepared``  two ``kr_prepare`` output directories: key set, columns, values and
              NULL/NaN/inf positions of ``feature_panel.parquet``; with ``--bundle`` also
              the model input matrix (bitwise), ``p_raw``, ranks, ties and the top 100.

The tolerance rule is ``|a-b| <= abs_tol + rel_tol * max(|a|, |b|)``; with both at 0 it is
exact equality (+0.0 and -0.0 count as equal, as in DuckDB). Exit code: 0 equal under the
given policy, 1 different, 2 usage or input error.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from modeler.etl.config import EngineOptions
from modeler.serving.build_profile import write_json_atomic

SCHEMA_VERSION = "kr-compare.v1"
DAILY_KEY = ("trade_date", "ticker", "market")
# Marts that are not (trade_date, ticker, market) daily. Grains are the winner
# partitions in the mart SQL (etl/marts/*.py), not what the data happens to satisfy.
KEY_OVERRIDES: dict[str, tuple[str, ...]] = {
    "stock_metric_fact": ("ticker", "metric_code", "bsns_year", "reprt_code"),
    "stock_metric_vintage_fact": (
        "ticker", "metric_code", "statement_period_end", "fs_basis", "rcept_no"),
    "fin_quarterly_metric_vintage": (
        "ticker", "metric_code", "fs_basis", "bsns_year", "quarter"),
    "common_feature_daily_fact": ("feature_date", "feature_code"),
    "dim_trading_calendar": ("market", "trade_date"),
    "dim_peer_monthly": ("month_end", "ticker", "market", "peer_ticker", "peer_market"),
}
FLOAT_TYPES = frozenset({"DOUBLE", "FLOAT"})
# Manifest fields that change on every run and say nothing about the inputs.
VOLATILE_MANIFEST_FIELDS = frozenset({"generated_at", "feature_build_completed_at"})
DEFAULT_MAX_TEMP_SIZE = "30GB"
SAMPLE_ROWS = 5


class UsageError(Exception):
    """Bad arguments or unreadable inputs; maps to exit code 2."""


@dataclass(frozen=True)
class Tolerance:
    abs_tol: float = 0.0
    rel_tol: float = 0.0

    def __post_init__(self) -> None:
        for value in (self.abs_tol, self.rel_tol):
            if not math.isfinite(value) or value < 0:
                raise UsageError("tolerances must be finite and non-negative")


def _ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _connect(engine: EngineOptions) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for key, value in engine.as_pragmas().items():
        con.execute(f"SET {key} = {_literal(value)}")
    return con


def _parquet(path_or_glob: str) -> str:
    return f"read_parquet({_literal(path_or_glob)}, hive_partitioning=false)"


def _describe(con, relation: str) -> list[tuple[str, str]]:
    rows = con.execute(f"DESCRIBE SELECT * FROM {relation}").fetchall()
    return [(name, kind) for name, kind, *_ in rows]


def _schema_diff(a: list[tuple[str, str]], b: list[tuple[str, str]]) -> dict:
    names_a, names_b = [n for n, _ in a], [n for n, _ in b]
    types_a, types_b = dict(a), dict(b)
    common = [n for n in names_a if n in types_b]
    return {
        "only_in_a": [n for n in names_a if n not in types_b],
        "only_in_b": [n for n in names_b if n not in types_a],
        "type_mismatch": [
            {"column": n, "a": types_a[n], "b": types_b[n]}
            for n in common if types_a[n] != types_b[n]
        ],
        "order_differs": (
            [n for n in names_a if n in types_b] != [n for n in names_b if n in types_a]),
    }


def _duplicates(con, view: str, key: tuple[str, ...]) -> dict:
    cols = ", ".join(_ident(c) for c in key)
    keys, extra = con.execute(
        f"SELECT count(*), coalesce(sum(n - 1), 0) FROM "
        f"(SELECT count(*) AS n FROM {view} GROUP BY {cols} HAVING count(*) > 1)"
    ).fetchone()
    return {"duplicate_keys": int(keys), "excess_rows": int(extra)}


def _count_except(con, cols: str, left: str, right: str) -> int:
    return int(con.execute(
        f"SELECT count(*) FROM (SELECT {cols} FROM {left} EXCEPT ALL SELECT {cols} FROM {right})"
    ).fetchone()[0])


def _sample_except(con, cols: str, left: str, right: str, limit: int) -> list[dict]:
    cursor = con.execute(
        f"SELECT * FROM (SELECT {cols} FROM {left} EXCEPT ALL SELECT {cols} FROM {right}) "
        f"LIMIT {int(limit)}")
    names = [d[0] for d in cursor.description]
    return [dict(zip(names, map(str, row), strict=True)) for row in cursor.fetchall()]


def _keyed_differences(
    con, common: list[tuple[str, str]], key: tuple[str, ...], tol: Tolerance
) -> dict:
    """Join on the key and count, per column, what differs and by how much."""
    cols = ", ".join(_ident(n) for n, _ in common)
    on = " AND ".join(f"a.{_ident(c)} IS NOT DISTINCT FROM b.{_ident(c)}" for c in key)
    matched = '(a."__pa" AND b."__pb")'
    abs_tol, rel_tol = f"CAST({tol.abs_tol!r} AS DOUBLE)", f"CAST({tol.rel_tol!r} AS DOUBLE)"
    parts: list[tuple[str | None, str, str]] = [
        (None, "only_in_a", 'count_if(a."__pa" AND b."__pb" IS NULL)'),
        (None, "only_in_b", 'count_if(b."__pb" AND a."__pa" IS NULL)'),
    ]
    for name, kind in common:
        if name in key:
            continue
        x, y = f"a.{_ident(name)}", f"b.{_ident(name)}"
        if kind not in FLOAT_TYPES:
            parts.append(
                (name, "value_mismatch", f"count_if({matched} AND {x} IS DISTINCT FROM {y})"))
            continue
        both = f"isfinite({x}) AND isfinite({y})"
        parts += [
            (name, "null_mismatch", f"count_if({matched} AND (({x} IS NULL) <> ({y} IS NULL)))"),
            (name, "nan_mismatch", f"count_if({matched} AND (coalesce(isnan({x}), false) "
                                   f"<> coalesce(isnan({y}), false)))"),
            (name, "inf_mismatch", f"count_if({matched} AND "
                                   f"(coalesce(isinf({x}), false) <> coalesce(isinf({y}), false) "
                                   f"OR (isinf({x}) AND isinf({y}) AND sign({x}) <> sign({y}))))"),
            (name, "over_tolerance", f"count_if({matched} AND {both} AND abs({x} - {y}) > "
                                     f"{abs_tol} + {rel_tol} * greatest(abs({x}), abs({y})))"),
            (name, "max_abs_diff", f"max(CASE WHEN {matched} AND {both} THEN abs({x} - {y}) END)"),
            (name, "max_rel_diff", f"max(CASE WHEN {matched} AND {both} THEN abs({x} - {y}) / "
                                   f"nullif(greatest(abs({x}), abs({y})), 0) END)"),
        ]
    sql = (
        f'SELECT {", ".join(expr for _, _, expr in parts)} '
        f'FROM (SELECT {cols}, TRUE AS "__pa" FROM __a) a '
        f'FULL OUTER JOIN (SELECT {cols}, TRUE AS "__pb" FROM __b) b ON {on}'
    )
    values = con.execute(sql).fetchone()
    result: dict = {"only_in_a": 0, "only_in_b": 0, "columns": {}}
    for (name, metric, _), value in zip(parts, values, strict=True):
        if value is not None:
            value = float(value) if metric.startswith("max_") else int(value)
        if name is None:
            result[metric] = value
        else:
            result["columns"].setdefault(name, {})[metric] = value
    flagged = {
        name: stats for name, stats in result["columns"].items()
        if any(v for m, v in stats.items() if not m.startswith("max_"))
    }
    result["columns"] = flagged
    result["clean"] = (
        result["only_in_a"] == 0 and result["only_in_b"] == 0 and not flagged
    )
    return result


def compare_tables(
    con, a_relation: str, b_relation: str, *, key: tuple[str, ...] | None,
    tol: Tolerance = Tolerance(), sample: int = SAMPLE_ROWS,
) -> dict:
    """Compare two relations; ``key`` None infers the daily key or reports it missing."""
    con.execute(f"CREATE OR REPLACE TEMP VIEW __a AS SELECT * FROM {a_relation}")
    con.execute(f"CREATE OR REPLACE TEMP VIEW __b AS SELECT * FROM {b_relation}")
    schema_a, schema_b = _describe(con, "__a"), _describe(con, "__b")
    names_a = {n for n, _ in schema_a}
    if key is None:
        key = DAILY_KEY if set(DAILY_KEY) <= names_a else None
    report: dict = {"schema": _schema_diff(schema_a, schema_b), "key": list(key) if key else None}
    schema = report["schema"]
    report["schema_equal"] = schema_a == schema_b
    report["row_count"] = {
        side: int(con.execute(f"SELECT count(*) FROM {view}").fetchone()[0])
        for side, view in (("a", "__a"), ("b", "__b"))
    }
    common = [(n, k) for n, k in schema_a if n in dict(schema_b)]
    problems: list[str] = []
    if key is None or not set(key) <= {n for n, _ in common}:
        report["key"] = list(key) if key else None
        problems.append("key columns are missing from one side; pass a key for this mart")
        report.update(problems=problems, equal=False)
        return report
    report["duplicates"] = {
        "a": _duplicates(con, "__a", key), "b": _duplicates(con, "__b", key)}
    cols = ", ".join(_ident(n) for n, _ in common)
    try:
        report["except_all"] = {
            "a_minus_b": _count_except(con, cols, "__a", "__b"),
            "b_minus_a": _count_except(con, cols, "__b", "__a"),
        }
    except duckdb.Error as exc:
        report["except_all"] = {"error": str(exc)}
        problems.append("EXCEPT ALL could not run (incompatible column types)")
    diffs = report["except_all"]
    exact = diffs.get("a_minus_b") == 0 and diffs.get("b_minus_a") == 0
    if not exact and "error" not in report["except_all"]:
        report["samples"] = {
            "a_minus_b": _sample_except(con, cols, "__a", "__b", sample),
            "b_minus_a": _sample_except(con, cols, "__b", "__a", sample),
        }
    keyed_clean = False
    dup_free = all(d["duplicate_keys"] == 0 for d in report["duplicates"].values())
    if not exact and dup_free and not schema["type_mismatch"]:
        report["keyed"] = _keyed_differences(con, common, key, tol)
        keyed_clean = report["keyed"]["clean"]
    elif not exact and not dup_free:
        problems.append("duplicate keys: per-column tolerance check skipped")
    report["tolerance"] = {"abs_tol": tol.abs_tol, "rel_tol": tol.rel_tol}
    same_dups = report["duplicates"]["a"] == report["duplicates"]["b"]
    report["problems"] = problems
    report["equal"] = bool(
        report["schema_equal"]
        and report["row_count"]["a"] == report["row_count"]["b"]
        and same_dups
        and (exact or keyed_clean)
    )
    return report


def _mart_relation(roots: list[Path], name: str) -> str | None:
    for root in roots:
        directory = root / name
        if directory.is_dir() and any(directory.rglob("*.parquet")):
            return _parquet(str(directory / "**" / "*.parquet"))
    return None


def _discover_marts(roots: list[Path]) -> set[str]:
    found: set[str] = set()
    for root in roots:
        if not root.is_dir():
            raise UsageError(f"mart root is not a directory: {root}")
        found |= {
            p.name for p in root.iterdir()
            if p.is_dir() and not p.name.startswith("_") and any(p.rglob("*.parquet"))
        }
    return found


def compare_marts(
    a_roots: list[Path], b_roots: list[Path], *, marts: list[str] | None,
    key_overrides: dict[str, tuple[str, ...]] | None = None, tol: Tolerance = Tolerance(),
    engine: EngineOptions, sample: int = SAMPLE_ROWS, log=None,
) -> dict:
    keys = {**KEY_OVERRIDES, **(key_overrides or {})}
    names = sorted(marts) if marts else sorted(_discover_marts(a_roots) | _discover_marts(b_roots))
    if not names:
        raise UsageError("no marts found under the given roots")
    con = _connect(engine)
    results: dict[str, dict] = {}
    try:
        for name in names:
            a_rel, b_rel = _mart_relation(a_roots, name), _mart_relation(b_roots, name)
            if a_rel is None or b_rel is None:
                results[name] = {
                    "equal": False, "problems": [
                        "missing in a" if a_rel is None else "missing in b"]}
            else:
                results[name] = compare_tables(
                    con, a_rel, b_rel, key=keys.get(name), tol=tol, sample=sample)
            if log:
                log(_summary_line(name, results[name]))
    finally:
        con.close()
    return {
        "schema_version": SCHEMA_VERSION, "mode": "marts",
        "generated_at": datetime.now(UTC).isoformat(),
        "a_roots": [str(p) for p in a_roots], "b_roots": [str(p) for p in b_roots],
        "tolerance": {"abs_tol": tol.abs_tol, "rel_tol": tol.rel_tol},
        "equal": all(r["equal"] for r in results.values()), "marts": results,
    }


def _flatten(value, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict):
        out: dict[str, object] = {}
        for key, item in value.items():
            if key in VOLATILE_MANIFEST_FIELDS:
                continue
            out.update(_flatten(item, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): value}


def _manifest_differences(a: dict, b: dict) -> list[dict]:
    flat_a, flat_b = _flatten(a), _flatten(b)
    return [
        {"path": path, "a": flat_a.get(path), "b": flat_b.get(path)}
        for path in sorted(flat_a.keys() | flat_b.keys())
        if flat_a.get(path) != flat_b.get(path)
    ]


def _prepared_files(directory: Path) -> tuple[Path, dict]:
    panel, manifest = directory / "feature_panel.parquet", directory / "prepare_manifest.json"
    for path in (panel, manifest):
        if not path.is_file():
            raise UsageError(f"prepared output is missing {path.name}: {directory}")
    return panel, json.loads(manifest.read_text(encoding="utf-8"))


def _ranks(symbols: list[tuple[str, str]], scores) -> dict[tuple[str, str], int]:
    # Same order as kr_serving.score_cross_section: score descending, ticker ascending.
    order = sorted(range(len(symbols)), key=lambda i: (-float(scores[i]), symbols[i][0]))
    return {symbols[i]: rank for rank, i in enumerate(order, start=1)}


def _tie_rows(scores) -> int:
    from collections import Counter

    return sum(n for n in Counter(float(s) for s in scores).values() if n > 1)


def _compare_scoring(a_panel: Path, b_panel: Path, bundle_dir: Path) -> dict:
    import numpy as np
    import polars as pl

    from modeler.serving.kr_model import load_bundle
    from modeler.serving.kr_serving import build_design_matrix

    bundle = load_bundle(bundle_dir, verify_golden=True)
    built = []
    for path in (a_panel, b_panel):
        selected, matrix = build_design_matrix(bundle, pl.read_parquet(path))
        keys = list(zip(
            selected.get_column("ticker").cast(pl.String).to_list(),
            selected.get_column("market").cast(pl.String).to_list(), strict=True))
        built.append((keys, np.ascontiguousarray(matrix, dtype=np.float64)))
    (keys_a, mat_a), (keys_b, mat_b) = built
    shared = sorted(set(keys_a) & set(keys_b))
    index_a, index_b = {k: i for i, k in enumerate(keys_a)}, {k: i for i, k in enumerate(keys_b)}
    rows_a, rows_b = [index_a[k] for k in shared], [index_b[k] for k in shared]
    mat_a, mat_b = mat_a[rows_a], mat_b[rows_b]
    design = list(bundle.design_columns)
    differs = mat_a.view(np.uint64) != mat_b.view(np.uint64)
    finite = np.isfinite(mat_a) & np.isfinite(mat_b)
    with np.errstate(invalid="ignore"):
        delta = np.where(finite, np.abs(mat_a - mat_b), 0.0)
    report: dict = {
        "bundle_manifest_sha256": bundle.manifest.get("manifest_sha256"),
        "rows_compared": len(shared), "rows_only_in_a": len(set(keys_a) - set(keys_b)),
        "rows_only_in_b": len(set(keys_b) - set(keys_a)),
        "matrix": {
            "shape": list(mat_a.shape), "bitwise_equal": bool(not differs.any()),
            "differing_cells": int(differs.sum()), "differing_rows": int(differs.any(axis=1).sum()),
            "differing_columns": [design[i] for i in np.flatnonzero(differs.any(axis=0))],
            "max_abs_diff": float(delta.max()) if delta.size else 0.0,
        },
    }
    score_a = bundle.model.predict_proba(mat_a)[:, 1]
    score_b = bundle.model.predict_proba(mat_b)[:, 1]
    rank_a, rank_b = _ranks(shared, score_a), _ranks(shared, score_b)
    rank_gap = [abs(rank_a[k] - rank_b[k]) for k in shared]
    top_a = {k for k, r in rank_a.items() if r <= 100}
    top_b = {k for k, r in rank_b.items() if r <= 100}
    report["p_raw"] = {
        "bitwise_equal": bool((score_a.view(np.uint64) == score_b.view(np.uint64)).all()),
        "differing_rows": int((score_a.view(np.uint64) != score_b.view(np.uint64)).sum()),
        "max_abs_diff": float(np.abs(score_a - score_b).max()) if len(shared) else 0.0,
    }
    report["ranks"] = {
        "equal": not any(rank_gap), "differing_rows": int(sum(1 for g in rank_gap if g)),
        "max_rank_shift": max(rank_gap, default=0),
        "tied_rows": {"a": _tie_rows(score_a), "b": _tie_rows(score_b)},
    }
    report["top100"] = {
        "set_equal": top_a == top_b, "only_in_a": sorted(map(list, top_a - top_b)),
        "only_in_b": sorted(map(list, top_b - top_a)),
        "order_equal": sorted(top_a, key=rank_a.get) == sorted(top_b, key=rank_b.get),
    }
    report["equal"] = bool(
        not report["rows_only_in_a"] and not report["rows_only_in_b"]
        and report["matrix"]["bitwise_equal"] and report["p_raw"]["bitwise_equal"]
        and report["ranks"]["equal"] and report["top100"]["set_equal"]
        and report["top100"]["order_equal"]
    )
    return report


def compare_prepared(
    a_dir: Path, b_dir: Path, *, bundle_dir: Path | None, engine: EngineOptions,
    sample: int = SAMPLE_ROWS,
) -> dict:
    a_panel, a_manifest = _prepared_files(a_dir)
    b_panel, b_manifest = _prepared_files(b_dir)
    con = _connect(engine)
    try:
        panel = compare_tables(
            con, _parquet(str(a_panel)), _parquet(str(b_panel)), key=DAILY_KEY, sample=sample)
    finally:
        con.close()
    report: dict = {
        "schema_version": SCHEMA_VERSION, "mode": "prepared",
        "generated_at": datetime.now(UTC).isoformat(),
        "a_dir": str(a_dir), "b_dir": str(b_dir), "panel": panel,
        # Informational: input markers and versions differ by design for a candidate.
        "manifest_differences": _manifest_differences(a_manifest, b_manifest),
    }
    equal = panel["equal"]
    if bundle_dir is None:
        report["scoring"] = {"skipped": "no --bundle given; matrix, p_raw and ranks not compared"}
    else:
        report["scoring"] = _compare_scoring(a_panel, b_panel, bundle_dir)
        equal = equal and report["scoring"]["equal"]
    report["equal"] = bool(equal)
    return report


def _summary_line(name: str, result: dict) -> str:
    if result["equal"]:
        rows = result.get("row_count", {}).get("a")
        return f"EQUAL {name}" + (f" rows={rows:,}" if rows is not None else "")
    why = list(result.get("problems", []))
    if result.get("schema_equal") is False:
        why.append("schema differs")
    counts = result.get("row_count")
    if counts and counts["a"] != counts["b"]:
        why.append(f"rows {counts['a']:,} vs {counts['b']:,}")
    dup = result.get("duplicates")
    if dup and dup["a"] != dup["b"]:
        why.append(f"duplicates a={dup['a']['duplicate_keys']} b={dup['b']['duplicate_keys']}")
    except_all = result.get("except_all", {})
    if except_all.get("a_minus_b") or except_all.get("b_minus_a"):
        why.append(f"except a-b={except_all.get('a_minus_b')} b-a={except_all.get('b_minus_a')}")
    keyed = result.get("keyed")
    if keyed and not keyed["clean"]:
        why.append(f"keyed: only_a={keyed['only_in_a']} only_b={keyed['only_in_b']} "
                   f"columns={sorted(keyed['columns'])[:6]}")
    return f"DIFF  {name}: " + "; ".join(why or ["see report"])


def _print_summary(report: dict, stream=None) -> None:
    stream = stream or sys.stdout
    if report["mode"] == "marts":
        for name, result in report["marts"].items():
            print(_summary_line(name, result), file=stream)
    else:
        print(_summary_line("feature_panel", report["panel"]), file=stream)
        scoring = report["scoring"]
        if "skipped" in scoring:
            print(f"NOTE  scoring skipped: {scoring['skipped']}", file=stream)
        else:
            print(
                f"{'EQUAL' if scoring['equal'] else 'DIFF '} scoring: "
                f"matrix_bitwise={scoring['matrix']['bitwise_equal']} "
                f"p_raw_bitwise={scoring['p_raw']['bitwise_equal']} "
                f"ranks_equal={scoring['ranks']['equal']} "
                f"top100_set_equal={scoring['top100']['set_equal']}", file=stream)
        changed = len(report["manifest_differences"])
        print(f"NOTE  manifest fields differing: {changed}", file=stream)
    print("RESULT " + ("equal" if report["equal"] else "different"), file=stream)


def _parse_keys(values: list[str]) -> dict[str, tuple[str, ...]]:
    keys: dict[str, tuple[str, ...]] = {}
    for item in values:
        name, sep, cols = item.partition("=")
        if not sep or not cols:
            raise UsageError(f"--key must look like mart=col1,col2: {item!r}")
        keys[name] = tuple(c for c in cols.split(",") if c)
    return keys


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--report", required=True, type=Path, help="JSON report path")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-limit", default="4GB")
    parser.add_argument("--max-temp-size", default=DEFAULT_MAX_TEMP_SIZE)
    parser.add_argument(
        "--temp-dir", type=Path,
        help="parent for the DuckDB spill directory (default: system temp)")
    parser.add_argument("--sample", type=int, default=SAMPLE_ROWS, help="differing rows to print")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    marts = modes.add_parser("marts", help="compare feature-mart roots")
    marts.add_argument("--a-root", required=True, type=Path, action="append",
                       help="repeatable: first root that holds the mart wins (e.g. metric lake)")
    marts.add_argument("--b-root", required=True, type=Path, action="append")
    marts.add_argument("--marts", help="comma-separated mart names (default: all found)")
    marts.add_argument("--key", action="append", default=[], metavar="MART=COL,COL",
                       help="key override for a non-daily mart")
    marts.add_argument("--abs-tol", type=float, default=0.0)
    marts.add_argument("--rel-tol", type=float, default=0.0)
    _add_common(marts)
    prepared = modes.add_parser("prepared", help="compare kr_prepare outputs and scoring")
    prepared.add_argument("--a-dir", required=True, type=Path)
    prepared.add_argument("--b-dir", required=True, type=Path)
    prepared.add_argument("--bundle", type=Path, help="pinned model bundle; omit to skip scoring")
    _add_common(prepared)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.threads < 1 or args.sample < 0:
        raise UsageError("--threads must be >= 1 and --sample >= 0")
    if args.temp_dir is not None and not args.temp_dir.is_absolute():
        raise UsageError("--temp-dir must be an absolute path")
    if args.temp_dir is not None:
        args.temp_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix="kr_compare_", dir=args.temp_dir))
    engine = EngineOptions(
        threads=args.threads, memory_limit=args.memory_limit, temp_directory=str(run_dir),
        max_temp_directory_size=args.max_temp_size)
    try:
        if args.mode == "marts":
            report = compare_marts(
                args.a_root, args.b_root,
                marts=[m for m in (args.marts or "").split(",") if m] or None,
                key_overrides=_parse_keys(args.key),
                tol=Tolerance(args.abs_tol, args.rel_tol), engine=engine, sample=args.sample)
        else:
            report = compare_prepared(
                args.a_dir, args.b_dir, bundle_dir=args.bundle, engine=engine, sample=args.sample)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
    write_json_atomic(args.report, report)
    _print_summary(report)
    return 0 if report["equal"] else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (UsageError, FileNotFoundError, NotADirectoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
