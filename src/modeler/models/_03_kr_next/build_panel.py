"""Guarded panel builder for the next KR model (G-22, prereg draft §16).

Two modes, one assembly (``dataset.build_kr_next_panel_sql``):

* ``serving`` — no labels, no returns. Needs no freeze tag. Used by the shadow scorer's input and
  by label-free measurements. The output never has a label or return column; the builder checks
  the written column list again and removes nothing silently: it refuses.
* ``labeled`` — the training panel (``y``, ``trainable``, exit kind / end date / lag). The first
  thing the CLI does is ``train.require_frozen_tag``: without the ``kr-next-frozen`` tag in
  ``--prereg-repo`` it exits 2 before the lake is opened (the labels must not be opened before
  the freeze).

Outputs are write-once: ``<out-dir>/panel_<mode>.parquet`` and ``<out-dir>/manifest_<mode>.json``
(the training CLI reads ``panel_labeled.parquet``). If either exists the CLI exits 2 and writes
nothing. The manifest records the mode, snapshot, date range, row count, column list, the modeler
git commit and dirty flag, the sha256 of every view SQL used (G-22: each inline mart, the tail or
label view, the universe view, the 정리매매 rule view, and the whole panel statement) and the
sha256 of the prebuilt derived views the lake does not hold as raw tables.

Inputs. Raw tables come from ``<lake>/raw/raw_postgres/snapshot_date=<snapshot>/source=<source>``.
The derived marts (``dim_price_quality_daily``, ``dim_stock_pit_daily`` (the non-empty PIT view),
``fin_quarterly_metric_vintage_dlr``) are built by their own materializers and handed in as parquet
files ``<views-dir>/<name>.parquet`` (default ``<lake>/derived/kr_next_views/snapshot_date=<s>``).
``dim_universe_broad_daily`` and the market-cap distortion view are cheap and registered here.
Missing files stop the build (exit 2): nothing is guessed.

Usage::

    python -m modeler.models._03_kr_next.build_panel --mode serving --lake <stock_data/kr> \
        --snapshot 2026-09-30 --start 2017-01-01 --end 2025-07-31 --out-dir <dir> \
        --prereg-repo <my repo>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb

from modeler.etl.config import DataRoot, LakeConfig
from modeler.etl.corporate_actions import register_mcap_distortion_view
from modeler.etl.lake import register_views
from modeler.etl.universe import UniverseFilter, broad_universe_filter, build_universe_sql
from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import dataset as ds
from modeler.models._03_kr_next.train import FrozenTagMissing, require_frozen_tag

MODES = ("serving", "labeled")
DEFAULT_SOURCE = "sj2_remote"
#: views the assembly reads that are handed in as parquet (derived marts, built elsewhere)
DERIVED_VIEW_FILES: tuple[str, ...] = (
    "dim_price_quality_daily",
    "dim_stock_pit_daily",
    "fin_quarterly_metric_vintage_dlr",
)
_RAW_COMMON = (
    "daily_ohlcv",
    "daily_market_cap",
    "dart_filing_receipt_raw",
    "stock_master",
    "dart_corp_master",
)
_BANNED_SERVING = re.compile(
    r"fwd_ret|raw_label|label_|bench_ret|^y$|trainable|in_terminal_tail|exit_kind"
)
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class BuildRefused(RuntimeError):
    """A guard stopped the build; the CLI turns it into exit code 2."""


def output_paths(out_dir: str | Path, mode: str) -> tuple[Path, Path]:
    out = Path(out_dir)
    return out / f"panel_{mode}.parquet", out / f"manifest_{mode}.json"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_outputs_free(out_dir: str | Path, mode: str) -> None:
    """Write-once guard: refuse if the panel or the manifest already exists."""
    taken = [str(p) for p in output_paths(out_dir, mode) if p.exists()]
    if taken:
        raise BuildRefused(f"output already exists, not overwriting: {', '.join(taken)}")


def build_panel_file(
    con: duckdb.DuckDBPyConnection,
    *,
    mode: str,
    snapshot: str,
    start: str,
    end: str | None,
    out_dir: str | Path,
    feature_set: ds.FeatureSet = "main",
    views: ds.KrNextViews | None = None,
    mart_overrides: Mapping[str, str] | None = None,
    universe_filter: UniverseFilter | None = None,
    keep_raw_label: bool = False,
    extra_manifest: Mapping[str, Any] | None = None,
    modeler_dir: Path | None = None,
) -> dict[str, Any]:
    """Write ``panel_<mode>.parquet`` + ``manifest_<mode>.json`` from a connection whose views
    are registered. Returns the manifest. Raises ``BuildRefused`` on any guard.

    The tag guard for ``labeled`` is the caller's (``main``); this function only refuses a bad
    mode, an existing output, or a serving panel that carries a label column.
    """
    if mode not in MODES:
        raise BuildRefused(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "serving" and keep_raw_label:
        raise BuildRefused("serving mode never emits label or return columns")
    check_outputs_free(out_dir, mode)
    labeled = mode == "labeled"
    parts: dict[str, str] = {}
    sql = ds.build_kr_next_panel_sql(
        feature_set,
        views=views,
        include_labels=labeled,
        start_date=start,
        end_date=end,
        keep_raw_label=keep_raw_label and labeled,
        mart_overrides=mart_overrides,
        universe_filter=universe_filter,
        view_sql_out=parts,
    )
    panel_path, manifest_path = output_paths(out_dir, mode)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tmp = panel_path.with_name(panel_path.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    try:
        con.execute(f"COPY ({sql}) TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        columns = [
            r[0]
            for r in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{tmp.as_posix()}')"
            ).fetchall()
        ]
        if mode == "serving":
            bad = [c for c in columns if _BANNED_SERVING.search(c)]
            if bad:
                raise BuildRefused(f"serving panel would carry label/return columns: {bad}")
        rows = con.execute(f"SELECT count(*) FROM read_parquet('{tmp.as_posix()}')").fetchone()[0]
        if panel_path.exists():
            raise BuildRefused(f"output already exists, not overwriting: {panel_path}")
        tmp.rename(panel_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    git = bundle_mod.git_state(modeler_dir or Path(__file__).resolve().parent)
    manifest: dict[str, Any] = {
        "mode": mode,
        "snapshot": snapshot,
        "start": start,
        "end": end,
        "feature_set": feature_set,
        "row_count": int(rows),
        "columns": columns,
        "panel_file": panel_path.name,
        "panel_sha256": _sha256_file(panel_path),
        "modeler_commit": git["commit"],
        "modeler_dirty": git["dirty"],
        "view_sql_sha256": {k: _sha256_text(v) for k, v in sorted(parts.items())},
        "panel_sql_sha256": _sha256_text(sql),
        "mart_overrides": dict(mart_overrides or {}),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **dict(extra_manifest or {}),
    }
    try:
        with manifest_path.open("x", encoding="utf-8") as f:  # "x": never overwrite
            f.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as e:
        raise BuildRefused(f"output already exists, not overwriting: {manifest_path}") from e
    return manifest


def register_inputs(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    views_dir: Path,
    *,
    labeled: bool,
    feature_set: str,
) -> dict[str, str]:
    """Register raw views, the prebuilt derived views and the two cheap registered views.

    Returns ``{derived view name: sha256 of its parquet}``. Raises ``BuildRefused`` for a missing
    derived file.
    """
    raw = list(_RAW_COMMON)
    if labeled:
        raw.append("krx_index_daily")
    if feature_set == "main":
        raw.append("krx_security_flow_raw")
    try:
        register_views(con, config, tables=raw)
    except FileNotFoundError as e:
        raise BuildRefused(str(e)) from e
    hashes: dict[str, str] = {}
    for name in DERIVED_VIEW_FILES:
        f = views_dir / f"{name}.parquet"
        if not f.is_file():
            raise BuildRefused(f"derived view file missing: {f}")
        quoted = str(f).replace("'", "''")
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{quoted}')")
        hashes[name] = _sha256_file(f)
    con.execute(
        f"CREATE OR REPLACE VIEW dim_universe_broad_daily AS "
        f"{build_universe_sql(broad_universe_filter())}"
    )
    register_mcap_distortion_view(
        con, receipt_view="dart_filing_receipt_raw", market_cap_view="daily_market_cap"
    )
    return hashes


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m modeler.models._03_kr_next.build_panel",
        description="Guarded panel builder for the next KR model",
    )
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--lake", required=True, help="stock_data lake root (kr)")
    ap.add_argument(
        "--snapshot", required=True, help="raw snapshot_date (recorded in the manifest)"
    )
    ap.add_argument("--start", default=ds.DEFAULT_START_DATE, help="first trade_date (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="last trade_date, inclusive (YYYY-MM-DD)")
    ap.add_argument("--out-dir", required=True, help="output directory (files must not exist)")
    ap.add_argument(
        "--prereg-repo", required=True, help="repository that holds the kr-next-frozen tag"
    )
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="raw lake source name")
    ap.add_argument("--feature-set", default="main", choices=("main", "noflow"))
    ap.add_argument("--views-dir", default=None, help="derived view parquet directory")
    ap.add_argument("--memory-limit", default="24GB")
    ap.add_argument("--threads", type=int, default=8)
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    for name, value in (("--start", args.start), ("--end", args.end)):
        if not _DATE.match(value):
            print(f"REFUSED: {name} must be YYYY-MM-DD, got {value!r}", file=sys.stderr)
            return 2
    labeled = args.mode == "labeled"
    # 1) guards first: the freeze tag before anything is read, then write-once
    extra: dict[str, Any] = {}
    try:
        if labeled:
            extra["prereg_tag"] = "kr-next-frozen"
            extra["prereg_commit"] = require_frozen_tag(args.prereg_repo)
        check_outputs_free(args.out_dir, args.mode)
    except FrozenTagMissing as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    except BuildRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    lake = Path(args.lake)
    views_dir = (
        Path(args.views_dir)
        if args.views_dir
        else lake / "derived" / "kr_next_views" / f"snapshot_date={args.snapshot}"
    )
    out_dir = Path(args.out_dir)
    tmp_dir = out_dir.parent / f".{out_dir.name}_duckdb_tmp"
    config = LakeConfig(root=DataRoot(base=lake), snapshot_date=args.snapshot, source=args.source)
    con = duckdb.connect()
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET threads={int(args.threads)}")
        con.execute(f"SET memory_limit='{args.memory_limit}'")
        con.execute(f"SET temp_directory='{tmp_dir.as_posix()}'")
        hashes = register_inputs(
            con, config, views_dir, labeled=labeled, feature_set=args.feature_set
        )
        extra["derived_views_sha256"] = hashes
        extra["raw_root"] = str(config.raw_root)
        manifest = build_panel_file(
            con,
            mode=args.mode,
            snapshot=args.snapshot,
            start=args.start,
            end=args.end,
            out_dir=out_dir,
            feature_set=args.feature_set,
            views=ds.KrNextViews(),
            extra_manifest=extra,
        )
    except BuildRefused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    finally:
        con.close()
        if tmp_dir.exists():
            for p in tmp_dir.glob("*"):
                p.unlink(missing_ok=True)
            tmp_dir.rmdir()
    print(f"panel written: {out_dir} rows={manifest['row_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
