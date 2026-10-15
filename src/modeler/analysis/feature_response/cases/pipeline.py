"""Run the pilot: select cases, build price series, distributions, parquet tables, manifest."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from . import config as C
from .data import (
    calendar_of,
    case_flags,
    case_series,
    corp_action_text,
    dataset_manifest_path,
    feature_frame,
    is_preferred,
    load_corp_actions,
    load_names,
    load_panel,
    load_prices,
    market_returns,
    panel_path,
    path_cum_excess,
)
from .dist import distribution_rows
from .guards import (
    HoldoutAccessError,
    check_formation_date,
    check_label_end,
    guard_not_in_repo,
    guard_path,
)
from .select import add_pct, build_grid, choose_dates, group_names, select_cases, with_groups

GROUP_RANK = {"good": 0, "top": 0, "opp": 1, "bottom": 1, "all": 2}


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def tree_digest(root: Path, pattern: str = "**/*.parquet") -> dict:
    """File-list digest (relative path + size) for a hive directory; not a content hash."""
    files = sorted(root.glob(pattern))
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(root).as_posix()}|{f.stat().st_size}\n".encode())
    return {"kind": "file-list digest (path+size)", "files": len(files), "sha256": h.hexdigest()}


def code_state() -> dict:
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(here), "status", "--porcelain", "--", str(here)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return {"commit": head, "dirty_in_cases_package": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty_in_cases_package": None}


def repo_root_of_code() -> Path | None:
    here = Path(__file__).resolve().parent
    try:
        top = subprocess.run(
            ["git", "-C", str(here), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return Path(top)
    except (OSError, subprocess.CalledProcessError):
        return None


def _pick_extremes(pool: pl.DataFrame, n: int) -> pl.DataFrame:
    p = pool.filter(pl.col("raw_label_20d").is_not_null() & pl.col("raw_label_20d").is_not_nan())
    top = (
        p.sort(["raw_label_20d", "ticker"], descending=[True, False])
        .head(n)
        .with_columns(pl.lit("extreme_top").alias("kind"))
    )
    bot = (
        p.sort(["raw_label_20d", "ticker"], descending=[False, False])
        .head(n)
        .with_columns(pl.lit("extreme_bottom").alias("kind"))
    )
    top = top.with_columns(pl.int_range(1, top.height + 1).alias("ext_rank"))
    bot = bot.with_columns(pl.int_range(1, bot.height + 1).alias("ext_rank"))
    return pl.concat([top, bot])


def run_pilot(
    stock_data: Path,
    out_dir: Path,
    features: list[str],
    seed: int = C.SEED,
    run_id: str = "run",
    log: Callable[[str], None] = print,
) -> dict:
    guard_not_in_repo(out_dir, repo_root_of_code())
    # the guard must actually fire; a silent guard is worse than none
    try:
        guard_path("x/holdout/y")
    except HoldoutAccessError:
        holdout_ok = True
    else:  # pragma: no cover
        raise RuntimeError("holdout guard did not fire")

    specs = {f: C.FEATURES[f] for f in features}
    ppath = panel_path(stock_data)
    log(f"panel {ppath}")
    panel = load_panel(ppath, list(specs))
    cal = calendar_of(panel)
    grid = build_grid(cal, data_end=C.DATA_END)
    for d, i, end in grid:
        check_formation_date(d)
        check_label_end(end)
    gidx = [i for _, i, _ in grid]
    assert all(b - a == C.HORIZON for a, b in zip(gidx, gidx[1:], strict=False)), "grid overlaps"
    grid_dates = [d for d, _, _ in grid]
    dates = choose_dates(grid_dates, seed)
    log(f"grid {len(grid)} dates, chosen {[d.isoformat() for d in dates]}")
    ci = {d: i for i, d in enumerate(cal)}

    mret = market_returns(panel, cal)
    prices = load_prices(stock_data, cal, mret)
    names = load_names(stock_data)
    actions = load_corp_actions(stock_data)
    log(f"prices {prices.close.shape}, corp actions {actions.height}")

    labels = panel.select("trade_date", "ticker", *["fwd_ret_20d", "raw_label_20d"])
    negeq = panel.select("trade_date", "ticker", "fin_is_negative_equity")
    base = (
        panel.filter(pl.col("trade_date").is_in(grid_dates))
        .select("trade_date", "ticker", "market")
        .join(labels, on=["trade_date", "ticker"], how="left")
        .with_row_index("ri")
    )
    tid = np.array([prices.tid.get(t, -1) for t in base["ticker"].to_list()])
    ci0 = np.array([ci[d] for d in base["trade_date"].to_list()])
    paths_all = path_cum_excess(prices, np.where(tid < 0, 0, tid), ci0, base["market"].to_list())
    paths_all[tid < 0] = np.nan

    case_frames: list[pl.DataFrame] = []
    dist_rows: list[dict] = []
    for f, spec in specs.items():
        log(f"feature {f}")
        pct = add_pct(feature_frame(panel, f, grid_dates))
        picks = select_cases(
            pct.filter(pl.col("trade_date").is_in(dates)), spec.direction, seed, f
        ).with_columns(
            pl.lit("random").alias("kind"), pl.lit(None, dtype=pl.Int64).alias("ext_rank")
        )
        gdf = with_groups(pct, spec.direction)
        pool_groups = list(group_names(spec.direction)) if spec.direction == 0 else ["good"]
        pool = gdf.filter(
            pl.col("trade_date").is_in(dates) & pl.col("group").is_in(pool_groups)
        ).join(labels, on=["trade_date", "ticker"], how="left")
        ext = _pick_extremes(pool, C.N_EXTREME).select(picks.columns)
        cases = pl.concat([picks, ext]).with_columns(pl.lit(f).alias("feature"))
        case_frames.append(cases)

        g = gdf.join(
            base.select("trade_date", "ticker", "ri", "raw_label_20d", "fwd_ret_20d"),
            on=["trade_date", "ticker"],
            how="inner",
        )
        dist_rows += distribution_rows(f, spec.direction, g, paths_all[g["ri"].to_numpy()])

    allc = (
        pl.concat(case_frames)
        .join(labels, on=["trade_date", "ticker"], how="left")
        .join(negeq, on=["trade_date", "ticker"], how="left")
    )

    # ---- price series and flags for every distinct (date, ticker) ------------------------------
    keys = allc.select("trade_date", "ticker", "market").unique().sort(["trade_date", "ticker"])
    series: dict[tuple, dict] = {}
    flag_rows = []
    for r in keys.iter_rows(named=True):
        d, t, m = r["trade_date"], r["ticker"], r["market"]
        c0 = ci[d]
        s = case_series(prices, t, m, c0)
        if s is not None:
            series[(d, t)] = s
        fl = case_flags(prices, t, c0)
        fl["corp_action"] = corp_action_text(actions, prices, t, c0)
        name = names.get(t)
        fl["name"] = name or t
        fl["preferred"] = is_preferred(t, name)
        fl["trade_date"], fl["ticker"] = d, t
        flag_rows.append(fl)
    flags = pl.DataFrame(flag_rows)
    allc = allc.join(flags, on=["trade_date", "ticker"], how="left").with_columns(
        pl.col("raw_label_20d").is_null().alias("label_missing"),
        pl.when(pl.col("group") == "good")
        .then(pl.col("raw_label_20d") > 0)
        .otherwise(None)
        .alias("hit"),
    )
    allc = (
        allc.with_columns(pl.col("group").replace_strict(GROUP_RANK, default=3).alias("_g"))
        .sort(["feature", "kind", "trade_date", "_g", "ticker"])
        .drop("_g")
    )
    allc = allc.with_columns(
        (
            pl.col("feature")
            + "|"
            + pl.col("kind")
            + "|"
            + pl.col("trade_date").cast(pl.String)
            + "|"
            + pl.col("ticker")
        ).alias("case_id")
    )
    for d in allc["trade_date"].unique().to_list():
        check_formation_date(d)
        check_label_end(cal[ci[d] + C.HORIZON])

    # day-20 path vs panel label agreement
    diffs = []
    gaps = []
    for r in allc.select("trade_date", "ticker", "fwd_ret_20d").unique().iter_rows(named=True):
        s = series.get((r["trade_date"], r["ticker"]))
        if s and r["fwd_ret_20d"] is not None and s["close_norm"][-1] is not None:
            gap = abs(s["close_norm"][-1] / 100 - 1 - r["fwd_ret_20d"])
            diffs.append(gap)
            gaps.append({"trade_date": r["trade_date"], "ticker": r["ticker"], "path_gap": gap})
    diffs_a = np.array(diffs)
    allc = allc.join(
        pl.DataFrame(
            gaps, schema={"trade_date": pl.Date, "ticker": pl.String, "path_gap": pl.Float64}
        ),
        on=["trade_date", "ticker"],
        how="left",
    )

    # ---- write --------------------------------------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    cases_out = allc.select(
        "case_id",
        "feature",
        "kind",
        "ext_rank",
        "group",
        "trade_date",
        "ticker",
        "name",
        "market",
        "value",
        "pct",
        "fwd_ret_20d",
        "raw_label_20d",
        "hit",
        "label_missing",
        "halt_fwd",
        "halt_prev20",
        "limit_hits",
        "jump_over_limit",
        "preferred",
        pl.col("fin_is_negative_equity").alias("neg_equity"),
        "corp_action",
        "path_gap",
    )
    cases_out.write_parquet(out_dir / "cases.parquet")
    dist_df = pl.DataFrame(
        dist_rows,
        schema={
            "feature": pl.String,
            "group": pl.String,
            "metric": pl.String,
            "stat": pl.String,
            "k": pl.Int64,
            "value": pl.Float64,
        },
    )
    dist_df.write_parquet(out_dir / "distribution.parquet")
    prow = []
    for (d, t), s in series.items():
        for i, rel in enumerate(s["rel"]):
            prow.append((d, t, rel, s["close_norm"][i], s["mkt_norm"][i], s["halt"][i]))
    paths_df = pl.DataFrame(
        prow,
        schema=["trade_date", "ticker", "rel_day", "close_norm", "mkt_norm", "halt"],
        orient="row",
    )
    paths_df.write_parquet(out_dir / "paths.parquet")

    raw = stock_data / C.RAW_REL
    manifest = {
        "run_id": run_id,
        "purpose": C.PURPOSE,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "labels_used": "dev_only",
        "seed": seed,
        "features": list(specs),
        "directions": {f: {"direction": s.direction, "basis": s.basis} for f, s in specs.items()},
        "formation_dates": [d.isoformat() for d in dates],
        "grid": {
            "n": len(grid),
            "first": grid_dates[0].isoformat(),
            "last": grid_dates[-1].isoformat(),
            "step_trading_days": C.HORIZON,
        },
        "rules": {
            "dates": "one per year 2016-2025 + 2 extra from the rest, random.Random(seed)",
            "per_date": {"good_or_top": C.N_GOOD, "opp_or_bottom": C.N_OPP, "all": C.N_ALL},
            "pct": "average rank / count within (trade_date, market)",
            "good_side": f"pct >= {C.TOP_PCT} or <= {C.BOTTOM_PCT} by direction",
            "hit": "raw_label_20d > 0, good side only",
            "extremes": f"top/bottom {C.N_EXTREME} raw_label_20d in the good-side group",
            "limit_hit": f"{C.LIMIT_LO} <= |close/prev-1| <= {C.LIMIT_HI}, non-halt, day +1..+20",
            "halt": "open = high = low = 0",
            "market_line": "per (date, market) equal-weight px_ret_1d mean, compounded (approx.)",
            "corp_actions": "DART dart_capital_change_raw isu_dcrs_stle in "
            + ", ".join(C.CORP_ACTIONS),
        },
        "inputs": {
            "panel": {"rel": C.PANEL_REL, "sha256": sha256_file(ppath)},
            "dataset_manifest": {
                "rel": C.DATASET_MANIFEST_REL,
                "sha256": sha256_file(dataset_manifest_path(stock_data)),
            },
            "stock_master": {
                "sha256": sha256_file(raw / "stock_master/schema_version=1/part-000000.parquet")
            },
            "daily_ohlcv": tree_digest(raw / "daily_ohlcv"),
            "dart_capital_change_raw": tree_digest(raw / "dart_capital_change_raw"),
        },
        "code": code_state(),
        "counts": {
            "cases_random": int(allc.filter(pl.col("kind") == "random").height),
            "cases_extreme": int(allc.filter(pl.col("kind") != "random").height),
            "per_feature": {
                f: {
                    k: int(v)
                    for k, v in allc.filter(pl.col("feature") == f)
                    .group_by("kind")
                    .len()
                    .iter_rows()
                }
                for f in specs
            },
            "label_missing_cases": int(allc.filter(pl.col("label_missing")).height),
            "corp_action_dart_events": int(actions.height),
        },
        "forbidden_checks": {
            "holdout_guard_fires": holdout_ok,
            "max_trade_date_read": str(panel["trade_date"].max()),
            "max_label_window_end": max(cal[ci[d] + C.HORIZON] for d in dates).isoformat(),
            "all_formation_dates_in_window": True,
            "label_windows_end_by_2025_07_31": True,
            "grid_non_overlapping": True,
            "selection_received_no_outcome_columns": True,
            "no_write_inside_repo": True,
            "holdout_files_opened": False,
        },
        "path_vs_label_check": {
            "n": int(diffs_a.size),
            "median_abs_diff": float(np.median(diffs_a)) if diffs_a.size else None,
            "p99_abs_diff": float(np.quantile(diffs_a, 0.99)) if diffs_a.size else None,
            "share_gt_1pct": float((diffs_a > 0.01).mean()) if diffs_a.size else None,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "manifest": manifest,
        "cases": cases_out,
        "dist": dist_df,
        "series": series,
        "dates": dates,
        "n_grid": len(grid),
    }
