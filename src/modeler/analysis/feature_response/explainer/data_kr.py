"""KR data builder: layer-3 run + frozen panel + daily prices -> the numbers one page draws."""

from __future__ import annotations

import datetime as dt
import glob
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl

from .common import clean_json, pick_example, quantiles, strip_md, supp_shape
from .paths import read_parquet, scan_parquet

Q_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
# Constants quoted from the result document (09 section 1 table); not in the manifest.
BASELINE_IC = 0.09986
SHAP_DATES = 53
SHAP_ROWS = 124022

_RE_CONTRAST = re.compile(
    r"^\| (\d+) \| `([a-z_0-9]+)` \| ([\d.]+) \| [^|]+\| ([^|]+)\| ([^|]+)\| ([YN]) \| "
    r"([^|]+)\| ([^|]+)\|$",
    re.M,
)
_RE_SUPP = re.compile(
    r"^\| (\d+) \| `([a-z_0-9]+)` \| ([\d.]+) \| ([^|]+)\| (\d+)/20 \| ([\d%—]+) \| ([^|]+)\|$",
    re.M,
)
_RE_SUMMARY = re.compile(r"^\| `([a-z_0-9]+)` \| ([^|]+)\| ([^|]+)\| ([^|]+)\| ([^|]+)\|$", re.M)
_RE_CARD = re.compile(r"^### `([a-z_0-9]+)`.*?(?=^### `|\Z)", re.M | re.S)
_RE_CARDLINE = re.compile(
    r"\| 모델 반응 \(층 3\) \| 섞기 ΔrankIC ([+\-−][\d.]+) \(단독 47개 중 (\d+)위, "
    r"(안정|불안정)\) · SHAP (\d+)위, 방향 ρ ([^,]+), 모양 ([^(,]+)\(보조 ([^)]+)\), "
    r"(안정|불안정) · ([^.|]+)"
)


def parse_result_md(text: str) -> tuple[dict, dict]:
    """SHAP table rows (rank, expected sign, contrast) and the supplementary-shape table."""
    contrast = {
        m.group(2): {
            "rank": int(m.group(1)),
            "expected": strip_md(m.group(7)),
            "result": strip_md(m.group(8)),
        }
        for m in _RE_CONTRAST.finditer(text)
    }
    supp = {
        m.group(2): {
            "flat": int(m.group(5)),
            "up": strip_md(m.group(6)),
            "verdict": strip_md(m.group(7)),
        }
        for m in _RE_SUPP.finditer(text)
    }
    return contrast, supp


def parse_cards_md(text: str) -> tuple[dict, dict]:
    """Summary-table cells and the "모델 반응 (층 3)" row of every card."""
    summary = {
        m.group(1): {
            "expected": strip_md(m.group(3)),
            "result": strip_md(m.group(4)),
            "level": strip_md(m.group(5)),
        }
        for m in _RE_SUMMARY.finditer(text)
    }
    cardline = {}
    for blk in _RE_CARD.finditer(text):
        m = _RE_CARDLINE.search(blk.group(0))
        if m:
            cardline[blk.group(1)] = {
                "dic": float(m.group(1).replace("−", "-")),
                "perm_rank": int(m.group(2)),
                "perm_stab": m.group(3),
                "shap_rank": int(m.group(4)),
                "rho": m.group(5),
                "shape": m.group(6),
                "supp": m.group(7),
                "shap_stab": m.group(8),
                "contrast": m.group(9),
            }
    return summary, cardline


def _prefixes(cfg: SimpleNamespace) -> tuple[str, ...]:
    pfx = getattr(cfg, "PREFIX", "")
    return tuple(pfx) if isinstance(pfx, (list, tuple)) else (pfx,)


def build_kr(cfg: SimpleNamespace) -> tuple[dict[str, Any], list[str]]:
    """Return ``(page data, log lines)``. A log line with ``MISMATCH`` is a real problem."""
    log: list[str] = []
    feats_all = [f for _, _, fs in cfg.THEMES for f in fs]
    pfx = _prefixes(cfg)
    ref = dt.date.fromisoformat(cfg.REF_DATE)
    run = Path(cfg.RUN_DIR)
    shap = read_parquet(run / "shap_summary.parquet")
    imp = read_parquet(run / "importance_summary.parquet")
    pdp = read_parquet(run / "pdp.parquet")
    ice = read_parquet(run / "ice_sample.parquet")
    grp = json.loads((run / "groups.json").read_text())
    man = json.loads((run / "manifest.json").read_text())

    shap = shap.sort("mean_abs_shap_combined", descending=True).with_row_index(
        "shap_rank", offset=1
    )
    sing = imp.filter(pl.col("level") == "feature").with_columns(
        pl.col("features").alias("feature")
    )
    sing = sing.sort("delta_rank_ic_full", descending=True).with_row_index("perm_rank", offset=1)
    shap_by = {r["feature"]: r for r in shap.to_dicts()}
    perm_by = {r["feature"]: r for r in sing.to_dicts()}

    contrast, supp_md = parse_result_md(Path(cfg.MD_RESULT).read_text(encoding="utf-8"))
    summary, cardline = parse_cards_md(Path(cfg.MD_CARDS).read_text(encoding="utf-8"))

    panel = read_parquet(cfg.PANEL).filter(pl.col("trade_date") == ref)
    assert panel.height > 0, "no panel rows on the reference date"
    grid21 = (
        pdp.filter((pl.col("feature") == feats_all[0]) & (pl.col("subset") == "full"))
        .sort("grid_idx")["grid"]
        .to_list()
    )
    markets = panel["market"].to_numpy()
    ticks = {}
    for f in feats_all:
        col = panel[f].cast(pl.Float64).to_numpy().astype(float)
        ticks[f] = {
            "grid": quantiles(col, grid21),
            "q": Q_LEVELS,
            "all": quantiles(col, Q_LEVELS),
            "KOSPI": quantiles(col[markets == "KOSPI"], Q_LEVELS),
            "KOSDAQ": quantiles(col[markets == "KOSDAQ"], Q_LEVELS),
            "nnull": int(np.isnan(col).sum()),
            "n": int(len(col)),
        }

    rng = np.random.default_rng(20261014)
    feats = []
    grid = None
    for f in feats_all:
        s, p = shap_by[f], perm_by[f]
        sub: dict[str, list[float]] = {}
        raw_sub: dict[str, list[float]] = {}
        for k in ("full", "first", "second"):
            pp = pdp.filter((pl.col("feature") == f) & (pl.col("subset") == k)).sort("grid_idx")
            raw_sub[k] = pp["pdp_p"].to_list()
            sub[k] = [round(v, 5) for v in raw_sub[k]]
            grid = pp["grid"].to_list()
        a = ice.filter(pl.col("feature") == f).sort(["trade_date", "ticker", "grid_idx"])
        keys = a.select("trade_date", "ticker").unique(maintain_order=True)
        pick = sorted(rng.choice(keys.height, size=min(cfg.N_ICE, keys.height), replace=False))
        arr = a["ice_p"].to_numpy().reshape(keys.height, len(grid))
        lines = [[round(float(v), 4) for v in arr[i]] for i in pick]
        flat, up, verdict = supp_shape(raw_sub["full"])
        feats.append(
            {
                "id": f,
                "shap": s["mean_abs_shap_combined"],
                "shap_rank": s["shap_rank"],
                "shap_rank_part": s["mean_abs_shap_rank"],
                "shap_isna": s["mean_abs_shap_isna"],
                "shap_ci": [s["ci_lo"], s["ci_hi"]],
                "shap_first": s["mean_abs_shap_combined_first"],
                "shap_second": s["mean_abs_shap_combined_second"],
                "rho": s["direction_spearman"],
                "rho_first": s["direction_spearman_first"],
                "rho_second": s["direction_spearman_second"],
                "stable_shap": bool(s["stable_shap"]),
                "shape": s["shape_full"],
                "pdp_range": s["pdp_range_full"],
                "flat_thr": s["flat_threshold_full"],
                "dic": p["delta_rank_ic_full"],
                "dic_ci": [p["delta_rank_ic_ci_lo"], p["delta_rank_ic_ci_hi"]],
                "dic_first": p["delta_rank_ic_first"],
                "dic_second": p["delta_rank_ic_second"],
                "dll": p["delta_logloss_full"],
                "perm_rank": p["perm_rank"],
                "stable_perm": bool(p["stable_permutation"]),
                "supp": {"flat": flat, "up": up, "verdict": verdict},
                "pdp": sub,
                "ice": lines,
                "ticks": ticks[f],
                "contrast": contrast.get(f),
                "md_expected": summary.get(f, {}).get("expected"),
                "md_result": summary.get(f, {}).get("result"),
                "md_level": summary.get(f, {}).get("level"),
            }
        )
        cl = cardline.get(f)
        if cl:
            ok = (
                cl["perm_rank"] == p["perm_rank"]
                and cl["shap_rank"] == s["shap_rank"]
                and abs(cl["dic"] - p["delta_rank_ic_full"]) < 6e-6
                and (cl["shap_stab"] == "안정") == bool(s["stable_shap"])
                and (cl["perm_stab"] == "안정") == bool(p["stable_permutation"])
                and cl["shape"].strip() == s["shape_full"]
            )
            if not ok:
                log.append(
                    f"MISMATCH card line {f}: {cl} vs perm_rank {p['perm_rank']} "
                    f"shap_rank {s['shap_rank']} dic {p['delta_rank_ic_full']} "
                    f"shape {s['shape_full']}"
                )
        else:
            log.append(f"no card line parsed for {f}")
        if f in supp_md and (supp_md[f]["flat"] != flat or supp_md[f]["verdict"] != verdict):
            log.append(f"MISMATCH supp {f}: md {supp_md[f]} vs calc flat {flat} verdict {verdict}")
        if f not in contrast:
            log.append(f"no contrast row for {f}")
        elif contrast[f]["rank"] != s["shap_rank"]:
            log.append(f"MISMATCH shap rank md09 {f}")

    scatter = [
        {
            "id": f,
            "px": f.startswith(pfx),
            "x": s["mean_abs_shap_combined"],
            "y": perm_by[f]["delta_rank_ic_full"],
            "lo": perm_by[f]["delta_rank_ic_ci_lo"],
            "hi": perm_by[f]["delta_rank_ic_ci_hi"],
            "stable_shap": bool(s["stable_shap"]),
            "stable_perm": bool(perm_by[f]["stable_permutation"]),
            "shap_rank": s["shap_rank"],
            "perm_rank": perm_by[f]["perm_rank"],
            "rho": s["direction_spearman"],
            "shape": s["shape_full"],
        }
        for f, s in shap_by.items()
    ]

    names = grp["features"]
    rho = grp["rho_mean_daily_spearman"]
    groups = []
    for g in imp.filter(pl.col("level") == "corr_group").to_dicts():
        mem = g["features"].split(",")
        if not any(m.startswith(pfx) for m in mem):
            continue
        edges = []
        allr = []
        for i, a_ in enumerate(mem):
            for b_ in mem[i + 1 :]:
                r = rho[names.index(a_)][names.index(b_)]
                allr.append(r)
                if r is not None and abs(r) >= grp["threshold_abs_rho"]:
                    edges.append({"a": a_, "b": b_, "rho": round(r, 2)})
        groups.append(
            {
                "id": g["group_id"],
                "members": [
                    {
                        "id": m,
                        "shap": shap_by[m]["mean_abs_shap_combined"],
                        "px": m.startswith(pfx),
                    }
                    for m in mem
                ],
                "edges": edges,
                "min_abs": round(min(abs(x) for x in allr), 2),
                "dic": g["delta_rank_ic_full"],
                "dic_ci": [g["delta_rank_ic_ci_lo"], g["delta_rank_ic_ci_hi"]],
                "dic_first": g["delta_rank_ic_first"],
                "dic_second": g["delta_rank_ic_second"],
                "shap_sum": sum(shap_by[m]["mean_abs_shap_combined"] for m in mem),
            }
        )
    singles = [g[0] for g in grp["corr_groups"] if len(g) == 1 and g[0].startswith(pfx)]

    examples = _examples_kr(cfg, panel, ref, log)
    data = {
        "market": "kr",
        "meta": {
            "run_id": man["run_id"],
            "commit": man["worktree_git"]["commit"][:7],
            "created_utc": man["created_utc"],
            "ref_date": cfg.REF_DATE,
            "sample": man["sample"],
            "baseline_ic": BASELINE_IC,
            "shap_dates": SHAP_DATES,
            "shap_rows": SHAP_ROWS,
            "n_features_total": len(shap_by),
        },
        "grid": grid,
        "features": feats,
        "scatter": scatter,
        "groups": groups,
        "singles": singles,
        "examples": examples,
        "ref_n": panel.height,
    }
    return clean_json(data), log


def _examples_kr(cfg, panel: pl.DataFrame, ref: dt.date, log: list[str]) -> dict:
    ohl = (
        scan_parquet(Path(cfg.RAW) / "daily_ohlcv/**/*.parquet", hive_partitioning=True)
        .filter(
            (pl.col("trade_date") <= ref) & (pl.col("trade_date") >= dt.date(ref.year - 2, 6, 1))
        )
        .collect()
    )
    assert ohl["trade_date"].max() <= ref
    master_file = sorted(
        glob.glob(str(Path(cfg.RAW) / "stock_master/**/*.parquet"), recursive=True)
    )
    master = read_parquet(master_file[0]).select("ticker", "name")
    cnt = ohl.group_by("ticker").agg(
        pl.len().alias("nrows"), pl.col("close").pct_change().abs().max().alias("mx")
    )
    px = panel.join(cnt, on="ticker", how="left").join(master, on="ticker", how="left")
    windows = getattr(cfg, "EX_WINDOWS", {})
    examples: dict[str, list[dict]] = {}
    for f, specs in getattr(cfg, "EXAMPLES", {}).items():
        out = []
        y = px.filter((pl.col("market") == cfg.EX_MARKET) & pl.col(f).is_not_null())
        n = y.height
        y = y.with_columns((pl.col(f).rank("average") / n).alias("pct"))
        for sp in specs:
            if sp.get("ticker"):
                cand = y.filter(pl.col("ticker") == sp["ticker"])
                row = cand.row(0, named=True) if cand.height else None
            else:
                base = y.filter((pl.col("nrows") >= cfg.EX_DAYS + 1) & (pl.col("mx") < 0.31))
                base = base.with_columns(pl.col("px_turnover_ma20").alias("size"))
                target = 0.95 if sp["kind"] == "p95" else 0.05
                row, fallback = pick_example(base, target)
                if fallback:
                    log.append(
                        f"EX {f} {sp['kind']}: 목표 백분위 근처 후보 없음 -> 가장 가까운 종목 사용"
                    )
            if row is None:
                log.append(f"EX {f} {sp['kind']}: 후보 없음, 건너뜀")
                continue
            out.append(_example_kr(f, sp, row, ohl, cfg, ref, windows.get(f, []), n, log))
        if out:
            examples[f] = out
    return examples


def _example_kr(f, sp, r, ohl, cfg, ref, window, n, log) -> dict:
    t = ohl.filter(pl.col("ticker") == r["ticker"]).sort("trade_date").tail(cfg.EX_DAYS)
    assert t["trade_date"].max() == ref, (r["ticker"], t["trade_date"].max())
    cl = t["close"].to_numpy().astype(float)
    halted = ((t["open"] == 0) & (t["high"] == 0) & (t["low"] == 0)).to_numpy()
    last = len(cl) - 1
    full = ohl.filter(pl.col("ticker") == r["ticker"]).sort("trade_date")
    fc = full["close"].to_numpy().astype(float)
    fh = full["high"].to_numpy().astype(float)
    fl = full["low"].to_numpy().astype(float)
    lf = len(fc) - 1
    chk: dict[str, float] = {}
    if f == "px_mom_20_60":
        chk["recalc"] = math.log(fc[lf - 60] / fc[lf - 20])
    if f == "px_high_low_range_20d":
        chk["recalc"] = (fh[lf - 19 :].max() - fl[lf - 19 :].min()) / fc[lf]
    if f == "px_dist_52w_high":
        chk["recalc"] = fc[lf] / fc[lf - 251 :].max() - 1
        chk["recalc_253"] = fc[lf] / fc[lf - 252 :].max() - 1
    log.append(
        f"EX {f} {sp['kind']} {r['ticker']} {r['name']} pct={r['pct']:.4f} "
        f"value={r[f]:.6f} n={n} chk={chk}"
    )
    e = {
        "feature": f,
        "kind": sp["kind"],
        "ticker": r["ticker"],
        "name": r["name"],
        "market": r["market"],
        "value": r[f],
        "pct": r["pct"],
        "n_market": n,
        "dates": [str(x) for x in t["trade_date"].to_list()],
        "close": [int(v) for v in cl],
        "norm": [round(float(v / cl[0] * 100), 2) for v in cl],
        "halted": [bool(h) for h in halted],
        "hi20": float(fh[lf - 19 :].max()),
        "lo20": float(fl[lf - 19 :].min()),
        "close_last": float(cl[-1]),
        "halt_ratio": r["px_halt_ratio_20d"],
        "turnover_ma20": r["px_turnover_ma20"],
        "recalc": chk,
        "window": window,
    }
    if f == "px_dist_52w_high":
        e["max_idx"] = int(last - 251 + int(np.argmax(cl[last - 251 :])))
    if f == "px_mom_20_60":
        e["c60"] = float(cl[last - 60])
        e["c20"] = float(cl[last - 20])
    return e
