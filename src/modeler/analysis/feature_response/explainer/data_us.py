"""US data builder (LightGBM + Ridge). Reads no labels; permutation does not exist for US."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl

from .common import clean_json, pick_example, quantiles, supp_shape, table_cells, unescape_md
from .paths import read_parquet, scan_parquet

SHAPE_KO = {"U": "U자", "inverted_U": "역U자", "flat": "평평", "기타": "기타", None: "선형"}
Q_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]

_RE_SUMMARY = re.compile(r"^\| `([a-z_0-9]+)` \| ([^|]+)\| ([^|]+)\| ([^|]+)\| ([^|]+)\|$", re.M)
_RE_CARD = re.compile(r"^### `([a-z_0-9]+)`.*?(?=^### `|\Z)", re.M | re.S)
_RE_LGB = re.compile(
    r"LightGBM SHAP (\d+)위, ρ ([^,]+), 모양 ([^(,]+)\(보조 ([^)]+)\), (안정|불안정)"
)
_RE_RIDGE = re.compile(r"Ridge (\d+)위, ρ ([^,]+), (안정|불안정)")


def _section(md: str, start: str, end: str) -> str:
    return md[md.index(start) : md.index(end)]


def parse_us_result_md(md: str) -> tuple[dict, dict]:
    """LightGBM/Ridge table rows (rank, reason, expected, contrast) and the 1.1 table."""
    contr: dict[str, dict] = {"lightgbm": {}, "ridge": {}}
    spans = {"lightgbm": ("## 2. ", "## 3. "), "ridge": ("## 3. ", "## 4. ")}
    for model, (a, b) in spans.items():
        for line in _section(md, a, b).splitlines():
            if line.startswith("| ") and "`" in line:
                c = table_cells(line)
                if c[0].isdigit():
                    contr[model][c[1]] = {
                        "rank": int(c[0]),
                        "expected": c[7],
                        "result": c[8],
                        "reason": c[6],
                    }
    supp: dict[str, dict] = {}
    for line in _section(md, "### 1.1", "## 2. ").splitlines():
        if line.startswith("| ") and "`" in line:
            c = table_cells(line)
            if c[0].isdigit():
                supp[c[1]] = {"flat": int(c[3].split("/")[0]), "up": c[4], "verdict": c[5]}
    return contr, supp


def parse_us_cards_md(md: str) -> tuple[dict, dict]:
    summ = {
        m.group(1): {
            "expected": unescape_md(m.group(3)),
            "result": unescape_md(m.group(4)),
            "level": unescape_md(m.group(5)),
        }
        for m in _RE_SUMMARY.finditer(md)
    }
    cardline = {}
    for blk in _RE_CARD.finditer(md):
        a = _RE_LGB.search(blk.group(0))
        b = _RE_RIDGE.search(blk.group(0))
        if a and b:
            cardline[blk.group(1)] = {
                "lr": int(a.group(1)),
                "lsupp": a.group(4),
                "lst": a.group(5),
                "rr": int(b.group(1)),
                "rst": b.group(3),
            }
    return summ, cardline


def build_us(cfg: SimpleNamespace) -> tuple[dict[str, Any], list[str]]:
    log: list[str] = []
    feats_all = [f for _, _, fs in cfg.THEMES for f in fs]
    run = Path(cfg.RUN_DIR)
    shap = read_parquet(run / "shap_summary.parquet")
    pdp = read_parquet(run / "pdp.parquet")
    ice = read_parquet(run / "ice_sample.parquet")
    coef = {r["feature"]: r for r in read_parquet(run / "ridge_coef.parquet").to_dicts()}
    grp = json.loads((run / "groups.json").read_text())
    man = json.loads((run / "manifest.json").read_text())
    assert man.get("labels_read") is False, "this run must not have read labels"

    def rows(model: str, comp: str) -> dict[str, dict]:
        sel = shap.filter((pl.col("model") == model) & (pl.col("component") == comp))
        return {r["feature"]: r for r in sel.to_dicts()}

    comb = {m: rows(m, "combined") for m in ("lightgbm", "ridge")}
    part = {m: {c: rows(m, c) for c in ("rank", "isna")} for m in ("lightgbm", "ridge")}
    grid = [round(g, 4) for g in man["pdp"]["grid"]]

    contr, supp_md = parse_us_result_md(Path(cfg.MD_RESULT).read_text(encoding="utf-8"))
    summ, cardline = parse_us_cards_md(Path(cfg.MD_CARDS).read_text(encoding="utf-8"))

    ref = dt.date.fromisoformat(cfg.REF_DATE)
    v2 = scan_parquet(cfg.PANEL_V2).filter(pl.col("date") == ref).collect()
    flow = scan_parquet(cfg.PANEL_FLOW).filter(pl.col("date") == ref).collect()
    panel = v2.join(flow, on=["date", "symbol"], how="left").filter(pl.col("price_ge_5"))
    ticks = {}
    for f in feats_all:
        if f in panel.columns:
            col = panel[f].cast(pl.Float64).to_numpy().astype(float)
        else:
            col = np.array([np.nan])
            log.append(f"panel에 {f} 열 없음")
        ticks[f] = {
            "grid": quantiles(col, grid),
            "q": Q_LEVELS,
            "all": quantiles(col, Q_LEVELS),
            "nnull": int(np.isnan(col).sum()),
            "n": int(len(col)),
        }

    rng = np.random.default_rng(20261014)
    feats = []
    for f in feats_all:
        lg, rd = comb["lightgbm"][f], comb["ridge"][f]
        sub: dict[str, list[float]] = {}
        raw_sub: dict[str, list[float]] = {}
        for key, half in (("full", "all"), ("first", "first"), ("second", "second")):
            pp = pdp.filter((pl.col("feature") == f) & (pl.col("half") == half)).sort("grid")
            raw_sub[key] = pp["pdp"].to_list()
            sub[key] = [round(v, 5) for v in raw_sub[key]]
        a = ice.filter(pl.col("feature") == f).sort(["sample_idx", "grid"])
        n = a["sample_idx"].n_unique()
        arr = a["ice"].to_numpy().reshape(n, len(grid))
        pick = sorted(rng.choice(n, size=min(cfg.N_ICE, n), replace=False).tolist())
        lines = [[round(float(v), 4) for v in arr[i]] for i in pick]
        flat, up, verdict = supp_shape(raw_sub["full"])
        cf = coef[f]
        feats.append(
            {
                "id": f,
                "md_level": summ.get(f, {}).get("level"),
                "md_expected": summ.get(f, {}).get("expected"),
                "md_result": summ.get(f, {}).get("result"),
                "lgb": {
                    "shap": lg["mean_abs"],
                    "rank": lg["rank_all"],
                    "ci": [lg["ci95_lo"], lg["ci95_hi"]],
                    "first": lg["mean_abs_first"],
                    "second": lg["mean_abs_second"],
                    "rho": lg["direction"],
                    "rho_first": lg["direction_first"],
                    "rho_second": lg["direction_second"],
                    "stable": bool(lg["stable"]),
                    "reason": lg["stability_reason"],
                    "shape": SHAPE_KO.get(lg["shape_all"], lg["shape_all"]),
                    "shape_raw": lg["shape_all"],
                    "pdp_range": lg["pdp_range"],
                    "rank_part": part["lightgbm"]["rank"][f]["mean_abs"],
                    "isna_part": part["lightgbm"]["isna"][f]["mean_abs"],
                },
                "ridge": {
                    "shap": rd["mean_abs"],
                    "rank": rd["rank_all"],
                    "ci": [rd["ci95_lo"], rd["ci95_hi"]],
                    "first": rd["mean_abs_first"],
                    "second": rd["mean_abs_second"],
                    "rho": rd["direction"],
                    "rho_first": rd["direction_first"],
                    "rho_second": rd["direction_second"],
                    "stable": bool(rd["stable"]),
                    "reason": rd["stability_reason"],
                    "coef": cf["rank_coef"],
                    "isna_coef": cf["isna_coef"],
                    "intercept": cf["intercept"],
                },
                "supp": {"flat": flat, "up": up, "verdict": verdict},
                "pdp": sub,
                "ice": lines,
                "ticks": ticks[f],
                "contrast_lgb": contr["lightgbm"].get(f),
                "contrast_ridge": contr["ridge"].get(f),
            }
        )
        cl = cardline.get(f)
        if cl:
            ok = (
                cl["lr"] == lg["rank_all"]
                and cl["rr"] == rd["rank_all"]
                and (cl["lst"] == "안정") == bool(lg["stable"])
                and (cl["rst"] == "안정") == bool(rd["stable"])
                and cl["lsupp"].strip() == verdict
            )
            if not ok:
                log.append(
                    f"MISMATCH card {f}: {cl} vs L{lg['rank_all']}/{lg['stable']} "
                    f"R{rd['rank_all']}/{rd['stable']} supp {verdict}"
                )
        else:
            log.append(f"no card line {f}")
        for model, row in (("lightgbm", lg), ("ridge", rd)):
            c = contr[model].get(f)
            if not c:
                log.append(f"no md row {model} {f}")
            elif c["rank"] != row["rank_all"]:
                log.append(f"MISMATCH md rank {model} {f}")
        if f in supp_md:
            if supp_md[f]["flat"] != flat or supp_md[f]["verdict"] != verdict:
                log.append(f"MISMATCH supp {f}: md {supp_md[f]} vs calc {flat}/{verdict}")
        else:
            log.append(f"no supp row {f}")

    in_doc = set(feats_all)
    scatter = []
    for f in comb["lightgbm"]:
        lg, rd = comb["lightgbm"][f], comb["ridge"][f]
        scatter.append(
            {
                "id": f,
                "own": f in in_doc,
                "x": lg["mean_abs"],
                "y": rd["mean_abs"],
                "stable_l": bool(lg["stable"]),
                "stable_r": bool(rd["stable"]),
                "rank_l": lg["rank_all"],
                "rank_r": rd["rank_all"],
                "rho_l": lg["direction"],
                "rho_r": rd["direction"],
                "shape": SHAPE_KO.get(lg["shape_all"], lg["shape_all"]),
            }
        )
    groups = []
    for gi, g in enumerate(grp["groups"]):
        mem = g["members"]
        if not any(m in in_doc for m in mem):
            continue
        edges = [
            {"a": p["a"], "b": p["b"], "rho": round(p["rho"], 2)}
            for p in grp["pairs_ge_threshold"]
            if p["a"] in mem and p["b"] in mem
        ]
        groups.append(
            {
                "id": f"g{gi + 1}",
                "members": [
                    {"id": m, "shap": comb["lightgbm"][m]["mean_abs"], "own": m in in_doc}
                    for m in mem
                ],
                "edges": edges,
                "sum_l": g["mean_abs_group_sum"]["lightgbm"],
                "mem_l": g["sum_member_mean_abs"]["lightgbm"],
                "sum_r": g["mean_abs_group_sum"]["ridge"],
                "mem_r": g["sum_member_mean_abs"]["ridge"],
            }
        )
    ingroup = {m for g in grp["groups"] for m in g["members"]}
    singles = [f for f in feats_all if f not in ingroup]

    examples = _examples_us(cfg, panel, ref, log)
    data = {
        "market": "us",
        "meta": {
            "run_id": man["run_id"],
            "commit": man["modeler"]["head"][:7],
            "created_utc": man["created_at"],
            "ref_date": cfg.REF_DATE,
            "dates": man["dates"],
            "n_rows": man["n_rows"],
            "n_features_total": man["n_features"],
            "base_l": man.get("lightgbm_base_value"),
            "base_r": man["ridge_base_value"],
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


def _examples_us(cfg, panel: pl.DataFrame, ref: dt.date, log: list[str]) -> dict:
    ohl = (
        scan_parquet(cfg.PRICES)
        .filter((pl.col("date") <= ref) & (pl.col("date") >= dt.date(ref.year - 2, 1, 1)))
        .select(
            "date",
            "symbol",
            pl.col("close").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
        )
        .collect()
    )
    assert ohl["date"].max() <= ref
    meta = read_parquet(cfg.COMPANY_META).select("cik", "name").unique("cik")
    cnt = ohl.group_by("symbol").agg(
        pl.len().alias("nrows"),
        pl.col("close").pct_change().abs().max().alias("mx"),
        pl.col("date").max().alias("last"),
    )
    px = panel.join(cnt, on="symbol", how="left").join(meta, on="cik", how="left")
    windows = getattr(cfg, "EX_WINDOWS", {})
    examples: dict[str, list[dict]] = {}
    for f, specs in getattr(cfg, "EXAMPLES", {}).items():
        out = []
        y = px.filter(pl.col(f).is_not_null())
        nn = y.height
        y = y.with_columns((pl.col(f).rank("average") / nn).alias("pct"))
        for sp in specs:
            if sp.get("ticker"):
                cand = y.filter(pl.col("symbol") == sp["ticker"])
                row = cand.row(0, named=True) if cand.height else None
            else:
                base = y.filter(
                    (pl.col("nrows") >= cfg.EX_DAYS + 1)
                    & (pl.col("mx") < 0.4)
                    & (pl.col("last") == ref)
                ).with_columns(pl.col("adv_20d").alias("size"))
                target = 0.95 if sp["kind"] == "p95" else 0.05
                row, fallback = pick_example(base, target)
                if fallback:
                    log.append(
                        f"EX {f} {sp['kind']}: 목표 백분위(동점·결측 많음) 근처 후보 없음 "
                        "-> 가장 가까운 종목 사용"
                    )
            if row is None:
                log.append(f"EX {f} {sp['kind']}: 후보 없음, 건너뜀")
                continue
            t = ohl.filter(pl.col("symbol") == row["symbol"]).sort("date").tail(cfg.EX_DAYS)
            assert t["date"].max() <= ref
            cl = t["close"].to_numpy()
            log.append(
                f"EX {f} {sp['kind']} {row['symbol']} {row['name']} pct={row['pct']:.4f} "
                f"value={row[f]:.6g} n={nn} last={t['date'].max()}"
            )
            out.append(
                {
                    "feature": f,
                    "kind": sp["kind"],
                    "ticker": row["symbol"],
                    "name": row["name"] or row["symbol"],
                    "market": "US",
                    "value": row[f],
                    "pct": row["pct"],
                    "n_market": nn,
                    "dates": [str(x) for x in t["date"].to_list()],
                    "close": [float(v) for v in cl],
                    "norm": [round(float(v / cl[0] * 100), 2) for v in cl],
                    "halted": [False] * len(cl),
                    "close_last": float(cl[-1]),
                    "window": windows.get(f, []),
                    "unit": "$",
                }
            )
        if out:
            examples[f] = out
    return examples
