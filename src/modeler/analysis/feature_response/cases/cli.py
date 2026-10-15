"""Command line: ``python -m modeler.analysis.feature_response.cases run``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

from . import config as C
from .guards import guard_path
from .pipeline import repo_root_of_code, run_pilot
from .render import render_feature_page, render_index

MY_SUBDIR = "milestones/common/20261012_feature_explainer/cases"


def _roots(args) -> tuple[Path, Path]:
    import os

    sd = Path(args.stock_data_root or os.environ.get("STOCK_DATA_ROOT", "../stock_data")).resolve()
    my = Path(args.my_root or os.environ.get("MY_ROOT", "../my")).resolve()
    return sd, my


def cmd_run(args) -> int:
    sd, my = _roots(args)
    features = args.features or list(C.DEFAULT_FEATURES)
    unknown = [f for f in features if f not in C.FEATURES]
    if unknown:
        print(f"unknown features: {unknown}", file=sys.stderr)
        return 2
    run_id = args.run_id or f"kr_cases_20261015_seed{args.seed}"
    out_root = Path(args.out_root) if args.out_root else sd / "kr" / "output" / "feature_cases"
    guard_path(out_root)
    html_dir = Path(args.html_dir) if args.html_dir else my / MY_SUBDIR
    out_dir = out_root / run_id
    res = run_pilot(sd, out_dir, features, seed=args.seed, run_id=run_id, log=print)
    from .guards import guard_not_in_repo

    guard_not_in_repo(html_dir, repo_root_of_code())
    html_dir.mkdir(parents=True, exist_ok=True)
    cases, dist = res["cases"], res["dist"]
    counts = {}
    for f in features:
        spec = C.FEATURES[f]
        c = cases.filter(pl.col("feature") == f)
        counts[f] = {
            "random": int(c.filter(pl.col("kind") == "random").height),
            "extreme": int(c.filter(pl.col("kind") != "random").height),
        }
        page = render_feature_page(
            spec,
            c,
            dist.filter(pl.col("feature") == f),
            res["series"],
            res["dates"],
            res["n_grid"],
            run_id,
            args.seed,
        )
        (html_dir / f"{f}.html").write_text(page, encoding="utf-8")
    index = render_index(
        [C.FEATURES[f] for f in features], counts, res["dates"], res["n_grid"], run_id, args.seed
    )
    (html_dir / "index.html").write_text(index, encoding="utf-8")
    print(f"run_id {run_id}\noutput {out_dir}\nhtml {html_dir}")
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m modeler.analysis.feature_response.cases",
        description="KR feature case gallery pilot (development window only, no holdout).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="select cases, compute distributions, write tables and HTML")
    r.add_argument("--features", nargs="+", help="default: all 9 pilot features")
    r.add_argument("--seed", type=int, default=C.SEED)
    r.add_argument("--run-id")
    r.add_argument("--out-root", help="default: <STOCK_DATA_ROOT>/kr/output/feature_cases")
    r.add_argument("--html-dir", help=f"default: <MY_ROOT>/{MY_SUBDIR}")
    r.add_argument("--stock-data-root", help="default: $STOCK_DATA_ROOT or ../stock_data")
    r.add_argument("--my-root", help="default: $MY_ROOT or ../my")
    r.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.func(args)
