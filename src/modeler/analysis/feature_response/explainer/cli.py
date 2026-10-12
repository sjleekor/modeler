"""Command line: ``python -m modeler.analysis.feature_response.explainer build kr_price``."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tempfile
from pathlib import Path

from .common import clean_json
from .config import CONFIG_NAMES, load_config, resolve_config
from .data_kr import build_kr
from .data_us import build_us
from .paths import Roots, resolve_roots
from .render import render_html


def default_cache_dir(roots: Roots, market: str) -> Path:
    return roots.stock_data / market / "output" / "feature_response_l3" / "explainer_cache"


def build_one(
    name_or_path: str,
    roots: Roots,
    out: str | Path | None = None,
    cache_dir: str | Path | None = None,
    generated: dt.date | None = None,
) -> tuple[Path, list[str]]:
    """Build one page. Returns the HTML path and the log lines (``MISMATCH`` = trouble)."""
    cfg_mod = load_config(name_or_path)
    cfg = resolve_config(cfg_mod, roots, out)
    name = Path(name_or_path).stem if name_or_path.endswith(".py") else name_or_path
    builder = build_kr if cfg.MARKET == "kr" else build_us
    data, log = builder(cfg)
    cache = Path(cache_dir) if cache_dir else default_cache_dir(roots, cfg.MARKET)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"data_{name}.json").write_text(
        json.dumps(clean_json(data), ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    (cache / f"build_log_{name}.txt").write_text("\n".join(log), encoding="utf-8")
    html_text = render_html(cfg, data, name, roots, generated)
    cfg.OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    cfg.OUT_HTML.write_text(html_text, encoding="utf-8")
    return cfg.OUT_HTML, log


def _problems(log: list[str]) -> list[str]:
    marks = ("MISMATCH", "no card line", "no md row", "no supp row")
    return [line for line in log if line.startswith(marks)]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m modeler.analysis.feature_response.explainer",
        description="Build the feature explainer HTML pages. Labels are never read.",
    )
    p.add_argument("--stock-data-root", help="default: $STOCK_DATA_ROOT or ../stock_data")
    p.add_argument("--my-root", help="default: $MY_ROOT or ../my")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build one page")
    b.add_argument("config", help=f"bundled name ({', '.join(CONFIG_NAMES)}) or a .py path")
    b.add_argument("--out", help="HTML path (default: <MY_ROOT>/<DOC_DIR>/<OUT_NAME>)")
    b.add_argument("--cache-dir", help="where data_<name>.json and the log go")
    a = sub.add_parser("build-all", help="build the six bundled pages")
    a.add_argument("--out-dir", help="write all HTML files here instead of next to the cards")
    a.add_argument("--cache-dir", help="where data_<name>.json and the logs go")
    sub.add_parser("list", help="list bundled config names")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "list":
        print("\n".join(CONFIG_NAMES))
        return 0
    roots = resolve_roots(args.stock_data_root, args.my_root)
    names = [args.config] if args.cmd == "build" else list(CONFIG_NAMES)
    cache = args.cache_dir or None
    rc = 0
    for name in names:
        out = args.out if args.cmd == "build" else None
        if args.cmd == "build-all" and args.out_dir:
            cfg = resolve_config(load_config(name), roots)
            out = Path(args.out_dir) / cfg.OUT_NAME
        path, log = build_one(name, roots, out=out, cache_dir=cache)
        bad = _problems(log)
        print(f"{name}: wrote {path} ({path.stat().st_size:,} bytes), problems={len(bad)}")
        for line in bad:
            print(f"  {line}", file=sys.stderr)
        rc = 1 if bad else rc
    return rc


def tmp_cache() -> str:
    """A throw-away cache directory for tests and dry runs."""
    return tempfile.mkdtemp(prefix="explainer_cache_")
