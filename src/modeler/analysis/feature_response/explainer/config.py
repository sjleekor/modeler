"""Config loading, path resolution and validation.

A config is a plain python module (see ``configs/``). It names things with *relative* names:
run id, dataset paths below ``STOCK_DATA_ROOT``, document names below ``MY_ROOT/DOC_DIR``.
``resolve_config`` turns them into the absolute attributes the builders read.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

from .paths import Roots, latest_match

CONFIG_NAMES = (
    "kr_price",
    "kr_flow",
    "kr_fin_event",
    "us_price_liquidity",
    "us_fundamental_event",
    "us_flow_holders",
)
_PKG = __package__ or "modeler.analysis.feature_response.explainer"

COMMON_FIELDS = (
    "MARKET",
    "DOC_DIR",
    "OUT_NAME",
    "DOC_MD",
    "RESULT_MD",
    "RUN_ID",
    "REF_DATE",
    "PAGE_TITLE",
    "N_ICE",
    "EX_DAYS",
    "THEMES",
    "CARDS",
    "FMT",
    "BASIS",
)
KR_FIELDS = ("PREFIX", "EX_MARKET", "PANEL_REL", "RAW_REL", "MODELER_COMMIT")
US_FIELDS = ("PANEL_V2_REL", "PANEL_FLOW_REL", "PRICES_GLOB", "COMPANY_META_GLOB")
PRINCIPLE_TYPES = ("flow", "timeline", "ratio")


class ConfigError(ValueError):
    pass


def load_config(name_or_path: str) -> ModuleType:
    """Load a bundled config by name (``kr_price``) or any ``.py`` file by path."""
    path = Path(name_or_path)
    if path.suffix == ".py" and path.exists():
        spec = importlib.util.spec_from_file_location(f"_explainer_cfg_{path.stem}", path)
        if spec is None or spec.loader is None:
            raise ConfigError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    if name_or_path not in CONFIG_NAMES:
        raise ConfigError(f"unknown config {name_or_path!r}; bundled: {', '.join(CONFIG_NAMES)}")
    return importlib.import_module(f"{_PKG}.configs.{name_or_path}")


def validate_principle(feature: str, spec: dict[str, Any]) -> list[str]:
    """Problems of one ``principle`` block (empty list = fine)."""
    kind = spec.get("type")
    if kind not in PRINCIPLE_TYPES:
        return [f"{feature}: principle.type must be one of {PRINCIPLE_TYPES}, got {kind!r}"]
    bad: list[str] = []
    if kind == "flow":
        steps = spec.get("steps") or []
        if not 2 <= len(steps) <= 5:
            bad.append(f"{feature}: flow needs 2-5 steps, got {len(steps)}")
        bad += [
            f"{feature}: flow step {i + 1} lacks title"
            for i, s in enumerate(steps)
            if not s.get("title")
        ]
    elif kind == "timeline":
        events = spec.get("events") or []
        if len(events) < 2:
            bad.append(f"{feature}: timeline needs at least 2 events")
        bad += [
            f"{feature}: timeline event {i + 1} lacks at/label"
            for i, e in enumerate(events)
            if "at" not in e or not e.get("label")
        ]
    else:
        rows = spec.get("rows") or []
        if not rows:
            bad.append(f"{feature}: ratio needs rows")
        for i, r in enumerate(rows):
            if not r.get("num") or not r.get("den"):
                bad.append(f"{feature}: ratio row {i + 1} needs non-zero num and den")
    return bad


def validate_config(cfg: ModuleType | SimpleNamespace) -> list[str]:
    """All problems found; an empty list means the config can be built."""
    bad = [f"missing field {k}" for k in COMMON_FIELDS if not hasattr(cfg, k)]
    market = getattr(cfg, "MARKET", None)
    if market not in ("kr", "us"):
        bad.append(f"MARKET must be 'kr' or 'us', got {market!r}")
    extra = KR_FIELDS if market == "kr" else US_FIELDS if market == "us" else ()
    bad += [f"missing {market} field {k}" for k in extra if not hasattr(cfg, k)]
    if bad:
        return bad
    feats = [f for _, _, fs in cfg.THEMES for f in fs]
    if len(set(feats)) != len(feats):
        bad.append("THEMES lists a feature twice")
    bad += [f"CARDS lacks {f}" for f in feats if f not in cfg.CARDS]
    bad += [f"FMT lacks {f}" for f in feats if f not in cfg.FMT]
    for f in feats:
        card = cfg.CARDS.get(f, {})
        bad += [
            f"{f}: card lacks {k}"
            for k in ("title", "basis", "intuition", "win", "what", "model", "caution")
            if k not in card
        ]
        if card.get("basis") not in cfg.BASIS:
            bad.append(f"{f}: basis {card.get('basis')!r} is not a BASIS key")
        if card.get("principle"):
            bad += validate_principle(f, card["principle"])
    bad += [
        f"EXAMPLES names unknown feature {f}"
        for f in getattr(cfg, "EXAMPLES", {})
        if f not in feats
    ]
    return bad


def resolve_config(cfg: ModuleType, roots: Roots, out: str | Path | None = None) -> SimpleNamespace:
    """Copy the config and add the absolute paths the builders read."""
    problems = validate_config(cfg)
    if problems:
        raise ConfigError("; ".join(problems))
    ns = SimpleNamespace(**{k: v for k, v in vars(cfg).items() if not k.startswith("_")})
    sd, my = roots.stock_data, roots.my
    doc_dir = my / ns.DOC_DIR
    ns.MD_CARDS = doc_dir / ns.DOC_MD
    ns.MD_RESULT = doc_dir / ns.RESULT_MD
    ns.OUT_HTML = Path(out) if out else doc_dir / ns.OUT_NAME
    ns.RUN_DIR = sd / ns.MARKET / "output" / "feature_response_l3" / ns.RUN_ID
    if ns.MARKET == "kr":
        ns.PANEL = sd / ns.PANEL_REL
        ns.RAW = sd / ns.RAW_REL
    else:
        ns.PANEL_V2 = sd / ns.PANEL_V2_REL
        ns.PANEL_FLOW = sd / ns.PANEL_FLOW_REL
        ns.PRICES = latest_match(sd, ns.PRICES_GLOB)
        ns.COMPANY_META = latest_match(sd, ns.COMPANY_META_GLOB)
    return ns
