"""Pure helpers shared by the KR and US data builders."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SUPP_EPS = 1e-12
SUPP_UP = 0.8
SUPP_DOWN = 0.2


def strip_md(text: str) -> str:
    """Drop markdown escapes and code ticks from a cell copied out of a card document."""
    return text.replace("\\~", "~").replace("`", "").strip()


def unescape_md(text: str) -> str:
    """Drop only the escaped tilde; code ticks stay so the page can show them as code."""
    return text.replace("\\~", "~").strip()


def table_cells(line: str) -> list[str]:
    """Cells of one markdown table row, with escapes and backticks removed."""
    return [strip_md(c) for c in line.strip().strip("|").split("|")]


def supp_shape(values: list[float]) -> tuple[int, float | None, str]:
    """Supplementary monotonicity of a PDP curve (result document, section "1.1").

    Only the steps that changed are counted: share of increases >= 0.8 means "증가",
    <= 0.2 "감소", otherwise "단조 아님"; no change at all is "변화 없음".
    Returns ``(unchanged steps, share of increases among changed steps, verdict)``.
    """
    diffs = np.diff(np.asarray(values, dtype=float))
    changed = np.abs(diffs) > SUPP_EPS
    flat = int((~changed).sum())
    if not changed.any():
        return flat, None, "변화 없음"
    up = float((diffs[changed] > 0).mean())
    verdict = "증가" if up >= SUPP_UP else "감소" if up <= SUPP_DOWN else "단조 아님"
    return flat, up, verdict


def clean_json(obj: Any) -> Any:
    """Replace NaN/inf by None recursively so the page JSON stays valid."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    return obj


def quantiles(values: np.ndarray, levels: list[float]) -> list[float | None]:
    x = values[~np.isnan(values)]
    return [float(np.quantile(x, q)) if len(x) else None for q in levels]


def pick_example(candidates, target: float, tol_steps=(0.005, 0.01)):
    """Pick the example row nearest to a target percentile.

    ``candidates`` is a polars frame with columns ``pct`` and ``size`` (already filtered by the
    history rules). Rows within the tolerance win, the largest ``size`` first; if none is that
    close the single nearest row is used. Returns ``(row or None, used_nearest_fallback)``.
    """
    if candidates.height == 0:
        return None, False
    for tol in tol_steps:
        near = candidates.filter((candidates["pct"] - target).abs() <= tol)
        if near.height:
            return near.sort("size", descending=True).row(0, named=True), False
    nearest = candidates.with_columns((candidates["pct"] - target).abs().alias("_d"))
    nearest = nearest.sort(["_d", "size"], descending=[False, True])
    return nearest.row(0, named=True), True
