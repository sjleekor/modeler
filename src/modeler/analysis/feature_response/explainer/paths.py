"""Roots, path resolution and the label guard.

Nothing here hard-codes a machine path. ``STOCK_DATA_ROOT`` (default ``../stock_data``) holds
the data, ``MY_ROOT`` (default ``../my``) holds the card documents and the HTML output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

_LABEL_MARK = "label"


class LabelAccessError(RuntimeError):
    """Raised when a code path tries to open a label file. This tool never reads labels."""


@dataclass(frozen=True)
class Roots:
    stock_data: Path
    my: Path

    def rel(self, path: Path | str) -> str:
        """Short, machine independent name of a path for the page footer."""
        p = Path(path)
        for base, prefix in ((self.stock_data, "stock_data"), (self.my, "my")):
            try:
                return f"{prefix}/{p.relative_to(base).as_posix()}"
            except ValueError:
                continue
        return p.as_posix()


def resolve_roots(stock_data: str | Path | None = None, my: str | Path | None = None) -> Roots:
    sd = stock_data or os.environ.get("STOCK_DATA_ROOT", "../stock_data")
    my_root = my or os.environ.get("MY_ROOT", "../my")
    return Roots(stock_data=Path(sd).resolve(), my=Path(my_root).resolve())


def guard_not_label(path: str | Path) -> Path:
    """Refuse any path with a component that names a label file or directory."""
    p = Path(path)
    for part in p.parts:
        if _LABEL_MARK in part.lower():
            raise LabelAccessError(f"label path is off limits: {p}")
    return p


def read_parquet(path: str | Path) -> pl.DataFrame:
    return pl.read_parquet(guard_not_label(path))


def scan_parquet(path: str | Path, **kwargs) -> pl.LazyFrame:
    return pl.scan_parquet(guard_not_label(path), **kwargs)


def latest_match(root: Path, pattern: str) -> Path:
    """Last (= newest snapshot_date) path matching ``pattern`` under ``root``."""
    hits = sorted(root.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"no file matches {pattern} under {root}")
    return guard_not_label(hits[-1])
