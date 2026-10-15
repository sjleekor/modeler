"""Hard guards. They raise; nothing here only warns."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from .config import DATA_END, FORMATION_END, FORMATION_START, LABEL_LIKE_PREFIXES


class HoldoutAccessError(RuntimeError):
    """A path names the holdout. This tool reads the development window only."""


class DateGuardError(RuntimeError):
    """A date lies outside the frozen window."""


class OutcomeLeakError(RuntimeError):
    """A selection function was handed outcome columns."""


class RepoWriteError(RuntimeError):
    """An output path points inside the code repository."""


def guard_path(path: str | Path) -> Path:
    """Refuse any path that mentions the holdout (case insensitive, any component)."""
    p = Path(path)
    if "holdout" in str(p).lower():
        raise HoldoutAccessError(f"holdout path is off limits: {p}")
    return p


def guard_not_in_repo(path: str | Path, repo_root: Path | None) -> Path:
    p = Path(path).resolve()
    if repo_root is not None:
        root = repo_root.resolve()
        if p == root or root in p.parents:
            raise RepoWriteError(f"refusing to write inside the code repository: {p}")
    return p


def check_formation_date(d: dt.date) -> dt.date:
    if not (FORMATION_START <= d <= FORMATION_END):
        raise DateGuardError(f"formation date {d} outside {FORMATION_START}..{FORMATION_END}")
    return d


def check_label_end(end: dt.date) -> dt.date:
    if end > DATA_END:
        raise DateGuardError(f"label window ends {end}, after {DATA_END}")
    return end


def assert_outcome_blind(columns: list[str]) -> None:
    bad = [c for c in columns if c.startswith(LABEL_LIKE_PREFIXES)]
    if bad:
        raise OutcomeLeakError(f"selection received outcome columns: {bad}")
