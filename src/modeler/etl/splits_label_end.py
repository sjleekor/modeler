"""Walk-forward folds purged by the *actual label end date* — quality-fix plan §9.3.

``etl/splits.py`` (``walk_forward_splits``) purges the last ``purge`` (= h)
sessions of train, which assumes a label ends h sessions after the row. That is
true for ``labels.py`` (c0) and for the ``regular`` kind of ``labels_entry.py``
(c1/o1), but not for the other valid kinds: a ``late`` exit moves to the h-th
*non-halt* session of the ticker, and a ``terminal`` / ``terminal_halted`` exit
is the series' last row, which can sit hundreds of sessions after t when a ticker
is halted for a long time before delisting. With the fixed purge such a row stays
in train although its label (the delisting value) is only known *after* the
validation window starts. The fixed purge absorbs the delay up to about h
sessions only (``splits.py:92-93,119-120``).

Rule here (plan §9.3, "학습 purge는 ``label_end_date`` 기준")

  ``boundary`` = the market session ``embargo`` sessions before ``valid_start``.
  A row is in the TRAIN set of a fold iff

    * its date is before ``boundary``, and
    * its ``label_end_date`` is not NULL and is before ``boundary``.

  A NULL ``label_end_date`` means there is no label (``censored`` / ``untradable``
  kinds), so the row cannot be trained on. ``label_end_date`` already is a trading
  date, so no calendar lookup is needed to apply the cut; the calendar only fixes
  the ``boundary`` itself (embargo is counted in sessions, as in ``splits.py``).

Equivalence with ``splits.py``. If every label ends exactly h sessions after its
row (``label_end = t + h`` sessions), the rule keeps ``t <= boundary - 1 - h`` —
the same rows as ``walk_forward_splits(..., purge=h)`` (``train_end_idx =
valid_start_idx - embargo - 1 - purge``). Fold boundaries, ``n_folds``,
``valid_len`` and the skipped early folds are identical (pinned by tests). The
holdout fold follows ``splits.py`` too: no embargo, ``boundary = valid_start``.

What is NOT a ``Fold``. ``splits.Fold`` carries ``train_end`` (the last kept
train date). Here the last kept date depends on the data, so :class:`LabelEndFold`
has ``boundary`` instead and no ``train_end``: slicing ``date <= something`` would
silently drop the purge, so the attribute does not exist. Always go through
:func:`fold_masks` / :func:`purged_train_mask` / :func:`purged_train_expr`.

Only train is purged. Validation rows are the dates ``valid_start..valid_end``;
which of them have a usable label is the label gate's job
(``etl/label_gate.py``).

``dates`` / ``calendar`` is one market calendar (sorted unique trading dates), as
for ``walk_forward_splits``. This module is new and imports nothing from the
model02 code-hash files (``splits.py``, ``universe.py``, ``labels.py`` ...).
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import polars as pl


def _as_dates(calendar: Sequence[date] | pl.Series, name: str) -> list[date]:
    """Calendar as a list of dates; strictly increasing or ``ValueError``."""
    out = calendar.to_list() if isinstance(calendar, pl.Series) else list(calendar)
    if any(b <= a for a, b in zip(out, out[1:], strict=False)):
        raise ValueError(f"{name} must be sorted ascending without duplicates")
    return out


def embargo_boundary(
    calendar: Sequence[date] | pl.Series, valid_start: date, embargo_sessions: int
) -> date:
    """First date of the embargo/validation boundary: ``valid_start`` minus
    ``embargo_sessions`` sessions of ``calendar``.

    ``valid_start`` must be a session of ``calendar`` (fail closed on anything
    else) and the calendar must reach back far enough.
    """
    if embargo_sessions < 0:
        raise ValueError("embargo_sessions must be >= 0")
    cal = _as_dates(calendar, "calendar")
    i = bisect_left(cal, valid_start)
    if i >= len(cal) or cal[i] != valid_start:
        raise ValueError(f"valid_start {valid_start} is not a session of the calendar")
    if i - embargo_sessions < 0:
        raise ValueError(
            f"calendar has only {i} sessions before valid_start {valid_start}; "
            f"embargo_sessions={embargo_sessions} reaches before its first date"
        )
    return cal[i - embargo_sessions]


def purged_train_expr(
    *,
    date_col: str,
    end_col: str,
    boundary: date,
    train_start: date | None = None,
) -> pl.Expr:
    """Boolean expression: row may be trained on for a fold with this ``boundary``.

    ``date < boundary AND end IS NOT NULL AND end < boundary`` (and
    ``date >= train_start`` when given). Never NULL — a NULL date counts as not
    kept. Works on ``DataFrame`` and ``LazyFrame``.
    """
    d = pl.col(date_col)
    e = pl.col(end_col)
    expr = d.lt(boundary) & e.is_not_null() & e.lt(boundary)
    if train_start is not None:
        expr = expr & d.ge(train_start)
    return expr.fill_null(False)


def _require_frame(frame: object) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(
            "frame must be a polars DataFrame; for a LazyFrame use purged_train_expr() "
            "inside .filter()/.with_columns()"
        )
    return frame


def purged_train_mask(
    frame: pl.DataFrame,
    *,
    date_col: str,
    end_col: str,
    valid_start: date,
    embargo_sessions: int,
    calendar: Sequence[date] | pl.Series,
    train_start: date | None = None,
) -> pl.Series:
    """Boolean mask (aligned with ``frame``) of train rows for a validation window
    starting at ``valid_start``.

    ``boundary = embargo_boundary(calendar, valid_start, embargo_sessions)``; a row
    is kept iff its date and its ``end_col`` (label end date) are both before the
    boundary and ``end_col`` is not NULL. See the module docstring for why.
    """
    boundary = embargo_boundary(calendar, valid_start, embargo_sessions)
    expr = purged_train_expr(
        date_col=date_col, end_col=end_col, boundary=boundary, train_start=train_start
    )
    return _require_frame(frame).select(expr.alias("train")).to_series()


@dataclass(frozen=True)
class LabelEndFold:
    """One fold over trading-date index positions (inclusive ranges).

    Mirrors ``splits.Fold`` except ``train_end*`` is replaced by ``boundary*``:
    the first embargo date (``valid_start - embargo`` sessions; for the holdout
    fold ``boundary == valid_start``). Train = rows from ``train_start`` whose
    date AND label end date are before ``boundary``. ``embargo`` is the gap that
    was applied (0 for the holdout fold). ``role`` is ``"fold"`` or ``"holdout"``.
    """

    fold_id: int
    role: str
    train_start: object
    boundary: object
    valid_start: object
    valid_end: object
    train_start_idx: int
    boundary_idx: int
    valid_start_idx: int
    valid_end_idx: int
    embargo: int

    def as_record(self) -> dict:
        """Flat dict for parquet/JSON (split_folds artifact, etl_00 §7)."""
        return {
            "fold_id": self.fold_id,
            "role": self.role,
            "train_start": self.train_start,
            "boundary": self.boundary,
            "valid_start": self.valid_start,
            "valid_end": self.valid_end,
            "embargo": self.embargo,
        }


def walk_forward_label_end_splits(
    dates: Sequence[date] | pl.Series,
    *,
    horizon: int = 20,
    embargo: int | None = None,
    n_folds: int = 5,
    valid_len: int | None = None,
    holdout_len: int = 0,
    min_train_sessions: int | None = None,
) -> list[LabelEndFold]:
    """Expanding-window folds whose train purge is by label end date.

    Same arguments and defaults as ``splits.walk_forward_splits`` minus ``purge``:

    dates: sorted unique trading dates of the market calendar.
    horizon: label horizon in sessions. Used for the ``embargo`` default and as
        the shortest label span in the feasibility checks; it never trims rows.
    embargo: gap (sessions) between the train boundary and valid start. Defaults
        to ``horizon``.
    n_folds, valid_len, holdout_len: as in ``walk_forward_splits`` (same
        ``valid_len`` default, same tiling, same trailing holdout fold).
    min_train_sessions: a fold needs at least this many dates before its boundary,
        else it is skipped like ``splits.py`` skips early folds. Default
        ``horizon + 1`` — the fewest dates in which a c0 label (ends t + h) can end
        before the boundary; that default is what makes the fold list identical to
        ``walk_forward_splits(purge=horizon)``. For c1/o1 labels (end t + 1 + h)
        pass ``horizon + 2``.

    Raises ``ValueError`` when there are too few dates (same test as
    ``walk_forward_splits`` with ``purge`` replaced by ``horizon``).
    """
    cal = _as_dates(dates, "dates")
    embargo = horizon if embargo is None else embargo
    min_train = horizon + 1 if min_train_sessions is None else min_train_sessions
    if embargo < 0:
        raise ValueError("embargo must be >= 0")
    if min_train < 1:
        raise ValueError("min_train_sessions must be >= 1")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    n = len(cal)
    usable = n - holdout_len
    if usable <= embargo + horizon + n_folds:
        raise ValueError(
            f"too few dates ({n}) for n_folds={n_folds}, embargo={embargo}, "
            f"horizon={horizon}, holdout_len={holdout_len}"
        )

    if valid_len is None:
        valid_region = max(n_folds, usable // 2)
        valid_len = max(1, valid_region // n_folds)

    folds: list[LabelEndFold] = []
    for k in range(n_folds):
        valid_end_idx = usable - 1 - (n_folds - 1 - k) * valid_len
        valid_start_idx = valid_end_idx - valid_len + 1
        boundary_idx = valid_start_idx - embargo
        if valid_start_idx < 0 or boundary_idx < min_train:
            continue  # not enough room for this (early) fold
        folds.append(
            LabelEndFold(
                fold_id=len(folds) + 1,
                role="fold",
                train_start=cal[0],
                boundary=cal[boundary_idx],
                valid_start=cal[valid_start_idx],
                valid_end=cal[valid_end_idx],
                train_start_idx=0,
                boundary_idx=boundary_idx,
                valid_start_idx=valid_start_idx,
                valid_end_idx=valid_end_idx,
                embargo=embargo,
            )
        )

    if not folds:
        raise ValueError("no valid folds produced; relax gaps or add more dates")

    if holdout_len > 0:
        if usable < min_train:
            raise ValueError(f"holdout fold has only {usable} train sessions (< {min_train})")
        folds.append(
            LabelEndFold(
                fold_id=len(folds) + 1,
                role="holdout",
                train_start=cal[0],
                boundary=cal[usable],
                valid_start=cal[usable],
                valid_end=cal[n - 1],
                train_start_idx=0,
                boundary_idx=usable,
                valid_start_idx=usable,
                valid_end_idx=n - 1,
                embargo=0,
            )
        )

    return folds


@dataclass(frozen=True)
class FoldMasks:
    """Row masks of one fold, aligned with the frame they were built from."""

    fold: LabelEndFold
    train: pl.Series
    valid: pl.Series

    def train_index(self) -> np.ndarray:
        """Row positions of the train rows (``int`` array, frame order)."""
        return self.train.arg_true().to_numpy()

    def valid_index(self) -> np.ndarray:
        """Row positions of the validation rows (``int`` array, frame order)."""
        return self.valid.arg_true().to_numpy()


def fold_masks(
    frame: pl.DataFrame,
    folds: Sequence[LabelEndFold],
    *,
    date_col: str,
    end_col: str,
) -> list[FoldMasks]:
    """Train/valid masks per fold.

    train: :func:`purged_train_expr` with the fold's ``boundary`` / ``train_start``.
    valid: ``valid_start <= date <= valid_end`` (label availability is the label
    gate's job).
    """
    frame = _require_frame(frame)
    out: list[FoldMasks] = []
    for f in folds:
        train = purged_train_expr(
            date_col=date_col, end_col=end_col, boundary=f.boundary, train_start=f.train_start
        )
        d = pl.col(date_col)
        valid = (d.ge(f.valid_start) & d.le(f.valid_end)).fill_null(False)
        t, v = frame.select(train.alias("train"), valid.alias("valid"))
        out.append(FoldMasks(fold=f, train=t, valid=v))
    return out


def purge_counts(
    frame: pl.DataFrame,
    fold: LabelEndFold,
    *,
    date_col: str,
    end_col: str,
    fixed_purge_horizon: int | None = None,
    calendar: Sequence[date] | pl.Series | None = None,
) -> dict[str, int]:
    """What the label-end purge did to one fold, for the evaluation report.

    Counts rows dated in ``[train_start, boundary)``:

    ``candidates``                rows before the boundary
    ``kept``                      train rows
    ``dropped_null_end``          no label end date (no label)
    ``dropped_crosses_boundary``  label ends on/after the boundary
    ``leaked_by_fixed_purge``     (only with ``fixed_purge_horizon`` and ``calendar``)
        of the crossing rows, those a fixed h-session purge would have KEPT
        (date <= ``boundary_idx - 1 - h``): late labels the old split leaked.
    """
    frame = _require_frame(frame)
    d, e = pl.col(date_col), pl.col(end_col)
    cand = d.ge(fold.train_start) & d.lt(fold.boundary)
    cross = cand & e.is_not_null() & e.ge(fold.boundary)
    exprs = [
        cand.sum().alias("candidates"),
        purged_train_expr(
            date_col=date_col, end_col=end_col, boundary=fold.boundary, train_start=fold.train_start
        )
        .sum()
        .alias("kept"),
        (cand & e.is_null()).sum().alias("dropped_null_end"),
        cross.sum().alias("dropped_crosses_boundary"),
    ]
    if fixed_purge_horizon is not None:
        if calendar is None:
            raise ValueError("calendar is required with fixed_purge_horizon")
        cal = _as_dates(calendar, "calendar")
        cut_idx = fold.boundary_idx - 1 - fixed_purge_horizon
        if cut_idx < 0:
            exprs.append(pl.lit(0).alias("leaked_by_fixed_purge"))
        else:
            exprs.append((cross & d.le(cal[cut_idx])).sum().alias("leaked_by_fixed_purge"))
    row = frame.select(exprs).row(0, named=True)
    return {k: int(v or 0) for k, v in row.items()}
