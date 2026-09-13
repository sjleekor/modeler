"""walk_forward_splits — purged, embargoed expanding-window CV (etl_00 §5).

The label looks H trading days into the future, so a naive train/valid boundary
leaks: training rows near the boundary carry labels that overlap the validation
window. Two guards (etl_00 §5):

  - embargo: leave an ``embargo`` trading-day gap between train end and valid
    start (default = label horizon, 20).
  - purge: drop training rows whose label horizon reaches into the embargo/valid
    region (the last ``purge`` trading days of train), since their labels peek
    forward. ``purge`` defaults to the horizon as well.

Splitting is on the ordered list of trading dates (a market calendar), so gaps
are counted in sessions, not calendar days — consistent with the d_idx label
logic. Expanding window: each fold trains on everything up to its boundary.

A final ``holdout`` fold (role="holdout") is reserved for a single post-selection
evaluation (etl_00 §5) and excluded from the walk-forward folds.

See ``etl_00`` §5, ``00_shared`` §3.3, and ``etl_03_implementation_plan.md`` §4 (P5).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """One CV fold over trading-date index positions (inclusive ranges).

    Dates are the actual ``date`` objects (or comparable keys) at each boundary;
    ``*_start_idx``/``*_end_idx`` are positions into the sorted date list for
    deterministic slicing. ``role`` is ``"fold"`` or ``"holdout"``.
    """

    fold_id: int
    role: str
    train_start: object
    train_end: object
    valid_start: object
    valid_end: object
    train_start_idx: int
    train_end_idx: int
    valid_start_idx: int
    valid_end_idx: int

    def as_record(self) -> dict:
        """Flat dict for parquet/JSON (split_folds artifact, etl_00 §7)."""
        return {
            "fold_id": self.fold_id,
            "role": self.role,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "valid_start": self.valid_start,
            "valid_end": self.valid_end,
        }


def walk_forward_splits(
    dates: Sequence,
    *,
    horizon: int = 20,
    embargo: int | None = None,
    purge: int | None = None,
    n_folds: int = 5,
    valid_len: int | None = None,
    holdout_len: int = 0,
) -> list[Fold]:
    """Build purged, embargoed expanding-window folds over ``dates``.

    Parameters
    ----------
    dates: sorted unique trading dates (ascending). Duplicates/order are the
        caller's responsibility; pass ``sorted(set(...))``.
    horizon: label horizon in sessions; default for embargo/purge.
    embargo: gap (sessions) between train end and valid start. Defaults to
        ``horizon``.
    purge: trailing train sessions dropped (labels peek into embargo/valid).
        Defaults to ``horizon``.
    n_folds: number of expanding walk-forward folds.
    valid_len: validation length (sessions) per fold. Defaults to an even split
        of the post-warmup region across folds.
    holdout_len: trailing sessions reserved as a single holdout fold (role=
        "holdout"); 0 disables.

    Returns a list of :class:`Fold`. The purge is expressed by trimming
    ``train_end_idx`` to ``boundary - purge`` (the kept train end). Raises
    ``ValueError`` if there are too few dates to honor the gaps.
    """
    embargo = horizon if embargo is None else embargo
    purge = horizon if purge is None else purge
    if embargo < 0 or purge < 0:
        raise ValueError("embargo and purge must be >= 0")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    n = len(dates)
    usable = n - holdout_len
    if usable <= embargo + purge + n_folds:
        raise ValueError(
            f"too few dates ({n}) for n_folds={n_folds}, embargo={embargo}, "
            f"purge={purge}, holdout_len={holdout_len}"
        )

    # The walk-forward region spans [0, usable). Reserve room so the first fold
    # has a non-empty train after purge. Validation blocks tile the tail.
    if valid_len is None:
        # leave ~half the region growing the train; tile the rest as valid.
        valid_region = max(n_folds, usable // 2)
        valid_len = max(1, valid_region // n_folds)

    folds: list[Fold] = []
    # valid_end for the last fold is usable-1; walk backwards in valid_len blocks.
    for k in range(n_folds):
        valid_end_idx = usable - 1 - (n_folds - 1 - k) * valid_len
        valid_start_idx = valid_end_idx - valid_len + 1
        boundary = valid_start_idx - embargo  # first index NOT in train (exclusive end is boundary)
        train_end_idx = boundary - 1 - purge  # purge trailing train labels
        train_start_idx = 0
        if train_end_idx < train_start_idx or valid_start_idx < 0:
            # not enough room for this (early) fold; skip it.
            continue
        folds.append(
            Fold(
                fold_id=len(folds) + 1,
                role="fold",
                train_start=dates[train_start_idx],
                train_end=dates[train_end_idx],
                valid_start=dates[valid_start_idx],
                valid_end=dates[valid_end_idx],
                train_start_idx=train_start_idx,
                train_end_idx=train_end_idx,
                valid_start_idx=valid_start_idx,
                valid_end_idx=valid_end_idx,
            )
        )

    if not folds:
        raise ValueError("no valid folds produced; relax gaps or add more dates")

    if holdout_len > 0:
        h_start = usable
        h_end = n - 1
        folds.append(
            Fold(
                fold_id=len(folds) + 1,
                role="holdout",
                train_start=dates[0],
                train_end=dates[usable - 1 - purge],
                valid_start=dates[h_start],
                valid_end=dates[h_end],
                train_start_idx=0,
                train_end_idx=usable - 1 - purge,
                valid_start_idx=h_start,
                valid_end_idx=h_end,
            )
        )

    return folds
