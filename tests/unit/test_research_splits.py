"""Unit tests for P5 — walk_forward_splits (purged, embargoed, expanding)."""

from __future__ import annotations

import datetime

import pytest
from modeler.etl.splits import walk_forward_splits


def _dates(n: int) -> list[datetime.date]:
    base = datetime.date(2020, 1, 1)
    return [base + datetime.timedelta(days=i) for i in range(n)]


def test_embargo_and_purge_gap_between_train_and_valid() -> None:
    dates = _dates(300)
    folds = walk_forward_splits(dates, horizon=20, embargo=20, purge=20, n_folds=3)
    assert folds
    for f in folds:
        # valid starts strictly after train ends, with an embargo+purge gap.
        gap = f.valid_start_idx - f.train_end_idx
        assert gap >= 20 + 1  # at least embargo sessions between
        # purge removed trailing train labels: train_end < boundary - purge
        assert f.train_end_idx < f.valid_start_idx - 20


def test_expanding_window_train_starts_at_zero() -> None:
    dates = _dates(300)
    folds = walk_forward_splits(dates, n_folds=4)
    for f in folds:
        if f.role == "fold":
            assert f.train_start_idx == 0  # expanding: always from the start


def test_folds_are_time_ordered() -> None:
    dates = _dates(400)
    folds = [f for f in walk_forward_splits(dates, n_folds=5) if f.role == "fold"]
    valid_starts = [f.valid_start_idx for f in folds]
    assert valid_starts == sorted(valid_starts)  # later folds validate later


def test_holdout_fold_is_trailing_and_labeled() -> None:
    dates = _dates(400)
    folds = walk_forward_splits(dates, n_folds=3, holdout_len=40)
    holdouts = [f for f in folds if f.role == "holdout"]
    assert len(holdouts) == 1
    h = holdouts[0]
    assert h.valid_end_idx == len(dates) - 1  # ends at the very last date
    assert h.valid_start_idx == len(dates) - 40


def test_too_few_dates_raises() -> None:
    with pytest.raises(ValueError):
        walk_forward_splits(_dates(30), horizon=20, n_folds=5)


def test_fold_as_record_has_date_fields() -> None:
    folds = walk_forward_splits(_dates(300), n_folds=2)
    rec = folds[0].as_record()
    assert set(rec) == {
        "fold_id",
        "role",
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
    }
