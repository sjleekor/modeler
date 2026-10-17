"""Walk-forward folds purged by label end date (``etl/splits_label_end.py``).

Synthetic polars frames plus one run on real ``labels_entry`` output. The
equivalence tests pin the module to ``splits.walk_forward_splits`` for labels that
end exactly h sessions later.
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from modeler.etl.label_gate import entry_label_columns
from modeler.etl.labels import LabelSpec
from modeler.etl.splits import walk_forward_splits
from modeler.etl.splits_label_end import (
    embargo_boundary,
    fold_masks,
    purge_counts,
    purged_train_expr,
    purged_train_mask,
    walk_forward_label_end_splits,
)

from ._entry_label_fixtures import delisted_after_long_halt, entry_frame, price_con


def _sessions(n: int) -> list[datetime.date]:
    """n weekday sessions from 2020-01-01 (weekends skipped: gaps are sessions, not days)."""
    out: list[datetime.date] = []
    d = datetime.date(2020, 1, 1)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += datetime.timedelta(days=1)
    return out


def _frame(cal: list[datetime.date], rows: list[tuple[int, int | None]]) -> pl.DataFrame:
    """rows: (date index, label end date index or None)."""
    return pl.DataFrame(
        {
            "trade_date": [cal[i] for i, _ in rows],
            "label_end": [None if j is None else cal[j] for _, j in rows],
        },
        schema={"trade_date": pl.Date, "label_end": pl.Date},
    )


# --- boundary and mask ----------------------------------------------------------


def test_embargo_boundary_counts_sessions_not_days() -> None:
    cal = _sessions(60)
    assert embargo_boundary(cal, cal[30], 5) == cal[25]
    assert embargo_boundary(cal, cal[30], 0) == cal[30]
    assert embargo_boundary(pl.Series(cal), cal[30], 5) == cal[25]
    saturday = datetime.date(2020, 1, 4)
    with pytest.raises(ValueError, match="not a session"):
        embargo_boundary(cal, saturday, 1)
    with pytest.raises(ValueError, match="reaches before"):
        embargo_boundary(cal, cal[3], 5)
    with pytest.raises(ValueError):
        embargo_boundary(cal, cal[30], -1)
    with pytest.raises(ValueError, match="sorted"):
        embargo_boundary(list(reversed(cal)), cal[30], 1)


def test_late_label_is_purged_when_it_crosses_the_boundary_and_kept_otherwise() -> None:
    cal = _sessions(200)
    valid_start, embargo = cal[100], 5  # boundary = cal[95]
    frame = _frame(
        cal,
        [
            (10, 30),  # 0 regular, long before the boundary            -> kept
            (60, 80),  # 1 regular                                        -> kept
            (60, 190),  # 2 late: ends 130 sessions after t, crosses    -> purged
            (60, 94),  # 3 late but ends one session before boundary    -> kept
            (60, 95),  # 4 ends ON the boundary                          -> purged
            (94, 114),  # 5 regular but runs into the embargo window    -> purged
            (96, 99),  # 6 dated inside the embargo window               -> purged
            (10, None),  # 7 NULL end: no label                           -> purged
            (105, 125),  # 8 dated in the validation window               -> not train
        ],
    )
    mask = purged_train_mask(
        frame,
        date_col="trade_date",
        end_col="label_end",
        valid_start=valid_start,
        embargo_sessions=embargo,
        calendar=cal,
    )
    assert mask.to_list() == [True, True, False, True, False, False, False, False, False]

    # The same late row is kept by a later fold whose boundary lies past its end.
    later = purged_train_mask(
        frame,
        date_col="trade_date",
        end_col="label_end",
        valid_start=cal[199],
        embargo_sessions=embargo,  # boundary = cal[194] > cal[190]
        calendar=cal,
    )
    assert later.to_list()[2] is True
    assert later.to_list()[7] is False  # NULL end is dropped in every fold


def test_fixed_purge_keeps_the_late_row_that_label_end_purge_drops() -> None:
    cal = _sessions(200)
    h, embargo = 20, 5
    # valid_start index 100 -> boundary 95 -> fixed purge keeps date index <= 95 - 1 - 20 = 74
    frame = _frame(cal, [(60, 190)])
    fixed_keep = 60 <= 100 - embargo - 1 - h
    assert fixed_keep
    mask = purged_train_mask(
        frame,
        date_col="trade_date",
        end_col="label_end",
        valid_start=cal[100],
        embargo_sessions=embargo,
        calendar=cal,
    )
    assert mask.to_list() == [False]


def test_null_end_is_dropped_and_null_date_is_not_kept() -> None:
    cal = _sessions(50)
    frame = pl.DataFrame(
        {"trade_date": [cal[1], cal[2], None], "label_end": [None, cal[3], cal[3]]},
        schema={"trade_date": pl.Date, "label_end": pl.Date},
    )
    mask = purged_train_mask(
        frame,
        date_col="trade_date",
        end_col="label_end",
        valid_start=cal[30],
        embargo_sessions=3,
        calendar=cal,
    )
    assert mask.to_list() == [False, True, False]


def test_train_start_and_lazy_expression() -> None:
    cal = _sessions(50)
    frame = _frame(cal, [(1, 5), (10, 15), (40, 41)])
    expr = purged_train_expr(
        date_col="trade_date", end_col="label_end", boundary=cal[30], train_start=cal[5]
    )
    assert frame.lazy().select(expr.alias("m")).collect()["m"].to_list() == [False, True, False]
    with pytest.raises(TypeError, match="DataFrame"):
        purged_train_mask(
            frame.lazy(),  # type: ignore[arg-type]
            date_col="trade_date",
            end_col="label_end",
            valid_start=cal[30],
            embargo_sessions=1,
            calendar=cal,
        )


# --- equivalence with splits.py ------------------------------------------------

_CASES = [
    # n, horizon, embargo, n_folds, valid_len, holdout_len
    (300, 20, None, 5, None, 0),
    (300, 20, 20, 3, None, 0),
    (400, 20, 10, 4, 30, 40),  # embargo != horizon, explicit valid_len, holdout
    (400, 5, 20, 5, None, 60),
    (70, 20, 20, 5, None, 0),  # early folds do not fit and are skipped
    (120, 10, 0, 6, 7, 0),  # no embargo
    (45, 3, 3, 1, None, 0),
]


@pytest.mark.parametrize("n,h,embargo,n_folds,valid_len,holdout_len", _CASES)
def test_equals_splits_py_when_every_label_ends_h_sessions_later(
    n: int, h: int, embargo: int | None, n_folds: int, valid_len: int | None, holdout_len: int
) -> None:
    cal = _sessions(n)
    old = walk_forward_splits(
        cal,
        horizon=h,
        embargo=embargo,
        purge=h,
        n_folds=n_folds,
        valid_len=valid_len,
        holdout_len=holdout_len,
    )
    new = walk_forward_label_end_splits(
        cal,
        horizon=h,
        embargo=embargo,
        n_folds=n_folds,
        valid_len=valid_len,
        holdout_len=holdout_len,
    )
    assert [(f.fold_id, f.role) for f in new] == [(f.fold_id, f.role) for f in old]
    assert [(f.valid_start_idx, f.valid_end_idx) for f in new] == [
        (f.valid_start_idx, f.valid_end_idx) for f in old
    ]
    assert [(f.valid_start, f.valid_end) for f in new] == [
        (f.valid_start, f.valid_end) for f in old
    ]
    assert all(f.train_start_idx == 0 for f in new)

    # one row per session, label ends exactly h sessions later (none for the last h)
    frame = _frame(cal, [(i, i + h if i + h < n else None) for i in range(n)])
    masks = fold_masks(frame, new, date_col="trade_date", end_col="label_end")
    for m, o in zip(masks, old, strict=True):
        assert m.train_index().tolist() == list(range(0, o.train_end_idx + 1)), m.fold
        assert m.valid_index().tolist() == list(range(o.valid_start_idx, o.valid_end_idx + 1))
        # the train mask never reaches the validation window
        assert not (m.train & m.valid).any()


def test_fold_fields_boundary_and_record() -> None:
    cal = _sessions(300)
    folds = walk_forward_label_end_splits(cal, horizon=20, embargo=8, n_folds=3, holdout_len=30)
    assert [f.role for f in folds] == ["fold", "fold", "fold", "holdout"]
    for f in folds[:-1]:
        assert f.boundary_idx == f.valid_start_idx - 8
        assert f.boundary == cal[f.boundary_idx] and f.embargo == 8
    h = folds[-1]  # holdout: no embargo, boundary is its first date (as in splits.py)
    assert h.boundary_idx == h.valid_start_idx == 270 and h.embargo == 0
    assert h.valid_end_idx == 299
    assert not hasattr(h, "train_end")  # a date slice would silently drop the purge
    rec = folds[0].as_record()
    assert set(rec) == {
        "fold_id",
        "role",
        "train_start",
        "boundary",
        "valid_start",
        "valid_end",
        "embargo",
    }


def test_min_train_sessions_skips_early_folds_like_splits_py() -> None:
    cal = _sessions(70)
    old = walk_forward_splits(cal, horizon=20, embargo=20, purge=20, n_folds=5)
    new = walk_forward_label_end_splits(cal, horizon=20, embargo=20, n_folds=5)
    assert len(old) == len(new) == 4  # the first of five folds had no room
    # a stricter minimum drops one more
    stricter = walk_forward_label_end_splits(
        cal, horizon=20, embargo=20, n_folds=5, min_train_sessions=old[0].train_end_idx + 22
    )
    assert len(stricter) == 3


def test_argument_validation() -> None:
    cal = _sessions(30)
    with pytest.raises(ValueError, match="too few"):
        walk_forward_label_end_splits(cal, horizon=20, n_folds=5)
    with pytest.raises(ValueError, match="too few"):
        walk_forward_splits(cal, horizon=20, n_folds=5)  # same test as splits.py
    with pytest.raises(ValueError):
        walk_forward_label_end_splits(_sessions(300), embargo=-1)
    with pytest.raises(ValueError):
        walk_forward_label_end_splits(_sessions(300), n_folds=0)
    with pytest.raises(ValueError):
        walk_forward_label_end_splits(_sessions(300), min_train_sessions=0)
    with pytest.raises(ValueError, match="sorted"):
        walk_forward_label_end_splits(list(reversed(_sessions(300))))


# --- masks and counts -----------------------------------------------------------


def test_fold_masks_are_aligned_with_a_shuffled_multi_ticker_frame() -> None:
    cal = _sessions(200)
    base = _frame(
        cal,
        [(i, i + 20 if i + 20 < 200 else None) for i in range(0, 200, 5)] + [(60, 190), (40, None)],
    )
    frame = base.with_columns(pl.Series("ticker", ["X"] * (base.height - 2) + ["Y", "Y"]))
    frame = frame.sample(fraction=1.0, shuffle=True, seed=1)
    (fold,) = walk_forward_label_end_splits(cal, horizon=20, embargo=5, n_folds=1, valid_len=40)
    (m,) = fold_masks(frame, [fold], date_col="trade_date", end_col="label_end")
    kept = frame.filter(m.train)
    assert kept.height == m.train.sum() == len(m.train_index())
    assert kept["label_end"].is_not_null().all()
    assert kept["label_end"].max() < fold.boundary and kept["trade_date"].max() < fold.boundary
    # the 130-session late row and the NULL row are not in train
    assert not ((frame["trade_date"] == cal[60]) & (frame["label_end"] == cal[190]) & m.train).any()
    valid = frame.filter(m.valid)
    assert valid["trade_date"].min() >= fold.valid_start
    assert valid["trade_date"].max() <= fold.valid_end


def test_purge_counts_report_what_a_fixed_purge_would_have_leaked() -> None:
    cal = _sessions(200)
    frame = _frame(
        cal,
        [
            (10, 30),  # kept
            (60, 190),  # late, crosses; fixed h=20 purge keeps it (60 <= 74)
            (80, 100),  # regular but crosses (80 > 74): the fixed purge drops it too
            (50, None),  # no label
            (96, 99),  # dated at/after the boundary: not a candidate
        ],
    )
    folds = walk_forward_label_end_splits(cal, horizon=20, embargo=5, n_folds=1, valid_len=100)
    (fold,) = folds
    assert fold.valid_start_idx == 100 and fold.boundary_idx == 95
    got = purge_counts(
        frame,
        fold,
        date_col="trade_date",
        end_col="label_end",
        fixed_purge_horizon=20,
        calendar=cal,
    )
    assert got == {
        "candidates": 4,
        "kept": 1,
        "dropped_null_end": 1,
        "dropped_crosses_boundary": 2,
        "leaked_by_fixed_purge": 1,
    }
    plain = purge_counts(frame, fold, date_col="trade_date", end_col="label_end")
    assert "leaked_by_fixed_purge" not in plain and plain["kept"] == 1
    with pytest.raises(ValueError, match="calendar"):
        purge_counts(
            frame, fold, date_col="trade_date", end_col="label_end", fixed_purge_horizon=20
        )


# --- real labels_entry output ---------------------------------------------------


def test_late_and_terminal_labels_from_labels_entry_are_purged() -> None:
    """A is halted for 30 days and delisted: its c1 labels end in the 정리매매 tail,
    far later than t + h. B is an ordinary flat ticker (labels end t + 1 + h)."""
    h = 5
    con = price_con(delisted_after_long_halt())
    df = entry_frame(con, LabelSpec(horizons=(h,), kind="abs"))
    cols = entry_label_columns("c1", h)
    cal = sorted(df.filter(pl.col("ticker") == "B")["trade_date"].unique().to_list())
    assert len(cal) == 60

    a = df.filter(pl.col("ticker") == "A").sort("trade_date")
    first = a.row(0, named=True)
    assert first[cols.kind] == "late"
    late_by = cal.index(first[cols.end_date]) - cal.index(first["trade_date"]) - (h + 1)
    assert late_by > 20  # it ends dozens of sessions after t + 1 + h

    valid_start, embargo = cal[20], 3  # boundary = cal[17]
    mask = purged_train_mask(
        df,
        date_col="trade_date",
        end_col=cols.end_date,
        valid_start=valid_start,
        embargo_sessions=embargo,
        calendar=cal,
    )
    boundary = embargo_boundary(cal, valid_start, embargo)
    kept = df.filter(mask)
    # no kept row has a label that ends in or after the embargo/validation window
    assert kept[cols.end_date].max() < boundary
    # A's first five rows (t = d1..d5, all late) are out; B's rows are in until t + 1 + h hits it
    assert kept.filter(pl.col("ticker") == "A").height == 0
    b_kept = kept.filter(pl.col("ticker") == "B")["trade_date"].to_list()
    assert b_kept == cal[: 17 - (h + 1)]

    # what the fixed h-session purge of splits.py would have kept instead
    (fold,) = walk_forward_label_end_splits(
        cal, horizon=h, embargo=embargo, n_folds=1, valid_len=40
    )
    assert fold.valid_start == cal[20]
    counts = purge_counts(
        df,
        fold,
        date_col="trade_date",
        end_col=cols.end_date,
        fixed_purge_horizon=h,
        calendar=cal,
    )
    # A's four late rows (d5 is untradable: NULL end) + B's row 11, whose c1 label ends t + 1 + h
    # = the boundary, one session past what a fixed h purge trims
    assert counts["leaked_by_fixed_purge"] == 5
    assert counts["kept"] == len(b_kept)
