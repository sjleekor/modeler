"""Training / evaluation row gate for the entry labels (``etl/label_gate.py``)."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from modeler.etl.label_gate import (
    DEFAULT_ALLOWED_KINDS,
    LABEL_EXIT_KINDS,
    apply_label_gate,
    entry_label_columns,
    exit_kind_counts,
    label_gate_expr,
)
from modeler.etl.labels import LabelSpec

from ._entry_label_fixtures import delisted_after_long_halt, entry_frame, price_con, series

H = 20


def _frame(kinds: list[str | None], values: list[float | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(len(kinds))],
            f"label_exit_kind_c1_{H}d": kinds,
            f"fwd_ret_c1_{H}d": values,
        },
        schema={
            "ticker": pl.String,
            f"label_exit_kind_c1_{H}d": pl.String,
            f"fwd_ret_c1_{H}d": pl.Float64,
        },
    )


def test_columns_follow_labels_entry_naming() -> None:
    c = entry_label_columns("o1", 5)
    assert (c.fwd, c.kind, c.end_date, c.exit_lag) == (
        "fwd_ret_o1_5d",
        "label_exit_kind_o1_5d",
        "label_end_date_o1_5d",
        "label_exit_lag_o1_5d",
    )
    for tag, h in (("c0", 5), ("c1", 0), ("c1", -1), ("c1", True), ("c1", 2.0)):
        with pytest.raises(ValueError):
            entry_label_columns(tag, h)  # type: ignore[arg-type]


def test_default_gate_keeps_value_kinds_and_drops_the_rest() -> None:
    assert DEFAULT_ALLOWED_KINDS == ("regular", "late", "terminal", "terminal_halted")
    kinds = list(LABEL_EXIT_KINDS)
    # every kind carries a value, to show the KIND decides, not only the value
    df = _frame(kinds, [0.1] * len(kinds))
    kept = apply_label_gate(df, "c1", H)[f"label_exit_kind_c1_{H}d"].to_list()
    assert kept == ["regular", "late", "terminal", "terminal_halted"]
    for gone in ("terminal_unpriced", "censored", "untradable", "ca_masked"):
        assert gone not in kept


def test_null_value_or_null_kind_is_not_usable_even_for_an_allowed_kind() -> None:
    df = _frame(["regular", "late", None, "terminal"], [None, 0.2, 0.3, 0.4])
    got = df.select(label_gate_expr("c1", H).alias("g"))["g"].to_list()
    assert got == [False, True, False, True]  # a boolean, never NULL
    assert apply_label_gate(df, "c1", H)["ticker"].to_list() == ["T1", "T3"]


def test_allowed_kinds_override_also_require_and_lazy_frames() -> None:
    df = _frame(["regular", "late", "terminal", "terminal_halted"], [0.1, 0.2, 0.3, 0.4])
    not_terminal = apply_label_gate(df, "c1", H, allowed_kinds=("regular", "late"))
    assert not_terminal["ticker"].to_list() == ["T0", "T1"]
    lazy = apply_label_gate(df.lazy(), "c1", H, allowed_kinds=("terminal",))
    assert isinstance(lazy, pl.LazyFrame) and lazy.collect()["ticker"].to_list() == ["T2"]
    extra = df.with_columns(pl.Series("raw_label_c1_20d", [None, 1.0, 1.0, None]))
    both = apply_label_gate(extra, "c1", H, also_require=["raw_label_c1_20d"])
    assert both["ticker"].to_list() == ["T1", "T2"]
    with pytest.raises(ValueError, match="unknown label_exit_kind"):
        label_gate_expr("c1", H, allowed_kinds=("regular", "lates"))


def test_exit_kind_counts_report() -> None:
    df = _frame(
        ["regular"] * 6 + ["late", "late", "terminal", "censored", "untradable", "weird", None],
        [0.1] * 6 + [0.2, None, 0.3, None, None, 0.9, 0.9],
    )
    out = exit_kind_counts(df, "c1", H)
    assert out.columns == ["kind", "n_rows", "n_gated", "share", "share_in_gate"]
    # canonical order, zero counts kept, then the unknown kind, then (null)
    assert out["kind"].to_list() == [*LABEL_EXIT_KINDS, "weird", "(null)"]
    rows = {r["kind"]: r for r in out.iter_rows(named=True)}
    assert (rows["regular"]["n_rows"], rows["regular"]["n_gated"]) == (6, 6)
    assert (rows["late"]["n_rows"], rows["late"]["n_gated"]) == (2, 1)  # one late has no value
    assert (rows["terminal"]["n_rows"], rows["terminal"]["n_gated"]) == (1, 1)
    assert rows["terminal_halted"]["n_rows"] == 0 and rows["ca_masked"]["n_gated"] == 0
    assert rows["weird"]["n_gated"] == 0 and rows["(null)"]["n_rows"] == 1
    assert out["n_rows"].sum() == df.height
    assert out["n_gated"].sum() == apply_label_gate(df, "c1", H).height == 8
    assert rows["regular"]["share"] == pytest.approx(6 / 13)
    assert rows["late"]["share_in_gate"] == pytest.approx(1 / 8)
    assert rows["terminal"]["share_in_gate"] == pytest.approx(1 / 8)
    assert out["share_in_gate"].sum() == pytest.approx(1.0)
    # an empty frame still reports every kind with zeros
    empty = exit_kind_counts(df.head(0), "c1", H)
    assert empty["kind"].to_list() == list(LABEL_EXIT_KINDS)
    assert empty["n_rows"].sum() == 0 and empty["share"].sum() == 0.0
    assert exit_kind_counts(df.lazy(), "c1", H).equals(out)


# --- real labels_entry output ---------------------------------------------------


def test_gate_on_labels_entry_output_keeps_delisted_names() -> None:
    """A is halted 30 days and delisted after 7 tail days. The c0 survivor gate
    (``universe.label_ok``) loses A's rows; the label gate keeps them as ``late``."""
    h = 5
    df = entry_frame(price_con(delisted_after_long_halt()), LabelSpec(horizons=(h,), kind="abs"))
    cols = entry_label_columns("c1", h)
    assert {cols.fwd, cols.kind, cols.end_date, cols.exit_lag} <= set(df.columns)

    kept = apply_label_gate(df, "c1", h)
    a_all = kept.filter(pl.col("ticker") == "A").sort("trade_date")
    a = a_all.filter(pl.col("trade_date") < datetime.date(2020, 1, 10))  # before the halt
    assert a[cols.kind].to_list() == ["late"] * 4  # d1..d4, all end in the tail
    # the 7 tail rows are themselves labelled (and are the universe's job to exclude)
    assert a_all.height == 4 + 6 and a_all["in_terminal_tail"].sum() == 6
    assert (a[cols.exit_lag] > 0).all() and a[cols.fwd].is_not_null().all()
    # d5 (2020-01-06): the next day is halted, A cannot be bought -> untradable
    d5 = df.filter((pl.col("ticker") == "A") & (pl.col("trade_date") == datetime.date(2020, 1, 6)))
    assert d5[cols.kind].to_list() == ["untradable"]
    # nothing the gate keeps lacks a value or has an excluded kind
    assert kept[cols.fwd].is_not_null().all()
    assert set(kept[cols.kind].unique()) <= set(DEFAULT_ALLOWED_KINDS)
    # report: B is regular until the last h+1 days (no exit yet: censored)
    counts = exit_kind_counts(df, "c1", h)
    got = dict(zip(counts["kind"], counts["n_rows"], strict=True))
    assert got["regular"] > 0 and got["late"] == 4 and got["censored"] > 0
    assert got["untradable"] >= 1
    assert counts["n_rows"].sum() == df.height


def test_unpriced_long_halt_end_needs_the_haircut_to_enter_the_gate() -> None:
    rows = series("A", "2020-01-02", [100, 101, 102, 103, 104] + [None] * 25)
    rows += series("B", "2020-01-02", [10] * 60)
    con = price_con(rows)
    spec = LabelSpec(horizons=(3,), kind="abs")
    is_a_d4 = (pl.col("ticker") == "A") & (pl.col("trade_date") == datetime.date(2020, 1, 5))

    unpriced = entry_frame(con, spec)  # end_halt_haircut=None: no value
    assert unpriced.filter(is_a_d4)["label_exit_kind_c1_3d"].to_list() == ["terminal_unpriced"]
    assert apply_label_gate(unpriced, "c1", 3).filter(is_a_d4).height == 0

    priced = entry_frame(con, spec, end_halt_haircut=0.0)
    assert priced.filter(is_a_d4)["label_exit_kind_c1_3d"].to_list() == ["terminal_halted"]
    assert apply_label_gate(priced, "c1", 3).filter(is_a_d4).height == 1
