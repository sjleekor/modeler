"""Training / evaluation row gate for the entry labels — quality-fix plan §9.3.

``etl/universe.py`` has ``label_ok``: "the ticker has t+H non-halt rows later".
That is a *survivor gate*: the last H rows of every delisted ticker fail it, so
the delisted names (and their losses) never reach train or evaluation. It is
TRAIN/EVAL ONLY by construction (it looks into the future) and it is the survivor
bias that ``labels_entry.py`` removes by closing a delisted series at its last
close (``terminal`` kinds).

For a model built on ``labels_entry`` this module REPLACES ``label_ok``. A row is
usable iff

  * ``fwd_ret_{tag}_{h}d`` is not NULL (a value exists), and
  * ``label_exit_kind_{tag}_{h}d`` is one of :data:`DEFAULT_ALLOWED_KINDS`
    (``regular``, ``late``, ``terminal``, ``terminal_halted``).

The kinds and why the rest are out:

  ``regular``            exit h sessions after entry
  ``late``               exit later than h (long halt) — value kept on purpose,
                         the loss of a halted name is the largest part of the bias
  ``terminal``           delisted, closed at the series' last close
  ``terminal_halted``    delisted after a long halt, closed at last close x (1 - s)
  ``terminal_unpriced``  excluded: no value (``end_halt_haircut=None``)
  ``censored``           excluded: panel end / still halted, outcome not known yet
  ``untradable``         excluded: entry halted / locked limit-up; a name that
                         could not be bought is not in the portfolio
  ``ca_masked``          excluded: corporate action inside the window

The gate looks at the future (kind, value), so it is for training and evaluation
rows only — never for the inference universe. Pair it with the formation universe
(``in_universe`` / ``universe_terminal.in_universe_v2``) and purge train by
``label_end_date`` (``etl/splits_label_end.py``).

Report ``exit_kind_counts`` next to every evaluation (plan §9.3: "평가 보고에
``late``·``terminal`` 비중을 같이 냅니다").

``tag`` is ``c1`` or ``o1``; ``h`` is the horizon in sessions. Column names are
those of ``labels_entry.build_entry_label_sql``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple

import polars as pl

LABEL_EXIT_KINDS: tuple[str, ...] = (
    "regular",
    "late",
    "terminal",
    "terminal_halted",
    "terminal_unpriced",
    "censored",
    "untradable",
    "ca_masked",
)
DEFAULT_ALLOWED_KINDS: tuple[str, ...] = ("regular", "late", "terminal", "terminal_halted")
_TAGS = ("c1", "o1")


class EntryLabelColumns(NamedTuple):
    """Column names of one (tag, horizon) label of ``labels_entry``."""

    fwd: str
    kind: str
    end_date: str
    exit_lag: str


def entry_label_columns(tag: str, h: int) -> EntryLabelColumns:
    """``fwd_ret_{tag}_{h}d`` and the three ``label_*_{tag}_{h}d`` columns."""
    if tag not in _TAGS:
        raise ValueError(f"tag must be one of {_TAGS}, got {tag!r}")
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError(f"h must be a positive int, got {h!r}")
    return EntryLabelColumns(
        fwd=f"fwd_ret_{tag}_{h}d",
        kind=f"label_exit_kind_{tag}_{h}d",
        end_date=f"label_end_date_{tag}_{h}d",
        exit_lag=f"label_exit_lag_{tag}_{h}d",
    )


def _check_kinds(allowed_kinds: Iterable[str]) -> tuple[str, ...]:
    kinds = tuple(allowed_kinds)
    bad = sorted(set(kinds) - set(LABEL_EXIT_KINDS))
    if bad:
        raise ValueError(f"unknown label_exit_kind {bad}; known: {list(LABEL_EXIT_KINDS)}")
    return kinds


def label_gate_expr(
    tag: str,
    h: int,
    *,
    allowed_kinds: Sequence[str] = DEFAULT_ALLOWED_KINDS,
    also_require: Sequence[str] = (),
) -> pl.Expr:
    """Boolean expression: the row is usable for training / evaluation.

    ``fwd_ret_{tag}_{h}d IS NOT NULL AND label_exit_kind_{tag}_{h}d IN allowed_kinds``.
    ``also_require`` adds more columns that must be non-NULL (e.g. a
    ``raw_label_idx_c1_{h}d`` target). Never NULL: a NULL kind or value is not usable.
    """
    cols = entry_label_columns(tag, h)
    kinds = _check_kinds(allowed_kinds)
    expr = pl.col(cols.fwd).is_not_null() & pl.col(cols.kind).is_in(list(kinds))
    for c in also_require:
        expr = expr & pl.col(c).is_not_null()
    return expr.fill_null(False)


def apply_label_gate[F: (pl.DataFrame, pl.LazyFrame)](
    frame: F,
    tag: str,
    h: int,
    *,
    allowed_kinds: Sequence[str] = DEFAULT_ALLOWED_KINDS,
    also_require: Sequence[str] = (),
) -> F:
    """Rows of ``frame`` (DataFrame or LazyFrame) that pass :func:`label_gate_expr`."""
    return frame.filter(
        label_gate_expr(tag, h, allowed_kinds=allowed_kinds, also_require=also_require)
    )


def exit_kind_counts(
    frame: pl.DataFrame | pl.LazyFrame,
    tag: str,
    h: int,
    *,
    allowed_kinds: Sequence[str] = DEFAULT_ALLOWED_KINDS,
    also_require: Sequence[str] = (),
) -> pl.DataFrame:
    """Row counts per ``label_exit_kind`` for the evaluation report.

    One row per known kind (zero counts included, canonical order), then any
    unknown kind, then ``"(null)"`` if present. Columns:

    ``kind``           exit kind
    ``n_rows``         all rows of that kind
    ``n_gated``        rows of that kind that pass the gate
    ``share``          ``n_rows`` / all rows
    ``share_in_gate``  ``n_gated`` / all gated rows (the ``late`` / ``terminal``
                       weight inside what train/evaluation actually uses)
    """
    cols = entry_label_columns(tag, h)
    gate = label_gate_expr(tag, h, allowed_kinds=allowed_kinds, also_require=also_require)
    agg = (
        frame.lazy()
        .group_by(pl.col(cols.kind).fill_null("(null)").alias("kind"))
        .agg(
            pl.len().cast(pl.Int64).alias("n_rows"),
            gate.cast(pl.Int64).sum().alias("n_gated"),
        )
        .collect()
    )
    order = {k: i for i, k in enumerate(LABEL_EXIT_KINDS)}
    known = pl.DataFrame({"kind": list(LABEL_EXIT_KINDS)}, schema={"kind": pl.String})
    out = (
        known.join(agg, on="kind", how="full", coalesce=True)
        .with_columns(pl.col("n_rows", "n_gated").fill_null(0).cast(pl.Int64))
        .with_columns(
            pl.col("kind")
            .replace_strict(order, default=len(order), return_dtype=pl.Int64)
            .alias("_order")
        )
        .with_columns(
            (pl.col("kind") == "(null)").cast(pl.Int64).alias("_null"),
        )
        .sort(["_order", "_null", "kind"])
        .drop("_order", "_null")
    )
    total = int(out["n_rows"].sum())
    gated = int(out["n_gated"].sum())
    return out.with_columns(
        (pl.col("n_rows") / total if total else pl.lit(0.0)).alias("share"),
        (pl.col("n_gated") / gated if gated else pl.lit(0.0)).alias("share_in_gate"),
    )
