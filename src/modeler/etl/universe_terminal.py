"""Formation universe without 정리매매 (terminal tail) rows — quality-fix plan §9.3.

``etl/universe.py`` does not filter 관리종목 / 정리매매 (``management_filter_available``
is always FALSE): on the lake 378 rows of the 정리매매 tail (281 with the liquidity
condition) pass ``in_universe``. This module adds the exclusion beside it without
touching ``universe.py`` (a model02 code-hash file):

    in_universe_v2 = in_universe AND NOT COALESCE(in_terminal_tail, FALSE)

``in_terminal_tail`` is the per-row flag of ``labels_entry`` (one value per
(trade_date, ticker, market), the same for every tag and horizon): the row is one
of the 1..``terminal_tail_max`` non-halt rows after the last halt of a series that
ENDED (delisted). Rows the label table does not have (halted days) have no flag
and count as FALSE; they are already out of ``in_universe``. A NULL ``in_universe``
counts as FALSE, so ``in_universe_v2`` is never NULL.

Two forms, same result (pinned by a test):

  * :func:`build_universe_v2_sql` — wraps an existing universe view (or a
    ``universe.build_universe_sql(...)`` statement) and the label view.
  * :func:`with_terminal_exclusion` — polars, DataFrame or LazyFrame.

TRAIN vs SERVING. ``in_terminal_tail`` is derived from the full future price
series. For training that is fine: 정리매매 is announced before it starts, so on
those days the fact was knowable. At serving the same fact has to come from a live
source (KIS stock status ``kis_stock_status_daily``, or exchange notices). That
decision rule is NOT implemented here — until it exists, serving would still
rank 정리매매 names that training excluded, so the preregistration has to name the
serving-side source and fix the rule.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import polars as pl

KEYS: tuple[str, ...] = ("trade_date", "ticker", "market")
TAIL_COL = "in_terminal_tail"
UNIVERSE_COL = "in_universe"
OUT_COL = "in_universe_v2"

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_STATEMENT = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def _ident(value: str, what: str) -> str:
    if not _IDENT.match(value):
        raise ValueError(f"{what} must be a plain SQL identifier, got {value!r}")
    return value


def _relation(value: str, what: str) -> str:
    """View/table name as is; a SELECT/WITH statement as a parenthesised subquery."""
    if _IDENT.match(value):
        return value
    if _STATEMENT.match(value):
        return f"(\n{value.strip()}\n)"
    raise ValueError(f"{what} must be a view name or a SELECT/WITH statement")


def build_universe_v2_sql(
    universe: str,
    label_view: str,
    *,
    keys: Sequence[str] = KEYS,
    universe_col: str = UNIVERSE_COL,
    tail_col: str = TAIL_COL,
    out_col: str = OUT_COL,
) -> str:
    """SQL: every universe column plus ``in_terminal_tail`` and ``in_universe_v2``.

    ``universe``: a registered view/table name (``dim_universe_daily``, the broad or
    tradable universe ...) or a SELECT/WITH statement such as
    ``universe.build_universe_sql(flt)``. ``label_view``: the ``labels_entry`` output
    (``label_entry_v2``) or any relation with ``keys`` and ``tail_col``. The label
    side is grouped per key (``BOOL_OR``), so the join never multiplies universe
    rows. The universe must not already carry ``tail_col`` / ``out_col``.
    """
    uni = _relation(universe, "universe")
    lab = _relation(label_view, "label_view")
    ks = [_ident(k, "key") for k in keys]
    u_col, t_col, o_col = (_ident(c, "column") for c in (universe_col, tail_col, out_col))
    if not ks:
        raise ValueError("keys must not be empty")
    key_list = ", ".join(ks)
    on = " AND ".join(f"t.{k} = u.{k}" for k in ks)
    return f"""
        WITH tail AS (
            SELECT {key_list}, BOOL_OR(COALESCE({t_col}, FALSE)) AS {t_col}
            FROM {lab}
            GROUP BY {key_list}
        )
        SELECT u.*,
               COALESCE(t.{t_col}, FALSE) AS {t_col},
               (COALESCE(u.{u_col}, FALSE) AND NOT COALESCE(t.{t_col}, FALSE)) AS {o_col}
        FROM {uni} u
        LEFT JOIN tail t ON {on}
    """


def with_terminal_exclusion[F: (pl.DataFrame, pl.LazyFrame)](
    universe: F,
    label: pl.DataFrame | pl.LazyFrame | None = None,
    *,
    keys: Sequence[str] = KEYS,
    universe_col: str = UNIVERSE_COL,
    tail_col: str = TAIL_COL,
    out_col: str = OUT_COL,
) -> F:
    """``universe`` plus ``in_terminal_tail`` and ``in_universe_v2`` (polars).

    ``label``: the ``labels_entry`` output (needs ``keys`` and ``tail_col``). It is
    grouped per key (any flag true), so universe rows are never multiplied, and the
    universe's row order is kept. Pass ``label=None`` when ``universe`` already
    carries ``tail_col`` (e.g. it is the label table joined to ``in_universe``).
    Same type back as ``universe`` (DataFrame in, DataFrame out).
    """
    lazy_in = isinstance(universe, pl.LazyFrame)
    u = universe.lazy()
    names = set(u.collect_schema().names())
    ks = list(keys)
    if out_col in names:
        raise ValueError(f"universe already has a column {out_col!r}")
    if label is None:
        if tail_col not in names:
            raise ValueError(f"universe has no {tail_col!r} column; pass the label table")
    else:
        if tail_col in names:
            raise ValueError(f"universe already has {tail_col!r}; pass label=None")
        tails = (
            label.lazy().group_by(ks).agg(pl.col(tail_col).fill_null(False).any().alias(tail_col))
        )
        u = u.join(tails, on=ks, how="left", maintain_order="left")
    u = u.with_columns(pl.col(tail_col).fill_null(False)).with_columns(
        (pl.col(universe_col).fill_null(False) & ~pl.col(tail_col)).alias(out_col)
    )
    return u if lazy_in else u.collect()  # type: ignore[return-value]


def terminal_exclusion_counts(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    universe_col: str = UNIVERSE_COL,
    tail_col: str = TAIL_COL,
    out_col: str = OUT_COL,
) -> dict[str, int]:
    """Counts for the report: rows in ``in_universe``, how many of them are
    정리매매 tail rows (dropped), and rows left in ``in_universe_v2``."""
    row = (
        frame.lazy()
        .select(
            pl.col(universe_col).fill_null(False).sum().alias("n_in_universe"),
            (pl.col(universe_col).fill_null(False) & pl.col(tail_col).fill_null(False))
            .sum()
            .alias("n_terminal_tail_dropped"),
            pl.col(out_col).fill_null(False).sum().alias("n_in_universe_v2"),
        )
        .collect()
        .row(0, named=True)
    )
    return {k: int(v or 0) for k, v in row.items()}
