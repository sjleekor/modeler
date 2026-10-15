"""Outcome-blind case selection.

The functions here take frames with ``trade_date``, ``ticker``, ``market``, ``value`` and (after
``add_pct``) ``pct`` only. They raise when handed any label-like column, so a later edit cannot
let results steer the draw.
"""

from __future__ import annotations

import datetime as dt
import random

import polars as pl

from .config import (
    BOTTOM_PCT,
    FORMATION_END,
    FORMATION_START,
    HORIZON,
    N_ALL,
    N_DATES,
    N_GOOD,
    N_OPP,
    TOP_PCT,
    YEARS,
)
from .guards import OutcomeLeakError, assert_outcome_blind, check_label_end

ALLOWED = {"trade_date", "ticker", "market", "value", "pct"}


def build_grid(
    calendar: list[dt.date],
    start: dt.date = FORMATION_START,
    last_formation: dt.date = FORMATION_END,
    horizon: int = HORIZON,
    data_end: dt.date | None = None,
) -> list[tuple[dt.date, int, dt.date]]:
    """Non-overlapping grid: every ``horizon``-th trading day from ``start``.

    Returns ``(formation_date, calendar_index, label_window_end)``. A point whose 20-day
    window would end after ``data_end`` is dropped; the window end is always asserted.
    """
    cal = sorted(calendar)
    first = next((i for i, d in enumerate(cal) if d >= start), None)
    if first is None:
        return []
    out = []
    i = first
    while i + horizon < len(cal) and cal[i] <= last_formation:
        end = cal[i + horizon]
        if data_end is not None:
            check_label_end(end)
        out.append((cal[i], i, end))
        i += horizon
    return out


def choose_dates(
    grid_dates: list[dt.date], seed: int, years: tuple[int, ...] = YEARS, n_total: int = N_DATES
) -> list[dt.date]:
    """One date per year, then extras drawn from the rest, up to ``n_total``. Sorted."""
    rng = random.Random(seed)
    picked: list[dt.date] = []
    for y in years:
        pool = [d for d in sorted(grid_dates) if d.year == y]
        if pool:
            picked.append(rng.choice(pool))
    rest = [d for d in sorted(grid_dates) if d not in picked]
    n_extra = max(0, n_total - len(picked))
    picked += rng.sample(rest, min(n_extra, len(rest)))
    return sorted(picked)


def add_pct(df: pl.DataFrame) -> pl.DataFrame:
    """Percentile of ``value`` within (trade_date, market): average rank / count, in (0, 1]."""
    assert_outcome_blind(df.columns)
    d = df.filter(pl.col("value").is_not_null() & pl.col("value").is_not_nan())
    return d.with_columns(
        (
            pl.col("value").rank(method="average").over(["trade_date", "market"])
            / pl.col("value").count().over(["trade_date", "market"])
        ).alias("pct")
    )


def group_names(direction: int) -> tuple[str, str]:
    return ("top", "bottom") if direction == 0 else ("good", "opp")


def with_groups(df: pl.DataFrame, direction: int) -> pl.DataFrame:
    assert_outcome_blind(df.columns)
    first, second = group_names(direction)
    hi = pl.col("pct") >= TOP_PCT
    lo = pl.col("pct") <= BOTTOM_PCT
    if direction == 0:
        g_hi, g_lo = "top", "bottom"
    elif direction > 0:
        g_hi, g_lo = first, second
    else:
        g_hi, g_lo = second, first
    return df.with_columns(
        pl.when(hi)
        .then(pl.lit(g_hi))
        .when(lo)
        .then(pl.lit(g_lo))
        .otherwise(pl.lit("mid"))
        .alias("group")
    )


def select_cases(
    cands: pl.DataFrame,
    direction: int,
    seed: int,
    feature: str,
    n_good: int = N_GOOD,
    n_opp: int = N_OPP,
    n_all: int = N_ALL,
) -> pl.DataFrame:
    """Draw cases for every date in ``cands``. Needs ``pct``; never sees outcomes.

    Per date: ``n_good`` from the good side, ``n_opp`` from the opposite side, ``n_all`` from the
    whole universe (not already drawn). A separate ``random.Random`` per (seed, feature, date)
    keeps the draw reproducible whatever the order of features or dates.
    """
    assert_outcome_blind(cands.columns)
    missing = {"trade_date", "ticker", "market", "value", "pct"} - set(cands.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    extra = set(cands.columns) - ALLOWED - {"group"}
    if extra:
        raise OutcomeLeakError(f"unexpected columns in selection input: {sorted(extra)}")
    g = with_groups(cands.select(sorted(ALLOWED & set(cands.columns))), direction)
    first, second = group_names(direction)
    rows = []
    for d in sorted(g["trade_date"].unique().to_list()):
        day = g.filter(pl.col("trade_date") == d).sort(["ticker", "market"])
        rng = random.Random(f"{seed}|{feature}|{d.isoformat()}")
        taken: set[str] = set()
        for name, n in ((first, n_good), (second, n_opp)):
            pool = day.filter(pl.col("group") == name)
            idx = sorted(rng.sample(range(pool.height), min(n, pool.height)))
            for r in pool[idx].iter_rows(named=True):
                rows.append({**r, "group": name})
                taken.add(r["ticker"])
        pool = day.filter(~pl.col("ticker").is_in(list(taken)))
        idx = sorted(rng.sample(range(pool.height), min(n_all, pool.height)))
        for r in pool[idx].iter_rows(named=True):
            rows.append({**r, "group": "all"})
    schema = {**{k: g.schema[k] for k in g.columns}}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def hit_flag(raw_label: float | None, group: str) -> bool | None:
    """Hit = excess return over the market equal-weight > 0. Only the good side is scored."""
    if group != "good" or raw_label is None or raw_label != raw_label:
        return None
    return raw_label > 0
