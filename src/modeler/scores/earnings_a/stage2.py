"""Research A stage 2 core: calendar, company market cap, signals, baskets, returns, statistics.

Pure functions over small frames and dicts (no lake access, no file output). The database glue
is ``stage2_run.py``. Rules come from ``my/milestones/kr/modeling/plan/20261103_earnings_a_stage2/
01_preregistration_draft.md`` (section 16 wins over the body text):

* decision date ``t`` = quarter end; financial inputs ``available_from <= t`` (stage 1 code);
* signal day ``u`` = last session on or before ``t + 14`` calendar days;
* entry = 5th session after ``u`` (close); exit = last session on or before entry + 12
  calendar months (close);
* signals ``M = yhat_HGB * A_t / mcap_u`` and ``B1 = yhat_PERS * A_t / mcap_u`` where ``mcap_u``
  is the company market cap at ``u`` (common line plus every line sharing its first five
  characters);
* basket = top 30 by signal desc, ties ticker asc, equal weight, no rebalance; an entry-day
  halted name is dropped and the weights renormalized (no replacement);
* basket return = mean holding return - 0.0065; B0 = KOSPI 200 price index, no cost.

Everything beyond that (B1u, EV denominator, capped weights, N 15 / 50, other costs, dividends)
is record-only.
"""

from __future__ import annotations

import calendar
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl

from modeler.models._03_kr_next.evaluate import cap_weights, holding_returns

SIGNAL_LAG_DAYS = 14
ENTRY_SESSIONS = 5
HOLD_MONTHS = 12
BASKET_N = 30
COST_MAIN = 0.0065
COSTS_RECORD = (0.0100, 0.0130)
WEIGHT_CAP = 0.10
PRICE_PER_NAME_LIMIT = 416_667  # 50,000,000 won / 4 tranches / 30 names (rounded up)
FORWARD_FIRST_MAIN = date(2026, 12, 31)  # first tranche that counts for the main estimate
BLOCK_TRANCHES = 4
BOOTSTRAP_SEED = 20261103
BOOTSTRAP_DRAWS = 10_000
INDEX_NAME = "코스피 200"
KODEX200 = "069500"

# name -> (signal column, N, weighting). Every one is built for every tranche.
BASKET_VARIANTS: dict[str, tuple[str, int, str]] = {
    "M": ("M", 30, "eq"),
    "B1": ("B1", 30, "eq"),
    "B1u": ("B1u", 30, "eq"),
    "M_EV": ("M_EV", 30, "eq"),
    "B1_EV": ("B1_EV", 30, "eq"),
    "M_cap": ("M", 30, "cap"),
    "B1_cap": ("B1", 30, "cap"),
    "M_N15": ("M", 15, "eq"),
    "B1_N15": ("B1", 15, "eq"),
    "M_N50": ("M", 50, "eq"),
    "B1_N50": ("B1", 50, "eq"),
}
BASKET_COLUMNS = [
    "variant",
    "rank",
    "ticker",
    "corp_code",
    "weight",
    "signal",
    "mcap_u",
    "div_yield",
    "ksic2",
]


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def _sorted(sessions: Iterable[date]) -> list[date]:
    return sorted(set(sessions))


def signal_day(sessions: Iterable[date], t: date) -> date | None:
    """Last session on or before ``t + 14`` calendar days (None when there is none)."""
    limit = t + timedelta(days=SIGNAL_LAG_DAYS)
    ok = [s for s in _sorted(sessions) if s <= limit]
    return ok[-1] if ok else None


def signal_day_known(sessions: Iterable[date], t: date) -> bool:
    """True when the data reaches ``t + 14`` days (a trailing weekend is allowed), so the signal
    day cannot still move later."""
    ss = _sorted(sessions)
    if not ss:
        return False
    limit = t + timedelta(days=SIGNAL_LAG_DAYS)
    last = ss[-1]
    return last >= limit or all(
        (last + timedelta(days=i)).weekday() >= 5 for i in range(1, (limit - last).days + 1)
    )


def entry_day(sessions: Iterable[date], u: date, n: int = ENTRY_SESSIONS) -> date | None:
    """The ``n``-th session after ``u`` (None when the data does not reach it)."""
    ss = _sorted(sessions)
    if u not in set(ss):
        return None
    i = ss.index(u) + n
    return ss[i] if i < len(ss) else None


def entry_day_estimate(
    sessions: Iterable[date], u: date, n: int = ENTRY_SESSIONS
) -> tuple[date, bool]:
    """Entry day, continuing with weekdays when the data ends first.

    Returns ``(date, estimated)``. A forward tranche is planned before the sessions exist; the
    weekday rule ignores exchange holidays, so an estimated date can be early.
    """
    known = entry_day(sessions, u, n)
    if known is not None:
        return known, False
    ss = [s for s in _sorted(sessions) if s > u]
    d = ss[-1] if ss else u
    count = len(ss)
    while count < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d, True


def add_months(d: date, n: int) -> date:
    """``d`` plus ``n`` calendar months, clamped to the month length (2024-02-29 + 12 -> 02-28)."""
    total = d.year * 12 + d.month - 1 + n
    y, m0 = divmod(total, 12)
    return date(y, m0 + 1, min(d.day, calendar.monthrange(y, m0 + 1)[1]))


def exit_target(entry: date) -> date:
    return add_months(entry, HOLD_MONTHS)


def exit_day(sessions: Iterable[date], entry: date) -> date | None:
    """Last session on or before entry + 12 months; None ("beyond data") when the data ends
    before that date, because a later session may still fall on or before it."""
    ss = _sorted(sessions)
    target = exit_target(entry)
    if not ss or ss[-1] < target:
        return None
    return [s for s in ss if s <= target][-1]


@dataclass(frozen=True)
class Schedule:
    decision_date: date
    signal_day: date | None
    entry: date | None
    exit: date | None


def schedule(sessions: Iterable[date], t: date) -> Schedule:
    ss = _sorted(sessions)
    u = signal_day(ss, t)
    e = entry_day(ss, u) if u is not None else None
    x = exit_day(ss, e) if e is not None else None
    return Schedule(t, u, e, x)


def late_flag(today: date, entry: date) -> bool:
    """A forward basket is late when the entry day has already come (``today >= entry``)."""
    return today >= entry


# ---------------------------------------------------------------------------
# Company market cap
# ---------------------------------------------------------------------------


def company_mcap(mcap: pd.DataFrame) -> pd.DataFrame:
    """Company market cap per common ticker from one day of ``daily_market_cap``.

    ``mcap`` has ``ticker`` and ``market_cap`` (listed lines of that day). A company is every
    line sharing the first five characters; its common ticker ends with ``0``. Returns one row
    per common ticker: ``ticker, mcap_common, pref_mcap, mcap_total, n_pref, has_pref,
    pref_ratio``. A missing common market cap gives NaN totals; a missing preferred one counts 0.
    """
    df = mcap[["ticker", "market_cap"]].copy()
    df["_key"] = df["ticker"].str[:5]
    df["_common"] = df["ticker"].str[-1] == "0"
    df["market_cap"] = df["market_cap"].astype(float)
    common = df[df["_common"]].drop_duplicates("ticker")
    pref = df[~df["_common"]]
    pg = pref.groupby("_key")["market_cap"].agg(
        pref_mcap=lambda s: float(s.fillna(0.0).sum()), n_pref="size"
    )
    out = common[["ticker", "_key", "market_cap"]].rename(columns={"market_cap": "mcap_common"})
    out = out.merge(pg, left_on="_key", right_index=True, how="left")
    out["pref_mcap"] = out["pref_mcap"].fillna(0.0)
    out["n_pref"] = out["n_pref"].fillna(0).astype(int)
    out["mcap_total"] = out["mcap_common"] + out["pref_mcap"]
    out["has_pref"] = out["n_pref"] > 0
    out["pref_ratio"] = np.where(
        out["mcap_common"] > 0, out["pref_mcap"] / out["mcap_common"], np.nan
    )
    return out.drop(columns="_key").sort_values("ticker").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Signals, universe, baskets
# ---------------------------------------------------------------------------


def as_bool(s: pd.Series) -> pd.Series:
    """Boolean Series with missing values as False (no pandas downcasting warning)."""
    return s.astype("object").where(s.notna(), False).astype(bool)


DROP_ORDER = ("not_listed", "halted", "p6_not_tradable", "liq_notice", "no_mcap", "no_prediction")


def flag_universe(df: pd.DataFrame, *, require_prediction: bool = True) -> pd.DataFrame:
    """Add ``drop_reason`` (first that applies, in ``DROP_ORDER``) and ``in_stage2``.

    ``df`` holds the stage-1 input-eligible rows of one decision date joined to the signal-day
    facts: ``listed_u, halted_u, p6_tradable_u, liq_notice_u, mcap_u`` and, when predictions are
    required, ``y_hat_hgb``. Labels are never read.
    """
    out = df.copy()
    flags = {
        "not_listed": ~as_bool(out["listed_u"]),
        "halted": as_bool(out["halted_u"]),
        "p6_not_tradable": ~as_bool(out["p6_tradable_u"]),
        "liq_notice": as_bool(out["liq_notice_u"]),
        "no_mcap": ~(out["mcap_u"].astype(float) > 0),
        "no_prediction": (
            out["y_hat_hgb"].isna() if require_prediction else pd.Series(False, index=out.index)
        ),
    }
    reason = pd.Series(None, index=out.index, dtype="object")
    for name in reversed(DROP_ORDER):
        reason = reason.where(~flags[name], name)
    out["drop_reason"] = reason
    out["drop_reason"] = reason.astype("object").where(reason.notna(), None)
    out["in_stage2"] = reason.isna()
    return out


def add_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Signals ``M, B1`` (primary) and ``B1u, M_EV, B1_EV`` (record only), all per mcap_u.

    Needs ``a_t, mcap_u, y_hat_hgb, y_hat_pers, oi_ttm_raw`` and the amounts ``debt_st_amt,
    debt_lt_amt, cash_amt`` (won). EV = mcap + borrowings - cash with null borrowings = 0; a
    null cash or EV <= 0 gives NaN for the EV signals.
    """
    out = df.copy()
    a = out["a_t"].astype(float)
    mc = out["mcap_u"].astype(float).where(lambda s: s > 0)
    out["M"] = out["y_hat_hgb"].astype(float) * a / mc
    out["B1"] = out["y_hat_pers"].astype(float) * a / mc
    out["B1u"] = out["oi_ttm_raw"].astype(float) * a / mc
    ev = (
        mc
        + out["debt_st_amt"].astype(float).fillna(0.0)
        + out["debt_lt_amt"].astype(float).fillna(0.0)
        - out["cash_amt"].astype(float)
    )
    ev = ev.where(ev > 0)
    out["ev_u"] = ev
    out["M_EV"] = out["y_hat_hgb"].astype(float) * a / ev
    out["B1_EV"] = out["y_hat_pers"].astype(float) * a / ev
    out["div_yield"] = (out["div_ttm"].astype(float) * a / mc).fillna(0.0)
    return out


def pick_top(df: pd.DataFrame, col: str, n: int = BASKET_N) -> pd.DataFrame:
    """Top ``n`` by ``col`` descending, ties ticker ascending; NaN and inf signals are skipped."""
    d = df[np.isfinite(df[col].astype(float))]
    d = d.sort_values([col, "ticker"], ascending=[False, True], kind="mergesort")
    return d.head(n)


def equal_weights(tickers: Sequence[str]) -> dict[str, float]:
    n = len(tickers)
    return {t: 1.0 / n for t in tickers} if n else {}


def capped_weights(
    mcap_by_ticker: Mapping[str, float], cap: float = WEIGHT_CAP
) -> dict[str, float]:
    """Market-cap weights with a 10% cap (water-filling; same function as P6)."""
    return cap_weights(dict(mcap_by_ticker), cap)


def build_baskets(df: pd.DataFrame) -> pd.DataFrame:
    """All variant baskets of one tranche as a long frame (``BASKET_COLUMNS``).

    ``df`` is the stage-2 universe (``in_stage2``) with signals; the caller filters.
    """
    parts = []
    for name, (col, n, wt) in BASKET_VARIANTS.items():
        top = pick_top(df, col, n)
        if top.empty:
            continue
        tickers = top["ticker"].tolist()
        if wt == "cap":
            w = capped_weights(dict(zip(tickers, top["mcap_u"], strict=True)))
        else:
            w = equal_weights(tickers)
        part = pd.DataFrame(
            {
                "variant": name,
                "rank": np.arange(1, len(top) + 1),
                "ticker": tickers,
                "corp_code": top["corp_code"].to_numpy() if "corp_code" in top else None,
                "weight": [w[t] for t in tickers],
                "signal": top[col].to_numpy(dtype=float),
                "mcap_u": top["mcap_u"].to_numpy(dtype=float),
                "div_yield": top["div_yield"].to_numpy(dtype=float),
                "ksic2": top["ksic2"].to_numpy() if "ksic2" in top else None,
            }
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=BASKET_COLUMNS)
    return pd.concat(parts, ignore_index=True)[BASKET_COLUMNS]


def size_quintile(mcap: pd.Series) -> pd.Series:
    """1 (smallest) .. 5 (largest) by rank of company market cap over the universe."""
    r = mcap.rank(method="first")
    n = int(mcap.notna().sum())
    if n == 0:
        return pd.Series(np.nan, index=mcap.index)
    return np.ceil(r / n * 5).clip(1, 5)


def tranche_exposure(universe: pd.DataFrame, baskets: pd.DataFrame, ksic_section) -> dict:
    """Record-only exposures of the M and B1 baskets: size quintile, KSIC section, overlap."""
    uni = universe.copy()
    uni["_q"] = size_quintile(uni["mcap_u"])
    q = uni.set_index("ticker")["_q"]
    out: dict = {}
    sets: dict[str, set[str]] = {}
    for v in ("M", "B1"):
        b = baskets[baskets["variant"] == v]
        sets[v] = set(b["ticker"])
        out[f"{v}_size_quintile"] = {
            int(k): int(c) for k, c in b["ticker"].map(q).value_counts().sort_index().items()
        }
        sec = b["ksic2"].map(ksic_section)
        out[f"{v}_ksic_section"] = {
            str(k): int(c) for k, c in sec.value_counts().sort_index().items()
        }
        out[f"{v}_median_mcap_u"] = float(b["mcap_u"].median()) if len(b) else float("nan")
    out["overlap_M_B1"] = len(sets["M"] & sets["B1"])
    out["n_over_price_limit"] = None  # filled by the caller (needs raw closes)
    return out


def assemble(
    feats_t: pd.DataFrame,
    info_u: pd.DataFrame,
    co_mcap: pd.DataFrame,
    preds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Stage-1 input-eligible rows of one decision date joined to the signal-day facts.

    ``feats_t``: rows of one ``decision_date`` (``input_eligible`` rows are kept).
    ``info_u``: per ticker ``halted_u, p6_tradable_u, liq_notice_u, raw_close_u`` on the signal
    day (a ticker with no row is not listed). ``co_mcap``: :func:`company_mcap` of that day.
    ``preds``: ``ticker, y_hat_hgb, y_hat_pers`` (optional). Returns the flagged frame
    (:func:`flag_universe`), with signals when predictions are given.
    """
    df = feats_t[as_bool(feats_t["input_eligible"])].copy()
    info = info_u.assign(listed_u=True)
    keep = ["ticker", "listed_u", "halted_u", "p6_tradable_u", "liq_notice_u", "raw_close_u"]
    df = df.merge(info[keep], on="ticker", how="left")
    df["listed_u"] = as_bool(df["listed_u"])
    cm = co_mcap[
        ["ticker", "mcap_common", "pref_mcap", "mcap_total", "n_pref", "has_pref", "pref_ratio"]
    ]
    df = df.merge(cm.rename(columns={"mcap_total": "mcap_u"}), on="ticker", how="left")
    if preds is not None:
        df = df.merge(preds[["ticker", "y_hat_hgb", "y_hat_pers"]], on="ticker", how="left")
    else:
        df["y_hat_hgb"] = np.nan
        df["y_hat_pers"] = np.nan
    df = flag_universe(df, require_prediction=preds is not None)
    if preds is not None:
        df = add_signals(df)
    return df.sort_values("ticker").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def closes_frame(df: pd.DataFrame) -> pl.DataFrame:
    """``trade_date, ticker, close, is_halted`` as the polars frame ``holding_returns`` takes."""
    d = df[["trade_date", "ticker", "close", "is_halted"]].copy()
    d["trade_date"] = pd.to_datetime(d["trade_date"]).dt.date
    d["close"] = d["close"].astype(float)
    d["is_halted"] = d["is_halted"].astype(bool)
    return pl.from_pandas(d)


def name_status(
    closes: pl.DataFrame, tickers: Sequence[str], entry: date, exit_: date
) -> dict[str, dict[str, bool]]:
    """Per ticker: no usable entry close, series ended before exit (delisted), halted at exit."""
    rows = closes.filter(pl.col("trade_date") <= exit_).to_pandas()
    rows["trade_date"] = pd.to_datetime(rows["trade_date"]).dt.date
    out: dict[str, dict[str, bool]] = {}
    for t in tickers:
        g = rows[rows["ticker"] == t]
        entry_ok = bool(((g["trade_date"] == entry) & ~g["is_halted"].astype(bool)).any())
        last = g["trade_date"].max() if len(g) else None
        at_exit = g[g["trade_date"] == exit_]
        out[t] = {
            "entry_unusable": not entry_ok,
            "delisted": last is None or last < exit_,
            "halted_at_exit": bool(len(at_exit) and bool(at_exit["is_halted"].iloc[0])),
        }
    return out


def basket_return(
    closes: pl.DataFrame,
    entry: date,
    exit_: date,
    weights: Mapping[str, float],
    *,
    cost: float = COST_MAIN,
    extra: Mapping[str, float] | None = None,
) -> dict:
    """Weighted holding return of a basket (P6 ``holding_returns`` semantics) minus ``cost``.

    Entry-unusable names drop out and the weights renormalize; a name halted at exit or with a
    series that ends early keeps its last non-halted close, then cash. ``extra`` is an optional
    per-ticker yield added to the gross return (dividend approximation), weighted the same way.
    """
    tickers = list(weights)
    rets = holding_returns(closes, entry, exit_, tickers)
    use = [t for t in tickers if rets[t] is not None]
    wsum = float(sum(weights[t] for t in use))
    if not use or wsum <= 0:
        gross = float("nan")
        yld = float("nan")
    else:
        gross = float(sum(weights[t] * rets[t] for t in use) / wsum)
        yld = (
            float(sum(weights[t] * extra.get(t, 0.0) for t in use) / wsum)
            if extra is not None
            else 0.0
        )
    st = name_status(closes, tickers, entry, exit_)
    return {
        "gross": gross,
        "net": gross - cost,
        "div_yield": yld,
        "n_names": len(tickers),
        "n_usable": len(use),
        "n_entry_dropped": len(tickers) - len(use),
        "n_delisted": sum(1 for t in use if st[t]["delisted"]),
        "n_halted_at_exit": sum(1 for t in use if st[t]["halted_at_exit"]),
        "worst_name": float(min(rets[t] for t in use)) if use else float("nan"),
    }


def basket_max_drawdown(
    closes: pl.DataFrame,
    sessions: Sequence[date],
    entry: date,
    exit_: date,
    weights: Mapping[str, float],
) -> float:
    """Max drawdown of the buy-and-hold basket value path (start 1.0 on the entry session).

    A halted day or a series that ended keeps the last non-halted close (the same freeze as the
    return). Names unusable at entry drop out and the weights renormalize. No cost.
    """
    days = [s for s in _sorted(sessions) if entry <= s <= exit_]
    if len(days) < 2:
        return float("nan")
    df = closes.filter(
        (pl.col("trade_date") >= entry)
        & (pl.col("trade_date") <= exit_)
        & ~pl.col("is_halted")
        & pl.col("close").is_not_null()
    ).to_pandas()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    wide = df.pivot_table(index="trade_date", columns="ticker", values="close", aggfunc="last")
    wide = wide.reindex(days).ffill()
    use = [
        t
        for t in weights
        if t in wide.columns and pd.notna(wide[t].iloc[0]) and wide[t].iloc[0] > 0
    ]
    if not use:
        return float("nan")
    w = np.array([weights[t] for t in use], dtype=float)
    w = w / w.sum()
    rel = wide[use].to_numpy(dtype=float) / wide[use].iloc[0].to_numpy(dtype=float)
    value = rel @ w
    peak = np.maximum.accumulate(value)
    return float((value / peak - 1.0).min())


def index_return(close: pd.Series, entry: date, exit_: date) -> float:
    """Close on ``exit_`` over close on ``entry`` minus 1 (last value on or before each date)."""
    s = close.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().dropna()
    v0 = s.loc[: pd.Timestamp(entry)]
    v1 = s.loc[: pd.Timestamp(exit_)]
    if v0.empty or v1.empty or v0.iloc[-1] <= 0:
        return float("nan")
    return float(v1.iloc[-1] / v0.iloc[-1] - 1.0)


def etf_total_return(close: pd.Series, dist: pd.DataFrame, entry: date, exit_: date) -> float:
    """Price return of an ETF with every distribution reinvested at the record-date close.

    ``dist`` has ``rgt_std_date`` and ``per_share_amount``. Only record dates in
    ``(entry, exit_]`` count. Approximation: the real ex-date is a session earlier.
    """
    s = close.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index().dropna()
    v0 = s.loc[: pd.Timestamp(entry)]
    v1 = s.loc[: pd.Timestamp(exit_)]
    if v0.empty or v1.empty or v0.iloc[-1] <= 0:
        return float("nan")
    units = 1.0
    d = dist.copy()
    d["rgt_std_date"] = pd.to_datetime(d["rgt_std_date"])
    d = d.dropna(subset=["rgt_std_date", "per_share_amount"]).drop_duplicates(
        "rgt_std_date", keep="last"
    )
    for r in d.itertuples(index=False):
        if pd.Timestamp(entry) < r.rgt_std_date <= pd.Timestamp(exit_) and r.per_share_amount > 0:
            px = s.loc[: r.rgt_std_date]
            if len(px) and px.iloc[-1] > 0:
                units *= 1.0 + float(r.per_share_amount) / float(px.iloc[-1])
    return float(units * v1.iloc[-1] / v0.iloc[-1] - 1.0)


def score_baskets(
    baskets: pd.DataFrame,
    closes: pl.DataFrame,
    sessions: Sequence[date],
    entry: date,
    exit_: date,
    *,
    index_close: pd.Series,
    etf_close: pd.Series | None = None,
    etf_dist: pd.DataFrame | None = None,
) -> dict:
    """Returns of every variant basket of one tranche plus B0. One flat dict.

    Keys ``<variant>_net`` (65bp, primary), ``<variant>_net100`` / ``_net130`` (record),
    ``<variant>_gross``, counts, ``<variant>_mdd`` and for M and B1 the dividend-included value
    ``<variant>_net_div``. B0 keys: ``B0`` (price, no cost) and ``B0_tr`` (KODEX 200 + distributions).
    """
    row: dict = {"entry": entry, "exit": exit_}
    for v, g in baskets.groupby("variant", sort=False):
        w = dict(zip(g["ticker"], g["weight"], strict=True))
        yld = dict(zip(g["ticker"], g["div_yield"], strict=True))
        r = basket_return(closes, entry, exit_, w, cost=COST_MAIN, extra=yld)
        row[f"{v}_gross"] = r["gross"]
        row[f"{v}_net"] = r["net"]
        for c in COSTS_RECORD:
            row[f"{v}_net{int(round(c * 1e4))}"] = r["gross"] - c
        row[f"{v}_mdd"] = basket_max_drawdown(closes, sessions, entry, exit_, w)
        for k in (
            "n_names",
            "n_usable",
            "n_entry_dropped",
            "n_delisted",
            "n_halted_at_exit",
            "worst_name",
        ):
            row[f"{v}_{k}"] = r[k]
        if v in ("M", "B1"):
            row[f"{v}_net_div"] = r["gross"] + r["div_yield"] - COST_MAIN
    row["B0"] = index_return(index_close, entry, exit_)
    if etf_close is not None and etf_dist is not None:
        row["B0_tr"] = etf_total_return(etf_close, etf_dist, entry, exit_)
    return row


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def block_bootstrap_ci(
    values: Sequence[float],
    *,
    block: int = BLOCK_TRANCHES,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile interval of the mean, resampling non-overlapping year blocks.

    ``values`` are tranche values in time order; consecutive ``block`` of them form one block
    (a short last block stays a block). Blocks are drawn with replacement; each draw's statistic
    is the mean over all drawn tranches. Fixed seed, so the interval is reproducible.
    """
    v = np.asarray(values, dtype=float)
    if len(v) == 0 or np.isnan(v).any():
        return float("nan"), float("nan")
    cuts = list(range(0, len(v), block))
    sums = np.array([v[i : i + block].sum() for i in cuts])
    cnts = np.array([len(v[i : i + block]) for i in cuts], dtype=float)
    rng = np.random.default_rng(seed)
    nb = len(cuts)
    idx = rng.integers(0, nb, size=(draws, nb))
    stats = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize_tranches(
    tab: pd.DataFrame,
    *,
    m: str = "M_net",
    b1: str = "B1_net",
    b0: str = "B0",
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """Mean over tranches of M - B1, M - B0, B1 - B0 with year-block bootstrap intervals.

    ``tab`` is one row per tranche (time order) with the return columns; rows missing any of
    the three values are dropped and counted.
    """
    t = tab.sort_values("decision_date") if "decision_date" in tab.columns else tab
    n_all = len(t)
    t = t.dropna(subset=[m, b1, b0])
    out: dict = {"n_tranches": int(len(t)), "n_dropped_incomplete": int(n_all - len(t))}
    pairs = {"M-B1": (m, b1), "M-B0": (m, b0), "B1-B0": (b1, b0)}
    for name, (a, b) in pairs.items():
        d = (t[a] - t[b]).to_numpy(dtype=float)
        if len(d) == 0:
            out[name] = {
                "mean": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "n": 0,
            }
            continue
        lo, hi = block_bootstrap_ci(d, draws=draws, seed=seed)
        out[name] = {
            "mean": float(d.mean()),
            "ci_low": lo,
            "ci_high": hi,
            "n": int(len(d)),
            "share_positive": float((d > 0).mean()),
        }
    out["seed"] = seed
    out["draws"] = draws
    out["block"] = BLOCK_TRANCHES
    return out


def decay_ratio(forward_mean: float, past_mean: float) -> float:
    """Forward mean divided by the past mean (NaN when the past mean is 0 or missing)."""
    if past_mean is None or forward_mean is None or math.isnan(past_mean) or past_mean == 0:
        return float("nan")
    return float(forward_mean / past_mean)
