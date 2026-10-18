"""Judgment computations for the next KR model (19 / spec G-3..G-9, G-18, G-24, C-5).

Plan: ``my/milestones/common/20261012_feature_explainer/19_*`` (prereg draft) and the C4 spec
(C-5 check rule, G-3 grid, G-5 basket, G-6 turnover, G-7 tail cell, G-9 G computation,
G-18 random basket, G-24 cost).

Pure functions over plain dicts, lists and small polars frames. No lake access, no file I/O,
no label reading: callers pass already assembled inputs. Every function is deterministic.

Pieces, in order:

1. **Grid** — :func:`grid_dates` (G-3), :func:`check_dates` (C-5, proposal).
2. **Basket** — :func:`cap_weights`, :func:`build_basket` (G-5), :func:`drift`,
   :func:`one_way_turnover`, :func:`turnover_warning` (G-6).
3. **G point** — :func:`holding_returns` (G-9), :func:`g_point` (G-9, G-24).
4. **Tail cell** — :func:`tail_flags`, :func:`tail_point` (G-7).
5. **Random basket** — :func:`random_basket` (G-18).
6. **Rank IC** — :func:`rank_ic` (C-5).
7. **Check rule** — :func:`check_decision` (C-5).

Conventions fixed here (spec ambiguities, see the module-level constants):

* Percent rank is ``(rank - 1) / (n - 1)`` with average ties over the non-null values of one
  ``(trade_date, market)`` group. ``n == 1`` gives NULL (not tail).
* Weights are fractions of NAV and sum to 1. Names without a usable return are excluded from a
  point and the remaining weights are renormalized; the excluded weight is reported.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import polars as pl
from scipy import stats

from modeler.etl.label_gate import DEFAULT_ALLOWED_KINDS

GRID_STEP = 20
BASKET_K = 30
BASKET_K_OUT = 90
WEIGHT_CAP = 0.10
COST_ROUND_TRIP = 0.0065  # G-24: 65bp round trip, applied to one-way turnover
TURNOVER_WINDOW = 6
TURNOVER_THRESHOLD = 0.5
TAIL_TOP_PCT = 0.9
TAIL_BOTTOM_PCT = 0.1
MIN_POINTS = 6  # C-5: fewer matured points -> "표본 부족"
SIGNIFICANCE_NS = (24, 36)
ALPHA = 0.025
_PCT_EPS = 1e-12

STATUS_INSUFFICIENT = "표본 부족"
STATUS_CONTINUE = "계속"
STATUS_UNDECIDED = "미정"
STATUS_STOP_PROPOSAL = "중단 제안"
STATUS_HOLD = "판정 보류"
TAIL_NOT_CONFIRMED = "가설 방향 미확인"
TAIL_DIRECTION_OK = "가설 방향 일치"


# ---------------------------------------------------------------------------
# 1. grid
# ---------------------------------------------------------------------------


def grid_dates(sessions: Sequence[date], d0: date, step: int = GRID_STEP) -> list[date]:
    """Grid dates ``D0 + step*k`` sessions, ``k = 0, 1, ...`` (19 / spec G-3).

    ``sessions`` are the KOSPI sessions in ascending order; ``d0`` must be one of them.
    Only dates present in ``sessions`` are returned.
    """
    ordered = sorted(sessions)
    if d0 not in ordered:
        raise ValueError(f"d0 {d0} is not a session")
    if step < 1:
        raise ValueError("step must be >= 1")
    return ordered[ordered.index(d0) :: step]


def check_dates(
    start_year: int = 2027,
    years: int = 5,
    sessions: Sequence[date] | None = None,
) -> list[date]:
    """Check dates: every January and July from ``start_year`` (19 / spec C-5).

    Day convention (main proposal): the 15th of the month, or the next session when the 15th
    is not one. Without ``sessions`` the plain 15th is returned. The project check dates
    2027-01 and 2027-07 are the first two.
    """
    ordered = sorted(sessions) if sessions is not None else None
    out: list[date] = []
    for year in range(start_year, start_year + years):
        for month in (1, 7):
            day = date(year, month, 15)
            if ordered is not None:
                nxt = [s for s in ordered if s >= day]
                if not nxt:
                    continue
                day = nxt[0]
            out.append(day)
    return out


# ---------------------------------------------------------------------------
# 2. basket
# ---------------------------------------------------------------------------


def _as_mapping(mcap: Mapping[str, float] | pl.Series) -> dict[str, float]:
    if isinstance(mcap, pl.Series):
        return {str(i): v for i, v in enumerate(mcap.to_list())}
    return dict(mcap)


def cap_weights(mcap: Mapping[str, float] | pl.Series, cap: float = WEIGHT_CAP) -> dict[str, float]:
    """Market-cap weights capped at ``cap`` by water-filling (19 / spec G-5).

    Names above the cap are fixed at ``cap`` and the excess goes to the rest in proportion to
    market cap, repeated until nothing exceeds. The weights sum to 1. When ``n * cap < 1``
    (fewer than 10 names at 10%) the cap cannot hold: equal weights ``1/n`` are returned.
    NULL or non-positive market caps get weight 0; if no positive value exists, equal weights.
    A ``pl.Series`` is keyed by position (``"0"``, ``"1"``, ...).
    """
    caps = _as_mapping(mcap)
    n = len(caps)
    if n == 0:
        return {}
    if n * cap < 1.0 - 1e-12:
        return {k: 1.0 / n for k in caps}
    vals = {
        k: (float(v) if v is not None and not math.isnan(float(v)) and v > 0 else 0.0)
        for k, v in caps.items()
    }
    if sum(vals.values()) <= 0:
        return {k: 1.0 / n for k in caps}
    fixed: dict[str, float] = {}
    free = dict(vals)
    while True:
        budget = 1.0 - cap * len(fixed)
        total = sum(free.values())
        if total <= 0:
            # free names all have zero cap-weight: give the leftover to nobody, spread equally
            share = budget / len(free) if free else 0.0
            w = {k: share for k in free}
            break
        w = {k: budget * v / total for k, v in free.items()}
        over = [k for k, x in w.items() if x > cap + 1e-15]
        if not over:
            break
        for k in over:
            fixed[k] = cap
            del free[k]
    out = {**fixed, **w}
    return {k: out[k] for k in caps}


def build_basket(
    prev: Mapping[str, float] | None,
    candidates: pl.DataFrame,
    k: int = BASKET_K,
    k_out: int = BASKET_K_OUT,
    cap: float = WEIGHT_CAP,
) -> dict[str, float]:
    """Target weights of the 30-name basket with the buffer rule (19 / spec G-5).

    ``candidates`` has ``ticker, score, tv20, mcap`` (already filtered to the scored
    universe). Rank = score desc, tv20 desc, ticker asc (1-based). Previous holdings with rank
    ``<= k_out`` stay (best rank first, at most ``k``); the rest is filled with the best
    ranked non-held names up to ``k``. With fewer than ``k`` candidates all are held.
    Weights are :func:`cap_weights` of the chosen names' market cap. ``prev=None`` is the
    first build. Empty candidates give an empty basket.
    """
    if candidates.height == 0:
        return {}
    ranked = candidates.sort(
        ["score", "tv20", "ticker"], descending=[True, True, False], nulls_last=True
    )
    tickers = ranked["ticker"].to_list()
    if len(tickers) <= k:
        chosen = tickers
    else:
        held = set(prev or {})
        keep = [t for t in tickers[:k_out] if t in held][:k]
        kept = set(keep)
        fill = [t for t in tickers if t not in kept][: k - len(keep)]
        chosen = keep + fill
    mcap = dict(zip(ranked["ticker"].to_list(), ranked["mcap"].to_list(), strict=True))
    return cap_weights({t: mcap[t] for t in chosen}, cap)


def drift(weights: Mapping[str, float], returns: Mapping[str, float | None]) -> dict[str, float]:
    """Weights after one holding period: ``w_i (1 + r_i)`` renormalized (19 / spec G-5).

    A name without a return (missing or None) is carried at 0%. Frozen (delisted) names keep
    their frozen return, so the total stays invested.
    """
    grown = {k: w * (1.0 + (returns.get(k) or 0.0)) for k, w in weights.items()}
    total = sum(grown.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: v / total for k, v in grown.items()}


def one_way_turnover(target: Mapping[str, float], drifted: Mapping[str, float] | None) -> float:
    """One-way turnover ``sum |w_target - w_drifted| / 2`` over the union (19 / spec G-5, G-6).

    ``drifted=None`` is the first build and counts as 1.0.
    """
    if drifted is None:
        return 1.0
    names = set(target) | set(drifted)
    return sum(abs(target.get(n, 0.0) - drifted.get(n, 0.0)) for n in names) / 2.0


def turnover_warning(
    turnovers: Sequence[float],
    window: int = TURNOVER_WINDOW,
    threshold: float = TURNOVER_THRESHOLD,
) -> bool:
    """True when the mean of the last ``window`` turnovers exceeds ``threshold`` (19 / spec G-6).

    Fewer than ``window`` points: the mean of what exists. Record only, never a stop.
    """
    if not turnovers:
        return False
    tail = list(turnovers)[-window:]
    return sum(tail) / len(tail) > threshold


# ---------------------------------------------------------------------------
# 3. G point
# ---------------------------------------------------------------------------


def holding_returns(
    closes: pl.DataFrame,
    start: date,
    end: date,
    tickers: Sequence[str] | None = None,
) -> dict[str, float | None]:
    """Holding return per ticker from ``start`` close to ``end`` close (19 / spec G-9).

    ``closes`` has ``trade_date, ticker, close, is_halted`` (one lake vintage, adjusted).

    * halted at ``end``, or the series ends before ``end`` (delisted): the last non-halt close
      on or before ``end`` is the exit value, then cash, so the return is frozen;
    * no non-halt close at ``start`` (untradable at entry) or ticker unknown: ``None``
      (excluded, weights renormalized by :func:`g_point`).

    ``tickers`` restricts and completes the result (unknown tickers map to None).
    """
    usable = closes.filter(
        (pl.col("trade_date") <= end) & ~pl.col("is_halted") & pl.col("close").is_not_null()
    )
    entry = {
        r["ticker"]: r["close"] for r in usable.filter(pl.col("trade_date") == start).to_dicts()
    }
    exit_ = {
        r["ticker"]: r["close"]
        for r in usable.sort("trade_date").group_by("ticker", maintain_order=True).last().to_dicts()
    }
    names = list(tickers) if tickers is not None else closes["ticker"].unique().sort().to_list()
    out: dict[str, float | None] = {}
    for t in names:
        c0 = entry.get(t)
        c1 = exit_.get(t)
        out[t] = None if c0 is None or c1 is None or c0 <= 0 else c1 / c0 - 1.0
    return out


def g_point(
    target: Mapping[str, float],
    holding_ret: Mapping[str, float | None],
    bench_ret: float,
    turnover: float,
    cost_rt: float = COST_ROUND_TRIP,
) -> dict[str, float | int | None]:
    """G of one grid point: basket excess over KOSPI 200 after cost (19 / spec G-9, G-24).

    ``gross`` is the weighted return over names that have a return (weights renormalized);
    ``excluded_weight`` is the target weight of the rest, ``n_excluded`` their count.
    ``cost = cost_rt * turnover`` (index holding cost is 0). ``g = gross - bench - cost``.
    No usable name: ``gross`` and ``g`` are None.
    """
    ok = {k: w for k, w in target.items() if holding_ret.get(k) is not None}
    w_ok = sum(ok.values())
    excluded = sum(target.values()) - w_ok
    cost = cost_rt * turnover
    gross: float | None = None
    g: float | None = None
    if ok and w_ok > 0:
        gross = sum(w * holding_ret[k] for k, w in ok.items()) / w_ok  # type: ignore[operator]
        g = gross - bench_ret - cost
    return {
        "gross": gross,
        "bench": bench_ret,
        "cost": cost,
        "g": g,
        "excluded_weight": excluded,
        "n_excluded": len(target) - len(ok),
    }


# ---------------------------------------------------------------------------
# 4. tail cell
# ---------------------------------------------------------------------------


def _pct_rank(col: str, keys: Sequence[str]) -> pl.Expr:
    rank = pl.col(col).rank("average").over(list(keys))
    n = pl.col(col).count().over(list(keys))
    return pl.when(n > 1).then((rank - 1) / (n - 1)).otherwise(None)


def tail_flags(df: pl.DataFrame) -> pl.DataFrame:
    """Add the tail flags of the avoid-tail cell (19 / spec G-7).

    Input columns: ``trade_date, market, ticker, px_vol_20d_v2, px_idio_vol_60d, fin_roa_v2,
    fin_capital_impaired_v2``. Within ``(trade_date, market)`` among non-null values the
    percent rank is ``(rank - 1) / (n - 1)`` with average ties.

    * ``tail1`` = vol pct >= 0.9 or idio-vol pct >= 0.9;
    * ``tail2`` = ROA pct <= 0.1 or ``fin_capital_impaired_v2`` is TRUE;
    * ``tail`` = ``tail1 | tail2``. NULL is never tail (flags are False, not NULL).

    The ``*_pct`` columns are added too (the shadow record keeps the inputs).
    """
    keys = ("trade_date", "market")
    out = df.with_columns(
        _pct_rank("px_vol_20d_v2", keys).alias("px_vol_20d_v2_pct"),
        _pct_rank("px_idio_vol_60d", keys).alias("px_idio_vol_60d_pct"),
        _pct_rank("fin_roa_v2", keys).alias("fin_roa_v2_pct"),
    )
    top = TAIL_TOP_PCT - _PCT_EPS
    bottom = TAIL_BOTTOM_PCT + _PCT_EPS
    out = out.with_columns(
        (
            (pl.col("px_vol_20d_v2_pct") >= top).fill_null(False)
            | (pl.col("px_idio_vol_60d_pct") >= top).fill_null(False)
        ).alias("tail1"),
        (
            (pl.col("fin_roa_v2_pct") <= bottom).fill_null(False)
            | (pl.col("fin_capital_impaired_v2") == True).fill_null(False)  # noqa: E712
        ).alias("tail2"),
    )
    return out.with_columns((pl.col("tail1") | pl.col("tail2")).alias("tail"))


def tail_point(
    df: pl.DataFrame,
    allowed_kinds: Sequence[str] = DEFAULT_ALLOWED_KINDS,
) -> dict[str, object]:
    """Tail cell value of one grid point: ``mean(tail) - mean(rest)`` (19 / spec G-7).

    ``df`` is the scored candidates of one point with ``tail`` (from :func:`tail_flags`),
    ``fwd_ret_c1_20d`` and ``label_exit_kind``. Equal-weight mean per group over rows whose
    kind is in ``allowed_kinds`` (regular, late, terminal, terminal_halted) and whose return is
    not NULL. Other rows are counted per group per kind in ``excluded_tail`` /
    ``excluded_rest`` (NULL kind is counted as ``"null"``, an allowed kind with a NULL
    return as ``"null_return"``). ``diff`` is None when a group is empty.
    """
    allowed = set(allowed_kinds)
    result: dict[str, object] = {}
    means: dict[str, float | None] = {}
    for name, flag in (("tail", True), ("rest", False)):
        grp = df.filter(pl.col("tail") == flag)
        excluded: dict[str, int] = {}
        vals: list[float] = []
        for r in grp.select("fwd_ret_c1_20d", "label_exit_kind").iter_rows():
            ret, kind = r
            if kind not in allowed:
                key = "null" if kind is None else str(kind)
            elif ret is None or (isinstance(ret, float) and math.isnan(ret)):
                key = "null_return"
            else:
                vals.append(float(ret))
                continue
            excluded[key] = excluded.get(key, 0) + 1
        means[name] = sum(vals) / len(vals) if vals else None
        result[f"n_{name}"] = len(vals)
        result[f"excluded_{name}"] = excluded
        result[f"{name}_mean"] = means[name]
    result["diff"] = (
        means["tail"] - means["rest"]  # type: ignore[operator]
        if means["tail"] is not None and means["rest"] is not None
        else None
    )
    return result


# ---------------------------------------------------------------------------
# 5. random basket
# ---------------------------------------------------------------------------


def date_seed(d: date) -> int:
    """Seed of the random basket: ``int(sha256(date.isoformat()) hex[:8], 16)`` (19 / spec G-18)."""
    return int(hashlib.sha256(d.isoformat().encode()).hexdigest()[:8], 16)


def random_basket(
    candidates: pl.DataFrame,
    date: date,  # noqa: A002 - public name from the spec
    holding_ret: Mapping[str, float | None],
    bench_ret: float,
    model_excess: float | None = None,
    n_draws: int = 1000,
    k: int = BASKET_K,
    cap: float = WEIGHT_CAP,
) -> dict[str, object]:
    """Random 30-name baskets from the same candidates (19 / spec G-18).

    ``candidates`` has ``ticker, mcap``. Each draw picks ``k`` names without replacement (no
    buffer; all names when fewer than ``k``), weights them with :func:`cap_weights`, and
    takes the gross return minus ``bench_ret`` (names without a return are dropped and
    weights renormalized, as for the model basket). ``seed = date_seed(date)``.

    Returns ``excess`` (draw array), ``mean``, ``sd`` and, when ``model_excess`` (the model
    basket's gross return minus ``bench_ret``) is given, its ``percentile`` among the draws
    (share below plus half the ties, in [0, 1]).
    """
    tickers = candidates["ticker"].to_list()
    mcaps = candidates["mcap"].to_list()
    n = len(tickers)
    rng = np.random.default_rng(date_seed(date))
    size = min(k, n)
    excess: list[float] = []
    for _ in range(n_draws):
        idx = rng.choice(n, size=size, replace=False) if n else []
        w = cap_weights({tickers[i]: mcaps[i] for i in idx}, cap)
        ok = {t: x for t, x in w.items() if holding_ret.get(t) is not None}
        tot = sum(ok.values())
        if tot <= 0:
            continue
        gross = sum(x * holding_ret[t] for t, x in ok.items()) / tot  # type: ignore[operator]
        excess.append(gross - bench_ret)
    arr = np.array(excess, dtype=float)
    out: dict[str, object] = {
        "seed": date_seed(date),
        "n_draws": int(arr.size),
        "excess": arr,
        "mean": float(arr.mean()) if arr.size else None,
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
        "percentile": None,
    }
    if model_excess is not None and arr.size:
        out["percentile"] = float((arr < model_excess).mean() + 0.5 * (arr == model_excess).mean())
    return out


# ---------------------------------------------------------------------------
# 6. rank IC
# ---------------------------------------------------------------------------


def rank_ic(scores: Sequence[float | None], labels: Sequence[float | None]) -> float | None:
    """Spearman correlation of one date's scores and labels, average ties (19 / C-5).

    Pairs with a NULL/NaN on either side are dropped. Fewer than 3 pairs or a constant side
    gives None.
    """
    pairs = [
        (s, y)
        for s, y in zip(scores, labels, strict=True)
        if s is not None and y is not None and not math.isnan(s) and not math.isnan(y)
    ]
    if len(pairs) < 3:
        return None
    s = pl.Series([p[0] for p in pairs]).rank("average").to_numpy()
    y = pl.Series([p[1] for p in pairs]).rank("average").to_numpy()
    if np.ptp(s) == 0 or np.ptp(y) == 0:
        return None
    return float(np.corrcoef(s, y)[0, 1])


# ---------------------------------------------------------------------------
# 7. check rule
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _t_test(values: Sequence[float], alternative: str) -> dict[str, float | int | bool | None]:
    """One-sided one-sample t-test against 0 (``greater``: H1 mean > 0, ``less``: H1 mean < 0)."""
    n = len(values)
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        t = math.copysign(math.inf, mean) if mean != 0 else 0.0
    else:
        t = mean / (sd / math.sqrt(n))
    p = float(stats.t.sf(t, n - 1) if alternative == "greater" else stats.t.cdf(t, n - 1))
    return {"n": n, "mean": mean, "sd": sd, "t": t, "p": p, "significant": p < ALPHA}


def _significance(
    pts: pl.DataFrame, col: str, alternative: str, check_date: date
) -> dict[str, object]:
    # A recorded point whose value for this cell is NULL/NaN counts as a missing point for the
    # cell (like an unrecorded grid date, G-4): the test takes the first n points that have a
    # value, so n stays 24/36 and one NULL cannot hold the test forever.
    out: dict[str, object] = {}
    ordered = pts.sort("grid_date")
    has_value = [v is not None and not math.isnan(v) for v in ordered[col].to_list()]
    valued = ordered.filter(pl.Series(has_value, dtype=pl.Boolean))
    for n in SIGNIFICANCE_NS:
        first = valued.head(n)
        matured = first.height == n and bool((first["matured_at"] <= check_date).all())
        if not matured:
            out[f"n{n}"] = {"status": STATUS_HOLD}
        else:
            last = first["grid_date"][-1]
            dates = ordered["grid_date"].to_list()
            skipped = sum(1 for d, ok in zip(dates, has_value) if d <= last and not ok)
            out[f"n{n}"] = {
                "status": "검정",
                "n_skipped_null": skipped,
                **_t_test(first[col].to_list(), alternative),
            }
    return out


def check_decision(points: pl.DataFrame, check_date: date) -> dict[str, object]:
    """Check-date decision exactly as C-5 (19 / spec C-5).

    ``points``: ``grid_date, matured_at, g, tail_diff, rank_ic`` (one row per recorded grid
    point; missing points are simply absent). Only points with ``matured_at <= check_date``
    count. All numbers used are in the returned dict.

    * ``n < 6`` matured points: model and tail ``표본 부족``;
    * model: ``중단 제안`` iff mean(G) <= 0 and mean(rank IC) <= 0; ``미정`` if exactly one is
      <= 0 (a mean that cannot be computed counts as undecided); else ``계속``;
    * tail: mean(tail - rest) >= 0 gives ``가설 방향 미확인`` (never stops the model);
    * significance: one-sided t-tests on the FIRST 24 and FIRST 36 points by ``grid_date``
      (G: H1 mean > 0; tail: H1 mean < 0), alpha 0.025 per cell, only when all of those
      points have matured by ``check_date``, else ``판정 보류``. A point whose value for the
      cell is NULL counts as a missing point for that cell (``n_skipped_null``). Beyond 36
      points nothing new is tested (record only until a new preregistration).
    """
    matured = points.filter(pl.col("matured_at") <= check_date).sort("grid_date")
    n = matured.height

    def clean(col: str) -> list[float]:
        return [v for v in matured[col].to_list() if v is not None and not math.isnan(v)]

    mean_g = _mean(clean("g"))
    mean_ic = _mean(clean("rank_ic"))
    mean_tail = _mean(clean("tail_diff"))
    result: dict[str, object] = {
        "check_date": check_date,
        "n_matured": n,
        "mean_g": mean_g,
        "mean_rank_ic": mean_ic,
        "mean_tail_diff": mean_tail,
        "alpha": ALPHA,
    }
    if n < MIN_POINTS:
        result["model"] = STATUS_INSUFFICIENT
        result["stop_proposal"] = False
        result["tail"] = STATUS_INSUFFICIENT
    else:
        if mean_g is None or mean_ic is None:
            model = STATUS_UNDECIDED
        else:
            neg = (mean_g <= 0) + (mean_ic <= 0)
            model = (
                STATUS_STOP_PROPOSAL
                if neg == 2
                else STATUS_UNDECIDED
                if neg == 1
                else STATUS_CONTINUE
            )
        result["model"] = model
        result["stop_proposal"] = model == STATUS_STOP_PROPOSAL
        result["tail"] = (
            TAIL_NOT_CONFIRMED if mean_tail is None or mean_tail >= 0 else TAIL_DIRECTION_OK
        )
    # significance counts the points by grid_date, and needs them matured by check_date
    sig_points = points.sort("grid_date")
    result["significance_g"] = _significance(sig_points, "g", "greater", check_date)
    result["significance_tail"] = _significance(sig_points, "tail_diff", "less", check_date)
    return result
