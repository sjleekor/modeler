"""metrics — ranking-centric evaluation for the return-rank model (etl_00 §6).

For stock selection, "did the names we ranked high actually do well?" matters
more than pointwise RMSE (etl_00 §6). The primary metrics are computed PER DATE
(cross-sectional) and then averaged across dates:

  - rank_ic            : Spearman corr(prediction, realized) within each date,
                         averaged over dates (the headline metric).
  - icir               : mean(rank_ic) / std(rank_ic) — information ratio of IC.
  - top_decile_spread  : mean realized excess of the top-decile-predicted names.
  - top_minus_bottom   : Q-top mean minus Q-bottom mean realized excess (Q5-Q1).
  - hit_ratio_top      : fraction of top-quantile names with positive realized excess.

A probability model needs a second half of this (20260907_model_experiment `03`
§2.1): a ranking is judged by whether the order was right, a probability also by
whether the *number* was right. :func:`classification_report`,
:func:`reliability_table` and :func:`threshold_economic_report` add log-loss,
Brier, AUC, ECE, precision/lift@k and the ``p >= tau`` portfolio to the same
per-date-then-average discipline.

All functions take per-row arrays/columns plus a date key for the per-date
grouping. Pure numpy/polars; sklearn is not needed here.

See ``etl_00`` §6, ``00_shared`` §3.3, and ``etl_03_implementation_plan.md`` §4 (P6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class RankICReport:
    """Aggregated walk-forward evaluation (etl_00 §6)."""

    n_dates: int
    n_obs: int
    rank_ic_mean: float
    rank_ic_std: float
    icir: float
    rank_ic_tstat: float
    top_decile_spread: float
    top_minus_bottom: float
    hit_ratio_top: float

    def as_dict(self) -> dict:
        return {
            "n_dates": self.n_dates,
            "n_obs": self.n_obs,
            "rank_ic_mean": self.rank_ic_mean,
            "rank_ic_std": self.rank_ic_std,
            "icir": self.icir,
            "rank_ic_tstat": self.rank_ic_tstat,
            "top_decile_spread": self.top_decile_spread,
            "top_minus_bottom": self.top_minus_bottom,
            "hit_ratio_top": self.hit_ratio_top,
        }


def per_date_market_rank_ic(
    df: pl.DataFrame,
    *,
    pred_col: str,
    realized_col: str,
    date_col: str = "trade_date",
    market_col: str = "market",
    min_names: int = 2,
    engine: str = "legacy",
) -> pl.DataFrame:
    """Spearman IC at the preregistered ``(date, market)`` unit.

    ``legacy`` remains available as the parity reference.  The native engine
    uses Polars' Rust expression path and fixes output order for downstream
    gap-aware HAC calculations.
    """
    if engine == "polars_native_v1":
        return _per_date_market_rank_ic_native(
            df,
            pred_col=pred_col,
            realized_col=realized_col,
            date_col=date_col,
            market_col=market_col,
            min_names=min_names,
        )
    if engine != "legacy":
        raise ValueError(f"unknown rank IC engine: {engine!r}")
    clean = df.select([date_col, market_col, pred_col, realized_col]).drop_nulls()
    clean = clean.filter(pl.col(pred_col).is_finite() & pl.col(realized_col).is_finite())
    out: list[tuple[object, object, float, int]] = []
    for (d, market), grp in clean.group_by([date_col, market_col], maintain_order=True):
        if grp.height < min_names:
            continue
        out.append(
            (
                d,
                market,
                _spearman(grp[pred_col].to_numpy(), grp[realized_col].to_numpy()),
                grp.height,
            )
        )
    return pl.DataFrame(
        out,
        schema={
            date_col: clean.schema[date_col],
            market_col: clean.schema[market_col],
            "rank_ic": pl.Float64,
            "n": pl.Int64,
        },
        orient="row",
    )


def _per_date_market_rank_ic_native(
    df: pl.DataFrame,
    *,
    pred_col: str,
    realized_col: str,
    date_col: str,
    market_col: str,
    min_names: int,
) -> pl.DataFrame:
    """Native date×market Spearman implementation used by Horizon Scan.

    Polars' Spearman correlation uses average ranks for ties, matching
    :func:`_spearman` — but *only* where at least one side actually varies.
    On a cross-section whose predictor (or realized value) is completely
    constant it returns a spurious non-zero correlation instead of a NaN, for
    some group sizes and not others (measured on polars 1.41.2: NaN at n=50,
    a number at n=745). A constant column carries no rank information, so the
    correlation is undefined and the date must drop out; the degeneracy guard
    below restores that, matching :func:`_spearman`'s own ``ps == 0`` check.

    This mattered in practice: a count feature's history begins with whole
    cross-sections at zero, and without the guard those dates entered the IC
    series with invented values (``10_known_issues.md`` I13).

    Null/non-finite filtering is deliberately kept before the group-by so the
    native and legacy paths have the same eligibility contract.
    """
    clean = df.select([date_col, market_col, pred_col, realized_col]).drop_nulls()
    clean = clean.filter(pl.col(pred_col).is_finite() & pl.col(realized_col).is_finite())
    if clean.is_empty():
        return pl.DataFrame(
            {
                date_col: pl.Series([], dtype=df.schema[date_col]),
                market_col: pl.Series([], dtype=df.schema[market_col]),
                "rank_ic": pl.Series([], dtype=pl.Float64),
                "n": pl.Series([], dtype=pl.Int64),
            }
        )
    return (
        clean.group_by([date_col, market_col])
        .agg(
            pl.corr(pred_col, realized_col, method="spearman").alias("rank_ic"),
            pl.len().alias("n"),
            pl.col(pred_col).n_unique().alias("_pred_unique"),
            pl.col(realized_col).n_unique().alias("_realized_unique"),
        )
        .filter(pl.col("n") >= min_names)
        # The legacy path keeps degenerate groups as a row with NaN and lets
        # callers decide whether to drop it. Match that contract instead of
        # silently changing the valid date×market population in the native
        # path (Polars represents the same correlation as null) — and force
        # the NaN where a constant column makes the correlation undefined,
        # which Polars does not always do on its own.
        .with_columns(
            pl.when((pl.col("_pred_unique") <= 1) | (pl.col("_realized_unique") <= 1))
            .then(float("nan"))
            .otherwise(pl.col("rank_ic").fill_null(float("nan")))
            .alias("rank_ic")
        )
        .drop(["_pred_unique", "_realized_unique"])
        .with_columns(pl.col("n").cast(pl.Int64))
        .sort([date_col, market_col])
    )


def daily_market_weighted_ic(
    market_ic: pl.DataFrame,
    *,
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """Collapse market ICs to one n-observation-weighted IC per date."""
    if market_ic.is_empty():
        return pl.DataFrame({date_col: [], "rank_ic": [], "n": []})
    return (
        market_ic.with_columns((pl.col("rank_ic") * pl.col("n")).alias("weighted_ic"))
        .group_by(date_col, maintain_order=True)
        .agg(
            (pl.col("weighted_ic").sum() / pl.col("n").sum()).alias("rank_ic"),
            pl.col("n").sum().alias("n"),
        )
    )


def market_weight_means(
    market_ic: pl.DataFrame,
    *,
    date_col: str = "trade_date",
    market_col: str = "market",
) -> dict[str, float]:
    """Mean KOSPI/KOSDAQ share of ``n`` across valid daily_ic dates (§4.1).

    A date where only one market is valid gets weight 1 for that market —
    daily IC is n-weighted, so this discloses how much an "overall" result is
    actually a KOSDAQ (or KOSPI) result in disguise.
    """
    if market_ic.is_empty():
        return {"kospi_weight_mean": float("nan"), "kosdaq_weight_mean": float("nan")}
    wide = market_ic.pivot(on=market_col, index=date_col, values="n").fill_null(0.0)
    kospi = wide["KOSPI"].to_numpy() if "KOSPI" in wide.columns else np.zeros(wide.height)
    kosdaq = wide["KOSDAQ"].to_numpy() if "KOSDAQ" in wide.columns else np.zeros(wide.height)
    total = kospi + kosdaq
    valid = total > 0
    if not valid.any():
        return {"kospi_weight_mean": float("nan"), "kosdaq_weight_mean": float("nan")}
    kospi_weight = kospi[valid] / total[valid]
    return {
        "kospi_weight_mean": float(kospi_weight.mean()),
        "kosdaq_weight_mean": float(1.0 - kospi_weight.mean()),
    }


def per_date_market_quantile_spread(
    df: pl.DataFrame,
    *,
    feature_col: str,
    raw_label_col: str,
    date_col: str = "trade_date",
    market_col: str = "market",
    n_quantiles: int = 5,
    min_names: int = 50,
) -> pl.DataFrame:
    """Per ``(date, market)`` equal-weighted Q-top minus Q-bottom raw spread.

    §4.3 steps 1-2: rank the *raw* feature (average-rank ties, matching
    :func:`per_date_market_rank_ic`'s Spearman convention) within each
    date×market cross-section, then difference the top/bottom quantile's
    equal-weighted raw excess return. A cross-section under ``min_names`` is
    dropped entirely (not zero-filled).
    """
    columns = (
        [date_col, market_col, feature_col]
        if feature_col == raw_label_col
        else [date_col, market_col, feature_col, raw_label_col]
    )
    clean = df.select(columns).drop_nulls()
    clean = clean.filter(pl.col(feature_col).is_finite() & pl.col(raw_label_col).is_finite())
    rows: list[tuple[object, object, float, int]] = []
    for (d, m), grp in clean.group_by([date_col, market_col], maintain_order=True):
        if grp.height < min_names:
            continue
        feature = grp[feature_col].to_numpy()
        realized = grp[raw_label_col].to_numpy()
        rank = _rankdata(feature) / feature.size
        top = realized[rank >= 1 - 1 / n_quantiles]
        bottom = realized[rank <= 1 / n_quantiles]
        if top.size and bottom.size:
            rows.append((d, m, float(top.mean() - bottom.mean()), grp.height))
    return pl.DataFrame(
        rows,
        schema={
            date_col: clean.schema[date_col],
            market_col: clean.schema[market_col],
            "spread": pl.Float64,
            "n": pl.Int64,
        },
        orient="row",
    )


def daily_market_weighted_spread(
    market_spread: pl.DataFrame,
    *,
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """Collapse per-market Q-top-minus-bottom spreads to one n-weighted spread
    per date (§4.3 step 3) — mirrors :func:`daily_market_weighted_ic`."""
    if market_spread.is_empty():
        return pl.DataFrame({date_col: [], "spread": [], "n": []})
    return (
        market_spread.with_columns((pl.col("spread") * pl.col("n")).alias("weighted"))
        .group_by(date_col, maintain_order=True)
        .agg(
            (pl.col("weighted").sum() / pl.col("n").sum()).alias("spread"),
            pl.col("n").sum().alias("n"),
        )
    )


def newey_west_tstat(
    values: np.ndarray | list[float],
    session_index: np.ndarray | list[int],
    lag: int,
) -> float:
    """Gap-aware HAC t-statistic using a vectorized pair lookup.

    The input contract is intentionally strict: after finite-value filtering,
    session indices must be strictly increasing.  A duplicate session means a
    daily/cohort grain violation and must not be silently folded into the
    covariance estimate.
    """
    x = np.asarray(values, dtype=float)
    idx = np.asarray(session_index, dtype=int)
    mask = np.isfinite(x) & np.isfinite(idx)
    x, idx = x[mask], idx[mask]
    n = x.size
    if n < 2:
        return float("nan")
    _validate_session_index(idx)
    mean = float(x.mean())
    centered = x - mean
    gamma0 = float(np.dot(centered, centered) / n)
    long_run = gamma0
    for distance, left, right in _hac_pair_indices(idx, lag):
        gamma = float(np.dot(centered[left], centered[right]) / n)
        long_run += 2.0 * (1.0 - distance / (lag + 1.0)) * gamma
    variance_mean = long_run / n
    if variance_mean <= 0 or not np.isfinite(variance_mean):
        return float("nan")
    return mean / float(np.sqrt(variance_mean))


def newey_west_ols(
    y: np.ndarray | list[float],
    x: np.ndarray | list[float] | None,
    session_index: np.ndarray | list[int],
    lag: int,
) -> dict[str, float]:
    """HAC-corrected OLS of ``y`` on a constant, optionally plus one regressor.

    Same gap-aware kernel as :func:`newey_west_tstat`: only pairs whose KRX
    session distance is within ``lag`` enter the autocovariance, so a market
    closure never lets two observations count as adjacent. With ``x=None`` the
    design is the constant alone and ``t_alpha`` reproduces
    ``newey_west_tstat(y, session_index, lag)`` exactly — the two are the same
    estimator written twice, and a test pins that down.

    Stage 1b estimates ``IC_t = alpha + delta * s_t + e_t`` with ``s_t`` the
    regime indicator, which makes ``delta`` exactly the difference of the two
    conditional means (§4.1) and ``alpha`` the mean in the ``s=0`` regime.
    Fitting it as a regression rather than differencing two means is what puts
    one HAC variance over both, on the session axis the cell's own ``t_nw``
    already uses.

    A degenerate regressor — one regime never occurring in the sample — gives
    NaN for the slope rather than an arbitrary number, and the intercept is
    still reported from the constant-only fit. G1 is what catches that case
    first; this only makes sure it cannot be papered over.
    """
    y_arr = np.asarray(y, dtype=float)
    idx = np.asarray(session_index, dtype=int)
    x_arr = None if x is None else np.asarray(x, dtype=float)
    if x_arr is not None and not (y_arr.size == x_arr.size == idx.size):
        raise ValueError("y, x and session_index must have the same length")
    if x_arr is None and y_arr.size != idx.size:
        raise ValueError("y and session_index must have the same length")

    mask = np.isfinite(y_arr) & np.isfinite(idx)
    if x_arr is not None:
        mask &= np.isfinite(x_arr)
    y_arr, idx = y_arr[mask], idx[mask]
    x_arr = None if x_arr is None else x_arr[mask]
    n = y_arr.size
    empty = {
        "alpha": float("nan"),
        "se_alpha": float("nan"),
        "t_alpha": float("nan"),
        "p_alpha": float("nan"),
        "delta": float("nan"),
        "se_delta": float("nan"),
        "t_delta": float("nan"),
        "p_delta": float("nan"),
        "n": n,
    }
    if n < 2:
        return empty
    _validate_session_index(idx)

    def _fit(design: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        xtx = design.T @ design
        if np.linalg.matrix_rank(xtx) < design.shape[1]:
            return None
        xtx_inv = np.linalg.inv(xtx)
        beta = xtx_inv @ (design.T @ y_arr)
        scores = design * (y_arr - design @ beta)[:, None]
        meat = scores.T @ scores
        for distance, left, right in _hac_pair_indices(idx, lag):
            cross = scores[left].T @ scores[right]
            meat = meat + (1.0 - distance / (lag + 1.0)) * (cross + cross.T)
        return beta, xtx_inv @ meat @ xtx_inv

    def _stats(beta: np.ndarray, cov: np.ndarray, position: int) -> dict[str, float]:
        variance = float(cov[position, position])
        coefficient = float(beta[position])
        if variance <= 0 or not math.isfinite(variance):
            return {"value": coefficient, "se": float("nan"), "t": float("nan"), "p": float("nan")}
        se = math.sqrt(variance)
        t_stat = coefficient / se
        return {"value": coefficient, "se": se, "t": t_stat, "p": two_sided_normal_p(t_stat)}

    ones = np.ones(n)
    full = None if x_arr is None else _fit(np.column_stack([ones, x_arr]))
    if full is not None:
        beta, cov = full
        alpha, delta = _stats(beta, cov, 0), _stats(beta, cov, 1)
    else:
        constant_only = _fit(ones[:, None])
        if constant_only is None:
            return empty
        beta, cov = constant_only
        alpha = _stats(beta, cov, 0)
        delta = {"value": float("nan"), "se": float("nan"), "t": float("nan"), "p": float("nan")}
    return {
        "alpha": alpha["value"],
        "se_alpha": alpha["se"],
        "t_alpha": alpha["t"],
        "p_alpha": alpha["p"],
        "delta": delta["value"],
        "se_delta": delta["se"],
        "t_delta": delta["t"],
        "p_delta": delta["p"],
        "n": n,
    }


def n_hac_pairs(session_index: np.ndarray | list[int], lag: int) -> int:
    """Count of (i, j) pairs within ``lag`` KRX sessions of each other.

    Diagnostic companion to :func:`newey_west_tstat` (§4.2: "각 lag의 pair 수를
    출력해 gap 영향을 진단한다") — a long calendar gap (regime break, delisting)
    starves some lags of pairs even when ``n_dates`` looks adequate.
    """
    idx = np.asarray(session_index, dtype=int)
    idx = idx[np.isfinite(idx)]
    if idx.size < 2:
        return 0
    _validate_session_index(idx)
    return sum(int(left.size) for _distance, left, _right in _hac_pair_indices(idx, lag))


def newey_west_tstat_legacy(
    values: np.ndarray | list[float], session_index: np.ndarray | list[int], lag: int
) -> float:
    """Reference implementation retained for randomized parity tests."""
    x = np.asarray(values, dtype=float)
    idx = np.asarray(session_index, dtype=int)
    mask = np.isfinite(x) & np.isfinite(idx)
    x, idx = x[mask], idx[mask]
    n = x.size
    if n < 2:
        return float("nan")
    _validate_session_index(idx)
    mean = float(x.mean())
    centered = x - mean
    long_run = float(np.dot(centered, centered) / n)
    if lag > 0:
        for i in range(n):
            distances = idx[i + 1 :] - idx[i]
            valid = (distances > 0) & (distances <= lag)
            for distance, product in zip(distances[valid], centered[i] * centered[i + 1 :][valid]):
                long_run += 2.0 * (1.0 - distance / (lag + 1.0)) * float(product / n)
    variance_mean = long_run / n
    if variance_mean <= 0 or not np.isfinite(variance_mean):
        return float("nan")
    return mean / float(np.sqrt(variance_mean))


def _validate_session_index(idx: np.ndarray) -> None:
    if idx.ndim != 1:
        raise ValueError("session_index must be one-dimensional")
    if idx.size > 1 and np.any(np.diff(idx) <= 0):
        raise ValueError("session_index must be strictly increasing and duplicate-free")


def _hac_pair_indices(idx: np.ndarray, lag: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return the exact pair positions shared by t-stat and diagnostics."""
    if lag <= 0 or idx.size < 2:
        return []
    pairs: list[tuple[int, np.ndarray, np.ndarray]] = []
    for distance in range(1, lag + 1):
        left = np.arange(idx.size, dtype=np.intp)
        right = np.searchsorted(idx, idx + distance, side="left")
        valid = (right < idx.size) & (idx[right.clip(max=idx.size - 1)] == idx + distance)
        pairs.append((distance, left[valid], right[valid]))
    return [(distance, left, right) for distance, left, right in pairs if left.size]


def exact_binomial_sign_test_p(n_success: int, n_trials: int) -> float:
    """One-sided exact binomial sign test: H0 p=0.5, H1 p>0.5 (§A-5).

    Used for non-overlap offset direction checks where ``n_trials`` (dates per
    offset, e.g. ~20-500) is far too small for a normal approximation to be
    trustworthy — ``math.comb`` keeps this exact via Python's arbitrary-
    precision integers rather than pulling in scipy for one test.
    """
    if n_trials <= 0:
        return float("nan")
    if not 0 <= n_success <= n_trials:
        raise ValueError(f"n_success={n_success} must be within [0, n_trials={n_trials}]")
    total = sum(math.comb(n_trials, i) for i in range(n_success, n_trials + 1))
    return total / (2**n_trials)


def two_sided_normal_p(tstat: float) -> float:
    """Two-sided asymptotic-normal p-value for an NW/HAC t-statistic (§4.2:
    ``stats.nw_p_value_distribution: asymptotic_normal``, not a t-distribution)."""
    if not math.isfinite(tstat):
        return float("nan")
    return math.erfc(abs(tstat) / math.sqrt(2.0))


def choose_nw_lag(
    *, scan_type: str, horizon: int | None = None, bucket_width: int | None = None
) -> int:
    """Use cumulative ``h-1`` and bucket-width ``width-1`` preregistered lags."""
    if scan_type == "cum":
        if horizon is None or horizon < 1:
            raise ValueError("cumulative scan requires a positive horizon")
        return horizon - 1
    if scan_type == "bucket":
        if bucket_width is None or bucket_width < 1:
            raise ValueError("bucket scan requires a positive bucket width")
        return bucket_width - 1
    raise ValueError(f"unknown scan type {scan_type!r}")


def benjamini_hochberg(pvalues: list[float] | np.ndarray) -> np.ndarray:
    """Return monotone BH q-values in the original hypothesis order.

    Tied p-values break by their position in ``pvalues`` (``kind="stable"``)
    — callers that must break ties by a specific key (§2.3 rule 5: hypothesis
    id) pre-sort their rows by that key before calling this.
    """
    p = np.asarray(pvalues, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    values = p[finite]
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    q[finite] = restored
    return q


def _spearman(pred: np.ndarray, realized: np.ndarray) -> float:
    """Spearman rank correlation = Pearson corr of ranks. NaN if degenerate."""
    if pred.size < 2:
        return float("nan")
    pr = _rankdata(pred)
    rr = _rankdata(realized)
    ps, rs = pr.std(), rr.std()
    if ps == 0 or rs == 0:
        return float("nan")
    return float(np.corrcoef(pr, rr)[0, 1])


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-rank of ``a`` (ties share the mean rank), like scipy.rankdata."""
    order = a.argsort()
    ranks = np.empty(a.size, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # resolve ties to average rank
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if counts.max() > 1:
        sums = np.zeros(counts.size)
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        ranks = avg[inv]
    return ranks


def _finite_clean(
    df: pl.DataFrame, date_col: str, pred_col: str, realized_col: str
) -> pl.DataFrame:
    """Keep only rows where pred and realized are non-null AND finite.

    ``drop_nulls`` alone leaves NaN/inf (e.g. a NaN realized label or an inf
    prediction), which would poison the rank/quantile stats; filter those too.
    """
    columns = (
        [date_col, pred_col] if pred_col == realized_col else [date_col, pred_col, realized_col]
    )
    clean = df.select(columns).drop_nulls()
    return clean.filter(pl.col(pred_col).is_finite() & pl.col(realized_col).is_finite())


def per_date_rank_ic(
    df: pl.DataFrame,
    *,
    pred_col: str,
    realized_col: str,
    date_col: str = "trade_date",
) -> pl.DataFrame:
    """Per-date Spearman IC between prediction and realized label.

    Returns a DataFrame with columns ``[date_col, "rank_ic", "n"]`` (one row per
    date). Rows with null/NaN/inf pred or realized are dropped before correlating.
    """
    out_dates = []
    out_ic = []
    out_n = []
    clean = _finite_clean(df, date_col, pred_col, realized_col)
    for (d,), grp in clean.group_by([date_col], maintain_order=True):
        pred = grp[pred_col].to_numpy()
        realized = grp[realized_col].to_numpy()
        out_dates.append(d)
        out_ic.append(_spearman(pred, realized))
        out_n.append(grp.height)
    return pl.DataFrame({date_col: out_dates, "rank_ic": out_ic, "n": out_n})


def _quantile_stats(
    df: pl.DataFrame,
    *,
    pred_col: str,
    realized_col: str,
    date_col: str,
    n_quantiles: int = 5,
) -> tuple[float, float, float, float]:
    """Per-date quantile portfolio stats, averaged across dates.

    Returns (top_decile_spread, top_minus_bottom, hit_ratio_top, _reserved).
    Top decile uses pred rank >= 0.9; top/bottom quantiles use n_quantiles.
    """
    clean = _finite_clean(df, date_col, pred_col, realized_col)
    top_dec, top_q, bot_q, hit = [], [], [], []
    for (_d,), grp in clean.group_by([date_col], maintain_order=True):
        if grp.height < n_quantiles:
            continue
        pred = grp[pred_col].to_numpy()
        realized = grp[realized_col].to_numpy()
        rank = _rankdata(pred) / pred.size  # in (0,1]
        top_mask = rank >= (1 - 1 / n_quantiles)
        bot_mask = rank <= (1 / n_quantiles)
        dec_mask = rank >= 0.9
        if dec_mask.any():
            top_dec.append(float(realized[dec_mask].mean()))
        if top_mask.any():
            top_q.append(float(realized[top_mask].mean()))
            hit.append(float((realized[top_mask] > 0).mean()))
        if bot_mask.any():
            bot_q.append(float(realized[bot_mask].mean()))

    def _m(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else float("nan")

    tmb = _m(top_q) - _m(bot_q) if top_q and bot_q else float("nan")
    return _m(top_dec), tmb, _m(hit), float("nan")


def per_date_quantile_spread(
    df: pl.DataFrame,
    *,
    score_col: str,
    realized_col: str,
    date_col: str = "trade_date",
    n_quantiles: int = 5,
    min_names: int = 20,
) -> pl.DataFrame:
    """Return raw-return top-minus-bottom spreads for each date.

    The score may be a model prediction or a precomputed rank.  Invalid rows
    are removed before both the score ordering and the realized-return mean,
    keeping the portfolio population explicit and auditable.
    """
    clean = _finite_clean(df, date_col, score_col, realized_col)
    rows: list[tuple[object, float, int]] = []
    for (d,), grp in clean.group_by([date_col], maintain_order=True):
        if grp.height < max(min_names, n_quantiles):
            continue
        score = grp[score_col].to_numpy()
        realized = grp[realized_col].to_numpy()
        ranks = _rankdata(score) / score.size
        top = realized[ranks >= 1 - 1 / n_quantiles]
        bottom = realized[ranks <= 1 / n_quantiles]
        if top.size and bottom.size:
            rows.append((d, float(top.mean() - bottom.mean()), grp.height))
    return pl.DataFrame(
        rows,
        schema={date_col: clean.schema[date_col], "spread": pl.Float64, "n": pl.Int64},
        orient="row",
    )


def raw_vs_rank_quantile_spread(
    df: pl.DataFrame,
    *,
    rank_col: str,
    raw_col: str,
    date_col: str = "trade_date",
    n_quantiles: int = 5,
    min_names: int = 20,
) -> pl.DataFrame:
    """Compare quantile spreads formed from a raw score and its rank score."""
    raw = per_date_quantile_spread(
        df,
        score_col=raw_col,
        realized_col=raw_col,
        date_col=date_col,
        n_quantiles=n_quantiles,
        min_names=min_names,
    ).rename({"spread": "raw_score_spread"})
    ranked = per_date_quantile_spread(
        df,
        score_col=rank_col,
        realized_col=raw_col,
        date_col=date_col,
        n_quantiles=n_quantiles,
        min_names=min_names,
    ).rename({"spread": "rank_score_spread"})
    return raw.join(ranked, on=[date_col, "n"], how="full", coalesce=True)


@dataclass(frozen=True)
class EconomicReport:
    """Non-overlapping-rebalance economic significance (acceptance gate §6.1 ⑤).

    Daily top-decile membership is not directly comparable across successive
    days because the label horizon overlaps (each day's realized return window
    covers the next ``horizon`` sessions) — turnover computed on daily
    snapshots would double-count the same holding period many times over. This
    report instead re-derives membership only on a grid spaced ``horizon``
    sessions apart (mirroring the non-overlap bucket grid in
    ``research/etl/labels.py``), so each rebalance is a genuinely distinct
    holding period.
    """

    horizon: int
    n_rebalances: int
    grid_top_decile_spread: float
    turnover: float
    cost_bps_roundtrip: float
    cost_adjusted_spread: float

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "n_rebalances": self.n_rebalances,
            "grid_top_decile_spread": self.grid_top_decile_spread,
            "turnover": self.turnover,
            "cost_bps_roundtrip": self.cost_bps_roundtrip,
            "cost_adjusted_spread": self.cost_adjusted_spread,
        }


def rebalance_grid(dates: list, horizon: int) -> list:
    """Every ``horizon``-th date of the sorted unique ``dates`` (non-overlap grid)."""
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    ordered = sorted(set(dates))
    return ordered[::horizon]


def decile_membership(
    df: pl.DataFrame,
    *,
    pred_col: str,
    date_col: str = "trade_date",
    ticker_col: str = "ticker",
    q: float = 0.9,
) -> dict:
    """Per-date set of tickers whose ``pred`` rank is at/above quantile ``q``.

    ``q=0.9`` is the top decile (matches ``_quantile_stats``'s ``top_decile_spread``
    convention); ``q=0.1`` with ``rank <= q`` would give the bottom decile — this
    helper always takes the upper tail, so pass ``1 - q`` and filter externally
    for a bottom-decile membership map if ever needed.
    """
    clean = df.select([date_col, ticker_col, pred_col]).drop_nulls()
    clean = clean.filter(pl.col(pred_col).is_finite())
    out: dict = {}
    for (d,), grp in clean.group_by([date_col], maintain_order=True):
        pred = grp[pred_col].to_numpy()
        if pred.size == 0:
            continue
        rank = _rankdata(pred) / pred.size
        mask = rank >= q
        if mask.any():
            out[d] = set(grp[ticker_col].to_numpy()[mask].tolist())
    return out


def portfolio_turnover(membership_by_date: dict, ordered_keys: list) -> float:
    """Mean pairwise turnover between consecutive membership snapshots.

    ``turnover = 1 - |A ∩ B| / max(|A|, |B|)`` for each consecutive pair present
    in ``membership_by_date`` — 0.0 means identical holdings, 1.0 means fully
    disjoint. Snapshots missing from ``membership_by_date`` (e.g. a rebalance
    date with too few names) are skipped rather than treated as empty.
    """
    present = [k for k in ordered_keys if k in membership_by_date]
    turnovers = []
    for prev, curr in zip(present, present[1:]):
        a, b = membership_by_date[prev], membership_by_date[curr]
        denom = max(len(a), len(b))
        if denom == 0:
            continue
        turnovers.append(1.0 - len(a & b) / denom)
    return float(np.mean(turnovers)) if turnovers else float("nan")


def economic_report(
    df: pl.DataFrame,
    *,
    pred_col: str,
    realized_col: str,
    horizon: int,
    date_col: str = "trade_date",
    ticker_col: str = "ticker",
    q: float = 0.9,
    cost_bps_roundtrip: float = 60.0,
) -> EconomicReport:
    """Grid-based top-decile spread net of an assumed round-trip transaction cost.

    ``cost_bps_roundtrip`` (default 60bp) is a business assumption, not derived
    from data — document it alongside any reported result. The realized spread
    is averaged only over the rebalance grid (not daily), so it is directly
    comparable to the turnover measured on that same grid.
    """
    clean = _finite_clean(df, date_col, pred_col, realized_col)
    grid = rebalance_grid(clean[date_col].to_list(), horizon)

    membership = decile_membership(
        df, pred_col=pred_col, date_col=date_col, ticker_col=ticker_col, q=q
    )
    turnover = portfolio_turnover(membership, grid)

    realized_means = []
    for d in grid:
        day = clean.filter(pl.col(date_col) == d)
        if day.height == 0:
            continue
        pred = day[pred_col].to_numpy()
        rank = _rankdata(pred) / pred.size
        mask = rank >= q
        if mask.any():
            realized_means.append(float(day[realized_col].to_numpy()[mask].mean()))
    grid_spread = float(np.mean(realized_means)) if realized_means else float("nan")

    cost = 0.0 if turnover != turnover else turnover * cost_bps_roundtrip / 10_000.0
    net = grid_spread - cost if grid_spread == grid_spread else float("nan")

    return EconomicReport(
        horizon=horizon,
        n_rebalances=len(grid),
        grid_top_decile_spread=grid_spread,
        turnover=turnover,
        cost_bps_roundtrip=cost_bps_roundtrip,
        cost_adjusted_spread=net,
    )


@dataclass(frozen=True)
class TopKEconomicReport:
    """Economics of the fixed top-k buy list, not the top decile.

    ``economic_report``'s decile is ~260 names on a ~2,600-name universe and
    turns over more gently than the k=100 list ``predict.select_topk`` actually
    trades. The Grade A acceptance gate (``07_phase1_acceptance_gate.md`` §6)
    adopted its candidates conditionally on exactly this gap: the improvement
    had only been shown at decile granularity. Same non-overlapping rebalance
    grid and same cost assumption as ``economic_report``, so the two are read
    side by side.

    ``mean_names_scored`` is below ``mean_names_held`` near the end of the
    sample, where the label horizon has not closed yet — those names are held
    but cannot contribute a realized return.
    """

    horizon: int
    k: int
    n_rebalances: int
    mean_names_held: float
    mean_names_scored: float
    grid_topk_mean_return: float
    turnover: float
    cost_bps_roundtrip: float
    cost_adjusted_return: float

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "k": self.k,
            "n_rebalances": self.n_rebalances,
            "mean_names_held": self.mean_names_held,
            "mean_names_scored": self.mean_names_scored,
            "grid_topk_mean_return": self.grid_topk_mean_return,
            "turnover": self.turnover,
            "cost_bps_roundtrip": self.cost_bps_roundtrip,
            "cost_adjusted_return": self.cost_adjusted_return,
        }


def _topk_ranked(
    df: pl.DataFrame,
    *,
    pred_col: str,
    k: int,
    date_col: str,
    ticker_col: str,
) -> pl.DataFrame:
    """Per-date top-k rows by ``pred``, mirroring ``predict.select_topk`` exactly.

    Cross-sectional ordinal rank on ``pred`` descending (ties broken
    deterministically), keeping ``rank <= k``. Rows with a null/non-finite
    ``pred`` are dropped first — an unscored name is never bought.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    clean = df.select([date_col, ticker_col, pred_col]).drop_nulls()
    clean = clean.filter(pl.col(pred_col).is_finite())
    rank = pl.col(pred_col).rank("ordinal", descending=True).over(date_col)
    return clean.with_columns(rank.alias("_topk_rank")).filter(pl.col("_topk_rank") <= k)


def topk_membership(
    df: pl.DataFrame,
    *,
    pred_col: str,
    k: int,
    date_col: str = "trade_date",
    ticker_col: str = "ticker",
) -> dict:
    """Per-date set of the k highest-``pred`` tickers — the deployed buy list."""
    picked = _topk_ranked(df, pred_col=pred_col, k=k, date_col=date_col, ticker_col=ticker_col)
    out: dict = {}
    for (d,), grp in picked.group_by([date_col], maintain_order=True):
        out[d] = set(grp[ticker_col].to_list())
    return out


def topk_economic_report(
    df: pl.DataFrame,
    *,
    pred_col: str,
    realized_col: str,
    horizon: int,
    k: int = 100,
    date_col: str = "trade_date",
    ticker_col: str = "ticker",
    cost_bps_roundtrip: float = 60.0,
) -> TopKEconomicReport:
    """Top-k mean realized return on the rebalance grid, net of turnover cost.

    ``cost_bps_roundtrip`` is the same business assumption ``economic_report``
    documents — not derived from data.
    """
    grid = rebalance_grid(df[date_col].to_list(), horizon)
    picked = _topk_ranked(df, pred_col=pred_col, k=k, date_col=date_col, ticker_col=ticker_col)
    membership = {
        d: set(grp[ticker_col].to_list())
        for (d,), grp in picked.group_by([date_col], maintain_order=True)
    }
    turnover = portfolio_turnover(membership, grid)

    labels = df.select([date_col, ticker_col, realized_col]).drop_nulls()
    labels = labels.filter(pl.col(realized_col).is_finite())
    scored = picked.join(labels, on=[date_col, ticker_col], how="inner")

    returns: list[float] = []
    scored_counts: list[int] = []
    for d in grid:
        day = scored.filter(pl.col(date_col) == d)
        if day.height == 0:
            continue
        returns.append(float(day[realized_col].mean()))
        scored_counts.append(day.height)
    held_counts = [len(membership[d]) for d in grid if d in membership]

    grid_return = float(np.mean(returns)) if returns else float("nan")
    cost = 0.0 if turnover != turnover else turnover * cost_bps_roundtrip / 10_000.0
    net = grid_return - cost if grid_return == grid_return else float("nan")

    return TopKEconomicReport(
        horizon=horizon,
        k=k,
        n_rebalances=len(held_counts),
        mean_names_held=float(np.mean(held_counts)) if held_counts else float("nan"),
        mean_names_scored=float(np.mean(scored_counts)) if scored_counts else float("nan"),
        grid_topk_mean_return=grid_return,
        turnover=turnover,
        cost_bps_roundtrip=cost_bps_roundtrip,
        cost_adjusted_return=net,
    )


def evaluate(
    df: pl.DataFrame,
    *,
    pred_col: str,
    realized_col: str,
    date_col: str = "trade_date",
    n_quantiles: int = 5,
) -> RankICReport:
    """Compute the full ranking report (etl_00 §6) over a predictions frame.

    ``realized_col`` is the realized label to rank against — typically the raw
    excess return (``raw_label_20d``) so the top-decile spread is in return
    units, or the rank label for a pure IC check.
    """
    ic_df = per_date_rank_ic(df, pred_col=pred_col, realized_col=realized_col, date_col=date_col)
    ics = ic_df["rank_ic"].drop_nulls().to_numpy()
    mean = float(ics.mean()) if ics.size else float("nan")
    std = float(ics.std(ddof=1)) if ics.size > 1 else float("nan")
    icir = mean / std if std and not np.isnan(std) and std != 0 else float("nan")
    tstat = mean / (std / np.sqrt(ics.size)) if std and ics.size > 1 and std != 0 else float("nan")

    top_dec, tmb, hit, _ = _quantile_stats(
        df,
        pred_col=pred_col,
        realized_col=realized_col,
        date_col=date_col,
        n_quantiles=n_quantiles,
    )

    clean = _finite_clean(df, date_col, pred_col, realized_col)
    return RankICReport(
        n_dates=int(ic_df.height),
        n_obs=int(clean.height),
        rank_ic_mean=mean,
        rank_ic_std=std,
        icir=icir,
        rank_ic_tstat=tstat,
        top_decile_spread=top_dec,
        top_minus_bottom=tmb,
        hit_ratio_top=hit,
    )


# --- probability metrics (20260907_model_experiment `03` §2.1) --------------

# log-loss is unbounded at p in {0, 1}: one confident miss would otherwise
# decide a whole fold. `03` §2.1 fixes the clip so every run reports the same
# number rather than one that depends on how extreme a model dares to be.
PROB_CLIP = 1e-6
DEFAULT_PROB_BINS = 10


@dataclass(frozen=True)
class ClassificationReport:
    """Probability quality for one slice — accuracy of order *and* of level.

    ``log_loss`` is the preregistered primary metric at every horizon (`03`
    §3.2); ``ece`` is what says whether the number can be read as a
    probability at all (a p=0.7 bucket that realizes 0.55 is a ranking with a
    probability's clothes on). ``auc_daily_mean`` is the honest ranking figure
    for a cross-sectional model — the pooled AUC also mixes *dates*, so a model
    that merely knows which days were good would score on it.

    ``base_rate`` is the mean of the per-date positive rate over the dates that
    entered ``precision_at_k``, so ``lift_at_k`` is the ratio of two quantities
    measured on the same dates.
    """

    n_obs: int
    n_dates: int
    base_rate: float
    log_loss: float
    brier: float
    auc_pooled: float
    auc_daily_mean: float
    n_dates_auc: int
    ece: float
    n_bins: int
    k: int
    precision_at_k: float
    lift_at_k: float

    def as_dict(self) -> dict:
        return {
            "n_obs": self.n_obs,
            "n_dates": self.n_dates,
            "base_rate": self.base_rate,
            "log_loss": self.log_loss,
            "brier": self.brier,
            "auc_pooled": self.auc_pooled,
            "auc_daily_mean": self.auc_daily_mean,
            "n_dates_auc": self.n_dates_auc,
            "ece": self.ece,
            "n_bins": self.n_bins,
            "k": self.k,
            "precision_at_k": self.precision_at_k,
            "lift_at_k": self.lift_at_k,
        }


def _binary_clean(df: pl.DataFrame, date_col: str, pred_col: str, y_col: str) -> pl.DataFrame:
    """Rows with a finite probability in [0,1] and a 0/1 label.

    A prediction outside [0,1] is rejected rather than clipped: it means a rank
    score or a raw margin was passed in, and every metric below would still
    return a plausible-looking number.
    """
    clean = df.select([date_col, pred_col, y_col]).drop_nulls()
    clean = clean.filter(pl.col(pred_col).is_finite())
    if clean.height == 0:
        return clean
    p_min, p_max = clean[pred_col].min(), clean[pred_col].max()
    if p_min < 0.0 or p_max > 1.0:
        raise ValueError(f"{pred_col!r} must be a probability in [0,1]; got [{p_min}, {p_max}]")
    labels = set(clean[y_col].unique().to_list())
    if not labels <= {0, 1, 0.0, 1.0, True, False}:
        raise ValueError(f"{y_col!r} must be binary 0/1; got values {sorted(labels)[:5]}")
    return clean.with_columns(pl.col(y_col).cast(pl.Float64).alias(y_col))


def binary_log_loss(p: np.ndarray, y: np.ndarray, *, clip: float = PROB_CLIP) -> float:
    """``-mean(y ln p + (1-y) ln(1-p))`` with ``p`` clipped to [clip, 1-clip]."""
    if p.size == 0:
        return float("nan")
    q = np.clip(p, clip, 1.0 - clip)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log1p(-q)))


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """``mean((p - y)^2)`` — the squared-error half of the calibration picture."""
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def binary_auc(p: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC; ties split credit, matching ``sklearn.roc_auc_score``.

    NaN when one class is absent — an AUC is undefined there, and a 0.5 would
    quietly drag a daily average toward "no skill".
    """
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(p)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_calibration_error(
    p: np.ndarray, y: np.ndarray, *, n_bins: int = DEFAULT_PROB_BINS
) -> float:
    """``sum_b (n_b/N) * |mean(p_b) - mean(y_b)|`` over equal-width bins on [0,1]."""
    table = _reliability_rows(p, y, n_bins=n_bins)
    if not table:
        return float("nan")
    total = sum(row["n"] for row in table)
    return float(sum(row["n"] / total * abs(row["gap"]) for row in table))


def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Equal-width bin index on [0,1]; ``p == 1`` lands in the last bin."""
    idx = np.floor(p * n_bins).astype(int)
    return np.clip(idx, 0, n_bins - 1)


def _reliability_rows(
    p: np.ndarray, y: np.ndarray, *, n_bins: int = DEFAULT_PROB_BINS
) -> list[dict]:
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if p.size == 0:
        return []
    idx = _bin_index(p, n_bins)
    rows: list[dict] = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        p_mean = float(p[mask].mean())
        y_mean = float(y[mask].mean())
        rows.append(
            {
                "bin": b,
                "p_lo": b / n_bins,
                "p_hi": (b + 1) / n_bins,
                "n": n,
                "p_mean": p_mean,
                "y_mean": y_mean,
                "gap": p_mean - y_mean,
            }
        )
    return rows


def reliability_table(
    df: pl.DataFrame,
    *,
    pred_col: str,
    y_col: str,
    date_col: str = "trade_date",
    n_bins: int = DEFAULT_PROB_BINS,
) -> pl.DataFrame:
    """Per-bin ``mean(p)`` vs ``mean(y)`` — the reliability curve, as a table.

    Empty bins are omitted rather than emitted as NaN rows: a probability model
    that never predicts above 0.8 has no 0.8-0.9 bucket, and saying so with a
    missing row is clearer than a row of NaNs.
    """
    clean = _binary_clean(df, date_col, pred_col, y_col)
    rows = _reliability_rows(clean[pred_col].to_numpy(), clean[y_col].to_numpy(), n_bins=n_bins)
    schema = {
        "bin": pl.Int64,
        "p_lo": pl.Float64,
        "p_hi": pl.Float64,
        "n": pl.Int64,
        "p_mean": pl.Float64,
        "y_mean": pl.Float64,
        "gap": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def _precision_at_k(
    clean: pl.DataFrame, *, pred_col: str, y_col: str, date_col: str, k: int
) -> tuple[float, float, int]:
    """Mean per-date precision@k, the matching mean base rate, and n_dates.

    Ordinal rank descending, exactly as ``predict.select_topk`` picks the buy
    list — so this is the precision of the list that would actually be traded,
    ties and all. A date with fewer than k names contributes all of them.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    rank = pl.col(pred_col).rank("ordinal", descending=True).over(date_col)
    picked = clean.with_columns(rank.alias("_pk_rank")).filter(pl.col("_pk_rank") <= k)
    per_date = picked.group_by(date_col).agg(pl.col(y_col).mean().alias("precision"))
    base = clean.group_by(date_col).agg(pl.col(y_col).mean().alias("base_rate"))
    joined = per_date.join(base, on=date_col, how="inner")
    if joined.height == 0:
        return float("nan"), float("nan"), 0
    return (
        float(joined["precision"].mean()),
        float(joined["base_rate"].mean()),
        int(joined.height),
    )


def classification_report(
    df: pl.DataFrame,
    *,
    pred_col: str,
    y_col: str,
    date_col: str = "trade_date",
    k: int = 100,
    n_bins: int = DEFAULT_PROB_BINS,
) -> ClassificationReport:
    """Probability metrics for one slice (`03` §2.1).

    ``pred_col`` must hold probabilities; ``y_col`` a 0/1 label. Rows missing
    either are dropped, which is how the last ``h`` sessions of a panel — held
    but not yet resolved — stay out of the metrics.
    """
    clean = _binary_clean(df, date_col, pred_col, y_col)
    p = clean[pred_col].to_numpy()
    y = clean[y_col].to_numpy()

    daily_auc: list[float] = []
    for (_d,), grp in clean.group_by([date_col], maintain_order=True):
        auc = binary_auc(grp[pred_col].to_numpy(), grp[y_col].to_numpy())
        if auc == auc:  # skip the NaN of a single-class date
            daily_auc.append(auc)

    precision, base_rate, n_dates_k = _precision_at_k(
        clean, pred_col=pred_col, y_col=y_col, date_col=date_col, k=k
    )
    lift = precision / base_rate if base_rate else float("nan")

    return ClassificationReport(
        n_obs=int(clean.height),
        n_dates=int(clean[date_col].n_unique()) if clean.height else 0,
        base_rate=base_rate if n_dates_k else float("nan"),
        log_loss=binary_log_loss(p, y),
        brier=brier_score(p, y),
        auc_pooled=binary_auc(p, y),
        auc_daily_mean=float(np.mean(daily_auc)) if daily_auc else float("nan"),
        n_dates_auc=len(daily_auc),
        ece=expected_calibration_error(p, y, n_bins=n_bins),
        n_bins=n_bins,
        k=k,
        precision_at_k=precision,
        lift_at_k=lift,
    )


@dataclass(frozen=True)
class ThresholdEconomicReport:
    """Economics of the ``p >= tau`` portfolio (`03` §2.3, D-4).

    The top-k list always holds k names; this one holds however many clear the
    threshold, which is the point — a calibrated probability should let the
    portfolio go small when nothing looks good. ``n_rebalances_cash`` counts the
    grid dates where it went to zero, and those dates enter
    ``grid_mean_return`` as 0.0 (cash), not as a skipped observation. Ignoring
    them would report the returns of a strategy that only trades when it likes
    the odds while pretending it was always invested.

    Same non-overlapping rebalance grid and same 60bp round-trip assumption as
    ``economic_report`` / ``topk_economic_report``, so the three read together.
    """

    horizon: int
    tau: float
    n_rebalances: int
    n_rebalances_held: int
    n_rebalances_cash: int
    mean_names_held: float
    min_names_held: int
    grid_mean_return: float
    turnover: float
    cost_bps_roundtrip: float
    cost_adjusted_return: float

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "tau": self.tau,
            "n_rebalances": self.n_rebalances,
            "n_rebalances_held": self.n_rebalances_held,
            "n_rebalances_cash": self.n_rebalances_cash,
            "mean_names_held": self.mean_names_held,
            "min_names_held": self.min_names_held,
            "grid_mean_return": self.grid_mean_return,
            "turnover": self.turnover,
            "cost_bps_roundtrip": self.cost_bps_roundtrip,
            "cost_adjusted_return": self.cost_adjusted_return,
        }


def threshold_membership(
    df: pl.DataFrame,
    *,
    pred_col: str,
    tau: float,
    date_col: str = "trade_date",
    ticker_col: str = "ticker",
) -> dict:
    """Per-date set of tickers whose probability clears ``tau``."""
    clean = df.select([date_col, ticker_col, pred_col]).drop_nulls()
    clean = clean.filter(pl.col(pred_col).is_finite() & (pl.col(pred_col) >= tau))
    return {
        d: set(grp[ticker_col].to_list())
        for (d,), grp in clean.group_by([date_col], maintain_order=True)
    }


def threshold_economic_report(
    df: pl.DataFrame,
    *,
    pred_col: str,
    realized_col: str,
    horizon: int,
    tau: float = 0.6,
    date_col: str = "trade_date",
    ticker_col: str = "ticker",
    cost_bps_roundtrip: float = 60.0,
) -> ThresholdEconomicReport:
    """Equal-weight ``p >= tau`` portfolio on the rebalance grid, net of cost."""
    grid = rebalance_grid(df[date_col].to_list(), horizon)
    membership = threshold_membership(
        df, pred_col=pred_col, tau=tau, date_col=date_col, ticker_col=ticker_col
    )
    turnover = portfolio_turnover(membership, grid)

    labels = df.select([date_col, ticker_col, realized_col]).drop_nulls()
    labels = labels.filter(pl.col(realized_col).is_finite())
    by_date: dict = {
        d: dict(zip(grp[ticker_col].to_list(), grp[realized_col].to_list(), strict=True))
        for (d,), grp in labels.group_by([date_col], maintain_order=True)
    }

    returns: list[float] = []
    held_counts: list[int] = []
    n_cash = 0
    for d in grid:
        names = membership.get(d, set())
        held_counts.append(len(names))
        if not names:
            n_cash += 1
            returns.append(0.0)  # cash
            continue
        realized = [by_date.get(d, {})[t] for t in names if t in by_date.get(d, {})]
        if realized:
            returns.append(float(np.mean(realized)))

    grid_return = float(np.mean(returns)) if returns else float("nan")
    cost = 0.0 if turnover != turnover else turnover * cost_bps_roundtrip / 10_000.0
    net = grid_return - cost if grid_return == grid_return else float("nan")

    return ThresholdEconomicReport(
        horizon=horizon,
        tau=tau,
        n_rebalances=len(grid),
        n_rebalances_held=sum(1 for n in held_counts if n > 0),
        n_rebalances_cash=n_cash,
        mean_names_held=float(np.mean(held_counts)) if held_counts else float("nan"),
        min_names_held=int(min(held_counts)) if held_counts else 0,
        grid_mean_return=grid_return,
        turnover=turnover,
        cost_bps_roundtrip=cost_bps_roundtrip,
        cost_adjusted_return=net,
    )
