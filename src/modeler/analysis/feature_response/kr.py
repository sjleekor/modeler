"""KR layer-3 model-response diagnostics for the serving bundle ``kr_daily_h20_v1``.

Diagnostic only, never a verdict (plan ``08_layer3_plan.md`` §0). The bundle is
used as shipped: no refit, no new search. Three questions:

1. how much does the model lean on a feature group (within-date permutation),
2. in which direction (SHAP, ``tree_path_dependent``),
3. in which shape (PDP / ICE on a 21 point percentile grid).

Everything lives in the per-date percentile space the model was trained in.
Rules from plan §4 (stability, shape classes, bootstrap) are frozen as the
constants below; do not change them after seeing results.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import polars as pl
from scipy.stats import rankdata

# ---------------------------------------------------------------- frozen spec
SNAPSHOT_DIR = (
    "kr/datasets/02_updown_prob/snapshot_date=2026-08-23/source=sj2_remote/"
    "FS1h_h20_lag1_rank_idxbench__holdout"
)
SAMPLE_START = dt.date(2025, 8, 1)
SAMPLE_END = dt.date(2026, 8, 28)
HARD_EXCLUDE_FROM = dt.date(2026, 9, 17)  # never load a row on/after this date
HALF1_END = dt.date(2026, 1, 30)
HALF2_START = dt.date(2026, 2, 2)
EXPECTED_ROWS = 613_234
EXPECTED_DATES = 262
TARGET = "y_up_20d"
RETURN_COL = "raw_label_20d"
REPEATS = 5
PERM_SEED = 20261012
BOOT_SEED = 20261013
ICE_SEED = 20261014
CHECK_SEED = 20261015
SHAP_DATE_STEP = 5
GRID_POINTS = 21
ICE_ROWS = 1000
N_BOOT = 200
CORR_THRESHOLD = 0.7
FLAT_FRACTION = 0.25
MONOTONE_FRACTION = 0.80
STABLE_TOP_K = 10
PERM_BUDGET_MIN = 60.0  # plan §7: above this, switch to every 2nd date (recorded)
SUBSETS = ("full", "first", "second")


# ---------------------------------------------------------------- small utils
def _log(msg: str, log_path: Path | None = None) -> None:
    line = f"{dt.datetime.now(dt.UTC).strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    if log_path is not None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stock_data_root() -> Path:
    env = os.environ.get("STOCK_DATA_ROOT")
    if env:
        return Path(env)
    return Path("../stock_data").resolve()  # convention: relative to the modeler checkout (cwd)


def _git_info() -> dict[str, Any]:
    here = Path(__file__).resolve().parent

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(here), *args], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def _versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "polars", "scikit-learn", "scipy", "shap", "pyarrow", "joblib"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


# ---------------------------------------------------------------- sample
class Sample:
    """Design matrix sorted by date, with per-date segment offsets."""

    def __init__(
        self,
        X: np.ndarray,
        dates: np.ndarray,
        keys: pl.DataFrame,
        design: list[str],
        y: np.ndarray | None = None,
        ret: np.ndarray | None = None,
    ) -> None:
        self.X = X
        self.dates = dates  # datetime64[D] per row, non-decreasing
        self.keys = keys  # trade_date, ticker, market (same row order)
        self.design = design
        self.y = y
        self.ret = ret
        self.unique_dates, self.offsets = _segments(dates)

    @property
    def n_dates(self) -> int:
        return len(self.unique_dates)

    def date_idx_of_rows(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_dates), np.diff(self.offsets))

    def select_dates(self, keep: np.ndarray) -> "Sample":
        """Sub-sample on whole dates (``keep``: sorted date indices)."""
        rows = np.concatenate([np.arange(self.offsets[i], self.offsets[i + 1]) for i in keep])
        return Sample(
            self.X[rows],
            self.dates[rows],
            self.keys[rows],
            self.design,
            None if self.y is None else self.y[rows],
            None if self.ret is None else self.ret[rows],
        )


def _segments(dates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(dates) == 0:
        return dates, np.zeros(1, dtype=np.int64)
    if np.any(dates[1:] < dates[:-1]):
        raise ValueError("rows must be sorted by trade_date")
    uniq, start = np.unique(dates, return_index=True)
    return uniq, np.append(start, len(dates)).astype(np.int64)


def half_masks(unique_dates: np.ndarray) -> dict[str, np.ndarray]:
    d = unique_dates.astype("datetime64[D]")
    first = d <= np.datetime64(HALF1_END)
    second = d >= np.datetime64(HALF2_START)
    if np.any(first & second) or not np.all(first | second):
        raise ValueError("every sample date must fall in exactly one half")
    return {"full": np.ones(len(d), dtype=bool), "first": first, "second": second}


def load_design(
    panel_path: Path,
    feature_columns: list[str],
    design_columns: list[str],
    *,
    dates_limit: int | None = None,
    log: Callable[[str], None] = print,
) -> tuple[pl.DataFrame, list[str]]:
    """Read raw panel rows (never >= HARD_EXCLUDE_FROM) and build the serving design.

    The panel holds RAW feature values. Ranking is done per (trade_date, market)
    over the full date cross-section (before any label filter), through the same
    ``kr_model._model_input`` the bundle was trained with.
    """
    from modeler.serving import kr_model as km

    lf = pl.scan_parquet(panel_path).filter(
        (pl.col("trade_date") >= SAMPLE_START)
        & (pl.col("trade_date") <= SAMPLE_END)
        & (pl.col("trade_date") < HARD_EXCLUDE_FROM)
    )
    frame = lf.select(["trade_date", "ticker", "market", *feature_columns]).collect()
    if frame.height and frame["trade_date"].max() >= HARD_EXCLUDE_FROM:
        raise AssertionError("rows on/after the hard-exclude date were loaded")
    if dates_limit is not None:
        keep = frame["trade_date"].unique().sort().head(dates_limit)
        frame = frame.filter(pl.col("trade_date") <= keep.max())
    frame = frame.sort(["trade_date", "ticker", "market"])
    log(f"panel rows loaded: {frame.height}, dates: {frame['trade_date'].n_unique()}")
    transformed, design, _feats = km._model_input(frame, list(feature_columns))
    if list(design) != list(design_columns):
        raise AssertionError("design column order differs from bundle manifest design_columns")
    return transformed.select(["trade_date", "ticker", "market", *design]), list(design)


def attach_labels(
    transformed: pl.DataFrame, label_path: Path, design: list[str]
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Join holdout labels (this label file only) and keep rows with a y_up_20d."""
    lab = (
        pl.scan_parquet(label_path)
        .filter(
            (pl.col("trade_date") >= SAMPLE_START)
            & (pl.col("trade_date") <= SAMPLE_END)
            & (pl.col("trade_date") < HARD_EXCLUDE_FROM)
        )
        .select(["trade_date", "ticker", "market", TARGET, RETURN_COL])
        .collect()
    )
    joined = transformed.join(lab, on=["trade_date", "ticker", "market"], how="left")
    if joined.height != transformed.height:
        raise AssertionError("label join changed the row count (duplicate label keys)")
    rows_before = joined.height
    joined = joined.filter(pl.col(TARGET).is_not_null()).sort(["trade_date", "ticker", "market"])
    info = {
        "panel_rows_before_label_filter": rows_before,
        "rows_with_label": joined.height,
        "dates": int(joined["trade_date"].n_unique()),
        "rows_null_raw_label_among_labeled": int(joined[RETURN_COL].null_count()),
        "positive_rate": float(joined[TARGET].mean()),
    }
    return joined, info


def to_sample(joined: pl.DataFrame, design: list[str], *, labeled: bool) -> Sample:
    keys = joined.select(["trade_date", "ticker", "market"])
    X = joined.select(design).to_numpy(order="c").astype(np.float64, copy=False)
    dates = joined["trade_date"].to_numpy().astype("datetime64[D]")
    y = ret = None
    if labeled:
        y = joined[TARGET].to_numpy().astype(np.int8)
        ret = joined[RETURN_COL].to_numpy().astype(np.float64)
    return Sample(X, dates, keys, design, y, ret)


# ---------------------------------------------------------------- metrics
def _proba(model: Any, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def logloss_rows(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


class DailyScorer:
    """Per-date rank IC and log-loss sums; return ranks are cached once."""

    def __init__(self, s: Sample) -> None:
        if s.y is None or s.ret is None:
            raise ValueError("DailyScorer needs labels")
        self.s = s
        self.valid = ~np.isnan(s.ret)
        self.ret_rank: list[np.ndarray | None] = []
        for i in range(s.n_dates):
            a, b = s.offsets[i], s.offsets[i + 1]
            v = self.valid[a:b]
            self.ret_rank.append(rankdata(s.ret[a:b][v]) if v.sum() > 2 else None)
        self.n_rows = np.diff(s.offsets).astype(np.float64)

    def score(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (daily rank IC [D], daily log-loss SUM [D])."""
        s = self.s
        ic = np.full(s.n_dates, np.nan)
        ll = np.zeros(s.n_dates)
        loss = logloss_rows(s.y, p)
        for i in range(s.n_dates):
            a, b = s.offsets[i], s.offsets[i + 1]
            ll[i] = loss[a:b].sum()
            rr = self.ret_rank[i]
            if rr is None:
                continue
            pr = rankdata(p[a:b][self.valid[a:b]])
            if np.std(pr) == 0:
                ic[i] = 0.0
            else:
                ic[i] = np.corrcoef(pr, rr)[0, 1]
        return ic, ll


def summarize(ic: np.ndarray, ll: np.ndarray, n: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    return float(np.nanmean(ic[mask])), float(ll[mask].sum() / n[mask].sum())


# ---------------------------------------------------------------- groups
class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)

    def groups(self, order: list[str]) -> list[list[str]]:
        out: dict[str, list[str]] = {}
        for item in order:
            out.setdefault(self.find(item), []).append(item)
        return list(out.values())


def correlation_groups(
    rho: np.ndarray, names: list[str], threshold: float = CORR_THRESHOLD
) -> list[list[str]]:
    """Union-find merge of |rho| >= threshold (NaN counts as 0)."""
    uf = UnionFind(names)
    r = np.nan_to_num(np.abs(rho))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if r[i, j] >= threshold:
                uf.union(names[i], names[j])
    return uf.groups(names)


def mean_daily_spearman(s: Sample, feature_columns: list[str], date_idx: np.ndarray) -> np.ndarray:
    cols = [s.design.index(c) for c in feature_columns]
    acc = np.zeros((len(cols), len(cols)))
    cnt = np.zeros((len(cols), len(cols)))
    for i in date_idx:
        blk = s.X[s.offsets[i] : s.offsets[i + 1]][:, cols]
        rk = rankdata(blk, axis=0)
        sd = rk.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.corrcoef(rk, rowvar=False)
        c[sd == 0, :] = np.nan
        c[:, sd == 0] = np.nan
        ok = ~np.isnan(c)
        acc[ok] += c[ok]
        cnt[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, acc / cnt, np.nan)


# ---------------------------------------------------------------- permutation
def permuted_block(
    X: np.ndarray, cols: list[int], offsets: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Columns ``cols`` shuffled within each date; one row permutation per date
    shared by every column of the group (feature and its ``_isna`` move together)."""
    out = np.empty((X.shape[0], len(cols)), dtype=X.dtype)
    for i in range(len(offsets) - 1):
        a, b = offsets[i], offsets[i + 1]
        perm = rng.permutation(b - a)
        out[a:b] = X[a:b][:, cols][perm]
    return out


def run_permutation(
    model: Any,
    s: Sample,
    groups: dict[str, list[str]],
    design: list[str],
    *,
    repeats: int,
    log: Callable[[str], None],
) -> dict[str, Any]:
    scorer = DailyScorer(s)
    base_ic, base_ll = scorer.score(_proba(model, s.X))
    D = s.n_dates
    gids = list(groups)
    ic = np.zeros((len(gids), repeats, D))
    ll = np.zeros((len(gids), repeats, D))
    work = s.X.copy()
    t0 = time.time()
    for gi, gid in enumerate(gids):
        feats = groups[gid]
        cols = [design.index(f) for f in feats] + [design.index(f"{f}_isna") for f in feats]
        for r in range(repeats):
            seed = PERM_SEED + 1000 * gi + r
            work[:, cols] = permuted_block(s.X, cols, s.offsets, np.random.default_rng(seed))
            ic[gi, r], ll[gi, r] = scorer.score(_proba(model, work))
        work[:, cols] = s.X[:, cols]
        el = time.time() - t0
        log(f"permutation {gi + 1}/{len(gids)} {gid}: {el / 60:.1f} min elapsed")
    return {
        "group_ids": gids,
        "base_ic": base_ic,
        "base_ll": base_ll,
        "perm_ic": ic,
        "perm_ll": ll,
        "n_rows": scorer.n_rows,
        "seconds": time.time() - t0,
    }


def importance_tables(
    res: dict[str, Any], unique_dates: np.ndarray, groups: dict[str, list[str]], kind: dict[str, str]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    masks = half_masks(unique_dates)
    n = res["n_rows"]
    rows = []
    for subset, m in masks.items():
        b_ic, b_ll = summarize(res["base_ic"], res["base_ll"], n, m)
        for gi, gid in enumerate(res["group_ids"]):
            for r in range(res["perm_ic"].shape[1]):
                p_ic, p_ll = summarize(res["perm_ic"][gi, r], res["perm_ll"][gi, r], n, m)
                rows.append(
                    {
                        "group_id": gid, "level": kind[gid], "n_features": len(groups[gid]),
                        "features": ",".join(groups[gid]), "subset": subset, "repeat": r,
                        "base_rank_ic": b_ic, "perm_rank_ic": p_ic, "delta_rank_ic": b_ic - p_ic,
                        "base_logloss": b_ll, "perm_logloss": p_ll, "delta_logloss": p_ll - b_ll,
                    }
                )
    long = pl.DataFrame(rows)
    # date bootstrap of repeat-averaged per-date drops (full sample)
    rng = np.random.default_rng(BOOT_SEED)
    D = len(unique_dates)
    idx = rng.integers(0, D, size=(N_BOOT, D))
    summ = []
    for gi, gid in enumerate(res["group_ids"]):
        d_ic = res["base_ic"] - res["perm_ic"][gi].mean(axis=0)
        d_ll = res["perm_ll"][gi].mean(axis=0) - res["base_ll"]
        rec: dict[str, Any] = {
            "group_id": gid, "level": kind[gid], "n_features": len(groups[gid]),
            "features": ",".join(groups[gid]),
        }
        for subset, m in masks.items():
            sub = long.filter((pl.col("group_id") == gid) & (pl.col("subset") == subset))
            rec[f"delta_rank_ic_{subset}"] = float(sub["delta_rank_ic"].mean())
            rec[f"delta_rank_ic_sd_{subset}"] = float(sub["delta_rank_ic"].std() or 0.0)
            rec[f"delta_logloss_{subset}"] = float(sub["delta_logloss"].mean())
            rec[f"delta_logloss_sd_{subset}"] = float(sub["delta_logloss"].std() or 0.0)
        boot_ic = np.nanmean(d_ic[idx], axis=1)
        boot_ll = d_ll[idx].sum(axis=1) / n[idx].sum(axis=1)
        rec["delta_rank_ic_ci_lo"], rec["delta_rank_ic_ci_hi"] = (
            float(v) for v in np.percentile(boot_ic, [2.5, 97.5])
        )
        rec["delta_logloss_ci_lo"], rec["delta_logloss_ci_hi"] = (
            float(v) for v in np.percentile(boot_ll, [2.5, 97.5])
        )
        summ.append(rec)
    return long, pl.DataFrame(summ)


# ---------------------------------------------------------------- shape / stability
def classify_shape(curve: np.ndarray, flat_threshold: float) -> str:
    """Plan §4.4. Labels: 평평 · 단조 증가 · 단조 감소 · U자 · 역U자 · 기타."""
    curve = np.asarray(curve, dtype=float)
    if float(curve.max() - curve.min()) < flat_threshold:
        return "평평"
    diffs = np.diff(curve)
    n = len(diffs)
    up, down = int((diffs > 0).sum()), int((diffs < 0).sum())
    if up >= MONOTONE_FRACTION * n:
        return "단조 증가"
    if down >= MONOTONE_FRACTION * n:
        return "단조 감소"
    last = len(curve) - 1
    # us.classify_shape 와 같은 규칙 (plan §6: 두 모듈은 같은 규칙). U자를 먼저 본다.
    imin, imax = int(np.argmin(curve)), int(np.argmax(curve))
    if 0 < imin < last and curve[0] > curve[imin] and curve[last] > curve[imin]:
        return "U자"
    if 0 < imax < last and curve[0] < curve[imax] and curve[last] < curve[imax]:
        return "역U자"
    return "기타"


def flat_threshold(pdp_ranges: np.ndarray) -> float:
    return FLAT_FRACTION * float(np.median(pdp_ranges))


def stable_flags(
    imp_h1: dict[str, float],
    imp_h2: dict[str, float],
    sign_h1: dict[str, float],
    sign_h2: dict[str, float],
    top_k: int = STABLE_TOP_K,
) -> dict[str, bool]:
    """Plan §4.1: top-k in both halves OR bottom half in both, AND same direction sign."""
    names = list(imp_h1)
    n = len(names)

    def ranks(d: dict[str, float]) -> dict[str, int]:
        order = sorted(names, key=lambda k: (-d[k], k))
        return {k: i + 1 for i, k in enumerate(order)}

    r1, r2 = ranks(imp_h1), ranks(imp_h2)
    out = {}
    for k in names:
        top = r1[k] <= top_k and r2[k] <= top_k
        bottom = r1[k] > n / 2 and r2[k] > n / 2
        same_sign = np.sign(sign_h1[k]) == np.sign(sign_h2[k]) and np.sign(sign_h1[k]) != 0
        out[k] = bool((top or bottom) and same_sign)
    return out


# ---------------------------------------------------------------- SHAP
def _positive_class(values: Any) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]
    return values


def _expected(ev: Any) -> float:
    return float(np.ravel(np.asarray(ev))[-1])


def run_shap(
    model: Any, s: Sample, *, log: Callable[[str], None], chunk: int = 20_000
) -> dict[str, Any]:
    import shap

    explainer = shap.TreeExplainer(model)  # default: tree_path_dependent
    out = np.empty(s.X.shape, dtype=np.float64)
    t0 = time.time()
    for a in range(0, s.X.shape[0], chunk):
        out[a : a + chunk] = _positive_class(
            explainer.shap_values(s.X[a : a + chunk], check_additivity=False)
        )
        log(f"shap {min(a + chunk, s.X.shape[0])}/{s.X.shape[0]}: {time.time() - t0:.0f}s")
    rng = np.random.default_rng(CHECK_SEED)
    sub = np.sort(rng.choice(s.X.shape[0], size=min(5000, s.X.shape[0]), replace=False))
    dfun = model.decision_function(s.X[sub])
    err = np.abs(out[sub].sum(axis=1) + _expected(explainer.expected_value) - dfun)
    return {
        "values": out,
        "expected_value": _expected(explainer.expected_value),
        "additivity_max_abs_error": float(err.max()),
        "seconds": time.time() - t0,
    }


def shap_summary(
    shap_values: np.ndarray, s: Sample, features: list[str]
) -> tuple[pl.DataFrame, dict[str, dict[str, float]]]:
    from scipy.stats import spearmanr

    masks = half_masks(s.unique_dates)
    didx = s.date_idx_of_rows()
    D = s.n_dates
    cnt = np.diff(s.offsets).astype(np.float64)
    rank_cols = [s.design.index(f) for f in features]
    isna_cols = [s.design.index(f"{f}_isna") for f in features]
    A = np.abs(shap_values[:, rank_cols])
    B = np.abs(shap_values[:, isna_cols])
    C = np.abs(shap_values[:, rank_cols] + shap_values[:, isna_cols])
    per_date = {}
    for name, M in (("rank", A), ("isna", B), ("combined", C)):
        acc = np.zeros((D, len(features)))
        acc[:] = np.add.reduceat(M, s.offsets[:-1], axis=0)
        per_date[name] = acc
    rng = np.random.default_rng(BOOT_SEED + 1)
    idx = rng.integers(0, D, size=(N_BOOT, D))
    boot = per_date["combined"][idx].sum(axis=1) / cnt[idx].sum(axis=1)[:, None]
    row_half = {k: m[didx] for k, m in masks.items()}
    rows = []
    half_imp: dict[str, dict[str, float]] = {"first": {}, "second": {}}
    half_dir: dict[str, dict[str, float]] = {"first": {}, "second": {}}
    for j, f in enumerate(features):
        x = s.X[:, rank_cols[j]]
        rec: dict[str, Any] = {"feature": f}
        for name in ("rank", "isna", "combined"):
            rec[f"mean_abs_shap_{name}"] = float(per_date[name][:, j].sum() / cnt.sum())
        for subset in SUBSETS:
            m = row_half[subset]
            dm = masks[subset]
            if subset != "full":
                rec[f"mean_abs_shap_combined_{subset}"] = float(
                    per_date["combined"][dm, j].sum() / cnt[dm].sum()
                )
                half_imp[subset][f] = rec[f"mean_abs_shap_combined_{subset}"]
            if np.std(x[m]) == 0 or np.std(shap_values[m, rank_cols[j]]) == 0:
                d = float("nan")
            else:
                d = float(spearmanr(x[m], shap_values[m, rank_cols[j]]).statistic)
            rec["direction_spearman" if subset == "full" else f"direction_spearman_{subset}"] = d
            if subset != "full":
                half_dir[subset][f] = 0.0 if np.isnan(d) else d
        rec["ci_lo"], rec["ci_hi"] = (float(v) for v in np.percentile(boot[:, j], [2.5, 97.5]))
        rows.append(rec)
    df = pl.DataFrame(rows)
    stable = stable_flags(half_imp["first"], half_imp["second"], half_dir["first"], half_dir["second"])
    df = df.with_columns(pl.Series("stable_shap", [stable[f] for f in features]))
    return df, {"imp_first": half_imp["first"], "imp_second": half_imp["second"]}


# ---------------------------------------------------------------- PDP / ICE
def run_pdp(
    model: Any, s: Sample, features: list[str], *, log: Callable[[str], None]
) -> pl.DataFrame:
    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    D = s.n_dates
    cnt = np.diff(s.offsets).astype(np.float64)
    masks = half_masks(s.unique_dates)
    work = s.X.copy()
    rows = []
    t0 = time.time()
    for fi, f in enumerate(features):
        c, cna = s.design.index(f), s.design.index(f"{f}_isna")
        keep_c, keep_na = s.X[:, c].copy(), s.X[:, cna].copy()
        pd_p = np.zeros((GRID_POINTS, D))
        pd_d = np.zeros((GRID_POINTS, D))
        work[:, cna] = 0.0
        for gi, g in enumerate(grid):
            work[:, c] = g
            dfun = model.decision_function(work)
            p = 1.0 / (1.0 + np.exp(-dfun))  # == predict_proba[:,1] for binary log_loss
            pd_p[gi] = np.add.reduceat(p, s.offsets[:-1])
            pd_d[gi] = np.add.reduceat(dfun, s.offsets[:-1])
        work[:, c], work[:, cna] = keep_c, keep_na
        for subset, m in masks.items():
            for gi, g in enumerate(grid):
                rows.append(
                    {
                        "feature": f, "subset": subset, "grid_idx": gi, "grid": float(g),
                        "pdp_p": float(pd_p[gi, m].sum() / cnt[m].sum()),
                        "pdp_decision": float(pd_d[gi, m].sum() / cnt[m].sum()),
                    }
                )
        log(f"pdp {fi + 1}/{len(features)} {f}: {time.time() - t0:.0f}s")
    return pl.DataFrame(rows)


def run_ice(model: Any, s: Sample, features: list[str]) -> pl.DataFrame:
    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    rng = np.random.default_rng(ICE_SEED)
    pick = np.sort(rng.choice(s.X.shape[0], size=min(ICE_ROWS, s.X.shape[0]), replace=False))
    base = s.X[pick]
    tick = s.keys["ticker"].to_numpy()[pick]
    day = s.dates[pick].astype(str)
    frames = []
    for f in features:
        c, cna = s.design.index(f), s.design.index(f"{f}_isna")
        blocks = []
        for g in grid:
            Z = base.copy()
            Z[:, c] = g
            Z[:, cna] = 0.0
            blocks.append(Z)
        Z = np.vstack(blocks)
        dfun = model.decision_function(Z)
        n = len(pick)
        frames.append(
            pl.DataFrame(
                {
                    "feature": f,
                    "trade_date": np.tile(day, GRID_POINTS),
                    "ticker": np.tile(tick, GRID_POINTS),
                    "actual_rank_value": np.tile(base[:, c], GRID_POINTS),
                    "grid_idx": np.repeat(np.arange(GRID_POINTS), n),
                    "grid": np.repeat(grid, n),
                    "ice_decision": dfun,
                    "ice_p": 1.0 / (1.0 + np.exp(-dfun)),
                }
            )
        )
    return pl.concat(frames)


def shape_table(pdp: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    rec: dict[str, dict[str, Any]] = {f: {"feature": f} for f in features}
    for subset in SUBSETS:
        sub = pdp.filter(pl.col("subset") == subset)
        curves = {f: sub.filter(pl.col("feature") == f).sort("grid_idx")["pdp_p"].to_numpy() for f in features}
        ranges = np.array([c.max() - c.min() for c in curves.values()])
        thr = flat_threshold(ranges)
        for f, c in curves.items():
            rec[f][f"pdp_range_{subset}"] = float(c.max() - c.min())
            rec[f][f"shape_{subset}"] = classify_shape(c, thr)
        rec[features[0]][f"flat_threshold_{subset}"] = thr
    return pl.DataFrame(list(rec.values()))


# ---------------------------------------------------------------- orchestration
def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    df.write_parquet(path)


def _cached(run_dir: Path, name: str, fn: Callable[[], Any], log: Callable[[str], None]) -> Any:
    import pickle

    p = run_dir / f"_stage_{name}.pkl"
    if p.exists():
        log(f"stage {name}: resumed from {p.name}")
        return pickle.loads(p.read_bytes())
    value = fn()
    p.write_bytes(pickle.dumps(value))
    return value


def _load_bundle_checked(bundle_dir: Path) -> tuple[Any, dict[str, Any]]:
    from modeler.serving import kr_model as km

    bundle = km.load_bundle(bundle_dir, verify_golden=False)  # manifest + file hashes checked
    try:
        km.verify_bundle_golden(bundle)
    except Exception as exc:  # abort: nothing is computed on a model that fails golden
        raise SystemExit(f"golden verification FAILED, aborting: {exc}") from exc
    info = {
        "golden_verify": "passed",
        "golden_rows": len(bundle.golden["symbols"]),
        "files_sha256": bundle.manifest["files"],
        "manifest_sha256": bundle.manifest["manifest_sha256"],
        "model_id": bundle.manifest["model_id"],
        "model_version": bundle.manifest["model_version"],
        "design_columns": len(bundle.design_columns),
    }
    return bundle, info


def default_bundle_dir() -> Path:
    return stock_data_root() / "kr/output/feature_response_l3/bundle_kr_daily_h20_v1"


def prepare_check(args: argparse.Namespace) -> int:
    """Load + design only. NEVER reads labels. For smoke checks."""
    out = Path(args.out_root) / f"kr_l3_check_{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: _log(m, out / "run.log")  # noqa: E731
    bundle, binfo = _load_bundle_checked(Path(args.bundle_dir))
    ds = stock_data_root() / SNAPSHOT_DIR
    t0 = time.time()
    tr, design = load_design(
        ds / "feat_panel.parquet", list(bundle.feature_columns), list(bundle.design_columns),
        dates_limit=args.dates_limit, log=log,
    )
    s = to_sample(tr, design, labeled=False)
    import resource

    rec: dict[str, Any] = {
        "labels_read": False,
        "bundle": binfo,
        "rows": int(s.X.shape[0]),
        "dates": s.n_dates,
        "design_shape": list(s.X.shape),
        "design_dtype": str(s.X.dtype),
        "design_matrix_mb": s.X.nbytes / 1e6,
        "n_nan": int(np.isnan(s.X).sum()),
        "rank_cols_min": float(np.nanmin(s.X[:, :47])),
        "rank_cols_max": float(np.nanmax(s.X[:, :47])),
        "isna_values": sorted(np.unique(s.X[:, 47:]).tolist()),
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "load_seconds": time.time() - t0,
    }
    t1 = time.time()
    p64 = _proba(bundle.model, s.X)
    rec["predict_seconds"] = time.time() - t1
    p32 = _proba(bundle.model, s.X.astype(np.float32).astype(np.float64))
    rec["float32_roundtrip_max_abs_diff"] = float(np.abs(p64 - p32).max())
    rec["p_raw_range"] = [float(p64.min()), float(p64.max())]
    (out / "check.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    log(json.dumps(rec, ensure_ascii=False))
    return 0


def run(args: argparse.Namespace) -> int:
    run_id = args.resume or f"kr_l3_{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}"
    out_root = Path(args.out_root) if args.out_root else stock_data_root() / "kr/output/feature_response_l3"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = lambda m: _log(m, run_dir / "run.log")  # noqa: E731
    T: dict[str, float] = {}
    t_all = time.time()
    repeats = args.repeats

    bundle, binfo = _load_bundle_checked(Path(args.bundle_dir))
    model = bundle.model
    feats = list(bundle.feature_columns)
    design = list(bundle.design_columns)
    ds = stock_data_root() / SNAPSHOT_DIR
    panel_p, label_p = ds / "feat_panel.parquet", ds / "label_daily.parquet"
    log(f"run_id={run_id} bundle golden verified")

    t0 = time.time()
    tr, design = load_design(panel_p, feats, design, dates_limit=args.dates_limit, log=log)
    joined, linfo = attach_labels(tr, label_p, design)
    del tr
    s = to_sample(joined, design, labeled=True)
    del joined
    T["load_seconds"] = time.time() - t0
    log(f"labeled sample: {linfo}")
    if args.dates_limit is None:
        if abs(linfo["rows_with_label"] - EXPECTED_ROWS) > 0.01 * EXPECTED_ROWS or linfo["dates"] != EXPECTED_DATES:
            raise AssertionError(f"sample differs from plan expectation: {linfo}")
    half_masks(s.unique_dates)

    # correlation groups (labels unused)
    shap_idx = np.arange(0, s.n_dates, SHAP_DATE_STEP)
    rho = mean_daily_spearman(s, feats, shap_idx)
    corr = correlation_groups(rho, feats)
    multi = [g for g in corr if len(g) > 1]
    groups: dict[str, list[str]] = {f"f:{f}": [f] for f in feats}
    kind = {f"f:{f}": "feature" for f in feats}
    for i, g in enumerate(multi):
        groups[f"c{i + 1:02d}"] = g
        kind[f"c{i + 1:02d}"] = "corr_group"
    (run_dir / "groups.json").write_text(
        json.dumps(
            {
                "threshold_abs_rho": CORR_THRESHOLD,
                "rho_dates": [str(d) for d in s.unique_dates[shap_idx]],
                "corr_groups": corr,
                "permutation_groups": groups,
                "features": feats,
                "rho_mean_daily_spearman": [[None if np.isnan(v) else round(float(v), 4) for v in r] for r in rho],
            },
            ensure_ascii=False, indent=1,
        ),
        encoding="utf-8",
    )
    log(f"groups: 47 single + {len(multi)} corr groups (size>=2)")

    # timing + sampling switch (plan §7)
    t1 = time.time()
    _proba(model, s.X)
    t_pred = time.time() - t1
    projected_min = len(groups) * repeats * t_pred / 60.0
    switch = projected_min > PERM_BUDGET_MIN
    log(f"one full predict_proba {t_pred:.1f}s; projected permutation {projected_min:.0f} min; switch={switch}")
    perm_sample = s
    if switch:
        perm_sample = s.select_dates(np.arange(0, s.n_dates, 2))
    T["predict_full_seconds"] = t_pred

    res = _cached(
        run_dir, "perm",
        lambda: run_permutation(model, perm_sample, groups, design, repeats=repeats, log=log), log,
    )
    T["permutation_seconds"] = res["seconds"]
    long, summ = importance_tables(res, perm_sample.unique_dates, groups, kind)
    _write_parquet(long, run_dir / "importance.parquet")
    # stability (permutation): single features only
    singles = summ.filter(pl.col("level") == "feature")
    names = [r["features"] for r in singles.iter_rows(named=True)]
    stab_perm = stable_flags(
        dict(zip(names, singles["delta_rank_ic_first"].to_list())),
        dict(zip(names, singles["delta_rank_ic_second"].to_list())),
        dict(zip(names, singles["delta_rank_ic_first"].to_list())),
        dict(zip(names, singles["delta_rank_ic_second"].to_list())),
    )
    summ = summ.with_columns(
        pl.Series("stable_permutation", [stab_perm.get(f) for f in summ["features"].to_list()]),
        pl.lit(None, dtype=pl.Utf8).alias("expected_sign"),
    )
    _write_parquet(summ, run_dir / "importance_summary.parquet")

    # SHAP / PDP / ICE on every 5th date
    ss = s.select_dates(shap_idx)
    log(f"shap sample: {ss.X.shape[0]} rows, {ss.n_dates} dates")
    sh = _cached(run_dir, "shap", lambda: run_shap(model, ss, log=log), log)
    T["shap_seconds"] = sh["seconds"]
    shap_df, _ = shap_summary(sh["values"], ss, feats)
    pdp = _cached(run_dir, "pdp", lambda: run_pdp(model, ss, feats, log=log), log)
    shapes = shape_table(pdp, feats)
    shap_df = shap_df.join(shapes, on="feature").with_columns(
        pl.lit(None, dtype=pl.Utf8).alias("expected_sign"),
        pl.Series("is_flag_feature", [f in ("px_is_halted", "fin_is_negative_equity", "fin_has_fs") for f in feats]),
    )
    _write_parquet(shap_df, run_dir / "shap_summary.parquet")
    _write_parquet(pdp, run_dir / "pdp.parquet")
    _write_parquet(run_ice(model, ss, feats), run_dir / "ice_sample.parquet")

    # PDP identity check (p == expit(decision)) on a subsample
    sub = ss.X[:5000]
    pid = float(np.abs(_proba(model, sub) - 1 / (1 + np.exp(-model.decision_function(sub)))).max())

    T["total_seconds"] = time.time() - t_all
    manifest = {
        "run_id": run_id,
        "purpose": "KR holdout third opening — serving-model diagnostic, not a verdict (plan 08 §2.2)",
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "bundle": binfo,
        "inputs": {
            "panel": str(panel_p), "panel_sha256": sha256_file(panel_p),
            "labels": str(label_p), "labels_sha256": sha256_file(label_p),
            "dataset_manifest_sha256": sha256_file(ds / "dataset_manifest.json"),
            "labels_used": "holdout label_daily.parquet only (y_up_20d, raw_label_20d)",
            "panel_values": "raw; ranked per (trade_date, market) by kr_model._model_input",
        },
        "sample": {
            **linfo,
            "formation_start": str(SAMPLE_START), "formation_end": str(SAMPLE_END),
            "hard_excluded_from": str(HARD_EXCLUDE_FROM),
            "dates_limit": args.dates_limit,
            "first_date": str(s.unique_dates[0]), "last_date": str(s.unique_dates[-1]),
            "half_first": f"{SAMPLE_START}..{HALF1_END}", "half_second": f"{HALF2_START}..{SAMPLE_END}",
            "n_dates_first": int(half_masks(s.unique_dates)["first"].sum()),
            "n_dates_second": int(half_masks(s.unique_dates)["second"].sum()),
        },
        "seeds": {"permutation_base": PERM_SEED, "bootstrap": BOOT_SEED, "ice": ICE_SEED, "check": CHECK_SEED},
        "method": {
            "permutation": "within trade_date; one row permutation per date shared by group columns incl. _isna; "
            f"R={repeats}; halves computed from the same full-sample permuted prediction (date subset)",
            "permutation_groups": len(groups), "corr_groups_size_ge2": len(multi),
            "corr_rule": f"mean daily Spearman over every {SHAP_DATE_STEP}th date, union-find |rho|>={CORR_THRESHOLD}",
            "shap": "shap.TreeExplainer(model) default tree_path_dependent, positive class log-odds; "
            f"every {SHAP_DATE_STEP}th date; combined = mean|shap_rank+shap_isna|",
            "shap_rows": int(ss.X.shape[0]), "shap_dates": ss.n_dates,
            "shap_additivity_max_abs_error": sh["additivity_max_abs_error"],
            "pdp": f"manual grid, {GRID_POINTS} points 0..1, _isna=0; p = expit(decision_function); same dates as SHAP",
            "pdp_proba_vs_expit_max_abs_diff": pid,
            "ice_rows": ICE_ROWS, "bootstrap": f"{N_BOOT} date resamples, 95% percentile",
            "flat_rule": f"range < {FLAT_FRACTION} x median range", "monotone_rule": f">= {MONOTONE_FRACTION} same-sign adjacent diffs",
            "stability_rule": f"top{STABLE_TOP_K} in both halves or bottom half in both, and same direction sign",
            "design_dtype": "float64",
        },
        "compute": {
            "single_predict_full_seconds": t_pred,
            "projected_permutation_minutes": projected_min,
            "permutation_budget_minutes": PERM_BUDGET_MIN,
            "sampling_switch_every_2nd_date": switch,
            "permutation_sample_dates": perm_sample.n_dates,
            "permutation_sample_rows": int(perm_sample.X.shape[0]),
        },
        "timings": T,
        "versions": _versions(),
        "worktree_git": _git_info(),
        "labels_read": True,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(manifest, summ, shap_df), encoding="utf-8")
    (run_dir / ".done").write_text(dt.datetime.now(dt.UTC).isoformat(), encoding="utf-8")
    log(f"done in {T['total_seconds'] / 60:.1f} min")
    return 0


def render_report(manifest: dict[str, Any], summ: pl.DataFrame, shap_df: pl.DataFrame) -> str:
    lines = [
        f"# KR layer 3 — {manifest['run_id']}",
        "",
        "진단입니다. 판정이 아닙니다. 모든 값은 날짜별 백분위 공간입니다.",
        "",
        f"- 표본: {manifest['sample']['rows_with_label']:,}행, {manifest['sample']['dates']}일, "
        f"표본 전환(매 2일): {manifest['compute']['sampling_switch_every_2nd_date']}",
        f"- 번들 golden: {manifest['bundle']['golden_verify']}, SHAP 가법성 최대 오차 "
        f"{manifest['method']['shap_additivity_max_abs_error']:.2e}",
        "",
        "## 묶음별 permutation importance (IC 감소 큰 순, 전체 표본)",
        "",
        "| 묶음 | 피쳐 수 | ΔrankIC | 95% 구간 | Δlog-loss | 앞 절반 ΔIC | 뒤 절반 ΔIC | 안정 |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ]
    for r in summ.sort("delta_rank_ic_full", descending=True).iter_rows(named=True):
        lines.append(
            f"| {r['group_id']} ({r['features'][:60]}) | {r['n_features']} | {r['delta_rank_ic_full']:.5f} | "
            f"[{r['delta_rank_ic_ci_lo']:.5f}, {r['delta_rank_ic_ci_hi']:.5f}] | {r['delta_logloss_full']:.5f} | "
            f"{r['delta_rank_ic_first']:.5f} | {r['delta_rank_ic_second']:.5f} | {r['stable_permutation']} |"
        )
    lines += [
        "",
        "## SHAP (평균 |SHAP| 큰 순)",
        "",
        "| 피쳐 | 평균 \\|SHAP\\| | 구간 | 방향 | 앞/뒤 방향 | 모양(전체) | 안정 |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for r in shap_df.sort("mean_abs_shap_combined", descending=True).iter_rows(named=True):
        lines.append(
            f"| {r['feature']} | {r['mean_abs_shap_combined']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | "
            f"{r['direction_spearman']:.3f} | {r['direction_spearman_first']:.2f}/{r['direction_spearman_second']:.2f} | "
            f"{r['shape_full']} | {r['stable_shap']} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="modeler.analysis.feature_response.kr")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "prepare-check"):
        p = sub.add_parser(name)
        p.add_argument("--bundle-dir", default=str(default_bundle_dir()))
        p.add_argument("--out-root", default=None)
        p.add_argument("--dates-limit", type=int, default=None)
        if name == "run":
            p.add_argument("--repeats", type=int, default=REPEATS)
            p.add_argument("--resume", default=None, help="existing run_id to resume finished stages")
    args = ap.parse_args(argv)
    if args.cmd == "prepare-check":
        if args.out_root is None:
            ap.error("prepare-check needs --out-root (scratch)")
        return prepare_check(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
