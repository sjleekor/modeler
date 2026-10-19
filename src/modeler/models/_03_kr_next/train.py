"""Training runner of the next KR model: grid CV, selection, final fit, bundle.

Spec: C-4, G-10..G-17.

Everything here is deterministic code that was fixed before any label was opened. The CLI
(``python -m modeler.models._03_kr_next.train``) is the only entry that touches real labels and it
REFUSES to run unless the preregistration repository has the git tag ``kr-next-frozen`` (F-5).
The functions below take a panel as a polars frame and never read the lake themselves.

Panel contract (columns): ``trade_date, market, ticker``, the feature columns (raw values, NULL
kept), ``y`` (percent rank of the label, NULL on rows without a usable label), ``trainable`` and
``label_end_date`` (the CLI renames ``label_end_date_c1_20d``). A row is used for training only if
``trainable`` and ``y`` is not NULL and its label end date is not NULL.

Choices where the spec was silent (also listed in the task report):

* Features are rank-transformed once, over *all rows of the panel on that date* (the kept
  universe, both markets), before any filtering for folds. That is the cross-section the serving
  side has; the label ``y`` is ranked over the trainable subset of the same rows.
* Early stopping uses an explicit time-ordered validation set (G-11): the last 20% of the fold's
  training dates; the fit rows end 20 sessions earlier and, in addition, their ``label_end_date``
  must precede the first validation date (c1 labels end t+21 sessions, so the 20-session gap alone
  would leave one session of overlap).
* sklearn keeps every tree it grew (it does not roll back to the best). The best iteration is the
  argmax of ``validation_score_`` (number of trees) and CV predictions are taken at that many
  trees (``staged_predict``), so the CV score matches a model refit with ``max_iter = best``.
* The score of a (point, fold) is the mean over validation dates of the per-date Spearman
  (both markets in one cross-section); validation rows need ``trainable`` and ``y``.
* Every (point, fold) is fit with seeds 0..4 (decisions 2026-10-19 #2). The point score is the mean
  over seeds and folds; the tie band is max(fold SE, seed SE) of the best point (ddof 1).
  The bundle model is one fit with random_state 0.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import subprocess
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from modeler.etl.splits_label_end import LabelEndFold, fold_masks, walk_forward_label_end_splits
from modeler.models._03_kr_next import bundle as bundle_mod
from modeler.models._03_kr_next import feature_set as fs
from modeler.models._03_kr_next import preprocess as pp

DATE_COL = "trade_date"
END_COL = "label_end_date"
Y_COL = "y"
TRAINABLE_COL = "trainable"
KEYS = ("trade_date", "market", "ticker")

HORIZON = 20
EMBARGO = 20
N_FOLDS = 5
#: c1 labels end t + 1 + h sessions (splits_label_end docstring: pass h + 2)
MIN_TRAIN_SESSIONS = HORIZON + 2
MAX_ITER = 500
N_ITER_NO_CHANGE = 20
INNER_VAL_FRACTION = 0.2
INNER_GAP_SESSIONS = 20
RIDGE_ALPHA = 100.0
TRAIN_START = date(2017, 1, 1)
CV_END_DEFAULT = date(2025, 7, 31)
PREREG_TAG = "kr-next-frozen"
SEEDS: tuple[int, ...] = tuple(range(5))
#: the bundle model is ONE fit with this seed, declared before any result (decisions #2)
FINAL_SEED = 0

GRID_AXES: dict[str, tuple[float | int, ...]] = {
    "learning_rate": (0.03, 0.1),
    "max_leaf_nodes": (15, 31),
    "min_samples_leaf": (200, 1000),
    "l2_regularization": (0.0, 1.0),
}
GRID: tuple[dict[str, float | int], ...] = tuple(
    dict(zip(GRID_AXES, vals, strict=True)) for vals in itertools.product(*GRID_AXES.values())
)
#: tie order of the selection rule: (key, preferred value), most important first (G-11)
TIE_PREFERENCE: tuple[tuple[str, float | int], ...] = (
    ("l2_regularization", 1.0),
    ("min_samples_leaf", 1000),
    ("max_leaf_nodes", 15),
    ("learning_rate", 0.03),
)
DEFAULT_FEATURES: tuple[str, ...] = fs.NOFLOW_COLS
#: R1 sensitivity (G-14): columns nulled where the flag column is TRUE
R1_FLAG_COL = "fin_ttm_conflict_v2"
R1_NULL_COLS: tuple[str, ...] = ("fin_roa_v2", "fin_roe_v2")

NullMask = tuple[str, Sequence[str]]


# --------------------------------------------------------------------------- preparation


def prepare_panel(
    panel: pl.DataFrame,
    feature_cols: Sequence[str],
    *,
    null_mask: NullMask | None = None,
) -> pl.DataFrame:
    """Keys + labels + rank-transformed feature columns.

    ``null_mask=(flag_col, cols)`` sets ``cols`` to NULL where ``flag_col`` is TRUE *before* the
    rank (R1 sensitivity, G-14). Feature columns end up Float64.
    """
    cols = list(feature_cols)
    need = [*KEYS, Y_COL, TRAINABLE_COL, END_COL, *cols]
    if null_mask is not None:
        need.append(null_mask[0])
    miss = [c for c in need if c not in panel.columns]
    if miss:
        raise ValueError(f"panel lacks columns: {sorted(set(miss))}")
    df = panel.select(list(dict.fromkeys(need)))
    df = df.with_columns([pl.col(c).cast(pl.Float64) for c in cols])
    if null_mask is not None:
        flag, mcols = null_mask
        bad = [c for c in mcols if c not in cols]
        if bad:
            raise ValueError(f"null_mask columns not in the feature list: {bad}")
        df = df.with_columns(
            [
                pl.when(pl.col(flag).fill_null(False)).then(None).otherwise(pl.col(c)).alias(c)
                for c in mcols
            ]
        )
    df = pp.rank_transform(df, cols)
    keep = [c for c in df.columns if c != (null_mask[0] if null_mask else None)]
    return df.select(keep)


def _usable(df: pl.DataFrame) -> np.ndarray:
    return (
        df.select(
            (
                pl.col(TRAINABLE_COL).fill_null(False)
                & pl.col(Y_COL).is_not_null()
                & pl.col(END_COL).is_not_null()
            ).alias("u")
        )
        .to_series()
        .to_numpy()
    )


def make_folds(prepared: pl.DataFrame) -> list[LabelEndFold]:
    """5 expanding folds, embargo 20, purge by ``label_end_date`` (the existing API)."""
    dates = prepared.get_column(DATE_COL).unique().sort()
    return walk_forward_label_end_splits(
        dates,
        horizon=HORIZON,
        embargo=EMBARGO,
        n_folds=N_FOLDS,
        min_train_sessions=MIN_TRAIN_SESSIONS,
    )


@dataclass
class _FoldIdx:
    fold: LabelEndFold
    fit: np.ndarray  # row positions: inner fit set
    inner_val: np.ndarray  # row positions: early-stopping validation
    valid: np.ndarray  # row positions: fold validation (scored)


@dataclass
class _Ctx:
    prepared: pl.DataFrame
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    folds: list[_FoldIdx]


def inner_split(
    dates: np.ndarray, ends: np.ndarray, train_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit / early-stopping rows inside one fold's training rows (G-11).

    Validation = rows of the last 20% of the training dates. Fit = rows on dates up to
    ``INNER_GAP_SESSIONS`` sessions before the validation start whose label ends before the first
    validation date.
    """
    tdates = np.unique(dates[train_idx])
    n = len(tdates)
    n_val = max(1, int(round(INNER_VAL_FRACTION * n)))
    last_fit = n - n_val - INNER_GAP_SESSIONS - 1
    if last_fit < 0:
        raise ValueError(
            f"fold has only {n} training dates; too few for the inner validation split"
        )
    val_start = tdates[n - n_val]
    d = dates[train_idx]
    e = ends[train_idx]
    fit = train_idx[(d <= tdates[last_fit]) & (e < val_start)]
    inner_val = train_idx[d >= val_start]
    return fit, inner_val


def _context(
    prepared: pl.DataFrame,
    feature_cols: Sequence[str],
    valid_label_end_max: date | None = None,
) -> _Ctx:
    folds = make_folds(prepared)
    masks = fold_masks(prepared, folds, date_col=DATE_COL, end_col=END_COL)
    usable = _usable(prepared)
    dates = prepared.get_column(DATE_COL).to_numpy().astype("datetime64[D]")
    ends = prepared.get_column(END_COL).to_numpy().astype("datetime64[D]")
    ends_ok = np.ones(len(dates), dtype=bool)
    if valid_label_end_max is not None:
        ends_ok = ends <= np.datetime64(valid_label_end_max)
    out: list[_FoldIdx] = []
    for m in masks:
        train_idx = np.flatnonzero(m.train.to_numpy() & usable)
        valid_idx = np.flatnonzero(m.valid.to_numpy() & usable & ends_ok)
        fit, inner_val = inner_split(dates, ends, train_idx)
        out.append(_FoldIdx(fold=m.fold, fit=fit, inner_val=inner_val, valid=valid_idx))
    return _Ctx(
        prepared=prepared,
        X=bundle_mod.model_matrix(prepared, list(feature_cols)),
        y=prepared.get_column(Y_COL).cast(pl.Float64).to_numpy(),
        dates=dates,
        folds=out,
    )


# --------------------------------------------------------------------------- model pieces


def _hgb(point: dict[str, Any], *, max_iter: int, early: bool, seed: int = 0):
    return HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=point["learning_rate"],
        max_iter=max_iter,
        max_leaf_nodes=int(point["max_leaf_nodes"]),
        min_samples_leaf=int(point["min_samples_leaf"]),
        l2_regularization=point["l2_regularization"],
        max_features=1.0,
        early_stopping=early,
        scoring="loss",
        n_iter_no_change=N_ITER_NO_CHANGE,
        random_state=seed,
    )


def _fit_early(ctx: _Ctx, fi: _FoldIdx, point: dict[str, Any], seed: int = 0):
    """Fit one fold with the explicit validation set; return (model, best_iter)."""
    model = _hgb(point, max_iter=MAX_ITER, early=True, seed=seed)
    model.fit(ctx.X[fi.fit], ctx.y[fi.fit], X_val=ctx.X[fi.inner_val], y_val=ctx.y[fi.inner_val])
    # validation_score_[i] is the score after i trees (index 0 = the initial prediction)
    best = max(1, int(np.argmax(model.validation_score_)))
    return model, best


def _predict_at(model, X: np.ndarray, n_trees: int) -> np.ndarray:
    for i, pred in enumerate(model.staged_predict(X), start=1):
        if i == n_trees:
            return pred
    raise ValueError(f"model has fewer than {n_trees} iterations")


def mean_date_spearman(dates: np.ndarray, pred: np.ndarray, y: np.ndarray) -> float:
    """Mean over dates of Spearman(pred, y); dates without a defined correlation are skipped."""
    if len(dates) == 0:
        return float("nan")
    per = (
        pl.DataFrame({"d": dates, "p": pred, "y": y})
        .group_by("d")
        .agg(pl.corr("p", "y", method="spearman").alias("ic"))
        .get_column("ic")
        .drop_nans()
        .drop_nulls()
    )
    return float(per.mean()) if len(per) else float("nan")


def _score_fold(ctx: _Ctx, fi: _FoldIdx, point: dict[str, Any], seed: int = 0) -> tuple[float, int]:
    model, best = _fit_early(ctx, fi, point, seed=seed)
    pred = _predict_at(model, ctx.X[fi.valid], best)
    return mean_date_spearman(ctx.dates[fi.valid], pred, ctx.y[fi.valid]), best


# --------------------------------------------------------------------------- grid CV


@dataclass
class ScoreMatrix:
    """Grid result: ``scores[p, f, s]`` and ``best_iter[p, f, s]`` (point, fold, seed).

    HGB is NOT seed-free: with more than 200k fit rows the binning thresholds come from a random
    subsample whose draw depends on ``random_state`` (decisions 2026-10-19 #2, reverting G-10).
    """

    points: list[dict[str, Any]]
    fold_ids: list[int]
    seeds: list[int]
    scores: np.ndarray
    best_iter: np.ndarray

    def __post_init__(self) -> None:
        if self.scores.ndim != 3 or self.best_iter.shape != self.scores.shape:
            raise ValueError("scores and best_iter must both be [point, fold, seed]")

    def fold_means(self) -> np.ndarray:
        """[point, fold]: mean over seeds."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a fold can be all-NaN
            return np.nanmean(self.scores, axis=2)

    def point_means(self) -> np.ndarray:
        """[point]: mean over folds and seeds."""
        return np.nanmean(self.scores, axis=(1, 2))

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": self.points,
            "fold_ids": self.fold_ids,
            "seeds": self.seeds,
            "axes": ["point", "fold", "seed"],
            "scores": self.scores.tolist(),
            "best_iter": self.best_iter.tolist(),
            "fold_mean_scores": self.fold_means().tolist(),
            "mean_scores": self.point_means().tolist(),
        }


def cv_grid(
    panel: pl.DataFrame,
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
    *,
    points: Sequence[dict[str, Any]] = GRID,
    seeds: Sequence[int] = SEEDS,
    null_mask: NullMask | None = None,
    valid_label_end_max: date | None = None,
    prepared: bool = False,
    progress: Callable[[str], None] | None = None,
) -> ScoreMatrix:
    """Score every grid point on every fold with every seed (G-11).

    ``prepared=True`` skips the rank transform.
    """
    prep = panel if prepared else prepare_panel(panel, feature_cols, null_mask=null_mask)
    ctx = _context(prep, feature_cols, valid_label_end_max)
    seeds = [int(s) for s in seeds]
    nf = len(ctx.folds)
    scores = np.full((len(points), nf, len(seeds)), np.nan)
    iters = np.zeros((len(points), nf, len(seeds)), dtype=int)
    for p, point in enumerate(points):
        for f, fi in enumerate(ctx.folds):
            for k, seed in enumerate(seeds):
                scores[p, f, k], iters[p, f, k] = _score_fold(ctx, fi, point, seed)
        if progress:
            progress(f"point {p + 1}/{len(points)} {point}: mean {np.nanmean(scores[p]):.5f}")
    return ScoreMatrix(
        points=[dict(p) for p in points],
        fold_ids=[fi.fold.fold_id for fi in ctx.folds],
        seeds=seeds,
        scores=scores,
        best_iter=iters,
    )


def _sd_over(values: np.ndarray) -> tuple[float, float]:
    """(sd with ddof 1, standard error) of the non-NaN values; (0, 0) with fewer than two."""
    v = values[~np.isnan(values)]
    if len(v) < 2:
        return 0.0, 0.0
    sd = float(np.std(v, ddof=1))
    return sd, sd / float(np.sqrt(len(v)))


def select_point(matrix: ScoreMatrix) -> dict[str, Any]:
    """Best seed-and-fold mean; among points within the tie band prefer the more regularized.

    Tie band = max(fold SE, seed SE) of the best point: fold SE = sd over folds of the seed-mean
    (ddof 1) / sqrt(n_folds); seed SE = sd over seeds of the fold-mean (ddof 1) / sqrt(n_seeds).
    Ties are then broken by ``TIE_PREFERENCE`` (G-11, decisions #2).

    Returns ``{"index", "point", "mean", "best_index", "se", "fold_se", "seed_se", "seed_sd",
    "candidates"}`` where ``se`` is the band width and ``seed_sd`` is the seed spread of the
    *chosen* point (sd over seeds of its fold-mean scores).
    """
    means = matrix.point_means()
    if np.all(np.isnan(means)):
        raise ValueError("no grid point has a score")
    best = int(np.nanargmax(means))
    _, fold_se = _sd_over(matrix.fold_means()[best])
    _, seed_se = _sd_over(np.nanmean(matrix.scores[best], axis=0))
    se = max(fold_se, seed_se)
    cand = [i for i in range(len(means)) if not np.isnan(means[i]) and means[i] >= means[best] - se]

    def key(i: int) -> tuple:
        pt = matrix.points[i]
        return (
            *(pt[k] != want for k, want in TIE_PREFERENCE),
            -float(means[i]),
            i,
        )

    chosen = min(cand, key=key)
    seed_sd, _ = _sd_over(np.nanmean(matrix.scores[chosen], axis=0))
    return {
        "index": chosen,
        "point": dict(matrix.points[chosen]),
        "mean": float(means[chosen]),
        "best_index": best,
        "se": se,
        "fold_se": fold_se,
        "seed_se": seed_se,
        "seed_sd": seed_sd,
        "candidates": cand,
    }


def final_max_iter(matrix: ScoreMatrix, index: int) -> int:
    """Median best iteration over folds and seeds of the chosen point, round half up (G-11)."""
    return max(1, int(np.floor(float(np.median(matrix.best_iter[index])) + 0.5)))


def determinism_check(
    panel: pl.DataFrame,
    point: dict[str, Any],
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
    *,
    seed: int = 0,
    null_mask: NullMask | None = None,
    valid_label_end_max: date | None = None,
    prepared: bool = False,
) -> dict[str, Any]:
    """Fit the last fold twice with the same seed; predictions and best_iter must be identical.

    The same seed must reproduce (other seeds may legitimately differ, decisions #2). A mismatch
    raises ``AssertionError`` (stop and report, no rule change).
    """
    prep = panel if prepared else prepare_panel(panel, feature_cols, null_mask=null_mask)
    ctx = _context(prep, feature_cols, valid_label_end_max)
    fi = ctx.folds[-1]
    runs: list[tuple[np.ndarray, int]] = []
    for _ in range(2):
        model, best = _fit_early(ctx, fi, point, seed=seed)
        runs.append((_predict_at(model, ctx.X[fi.valid], best), best))
    (p0, b0), (p1, b1) = runs
    if b0 != b1 or not np.array_equal(p0, p1):
        raise AssertionError(
            f"two fits with random_state={seed} differ "
            f"(best iter {b0} vs {b1}, max abs diff {float(np.max(np.abs(p0 - p1))):.3g})"
        )
    return {
        "seed": seed,
        "repeats": 2,
        "identical": True,
        "fold_id": fi.fold.fold_id,
        "n_valid_rows": int(len(fi.valid)),
        "best_iter": b0,
    }


# --------------------------------------------------------------------------- final fits


def _final_rows(prep: pl.DataFrame, start_date: date, end_date: date | None) -> pl.DataFrame:
    keep = _usable(prep) & (
        prep.get_column(DATE_COL).to_numpy().astype("datetime64[D]") >= np.datetime64(start_date)
    )
    if end_date is not None:
        keep &= prep.get_column(DATE_COL).to_numpy().astype("datetime64[D]") <= np.datetime64(
            end_date
        )
    rows = prep.filter(pl.Series(keep))
    if rows.is_empty():
        raise ValueError("no trainable rows for the final fit")
    return rows


def fit_final(
    panel: pl.DataFrame,
    point: dict[str, Any],
    max_iter: int,
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
    *,
    null_mask: NullMask | None = None,
    start_date: date = TRAIN_START,
    end_date: date | None = None,
    prepared: bool = False,
) -> HistGradientBoostingRegressor:
    """All trainable rows from ``start_date``, fixed ``max_iter``, no early stopping (G-12)."""
    prep = panel if prepared else prepare_panel(panel, feature_cols, null_mask=null_mask)
    rows = _final_rows(prep, start_date, end_date)
    model = _hgb(point, max_iter=int(max_iter), early=False, seed=FINAL_SEED)
    model.fit(
        bundle_mod.model_matrix(rows, list(feature_cols)),
        rows.get_column(Y_COL).cast(pl.Float64).to_numpy(),
    )
    return model


def fit_ridge(
    panel: pl.DataFrame,
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
    *,
    alpha: float = RIDGE_ALPHA,
    start_date: date = TRAIN_START,
    end_date: date | None = None,
    prepared: bool = False,
) -> Ridge:
    """Ridge baseline on the rank inputs with NULL -> 0.5, same rows as ``fit_final`` (G-17)."""
    prep = panel if prepared else prepare_panel(panel, feature_cols)
    rows = _final_rows(prep, start_date, end_date)
    model = Ridge(alpha=alpha)
    model.fit(
        bundle_mod.ridge_matrix(rows, list(feature_cols)),
        rows.get_column(Y_COL).cast(pl.Float64).to_numpy(),
    )
    return model


def cv_ridge(
    panel: pl.DataFrame,
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
    *,
    alpha: float = RIDGE_ALPHA,
    valid_label_end_max: date | None = None,
    prepared: bool = False,
) -> list[float]:
    """Per-fold mean date Spearman of the Ridge baseline on the same folds (record only)."""
    prep = panel if prepared else prepare_panel(panel, feature_cols)
    ctx = _context(prep, feature_cols, valid_label_end_max)
    Xr = bundle_mod.ridge_matrix(prep, list(feature_cols))
    out: list[float] = []
    for fi in ctx.folds:
        train = np.concatenate([fi.fit, fi.inner_val])
        model = Ridge(alpha=alpha).fit(Xr[train], ctx.y[train])
        out.append(
            mean_date_spearman(ctx.dates[fi.valid], model.predict(Xr[fi.valid]), ctx.y[fi.valid])
        )
    return out


def cv_point(
    panel: pl.DataFrame,
    point: dict[str, Any],
    feature_cols: Sequence[str] = DEFAULT_FEATURES,
    *,
    seeds: Sequence[int] = SEEDS,
    null_mask: NullMask | None = None,
    valid_label_end_max: date | None = None,
) -> dict[str, Any]:
    """CV of one point (R1 sensitivity): seed-mean fold scores, median-over-seeds best iters."""
    m = cv_grid(
        panel,
        feature_cols,
        points=[point],
        seeds=seeds,
        null_mask=null_mask,
        valid_label_end_max=valid_label_end_max,
    )
    return {
        "fold_scores": m.fold_means()[0].tolist(),
        "mean": float(m.point_means()[0]),
        "best_iter": np.median(m.best_iter[0], axis=1).tolist(),
    }


# --------------------------------------------------------------------------- CLI guard


class FrozenTagMissing(RuntimeError):
    """The preregistration repository has no ``kr-next-frozen`` tag: labels must stay closed."""


def require_frozen_tag(prereg_repo: str | Path, tag: str = PREREG_TAG) -> str:
    """Return the commit the tag points to, or raise ``FrozenTagMissing``."""
    repo = Path(prereg_repo)
    if not repo.is_dir():
        raise FrozenTagMissing(f"prereg repo not found: {repo}")
    ok = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "-q", "--verify", f"refs/tags/{tag}"],
        capture_output=True,
        text=True,
    )
    if ok.returncode != 0:
        raise FrozenTagMissing(
            f"git tag {tag!r} does not exist in {repo}; "
            "the labels must not be opened before the freeze"
        )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"refs/tags/{tag}^{{commit}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return commit.stdout.strip()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="python -m modeler.models._03_kr_next.train", description=__doc__.split("\n")[0]
    )
    ap.add_argument("--lake", required=True, help="stock_data lake root (kr)")
    ap.add_argument("--snapshot", required=True, help="snapshot id S (recorded in the manifest)")
    ap.add_argument("--out", required=True, help="new bundle directory (must not exist)")
    ap.add_argument(
        "--prereg-repo", required=True, help="repository that holds the kr-next-frozen tag"
    )
    ap.add_argument(
        "--panel",
        default=None,
        help="labeled panel parquet (default <lake>/datasets/kr_next/<snapshot>/"
        "panel_labeled.parquet)",
    )
    ap.add_argument(
        "--prereg-doc", default=None, help="preregistration document (sha256 is recorded)"
    )
    ap.add_argument(
        "--cv-end",
        default=CV_END_DEFAULT.isoformat(),
        help="last date of the grid-selection period",
    )
    ap.add_argument(
        "--r1-sensitivity", action="store_true", help="also run the R1 sensitivity CV (G-14)"
    )
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    # 1) guards first: nothing is read before the freeze tag exists and the target is free
    try:
        prereg_commit = require_frozen_tag(args.prereg_repo)
    except FrozenTagMissing as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.exists():
        print(f"REFUSED: --out already exists, not overwriting: {out}", file=sys.stderr)
        return 2

    panel_path = (
        Path(args.panel)
        if args.panel
        else (Path(args.lake) / "datasets" / "kr_next" / args.snapshot / "panel_labeled.parquet")
    )
    if not panel_path.is_file():
        print(f"REFUSED: labeled panel not found: {panel_path}", file=sys.stderr)
        return 2
    doc_sha = None
    if args.prereg_doc:
        doc_sha = hashlib.sha256(Path(args.prereg_doc).read_bytes()).hexdigest()
    cv_end = date.fromisoformat(args.cv_end)

    panel = pl.read_parquet(panel_path)
    legacy = "label_end_date_c1_20d"
    if END_COL not in panel.columns and legacy in panel.columns:
        panel = panel.rename({legacy: END_COL})
    cols = list(DEFAULT_FEATURES)

    def log(msg: str) -> None:
        print(msg, flush=True)

    # 2) selection period: grid CV on dates up to cv_end only (the later period stays closed)
    cv_panel = panel.filter(pl.col(DATE_COL) <= cv_end)
    prep_cv = prepare_panel(cv_panel, cols)
    matrix = cv_grid(prep_cv, cols, valid_label_end_max=cv_end, prepared=True, progress=log)
    chosen = select_point(matrix)
    point = chosen["point"]
    max_iter = final_max_iter(matrix, chosen["index"])
    log(f"chosen {point} max_iter={max_iter}")
    det = determinism_check(
        prep_cv, point, cols, seed=FINAL_SEED, valid_label_end_max=cv_end, prepared=True
    )
    ridge_cv = cv_ridge(prep_cv, cols, valid_label_end_max=cv_end, prepared=True)
    extra: dict[str, Any] = {
        "selection": {k: v for k, v in chosen.items() if k != "point"},
        "seeds": list(SEEDS),
        "final_seed": FINAL_SEED,
        "seed_spread": {
            "seed_sd_of_fold_mean": chosen["seed_sd"],
            "seed_se": chosen["seed_se"],
            "fold_se": chosen["fold_se"],
            "note": "HGB binning subsample (n > 2e5 rows) depends on random_state",
        },
        "score_matrix_shape": list(matrix.scores.shape),
        "ridge_cv_fold_scores": ridge_cv,
        "cv_end": cv_end.isoformat(),
        "trials": {"hgb_grid_points": len(GRID), "ridge": 1},
        "panel_path": str(panel_path),
    }
    if args.r1_sensitivity:
        extra["r1_sensitivity"] = cv_point(
            cv_panel,
            point,
            cols,
            null_mask=(R1_FLAG_COL, R1_NULL_COLS),
            valid_label_end_max=cv_end,
        )

    # 3) final fit on every matured trainable row from 2017-01-01 (G-12), then the bundle
    prep_all = prepare_panel(panel, cols)
    model = fit_final(prep_all, point, max_iter, cols, prepared=True)
    ridge = fit_ridge(prep_all, cols, prepared=True)
    golden = prep_all.filter(pl.Series(_usable(prep_all)))
    git = bundle_mod.git_state(Path(__file__).resolve().parent)
    bundle_mod.write_bundle(
        out,
        model=model,
        ridge=ridge,
        ridge_alpha=RIDGE_ALPHA,
        feature_cols=cols,
        golden_frame=golden,
        params={
            **point,
            "max_iter_grid": MAX_ITER,
            "max_features": 1.0,
            "n_iter_no_change": N_ITER_NO_CHANGE,
            "scoring": "loss",
        },
        max_iter=max_iter,
        score_matrix=matrix.to_dict(),
        determinism=det,
        snapshot_id=args.snapshot,
        prereg_tag=PREREG_TAG,
        prereg_commit=prereg_commit,
        prereg_doc_sha256=doc_sha,
        modeler_commit=git["commit"],
        modeler_dirty=git["dirty"],
        preprocessing={
            "rank": "per trade_date, both markets one cross-section, average ties, "
            "(rank-1)/(n-1), n=1 -> 0.5, NULL kept",
            "ridge_fill": pp.RIDGE_FILL_VALUE,
            "train_start": TRAIN_START.isoformat(),
            "horizon": HORIZON,
            "embargo": EMBARGO,
            "n_folds": N_FOLDS,
        },
        extra=extra,
    )
    log(f"bundle written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
