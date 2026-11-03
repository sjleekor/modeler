"""Models and walk-forward training for research A stage 1 (preregistration section 6).

Predictors: PERS (persistence), RIDGE, HGB, IND (industry shrinkage, record only).
Training rows for an evaluation date ``T``: ``input_eligible & label_ok &
label_pub_date <= T & decision_date < T`` (A20). Evaluation rows: ``decision_date == T &
input_eligible``; evaluation later restricts to ``label_ok``.

Every public function is ``@seal.guarded``. On top of the guard's check of the passed
``decision_dates`` against the grant, each function checks that every ``decision_date`` of the
input frame lies inside the passed list, so a DEV grant can never train on sealed rows.

RIDGE design notes: each feature (binary ones too, for simplicity and consistency) is turned
into a per-decision-date percentile rank (P6 ``rank_transform``; train and eval rows are ranked
separately, which is leak-free because the rank is a within-date cross-section), a
``<f>_isna`` indicator (1.0 when the raw value is NaN) is added for every feature, and ranked
NaN is filled with 0.5 (P6 ``ridge_fill``). Train ranks are taken over the train rows
(``label_ok`` only) of each date, eval ranks over the eval rows of the date.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from modeler.models._03_kr_next.preprocess import rank_transform, ridge_fill
from modeler.scores.earnings_a import seal
from modeler.scores.earnings_a.spec import FEATURES, SEEDS

MODEL_NAMES: tuple[str, ...] = ("PERS", "RIDGE", "HGB", "IND")
IND_MIN_GROUP_ROWS = 20
IND_W_GRID = np.linspace(0.0, 1.0, 21)
_RANK_DATE = "trade_date"
_REQUIRED = (
    "decision_date",
    "ticker",
    "corp_code",
    "ksic2",
    "input_eligible",
    "label_ok",
    "y_clip",
    "label_pub_date",
    *FEATURES,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in MODEL_NAMES:
            raise ValueError(f"unknown model {self.name!r}; expected one of {MODEL_NAMES}")

    @property
    def spec_id(self) -> str:
        return f"{self.name}|{json.dumps(self.params, sort_keys=True, default=str)}"


def _check_ds(ds: pd.DataFrame, decision_dates: Sequence[Any], eval_dates: Sequence[Any]) -> None:
    """Raise before any fit when ``ds`` or the eval dates reach outside ``decision_dates``."""
    missing = [c for c in _REQUIRED if c not in ds.columns]
    if missing:
        raise ValueError(f"dataset columns missing: {missing}")
    allowed = {seal._to_date(d) for d in decision_dates}
    n_null = int(ds["decision_date"].isna().sum())
    if n_null:
        raise seal.SealViolation(f"{n_null} row(s) with empty 'decision_date'")
    bad = sorted({seal._to_date(d) for d in ds["decision_date"].unique()} - allowed)
    if bad:
        raise seal.SealViolation(
            f"dataset has {len(bad)} decision date(s) outside the passed list: "
            + ", ".join(d.isoformat() for d in bad)
        )
    bad_eval = sorted({seal._to_date(d) for d in eval_dates} - allowed)
    if bad_eval:
        raise seal.SealViolation(
            "eval date(s) outside the passed list: " + ", ".join(d.isoformat() for d in bad_eval)
        )


def _split(ds: pd.DataFrame, eval_date: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train rows, eval rows) for ``eval_date`` under rule A20."""
    t = pd.Timestamp(seal._to_date(eval_date))
    dd = pd.to_datetime(ds["decision_date"])
    lp = pd.to_datetime(ds["label_pub_date"])
    eligible = ds["input_eligible"].fillna(False).astype(bool)
    label_ok = ds["label_ok"].fillna(False).astype(bool)
    train_mask = eligible & label_ok & (lp <= t) & (dd < t)
    eval_mask = eligible & (dd == t)
    return ds.loc[train_mask], ds.loc[eval_mask]


def _ridge_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank within date, fill 0.5, append ``<f>_isna`` flags (42 columns)."""
    feats = list(FEATURES)
    raw = frame[feats].astype(float)
    flags = raw.isna().astype(float)
    flags.columns = [f"{c}_isna" for c in feats]
    if len(frame) == 0:
        return pd.concat([raw, flags], axis=1)
    codes = pd.to_datetime(frame["decision_date"]).factorize()[0]
    pl_df = pl.DataFrame({_RANK_DATE: codes, **{c: raw[c].to_numpy() for c in feats}}).with_columns(
        [pl.col(c).fill_nan(None) for c in feats]
    )
    filled = ridge_fill(rank_transform(pl_df, feats, date_col=_RANK_DATE), feats)
    ranked = pd.DataFrame(filled.select(feats).to_numpy(), columns=feats, index=frame.index)
    return pd.concat([ranked, flags], axis=1)


def _predict_pers(ev: pd.DataFrame) -> np.ndarray:
    return ev["oi_ttm"].to_numpy(dtype=float)


def _predict_ridge(tr: pd.DataFrame, ev: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    model = Ridge(alpha=float(params["alpha"]))
    model.fit(_ridge_matrix(tr), tr["y_clip"].to_numpy(dtype=float))
    return np.asarray(model.predict(_ridge_matrix(ev)), dtype=float)


def _predict_hgb(tr: pd.DataFrame, ev: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    x_tr = tr[list(FEATURES)].astype(float)
    y_tr = tr["y_clip"].to_numpy(dtype=float)
    x_ev = ev[list(FEATURES)].astype(float)
    preds = []
    for seed in SEEDS:
        model = HistGradientBoostingRegressor(**params, random_state=seed)
        model.fit(x_tr, y_tr)
        preds.append(np.asarray(model.predict(x_ev), dtype=float))
    return np.mean(preds, axis=0)


def _industry_means(tr: pd.DataFrame) -> tuple[pd.Series, float]:
    overall = float(tr["y_clip"].mean())
    grp = tr.groupby("ksic2")["y_clip"].agg(["mean", "size"])
    means = grp.loc[grp["size"] >= IND_MIN_GROUP_ROWS, "mean"]
    return means, overall


def _predict_ind(tr: pd.DataFrame, ev: pd.DataFrame) -> np.ndarray:
    means, overall = _industry_means(tr)

    def m_of(frame: pd.DataFrame) -> np.ndarray:
        return frame["ksic2"].map(means).astype(float).fillna(overall).to_numpy()

    tr_ok = tr.loc[tr["oi_ttm"].notna()]
    oi = tr_ok["oi_ttm"].to_numpy(dtype=float)
    m = m_of(tr_ok)
    y = tr_ok["y_clip"].to_numpy(dtype=float)
    maes = [float(np.mean(np.abs(y - (w * oi + (1.0 - w) * m)))) for w in IND_W_GRID]
    w_best = float(IND_W_GRID[int(np.argmin(maes))])  # ties -> smallest w
    return w_best * ev["oi_ttm"].to_numpy(dtype=float) + (1.0 - w_best) * m_of(ev)


def _fit_predict(tr: pd.DataFrame, ev: pd.DataFrame, spec: ModelSpec) -> np.ndarray:
    if spec.name == "PERS":
        return _predict_pers(ev)
    if len(ev) == 0:
        return np.empty(0, dtype=float)
    if len(tr) == 0:
        raise ValueError(f"no training rows for {spec.name}")
    if spec.name == "RIDGE":
        return _predict_ridge(tr, ev, spec.params)
    if spec.name == "HGB":
        return _predict_hgb(tr, ev, spec.params)
    return _predict_ind(tr, ev)


def _one_date(ds: pd.DataFrame, eval_date: Any, spec: ModelSpec) -> pd.DataFrame:
    tr, ev = _split(ds, eval_date)
    y_hat = _fit_predict(tr, ev, spec)
    return pd.DataFrame(
        {
            "decision_date": ev["decision_date"].to_numpy(),
            "ticker": ev["ticker"].to_numpy(),
            "corp_code": ev["corp_code"].to_numpy(),
            "model": spec.name,
            "spec_id": spec.spec_id,
            "y_hat": y_hat,
        }
    )


@seal.guarded
def fit_predict_date(
    ds: pd.DataFrame,
    *,
    eval_date: date,
    spec: ModelSpec,
    decision_dates: Sequence[date],
    grant: seal.SealGrant,
) -> pd.DataFrame:
    """Train on rows known by ``eval_date`` and predict every input-eligible row of it.

    Returns [decision_date, ticker, corp_code, model, spec_id, y_hat].
    """
    _check_ds(ds, decision_dates, [eval_date])
    return _one_date(ds, eval_date, spec)


@seal.guarded
def walk_forward(
    ds: pd.DataFrame,
    *,
    eval_dates: Sequence[date],
    specs: Sequence[ModelSpec],
    decision_dates: Sequence[date],
    grant: seal.SealGrant,
) -> pd.DataFrame:
    """Concatenate ``fit_predict_date`` over eval_dates x specs (dates ascending, grid order)."""
    _check_ds(ds, decision_dates, eval_dates)
    parts = [_one_date(ds, d, s) for d in sorted(eval_dates, key=seal._to_date) for s in specs]
    cols = ["decision_date", "ticker", "corp_code", "model", "spec_id", "y_hat"]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


def select_spec(
    frame: pd.DataFrame, specs: Sequence[ModelSpec], *, y_col: str = "y_clip"
) -> ModelSpec:
    """Pick the spec with the lowest mean over dates of per-date MAE (pure; no data access).

    ``frame`` holds predictions joined with labels: decision_date, spec_id, y_hat, ``y_col`` and
    optionally label_ok. Ties go to the earlier spec in ``specs`` (grid order).
    """
    df = frame
    if "label_ok" in df.columns:
        df = df.loc[df["label_ok"].fillna(False).astype(bool)]
    df = df.loc[df["y_hat"].notna() & df[y_col].notna()]
    df = df.assign(_ae=(df[y_col] - df["y_hat"]).abs())
    per_date = df.groupby(["spec_id", "decision_date"])["_ae"].mean()
    score = per_date.groupby(level="spec_id").mean()
    best: ModelSpec | None = None
    best_score = np.inf
    for s in specs:
        if s.spec_id not in score.index:
            raise ValueError(f"no predictions for spec {s.spec_id}")
        v = float(score[s.spec_id])
        if v < best_score:
            best, best_score = s, v
    assert best is not None
    return best
