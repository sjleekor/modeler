"""H1 / H2 judgement (preregistration section 7.1-7.2).

H1: HGB against PERS. All three must hold:
  1. pooled MAE_HGB / MAE_PERS <= 0.95
  2. MAE_HGB < MAE_PERS on at least 7 of the 9 decision dates
  3. firm-cluster bootstrap one-sided 97.5% lower bound of mean d > 0,
     d = |y - yhat_PERS| - |y - yhat_HGB|
H2 (only when H1 passes): HGB against RIDGE, same d with RIDGE in place of PERS.
  wins >= 6 of 9 and lower bound > 0; no MAE ratio threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from modeler.scores.earnings_a import seal
from modeler.scores.earnings_a.spec import BOOTSTRAP_DRAWS, BOOTSTRAP_SEED

H1_MAX_MAE_RATIO = 0.95  # section 7.1 (1)
H1_MIN_WINS = 7  # section 7.1 (2), out of 9 dates
H2_MIN_WINS = 6  # section 7.2 (1), out of 9 dates
N_DATES = 9
LOWER_PERCENTILE = 2.5  # one-sided 97.5% lower bound
BOOTSTRAP_CHUNK = 500
PREDICTORS = ("PERS", "RIDGE", "HGB")


def _prepare(preds: pd.DataFrame, labels: pd.DataFrame, y_col: str) -> pd.DataFrame:
    p = preds.copy()
    p["_d"] = p["decision_date"].map(seal._to_date)
    if p.duplicated(["_d", "ticker", "corp_code", "model"]).any():
        raise ValueError("more than one prediction per (date, ticker, model)")
    wide = p.pivot(index=["_d", "ticker", "corp_code"], columns="model", values="y_hat")
    wide = wide.reset_index()
    for m in PREDICTORS:
        if m not in wide.columns:
            raise ValueError(f"predictor {m} missing from predictions")
    lab = labels[["decision_date", "ticker", "label_ok", y_col]].copy()
    lab["_d"] = lab["decision_date"].map(seal._to_date)
    lab = lab.drop(columns="decision_date")
    df = wide.merge(lab, on=["_d", "ticker"], how="inner")
    keep = df["label_ok"].fillna(False).astype(bool) & df[y_col].notna()
    for m in PREDICTORS:
        keep &= df[m].notna()
    return df.loc[keep].reset_index(drop=True)


def _lower_bound(
    firm_codes: np.ndarray, d: np.ndarray, seed: int, draws: int
) -> tuple[float, float]:
    """Firm-cluster bootstrap of the row-weighted mean of ``d``; returns (mean, 2.5th pct)."""
    _, inv = np.unique(firm_codes, return_inverse=True)
    n_firms = int(inv.max()) + 1
    sums = np.bincount(inv, weights=d, minlength=n_firms)
    cnts = np.bincount(inv, minlength=n_firms).astype(float)
    rng = np.random.default_rng(seed)
    probs = np.full(n_firms, 1.0 / n_firms)
    stats = np.empty(draws)
    done = 0
    while done < draws:
        k = min(BOOTSTRAP_CHUNK, draws - done)
        counts = rng.multinomial(n_firms, probs, size=k).astype(float)
        stats[done : done + k] = (counts @ sums) / (counts @ cnts)
        done += k
    return float(d.mean()), float(np.percentile(stats, LOWER_PERCENTILE))


def _compare(
    df: pd.DataFrame, base: str, y_col: str, seed: int, draws: int
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    y = df[y_col].to_numpy(dtype=float)
    e_base = np.abs(y - df[base].to_numpy(dtype=float))
    e_hgb = np.abs(y - df["HGB"].to_numpy(dtype=float))
    work = pd.DataFrame({"_d": df["_d"], "b": e_base, "h": e_hgb})
    per_date = work.groupby("_d")[["b", "h"]].mean()
    wins = int((per_date["h"] < per_date["b"]).sum())
    mean_d, lower = _lower_bound(df["corp_code"].to_numpy(), e_base - e_hgb, seed, draws)
    mae_b, mae_h = float(e_base.mean()), float(e_hgb.mean())
    res = {
        f"mae_{base.lower()}": mae_b,
        "mae_hgb": mae_h,
        "mae_ratio": mae_h / mae_b if mae_b > 0 else float("inf"),
        "wins": wins,
        "mean_d": mean_d,
        "lower": lower,
    }
    return (
        res,
        {str(k): float(v) for k, v in per_date["b"].items()},
        {str(k): float(v) for k, v in per_date["h"].items()},
    )


@seal.guarded
def h1_h2(
    preds: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    decision_dates: Sequence[date],
    grant: seal.SealGrant,
    seed: int = BOOTSTRAP_SEED,
    draws: int = BOOTSTRAP_DRAWS,
    y_col: str = "y_clip",
    n_dates: int = N_DATES,
    min_wins_h1: int = H1_MIN_WINS,
    min_wins_h2: int = H2_MIN_WINS,
) -> dict[str, Any]:
    """Judge H1 (and H2 when H1 passes) on rows with label and all of PERS, RIDGE, HGB."""
    allowed = {seal._to_date(d) for d in decision_dates}
    for name, frame in (("predictions", preds), ("labels", labels)):
        bad = sorted({seal._to_date(d) for d in frame["decision_date"].unique()} - allowed)
        if bad:
            raise seal.SealViolation(
                f"{name} have {len(bad)} decision date(s) outside the passed list: "
                + ", ".join(d.isoformat() for d in bad)
            )
    df = _prepare(preds, labels, y_col)
    dates = sorted(df["_d"].unique())
    if len(dates) != n_dates:
        raise ValueError(f"expected {n_dates} evaluation dates, found {len(dates)}")
    h1, pd_pers, pd_hgb = _compare(df, "PERS", y_col, seed, draws)
    h1["pass"] = bool(
        h1["mae_ratio"] <= H1_MAX_MAE_RATIO and h1["wins"] >= min_wins_h1 and h1["lower"] > 0
    )
    out: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_firms": int(df["corp_code"].nunique()),
        "n_dates": len(dates),
        "seed": int(seed),
        "draws": int(draws),
        "h1": h1,
        "h2": None,
        "per_date_mae": {"PERS": pd_pers, "HGB": pd_hgb},
    }
    if h1["pass"]:
        h2, pd_ridge, _ = _compare(df, "RIDGE", y_col, seed, draws)
        h2["pass"] = bool(h2["wins"] >= min_wins_h2 and h2["lower"] > 0)
        out["h2"] = h2
        out["per_date_mae"]["RIDGE"] = pd_ridge
    return out
