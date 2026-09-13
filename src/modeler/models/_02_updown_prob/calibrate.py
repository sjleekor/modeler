"""calibrate — map a score to a probability without touching valid data.

Plan: ``docs/dev/20260907_model_experiment/04_models_and_calibration.md`` §3.

Two rules decide everything here.

**Valid data is never fit on.** A calibrator fit on the slice it is judged on
would report a calibration the model does not have. So the fit slice is carved
out of the *train* range: its trailing ``frac`` of sessions, with the model
itself trained on what comes before, minus an ``h``-session embargo. The embargo
is not decoration — an h-session label formed on the last model-train session
resolves inside the calibration slice, and without the gap the calibrator would
be reading labels the model already saw.

**Calibration is per (horizon, model, fold).** Nothing is shared across folds:
each fold's calibrator is fit on that fold's own trailing sessions.

``isotonic`` is the plan's default because it is free to be non-monotonic in
*shape* while monotonic in order — it can pull an overconfident boosting tail
back without assuming a sigmoid. ``platt`` is kept beside it for comparison.
On L-B (positive rate 0.2) isotonic's upper steps rest on few names, which is
why the reliability table records the per-bin ``n`` rather than only the gap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from modeler.etl.metrics import PROB_CLIP

VALID_METHODS: tuple[str, ...] = ("none", "isotonic", "platt")
DEFAULT_CAL_FRAC = 0.2


class Calibrator(Protocol):
    """Anything that maps raw scores to probabilities."""

    def transform(self, scores: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class CalSplit:
    """How one fold's train range was divided for calibration.

    ``model_dates`` fits the model, ``cal_dates`` fits the calibrator, and the
    ``embargo`` sessions between them belong to neither.
    """

    model_dates: list
    cal_dates: list
    embargo: int

    @property
    def n_dropped(self) -> int:
        return self.embargo


def split_cal_slice(train_dates: list, horizon: int, frac: float = DEFAULT_CAL_FRAC) -> CalSplit:
    """Carve a calibration slice off the end of a fold's train sessions.

    ``frac`` of the sessions go to the calibrator; the model gets everything
    before them except the last ``horizon`` sessions (the embargo). Raises when
    the split would leave the model with nothing — a fold that small should be
    reported as skipped, not silently trained on a handful of days.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0,1), got {frac}")
    if horizon < 0:
        raise ValueError(f"horizon must be >= 0, got {horizon}")
    dates = sorted(set(train_dates))
    n_cal = max(1, int(round(len(dates) * frac)))
    cal_start = len(dates) - n_cal
    model_end = cal_start - horizon
    if model_end <= 0:
        raise ValueError(
            f"cal slice leaves no model-train sessions: {len(dates)} dates, "
            f"frac={frac}, embargo={horizon}"
        )
    return CalSplit(model_dates=dates[:model_end], cal_dates=dates[cal_start:], embargo=horizon)


@dataclass(frozen=True)
class IsotonicCalibrator:
    """Isotonic map, clipped at the ends (``out_of_bounds="clip"``)."""

    model: IsotonicRegression

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return _clip(self.model.predict(scores))


@dataclass(frozen=True)
class PlattCalibrator:
    """One-variable logistic map (Platt scaling)."""

    model: LogisticRegression

    def transform(self, scores: np.ndarray) -> np.ndarray:
        return _clip(self.model.predict_proba(scores.reshape(-1, 1))[:, 1])


def fit_isotonic(scores: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
    """Fit an isotonic score -> probability map on the calibration slice."""
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(scores, y)
    return IsotonicCalibrator(model=model)


def fit_platt(scores: np.ndarray, y: np.ndarray) -> PlattCalibrator:
    """Fit a one-variable logistic score -> probability map."""
    model = LogisticRegression(solver="lbfgs", max_iter=500)
    model.fit(scores.reshape(-1, 1), y)
    return PlattCalibrator(model=model)


def fit_calibrator(method: str, scores: np.ndarray, y: np.ndarray) -> Calibrator | None:
    """Dispatch on the method name; ``"none"`` returns None.

    Returns None as well when the calibration slice holds a single class —
    there is nothing to map, and an isotonic fit on one class would collapse
    every probability onto that class's rate.
    """
    if method not in VALID_METHODS:
        raise ValueError(f"method must be one of {VALID_METHODS}, got {method!r}")
    if method == "none":
        return None
    if scores.size == 0 or len(np.unique(y)) < 2:
        return None
    if method == "isotonic":
        return fit_isotonic(scores, y)
    return fit_platt(scores, y)


def _clip(p: np.ndarray) -> np.ndarray:
    """Keep probabilities inside the log-loss clip, so a 0 cannot become -inf."""
    return np.clip(p, PROB_CLIP, 1.0 - PROB_CLIP)
