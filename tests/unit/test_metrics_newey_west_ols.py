"""Tests for the HAC-corrected OLS Phase C estimates its delta with.

Design: ``docs/dev/20260829_macro_features/01_design/03_stage1b_conditional_ic_phase_c.md``
§4.2, §7.4.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from modeler.etl.metrics import newey_west_ols, newey_west_tstat


def _sessions(n: int) -> np.ndarray:
    return np.arange(1, n + 1)


@pytest.mark.parametrize("lag", [0, 5, 19, 59])
def test_a_constant_only_fit_reproduces_newey_west_tstat(lag: int) -> None:
    """The two are the same estimator written twice: a constant-only HAC
    regression's intercept t is exactly ``newey_west_tstat``'s. Pinning it here
    is what lets Phase C claim its delta uses "the cell's own convention"."""
    rng = np.random.default_rng(11)
    y = rng.normal(0.01, 0.05, 600)
    sessions = _sessions(600)

    assert newey_west_ols(y, None, sessions, lag)["t_alpha"] == pytest.approx(
        newey_west_tstat(y, sessions, lag), rel=1e-12, abs=1e-12
    )


def test_the_parity_holds_on_a_gapped_session_index() -> None:
    """A market closure must not let two observations count as adjacent — the
    same gap rule governs both functions."""
    rng = np.random.default_rng(3)
    n = 400
    sessions = np.sort(rng.choice(np.arange(1, 900), size=n, replace=False))
    y = rng.normal(0.0, 0.03, n)
    for lag in (5, 19):
        assert newey_west_ols(y, None, sessions, lag)["t_alpha"] == pytest.approx(
            newey_west_tstat(y, sessions, lag), rel=1e-12, abs=1e-12
        )


def test_the_slope_on_a_binary_regressor_is_the_difference_of_means() -> None:
    """§4.1: ``delta`` is exactly ``mean(IC | s=1) - mean(IC | s=0)``.

    That identity is why the contract can state the estimand as a difference of
    conditional means while the code fits a regression — one HAC variance then
    covers both sides.
    """
    rng = np.random.default_rng(5)
    n = 500
    sessions = _sessions(n)
    flags = (sessions % 80) < 40
    y = 0.01 + 0.03 * flags + rng.normal(0, 0.02, n)

    fit = newey_west_ols(y, flags.astype(float), sessions, 19)
    assert fit["delta"] == pytest.approx(y[flags].mean() - y[~flags].mean(), abs=1e-12)
    assert fit["alpha"] == pytest.approx(y[~flags].mean(), abs=1e-12)
    assert fit["n"] == n


def test_hac_widens_the_standard_error_on_autocorrelated_residuals() -> None:
    """The reason the HAC is there at all: an overlapping-window IC series is
    strongly autocorrelated, and a plain OLS error would overstate the
    evidence."""
    rng = np.random.default_rng(9)
    n = 800
    sessions = _sessions(n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + rng.normal(0, 0.01)
    flags = (sessions % 200) < 100
    y = 0.01 + 0.002 * flags + noise

    naive = newey_west_ols(y, flags.astype(float), sessions, 0)
    corrected = newey_west_ols(y, flags.astype(float), sessions, 59)
    assert corrected["se_delta"] > naive["se_delta"]
    assert abs(corrected["t_delta"]) < abs(naive["t_delta"])
    assert corrected["delta"] == pytest.approx(naive["delta"], abs=1e-12)


def test_a_regime_that_never_switches_gives_no_slope() -> None:
    """One regime never occurring is G1's failure case; the estimator must not
    invent a number for it. The intercept is still reported."""
    rng = np.random.default_rng(2)
    n = 300
    y = rng.normal(0.01, 0.02, n)
    fit = newey_west_ols(y, np.ones(n), _sessions(n), 5)

    assert math.isnan(fit["delta"])
    assert math.isnan(fit["t_delta"])
    assert fit["alpha"] == pytest.approx(y.mean(), abs=1e-12)
    assert math.isfinite(fit["t_alpha"])


def test_non_finite_rows_are_dropped_before_fitting() -> None:
    rng = np.random.default_rng(4)
    n = 300
    sessions = _sessions(n)
    flags = (sessions % 40) < 20
    y = 0.01 + 0.02 * flags + rng.normal(0, 0.01, n)
    holed = y.copy()
    holed[[7, 88, 190]] = np.nan

    fit = newey_west_ols(holed, flags.astype(float), sessions, 5)
    keep = ~np.isnan(holed)
    reference = newey_west_ols(y[keep], flags[keep].astype(float), sessions[keep], 5)
    assert fit["n"] == int(keep.sum())
    assert fit["delta"] == pytest.approx(reference["delta"], abs=1e-12)


def test_a_duplicate_session_is_rejected() -> None:
    """A repeated session index means a grain violation upstream; folding it
    into the covariance would silently double-count a date."""
    y = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="strictly increasing"):
        newey_west_ols(y, np.array([0.0, 1.0, 0.0, 1.0]), np.array([1, 2, 2, 3]), 2)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        newey_west_ols([0.1, 0.2], [1.0], [1, 2], 1)
    with pytest.raises(ValueError, match="same length"):
        newey_west_ols([0.1, 0.2], None, [1, 2, 3], 1)


def test_too_few_observations_return_nan_rather_than_a_fit() -> None:
    fit = newey_west_ols([0.1], [1.0], [1], 0)
    assert math.isnan(fit["delta"])
    assert fit["n"] == 1
