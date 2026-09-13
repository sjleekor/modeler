from __future__ import annotations

import math

from modeler.analysis.horizon_scan_runner import (
    compute_available_direction_pass,
    compute_delay_pass,
    compute_period_sign_pass,
    compute_tradable_pass,
)


def test_tradable_pass_requires_same_sign_and_retention_threshold() -> None:
    passed = compute_tradable_pass(ic_broad=0.04, ic_tradable=0.03)
    assert passed["tradable_retention"] == 0.75
    assert passed["tradable_pass"] is True

    below_threshold = compute_tradable_pass(ic_broad=0.04, ic_tradable=0.01)
    assert below_threshold["tradable_pass"] is False

    wrong_sign = compute_tradable_pass(ic_broad=0.04, ic_tradable=-0.03)
    assert wrong_sign["tradable_pass"] is False


def test_tradable_pass_exact_threshold_boundary_passes() -> None:
    # retention exactly 0.50 must pass (>=, not >)
    result = compute_tradable_pass(ic_broad=0.04, ic_tradable=0.02)
    assert result["tradable_retention"] == 0.50
    assert result["tradable_pass"] is True


def test_tradable_pass_null_or_zero_broad_ic_fails_not_na() -> None:
    zero = compute_tradable_pass(ic_broad=0.0, ic_tradable=0.02)
    assert zero["tradable_retention"] is None
    assert zero["tradable_pass"] is False

    missing = compute_tradable_pass(ic_broad=None, ic_tradable=0.02)
    assert missing["tradable_pass"] is False

    nonfinite = compute_tradable_pass(ic_broad=float("nan"), ic_tradable=0.02)
    assert nonfinite["tradable_pass"] is False


def test_period_sign_pass_requires_strict_majority() -> None:
    # 2 of 3 valid periods aligned positive -> majority (2 > 1.5)
    result = compute_period_sign_pass([0.03, 0.02, -0.01], expected_sign="+")
    assert result["valid_subperiods"] == 3
    assert result["sign_consistent_subperiods"] == 2
    assert result["period_sign_pass"] is True

    # exactly half aligned (2 of 4) is not a strict majority
    tie = compute_period_sign_pass([0.03, 0.02, -0.01, -0.02], expected_sign="+")
    assert tie["period_sign_pass"] is False


def test_period_sign_pass_ignores_insufficient_none_periods() -> None:
    result = compute_period_sign_pass([0.03, None, None, -0.01], expected_sign="+")
    assert result["valid_subperiods"] == 2
    assert result["sign_consistent_subperiods"] == 1
    assert result["period_sign_pass"] is False  # 1 of 2 is not > half


def test_period_sign_pass_negative_expected_sign_flips_alignment() -> None:
    result = compute_period_sign_pass([-0.03, -0.02, 0.01], expected_sign="-")
    assert result["sign_consistent_subperiods"] == 2
    assert result["period_sign_pass"] is True


def test_period_sign_pass_no_valid_periods_fails() -> None:
    result = compute_period_sign_pass([None, None], expected_sign="+")
    assert result["valid_subperiods"] == 0
    assert result["period_sign_pass"] is False


def test_available_direction_pass_agrees_or_flips() -> None:
    agree = compute_available_direction_pass(ic_common_survivor=0.03, ic_available=0.02)
    assert agree["available_direction_pass"] is True

    flip = compute_available_direction_pass(ic_common_survivor=0.03, ic_available=-0.01)
    assert flip["available_direction_pass"] is False

    missing = compute_available_direction_pass(ic_common_survivor=None, ic_available=0.02)
    assert missing["available_direction_pass"] is None

    nonfinite = compute_available_direction_pass(ic_common_survivor=0.03, ic_available=float("inf"))
    assert nonfinite["available_direction_pass"] is None


def test_delay_pass_requires_direction_retention_and_significance() -> None:
    passed = compute_delay_pass(ic_native=0.04, ic_lag1=0.03, p_nw_lag1=0.01)
    assert passed["delay_retention"] == 0.75
    assert passed["delay_pass"] is True

    weak_retention = compute_delay_pass(ic_native=0.04, ic_lag1=0.01, p_nw_lag1=0.01)
    assert weak_retention["delay_pass"] is False

    not_significant = compute_delay_pass(ic_native=0.04, ic_lag1=0.03, p_nw_lag1=0.20)
    assert not_significant["delay_pass"] is False

    wrong_sign = compute_delay_pass(ic_native=0.04, ic_lag1=-0.03, p_nw_lag1=0.01)
    assert wrong_sign["delay_pass"] is False


def test_delay_pass_null_or_zero_native_ic_fails() -> None:
    zero = compute_delay_pass(ic_native=0.0, ic_lag1=0.02, p_nw_lag1=0.01)
    assert zero["delay_retention"] is None
    assert zero["delay_pass"] is False

    missing_p = compute_delay_pass(ic_native=0.04, ic_lag1=0.03, p_nw_lag1=None)
    assert missing_p["delay_pass"] is False

    nonfinite = compute_delay_pass(ic_native=float("nan"), ic_lag1=0.03, p_nw_lag1=0.01)
    assert nonfinite["delay_pass"] is False
    assert math.isnan(float("nan"))  # sanity check on the helper used above
