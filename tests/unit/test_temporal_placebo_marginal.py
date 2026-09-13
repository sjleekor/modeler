"""F-9.11 — the temporal placebo's verdict has a Monte Carlo error, and it shows.

100 replicates put the standard error of `p_temporal_nw` at 0.030 when p is
near 0.10, and the shift seed is derived from `config_hash`, so every new
preregistration layer redraws the null. On the 2026-09-09 run two cells changed
side with every other statistic bit-identical:

    ev_payout_yield|bucket|60|120   0.0891 -> 0.1287   pass -> fail
    mcap_krx_log|cum|0|120          0.1386 -> 0.0495   fail -> pass

The rule is preregistered and stays exactly as it is -- `temporal_p_max` and
`temporal_long_cell_repeats` are in the frozen config, and moving either after
seeing a result is the thing preregistration exists to stop. What changes is
that a verdict landing inside the noise is *labelled* instead of being read as
evidence.
"""

from __future__ import annotations

import math

from modeler.analysis.horizon_scan_permutation import (
    TEMPORAL_MARGINAL_SE_MULTIPLE,
    temporal_placebo_p,
)
from modeler.analysis.horizon_scan_report import _marginal_note


def _shifted(count_at_least: int, repeats: int = 100) -> list[float]:
    """`count_at_least` shifted replicates beat the real |t|, the rest do not."""
    return [2.0] * count_at_least + [0.5] * (repeats - count_at_least)


def test_the_pass_rule_is_unchanged() -> None:
    # (1 + at_least) / (repeats + 1) < p_max, exactly as §A-6b states it.
    result = temporal_placebo_p(1.0, _shifted(4), p_max=0.10)

    assert result["p_temporal_nw"] == 5 / 101
    assert result["temporal_null_pass"] is True
    assert temporal_placebo_p(1.0, _shifted(20), p_max=0.10)["temporal_null_pass"] is False


def test_the_monte_carlo_error_matches_the_binomial_one() -> None:
    result = temporal_placebo_p(1.0, _shifted(9), p_max=0.10)

    p = result["p_temporal_nw"]
    assert result["temporal_repeats"] == 100
    assert result["p_temporal_nw_se"] == math.sqrt(p * (1 - p) / 100)
    # The number the F-9.11 note quotes: 0.030 at p_max=0.10 with 100 replicates.
    assert round(result["temporal_marginal_band"] / TEMPORAL_MARGINAL_SE_MULTIPLE, 3) == 0.030


def test_the_band_is_taken_at_the_threshold_not_at_the_estimate() -> None:
    # Taken at the observed p the band shrinks exactly where p is small, and it
    # misses the measured 0.1386 -> 0.0495 flip: 0.0495 is 0.0505 from p_max
    # against a 2-SE-at-p-hat band of 0.0434. At p_max the band is 0.060.
    low = temporal_placebo_p(1.0, _shifted(4), p_max=0.10)
    high = temporal_placebo_p(1.0, _shifted(13), p_max=0.10)

    assert low["temporal_marginal_band"] == high["temporal_marginal_band"]
    assert round(low["temporal_marginal_band"], 3) == 0.060
    assert low["p_temporal_nw_se"] < low["temporal_marginal_band"] / 2


def test_both_measured_flips_are_flagged_marginal() -> None:
    # ev_payout_yield|bucket|60|120: 0.0891 then 0.1287. With 100 replicates the
    # attainable p values either side are 9/101 and 13/101.
    before = temporal_placebo_p(1.0, _shifted(8), p_max=0.10)
    after = temporal_placebo_p(1.0, _shifted(12), p_max=0.10)

    assert (before["temporal_null_pass"], after["temporal_null_pass"]) == (True, False)
    assert before["temporal_null_marginal"] is True
    assert after["temporal_null_marginal"] is True

    # mcap_krx_log|cum|0|120: 0.1386 then 0.0495 -- the same band, other way.
    fail = temporal_placebo_p(1.0, _shifted(13), p_max=0.10)
    passed = temporal_placebo_p(1.0, _shifted(4), p_max=0.10)

    assert (fail["temporal_null_pass"], passed["temporal_null_pass"]) == (False, True)
    assert fail["temporal_null_marginal"] is True
    assert passed["temporal_null_marginal"] is True


def test_a_decisive_verdict_is_not_flagged() -> None:
    # p = 1/101: no shifted replicate came close, so the flip risk is not the
    # story and the label would be noise of its own.
    decisive_pass = temporal_placebo_p(1.0, _shifted(0), p_max=0.10)
    decisive_fail = temporal_placebo_p(1.0, _shifted(80), p_max=0.10)

    assert decisive_pass["temporal_null_marginal"] is False
    assert decisive_fail["temporal_null_marginal"] is False


def test_the_flag_never_changes_the_verdict() -> None:
    # The preregistered rule is p < p_max and nothing else. A marginal cell
    # that passed still passes; one that failed still fails.
    for at_least in range(0, 25):
        result = temporal_placebo_p(1.0, _shifted(at_least), p_max=0.10)
        assert result["temporal_null_pass"] == (result["p_temporal_nw"] < 0.10)


def test_more_replicates_narrow_the_band() -> None:
    # The other remedy the F-9.11 note names. Raising the replicate count is a
    # config change and therefore a new layer, but the mechanism is asserted so
    # the trade-off is not guesswork: the same p at 400 replicates halves the
    # band and stops being marginal.
    hundred = temporal_placebo_p(1.0, _shifted(13, 100), p_max=0.10)
    four_hundred = temporal_placebo_p(1.0, _shifted(55, 400), p_max=0.10)

    assert round(hundred["p_temporal_nw"], 2) == round(four_hundred["p_temporal_nw"], 2)
    assert four_hundred["temporal_marginal_band"] < hundred["temporal_marginal_band"] / 1.9
    assert hundred["temporal_null_marginal"] is True
    assert four_hundred["temporal_null_marginal"] is False


def test_the_report_labels_a_marginal_cell_and_says_by_how_much() -> None:
    marginal = temporal_placebo_p(1.0, _shifted(12), p_max=0.10)

    note = _marginal_note(marginal)

    assert "marginal" in note
    assert f"within {TEMPORAL_MARGINAL_SE_MULTIPLE:.0f} MC SE of p_max" in note
    assert "±0.060" in note


def test_the_report_says_nothing_for_a_decisive_cell() -> None:
    assert _marginal_note(temporal_placebo_p(1.0, _shifted(0), p_max=0.10)) == ""
    # A legacy row from before F-9.11 carries neither key.
    assert _marginal_note({"p_temporal_nw": 0.05, "temporal_null_pass": True}) == ""


def test_a_marginal_row_without_the_error_still_gets_labelled() -> None:
    # Defensive: an artifact that recorded the flag but not the SE must not
    # lose the label to a formatting error.
    assert _marginal_note({"temporal_null_marginal": True}) == " (marginal)"
