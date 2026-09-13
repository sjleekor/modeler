from __future__ import annotations

import pytest
from modeler.analysis.horizon_scan_runner import (
    apply_global_bh,
    assert_rows_match_registry,
    compute_isolated_spikes,
)


def _row(
    hypothesis_id,
    family,
    scan_type,
    h_end,
    *,
    status="valid",
    p_nw=None,
    ic_mean=None,
    expected_sign="+",
):
    return {
        "hypothesis_id": hypothesis_id,
        "family": family,
        "scan_type": scan_type,
        "h_end": h_end,
        "status": status,
        "p_nw": p_nw,
        "ic_mean": ic_mean,
        "expected_sign": expected_sign,
    }


def test_assert_rows_match_registry_detects_missing_and_extra() -> None:
    registry = [_row("a", "f", "cum", 5), _row("b", "f", "cum", 10)]
    assert_rows_match_registry(registry, registry)  # exact match: no error

    with pytest.raises(ValueError, match="missing"):
        assert_rows_match_registry([registry[0]], registry)

    extra = registry + [_row("c", "f", "cum", 20)]
    with pytest.raises(ValueError, match="extra"):
        assert_rows_match_registry(extra, registry)


def test_assert_rows_match_registry_detects_duplicates() -> None:
    registry = [_row("a", "f", "cum", 5)]
    dup_rows = [_row("a", "f", "cum", 5), _row("a", "f", "cum", 5)]
    with pytest.raises(ValueError, match="duplicate"):
        assert_rows_match_registry(dup_rows, registry)


def test_insufficient_cells_get_p_for_bh_one_and_do_not_shrink_m() -> None:
    rows = [
        _row("a", "f", "cum", 5, p_nw=0.001, ic_mean=0.05),
        _row("b", "f", "cum", 10, p_nw=0.5, ic_mean=0.01),
        _row("c", "f", "cum", 20, status="insufficient", p_nw=None, ic_mean=None),
        _row("d", "g", "cum", 5, p_nw=0.9, ic_mean=-0.01),
    ]
    out = {r["hypothesis_id"]: r for r in apply_global_bh(rows)}
    assert out["c"]["p_for_bh"] == 1.0
    assert out["c"]["q_fdr_global"] is not None
    # m is still 4: rank-1 (p=0.001) q = 0.001*4/1, not 0.001*3/1
    assert out["a"]["q_fdr_global"] == pytest.approx(0.001 * 4 / 1)


def test_bh_threshold_is_strict_not_less_or_equal() -> None:
    # p=[0.05, 0.10] with m=2 both land on q=0.10 exactly after BH.
    rows = [
        _row("a", "f", "cum", 5, p_nw=0.05, ic_mean=0.02),
        _row("b", "f", "cum", 10, p_nw=0.10, ic_mean=0.02),
    ]
    out = {r["hypothesis_id"]: r for r in apply_global_bh(rows, q_threshold=0.10)}
    assert out["a"]["q_fdr_global"] == pytest.approx(0.10)
    assert out["a"]["bh_pass"] is False
    assert out["b"]["bh_pass"] is False


def test_expected_sign_pass_blocks_wrong_direction_discovery() -> None:
    rows = [
        _row("a", "f", "cum", 5, p_nw=0.0001, ic_mean=-0.05, expected_sign="+"),
        _row("b", "g", "cum", 5, p_nw=0.0001, ic_mean=0.05, expected_sign="+"),
    ]
    out = {r["hypothesis_id"]: r for r in apply_global_bh(rows)}
    assert out["a"]["expected_sign_pass"] is False
    assert out["a"]["primary_discovery"] is False
    assert out["b"]["expected_sign_pass"] is True
    assert out["b"]["primary_discovery"] is True


def test_two_sided_family_has_no_directional_gate() -> None:
    rows = [_row("a", "f", "cum", 5, p_nw=0.0001, ic_mean=-0.05, expected_sign=None)]
    out = apply_global_bh(rows)[0]
    assert out["expected_sign_pass"] is None
    assert out["primary_discovery"] is True  # BH pass alone is enough — no sign gate


def test_isolated_spike_when_all_neighbors_are_opposite_signed() -> None:
    rows = [
        _row("h5", "fam_b", "cum", 5, p_nw=0.5, ic_mean=-0.01),
        _row("h10", "fam_b", "cum", 10, p_nw=0.0001, ic_mean=0.06),  # candidate, isolated
        _row("h20", "fam_b", "cum", 20, p_nw=0.5, ic_mean=-0.02),
    ]
    spikes = compute_isolated_spikes(rows)
    assert spikes["h10"] is True
    assert spikes["h5"] is False  # not a positive candidate itself
    assert spikes["h20"] is False


def test_isolated_spike_false_when_a_neighbor_is_aligned() -> None:
    rows = [
        _row("h5", "fam_a", "cum", 5, p_nw=0.001, ic_mean=0.05),
        _row("h10", "fam_a", "cum", 10, p_nw=0.002, ic_mean=0.04),
        _row("h20", "fam_a", "cum", 20, p_nw=0.5, ic_mean=-0.01),
    ]
    spikes = compute_isolated_spikes(rows)
    assert spikes["h5"] is False  # its only neighbor (h10) is aligned positive
    assert spikes["h10"] is False  # has an aligned neighbor on each side... one is enough


def test_isolated_spike_grid_end_uses_single_neighbor_and_null_counts_as_bad() -> None:
    rows = [
        _row("h5", "fam_c", "cum", 5, p_nw=0.001, ic_mean=0.03),
        _row("h10", "fam_c", "cum", 10, status="insufficient", p_nw=None, ic_mean=None),
    ]
    spikes = compute_isolated_spikes(rows)
    assert spikes["h5"] is True  # its only neighbor is NULL -> counts as bad
    assert spikes["h10"] is False


def test_isolated_spike_and_bh_pass_together_gate_primary_discovery() -> None:
    rows = [
        _row("h5", "fam_b", "cum", 5, p_nw=0.5, ic_mean=-0.01),
        _row("h10", "fam_b", "cum", 10, p_nw=0.0001, ic_mean=0.06),
        _row("h20", "fam_b", "cum", 20, p_nw=0.5, ic_mean=-0.02),
    ]
    out = {r["hypothesis_id"]: r for r in apply_global_bh(rows)}
    assert out["h10"]["bh_pass"] is True
    assert out["h10"]["isolated_spike"] is True
    # isolated spike vetoes an otherwise BH-passing discovery
    assert out["h10"]["primary_discovery"] is False
