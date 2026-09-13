"""k=100 cost check — report rendering and the candidate-vs-baseline verdict."""

from __future__ import annotations

from modeler.models._01_20_access_return_rank.experiments.run_topk_cost_check import (
    render_markdown,
)


def _record(
    name: str,
    horizon: int,
    *,
    decile_net: float,
    topk_net: float,
    gate_net: float | None = None,
) -> dict:
    return {
        "name": name,
        "horizon": horizon,
        "best_params": {},
        "decile": {
            "horizon": horizon,
            "n_rebalances": 100,
            "grid_top_decile_spread": decile_net + 0.004,
            "turnover": 0.68,
            "cost_bps_roundtrip": 60.0,
            "cost_adjusted_spread": decile_net,
        },
        "topk": {
            "horizon": horizon,
            "k": 100,
            "n_rebalances": 100,
            "mean_names_held": 100.0,
            "mean_names_scored": 98.0,
            "grid_topk_mean_return": topk_net + 0.006,
            "turnover": 0.95,
            "cost_bps_roundtrip": 60.0,
            "cost_adjusted_return": topk_net,
        },
        "gate_decile_cost_adjusted_spread": gate_net,
        "fit_seconds": 1.0,
    }


def test_verdict_counts_only_horizons_where_the_candidate_wins_at_k() -> None:
    records = [
        _record("baseline", 5, decile_net=0.001, topk_net=0.002),
        _record("candidate", 5, decile_net=0.002, topk_net=0.003),  # improves
        _record("baseline", 20, decile_net=0.001, topk_net=0.004),
        _record("candidate", 20, decile_net=0.002, topk_net=0.001),  # decile up, k down
    ]

    md = render_markdown(records, k=100, cost_bps=60.0)

    assert "2개 horizon 중 1개에서 k=100 비용차감 후 candidate가 baseline보다 높다." in md
    assert "| 5d | 0.0010 | 0.0010 | 예 |" in md
    assert "| 20d | 0.0010 | -0.0030 | 아니오 |" in md


def test_reproduction_row_shows_the_drift_from_the_gate_value() -> None:
    records = [
        _record("baseline", 5, decile_net=0.0012, topk_net=0.002, gate_net=0.0010),
        _record("candidate", 5, decile_net=0.0020, topk_net=0.003, gate_net=0.0020),
    ]

    md = render_markdown(records, k=100, cost_bps=60.0)

    assert "| baseline | 5d | 0.0012 | 0.0010 | 0.0002 |" in md
    assert "| candidate | 5d | 0.0020 | 0.0020 | 0.0000 |" in md


def test_missing_gate_value_renders_as_a_dash_not_a_crash() -> None:
    records = [_record("baseline", 5, decile_net=0.001, topk_net=0.002, gate_net=None)]

    md = render_markdown(records, k=100, cost_bps=60.0)

    assert "| baseline | 5d | 0.0010 | - | - |" in md
