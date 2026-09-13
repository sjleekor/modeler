from pathlib import Path

import polars as pl

import modeler.models._01_20_access_return_rank.experiments.run_phase_b_acceptance_gate as gate


def test_select_screen_pass_features_uses_successful_cells_only(tmp_path: Path) -> None:
    (tmp_path / "_SUCCESS.json").write_text(
        '{"status":"success","run_id":"ab","content_hash":"x","config_hash":"cfg"}'
    )
    pl.DataFrame(
        {
            "family": ["size", "filing", "discarded"],
            "feature": ["mcap_krx_log", "ev_filing_burst_60d", "fin_sue"],
            "screen_pass": [True, True, False],
        }
    ).write_parquet(tmp_path / "combined_ab_primary_hypotheses.parquet")

    families, features, success = gate.select_screen_pass_features(tmp_path)

    assert families == ["filing", "size"]
    assert features == ["ev_filing_burst_60d", "mcap_krx_log"]
    assert success["run_id"] == "ab"


def test_group_features_by_mart_is_stable() -> None:
    grouped = gate.group_features_by_mart(
        ["mcap_krx_log", "fin_value_z", "fin_log_mcap", "own_major_stake"]
    )

    assert grouped == {
        "feat_fin_scan_daily": ["fin_log_mcap", "fin_value_z"],
        "feat_market_cap": ["mcap_krx_log"],
        "feat_periodic_extras": ["own_major_stake"],
    }


def test_summarize_validation_keeps_final_acceptance_deferred() -> None:
    records = []
    for horizon in gate.TARGET_HORIZONS:
        records.extend(
            [
                {
                    "name": "baseline",
                    "horizon": horizon,
                    "mean_rank_ic": 0.10,
                    "economic": {"cost_adjusted_spread": 0.01},
                },
                {
                    "name": "phase_b_candidate",
                    "horizon": horizon,
                    "mean_rank_ic": 0.11,
                    "economic": {"cost_adjusted_spread": 0.02},
                },
            ]
        )

    summary = gate.summarize_validation(records)

    assert summary["status"] == "improved_all_horizons"
    assert summary["final_acceptance"] == "deferred_until_new_mature_h60_holdout"
    assert [row["horizon"] for row in summary["deltas"]] == list(gate.TARGET_HORIZONS)
