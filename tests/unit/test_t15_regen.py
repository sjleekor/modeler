"""t15_regen — config 복원, 데이터셋 경로, 대조 분류. 학습은 하지 않는다."""

from __future__ import annotations

import json
from types import SimpleNamespace

import polars as pl

from modeler.models._02_updown_prob.experiments import t15_regen as t


def _write_run(root, stage="E4", run_id="E4_x"):
    d = root / stage / run_id
    d.mkdir(parents=True)
    params = {"max_iter": 200, "learning_rate": 0.03, "max_leaf_nodes": 31, "l2_regularization": 1.0}
    (d / "run_spec.json").write_text(
        json.dumps(
            {
                "run": {"stage": stage, "feature_set": "FS1h", "flow_variant": "lag1",
                        "preprocess_profile": "rank", "seed": 1},
                "train_config": {"model": "hgb_clf", "target": "y_up", "horizon": 20,
                                 "grid": [params, {**params, "l2_regularization": 0.0}], "seed": 1,
                                 "calibrate": "none", "cal_target": None, "cal_frac": 0.2,
                                 "monotonic": False, "date_col": "trade_date",
                                 "id_cols": ["ticker", "market"]},
                "model_code_hash": "abc",
            }
        )
    )
    (d / "summary.json").write_text(
        json.dumps({"best_params": params, "primary_pred_col": "p_raw", "k": 100,
                    "cost_bps_roundtrip": 60.0, "tau": 0.6, "dataset_dir": "/gone"})
    )
    return params


def test_restore_config_and_dataset_path(tmp_path):
    params = _write_run(tmp_path)
    rec = t.load_recorded("E4_x", results_root=tmp_path)
    assert rec.stage == "E4"
    cfg = t.restore_train_config(rec)
    assert cfg.grid == (params,) and cfg.seed == 1 and cfg.target_column == "y_up_20d"
    lake = SimpleNamespace(dataset_dir=lambda model_id: tmp_path / "ds" / model_id)
    path = t.dataset_path(rec, lake=lake)
    assert path.parent == tmp_path / "ds" / "02_updown_prob"
    assert path.name == "FS1h_h20_lag1_rank"
    assert "gone" not in str(path)


def _row(**over):
    base = {"fold_id": 1, "pred_col": "p_raw", "n_obs": 5, "log_loss": 0.66, "base_rate": 0.37,
            "tau_turnover": float("nan")}
    base.update(over)
    return base


SCHEMA = {"fold_id": pl.Int64, "pred_col": pl.String, "n_obs": pl.Int64, "log_loss": pl.Float64,
          "base_rate": pl.Float64, "tau_turnover": pl.Float64}


def test_compare_pass():
    cmp = t.compare_rows(_row(), _row(base_rate=0.37 * (1 + 1e-14)), SCHEMA)
    assert cmp.passed and not cmp.unclassified


def test_compare_integer_and_bitwise_and_close_fail():
    cmp = t.compare_rows(_row(), _row(n_obs=6, log_loss=0.66 + 1e-16, base_rate=0.37 * (1 + 1e-9)), SCHEMA)
    assert set(cmp.failures) == {"n_obs", "log_loss", "base_rate"}


def test_unclassified_float_fails():
    schema = {**SCHEMA, "new_float": pl.Float64}
    cmp = t.compare_rows(_row(new_float=1.0), _row(new_float=1.0), schema)
    assert cmp.unclassified == ["new_float"] and not cmp.passed
