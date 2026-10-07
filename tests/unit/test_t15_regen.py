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


def test_new_column_on_fresh_side_is_not_compared_and_passes():
    schema = {**SCHEMA, "new_float": pl.Float64}
    fresh = {**_row(), "topk_max_drawdown": -0.03, "topk_longest_drawdown_rebalances": 12}
    cmp = t.compare_rows(_row(), fresh, SCHEMA)
    assert cmp.passed and not cmp.failures
    assert set(cmp.new_columns) == {"topk_max_drawdown", "topk_longest_drawdown_rebalances"}
    kinds = {r["column"]: r["kind"] for r in cmp.table.iter_rows(named=True)}
    assert kinds["topk_max_drawdown"] == "new"
    assert schema  # 새 열은 schema 에 없어도 분류 실패가 아니다


def test_column_only_in_recorded_fails():
    fresh = _row()
    del fresh["log_loss"]
    cmp = t.compare_rows(_row(), fresh, SCHEMA)
    assert not cmp.passed and cmp.failures == ["log_loss: 기록에만 있음"]


def test_regen_run_list_is_15_and_excludes_adopted():
    ids = [r for _, r in t.REGEN_RUNS]
    assert len(ids) == len(set(ids)) == 15
    assert t.ADOPTED_RUN_ID not in ids
    assert ids == list(t.REGEN_RUN_IDS)
    assert sum(1 for s, _ in t.REGEN_RUNS if s == "E5") == 3
    assert all("_h20_" in r for r in ids)


def test_e5_uses_isolated_lake_root_and_others_shared(tmp_path):
    _write_run(tmp_path, stage="E5", run_id="E5_x")
    _write_run(tmp_path, stage="E4", run_id="E4_x")
    shared = SimpleNamespace(dataset_dir=lambda m: tmp_path / "shared" / m)
    e5 = SimpleNamespace(dataset_dir=lambda m: tmp_path / "_e5" / m)
    r5 = t.load_recorded("E5_x", results_root=tmp_path)
    r4 = t.load_recorded("E4_x", results_root=tmp_path)
    assert t.lake_for(r5, shared, e5) is e5
    assert t.lake_for(r4, shared, e5) is shared
    assert "_e5" in str(t.dataset_path(r5, lake=e5))


def test_e5_lake_root_is_derived_e5(tmp_path, monkeypatch):
    from modeler.models._02_updown_prob.experiments import isolated_lake

    monkeypatch.setenv("STOCK_DATA_ROOT", str(tmp_path))
    isolated_lake._shared_root.cache_clear()
    isolated_lake._isolated_root.cache_clear()
    try:
        path = t.e5_lake().dataset_dir("02_updown_prob")
        assert "/kr/derived/_e5/" in str(path)
    finally:
        isolated_lake._shared_root.cache_clear()
        isolated_lake._isolated_root.cache_clear()


def test_is_done_and_resume_skips(tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert not t.is_done(out, "E4_x")
    d = t.run_out_dir(out, "E4_x")
    d.mkdir(parents=True)
    (d / "regen_spec.json").write_text(json.dumps({"status": "running"}))
    assert not t.is_done(out, "E4_x")
    (d / "regen_spec.json").write_text(json.dumps({"status": "done", "all_folds_pass": True}))
    assert t.is_done(out, "E4_x")
    (d / "regen_spec.json").write_text("{broken")
    assert not t.is_done(out, "E4_x")

    # 끝난 run 은 건너뛰고(regenerate_run 을 부르지 않는다), 안 끝난 run 은 부른다. 오류는 기록하고 이어간다.
    monkeypatch.setattr(t, "_check_out_root", lambda _o: None)
    (d / "regen_spec.json").write_text(json.dumps({"status": "done", "all_folds_pass": False}))
    called = []

    def fake(run_id, out_root, **kw):
        called.append(run_id)
        if run_id == "boom":
            raise RuntimeError("x")
        return {"run_id": run_id, "status": "pass", "elapsed": 1.0, "bad_folds": []}

    monkeypatch.setattr(t, "regenerate_run", fake)
    rows = t.regen_many(["E4_x", "boom", "next"], out, log=lambda m: None)
    assert called == ["boom", "next"]
    assert [r["status"] for r in rows] == ["skip_done", "error", "pass"]


def test_out_root_inside_res_is_refused():
    import pytest

    with pytest.raises(SystemExit):
        t._check_out_root(t.RESULTS_ROOT / "E2" / "x")


def test_frames_bit_equal_detects_one_ulp():
    a = pl.DataFrame({"x": [1.0, None, 3.0], "s": ["a", "b", "c"]})
    assert t.frames_bit_equal(a, a.clone())[0]
    b = a.with_columns(pl.Series("x", [1.0, None, 3.0000000000000004]))
    ok, problems = t.frames_bit_equal(a, b)
    assert not ok and problems == ["x: 값 다름"]


def test_unclassified_float_fails():
    schema = {**SCHEMA, "new_float": pl.Float64}
    cmp = t.compare_rows(_row(new_float=1.0), _row(new_float=1.0), schema)
    assert cmp.unclassified == ["new_float"] and not cmp.passed
