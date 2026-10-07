import json

import numpy as np
import pandas as pd
import pytest

from modeler.models._02_updown_prob.experiments import t15_report as t


def _make(out, res, seed_shift=0, pred_col=lambda r: "p_raw"):
    dates = pd.bdate_range("2020-04-23", periods=65 * 20, freq="B")[::20][:65]
    fold = np.repeat(np.arange(1, 6), 13)
    for i, r in enumerate(t.RUN_IDS):
        rng = np.random.default_rng(i)
        g = rng.normal(0.004 + 0.0005 * i, 0.02, 65)
        df = pd.DataFrame(
            {"fold_id": fold, "pred_col": pred_col(r), "rebalance_date": dates, "horizon": 20, "k": 100,
             "gross_return": g, "turnover": 0.5, "cost_bps_roundtrip": 60.0, "net_return": g - 0.003}
        )
        d = out / "runs" / r
        d.mkdir(parents=True)
        df.to_parquet(d / "rebalance_returns.parquet")
        sd = res / "E9" / r
        sd.mkdir(parents=True)
        (sd / "summary.json").write_text(json.dumps({"primary_pred_col": pred_col(r)}))


@pytest.fixture
def env(tmp_path):
    out, res = tmp_path / "out", tmp_path / "res"
    _make(out, res)
    return out, res, tmp_path / "copy"


def test_end_to_end(env):
    out, res, copy = env
    text = t.run_report(out, res, copy_to_res=True, res_copy_dir=copy)
    for name in ("returns_matrix.parquet", "dsr_pbo.json", "cscv_combinations.parquet"):
        assert (out / "result" / name).is_file()
    assert (copy / "dsr_pbo.json").is_file() and (copy / "returns_matrix.parquet").is_file()
    assert not (copy / "cscv_combinations.parquet").exists()
    j = json.loads((out / "result" / "dsr_pbo.json").read_text())
    assert j["n_runs"] == 16 and j["n_obs"] == 65 and j["pbo"]["n_combinations"] == 70
    assert j["pbo"]["block_sizes"] == [9] + [8] * 7
    m = pd.read_parquet(out / "result" / "returns_matrix.parquet")
    assert m.shape == (65, 16)
    assert len(pd.read_parquet(out / "result" / "cscv_combinations.parquet")) == 70
    assert "DSR" in text and "PBO" in text


def test_no_copy_by_default(env):
    out, res, copy = env
    t.run_report(out, res, res_copy_dir=copy)
    assert not copy.exists()


def test_missing_run_stops(env):
    out, res, _ = env
    (out / "runs" / t.RUN_IDS[3] / "rebalance_returns.parquet").unlink()
    with pytest.raises(t.T15Error, match="없는 run"):
        t.run_report(out, res)
    assert not (out / "result").exists()


def test_different_dates_stop(env):
    out, res, _ = env
    p = out / "runs" / t.RUN_IDS[5] / "rebalance_returns.parquet"
    df = pd.read_parquet(p)
    df["rebalance_date"] = df["rebalance_date"] + pd.Timedelta(days=1)
    df.to_parquet(p)
    with pytest.raises(t.T15Error, match="rebalance_date"):
        t.run_report(out, res)


def test_pred_col_mismatch_stops(env):
    out, res, _ = env
    s = res / "E9" / t.RUN_IDS[0] / "summary.json"
    s.write_text(json.dumps({"primary_pred_col": "p_cal"}))
    with pytest.raises(t.T15Error, match="primary_pred_col"):
        t.run_report(out, res)


def test_failed_regen_spec(env):
    out, res, _ = env
    (out / "runs" / t.RUN_IDS[2] / "regen_spec.json").write_text(json.dumps({"check": {"verdict": "FAIL"}}))
    with pytest.raises(t.T15Error, match="FAIL"):
        t.run_report(out, res)
    text = t.run_report(out, res, allow_failed=True)
    assert "경고" in text


def test_nan_in_returns_stops(env):
    out, res, _ = env
    p = out / "runs" / t.RUN_IDS[1] / "rebalance_returns.parquet"
    df = pd.read_parquet(p)
    df.loc[3, "net_return"] = np.nan
    df.to_parquet(p)
    with pytest.raises(ValueError, match="NaN"):
        t.run_report(out, res)
