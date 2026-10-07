import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from modeler.etl.multiple_testing import (
    annualize_sharpe,
    deflated_sharpe,
    expected_max_sharpe,
    pbo_cscv,
    probabilistic_sharpe,
    sharpe,
    strategy_is_best_summary,
)


def test_sharpe_and_annualize():
    r = np.array([0.01, 0.03, -0.01, 0.02])
    assert sharpe(r) == pytest.approx(r.mean() / r.std(ddof=1))
    assert annualize_sharpe(1.0, 20) == pytest.approx(math.sqrt(12.5))
    assert math.isnan(sharpe([0.1, 0.1, 0.1]))
    assert math.isnan(sharpe([0.1]))


def test_nan_fails_loudly():
    with pytest.raises(ValueError):
        sharpe([0.1, np.nan, 0.2])
    with pytest.raises(ValueError):
        pbo_cscv(np.full((40, 3), np.nan))


def test_expected_max_sharpe_hand_values():
    assert expected_max_sharpe(0.5, 1) == 0.0
    g = 0.5772156649015329
    n = 16
    want = math.sqrt(0.01) * (
        (1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * math.e))
    )
    assert expected_max_sharpe(0.01, n) == pytest.approx(want)
    assert expected_max_sharpe(0.01, 100) > expected_max_sharpe(0.01, 16) > 0


def test_psr_normal_denominator():
    sr = 0.3
    # skew 0, kurt 3 -> sqrt(1 + sr^2/2)
    got = probabilistic_sharpe(sr, 0.0, 65, 0.0, 3.0)
    want = stats.norm.cdf(sr * math.sqrt(64) / math.sqrt(1 + sr**2 / 2))
    assert got == pytest.approx(want)
    with pytest.raises(ValueError):
        probabilistic_sharpe(5.0, 0.0, 65, 3.0, 3.0)  # 근호 안 <= 0


def test_literature_example_bailey_ldp_2014():
    # 논문 예: 연 SR 2.5, 연 분산 0.5, N=100, T=1250 일간, skew -3, kurt 10 -> DSR 약 0.90
    sr = 2.5 / math.sqrt(250)
    var = 0.5 / 250
    sr0 = expected_max_sharpe(var, 100)
    assert sr0 == pytest.approx(0.1132, abs=5e-4)
    assert probabilistic_sharpe(sr, sr0, 1250, -3.0, 10.0) == pytest.approx(0.9004, abs=2e-3)


def test_deflated_sharpe_n1_equals_psr0():
    rng = np.random.default_rng(0)
    r = rng.normal(0.004, 0.02, 65)
    d = deflated_sharpe(r, [sharpe(r)], n_trials=1)
    assert d.sr0 == 0.0
    psr0 = probabilistic_sharpe(d.sr_hat, 0.0, 65, d.skew, d.kurtosis)
    assert d.dsr == pytest.approx(psr0)
    assert d.p_value == pytest.approx(1 - d.dsr)
    assert d.sr_hat_annual == pytest.approx(d.sr_hat * math.sqrt(12.5))


def test_deflated_sharpe_more_trials_lowers_dsr():
    rng = np.random.default_rng(1)
    r = rng.normal(0.004, 0.02, 65)
    trials = rng.normal(0.1, 0.1, 16)
    trials[0] = sharpe(r)
    d16 = deflated_sharpe(r, trials)
    assert d16.n_trials == 16
    assert d16.sr_var == pytest.approx(np.var(trials, ddof=1))
    d4 = deflated_sharpe(r, trials, n_trials=4)
    assert d16.sr0 > d4.sr0
    assert d16.dsr < d4.dsr
    assert d16.kurtosis == pytest.approx(stats.kurtosis(r, fisher=False))
    with pytest.raises(ValueError):
        deflated_sharpe([0.01, 0.01, 0.01, 0.01], trials)


def test_pbo_split_sizes_and_combinations():
    x = np.random.default_rng(2).normal(size=(65, 16))
    res = pbo_cscv(x, n_splits=8)
    assert res.block_sizes == [9] + [8] * 7
    assert res.n_combinations == 70 == len(res.table) == len(res.lambdas)
    assert res.table["oos_rank"].between(1, 16).all()


def test_pbo_pure_noise_near_half():
    x = np.random.default_rng(3).normal(size=(65, 16))
    assert 0.25 <= pbo_cscv(x).pbo <= 0.75
    pbos = [pbo_cscv(np.random.default_rng(s).normal(size=(65, 16))).pbo for s in range(20)]
    assert 0.35 <= float(np.mean(pbos)) <= 0.65


def test_pbo_dominant_strategy_near_zero():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 0.02, size=(65, 16))
    x[:, 5] += 0.05  # 모든 블록에서 지배
    df = pd.DataFrame(x, columns=[f"r{i}" for i in range(16)])
    res = pbo_cscv(df)
    assert res.pbo == pytest.approx(0.0)
    s = strategy_is_best_summary(res, "r5")
    assert s["n_is_best"] == 70
    assert s["oos_rank_from_top_median"] == 1.0
    s0 = strategy_is_best_summary(res, "r0")
    assert s0["n_is_best"] == 0 and s0["oos_rank_median"] is None
    assert len(s0["oos_rank_from_top_range_all"]) == 2


def test_pbo_degenerate():
    with pytest.raises(ValueError):
        pbo_cscv(np.random.default_rng(0).normal(size=(65, 1)))
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((65, 3)), n_splits=7)
    with pytest.raises(ValueError):
        pbo_cscv(np.zeros((10, 3)), n_splits=8)
    # 모든 열이 같다: 동점이라 omega 0.5, lambda 0, PBO 1
    col = np.random.default_rng(5).normal(size=(65, 1))
    res = pbo_cscv(np.hstack([col] * 4))
    assert res.pbo == 1.0
    assert np.allclose(res.table["omega"], 0.5)
    assert (res.table["best_col"] == 0).all()
    # 표준편차 0 열(상수)도 멈추지 않는다
    assert pbo_cscv(np.zeros((65, 3))).pbo == 1.0
