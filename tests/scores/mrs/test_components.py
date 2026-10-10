"""MRS 성분 산식 — 손으로 계산한 작은 예와 ``lookback`` 선언 검증 (합성 데이터만)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from modeler.scores.mrs import components as C
from modeler.scores.mrs import config as cfg

NAN = np.nan


def close(a: np.ndarray, b: list[float]) -> None:
    np.testing.assert_allclose(a, np.array(b, dtype=float), rtol=1e-12, atol=1e-12, equal_nan=True)


def test_trend_lag_hand_example():
    p = np.array([100.0, 110.0, 121.0, 133.1])
    x = C.trend_lag(p, lag=2)
    close(x, [NAN, NAN, math.log(1.21), math.log(133.1 / 110.0)])


def test_trend_lag_default_is_config_252():
    p = np.arange(1.0, 300.0)
    x = C.trend_lag(p)
    assert np.isnan(x[:252]).all() and not np.isnan(x[252])
    assert x[252] == pytest.approx(math.log(253.0 / 1.0))


def test_trend_ma_hand_example():
    p = np.array([1.0, 2.0, 3.0, 6.0])
    close(C.trend_ma(p, window=3), [NAN, NAN, 3.0 / 2.0 - 1.0, 6.0 / (11.0 / 3.0) - 1.0])


def test_rvol_hand_example():
    rets = [0.1, 0.2, -0.1, 0.3]
    p = np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
    # σ_2(s) = |r_s - r_{s-1}| / sqrt(2) (ddof=1) -> idx2: .1/√2, idx3: .3/√2, idx4: .4/√2
    s2, s3, s4 = 0.1 / math.sqrt(2), 0.3 / math.sqrt(2), 0.4 / math.sqrt(2)
    x = C.rvol_neg(p, window=2, median_window=2)
    close(x, [NAN, NAN, NAN, -math.log(s3 / ((s2 + s3) / 2)), -math.log(s4 / ((s3 + s4) / 2))])
    # 첫 수익은 s=1이므로 σ_2 첫 값은 s=2 — 연율화하지 않는다 (MI10)
    sig = C.sigma_20(p, window=2)
    close(sig, [NAN, NAN, s2, s3, s4])


def test_sigma20_matches_numpy_std_ddof1():
    rng = np.random.default_rng(0)
    p = np.exp(np.cumsum(rng.normal(0, 0.01, 60)))
    r = np.diff(np.log(p))
    sig = C.sigma_20(p)
    assert sig[20] == pytest.approx(np.std(r[:20], ddof=1))
    assert np.isnan(sig[:20]).all()


def test_vix_level_hand_example():
    v = np.array([10.0, 12.0, 11.0, 15.0])
    close(C.level_vs_median_neg(v, window=3), [NAN, NAN, 0.0, -3.0])


def test_liq_hand_example():
    tvk = np.array([1.0, 2.0, 3.0, 4.0])
    tvd = np.array([1.0, 2.0, 3.0, 4.0])  # TV = 2,4,6,8
    x = C.liq_20(tvk, tvd, mean_window=2, median_window=3)
    close(x, [NAN, NAN, math.log(5.0 / 4.0), math.log(7.0 / 6.0)])


def test_liq_needs_both_markets():
    tvk = np.array([1.0, 2.0, 3.0, 4.0])
    tvd = np.array([1.0, NAN, 3.0, 4.0])
    x = C.liq_20(tvk, tvd, mean_window=2, median_window=2)
    # idx1: TV NaN -> idx1·idx2의 창이 모두 NaN
    assert np.isnan(x[:3]).all() and not np.isnan(x[3])


def test_foreign_hand_example():
    fk = np.array([1.0, 1.0, 2.0, 2.0])
    fd = np.array([1.0, 1.0, 0.0, 0.0])  # 합계 2,2,2,2
    tvk = np.array([1.0, 2.0, 3.0, 4.0])
    tvd = np.array([1.0, 2.0, 3.0, 4.0])  # TV 2,4,6,8
    x = C.foreign_20(fk, fd, tvk, tvd, window=2)
    close(x, [NAN, 4.0 / 6.0, 4.0 / 10.0, 4.0 / 14.0])


def test_foreign_zero_denominator_is_null():
    z = np.zeros(4)
    assert np.isnan(C.foreign_20(z + 1, z, z, z, window=2)).all()


def test_krw_hand_example():
    fx = np.array([1000.0, 1010.0, 990.0, 1000.0])
    close(C.krw_20(fx, lag=2), [NAN, NAN, -math.log(990.0 / 1000.0), -math.log(1000.0 / 1010.0)])


def test_kr_term_difference():
    close(C.term_spread(np.array([3.0, 3.5]), np.array([2.0, 2.1])), [1.0, 1.4])
    assert np.isnan(C.term_spread(np.array([3.0, NAN]), np.array([2.0, 2.0]))[1])


def test_credit_hand_examples():
    baa = np.array([2.0, 2.5, 3.0, 2.0])
    close(C.credit_chg_neg(baa, lag=2), [NAN, NAN, -1.0, 0.5])
    close(C.level_vs_median_neg(baa, window=3), [NAN, NAN, -(3.0 - 2.5), -(2.0 - 2.5)])


def test_us_term_identity():
    a = np.array([0.5, NAN, -0.2])
    out = C.identity(a)
    close(out, [0.5, NAN, -0.2])
    assert out is not a


def test_strict_windows_nan_propagates():
    p = np.array([1.0, 2.0, NAN, 4.0, 5.0, 6.0])
    # 창 3 이동평균: NaN을 포함한 창(idx 2~4)은 모두 NaN (MI05)
    assert np.isnan(C.rolling_mean(p, 3)[2:5]).all()
    assert not np.isnan(C.rolling_mean(p, 3)[5])
    # 앞선 값이 NaN이면 lag 값도 NaN
    assert np.isnan(C.trend_lag(p, lag=2)[4])


def test_nonpositive_is_null_not_inf():
    p = np.array([1.0, 0.0, 2.0, 3.0])
    out = C.trend_lag(p, lag=1)
    assert np.isnan(out[1]) and np.isnan(out[2]) and not np.isinf(out).any()


def _random_inputs(rng: np.random.Generator, specs, n: int) -> dict[str, np.ndarray]:
    keys = {k for sp in specs.values() for k in sp.inputs}
    return {k: np.exp(np.cumsum(rng.normal(0, 0.01, n))) * 100.0 for k in keys}


@pytest.mark.parametrize(
    "specs",
    [
        C.kr_components(),
        C.us_components(),
        C.kr_components(
            trend_lag_n=5,
            ma_window=4,
            rvol_window=3,
            median_window=6,
            liq_window=3,
            flow_window=3,
            fx_lag=4,
        ),
    ],
    ids=["kr_default", "us_default", "kr_small"],
)
def test_lookback_declaration_is_sufficient(specs):
    """x[s]를 s-lookback부터 자른 배열로 계산해도 전체 배열로 계산한 값과 비트까지 같다."""
    rng = np.random.default_rng(1)
    n = 700
    arrays = _random_inputs(rng, specs, n)
    # 일부 칸을 NaN으로 — 창 안 NaN 전파도 같아야 한다
    for a in arrays.values():
        a[rng.choice(n, 25, replace=False)] = NAN
    for name, sp in specs.items():
        full = sp.fn(arrays)
        for s in list(range(n - 40, n)) + [sp.lookback, sp.lookback + 3]:
            lo = max(0, s - sp.lookback)
            part = sp.fn({k: a[lo : s + 1] for k, a in arrays.items()})
            a, b = part[-1], full[s]
            assert (np.isnan(a) and np.isnan(b)) or a == b, (name, s)


def test_component_tables_match_config():
    assert set(C.kr_components()) == set(cfg.KR_COMPONENTS)
    assert set(C.us_components()) == set(cfg.US_COMPONENTS)
    assert C.kr_components()["kr_trend_252"].lookback == cfg.TREND_LAG
    assert C.kr_components()["kr_rvol_20"].lookback == cfg.RVOL_WINDOW + cfg.MEDIAN_WINDOW - 1
