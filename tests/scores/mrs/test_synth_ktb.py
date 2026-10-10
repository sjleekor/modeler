"""합성 국고채 (사전등록 §0-5) — 합성 데이터만."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from modeler.scores.mrs import synth_ktb as K


def _price(y_obs: float, coupon: float, years: int) -> float:
    q = 1.0 + y_obs / 4
    n = 4 * years
    return sum((coupon / 4) * q ** (-k) for k in range(1, n + 1)) + q ** (-n)


def test_textbook_ten_year_par_5pct():
    d, c = K.par_duration_convexity(0.05, 10)
    assert 7.7 < d[0] < 7.9  # 교과서: 10년 par 5% 수정 듀레이션 약 7.8
    assert c[0] > 0


def test_duration_convexity_vs_finite_difference():
    h = 1e-5
    for y, m in [(0.03, 3), (0.05, 10), (0.08, 10), (0.015, 3)]:
        p0 = _price(y, y, m)
        assert abs(p0 - 1.0) < 1e-12  # par
        pu, pd = _price(y + h, y, m), _price(y - h, y, m)
        d_fd = -(pu - pd) / (2 * h) / p0
        c_fd = (pu - 2 * p0 + pd) / h**2 / p0
        d, c = K.par_duration_convexity(y, m)
        assert abs(d[0] - d_fd) < 1e-6
        assert abs(c[0] - c_fd) < 1e-3 * c_fd


def test_duration_nan_passthrough():
    d, c = K.par_duration_convexity(np.array([0.04, np.nan]), 3)
    assert np.isfinite(d[0]) and np.isnan(d[1]) and np.isnan(c[1])


def _grid(n: int, start: date = date(2000, 1, 3)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_flat_yield_return_equals_carry():
    g = _grid(10)
    rows = pl.DataFrame({"date": g, "value": [5.0] * len(g)})
    r = K.synth_returns(g, rows, 3)
    assert np.isnan(r[-1])
    for i in range(len(g) - 1):
        days = (g[i + 1] - g[i]).days
        assert abs(r[i] - 0.05 * days / 365.0) < 1e-15


def test_rate_change_uses_prior_day_duration_and_convexity():
    g = _grid(3)
    rows = pl.DataFrame({"date": g, "value": [4.0, 4.5, 4.5]})
    r = K.synth_returns(g, rows, 10)
    d, c = K.par_duration_convexity(0.04, 10)
    dy = 0.005
    expect = 0.04 * (g[1] - g[0]).days / 365.0 - d[0] * dy + 0.5 * c[0] * dy**2
    assert abs(r[0] - expect) < 1e-14
    assert r[0] < 0.04 * 1 / 365.0  # 금리 상승이라 손실이 이표를 넘는다


def test_stale_yield_is_null():
    g = _grid(30)
    # 관측이 처음 5일뿐 -> 마지막 관측 뒤 14일이 지나면 y가 null
    rows = pl.DataFrame({"date": g[:5], "value": [3.0] * 5})
    y = K.yields_on_grid(g, rows)
    for i, d in enumerate(g):
        age = (d - g[4]).days
        if i >= 5 and age > 14:
            assert np.isnan(y[i])
        else:
            assert np.isfinite(y[i])
    r = K.synth_returns(g, rows, 3)
    assert np.isfinite(r[3]) and np.isnan(r[28])
    first_bad = next(i for i in range(5, 30) if np.isnan(y[i]))
    assert np.isnan(r[first_bad - 1])  # 도착 격자의 y가 null이면 그 구간도 null
    assert np.isnan(r[first_bad])


def test_before_first_observation_is_null():
    g = _grid(5)
    rows = pl.DataFrame({"date": [g[2], g[3], g[4]], "value": [3.0, 3.0, 3.0]})
    y = K.yields_on_grid(g, rows)
    assert np.isnan(y[0]) and np.isnan(y[1]) and np.isfinite(y[2])


def test_mix_ps1_and_ps2_start():
    g = [
        date(2000, 12, 27),
        date(2000, 12, 28),
        date(2001, 1, 2),
        date(2001, 1, 3),
        date(2001, 1, 4),
    ]
    r_bh = np.array([0.01, 0.02, -0.01, 0.0, 0.03])
    r_k = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    ps1 = K.mix_synth("mix_synth_ps1", g, r_bh, r_k)
    np.testing.assert_allclose(ps1, 0.3 * r_bh + 0.7 * r_k)
    ps2 = K.mix_synth("mix_synth_ps2", g, r_bh, r_k)
    assert np.all(np.isnan(ps2[:2]))  # 2001-01-01 이전은 비어 있다
    np.testing.assert_allclose(ps2[2:], 0.6 * r_bh[2:] + 0.4 * r_k[2:])
