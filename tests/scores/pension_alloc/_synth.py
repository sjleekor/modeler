"""합성 자료 도우미. 실제 가격·수익률 파일은 어디서도 읽지 않는다."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np


def bdays(n: int, start: date = date(2000, 1, 3)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def month_ends(days: list[date]) -> list[int]:
    return [
        i
        for i in range(len(days) - 1)
        if (days[i].year, days[i].month) != (days[i + 1].year, days[i + 1].month)
    ]


def two_regime(n: int, block: int = 150, seed: int = 0):
    """블록마다 강세/약세가 번갈아 나오는 일수익. 반환: (수익, 심은 상태(1=약세))."""
    rng = np.random.default_rng(seed)
    st = (np.arange(n) // block) % 2
    mu = np.where(st == 1, -0.0015, 0.0010)
    sd = np.where(st == 1, 0.020, 0.006)
    return mu + sd * rng.standard_normal(n), st


def slot_returns(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = np.array([4e-4, 4e-4, 1e-4, 1.5e-4, 2e-4, 0.5e-4])
    sd = np.array([0.011, 0.012, 0.002, 0.004, 0.009, 0.0002])
    return mu + sd * rng.standard_normal((n, 6))
