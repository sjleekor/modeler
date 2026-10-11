"""ETF E3 — 총수익·보수 조정 결과 값 (사양 §3·§5·§6). **판정 실행에서만 부른다.**

동결 전에는 이 모듈의 값 함수(``pair_rtr``·``own_rtr``·``gy_values``)를 판정 구간에 돌리지 않는다.
(동결 전에 쓰는 개수·존재 확인은 ``etf_e3_sample`` 에 있다.)

- ETF 일 총수익 ``r_t = (NAV_t + D_t) / NAV_{t−1} − 1``. ``t−1`` 은 KRX 거래일 달력의 직전 거래일.
- 연 y의 창 = y−1년 마지막 거래일 다음 날 ~ y년 마지막 거래일(= 달력 연 y의 거래일).
- 쌍 (i, j)의 연 총수익은 두 ETF 모두 r_t 가 있는 날(공통일)만 곱한다.
- 보수 조정: ``G_i = RTR_i^(ij)(Y) + F_i(Y)/100`` , ``H_i = RTR_i^(ij)(Y+1) + A_i(Y+1)/100``.
  보수 F·A 는 연 % 이므로 100으로 나눈다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modeler.scores.quality import etf_e3_fees as fees
from modeler.scores.quality.etf_e3_sample import Ctx, GroupYear, pair_window_ok


def _factor(ctx: Ctx, s: int, e: int) -> np.ndarray:
    """(e−s+1, N): 1 + r_t. r_t 가 정의되지 않은 날은 NaN."""
    d = ctx.dense
    nav = d.nav[s : e + 1]
    prev = d.nav[s - 1 : e]
    f = (nav + ctx.D[s : e + 1]) / prev
    return np.where(ctx.rfin[s : e + 1], f, np.nan)


def pair_rtr(ctx: Ctx, i: int, j: int, y: int) -> tuple[float, float]:
    """연 y 창에서 두 ETF 공통일만 곱한 총수익 (RTR_i^(ij), RTR_j^(ij))."""
    s, e = ctx.dense.year_range(y)
    f = _factor(ctx, s, e)
    common = np.isfinite(f[:, i]) & np.isfinite(f[:, j])
    return float(np.prod(f[common, i]) - 1.0), float(np.prod(f[common, j]) - 1.0)


def own_rtr(ctx: Ctx, i: int, y: int) -> tuple[float, bool]:
    """연 y 창 자기 r_t 가 있는 날 전부의 총수익, 그리고 분배락일이 r_t 없는 날에 있어 분배가 빠졌는지(own_lost)."""
    s, e = ctx.dense.year_range(y)
    f = _factor(ctx, s, e)[:, i]
    ok = np.isfinite(f)
    lost = bool((ctx.exmask[s : e + 1, i] & ~ok).any())
    return float(np.prod(f[ok]) - 1.0), lost


@dataclass
class GyValues:
    F: np.ndarray  # %
    A: np.ndarray  # %
    own_y1: np.ndarray  # RTR_i(Y+1) 자기 창
    own_lost: np.ndarray  # bool
    G: np.ndarray  # (n, n) G[a, b] = G_a^(ab)
    H: np.ndarray  # (n, n) H[a, b] = H_a^(ab) (Y+1 창 쌍 규칙 실패면 NaN)
    R: np.ndarray  # (n, n) R[a, b] = RTR_a^(ab)(Y+1) 보수 조정 안 한 값
    ok_y: np.ndarray  # (n, n) bool
    ok_y1: np.ndarray  # (n, n) bool
    operator: list[str | None]


def gy_values(
    ctx: Ctx, gy: GroupYear, fee_field: str = "ter", lag_days: int = fees.LAG_DAYS
) -> GyValues:
    d = ctx.dense
    n = len(gy.cols)
    y = gy.year
    t_last = d.dates[d.year_range(y)[1]]
    fund = [ctx.code_map.get(c) for c in gy.isu]
    F = np.array([fees.formation_value(ctx.fee, f, t_last, lag_days, fee_field) for f in fund])
    A = np.array([fees.outcome_value(ctx.fee, f, y + 1, fee_field) for f in fund])
    own = [own_rtr(ctx, int(c), y + 1) for c in gy.cols]
    G = np.full((n, n), np.nan)
    H = np.full((n, n), np.nan)
    R = np.full((n, n), np.nan)
    oky = np.zeros((n, n), dtype=bool)
    oky1 = np.zeros((n, n), dtype=bool)
    for a in range(n):
        for b in range(a + 1, n):
            ca, cb = int(gy.cols[a]), int(gy.cols[b])
            oy = pair_window_ok(ctx, ca, cb, y)
            oy1 = pair_window_ok(ctx, ca, cb, y + 1)
            oky[a, b] = oky[b, a] = oy
            oky1[a, b] = oky1[b, a] = oy1
            if oy:
                ra, rb = pair_rtr(ctx, ca, cb, y)
                G[a, b] = ra + F[a] / 100.0
                G[b, a] = rb + F[b] / 100.0
            if oy1:
                ra, rb = pair_rtr(ctx, ca, cb, y + 1)
                R[a, b], R[b, a] = ra, rb
                H[a, b] = ra + A[a] / 100.0
                H[b, a] = rb + A[b] / 100.0
    return GyValues(
        F=F,
        A=A,
        own_y1=np.array([o[0] for o in own]),
        own_lost=np.array([o[1] for o in own]),
        G=G,
        H=H,
        R=R,
        ok_y=oky,
        ok_y1=oky1,
        operator=[ctx.fee.operator.get(f) if f else None for f in fund],
    )
