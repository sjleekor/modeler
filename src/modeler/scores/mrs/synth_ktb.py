"""합성(추정) 국고채 일별 수익과 혼합 대용 (사전등록 §0-5).

::

    r_ktb(t) = y(t-1) * 일수(t-1, t)/365 - D(t-1) * dy(t) + 1/2 * C(t-1) * dy(t)^2
    dy(t)    = y(t) - y(t-1)                 (y는 소수 수익률)

* ``D``·``C``는 만기 M년 par 채권(분기 이표, 이표율 = 수익률 = ``y(t-1)``)의
  수정 듀레이션·볼록성이고 매일 다시 계산한다.
  가격함수 ``P(y') = sum_k (c/4)(1+y'/4)^-k + (1+y'/4)^-4M``를 해석적으로 미분한다
  (MI26 — 분기 복리 관례).
* 롤다운·이표 재투자·자유 모수·보정·5년 합성은 없다 (§0-5.3, §0-5.4).
* 인덱스는 KR 격자다 (MI27). ``y_i``는 격자일 ``g_i`` 기준 as-of 최신 값(관측일 <= ``g_i``)이고,
  나이가 14일을 넘으면 null이다. 실현값이라 PIT가 아니다.
* 반환 배열은 ``r_cash``와 같은 관례로 정렬한다. ``out[i]``는 ``g_i -> g_{i+1}`` 구간의 수익이고
  마지막 원소는 NaN이다. 그래서 ``ktb_synth_cash_sens``는 이 배열을 ``r_cash``로 그대로 넘기면 된다.
* 합성 값은 **합성(추정)** 이다. 판정·등급·읽는 법에 쓰지 않는다 (§0-5.6).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence
from datetime import date

import numpy as np
import polars as pl

from modeler.scores.mrs import config as C

#: as-of 나이 상한(달력일). cash.py 기존 규칙과 같은 14일 (§0-5 / 사양 4.6).
YIELD_STALENESS_DAYS = C.CASH_STALENESS_DAYS


def par_duration_convexity(
    y: np.ndarray | float, maturity_years: int, freq: int = C.KTB_COUPONS_PER_YEAR
) -> tuple[np.ndarray, np.ndarray]:
    """par 채권의 수정 듀레이션 ``D = -P'(y)/P`` 와 볼록성 ``C = P''(y)/P``.

    ``y``는 소수(0.05 = 5%). 이표율 = ``y``라 ``P(y) = 1``이다. NaN은 NaN으로 낸다.
    """
    yy = np.atleast_1d(np.asarray(y, dtype=float))
    n_pay = int(maturity_years * freq)
    k = np.arange(1, n_pay + 1, dtype=float)[None, :]
    q = 1.0 + yy[:, None] / freq  # 기간 할인 인자 밑
    cpn = yy[:, None] / freq
    with np.errstate(invalid="ignore", divide="ignore"):
        price = (cpn * q ** (-k)).sum(axis=1) + q[:, 0] ** (-n_pay)
        d1 = (cpn * (-k / freq) * q ** (-k - 1)).sum(axis=1) + (-(n_pay / freq)) * q[:, 0] ** (
            -n_pay - 1
        )
        d2 = (cpn * (k * (k + 1) / freq**2) * q ** (-k - 2)).sum(axis=1) + (
            n_pay * (n_pay + 1) / freq**2
        ) * q[:, 0] ** (-n_pay - 2)
    return -d1 / price, d2 / price


def yields_on_grid(
    grid_dates: Sequence[date],
    yield_rows: pl.DataFrame,
    *,
    staleness_days: int = YIELD_STALENESS_DAYS,
) -> np.ndarray:
    """격자일마다 as-of 최신 수익률(소수). 나이가 ``staleness_days``를 넘으면 NaN.

    ``yield_rows``: ``date``(관측일), ``value``(%). 같은 관측일 중복은 마지막 행을 쓴다.
    """
    rows = yield_rows.drop_nulls(["date", "value"]).unique("date", keep="last").sort("date")
    ds = rows["date"].to_list()
    vs = rows["value"].to_list()
    out = np.full(len(grid_dates), np.nan)
    for i, g in enumerate(grid_dates):
        k = bisect_right(ds, g) - 1
        if k >= 0 and (g - ds[k]).days <= staleness_days:
            out[i] = vs[k] / 100.0
    return out


def synth_returns(
    grid_dates: Sequence[date],
    yield_rows: pl.DataFrame,
    maturity_years: int,
    *,
    staleness_days: int = YIELD_STALENESS_DAYS,
) -> np.ndarray:
    """격자 구간 ``g_i -> g_{i+1}``의 합성 국고채 수익 (마지막 원소 NaN, 계산 불가 구간 NaN)."""
    n = len(grid_dates)
    y = yields_on_grid(grid_dates, yield_rows, staleness_days=staleness_days)
    out = np.full(n, np.nan)
    if n < 2:
        return out
    days = np.array([(grid_dates[i + 1] - grid_dates[i]).days for i in range(n - 1)], dtype=float)
    dur, conv = par_duration_convexity(y, maturity_years)
    dy = y[1:] - y[:-1]
    out[:-1] = y[:-1] * days / C.KTB_DAYCOUNT - dur[:-1] * dy + 0.5 * conv[:-1] * dy**2
    return out


def mix_synth(
    name: str,
    grid_dates: Sequence[date],
    r_bh: np.ndarray,
    r_ktb: np.ndarray,
) -> np.ndarray:
    """혼합 대용 ``w*r_bh + (1-w)*r_ktb`` (매일 재조정, 비용 0 — §0-2와 같은 단순화).

    ``name``은 ``mix_synth_ps1``(0.30 + 합성 3년) 또는 ``mix_synth_ps2``(0.60 + 합성 10년).
    ``r_ktb``는 ``name``에 맞는 만기의 :func:`synth_returns` 결과다.
    ps2는 2001-01-01 이후 첫 격자일부터만 값이 있고 그 앞은 NaN이다
    (MI28, §0-5.4 — 5년 등으로 메우지 않는다).
    """
    w_risk, leg = C.MIX_SYNTH[name]
    del leg  # 호출자가 맞는 만기 배열을 넘긴다. 이름 확인용 상수.
    out = w_risk * np.asarray(r_bh, dtype=float) + (1.0 - w_risk) * np.asarray(r_ktb, dtype=float)
    if name == "mix_synth_ps2":
        first = next(
            (i for i, g in enumerate(grid_dates) if g >= C.MIX_SYNTH_PS2_START), len(grid_dates)
        )
        out[:first] = np.nan
    return out
