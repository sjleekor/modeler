"""지표 (09b I11): CAGR·연 변동성·샤프·MDD·최장 손실 기간·칼마·연 회전율, 연도별 표."""

from __future__ import annotations

from datetime import date

import numpy as np

TD = 252


def max_drawdown(nav: np.ndarray) -> float:
    """``nav``(시작 NAV 포함)의 최대 낙폭(음수 또는 0)."""
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1.0).min())


def longest_underwater(dates: list[date], nav: np.ndarray) -> dict:
    """NAV가 직전 최고점 아래 머문 가장 긴 기간.

    기간은 최고점 날짜에서 그 값을 되찾은(NAV >= 최고점) 첫 날까지다. 끝까지 못 되찾으면
    ``ongoing=True``이고 마지막 날까지로 센다. 달력일 기준으로 가장 긴 구간을 고른다.
    """
    best = {
        "calendar_days": 0,
        "trading_days": 0,
        "ongoing": False,
        "peak_date": None,
        "end_date": None,
    }
    peak_i = 0
    for i in range(1, len(nav)):
        if nav[i] >= nav[peak_i]:
            if i - 1 > peak_i or nav[i - 1] < nav[peak_i]:
                _consider(best, dates, peak_i, i, False)
            peak_i = i
    if len(nav) and nav[-1] < nav[peak_i]:
        _consider(best, dates, peak_i, len(nav) - 1, True)
    return best


def _consider(best: dict, dates: list[date], p: int, q: int, ongoing: bool) -> None:
    cal = (dates[q] - dates[p]).days
    if cal > best["calendar_days"]:
        best.update(
            calendar_days=cal,
            trading_days=q - p,
            ongoing=ongoing,
            peak_date=dates[p],
            end_date=dates[q],
        )


def compute_metrics(
    dates: list[date],  # 성과 첫날 직전(시작) 날짜 + 성과 날들 (len = n+1)
    nav: np.ndarray,  # 시작 NAV(=1) 포함 (len = n+1)
    cash_ret: np.ndarray,  # 성과 날들의 현금 일간 수익 (len = n)
    turnovers: list[tuple[date, float]],  # (체결일, Σ_ETF|Δw|) — 시작 매수 제외
) -> dict:
    n = len(nav) - 1
    r = nav[1:] / nav[:-1] - 1.0
    cagr = float(nav[-1] ** (TD / n) - 1.0)
    vol = float(r.std(ddof=1) * np.sqrt(TD))
    ex = r - cash_ret
    sd_ex = ex.std(ddof=1)
    sharpe = float(ex.mean() * TD / (sd_ex * np.sqrt(TD))) if sd_ex > 0 else float("nan")
    mdd = max_drawdown(nav)
    uw = longest_underwater(dates, nav)
    calmar = float(cagr / abs(mdd)) if mdd < 0 else float("nan")
    by_year: dict[int, float] = {}
    for d, t in turnovers:
        by_year[d.year] = by_year.get(d.year, 0.0) + t / 2.0
    turnover_ann = float(np.mean(list(by_year.values()))) if by_year else 0.0
    total_to = sum(t for _, t in turnovers) / 2.0
    turnover_alt = float(total_to / (n / TD)) if n else 0.0
    return {
        "n_days": n,
        "start": dates[0],
        "end": dates[-1],
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "mdd": mdd,
        "longest_underwater_cal_days": uw["calendar_days"],
        "longest_underwater_trading_days": uw["trading_days"],
        "longest_underwater_ongoing": uw["ongoing"],
        "longest_underwater_peak": uw["peak_date"],
        "longest_underwater_end": uw["end_date"],
        "calmar": calmar,
        "turnover_ann": turnover_ann,
        "turnover_ann_alt_total_over_years": turnover_alt,
        "terminal_nav": float(nav[-1]),
    }


def yearly_table(dates: list[date], nav: np.ndarray) -> list[dict]:
    """연도별 수익과 연중 MDD. 한 해의 기준 NAV는 전년 마지막 NAV(첫 해는 시작 NAV)."""
    rows = []
    years = sorted({d.year for d in dates[1:]})
    for y in years:
        idx = [i for i in range(1, len(dates)) if dates[i].year == y]
        base_i = idx[0] - 1
        seg = nav[base_i : idx[-1] + 1]
        rows.append(
            {
                "year": y,
                "ret": float(nav[idx[-1]] / nav[base_i] - 1.0),
                "mdd": max_drawdown(seg),
                "days": len(idx),
            }
        )
    return rows
