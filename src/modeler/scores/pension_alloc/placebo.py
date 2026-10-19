"""G-d placebo: 월 비중 경로의 순환 이동 (사전등록 §6 G-d·§10 Q6 ②).

통계량은 Calmar 하나. 가능한 모든 순환 이동(이동량 s는 6 이상, n−6 이하 개월)을 쓰고
``p = (1 + #{이동의 Calmar ≥ 실제}) / (1 + #이동)``. 비용마다 따로 구한다. 후보 둘 사이는 Holm.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

MIN_SHIFT = 6


def circular_shifts(n: int, min_shift: int = MIN_SHIFT) -> list[int]:
    """이동량 ``min_shift ≤ s ≤ n − min_shift``. 월 수가 모자라면 빈 목록."""
    return list(range(min_shift, n - min_shift + 1))


def placebo_pvalue(actual: float, shifted: list[float]) -> float:
    """``(1 + #{shifted ≥ actual}) / (1 + len(shifted))``. NaN 이동은 이긴 것으로 세지 않는다.

    실제 값이 NaN이면 판정 불가이므로 1.0(통과 못 함)을 돌려준다.
    """
    if np.isnan(actual):
        return 1.0
    wins = sum(1 for v in shifted if not np.isnan(v) and v >= actual)
    return (1.0 + wins) / (1.0 + len(shifted))


def placebo_calmar(
    weights: np.ndarray,
    calmar_fn: Callable[[np.ndarray], float],
    min_shift: int = MIN_SHIFT,
) -> dict:
    """``weights``: (n개월, 슬롯) 결정 비중 경로. ``calmar_fn(경로) -> Calmar``(비용 고정).

    이동은 결정일은 그대로 두고 비중 행만 ``np.roll``한다.
    """
    n = len(weights)
    shifts = circular_shifts(n, min_shift)
    actual = calmar_fn(weights)
    vals = [calmar_fn(np.roll(weights, s, axis=0)) for s in shifts]
    return {
        "actual": actual,
        "n_shifts": len(shifts),
        "p": placebo_pvalue(actual, vals),
        "shifted": vals,
    }


def holm(pvals: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm 단계 하향 보정. 키마다 ``{"adj_p", "reject"}``.

    작은 p부터 ``adj = max(이전 adj, (m − rank)·p)``(1로 자름).
    한 번 못 넘으면 이후는 모두 기각 안 함.
    """
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict[str, dict] = {}
    run_max = 0.0
    stopped = False
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, max(run_max, (m - rank) * p))
        run_max = adj
        rej = (not stopped) and adj < alpha
        if not rej:
            stopped = True
        out[k] = {"adj_p": adj, "reject": rej}
    return out
