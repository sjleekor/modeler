"""평가 하네스와 실제 자료 진입점 (사전등록 §4.1·§6·§10 Q6).

* :func:`evaluate_candidate`는 배열만 받는다(합성 자료로 시험 가능). 실제 파일을 열지 않는다.
  기존 기준선 장부(``etf_baseline.ledger.simulate``)와 지표(``compute_metrics``)를 그대로 쓴다.
* :func:`main`은 실제 자료를 읽는 유일한 진입점이고, **첫 줄에서** 동결 태그 가드를 부른다.
  가드가 실패하면 자료를 열기 전에 멈춘다. 이 모듈 밖에서 ``etf_baseline.run.prepare``
  (실제 자료를 읽는 함수)를 부르지 않는다.
"""

from __future__ import annotations

from datetime import date

import numpy as np

from modeler.scores.etf_baseline.ledger import LedgerResult, simulate
from modeler.scores.etf_baseline.metrics import compute_metrics
from modeler.scores.pension_alloc.guard import require_frozen_tag
from modeler.scores.pension_alloc.placebo import placebo_calmar

COSTS = {"low": 0.0005, "high": 0.0030}  # 편도 (왕복 10bp·60bp, §4.1 E4)


def targets_from_weights(decisions: list[int], weights: np.ndarray) -> dict[int, np.ndarray]:
    """결정일 d의 목표 비중 -> 체결일 d+1 종가(§4.1 E3)."""
    return {d + 1: np.asarray(w, dtype=float) for d, w in zip(decisions, weights, strict=True)}


def run_ledger(
    asset_ret: np.ndarray,
    cash_ret: np.ndarray,
    decisions: list[int],
    weights: np.ndarray,
    one_way: float,
    end_idx: int,
) -> LedgerResult:
    tg = targets_from_weights(decisions, weights)
    return simulate(asset_ret, cash_ret, decisions[0] + 1, end_idx, tg, one_way)


def _metrics(days: list[date], res: LedgerResult, cash_ret: np.ndarray, start_idx: int) -> dict:
    dates_ = [days[start_idx]] + [days[i] for i in res.idx]
    nav = np.concatenate([[1.0], res.nav])
    tos = [(days[r["idx"]], r["turnover"]) for r in res.rebalances]
    return compute_metrics(dates_, nav, cash_ret[res.idx], tos)


def b5r_weights(res: LedgerResult) -> np.ndarray:
    """B5_R: 후보 비중의 기간 시간평균(슬롯별, 흘러간 비중 기준 — 09b I9와 같은 정의)."""
    w = res.weights_held.mean(axis=0)
    return w / w.sum()


def evaluate_candidate(
    days: list[date],
    asset_ret: np.ndarray,
    cash_ret: np.ndarray,
    decisions: list[int],
    weights: np.ndarray,
    end_idx: int,
    costs: dict[str, float] | None = None,
    placebo: bool = True,
) -> dict:
    """후보 하나를 비용마다 평가한다. ``weights``: (결정 수, K+1) 목표 비중(마지막 열 현금).

    반환 ``{비용이름: {"metrics", "b5r_metrics", "b5r_weights", "placebo"}}``.
    B5_R과 placebo는 비용 수준마다 따로 구한다(§10 Q6 ②·③).
    """
    out: dict = {}
    start_idx = decisions[0] + 1
    for name, ow in (costs or COSTS).items():
        res = run_ledger(asset_ret, cash_ret, decisions, weights, ow, end_idx)
        const = b5r_weights(res)
        res_b5 = run_ledger(
            asset_ret, cash_ret, decisions, np.tile(const, (len(decisions), 1)), ow, end_idx
        )

        def calmar(path: np.ndarray, ow: float = ow) -> float:
            r = run_ledger(asset_ret, cash_ret, decisions, path, ow, end_idx)
            return _metrics(days, r, cash_ret, start_idx)["calmar"]

        out[name] = {
            "metrics": _metrics(days, res, cash_ret, start_idx),
            "b5r_weights": const,
            "b5r_metrics": _metrics(days, res_b5, cash_ret, start_idx),
            "placebo": placebo_calmar(weights, calmar) if placebo else None,
        }
    return out


def main() -> None:
    """실제 자료 진입점. 동결 태그가 없으면 자료를 열기 전에 멈춘다."""
    require_frozen_tag()  # 반드시 첫 호출 — 이 줄 앞에 자료 접근이 없다
    raise NotImplementedError(
        "동결 전에는 구현하지 않습니다: KOSPI 가격지수 적재·결정일 정렬·진단 구간 날짜는 "
        "동결 때 정합니다(사전등록 §10 Q3·Q8)."
    )


if __name__ == "__main__":
    main()
