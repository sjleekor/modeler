"""일간 장부 (09b I5·I6).

비중은 매일 가격 따라 흘러간다. 결정일 t의 정보로 정한 목표 비중은 t+1 종가에 맞추고
(t+1 하루 수익은 옛 비중, t+2부터 새 비중), 비용은 ETF 쪽 ``Σ|목표 − 흘러간 비중|``에
편도 비용을 곱해 t+1 종가에 뺀다. 현금 계정 쪽 이동은 비용이 없다. 시작 매수 비용은 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LedgerResult:
    #: 성과 구간 첫날 직전(시작 종가) 인덱스. NAV = 1.
    start_idx: int
    #: 성과에 들어간 날(인덱스, 오름차순). 현금 null 세션은 빠진다.
    idx: np.ndarray
    nav_start: float
    nav: np.ndarray  # idx와 같은 길이, 그날 종가 NAV(비용 차감 뒤)
    ret: np.ndarray  # 그날 수익률(비용 반영 — 비용이 든 날은 비용 포함)
    weights_held: np.ndarray  # (len(idx), K+1) 그날 수익에 쓴 비중(날 시작)
    rebalances: list[dict] = field(default_factory=list)  # idx, turnover(Σ|Δw|), cost


def simulate(
    asset_ret: np.ndarray,
    cash_ret: np.ndarray,
    start_idx: int,
    end_idx: int,
    targets: dict[int, np.ndarray],
    one_way: float,
) -> LedgerResult:
    """``asset_ret``: (T, K) ETF 일간 총수익. ``cash_ret``: (T,) 현금 일간 수익(null=NaN).

    ``targets[i]``: i일 종가에 맞출 목표 비중 (K+1, 마지막이 현금, 합 1).
    ``start_idx``의 목표는 비용 없이 시작 종가에 잡는다. 성과는 ``start_idx+1``부터.
    """
    T, K = asset_ret.shape
    if start_idx not in targets:
        raise ValueError("start_idx의 목표 비중이 없습니다")
    w = np.asarray(targets[start_idx], dtype=float).copy()
    if abs(w.sum() - 1.0) > 1e-9 or (w < -1e-12).any():
        raise ValueError(f"목표 비중이 잘못됐습니다: {w}")
    nav = 1.0
    idx_l, nav_l, w_l = [], [], []
    rebs: list[dict] = []
    for t in range(start_idx + 1, end_idx + 1):
        cr = cash_ret[t]
        ok = not np.isnan(cr)
        if ok:
            rr = np.append(asset_ret[t], cr)
            # 비중 0인 자산의 NaN(상장 전)은 쓰지 않는다.
            if (np.isnan(rr) & (w != 0)).any():
                raise ValueError(f"{t}일: 비중이 있는 자산의 수익이 없습니다")
            rr = np.where(w == 0, 0.0, rr)
            w_l.append(w.copy())
            gross = float(w @ (1.0 + rr))
            w = w * (1.0 + rr) / gross
            nav *= gross
        if t in targets:
            tgt = np.asarray(targets[t], dtype=float)
            if abs(tgt.sum() - 1.0) > 1e-9 or (tgt < -1e-12).any():
                raise ValueError(f"{t}일 목표 비중이 잘못됐습니다: {tgt}")
            turnover = float(np.abs(tgt[:K] - w[:K]).sum())
            cost = turnover * one_way
            nav *= 1.0 - cost
            w = tgt.copy()
            rebs.append({"idx": t, "turnover": turnover, "cost": cost})
        if ok:
            # 현금 null 세션은 성과 표본에서 빠진다(그날의 수익은 없다).
            idx_l.append(t)
            nav_l.append(nav)
    nav_arr = np.array(nav_l)
    prev = np.concatenate([[1.0], nav_arr[:-1]]) if len(nav_arr) else nav_arr
    return LedgerResult(
        start_idx=start_idx,
        idx=np.array(idx_l, dtype=int),
        nav_start=1.0,
        nav=nav_arr,
        ret=nav_arr / prev - 1.0,
        weights_held=np.array(w_l).reshape(-1, K + 1),
        rebalances=rebs,
    )
