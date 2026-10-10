"""MRS 비중 규칙과 변동성 관리 기준선 비중 (사전등록 §6, §0-3.2).

::

    w(MRS) = 0.5                        MRS <= 20
           = 0.5 + 0.5 * (MRS - 20)/60  20 < MRS < 80
           = 1.0                        MRS >= 80
    w_vm   = clip(sigma_target / sigma_20, 0.5, 1.0)      (바닥 있음, 기준선)
    w_vm_nofloor = min(1, sigma_target / sigma_20)        (바닥 없음, 민감도)

* MRS가 null이면 비중 1.0이다 ("모르면 하던 대로 한다", §3.3).
* 변동성 관리에서 ``sigma_20``이나 ``sigma_target``이 null이거나 ``sigma_20 <= 0``이면 비중 1.0이다
  (MI08 — 변동성을 모르면 하던 대로 한다. MRS null과 같은 쓰임).
* IRP 상한(0.7)은 MRS·기준선 비중 모두에 ``min(w, 0.7)``로 건다 (§0-3.2).
* null은 ``NaN``으로 받는다.
"""

from __future__ import annotations

import numpy as np

from modeler.scores.mrs import config as C


def mrs_weight(mrs: np.ndarray) -> np.ndarray:
    """MRS(0~100, NaN=null) -> 비중 w. 사전등록 §6."""
    m = np.asarray(mrs, dtype=float)
    span = C.WEIGHT_HIGH_CUT - C.WEIGHT_LOW_CUT
    lin = C.WEIGHT_FLOOR + (C.WEIGHT_CAP - C.WEIGHT_FLOOR) * (m - C.WEIGHT_LOW_CUT) / span
    w = np.clip(lin, C.WEIGHT_FLOOR, C.WEIGHT_CAP)
    return np.where(np.isfinite(m), w, C.WEIGHT_WHEN_NULL)


def vm_weight(sigma20: np.ndarray, sigma_target: np.ndarray, *, floor: bool = True) -> np.ndarray:
    """변동성 관리 비중. ``floor=True``면 [0.5, 1.0] 클립, ``False``면 ``min(1, ratio)``.

    입력 null(NaN)이거나 ``sigma20 <= 0``이면 1.0 (MI08).
    """
    s = np.asarray(sigma20, dtype=float)
    t = np.asarray(sigma_target, dtype=float)
    ok = np.isfinite(s) & np.isfinite(t) & (s > 0)
    ratio = np.divide(t, s, out=np.ones_like(s, dtype=float), where=ok)
    hi = C.VM_CAP
    w = np.clip(ratio, C.VM_FLOOR, hi) if floor else np.minimum(ratio, hi)
    return np.where(ok, w, C.WEIGHT_WHEN_NULL)


def cap(w: np.ndarray, limit: float = C.IRP_CAP) -> np.ndarray:
    """IRP 위험자산 상한 ``min(w, limit)`` (§0-3.2)."""
    return np.minimum(np.asarray(w, dtype=float), limit)
