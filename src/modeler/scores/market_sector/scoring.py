"""점수 변환 (사양 02 §5.2).

* Opportunity 점수 = 예측값이 **이전 fold들의 expanding OOF 예측** 안에서 차지하는 백분위(0~100,
  동점은 중간 순위). 시장 그룹별 reference. 현재 fold 예측은 자기 reference에 들어가지 않는다.
  reference 예측 수가 ``opportunity_reference_min_oof`` 미만이면 원예측은 남기고 점수는 null,
  ``score_status="warmup"``.
* Stability v1 = ``100 * (1 - p_hat)``. 원확률 그대로이므로 ``calibration_method="none"``,
  ``validation_status="research"``.
"""

from __future__ import annotations

import numpy as np

OPPORTUNITY_REFERENCE_VERSION = "expanding_oof_previous_folds_v1"
STABILITY_CALIBRATION_METHOD = "none"
STABILITY_VALIDATION_STATUS = "research"
STATUS_OK = "ok"
STATUS_WARMUP = "warmup"
STATUS_NO_PREDICTION = "no_prediction"


def percentile_against(ref: np.ndarray, x: np.ndarray, min_ref: int) -> np.ndarray:
    """``ref``(NaN 제외) 안에서 ``x``의 백분위. ``len(ref) < min_ref``이면 전부 NaN."""
    r = np.sort(ref[~np.isnan(ref)])
    out = np.full(len(x), np.nan)
    if len(r) < max(min_ref, 1):
        return out
    lo = np.searchsorted(r, x, side="left")
    hi = np.searchsorted(r, x, side="right")
    out = 100.0 * (lo + hi) / 2.0 / len(r)
    return np.where(np.isnan(x), np.nan, out)


def opportunity_scores(
    pred: np.ndarray,
    fold_year: np.ndarray,
    market: np.ndarray,
    min_ref: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OOF 점수·상태·reference 크기. fold Y의 reference는 같은 시장의 ``fold_year < Y`` 예측."""
    n = len(pred)
    score = np.full(n, np.nan)
    ref_n = np.zeros(n, dtype=np.int64)
    for m in np.unique(market):
        mm = market == m
        for y in np.unique(fold_year[mm]):
            cur = mm & (fold_year == y)
            ref = pred[mm & (fold_year < y)]
            ref = ref[~np.isnan(ref)]
            ref_n[cur] = len(ref)
            score[cur] = percentile_against(ref, pred[cur], min_ref)
    status = np.where(
        np.isnan(pred), STATUS_NO_PREDICTION, np.where(np.isnan(score), STATUS_WARMUP, STATUS_OK)
    )
    return score, status.astype(object), ref_n


def stability_scores(p_hat: np.ndarray) -> np.ndarray:
    return 100.0 * (1.0 - p_hat)
