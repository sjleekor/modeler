"""Baseline (사양 02 §3, §5.3).

* Opportunity: **자산별 과거 평균**. 결정 시각 ``D``마다 ``label_end_at < D``인(=D 이전에 만기된)
  자산의 학습 후보 행 타깃 평균. 하루 단위로 갱신된다(fold 단위 고정보다 baseline에 유리하다).
* Stability (a): 자산별 기저율을 pooled로 수축 ``(events + k*pooled)/(n + k)``. 같은 PIT 규칙.
* Stability (b): 자산 절편 + ``rvol_20`` L2 Logistic (주 모델과 같은 C). fold마다 학습 경계
  안의 행으로 맞춘다. 사건이 30개 미만이거나 한 클래스뿐인 자산은 null + 사유.
* pooled 기저율은 보고만 하고 주 baseline으로 쓰지 않는다.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from modeler.scores.market_sector.config import MsConfig

NAN = float("nan")


def epoch_us(s: pl.Series) -> np.ndarray:
    return s.dt.epoch("us").fill_null(np.iinfo(np.int64).min).to_numpy()


def pit_expanding_mean(
    asset: np.ndarray,
    decision_us: np.ndarray,
    label_end_us: np.ndarray,
    y: np.ndarray,
    pool: np.ndarray,
) -> np.ndarray:
    """자산별, 각 행의 ``decision_us`` 이전에 만기된 ``pool`` 행 ``y``의 평균. 없으면 NaN."""
    out = np.full(len(y), NAN)
    valid = pool & ~np.isnan(y)
    for a in np.unique(asset):
        idx = np.flatnonzero(asset == a)
        src = idx[valid[idx]]
        if len(src) == 0:
            continue
        order = np.argsort(label_end_us[src], kind="stable")
        e = label_end_us[src][order]
        cs = np.cumsum(y[src][order])
        n = np.searchsorted(e, decision_us[idx], side="left")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = np.where(n > 0, cs[np.maximum(n - 1, 0)] / np.maximum(n, 1), NAN)
    return out


def _pit_counts(
    decision_us: np.ndarray,
    label_end_us: np.ndarray,
    ev: np.ndarray,
    sel: np.ndarray,
    at: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """``sel`` 행의 사건 수·표본 수 누적을 ``at`` 행들의 결정 시각에서 읽는다."""
    src = np.flatnonzero(sel)
    if len(src) == 0:
        z = np.zeros(len(at))
        return z, z
    order = np.argsort(label_end_us[src], kind="stable")
    e = label_end_us[src][order]
    cs = np.cumsum(ev[src][order])
    n = np.searchsorted(e, decision_us[at], side="left")
    events = np.where(n > 0, cs[np.maximum(n - 1, 0)], 0.0)
    return events, n.astype(float)


def pit_smoothed_rate(
    asset: np.ndarray,
    decision_us: np.ndarray,
    label_end_us: np.ndarray,
    event: np.ndarray,
    pool: np.ndarray,
    k: int,
) -> dict[str, np.ndarray]:
    """자산별 수축 기저율(a)과 pooled 기저율. 반환 배열은 전체 행 길이."""
    valid = pool & ~np.isnan(event)
    all_rows = np.arange(len(event))
    ev_pool, n_pool = _pit_counts(decision_us, label_end_us, event, valid, all_rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.where(n_pool > 0, ev_pool / np.maximum(n_pool, 1), NAN)
    asset_rate = np.full(len(event), NAN)
    n_asset = np.zeros(len(event))
    ev_asset = np.zeros(len(event))
    for a in np.unique(asset):
        idx = np.flatnonzero(asset == a)
        ev, n = _pit_counts(decision_us, label_end_us, event, valid & (asset == a), idx)
        ev_asset[idx], n_asset[idx] = ev, n
        asset_rate[idx] = (ev + k * pooled[idx]) / (n + k)
    asset_rate[np.isnan(pooled)] = NAN
    return {
        "asset_rate": asset_rate,
        "pooled_rate": pooled,
        "n_asset": n_asset,
        "events_asset": ev_asset,
    }


def train_event_status(
    asset: np.ndarray, y: np.ndarray, train_idx: np.ndarray, cfg: MsConfig
) -> dict[str, str | None]:
    """학습 창 안 자산별 사건 수·클래스. 문제가 있으면 사유 문자열, 없으면 ``None``."""
    out: dict[str, str | None] = {}
    yt, at = y[train_idx], asset[train_idx]
    for a in np.unique(asset):
        ya = yt[(at == a) & ~np.isnan(yt)]
        ev = int((ya == 1).sum())
        if len(ya) == 0:
            out[str(a)] = "no_train_rows"
        elif ev == 0 or ev == len(ya):
            out[str(a)] = "single_class_in_train"
        elif ev < cfg.min_train_events_per_asset:
            out[str(a)] = f"events_in_train<{cfg.min_train_events_per_asset}"
        else:
            out[str(a)] = None
    return out
