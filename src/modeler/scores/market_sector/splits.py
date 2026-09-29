"""연도별 expanding walk-forward 분할.

fold Y의 test = 그 해 달력 연도의 결정 세션. 학습 행은 **행마다** ``label_end_at <
fold_start``(fold Y의 첫 ``decision_at``)를 만족해야 한다. "12월 31일까지 train"으로 자르지
않는다 — 60세션 라벨이 연말 넘어 만기되는 행이 test로 새기 때문이다. 경계는 모든 자산에
같은 실제 시각이고, scaler·imputer·calibrator를 맞추는 행도 같은 경계 안이다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from modeler.scores.market_sector.config import MsConfig


class SplitBoundaryError(AssertionError):
    """학습 행의 ``label_end_at``이 fold 시작 시각 이후다."""


def _us(s: pl.Series) -> np.ndarray:
    """tz-aware Datetime Series -> epoch µs (null은 최솟값 센티넬)."""
    return s.dt.epoch("us").fill_null(np.iinfo(np.int64).min).to_numpy()


def assert_train_boundary(
    label_end_at: pl.Series, idx: np.ndarray, fold_start: datetime, *, what: str = "train"
) -> None:
    """``idx`` 행이 전부 ``label_end_at < fold_start``인지 확인. null도 위반이다."""
    if len(idx) == 0:
        return
    sub = label_end_at.gather(idx.tolist())
    if sub.null_count():
        raise SplitBoundaryError(f"{what}: label_end_at이 null인 행이 있습니다")
    bad = sub.filter(sub >= fold_start)
    if bad.len():
        raise SplitBoundaryError(
            f"{what}: label_end_at >= fold_start {fold_start} 인 행 {bad.len()}개 "
            f"(가장 이른 {bad.min()})"
        )


@dataclass(frozen=True)
class Fold:
    year: int | str  # 정수 연도 또는 "live"
    fold_start: datetime
    train_idx: np.ndarray
    test_idx: np.ndarray
    first_decision: datetime | None
    last_decision: datetime | None
    n_test_matured: int
    skipped_reason: str | None

    @property
    def n_train(self) -> int:
        return len(self.train_idx)

    @property
    def n_test(self) -> int:
        return len(self.test_idx)


def _train_pool(df: pl.DataFrame, market: str, cfg: MsConfig) -> np.ndarray:
    """학습 후보: 라벨 만기·피쳐 준비·train_start 이후. 반환: bool 마스크."""
    start = cfg.train_start[market.upper()]
    m = (
        df["label_matured"].fill_null(False)
        & df["feature_ready"].fill_null(False)
        & (df["session"] >= start)
        & df["label_end_at"].is_not_null()
    )
    return m.to_numpy()


def make_fold(
    df: pl.DataFrame,
    year: int | str,
    fold_start: datetime,
    test_mask: np.ndarray,
    market: str,
    cfg: MsConfig,
) -> Fold:
    pool = _train_pool(df, market, cfg)
    end_us = _us(df["label_end_at"])
    start_us = int(fold_start.timestamp() * 1_000_000)
    train_mask = pool & (end_us < start_us)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)
    assert_train_boundary(df["label_end_at"], train_idx, fold_start)
    if len(np.intersect1d(train_idx, test_idx)):
        raise SplitBoundaryError("train과 test 행이 겹칩니다")
    dec = df["decision_at"].gather(test_idx.tolist()) if len(test_idx) else None
    matured = int(df["label_matured"].fill_null(False).to_numpy()[test_idx].sum())
    reason = None
    if len(train_idx) < cfg.min_train_rows:
        reason = f"train_rows<{cfg.min_train_rows}"
    return Fold(
        year=year,
        fold_start=fold_start,
        train_idx=train_idx,
        test_idx=test_idx,
        first_decision=dec.min() if dec is not None else None,
        last_decision=dec.max() if dec is not None else None,
        n_test_matured=matured,
        skipped_reason=reason,
    )


def last_test_year(df: pl.DataFrame) -> int:
    m = df.filter(pl.col("label_matured").fill_null(False))
    if m.height == 0:
        raise ValueError("만기된 라벨이 없습니다")
    return int(m["session"].max().year)


def expanding_annual_folds(df: pl.DataFrame, market: str, cfg: MsConfig) -> list[Fold]:
    """``df``: 열 ``session, decision_at, label_end_at, label_matured, feature_ready``."""
    years = df["session"].dt.year().to_numpy()
    ready = df["feature_ready"].fill_null(False).to_numpy()
    out: list[Fold] = []
    for y in range(cfg.first_test_year[market.upper()], last_test_year(df) + 1):
        in_year = years == y
        if not in_year.any():
            continue
        fold_start = df.filter(pl.Series(in_year))["decision_at"].min()
        out.append(make_fold(df, y, fold_start, in_year & ready, market, cfg))
    return out


def live_fold(df: pl.DataFrame, market: str, cfg: MsConfig) -> Fold:
    """가장 최근 결정 시각에서 학습한 모델로 자산별 마지막 행을 예측한다."""
    fold_start = df["decision_at"].max()
    ready = df["feature_ready"].fill_null(False).to_numpy()
    at_last = (
        df.select(pl.col("session") == pl.col("session").max().over("asset_id"))
        .to_series()
        .to_numpy()
    )
    return make_fold(df, "live", fold_start, at_last & ready, market, cfg)
