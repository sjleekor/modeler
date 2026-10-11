"""ETF E3 — 일별 패널을 (거래일 × ETF) 조밀 행렬로 바꾼다 (사양 §3·§4의 계산 바탕).

값 계산이 아니라 모양 바꾸기다. 거래일 달력은 ``etf_panel.read_panel`` 의 달력(종가가 있는 행이 하나라도 있는
날)이고, 그 ETF의 행이 없거나 값이 비면 NaN이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from modeler.scores.quality import etf_panel as ep


@dataclass
class Dense:
    isu: list[str]
    col: dict[str, int]
    dates: np.ndarray  # datetime64[D], 길이 T
    years: np.ndarray  # int, 길이 T
    nav: np.ndarray  # (T, N)
    close: np.ndarray
    prev_ref: np.ndarray  # 종가 − 전일 대비(KRX 기준가)
    shares: np.ndarray  # 상장좌수
    has_row: np.ndarray  # bool (T, N): 그 ETF 행이 있는 날

    @property
    def n_days(self) -> int:
        return len(self.dates)

    def year_range(self, y: int) -> tuple[int, int]:
        """연 y의 첫·마지막 거래일 번호(양 끝 포함). 없으면 (-1, -1)."""
        idx = np.flatnonzero(self.years == y)
        if len(idx) == 0:
            return -1, -1
        return int(idx[0]), int(idx[-1])


def build_dense(panel: ep.Panel) -> Dense:
    rows = panel.rows
    isu = sorted(rows["ISU_CD"].unique().to_list())
    col = {c: i for i, c in enumerate(isu)}
    T = panel.n_days
    N = len(isu)
    t = rows["day_idx"].to_numpy()
    j = rows["ISU_CD"].replace_strict(col, return_dtype=pl.Int64).to_numpy()

    def mat(name: str) -> np.ndarray:
        m = np.full((T, N), np.nan)
        m[t, j] = rows[name].to_numpy().astype(float)
        return m

    close = mat("TDD_CLSPRC")
    cmp_ = mat("CMPPREVDD_PRC")
    has = np.zeros((T, N), dtype=bool)
    has[t, j] = True
    dates = panel.calendar["date"].to_numpy().astype("datetime64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    return Dense(
        isu=isu,
        col=col,
        dates=dates,
        years=years,
        nav=mat("NAV"),
        close=close,
        prev_ref=close - cmp_,
        shares=mat("LIST_SHRS"),
        has_row=has,
    )
