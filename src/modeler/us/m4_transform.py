"""M4 입력 변환 — ``03_model_candidates.md`` §4.

    피쳐를 그날 횡단면 순위 [0,1]로 → 결측은 0.5 + ``_isna`` 플래그

``us_features_v1``은 44개 피쳐마다 ``<feature>_isna`` 동반 컬럼을 이미 갖고
있다(``01_isna == is_null()``, 2026-09-21 실측 — 어느 컬럼도 어긋나지
않았다). 이 모듈은 그 플래그를 다시 계산하지 않고 그대로 쓴다.

순위 정의는 ``modeler.us.labels._percentile_rank``와 같다 — ``(rank(method=
"min")-1)/(n-1)``, 그날 비결측 이름 사이에서만. 결측 행은 순위를 매기지
않고(정의상 불가능하다) 0.5로 채운다 — "중간값" 취급이며, 모델이 실제
결측인지는 ``_isna`` 플래그로 따로 안다.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

#: 결측 피쳐에 채우는 값 — 순위 공간의 중간값.
MISSING_FILL = 0.5


def rank_transform(
    df: pl.DataFrame, feature_cols: Sequence[str], *, date_col: str = "date"
) -> pl.DataFrame:
    """``feature_cols`` 각각에 ``<col>_rank`` 컬럼을 붙인다.

    ``<col>_isna``가 이미 ``df``에 있어야 한다(``us_features_v1`` 스키마).
    비결측 행끼리만 ``date_col`` 그룹 안에서 백분위 순위를 매기고([0,1]),
    결측 행은 ``MISSING_FILL``로 채운다. 원본 ``feature_cols``·``_isna``
    컬럼은 그대로 남는다 — 나중에 어떤 걸 썼는지 추적하기 위해서다.
    """
    exprs = []
    for col in feature_cols:
        isna_col = f"{col}_isna"
        if isna_col not in df.columns:
            raise KeyError(f"{isna_col!r}가 없습니다 — us_features_v1 스키마를 확인하십시오")
        n_present = pl.col(col).is_not_null().sum().over(date_col)
        rank = pl.col(col).rank(method="min").over(date_col)
        denom = (n_present - 1).clip(lower_bound=1).cast(pl.Float64)
        pct = ((rank.cast(pl.Float64) - 1.0) / denom).fill_null(MISSING_FILL)
        exprs.append(pct.alias(f"{col}_rank"))
    return df.with_columns(exprs)


def design_matrix_columns(feature_cols: Sequence[str]) -> list[str]:
    """모델 입력 컬럼 이름 — ``<col>_rank`` 전부 뒤에 ``<col>_isna`` 전부.

    순서를 고정해 두면 여러 fold·시드에서 같은 열 순서로 numpy 배열을
    뽑을 수 있다(디버깅 시 어느 열이 무엇인지 헷갈리지 않는다).
    """
    return [f"{c}_rank" for c in feature_cols] + [f"{c}_isna" for c in feature_cols]


def to_design_arrays(df: pl.DataFrame, feature_cols: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    """``df``에서 설계행렬(``float64``, ``_isna``는 0.0/1.0)과 컬럼 이름을 뽑는다."""
    cols = design_matrix_columns(feature_cols)
    arrays = [df[c].cast(pl.Float64).to_numpy() for c in cols]
    x = np.column_stack(arrays) if arrays else np.zeros((df.height, 0))
    return x, cols


def cross_sectional_percentile(
    df: pl.DataFrame, value_col: str, *, date_col: str = "date", out_col: str | None = None
) -> pl.DataFrame:
    """``value_col``의 그날 횡단위 백분위 순위(``[0,1]``)를 ``out_col``에 붙인다.

    ``modeler.us.labels._percentile_rank``와 같은 정의(``(rank(min)-1)/(n-1)``)
    지만, 이 함수는 결측이 없다고 가정한다(``L1``은 개발 구간에 결측이 없다
    — 2026-09-21 실측) — 결측이 섞이면 ``rank_transform``을 쓴다.
    """
    out_col = out_col or f"{value_col}_rank"
    n = pl.col(value_col).count().over(date_col)
    rank = pl.col(value_col).rank(method="min").over(date_col)
    denom = (n - 1).clip(lower_bound=1).cast(pl.Float64)
    pct = (rank.cast(pl.Float64) - 1.0) / denom
    return df.with_columns(pct.alias(out_col))
