"""``modeler.us.build_features``의 순수 함수 단위 테스트.

레이크를 읽지 않는다 — ``main()``은 ``UsLake.resolve()``(``STOCK_DATA_ROOT``)와
실제 데이터셋 생성을 부르므로 통합 실행(``build_features.py`` 작업 보고 참고)으로만
검증하고, 여기서는 합성 ``pl.DataFrame``으로 재현 가능한 ``_missing_rate``만 본다.
"""

from __future__ import annotations

import math

import polars as pl

from modeler.us.build_features import _missing_rate


def test_missing_rate_computes_null_fraction_per_column() -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, None, None, 4.0],
            "b": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = _missing_rate(df, ["a", "b"])

    assert result == {"a": 0.5, "b": 0.0}


def test_missing_rate_all_null_column_is_one() -> None:
    df = pl.DataFrame({"a": pl.Series("a", [None, None], dtype=pl.Float64)})

    result = _missing_rate(df, ["a"])

    assert result == {"a": 1.0}


def test_missing_rate_empty_frame_returns_nan_for_each_column() -> None:
    df = pl.DataFrame(schema={"a": pl.Float64, "b": pl.Float64})

    result = _missing_rate(df, ["a", "b"])

    assert math.isnan(result["a"])
    assert math.isnan(result["b"])
