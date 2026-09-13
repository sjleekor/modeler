from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from modeler.analysis.horizon_scan_mapping import (
    apply_group_permutation,
    build_and_apply_group_permutation,
    build_group_permutation_mapping,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 2)],
            "market": ["KOSPI", "KOSPI", "KOSPI", "KOSDAQ"],
            "ticker": ["B", "B", "A", "A"],
            "feature_a": [20.0, 11.0, 10.0, 21.0],
            "feature_b": [200.0, 110.0, 100.0, 210.0],
            "fixed": [2, 1, 0, 3],
        }
    )


def test_build_and_apply_reuses_one_canonical_sort_and_matches_explicit_steps() -> None:
    frame = _frame()
    mappings, expected_hash = build_group_permutation_mapping(
        frame, replicate_index=3, config_hash="cfg"
    )
    explicit = apply_group_permutation(
        frame, permute_cols=["feature_a", "feature_b"], mappings=mappings
    )
    combined, actual_hash = build_and_apply_group_permutation(
        frame,
        permute_cols=["feature_a", "feature_b"],
        replicate_index=3,
        config_hash="cfg",
    )

    assert actual_hash == expected_hash
    assert combined.equals(explicit)
    assert combined.select("fixed").to_series().to_list() == [0, 1, 3, 2]


def test_mapping_rejects_duplicate_date_market_ticker_keys() -> None:
    frame = _frame().vstack(_frame().head(1))
    with pytest.raises(ValueError, match="duplicate"):
        build_group_permutation_mapping(frame, replicate_index=0, config_hash="cfg")
