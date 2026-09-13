from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from modeler.analysis.horizon_scan_config import HorizonScanConfig, load_config
from modeler.analysis.horizon_scan_readiness import (
    build_primary_hypothesis_registry,
    build_short_exploratory_registry,
)

_BUCKETS = [[0, 5], [5, 10], [10, 20], [20, 40], [40, 60], [60, 120]]


def _synthetic_config(*, horizon_count: int, role: str, fdr_include: bool) -> HorizonScanConfig:
    """A minimal one-family config isolating the registry size check from any
    real family's bucket-grid alignment: ``include_bucket_primary=False``
    means the cell count is exactly ``horizon_count``, whatever the values."""
    family = {
        "family": "synthetic_family",
        "fdr_family": "synthetic",
        "role": role,
        "expected_sign": "+",
        "features": [{"column": "synthetic_feature", "role": "primary"}],
        "primary_horizon_set": list(range(1, horizon_count + 1)),
        "include_bucket_primary": False,
        "fdr_include": fdr_include,
        "official_feature_variant": "native_t",
        "variant_columns": {
            "native_t": "synthetic_feature", "lag1": "synthetic_feature_lag1",
        },
    }
    raw = {"families": [family], "buckets": _BUCKETS}
    return HorizonScanConfig(raw=raw, config_hash="test", path=Path("."))


def test_primary_registry_has_75_unique_cells_with_expected_id_format() -> None:
    config = load_config()
    rows = build_primary_hypothesis_registry(config)
    assert len(rows) == 75
    ids = [row["hypothesis_id"] for row in rows]
    assert len(set(ids)) == 75
    assert all(row["hypothesis_role"] == "primary" for row in rows)

    reversal_cum = next(
        r
        for r in rows
        if r["family"] == "px_reversal_5d" and r["scan_type"] == "cum" and r["h_end"] == 5
    )
    assert reversal_cum["hypothesis_id"] == "px_reversal_5d|px_reversal_5d|cum|0|5"
    reversal_bucket = next(
        r
        for r in rows
        if r["family"] == "px_reversal_5d" and r["scan_type"] == "bucket" and r["h_end"] == 10
    )
    assert reversal_bucket["hypothesis_id"] == "px_reversal_5d|px_reversal_5d|bucket|5|10"

    # px_reversal_5d's cumulative {1,2,3,5,10} only meets the bucket grid at
    # {5,10}: 5 cum + 2 bucket, not 5 cum + 5 bucket.
    reversal_rows = [r for r in rows if r["family"] == "px_reversal_5d"]
    assert len(reversal_rows) == 7
    assert sum(1 for r in reversal_rows if r["scan_type"] == "bucket") == 2


def test_primary_registry_scans_the_official_feature_variant_not_native() -> None:
    # flow_foreign_netbuy_to_volume's official_feature_variant is "lag1" (§1.1
    # condition 7: same-day-unverified flow uses lag1 as the frozen official
    # variant) — the registry's scanned "feature" column must be the lag1
    # variant, even though hypothesis_id still names the family by its native
    # column for stable identity.
    config = load_config()
    rows = build_primary_hypothesis_registry(config)
    flow_rows = [r for r in rows if r["family"] == "flow_foreign_netbuy_to_volume"]
    assert flow_rows
    assert all(r["feature"] == "flow_foreign_netbuy_to_volume_20d_lag1" for r in flow_rows)
    assert all(r["feature_variant"] == "lag1" for r in flow_rows)
    assert all(
        r["hypothesis_id"].startswith("flow_foreign_netbuy_to_volume|"
        "flow_foreign_netbuy_to_volume_20d|")
        for r in flow_rows
    )

    # px_reversal_5d's official variant is native_t, so feature == the native
    # column exactly as before this fix.
    reversal_rows = [r for r in rows if r["family"] == "px_reversal_5d"]
    assert all(r["feature"] == "px_reversal_5d" for r in reversal_rows)


def test_short_exploratory_registry_has_28_cells_disjoint_from_primary() -> None:
    config = load_config()
    primary = build_primary_hypothesis_registry(config)
    short = build_short_exploratory_registry(config)
    assert len(short) == 28
    assert all(r["hypothesis_role"] == "exploratory_short_regime" for r in short)
    assert set(r["hypothesis_id"] for r in primary).isdisjoint(r["hypothesis_id"] for r in short)
    short_families = {r["family"] for r in short}
    assert short_families == {
        "flow_short_turnover",
        "flow_short_interest",
        "flow_days_to_cover",
        "flow_nat_proxy_20d",
    }


@pytest.mark.parametrize("horizon_count", [74, 76])
def test_primary_registry_rejects_74_or_76_cells(horizon_count: int) -> None:
    config = _synthetic_config(horizon_count=horizon_count, role="ready", fdr_include=True)
    with pytest.raises(ValueError, match=f"must have 75 cells, got {horizon_count}"):
        build_primary_hypothesis_registry(config)


@pytest.mark.parametrize("horizon_count", [27, 29])
def test_short_exploratory_registry_rejects_27_or_29_cells(horizon_count: int) -> None:
    config = _synthetic_config(
        horizon_count=horizon_count, role="exploratory_short_regime", fdr_include=False
    )
    with pytest.raises(ValueError, match=f"must have 28 cells, got {horizon_count}"):
        build_short_exploratory_registry(config)


def test_hypothesis_registry_rejects_duplicate_ids() -> None:
    config = load_config()
    raw = deepcopy(config.raw)
    reversal = next(f for f in raw["families"] if f["family"] == "px_reversal_5d")
    raw["families"].append(deepcopy(reversal))
    dup_config = HorizonScanConfig(raw=raw, config_hash="test", path=config.path)
    with pytest.raises(ValueError, match="duplicate ids"):
        build_primary_hypothesis_registry(dup_config)
