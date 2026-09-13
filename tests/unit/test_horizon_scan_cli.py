from __future__ import annotations

from modeler.analysis.horizon_scan import (
    delay_gate_required,
    filter_registry_to_family,
    scan_kwargs_from_config,
)
from modeler.analysis.horizon_scan_config import load_config


def test_filter_registry_to_family_returns_everything_when_family_is_none() -> None:
    registry = [{"family": "a"}, {"family": "b"}]
    assert filter_registry_to_family(registry, None) == registry


def test_filter_registry_to_family_restricts_to_the_named_family() -> None:
    registry = [
        {"family": "a", "h_end": 5},
        {"family": "b", "h_end": 10},
        {"family": "a", "h_end": 20},
    ]
    result = filter_registry_to_family(registry, "a")
    assert result == [{"family": "a", "h_end": 5}, {"family": "a", "h_end": 20}]


def test_filter_registry_to_family_returns_empty_for_an_unknown_family() -> None:
    assert filter_registry_to_family([{"family": "a"}], "does-not-exist") == []


def test_scan_kwargs_from_config_reads_the_real_config() -> None:
    config = load_config()
    kwargs = scan_kwargs_from_config(config)
    assert kwargs["sample_start"] == str(config.raw["sample"]["start"])
    assert kwargs["min_names"] == int(config.raw["stats"]["min_names_per_date_market"])
    assert kwargs["min_names_for_spread"] == int(config.raw["stats"]["min_names_for_spread"])
    assert kwargs["quantile_count"] == int(config.raw["stats"]["quantile_count"])
    assert kwargs["min_dates_per_cell"] == int(config.raw["stats"]["min_dates_per_cell"])
    assert kwargs["scan_engine"] == "polars_native_v1"


def test_delay_gate_required_for_cumulative_horizons() -> None:
    assert delay_gate_required({"scan_type": "cum", "h_start": 0, "h_end": 5}) is True
    assert delay_gate_required({"scan_type": "cum", "h_start": 0, "h_end": 3}) is True
    assert delay_gate_required({"scan_type": "cum", "h_start": 0, "h_end": 10}) is False


def test_delay_gate_required_only_for_the_0_5_bucket() -> None:
    assert delay_gate_required({"scan_type": "bucket", "h_start": 0, "h_end": 5}) is True
    assert delay_gate_required({"scan_type": "bucket", "h_start": 5, "h_end": 10}) is False
