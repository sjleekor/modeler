import copy

import pytest

from modeler.feedback_outcomes.contracts import OutcomeError
from modeler.feedback_outcomes.prices import EndpointExtract
from modeler.feedback_outcomes.synthetic import fixture


def case():
    data = fixture()
    slot = data["events"][0]["rows"][0]
    return data, slot, "2026-11-04", "2026-12-02"


def test_exact_regular_adjusted_endpoint_and_index_mapping():
    data, slot, entry, exit_ = case()
    adapter = EndpointExtract(data["extract"], cutoff=data["cutoff"], synthetic=True)
    observed = adapter.observations(slot, entry, exit_)
    assert observed["entry"]["adjusted_close"] == 100
    assert observed["exit"]["adjusted_close"] == 110
    assert observed["benchmark_entry"]["index_group"] == "kospi"
    assert observed["same_adjustment_vintage"] is True
    assert observed["identity_verified"] is True


@pytest.mark.parametrize(
    "key", ["regular_close_binding_sha256", "adjustment_code_sha256", "identity_registry_sha256"]
)
def test_missing_proof_never_promotes_existing_close(key):
    data, slot, entry, exit_ = case()
    del data["extract"]["proof"][key]
    adapter = EndpointExtract(data["extract"], cutoff=data["cutoff"], synthetic=True)
    observed = adapter.observations(slot, entry, exit_)
    assert observed["same_adjustment_vintage"] is False
    assert observed["entry"]["regular_session_corrected"] is False


def test_regular_krx_close_factor_must_bind_adjusted_close():
    data, slot, entry, exit_ = case()
    data["extract"]["daily_ohlcv"][0]["close"] = 200
    adapter = EndpointExtract(data["extract"], cutoff=data["cutoff"], synthetic=True)
    assert adapter.observations(slot, entry, exit_)["entry"]["regular_session_corrected"] is False


@pytest.mark.parametrize("table", ["daily_ohlcv", "daily_market_cap", "krx_index_daily"])
def test_duplicate_source_grain_is_rejected(table):
    data, *_ = case()
    data["extract"][table].append(copy.deepcopy(data["extract"][table][0]))
    with pytest.raises(OutcomeError, match="duplicate"):
        EndpointExtract(data["extract"], cutoff=data["cutoff"], synthetic=True)


def test_unavailable_and_out_of_plan_sessions_are_rejected():
    data, *_ = case()
    data["extract"]["daily_ohlcv"][0]["available_at"] = "2027-01-01T16:00:00+09:00"
    with pytest.raises(OutcomeError, match="unavailable"):
        EndpointExtract(data["extract"], cutoff=data["cutoff"], synthetic=True)
    data, *_ = case()
    with pytest.raises(OutcomeError, match="outside the read plan"):
        EndpointExtract(
            data["extract"], cutoff=data["cutoff"], synthetic=True, sessions=["2026-11-04"]
        )


def test_ticker_reuse_or_market_transfer_is_unresolved():
    data, slot, entry, exit_ = case()
    second = copy.deepcopy(data["extract"]["listing_periods"][0])
    second.update(entity_id="another-synthetic-entity", valid_from="2026-11-20", market="KOSDAQ")
    data["extract"]["listing_periods"].append(second)
    observed = EndpointExtract(data["extract"], cutoff=data["cutoff"], synthetic=True).observations(
        slot, entry, exit_
    )
    assert observed["identity_verified"] is True
    assert observed["exit"]["entity_id"] is None


def test_real_and_fixture_inputs_never_mix():
    data, *_ = case()
    with pytest.raises(OutcomeError, match="cannot be mixed"):
        EndpointExtract(data["extract"], cutoff=data["cutoff"], synthetic=False)


def test_same_row_override_vintage_cannot_override_the_bound_snapshot():
    data, slot, entry, exit_ = case()
    for price in data["extract"]["daily_ohlcv"]:
        price["snapshot_id"] = "unbound-other-snapshot"
    adapter = EndpointExtract(data["extract"], cutoff=data["cutoff"], synthetic=True)
    result = adapter.observations(slot, entry, exit_)
    assert result["entry"]["regular_session_corrected"] is False
    assert result["exit"]["regular_session_corrected"] is False
