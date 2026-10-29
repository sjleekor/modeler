"""Generated dictionaries only: no reports, historical prices, or label files."""

from copy import deepcopy

import pytest

from modeler.feedback_outcomes.calculation import aggregate_cohort, calculate_row, summarize

ENTRY = "2026-11-04"
EXIT = "2026-12-02"
CUTOFF = "2026-12-04T13:00:00+09:00"


def slot(rank=1, market="kospi", symbol="SYNTH0001"):
    return {
        "rank": rank,
        "market": market,
        "symbol": symbol,
        "cohort_id": "synthetic-c1",
        "report_sha256": "synthetic-report-hash",
        "rule_hash": "synthetic-rule-hash",
        "score": 0.75,
        "report_date": ENTRY,
        "verified_by": "2026-11-04T14:00:00+09:00",
    }


def observations(market="kospi"):
    def price(day, close):
        return {
            "adjusted_close": close,
            "source_open": 100,
            "source_high": 125,
            "source_low": 90,
            "source_close": close,
            "volume": 1000,
            "trading_value": 100000,
            "explicit_untradeable": False,
            "tradeability_conflict": False,
            "snapshot_id": "synthetic-snapshot",
            "adjustment_revision": "r1",
            "regular_session_corrected": True,
            "corporate_action": "none",
            "entity_id": "synthetic-entity-1",
            "listing_valid_from": "2020-01-01",
            "listing_valid_to": None,
            "market": market,
            "symbol": "SYNTH0001",
            "session": day,
            "available_at": day + "T18:00:00+09:00",
            "source_ref": "synthetic-memory-only",
        }

    def index(day, close):
        return {
            "close_idx": close,
            "index_group": market,
            "idx_nm": "코스피" if market == "kospi" else "코스닥",
            "session": day,
            "available_at": day + "T18:00:00+09:00",
            "source_ref": "synthetic-memory-only",
        }

    return {
        "entry": price(ENTRY, 100),
        "exit": price(EXIT, 110),
        "benchmark_entry": index(ENTRY, 200),
        "benchmark_exit": index(EXIT, 210),
        "identity_verified": True,
        "same_adjustment_vintage": True,
    }


def calculate(data=None, **kwargs):
    return calculate_row(
        slot(),
        entry_session=ENTRY,
        exit_session=EXIT,
        observations=observations() if data is None else data,
        cutoff=kwargs.pop("cutoff", CUTOFF),
        **kwargs,
    )


def rows_for(size=20, **kwargs):
    output = []
    for rank in range(1, size + 1):
        row = calculate(**kwargs)
        row["rank"] = rank
        row["symbol"] = f"SYNTH{rank:04d}"
        output.append(row)
    return output


def test_manual_return_arithmetic_and_original_provenance():
    original_slot, data = slot(), observations()
    original = deepcopy(data)
    result = calculate_row(
        original_slot, entry_session=ENTRY, exit_session=EXIT, observations=data, cutoff=CUTOFF
    )
    assert result["status"] == "available"
    assert result["gross"] == pytest.approx(110 / 100 - 1)
    assert result["net"] == pytest.approx(0.094)
    assert result["cost"] == 0.006
    assert result["benchmark"] == pytest.approx(0.05)
    assert result["excess"] == pytest.approx(0.044)
    assert all(result[key] == value for key, value in original_slot.items())
    assert result["observations"] == original
    assert data == original


@pytest.mark.parametrize("cutoff", ["2026-12-02T15:29:59+09:00", "2026-12-02T15:30:00+09:00"])
def test_immature_rows_do_not_calculate_even_with_supplied_future_values(cutoff):
    data = observations()
    data["entry"]["adjusted_close"] = 1
    data["exit"]["adjusted_close"] = 999999
    result = calculate(data, cutoff=cutoff)
    assert result["status"] == "pending"
    assert result["mature"] is False
    assert all(result[key] is None for key in ("gross", "net", "benchmark", "excess", "cost"))


def test_timezone_required_and_session_order_checked():
    with pytest.raises(ValueError, match="timezone-aware"):
        calculate(cutoff="2026-12-04T13:00:00")
    with pytest.raises(ValueError, match="follow"):
        calculate_row(
            slot(),
            entry_session=EXIT,
            exit_session=ENTRY,
            observations=observations(),
            cutoff=CUTOFF,
        )


def test_session_close_metadata_controls_maturity():
    result = calculate_row(
        slot(),
        entry_session={"date": ENTRY, "close_at": ENTRY + "T15:30:00+09:00"},
        exit_session={"session": EXIT, "close_at": EXIT + "T16:30:00+09:00"},
        observations=observations(),
        cutoff=EXIT + "T16:00:00+09:00",
    )
    assert result["status"] == "pending"


def test_explicit_halt_is_cash_while_benchmark_keeps_original_market_weight():
    data = observations()
    data["entry"]["explicit_untradeable"] = True
    data["entry"]["source_close"] = None
    data.pop("exit")
    result = calculate(data)
    assert result["status"] == "not_entered"
    assert (result["gross"], result["net"], result["cost"]) == (0, 0, 0)
    assert result["benchmark"] == pytest.approx(0.05)
    assert result["excess"] == pytest.approx(-0.05)
    assert result["executed"] is False


def test_withdrawal_is_cash_and_never_promotes_another_report():
    result = calculate({}, withdrawn_before_entry=True)
    assert result["status"] == "withdrawn_before_entry"
    assert result["net"] == 0
    assert result["cost"] == 0
    assert result["benchmark"] is None
    assert result["excess"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("volume", 0),
        ("volume", None),
        ("volume", True),
        ("trading_value", 0),
        ("source_open", float("nan")),
        ("source_high", float("inf")),
        ("source_low", -1),
        ("source_close", 0),
        ("source_close", "110"),
        ("source_close", 126),
        ("source_open", 89),
    ],
)
def test_invalid_entry_proxy_never_becomes_cash(field, value):
    data = observations()
    data["entry"][field] = value
    result = calculate(data)
    assert result["status"] == "entry_unresolved"
    assert result["net"] is None and result["cost"] is None


def test_locked_price_and_conflicting_halt_are_unresolved():
    data = observations()
    data["entry"].update(source_open=100, source_high=100, source_low=100, source_close=100)
    assert calculate(data)["status"] == "entry_unresolved"
    data = observations()
    data["entry"].update(explicit_untradeable=True, tradeability_conflict=True)
    assert calculate(data)["status"] == "entry_unresolved"


@pytest.mark.parametrize("field", ["explicit_untradeable", "volume", "source_close"])
def test_exit_halt_or_missing_price_stays_unknown_without_fill(field):
    data = observations()
    data["exit"][field] = True if field == "explicit_untradeable" else None
    result = calculate(data)
    assert result["status"] == "exit_unresolved"
    assert result["net"] is None
    assert result["entry_executed"] is True
    assert result["exit_executed"] is False


def test_absent_entry_exit_and_adapter_contract_failure_are_distinct():
    assert calculate({})["status"] == "entry_unresolved"
    data = observations()
    data.pop("exit")
    assert calculate(data)["status"] == "exit_unresolved"
    assert (
        calculate({"status": "source_missing", "reason": "metadata_missing"})["status"]
        == "source_missing"
    )


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        (None, "identity_verified", False),
        ("exit", "entity_id", "ticker-reused"),
        ("exit", "market", "kosdaq"),
        ("exit", "symbol", "OTHER"),
        ("entry", "listing_valid_from", "2027-01-01"),
        ("exit", "listing_valid_to", "2026-12-01"),
    ],
)
def test_identity_unknown_moved_or_reused_is_unresolved(location, field, value):
    data = observations()
    (data if location is None else data[location])[field] = value
    result = calculate(data)
    assert result["status"] == "identity_unresolved"
    assert result["net"] is None


@pytest.mark.parametrize("kind", ["split", "reverse_split"])
def test_split_and_reverse_split_use_verified_same_adjustment_chain(kind):
    data = observations()
    data["exit"]["corporate_action"] = kind
    # Adjusted prices already account for the split; never adjust a second time.
    assert calculate(data)["net"] == pytest.approx(0.094)
    data["exit"]["adjustment_revision"] = "r2"
    assert calculate(data)["status"] == "source_missing"


@pytest.mark.parametrize(
    "kind", ["merger", "capital_reduction", "spinoff", "delisting_cash_settlement"]
)
def test_other_corporate_actions_remain_unknown_even_if_prices_exist(kind):
    data = observations()
    data["exit"]["corporate_action"] = kind
    assert calculate(data)["status"] == "corporate_action_unresolved"


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [
        (None, "same_adjustment_vintage", False),
        ("exit", "snapshot_id", "different"),
        ("entry", "adjustment_revision", None),
        ("exit", "regular_session_corrected", False),
        ("entry", "adjusted_close", float("nan")),
        ("exit", "adjusted_close", False),
    ],
)
def test_adjustment_metadata_contract_is_required(location, field, value):
    data = observations()
    (data if location is None else data[location])[field] = value
    assert calculate(data)["status"] == "source_missing"


def test_corrected_snapshot_is_new_input_and_does_not_mutate_old_result():
    original = observations()
    first = calculate(original)
    correction = deepcopy(original)
    for key in ("entry", "exit"):
        correction[key].update(snapshot_id="synthetic-correction", adjustment_revision="r2")
    correction["exit"]["adjusted_close"] = 120
    second = calculate(correction)
    assert first["net"] == pytest.approx(0.094)
    assert second["net"] == pytest.approx(0.194)
    assert first["observations"]["exit"]["snapshot_id"] == "synthetic-snapshot"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("index_group", "kosdaq"),
        ("idx_nm", "코스피 총수익"),
        ("unique", False),
        ("close_idx", float("inf")),
        ("close_idx", 0),
        ("session", "2026-12-03"),
        ("available_at", "2027-01-01T00:00:00+09:00"),
    ],
)
def test_wrong_or_unknown_benchmark_has_no_fallback(field, value):
    data = observations()
    data["benchmark_exit"][field] = value
    result = calculate(data)
    assert result["net"] == pytest.approx(0.094)
    assert result["benchmark"] is None and result["excess"] is None


def test_kosdaq_uses_exact_kosdaq_price_index():
    result = calculate_row(
        slot(market="kosdaq"),
        entry_session=ENTRY,
        exit_session=EXIT,
        observations=observations("kosdaq"),
        cutoff=CUTOFF,
    )
    assert result["benchmark"] == pytest.approx(0.05)


def test_original_equal_weight_includes_cash_slot_and_fixed_cost():
    rows = rows_for()
    cash = calculate(withdrawn_before_entry=True)
    cash["rank"] = 20
    rows[-1] = cash
    result = aggregate_cohort({"top_n": 20, "cohort_id": "synthetic-c1"}, rows)
    assert result["status"] == "available"
    assert result["net"] == pytest.approx(19 / 20 * 0.094)
    assert result["cost"] == pytest.approx(19 / 20 * 0.006)
    assert result["benchmark"] == pytest.approx(0.05)
    assert result["excess"] == pytest.approx(19 / 20 * 0.094 - 0.05)
    assert result["nominal_buy_weight"] == 0.95
    assert result["nominal_sell_weight"] == 0.95


def test_unresolved_slot_prevents_survivor_mean_and_preserves_original_weight():
    rows = rows_for()
    rows[-1].update(net=None, gross=None, cost=None, status="exit_unresolved")
    result = aggregate_cohort({"size": 20}, rows)
    assert result["net"] is None and result["excess"] is None
    assert result["benchmark"] is None
    assert result["gross"] is None and result["cost"] is None
    assert result["unresolved_weight"] == 0.05
    assert result["status"] == "unresolved"


def test_cash_with_unknown_benchmark_does_not_claim_available_cohort():
    rows = rows_for(withdrawn_before_entry=True)
    rows[0].update(benchmark=None, excess=None)
    result = aggregate_cohort({"size": 20}, rows)
    assert result["net"] is None
    assert result["benchmark"] is None and result["excess"] is None
    assert result["status"] == "unresolved"
    assert result["unresolved_weight"] == 0.05
    assert result["resolved_slot_count"] == 19


@pytest.mark.parametrize(
    "ranks", [list(range(1, 20)), list(range(1, 20)) + [19], list(range(2, 22))]
)
def test_incomplete_original_slots_never_shrink_or_promote(ranks):
    rows = rows_for()
    for row, rank in zip(rows, ranks, strict=False):
        row["rank"] = rank
    rows = rows[: len(ranks)]
    result = aggregate_cohort({"size": 20}, rows)
    assert result["status"] == "source_incomplete"
    assert result["net"] is None


def test_top100_secondary_and_pending_cohort():
    result = aggregate_cohort({"size": 100}, rows_for(100))
    assert result["net"] == pytest.approx(0.094)
    pending = aggregate_cohort({"size": 20}, rows_for(cutoff=ENTRY + "T18:00:00+09:00"))
    assert pending["status"] == "pending"
    assert pending["net"] is None
    with pytest.raises(ValueError, match="top20 or top100"):
        aggregate_cohort({"size": 19}, [])


def test_summary_conditional_means_counts_repeat_symbols_and_small_sample():
    rows = rows_for()
    available = aggregate_cohort({"size": 20}, rows)
    unresolved = dict(available, status="unresolved", net=None, excess=None, unresolved_weight=0.05)
    pending = dict(
        available, status="pending", mature=False, net=None, excess=None, unresolved_weight=1
    )
    secondary = aggregate_cohort({"size": 100}, rows_for(100))
    repeated = dict(rows[0], report_sha256="synthetic-second-report-hash")
    result = summarize([available, unresolved, pending, secondary], rows + [repeated])
    top20 = result["top20"]
    assert top20["role"] == "primary"
    assert result["top100"]["role"] == "secondary"
    assert top20["total_cohort_count"] == 3
    assert top20["mature_cohort_count"] == 2
    assert top20["resolved_cohort_count"] == 1
    assert top20["unresolved_cohort_count"] == 2
    assert top20["unresolved_weight"] == pytest.approx(1.05)
    assert top20["mean_net"] == pytest.approx(0.094)
    assert top20["mean_excess"] == pytest.approx(0.044)
    assert top20["insufficient_sample"] is True
    assert result["unique_symbol_count"] == 20
    assert result["repeated_recommendation_count"] == 1
    assert result["repeated_symbol_count"] == 1
    assert result["publication_count"] == 2
    assert "not independent" in result["interpretation"]
    assert "Conditional" in result["interpretation"]
    assert not any(
        key in str(result).lower() for key in ("sharpe", "annualized", "significance", "adoption")
    )


def test_insufficient_sample_uses_fixed_spec_mature_count_not_independent_samples():
    available = aggregate_cohort({"size": 20}, rows_for())
    unresolved = dict(available, status="unresolved", net=None, excess=None, unresolved_weight=0.1)
    result = summarize([available] + [unresolved] * 19, [])
    assert result["top20"]["insufficient_sample"] is False
    assert result["top20"]["resolved_cohort_count"] == 1
    assert result["top20"]["sample_status"] == "descriptive_only"


def test_empty_summary_has_null_means_instead_of_zero():
    result = summarize([], [])
    assert result["top20"]["mean_net"] is None
    assert result["top100"]["mean_excess"] is None
    assert result["top20"]["insufficient_sample"] is True


def test_f3_market_is_preserved_and_security_market_selects_identity_and_index():
    original = dict(slot(), market="KR", security_market="KOSPI")
    result = calculate_row(
        original, entry_session=ENTRY, exit_session=EXIT, observations=observations(), cutoff=CUTOFF
    )
    assert result["market"] == "KR"
    assert result["security_market"] == "KOSPI"
    assert result["status"] == "available"
    assert result["benchmark"] == pytest.approx(0.05)
    original["security_market"] = "KOSDAQ"
    wrong = calculate_row(
        original, entry_session=ENTRY, exit_session=EXIT, observations=observations(), cutoff=CUTOFF
    )
    assert wrong["status"] == "identity_unresolved"
    assert wrong["benchmark"] is None


def test_other_securitys_halt_never_turns_into_cash():
    data = observations()
    data["entry"].update(explicit_untradeable=True, symbol="OTHER")
    result = calculate(data)
    assert result["status"] == "identity_unresolved"
    assert result["net"] is None
    assert result["entry_executed"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("regular_session_corrected", False),
        ("snapshot_id", None),
        ("adjustment_revision", None),
        ("adjusted_close", None),
        ("adjusted_close", 0),
        ("adjusted_close", float("nan")),
    ],
)
def test_missing_entry_price_proof_cannot_record_a_buy(key, value):
    data = observations()
    data["entry"][key] = value
    result = calculate(data)
    assert result["status"] == "source_missing"
    assert result["entry_executed"] is False
    assert result["exit_executed"] is False


@pytest.mark.parametrize(
    "close",
    [
        "2026-12-01T15:30:00+09:00",
        "2026-12-03T15:30:00+09:00",
        "2026-12-02T08:59:59+09:00",
        "2026-12-02T18:00:01+09:00",
        "2026-12-02T15:30:00",
    ],
)
def test_override_close_cannot_change_session_date_or_regular_session_window(close):
    with pytest.raises(ValueError):
        calculate_row(
            slot(),
            entry_session=ENTRY,
            exit_session={"session": EXIT, "close_at": close},
            observations=observations(),
            cutoff=CUTOFF,
        )


@pytest.mark.parametrize("bad_day", ["20261202", "2026-12-02T15:30:00+09:00"])
def test_session_requires_canonical_date(bad_day):
    with pytest.raises(ValueError):
        calculate_row(
            slot(),
            entry_session=ENTRY,
            exit_session=bad_day,
            observations=observations(),
            cutoff=CUTOFF,
        )


@pytest.mark.parametrize("bad_rank", [True, 1.0, "1", None, 0, 101])
def test_non_original_or_noninteger_rank_cannot_fill_missing_slot(bad_rank):
    rows = rows_for()
    rows[0]["rank"] = bad_rank
    assert aggregate_cohort({"size": 20}, rows)["status"] == "source_incomplete"


def test_summary_does_not_count_same_report_in_top20_and_top100_twice():
    rows = rows_for()
    available20 = aggregate_cohort({"size": 20, "cohort_id": "report-top20"}, rows)
    available100 = aggregate_cohort({"size": 100, "cohort_id": "report-top100"}, rows_for(100))
    # The same original prediction is intentionally supplied from two cohort views.
    result = summarize([available20, available100], rows + rows)
    assert result["row_count"] == 40
    assert result["recommendation_count"] == 20
    assert result["publication_count"] == 1
    assert result["repeated_recommendation_count"] == 0
    assert result["top20"]["mature_cohort_count"] == 1
    assert result["top100"]["mature_cohort_count"] == 1
    assert result["top20"]["insufficient_sample"] is True


def test_summary_publication_ref_fallback_and_zero_counts_remain_known():
    row = slot()
    row.pop("report_sha256")
    row["publication_ref"] = "synthetic-publication-receipt"
    result = summarize([], [row, dict(row, cohort_id="other-view")])
    assert result["publication_count"] == 1
    assert result["recommendation_count"] == 1
    assert result["repeated_recommendation_count"] == 0
    assert result["top20"]["total_cohort_count"] == 0
    assert result["top20"]["unresolved_weight"] == 0
