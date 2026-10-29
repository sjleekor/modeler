"""Fixtures are generated from the protocol grammar, never captured reports/prices."""

from copy import deepcopy
from datetime import date, timedelta

import pytest

from modeler.feedback.core import identity
from modeler.feedback_outcomes.selection import BUNDLE, Calendar, entry_exit, select_cohorts


def manifest():
    start, end = date(2026, 10, 20), date(2027, 3, 10)
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return {
        "market": "KR",
        "timezone": "Asia/Seoul",
        "coverage_start": start.isoformat(),
        "coverage_end": end.isoformat(),
        "sessions": [d.isoformat() for d in days if d.weekday() < 5],
        "default_open_at": "09:00:00",
        "default_close_at": "15:30:00",
        "overrides": [],
        "unconfirmed_dates": [],
        "calendar_sha256": "c" * 64,
        "calendar_code_version": "synthetic-test-v1",
        "synthetic_fixture": True,
    }


@pytest.fixture
def calendar():
    return Calendar.from_manifest(manifest())


def publication(**changes):
    event = {
        "kind": "publication",
        "market": "KR",
        "model_id": "kr_daily_h20_v1",
        "model_version": "1.0.0",
        "bundle_hash": BUNDLE,
        "input_sha256": "a" * 64,
        "feature_asof_date": "2026-11-03",
        "freshness_lag_sessions": 0,
        "model_status": "partial",
        "report_date": "2026-11-04",
        "decision_at": "2026-11-04T10:00:00+09:00",
        "verified_by": "2026-11-04T10:05:00+09:00",
        "collection_started_at": "2026-11-04T10:01:00+09:00",
        "coverage_status": "covered",
        "publication_ref": "1" * 64,
        "receipt_ref": "2" * 64,
        "report_sha256": "b" * 64,
        "invocation_id": "synthetic-one",
        "reports_commit": "d" * 40,
        "unit_sha256": "e" * 64,
        "revision": 1,
        "synthetic_fixture": True,
        "historical_replay": False,
        "rows": [
            {
                "prediction_ref": f"synthetic-prediction-{i}",
                "symbol": f"{i:06d}",
                "security_market": "KOSPI" if i % 2 else "KOSDAQ",
                "rank": i,
                "score": i / 1000,
                "quality_reasons": ["synthetic-warning"] if i == 1 else [],
            }
            for i in range(1, 101)
        ],
    }
    event.update(changes)
    for row in event["rows"]:
        row["prediction_ref"] = identity(
            "prediction",
            {
                "market": "KR",
                "security_market": row["security_market"],
                "model_id": event["model_id"],
                "bundle_hash": event["bundle_hash"],
                "report_sha256": event["report_sha256"],
                "invocation_id": event["invocation_id"],
                "symbol": row["symbol"],
            },
        )
    return event


def choose(events, calendar):
    return select_cohorts(events, calendar, synthetic=True)


@pytest.mark.parametrize(
    "observed, expected",
    [
        ("2026-11-04T14:29:59+09:00", "2026-11-04"),
        ("2026-11-04T14:30:00+09:00", "2026-11-04"),
        ("2026-11-04T14:30:00.000001+09:00", "2026-11-05"),
        ("2026-11-04T05:30:00Z", "2026-11-04"),
        ("2026-11-06T23:00:00+09:00", "2026-11-09"),
        ("2026-11-07T09:00:00+09:00", "2026-11-09"),
    ],
)
def test_entry_cutoff_and_observation_day(calendar, observed, expected):
    assert entry_exit("2026-11-04", observed, calendar)["entry_session"] == expected


def test_h20_is_twenty_following_sessions_not_twenty_observations(calendar):
    schedule = entry_exit("2026-11-04", "2026-11-04T14:30:00+09:00", calendar)
    assert schedule == {
        "status": "scheduled",
        "entry_session": "2026-11-04",
        "exit_session": "2026-12-02",
        "reasons": [],
    }


def test_explicit_holiday_skipped_and_unconfirmed_session_pending():
    data = manifest()
    data["sessions"].remove("2026-11-05")
    cal = Calendar.from_manifest(data)
    result = entry_exit("2026-11-04", "2026-11-04T14:30:01+09:00", cal)
    assert (result["entry_session"], result["exit_session"]) == ("2026-11-06", "2026-12-04")
    data["unconfirmed_dates"] = ["2026-11-06"]
    result = entry_exit("2026-11-04", "2026-11-04T14:30:01+09:00", Calendar.from_manifest(data))
    assert result["status"] == "pending"
    assert result["entry_session"] is None


def test_unknown_exit_and_coverage_never_invent_days():
    data = manifest()
    data["coverage_end"] = "2026-11-10"
    data["sessions"] = [s for s in data["sessions"] if s <= data["coverage_end"]]
    cal = Calendar.from_manifest(data)
    result = entry_exit("2026-11-04", "2026-11-04T14:30:00+09:00", cal)
    assert result["entry_session"] == "2026-11-04"
    assert result["exit_session"] is None
    assert result["reasons"] == ["exit_session_unknown"]
    assert entry_exit("2026-11-11", "2026-11-11T10:01:00+09:00", cal)["reasons"] == [
        "calendar_coverage_unknown"
    ]


def test_2027_provisional_holidays_remain_pending_despite_regular_confirmed_hours():
    data = manifest()
    data["synthetic_fixture"] = False
    data["source"] = {
        "holiday_source_status": (
            "project_derived_integrity_pinned_2026_primary_crosschecked_"
            "2027_provisional_pre_notice"
        )
    }
    cal = Calendar.from_manifest(data)
    assert cal.calendar.session(date(2027, 1, 5)).confirmed
    result = entry_exit("2027-01-05", "2027-01-05T10:05:00+09:00", cal)
    assert result["status"] == "pending"
    assert result["entry_session"] is None
    assert result["reasons"] == ["calendar_holiday_provisional"]


def test_2026_entry_keeps_entry_when_exit_crosses_into_provisional_year():
    data = manifest()
    data["synthetic_fixture"] = False
    data["source"] = {"holiday_source_status": "2027_provisional_pre_notice"}
    result = entry_exit("2026-12-21", "2026-12-21T10:05:00+09:00", Calendar.from_manifest(data))
    assert result["entry_session"] == "2026-12-21"
    assert result["exit_session"] is None
    assert result["reasons"] == ["calendar_holiday_provisional"]


def test_explicit_confirmed_through_controls_future_holiday_coverage():
    data = manifest()
    data["synthetic_fixture"] = False
    result = entry_exit("2027-01-05", "2027-01-05T10:05:00+09:00", Calendar.from_manifest(data))
    assert result["reasons"] == ["calendar_holiday_coverage_unconfirmed"]
    data["source"] = {"confirmed_through": "2027-02-05"}
    result = entry_exit("2027-01-05", "2027-01-05T10:05:00+09:00", Calendar.from_manifest(data))
    assert result["status"] == "scheduled"
    data["source"]["holiday_source_status"] = "2027_provisional_pre_notice"
    result = entry_exit("2027-01-05", "2027-01-05T10:05:00+09:00", Calendar.from_manifest(data))
    assert result["reasons"] == ["calendar_holiday_provisional"]


def test_synthetic_calendar_only_bypasses_holiday_uncertainty_not_hours():
    data = manifest()
    data["source"] = {"holiday_source_status": "2027_provisional_pre_notice"}
    assert (
        entry_exit("2027-01-05", "2027-01-05T10:05:00+09:00", Calendar.from_manifest(data))[
            "status"
        ]
        == "scheduled"
    )
    data["unconfirmed_dates"] = ["2027-01-05"]
    result = entry_exit("2027-01-05", "2027-01-05T10:05:00+09:00", Calendar.from_manifest(data))
    assert result["status"] == "pending"
    assert result["reasons"] == ["calendar_hours_unconfirmed"]


@pytest.mark.parametrize(
    "update",
    [
        {"market": "US"},
        {"timezone": "UTC"},
        {"sessions": ["20261104"]},
        {"sessions": ["2026-11-04", "2026-11-04"]},
        {"calendar_sha256": "../sealed"},
    ],
)
def test_calendar_rejects_noncanonical_or_wrong_market_metadata(update):
    data = manifest()
    data.update(update)
    with pytest.raises(ValueError):
        Calendar.from_manifest(data)


def test_first_valid_report_and_corrections_preserve_slots(calendar):
    original = publication()
    wording = publication(verified_by="2026-11-04T10:06:00+09:00", revision=2, unit_sha256="f" * 64)
    corrected = publication(
        report_sha256="f" * 64,
        invocation_id="synthetic-two",
        verified_by="2026-11-04T10:07:00+09:00",
    )
    result = choose([corrected, wording, original], calendar)
    assert len(result["cohorts"]) == 2
    assert {c["report_sha256"] for c in result["cohorts"]} == {original["report_sha256"]}
    assert [d["status"] for d in result["decisions"]] == ["correction", "duplicate", "selected"]
    assert result["cohorts"][0]["slots"][0]["quality_reasons"] == ["synthetic-warning"]


def test_invalid_report_does_not_take_first_slot(calendar):
    invalid = publication(model_status="failed", verified_by="2026-11-04T10:04:00+09:00")
    result = choose([invalid, publication()], calendar)
    assert len(result["cohorts"]) == 2
    assert result["decisions"][0]["status"] == "excluded"
    assert result["decisions"][1]["status"] == "selected"


def test_ambiguous_simultaneous_reports_hold_entire_date(calendar):
    result = choose(
        [
            publication(),
            publication(report_sha256="f" * 64),
            publication(report_sha256="1" * 64, verified_by="2026-11-04T10:08:00+09:00"),
        ],
        calendar,
    )
    assert not result["cohorts"]
    assert all(d["reasons"] == ["ambiguous_first_publication"] for d in result["decisions"])


def test_exact_same_observation_is_one_sample(calendar):
    result = choose([publication(), publication()], calendar)
    assert len(result["cohorts"]) == 2
    assert [d["status"] for d in result["decisions"]] == ["selected", "duplicate"]


def test_same_hash_with_different_invocation_is_rejected(calendar):
    result = choose([publication(), publication(invocation_id="synthetic-conflicting")], calendar)
    assert not result["cohorts"]
    assert all(d["reasons"] == ["report_binding_conflict"] for d in result["decisions"])


def test_same_prediction_input_has_only_first_valid_publication(calendar):
    saturday = publication(
        report_date="2026-11-07",
        feature_asof_date="2026-11-06",
        decision_at="2026-11-07T10:00:00+09:00",
        verified_by="2026-11-07T10:05:00+09:00",
        collection_started_at="2026-11-07T10:01:00+09:00",
    )
    sunday = publication(
        report_date="2026-11-08",
        feature_asof_date="2026-11-06",
        decision_at="2026-11-08T10:00:00+09:00",
        verified_by="2026-11-08T10:05:00+09:00",
        collection_started_at="2026-11-08T10:01:00+09:00",
        report_sha256="f" * 64,
    )
    result = choose([sunday, saturday], calendar)
    assert len(result["cohorts"]) == 2
    assert result["decisions"][0]["reasons"] == ["same_prediction_input"]
    assert all(c["entry_session"] == "2026-11-09" for c in result["cohorts"])


def test_repeated_symbol_at_new_signal_session_is_separate_cohort(calendar):
    next_day = publication(
        report_date="2026-11-05",
        feature_asof_date="2026-11-04",
        report_sha256="f" * 64,
        decision_at="2026-11-05T10:00:00+09:00",
        verified_by="2026-11-05T10:05:00+09:00",
        collection_started_at="2026-11-05T10:01:00+09:00",
    )
    result = choose([publication(), next_day], calendar)
    assert len(result["cohorts"]) == 4
    assert len({c["cohort_ref"] for c in result["cohorts"]}) == 4


@pytest.mark.parametrize(
    "field,value",
    [
        ("score", float("nan")),
        ("rank", True),
        ("symbol", "not-a-security"),
        ("quality_reasons", "warning"),
    ],
)
def test_invalid_slot_metadata_rejects_publication(calendar, field, value):
    event = publication()
    event["rows"][0][field] = value
    result = choose([event], calendar)
    assert not result["cohorts"]
    assert result["decisions"][0]["reasons"] == ["invalid_published_row"]


def test_incomplete_sources_keep_original_ranks(calendar):
    event = publication()
    event["rows"] = [r for r in event["rows"] if r["rank"] not in (3, 25)]
    result = choose([event], calendar)
    top20, top100 = result["cohorts"]
    assert top20["missing_ranks"] == [3]
    assert top100["missing_ranks"] == [3, 25]
    assert all(c["status"] == "source_incomplete" for c in result["cohorts"])
    assert [r["rank"] for r in top20["slots"]] == [1, 2, *range(4, 21)]


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"market": "US"}, "sealed_model_or_bundle"),
        ({"model_id": "model02"}, "sealed_model_or_bundle"),
        ({"bundle_hash": "f" * 64}, "sealed_model_or_bundle"),
        ({"verified_by": "2026-11-04T09:59:59+09:00"}, "verified_before_decision"),
        ({"verified_by": "2026-11-04T10:05:00"}, "invalid_observation_timestamp"),
        ({"report_date": "2026-11-03"}, "outside_report_window"),
        ({"feature_asof_date": "2026-11-02"}, "stale_signal"),
        ({"freshness_lag_sessions": 1}, "stale_signal"),
        ({"freshness_lag_sessions": False}, "stale_signal"),
        ({"coverage_status": "unknown"}, "publication_unverified"),
        ({"historical_replay": True}, "historical_replay"),
        ({"report_sha256": "../P6/sealed"}, "invalid_report_sha256"),
        ({"rows": []}, "rankings_not_published"),
    ],
)
def test_fail_closed_publication_contract(calendar, change, reason):
    result = choose([publication(**change)], calendar)
    assert not result["cohorts"]
    assert reason in result["decisions"][0]["reasons"]


def test_synthetic_flags_are_explicit_and_real_mode_excludes_fixture(calendar):
    assert not select_cohorts([publication()], calendar)["cohorts"]
    assert not choose([publication(synthetic_fixture=False)], calendar)["cohorts"]
    assert all(c["synthetic_fixture"] for c in choose([publication()], calendar)["cohorts"])


@pytest.mark.parametrize(
    "field,value",
    [
        ("publication_ref", "synthetic://unbound"),
        ("publication_ref", 123),
        ("receipt_ref", "a" * 63),
        ("receipt_ref", "A" * 64),
        ("revision", -1),
        ("revision", 0),
        ("revision", "1"),
        ("revision", True),
    ],
)
def test_source_refs_and_revision_require_frozen_evidence_format(calendar, field, value):
    result = choose([publication(**{field: value})], calendar)
    assert not result["cohorts"]
    assert f"invalid_{field}" in result["decisions"][0]["reasons"]


@pytest.mark.parametrize(
    "value,reason",
    [
        ("synthetic://prediction", "invalid_published_row"),
        ("0" * 64, "prediction_identity_mismatch"),
    ],
)
def test_prediction_ref_is_hash_and_binds_original_identity(calendar, value, reason):
    event = publication()
    event["rows"][0]["prediction_ref"] = value
    result = choose([event], calendar)
    assert not result["cohorts"]
    assert reason in result["decisions"][0]["reasons"]


def test_security_market_requires_canonical_f3_identity(calendar):
    event = publication()
    event["rows"][0]["security_market"] = "kospi"
    result = choose([event], calendar)
    assert not result["cohorts"]
    assert result["decisions"][0]["reasons"] == ["invalid_published_row"]


def withdrawal(**changes):
    event = publication(
        kind="withdrawal",
        original_report_sha256="b" * 64,
        verified_by="2026-11-04T15:29:59+09:00",
        publication_ref="3" * 64,
    )
    event.update(changes)
    return event


def test_withdrawal_before_entry_close_marks_all_original_slots_cash(calendar):
    original = publication()
    corrected = publication(report_sha256="f" * 64, verified_by="2026-11-04T16:00:00+09:00")
    result = choose([withdrawal(), original, corrected], calendar)
    assert len(result["cohorts"]) == 2
    for cohort in result["cohorts"]:
        assert cohort["withdrawn_before_entry"]
        assert cohort["status"] == "withdrawn_before_entry"
        assert len(cohort["slots"]) == cohort["top_n"]
        assert cohort["entry_session"] == "2026-11-04"
        assert cohort["exit_session"] == "2026-12-02"


@pytest.mark.parametrize("stamp", ["2026-11-04T15:30:00+09:00", "2026-11-05T10:00:00+09:00"])
def test_withdrawal_at_or_after_entry_close_cannot_change_cohort(calendar, stamp):
    result = choose([publication(), withdrawal(verified_by=stamp)], calendar)
    assert not any(c["withdrawn_before_entry"] for c in result["cohorts"])
    assert result["decisions"][1]["status"] == "withdrawal_after_entry"


def test_withdrawal_uses_confirmed_operating_close_override():
    data = manifest()
    data["overrides"] = [
        {"date": "2026-11-04", "open_at": "10:00:00", "close_at": "16:30:00", "confirmed": True}
    ]
    result = choose(
        [publication(), withdrawal(verified_by="2026-11-04T16:00:00+09:00")],
        Calendar.from_manifest(data),
    )
    assert all(c["withdrawn_before_entry"] for c in result["cohorts"])
    assert all(c["entry_close_at"] == "2026-11-04T16:30:00+09:00" for c in result["cohorts"])


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"market": "US"}, "sealed_model_or_bundle"),
        ({"model_id": "different-model"}, "sealed_model_or_bundle"),
        ({"model_version": "2.0.0"}, "sealed_model_or_bundle"),
        ({"bundle_hash": "0" * 64}, "sealed_model_or_bundle"),
        ({"report_date": "2026-11-05"}, "withdrawal_report_date_mismatch"),
        ({"report_sha256": "f" * 64}, "withdrawal_report_binding_mismatch"),
        ({"invocation_id": "different-invocation"}, "withdrawal_report_binding_mismatch"),
        ({"input_sha256": "f" * 64}, "withdrawal_report_binding_mismatch"),
        ({"publication_ref": "not-a-hash"}, "invalid_publication_ref"),
        ({"receipt_ref": 123}, "invalid_receipt_ref"),
        ({"revision": False}, "invalid_revision"),
        ({"verified_by": "2026-11-04T10:04:00+09:00"}, "withdrawal_before_original_observation"),
        ({"coverage_status": "unknown"}, "publication_unverified"),
    ],
)
def test_withdrawal_requires_original_context_and_verified_evidence(calendar, change, reason):
    result = choose([publication(), withdrawal(**change)], calendar)
    assert len(result["cohorts"]) == 2
    assert not any(c["withdrawn_before_entry"] for c in result["cohorts"])
    assert reason in result["decisions"][1]["reasons"]


def test_pure_selection_no_external_reads_and_stable_identity(calendar, monkeypatch):
    event = publication()
    original = deepcopy(event)

    def forbidden(*args, **kwargs):
        raise AssertionError("pure selection must not read files")

    monkeypatch.setattr("builtins.open", forbidden)
    first = choose([event], calendar)
    second = choose([event], calendar)
    assert first == second
    assert event == original
    assert len({c["cohort_ref"] for c in first["cohorts"]}) == 2
