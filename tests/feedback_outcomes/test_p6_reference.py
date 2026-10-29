"""Synthetic JSON grammar only; no registered check or point artifact is opened."""

import hashlib
import json
from copy import deepcopy

import pytest

from modeler.feedback_outcomes.contracts import OutcomeError
from modeler.feedback_outcomes.p6_reference import (
    ARTIFACT_KIND,
    CHECK_SCHEMA,
    P6_BUNDLE_SHA256,
    P6_MODEL_ID,
    PREREG_DOCUMENT_SHA256,
    REGISTRATION_SCHEMA,
    link_registered_check,
)


def synthetic_check():
    return {
        "schema": CHECK_SCHEMA,
        "check_date": "2099-01-15",
        "data_date": "2099-01-15",
        "decision": {
            "check_date": "2099-01-15",
            "n_matured": 1,
            "mean_g": 0.0123,
            "mean_rank_ic": None,
            "mean_tail_diff": -0.0456,
            "alpha": 0.025,
            "model": "표본 부족",
            "stop_proposal": False,
            "tail": "표본 부족",
            "significance_g": {"n24": {"status": "판정 보류"}, "n36": {"status": "판정 보류"}},
            "significance_tail": {"n24": {"status": "판정 보류"}, "n36": {"status": "판정 보류"}},
        },
        "points": [
            {
                "grid_date": "2098-11-01",
                "matured_at": "2098-12-01",
                "g_at_check": 0.0123,
                "tail_diff_at_check": -0.0456,
                "rank_ic_at_check": None,
                "n_pending_at_check": 2,
                "pending_tickers_at_check": ["synthetic-A", "synthetic-B"],
                "excluded_tail_at_check": '{"pending":1}',
                "excluded_rest_at_check": '{"pending":1}',
                "final": {"g": 0.999, "tail_diff": 0.888, "rank_ic": 0.777, "n_pending": 0},
                "final_revision": 3,
                "differs_from_final": {
                    "g": True,
                    "tail_diff": True,
                    "rank_ic": True,
                    "n_pending": True,
                },
                "g_final": 0.999,
                "g_final_differs": True,
                "turnover": 1.0,
            }
        ],
        "missing": [{"grid_date": "2098-10-01", "reason": "synthetic-no-record"}],
        "turnover_warning": False,
        "requires_user_confirmation": False,
        "cell_stats": {
            "g": {
                "n": 1,
                "mean": 0.0123,
                "power_base": None,
                "power_scenario": {
                    "effect": 0.005,
                    "sd": 0.04,
                    "alpha": 0.025,
                    "test": "one-sided t",
                },
                "lower_975_one_sided": None,
            },
            "tail_diff": {
                "n": 1,
                "mean": -0.0456,
                "power_base": None,
                "power_scenario": {
                    "effect": -0.01,
                    "sd": 0.03,
                    "alpha": 0.025,
                    "test": "one-sided t",
                },
                "upper_975_one_sided": None,
            },
            "rank_ic": {"n": 0, "mean": None, "ci95_two_sided": None},
            "record_only": True,
        },
        "note": "synthetic aggregate fixture; never a real observation",
        "inputs": {
            "bundle_tag_time": "2098-01-01T10:00:00+09:00",
            "prereg_tag": "kr-next-frozen",
            "ledger_lines": 2,
            "code_inventory_sha256": "a" * 64,
        },
    }


def encode(check):
    return json.dumps(check, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def registration(raw):
    return {
        "schema": REGISTRATION_SCHEMA,
        "model_id": P6_MODEL_ID,
        "bundle_sha256": P6_BUNDLE_SHA256,
        "registered_check_date": "2099-01-15",
        "not_before": "2099-01-15T13:00:00+09:00",
        "expected_check_sha256": hashlib.sha256(raw).hexdigest(),
        "registration_sha256": PREREG_DOCUMENT_SHA256,
        "allowed_artifact_kind": ARTIFACT_KIND,
    }


AFTER = "2099-01-15T13:00:00+09:00"


def test_links_exact_bytes_without_recomputing_or_exposing_final_per_security_values():
    check = synthetic_check()
    raw = encode(check)
    reg = registration(raw)
    original = deepcopy(reg)
    result = link_registered_check(reg, raw, observed_at=AFTER)
    assert result["kind"] == "model_aggregate_check"
    assert result["status"] == "linked"
    assert result["per_security_outcome"] is False
    assert result["recomputed"] is False
    assert result["model_binding_in_check"] is False
    assert result["check_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["values"]["decision"] == check["decision"]
    assert result["values"]["cell_stats"] == check["cell_stats"]
    point = result["values"]["points"][0]
    assert point["g_at_check"] == 0.0123
    assert "final" not in point and "g_final" not in point and "final_revision" not in point
    assert "pending_tickers_at_check" not in point
    assert reg == original


def test_missing_check_remains_awaiting_without_guessing_or_running_forward():
    reg = registration(encode(synthetic_check()))
    result = link_registered_check(reg, None, observed_at=AFTER)
    assert result["status"] == "awaiting_check"
    assert result["check_sha256"] is None and result["values"] is None


def test_early_observation_does_not_hash_parse_or_inspect_check(monkeypatch):
    reg = registration(encode(synthetic_check()))

    def forbidden(*args, **kwargs):
        raise AssertionError("check bytes were touched before their registration cutoff")

    monkeypatch.setattr("modeler.feedback_outcomes.p6_reference.hashlib.sha256", forbidden)
    monkeypatch.setattr("modeler.feedback_outcomes.p6_reference.json.loads", forbidden)
    result = link_registered_check(reg, object(), observed_at="2099-01-15T12:59:59+09:00")
    assert result["status"] == "awaiting_check"
    assert result["reason"] == "registered_time_not_reached"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "unknown.v1"),
        ("model_id", "US-model"),
        ("bundle_sha256", "f" * 64),
        ("registration_sha256", "f" * 64),
        ("expected_check_sha256", "bad-hash"),
        ("allowed_artifact_kind", "per_security_outcome"),
        ("registered_check_date", "20990115"),
        ("not_before", "2099-01-15T13:00:00"),
        ("not_before", "2099-01-14T13:00:00+09:00"),
    ],
)
def test_rejects_unpinned_or_invalid_registration_even_without_check(field, value):
    raw = encode(synthetic_check())
    reg = registration(raw)
    reg[field] = value
    with pytest.raises(OutcomeError):
        link_registered_check(reg, None, observed_at=AFTER)


@pytest.mark.parametrize("field", ["points_revision", "points_path", "label_path", "new_rank"])
def test_registration_cannot_add_revision_label_or_security_inputs(field):
    raw = encode(synthetic_check())
    reg = registration(raw)
    reg[field] = "forbidden"
    with pytest.raises(OutcomeError, match="registration keys"):
        link_registered_check(reg, raw, observed_at=AFTER)


def test_mismatched_check_byte_hash_rejected_before_json_parse(monkeypatch):
    raw = encode(synthetic_check())
    reg = registration(raw)

    def forbidden(*args, **kwargs):
        raise AssertionError("changed bytes reached the parser")

    monkeypatch.setattr("modeler.feedback_outcomes.p6_reference.json.loads", forbidden)
    with pytest.raises(OutcomeError, match="byte hash"):
        link_registered_check(reg, raw + b" ", observed_at=AFTER)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda c: c.update(schema="other-schema"),
        lambda c: c.update(check_date="2099-01-16"),
        lambda c: c.update(data_date="2099-01-16"),
        lambda c: c.update(points_revision=4),
        lambda c: c.update(model_id="US-model"),
        lambda c: c["decision"].update(check_date="2099-01-16"),
        lambda c: c["decision"].update(new_rank=1),
        lambda c: c["points"][0].update(matured_at="2099-01-16"),
        lambda c: c["points"][0].update(per_security_return=0.25),
        lambda c: c["points"].append(deepcopy(c["points"][0])),
        lambda c: c["cell_stats"].update(record_only=False),
        lambda c: c["cell_stats"].update(per_security_outcomes=[]),
        lambda c: c["cell_stats"]["g"].update(new_rank=1),
        lambda c: c["decision"]["significance_g"].update(new_security_return=0.1),
        lambda c: c["inputs"].update(prereg_tag="US-frozen"),
    ],
)
def test_rejects_wrong_check_date_schema_or_unregistered_additions(mutation):
    check = synthetic_check()
    mutation(check)
    raw = encode(check)
    with pytest.raises(OutcomeError):
        link_registered_check(registration(raw), raw, observed_at=AFTER)


@pytest.mark.parametrize("raw", [b"not-json", b'{"schema":1,"schema":2}', b'{"value":NaN}'])
def test_rejects_invalid_or_ambiguous_json(raw):
    with pytest.raises(OutcomeError):
        link_registered_check(registration(raw), raw, observed_at=AFTER)


def test_no_input_file_reads_and_no_producer_import(monkeypatch):
    raw = encode(synthetic_check())
    reg = registration(raw)

    def forbidden(*args, **kwargs):
        raise AssertionError("aggregate reference must never read files")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("pathlib.Path.open", forbidden)
    result = link_registered_check(reg, raw, observed_at=AFTER)
    assert result["status"] == "linked"
