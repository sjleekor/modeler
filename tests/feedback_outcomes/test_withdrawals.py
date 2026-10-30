"""Invented code-generated withdrawal bytes only; no source or outcome reads."""

import json
from copy import deepcopy

import pytest

from modeler.feedback.core import FeedbackError, canonical_bytes, sha256_bytes
from modeler.feedback_outcomes import withdrawals
from modeler.feedback_outcomes.calculation import calculate_row
from modeler.feedback_outcomes.pipeline import evaluate
from modeler.feedback_outcomes.selection import Calendar, select_cohorts
from modeler.feedback_outcomes.synthetic import fixture
from modeler.feedback_outcomes.withdrawals import BINDING_KEYS, PAYLOAD_SCHEMA, SCHEMA

START = "2026-11-04T10:01:00+09:00"
OBSERVED = "2026-11-04T14:00:00+09:00"
CUTOFF = "2026-11-04T16:00:00+09:00"


def seal(sidecar, *, rechain=True):
    previous = None
    for record in sidecar["records"]:
        if rechain:
            record["previous_ref"] = previous
        if "receipt" in record:
            record["receipt_ref"] = sha256_bytes(canonical_bytes(record["receipt"]))
        body = {key: value for key, value in record.items() if key != "record_ref"}
        previous = record["record_ref"] = sha256_bytes(canonical_bytes(body))
    sidecar["head_ref"] = previous
    body = {key: value for key, value in sidecar.items() if key not in {"sidecar_ref", "evidence"}}
    sidecar["sidecar_ref"] = sha256_bytes(canonical_bytes(body))
    return sidecar


def rewrite_payload(sidecar, transform):
    record = sidecar["records"][1]
    raw = sidecar["evidence"].pop(record["source_sha256"])
    payload = json.loads(raw)
    transform(payload)
    raw = canonical_bytes(payload)
    digest = sha256_bytes(raw)
    sidecar["evidence"][digest] = raw
    record["source_sha256"] = record["receipt"]["unit_sha256"] = digest
    seal(sidecar)


def generated_sidecar(original, *, verified=OBSERVED):
    binding = {key: original[key] for key in BINDING_KEYS}
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "synthetic_fixture": True,
        "declaration": "withdrawal",
        "original_binding": deepcopy(binding),
        "revision": 2,
        "reports_commit": "c" * 40,
    }
    raw = canonical_bytes(payload)
    digest = sha256_bytes(raw)
    return seal(
        {
            "schema": SCHEMA,
            "synthetic_fixture": True,
            "coverage": {"started_at": START, "through": CUTOFF, "status": "covered", "gaps": []},
            "records": [
                {"kind": "baseline", "sequence": 0, "previous_ref": None, "observed_at": START},
                {
                    "kind": "withdrawal_evidence",
                    "sequence": 1,
                    "previous_ref": None,
                    "observed_at": verified,
                    "source_sha256": digest,
                    "receipt": {
                        "verified_by": verified,
                        "remote_verified": True,
                        "revision": 2,
                        "reports_commit": "c" * 40,
                        "unit_sha256": digest,
                        "original_binding": deepcopy(binding),
                    },
                },
            ],
            "evidence": {digest: raw},
        }
    )


def adapt(sidecar, original, **kwargs):
    return withdrawals.normalize_withdrawals(
        sidecar, [original], observed_at=kwargs.pop("observed_at", CUTOFF), synthetic=True, **kwargs
    )


def test_preentry_withdrawal_selects_cash_and_preserves_original_index_schedule():
    data = fixture()
    original = data["events"][0]
    sidecar = generated_sidecar(original)
    before = deepcopy((sidecar, original))
    events = adapt(sidecar, original)
    assert (sidecar, original) == before
    event = events[0]
    assert event["synthetic_fixture"] is True
    assert event["explicit_withdrawal_adapter"] == "synthetic_sidecar_proposal"
    assert event["original_publication_ref"] == original["publication_ref"]
    assert event["original_report_sha256"] == original["report_sha256"]
    assert event["withdrawal_evidence_sha256"] == event["unit_sha256"]
    assert event["absence_claim"] == "none"
    assert "source_bound_v1" not in str(event)
    calendar = Calendar.from_manifest(data["calendar"])
    selected = select_cohorts([original, *events], calendar, synthetic=True)
    assert selected["decisions"][-1]["status"] == "withdrawn_before_entry"
    for cohort in selected["cohorts"]:
        assert cohort["withdrawn_before_entry"] is True
        assert cohort["report_sha256"] == original["report_sha256"]
        assert cohort["entry_session"] == "2026-11-04"
        assert cohort["exit_session"] == "2026-12-02"
        row = cohort["slots"][0]
        market = row["security_market"].lower()
        observations = {
            key: {
                "index_group": market,
                "idx_nm": "코스피" if market == "kospi" else "코스닥",
                "session": cohort[session],
                "close_idx": close,
                "available_at": cohort[session] + "T17:00:00+09:00",
            }
            for key, session, close in (
                ("benchmark_entry", "entry_session", 1000),
                ("benchmark_exit", "exit_session", 1050),
            )
        }
        cash = calculate_row(
            row,
            entry_session=cohort["entry_session"],
            exit_session=cohort["exit_session"],
            observations=observations,
            cutoff=data["cutoff"],
            withdrawn_before_entry=cohort["withdrawn_before_entry"],
        )
        assert (cash["gross"], cash["net"], cash["cost"]) == (0, 0, 0)
        assert cash["benchmark"] == pytest.approx(0.05)


@pytest.mark.parametrize("verified", ["2026-11-04T15:30:00+09:00", CUTOFF])
def test_postentry_withdrawal_preserves_original_cohorts(verified):
    data = fixture()
    original = data["events"][0]
    calendar = Calendar.from_manifest(data["calendar"])
    expected = select_cohorts([original], calendar, synthetic=True)["cohorts"]
    events = adapt(generated_sidecar(original, verified=verified), original)
    selected = select_cohorts([original, *events], calendar, synthetic=True)
    assert selected["cohorts"] == expected
    assert selected["decisions"][-1]["status"] == "withdrawal_after_entry"


@pytest.mark.parametrize("missing_rank", [None, 20])
def test_adapter_to_synthetic_evaluation_preserves_cash_benchmark_and_incomplete_slots(
    missing_rank,
):
    data = fixture()
    original = data["events"][0]
    if missing_rank is not None:
        original["rows"] = [row for row in original["rows"] if row["rank"] != missing_rank]
    events = adapt(generated_sidecar(original), original)
    result = evaluate(
        data["calendar"],
        [*data["events"], *events],
        data["extract"],
        cutoff=data["cutoff"],
        synthetic=True,
    )
    assert result["synthetic_fixture"] is True
    assert result["real_price_reads"] == 0
    for row in result["rows"]:
        if row["report_sha256"] != original["report_sha256"]:
            continue
        assert row["status"] == "withdrawn_before_entry"
        assert (row["gross"], row["net"], row["cost"]) == (0, 0, 0)
        assert row["benchmark"] == pytest.approx(0.05)
        assert row["excess"] == pytest.approx(-0.05)
    for cohort in result["cohorts"]:
        if cohort["report_sha256"] != original["report_sha256"]:
            continue
        if missing_rank is None:
            assert cohort["net"] == 0
            assert cohort["benchmark"] == pytest.approx(0.05)
            assert cohort["status"] == "available"
        else:
            assert cohort["status"] == "source_incomplete"
            assert cohort["net"] is None
            assert cohort["missing_rank_count"] == 1
            assert missing_rank in cohort["missing_ranks"]
            assert len(cohort["slots"]) == cohort["top_n"] - 1


def test_real_mode_and_unmarked_input_reject_before_hashing_or_parsing(monkeypatch):
    original = fixture()["events"][0]
    sidecar = generated_sidecar(original)

    def forbidden(*args, **kwargs):
        pytest.fail("real-mode guard must run before parsing/hashing")

    monkeypatch.setattr(withdrawals, "sha256_bytes", forbidden)
    monkeypatch.setattr(withdrawals.json, "loads", forbidden)
    for synthetic in (False, None, 1, "true"):
        with pytest.raises(FeedbackError, match="synthetic-only"):
            withdrawals.normalize_withdrawals(
                sidecar, [original], observed_at="invalid", synthetic=synthetic
            )
    for flag in (False, None, 1):
        with pytest.raises(FeedbackError, match="fixture flag"):
            adapt(dict(sidecar, synthetic_fixture=flag), original)
    with pytest.raises(FeedbackError, match="original publications"):
        adapt(sidecar, dict(original, synthetic_fixture=False))


@pytest.mark.parametrize("key", BINDING_KEYS)
def test_every_original_identity_field_is_bound_to_supplied_publication(key):
    original = fixture()["events"][0]
    sidecar = generated_sidecar(original)
    changed = deepcopy(original)
    replacements = {
        "publication_ref": "1" * 64,
        "report_sha256": "2" * 64,
        "invocation_id": "another-invocation",
        "input_sha256": "3" * 64,
        "feature_asof_date": "2026-11-02",
        "report_date": "2026-11-05",
        "decision_at": "2026-11-04T10:00:01+09:00",
    }
    changed[key] = replacements.get(key, "different")
    with pytest.raises(FeedbackError):
        adapt(sidecar, changed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(schema="unknown"),
        lambda s: s.update(extra=True),
        lambda s: s["coverage"].update(gaps=[{"from": START, "through": OBSERVED}]),
        lambda s: s["coverage"].update(status="gap"),
        lambda s: s["coverage"].update(through=OBSERVED),
        lambda s: s["coverage"].update(started_at="2026-11-04T14:01:00+09:00"),
        lambda s: s["records"][0].update(kind="gap"),
        lambda s: s["records"][1].update(kind="report_missing"),
        lambda s: s["records"][1].update(sequence=2),
        lambda s: s["records"][1].update(sequence=True),
        lambda s: s["records"][1].update(observed_at="2026-11-04T16:00:01+09:00"),
        lambda s: s["records"][1]["receipt"].update(remote_verified=False),
        lambda s: s["records"][1]["receipt"].update(verified_by="2026-11-04T14:01:00+09:00"),
        lambda s: s["records"][1]["receipt"].update(verified_by="2026-11-04T10:02:00+09:00"),
        lambda s: s["records"][1]["receipt"].update(revision=True),
        lambda s: s["records"][1]["receipt"].update(reports_commit="bad"),
        lambda s: s["records"][1]["receipt"].update(unit_sha256="4" * 64),
        lambda s: s["records"][1]["receipt"]["original_binding"].update(invocation_id="wrong"),
    ],
)
def test_invalid_schema_coverage_chain_observation_and_receipt_fail_closed(mutation):
    original = fixture()["events"][0]
    sidecar = generated_sidecar(original)
    mutation(sidecar)
    seal(sidecar)
    with pytest.raises(FeedbackError):
        adapt(sidecar, original)


@pytest.mark.parametrize(
    "transform",
    [
        lambda p: p.update(declaration="publication"),
        lambda p: p.update(declaration="withdrawn"),
        lambda p: p.update(declaration="document_deleted"),
        lambda p: p.update(schema="purported-real-producer-format"),
        lambda p: p.update(synthetic_fixture=False),
        lambda p: p.update(revision=True),
        lambda p: p["original_binding"].update(input_sha256="d" * 64),
        lambda p: p.update(reports_commit="d" * 40),
        lambda p: p.update(extra="unapproved"),
    ],
)
def test_nonwithdrawal_and_tampered_payload_never_infer_withdrawal(transform):
    original = fixture()["events"][0]
    sidecar = generated_sidecar(original)
    rewrite_payload(sidecar, transform)
    with pytest.raises(FeedbackError):
        adapt(sidecar, original)


@pytest.mark.parametrize(
    "target", ["sidecar_ref", "record_ref", "receipt_ref", "head_ref", "raw", "previous_ref"]
)
def test_each_hash_and_chain_link_is_checked(target):
    original = fixture()["events"][0]
    sidecar = generated_sidecar(original)
    if target == "raw":
        digest = sidecar["records"][1]["source_sha256"]
        sidecar["evidence"][digest] += b" "
    elif target == "previous_ref":
        sidecar["records"][1][target] = "0" * 64
        seal(sidecar, rechain=False)
    elif target in {"record_ref", "receipt_ref"}:
        sidecar["records"][1][target] = "0" * 64
        body = {k: v for k, v in sidecar.items() if k not in {"sidecar_ref", "evidence"}}
        sidecar["sidecar_ref"] = sha256_bytes(canonical_bytes(body))
    else:
        sidecar[target] = "0" * 64
        if target == "head_ref":
            body = {k: v for k, v in sidecar.items() if k not in {"sidecar_ref", "evidence"}}
            sidecar["sidecar_ref"] = sha256_bytes(canonical_bytes(body))
    with pytest.raises(FeedbackError):
        adapt(sidecar, original)


def test_baseline_only_does_not_claim_absence_or_replace_any_report():
    data = fixture()
    original = data["events"][0]
    sidecar = generated_sidecar(original)
    sidecar["records"] = sidecar["records"][:1]
    sidecar["evidence"] = {}
    seal(sidecar)
    assert adapt(sidecar, original) == []
    calendar = Calendar.from_manifest(data["calendar"])
    selected = select_cohorts([original, *adapt(sidecar, original)], calendar, synthetic=True)
    assert len(selected["cohorts"]) == 2
    assert all(cohort["withdrawn_before_entry"] is False for cohort in selected["cohorts"])


def test_missing_baseline_evidence_and_excess_inventory_are_rejected():
    original = fixture()["events"][0]
    sidecar = generated_sidecar(original)
    for bad in (
        dict(sidecar, records=[]),
        dict(sidecar, evidence={}),
        dict(sidecar, evidence={**sidecar["evidence"], "f" * 64: b"extra"}),
        dict(sidecar, evidence={"f" * 64: "path/to/report.json"}),
        dict(sidecar, evidence={"f" * 64: b"x" * (withdrawals.MAX_EVIDENCE_BYTES + 1)}),
    ):
        with pytest.raises(FeedbackError):
            adapt(bad, original)


def test_duplicate_originals_or_evidence_and_original_future_are_rejected():
    original = fixture()["events"][0]
    sidecar = generated_sidecar(original)
    with pytest.raises(FeedbackError, match="duplicate original"):
        withdrawals.normalize_withdrawals(
            sidecar, [original, original], observed_at=CUTOFF, synthetic=True
        )
    with pytest.raises(FeedbackError, match="after observation cutoff"):
        adapt(sidecar, dict(original, verified_by="2026-11-05T10:00:00+09:00"))
    duplicate = deepcopy(sidecar["records"][1])
    duplicate["sequence"] = 2
    sidecar["records"].append(duplicate)
    seal(sidecar)
    with pytest.raises(FeedbackError, match="duplicate withdrawal source"):
        adapt(sidecar, original)


def test_prospective_baseline_must_start_by_original_observation():
    original = fixture()["events"][0]
    sidecar = generated_sidecar(original)
    late_start = "2026-11-04T10:04:00+09:00"
    sidecar["coverage"]["started_at"] = late_start
    sidecar["records"][0]["observed_at"] = late_start
    seal(sidecar)
    with pytest.raises(FeedbackError, match="starts after original publication"):
        adapt(sidecar, original)


@pytest.mark.parametrize("raw", [b"{", b"\xff", b"[]", b'{"schema":1,"schema":2}'])
def test_malformed_and_duplicate_key_evidence_is_rejected(raw):
    original = fixture()["events"][0]
    sidecar = generated_sidecar(original)
    digest = sha256_bytes(raw)
    sidecar["evidence"] = {digest: raw}
    record = sidecar["records"][1]
    record["source_sha256"] = record["receipt"]["unit_sha256"] = digest
    seal(sidecar)
    with pytest.raises(FeedbackError):
        adapt(sidecar, original)
