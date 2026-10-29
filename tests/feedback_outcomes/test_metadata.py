"""Generated F3 report/git captures only; never a saved operational artifact."""

from __future__ import annotations

import base64
import subprocess
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import pytest

from modeler.feedback import capture
from modeler.feedback.core import FeedbackError, canonical_bytes, sha256_bytes
from modeler.feedback_outcomes import metadata
from modeler.feedback_outcomes.contracts import BUNDLE_SHA256
from modeler.feedback_outcomes.selection import Calendar, select_cohorts
from modeler.reporting.markdown import front_matter, unit_sha256

DAY = "2026-11-04"
OBSERVED = "2026-12-04T13:00:00+09:00"
SCOPE = {
    "reviewed": True,
    "registry_complete": True,
    "controlled_epoch": True,
    "registry_sha256": "e" * 64,
    "producers": ["reports-publisher"],
}


def _git(root, *args):
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, text=True, capture_output=True
    ).stdout.strip()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))


def synthetic_capture(
    tmp_path,
    *,
    count=100,
    shown_ranks=None,
    status="partial",
    lag=0,
    missing_lag=False,
    synthetic=True,
    historical=False,
    model_changes=None,
    report_changes=None,
    extra_models=(),
    malformed_explain=False,
):
    """Create bounded synthetic producer bytes, local git, then genuine F3 captures.

    This fixture follows F3's existing test_capture/test_briefing grammar, with a
    prospective fixed-rule date and generated 1..100 ranks for F4a integration.
    No remote is configured or contacted.
    """
    day, checkout = tmp_path / "producer", tmp_path / "reports"
    day.mkdir(parents=True)
    checkout.mkdir()
    _git(checkout, "init")
    _git(
        checkout,
        "-c",
        "user.name=Synthetic",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "Synthetic baseline",
    )
    baseline = _git(checkout, "rev-parse", "HEAD")
    config = {
        "checkout_dir": str(checkout),
        "run_dir": str(day),
        "report_date": DAY,
        "feedback_capture": {
            "output_root": str(tmp_path / "capture"),
            "session_id": "synthetic-session",
        },
    }
    capture.begin_session(
        config,
        started_at=DAY + "T10:01:00+09:00",
        baseline_remote_head=baseline,
        day_unit_absent=True,
        no_prior_unit_history=True,
        producer_scope=SCOPE,
    )
    model = {
        "market": "KR",
        "report_date": DAY,
        "decision_at": DAY + "T10:00:00+09:00",
        "feature_asof_date": "2026-11-03",
        "model_id": "kr_daily_h20_v1",
        "model_version": "1.0.0",
        "status": status,
        "rankings": [
            {
                "symbol": f"{i:06d}",
                "market": "KOSPI" if i % 2 else "KOSDAQ",
                "rank": i,
                "score": 0.9 - i / 1000,
                "quality_reasons": ["synthetic-quality-warning"] if i == 3 else [],
            }
            for i in range(1, count + 1)
        ],
        "provenance": {
            "bundle_manifest_sha256": BUNDLE_SHA256,
            "input_sha256": "a" * 64,
            "code_inventory_sha256": "b" * 64,
            "native_prepare_manifest_sha256": "c" * 64,
            "freshness": {
                "status": "ok" if lag == 0 else "stale",
                "delivery_lag": lag,
                "market_lag": 0 if lag == 0 else None,
            },
        },
    }
    if missing_lag:
        model["provenance"].pop("freshness")
    model.update(model_changes or {})
    report = {
        "schema_version": "1.0",
        "report_date": DAY,
        "decision_at": DAY + "T10:00:00+09:00",
        "markets": [model, *extra_models],
        "synthetic_fixture": synthetic,
        "historical_replay": historical,
    }
    report.update(report_changes or {})
    _write(day / f"report-{DAY}.json", report)
    report_hash = sha256_bytes((day / f"report-{DAY}.json").read_bytes())
    invocation = "generated-f4a-invocation"
    _write(day / "run-state.json", {"invocation_id": invocation})
    _write(
        day / "coordinator-inference.json",
        {
            "report_date": DAY,
            "report_sha256": report_hash,
            "invocation_id": invocation,
            "status": "inferred",
        },
    )
    _write(day / "reports-publisher-config.json", {"release": "synthetic-release"})
    if malformed_explain:
        (day / "coordinator-explain.json").write_bytes(b"invalid synthetic optional JSON")
        (day / f"explain-{DAY}.json").write_bytes(b"invalid synthetic optional explanation")
    files = {}
    for name in sorted(capture._UNIT_FILES):
        is_kr = name == "kr-stocks.md"
        meta = {
            "schema": "stock-reports.v1",
            "unit": DAY,
            "revision": 1,
            "section": "summary" if name == "README.md" else name.removesuffix(".md"),
            "models": [model["model_id"]] if is_kr else [],
            "source": {"release": "synthetic-release", "report_sha256": report_hash},
        }
        body = "Synthetic metadata only\n"
        if is_kr:
            body += (
                f"\n## {model['model_id']}\n\n"
                f"- 모델: [Synthetic](../../reference/models/{model['model_id']}.md) "
                f"(`{model['model_id']}`)\n\n| 순위 | 코드 | 점수 |\n| ---: | --- | ---: |\n"
            )
            body += "".join(
                f"| {row['rank']} | {row['symbol']} | {row['score']} |\n"
                for row in model["rankings"]
                if shown_ranks is None or row["rank"] in shown_ranks
            )
        files[name] = front_matter(meta) + "\n" + body
        path = checkout / capture._unit_path(DAY) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[name], encoding="utf-8")
    _git(checkout, "add", "reports")
    _git(
        checkout,
        "-c",
        "user.name=Synthetic",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "Synthetic publication",
    )
    commit = _git(checkout, "rev-parse", "HEAD")
    digest = unit_sha256(files)
    _write(
        day / "markdown/manifest.json",
        {
            "unit": DAY,
            "release": "synthetic-release",
            "report_sha256": report_hash,
            "invocation_id": invocation,
            "unit_sha256": digest,
        },
    )
    entry = {
        "unit": DAY,
        "run_dir": str(day),
        "markdown_dir": str(day / "markdown"),
        "report_sha256": report_hash,
        "invocation_id": invocation,
        "release": "synthetic-release",
        "unit_sha256": digest,
        "commit": commit,
    }
    pending = capture.record_intent(
        config, entry, observation_started_at=DAY + "T10:02:00+09:00", previous_remote_head=baseline
    )
    result = capture.capture_published(
        config,
        entry,
        commit,
        DAY + "T10:03:00+09:00",
        producer_scope=SCOPE,
        baseline_evidence={
            "method": "publisher-verified-push.v1",
            "remote_head": commit,
            "previous_remote_head": baseline,
            "observation_started_at": DAY + "T10:02:00+09:00",
            "intent_ref": pending["intent_ref"],
        },
    )
    return config, entry, report, result


def _bundle(tmp_path, **changes):
    config, entry, report, receipt = synthetic_capture(tmp_path, **changes)
    return metadata.export_capture_metadata(config), config, entry, report, receipt


def _normalize(bundle, **kwargs):
    return metadata.normalize_publications(bundle, observed_at=OBSERVED, synthetic=True, **kwargs)


def _rehash_chain(session):
    """Generate a deliberately different synthetic chain to test semantic guards."""
    previous = None
    mapping = {}
    for record in session["records"]:
        old = record["capture_ref"]
        record["previous_ref"] = previous
        evidence = record.get("remote_evidence", {})
        if evidence.get("intent_ref") in mapping:
            evidence["intent_ref"] = mapping[evidence["intent_ref"]]
        record["capture_ref"] = sha256_bytes(
            canonical_bytes({k: v for k, v in record.items() if k != "capture_ref"})
        )
        mapping[old] = record["capture_ref"]
        previous = record["capture_ref"]
    session["state"]["head_ref"] = previous


def test_f3_bytes_to_normalized_events_to_selection(tmp_path):
    bundle, _, entry, report, receipt = _bundle(tmp_path)
    events = _normalize(bundle)
    assert len(events) == 1
    event = events[0]
    assert event["report_sha256"] == entry["report_sha256"]
    assert event["receipt_ref"] == receipt["capture_ref"]
    assert event["model_status"] == report["markets"][0]["status"]
    assert event["freshness_lag_sessions"] == 0
    assert event["collection_started_at"] == DAY + "T10:01:00+09:00"
    assert event["coverage_status"] == "covered"
    assert len(event["rows"]) == 100
    assert event["rows"][2]["quality_reasons"] == ["synthetic-quality-warning"]
    start = date(2026, 11, 2)
    days = [start + timedelta(days=i) for i in range(70)]
    calendar = Calendar.from_manifest(
        {
            "market": "KR",
            "timezone": "Asia/Seoul",
            "coverage_start": start.isoformat(),
            "coverage_end": days[-1].isoformat(),
            "sessions": [day.isoformat() for day in days if day.weekday() < 5],
            "default_open_at": "09:00:00",
            "default_close_at": "15:30:00",
        }
    )
    selection = select_cohorts(events, calendar, synthetic=True)
    assert selection["decisions"][0]["status"] == "selected"
    assert [cohort["size"] for cohort in selection["cohorts"]] == [20, 100]
    assert all(cohort["entry_session"] == DAY for cohort in selection["cohorts"])
    assert all(cohort["exit_session"] == "2026-12-02" for cohort in selection["cohorts"])


def test_archive_replica_survives_source_disappearance_and_preserves_original_bytes(tmp_path):
    bundle, config, entry, _, _ = _bundle(tmp_path, count=2)
    original = deepcopy(bundle)
    expected = (Path(entry["run_dir"]) / f"report-{DAY}.json").read_bytes()
    Path(entry["run_dir"]).rename(tmp_path / "producer-removed")
    Path(config["checkout_dir"]).rename(tmp_path / "checkout-removed")
    assert len(_normalize(bundle)[0]["rows"]) == 2
    digest = entry["report_sha256"]
    assert base64.b64decode(bundle["sessions"][0]["artifacts"][digest]) == expected
    assert bundle == original


def test_only_originally_exposed_top100_refs_are_normalized(tmp_path):
    bundle, *_ = _bundle(tmp_path, count=105, shown_ranks={1, 3, 100, 101})
    event = _normalize(bundle)[0]
    assert [row["rank"] for row in event["rows"]] == [1, 3, 100]


@pytest.mark.parametrize("status", ["partial", "failed", "stale", None])
def test_original_model_status_is_never_upgraded(tmp_path, status):
    bundle, *_ = _bundle(tmp_path, count=1, status=status)
    assert _normalize(bundle)[0]["model_status"] == status


def test_missing_original_lag_never_infers_zero_from_other_fields(tmp_path):
    bundle, *_ = _bundle(
        tmp_path,
        count=1,
        missing_lag=True,
        model_changes={"freshness": {"delivery_lag": 0, "market_lag": 0}},
    )
    assert _normalize(bundle)[0]["freshness_lag_sessions"] is None


def test_provenance_freshness_keeps_original_lag_without_deriving_one(tmp_path):
    bundle, *_ = _bundle(tmp_path, count=1, model_changes={"freshness": None})
    event = _normalize(bundle)[0]
    # A null legacy field cannot erase the independently explicit producer proof.
    assert event["freshness_lag_sessions"] == 0
    assert event["freshness_lag_source"] == "provenance.freshness.delivery_lag"
    assert event["original_freshness"] is None


def test_real_normalization_excludes_historical_replay_without_withdrawal(tmp_path):
    bundle, *_ = _bundle(tmp_path, count=1, synthetic=False, historical=True)
    assert metadata.normalize_publications(bundle, observed_at=OBSERVED) == []


def test_fixture_and_real_bundle_flags_cannot_be_mixed(tmp_path):
    bundle, *_ = _bundle(tmp_path, count=1)
    with pytest.raises(FeedbackError, match="cannot be mixed"):
        metadata.normalize_publications(bundle, observed_at=OBSERVED)


@pytest.mark.parametrize("target", ["capture_ref", "sequence", "previous_ref", "schema"])
def test_changed_original_record_is_rejected(tmp_path, target):
    bundle, *_ = _bundle(tmp_path, count=1)
    record = bundle["sessions"][0]["records"][-1]
    record[target] = "wrong"
    with pytest.raises(FeedbackError):
        _normalize(bundle)


def test_chain_discontinuity_rejected_even_when_synthetic_body_hash_recomputed(tmp_path):
    bundle, *_ = _bundle(tmp_path, count=1)
    session = bundle["sessions"][0]
    session["records"][-1]["sequence"] = 10
    _rehash_chain(session)
    with pytest.raises(FeedbackError, match="discontinuity"):
        _normalize(bundle)


def test_state_head_and_duplicate_refs_are_rejected(tmp_path):
    bundle, *_ = _bundle(tmp_path, count=1)
    original = deepcopy(bundle)
    bundle["sessions"][0]["state"]["head_ref"] = "0" * 64
    with pytest.raises(FeedbackError, match="head"):
        _normalize(bundle)
    original["sessions"].append(deepcopy(original["sessions"][0]))
    with pytest.raises(FeedbackError, match="duplicate"):
        _normalize(original)


@pytest.mark.parametrize(
    "bad_path",
    [
        "prices.parquet",
        "labels/us.parquet",
        "../report.json",
        "evaluation/result.json",
        "git:unknown:reports/daily-briefing/us/kr-stocks.md",
        "report-2026-11-03.json",
        "raw/secret.json",
    ],
)
def test_forbidden_source_rejected_before_any_base64_decode(tmp_path, monkeypatch, bad_path):
    bundle, *_ = _bundle(tmp_path, count=1)
    session = bundle["sessions"][0]
    session["records"][-1]["sources"][0]["path"] = bad_path
    _rehash_chain(session)

    def forbidden(*args, **kwargs):
        raise AssertionError("base64 decoded before source allowlist")

    monkeypatch.setattr(metadata.base64, "b64decode", forbidden)
    with pytest.raises(FeedbackError, match="allowlist"):
        _normalize(bundle)


def test_source_allowlist_applies_to_all_sessions_before_any_decode(tmp_path, monkeypatch):
    bundle, *_ = _bundle(tmp_path, count=1)
    duplicate = deepcopy(bundle["sessions"][0])
    duplicate["session_id"] = "other-session"
    duplicate["state"]["session_id"] = "other-session"
    for record in duplicate["records"]:
        record["session_id"] = "other-session"
    duplicate["records"][-1]["sources"][0]["path"] = "prices.parquet"
    _rehash_chain(duplicate)
    bundle["sessions"].append(duplicate)
    monkeypatch.setattr(metadata.base64, "b64decode", lambda *a, **k: pytest.fail("decoded first"))
    with pytest.raises(FeedbackError, match="allowlist"):
        _normalize(bundle)


@pytest.mark.parametrize(
    "replacement", [None, "not base64!", base64.b64encode(b"wrong bytes").decode()]
)
def test_null_bad_base64_or_changed_evidence_is_rejected(tmp_path, replacement):
    bundle, *_ = _bundle(tmp_path, count=1)
    digest = next(iter(bundle["sessions"][0]["artifacts"]))
    bundle["sessions"][0]["artifacts"][digest] = replacement
    with pytest.raises(FeedbackError):
        _normalize(bundle)


def test_unreferenced_evidence_and_unknown_bundle_keys_are_rejected(tmp_path):
    bundle, *_ = _bundle(tmp_path, count=1)
    original = deepcopy(bundle)
    bundle["sessions"][0]["artifacts"][sha256_bytes(b"unreferenced")] = "dW5yZWZlcmVuY2Vk"
    with pytest.raises(FeedbackError, match="inventory"):
        _normalize(bundle)
    original["price_path"] = "never-open"
    with pytest.raises(FeedbackError, match="schema"):
        _normalize(original)


def test_bundle_cap_is_enforced_before_decode(tmp_path, monkeypatch):
    bundle, *_ = _bundle(tmp_path, count=1)
    monkeypatch.setattr(metadata, "MAX_BUNDLE_BYTES", len(canonical_bytes(bundle)) - 1)
    monkeypatch.setattr(metadata.base64, "b64decode", lambda *a, **k: pytest.fail("decoded first"))
    with pytest.raises(FeedbackError, match="32 MiB"):
        _normalize(bundle)


def test_collection_gap_stays_unverified_and_never_becomes_withdrawal(tmp_path):
    _, config, *_ = _bundle(tmp_path, count=1)
    capture.record_gap(config, "synthetic-gap", observed_at=DAY + "T11:00:00+09:00")
    bundle = metadata.export_capture_metadata(config)
    event = _normalize(bundle)[0]
    assert event["coverage_status"] == "awaiting_coverage"
    assert event["kind"] == "publication"
    assert event["explicit_withdrawal_adapter"] == "not_implemented_no_original_schema"


def test_future_verification_cannot_enter_observed_metadata_cutoff(tmp_path):
    bundle, *_ = _bundle(tmp_path, count=1)
    with pytest.raises(FeedbackError, match="cutoff"):
        metadata.normalize_publications(bundle, observed_at=DAY + "T10:02:00+09:00", synthetic=True)


def test_optional_explain_errors_are_preserved_without_fabricating_validity(tmp_path):
    bundle, *_ = _bundle(tmp_path, count=1, malformed_explain=True)
    event = _normalize(bundle)[0]
    assert len(event["rows"]) == 1
    assert {error["source"] for error in event["adapter_errors"]} == {
        "coordinator-explain.json",
        f"explain-{DAY}.json",
    }


def test_invalid_original_quality_reasons_are_not_silently_replaced_with_valid_empty_list(tmp_path):
    bundle, *_ = _bundle(
        tmp_path,
        count=1,
        model_changes={
            "rankings": [
                {
                    "symbol": "000001",
                    "market": "KOSPI",
                    "rank": 1,
                    "score": 0.5,
                    "quality_reasons": "invalid-original-type",
                }
            ]
        },
    )
    event = _normalize(bundle)[0]
    assert event["rows"][0]["quality_reasons"] == "invalid-original-type"
    assert event["rows"][0]["quality_validation"] == "source_invalid"
    assert any(error["reason"] == "quality_reasons_invalid" for error in event["adapter_errors"])


def test_original_invalid_report_flag_is_rejected(tmp_path):
    bundle, *_ = _bundle(tmp_path, count=1, report_changes={"synthetic_fixture": "yes"})
    with pytest.raises(FeedbackError, match="original report fixture flag"):
        metadata.normalize_publications(bundle, observed_at=OBSERVED, synthetic=False)


def test_common_envelope_us_metadata_never_emits_us_outcome_events(tmp_path):
    us = {
        "market": "US",
        "model_id": "synthetic-us-envelope-only",
        "model_version": "1",
        "report_date": DAY,
        "decision_at": DAY + "T10:00:00+09:00",
        "feature_asof_date": "2026-11-03",
        "status": "ok",
        "rankings": [{"symbol": "US-SYNTHETIC", "rank": 1, "score": 0.3}],
        "provenance": {
            key: "d" * 64
            for key in (
                "bundle_manifest_sha256",
                "input_sha256",
                "code_inventory_sha256",
                "native_prepare_manifest_sha256",
            )
        },
    }
    bundle, *_ = _bundle(tmp_path, count=1, extra_models=[us])
    events = _normalize(bundle)
    assert len(events) == 1
    assert all(event["market"] == "KR" for event in events)
    assert [row["symbol"] for row in events[0]["rows"]] == ["000001"]


def test_original_provenance_freshness_lag_is_used_only_when_explicitly_present(tmp_path):
    provenance = {
        "bundle_manifest_sha256": BUNDLE_SHA256,
        "input_sha256": "a" * 64,
        "code_inventory_sha256": "b" * 64,
        "native_prepare_manifest_sha256": "c" * 64,
        "freshness": {"lag_sessions": 2},
    }
    bundle, *_ = _bundle(
        tmp_path, count=1, missing_lag=True, model_changes={"provenance": provenance}
    )
    event = _normalize(bundle)[0]
    assert event["freshness_lag_sessions"] == 2
    assert event["original_freshness"] == {"lag_sessions": 2}


def test_export_source_allowlist_runs_before_f3_artifact_reads(tmp_path, monkeypatch):
    bundle, config, *_ = _bundle(tmp_path, count=1)
    session = bundle["sessions"][0]
    session["records"][-1]["sources"][0]["path"] = "labels/sealed-us.parquet"
    _rehash_chain(session)
    calls = []

    def guarded_read(settings, capture_ref=None):
        calls.append(capture_ref)
        if capture_ref is not None:
            pytest.fail("export opened artifact evidence before checking source allowlist")
        return {"state": session["state"], "records": session["records"]}

    monkeypatch.setattr(metadata.capture, "read_capture", guarded_read)
    with pytest.raises(FeedbackError, match="allowlist"):
        metadata.export_capture_metadata(config)
    assert calls == [None]


@pytest.mark.parametrize(
    ("freshness", "expected", "conflict"),
    [
        ({"status": "ok", "delivery_lag": 0, "market_lag": 0}, 0, False),
        ({"status": "stale", "delivery_lag": 1, "market_lag": None}, 1, False),
        ({"status": "ok", "delivery_lag": None, "market_lag": 0}, None, False),
        ({"status": "ok", "market_lag": 0}, None, False),
        ({"status": "ok"}, None, False),
        ({"delivery_lag": 0, "market_lag": 1}, None, True),
        ({"delivery_lag": 1, "market_lag": 0}, None, True),
        ({"lag_sessions": 2, "delivery_lag": 1, "market_lag": None}, None, True),
    ],
)
def test_actual_kr_producer_freshness_schema_maps_only_explicit_delivery_lag(
    tmp_path, freshness, expected, conflict
):
    provenance = {
        "bundle_manifest_sha256": BUNDLE_SHA256,
        "input_sha256": "a" * 64,
        "code_inventory_sha256": "b" * 64,
        "native_prepare_manifest_sha256": "c" * 64,
        "freshness": freshness,
    }
    bundle, *_ = _bundle(
        tmp_path,
        count=1,
        model_changes={"provenance": provenance, "quality": {"reference_lag_sessions": 0}},
    )
    event = _normalize(bundle)[0]
    assert event["freshness_lag_sessions"] == expected
    assert event["freshness_lag_source"] == (
        "provenance.freshness.delivery_lag" if expected is not None else None
    )
    assert event["original_freshness"] == freshness
    assert (
        any(error["reason"] == "freshness_lag_conflict" for error in event["adapter_errors"])
        is conflict
    )


def test_legacy_lag_and_producer_delivery_conflict_is_explicit(tmp_path):
    bundle, *_ = _bundle(tmp_path, count=1, model_changes={"freshness": {"lag_sessions": 1}})
    event = _normalize(bundle)[0]
    assert event["freshness_lag_sessions"] is None
    assert event["freshness_lag_source"] is None
    error = next(
        error for error in event["adapter_errors"] if error["reason"] == "freshness_lag_conflict"
    )
    assert error["fields"]["model.freshness.lag_sessions"] == 1
    assert error["fields"]["provenance.freshness.delivery_lag"] == 0


def test_legacy_model_lag_sessions_source_is_supported_when_producer_lag_absent(tmp_path):
    bundle, *_ = _bundle(
        tmp_path, count=1, missing_lag=True, model_changes={"freshness": {"lag_sessions": 0}}
    )
    event = _normalize(bundle)[0]
    assert event["freshness_lag_sessions"] == 0
    assert event["freshness_lag_source"] == "model.freshness.lag_sessions"
