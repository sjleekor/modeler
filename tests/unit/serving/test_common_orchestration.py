from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from modeler.serving.orchestration import InferenceJob, code_inventory_sha256, retry_publication, run_daily
from modeler.serving.schema import report_template

SEOUL = ZoneInfo("Asia/Seoul")
REPORT_DATE = date(2026, 9, 29)
DECISION = datetime(2026, 9, 29, 10, tzinfo=SEOUL)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _job(tmp_path: Path, *, market="KR", model_id="kr_daily_h20_v1", source_time="2026-09-29T09:25:00+09:00"):
    prepared = tmp_path / "prepared"
    prepared.mkdir(exist_ok=True)
    input_path = prepared / "input.bin"
    input_path.write_bytes(b"synthetic feature matrix")
    input_hash = _hash(input_path)
    native = prepared / "native.json"
    native.write_text(json.dumps({"market": market, "feature_asof_date": "2026-09-28",
        "input_sha256": input_hash, "features_sha256": input_hash,
        "availability_evidence_type": "prepared_features_completion",
        "synthetic_fixture": True}))
    marker = prepared / "completion.json"
    marker.write_text(json.dumps({"schema_version": "prepared-features-completion.v1",
        "verified_available_by": source_time,
        "availability_evidence_type": "prepared_features_completion",
        "features_sha256": input_hash, "native_prepare_manifest_sha256": _hash(native)}))
    marker_hash = _hash(marker)
    calendar = {"market": "KR", "timezone": "Asia/Seoul", "coverage_start": "2026-09-28",
        "coverage_end": "2026-09-29", "sessions": ["2026-09-28", "2026-09-29"],
        "default_open_at": "09:00", "default_close_at": "15:30", "overrides": [],
        "unconfirmed_dates": []}
    manifest = prepared / "prepared.json"
    manifest.write_text(json.dumps({
        "market": market, "report_date": REPORT_DATE.isoformat(),
        "feature_asof_date": "2026-09-28", "input_sha256": input_hash,
        "native_prepare_manifest_sha256": _hash(native),
        "calendar": calendar, "kr_session": "2026-09-28",
        "input_cutoff": "2026-09-29T09:30:00+09:00", "verified_available_by": source_time,
        "completion_marker_path": marker.name, "completion_marker_sha256": marker_hash,
        "availability_evidence_type": "prepared_features_completion",
        "availability_evidence": {"features_sha256": input_hash,
            "native_prepare_manifest_sha256": _hash(native),
            "completion_marker_sha256": marker_hash}, "source_first_available_at": None,
        "completed_at": "2026-09-29T09:35:00+09:00", "freshness_status": "ok",
    }))
    bundle = tmp_path / "model.bundle"
    bundle.write_bytes(b"synthetic immutable model bundle")
    code = tmp_path / "inference.py"
    code.write_text("synthetic inference implementation")
    code_files = ((code, _hash(code)),)
    code_hash = code_inventory_sha256(code_files)
    model_version = "1"

    def infer(context) -> dict:
        assert context.prepared_input.read_bytes() == input_path.read_bytes()
        assert context.bundle_manifest.read_bytes() == bundle.read_bytes()
        assert context.prepared_manifest == manifest
        assert context.native_manifest == native
        assert context.input_sha256 == _hash(context.prepared_input)
        assert context.selection["input_sha256"] == context.input_sha256
        assert context.native_preparation["features_sha256"] == context.input_sha256
        assert context.freshness_status == "ok"
        assert context.fixture_mode is True
        assert context.selection["market"] == market
        assert context.native_preparation["market"] == market
        assert context.code_sha256 == code_hash
        report = report_template(market=market, report_date=REPORT_DATE.isoformat(),
            decision_at=DECISION, feature_asof_date="2026-09-28", model_id=model_id,
            model_version=model_version)
        report["status"] = "ok"
        return report

    job = InferenceJob(market=market, model_id=model_id, model_version=model_version,
        prepared_input=input_path, input_sha256=input_hash, prepared_manifest=manifest,
        prepared_manifest_sha256=_hash(manifest), native_manifest=native,
        native_manifest_sha256=_hash(native), bundle_path=bundle, bundle_sha256=_hash(bundle),
        code_path=code, code_sha256=code_hash, code_files=code_files, infer=infer)
    return job, input_path, manifest


def test_code_inventory_pins_every_runtime_source_file(tmp_path):
    first = tmp_path / "adapter.py"
    second = tmp_path / "feature.py"
    first.write_text("adapter v1")
    second.write_text("feature v1")
    records = ((first, _hash(first)), (second, _hash(second)))
    assert code_inventory_sha256(records) == code_inventory_sha256(tuple(reversed(records)))
    first.write_text("adapter changed after inventory")
    assert _hash(first) != records[0][1]


def test_run_daily_fails_closed_when_a_pinned_runtime_file_changes(tmp_path):
    job, _, _ = _job(tmp_path)
    job.code_path.write_text("runtime source changed")
    envelope = run_daily(report_date=REPORT_DATE, decision_at=DECISION,
        prepared_root=tmp_path / "prepared", run_root=tmp_path / "run", jobs=[job],
        expected_identities={("KR", "kr_daily_h20_v1")}, fixture_mode=True,
        now=datetime(2026, 9, 29, 10, 2, tzinfo=SEOUL))
    assert envelope["status"] == "failed"
    assert envelope["failures"][0]["error_class"] == "ValueError"


def test_run_daily_rejects_a_selection_for_the_wrong_feature_session(tmp_path):
    job, _, selection_path = _job(tmp_path)
    calls = []
    original = job.infer
    job = InferenceJob(**{**job.__dict__, "infer": lambda context: calls.append(1) or original(context)})
    selection = json.loads(selection_path.read_text())
    native = json.loads(job.native_manifest.read_text())
    native["feature_asof_date"] = "2026-09-27"
    job.native_manifest.write_text(json.dumps(native))
    marker_path = job.native_manifest.parent / "completion.json"
    marker = json.loads(marker_path.read_text())
    marker["native_prepare_manifest_sha256"] = _hash(job.native_manifest)
    marker_path.write_text(json.dumps(marker))
    marker_hash = _hash(marker_path)
    selection["native_prepare_manifest_sha256"] = _hash(job.native_manifest)
    selection["feature_asof_date"] = "2026-09-27"
    selection["completion_marker_sha256"] = marker_hash
    selection["availability_evidence"]["native_prepare_manifest_sha256"] = _hash(job.native_manifest)
    selection["availability_evidence"]["completion_marker_sha256"] = marker_hash
    selection_path.write_text(json.dumps(selection))
    job = InferenceJob(**{**job.__dict__, "native_manifest_sha256": _hash(job.native_manifest),
                          "prepared_manifest_sha256": _hash(selection_path)})
    result = run_daily(report_date=REPORT_DATE, decision_at=DECISION,
        prepared_root=tmp_path / "prepared", run_root=tmp_path / "run", jobs=[job],
        expected_identities={("KR", "kr_daily_h20_v1")}, fixture_mode=True,
        now=datetime(2026, 9, 29, 10, 2, tzinfo=SEOUL))
    assert calls == []
    assert result["status"] == "failed"
    assert result["failures"][0]["error_class"] == "ValueError"


def test_run_daily_pins_and_reuses_success_then_keeps_new_hash_as_new_revision(tmp_path):
    job, input_path, manifest = _job(tmp_path)
    calls = []
    original_infer = job.infer

    def counted(*args):
        calls.append(1)
        return original_infer(*args)

    job = InferenceJob(**{**job.__dict__, "infer": counted})
    run_root = tmp_path / "run"
    kwargs = dict(report_date=REPORT_DATE, decision_at=DECISION,
        prepared_root=tmp_path / "prepared", run_root=run_root, jobs=[job],
        expected_identities={("KR", "kr_daily_h20_v1")}, fixture_mode=True,
        now=datetime(2026, 9, 29, 10, 2, tzinfo=SEOUL))
    first = run_daily(**kwargs)
    first_paths = list((run_root / "inference" / REPORT_DATE.isoformat() / "KR" / job.model_id).glob("*.json"))
    second = run_daily(**kwargs)
    assert first["markets"][0]["inference_started_at"] == second["markets"][0]["inference_started_at"]
    assert len(calls) == 1
    assert len(first_paths) == 1
    input_path.write_bytes(b"synthetic corrected feature matrix")
    new_hash = _hash(input_path)
    manifest_data = json.loads(manifest.read_text())
    manifest_data["input_sha256"] = new_hash
    native_path = job.native_manifest
    native_data = json.loads(native_path.read_text())
    native_data["input_sha256"] = new_hash
    native_data["features_sha256"] = new_hash
    native_path.write_text(json.dumps(native_data))
    marker_path = native_path.parent / "completion.json"
    marker_data = json.loads(marker_path.read_text())
    marker_data["features_sha256"] = new_hash
    manifest_data["native_prepare_manifest_sha256"] = _hash(native_path)
    marker_data["native_prepare_manifest_sha256"] = _hash(native_path)
    marker_path.write_text(json.dumps(marker_data))
    marker_hash = _hash(marker_path)
    manifest_data["completion_marker_sha256"] = marker_hash
    manifest_data["availability_evidence"] = {"features_sha256": new_hash,
        "native_prepare_manifest_sha256": _hash(native_path),
        "completion_marker_sha256": marker_hash}
    manifest.write_text(json.dumps(manifest_data))
    new_job = InferenceJob(**{**job.__dict__, "input_sha256": new_hash,
                              "prepared_manifest_sha256": _hash(manifest),
                              "native_manifest_sha256": _hash(native_path)})
    run_daily(**{**kwargs, "jobs": [new_job]})
    all_paths = list((run_root / "inference" / REPORT_DATE.isoformat() / "KR" / job.model_id).glob("*.json"))
    assert len(calls) == 2
    assert len(all_paths) == 2


def test_run_daily_blocks_source_after_cutoff_and_records_failure_separately(tmp_path):
    job, _, _ = _job(tmp_path, source_time="2026-09-29T09:31:00+09:00")
    calls = []
    original = job.infer
    job = InferenceJob(**{**job.__dict__, "infer": lambda *args: calls.append(1) or original(*args)})
    envelope = run_daily(report_date=REPORT_DATE, decision_at=DECISION,
        prepared_root=tmp_path / "prepared", run_root=tmp_path / "run", jobs=[job],
        expected_identities={("KR", "kr_daily_h20_v1")}, fixture_mode=True,
        now=datetime(2026, 9, 29, 10, 2, tzinfo=SEOUL))
    state = json.loads((tmp_path / "run" / "run-state.json").read_text())
    assert calls == []
    assert envelope["status"] == "failed"
    assert envelope["failures"][0]["error_class"] == "ValueError"
    assert state["jobs"][0]["status"] == "failed"


def test_retry_publication_uses_saved_report_and_never_calls_inference(tmp_path):
    job, _, _ = _job(tmp_path)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps({"report_date": REPORT_DATE.isoformat(), "markets": [], "synthetic_fixture": True}))
    seen = []
    result = retry_publication(report_path=report_path,
        publish=lambda saved: seen.append(saved["report_date"]) or {"status": "ok"})
    assert seen == [REPORT_DATE.isoformat()]
    assert result["status"] == "ok"
    assert json.loads((tmp_path / "publication-state.json").read_text())["status"] == "ok"


def test_run_daily_requires_fixture_or_replay_before_future_cutoff(tmp_path):
    job, _, _ = _job(tmp_path)
    try:
        run_daily(report_date=REPORT_DATE, decision_at=DECISION,
            prepared_root=tmp_path / "prepared", run_root=tmp_path / "run", jobs=[job],
            expected_identities={("KR", "kr_daily_h20_v1")}, now=datetime(2026, 9, 29, 9, tzinfo=SEOUL))
    except ValueError as exc:
        assert "fixture_mode" in str(exc)
    else:
        raise AssertionError("future decision cutoff must require explicit fixture/replay mode")


# ---- R3: time checks (change 5), stale KR input (3), envelope status (6), five-session cap (Q5) ----

def _rewrite_manifest(job, manifest: Path, **changes) -> InferenceJob:
    data = json.loads(manifest.read_text())
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    manifest.write_text(json.dumps(data))
    return InferenceJob(**{**job.__dict__, "prepared_manifest_sha256": _hash(manifest)})


def _metadata(job, root: Path):
    from modeler.serving.orchestration import _prepared_metadata

    return _prepared_metadata(job, root, REPORT_DATE, DECISION)


def test_a_selection_made_after_ten_is_refused_unless_the_run_made_it_itself(tmp_path):
    import pytest

    job, _, manifest = _job(tmp_path)
    late = "2026-09-29T10:05:00+09:00"
    scheduled = _rewrite_manifest(job, manifest, selected_at=late, completed_at=None,
                                  selection_mode="scheduled")
    with pytest.raises(ValueError, match="completed after decision_at"):
        _metadata(scheduled, tmp_path / "prepared")
    fallback = _rewrite_manifest(job, manifest, selection_mode="run_fallback")
    metadata = _metadata(fallback, tmp_path / "prepared")
    assert metadata["selection_mode"] == "run_fallback" and metadata["selected_at"] == late
    assert metadata["producer_completed_at"] == "2026-09-29T09:25:00+09:00"
    # A legacy selection (one completed_at, no mode) still reads as a scheduled one.
    legacy = _rewrite_manifest(job, manifest, selected_at=None, selection_mode=None,
                               completed_at="2026-09-29T09:35:00+09:00")
    assert _metadata(legacy, tmp_path / "prepared")["selection_mode"] == "scheduled"
    for bad in ("sometimes", None):
        wrong = _rewrite_manifest(job, manifest, selection_mode=bad if bad else "")
        with pytest.raises(ValueError, match="scheduled or run_fallback"):
            _metadata(wrong, tmp_path / "prepared")


def test_an_input_that_finished_after_the_cutoff_is_refused_in_every_mode(tmp_path):
    import pytest

    job, _, manifest = _job(tmp_path, source_time="2026-09-29T09:31:00+09:00")
    for mode in ("scheduled", "run_fallback"):
        candidate = _rewrite_manifest(job, manifest, selection_mode=mode,
                                      selected_at="2026-09-29T09:45:00+09:00", completed_at=None)
        with pytest.raises(ValueError, match="after the 09:30 input cutoff"):
            _metadata(candidate, tmp_path / "prepared")


def test_a_selection_cannot_claim_another_producer_completion_time(tmp_path):
    import pytest

    job, _, manifest = _job(tmp_path)
    forged = _rewrite_manifest(job, manifest, producer_completed_at="2026-09-29T08:00:00+09:00")
    with pytest.raises(ValueError, match="producer_completed_at disagrees"):
        _metadata(forged, tmp_path / "prepared")


def test_run_daily_records_selection_fields_in_the_report_quality(tmp_path):
    job, _, manifest = _job(tmp_path)
    job = _rewrite_manifest(job, manifest, selection_mode="run_fallback",
                            selected_at="2026-09-29T10:05:00+09:00", completed_at=None)
    envelope = run_daily(report_date=REPORT_DATE, decision_at=DECISION,
        prepared_root=tmp_path / "prepared", run_root=tmp_path / "run", jobs=[job],
        expected_identities={("KR", "kr_daily_h20_v1")}, fixture_mode=True,
        now=datetime(2026, 9, 29, 10, 6, tzinfo=SEOUL))
    quality = envelope["markets"][0]["quality"]
    assert quality["selection_mode"] == "run_fallback"
    assert quality["selected_at"] == "2026-09-29T10:05:00+09:00"
    assert quality["producer_completed_at"] == "2026-09-29T09:25:00+09:00"
    assert quality["lag_sessions"] == 0 and quality["freshness_status"] == "ok"


def _report(status, market="KR", model_id="kr_daily_h20_v1", *, rankings=0, publication="unresolved"):
    report = report_template(market=market, report_date=REPORT_DATE.isoformat(), decision_at=DECISION,
                             feature_asof_date="2026-09-28", model_id=model_id, model_version="1")
    report["status"] = status
    report["rankings"] = [{"rank": rank, "symbol": f"S{rank}", "name": "n", "score": 1.0}
                          for rank in range(1, rankings + 1)]
    report["publication"] = {"status": publication, "evidence": []}
    return report


US_IDS = ("us_exploratory_20260929_r1_lightgbm", "us_exploratory_20260929_r1_ridge")


def _status(*reports, failures=()):
    from modeler.serving.orchestration import combine_reports

    return combine_reports(report_date=REPORT_DATE, decision_at=DECISION, reports=list(reports),
                           failures=failures)["status"]


def test_envelope_status_is_ok_only_when_every_section_is_ok():
    kr = _report("ok")
    us = [_report("ok", "US", model) for model in US_IDS]
    # Publication is unresolved on every private-repo report: that no longer keeps the unit from ok.
    assert _status(kr, *us) == "ok"
    assert _status(kr, *us[:1], failures=[{"market": "US", "model_id": US_IDS[1],
                                           "error_class": "ValueError"}]) == "partial"
    assert _status(_report("partial"), *us) == "partial"
    assert _status(_report("stale"), *us) == "partial"
    assert _status(kr, _report("stale", "US", US_IDS[0]), us[1]) == "partial"


def test_envelope_is_failed_only_when_no_section_produced_anything():
    """The 10-05 defect: only KR succeeded and the envelope said failed."""
    assert _status(_report("partial")) == "partial"  # KR alone, US missing -> MissingInference x2
    assert _status() == "failed"
    assert _status(_report("unavailable"), _report("failed", "US", US_IDS[0]),
                   _report("withheld", "US", US_IDS[1])) == "failed"
    assert _status(failures=[{"market": "KR", "model_id": "kr_daily_h20_v1",
                              "error_class": "RunnerTimeout"}]) == "failed"


def test_session_lag_reads_kr_delivery_lag_and_us_market_lag():
    from modeler.serving.orchestration import session_lag

    assert session_lag("KR", {"delivery_lag": 3, "market_lag": None}) == 3
    assert session_lag("US", {"delivery_lag": 1, "market_lag": 4}) == 4
    assert session_lag("US", {"market_lag": None}) is None
    assert session_lag("KR", {"delivery_lag": True}) is None


def test_stale_reports_keep_the_ranking_up_to_five_sessions_and_drop_it_beyond():
    from modeler.serving.orchestration import _annotate_selection

    meta = {"freshness_status": "stale", "selection_mode": "scheduled",
            "selected_at": "2026-09-29T09:30:00+09:00",
            "producer_completed_at": "2026-09-29T09:20:00+09:00"}
    five = _report("partial", rankings=3)
    _annotate_selection(five, {**meta, "freshness": {"delivery_lag": 5}}, "KR")
    assert five["status"] == "stale" and len(five["rankings"]) == 3
    assert five["quality"]["lag_sessions"] == 5 and five["quality"]["status_before_stale"] == "partial"
    six = _report("partial", rankings=3)
    _annotate_selection(six, {**meta, "freshness": {"delivery_lag": 6}}, "KR")
    assert six["status"] == "stale" and six["rankings"] == []
    assert six["quality"]["rankings_withheld"] == {
        "reason": "stale_lag_exceeds_limit", "lag_sessions": 6, "limit_sessions": 5, "ranking_count": 3}
    us = _report("stale", "US", US_IDS[0], rankings=2)
    _annotate_selection(us, {**meta, "freshness": {"market_lag": 7, "delivery_lag": 1}}, "US")
    assert us["rankings"] == [] and us["quality"]["lag_sessions"] == 7
    fresh = _report("ok", rankings=2)
    _annotate_selection(fresh, {**meta, "freshness_status": "ok", "freshness": {"delivery_lag": 0}}, "KR")
    assert fresh["status"] == "ok" and len(fresh["rankings"]) == 2
    assert "rankings_withheld" not in fresh["quality"]
