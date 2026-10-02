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
