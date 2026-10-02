from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from modeler.serving.daily_inputs import select
from modeler.serving.orchestration import InferenceJob, _prepared_metadata, code_inventory_sha256, run_daily
from modeler.serving.schema import report_template

SEOUL = ZoneInfo("Asia/Seoul")
D = date(2026, 9, 29)


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    return path


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calendar(market: str) -> dict:
    return {"market": market, "timezone": "Asia/Seoul" if market == "KR" else "America/New_York",
            "coverage_start": "2026-09-28", "coverage_end": "2026-09-29",
            "sessions": ["2026-09-28", "2026-09-29"],
            "default_open_at": "09:00" if market == "KR" else "09:30",
            "default_close_at": "15:30" if market == "KR" else "16:00",
            "overrides": [], "unconfirmed_dates": []}


def _native(root: Path, market: str, *, ready_at: str = "2026-09-29T09:20:00+09:00") -> None:
    directory = root / market.lower() / "score_date=2026-09-28" / "prep_id=fixture"
    directory.mkdir(parents=True)
    feature_name = "feature_panel.parquet" if market == "KR" else "features.parquet"
    native_name = "prepare_manifest.json" if market == "KR" else "manifest.json"
    feature = directory / feature_name
    feature.write_bytes(b"synthetic features " + market.encode())
    native = _write(directory / native_name, {"market": market,
        "feature_asof_date": "2026-09-28", "input_sha256": _hash(feature),
        "features_sha256": _hash(feature),
        "availability_evidence_type": "prepared_features_completion",
        "synthetic_fixture": True})
    _write(directory / "completion.json", {"schema_version": "prepared-features-completion.v1",
        "verified_available_by": ready_at,
        "availability_evidence_type": "prepared_features_completion",
        "features_sha256": _hash(feature), "native_prepare_manifest_sha256": _hash(native)})


def _release(root: Path) -> Path:
    release_root = root / "release-1"
    code = release_root / "src" / "adapter.py"
    code.parent.mkdir(parents=True)
    code.write_text("# synthetic adapter\n")
    code_sha = _hash(code)
    jobs = []
    for market, model_id in (("KR", "kr_daily_h20_v1"),
                             ("US", "us_exploratory_20260929_r1_lightgbm"),
                             ("US", "us_exploratory_20260929_r1_ridge")):
        bundle = release_root / "bundles" / (model_id + ".json")
        _write(bundle, {"synthetic_fixture": True, "model_id": model_id})
        jobs.append({"market": market, "model_id": model_id, "model_version": "1",
            "entrypoint": "modeler.serving.adapters:infer_kr_daily" if market == "KR"
                          else "modeler.serving.adapters:infer_us_model",
            "bundle_path": str(bundle.relative_to(release_root)), "bundle_sha256": _hash(bundle),
            "code_path": str(code.relative_to(release_root)), "code_path_sha256": code_sha,
            "code_files": [{"path": str(code.relative_to(release_root)), "sha256": code_sha}],
            "code_sha256": code_inventory_sha256(((code, code_sha),))})
    return _write(release_root / "release.json", {"schema_version": "daily-briefing-release.v1",
        "frozen": True, "synthetic_fixture": True, "release_root": str(release_root), "jobs": jobs})


def _inputs(tmp_path: Path, *, expected: str | None = "2026-09-28") -> dict:
    prepared = tmp_path / "prepared"
    _native(prepared, "KR")
    _native(prepared, "US")
    return {"report_date": D, "selected_at": datetime(2026, 9, 29, 9, 30, tzinfo=SEOUL),
        "prepared_root": prepared, "output_root": prepared / "selected",
        "release_manifest": _release(tmp_path),
        "kr_calendar_path": _write(tmp_path / "kr-calendar.json", _calendar("KR")),
        "us_calendar_path": _write(tmp_path / "us-calendar.json", _calendar("US")),
        "us_expected_path": _write(tmp_path / "us-expected.json", {
            "schema_version": "us-expected-source.v1", "reviewed_status": "synthetic_fixture",
            "source_reference": "synthetic-fixture-only",
            "expected_session_by_report_date": {D.isoformat(): expected} if expected else {},
            "market_lag_limit_sessions": 1})}


def test_select_pins_three_jobs_and_is_idempotent(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    result = select(**args)
    assert result["status"] == "selected"
    assert len(result["jobs"]) == 3
    assert result["markets"]["KR"]["feature_asof_date"] == "2026-09-28"
    assert result["markets"]["US"]["status"] == "ok"
    args["selected_at"] = datetime(2026, 9, 29, 10, 5, tzinfo=SEOUL)
    assert select(**args) == result


def test_us_expected_session_is_never_guessed_from_actual(tmp_path: Path) -> None:
    args = _inputs(tmp_path, expected=None)
    result = select(**args)
    assert result["status"] == "partial"
    assert len(result["jobs"]) == 1
    assert result["markets"]["US"]["reason"] == "us_expected_source_schedule_missing"


def test_late_native_completion_is_excluded(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    _native_late = args["prepared_root"] / "kr" / "score_date=2026-09-28" / "prep_id=fixture" / "completion.json"
    marker = json.loads(_native_late.read_text())
    marker["verified_available_by"] = "2026-09-29T09:30:01+09:00"
    _write(_native_late, marker)
    result = select(**args)
    assert result["status"] == "partial"
    assert len(result["jobs"]) == 2
    assert result["markets"]["KR"]["status"] == "unavailable"


def test_missing_us_policy_keeps_kr_job(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    args["us_expected_path"].unlink()
    result = select(**args)
    assert result["status"] == "partial"
    assert [job["market"] for job in result["jobs"]] == ["KR"]
    assert result["markets"]["US"]["status"] == "unavailable"


def test_unreviewed_us_expectation_keeps_kr_job(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    policy = json.loads(args["us_expected_path"].read_text())
    policy["reviewed_status"] = "unreviewed"
    _write(args["us_expected_path"], policy)
    result = select(**args)
    assert [job["market"] for job in result["jobs"]] == ["KR"]


def test_existing_selection_rejects_tampered_manifest(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    result = select(**args)
    selected = Path(result["jobs"][0]["prepared_manifest"])
    selected.write_text("{}")
    import pytest
    with pytest.raises(ValueError, match="changed"):
        select(**args)


@pytest.mark.parametrize("field,value", [
    ("raw_feature_parity_status", "failed"),
    ("diagnostic_only", True),
    ("serving_eligible", False),
])
def test_diagnostic_us_native_is_excluded_without_excluding_kr(
        tmp_path: Path, field: str, value: object) -> None:
    args = _inputs(tmp_path)
    native_path = args["prepared_root"] / "us/score_date=2026-09-28/prep_id=fixture/manifest.json"
    native = json.loads(native_path.read_text())
    native[field] = value
    _write(native_path, native)
    marker_path = native_path.parent / "completion.json"
    marker = json.loads(marker_path.read_text())
    marker["native_prepare_manifest_sha256"] = _hash(native_path)
    _write(marker_path, marker)
    result = select(**args)
    assert result["status"] == "partial"
    assert [job["market"] for job in result["jobs"]] == ["KR"]
    assert result["markets"]["US"]["status"] == "unavailable"


@pytest.mark.parametrize("field,value", [
    ("raw_feature_parity_status", "failed"),
    ("diagnostic_only", True),
    ("serving_eligible", False),
])
def test_direct_runner_rejects_hash_consistent_diagnostic_us(
        tmp_path: Path, field: str, value: object) -> None:
    args = _inputs(tmp_path)
    selected = select(**args)
    raw_jobs = selected["jobs"]
    native_path = Path(next(job["native_manifest"] for job in raw_jobs if job["market"] == "US"))
    native = json.loads(native_path.read_text())
    native[field] = value
    _write(native_path, native)
    marker_path = native_path.parent / "completion.json"
    marker = json.loads(marker_path.read_text())
    marker["native_prepare_manifest_sha256"] = _hash(native_path)
    _write(marker_path, marker)
    selection_path = Path(next(job["prepared_manifest"] for job in raw_jobs if job["market"] == "US"))
    selection = json.loads(selection_path.read_text())
    selection["native_prepare_manifest_sha256"] = _hash(native_path)
    selection["completion_marker_sha256"] = _hash(marker_path)
    selection["availability_evidence"]["native_prepare_manifest_sha256"] = _hash(native_path)
    selection["availability_evidence"]["completion_marker_sha256"] = _hash(marker_path)
    _write(selection_path, selection)

    def infer(context):
        report = report_template(market=context.market, report_date=context.report_date.isoformat(),
            decision_at=context.decision_at, feature_asof_date=context.feature_asof_date,
            model_id=context.model_id, model_version=context.model_version)
        report["status"] = "partial"
        report["synthetic_fixture"] = True
        return report

    jobs = []
    for raw in raw_jobs:
        is_us = raw["market"] == "US"
        jobs.append(InferenceJob(market=raw["market"], model_id=raw["model_id"],
            model_version=raw["model_version"], prepared_input=Path(raw["prepared_input"]),
            input_sha256=raw["input_sha256"], prepared_manifest=Path(raw["prepared_manifest"]),
            prepared_manifest_sha256=_hash(selection_path) if is_us else raw["prepared_manifest_sha256"],
            native_manifest=Path(raw["native_manifest"]),
            native_manifest_sha256=_hash(native_path) if is_us else raw["native_manifest_sha256"],
            bundle_path=Path(raw["bundle_path"]), bundle_sha256=raw["bundle_sha256"],
            code_path=Path(raw["code_path"]), code_sha256=raw["code_sha256"],
            code_files=tuple((Path(item["path"]), item["sha256"]) for item in raw["code_files"]),
            infer=infer))
    decision = datetime(2026, 9, 29, 10, tzinfo=SEOUL)
    for job in jobs:
        if job.market == "US":
            with pytest.raises(ValueError, match="diagnostic native input"):
                _prepared_metadata(job, args["prepared_root"], D, decision)
    report = run_daily(report_date=D, decision_at=decision, prepared_root=args["prepared_root"],
        run_root=tmp_path / "direct-run", jobs=jobs, fixture_mode=True, now=decision)
    assert [(row["market"], row["model_id"]) for row in report["markets"]] == [
        ("KR", "kr_daily_h20_v1")]
    assert len(report["failures"]) == 2
