from __future__ import annotations

import hashlib
import json
import shutil
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


# ---- R3: selection modes, separate time fields, KR prepared input older than K ----

def _wide_calendars(args: dict, start: str = "2026-09-14") -> None:
    """Weekday sessions from ``start`` through D for both markets (K = 09-28)."""
    import r3_world

    for key, market in (("kr_calendar_path", "KR"), ("us_calendar_path", "US")):
        _write(args[key], r3_world._calendar(market, date.fromisoformat(start), D))


def test_selection_records_both_times_and_the_mode_and_no_longer_a_completed_at(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    result = select(**args)
    assert result["selection_mode"] == "scheduled"
    assert result["selected_at"] == "2026-09-29T09:30:00+09:00"
    for market in ("kr", "us"):
        selection = json.loads((args["output_root"] / "2026-09-29" / f"{market}-selection.json").read_text())
        assert selection["selected_at"] == "2026-09-29T09:30:00+09:00"
        assert selection["selection_mode"] == "scheduled"
        assert selection["producer_completed_at"] == "2026-09-29T09:20:00+09:00"
        assert selection["verified_available_by"] == selection["producer_completed_at"]
        assert "completed_at" not in selection
        assert selection["lag_sessions"] == 0
    assert result["markets"]["KR"]["producer_completed_at"] == "2026-09-29T09:20:00+09:00"


@pytest.mark.parametrize("mode,minute,ok", [
    ("scheduled", (9, 30), True), ("scheduled", (10, 0), True), ("scheduled", (10, 5), False),
    ("run_fallback", (10, 5), True), ("run_fallback", (14, 0), True), ("run_fallback", (9, 29), False),
])
def test_selection_time_window_depends_on_the_mode(
        tmp_path: Path, mode: str, minute: tuple, ok: bool) -> None:
    args = {**_inputs(tmp_path), "mode": mode,
            "selected_at": datetime(2026, 9, 29, *minute, tzinfo=SEOUL)}
    if ok:
        assert select(**args)["selection_mode"] == mode
    else:
        with pytest.raises(ValueError, match="selection must occur"):
            select(**args)


def test_unknown_selection_mode_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scheduled or run_fallback"):
        select(**{**_inputs(tmp_path), "mode": "whenever"})


def test_run_fallback_never_admits_an_input_that_finished_after_the_cutoff(tmp_path: Path) -> None:
    args = {**_inputs(tmp_path), "mode": "run_fallback",
            "selected_at": datetime(2026, 9, 29, 10, 5, tzinfo=SEOUL)}
    marker = args["prepared_root"] / "kr" / "score_date=2026-09-28" / "prep_id=fixture" / "completion.json"
    body = json.loads(marker.read_text())
    body["verified_available_by"] = "2026-09-29T09:40:00+09:00"
    _write(marker, body)
    result = select(**args)
    assert result["markets"]["KR"]["status"] == "unavailable" and result["markets"]["US"]["status"] == "ok"


def test_kr_prepared_input_before_k_is_selected_as_stale_and_the_lag_is_recorded(tmp_path: Path) -> None:
    import r3_world

    args = _inputs(tmp_path)
    shutil.rmtree(args["prepared_root"] / "kr")
    _wide_calendars(args)
    for asof in ("2026-09-22", "2026-09-24"):
        r3_world.write_native(args["prepared_root"], "KR", asof, "2026-09-29T09:20:00+09:00")
    result = select(**args)
    assert result["markets"]["KR"] == {
        "status": "stale", "feature_asof_date": "2026-09-24", "lag_sessions": 2,
        "freshness_reason": "KR feature session does not match K",
        "producer_completed_at": "2026-09-29T09:20:00+09:00"}
    selection = json.loads((args["output_root"] / "2026-09-29" / "kr-selection.json").read_text())
    assert selection["freshness_status"] == "stale" and selection["lag_sessions"] == 2
    assert selection["kr_session"] == "2026-09-28"
    assert result["status"] == "selected"  # all three jobs are pinned, KR on its older input


def test_kr_prepared_input_newer_than_k_is_never_selected(tmp_path: Path) -> None:
    import r3_world

    args = _inputs(tmp_path)
    shutil.rmtree(args["prepared_root"] / "kr")
    _wide_calendars(args)
    r3_world.write_native(args["prepared_root"], "KR", "2026-09-29", "2026-09-29T09:20:00+09:00")
    result = select(**args)
    assert result["markets"]["KR"]["status"] == "unavailable"
    assert result["markets"]["KR"]["reason"] == "kr_prepared_newer_than_k"


def test_the_newest_older_prepared_input_wins_over_an_older_one_even_when_finished_later(
        tmp_path: Path) -> None:
    import r3_world

    args = _inputs(tmp_path)
    shutil.rmtree(args["prepared_root"] / "kr")
    _wide_calendars(args)
    r3_world.write_native(args["prepared_root"], "KR", "2026-09-23", "2026-09-29T09:25:00+09:00")
    r3_world.write_native(args["prepared_root"], "KR", "2026-09-25", "2026-09-29T08:00:00+09:00")
    assert select(**args)["markets"]["KR"]["feature_asof_date"] == "2026-09-25"


# ---- R3: what stays blocked on the relaxed (stale, fallback) paths ----

def _stale_kr_args(tmp_path: Path) -> dict:
    import r3_world

    args = _inputs(tmp_path)
    shutil.rmtree(args["prepared_root"] / "kr")
    _wide_calendars(args)
    r3_world.write_native(args["prepared_root"], "KR", "2026-09-25", "2026-09-29T09:20:00+09:00")
    return args


def test_stale_kr_input_still_needs_matching_hashes_and_marker(tmp_path: Path) -> None:
    args = _stale_kr_args(tmp_path)
    assert select(**{**args, "output_root": args["output_root"] / "ok"})["markets"]["KR"]["status"] == "stale"
    directory = args["prepared_root"] / "kr" / "score_date=2026-09-25" / "prep_id=r3-2026-09-25"
    # A changed feature file no longer matches the completion marker's hash.
    (directory / "feature_panel.parquet").write_bytes(b"tampered after completion")
    tampered = select(**{**args, "output_root": args["output_root"] / "tampered"})
    assert tampered["markets"]["KR"]["status"] == "unavailable"
    (directory / "feature_panel.parquet").write_bytes(b"synthetic model input KR 2026-09-25")
    # No completion marker at all: the input does not exist for the selector.
    (directory / "completion.json").unlink()
    missing = select(**{**args, "output_root": args["output_root"] / "missing"})
    assert missing["markets"]["KR"]["status"] == "unavailable"


def test_stale_kr_input_outside_the_calendar_coverage_is_unavailable_not_guessed(
        tmp_path: Path) -> None:
    import r3_world

    args = _inputs(tmp_path)
    shutil.rmtree(args["prepared_root"] / "kr")
    _wide_calendars(args, start="2026-09-24")  # the calendar starts after the prepared input's date
    r3_world.write_native(args["prepared_root"], "KR", "2026-09-22", "2026-09-29T09:20:00+09:00")
    result = select(**args)
    assert result["markets"]["KR"]["status"] == "unavailable"
    assert "KR feature session is not a session of the KR calendar" in result["markets"]["KR"]["reason"]


def test_stale_us_input_still_goes_through_the_serving_gate(tmp_path: Path) -> None:
    """A' (older than E) is shown as stale, but a native that fails the serving gate is not served."""
    import r3_world

    args = _inputs(tmp_path)
    shutil.rmtree(args["prepared_root"] / "us")
    _wide_calendars(args)
    directory = r3_world.write_native(
        args["prepared_root"], "US", "2026-09-23", "2026-09-29T09:20:00+09:00",
        extra_native={"raw_feature_parity_status": "failed"})
    served = select(**{**args, "output_root": args["output_root"] / "failed-parity"})
    assert served["markets"]["US"]["status"] == "unavailable"
    shutil.rmtree(directory.parent)
    r3_world.write_native(args["prepared_root"], "US", "2026-09-23", "2026-09-29T09:20:00+09:00")
    ok = select(**{**args, "output_root": args["output_root"] / "no-parity-claim"})
    assert ok["markets"]["US"]["status"] == "stale" and ok["markets"]["US"]["lag_sessions"] == 3


# ---- 운영 배치: prepared/us가 prepared 밖 디렉터리로 가는 symlink (provision_serving) ----

def _linked_us_world(tmp_path: Path) -> dict:
    """prepared/kr은 실제 디렉터리, prepared/us는 레이크 쪽 디렉터리로 가는 symlink."""
    args = _inputs(tmp_path)
    lake_us = tmp_path / "lake" / "us_scoring_daily_v1" / "prepared"
    lake_us.parent.mkdir(parents=True)
    shutil.move(str(args["prepared_root"] / "us"), str(lake_us))
    (args["prepared_root"] / "us").symlink_to(lake_us)
    assert (args["prepared_root"] / "us").is_symlink()
    assert not (args["prepared_root"] / "kr").is_symlink()
    return args


def _jobs_from(selected: dict, infer) -> list[InferenceJob]:
    return [InferenceJob(market=raw["market"], model_id=raw["model_id"],
            model_version=raw["model_version"], prepared_input=Path(raw["prepared_input"]),
            input_sha256=raw["input_sha256"], prepared_manifest=Path(raw["prepared_manifest"]),
            prepared_manifest_sha256=raw["prepared_manifest_sha256"],
            native_manifest=Path(raw["native_manifest"]),
            native_manifest_sha256=raw["native_manifest_sha256"],
            bundle_path=Path(raw["bundle_path"]), bundle_sha256=raw["bundle_sha256"],
            code_path=Path(raw["code_path"]), code_sha256=raw["code_sha256"],
            code_files=tuple((Path(i["path"]), i["sha256"]) for i in raw["code_files"]),
            infer=infer) for raw in selected["jobs"]]


def _ok_infer(context):
    report = report_template(market=context.market, report_date=context.report_date.isoformat(),
        decision_at=context.decision_at, feature_asof_date=context.feature_asof_date,
        model_id=context.model_id, model_version=context.model_version)
    report["status"] = "partial"
    report["synthetic_fixture"] = True
    return report


def test_run_accepts_us_inputs_behind_a_prepared_symlink(tmp_path: Path) -> None:
    args = _linked_us_world(tmp_path)
    selected = select(**args)
    assert selected["status"] == "selected"
    # select는 symlink를 푼 경로를 jobs에 쓰므로 US 파일은 prepared 바깥이다(결함 재현 조건).
    us_input = Path(next(j["prepared_input"] for j in selected["jobs"] if j["market"] == "US"))
    assert args["prepared_root"].resolve() not in us_input.parents
    jobs = _jobs_from(selected, _ok_infer)
    decision = datetime(2026, 9, 29, 10, tzinfo=SEOUL)
    for job in jobs:
        _prepared_metadata(job, args["prepared_root"], D, decision)
    report = run_daily(report_date=D, decision_at=decision, prepared_root=args["prepared_root"],
        run_root=tmp_path / "linked-run", jobs=jobs, fixture_mode=True, now=decision)
    assert report["failures"] == []
    assert len(report["markets"]) == 3


def test_prepared_symlink_does_not_open_files_outside_the_market_directory(tmp_path: Path) -> None:
    args = _linked_us_world(tmp_path)
    selected = select(**args)
    jobs = _jobs_from(selected, _ok_infer)
    us = next(job for job in jobs if job.market == "US")
    stray = tmp_path / "lake" / "other" / "features.parquet"
    stray.parent.mkdir(parents=True)
    shutil.copy(us.prepared_input, stray)
    decision = datetime(2026, 9, 29, 10, tzinfo=SEOUL)
    outside = InferenceJob(**{**us.__dict__, "prepared_input": stray})
    with pytest.raises(ValueError, match="outside its allowed directory"):
        _prepared_metadata(outside, args["prepared_root"], D, decision)
    # symlink 파일 자체는 여전히 거부한다.
    link = us.prepared_input.parent / "alias.parquet"
    link.symlink_to(us.prepared_input)
    aliased = InferenceJob(**{**us.__dict__, "prepared_input": link})
    with pytest.raises(ValueError, match="symlink"):
        _prepared_metadata(aliased, args["prepared_root"], D, decision)
