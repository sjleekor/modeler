"""Run the real coordinator CLI in a frozen, synthetic-only local release."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import nullcontext
from datetime import date
from pathlib import Path

import pytest

from modeler.serving.orchestration import code_inventory_sha256

D = date(2026, 9, 29)
PROJECT = Path(__file__).resolve().parents[3]
MODULES = (
    "modeler/__init__.py", "modeler/serving/__init__.py",
    "modeler/serving/runner.py", "modeler/serving/orchestration.py",
    "modeler/serving/opening_validation.py", "modeler/serving/calendars.py",
    "modeler/serving/freshness.py", "modeler/serving/schema.py",
    "modeler/serving/daily_inputs.py", "modeler/serving/daily_coordinator.py",
    "modeler/serving/daily_wrapper.py",
    "modeler/serving/daily_opening.py",
    "modeler/serving/opening_prepare.py",
    "modeler/serving/runtime_contract.py", "modeler/reporting/__init__.py",
    "modeler/reporting/site.py",
)
ADAPTER = '''from modeler.serving.schema import report_template

def _infer(context):
    report = report_template(market=context.market, report_date=context.report_date.isoformat(),
        decision_at=context.decision_at, feature_asof_date=context.feature_asof_date,
        model_id=context.model_id, model_version=context.model_version)
    report["status"] = "partial"
    report["synthetic_fixture"] = True
    return report

def infer_kr_daily(context):
    return _infer(context)

def infer_us_model(context):
    return _infer(context)
'''


def _write(path: Path, body: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calendar(market: str) -> dict:
    return {"market": market, "timezone": "Asia/Seoul" if market == "KR" else "America/New_York",
        "coverage_start": "2026-09-28", "coverage_end": "2026-09-29",
        "sessions": ["2026-09-28", "2026-09-29"],
        "default_open_at": "09:00" if market == "KR" else "09:30",
        "default_close_at": "15:30" if market == "KR" else "16:00",
        "overrides": [], "unconfirmed_dates": []}


def _native(prepared: Path, market: str) -> None:
    directory = prepared / market.lower() / "score_date=2026-09-28" / "prep_id=fixture"
    directory.mkdir(parents=True)
    feature = directory / ("feature_panel.parquet" if market == "KR" else "features.parquet")
    feature.write_bytes(b"synthetic model input: " + market.encode())
    native = _write(directory / ("prepare_manifest.json" if market == "KR" else "manifest.json"),
        {"market": market, "feature_asof_date": "2026-09-28", "input_sha256": _sha(feature),
         "features_sha256": _sha(feature), "synthetic_fixture": True,
         "availability_evidence_type": "prepared_features_completion"})
    _write(directory / "completion.json", {"schema_version": "prepared-features-completion.v1",
        "verified_available_by": "2026-09-29T09:20:00+09:00",
        "availability_evidence_type": "prepared_features_completion",
        "features_sha256": _sha(feature), "native_prepare_manifest_sha256": _sha(native)})


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    release = tmp_path / "frozen-release"
    source = release / "src"
    for relative in MODULES:
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PROJECT / "src" / relative, target)
    adapter = source / "modeler" / "serving" / "adapters.py"
    adapter.write_text(ADAPTER)
    shutil.copy2(PROJECT / "uv.lock", release / "uv.lock")
    shutil.copy2(PROJECT / "deploy" / "prod" / "runtime-verified-sj2-20260930.json",
                 release / "runtime.json")
    cards = _write(release / "model-cards.json", {
        model_id: {"title": model_id, "summary": "Synthetic fixture model card."}
        for model_id in ("kr_daily_h20_v1", "us_exploratory_20260929_r1_lightgbm",
                         "us_exploratory_20260929_r1_ridge")})
    files = tuple((path, _sha(path)) for path in source.rglob("*.py"))
    inventory = [{"path": str(path.relative_to(release)), "sha256": digest}
                 for path, digest in files]
    code_sha = code_inventory_sha256(files)
    jobs = []
    for market, model_id in (("KR", "kr_daily_h20_v1"),
                             ("US", "us_exploratory_20260929_r1_lightgbm"),
                             ("US", "us_exploratory_20260929_r1_ridge")):
        bundle = _write(release / "bundles" / (model_id + ".json"),
                        {"synthetic_fixture": True, "model_id": model_id})
        jobs.append({"market": market, "model_id": model_id, "model_version": "1",
            "entrypoint": "modeler.serving.adapters:infer_kr_daily" if market == "KR"
                          else "modeler.serving.adapters:infer_us_model",
            "bundle_path": str(bundle.relative_to(release)), "bundle_sha256": _sha(bundle),
            "code_path": str(adapter.relative_to(release)), "code_path_sha256": _sha(adapter),
            "code_files": inventory, "code_sha256": code_sha})
    release_manifest = _write(release / "release.json", {"schema_version": "daily-briefing-release.v1",
        "frozen": True, "synthetic_fixture": True, "release_root": str(release), "jobs": jobs})
    prepared = tmp_path / "prepared"
    _native(prepared, "KR")
    _native(prepared, "US")
    kr_calendar = _write(tmp_path / "kr-calendar.json", _calendar("KR"))
    us_calendar = _write(tmp_path / "us-calendar.json", _calendar("US"))
    expected = _write(tmp_path / "us-expected.json", {"schema_version": "us-expected-source.v1",
        "reviewed_status": "synthetic_fixture", "source_reference": "synthetic-fixture-only",
        "expected_session_by_report_date": {D.isoformat(): "2026-09-28"},
        "market_lag_limit_sessions": 1})
    config = _write(tmp_path / "ops.json", {"schema_version": "daily-briefing-ops.v1",
        "prepared_root": str(prepared), "selection_root": str(prepared / "selections"),
        "run_root": str(tmp_path / "runs"), "projection_root": str(tmp_path / "site"),
        "previous_projection_dir": None, "release_manifest": str(release_manifest),
        "python": sys.executable, "python_sha256": _sha(Path(sys.executable)),
        "runtime_lock": str(release / "uv.lock"), "runtime_lock_sha256": _sha(release / "uv.lock"),
        "runtime_manifest": str(release / "runtime.json"), "runtime_manifest_sha256": _sha(release / "runtime.json"),
        "model_cards_path": str(cards), "model_cards_sha256": _sha(cards),
        "kr_calendar": str(kr_calendar), "us_calendar": str(us_calendar),
        "us_expected_source": str(expected), "opening_artifact": None,
        "base_path": "/market-briefing/", "publisher_enabled": False,
        "external_verification_enabled": False, "publisher_script": None,
        "publisher_script_sha256": None, "publisher_config": None, "site_checkout": None,
        "actions_repository": None, "actions_workflow": None, "public_manifest_url": None})
    return release, config


def _cli(release: Path, config: Path, stage: str, *, fixture_now: str = "2026-09-29T10:00:00+09:00") -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release / "src")
    cmd = [sys.executable, "-m", "modeler.serving.daily_coordinator", stage,
           "--config", str(config), "--report-date", D.isoformat(), "--fixture-now", fixture_now]
    result = subprocess.run(cmd, cwd=release, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, (stage, result.stdout, result.stderr)
    return json.loads(result.stdout)


def test_three_model_synthetic_coordinator_cli_and_private_site(tmp_path: Path) -> None:
    release, config = _setup(tmp_path)
    assert _cli(release, config, "select", fixture_now="2026-09-29T09:30:00+09:00")["status"] == "selected"
    assert _cli(release, config, "infer")["status"] == "inferred"
    assert _cli(release, config, "render")["status"] == "rendered"
    assert _cli(release, config, "publish")["status"] == "publication_withheld"
    report = json.loads((tmp_path / "runs" / D.isoformat() / f"report-{D}.json").read_text())
    assert len(report["markets"]) == 3
    assert report["synthetic_fixture"] is True
    manifest = json.loads((tmp_path / "site" / D.isoformat() / "site-manifest.json").read_text())
    assert manifest["latest_report_date"] == D.isoformat()
    assert manifest["synthetic_fixture"] is True


def test_missing_us_policy_keeps_kr_cli_inference(tmp_path: Path) -> None:
    release, config = _setup(tmp_path)
    policy = Path(json.loads(config.read_text())["us_expected_source"])
    policy.unlink()
    assert _cli(release, config, "select", fixture_now="2026-09-29T09:30:00+09:00")["status"] == "partial"
    assert _cli(release, config, "infer")["status"] == "inferred_partial"
    assert _cli(release, config, "render")["status"] == "rendered"
    report = json.loads((tmp_path / "runs" / D.isoformat() / f"report-{D}.json").read_text())
    assert [row["market"] for row in report["markets"]] == ["KR"]
    assert len(report["failures"]) == 2
    public = json.loads((tmp_path / "site" / D.isoformat() / "reports" / D.isoformat() /
                         "report.json").read_text())
    assert len(public["markets"]) == 3
    assert [row["status"] for row in public["markets"] if row["market"] == "US"] == ["unavailable"] * 2


def test_no_native_jobs_still_renders_three_unavailable_sections(tmp_path: Path) -> None:
    release, config = _setup(tmp_path)
    shutil.rmtree(tmp_path / "prepared" / "kr")
    shutil.rmtree(tmp_path / "prepared" / "us")
    assert _cli(release, config, "select", fixture_now="2026-09-29T09:30:00+09:00")["status"] == "partial"
    # The runner correctly exits nonzero; the current D report still renders a status page.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release / "src")
    cmd = [sys.executable, "-m", "modeler.serving.daily_coordinator", "infer",
           "--config", str(config), "--report-date", D.isoformat(),
           "--fixture-now", "2026-09-29T10:00:00+09:00"]
    result = subprocess.run(cmd, cwd=release, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "inference_failed"
    assert _cli(release, config, "render")["status"] == "rendered"
    public = json.loads((tmp_path / "site" / D.isoformat() / "reports" / D.isoformat() /
                         "report.json").read_text())
    assert len(public["markets"]) == 3
    assert all(row["status"] == "unavailable" for row in public["markets"])


def test_failed_second_invocation_does_not_reuse_old_report(tmp_path: Path, monkeypatch) -> None:
    from modeler.serving import daily_coordinator

    release, config_path = _setup(tmp_path)
    assert _cli(release, config_path, "select", fixture_now="2026-09-29T09:30:00+09:00")["status"] == "selected"
    assert _cli(release, config_path, "infer")["status"] == "inferred"
    config = daily_coordinator._config(config_path)
    monkeypatch.setattr(daily_coordinator.subprocess, "run",
                        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1))
    status = daily_coordinator.infer_stage(config, D, daily_coordinator.datetime.fromisoformat(
        "2026-09-29T10:00:00+09:00"))
    assert status["status"] == "inference_failed"
    assert status["this_invocation_completed"] is False
    assert status["successful_jobs"] == 0


def test_wrapper_renders_status_page_when_no_models_run(tmp_path: Path) -> None:
    release, config = _setup(tmp_path)
    shutil.rmtree(tmp_path / "prepared" / "kr")
    shutil.rmtree(tmp_path / "prepared" / "us")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release / "src")
    def wrapper(stage: str, now: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, "-m", "modeler.serving.daily_wrapper", stage,
            "--config", str(config), "--report-date", D.isoformat(), "--fixture-now", now],
            cwd=release, env=env, text=True, capture_output=True, timeout=60)
    selected = wrapper("select", "2026-09-29T09:30:00+09:00")
    assert selected.returncode == 0, selected.stderr
    executed = wrapper("run", "2026-09-29T10:00:00+09:00")
    assert executed.returncode == 1, executed.stderr
    assert json.loads(executed.stdout)["stages"] == {
        "opening": "not_configured", "inference": "inference_failed",
        "render": "rendered", "publication": "publication_withheld"}
    public = json.loads((tmp_path / "site" / D.isoformat() / "reports" / D.isoformat() /
                         "report.json").read_text())
    assert len(public["markets"]) == 3


def test_publish_rejects_base_config_with_projection_before_running_publisher(tmp_path: Path) -> None:
    from modeler.serving import daily_coordinator

    release, config_path = _setup(tmp_path)
    assert _cli(release, config_path, "select", fixture_now="2026-09-29T09:30:00+09:00")["status"] == "selected"
    assert _cli(release, config_path, "infer")["status"] == "inferred"
    assert _cli(release, config_path, "render")["status"] == "rendered"
    config = daily_coordinator._config(config_path)
    wrong = _write(tmp_path / "publisher.json", {
        "projection_dir": str(tmp_path / "site" / "2026-09-28"),
        "checkout_dir": str(tmp_path / "site-checkout"), "base_path": "/market-briefing/"})
    config.update({"publisher_enabled": True, "publisher_config": str(wrong),
                   "site_checkout": str(tmp_path / "site-checkout")})
    with pytest.raises(ValueError, match="must not set projection_dir"):
        daily_coordinator.publish_stage(config, D)
    assert not (tmp_path / "runs" / D.isoformat() / "coordinator-publication.json").exists()


def test_monitor_keeps_rights_and_transport_status_separate(tmp_path: Path) -> None:
    release, config = _setup(tmp_path)
    assert _cli(release, config, "select", fixture_now="2026-09-29T09:30:00+09:00")["status"] == "selected"
    assert _cli(release, config, "infer")["status"] == "inferred"
    assert _cli(release, config, "render")["status"] == "rendered"
    assert _cli(release, config, "publish")["status"] == "publication_withheld"
    assert _cli(release, config, "monitor")["status"] == "publication_withheld"
    state = json.loads((tmp_path / "runs" / D.isoformat() / "monitor-0.json").read_text())
    assert state["axes"]["rights"] == "withheld"
    assert state["axes"]["publisher"] == "publication_withheld"
    assert state["axes"]["actions"] == "unknown"
    assert state["axes"]["public_url"] == "unknown"


def test_holiday_monitor_skips_without_retry(tmp_path: Path) -> None:
    release, config = _setup(tmp_path)
    calendar_path = Path(json.loads(config.read_text())["kr_calendar"])
    calendar = json.loads(calendar_path.read_text())
    calendar["sessions"] = ["2026-09-28"]
    _write(calendar_path, calendar)
    assert _cli(release, config, "select", fixture_now="2026-09-29T09:30:00+09:00")["status"] == "holiday"
    assert _cli(release, config, "monitor")["status"] == "holiday_skipped"
    state = json.loads((tmp_path / "runs" / D.isoformat() / "monitor-0.json").read_text())
    assert state["retry_allowed"] is False
    assert state["axes"]["inference"] == "skipped"


def test_wrapper_exits_nonzero_on_publisher_failure(tmp_path: Path, monkeypatch, capsys) -> None:
    from modeler.serving import daily_wrapper

    monkeypatch.setattr(daily_wrapper, "_config", lambda path: {"release_manifest": str(tmp_path / "release.json")})
    monkeypatch.setattr(daily_wrapper, "_read", lambda path: {"synthetic_fixture": True})
    monkeypatch.setattr(daily_wrapper, "_day_dir", lambda config, day: tmp_path)
    monkeypatch.setattr(daily_wrapper, "_lock", lambda path: nullcontext())
    monkeypatch.setattr(daily_wrapper, "_selected", lambda config, day: {"status": "selected"})
    monkeypatch.setattr(daily_wrapper, "infer_stage", lambda config, day, now: {
        "status": "inferred", "this_invocation_completed": True})
    monkeypatch.setattr(daily_wrapper, "render_stage", lambda config, day: {"status": "rendered"})
    monkeypatch.setattr(daily_wrapper, "publish_stage", lambda config, day: {"status": "publisher_failed"})
    exit_code = daily_wrapper.main(["run", "--config", str(tmp_path / "ops.json"),
        "--report-date", D.isoformat(), "--fixture-now", "2026-09-29T10:00:00+09:00"])
    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["stages"]["publication"] == "publisher_failed"


def test_render_rejects_runner_invocation_swap(tmp_path: Path) -> None:
    from modeler.serving import daily_coordinator

    release, config_path = _setup(tmp_path)
    assert _cli(release, config_path, "select", fixture_now="2026-09-29T09:30:00+09:00")["status"] == "selected"
    assert _cli(release, config_path, "infer")["status"] == "inferred"
    run_root = tmp_path / "runs" / D.isoformat()
    inference = json.loads((run_root / "coordinator-inference.json").read_text())
    runner = json.loads((run_root / "run-state.json").read_text())
    assert inference["invocation_id"] == runner["invocation_id"]
    runner["invocation_id"] = "another-invocation"
    _write(run_root / "run-state.json", runner)
    with pytest.raises(ValueError, match="invocation changed"):
        daily_coordinator.render_stage(daily_coordinator._config(config_path), D)


def test_opening_builder_only_passes_slots_received_by_ten(tmp_path: Path, monkeypatch) -> None:
    from modeler.serving import daily_coordinator, daily_opening

    release, config_path = _setup(tmp_path)
    config = daily_coordinator._config(config_path)
    snapshot_root = tmp_path / "slots"
    before = _write(snapshot_root / f"report_date={D}" / "slot-0930-a.json",
                    {"received_at": "2026-09-29T09:30:01+09:00"})
    after = _write(snapshot_root / f"report_date={D}" / "slot-1001-b.json",
                   {"received_at": "2026-09-29T10:01:00+09:00"})
    output_root = tmp_path / "opening"
    artifact = _write(output_root / f"report_date={D}" / "opening-fixture.json",
        {"report_date": D.isoformat(), "decision_at": "2026-09-29T10:00:00+09:00",
         "status": "unavailable"})
    config.update({"opening_snapshot_root": str(snapshot_root),
                   "opening_output_root": str(output_root), "opening_max_age_seconds": 1800})
    observed_argv = []
    def fake_run(argv, **kwargs):
        observed_argv.extend(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps({"path": str(artifact)}), "")
    monkeypatch.setattr(daily_opening.subprocess, "run", fake_run)
    state = daily_opening.prepare_opening(config, D)
    assert state["snapshot_count"] == 1
    assert str(before) in observed_argv
    assert str(after) not in observed_argv
    assert config["opening_artifact"] == str(artifact)


def test_opening_child_never_runs_with_tampered_frozen_source(tmp_path: Path, monkeypatch) -> None:
    from modeler.serving import daily_coordinator, daily_opening

    release, config_path = _setup(tmp_path)
    config = daily_coordinator._config(config_path)
    snapshot_root = tmp_path / "slots"
    snapshot_root.mkdir()
    config.update({"opening_snapshot_root": str(snapshot_root),
                   "opening_output_root": str(tmp_path / "opening"),
                   "opening_max_age_seconds": 1800})
    opening_source = release / "src/modeler/serving/opening_prepare.py"
    opening_source.write_text(opening_source.read_text() + "\n# tampered after release pin\n")
    def forbidden_child(*args, **kwargs):
        pytest.fail("opening child ran with unpinned source")
    monkeypatch.setattr(daily_opening.subprocess, "run", forbidden_child)
    state = daily_opening.prepare_opening(config, D)
    assert state == {"status": "unavailable", "report_date": D.isoformat(),
                     "error_class": "ValueError"}
