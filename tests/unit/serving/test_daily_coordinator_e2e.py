"""Run the real coordinator CLI in a frozen, synthetic-only local release."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import nullcontext
from datetime import date
from pathlib import Path

import pytest

from modeler.serving.orchestration import code_inventory_sha256
from modeler.serving.runtime_contract import PROBE

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
    "modeler/reporting/site.py", "modeler/reporting/markdown.py",
    "modeler/reporting/security_names.py",
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


def _runtime_manifest(target: Path) -> None:
    """Write the frozen runtime contract these synthetic releases pin.

    The checked-in manifest records the sj2 venv (Linux x86_64).  On the same OS and CPU it is
    copied as is, so version drift in that venv still fails here.  On another platform (a Mac) the
    interpreter identity can never match it, which used to fail every coordinator test before the
    flow under test even ran.  There the contract is taken from the running interpreter instead.
    DAILY_E2E_PROBE_RUNTIME=1 forces that on any platform.
    """
    pinned = PROJECT / "deploy" / "prod" / "runtime-verified-sj2-20260930.json"
    recorded = json.loads(pinned.read_text(encoding="utf-8"))
    same_platform = (recorded["system"], recorded["machine"]) == (
        platform.system(), platform.machine())
    if same_platform and os.environ.get("DAILY_E2E_PROBE_RUNTIME") != "1":
        shutil.copy2(pinned, target)
        return
    probe = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True,
                           check=True)
    body = {"schema_version": "daily-briefing-runtime.v1", **json.loads(probe.stdout)}
    target.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")


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
    _runtime_manifest(release / "runtime.json")
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
        "run_root": str(tmp_path / "runs"), "release_manifest": str(release_manifest),
        "python": sys.executable, "python_sha256": _sha(Path(sys.executable)),
        "runtime_lock": str(release / "uv.lock"), "runtime_lock_sha256": _sha(release / "uv.lock"),
        "runtime_manifest": str(release / "runtime.json"), "runtime_manifest_sha256": _sha(release / "runtime.json"),
        "model_cards_path": str(cards), "model_cards_sha256": _sha(cards),
        "kr_calendar": str(kr_calendar), "us_calendar": str(us_calendar),
        "us_expected_source": str(expected), "opening_artifact": None,
        "publisher_enabled": False, "external_verification_enabled": False,
        "reports_publisher": None, "reports_publisher_sha256": None, "reports_checkout": None,
        "reports_remote_url": None, "reports_repository": None, "reports_audience": None,
        "reports_branch": None, "reports_top_n": 100})
    return release, config


def _cli(release: Path, config: Path, stage: str, *, fixture_now: str = "2026-09-29T10:00:00+09:00") -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release / "src")
    cmd = [sys.executable, "-m", "modeler.serving.daily_coordinator", stage,
           "--config", str(config), "--report-date", D.isoformat(), "--fixture-now", fixture_now]
    result = subprocess.run(cmd, cwd=release, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, (stage, result.stdout, result.stderr)
    return json.loads(result.stdout)


def _unit_context(tmp_path: Path, day: date = D):
    """Render context of the saved report: what the publisher's local step renders as markdown."""
    from modeler.reporting import markdown

    saved = tmp_path / "runs" / day.isoformat() / f"report-{day}.json"
    return markdown.build_context(tmp_path / "no-reports-checkout", saved.read_bytes(), None,
                                  "frozen-release", None, 100, lambda message: None)


def test_three_model_synthetic_coordinator_cli_and_withheld_publication(tmp_path: Path) -> None:
    release, config = _setup(tmp_path)
    assert _cli(release, config, "select", fixture_now="2026-09-29T09:30:00+09:00")["status"] == "selected"
    assert _cli(release, config, "infer")["status"] == "inferred"
    assert _cli(release, config, "render")["status"] == "rendered"
    assert _cli(release, config, "publish")["status"] == "publication_withheld"
    report = json.loads((tmp_path / "runs" / D.isoformat() / f"report-{D}.json").read_text())
    assert len(report["markets"]) == 3
    assert report["synthetic_fixture"] is True
    run_dir = tmp_path / "runs" / D.isoformat()
    render = json.loads((run_dir / "coordinator-render.json").read_text())
    inference = json.loads((run_dir / "coordinator-inference.json").read_text())
    assert render["invocation_id"] == inference["invocation_id"]
    assert render["report_sha256"] == inference["report_sha256"]
    assert render["synthetic_fixture"] is True
    assert not any(key in render for key in ("projection_dir", "site_manifest_sha256"))
    assert not (tmp_path / "site").exists()  # the public Pages projection is gone
    publication = json.loads((run_dir / "coordinator-publication.json").read_text())
    assert publication["reason"] == "publisher_disabled"
    assert not (run_dir / "markdown").exists()  # a disabled publisher renders nothing


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
    ctx = _unit_context(tmp_path)
    assert ctx["kr"]["status"] == "partial"
    assert [m["status"] for m in ctx["us"]["models"]] == ["failed", "failed"]
    assert [m["internal"] for m in ctx["us"]["models"]] == ["failed", "failed"]


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
    ctx = _unit_context(tmp_path)
    assert ctx["status"] == "failed"
    assert [m["status"] for m in ctx["kr"]["models"] + ctx["us"]["models"]] == ["failed"] * 3


def test_failed_second_invocation_does_not_reuse_old_report(tmp_path: Path, monkeypatch) -> None:
    from modeler.serving import daily_coordinator

    release, config_path = _setup(tmp_path)
    assert _cli(release, config_path, "select", fixture_now="2026-09-29T09:30:00+09:00")["status"] == "selected"
    assert _cli(release, config_path, "infer")["status"] == "inferred"
    run_dir = tmp_path / "runs" / D.isoformat()
    old_bytes = (run_dir / f"report-{D}.json").read_bytes()
    config = daily_coordinator._config(config_path)
    monkeypatch.setattr(daily_coordinator, "run_group",
                        lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1))
    status = daily_coordinator.infer_stage(config, D, daily_coordinator.datetime.fromisoformat(
        "2026-09-29T10:00:00+09:00"))
    assert status["status"] == "inference_failed"
    assert status["this_invocation_completed"] is False
    assert status["successful_jobs"] == 0
    # The unit is still made, from this invocation's own failure report, not from the old success.
    assert status["report_ready"] is True and status["failure_report"] is True
    report = json.loads((run_dir / f"report-{D}.json").read_text())
    assert report["status"] == "failed" and report["markets"] == []
    assert report["invocation_id"] == status["invocation_id"]
    assert json.loads((run_dir / "run-state.json").read_text())["invocation_id"] == status["invocation_id"]
    (kept,) = run_dir.glob(f"report-{D}.superseded-*.json")
    assert kept.read_bytes() == old_bytes


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
    assert _unit_context(tmp_path)["status"] == "failed"


def test_enabled_publisher_requires_confirmed_reports_settings(tmp_path: Path) -> None:
    from modeler.serving import daily_coordinator

    release, config_path = _setup(tmp_path)
    checkout = tmp_path / "reports-checkout"
    checkout.mkdir()
    script = tmp_path / "publish_reports.py"
    script.write_text("raise SystemExit(0)\n")
    good = {**json.loads(config_path.read_text()), "publisher_enabled": True,
            "reports_publisher": str(script), "reports_publisher_sha256": _sha(script),
            "reports_checkout": str(checkout),
            "reports_remote_url": "git@github.com:sjleekor/stock_reports.git",
            "reports_repository": "sjleekor/stock_reports", "reports_audience": "owner_only",
            "reports_branch": "main", "reports_top_n": 100}
    accepted = daily_coordinator._config(_write(tmp_path / "good.json", good))
    assert accepted["publisher_enabled"] is True
    for key, value, message in (
            ("reports_repository", "sjleekor/market-briefing", "stock_reports"),
            ("reports_audience", "public", "owner_only"),
            ("reports_audience", None, "owner_only"),
            ("reports_branch", "site", "main"),
            ("reports_remote_url", "", "remote URL"),
            ("reports_publisher_sha256", "0" * 64, "SHA-256"),
            ("reports_top_n", 0, "reports_top_n"),
            ("reports_checkout", str(tmp_path / "missing"), "reports_checkout"),
            ("reports_checkout", None, "reports_checkout")):
        with pytest.raises(ValueError, match=message):
            daily_coordinator._config(_write(tmp_path / "bad.json", {**good, key: value}))
    # A disabled publisher needs none of these settings.
    assert daily_coordinator._config(config_path)["publisher_enabled"] is False


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
    assert state["axes"]["remote"] == "unknown"
    assert set(state["axes"]) == {"input", "inference", "rights", "publisher", "remote"}


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
    monkeypatch.setattr(daily_wrapper, "ensure_selection", lambda config, day, now: {"status": "selected"})
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
    monkeypatch.setattr(daily_opening, "run_group", fake_run)
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
    monkeypatch.setattr(daily_opening, "run_group", forbidden_child)
    state = daily_opening.prepare_opening(config, D)
    assert state == {"status": "unavailable", "report_date": D.isoformat(),
                     "error_class": "ValueError"}
