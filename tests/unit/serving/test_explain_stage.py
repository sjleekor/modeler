"""The explain stage in the daily flow: switch, time limit, stale files, and that a run never fails on it."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import ms_fixtures as F
import pytest

from modeler.serving import daily_coordinator as dc
from modeler.serving import daily_wrapper
from modeler.serving import explain as ex

D = F.D
NOW = datetime.fromisoformat(F.RUN_AT)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs" / D.isoformat()


def _set(config: Path, **changes) -> None:
    body = json.loads(config.read_text())
    body.update(changes)
    config.write_text(json.dumps(body, sort_keys=True))


def _wrapper(release: Path, config: Path, stage: str, now: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(release / "src")}
    return subprocess.run(
        [sys.executable, "-m", "modeler.serving.daily_wrapper", stage, "--config", str(config),
         "--report-date", D.isoformat(), "--fixture-now", now],
        cwd=release, env=env, text=True, capture_output=True, timeout=240)


@pytest.fixture()
def world(tmp_path, monkeypatch):
    release, config, data = F.build_release_world(tmp_path, monkeypatch)
    assert _wrapper(release, config, "select", F.SELECT_AT).returncode == 0
    return release, config


def _infer(config_path: Path):
    config = dc._config(config_path)
    state = dc.infer_stage(config, D, NOW)
    assert state["status"] == "inferred", state
    return config, state


def test_config_requires_an_explicit_switch_and_a_sane_time_limit(tmp_path, world):
    release, config_path = world
    body = json.loads(config_path.read_text())
    del body["explain_enabled"]
    config_path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="explain_enabled must be explicit"):
        dc._config(config_path)
    _set(config_path, explain_enabled=True, explain_timeout_seconds=5)
    with pytest.raises(ValueError, match="explain_timeout_seconds"):
        dc._config(config_path)
    _set(config_path, explain_enabled=True, explain_timeout_seconds=120)
    assert dc._config(config_path)["explain_enabled"] is True


def test_disabled_writes_nothing_sets_a_stale_file_aside_and_the_run_output_is_as_before(tmp_path, world):
    release, config_path = world
    run = _run_dir(tmp_path)
    run.mkdir(parents=True, exist_ok=True)
    stale = run / f"explain-{D}.json"
    stale.write_text('{"invocation_id": "old"}')
    done = _wrapper(release, config_path, "run", F.RUN_AT)
    assert done.returncode == 0, done.stderr
    stages = json.loads(done.stdout)["stages"]
    assert "explain" not in stages and stages["render"] == "rendered"  # a disabled release reports as before
    assert not stale.exists() and not list(run.glob(".explain-*"))
    (old,) = run.glob(f"explain-{D}.superseded-*.json")  # moved, not deleted
    assert json.loads(old.read_text()) == {"invocation_id": "old"}
    assert json.loads((run / "coordinator-explain.json").read_text())["status"] == "disabled"


def test_a_failing_explain_never_fails_the_run_and_leaves_the_report_bytes_alone(tmp_path, world):
    release, config_path = world
    _set(config_path, explain_enabled=True)
    done = _wrapper(release, config_path, "run", F.RUN_AT)
    assert done.returncode == 0, done.stderr
    stages = json.loads(done.stdout)["stages"]
    # the synthetic release has no real bundles: every model fails, the document still says so
    assert stages["explain"] == "failed" and stages["inference"] == "inferred" and stages["render"] == "rendered"
    run = _run_dir(tmp_path)
    doc = json.loads((run / f"explain-{D}.json").read_text())
    run_state = json.loads((run / "run-state.json").read_text())
    report = run / f"report-{D}.json"
    assert doc["schema"] == ex.SCHEMA and doc["invocation_id"] == run_state["invocation_id"]
    assert doc["report_sha256"] == _sha(report) == json.loads((run / "coordinator-inference.json").read_text())["report_sha256"]
    assert doc["unit"] == D.isoformat() and doc["release"] == "frozen-release" and doc["status"] == "failed"
    assert set(doc["models"]) == set(ex.MODEL_MARKET)
    assert all(m["status"] == "failed" and m["reason"] for m in doc["models"].values())


def test_explain_does_not_change_the_report_and_a_rerun_sets_the_old_document_aside(tmp_path, world):
    release, config_path = world
    _set(config_path, explain_enabled=True)
    config, first_infer = _infer(config_path)
    report = _run_dir(tmp_path) / f"report-{D}.json"
    before = _sha(report)
    first = dc.explain_stage(config, D, NOW)
    assert _sha(report) == before and first["status"] == "failed"
    doc_path = _run_dir(tmp_path) / f"explain-{D}.json"
    first_doc = json.loads(doc_path.read_text())
    assert first_doc["invocation_id"] == first_infer["invocation_id"]
    # a new invocation of the same D: the earlier document must not stand in for it
    _, second_infer = _infer(config_path)
    assert second_infer["invocation_id"] != first_infer["invocation_id"]
    dc.explain_stage(config, D, NOW)
    second_doc = json.loads(doc_path.read_text())
    assert second_doc["invocation_id"] == second_infer["invocation_id"] != first_doc["invocation_id"]
    (old,) = _run_dir(tmp_path).glob(f"explain-{D}.superseded-*.json")
    assert json.loads(old.read_text())["invocation_id"] == first_doc["invocation_id"]


def test_a_stale_file_cannot_survive_a_crashed_child(tmp_path, world, monkeypatch):
    release, config_path = world
    _set(config_path, explain_enabled=True)
    config, _ = _infer(config_path)
    run = _run_dir(tmp_path)
    (run / f"explain-{D}.json").write_text(json.dumps({"invocation_id": "old", "report_sha256": "x", "status": "ok"}))
    orig = dc.run_group
    monkeypatch.setattr(dc, "run_group", lambda argv, **kw: orig([sys.executable, "-c", "raise SystemExit(3)"], **kw))
    state = dc.explain_stage(config, D, NOW)
    doc = json.loads((run / f"explain-{D}.json").read_text())
    assert state["status"] == "failed" and state["reason"] == "exit_nonzero" and state["exit_code"] == 3
    assert doc["invocation_id"] == state["invocation_id"] != "old" and doc["status"] == "failed"


def test_a_timeout_kills_the_child_and_records_failed(tmp_path, world, monkeypatch):
    release, config_path = world
    _set(config_path, explain_enabled=True)
    config, _ = _infer(config_path)
    config["explain_timeout_seconds"] = 1
    marker = tmp_path / "child.pid"
    script = f"import os, time; open({str(marker)!r}, 'w').write(str(os.getpid())); time.sleep(120)"
    orig = dc.run_group
    monkeypatch.setattr(dc, "run_group", lambda argv, **kw: orig([sys.executable, "-c", script], **kw))
    started = time.monotonic()
    state = dc.explain_stage(config, D, NOW)
    assert time.monotonic() - started < 30
    assert state["status"] == "failed" and state["timed_out"] is True and state["reason"] == "timeout"
    pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the child is gone, not left running
    run = _run_dir(tmp_path)
    doc = json.loads((run / f"explain-{D}.json").read_text())
    assert doc["status"] == "failed" and {m["reason"] for m in doc["models"].values()} == {"timeout"}
    assert not (run / f".explain-{D}.new").exists()


def test_the_wrapper_goes_on_when_the_stage_itself_blows_up(tmp_path, world, monkeypatch, capsys):
    release, config_path = world
    _set(config_path, explain_enabled=True)

    def boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(daily_wrapper, "explain_stage", boom)
    code = daily_wrapper.main(["run", "--config", str(config_path), "--report-date", D.isoformat(),
                               "--fixture-now", F.RUN_AT])
    stages = json.loads(capsys.readouterr().out)["stages"]
    assert code == 0 and stages["explain"] == "failed" and stages["render"] == "rendered"
    assert stages["inference"] == "inferred"


def test_a_document_of_another_invocation_is_refused_and_a_good_one_is_installed_whole(tmp_path, world, monkeypatch):
    release, config_path = world
    _set(config_path, explain_enabled=True)
    config, infer_state = _infer(config_path)
    run = _run_dir(tmp_path)
    orig = dc.run_group

    def fake_child(identity):
        def run_group(argv, **kw):
            out = Path(argv[argv.index("--output") + 1])
            doc = ex.build_document(unit=D.isoformat(), invocation_id=identity, report_sha256=infer_state["report_sha256"],
                                    release="frozen-release", models={m: ex.failed_model(k, "x") for m, k in ex.MODEL_MARKET.items()})
            ex.atomic_write_json(out, doc)
            return orig([sys.executable, "-c", "print('{}')"], **kw)
        return run_group

    monkeypatch.setattr(dc, "run_group", fake_child("someone-else"))
    state = dc.explain_stage(config, D, NOW)
    assert state["status"] == "failed" and state["reason"] == "output_invalid"
    assert json.loads((run / f"explain-{D}.json").read_text())["invocation_id"] == infer_state["invocation_id"]
    monkeypatch.setattr(dc, "run_group", fake_child(infer_state["invocation_id"]))
    state = dc.explain_stage(config, D, NOW)
    final = run / f"explain-{D}.json"
    assert state["status"] == "failed"  # every model of the fake document failed: the rollup is honest
    assert state["document_sha256"] == _sha(final)
    assert not (run / f".explain-{D}.new").exists()


def test_a_failure_report_has_nothing_to_explain(tmp_path, world):
    release, config_path = world
    _set(config_path, explain_enabled=True)
    config, state = _infer(config_path)
    dc._atomic(_run_dir(tmp_path) / "coordinator-inference.json", {**state, "failure_report": True})
    out = dc.explain_stage(config, D, NOW)
    assert out["status"] == "skipped" and out["reason"] == "no_model_report"
    assert not (_run_dir(tmp_path) / f"explain-{D}.json").exists()
