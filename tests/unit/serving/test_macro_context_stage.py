"""The macro_context stage in the daily flow: switch, time limit, stale files, failure documents,
and that a run never fails on it.
Plan: my/milestones/common/20261031_briefing_macro_context §5.5, §9 M2-c.

The calculation module (``modeler.scores.macro_context``) is built on another branch: the release
gets the stand-in from ``macro_fakes`` (child process) and this process gets it through
``sys.modules`` (the coordinator's own ``failure_document`` call).
"""

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reporting"))
import macro_fakes as mk  # noqa: E402

from modeler.serving import daily_coordinator as dc  # noqa: E402
from modeler.serving import daily_wrapper  # noqa: E402

D = F.D
NOW = datetime.fromisoformat(F.RUN_AT)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs" / D.isoformat()


def _state_file(tmp_path: Path) -> Path:
    return tmp_path / "prepared" / "selections" / D.isoformat() / "selection-state.json"


def _ms_block(tmp_path: Path) -> dict:
    """The market-sector block 09:30 select wrote: the selection document the stage hands over."""
    return json.loads(_state_file(tmp_path).read_text())["market_sector"]


def _edit_block(tmp_path: Path, block) -> None:
    state = json.loads(_state_file(tmp_path).read_text())
    if block is None:
        del state["market_sector"]
    else:
        state["market_sector"] = block
    _state_file(tmp_path).write_text(json.dumps(state, sort_keys=True))


def _set(config: Path, **changes) -> None:
    body = json.loads(config.read_text())
    body.update(changes)
    config.write_text(json.dumps(body, sort_keys=True))


def _wrapper(release: Path, config: Path, stage: str, now: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(release / "src")}
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "modeler.serving.daily_wrapper",
            stage,
            "--config",
            str(config),
            "--report-date",
            D.isoformat(),
            "--fixture-now",
            now,
        ],
        cwd=release,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )


@pytest.fixture()
def world(tmp_path, monkeypatch):
    release, config, _data = F.build_release_world(
        tmp_path, monkeypatch, extra_sources=mk.RELEASE_SOURCES
    )
    assert _wrapper(release, config, "select", F.SELECT_AT).returncode == 0
    return release, config


@pytest.fixture()
def fakes(monkeypatch):
    return mk.install(monkeypatch)


def _infer(config_path: Path):
    config = dc._config(config_path)
    state = dc.infer_stage(config, D, NOW)
    assert state["status"] == "inferred", state
    return config, state


def _doc(tmp_path: Path) -> dict:
    return json.loads((_run_dir(tmp_path) / f"macro-{D}.json").read_text())


def test_config_requires_an_explicit_switch_and_a_sane_time_limit(tmp_path, world):
    _release, config_path = world
    body = json.loads(config_path.read_text())
    del body["macro_context_enabled"]
    config_path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="macro_context_enabled must be explicit"):
        dc._config(config_path)
    _set(config_path, macro_context_enabled="yes")
    with pytest.raises(ValueError, match="macro_context_enabled must be explicit"):
        dc._config(config_path)
    for bad in (5, 1801, True, "300"):
        _set(config_path, macro_context_enabled=True, macro_context_timeout_seconds=bad)
        with pytest.raises(ValueError, match="macro_context_timeout_seconds"):
            dc._config(config_path)
    _set(config_path, macro_context_enabled=True, macro_context_timeout_seconds=300)
    assert dc._config(config_path)["macro_context_enabled"] is True


def test_disabled_writes_nothing_sets_a_stale_file_aside_and_the_run_output_is_as_before(
    tmp_path, world
):
    release, config_path = world
    run = _run_dir(tmp_path)
    run.mkdir(parents=True, exist_ok=True)
    stale = run / f"macro-{D}.json"
    stale.write_text('{"invocation_id": "old"}')
    done = _wrapper(release, config_path, "run", F.RUN_AT)
    assert done.returncode == 0, done.stderr
    stages = json.loads(done.stdout)["stages"]
    assert "macro_context" not in stages and stages["render"] == "rendered"
    assert not stale.exists() and not list(run.glob(".macro-*"))
    (old,) = run.glob(f"macro-{D}.superseded-*.json")  # moved, not deleted
    assert json.loads(old.read_text()) == {"invocation_id": "old"}
    assert json.loads((run / "coordinator-macro-context.json").read_text())["status"] == "disabled"


def test_wrapper_run_computes_the_document_from_the_selection(tmp_path, world):
    release, config_path = world
    _set(config_path, macro_context_enabled=True)
    done = _wrapper(release, config_path, "run", F.RUN_AT)
    assert done.returncode == 0, done.stderr
    stages = json.loads(done.stdout)["stages"]
    assert stages["macro_context"] == "ok" and stages["render"] == "rendered"
    run = _run_dir(tmp_path)
    doc = _doc(tmp_path)
    run_state = json.loads((run / "run-state.json").read_text())
    state = json.loads((run / "coordinator-macro-context.json").read_text())
    assert doc["schema"] == mk.SCHEMA and doc["unit"] == D.isoformat() and doc["state"] == "ok"
    assert doc["invocation_id"] == run_state["invocation_id"] == state["invocation_id"]
    block = _ms_block(tmp_path)
    assert block["status"] in {"selected", "partial"}
    assert doc["selection_sha256"] == block["selection_sha256"] == state["selection_sha256"]
    assert _sha(Path(block["selection"])) == block["selection_sha256"]
    assert doc["seen_selection_schema"] == mk.MS_SELECTION_SCHEMA  # not selection-state.json
    assert doc["release"] == "frozen-release"
    assert state["document_sha256"] == _sha(run / f"macro-{D}.json")
    assert not list(run.glob(".macro-*"))


def test_macro_leaves_report_market_sector_and_explain_bytes_alone(tmp_path, world, fakes):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True, explain_enabled=True)
    config, _ = _infer(config_path)
    dc.market_sector_stage(config, D, NOW)
    dc.explain_stage(config, D, NOW)
    run = _run_dir(tmp_path)
    pinned = [
        run / f"report-{D}.json",
        run / f"market-sector-{D}.json",
        run / f"explain-{D}.json",
        _state_file(tmp_path),
        Path(_ms_block(tmp_path)["selection"]),
    ]
    before = {path: _sha(path) for path in pinned}
    state = dc.macro_context_stage(config, D, NOW)
    assert state["status"] == "ok" and _doc(tmp_path)["state"] == "ok"
    assert {path: _sha(path) for path in pinned} == before


def test_a_failure_report_still_gets_the_section(tmp_path, world, fakes):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config, state = _infer(config_path)
    dc._atomic(_run_dir(tmp_path) / "coordinator-inference.json", {**state, "failure_report": True})
    out = dc.macro_context_stage(config, D, NOW)
    assert out["status"] == "ok" and _doc(tmp_path)["invocation_id"] == state["invocation_id"]


def test_a_rerun_sets_the_old_document_aside(tmp_path, world, fakes):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config, first = _infer(config_path)
    dc.macro_context_stage(config, D, NOW)
    first_doc = _doc(tmp_path)
    _, second = _infer(config_path)
    assert second["invocation_id"] != first["invocation_id"]
    dc.macro_context_stage(config, D, NOW)
    assert _doc(tmp_path)["invocation_id"] == second["invocation_id"]
    (old,) = _run_dir(tmp_path).glob(f"macro-{D}.superseded-*.json")
    assert json.loads(old.read_text())["invocation_id"] == first_doc["invocation_id"]


def test_a_crashed_child_leaves_a_failed_document_of_this_invocation(
    tmp_path, world, fakes, monkeypatch
):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config, infer_state = _infer(config_path)
    run = _run_dir(tmp_path)
    (run / f"macro-{D}.json").write_text(json.dumps({"invocation_id": "old", "state": "ok"}))
    orig = dc.run_group
    monkeypatch.setattr(
        dc,
        "run_group",
        lambda argv, **kw: orig([sys.executable, "-c", "raise SystemExit(3)"], **kw),
    )
    state = dc.macro_context_stage(config, D, NOW)
    doc = _doc(tmp_path)
    assert (
        state["status"] == "failed"
        and state["reason"] == "exit_nonzero"
        and state["exit_code"] == 3
    )
    assert doc["state"] == "failed" and doc["reason"] == "exit_nonzero"
    assert doc["invocation_id"] == infer_state["invocation_id"] != "old"
    assert doc["selection_sha256"] == _ms_block(tmp_path)["selection_sha256"]
    assert state["document_sha256"] == _sha(run / f"macro-{D}.json")


def test_a_timeout_kills_the_child_and_records_failed(tmp_path, world, fakes, monkeypatch):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config, _ = _infer(config_path)
    config["macro_context_timeout_seconds"] = 1
    marker = tmp_path / "child.pid"
    script = f"import os, time; open({str(marker)!r}, 'w').write(str(os.getpid())); time.sleep(120)"
    orig = dc.run_group
    monkeypatch.setattr(
        dc, "run_group", lambda argv, **kw: orig([sys.executable, "-c", script], **kw)
    )
    started = time.monotonic()
    state = dc.macro_context_stage(config, D, NOW)
    assert time.monotonic() - started < 30
    assert (
        state["status"] == "failed" and state["timed_out"] is True and state["reason"] == "timeout"
    )
    pid = int(marker.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)  # the child is gone, not left running
    assert _doc(tmp_path)["reason"] == "timeout"
    assert not (_run_dir(tmp_path) / f".macro-{D}.new").exists()


def test_the_default_time_limit_is_300_seconds(tmp_path, world, fakes, monkeypatch):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config, _ = _infer(config_path)
    seen = {}
    orig = dc.run_group

    def spy(argv, **kw):
        seen.update(argv=argv, timeout=kw["timeout"])
        return orig(argv, **kw)

    monkeypatch.setattr(dc, "run_group", spy)
    assert dc.macro_context_stage(config, D, NOW)["status"] == "ok"
    assert seen["timeout"] == 300 and dc.MACRO_CONTEXT_TIMEOUT_SECONDS == 300
    assert seen["argv"][1:4] == ["-m", dc.MACRO_CONTEXT_ENTRYPOINT, "run"]


def test_a_document_of_another_invocation_is_refused(tmp_path, world, fakes, monkeypatch):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config, infer_state = _infer(config_path)
    orig = dc.run_group

    def fake_child(identity):
        def run_group(argv, **kw):
            out = Path(argv[argv.index("--output") + 1])
            out.write_bytes(mk.dump(mk.doc(D.isoformat(), identity)))
            return orig([sys.executable, "-c", "print('{}')"], **kw)

        return run_group

    monkeypatch.setattr(dc, "run_group", fake_child("someone-else"))
    state = dc.macro_context_stage(config, D, NOW)
    assert state["status"] == "failed" and state["reason"] == "output_invalid"
    assert _doc(tmp_path)["invocation_id"] == infer_state["invocation_id"]
    assert not (_run_dir(tmp_path) / f".macro-{D}.new").exists()
    monkeypatch.setattr(dc, "run_group", fake_child(infer_state["invocation_id"]))
    state = dc.macro_context_stage(config, D, NOW)
    assert state["status"] == "ok" and state["document_sha256"] == _sha(
        _run_dir(tmp_path) / f"macro-{D}.json"
    )


def test_no_selection_means_a_failed_document_without_a_selection_hash(tmp_path, world, fakes):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config, infer_state = _infer(config_path)
    _state_file(tmp_path).unlink()
    early = datetime.fromisoformat(F.SELECT_AT)  # before 10:00 the stage cannot select by itself
    state = dc.macro_context_stage(config, D, early)
    doc = _doc(tmp_path)
    assert state["status"] == "failed" and state["reason"] == "selection_unavailable"
    assert doc["reason"] == "selection_unavailable" and doc["selection_sha256"] is None
    assert doc["invocation_id"] == infer_state["invocation_id"]


def test_the_child_gets_the_market_sector_selection_document(tmp_path, world, fakes, monkeypatch):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config, _ = _infer(config_path)
    seen = {}
    orig = dc.run_group

    def spy(argv, **kw):
        seen["argv"] = argv
        return orig(argv, **kw)

    monkeypatch.setattr(dc, "run_group", spy)
    assert dc.macro_context_stage(config, D, NOW)["status"] == "ok"
    argv, block = seen["argv"], _ms_block(tmp_path)
    selection = Path(argv[argv.index("--selection") + 1])
    assert selection == Path(block["selection"]) and selection != _state_file(tmp_path)
    assert argv[argv.index("--selection-sha256") + 1] == block["selection_sha256"]
    body = json.loads(selection.read_text())
    assert body["schema_version"] == mk.MS_SELECTION_SCHEMA and body["report_date"] == D.isoformat()
    assert _doc(tmp_path)["seen_selection_schema"] == mk.MS_SELECTION_SCHEMA


@pytest.mark.parametrize(
    "block,reason",
    [
        (None, "not_in_selection"),
        ("not-a-dict", "not_in_selection"),
        ({"status": "unavailable", "reason": "kr_snapshot_missing"}, "kr_snapshot_missing"),
        ({"status": "unavailable"}, "inputs_unavailable"),
    ],
)
def test_without_a_usable_market_sector_block_the_section_fails(
    tmp_path, world, fakes, block, reason
):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config, infer_state = _infer(config_path)
    _edit_block(tmp_path, block)
    state = dc.macro_context_stage(config, D, NOW)
    doc = _doc(tmp_path)
    assert state["status"] == "failed" and state["reason"] == reason
    assert doc["reason"] == reason and doc["selection_sha256"] is None
    assert doc["invocation_id"] == infer_state["invocation_id"]


def test_a_changed_selection_document_is_not_used(tmp_path, world, fakes, monkeypatch):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config, _ = _infer(config_path)
    path = Path(_ms_block(tmp_path)["selection"])
    path.write_bytes(path.read_bytes() + b" ")
    called = []
    monkeypatch.setattr(dc, "run_group", lambda argv, **kw: called.append(argv))
    state = dc.macro_context_stage(config, D, NOW)
    assert state["status"] == "failed" and state["reason"] == "selection_changed" and not called
    assert _doc(tmp_path)["reason"] == "selection_changed"


def test_wrapper_run_publishes_the_page_and_svg(tmp_path, world):
    """Selection pinned at 09:30 → child → macro-D.json → publisher → macro.md and SVG pushed."""
    import test_daily_multiday_publish as mp

    release, config_path = world
    reports = mp._enable_publisher(tmp_path, config_path)
    _set(config_path, macro_context_enabled=True)
    done = _wrapper(release, config_path, "run", F.RUN_AT)
    assert done.returncode == 0, done.stderr
    stages = json.loads(done.stdout)["stages"]
    assert stages["macro_context"] == "ok" and stages["publication"] == "published"
    run = _run_dir(tmp_path)
    unit = reports.checkout / "reports" / "daily-briefing" / "2026" / "10" / D.isoformat()
    assert (unit / "macro.md").is_file() and (unit / "assets" / "macro-rates.svg").is_file()
    page = (unit / "macro.md").read_text()
    assert f"macro_sha256: {_sha(run / f'macro-{D}.json')}" in page
    assert "| [거시 흐름](macro.md) | 정상 |" in (unit / "README.md").read_text()
    assert "거시 흐름은 [macro.md](macro.md)" in (unit / "market-sector.md").read_text()
    manifest = json.loads((run / "markdown" / "manifest.json").read_text())
    assert manifest["macro_state"] == "ok"


def test_without_an_invocation_there_is_nothing_to_bind_to(tmp_path, world, fakes):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config = dc._config(config_path)
    state = dc.macro_context_stage(config, D, NOW)
    assert state == {"report_date": D.isoformat(), "status": "skipped", "reason": "no_invocation"}
    assert not (_run_dir(tmp_path) / f"macro-{D}.json").exists()


def test_without_the_calculation_module_the_stage_fails_quietly(tmp_path, world, monkeypatch):
    """Before the calculation branch is merged the coordinator cannot even write a failed document:
    the stage says so and the run goes on without a macro page."""
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)
    config, _ = _infer(config_path)
    monkeypatch.setitem(sys.modules, mk.CALC_MODULE, None)  # import fails in this process
    orig = dc.run_group
    monkeypatch.setattr(
        dc,
        "run_group",
        lambda argv, **kw: orig([sys.executable, "-c", "raise SystemExit(1)"], **kw),
    )
    state = dc.macro_context_stage(config, D, NOW)
    assert state["status"] == "failed" and state["document_error_class"] in {
        "ImportError",
        "ModuleNotFoundError",
    }
    assert not (_run_dir(tmp_path) / f"macro-{D}.json").exists()


def test_the_wrapper_goes_on_when_the_stage_itself_blows_up(tmp_path, world, monkeypatch, capsys):
    _release, config_path = world
    _set(config_path, macro_context_enabled=True)

    def boom(*args, **kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(daily_wrapper, "macro_context_stage", boom)
    code = daily_wrapper.main(
        [
            "run",
            "--config",
            str(config_path),
            "--report-date",
            D.isoformat(),
            "--fixture-now",
            F.RUN_AT,
        ]
    )
    stages = json.loads(capsys.readouterr().out)["stages"]
    assert code == 0 and stages["macro_context"] == "failed" and stages["render"] == "rendered"
    assert stages["inference"] == "inferred"


def test_the_coordinator_cli_runs_the_stage_alone(tmp_path, world):
    release, config_path = world
    _set(config_path, macro_context_enabled=True)
    env = {**os.environ, "PYTHONPATH": str(release / "src")}
    base = [sys.executable, "-m", "modeler.serving.daily_coordinator"]
    tail = ["--config", str(config_path), "--report-date", D.isoformat(), "--fixture-now", F.RUN_AT]
    infer = subprocess.run(
        [*base, "infer", *tail], cwd=release, env=env, text=True, capture_output=True, timeout=240
    )
    assert infer.returncode == 0, infer.stderr
    done = subprocess.run(
        [*base, "macro-context", *tail],
        cwd=release,
        env=env,
        text=True,
        capture_output=True,
        timeout=240,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {"status": "ok", "report_date": D.isoformat()}
