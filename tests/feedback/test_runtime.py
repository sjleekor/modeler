"""Preflight, disabled configuration, and real process-group interruption checks."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time as clock
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from modeler.feedback import runtime
from modeler.feedback.core import FeedbackError, sha256_bytes

NOW = datetime(2026, 10, 30, 1, tzinfo=UTC)


def config(tmp_path):
    package = Path(runtime.__file__).resolve().parents[1]
    paths = list((package / "feedback").glob("*.py")) + [
        package / "__init__.py",
        package / "reporting/__init__.py",
        package / "reporting/markdown.py",
    ]
    pins = {p.relative_to(package).as_posix(): sha256_bytes(p.read_bytes()) for p in paths}
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "observed_at": NOW.isoformat(),
                "api_code": 0,
                "active_jobs": [],
                "mem_available_bytes": 2**32,
                "load1": 0.1,
            }
        )
    )
    return {
        "schema": "prediction-feedback-runtime.v1",
        "enabled": True,
        "mode": "isolated",
        "source_root": str(tmp_path / "source/2026-10-29"),
        "allowed_dates": ["2026-10-29"],
        "checkout": str(tmp_path / "reports"),
        "output_root": str(tmp_path / "out"),
        "mapping_file": str(tmp_path / "mapping.json"),
        "mapping_sha256": "a" * 64,
        "my_commit": "b" * 40,
        "issues_blob": "c" * 40,
        "preflight_file": str(preflight),
        "memory_limit_bytes": 256 * 1024**2,
        "timeout_seconds": 20,
        "code_files": pins,
    }


def write_config(tmp_path, value):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(value))
    return path


def test_disabled_does_not_read_any_other_input(tmp_path, monkeypatch):
    def no_read(*args):
        pytest.fail("disabled runner opened an operational input")

    monkeypatch.setattr(runtime, "_code_pin", no_read)
    assert (
        runtime.run_config(
            write_config(tmp_path, {"schema": "prediction-feedback-runtime.v1", "enabled": False})
        )
        == 0
    )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("change", ["old", "future", "active", "api", "memory", "load"])
def test_unready_preflight_never_spawns(tmp_path, change):
    value = config(tmp_path)
    path = Path(value["preflight_file"])
    state = json.loads(path.read_text())
    changes = {
        "old": {"observed_at": (NOW - timedelta(seconds=61)).isoformat()},
        "future": {"observed_at": (NOW + timedelta(seconds=1)).isoformat()},
        "active": {"active_jobs": ["other-job"]},
        "api": {"api_code": 1},
        "memory": {"mem_available_bytes": 100},
        "load": {"load1": 3.0},
    }
    state.update(changes[change])
    path.write_text(json.dumps(state))

    def no_spawn(*args, **kwargs):
        pytest.fail("preflight failure started a child")

    with pytest.raises(FeedbackError):
        runtime.run_config(write_config(tmp_path, value), now=NOW, popen=no_spawn)


def test_unpinned_code_and_unallowed_backfill_rejected(tmp_path):
    value = config(tmp_path)
    value["code_files"]["feedback/runtime.py"] = "f" * 64
    with pytest.raises(FeedbackError, match="pin"):
        runtime.run_config(write_config(tmp_path, value), now=NOW)
    value = config(tmp_path)
    value["source_root"] = str(tmp_path / "source/2026-10-28")
    with pytest.raises(FeedbackError, match="10-29"):
        runtime.command(value, NOW)


def test_freshness_is_rechecked_immediately_before_spawn(tmp_path):
    value = config(tmp_path)
    ticks = iter([NOW, NOW + timedelta(seconds=61), NOW + timedelta(seconds=61)])

    def no_spawn(*args, **kwargs):
        pytest.fail("stale preflight reached spawn")

    with pytest.raises(FeedbackError, match="stale"):
        runtime.run_config(write_config(tmp_path, value), clock=lambda: next(ticks), popen=no_spawn)


def test_incremental_cannot_start_before_existing_change_window(tmp_path):
    value = config(tmp_path)
    value.update(mode="incremental", not_before_date="2026-11-07")
    with pytest.raises(FeedbackError, match="date has not arrived"):
        runtime.command(value, NOW)


@pytest.mark.parametrize("hour", [0, 9, 22])
def test_incremental_is_gated_by_time_and_session(tmp_path, hour):
    value = config(tmp_path)
    value.update(mode="incremental", not_before_date="2026-11-07", sessions=[])
    at = datetime(2026, 11, 9, hour, tzinfo=runtime.SEOUL)
    with pytest.raises(FeedbackError):
        runtime.command(value, at)


def _alive(pid):
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


@pytest.mark.parametrize("stop", ["TERM", "repeated-TERM", "timeout"])
def test_real_interruption_kills_child_group_and_leaves_index_unchanged(tmp_path, stop):
    value = config(tmp_path)
    value["timeout_seconds"] = 1 if stop == "timeout" else 20
    path = write_config(tmp_path, value)
    out = Path(value["output_root"])
    out.mkdir()
    (out / "index").mkdir()
    pointer = out / "index/current.json"
    pointer.write_text('{"previous":"complete"}')
    before = pointer.read_bytes()
    marker = tmp_path / "pids.json"
    child_code = (
        "import subprocess,sys,time,json,os,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "p=subprocess.Popen([sys.executable,'-c','import signal,time; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)']); "
        f"open({str(marker)!r},'w').write(json.dumps([os.getpid(),p.pid]));time.sleep(120)"
    )
    parent_code = (
        "from modeler.feedback.runtime import run_config; import subprocess; "
        "from datetime import datetime; "
        f"spawn=lambda args,**kw: subprocess.Popen([{sys.executable!r},'-c',{child_code!r}],**kw); "
        f"raise SystemExit(run_config({str(path)!r},"
        f"now=datetime.fromisoformat({NOW.isoformat()!r}),popen=spawn))"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code], env=os.environ.copy())
    pids = []
    try:
        deadline = clock.monotonic() + 5
        while not marker.exists() and clock.monotonic() < deadline:
            clock.sleep(0.02)
        assert marker.exists()
        pids = json.loads(marker.read_text())
        if stop in {"TERM", "repeated-TERM"}:
            parent.send_signal(signal.SIGTERM)
        if stop == "repeated-TERM":
            clock.sleep(0.05)
            parent.send_signal(signal.SIGTERM)
        assert parent.wait(timeout=8) == (124 if stop == "timeout" else 143)
        deadline = clock.monotonic() + 3
        while any(_alive(pid) for pid in pids) and clock.monotonic() < deadline:
            clock.sleep(0.02)
        assert not any(_alive(pid) for pid in pids)
        assert pointer.read_bytes() == before
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        for pid in pids:
            if _alive(pid):
                os.kill(pid, signal.SIGKILL)
