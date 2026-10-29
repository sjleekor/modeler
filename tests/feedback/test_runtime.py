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


INCREMENTAL_NOW = datetime(2026, 11, 9, 2, tzinfo=UTC)


def provider_config(tmp_path, monkeypatch):
    value = config(tmp_path)
    state = tmp_path / "state/preflight.json"
    state.parent.mkdir()
    Path(value["preflight_file"]).replace(state)
    value.update(
        mode="incremental",
        not_before_date="2026-11-07",
        sessions=["2026-11-09"],
        preflight_file=str(state),
        preflight_provider={"enabled": True, "expected_event_id": "feedback-event"},
        require_preflight_provider=True,
    )
    monkeypatch.setattr(runtime, "_completion", lambda source: None)
    monkeypatch.setenv("FEEDBACK_JOB_ID", "feedback-self")
    monkeypatch.setenv("FEEDBACK_CRONICLE_API_KEY", "synthetic-env-key")
    actual_collect = runtime.preflight_provider.collect_preflight

    def injected_collect(**kwargs):
        return actual_collect(**kwargs, read_memory=lambda: 2**32, load=lambda: 0.1)

    monkeypatch.setattr(runtime.preflight_provider, "collect_preflight", injected_collect)
    return value


def _self_payload():
    return {
        "code": 0,
        "jobs": {"feedback-self": {"id": "feedback-self", "event": "feedback-event"}},
    }


def test_incremental_provider_refreshes_stale_snapshot_and_scrubs_build_environment(
    tmp_path, monkeypatch
):
    value = provider_config(tmp_path, monkeypatch)
    observations = []
    proxy_names = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
    proxy_state = {name: os.environ.get(name) for name in proxy_names}

    def poll(key):
        assert key == "synthetic-env-key"
        observations.append("fresh-api")
        return _self_payload()

    monkeypatch.setattr(runtime.preflight_provider, "poll_active_jobs", poll)

    class Child:
        def wait(self, timeout):
            return 0

    def spawn(args, **kwargs):
        assert observations == ["fresh-api"]
        state = json.loads(Path(value["preflight_file"]).read_bytes())
        assert state["observed_at"] == INCREMENTAL_NOW.isoformat()
        assert state["excluded_own_job"] is True
        assert state["active_jobs"] == []
        assert "FEEDBACK_CRONICLE_API_KEY" not in kwargs["env"]
        assert "FEEDBACK_CRONICLE_KEY_FD" not in kwargs["env"]
        assert {name: kwargs["env"].get(name) for name in proxy_names} == proxy_state
        assert kwargs["close_fds"] is True
        assert kwargs["start_new_session"] is True
        return Child()

    assert runtime.run_config(write_config(tmp_path, value), now=INCREMENTAL_NOW, popen=spawn) == 0
    assert os.environ["FEEDBACK_CRONICLE_API_KEY"] == "synthetic-env-key"


@pytest.mark.parametrize("failure", ["api", "missing-self", "event", "other", "request"])
def test_provider_failure_never_uses_old_success_or_spawns(tmp_path, monkeypatch, failure):
    value = provider_config(tmp_path, monkeypatch)
    snapshot = Path(value["preflight_file"])
    old = json.loads(snapshot.read_bytes())
    old["observed_at"] = INCREMENTAL_NOW.isoformat()
    snapshot.write_text(json.dumps(old))

    def poll(key):
        payload = _self_payload()
        if failure == "request":
            raise OSError("synthetic-credential-in-exception")
        if failure == "api":
            payload["code"] = 1
        elif failure == "missing-self":
            payload["jobs"] = {}
        elif failure == "event":
            payload["jobs"]["feedback-self"]["event"] = "wrong-event"
        elif failure == "other":
            payload["jobs"]["other"] = {"id": "other", "event": "other-event"}
        return payload

    monkeypatch.setattr(runtime.preflight_provider, "poll_active_jobs", poll)

    def no_spawn(*args, **kwargs):
        pytest.fail("unverified provider spawned build child")

    with pytest.raises(FeedbackError):
        runtime.run_config(write_config(tmp_path, value), now=INCREMENTAL_NOW, popen=no_spawn)
    if failure != "other":
        assert json.loads(snapshot.read_bytes()) == old


def test_provider_self_job_is_explicit_and_does_not_infer_cronicle_job_id(tmp_path, monkeypatch):
    value = provider_config(tmp_path, monkeypatch)
    monkeypatch.delenv("FEEDBACK_JOB_ID")
    monkeypatch.setenv("JOB_ID", "feedback-self")
    monkeypatch.setattr(
        runtime.preflight_provider,
        "poll_active_jobs",
        lambda key: pytest.fail("missing explicit identity must fail before polling"),
    )
    with pytest.raises(FeedbackError, match="provider verification failed"):
        runtime.run_config(write_config(tmp_path, value), now=INCREMENTAL_NOW)


def test_provider_fd_precedes_environment_and_is_not_inherited_by_build(tmp_path, monkeypatch):
    value = provider_config(tmp_path, monkeypatch)
    key = tmp_path / "synthetic-key-fd"
    key.write_bytes(b"synthetic-fd-key")

    def poll(secret):
        assert secret == "synthetic-fd-key"
        return _self_payload()

    monkeypatch.setattr(runtime.preflight_provider, "poll_active_jobs", poll)

    class Child:
        def wait(self, timeout):
            return 0

    def spawn(args, **kwargs):
        assert kwargs["close_fds"] is True
        assert "pass_fds" not in kwargs
        assert "FEEDBACK_CRONICLE_KEY_FD" not in kwargs["env"]
        return Child()

    with key.open("rb") as stream:
        monkeypatch.setenv("FEEDBACK_CRONICLE_KEY_FD", str(stream.fileno()))
        assert (
            runtime.run_config(write_config(tmp_path, value), now=INCREMENTAL_NOW, popen=spawn) == 0
        )


def test_operational_config_can_require_provider_without_changing_legacy_isolated(tmp_path):
    value = config(tmp_path)
    value.update(mode="incremental", require_preflight_provider=True)
    with pytest.raises(FeedbackError, match="requires an enabled preflight provider"):
        runtime.run_config(write_config(tmp_path, value), now=INCREMENTAL_NOW)
    value["mode"] = "isolated"
    assert runtime._provider_enabled(value) is False


def test_disabled_runtime_does_not_read_provider_identity_or_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runtime,
        "_provider_enabled",
        lambda config: pytest.fail("disabled runner inspected provider"),
    )
    assert (
        runtime.run_config(
            write_config(
                tmp_path,
                {
                    "schema": "prediction-feedback-runtime.v1",
                    "enabled": False,
                    "preflight_provider": {"enabled": True},
                },
            )
        )
        == 0
    )


def test_runtime_main_does_not_log_exception_credentials(tmp_path, monkeypatch, capsys):
    def fail(path):
        raise OSError("synthetic-secret-must-stay-private")

    monkeypatch.setattr(runtime, "run_config", fail)
    assert runtime.main(["--config", str(tmp_path / "runtime.json")]) == 2
    assert capsys.readouterr().err == "feedback-runtime: execution failed\n"


def test_term_during_provider_unwinds_before_build_and_restores_handlers(tmp_path, monkeypatch):
    value = provider_config(tmp_path, monkeypatch)
    previous = signal.getsignal(signal.SIGTERM)

    def poll(key):
        # Exercise the real runtime signal handler with a mocked network boundary.
        handler = signal.getsignal(signal.SIGTERM)
        assert handler is not previous
        handler(signal.SIGTERM, None)

    monkeypatch.setattr(runtime.preflight_provider, "poll_active_jobs", poll)

    def no_spawn(*args, **kwargs):
        pytest.fail("interrupted provider spawned build")

    assert (
        runtime.run_config(write_config(tmp_path, value), now=INCREMENTAL_NOW, popen=no_spawn)
        == 143
    )
    assert signal.getsignal(signal.SIGTERM) is previous


def test_term_during_provider_reaps_curl_before_restoring_handlers(tmp_path, monkeypatch):
    import io

    value = provider_config(tmp_path, monkeypatch)
    actual_poll = runtime.preflight_provider.poll_active_jobs
    events = []

    class InterruptedOutput(io.BytesIO):
        def read(self, size):
            handler = signal.getsignal(signal.SIGTERM)
            handler(signal.SIGTERM, None)

    class Curl:
        stdin = io.BytesIO()
        stdout = InterruptedOutput()
        finished = False

        def poll(self):
            return 0 if self.finished else None

        def kill(self):
            assert signal.getsignal(signal.SIGTERM) is signal.SIG_IGN
            events.append("killed")

        def wait(self):
            self.finished = True
            events.append("reaped")
            return 0

    child = Curl()
    monkeypatch.setattr(
        runtime.preflight_provider,
        "poll_active_jobs",
        lambda key: actual_poll(key, popen=lambda *a, **k: child),
    )
    assert runtime.run_config(write_config(tmp_path, value), now=INCREMENTAL_NOW) == 143
    assert events == ["killed", "reaped"]
    assert child.stdout.closed


def test_provider_curl_timeout_is_reaped_without_spawning_build(tmp_path, monkeypatch):
    import io

    value = provider_config(tmp_path, monkeypatch)
    actual_poll = runtime.preflight_provider.poll_active_jobs
    events = []

    class Curl:
        stdin = io.BytesIO()
        stdout = io.BytesIO(b'{"code":0,"jobs":{}}')
        killed = False
        finished = False

        def poll(self):
            return 0 if self.finished else None

        def kill(self):
            self.killed = True
            events.append("killed")

        def wait(self, timeout=None):
            if not self.killed:
                raise subprocess.TimeoutExpired("synthetic-curl", timeout)
            self.finished = True
            events.append("reaped")
            return 0

    child = Curl()
    monkeypatch.setattr(
        runtime.preflight_provider,
        "poll_active_jobs",
        lambda key: actual_poll(key, popen=lambda *a, **k: child),
    )

    def no_spawn(*args, **kwargs):
        pytest.fail("timed out provider spawned build")

    with pytest.raises(FeedbackError, match="provider verification failed"):
        runtime.run_config(write_config(tmp_path, value), now=INCREMENTAL_NOW, popen=no_spawn)
    assert events == ["killed", "reaped"]


def test_provider_poll_cannot_carry_execution_past_change_window(tmp_path, monkeypatch):
    value = provider_config(tmp_path, monkeypatch)
    before = datetime(2026, 11, 9, 13, 44, 59, tzinfo=UTC)
    after = before + timedelta(seconds=15)
    ticks = iter([before, before, after, after])
    monkeypatch.setattr(runtime.preflight_provider, "poll_active_jobs", lambda key: _self_payload())

    def no_spawn(*args, **kwargs):
        pytest.fail("provider polling crossed the allowed execution window")

    with pytest.raises(FeedbackError, match="outside the post-briefing window"):
        runtime.run_config(write_config(tmp_path, value), clock=lambda: next(ticks), popen=no_spawn)
