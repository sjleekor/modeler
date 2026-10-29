"""Synthetic Cronicle payloads only: no live API, credentials, or source records."""

from __future__ import annotations

import io
import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from modeler.feedback import preflight
from modeler.feedback.core import FeedbackError

NOW = datetime(2026, 11, 9, 2, tzinfo=UTC)
EVENT = "feedback-event"


def collect(jobs, own="self", **kwargs):
    return preflight.collect_preflight(
        now=NOW,
        jobs_payload={"code": 0, "jobs": jobs},
        own_job_id=own,
        expected_event_id=EVENT,
        read_memory=lambda: 2**32,
        load=lambda: 0.1,
        **kwargs,
    )


def job(identifier, event=EVENT):
    return {"id": identifier, "event": event, "params": {"private": "never-copy-this"}}


def test_exact_self_excluded_and_other_jobs_remain_blocking():
    state = collect({"self": job("self"), "other": job("other", "other-event")})
    assert state == {
        "schema": "prediction-feedback-preflight.v1",
        "observed_at": NOW.isoformat(),
        "api_code": 0,
        "active_jobs": ["other"],
        "own_job_id": "self",
        "excluded_own_job": True,
        "mem_available_bytes": 2**32,
        "load1": 0.1,
    }
    assert "never-copy-this" not in json.dumps(state)


def test_same_event_does_not_exclude_another_job():
    assert collect({"self": job("self"), "other": job("other")})["active_jobs"] == ["other"]


def test_no_self_does_not_remove_any_job():
    state = collect({"self": job("self"), "other": job("other")}, own=None)
    assert state["active_jobs"] == ["other", "self"]
    assert state["excluded_own_job"] is False
    assert collect({}, own=None)["active_jobs"] == []


@pytest.mark.parametrize(
    "jobs",
    [
        {},
        {"self": job("self", "wrong-event")},
        {"self": {"id": "self"}},
        {"self": job("another")},
        {"self": job("self"), "other": job("self")},
        {"self": job("self"), "other": {"event": EVENT}},
        {"self": job("self"), "other": None},
    ],
)
def test_missing_self_mismatched_event_and_id_collisions_fail_closed(jobs):
    with pytest.raises(FeedbackError):
        collect(jobs)


@pytest.mark.parametrize(
    "payload",
    [{"code": 1, "jobs": {}}, {"code": False, "jobs": {}}, {"code": 0, "jobs": []}, {}],
)
def test_api_failure_and_invalid_shape_are_not_observations(payload):
    with pytest.raises(FeedbackError):
        preflight.collect_preflight(
            now=NOW,
            jobs_payload=payload,
            own_job_id=None,
            expected_event_id=EVENT,
            read_memory=lambda: 2**32,
            load=lambda: 0.1,
        )


@pytest.mark.parametrize("memory,load", [(0, 0), (True, 0), (1, -1), (1, float("nan"))])
def test_invalid_host_counters_fail_closed(memory, load):
    with pytest.raises(FeedbackError):
        preflight.collect_preflight(
            now=NOW,
            jobs_payload={"code": 0, "jobs": {}},
            own_job_id=None,
            expected_event_id=EVENT,
            read_memory=lambda: memory,
            load=lambda: load,
        )


class Input(io.BytesIO):
    def close(self):
        self.saved = self.getvalue()
        super().close()


class Process:
    def __init__(self, raw, returncode=0):
        self.stdin = Input()
        self.stdout = io.BytesIO(raw)
        self.returncode = returncode
        self.killed = False
        self.finished = False

    def wait(self, timeout=None):
        self.finished = True
        return self.returncode

    def poll(self):
        return self.returncode if self.finished else None

    def kill(self):
        self.killed = True


def test_poll_uses_compressed_get_fixed_endpoint_and_private_stdin(monkeypatch):
    process = Process(b'{"code":0,"jobs":{}}')
    calls = []
    # Values are synthetic fixtures; check that every managed proxy variable is inherited.
    proxy_names = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "no_proxy"]
    before = {name: os.environ.get(name) for name in proxy_names}
    monkeypatch.setenv("FEEDBACK_CRONICLE_API_KEY", "synthetic-env-secret")

    def spawn(args, **kwargs):
        calls.append((args, kwargs))
        return process

    key = 'synthetic-quote"-slash\\-key'
    assert preflight.poll_active_jobs(key, popen=spawn) == {"code": 0, "jobs": {}}
    args, kwargs = calls[0]
    assert "--compressed" in args
    assert args[args.index("--request") + 1] == "GET"
    assert args[args.index("--config") + 1] == "-"
    assert args[-1] == preflight.ENDPOINT
    assert "--proxy" not in args and "--noproxy" not in args and "--insecure" not in args
    assert key not in repr(args) and key not in repr(kwargs)
    assert "FEEDBACK_CRONICLE_API_KEY" not in kwargs["env"]
    assert {name: kwargs["env"].get(name) for name in proxy_names} == before
    assert {name: os.environ.get(name) for name in proxy_names} == before
    assert process.stdin.saved == b'header = "X-API-Key: synthetic-quote\\"-slash\\\\-key"\n'


@pytest.mark.parametrize("key", [None, "", "bad\nheader", "bad\rheader", "bad\x00key", "x" * 4097])
def test_invalid_key_never_starts_curl(key):
    def no_spawn(*args, **kwargs):
        pytest.fail("invalid credential reached curl")

    with pytest.raises(FeedbackError):
        preflight.poll_active_jobs(key, popen=no_spawn)


@pytest.mark.parametrize("raw,code", [(b"not-json", 0), (b"[]", 0), (b"{}", 22)])
def test_poll_failures_are_generic(raw, code):
    with pytest.raises(FeedbackError, match="verification failed"):
        preflight.poll_active_jobs("synthetic-key", popen=lambda *a, **k: Process(raw, code))


def test_decompressed_payload_is_bounded_and_oversized_process_reaped():
    process = Process(b" " * (preflight.MAX_PAYLOAD_BYTES + 100))
    with pytest.raises(FeedbackError):
        preflight.poll_active_jobs("synthetic-key", popen=lambda *a, **k: process)
    assert process.killed and process.finished


def test_private_atomic_snapshot_and_output_boundary(tmp_path):
    root = tmp_path / "state"
    path = root / "preflight.json"
    state = collect({"self": job("self")})
    preflight.write_snapshot(path, state, output_root=root)
    assert json.loads(path.read_bytes()) == state
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    state["load1"] = 0.2
    preflight.write_snapshot(path, state, output_root=root)
    assert json.loads(path.read_bytes())["load1"] == 0.2
    assert list(root.iterdir()) == [path]
    with pytest.raises(FeedbackError):
        preflight.write_snapshot(tmp_path / "escape.json", state, output_root=root)
    with pytest.raises(FeedbackError):
        preflight.write_snapshot(Path("relative.json"), state, output_root=root)
    with pytest.raises(FeedbackError):
        preflight.write_snapshot(Path("/file.json"), state, output_root=Path("/"))


def test_snapshot_symlink_and_parent_symlink_rejected(tmp_path):
    state = collect({}, own=None)
    target = tmp_path / "target"
    target.write_text("preserve")
    path = tmp_path / "snapshot"
    path.symlink_to(target)
    with pytest.raises(FeedbackError):
        preflight.write_snapshot(path, state, output_root=tmp_path)
    assert target.read_text() == "preserve"
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(FeedbackError):
        preflight.write_snapshot(alias / "new.json", state, output_root=alias)


def cli_args(tmp_path, *extra):
    return [
        "--output",
        str(tmp_path / "state/preflight.json"),
        "--output-root",
        str(tmp_path / "state"),
        "--expected-event-id",
        EVENT,
        *extra,
    ]


def test_scheduled_mode_requires_explicit_job_id_not_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("JOB_ID", "self")
    monkeypatch.setattr(preflight, "poll_active_jobs", lambda key: pytest.fail("must fail first"))
    assert preflight.main(cli_args(tmp_path)) == 2
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("failure", ["request", "api", "self", "event"])
def test_cli_failure_does_not_write_success_snapshot(tmp_path, monkeypatch, capsys, failure):
    monkeypatch.setattr(preflight, "_read_key", lambda fd: "synthetic-secret")

    def poll(key):
        if failure == "request":
            raise OSError("synthetic-secret must never appear")
        if failure == "api":
            return {"code": 1, "jobs": {}}
        if failure == "self":
            return {"code": 0, "jobs": {}}
        return {"code": 0, "jobs": {"self": job("self", "wrong-event")}}

    monkeypatch.setattr(preflight, "poll_active_jobs", poll)
    assert preflight.main(cli_args(tmp_path, "--own-job-id", "self")) == 2
    assert not (tmp_path / "state").exists()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "feedback-preflight: observation failed\n"


def test_cli_timestamp_after_poll_and_fd_overrides_environment(tmp_path, monkeypatch):
    key_file = tmp_path / "synthetic-fd"
    key_file.write_bytes(b"synthetic-fd-key")
    monkeypatch.setenv("FEEDBACK_CRONICLE_API_KEY", "synthetic-env-key")
    ticks = []

    class Clock:
        @staticmethod
        def now(tz):
            assert ticks == ["polled"]
            return NOW + timedelta(seconds=20)

    def poll(key):
        assert key == "synthetic-fd-key"
        ticks.append("polled")
        return {"code": 0, "jobs": {"self": job("self")}}

    actual_collect = preflight.collect_preflight

    def collect_injected(**kwargs):
        return actual_collect(**kwargs, read_memory=lambda: 2**32, load=lambda: 0.1)

    monkeypatch.setattr(preflight, "datetime", Clock)
    monkeypatch.setattr(preflight, "poll_active_jobs", poll)
    monkeypatch.setattr(preflight, "collect_preflight", collect_injected)
    with key_file.open("rb") as stream:
        assert (
            preflight.main(
                cli_args(tmp_path, "--own-job-id", "self", "--key-fd", str(stream.fileno()))
            )
            == 0
        )
    state = json.loads((tmp_path / "state/preflight.json").read_bytes())
    assert state["observed_at"] == (NOW + timedelta(seconds=20)).isoformat()
