"""Isolated/disabled operational entrypoint with preflight and process-group cleanup."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

from . import preflight as preflight_provider
from .core import Batch, FeedbackError, read_json, require_hash, require_timestamp, sha256_bytes
from .store import check_output_root

SEOUL = ZoneInfo("Asia/Seoul")


class _Stopped(BaseException):
    def __init__(self, signum):
        self.signum = signum


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise FeedbackError(f"{name} must be a positive integer")
    return value


def _absolute(value, name):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise FeedbackError(f"{name} must be an explicit absolute path")
    path = Path(value)
    if ".." in path.parts:
        raise FeedbackError(f"{name} cannot contain parent traversal")
    return path


def preflight(config, now):
    source = _absolute(config.get("preflight_file"), "preflight_file")
    if source.stat().st_size > 16 * 1024:
        raise FeedbackError("preflight snapshot is oversized")
    state = read_json(source, root=source.parent, batch=Batch())
    require_timestamp(state.get("observed_at"))
    elapsed = (
        now - datetime.fromisoformat(state["observed_at"].replace("Z", "+00:00"))
    ).total_seconds()
    if not 0 <= elapsed <= 60 or state.get("api_code") != 0:
        raise FeedbackError("preflight is stale or API verification failed")
    if state.get("active_jobs") != []:
        raise FeedbackError("active jobs prevent feedback execution")
    available = _positive(state.get("mem_available_bytes"), "mem_available_bytes")
    limit = _positive(config.get("memory_limit_bytes"), "memory_limit_bytes")
    if available < max(2 * limit, 512 * 1024 * 1024):
        raise FeedbackError("insufficient available memory")
    load = state.get("load1")
    if type(load) not in (int, float) or not 0 <= load <= 2.0:
        raise FeedbackError("server load prevents feedback execution")


def _code_pin(config):
    root = Path(__file__).resolve().parents[1]
    pins = config.get("code_files")
    if not isinstance(pins, dict) or "feedback/runtime.py" not in pins:
        raise FeedbackError("code file pins are required")
    required = {p.relative_to(root).as_posix() for p in (root / "feedback").glob("*.py")}
    required.update({"__init__.py", "reporting/__init__.py", "reporting/markdown.py"})
    if set(pins) != required:
        raise FeedbackError("code inventory does not match the feedback closure")
    for name, digest in pins.items():
        path = root / name
        if path.is_symlink() or sha256_bytes(path.read_bytes()) != require_hash(digest):
            raise FeedbackError("feedback code pin mismatch")


def command(config, now):
    mode = config.get("mode")
    if mode not in {"isolated", "incremental"}:
        raise FeedbackError("unsupported runtime mode")
    source = _absolute(config.get("source_root"), "source_root")
    if mode == "isolated":
        if source.name != "2026-10-29" or config.get("allowed_dates") != ["2026-10-29"]:
            raise FeedbackError("isolated input is limited to the fixed 10-29 source")
        action = "build"
    else:
        local = now.astimezone(SEOUL)
        if config.get("require_calendar_pin"):
            from .bundle import pinned_sessions

            sessions = pinned_sessions(
                _absolute(config.get("calendar_file"), "calendar_file"),
                config.get("calendar_sha256"),
            )
            if config.get("sessions") != sessions:
                raise FeedbackError("runtime sessions differ from the pinned confirmed calendar")
        if config.get("not_before_date", "") < "2026-11-07":
            raise FeedbackError("incremental activation date cannot precede the F3 change window")
        if local.date().isoformat() < config["not_before_date"]:
            raise FeedbackError("incremental activation date has not arrived")
        if not time(10, 35) <= local.time().replace(tzinfo=None) < time(22, 45):
            raise FeedbackError("incremental execution is outside the post-briefing window")
        if local.date().isoformat() not in config.get("sessions", []):
            raise FeedbackError("today is not a pinned KR session")
        source = source / local.date().isoformat()
        action = "observe"
        if config.get("source_mode") == "publisher-capture":
            action = "capture"
        elif config.get("source_mode") is not None:
            raise FeedbackError("unsupported incremental source mode")
        if action == "observe":
            _completion(source)
    output = _absolute(config.get("output_root"), "output_root")
    checkout = _absolute(config.get("checkout"), "checkout")
    mapping = _absolute(config.get("mapping_file"), "mapping_file")
    check_output_root(output, [source, checkout, mapping])
    require_hash(config.get("mapping_sha256"))
    require_hash(config.get("my_commit"), length=40)
    require_hash(config.get("issues_blob"), length=40)
    args = [
        sys.executable,
        "-s",
        "-P",
        "-m",
        "modeler.feedback",
        action,
        "--kind",
        "briefing",
        "--market",
        "KR",
        "--source-root",
        str(source),
        "--checkout",
        str(checkout),
        "--output-root",
        str(output),
        "--mapping-file",
        str(mapping),
        "--mapping-sha256",
        config["mapping_sha256"],
        "--my-commit",
        config["my_commit"],
        "--issues-blob",
        config["issues_blob"],
    ]
    if action == "capture":
        settings = _absolute(config.get("capture_settings"), "capture_settings")
        from .captured import session_config

        capture = session_config(settings, source.name)
        _capture_completion(capture, source.name)
        check_output_root(output, [settings, Path(capture["feedback_capture"]["output_root"])])
        args += ["--capture-settings", str(settings)]
    layout = config.get("storage_layout")
    if layout == "day-partitioned.v1":
        args.append("--partitioned")
    elif layout is not None:
        raise FeedbackError("unsupported storage layout")
    return args


def _capture_completion(config, day):
    from .capture import read_capture

    records = read_capture(config)["records"]
    if not any(r.get("kind") == "publication_captured" and r.get("unit") == day for r in records):
        raise FeedbackError("today's committed publication capture is missing")


def _completion(source):
    batch = Batch()

    def read(name):
        return read_json(source / name, root=source, batch=batch)

    inference, render, publication = (
        read(n)
        for n in (
            "coordinator-inference.json",
            "coordinator-render.json",
            "coordinator-publication.json",
        )
    )
    remote = read("remote-verification.json")
    monitors = [
        read(f"monitor-{i}.json") for i in range(4) if (source / f"monitor-{i}.json").is_file()
    ]
    if (
        inference.get("status") != "inferred"
        or render.get("status") != "rendered"
        or publication.get("status") != "published"
        or len(
            {
                inference.get("report_sha256"),
                render.get("report_sha256"),
                publication.get("report_sha256"),
            }
        )
        != 1
        or inference.get("invocation_id") != render.get("invocation_id")
        or remote.get("status") != "contained"
        or remote.get("reports_commit") != publication.get("reports_commit")
        or not any(
            m.get("status") == "verified"
            and m.get("report_date") == source.name
            and m.get("axes", {}).get("remote") == "contained"
            for m in monitors
        )
    ):
        raise FeedbackError("today's briefing completion signals are incomplete")


def _stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass
    # Kill descendants even if their group leader already exited.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def _provider_enabled(config):
    if config.get("mode") != "incremental":
        return False
    provider = config.get("preflight_provider")
    enabled = False
    if provider is not None:
        if not isinstance(provider, dict) or type(provider.get("enabled")) is not bool:
            raise FeedbackError("invalid preflight provider configuration")
        enabled = provider["enabled"]
    required = config.get("require_preflight_provider", False)
    if type(required) is not bool or (required and not enabled):
        raise FeedbackError("incremental runtime requires an enabled preflight provider")
    return enabled


def _refresh_preflight(config, clock, *, config_path):
    """A failed observation never falls back to an older successful snapshot."""
    try:
        own_job_id = os.environ.get("FEEDBACK_JOB_ID")
        if not own_job_id:
            raise FeedbackError("scheduled preflight requires an explicit self job")
        descriptor = os.environ.get("FEEDBACK_CRONICLE_KEY_FD")
        key_fd = int(descriptor) if descriptor is not None else None
        key = preflight_provider._read_key(key_fd)
        payload = preflight_provider.poll_active_jobs(key)
        state = preflight_provider.collect_preflight(
            now=clock(),
            jobs_payload=payload,
            own_job_id=own_job_id,
            expected_event_id=config["preflight_provider"].get("expected_event_id"),
        )
        path = _absolute(config.get("preflight_file"), "preflight_file")
        protected = [
            _absolute(config.get(name), name)
            for name in ("source_root", "checkout", "mapping_file", "output_root")
        ]
        protected.append(config_path)
        check_output_root(path.parent, protected)
        preflight_provider.write_snapshot(path, state, output_root=path.parent)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        raise FeedbackError("preflight provider verification failed") from exc


def _build_environment():
    environment = os.environ.copy()
    for name in ("FEEDBACK_CRONICLE_API_KEY", "FEEDBACK_CRONICLE_KEY_FD"):
        environment.pop(name, None)
    return environment


def run_config(path, *, now=None, clock=None, popen=subprocess.Popen):
    clock = clock or ((lambda: now) if now is not None else (lambda: datetime.now(UTC)))
    now = clock()
    path = Path(path)
    config = read_json(path, root=path.parent, batch=Batch())
    if (
        config.get("schema") != "prediction-feedback-runtime.v1"
        or type(config.get("enabled")) is not bool
    ):
        raise FeedbackError("invalid runtime schema/enabled flag")
    if not config["enabled"]:
        print(json.dumps({"status": "disabled"}))
        return 0
    _code_pin(config)
    use_provider = _provider_enabled(config)
    if not use_provider:
        preflight(config, now)
    args = command(config, now)
    output = Path(config["output_root"])
    check_output_root(output, [path, Path(config["preflight_file"])])
    output.mkdir(parents=True, exist_ok=True)
    timeout = _positive(config.get("timeout_seconds"), "timeout_seconds")
    with (output / ".runtime.lock").open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        previous = {}

        def stop(signum, _frame):
            # poll_active_jobs reaps curl in its finally block. Ignore repeated signals
            # before unwinding that block so cleanup cannot itself be interrupted.
            for other in previous:
                signal.signal(other, signal.SIG_IGN)
            raise _Stopped(signum)

        process = None
        try:
            for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                previous[signum] = signal.signal(signum, stop)
            args = command(config, clock())
            if use_provider:
                _refresh_preflight(config, clock, config_path=path)
                args = command(config, clock())
            preflight(config, clock())
            process = popen(args, start_new_session=True, close_fds=True, env=_build_environment())
            return process.wait(timeout=timeout)
        except _Stopped as exc:
            for signum in previous:
                signal.signal(signum, signal.SIG_IGN)
            if process is not None:
                _stop_group(process)
            return 128 + exc.signum
        except subprocess.TimeoutExpired:
            for signum in previous:
                signal.signal(signum, signal.SIG_IGN)
            if process is not None:
                _stop_group(process)
            return 124
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        return run_config(args.config)
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        print("feedback-runtime: execution failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
