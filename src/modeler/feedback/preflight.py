"""Bounded loopback Cronicle observation for the feedback runtime's freshness guard."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from .core import FeedbackError, canonical_bytes, require_timestamp

ENDPOINT = "http://127.0.0.1:3012/api/app/get_active_jobs/v1"
MAX_PAYLOAD_BYTES = 1024 * 1024
MAX_SNAPSHOT_BYTES = 16 * 1024
MAX_KEY_BYTES = 4096
_ID = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise FeedbackError("invalid Cronicle identifier")
    return value


def read_available_memory():
    """Read only the host memory counter, never process environments or job parameters."""
    with Path("/proc/meminfo").open() as stream:
        for line in stream:
            fields = line.split()
            if fields and fields[0] == "MemAvailable:":
                if len(fields) != 3 or fields[2] != "kB":
                    break
                return int(fields[1]) * 1024
    raise FeedbackError("available memory counter is unavailable")


def collect_preflight(
    *,
    now,
    jobs_payload: dict,
    own_job_id: str | None,
    expected_event_id: str,
    read_memory=read_available_memory,
    load=lambda: os.getloadavg()[0],
) -> dict:
    """Exclude a single self record only after both its job and event identities match."""
    require_timestamp(now.isoformat())
    _identifier(expected_event_id)
    if own_job_id is not None:
        _identifier(own_job_id)
    if (
        not isinstance(jobs_payload, dict)
        or type(jobs_payload.get("code")) is not int
        or jobs_payload["code"] != 0
        or not isinstance(jobs_payload.get("jobs"), dict)
    ):
        raise FeedbackError("Cronicle active-job verification failed")
    active = []
    excluded = False
    for key, job in jobs_payload["jobs"].items():
        _identifier(key)
        if not isinstance(job, dict) or _identifier(job.get("id")) != key:
            raise FeedbackError("Cronicle job identity is inconsistent")
        if key == own_job_id:
            if job.get("event") != expected_event_id:
                raise FeedbackError("Cronicle self event does not match")
            excluded = True
        else:
            active.append(key)
    if own_job_id is not None and not excluded:
        raise FeedbackError("Cronicle self job is absent")
    memory = read_memory()
    load1 = load()
    if type(memory) is not int or memory <= 0:
        raise FeedbackError("available memory counter is invalid")
    if type(load1) not in (int, float) or not math.isfinite(load1) or load1 < 0:
        raise FeedbackError("host load counter is invalid")
    result = {
        "schema": "prediction-feedback-preflight.v1",
        "observed_at": now.astimezone(UTC).isoformat(),
        "api_code": 0,
        "active_jobs": sorted(active),
        "own_job_id": own_job_id,
        "excluded_own_job": excluded,
        "mem_available_bytes": memory,
        "load1": load1,
    }
    if len(canonical_bytes(result)) > MAX_SNAPSHOT_BYTES:
        raise FeedbackError("preflight snapshot is oversized")
    return result


def _key_config(key):
    if (
        not isinstance(key, str)
        or not key
        or len(key.encode("utf-8")) > MAX_KEY_BYTES
        or any(ord(c) < 32 or ord(c) == 127 for c in key)
    ):
        raise FeedbackError("Cronicle credential is unavailable or invalid")
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'header = "X-API-Key: {escaped}"\n'.encode()


def poll_active_jobs(key, *, popen=subprocess.Popen):
    """curl handles gzip and the inherited proxy/NO_PROXY policy; credentials use stdin."""
    config = _key_config(key)
    child_env = os.environ.copy()
    child_env.pop("FEEDBACK_CRONICLE_API_KEY", None)
    process = popen(
        [
            "curl",
            "--disable",
            "--compressed",
            "-fsS",
            "--max-time",
            "15",
            "--request",
            "GET",
            "--config",
            "-",
            ENDPOINT,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=child_env,
    )
    try:
        process.stdin.write(config)
        process.stdin.close()
        raw = process.stdout.read(MAX_PAYLOAD_BYTES + 1)
        if len(raw) > MAX_PAYLOAD_BYTES:
            raise FeedbackError("Cronicle active-job payload is oversized")
        if process.wait(timeout=20) != 0:
            raise FeedbackError("Cronicle active-job request failed")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise FeedbackError("Cronicle active-job payload is invalid")
        return payload
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise FeedbackError("Cronicle active-job verification failed") from exc
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        if not process.stdin.closed:
            process.stdin.close()


def write_snapshot(path, state, *, output_root):
    """Publish one private file atomically inside an explicit, symlink-free directory."""
    path, root = Path(path), Path(output_root)
    if (
        not path.is_absolute()
        or not root.is_absolute()
        or ".." in path.parts
        or ".." in root.parts
        or root == Path(root.anchor)
        or path.parent != root
    ):
        raise FeedbackError("snapshot must be a direct child of an explicit output root")
    if any(p.is_symlink() for p in [path, root, *root.parents]):
        raise FeedbackError("snapshot output cannot traverse symlinks")
    code_root = Path(__file__).resolve().parents[2]
    if root.is_relative_to(code_root) or code_root.is_relative_to(root):
        raise FeedbackError("snapshot output and code root must be disjoint")
    raw = canonical_bytes(state)
    if len(raw) > MAX_SNAPSHOT_BYTES:
        raise FeedbackError("preflight snapshot is oversized")
    root.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".preflight-", dir=root)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if path.is_symlink():
            raise FeedbackError("snapshot output cannot be a symlink")
        os.replace(name, path)
        directory = os.open(root, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)


def _read_key(key_fd):
    if key_fd is None:
        return os.environ.get("FEEDBACK_CRONICLE_API_KEY")
    if key_fd < 0:
        raise FeedbackError("invalid credential descriptor")
    with os.fdopen(os.dup(key_fd), "rb") as stream:
        raw = stream.read(MAX_KEY_BYTES + 1)
    return raw.decode("utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-event-id", required=True)
    parser.add_argument("--own-job-id")
    parser.add_argument("--mode", choices=("scheduled", "standalone"), default="scheduled")
    parser.add_argument("--key-fd", type=int)
    args = parser.parse_args(argv)
    try:
        if args.mode == "scheduled" and args.own_job_id is None:
            raise FeedbackError("scheduled preflight requires an explicit self job")
        payload = poll_active_jobs(_read_key(args.key_fd))
        # The observation starts when the API response is complete, never before polling.
        state = collect_preflight(
            now=datetime.now(UTC),
            jobs_payload=payload,
            own_job_id=args.own_job_id,
            expected_event_id=args.expected_event_id,
        )
        write_snapshot(args.output, state, output_root=args.output_root)
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        # Neither credentials, raw payloads, curl stderr nor exception messages are logged.
        print("feedback-preflight: observation failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
