"""Per-step build profile for the KR serving mart builder.

Records, for every build step in order, wall time, output row count, the peak
size of the DuckDB temp directory and the peak process RSS. A background thread
samples both while a step runs; sampling never raises into the build.

RSS comes from ``/proc/self/status`` (``VmRSS``) on Linux. Elsewhere only
``resource.getrusage`` is available, which is the *lifetime* maximum of the
process, so per-step peaks there are non-decreasing and ``rss_source`` says so.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import duckdb

SCHEMA_VERSION = "kr-build-profile.v1"
SAMPLE_INTERVAL_SECONDS = 2.0
PROFILE_NAME = "build_profile.json"
PARTIAL_PROFILE_NAME = "build_profile.partial.json"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _log_stderr(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def directory_bytes(path: Path | None) -> int:
    """Total size of regular files under ``path``; files vanishing mid-walk are skipped."""
    if path is None:
        return 0
    total = 0
    stack = [str(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return total


def rss_source() -> str:
    if Path("/proc/self/status").is_file():
        return "proc_self_status_vmrss"
    return "getrusage_lifetime_max"


def current_rss_bytes() -> int | None:
    """Current RSS on Linux, lifetime-max RSS elsewhere, None if unreadable."""
    try:
        if rss_source() == "proc_self_status_vmrss":
            for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
            return None
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # ru_maxrss is bytes on macOS and KiB elsewhere.
        return peak if sys.platform == "darwin" else peak * 1024
    except (OSError, ValueError, ImportError):
        return None


def write_json_atomic(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(body, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class NullProfiler:
    """Drop-in used when a caller does not profile; ``step`` just runs the body."""

    @contextmanager
    def step(self, name: str, output: str | None = None, con=None) -> Iterator[None]:
        yield


class BuildProfiler:
    """Collects ordered step records and a sampled temp/RSS peak per step."""

    def __init__(
        self, *, temp_dir: Path | None, engine: dict, context: dict,
        interval: float = SAMPLE_INTERVAL_SECONDS, log=None,
    ) -> None:
        self._temp_dir = temp_dir
        self._interval = interval
        self._log = log if log is not None else _log_stderr
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._current: dict | None = None
        self._started_at = _now()
        self._started = time.monotonic()
        self.steps: list[dict] = []
        self.status = "running"
        self.error: str | None = None
        self._header = {
            "engine": engine,
            "environment": {
                "duckdb_version": duckdb.__version__, "python_version": platform.python_version(),
                "platform": platform.platform(), "machine": platform.machine(),
                "cpu_count": os.cpu_count(), "rss_source": rss_source(),
            },
            **context,
        }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="build-profile-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 5)

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    def _sample(self) -> None:
        # Sampling must never take the build down: swallow everything.
        try:
            temp, rss = directory_bytes(self._temp_dir), current_rss_bytes()
            with self._lock:
                step = self._current
                if step is None:
                    return
                step["peak_temp_bytes"] = max(step["peak_temp_bytes"], temp)
                if rss is not None:
                    step["peak_rss_bytes"] = max(step["peak_rss_bytes"] or 0, rss)
        except Exception:  # noqa: BLE001
            return

    @contextmanager
    def step(self, name: str, output: str | None = None, con=None) -> Iterator[None]:
        """Profile one step. ``output`` is the mart/view whose rows are counted afterwards."""
        record = {
            "order": len(self.steps) + 1, "name": name, "output": output, "status": "running",
            "started_at": _now(), "finished_at": None, "elapsed_seconds": None,
            "row_count": None, "peak_temp_bytes": 0, "peak_rss_bytes": None,
        }
        self.steps.append(record)
        began = time.monotonic()
        with self._lock:
            self._current = record
        self._sample()
        try:
            yield
        except BaseException:
            record["status"] = "failed"
            raise
        else:
            record["status"] = "success"
        finally:
            self._sample()
            with self._lock:
                self._current = None
            record["elapsed_seconds"] = round(time.monotonic() - began, 3)
            record["finished_at"] = _now()
            if output is not None and con is not None and record["status"] == "success":
                # Counted outside the timed window; a failure only loses the number.
                try:
                    count = con.execute(f"SELECT count(*) FROM {output}").fetchone()[0]
                    record["row_count"] = int(count)
                except Exception:  # noqa: BLE001
                    record["row_count"] = None
            self._emit(record)

    def _emit(self, record: dict) -> None:
        try:
            rows = "n/a" if record["row_count"] is None else f"{record['row_count']:,}"
            rss = record["peak_rss_bytes"]
            self._log(
                f"[kr-build] {record['order']:02d} {record['name']} {record['status']} "
                f"{record['elapsed_seconds']:.1f}s rows={rows} "
                f"temp_peak={record['peak_temp_bytes'] / 1e9:.2f}GB "
                f"rss_peak={'n/a' if rss is None else f'{rss / 1e9:.2f}GB'}"
            )
        except Exception:  # noqa: BLE001
            return

    def fail(self, exc: BaseException) -> None:
        self.status = "failed"
        self.error = f"{type(exc).__name__}: {exc}"

    def finish(self) -> None:
        if self.status == "running":
            self.status = "success"

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION, "status": self.status, "error": self.error,
            "started_at": self._started_at, "finished_at": _now(),
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
            "sample_interval_seconds": self._interval,
            **self._header, "steps": self.steps,
        }
