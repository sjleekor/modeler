"""Stage timing and lightweight process resource measurements."""

from __future__ import annotations

import json
import resource
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def _peak_rss_bytes() -> int | None:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if value <= 0:
        return None
    # macOS reports bytes; Linux reports KiB.
    return int(value if value > 10_000_000 else value * 1024)


class StageTimingRecorder:
    """Record completed stages without making timing a scan dependency."""

    def __init__(self) -> None:
        self._active: dict[str, tuple[float, float]] = {}
        self._records: dict[str, dict[str, float | int | None]] = {}

    def start(self, name: str) -> None:
        if name in self._active:
            raise ValueError(f"stage is already running: {name}")
        self._active[name] = (time.monotonic(), time.process_time())

    def stop(self, name: str) -> dict[str, float | int | None]:
        try:
            wall_start, cpu_start = self._active.pop(name)
        except KeyError as exc:
            raise ValueError(f"stage was not started: {name}") from exc
        record: dict[str, float | int | None] = {
            "wall_seconds": time.monotonic() - wall_start,
            "cpu_seconds": time.process_time() - cpu_start,
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        self._records[name] = record
        return record

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def as_dict(self) -> dict[str, object]:
        return {
            "stages": {name: self._records[name] for name in sorted(self._records)},
            "total_wall_seconds": sum(
                float(record["wall_seconds"] or 0) for record in self._records.values()
            ),
            "completed_stage_count": len(self._records),
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp.replace(path)
        return path
