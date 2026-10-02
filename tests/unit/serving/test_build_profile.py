from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from modeler.serving import build_profile as bp


def test_step_samples_temp_and_rss_peaks(tmp_path: Path) -> None:
    lines: list[str] = []
    profiler = bp.BuildProfiler(
        temp_dir=tmp_path, engine={"threads": "2"}, context={"run_id": "r"}, interval=0.01,
        log=lines.append)
    profiler.start()
    with profiler.step("spill"):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "block.tmp").write_bytes(b"x" * 300_000)
        time.sleep(0.15)
        (tmp_path / "sub" / "block.tmp").unlink()
    with profiler.step("idle"):
        pass
    profiler.finish()
    profiler.stop()
    spill, idle = profiler.to_dict()["steps"]
    assert spill["peak_temp_bytes"] >= 300_000 and idle["peak_temp_bytes"] == 0
    assert spill["peak_rss_bytes"] and spill["peak_rss_bytes"] > 0
    assert len(lines) == 2 and lines[0].startswith("[kr-build] 01 spill success")


def test_failed_step_is_recorded_and_sampling_survives_a_missing_temp_dir(tmp_path: Path) -> None:
    profiler = bp.BuildProfiler(
        temp_dir=tmp_path / "gone", engine={}, context={}, interval=0.01, log=lambda _line: None)
    profiler.start()
    with pytest.raises(RuntimeError):
        with profiler.step("bad"):
            time.sleep(0.05)
            raise RuntimeError("boom")
    profiler.fail(RuntimeError("boom"))
    profiler.stop()
    body = profiler.to_dict()
    assert body["status"] == "failed" and body["steps"][0]["status"] == "failed"
    assert body["steps"][0]["peak_temp_bytes"] == 0


def test_row_count_failure_only_loses_the_number(tmp_path: Path) -> None:
    class _Broken:
        def execute(self, _sql):
            raise RuntimeError("no view")

    profiler = bp.BuildProfiler(temp_dir=None, engine={}, context={}, log=lambda _line: None)
    with profiler.step("m", "missing_view", _Broken()):
        pass
    assert profiler.steps[0]["status"] == "success" and profiler.steps[0]["row_count"] is None


def test_profile_is_written_atomically(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "build_profile.json"
    bp.write_json_atomic(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    assert [p.name for p in target.parent.iterdir()] == ["build_profile.json"]
