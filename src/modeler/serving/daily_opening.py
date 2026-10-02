"""Build a private D 10:00 opening artifact from already captured KIS slots."""
from __future__ import annotations

import json
import os
import subprocess
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from .daily_coordinator import SEOUL, _absolute, _atomic, _day_dir, _hash, _read
from .daily_inputs import _release_jobs


def prepare_opening(config: dict[str, Any], day: date) -> dict[str, Any]:
    if config.get("opening_snapshot_root") is None:
        return {"status": "not_configured"}
    state_path = _day_dir(config, day) / "coordinator-opening.json"
    try:
        snapshot_root = _absolute(config, "opening_snapshot_root", exists=True)
        output_root = _absolute(config, "opening_output_root")
        max_age = config.get("opening_max_age_seconds")
        if not isinstance(max_age, int) or not 1 <= max_age <= 3600:
            raise ValueError("opening_max_age_seconds must be 1..3600")
        decision = datetime.combine(day, time(10), SEOUL)
        snapshot_dir = snapshot_root / f"report_date={day.isoformat()}"
        snapshots = []
        if snapshot_dir.is_dir() and not snapshot_dir.is_symlink():
            for path in sorted(snapshot_dir.glob("slot-*.json")):
                if path.is_symlink() or not path.is_file():
                    raise ValueError("opening snapshot path is not a regular file")
                body = _read(path)
                received = datetime.fromisoformat(str(body["received_at"]))
                if received.tzinfo is None or received.utcoffset() is None:
                    raise ValueError("opening snapshot receipt has no timezone")
                if received.astimezone(SEOUL).date() == day and received <= decision:
                    snapshots.append(path)
        release = _read(_absolute(config, "release_manifest"))
        release_root = Path(release["release_root"]).resolve(strict=True)
        _release_jobs(_absolute(config, "release_manifest"))
        argv = [str(_absolute(config, "python")), "-m", "modeler.serving.opening_prepare",
                "--report-date", day.isoformat(), "--calendar-json", str(_absolute(config, "kr_calendar")),
                "--decision-at", decision.isoformat(), "--max-age-seconds", str(max_age),
                "--output-root", str(output_root)]
        for path in snapshots:
            argv.extend(("--snapshot-json", str(path)))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(release_root / "src")
        done = subprocess.run(argv, cwd=release_root, env=env, check=False,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              text=True, timeout=90)
        if done.returncode:
            raise RuntimeError("opening mapper exited nonzero")
        result = json.loads(done.stdout)
        artifact = Path(result["path"])
        if artifact.is_symlink() or output_root.resolve(strict=True) not in artifact.resolve(strict=True).parents:
            raise ValueError("opening artifact escaped its isolated output root")
        body = _read(artifact)
        if body.get("report_date") != day.isoformat() or body.get("decision_at") != decision.isoformat():
            raise ValueError("opening artifact date or cutoff differs")
        state = {"status": body.get("status", "unavailable"), "report_date": day.isoformat(),
                 "artifact_path": str(artifact), "artifact_sha256": _hash(artifact),
                 "snapshot_count": len(snapshots)}
        config["opening_artifact"] = str(artifact)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
        state = {"status": "unavailable", "report_date": day.isoformat(),
                 "error_class": type(exc).__name__}
    _atomic(state_path, state)
    return state
