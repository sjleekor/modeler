"""Real capture CLI, partition catalog and activation configuration boundaries."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_briefing import DAY, write
from test_cli import pinned_document
from test_publisher_capture import hook, published, request_case
from test_runtime import config

from modeler.feedback import runtime
from modeler.feedback.__main__ import main
from modeler.feedback.bundle import pinned_sessions
from modeler.feedback.core import FeedbackError, sha256_bytes
from modeler.feedback.partition import load_catalog
from modeler.feedback.store import load_index


def test_cli_indexes_captured_bytes_twice_after_daily_source_disappears(tmp_path):
    request, _ = request_case(tmp_path)
    pending = hook.handle(request)
    assert published(request, pending)["status"] == "captured"
    source = Path(request["entry"]["run_dir"])
    source.rename(tmp_path / "original-source-removed")
    doc, commit, blob = pinned_document(tmp_path)
    output = tmp_path / "index"
    args = [
        "capture",
        "--source-root",
        str(source),
        "--capture-settings",
        request["settings_path"],
        "--output-root",
        str(output),
        "--partitioned",
        "--issues-document",
        str(doc),
        "--my-commit",
        commit,
        "--issues-blob",
        blob,
    ]
    assert main(args) == main(args) == 0
    entry = load_catalog(output)["days"][DAY]
    assert entry["counts"] == {"predictions": 1, "publications": 1, "occurrences": 1, "errors": 0}
    index = load_index(output / entry["root"])
    assert index["publications"][0]["coverage_status"] == "covered"
    assert index["publications"][0]["eligibility_status"] == "not_evaluated"
    assert len(index["publications"][0]["verification_observations"]) == 1


def test_runtime_requires_matching_calendar_and_builds_capture_partition_command(
    tmp_path, monkeypatch
):
    value = config(tmp_path)
    now = datetime(2026, 11, 9, 1, 36, tzinfo=UTC)
    calendar = tmp_path / "calendar.json"
    write(
        calendar,
        {
            "market": "KR",
            "timezone": "Asia/Seoul",
            "coverage_start": "2026-11-01",
            "coverage_end": "2026-11-30",
            "sessions": ["2026-11-09"],
            "default_open_at": "09:00",
            "default_close_at": "15:30",
        },
    )
    digest = sha256_bytes(calendar.read_bytes())
    settings = tmp_path / "capture.json"
    write(
        settings,
        {
            "schema": "prediction-feedback-capture-settings.v1",
            "enabled": True,
            "epoch_id": "synthetic",
            "output_root": str(tmp_path / "captures"),
        },
    )
    value.update(
        mode="incremental",
        not_before_date="2026-11-07",
        source_mode="publisher-capture",
        storage_layout="day-partitioned.v1",
        capture_settings=str(settings),
        require_calendar_pin=True,
        calendar_file=str(calendar),
        calendar_sha256=digest,
        sessions=pinned_sessions(calendar, digest),
    )
    monkeypatch.setattr(runtime, "_completion", lambda source: None)
    monkeypatch.setattr(runtime, "_capture_completion", lambda capture, day: None)
    command = runtime.command(value, now)
    assert command[5] == "capture"
    assert command[-1] == "--partitioned"
    assert command[command.index("--source-root") + 1].endswith("2026-11-09")
    assert command[command.index("--capture-settings") + 1] == str(settings)
    value["sessions"] = []
    with pytest.raises(FeedbackError, match="pinned confirmed calendar"):
        runtime.command(value, now)
    value["sessions"] = ["2026-11-09"]
    calendar.write_bytes(calendar.read_bytes() + b" ")
    with pytest.raises(FeedbackError, match="pin mismatch"):
        runtime.command(value, now)


def test_capture_output_can_never_be_its_index_destination(tmp_path, monkeypatch):
    request, _ = request_case(tmp_path)
    pending = hook.handle(request)
    assert published(request, pending)["status"] == "captured"
    doc, commit, blob = pinned_document(tmp_path)
    settings = __import__("json").loads(Path(request["settings_path"]).read_bytes())
    assert (
        main(
            [
                "capture",
                "--source-root",
                request["entry"]["run_dir"],
                "--capture-settings",
                request["settings_path"],
                "--output-root",
                settings["output_root"],
                "--partitioned",
                "--issues-document",
                str(doc),
                "--my-commit",
                commit,
                "--issues-blob",
                blob,
            ]
        )
        == 2
    )
    assert not (Path(settings["output_root"]) / "catalog.json").exists()
