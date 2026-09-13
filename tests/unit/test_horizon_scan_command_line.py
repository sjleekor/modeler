"""F-9.10 — run_spec must record the arguments the run actually used.

`command_line = ["horizon_scan", *(argv or [])]` looked right and was wrong for
every real invocation: `__main__` calls `main()` with `argv=None`,
`parse_args(None)` then reads `sys.argv` on its own, and the empty list is what
reached `run_spec.json`. All four 2026-08-30 runs recorded `['horizon_scan']`.

`config_hash` pins the analysis contract either way, so no run's lineage was
ever ambiguous. What was lost is narrower and still matters: you cannot
reproduce a run from its own spec. These tests pin both call shapes, because
the passed-argv one always worked and must keep working.
"""

from __future__ import annotations

import pytest

from modeler.analysis import horizon_scan, horizon_scan_phase_c


def _capture(monkeypatch: pytest.MonkeyPatch, module, attr: str) -> dict:
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return "published/run-dir"

    monkeypatch.setattr(module, attr, fake)
    return seen


def test_argv_from_sys_reaches_the_run_spec(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, horizon_scan, "run_phase_a")
    monkeypatch.setattr(
        "sys.argv",
        ["horizon_scan", "--phase", "A", "--snapshot-date", "2026-09-08", "--workers", "4"],
    )

    assert horizon_scan.main(None) == 0
    assert seen["command_line"] == [
        "horizon_scan",
        "--phase",
        "A",
        "--snapshot-date",
        "2026-09-08",
        "--workers",
        "4",
    ]


def test_explicitly_passed_argv_still_reaches_the_run_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = _capture(monkeypatch, horizon_scan, "run_phase_b_core")

    assert horizon_scan.main(["--phase", "B", "--snapshot-date", "2026-09-08"]) == 0
    assert seen["command_line"] == [
        "horizon_scan",
        "--phase",
        "B",
        "--snapshot-date",
        "2026-09-08",
    ]


def test_an_empty_argv_is_not_confused_with_an_absent_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `argv or []` collapsed these two: `main([])` and `main(None)` produced the
    # same record even though only the first one really had no arguments.
    seen = _capture(monkeypatch, horizon_scan, "run_phase_a")
    monkeypatch.setattr("sys.argv", ["horizon_scan", "--workers", "2"])

    assert horizon_scan.main([]) == 0
    assert seen["command_line"] == ["horizon_scan"]


def test_phase_c_records_its_own_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _capture(monkeypatch, horizon_scan_phase_c, "run_phase_c")
    monkeypatch.setattr(
        "sys.argv",
        [
            "horizon_scan_phase_c",
            "--phase-a-run-dir",
            "research/output/horizon_scan/a",
            "--snapshot-date",
            "2026-09-08",
        ],
    )

    assert horizon_scan_phase_c.main(None) == 0
    assert seen["command_line"] == [
        "horizon_scan_phase_c",
        "--phase-a-run-dir",
        "research/output/horizon_scan/a",
        "--snapshot-date",
        "2026-09-08",
    ]
