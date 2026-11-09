"""Multi-day behaviour of the coordinator's stock_reports publication (publish and monitor stages).

The remote is a local bare repository; GitHub is never contacted.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest
import test_daily_coordinator_e2e as e2e

from modeler.reporting import markdown as md

sys.path.insert(0, str(Path(e2e.PROJECT) / "tests" / "unit" / "reporting"))
from reports_world import REPORTS_DIR, World, git  # noqa: E402

D1 = date(2026, 9, 29)
D2 = date(2026, 9, 30)
WRAPPER = f"""import sys
from pathlib import Path
sys.path.insert(0, {str(REPORTS_DIR)!r})
import publish_reports
flag = Path({{flag!r}})
if flag.exists():
    flag.unlink()
    raise SystemExit(3)
raise SystemExit(publish_reports.main(sys.argv[1:], strict_remote=False))
"""


def _git(*args: str) -> str:
    return git(*args)


def _cli(
    release: Path,
    config: Path,
    stage: str,
    day: date,
    now: str,
    *,
    attempt: int | None = None,
    expect: int | None = 0,
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release / "src")
    cmd = [
        sys.executable,
        "-m",
        "modeler.serving.daily_coordinator",
        stage,
        "--config",
        str(config),
        "--report-date",
        day.isoformat(),
        "--fixture-now",
        now,
    ]
    if attempt is not None:
        cmd += ["--attempt", str(attempt)]
    done = subprocess.run(cmd, cwd=release, env=env, text=True, capture_output=True, timeout=120)
    if expect is not None:
        assert done.returncode == expect, (stage, day, done.stdout, done.stderr)
    return json.loads(done.stdout) if done.stdout else {"returncode": done.returncode}


def _prepare_d2(tmp_path: Path, config: Path) -> None:
    for market in ("kr", "us"):
        name = "KR" if market == "kr" else "US"
        directory = tmp_path / "prepared" / market / "score_date=2026-09-29" / "prep_id=fixture"
        directory.mkdir(parents=True)
        feature = directory / ("feature_panel.parquet" if name == "KR" else "features.parquet")
        feature.write_bytes(b"synthetic model input D2: " + name.encode())
        native = e2e._write(
            directory / ("prepare_manifest.json" if name == "KR" else "manifest.json"),
            {
                "market": name,
                "feature_asof_date": "2026-09-29",
                "input_sha256": e2e._sha(feature),
                "features_sha256": e2e._sha(feature),
                "synthetic_fixture": True,
                "availability_evidence_type": "prepared_features_completion",
            },
        )
        e2e._write(
            directory / "completion.json",
            {
                "schema_version": "prepared-features-completion.v1",
                "verified_available_by": "2026-09-30T09:20:00+09:00",
                "availability_evidence_type": "prepared_features_completion",
                "features_sha256": e2e._sha(feature),
                "native_prepare_manifest_sha256": e2e._sha(native),
            },
        )
    body = json.loads(config.read_text())
    for key in ("kr_calendar", "us_calendar"):
        path = Path(body[key])
        calendar = json.loads(path.read_text())
        calendar["sessions"].append("2026-09-30")
        calendar["coverage_end"] = "2026-09-30"
        e2e._write(path, calendar)
    path = Path(body["us_expected_source"])
    expected = json.loads(path.read_text())
    expected["expected_session_by_report_date"][D2.isoformat()] = "2026-09-29"
    e2e._write(path, expected)


def _enable_publisher(
    tmp_path: Path, config: Path, *, flag: Path | None = None, external_verification: bool = False
) -> World:
    world = World(tmp_path)
    script = tmp_path / "publisher_wrapper.py"
    script.write_text(WRAPPER.replace("{flag!r}", repr(str(flag or tmp_path / "no-such-flag"))))
    body = json.loads(config.read_text())
    body.update(
        {
            "publisher_enabled": True,
            "external_verification_enabled": external_verification, "explain_enabled": False,
            "macro_context_enabled": False,
            "reports_publisher": str(script),
            "reports_publisher_sha256": e2e._sha(script),
            "reports_checkout": str(world.checkout),
            "reports_remote_url": str(world.bare),
            "reports_repository": "sjleekor/stock_reports",
            "reports_audience": "owner_only",
            "reports_branch": "main",
            "reports_top_n": 100,
        }
    )
    e2e._write(config, body)
    return world


def _day(release: Path, config: Path, day: date, now_select: str, now_run: str) -> None:
    assert _cli(release, config, "select", day, now_select)["status"] == "selected"
    assert _cli(release, config, "infer", day, now_run)["status"] == "inferred"
    assert _cli(release, config, "render", day, now_run)["status"] == "rendered"


def _unit(day: date) -> str:
    return f"reports/daily-briefing/{day:%Y}/{day:%m}/{day}"


def _publication(tmp_path: Path, day: date) -> dict:
    return json.loads(
        (tmp_path / "runs" / day.isoformat() / "coordinator-publication.json").read_text()
    )


def _two_days(tmp_path: Path):
    release, config = e2e._setup(tmp_path)
    world = _enable_publisher(tmp_path, config)
    _day(release, config, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    assert (
        _cli(release, config, "publish", D1, "2026-09-29T10:00:00+09:00")["status"] == "published"
    )
    _prepare_d2(tmp_path, config)
    _day(release, config, D2, "2026-09-30T09:30:00+09:00", "2026-09-30T10:00:00+09:00")
    return release, config, world


def test_second_day_publishes_a_fast_forward_commit_on_top_of_the_first(tmp_path: Path) -> None:
    release, config, world = _two_days(tmp_path)
    runs = tmp_path / "runs"
    first = world.remote_head()
    assert _publication(tmp_path, D1)["reports_commit"] == first
    assert world.remote_log()[0] == "daily-briefing 2026-09-29"
    assert (
        _cli(release, config, "publish", D2, "2026-09-30T10:00:00+09:00")["status"] == "published"
    )
    second = world.remote_head()
    assert second != first
    assert _git("--git-dir", str(world.bare), "merge-base", "--is-ancestor", first, second) == ""
    assert _git("--git-dir", str(world.bare), "rev-parse", second + "^") == first
    assert world.remote_log()[:2] == ["daily-briefing 2026-09-30", "daily-briefing 2026-09-29"]
    for day in (D1, D2):
        assert world.remote_has(f"{_unit(day)}/README.md")
        assert (runs / day.isoformat() / "markdown" / "unit" / "README.md").is_file()
    month = world.remote_file("reports/daily-briefing/2026/09/README.md")
    assert "2026-09-29" in month and "2026-09-30" in month
    # The day-scoped publisher input carries only this day's pins and no private or Pages values.
    c1 = runs / "2026-09-29" / "reports-publisher-config.json"
    c2 = runs / "2026-09-30" / "reports-publisher-config.json"
    j1, j2 = json.loads(c1.read_text()), json.loads(c2.read_text())
    assert j1["report_date"] == "2026-09-29" and j2["report_date"] == "2026-09-30"
    assert j1["report_sha256"] != j2["report_sha256"] and j1["invocation_id"] != j2["invocation_id"]
    assert j1["audience"] == "owner_only" and j1["release"] == "frozen-release"
    assert {
        k: v
        for k, v in j1.items()
        if k not in {"run_dir", "report_date", "report_sha256", "invocation_id"}
    } == {
        k: v
        for k, v in j2.items()
        if k not in {"run_dir", "report_date", "report_sha256", "invocation_id"}
    }
    publication = _publication(tmp_path, D2)
    assert publication["publisher_config_path"] == str(c2)
    assert publication["publisher_config_sha256"] == hashlib.sha256(c2.read_bytes()).hexdigest()
    assert publication["reports_commit"] == second and publication["publisher_mode"] == "run"
    assert publication["local_done"] is True and publication["publisher_exit_code"] == 0


def test_publish_does_not_run_again_for_a_changed_report(tmp_path: Path) -> None:
    release, config = e2e._setup(tmp_path)
    _enable_publisher(tmp_path, config)
    _day(release, config, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    report = tmp_path / "runs" / "2026-09-29" / "report-2026-09-29.json"
    report.write_text(report.read_text() + " ")
    result = _cli(release, config, "publish", D1, "2026-09-29T10:00:00+09:00", expect=1)
    assert result["returncode"] == 1
    assert not (tmp_path / "runs" / "2026-09-29" / "reports-publisher-config.json").exists()


def test_publisher_failure_is_retried_by_the_monitor_with_the_same_day_config(
    tmp_path: Path,
) -> None:
    release, config = e2e._setup(tmp_path)
    flag = tmp_path / "fail-once"
    flag.write_text("x")
    world = _enable_publisher(tmp_path, config, flag=flag)
    _day(release, config, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    failed = _cli(release, config, "publish", D1, "2026-09-29T10:00:00+09:00", expect=1)
    assert failed["status"] == "publisher_failed"
    daily = tmp_path / "runs" / "2026-09-29" / "reports-publisher-config.json"
    first_bytes = daily.read_bytes()
    assert (
        _publication(tmp_path, D1)["local_done"] is False
    )  # the publisher died before its local step
    monitor = _cli(release, config, "monitor", D1, "2026-09-29T10:15:00+09:00", attempt=1, expect=1)
    assert monitor["status"] == "remote_pending"
    publication = _publication(tmp_path, D1)
    assert publication["status"] == "published" and publication["publisher_mode"] == "run"
    assert daily.read_bytes() == first_bytes
    assert publication["publisher_config_sha256"] == hashlib.sha256(first_bytes).hexdigest()
    assert world.remote_head() == publication["reports_commit"]


def test_remote_outage_keeps_the_unit_then_monitor_retries_only_the_sync_step(
    tmp_path: Path,
) -> None:
    release, config = e2e._setup(tmp_path)
    world = _enable_publisher(tmp_path, config, external_verification=True)
    _day(release, config, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    world.go_offline()
    failed = _cli(release, config, "publish", D1, "2026-09-29T10:00:00+09:00", expect=1)
    assert failed["status"] == "publisher_failed"
    publication = _publication(tmp_path, D1)
    assert (
        publication["publisher_status"] == "sync_pending"
        and publication["publisher_exit_code"] == 20
    )
    assert publication["local_done"] is True and "reports_commit" not in publication
    assert (tmp_path / "runs" / "2026-09-29" / "markdown" / "unit" / "README.md").is_file()
    assert world.journal()["units"]["2026-09-29"]["state"] == "sync_pending"
    # Still down at 10:15: the monitor tries the sync step again and reports the publisher failure.
    still = _cli(release, config, "monitor", D1, "2026-09-29T10:15:00+09:00", attempt=1, expect=1)
    assert still["status"] == "publisher_failed"
    assert _publication(tmp_path, D1)["publisher_mode"] == "sync"
    world.come_online()
    done = _cli(release, config, "monitor", D1, "2026-09-29T10:17:00+09:00", attempt=2)
    assert done["status"] == "verified"
    publication = _publication(tmp_path, D1)
    assert (
        publication["publisher_mode"] == "sync"
        and publication["reports_commit"] == world.remote_head()
    )
    assert world.journal() is None


def test_monitor_checks_that_remote_main_contains_the_pushed_commit(tmp_path: Path) -> None:
    release, config = e2e._setup(tmp_path)
    world = _enable_publisher(tmp_path, config, external_verification=True)
    seed_head = world.remote_head()
    _day(release, config, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    assert (
        _cli(release, config, "publish", D1, "2026-09-29T10:00:00+09:00")["status"] == "published"
    )
    pushed = world.remote_head()
    verified = _cli(release, config, "monitor", D1, "2026-09-29T10:15:00+09:00", attempt=0)
    assert verified["status"] == "verified"
    state = json.loads((tmp_path / "runs" / "2026-09-29" / "monitor-0.json").read_text())
    assert state["axes"]["remote"] == "contained" and state["retry_allowed"] is False
    # The owner pushes another commit later: the publisher's commit is still contained.
    world.user_push("reference/after.md")
    assert world.remote_head() != pushed
    later = _cli(release, config, "monitor", D1, "2026-09-29T10:17:00+09:00", attempt=1)
    assert later["status"] == "verified"
    # If the commit is gone from remote main, the monitor says so instead of verifying.
    _git("--git-dir", str(world.bare), "update-ref", "refs/heads/main", seed_head)
    gone = _cli(release, config, "monitor", D1, "2026-09-29T10:22:00+09:00", attempt=2, expect=1)
    assert gone["status"] == "remote_missing_commit"
    # An unreachable remote is "unavailable": the monitor stays pending instead of failing the run.
    world.go_offline()
    down = _cli(release, config, "monitor", D1, "2026-09-29T10:32:00+09:00", attempt=3, expect=1)
    assert down["status"] == "remote_pending"
    assert (
        json.loads((tmp_path / "runs" / "2026-09-29" / "monitor-3.json").read_text())["axes"][
            "remote"
        ]
        == "unavailable"
    )


def test_monitor_stays_pending_without_external_verification(tmp_path: Path) -> None:
    release, config = e2e._setup(tmp_path)
    _enable_publisher(tmp_path, config, external_verification=False)
    _day(release, config, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    assert (
        _cli(release, config, "publish", D1, "2026-09-29T10:00:00+09:00")["status"] == "published"
    )
    result = _cli(release, config, "monitor", D1, "2026-09-29T10:15:00+09:00", attempt=0, expect=1)
    assert result["status"] == "remote_pending"
    state = json.loads((tmp_path / "runs" / "2026-09-29" / "monitor-0.json").read_text())
    assert state["axes"]["remote"] == "pending" and state["retry_allowed"] is True


def test_next_day_run_flushes_a_day_that_never_reached_the_remote(tmp_path: Path) -> None:
    release, config = e2e._setup(tmp_path)
    world = _enable_publisher(tmp_path, config)
    _day(release, config, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    world.go_offline()
    assert (
        _cli(release, config, "publish", D1, "2026-09-29T10:00:00+09:00", expect=1)["status"]
        == "publisher_failed"
    )
    _prepare_d2(tmp_path, config)
    _day(release, config, D2, "2026-09-30T09:30:00+09:00", "2026-09-30T10:00:00+09:00")
    world.come_online()
    assert (
        _cli(release, config, "publish", D2, "2026-09-30T10:00:00+09:00")["status"] == "published"
    )
    assert world.remote_log()[:2] == ["daily-briefing 2026-09-30", "daily-briefing 2026-09-29"]
    assert world.journal() is None
    assert _publication(tmp_path, D2)["reports_commit"] == world.remote_head()


def test_rerunning_a_published_day_with_the_same_report_makes_no_new_commit(tmp_path: Path) -> None:
    release, config = e2e._setup(tmp_path)
    world = _enable_publisher(tmp_path, config)
    _day(release, config, D1, "2026-09-29T09:30:00+09:00", "2026-09-29T10:00:00+09:00")
    assert (
        _cli(release, config, "publish", D1, "2026-09-29T10:00:00+09:00")["status"] == "published"
    )
    head, count = world.remote_head(), len(world.remote_log())
    # The runner reuses the finished jobs, so the second inference writes the same report.
    assert _cli(release, config, "infer", D1, "2026-09-29T10:40:00+09:00")["status"] == "inferred"
    assert _cli(release, config, "render", D1, "2026-09-29T10:40:00+09:00")["status"] == "rendered"
    assert (
        _cli(release, config, "publish", D1, "2026-09-29T10:40:00+09:00")["status"] == "published"
    )
    publication = _publication(tmp_path, D1)
    assert publication["publisher_status"] == "unchanged" and publication["reports_commit"] == head
    assert world.remote_head() == head and len(world.remote_log()) == count
    assert world.journal() is None


def _wrapper_run(release: Path, config: Path, day: date, now: str) -> dict:
    env = {**os.environ, "PYTHONPATH": str(release / "src")}
    base = [sys.executable, "-m", "modeler.serving.daily_wrapper"]
    tail = ["--config", str(config), "--report-date", day.isoformat(), "--fixture-now"]
    select = subprocess.run([*base, "select", *tail, "2026-09-29T09:30:00+09:00"], cwd=release, env=env,
                            text=True, capture_output=True, timeout=120)
    assert select.returncode == 0, (select.stdout, select.stderr)
    done = subprocess.run([*base, "run", *tail, now], cwd=release, env=env, text=True,
                          capture_output=True, timeout=300)
    assert done.returncode == 0, (done.stdout, done.stderr)
    return json.loads(done.stdout)["stages"]


def _unit_files(world: World, day: date) -> dict[str, bytes]:
    root = world.checkout / _unit(day)
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _pushed_files(world: World, day: date) -> set[str]:
    listing = _git("--git-dir", str(world.bare), "ls-tree", "-r", "--name-only", "main", _unit(day))
    return {line[len(_unit(day)) + 1:] for line in listing.splitlines()}


def test_publisher_input_carries_the_explain_switch(tmp_path: Path) -> None:
    from modeler.serving import daily_coordinator as dc

    _release, config_path = e2e._setup(tmp_path)
    _enable_publisher(tmp_path, config_path)
    for value in (True, False):
        body = {**json.loads(config_path.read_text()), "explain_enabled": value}
        built = dc._reports_publisher_input(
            body, D1, {"report_sha256": "a" * 64, "invocation_id": "i"}, tmp_path / "run")
        assert built["explain_enabled"] is value


def test_wrapper_run_with_explain_on_documents_the_synthetic_failure_and_still_publishes(
        tmp_path: Path) -> None:
    """Synthetic bundles are not the three real models, so explain ends in the documented failed
    document (no svg assets); the unit still publishes and records the explain file it read."""
    release, config = e2e._setup(tmp_path)
    world = _enable_publisher(tmp_path, config)
    body = json.loads(config.read_text())
    body["explain_enabled"] = True
    e2e._write(config, body)
    stages = _wrapper_run(release, config, D1, "2026-09-29T10:00:00+09:00")
    assert stages["explain"] == "failed" and stages["publication"] == "published"
    run = tmp_path / "runs" / D1.isoformat()
    state = json.loads((run / "coordinator-explain.json").read_text())
    document = json.loads((run / f"explain-{D1}.json").read_text())
    inference = json.loads((run / "coordinator-inference.json").read_text())
    assert state["status"] == "failed"
    assert document["report_sha256"] == inference["report_sha256"] == e2e._sha(run / f"report-{D1}.json")
    assert document["invocation_id"] == inference["invocation_id"]
    assert json.loads((run / "reports-publisher-config.json").read_text())["explain_enabled"] is True
    local = _unit_files(world, D1)
    assert not any(name.startswith("assets/") for name in local)
    assert not any(name.startswith("assets/") for name in _pushed_files(world, D1))
    document_sha = e2e._sha(run / f"explain-{D1}.json")  # the unit records which explain file it read
    assert all(f"explain_sha256: {document_sha}" in data.decode()
               for name, data in local.items() if name.endswith(".md"))


def test_wrapper_run_with_explain_off_writes_no_explain_state_or_file(tmp_path: Path) -> None:
    release, config = e2e._setup(tmp_path)
    world = _enable_publisher(tmp_path, config)  # explain_enabled False
    stages = _wrapper_run(release, config, D1, "2026-09-29T10:00:00+09:00")
    assert "explain" not in stages and stages["publication"] == "published"
    run = tmp_path / "runs" / D1.isoformat()
    assert not (run / f"explain-{D1}.json").exists()
    assert json.loads((run / "reports-publisher-config.json").read_text())["explain_enabled"] is False
    local = _unit_files(world, D1)
    assert not any(name.startswith("assets/") for name in local)
    assert not any("explain_sha256" in data.decode() for data in local.values())
    assert set(local) == _pushed_files(world, D1)


def test_publisher_input_carries_the_macro_switch(tmp_path: Path) -> None:
    from modeler.serving import daily_coordinator as dc

    _release, config_path = e2e._setup(tmp_path)
    _enable_publisher(tmp_path, config_path)
    base = json.loads(config_path.read_text())
    render = {"report_sha256": "a" * 64, "invocation_id": "i"}
    built = dc._reports_publisher_input(
        {**base, "macro_context_enabled": True}, D1, render, tmp_path / "run")
    assert built["macro_context_enabled"] is True
    for off in ({**base, "macro_context_enabled": False},
                {k: v for k, v in base.items() if k != "macro_context_enabled"}):
        assert "macro_context_enabled" not in dc._reports_publisher_input(
            off, D1, render, tmp_path / "run")


def _old_publisher(tmp_path: Path):
    """publish_reports.py as of b0262b0 (before the macro section), loaded from git."""
    import importlib.util

    done = subprocess.run(["git", "-C", str(e2e.PROJECT), "show",
                           "b0262b0:deploy/reports/publish_reports.py"],
                          capture_output=True, text=True, check=False)
    if done.returncode:
        pytest.skip("b0262b0 is not in this checkout")
    path = tmp_path / "publish_reports_b0262b0.py"
    path.write_text(done.stdout, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("publish_reports_b0262b0", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_an_old_publisher_accepts_the_input_while_the_section_is_off(tmp_path: Path) -> None:
    from modeler.serving import daily_coordinator as dc

    old = _old_publisher(tmp_path)
    _release, config_path = e2e._setup(tmp_path)
    _enable_publisher(tmp_path, config_path)
    base = json.loads(config_path.read_text())
    render = {"report_sha256": "a" * 64, "invocation_id": "inv"}
    for off in ({**base, "macro_context_enabled": False},
                {k: v for k, v in base.items() if k != "macro_context_enabled"}):
        built = dc._reports_publisher_input(off, D1, render, tmp_path / "run")
        old.check_config(built, need_local=True, strict_remote=False)  # does not raise
    on = dc._reports_publisher_input({**base, "macro_context_enabled": True}, D1, render,
                                     tmp_path / "run")
    with pytest.raises(old.PublishError, match="unknown config keys"):
        old.check_config(on, need_local=True, strict_remote=False)  # needs the new publisher


def test_wrapper_run_without_the_switch_key_is_the_unit_without_the_section(
        tmp_path: Path) -> None:
    """ops.json without macro_context_enabled (as before the section): select/run go on, the
    publisher gets no key and the unit has no trace of the section (the renderer's byte identity
    with the golden hashes is in test_macro_section_render)."""
    import macro_fakes as mk

    release, config = e2e._setup(tmp_path, extra_sources=mk.RELEASE_SOURCES)
    world = _enable_publisher(tmp_path, config)
    body = json.loads(config.read_text())
    del body["macro_context_enabled"]
    e2e._write(config, body)
    stages = _wrapper_run(release, config, D1, "2026-09-29T10:00:00+09:00")
    assert "macro_context" not in stages and stages["publication"] == "published"
    run = tmp_path / "runs" / D1.isoformat()
    assert not (run / f"macro-{D1}.json").exists()
    assert json.loads((run / "coordinator-macro-context.json").read_text())["status"] == "disabled"
    assert "macro_context_enabled" not in json.loads(
        (run / "reports-publisher-config.json").read_text())
    unit = _unit_files(world, D1)
    assert set(unit) == set(md.SECTION_FILES.values()) == _pushed_files(world, D1)
    assert all(b"macro" not in data for data in unit.values())
    manifest = json.loads((run / "markdown" / "manifest.json").read_text())
    assert not any(key.startswith("macro_") for key in manifest)


def test_wrapper_run_with_macro_on_but_no_market_sector_selection_publishes_a_notice(
    tmp_path: Path,
) -> None:
    """This release has no market-sector section, so 09:30 select pins no selection document for
    the macro calculation: the section fails with ``not_in_selection`` and the unit still goes out
    with the failure notice (the ok path with a pinned document is in test_macro_context_stage)."""
    import macro_fakes as mk

    release, config = e2e._setup(tmp_path, extra_sources=mk.RELEASE_SOURCES)
    world = _enable_publisher(tmp_path, config)
    body = json.loads(config.read_text())
    body["macro_context_enabled"] = True
    e2e._write(config, body)
    stages = _wrapper_run(release, config, D1, "2026-09-29T10:00:00+09:00")
    assert stages["macro_context"] == "failed" and stages["publication"] == "published"
    run = tmp_path / "runs" / D1.isoformat()
    document = json.loads((run / f"macro-{D1}.json").read_text())
    inference = json.loads((run / "coordinator-inference.json").read_text())
    assert document["invocation_id"] == inference["invocation_id"]
    assert document["state"] == "failed" and document["reason"] == "not_in_selection"
    daily = json.loads((run / "reports-publisher-config.json").read_text())
    assert daily["macro_context_enabled"] is True
    pushed = _pushed_files(world, D1)
    assert "macro.md" in pushed and not any(name.startswith("assets/") for name in pushed)
    local = _unit_files(world, D1)
    assert "거시 흐름을 내지 못했습니다: not_in_selection." in local["macro.md"].decode()
    assert "| [거시 흐름](macro.md) | 실패 |" in local["README.md"].decode()
    manifest = json.loads((run / "markdown" / "manifest.json").read_text())
    assert manifest["macro_state"] == "failed"
    assert manifest["macro_path"] == str(run / f"macro-{D1}.json")


def test_wrapper_run_with_macro_off_writes_no_macro_state_or_file(tmp_path: Path) -> None:
    release, config = e2e._setup(tmp_path)
    world = _enable_publisher(tmp_path, config)  # macro_context_enabled False
    stages = _wrapper_run(release, config, D1, "2026-09-29T10:00:00+09:00")
    assert "macro_context" not in stages and stages["publication"] == "published"
    run = tmp_path / "runs" / D1.isoformat()
    assert not (run / f"macro-{D1}.json").exists()
    daily = json.loads((run / "reports-publisher-config.json").read_text())
    assert "macro_context_enabled" not in daily  # off adds no key (old publishers refuse it)
    pushed = _pushed_files(world, D1)
    assert "macro.md" not in pushed and not any(name.startswith("assets/") for name in pushed)
    manifest = json.loads((run / "markdown" / "manifest.json").read_text())
    assert not any(key.startswith("macro_") for key in manifest)
