"""R3: nine "what comes out that day" scenarios of 04_run_without_waiting section 6, end to end.

Each one runs the real ``daily_wrapper`` stages (select 09:30, run 10:00) on a frozen synthetic
release with a fixture clock, then looks at the internal report and at the markdown renderer's
reading of it.  Report date D = 2026-09-29, K = 09-28.  Weekdays are the sessions.

  1 normal day            2 DART chain failed (K' prepared)    3 K only partly collected (K')
  4 runner timeout        5 KR prepare failed (older prepared) 6 US A missing (A')
  7 select did not run    8 GitHub unreachable                 9 everything failed

Scenario 8 reuses the publisher outage flow of ``test_daily_multiday_publish``; here it is shown
for the failure unit of scenario 9, which has to pass the publisher's own checks too.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import r3_world as w
import test_daily_coordinator_e2e as e2e
import test_daily_multiday_publish as mday

from modeler.reporting import markdown
from modeler.serving import daily_coordinator, daily_wrapper

KR, US_LGBM, US_RIDGE = (model for _, model in w.MODELS)
EARLY = "2026-09-29T09:20:00+09:00"  # an input that finished before the 09:30 cutoff
LATE = "2026-09-29T09:40:00+09:00"  # one that finished after it


def _states(done: subprocess.CompletedProcess[str]) -> dict:
    return json.loads(done.stdout)["stages"]


def _day(
    release: Path, config: Path, *, select_stage: bool = True
) -> subprocess.CompletedProcess[str]:
    if select_stage:
        selected = w.wrapper(release, config, "select", w.SELECT_AT)
        assert selected.returncode == 0, selected.stderr
    return w.wrapper(release, config, "run", w.RUN_AT)


def _context(tmp_path: Path) -> dict:
    saved = w.run_dir(tmp_path) / f"report-{w.D}.json"
    return markdown.build_context(
        tmp_path / "no-checkout",
        saved.read_bytes(),
        None,
        "r3-test",
        None,
        100,
        lambda message: None,
    )


def _lines(rendered: tuple[list, dict]) -> str:
    return "\n".join(rendered[0])


# ---- 1. normal day ------------------------------------------------------------------------


def test_1_normal_day_all_sections_ok(tmp_path: Path, monkeypatch) -> None:
    release, config = w.build(
        tmp_path, monkeypatch, kr=[("2026-09-28", EARLY)], us=[("2026-09-28", EARLY)]
    )
    done = _day(release, config)
    assert done.returncode == 0, done.stderr
    assert _states(done)["inference"] == "inferred" and _states(done)["render"] == "rendered"
    rep = w.report(tmp_path)
    assert rep["status"] == "ok" and rep["failures"] == []  # every section ok -> ok (change 6)
    assert {item["status"] for item in rep["markets"]} == {"ok"}
    quality = w.section(rep, KR)["quality"]
    assert quality["selection_mode"] == "scheduled" and quality["lag_sessions"] == 0
    assert quality["producer_completed_at"] == EARLY and quality["freshness_status"] == "ok"
    state = json.loads((w.run_dir(tmp_path) / "coordinator-inference.json").read_text())
    assert state["report_ready"] is True and state["failure_report"] is False
    assert (
        state["invocation_id"]
        == json.loads((w.run_dir(tmp_path) / "run-state.json").read_text())["invocation_id"]
    )
    ctx = _context(tmp_path)
    assert (ctx["kr"]["status"], ctx["us"]["status"]) == ("ok", "ok")
    assert "세션 전 기준" not in _lines(markdown.render_kr(ctx))


# ---- 2 + 3. K is not usable: the fallback session K' is served, marked, with its lag ----------


@pytest.mark.parametrize("kprime,lag", [("2026-09-25", 1), ("2026-09-23", 3)])
def test_2_3_kr_fallback_session_is_stale_with_the_lag_in_the_banner(
    tmp_path: Path, monkeypatch, kprime: str, lag: int
) -> None:
    """DART chain failed / K only partly collected: kr_prepare built K' (here: the only prepared
    input), the selector serves it as stale and the report says how many sessions it is behind."""
    release, config = w.build(
        tmp_path, monkeypatch, kr=[(kprime, EARLY)], us=[("2026-09-28", EARLY)]
    )
    done = _day(release, config)
    assert done.returncode == 0, done.stderr
    kr = w.section(w.report(tmp_path), KR)
    assert kr["status"] == "stale" and kr["feature_asof_date"] == kprime
    assert kr["quality"]["lag_sessions"] == lag and kr["quality"]["freshness_status"] == "stale"
    assert len(kr["rankings"]) == 3 and "rankings_withheld" not in kr["quality"]
    assert (
        w.section(w.report(tmp_path), US_LGBM)["status"] == "ok"
    )  # the other market is unaffected
    assert w.report(tmp_path)["status"] == "partial"
    ctx = _context(tmp_path)
    assert ctx["kr"]["status"] == "stale" and ctx["kr"]["models"][0]["lag"] == lag
    text = _lines(markdown.render_kr(ctx))
    assert f"{lag}세션 전 기준 순위입니다" in text and kprime in text


# ---- 4. runner timeout -----------------------------------------------------------------------


def test_4_runner_timeout_still_makes_this_invocations_failed_unit(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    release, config = w.build(
        tmp_path, monkeypatch, kr=[("2026-09-28", EARLY)], us=[("2026-09-28", EARLY)]
    )
    assert _day(release, config).returncode == 0  # the first, healthy run of D
    first = w.report(tmp_path)
    first_bytes = (w.run_dir(tmp_path) / f"report-{w.D}.json").read_bytes()
    assert first["status"] == "ok"
    # The saved per-model results would be reused (and finish at once); remove them so the second
    # run really has to infer again, while the first run's report-D.json stays in place.
    shutil.rmtree(w.run_dir(tmp_path) / "inference")
    slow = tmp_path / "slow.flag"
    slow.write_text("x")
    monkeypatch.setenv("R3_SLOW_FLAG", str(slow))
    monkeypatch.setattr(daily_coordinator, "RUNNER_TIMEOUT_SECONDS", 2)
    started = time.monotonic()
    code = daily_wrapper.main(
        [
            "run",
            "--config",
            str(config),
            "--report-date",
            w.D.isoformat(),
            "--fixture-now",
            w.RUN_AT,
        ]
    )
    assert time.monotonic() - started < 60
    stages = json.loads(capsys.readouterr().out)["stages"]
    assert code == 1 and stages["inference"] == "inference_failed"
    assert stages["render"] == "rendered"  # the unit is still made
    second = w.report(tmp_path)
    assert second["invocation_id"] != first.get("invocation_id")  # not the earlier run's report
    assert second["status"] == "failed" and second["markets"] == []
    assert second["failure"] == {
        "stage": "infer",
        "error_class": "RunnerTimeout",
        "runner_exit_code": 124,
        "timed_out": True,
        "synthesized_by": "coordinator",
    }
    assert {item["error_class"] for item in second["failures"]} == {"RunnerTimeout"}
    assert {(item["market"], item["model_id"]) for item in second["failures"]} == set(w.MODELS)
    # The earlier report was set aside, never reused as this run's.
    (kept,) = list(w.run_dir(tmp_path).glob(f"report-{w.D}.superseded-*.json"))
    assert kept.read_bytes() == first_bytes
    state = json.loads((w.run_dir(tmp_path) / "coordinator-inference.json").read_text())
    assert state["failure_report"] is True and state["timed_out"] is True
    run_state = json.loads((w.run_dir(tmp_path) / "run-state.json").read_text())
    assert run_state["invocation_id"] == second["invocation_id"]
    ctx = _context(tmp_path)
    assert ctx["status"] == "failed" and [m["status"] for m in ctx["kr"]["models"]] == ["failed"]
    assert "RunnerTimeout" in _lines(markdown.render_kr(ctx))


def test_4_timeout_unit_passes_the_publisher_and_the_stock_reports_validator(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    release, config = w.build(
        tmp_path, monkeypatch, kr=[("2026-09-28", EARLY)], us=[("2026-09-28", EARLY)]
    )
    world = mday._enable_publisher(tmp_path, config)
    assert w.wrapper(release, config, "select", w.SELECT_AT).returncode == 0
    slow = tmp_path / "slow.flag"
    slow.write_text("x")
    monkeypatch.setenv("R3_SLOW_FLAG", str(slow))
    monkeypatch.setattr(daily_coordinator, "RUNNER_TIMEOUT_SECONDS", 2)
    code = daily_wrapper.main(
        [
            "run",
            "--config",
            str(config),
            "--report-date",
            w.D.isoformat(),
            "--fixture-now",
            w.RUN_AT,
        ]
    )
    stages = json.loads(capsys.readouterr().out)["stages"]
    assert code == 1 and stages["inference"] == "inference_failed"
    assert stages["publication"] == "published", stages
    unit = world.checkout / mday._unit(w.D)
    assert "RunnerTimeout" in (unit / "data-status.md").read_text()
    assert markdown.validate(world.checkout, allow_synthetic=True) == []
    assert world.remote_head()


def test_4_runner_crash_and_missing_report_also_make_a_failed_unit(
    tmp_path: Path, monkeypatch
) -> None:
    release, config = w.build(tmp_path, monkeypatch, kr=[("2026-09-28", EARLY)])
    assert w.wrapper(release, config, "select", w.SELECT_AT).returncode == 0
    monkeypatch.setattr(
        daily_coordinator,
        "run_group",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 1, None, None),
    )
    state = daily_coordinator.infer_stage(
        daily_coordinator._config(config), w.D, daily_coordinator.datetime.fromisoformat(w.RUN_AT)
    )
    assert state["status"] == "inference_failed" and state["error_class"] == "RunnerExitNonzero"
    assert state["report_ready"] is True and state["runner_exit_code"] == 1
    assert w.report(tmp_path)["failure"]["error_class"] == "RunnerExitNonzero"
    # exit 0 but no report of this invocation: not a success either
    monkeypatch.setattr(
        daily_coordinator,
        "run_group",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, None, None),
    )
    state = daily_coordinator.infer_stage(
        daily_coordinator._config(config), w.D, daily_coordinator.datetime.fromisoformat(w.RUN_AT)
    )
    assert state["error_class"] == "RunnerReportMissing"
    assert (
        daily_coordinator.render_stage(daily_coordinator._config(config), w.D)["status"]
        == "rendered"
    )


def test_4_failure_report_also_renders_the_private_view(tmp_path: Path, monkeypatch) -> None:
    release, config = w.build(tmp_path, monkeypatch, kr=[("2026-09-28", EARLY)])
    body = json.loads(config.read_text())
    body["private_projection_root"] = str(tmp_path / "private")
    e2e._write(config, body)
    assert w.wrapper(release, config, "select", w.SELECT_AT).returncode == 0
    monkeypatch.setattr(
        daily_coordinator,
        "run_group",
        lambda argv, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(argv, 1)),
    )
    now = daily_coordinator.datetime.fromisoformat(w.RUN_AT)
    config_body = daily_coordinator._config(config)
    assert daily_coordinator.infer_stage(config_body, w.D, now)["error_class"] == "RunnerTimeout"
    assert daily_coordinator.render_stage(config_body, w.D)["status"] == "rendered"
    assert (tmp_path / "private" / "2026-09-29" / "PRIVATE_DO_NOT_PUBLISH.txt").is_file()


def test_run_group_ends_the_whole_group_on_timeout_and_on_exceptions(tmp_path: Path) -> None:
    pids = tmp_path / "pids"
    child = (
        "import os, subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pids)!r}, 'w').write(f'{{os.getpid()}} {{p.pid}}')\ntime.sleep(60)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        daily_coordinator.run_group([sys.executable, "-c", child], timeout=2)
    parent, grandchild = (int(x) for x in pids.read_text().split())
    deadline = time.time() + 10
    import os

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True

    while (alive(parent) or alive(grandchild)) and time.time() < deadline:
        time.sleep(0.1)
    assert not alive(parent) and not alive(grandchild)


def test_term_to_the_daily_wrapper_ends_the_runner_it_started(tmp_path: Path, monkeypatch) -> None:
    """Cronicle aborting ``briefing-stage.sh run`` must not leave the runner Python alive."""
    import os
    import signal

    release, config = w.build(
        tmp_path, monkeypatch, kr=[("2026-09-28", EARLY)], us=[("2026-09-28", EARLY)]
    )
    assert w.wrapper(release, config, "select", w.SELECT_AT).returncode == 0
    slow = tmp_path / "slow.flag"
    slow.write_text("x")
    env = {**os.environ, "PYTHONPATH": str(release / "src"), "R3_SLOW_FLAG": str(slow)}
    wrapper = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "modeler.serving.daily_wrapper",
            "run",
            "--config",
            str(config),
            "--report-date",
            w.D.isoformat(),
            "--fixture-now",
            w.RUN_AT,
        ],
        cwd=release,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pid_file = Path(str(slow) + ".pid")
    deadline = time.time() + 30
    while not pid_file.exists():
        assert time.time() < deadline and wrapper.poll() is None, "the runner never started"
        time.sleep(0.05)
    runner = int(pid_file.read_text())
    os.kill(runner, 0)  # alive
    wrapper.send_signal(signal.SIGTERM)
    wrapper.communicate(timeout=30)
    assert wrapper.returncode == 128 + signal.SIGTERM
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            os.kill(runner, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    with pytest.raises(ProcessLookupError):
        os.kill(runner, 0)


# ---- 5. KR prepare failed: the newest older prepared input, then no ranking after five ------


def test_5_kr_prepare_failed_older_prepared_is_stale_and_ranking_stops_after_five_sessions(
    tmp_path: Path, monkeypatch
) -> None:
    release, config = w.build(
        tmp_path,
        monkeypatch,
        kr=[("2026-09-22", EARLY), ("2026-09-24", EARLY)],  # K=09-28 itself is missing
        us=[("2026-09-28", EARLY)],
    )
    done = _day(release, config)
    assert done.returncode == 0, done.stderr
    kr = w.section(w.report(tmp_path), KR)
    # The most recent valid one (score_date < K) is picked: 09-24, two sessions before K.
    assert kr["feature_asof_date"] == "2026-09-24" and kr["quality"]["lag_sessions"] == 2
    assert kr["status"] == "stale" and len(kr["rankings"]) == 3
    selection = json.loads(
        (tmp_path / "prepared" / "selections" / "2026-09-29" / "selection-state.json").read_text()
    )
    assert selection["markets"]["KR"]["lag_sessions"] == 2
    assert selection["markets"]["KR"]["freshness_reason"] == "KR feature session does not match K"


def test_5_kr_older_than_five_sessions_shows_no_ranking_but_stays_a_stale_section(
    tmp_path: Path, monkeypatch
) -> None:
    release, config = w.build(
        tmp_path, monkeypatch, kr=[("2026-09-18", EARLY)], us=[("2026-09-28", EARLY)]
    )
    done = _day(release, config)
    assert done.returncode == 0, done.stderr
    kr = w.section(w.report(tmp_path), KR)
    assert kr["quality"]["lag_sessions"] == 6 and kr["status"] == "stale"
    assert kr["rankings"] == []
    assert kr["quality"]["rankings_withheld"] == {
        "reason": "stale_lag_exceeds_limit",
        "lag_sessions": 6,
        "limit_sessions": 5,
        "ranking_count": 3,
    }
    ctx = _context(tmp_path)
    text = _lines(markdown.render_kr(ctx))
    assert "5세션 초과" in text and "| 1 |" not in text  # a reason instead of the table
    assert ctx["kr"]["status"] == "stale"


def test_5_exactly_five_sessions_still_shows_the_ranking(tmp_path: Path, monkeypatch) -> None:
    release, config = w.build(
        tmp_path, monkeypatch, kr=[("2026-09-21", EARLY)], us=[("2026-09-28", EARLY)]
    )
    assert _day(release, config).returncode == 0
    kr = w.section(w.report(tmp_path), KR)
    assert kr["quality"]["lag_sessions"] == 5 and len(kr["rankings"]) == 3


# ---- 6. US A missing: A' is served as stale, lag shown, no ranking after five ---------------


@pytest.mark.parametrize("a_prime,lag", [("2026-09-25", 1), ("2026-09-24", 2), ("2026-09-22", 4)])
def test_6_us_a_prime_is_stale_not_blocked_whatever_the_old_two_session_ceiling_said(
    tmp_path: Path, monkeypatch, a_prime: str, lag: int
) -> None:
    release, config = w.build(
        tmp_path, monkeypatch, kr=[("2026-09-28", EARLY)], us=[(a_prime, EARLY)], lag_limit=2
    )
    done = _day(release, config)
    assert done.returncode == 0, done.stderr
    for model in (US_LGBM, US_RIDGE):
        us = w.section(w.report(tmp_path), model)
        assert us["status"] == "stale" and us["feature_asof_date"] == a_prime
        assert us["quality"]["lag_sessions"] == lag and len(us["rankings"]) == 3
    assert w.section(w.report(tmp_path), KR)["status"] == "ok"
    ctx = _context(tmp_path)
    assert {m["lag"] for m in ctx["us"]["models"]} == {lag}


def test_6_us_older_than_five_sessions_shows_no_ranking(tmp_path: Path, monkeypatch) -> None:
    release, config = w.build(
        tmp_path, monkeypatch, kr=[("2026-09-28", EARLY)], us=[("2026-09-18", EARLY)]
    )
    assert _day(release, config).returncode == 0
    rep = w.report(tmp_path)
    for model in (US_LGBM, US_RIDGE):
        us = w.section(rep, model)
        assert (
            us["quality"]["lag_sessions"] == 6 and us["rankings"] == [] and us["status"] == "stale"
        )
        assert us["quality"]["rankings_withheld"]["ranking_count"] == 3
    assert "5세션 초과" in _lines(markdown.render_us(_context(tmp_path)))


# ---- 7. select did not run: the run selects for itself ----------------------------------------


def test_7_run_selects_for_itself_after_ten_and_scores(tmp_path: Path, monkeypatch) -> None:
    release, config = w.build(
        tmp_path, monkeypatch, kr=[("2026-09-28", EARLY)], us=[("2026-09-28", EARLY)]
    )
    done = _day(release, config, select_stage=False)  # no 09:30 select at all
    assert done.returncode == 0, done.stderr
    assert _states(done)["inference"] == "inferred"
    selection = json.loads(
        (tmp_path / "prepared" / "selections" / "2026-09-29" / "selection-state.json").read_text()
    )
    assert selection["selection_mode"] == "run_fallback"
    assert selection["selected_at"] == w.RUN_AT  # 10:00 sharp: the run's own clock
    for market in ("kr", "us"):
        one = json.loads(
            (
                tmp_path / "prepared" / "selections" / "2026-09-29" / f"{market}-selection.json"
            ).read_text()
        )
        assert one["selection_mode"] == "run_fallback" and one["producer_completed_at"] == EARLY
        assert "completed_at" not in one  # selection time and input completion are separate fields
    rep = w.report(tmp_path)
    assert rep["status"] == "ok"
    assert w.section(rep, KR)["quality"]["selection_mode"] == "run_fallback"


def test_7_a_selection_made_at_10_05_scores_but_an_input_finished_at_09_40_is_still_refused(
    tmp_path: Path, monkeypatch
) -> None:
    """The mode widens when a selection may be made, never which inputs qualify."""
    release, config = w.build(
        tmp_path,
        monkeypatch,
        kr=[("2026-09-28", LATE), ("2026-09-25", EARLY)],  # K finished 09:40, K' at 09:20
        us=[("2026-09-28", EARLY)],
    )
    done = w.wrapper(release, config, "run", "2026-09-29T10:05:00+09:00")
    assert done.returncode == 0, done.stderr
    kr = w.section(w.report(tmp_path), KR)
    assert kr["feature_asof_date"] == "2026-09-25" and kr["status"] == "stale"  # not the late K
    assert kr["quality"]["selected_at"] == "2026-09-29T10:05:00+09:00"
    assert kr["quality"]["producer_completed_at"] == EARLY
    assert w.section(w.report(tmp_path), US_LGBM)["status"] == "ok"


def test_7_run_before_ten_cannot_make_a_selection(tmp_path: Path, monkeypatch) -> None:
    release, config = w.build(tmp_path, monkeypatch, kr=[("2026-09-28", EARLY)])
    done = w.wrapper(release, config, "run", "2026-09-29T09:45:00+09:00")
    assert done.returncode == 1  # inference cannot start before 10:00 either
    assert not (tmp_path / "prepared" / "selections" / "2026-09-29").exists()


# ---- 8. GitHub unreachable (publisher flow of R2) and 9. everything failed -------------------


def test_9_everything_failed_still_makes_a_failed_unit_that_the_publisher_accepts(
    tmp_path: Path, monkeypatch
) -> None:
    release, config = w.build(tmp_path, monkeypatch)  # nothing prepared at all
    world = mday._enable_publisher(tmp_path, config)
    done = _day(release, config)
    assert done.returncode == 1  # Cronicle sees the failed inference ...
    states = _states(done)
    assert (
        states["publication"] == "published"
    ), states  # ... and the unit is made and pushed anyway
    rep = w.report(tmp_path)
    assert rep["status"] == "failed" and rep["markets"] == []
    assert {item["error_class"] for item in rep["failures"]} == {"MissingInference"}
    unit = world.checkout / mday._unit(w.D)
    assert (unit / "README.md").is_file() and "실패" in (unit / "README.md").read_text()
    assert (
        markdown.validate(world.checkout, allow_synthetic=True) == []
    )  # the stock_reports validator is satisfied
    assert world.remote_head()  # pushed to the (local bare) remote


def test_8_failure_unit_is_kept_locally_while_the_remote_is_unreachable(
    tmp_path: Path, monkeypatch
) -> None:
    release, config = w.build(tmp_path, monkeypatch)
    world = mday._enable_publisher(tmp_path, config)
    world.go_offline()
    done = _day(release, config)
    assert done.returncode == 1  # publisher_failed is reported to Cronicle ...
    assert _states(done)["publication"] == "publisher_failed"
    publication = json.loads((w.run_dir(tmp_path) / "coordinator-publication.json").read_text())
    assert publication["local_done"] is True and publication["publisher_status"] == "sync_pending"
    assert (
        w.run_dir(tmp_path) / "markdown" / "unit" / "README.md"
    ).is_file()  # ... the unit exists
    assert world.journal()["units"]["2026-09-29"]["state"] == "sync_pending"


def test_main_scenarios_are_exercised_through_the_shared_helpers() -> None:
    assert e2e.D == w.D and sys.version_info >= (3, 11)
