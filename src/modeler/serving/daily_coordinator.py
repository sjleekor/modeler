"""Date-scoped selector, inference, private rendering, and bounded publication watch."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from collections.abc import Iterator
from zoneinfo import ZoneInfo

from modeler.reporting.site import PRIVATE_MANIFEST_NAME, PrivateViewBuilder, _read_tree
from modeler.serving.schema import report_template

from .daily_inputs import MS_ENTRYPOINT, _release_market_sector, select
from .orchestration import combine_reports, selection_summary
from .runtime_contract import verify_runtime

SEOUL = ZoneInfo("Asia/Seoul")
RETRY_MINUTES = (15, 17, 22, 32)
REPORTS_REPOSITORY = "sjleekor/stock_reports"
REPORTS_AUDIENCE = "owner_only"
REPORTS_BRANCH = "main"
PUBLISHER_TIMEOUT_SECONDS = 300
RUNNER_TIMEOUT_SECONDS = 1800
MARKET_SECTOR_TIMEOUT_SECONDS = 900
MARKET_SECTOR_ROOT_KEYS = ("market_sector_kr_root", "market_sector_us_root")
# 증권 이름 입력을 만들 레이크 root. 전용 키가 없으면 같은 레이크를 가리키는 시장·섹터 root를 씁니다.
SECURITY_NAMES_ROOT_KEYS = (
    ("security_names_us_root", "market_sector_us_root"),
    ("security_names_kr_root", "market_sector_kr_root"),
)


def _stop_group(process: "subprocess.Popen[Any]", grace_seconds: float = 5.0) -> None:
    """End the child's whole process group: TERM, then KILL after a short grace."""
    try:
        group = os.getpgid(process.pid)
    except ProcessLookupError:
        group = process.pid
    for sig, wait in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 30.0)):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=wait)
            break
        except subprocess.TimeoutExpired:
            continue
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        pass


def run_group(argv: list[str], *, timeout: float, **kwargs: Any) -> "subprocess.CompletedProcess[Any]":
    """``subprocess.run`` whose child leads its own process group.

    A timeout, a TERM/INT to this process or any other exception ends the child's whole group
    (the 2026-10-06 incident: the wrapper died when Cronicle aborted the job and the Python below
    it kept running, outside the lock and every watch).  ``TimeoutExpired`` is re-raised after the
    group is gone.
    """
    process = subprocess.Popen(argv, start_new_session=True, **kwargs)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        _stop_group(process)
        raise
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError("JSON object required")
    return body


def _atomic(path: Path, body: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".state-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(body, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _absolute(config: dict[str, Any], key: str, *, exists: bool = False) -> Path:
    raw = config.get(key)
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise ValueError(f"{key} requires a concrete absolute path")
    path = Path(raw)
    if (path.is_symlink() and key != "python") or (exists and not path.exists()):
        raise ValueError(f"{key} does not identify a permitted path")
    return path


def _config(path: Path) -> dict[str, Any]:
    config = _read(path)
    if config.get("schema_version") != "daily-briefing-ops.v1":
        raise ValueError("ops config schema mismatch")
    for key in ("prepared_root", "selection_root", "run_root",
                "release_manifest", "python", "runtime_lock", "runtime_manifest", "model_cards_path", "kr_calendar",
                "us_calendar", "us_expected_source"):
        _absolute(config, key, exists=key in {"release_manifest", "python", "runtime_lock",
                                              "runtime_manifest", "model_cards_path", "kr_calendar"})
    release = _read(_absolute(config, "release_manifest"))
    release_root = Path(release["release_root"]).resolve(strict=True)
    if release.get("frozen") is not True or any(
            release_root not in _absolute(config, key).resolve().parents
            for key in ("runtime_lock", "runtime_manifest", "model_cards_path")):
        raise ValueError("runtime lock and manifest must belong to the frozen release")
    if _hash(_absolute(config, "runtime_lock")) != config.get("runtime_lock_sha256"):
        raise ValueError("runtime lock SHA-256 changed")
    if _hash(_absolute(config, "runtime_manifest")) != config.get("runtime_manifest_sha256"):
        raise ValueError("runtime manifest SHA-256 changed")
    if _hash(_absolute(config, "model_cards_path")) != config.get("model_cards_sha256"):
        raise ValueError("model card SHA-256 changed")
    if _hash(_absolute(config, "python")) != config.get("python_sha256"):
        raise ValueError("serving Python executable SHA-256 changed")
    verify_runtime(_absolute(config, "python"), _absolute(config, "runtime_manifest"))
    opening_keys = ("opening_snapshot_root", "opening_output_root", "opening_max_age_seconds")
    if any(config.get(key) is not None for key in opening_keys):
        if any(config.get(key) is None for key in opening_keys):
            raise ValueError("opening snapshot root, output root, and max age must be set together")
        _absolute(config, "opening_snapshot_root", exists=True)
        _absolute(config, "opening_output_root")
        if (not isinstance(config["opening_max_age_seconds"], int) or
                not 1 <= config["opening_max_age_seconds"] <= 3600):
            raise ValueError("opening max age must be 1..3600 seconds")
    if _market_sector_roots(config) is not None and _release_market_sector(
            _absolute(config, "release_manifest"), jobs={}) is None:
        raise ValueError("market sector inputs are configured but the frozen release has no bundle")
    _security_names_roots(config)  # 절대 경로인지만 봅니다. 없는 경로는 publisher가 열 생략으로 처리합니다
    if config.get("private_projection_root") is not None:
        private = _absolute(config, "private_projection_root").resolve()
        others = [_absolute(config, "run_root").resolve()]
        if config.get("reports_checkout") is not None:
            others.append(_absolute(config, "reports_checkout").resolve())
        for other in others:
            if private == other or private in other.parents or other in private.parents:
                raise ValueError("private_projection_root must not overlap the run root or the reports checkout")
    if not isinstance(config.get("publisher_enabled"), bool):
        raise ValueError("publisher_enabled must be explicit")
    if not isinstance(config.get("external_verification_enabled"), bool):
        raise ValueError("external_verification_enabled must be explicit")
    if config["publisher_enabled"]:
        _reports_publisher_config(config)
    return config


def _reports_publisher_config(config: dict[str, Any]) -> None:
    """Check the stock_reports publisher settings. Only needed once the publisher is enabled."""
    for key in ("reports_publisher", "reports_checkout"):
        _absolute(config, key, exists=True)
    if _hash(_absolute(config, "reports_publisher")) != config.get("reports_publisher_sha256"):
        raise ValueError("publisher source SHA-256 changed")
    if config.get("reports_repository") != REPORTS_REPOSITORY:
        raise ValueError("reports repository must be confirmed as sjleekor/stock_reports")
    if config.get("reports_audience") != REPORTS_AUDIENCE:
        raise ValueError("reports audience must be owner_only; review the repository settings first")
    if config.get("reports_branch") != REPORTS_BRANCH:
        raise ValueError("reports branch must be main")
    if not isinstance(config.get("reports_remote_url"), str) or not config["reports_remote_url"]:
        raise ValueError("reports remote URL is required")
    top_n = config.get("reports_top_n", 100)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 500:
        raise ValueError("reports_top_n must be 1..500")


def _day_dir(config: dict[str, Any], day: date) -> Path:
    return _absolute(config, "run_root") / day.isoformat()


def _selected(config: dict[str, Any], day: date) -> dict[str, Any]:
    path = _absolute(config, "selection_root") / day.isoformat() / "selection-state.json"
    state = _read(path)
    if state.get("report_date") != day.isoformat():
        raise ValueError("selection report date mismatch")
    if state.get("release_manifest_sha256") != _hash(_absolute(config, "release_manifest")):
        raise ValueError("frozen release changed since 09:30 selection")
    jobs = Path(state["jobs_config"])
    if _hash(jobs) != state.get("jobs_config_sha256"):
        raise ValueError("selected jobs config changed since 09:30")
    return state


def _market_sector_roots(config: dict[str, Any]) -> dict[str, Path] | None:
    """The KR and US data roots the market-sector section reads; ``None`` when it is not configured.

    Both keys are optional and must be set together.  They are the ``DataRoot`` bases
    (``<stock_data>/kr``, ``<stock_data>/us``), read only.
    """
    kr, us = (config.get(key) for key in MARKET_SECTOR_ROOT_KEYS)
    if kr is None and us is None:
        return None
    if kr is None or us is None:
        raise ValueError("market_sector_kr_root and market_sector_us_root must be set together")
    return {"kr": _absolute(config, MARKET_SECTOR_ROOT_KEYS[0], exists=True),
            "us": _absolute(config, MARKET_SECTOR_ROOT_KEYS[1], exists=True)}


def _security_names_roots(config: dict[str, Any]) -> dict[str, str]:
    """publisher가 순위 종목의 이름을 읽을 레이크 root. 설정이 없으면 빈 dict입니다.

    ``security_names_us_root``·``security_names_kr_root``(``<stock_data>/us``·``<stock_data>/kr``,
    읽기만)가 있으면 그것을, 없으면 같은 레이크를 가리키는 ``market_sector_*_root``를 씁니다.
    경로가 실제로 있는지는 여기서 보지 않습니다. 이름은 표시용이라 읽지 못해도 단위는 나가고,
    publisher가 데이터 상태에 사유를 적습니다.
    """
    roots: dict[str, str] = {}
    for key, fallback in SECURITY_NAMES_ROOT_KEYS:
        source = key if config.get(key) is not None else fallback
        if config.get(source) is not None:
            roots[key] = str(_absolute(config, source))
    return roots


def select_stage(config: dict[str, Any], day: date, now: datetime, *,
                 mode: str = "scheduled") -> dict[str, Any]:
    return select(report_date=day, selected_at=now, market_sector=_market_sector_roots(config),
        prepared_root=_absolute(config, "prepared_root"),
        output_root=_absolute(config, "selection_root"),
        release_manifest=_absolute(config, "release_manifest"),
        kr_calendar_path=_absolute(config, "kr_calendar"),
        us_calendar_path=_absolute(config, "us_calendar"),
        us_expected_path=_absolute(config, "us_expected_source"), mode=mode)


def ensure_selection(config: dict[str, Any], day: date, now: datetime) -> dict[str, Any]:
    """D의 selection을 읽고, 09:30 select가 안 돈 날은 run이 같은 규칙으로 직접 고릅니다.

    selection은 불변이라 이미 있으면 그대로 돌려줍니다. 직접 고른 것은 `selection_mode`가
    `run_fallback`이고 `selected_at`이 D 10:00 뒤일 수 있습니다. 입력 선택 규칙은 같아서
    D 09:30 뒤에 끝난 입력은 이때도 들어가지 않습니다 (2026-10-05 변경 5).
    """
    path = _absolute(config, "selection_root") / day.isoformat() / "selection-state.json"
    if not path.is_file():
        if now.astimezone(SEOUL) < datetime.combine(day, time(10), SEOUL):
            raise ValueError("a missing selection can only be made by the run stage after D 10:00 KST")
        select_stage(config, day, now, mode="run_fallback")
    return _selected(config, day)


def _release_jobs_brief(config: dict[str, Any]) -> tuple[list[dict[str, str]], bool]:
    release = _read(_absolute(config, "release_manifest"))
    jobs = [{"market": job["market"], "model_id": job["model_id"]} for job in release.get("jobs", [])]
    return jobs, release.get("synthetic_fixture") is True


def _failed_inference(config: dict[str, Any], day: date, run_root: Path, invocation_id: str, *,
                      stage: str, error_class: str, exit_code: int | None, timed_out: bool,
                      fixture: bool,
                      select_summary: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Make this invocation's own failure report when the runner left none (변경 6).

    Every model section is ``failed`` with the cause class; the unit still renders and publishes.
    ``select_summary`` (what the 09:30 select recorded per market, when a selection exists) goes
    into the envelope so the failed sections can say why select found no input.
    A report of an earlier invocation on the same D is never reused: it is set aside under
    ``report-D.superseded-<sha>.json`` and replaced by this one, which carries this invocation id.
    """
    decision = datetime.combine(day, time(10), SEOUL)
    jobs, _ = _release_jobs_brief(config)
    envelope = combine_reports(report_date=day, decision_at=decision, reports=[], opening=None,
        failures=[{**job, "error_class": error_class} for job in jobs],
        selection_summary=select_summary)
    envelope.update(historical_replay=False, synthetic_fixture=fixture, invocation_id=invocation_id,
        failure={"stage": stage, "error_class": error_class, "runner_exit_code": exit_code,
                 "timed_out": timed_out, "synthesized_by": "coordinator"})
    report_path = run_root / f"report-{day.isoformat()}.json"
    if report_path.is_file():
        shutil.copy2(report_path, run_root / f"report-{day.isoformat()}.superseded-{_hash(report_path)[:12]}.json")
    _atomic(report_path, envelope)
    # The publisher accepts a unit only when run-state.json names this invocation.
    _atomic(run_root / "run-state.json", {
        "report_date": day.isoformat(), "invocation_id": invocation_id, "jobs": [],
        "synthesized_by": "coordinator", "failure_stage": stage, "error_class": error_class})
    return {"report_path": str(report_path), "report_sha256": _hash(report_path)}


def infer_stage(config: dict[str, Any], day: date, now: datetime) -> dict[str, Any]:
    if now.astimezone(SEOUL) < datetime.combine(day, time(10), SEOUL):
        raise ValueError("inference cannot start before D 10:00 KST")
    run_root = _day_dir(config, day)
    run_root.mkdir(parents=True, exist_ok=True)
    invocation_id = secrets.token_hex(16)
    _, fixture = _release_jobs_brief(config)
    select_summary: dict[str, dict[str, Any]] | None = None  # set once a selection exists

    def failed(stage: str, error_class: str, *, exit_code: int | None = None,
               timed_out: bool = False) -> dict[str, Any]:
        written = _failed_inference(config, day, run_root, invocation_id, stage=stage,
            error_class=error_class, exit_code=exit_code, timed_out=timed_out, fixture=fixture,
            select_summary=select_summary)
        state = {"report_date": day.isoformat(), "status": "inference_failed",
                 "invocation_id": invocation_id, "runner_exit_code": exit_code,
                 "report_path": written["report_path"], "report_sha256": written["report_sha256"],
                 "synthetic_fixture": fixture, "successful_jobs": 0,
                 "this_invocation_completed": False, "report_ready": True,
                 "failure_report": True, "failure_stage": stage, "timed_out": timed_out,
                 "error_class": error_class}
        _atomic(run_root / "coordinator-inference.json", state)
        return state

    try:
        selected = ensure_selection(config, day, now)
    except Exception as exc:  # the unit is still made: every section fails with the cause class
        return failed("select", type(exc).__name__)
    if selected.get("status") == "holiday":
        return {"status": "holiday", "report_date": day.isoformat()}
    select_summary = selection_summary(selected)
    state_path = _absolute(config, "selection_root") / day.isoformat() / "selection-state.json"
    release = _read(_absolute(config, "release_manifest"))
    release_root = Path(release["release_root"]).resolve(strict=True)
    argv = [str(_absolute(config, "python")), "-m", "modeler.serving.runner", "infer",
            "--report-date", day.isoformat(), "--prepared-root", str(_absolute(config, "prepared_root")),
            "--run-root", str(run_root), "--jobs-config", selected["jobs_config"],
            "--kr-calendar-json", str(_absolute(config, "kr_calendar")),
            "--kr-calendar-sha256", selected["kr_calendar_sha256"],
            "--invocation-id", invocation_id]
    if select_summary:
        argv += ["--selection-state", str(state_path),
                 "--selection-state-sha256", _hash(state_path)]
    opening = config.get("opening_artifact")
    if opening is not None:
        opening_path = _absolute(config, "opening_artifact", exists=True)
        argv.extend(("--opening-json", str(opening_path)))
    if selected.get("synthetic_fixture") is True:
        argv.append("--fixture-mode")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release_root / "src")
    timed_out, spawn_error = False, None
    try:
        completed = run_group(argv, cwd=release_root, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=RUNNER_TIMEOUT_SECONDS)
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        exit_code, timed_out = 124, True
    except OSError as exc:
        exit_code, spawn_error = 127, type(exc).__name__
    report_path = run_root / f"report-{day.isoformat()}.json"
    report = _read(report_path) if report_path.is_file() else None
    # The report must be this invocation's own: an earlier run's file is never reused.  The runner
    # writes report-D.json first and run-state.json (with this invocation id) last, so a report that
    # an earlier invocation left behind is only accepted together with a run-state of this one.  The
    # id is not written into the runner's report: its bytes (and sha256, which the rendered unit
    # carries) must stay the same when a rerun reuses the saved results.
    report_valid = bool(report and report.get("report_date") == day.isoformat() and
                        bool(report.get("synthetic_fixture")) == selected["synthetic_fixture"])
    run_state_path = run_root / "run-state.json"
    latest_run = _read(run_state_path) if run_state_path.is_file() else {}
    this_invocation_completed = latest_run.get("invocation_id") == invocation_id
    if not (this_invocation_completed and report_valid):
        if timed_out:
            error_class = "RunnerTimeout"
        elif spawn_error:
            error_class = "RunnerSpawnError"
        elif exit_code != 0:
            error_class = "RunnerExitNonzero"
        else:
            error_class = "RunnerReportMissing"
        return failed("infer", error_class, exit_code=exit_code, timed_out=timed_out)
    job_records = latest_run.get("jobs", [])
    successful_jobs = sum(row.get("status") in {"ok", "reused"} for row in job_records)
    if exit_code == 0 and successful_jobs and not report.get("failures"):
        inference_status = "inferred"
    elif successful_jobs and report.get("markets"):
        inference_status = "inferred_partial"
    else:
        inference_status = "inference_failed"
    state = {"report_date": day.isoformat(), "status": inference_status,
             "invocation_id": invocation_id,
             "runner_exit_code": exit_code, "report_path": str(report_path),
             "report_sha256": _hash(report_path),
             "synthetic_fixture": selected["synthetic_fixture"], "successful_jobs": successful_jobs,
             "this_invocation_completed": True, "report_ready": True, "failure_report": False}
    _atomic(run_root / "coordinator-inference.json", state)
    return state


def _retire_ms_document(path: Path) -> None:
    """Set an earlier market-sector document of the same D aside; it is never reused."""
    if path.is_file():
        digest = _hash(path)[:12]
        os.replace(path, path.with_name(f"{path.stem}.superseded-{digest}{path.suffix}"))


def _install_ms_document(path: Path, body: bytes) -> str:
    fd, temp = tempfile.mkstemp(prefix=".market-sector-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)
    return hashlib.sha256(body).hexdigest()


def market_sector_stage(config: dict[str, Any], day: date, now: datetime) -> dict[str, Any]:
    """Score the market-sector section into ``runs/D/market-sector-D.json`` (own process group).

    It is not a fourth ranking job: the section is not a ranking, and a failure here must never
    touch the three model jobs.  It reads only the files ``ms-selection.json`` pinned at select
    time.  Whatever goes wrong, the section ends up ``failed`` with its cause in the document (the
    renderer writes it into the unit); the units still render and publish.  Not configured means
    no document, and the renderer says the input is missing.
    """
    run_root = _day_dir(config, day)
    run_root.mkdir(parents=True, exist_ok=True)
    state_path = run_root / "coordinator-market-sector.json"
    document = run_root / f"market-sector-{day.isoformat()}.json"
    if _market_sector_roots(config) is None:
        state = {"report_date": day.isoformat(), "status": "disabled"}
        _atomic(state_path, state)
        return state
    _retire_ms_document(document)

    def failed(stage: str, error_class: str, reason: str, *, exit_code: int | None = None,
               timed_out: bool = False, markets: Any = None) -> dict[str, Any]:
        from modeler.scores.market_sector.daily_doc import failure_document

        body = failure_document(day, stage=stage, error_class=error_class, reason=reason,
                                exit_code=exit_code, timed_out=timed_out,
                                markets=markets if isinstance(markets, dict) else None)
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False, indent=2) + "\n"
        sha = _install_ms_document(document, raw.encode())
        state = {"report_date": day.isoformat(), "status": "market_sector_failed", "stage": stage,
                 "error_class": error_class, "reason": reason, "timed_out": timed_out,
                 "document_sha256": sha}
        _atomic(state_path, state)
        return state

    try:
        selected = ensure_selection(config, day, now)
    except Exception as exc:
        return failed("select", type(exc).__name__, "selection_unavailable")
    if selected.get("status") == "holiday":
        state = {"report_date": day.isoformat(), "status": "holiday"}
        _atomic(state_path, state)
        return state
    block = selected.get("market_sector")
    if not isinstance(block, dict):
        return failed("select", "MarketSectorNotSelected", "not_in_selection")
    if block.get("status") not in {"selected", "partial"}:
        return failed("select", "MarketSectorInputsUnavailable",
                      str(block.get("reason") or "inputs_unavailable"),
                      markets=block.get("markets"))
    release = _read(_absolute(config, "release_manifest"))
    release_root = Path(release["release_root"]).resolve(strict=True)
    try:
        pinned = _release_market_sector(_absolute(config, "release_manifest"), jobs={})
    except Exception as exc:
        return failed("release", type(exc).__name__, "release_invalid")
    if pinned is None:
        return failed("release", "MarketSectorNotInRelease", "release_without_market_sector")
    scratch = run_root / f".market-sector-{day.isoformat()}.new"
    scratch.unlink(missing_ok=True)
    argv = [str(_absolute(config, "python")), "-m", MS_ENTRYPOINT, "score",
            "--report-date", day.isoformat(), "--selection", block["selection"],
            "--selection-sha256", block["selection_sha256"],
            "--bundle", str(Path(pinned["bundle_path"]).parent),
            "--bundle-sha256", pinned["bundle_sha256"], "--output", str(scratch)]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release_root / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        done = run_group(argv, cwd=release_root, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True,
                         timeout=MARKET_SECTOR_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        scratch.unlink(missing_ok=True)
        return failed("score", "MarketSectorTimeout", "timeout", exit_code=124, timed_out=True)
    except OSError as exc:
        return failed("score", type(exc).__name__, "spawn_error", exit_code=127)
    summary = _last_json_line(done.stdout or "")
    if done.returncode != 0 or not scratch.is_file():
        scratch.unlink(missing_ok=True)
        error_class = "MarketSectorExitNonzero" if done.returncode else "MarketSectorOutputMissing"
        return failed("score", error_class,
                      str(summary.get("reason") or summary.get("status") or "score_failed"),
                      exit_code=done.returncode, markets=summary.get("failures"))
    sha = _install_ms_document(document, scratch.read_bytes())
    scratch.unlink(missing_ok=True)
    partial = summary.get("status") == "partial"
    scored = "market_sector_partial" if partial else "market_sector_ready"
    state = {"report_date": day.isoformat(), "status": scored, "markets": summary.get("markets"),
             "document_sha256": sha, "selection_sha256": block["selection_sha256"],
             "bundle_sha256": pinned["bundle_sha256"]}
    _atomic(state_path, state)
    return state


def _previous_private(config: dict[str, Any], day: date,
                      synthetic: bool) -> tuple[Path | None, dict[str, bytes] | None]:
    """Newest strictly earlier private view with the same fixture mode; never a public projection."""
    root = _absolute(config, "private_projection_root")
    chosen, newest = None, None
    for entry in (sorted(root.iterdir()) if root.is_dir() else []):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry.name):
            continue
        try:
            entry_day = date.fromisoformat(entry.name)
        except ValueError:
            continue
        if entry_day >= day:
            continue
        if entry.is_symlink():
            raise ValueError("previous private view candidate cannot be a symlink")
        manifest_path = entry / PRIVATE_MANIFEST_NAME
        if not entry.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = _read(manifest_path)
        except (OSError, ValueError):
            continue
        if manifest.get("private") is not True or manifest.get("latest_report_date") != entry.name:
            continue
        if newest is None or entry_day > newest:
            chosen, newest = entry, entry_day
    if chosen is None:
        return None, None
    files = _read_tree(chosen)
    if bool(json.loads(files[PRIVATE_MANIFEST_NAME]).get("synthetic_fixture")) != synthetic:
        raise ValueError("previous private view fixture mode differs from this run")
    return chosen, files


def render_stage(config: dict[str, Any], day: date) -> dict[str, Any]:
    """Validate this invocation's report and, when configured, write the owner-only private view.

    The markdown unit for ``stock_reports`` is rendered by the publisher's local step (publish stage),
    from the same saved report, so a failed push never blocks the report itself.
    """
    run_root = _day_dir(config, day)
    infer_state = _read(run_root / "coordinator-inference.json")
    if infer_state.get("report_ready", infer_state.get("this_invocation_completed")) is not True:
        raise ValueError("render requires a report from the latest completed runner invocation")
    if not isinstance(infer_state.get("invocation_id"), str):
        raise ValueError("runner invocation changed after inference")
    synthesized = infer_state.get("failure_report") is True
    if not synthesized:
        # The selection (and its release pin) must still be the frozen one the runner used.
        _selected(config, day)
    run_state = _read(run_root / "run-state.json")
    if run_state.get("invocation_id") != infer_state["invocation_id"]:
        raise ValueError("runner invocation changed after inference")
    fixture_mode = bool(infer_state.get("synthetic_fixture"))
    report_path = Path(infer_state["report_path"])
    if _hash(report_path) != infer_state["report_sha256"]:
        raise ValueError("saved report changed after inference")
    report = _read(report_path)
    if report.get("report_date") != day.isoformat() or bool(report.get("synthetic_fixture")) != fixture_mode:
        raise ValueError("report date or fixture mode mismatch")
    if report.get("invocation_id", infer_state["invocation_id"]) != infer_state["invocation_id"]:
        raise ValueError("report belongs to another invocation")
    market_reports = list(report["markets"])
    present = {(item["market"], item["model_id"]) for item in market_reports}
    release = _read(_absolute(config, "release_manifest"))
    for job in release["jobs"]:
        identity = (job["market"], job["model_id"])
        if identity not in present:
            empty = report_template(market=job["market"], report_date=day.isoformat(),
                decision_at=datetime.fromisoformat(report["decision_at"]),
                feature_asof_date=None, model_id=job["model_id"],
                model_version=job["model_version"])
            empty["synthetic_fixture"] = fixture_mode
            market_reports.append(empty)
    cards = _read(_absolute(config, "model_cards_path"))
    if set(cards) != {job["model_id"] for job in release["jobs"]}:
        raise ValueError("model cards must match the frozen three-model release")
    private_files = private_target = private_previous = None
    if config.get("private_projection_root") is not None:
        private_previous, private_previous_files = _previous_private(config, day, fixture_mode)
        private_builder = PrivateViewBuilder(synthetic_fixture=fixture_mode)
        private_files = private_builder.render(report_date=day, reports=market_reports,
            opening=report["opening"], previous_files=private_previous_files)
        private_target = _absolute(config, "private_projection_root") / day.isoformat()
        private_builder.write_atomic(private_target, private_files)
    state = {"report_date": day.isoformat(), "status": "rendered", "report_sha256": _hash(report_path),
             "invocation_id": infer_state["invocation_id"],
             "synthetic_fixture": fixture_mode,
             "private_projection_dir": str(private_target) if private_target else None,
             "private_manifest_sha256": (hashlib.sha256(private_files[PRIVATE_MANIFEST_NAME]).hexdigest()
                                         if private_files is not None else None),
             "private_previous_projection_dir": str(private_previous) if private_previous else None}
    _atomic(run_root / "coordinator-render.json", state)
    return state


def _reports_publisher_input(config: dict[str, Any], day: date, render: dict[str, Any],
                             run_root: Path) -> dict[str, Any]:
    """The date-scoped settings handed to the stock_reports publisher."""
    return {
        "audience": config["reports_audience"],
        "checkout_dir": str(_absolute(config, "reports_checkout")),
        "remote_name": "origin",
        "expected_remote_url": config["reports_remote_url"],
        "branch": config["reports_branch"],
        "release": Path(config["release_manifest"]).parent.name,
        "run_dir": str(run_root),
        "report_date": day.isoformat(),
        "report_sha256": render["report_sha256"],
        "invocation_id": render["invocation_id"],
        "model_cards_path": str(_absolute(config, "model_cards_path")),
        "top_n": config.get("reports_top_n", 100),
        **_security_names_roots(config),
    }


def _last_json_line(text: str) -> dict[str, Any]:
    for line in reversed(text.strip().splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def publish_stage(config: dict[str, Any], day: date, *, sync_only: bool = False) -> dict[str, Any]:
    """Run the stock_reports publisher: local render + validation, then fetch/commit/push.

    ``sync_only`` asks for the sync step alone (monitor retries).  It only applies once the local step
    finished before; otherwise the whole publisher runs again.
    """
    run_root = _day_dir(config, day)
    render = _read(run_root / "coordinator-render.json")
    report = run_root / f"report-{day.isoformat()}.json"
    if _hash(report) != render.get("report_sha256"):
        raise ValueError("publication retry cannot use a changed report")
    publication_path = run_root / "coordinator-publication.json"
    previous = _read(publication_path) if publication_path.is_file() else {}
    if not config["publisher_enabled"]:
        state = {"report_date": day.isoformat(), "status": "publication_withheld",
                 "reason": "publisher_disabled", "report_sha256": _hash(report)}
        _atomic(publication_path, state)
        return state
    daily_config = _reports_publisher_input(config, day, render, run_root)
    daily_path = run_root / "reports-publisher-config.json"
    if daily_path.is_symlink():
        raise ValueError("date-scoped publisher config cannot be a symlink")
    if not (daily_path.is_file() and _read(daily_path) == daily_config):
        _atomic(daily_path, daily_config)
    release_root = Path(_read(_absolute(config, "release_manifest"))["release_root"]).resolve(strict=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release_root / "src")  # the publisher renders with the frozen release code
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    mode = "sync" if sync_only and previous.get("local_done") is True else "run"
    exit_code, outcome = _run_reports_publisher(config, daily_path, mode, render, env)
    if mode == "sync" and outcome.get("status") == "nothing_to_do":
        mode = "run"  # nothing was waiting in the journal, so render and check the remote again
        exit_code, outcome = _run_reports_publisher(config, daily_path, mode, render, env)
    state = {"report_date": day.isoformat(),
             "status": "published" if exit_code == 0 else "publisher_failed",
             "publisher_exit_code": exit_code, "publisher_mode": mode,
             "publisher_status": outcome.get("status"),
             "local_done": bool(outcome.get("local_done")) or previous.get("local_done") is True,
             "report_sha256": _hash(report),
             "publisher_config_path": str(daily_path), "publisher_config_sha256": _hash(daily_path)}
    if exit_code != 0 and outcome.get("detail"):
        state["publisher_detail"] = outcome["detail"] if isinstance(outcome["detail"], (str, list)) else None
    if state["status"] == "published":
        commit = outcome.get("commit")
        checkout = _absolute(config, "reports_checkout")
        head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "--verify",
                               f"refs/remotes/origin/{config['reports_branch']}^{{commit}}"],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                              check=False, timeout=10)
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            state["status"] = "publisher_failed"
            state["reason"] = "reports_commit_unavailable"
        elif head.returncode or head.stdout.strip() != commit:
            state["status"] = "publisher_failed"
            state["reason"] = "reports_commit_mismatch"
        else:
            state["reports_commit"] = commit
    _atomic(publication_path, state)
    return state


def _run_reports_publisher(config: dict[str, Any], daily_path: Path, mode: str,
                           render: dict[str, Any], env: dict[str, str]) -> tuple[int, dict[str, Any]]:
    argv = [str(_absolute(config, "python")), str(_absolute(config, "reports_publisher")),
            mode, "--config", str(daily_path)]
    if render["synthetic_fixture"]:
        argv.append("--allow-synthetic")
    try:
        done = run_group(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True, timeout=PUBLISHER_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return 124, {}
    return done.returncode, _last_json_line(done.stdout)


def _remote_contains(config: dict[str, Any], commit: str) -> str:
    """Does remote main contain the commit the publisher pushed?  Uses ``git ls-remote``.

    If remote main moved on (for example, the owner pushed a later commit), the commit counts as
    contained when it is an ancestor of the new head.  The ancestry check fetches into a private
    ref (``refs/monitor/main``), never into the ref the publisher uses.
    """
    checkout = _absolute(config, "reports_checkout")
    branch = config["reports_branch"]
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    def git(*args: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(checkout), *args], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True, check=False, env=env,
                              timeout=timeout)

    try:
        listed = git("ls-remote", config["reports_remote_url"], f"refs/heads/{branch}")
        if listed.returncode or not listed.stdout.split():
            return "unavailable"
        head = listed.stdout.split()[0]
        if head == commit:
            return "contained"
        if git("cat-file", "-e", f"{head}^{{commit}}").returncode:
            fetched = git("fetch", "--no-tags", config["reports_remote_url"],
                          f"+refs/heads/{branch}:refs/monitor/{branch}", timeout=120)
            if fetched.returncode or git("cat-file", "-e", f"{head}^{{commit}}").returncode:
                return "unavailable"
        ancestry = git("merge-base", "--is-ancestor", commit, head)
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    if ancestry.returncode == 0:
        return "contained"
    return "not_contained" if ancestry.returncode == 1 else "unavailable"


def monitor_stage(config: dict[str, Any], day: date, *, attempt: int) -> dict[str, Any]:
    if attempt not in range(len(RETRY_MINUTES)):
        raise ValueError("monitor attempt exceeds the bounded retry plan")
    root = _day_dir(config, day)
    selection_path = _absolute(config, "selection_root") / day.isoformat() / "selection-state.json"
    axes: dict[str, str] = {"input": "unknown", "inference": "unknown", "rights": "unknown",
                            "publisher": "unknown", "remote": "unknown"}
    if not selection_path.is_file():
        axes["input"] = "missing"
    else:
        selected = _read(selection_path)
        axes["input"] = "ready" if selected.get("status") == "selected" else selected.get("status", "missing")
        if selected.get("status") == "holiday":
            axes.update({"inference": "skipped", "rights": "not_applicable",
                         "publisher": "skipped", "remote": "skipped"})
            state = {"report_date": day.isoformat(), "attempt": attempt,
                     "scheduled_minute": RETRY_MINUTES[attempt], "status": "holiday_skipped",
                     "axes": axes, "retry_allowed": False}
            _atomic(root / f"monitor-{attempt}.json", state)
            return state
    inference_path = root / "coordinator-inference.json"
    if inference_path.is_file():
        axes["inference"] = _read(inference_path).get("status", "failed")
    report_path = root / f"report-{day.isoformat()}.json"
    if report_path.is_file():
        # Informational only: the owner-only repository does not use the public-source gate.
        report = _read(report_path)
        gates = [item.get("publication", {}).get("status") for item in report.get("markets", [])]
        gates.append(report.get("opening", {}).get("publication", {}).get("status"))
        axes["rights"] = "allowed" if all(gate == "allowed" for gate in gates) else "withheld"
    publication_path = root / "coordinator-publication.json"
    if publication_path.is_file():
        axes["publisher"] = _read(publication_path).get("status", "publisher_failed")
    if axes["publisher"] == "publisher_failed" and attempt > 0 and config["publisher_enabled"]:
        # Retry the sync step only; the publisher's journal finishes the same unit.
        axes["publisher"] = publish_stage(config, day, sync_only=True)["status"]
    if axes["publisher"] == "published":
        published = _read(publication_path)
        commit = published.get("reports_commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            axes["remote"] = "pending"
        elif config["external_verification_enabled"]:
            axes["remote"] = _remote_contains(config, commit)
            _atomic(root / "remote-verification.json",
                    {"reports_commit": commit, "status": axes["remote"]})
        else:
            axes["remote"] = "pending"
    if axes["input"] == "missing":
        status = "input_missing"
    elif axes["inference"] in {"unknown", "inference_failed"}:
        status = "inference_failed"
    elif axes["publisher"] in {"publisher_failed", "unknown"}:
        status = "publisher_failed"
    elif axes["publisher"] == "publication_withheld":
        status = "publication_withheld"
    elif axes["remote"] == "contained":
        status = "verified"
    elif axes["remote"] == "not_contained":
        status = "remote_missing_commit"
    else:
        status = "remote_pending"
    state = {"report_date": day.isoformat(), "attempt": attempt,
             "scheduled_minute": RETRY_MINUTES[attempt], "status": status, "axes": axes,
             "retry_allowed": attempt < len(RETRY_MINUTES) - 1 and status != "verified"}
    _atomic(root / f"monitor-{attempt}.json", state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modeler-daily-coordinator")
    parser.add_argument("stage", choices=("select", "infer", "market-sector", "render", "publish",
                                          "monitor"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-date", type=date.fromisoformat, required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--fixture-now", type=datetime.fromisoformat,
                        help="synthetic release tests only; never valid for a real release")
    args = parser.parse_args(argv)
    try:
        config = _config(args.config)
        if args.fixture_now and _read(_absolute(config, "release_manifest")).get("synthetic_fixture") is not True:
            raise ValueError("fixture clock is forbidden for a real release")
        now = args.fixture_now or datetime.now(SEOUL)
        with _lock(_day_dir(config, args.report_date) / ".coordinator.lock"):
            if args.stage == "select":
                result = select_stage(config, args.report_date, now)
            elif args.stage == "infer":
                result = infer_stage(config, args.report_date, now)
            elif args.stage == "market-sector":
                result = market_sector_stage(config, args.report_date, now)
            elif args.stage == "render":
                result = render_stage(config, args.report_date)
            elif args.stage == "publish":
                result = publish_stage(config, args.report_date)
            else:
                result = monitor_stage(config, args.report_date, attempt=args.attempt)
        print(json.dumps({"status": result["status"], "report_date": args.report_date.isoformat()}, sort_keys=True))
        return 0 if result["status"] in {"selected", "partial", "holiday", "holiday_skipped",
                                          "inferred", "inferred_partial", "rendered",
                                          "published", "verified", "publication_withheld",
                                          "disabled", "market_sector_ready",
                                          "market_sector_partial"} else 1
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
