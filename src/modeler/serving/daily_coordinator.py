"""Date-scoped selector, inference, private rendering, and bounded publication watch."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import secrets
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo
from urllib.parse import urlencode

from modeler.reporting.site import PRIVATE_MANIFEST_NAME, PrivateViewBuilder, SiteBuilder, _read_tree
from modeler.serving.schema import report_template

from .daily_inputs import select
from .runtime_contract import verify_runtime

SEOUL = ZoneInfo("Asia/Seoul")
RETRY_MINUTES = (15, 17, 22, 32)


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
    for key in ("prepared_root", "selection_root", "run_root", "projection_root",
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
    if config.get("private_projection_root") is not None:
        private = _absolute(config, "private_projection_root").resolve()
        others = [_absolute(config, "projection_root").resolve()]
        if config.get("site_checkout") is not None:
            others.append(_absolute(config, "site_checkout").resolve())
        for other in others:
            if private == other or private in other.parents or other in private.parents:
                raise ValueError("private_projection_root must not overlap the public projection or site checkout")
    if not isinstance(config.get("publisher_enabled"), bool):
        raise ValueError("publisher_enabled must be explicit")
    if not isinstance(config.get("external_verification_enabled"), bool):
        raise ValueError("external_verification_enabled must be explicit")
    base = config.get("base_path")
    if not isinstance(base, str) or not base.startswith("/") or not base.endswith("/"):
        raise ValueError("base_path requires a concrete Pages project path")
    if config["publisher_enabled"]:
        for key in ("publisher_script", "publisher_config", "site_checkout", "public_manifest_url"):
            if key == "public_manifest_url":
                if not isinstance(config.get(key), str) or not config[key].startswith("https://"):
                    raise ValueError("public manifest URL must be HTTPS")
            else:
                _absolute(config, key, exists=True)
        if _hash(_absolute(config, "publisher_script")) != config.get("publisher_script_sha256"):
            raise ValueError("publisher source SHA-256 changed")
        if (config.get("actions_repository") != "sjleekor/market-briefing" or
                not isinstance(config.get("actions_workflow"), str) or
                not re.fullmatch(r"[A-Za-z0-9._-]+\.ya?ml", config["actions_workflow"])):
            raise ValueError("Actions repository and workflow must be confirmed")
    return config


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


def select_stage(config: dict[str, Any], day: date, now: datetime) -> dict[str, Any]:
    return select(report_date=day, selected_at=now,
        prepared_root=_absolute(config, "prepared_root"),
        output_root=_absolute(config, "selection_root"),
        release_manifest=_absolute(config, "release_manifest"),
        kr_calendar_path=_absolute(config, "kr_calendar"),
        us_calendar_path=_absolute(config, "us_calendar"),
        us_expected_path=_absolute(config, "us_expected_source"))


def infer_stage(config: dict[str, Any], day: date, now: datetime) -> dict[str, Any]:
    if now.astimezone(SEOUL) < datetime.combine(day, time(10), SEOUL):
        raise ValueError("inference cannot start before D 10:00 KST")
    selected = _selected(config, day)
    if selected.get("status") == "holiday":
        return {"status": "holiday", "report_date": day.isoformat()}
    release = _read(_absolute(config, "release_manifest"))
    release_root = Path(release["release_root"]).resolve(strict=True)
    run_root = _day_dir(config, day)
    run_root.mkdir(parents=True, exist_ok=True)
    invocation_id = secrets.token_hex(16)
    argv = [str(_absolute(config, "python")), "-m", "modeler.serving.runner", "infer",
            "--report-date", day.isoformat(), "--prepared-root", str(_absolute(config, "prepared_root")),
            "--run-root", str(run_root), "--jobs-config", selected["jobs_config"],
            "--kr-calendar-json", str(_absolute(config, "kr_calendar")),
            "--kr-calendar-sha256", selected["kr_calendar_sha256"],
            "--invocation-id", invocation_id]
    opening = config.get("opening_artifact")
    if opening is not None:
        opening_path = _absolute(config, "opening_artifact", exists=True)
        argv.extend(("--opening-json", str(opening_path)))
    if selected.get("synthetic_fixture") is True:
        argv.append("--fixture-mode")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(release_root / "src")
    try:
        completed = subprocess.run(argv, cwd=release_root, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False, timeout=1800)
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    report_path = run_root / f"report-{day.isoformat()}.json"
    report = _read(report_path) if report_path.is_file() else None
    report_valid = bool(report and report.get("report_date") == day.isoformat() and
                        bool(report.get("synthetic_fixture")) == selected["synthetic_fixture"])
    run_state_path = run_root / "run-state.json"
    latest_run = _read(run_state_path) if run_state_path.is_file() else {}
    this_invocation_completed = latest_run.get("invocation_id") == invocation_id
    job_records = latest_run.get("jobs", []) if this_invocation_completed else []
    successful_jobs = sum(row.get("status") in {"ok", "reused"} for row in job_records)
    if exit_code == 0 and this_invocation_completed and report_valid and successful_jobs:
        inference_status = "inferred"
    elif this_invocation_completed and report_valid and successful_jobs and report.get("markets"):
        inference_status = "inferred_partial"
    else:
        inference_status = "inference_failed"
    state = {"report_date": day.isoformat(), "status": inference_status,
             "invocation_id": invocation_id,
             "runner_exit_code": exit_code, "report_path": str(report_path) if report_path.is_file() else None,
             "report_sha256": _hash(report_path) if report_path.is_file() else None,
             "synthetic_fixture": selected["synthetic_fixture"], "successful_jobs": successful_jobs,
             "this_invocation_completed": this_invocation_completed}
    _atomic(run_root / "coordinator-inference.json", state)
    return state


def _previous_projection(config: dict[str, Any], day: date,
                         synthetic: bool) -> tuple[Path | None, dict[str, bytes] | None]:
    """Pick the newest strictly earlier, self-consistent projection (or the explicit override)."""
    if config.get("previous_projection_dir"):
        chosen: Path | None = _absolute(config, "previous_projection_dir", exists=True)
    else:
        chosen, newest = None, None
        root = _absolute(config, "projection_root")
        candidates = sorted(root.iterdir()) if root.is_dir() else []
        for entry in candidates:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", entry.name):
                continue
            try:
                entry_day = date.fromisoformat(entry.name)
            except ValueError:
                continue
            if entry_day >= day:
                continue
            if entry.is_symlink():
                raise ValueError("previous projection candidate cannot be a symlink")
            manifest_path = entry / "site-manifest.json"
            if not entry.is_dir() or not manifest_path.is_file():
                continue
            try:
                manifest = _read(manifest_path)
            except (OSError, ValueError):
                continue
            if manifest.get("latest_report_date") != entry.name:
                continue
            if newest is None or entry_day > newest:
                chosen, newest = entry, entry_day
    if chosen is None:
        return None, None
    files = _read_tree(chosen)
    manifest_bytes = files.get("site-manifest.json")
    if manifest_bytes is not None:
        try:
            previous_manifest = json.loads(manifest_bytes)
        except ValueError as exc:
            raise ValueError("previous projection site manifest is unreadable") from exc
        if (not isinstance(previous_manifest, dict) or
                bool(previous_manifest.get("synthetic_fixture")) != synthetic):
            raise ValueError("previous projection fixture mode differs from this run")
    return chosen, files


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
    selected = _selected(config, day)
    run_root = _day_dir(config, day)
    infer_state = _read(run_root / "coordinator-inference.json")
    if infer_state.get("this_invocation_completed") is not True:
        raise ValueError("render requires a report from the latest completed runner invocation")
    run_state = _read(run_root / "run-state.json")
    if (not isinstance(infer_state.get("invocation_id"), str) or
            run_state.get("invocation_id") != infer_state["invocation_id"]):
        raise ValueError("runner invocation changed after inference")
    report_path = Path(infer_state["report_path"])
    if _hash(report_path) != infer_state["report_sha256"]:
        raise ValueError("saved report changed after inference")
    report = _read(report_path)
    if report.get("report_date") != day.isoformat() or bool(report.get("synthetic_fixture")) != selected["synthetic_fixture"]:
        raise ValueError("report date or fixture mode mismatch")
    previous_dir, previous_files = _previous_projection(config, day, selected["synthetic_fixture"])
    builder = SiteBuilder(base_path=config["base_path"],
                          synthetic_fixture=selected["synthetic_fixture"])
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
            empty["synthetic_fixture"] = selected["synthetic_fixture"]
            market_reports.append(empty)
    cards = _read(_absolute(config, "model_cards_path"))
    if set(cards) != {job["model_id"] for job in release["jobs"]}:
        raise ValueError("model cards must match the frozen three-model release")
    files = builder.render(report_date=day, reports=market_reports, model_cards=cards,
        opening=report["opening"], previous_files=previous_files,
        inference_started_at=datetime.fromisoformat(run_state["started_at"]))
    private_files = private_target = private_previous = None
    if config.get("private_projection_root") is not None:
        private_previous, private_previous_files = _previous_private(config, day, selected["synthetic_fixture"])
        private_builder = PrivateViewBuilder(synthetic_fixture=selected["synthetic_fixture"])
        private_files = private_builder.render(report_date=day, reports=market_reports,
            opening=report["opening"], previous_files=private_previous_files)
        private_target = _absolute(config, "private_projection_root") / day.isoformat()
    target = _absolute(config, "projection_root") / day.isoformat()
    builder.write_atomic(target, files)
    if private_files is not None:
        private_builder.write_atomic(private_target, private_files)
    state = {"report_date": day.isoformat(), "status": "rendered", "report_sha256": _hash(report_path),
             "projection_dir": str(target), "site_manifest_sha256": _hash(target / "site-manifest.json"),
             "synthetic_fixture": selected["synthetic_fixture"],
             "previous_projection_dir": str(previous_dir) if previous_dir else None,
             "previous_site_manifest_sha256": (
                 hashlib.sha256(previous_files["site-manifest.json"]).hexdigest()
                 if previous_files and "site-manifest.json" in previous_files else None),
             "private_projection_dir": str(private_target) if private_target else None,
             "private_manifest_sha256": (hashlib.sha256(private_files[PRIVATE_MANIFEST_NAME]).hexdigest()
                                         if private_files is not None else None),
             "private_previous_projection_dir": str(private_previous) if private_previous else None}
    _atomic(run_root / "coordinator-render.json", state)
    return state


def publish_stage(config: dict[str, Any], day: date) -> dict[str, Any]:
    run_root = _day_dir(config, day)
    render = _read(run_root / "coordinator-render.json")
    report = run_root / f"report-{day.isoformat()}.json"
    if _hash(report) != render.get("report_sha256"):
        raise ValueError("publication retry cannot use a changed report")
    if not config["publisher_enabled"]:
        state = {"report_date": day.isoformat(), "status": "publication_withheld",
                 "reason": "publisher_target_unconfirmed", "report_sha256": _hash(report)}
        _atomic(run_root / "coordinator-publication.json", state)
        return state
    base = _read(_absolute(config, "publisher_config"))
    if base.get("projection_dir") is not None:
        raise ValueError("publisher base config must not set projection_dir")
    if render.get("projection_dir") != str(_absolute(config, "projection_root") / day.isoformat()):
        raise ValueError("publisher target does not match the saved D projection")
    if (base.get("checkout_dir") != str(_absolute(config, "site_checkout")) or
            base.get("base_path") != config["base_path"] or
            _hash(Path(render["projection_dir"]) / "site-manifest.json") != render["site_manifest_sha256"]):
        raise ValueError("publisher target does not match the saved D projection")
    daily_config = {**base, "projection_dir": render["projection_dir"]}
    daily_path = run_root / "pages-publisher-config.json"
    if daily_path.is_symlink():
        raise ValueError("date-scoped publisher config cannot be a symlink")
    if not (daily_path.is_file() and _read(daily_path) == daily_config):
        _atomic(daily_path, daily_config)
    argv = [str(_absolute(config, "python")), str(_absolute(config, "publisher_script")),
            "--config", str(daily_path), "--publish"]
    if render["synthetic_fixture"]:
        argv.append("--allow-synthetic")
    try:
        done = subprocess.run(argv, check=False, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL, timeout=300)
        exit_code = done.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    state = {"report_date": day.isoformat(), "status": "published" if exit_code == 0 else "publisher_failed",
             "publisher_exit_code": exit_code, "report_sha256": _hash(report),
             "site_manifest_sha256": render["site_manifest_sha256"],
             "publisher_config_path": str(daily_path), "publisher_config_sha256": _hash(daily_path)}
    checkout_manifest = _absolute(config, "site_checkout") / "public" / "site-manifest.json"
    if exit_code == 0 and (not checkout_manifest.is_file() or
                           _hash(checkout_manifest) != render["site_manifest_sha256"]):
        state["status"] = "publisher_failed"
        state["reason"] = "published_checkout_manifest_mismatch"
    if state["status"] == "published":
        commit = subprocess.run(["git", "-C", str(_absolute(config, "site_checkout")),
                                 "rev-parse", "HEAD"], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True, check=False, timeout=10)
        if commit.returncode or not re.fullmatch(r"[0-9a-f]{40}", commit.stdout.strip()):
            state["status"] = "publisher_failed"
            state["reason"] = "site_commit_unavailable"
        else:
            state["site_commit"] = commit.stdout.strip()
    _atomic(run_root / "coordinator-publication.json", state)
    return state


def _actions_status(config: dict[str, Any], commit: str) -> dict[str, Any]:
    repository = config["actions_repository"]
    workflow = config["actions_workflow"]
    query = urlencode({"head_sha": commit, "branch": "site", "per_page": "20"})
    url = f"https://api.github.com/repos/{repository}/actions/workflows/{workflow}/runs?{query}"
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "daily-market-briefing"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read(2_000_000))
    except (OSError, ValueError, urllib.error.URLError):
        return {"status": "unavailable", "site_commit": commit}
    runs = [row for row in payload.get("workflow_runs", [])
            if row.get("head_sha") == commit and row.get("head_branch") == "site"]
    if not runs:
        return {"status": "pending", "site_commit": commit}
    newest = max(runs, key=lambda row: (row.get("run_number", 0), row.get("run_attempt", 0)))
    status = ("success" if newest.get("status") == "completed" and newest.get("conclusion") == "success"
              else "failed" if newest.get("status") == "completed" else "pending")
    return {"status": status, "site_commit": commit,
            "run_id": newest.get("id"), "conclusion": newest.get("conclusion")}


def monitor_stage(config: dict[str, Any], day: date, *, attempt: int) -> dict[str, Any]:
    if attempt not in range(len(RETRY_MINUTES)):
        raise ValueError("monitor attempt exceeds the bounded retry plan")
    root = _day_dir(config, day)
    selection_path = _absolute(config, "selection_root") / day.isoformat() / "selection-state.json"
    axes: dict[str, str] = {"input": "unknown", "inference": "unknown", "rights": "unknown",
                            "publisher": "unknown", "actions": "unknown", "public_url": "unknown"}
    if not selection_path.is_file():
        axes["input"] = "missing"
    else:
        selected = _read(selection_path)
        axes["input"] = "ready" if selected.get("status") == "selected" else selected.get("status", "missing")
        if selected.get("status") == "holiday":
            axes.update({"inference": "skipped", "rights": "not_applicable",
                         "publisher": "skipped", "actions": "skipped", "public_url": "skipped"})
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
        report = _read(report_path)
        gates = [item.get("publication", {}).get("status") for item in report.get("markets", [])]
        gates.append(report.get("opening", {}).get("publication", {}).get("status"))
        axes["rights"] = "allowed" if all(gate == "allowed" for gate in gates) else "withheld"
    publication_path = root / "coordinator-publication.json"
    if publication_path.is_file():
        axes["publisher"] = _read(publication_path).get("status", "publisher_failed")
    if axes["publisher"] == "publisher_failed" and attempt > 0 and config["publisher_enabled"]:
        # The publisher's journal retries the same commit from the saved report.
        axes["publisher"] = publish_stage(config, day)["status"]
    if axes["publisher"] == "published":
        published = _read(publication_path)
        commit = published.get("site_commit")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            axes["actions"] = "pending"
        elif config["external_verification_enabled"]:
            actions = _actions_status(config, commit)
            _atomic(root / "actions-deployment.json", actions)
            axes["actions"] = actions["status"]
        else:
            axes["actions"] = "pending"
        if axes["actions"] == "success":
            try:
                with urllib.request.urlopen(config["public_manifest_url"], timeout=10) as response:
                    public = json.loads(response.read(2_000_000))
                expected = _read(Path(_read(root / "coordinator-render.json")["projection_dir"]) /
                                 "site-manifest.json")
                axes["public_url"] = ("verified" if public.get("latest_report_date") == day.isoformat()
                                      and public == expected else "stale")
            except (OSError, ValueError, urllib.error.URLError):
                axes["public_url"] = "unreachable"
    if axes["input"] == "missing":
        status = "input_missing"
    elif axes["inference"] in {"unknown", "inference_failed"}:
        status = "inference_failed"
    elif axes["publisher"] in {"publisher_failed", "unknown"}:
        status = "publisher_failed"
    elif axes["publisher"] == "publication_withheld":
        status = "publication_withheld"
    elif axes["actions"] == "failed":
        status = "actions_failed"
    elif axes["actions"] != "success":
        status = "actions_pending"
    elif axes["public_url"] == "stale":
        status = "public_url_stale"
    elif axes["public_url"] == "unreachable":
        status = "public_url_unreachable"
    elif axes["public_url"] == "verified":
        status = "verified_withheld" if axes["rights"] == "withheld" else "verified"
    else:
        status = "actions_pending"
    state = {"report_date": day.isoformat(), "attempt": attempt,
             "scheduled_minute": RETRY_MINUTES[attempt], "status": status, "axes": axes,
             "retry_allowed": attempt < len(RETRY_MINUTES) - 1 and status not in {"verified", "verified_withheld"}}
    _atomic(root / f"monitor-{attempt}.json", state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modeler-daily-coordinator")
    parser.add_argument("stage", choices=("select", "infer", "render", "publish", "monitor"))
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
            elif args.stage == "render":
                result = render_stage(config, args.report_date)
            elif args.stage == "publish":
                result = publish_stage(config, args.report_date)
            else:
                result = monitor_stage(config, args.report_date, attempt=args.attempt)
        print(json.dumps({"status": result["status"], "report_date": args.report_date.isoformat()}, sort_keys=True))
        return 0 if result["status"] in {"selected", "partial", "holiday", "holiday_skipped",
                                          "inferred", "inferred_partial", "rendered",
                                          "published", "verified", "publication_withheld"} else 1
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
