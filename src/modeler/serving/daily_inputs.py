"""Freeze D 09:30 native inputs and a pinned three-model serving release."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .calendars import SessionCalendar
from .freshness import assess_freshness
from .orchestration import (
    EXPECTED_MODEL_IDENTITIES, SELECTION_MODES, code_inventory_sha256, session_lag,
    us_native_block_reason,
)

SEOUL = ZoneInfo("Asia/Seoul")
HEX = re.compile(r"[0-9a-f]{64}\Z")
NATIVE_NAMES = {"KR": ("feature_panel.parquet", "prepare_manifest.json"),
                "US": ("features.parquet", "manifest.json")}
ENTRYPOINTS = {"KR": "modeler.serving.adapters:infer_kr_daily",
               "US": "modeler.serving.adapters:infer_us_model"}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _under(path: Path, root: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError("pinned file must be regular and cannot be a symlink")
    resolved = path.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError("pinned file is outside its declared root")
    return resolved


def _pinned(path: Path, digest: str, root: Path) -> Path:
    if not isinstance(digest, str) or not HEX.fullmatch(digest):
        raise ValueError("release file SHA-256 is missing")
    checked = _under(path, root)
    if _hash(checked) != digest:
        raise ValueError("release file SHA-256 changed")
    return checked


def _time(raw: str) -> datetime:
    instant = datetime.fromisoformat(raw)
    if instant.tzinfo is None or instant.utcoffset() is None:
        raise ValueError("timestamp requires timezone")
    return instant


def _candidate(path: Path, market: str, prepared_root: Path, cutoff: datetime) -> dict[str, Any]:
    feature_name, manifest_name = NATIVE_NAMES[market]
    feature = _under(path.parent / feature_name, prepared_root)
    native_path = _under(path.parent / manifest_name, prepared_root)
    marker_path = _under(path.parent / "completion.json", prepared_root)
    marker = _load(marker_path)
    native = _load(native_path)
    if marker.get("schema_version") != "prepared-features-completion.v1":
        raise ValueError("native completion marker version is invalid")
    if marker.get("availability_evidence_type") != "prepared_features_completion" or native.get(
            "availability_evidence_type") != "prepared_features_completion":
        raise ValueError("native completion evidence is missing")
    feature_hash, native_hash, marker_hash = _hash(feature), _hash(native_path), _hash(marker_path)
    if (marker.get("features_sha256") != feature_hash or
            marker.get("native_prepare_manifest_sha256") != native_hash or
            native.get("features_sha256", native.get("input_sha256")) != feature_hash or
            native.get("input_sha256", feature_hash) != feature_hash):
        raise ValueError("native completion marker does not bind the prepared input")
    available = _time(marker["verified_available_by"])
    if available > cutoff:
        raise ValueError("native prepared input completed after 09:30 cutoff")
    if native.get("market") != market:
        raise ValueError("native prepared input market mismatch")
    if market == "US" and us_native_block_reason(native) is not None:
        raise ValueError("US diagnostic native input is not serving eligible: "
                         + str(us_native_block_reason(native)))
    asof = date.fromisoformat(native["feature_asof_date"])
    return {"feature": feature, "native_path": native_path, "marker_path": marker_path,
            "feature_hash": feature_hash, "native_hash": native_hash, "marker_hash": marker_hash,
            "available": available, "asof": asof, "native": native}


def _candidates(root: Path, market: str, cutoff: datetime) -> list[dict[str, Any]]:
    if not root.is_dir():
        return []
    manifest_name = NATIVE_NAMES[market][1]
    accepted: list[dict[str, Any]] = []
    for path in root.glob(f"**/{manifest_name}"):
        try:
            accepted.append(_candidate(path, market, root.resolve(strict=True), cutoff))
        except (ValueError, KeyError, OSError, json.JSONDecodeError):
            continue
    return sorted(accepted, key=lambda item: (item["asof"], item["available"], str(item["native_path"])), reverse=True)


def _check_data_files(items: Any, root: Path) -> None:
    """Verify packaged data files when the release declares them."""
    if not isinstance(items, list):
        raise ValueError("release data_files must be a list")
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("release data_files entry is invalid")
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts or not rel.parts or rel.as_posix() in seen:
            raise ValueError("release data_files path is invalid or duplicated")
        seen.add(rel.as_posix())
        _pinned(root / rel, item.get("sha256"), root)


def _release_jobs(release_path: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], bool]:
    release = _load(release_path)
    root = Path(release["release_root"]).resolve(strict=True)
    if root not in release_path.resolve(strict=True).parents or root.name in {"source", "src"}:
        raise ValueError("release manifest must live in its frozen release root")
    if release.get("schema_version") != "daily-briefing-release.v1" or release.get("frozen") is not True:
        raise ValueError("a frozen serving release is required")
    fixture = release.get("synthetic_fixture")
    if not isinstance(fixture, bool):
        raise ValueError("release synthetic_fixture marker is required")
    jobs = {}
    source_root = root / "src"
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError("frozen release source tree is missing")
    source_files = {path.resolve(strict=True) for path in source_root.rglob("*.py")
                    if "__pycache__" not in path.parts}
    if not source_files or any(path.is_symlink() for path in source_root.rglob("*.py")):
        raise ValueError("frozen release source inventory is invalid")
    if "data_files" in release:
        _check_data_files(release["data_files"], root)
    for raw in release.get("jobs", []):
        key = raw["market"], raw["model_id"]
        if key not in EXPECTED_MODEL_IDENTITIES or key in jobs:
            raise ValueError("unexpected or duplicate serving release job")
        if raw.get("entrypoint") != ENTRYPOINTS[key[0]]:
            raise ValueError("serving release entrypoint mismatch")
        files = []
        for item in raw["code_files"]:
            checked = _pinned(root / item["path"], item["sha256"], root)
            files.append((checked, item["sha256"]))
        if {path for path, _ in files} != source_files or len(files) != len(source_files):
            raise ValueError("release code inventory does not cover every Python source file")
        code_path = _pinned(root / raw["code_path"], raw["code_path_sha256"], root)
        if code_path not in {path for path, _ in files}:
            raise ValueError("entrypoint code is absent from release inventory")
        code_hash = code_inventory_sha256(files)
        if code_hash != raw.get("code_sha256"):
            raise ValueError("release code inventory SHA-256 changed")
        bundle = _pinned(root / raw["bundle_path"], raw["bundle_sha256"], root)
        jobs[key] = {"market": key[0], "model_id": key[1],
                     "model_version": raw["model_version"], "entrypoint": raw["entrypoint"],
                     "bundle_path": str(bundle), "bundle_sha256": raw["bundle_sha256"],
                     "code_path": str(code_path), "code_sha256": code_hash,
                     "code_files": [{"path": str(path), "sha256": sha} for path, sha in files]}
    if set(jobs) != EXPECTED_MODEL_IDENTITIES:
        raise ValueError("frozen release must pin exactly three model jobs")
    return jobs, fixture


def _atomic_new(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError("D selection is immutable; existing content differs")
        return
    fd, temp = tempfile.mkstemp(prefix=".selection-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


@contextmanager
def _lock(path: Path):
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def select(*, report_date: date, selected_at: datetime, prepared_root: Path,
           output_root: Path, release_manifest: Path, kr_calendar_path: Path,
           us_calendar_path: Path, us_expected_path: Path,
           mode: str = "scheduled") -> dict[str, Any]:
    """Select only producer-complete inputs; never infer E from observed A.

    ``mode`` is ``scheduled`` (the 09:30 event, D 09:30 through 10:00) or ``run_fallback`` (the run
    stage selects for itself because no selection exists; any time from D 09:30).  The mode never
    changes which inputs qualify: a producer completion after D 09:30 is excluded in both.
    """
    if mode not in SELECTION_MODES:
        raise ValueError("selection mode must be scheduled or run_fallback")
    prepared = prepared_root.resolve(strict=True)
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root.resolve(strict=True)
    if prepared not in output.parents:
        raise ValueError("selection output must be inside runner prepared_root")
    with _lock(output / f".{report_date.isoformat()}.lock"):
        return _select_once(report_date=report_date, selected_at=selected_at,
            prepared_root=prepared, output_root=output, release_manifest=release_manifest,
            kr_calendar_path=kr_calendar_path, us_calendar_path=us_calendar_path,
            us_expected_path=us_expected_path, mode=mode)


def _select_once(*, report_date: date, selected_at: datetime, prepared_root: Path,
                 output_root: Path, release_manifest: Path, kr_calendar_path: Path,
                 us_calendar_path: Path, us_expected_path: Path,
                 mode: str = "scheduled") -> dict[str, Any]:
    target = output_root / report_date.isoformat()
    if target.is_dir():
        state = _load(target / "selection-state.json")
        if (state.get("report_date") != report_date.isoformat() or
                state.get("release_manifest_sha256") != _hash(release_manifest) or
                state.get("jobs_config_sha256") != _hash(target / "jobs.json")):
            raise ValueError("existing D selection does not match frozen inputs")
        if state.get("status") != "holiday" and (
                state.get("kr_calendar_sha256") != _hash(kr_calendar_path) or
                state.get("us_calendar_sha256") != (_hash(us_calendar_path) if us_calendar_path.is_file() else None) or
                state.get("us_expected_source_sha256") != (_hash(us_expected_path) if us_expected_path.is_file() else None)):
            raise ValueError("existing D source schedule or calendar changed")
        for market in ("KR", "US"):
            for job in state.get("jobs", []):
                if job.get("market") == market and _hash(Path(job["prepared_manifest"])) != job.get("prepared_manifest_sha256"):
                    raise ValueError("existing selected manifest changed")
        return state
    if target.exists() or target.is_symlink():
        raise ValueError("D selection path is not a regular directory")
    decision = datetime.combine(report_date, time(10), SEOUL)
    cutoff = datetime.combine(report_date, time(9, 30), SEOUL)
    if selected_at.tzinfo is None or selected_at < cutoff or (mode == "scheduled" and selected_at > decision):
        raise ValueError("selection must occur from D 09:30 through D 10:00 KST"
                         if mode == "scheduled" else "selection must occur at or after D 09:30 KST")
    calendars = {"KR": SessionCalendar.from_manifest(_load(kr_calendar_path))}
    if calendars["KR"].is_session(report_date) is None:
        raise ValueError("KR calendar does not cover report date")
    if calendars["KR"].is_session(report_date) is False:
        stage = Path(tempfile.mkdtemp(prefix=f".{report_date.isoformat()}-", dir=output_root))
        _atomic_new(stage / "jobs.json", {"jobs": []})
        holiday = {"status": "holiday", "report_date": report_date.isoformat(), "jobs": [],
                   "jobs_config": str(target / "jobs.json"),
                   "jobs_config_sha256": _hash(stage / "jobs.json"),
                   "release_manifest_sha256": _hash(release_manifest),
                   "synthetic_fixture": False}
        _atomic_new(stage / "selection-state.json", holiday)
        os.rename(stage, target)
        return holiday
    release_jobs, fixture_mode = _release_jobs(release_manifest)
    us_policy_error = None
    try:
        calendars["US"] = SessionCalendar.from_manifest(_load(us_calendar_path))
        expected_source = _load(us_expected_path)
        reviewed = expected_source.get("reviewed_status")
        if (expected_source.get("schema_version") != "us-expected-source.v1" or
                reviewed not in ({"synthetic_fixture"} if fixture_mode else {"confirmed"}) or
                not isinstance(expected_source.get("source_reference"), str) or
                not expected_source["source_reference"]):
            raise ValueError("US source expectation lacks independent reviewed evidence")
        expected_raw = expected_source.get("expected_session_by_report_date", {}).get(report_date.isoformat())
        expected = date.fromisoformat(expected_raw) if isinstance(expected_raw, str) else None
        lag_limit = expected_source.get("market_lag_limit_sessions")
        if isinstance(lag_limit, bool) or not isinstance(lag_limit, int) or lag_limit < 0:
            raise ValueError("US source schedule requires an explicit market lag limit")
        latest_us = calendars["US"].latest_completed_before(decision)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        us_policy_error = "us_expected_source_or_calendar_unavailable"
        expected, lag_limit, latest_us = None, None, None
    k_day = calendars["KR"].previous_session(report_date)
    stage = Path(tempfile.mkdtemp(prefix=f".{report_date.isoformat()}-", dir=output_root))
    result = {"schema_version": "daily-input-selection.v1", "report_date": report_date.isoformat(),
              "selected_at": selected_at.isoformat(), "selection_mode": mode,
              "input_cutoff": cutoff.isoformat(),
              "release_manifest": str(release_manifest.resolve(strict=True)),
              "release_manifest_sha256": _hash(release_manifest),
              "kr_calendar_sha256": _hash(kr_calendar_path),
              "us_calendar_sha256": _hash(us_calendar_path) if us_calendar_path.is_file() else None,
              "us_expected_source_sha256": _hash(us_expected_path) if us_expected_path.is_file() else None,
              "synthetic_fixture": fixture_mode, "jobs": [], "markets": {}}
    for market, calendar, native_root in (
            ("KR", calendars["KR"], prepared_root / "kr"),
            ("US", calendars.get("US"), prepared_root / "us")):
        selected = None
        reason = "producer_completion_missing"
        if market == "US" and (us_policy_error or expected is None):
            result["markets"][market] = {"status": "unavailable", "reason": us_policy_error or "us_expected_source_schedule_missing"}
            continue
        for candidate in _candidates(native_root, market, cutoff):
            if bool(candidate["native"].get("synthetic_fixture", False)) != fixture_mode:
                reason = "native_fixture_mode_mismatch"
                continue
            if market == "KR" and (k_day is None or candidate["asof"] > k_day):
                # K보다 이른 prepared는 stale로 고릅니다(변경 3). K보다 늦은 것은 쓰지 않습니다.
                reason = "kr_prepared_newer_than_k"
                continue
            if market == "US" and (expected is None or latest_us is None or candidate["asof"] > latest_us):
                reason = "us_expected_source_schedule_missing" if expected is None else "us_calendar_unavailable"
                continue
            evidence = {"features_sha256": candidate["feature_hash"],
                        "native_prepare_manifest_sha256": candidate["native_hash"],
                        "completion_marker_sha256": candidate["marker_hash"]}
            freshness = assess_freshness(report_date=report_date, market=market,
                feature_asof_date=candidate["asof"], decision_at=decision, input_cutoff=cutoff,
                calendar=calendar, latest_us_session=latest_us if market == "US" else None,
                expected_us_session=expected if market == "US" else None,
                actual_us_session=candidate["asof"] if market == "US" else None,
                verified_available_by=candidate["available"],
                availability_evidence_type="prepared_features_completion",
                availability_evidence=evidence, max_us_market_lag=lag_limit if market == "US" else None)
            if freshness.status not in {"ok", "stale"}:
                reason = freshness.reason or freshness.status
                continue
            selected = candidate, freshness, evidence
            break
        if selected is None:
            result["markets"][market] = {"status": "unavailable", "reason": reason}
            continue
        candidate, freshness, evidence = selected
        selection_path = target / f"{market.lower()}-selection.json"
        selection = {"market": market, "report_date": report_date.isoformat(),
            "feature_asof_date": candidate["asof"].isoformat(),
            "input_sha256": candidate["feature_hash"],
            "native_prepare_manifest_sha256": candidate["native_hash"],
            "completion_marker_path": candidate["marker_path"].name,
            "completion_marker_sha256": candidate["marker_hash"],
            "availability_evidence_type": "prepared_features_completion",
            "availability_evidence": evidence,
            "verified_available_by": candidate["available"].isoformat(),
            "source_first_available_at": None, "input_cutoff": cutoff.isoformat(),
            # 입력이 끝난 시각(producer)과 selection을 만든 시각을 따로 적습니다(변경 5).
            "producer_completed_at": candidate["available"].isoformat(),
            "selected_at": selected_at.isoformat(), "selection_mode": mode,
            "calendar": _load(kr_calendar_path if market == "KR" else us_calendar_path),
            "freshness_status": freshness.status, "freshness_reason": freshness.reason,
            "lag_sessions": session_lag(market, freshness.as_dict())}
        if market == "KR":
            selection["kr_session"] = k_day.isoformat()
        else:
            selection.update(latest_us_session=latest_us.isoformat(),
                             expected_us_session=expected.isoformat(),
                             actual_us_session=candidate["asof"].isoformat(),
                             market_lag_limit_sessions=lag_limit)
        staged_selection_path = stage / selection_path.name
        _atomic_new(staged_selection_path, selection)
        for key, pinned in sorted(release_jobs.items()):
            if key[0] != market:
                continue
            result["jobs"].append({**pinned,
                "prepared_input": str(candidate["feature"]),
                "input_sha256": candidate["feature_hash"],
                "prepared_manifest": str(selection_path),
                "prepared_manifest_sha256": _hash(staged_selection_path),
                "native_manifest": str(candidate["native_path"]),
                "native_manifest_sha256": candidate["native_hash"]})
        result["markets"][market] = {"status": freshness.status,
                                      "feature_asof_date": candidate["asof"].isoformat(),
                                      "lag_sessions": session_lag(market, freshness.as_dict()),
                                      "freshness_reason": freshness.reason,
                                      "producer_completed_at": candidate["available"].isoformat()}
    jobs_path = target / "jobs.json"
    _atomic_new(stage / "jobs.json", {"jobs": result["jobs"]})
    result["jobs_config"] = str(jobs_path)
    result["jobs_config_sha256"] = _hash(stage / "jobs.json")
    result["status"] = "selected" if len(result["jobs"]) == 3 else "partial"
    _atomic_new(stage / "selection-state.json", result)
    try:
        os.rename(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modeler-serving-inputs")
    parser.add_argument("--report-date", type=date.fromisoformat, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--kr-calendar", type=Path, required=True)
    parser.add_argument("--us-calendar", type=Path, required=True)
    parser.add_argument("--us-expected-source", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = select(report_date=args.report_date, selected_at=datetime.now(SEOUL),
            prepared_root=args.prepared_root, output_root=args.output_root,
            release_manifest=args.release_manifest, kr_calendar_path=args.kr_calendar,
            us_calendar_path=args.us_calendar, us_expected_path=args.us_expected_source)
        print(json.dumps({"status": result["status"], "report_date": result["report_date"],
                          "job_count": len(result["jobs"])}, sort_keys=True))
        return 0 if result["status"] in {"selected", "holiday", "partial"} else 1
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}), file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
