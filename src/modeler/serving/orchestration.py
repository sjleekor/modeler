"""Pinned daily inference execution and report aggregation."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from zoneinfo import ZoneInfo

from .calendars import SessionCalendar
from .freshness import MAX_STALE_SESSIONS, assess_freshness
from .opening_validation import validate_opening
from .schema import validate_report

SEOUL = ZoneInfo("Asia/Seoul")
IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]+$")
SELECTION_MODES = frozenset({"scheduled", "run_fallback"})
EXPECTED_MODEL_IDENTITIES = frozenset({
    ("KR", "kr_daily_h20_v1"),
    ("US", "us_exploratory_20260929_r1_lightgbm"),
    ("US", "us_exploratory_20260929_r1_ridge"),
})


@dataclass(frozen=True)
class InferenceJob:
    market: str
    model_id: str
    model_version: str
    prepared_input: Path
    input_sha256: str
    prepared_manifest: Path
    prepared_manifest_sha256: str
    native_manifest: Path
    native_manifest_sha256: str
    bundle_path: Path
    bundle_sha256: str
    code_path: Path
    code_sha256: str
    infer: Callable[["InferenceContext"], dict[str, Any]]
    code_files: tuple[tuple[Path, str], ...] = ()


@dataclass(frozen=True)
class InferenceContext:
    """Validated D-selection and pinned artifacts passed to model adapters."""

    report_date: date
    decision_at: datetime
    market: str
    model_id: str
    model_version: str
    prepared_input: Path
    prepared_manifest: Path
    native_manifest: Path
    bundle_manifest: Path
    input_sha256: str
    prepared_manifest_sha256: str
    native_manifest_sha256: str
    bundle_sha256: str
    code_sha256: str
    feature_asof_date: str
    freshness_status: str
    freshness: dict[str, Any]
    selection: dict[str, Any]
    native_preparation: dict[str, Any]
    fixture_mode: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_inventory_sha256(files: Iterable[tuple[Path, str]]) -> str:
    """Hash a stable list of source file paths and their expected SHA-256 values."""
    records = sorted((Path(path).as_posix(), digest) for path, digest in files)
    if not records or len({path for path, _ in records}) != len(records):
        raise ValueError("code inventory must contain unique source files")
    if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for _, digest in records):
        raise ValueError("code inventory entries require SHA-256 values")
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


@contextmanager
def _run_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _checked_file(path: Path, expected_sha256: str, *, root: Path | None = None) -> Path:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ValueError("all pinned files require a SHA-256 hash")
    if path.is_symlink():
        raise ValueError("pinned files cannot be symlinks")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or (root is not None and root not in resolved.parents):
        raise ValueError("pinned file is outside its allowed directory")
    if _sha256(resolved) != expected_sha256:
        raise ValueError("pinned file hash does not match")
    return resolved


def _prepared_metadata(job: InferenceJob, prepared_root: Path,
                       report_date: date, decision_at: datetime) -> dict[str, Any]:
    if job.market not in {"KR", "US"} or not IDENTIFIER.fullmatch(job.model_id):
        raise ValueError("invalid market/model identity")
    root = prepared_root.resolve(strict=True)
    input_path = _checked_file(job.prepared_input, job.input_sha256, root=root)
    manifest_path = _checked_file(job.prepared_manifest, job.prepared_manifest_sha256, root=root)
    native_manifest_path = _checked_file(job.native_manifest, job.native_manifest_sha256, root=root)
    bundle_path = _checked_file(job.bundle_path, job.bundle_sha256)
    code_files = tuple(job.code_files)
    if code_files:
        verified_code = {Path(path): _checked_file(path, expected)
                         for path, expected in code_files}
        if Path(job.code_path) not in verified_code:
            raise ValueError("code_path must be included in the pinned code inventory")
        code_path = verified_code[Path(job.code_path)]
        if code_inventory_sha256(code_files) != job.code_sha256:
            raise ValueError("code inventory hash does not match")
    else:
        code_path = _checked_file(job.code_path, job.code_sha256)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    native = json.loads(native_manifest_path.read_text(encoding="utf-8"))
    if job.market == "US" and us_native_block_reason(native) is not None:
        raise ValueError("US diagnostic native input is not serving eligible: "
                         + str(us_native_block_reason(native)))
    if manifest.get("market") != job.market or manifest.get("report_date") != report_date.isoformat():
        raise ValueError("prepared manifest does not match report date/market")
    if manifest.get("input_sha256") != job.input_sha256:
        raise ValueError("prepared manifest input hash does not match")
    native_digest = manifest.get("native_prepare_manifest_sha256")
    if (native_digest != job.native_manifest_sha256 or native.get("market") != job.market or
            native.get("feature_asof_date") != manifest.get("feature_asof_date")):
        raise ValueError("selection manifest does not match its native prepared input")
    if native.get("input_sha256") not in {None, job.input_sha256}:
        raise ValueError("native prepared manifest input hash does not match")
    asof = manifest.get("feature_asof_date")
    if not isinstance(asof, str):
        raise ValueError("prepared manifest feature_asof_date is missing")
    date.fromisoformat(asof)
    cutoff = datetime.fromisoformat(manifest.get("input_cutoff", ""))
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("prepared manifest timestamps must include timezones")
    expected_cutoff = decision_at.astimezone(SEOUL).replace(hour=9, minute=30, second=0, microsecond=0)
    if cutoff.astimezone(SEOUL) != expected_cutoff:
        raise ValueError("prepared input cutoff is not D 09:30 Asia/Seoul")
    # 입력 완료 시각과 selection을 만든 시각은 다른 시각입니다(2026-10-05 변경 5). 미래 정보는
    # **입력 완료 시각 <= cutoff**로 막고, selection을 D 10:00 뒤에 만든 것은 run이 대신 고른
    # `run_fallback`일 때만 받아서 기록합니다. 옛 selection은 `completed_at` 하나였고 그 값이
    # selection을 만든 시각이었습니다.
    selected_raw = manifest.get("selected_at", manifest.get("completed_at", ""))
    selected_at = datetime.fromisoformat(selected_raw)
    if selected_at.tzinfo is None or selected_at.utcoffset() is None:
        raise ValueError("prepared manifest timestamps must include timezones")
    selection_mode = manifest.get("selection_mode", "scheduled")
    if selection_mode not in SELECTION_MODES:
        raise ValueError("selection_mode must be scheduled or run_fallback")
    if selection_mode == "scheduled" and selected_at > decision_at:
        raise ValueError("selection manifest was completed after decision_at")
    calendar_raw = manifest.get("calendar")
    if not isinstance(calendar_raw, dict):
        raise ValueError("selection manifest requires an explicit session calendar")
    calendar = SessionCalendar.from_manifest(calendar_raw)
    marker_path_raw = manifest.get("completion_marker_path")
    marker_sha = manifest.get("completion_marker_sha256")
    if (not isinstance(marker_path_raw, str) or not marker_path_raw or
            not isinstance(marker_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", marker_sha)):
        raise ValueError("selection completion marker path or SHA-256 is missing")
    marker_path = Path(marker_path_raw)
    if not marker_path.is_absolute():
        marker_path = native_manifest_path.parent / marker_path
    marker_path = _checked_file(marker_path, marker_sha, root=root)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    verified_raw = manifest.get("verified_available_by")
    verified_available = datetime.fromisoformat(verified_raw) if isinstance(verified_raw, str) else None
    evidence_type = manifest.get("availability_evidence_type")
    availability_evidence = manifest.get("availability_evidence")
    if (verified_raw is None or marker.get("verified_available_by") != verified_raw or
            manifest.get("availability_evidence_type") != marker.get("availability_evidence_type") or
            evidence_type != "prepared_features_completion" or
            marker.get("schema_version") != "prepared-features-completion.v1" or
            marker.get("native_prepare_manifest_sha256") != job.native_manifest_sha256 or
            marker.get("features_sha256") != job.input_sha256 or
            marker.get("availability_evidence_type") != native.get("availability_evidence_type") or
            not isinstance(availability_evidence, dict) or
            availability_evidence.get("features_sha256") != job.input_sha256 or
            availability_evidence.get("native_prepare_manifest_sha256") != job.native_manifest_sha256 or
            availability_evidence.get("completion_marker_sha256") != manifest.get("completion_marker_sha256") or
            native.get("features_sha256", native.get("input_sha256")) != job.input_sha256):
        raise ValueError("selection lacks matching immutable prepared-feature completion evidence")
    if verified_available is None or verified_available.tzinfo is None or verified_available.utcoffset() is None:
        raise ValueError("verified_available_by must include a timezone")
    if verified_available > cutoff:
        # run_fallback이어도 입력 완료가 D 09:30보다 늦으면 쓰지 않습니다(PIT).
        raise ValueError("input producer completed after the 09:30 input cutoff")
    producer_raw = manifest.get("producer_completed_at", verified_raw)
    if producer_raw != verified_raw:
        raise ValueError("producer_completed_at disagrees with the verified availability time")
    first_raw = manifest.get("source_first_available_at")
    first_available = datetime.fromisoformat(first_raw) if isinstance(first_raw, str) else None
    if first_available and (first_available.tzinfo is None or first_available.utcoffset() is None):
        raise ValueError("source_first_available_at must include a timezone")
    if job.market == "US":
        def _manifest_session(field: str) -> date | None:
            value = manifest.get(field)
            return date.fromisoformat(value) if isinstance(value, str) else None
        freshness = assess_freshness(report_date=report_date, market="US",
            feature_asof_date=date.fromisoformat(asof), decision_at=decision_at,
            input_cutoff=cutoff, calendar=calendar,
            latest_us_session=_manifest_session("latest_us_session"),
            expected_us_session=_manifest_session("expected_us_session"),
            actual_us_session=_manifest_session("actual_us_session"),
            verified_available_by=verified_available,
            availability_evidence_type=evidence_type,
            availability_evidence=availability_evidence,
            source_first_available_at=first_available,
            max_us_market_lag=manifest.get("market_lag_limit_sessions"))
    else:
        # 기준일이 K보다 이른 prepared(K')도 받습니다. 달력이 K를 다시 계산하고
        # assess_freshness가 이른 만큼을 stale 지연 세션 수로 적습니다(2026-10-05 변경 3).
        k_raw = manifest.get("kr_session")
        k_session = date.fromisoformat(k_raw) if isinstance(k_raw, str) else None
        if k_session is None or date.fromisoformat(asof) > k_session:
            raise ValueError("prepared KR feature date is newer than K")
        freshness = assess_freshness(report_date=report_date, market="KR",
            feature_asof_date=date.fromisoformat(asof), decision_at=decision_at,
            input_cutoff=cutoff, calendar=calendar,
            verified_available_by=verified_available,
            availability_evidence_type=evidence_type,
            availability_evidence=availability_evidence,
            source_first_available_at=first_available)
    freshness_status = freshness.status
    if manifest.get("freshness_status") != freshness_status:
        raise ValueError("selection freshness status disagrees with calendar assessment")
    if freshness_status not in {"ok", "stale"}:
        raise ValueError("prepared input freshness is unavailable for inference")
    return {"input_path": input_path, "manifest_path": manifest_path,
            "native_manifest_path": native_manifest_path,
            "bundle_path": bundle_path, "code_path": code_path,
            "input_sha256": job.input_sha256, "prepared_manifest_sha256": job.prepared_manifest_sha256,
            "native_manifest_sha256": job.native_manifest_sha256,
            "bundle_sha256": job.bundle_sha256, "code_sha256": job.code_sha256,
            "code_files": tuple((Path(path), digest) for path, digest in code_files),
            "feature_asof_date": asof, "freshness_status": freshness_status,
            "selected_at": selected_at.isoformat(), "selection_mode": selection_mode,
            "producer_completed_at": verified_available.isoformat(),
            "input_cutoff": cutoff.isoformat(), "verified_available_by": verified_available.isoformat(),
            "availability_evidence_type": evidence_type,
            "availability_evidence": availability_evidence,
            "source_first_available_at": first_available.isoformat() if first_available else None,
            "latest_us_session": freshness.latest_us_session,
            "expected_us_session": freshness.expected_us_session,
            "actual_us_session": freshness.actual_us_session,
            "delivery_lag_sessions": freshness.delivery_lag,
            "market_lag_sessions": freshness.market_lag,
            "market_lag_limit_sessions": manifest.get("market_lag_limit_sessions"),
            "freshness": freshness.as_dict()}


def _job_run_id(job: InferenceJob, metadata: dict[str, Any], report_date: date,
                decision_at: datetime) -> str:
    key = {"report_date": report_date.isoformat(), "decision_at": decision_at.isoformat(),
           "market": job.market, "model_id": job.model_id, "model_version": job.model_version,
           **{key: metadata[key] for key in ("input_sha256", "prepared_manifest_sha256", "native_manifest_sha256", "bundle_sha256", "code_sha256")}}
    raw = json.dumps(key, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


#: 순위를 낸 섹션의 상태. `stale`은 오래된 입력으로 만들었어도 낸 것입니다.
PRODUCED_STATES = frozenset({"ok", "partial", "stale"})


def envelope_status(markets: list[dict[str, Any]], failures: list[dict[str, str]]) -> str:
    """전 섹션이 ok면 ok, 낸 섹션이 하나도 없을 때만 failed, 나머지는 partial (2026-10-05 변경 6).

    섹션은 모델 report 하나입니다. 실패한 모델(`failures`)은 낸 섹션이 아니라서 ok를 막습니다.
    KR adapter가 늘 `partial`을 내므로(관리종목·거래정지 미확인) KR이 들어간 날은 `ok`가 안 나옵니다.
    공개 게이트와 opening은 이 상태에 넣지 않습니다. Pages 공개가 없어졌고, 게이트는
    `monitor`의 rights 축이 따로 봅니다.
    """
    if not any(item["status"] in PRODUCED_STATES for item in markets):
        return "failed"
    if not failures and all(item["status"] == "ok" for item in markets):
        return "ok"
    return "partial"


def session_lag(market: str, freshness: dict[str, Any]) -> int | None:
    """표시용 지연 세션 수: KR은 K와의 차이, US는 최신 완료 세션 U와의 차이입니다."""
    value = freshness.get("delivery_lag") if market == "KR" else freshness.get("market_lag")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _annotate_selection(report: dict[str, Any], metadata: dict[str, Any], market: str) -> None:
    """신선도와 selection 시각을 report에 적고, 5세션을 넘은 순위는 쓰지 않습니다 (Q5).

    신선도는 막지 않고 표시합니다(2026-10-05). 기준일이 K·U보다 늦은 입력은 `stale`이고
    지연 세션 수를 `quality.lag_sessions`에 적습니다. 지연이 `MAX_STALE_SESSIONS`를 넘으면
    순위 목록을 비우고 사유만 남깁니다. 비운 순위 수는 `rankings_withheld`에 있습니다.
    """
    quality = dict(report["quality"])
    quality["freshness_status"] = metadata["freshness_status"]
    quality["selection_mode"] = metadata["selection_mode"]
    quality["selected_at"] = metadata["selected_at"]
    quality["producer_completed_at"] = metadata["producer_completed_at"]
    lag = session_lag(market, metadata["freshness"])
    if lag is not None:
        quality["lag_sessions"] = lag
    if metadata["freshness_status"] == "stale":
        if report["status"] in {"ok", "partial"}:
            quality["status_before_stale"] = report["status"]
            report["status"] = "stale"
        if lag is not None and lag > MAX_STALE_SESSIONS:
            quality["rankings_withheld"] = {
                "reason": "stale_lag_exceeds_limit", "lag_sessions": lag,
                "limit_sessions": MAX_STALE_SESSIONS, "ranking_count": len(report["rankings"])}
            report["rankings"] = []
    report["quality"] = quality


#: Keys of ``selection-state.json["markets"][market]`` that ``selection_summary`` copies.
SELECTION_SUMMARY_STR_KEYS = ("status", "reason", "feature_asof_date", "freshness_reason")


def selection_summary(state: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """What the 09:30 select recorded per market, for the unit's failed and stale sections.

    The envelope only names the cause class of a failed model (``MissingInference`` when select
    found no input at all), so a reader cannot tell why.  This copies the few keys select itself
    wrote (``status``, ``reason`` such as ``producer_completion_missing``, ``feature_asof_date``,
    ``lag_sessions``, ``freshness_reason``) and nothing else: no paths, hashes or free text.
    ``None`` when the state names no market (a holiday, or an old state), so the envelope then has
    no ``selection_summary`` field at all.
    """
    markets = state.get("markets")
    if not isinstance(markets, dict):
        return None
    summary: dict[str, dict[str, Any]] = {}
    for market in ("KR", "US"):
        block = markets.get(market)
        if not isinstance(block, dict):
            continue
        row: dict[str, Any] = {key: block[key] for key in SELECTION_SUMMARY_STR_KEYS
                               if isinstance(block.get(key), str) and block[key]}
        lag = block.get("lag_sessions")
        if isinstance(lag, int) and not isinstance(lag, bool):
            row["lag_sessions"] = lag
        if row:
            summary[market] = row
    return summary or None


def combine_reports(*, report_date: date, decision_at: datetime,
                    reports: Iterable[dict[str, Any]],
                    opening: dict[str, Any] | None = None,
                    failures: Iterable[dict[str, str]] = (),
                    expected_identities: Iterable[tuple[str, str]] = EXPECTED_MODEL_IDENTITIES,
                    selection_summary: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    markets = []
    for report in reports:
        validate_report(report)
        if report["report_date"] != report_date.isoformat():
            raise ValueError("all market reports must match report_date")
        if report["decision_at"] != decision_at.isoformat():
            raise ValueError("all market reports must match decision_at")
        markets.append(report)
    failures = list(failures)
    actual = {(item["market"], item["model_id"]) for item in markets}
    expected = set(expected_identities)
    known_failures = {(x.get("market"), x.get("model_id")) for x in failures}
    unexpected = actual - expected
    if unexpected:
        failures.extend({"market": market, "model_id": model_id, "error_class": "UnexpectedInference"}
                        for market, model_id in sorted(unexpected))
    missing = expected - actual - known_failures
    if missing:
        failures.extend({"market": market, "model_id": model_id, "error_class": "MissingInference"}
                        for market, model_id in sorted(missing))
    status = envelope_status(markets, failures)
    fixture_marks = {bool(item.get("synthetic_fixture", False)) for item in markets}
    envelope = {"schema_version": "1.0", "report_date": report_date.isoformat(),
                "decision_at": decision_at.isoformat(), "status": status,
                "markets": sorted(markets, key=lambda item: (item["market"], item["model_id"])),
                "failures": failures,
                "opening": opening or {"status": "unavailable",
                                       "publication": {"status": "unresolved", "evidence": []}},
                "synthetic_fixture": fixture_marks == {True}}
    if selection_summary:  # without it the envelope is byte-identical to the earlier one
        envelope["selection_summary"] = selection_summary
    return envelope


def run_daily(*, report_date: date, decision_at: datetime, prepared_root: Path,
              run_root: Path, jobs: Iterable[InferenceJob],
              opening: dict[str, Any] | None = None,
              opening_calendar: SessionCalendar | None = None,
              expected_identities: Iterable[tuple[str, str]] = EXPECTED_MODEL_IDENTITIES,
              historical_replay: bool = False, fixture_mode: bool = False,
              invocation_id: str | None = None,
              now: datetime | None = None,
              selection_summary: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """Infer from hash-pinned inputs, preserve revisions, and record failures safely."""
    if decision_at.tzinfo is None or decision_at.utcoffset() is None:
        raise ValueError("decision_at requires a timezone")
    decision = decision_at.astimezone(SEOUL)
    if decision.date() != report_date or decision.time().replace(tzinfo=None) != time(10):
        raise ValueError("daily inference must use report_date at 10:00 Asia/Seoul")
    actual_now = (now or datetime.now(timezone.utc))
    if actual_now.tzinfo is None or actual_now.utcoffset() is None:
        raise ValueError("now requires a timezone")
    if actual_now < decision and not (historical_replay or fixture_mode):
        raise ValueError("future decision cutoff requires fixture_mode or historical_replay")
    prepared_root = prepared_root.resolve(strict=True)
    run_root.mkdir(parents=True, exist_ok=True)
    jobs = list(jobs)
    identities = [(job.market, job.model_id) for job in jobs]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate market/model inference job")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    validated_opening_calendar = opening_calendar
    run_meta: dict[str, Any] = {"report_date": report_date.isoformat(),
        "decision_at": decision.isoformat(), "started_at": actual_now.astimezone(timezone.utc).isoformat(),
        "historical_replay": historical_replay, "fixture_mode": fixture_mode,
        "invocation_id": invocation_id, "jobs": []}
    with _run_lock(run_root / ".run.lock"):
        for job in jobs:
            record = {"market": job.market, "model_id": job.model_id, "status": "failed"}
            try:
                metadata = _prepared_metadata(job, prepared_root, report_date, decision)
                native_metadata = json.loads(metadata["native_manifest_path"].read_text(encoding="utf-8"))
                if bool(native_metadata.get("synthetic_fixture", False)) != fixture_mode:
                    raise ValueError("native input fixture mode does not match run mode")
                if job.market == "KR":
                    selected_metadata = json.loads(metadata["manifest_path"].read_text(encoding="utf-8"))
                    job_calendar = SessionCalendar.from_manifest(selected_metadata["calendar"])
                    if validated_opening_calendar is not None and job_calendar != validated_opening_calendar:
                        raise ValueError("KR job calendar differs from pinned opening calendar")
                    validated_opening_calendar = job_calendar
                run_id = _job_run_id(job, metadata, report_date, decision)
                result_path = (run_root / "inference" / report_date.isoformat() / job.market /
                               job.model_id / f"{run_id}.json")
                if result_path.exists():
                    saved = json.loads(result_path.read_text(encoding="utf-8"))
                    report = saved["report"]
                    validate_report(report)
                    expected_identity = (job.market, job.model_id, job.model_version,
                                         report_date.isoformat(), decision.isoformat())
                    actual_identity = (report["market"], report["model_id"], report["model_version"],
                                       report["report_date"], report["decision_at"])
                    hashes_match = all(saved.get(key) == metadata[key] for key in
                                       ("input_sha256", "prepared_manifest_sha256", "native_manifest_sha256",
                                        "bundle_sha256", "code_sha256"))
                    hashes_match = hashes_match and saved.get("freshness") == metadata["freshness"]
                    if expected_identity != actual_identity or not hashes_match or saved.get("run_id") != run_id:
                        raise ValueError("saved inference does not match current identity or pinned hashes")
                    results.append(report)
                    record.update(status="reused", run_id=run_id,
                                  inference_started_at=saved["inference_started_at"])
                    run_meta["jobs"].append(record)
                    continue
                started = datetime.now(timezone.utc)
                context = InferenceContext(
                    report_date=report_date, decision_at=decision, market=job.market,
                    model_id=job.model_id, model_version=job.model_version,
                    prepared_input=metadata["input_path"],
                    prepared_manifest=metadata["manifest_path"],
                    native_manifest=metadata["native_manifest_path"],
                    bundle_manifest=metadata["bundle_path"],
                    input_sha256=metadata["input_sha256"],
                    prepared_manifest_sha256=metadata["prepared_manifest_sha256"],
                    native_manifest_sha256=metadata["native_manifest_sha256"],
                    bundle_sha256=metadata["bundle_sha256"], code_sha256=metadata["code_sha256"],
                    feature_asof_date=metadata["feature_asof_date"],
                    freshness_status=metadata["freshness_status"], freshness=metadata["freshness"],
                    selection=json.loads(metadata["manifest_path"].read_text(encoding="utf-8")),
                    native_preparation=json.loads(metadata["native_manifest_path"].read_text(encoding="utf-8")),
                    fixture_mode=fixture_mode)
                report = job.infer(context)
                validate_report(report)
                if (report["market"], report["model_id"], report["model_version"],
                        report["report_date"], report["decision_at"]) != (
                        job.market, job.model_id, job.model_version, report_date.isoformat(), decision.isoformat()):
                    raise ValueError("inference result identity/cutoff does not match its pinned job")
                if report["feature_asof_date"] != metadata["feature_asof_date"]:
                    raise ValueError("inference result feature date does not match prepared input manifest")
                _annotate_selection(report, metadata, job.market)
                report["inference_started_at"] = started.isoformat()
                saved = {**{key: metadata[key] for key in
                            ("input_sha256", "prepared_manifest_sha256", "native_manifest_sha256",
                             "bundle_sha256", "code_sha256")},
                         "freshness": metadata["freshness"],
                         "run_id": run_id, "inference_started_at": started.isoformat(), "report": report}
                _atomic_json(result_path, saved)
                results.append(report)
                record.update(status="ok", run_id=run_id, inference_started_at=started.isoformat())
            except Exception as exc:  # Record only the class; messages can contain paths/data.
                record["error_class"] = type(exc).__name__
                failures.append({"market": job.market, "model_id": job.model_id,
                                 "error_class": type(exc).__name__})
            run_meta["jobs"].append(record)
        safe_opening = validate_opening(opening, report_date=report_date,
                                        decision_at=decision, calendar=validated_opening_calendar,
                                        fixture_mode=fixture_mode)
        envelope = combine_reports(report_date=report_date, decision_at=decision,
                                   reports=results, opening=safe_opening, failures=failures,
                                   expected_identities=expected_identities,
                                   selection_summary=selection_summary)
        envelope["historical_replay"] = historical_replay
        envelope["synthetic_fixture"] = fixture_mode
        _atomic_json(run_root / f"report-{report_date.isoformat()}.json", envelope)
        _atomic_json(run_root / "run-state.json", run_meta)
    return envelope


def retry_publication(*, report_path: Path,
                      publish: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
    """Call delivery from a saved envelope; this path never invokes inference."""
    envelope = json.loads(report_path.read_text(encoding="utf-8"))
    attempted_at = datetime.now(timezone.utc).isoformat()
    try:
        publish(envelope)
        state = {"status": "ok", "attempted_at": attempted_at}
    except Exception as exc:
        state = {"status": "failed", "attempted_at": attempted_at,
                 "error_class": type(exc).__name__}
        _atomic_json(report_path.with_name("publication-state.json"), state)
        return state
    _atomic_json(report_path.with_name("publication-state.json"), state)
    return state


# --- US 재무 feature 점수 수준 동등성 판정 -------------------------------------------------
# 서빙은 재무 11개를 규칙 A(det_a)로 고정한다. frozen과 셀 단위로 같지 않으므로 parity는
# `passed`가 아니라 `score_equivalent`로 적고, 아래 기준을 증거 JSON이 넘을 때만 서빙을 연다.
US_FIN_SERVING_RULE = "det_a"
US_PARITY_SCORE_EQUIVALENT = "score_equivalent"
US_PARITY_EVIDENCE_SCHEMA = "us-fin-parity-evidence.v1"
US_PARITY_CRITERIA: dict[str, float | int] = {
    "financial_determinism_diff_cells": 0,
    "non_financial_diff_cells": 0,
    "min_daily_spearman": 0.9999,
    "min_top50_overlap": 1.0,
    "min_top100_overlap": 1.0,
    "max_p99_abs_rank_shift": 1,
    "max_frozen_top100_abs_rank_shift": 1,
}
US_PARITY_MODELS = ("lightgbm", "ridge")


def us_parity_failures(evidence: dict[str, Any]) -> list[str]:
    """Re-derive pass/fail from the recorded numbers; never trust the recorded status."""
    failures: list[str] = []
    if evidence.get("schema_version") != US_PARITY_EVIDENCE_SCHEMA:
        return ["schema_version"]
    if evidence.get("criteria") != US_PARITY_CRITERIA:
        failures.append("criteria differ from code constants")
    if evidence.get("labels_read") is not False:
        failures.append("labels_read")
    c = US_PARITY_CRITERIA
    runs = (evidence.get("determinism") or {}).get("runs")
    if not isinstance(runs, list) or len(runs) < 4:
        failures.append("determinism runs missing")
    else:
        for run in runs:
            if run.get("diff_cells") != c["financial_determinism_diff_cells"] or not run.get("keys_equal"):
                failures.append(f"determinism:{run.get('name')}")
    if (evidence.get("inputs") or {}).get("bundle_matches_frozen") is not True:
        failures.append("bundle_matches_frozen")
    non_fin = evidence.get("non_financial") or {}
    if (non_fin.get("diff_cells") != c["non_financial_diff_cells"] or non_fin.get("keys_equal") is not True
            or non_fin.get("eligible_equal") is not True):
        failures.append("non_financial")
    models = evidence.get("models") or {}
    for name in US_PARITY_MODELS:
        m = models.get(name)
        if not isinstance(m, dict):
            failures.append(f"{name}:missing")
            continue
        try:
            checks = {
                "keys_equal": m["keys_equal"] is True,
                "spearman": m["min_daily_spearman"] >= c["min_daily_spearman"],
                "top50": m["min_top50_overlap"] >= c["min_top50_overlap"],
                "top100": m["min_top100_overlap"] >= c["min_top100_overlap"],
                "p99": m["p99_abs_rank_shift"] <= c["max_p99_abs_rank_shift"],
                "frozen_top100": m["frozen_top100_max_abs_rank_shift"]
                <= c["max_frozen_top100_abs_rank_shift"],
            }
        except (KeyError, TypeError):
            failures.append(f"{name}:malformed")
            continue
        failures.extend(f"{name}:{key}" for key, ok in checks.items() if not ok)
    return failures


def us_native_block_reason(native: dict[str, Any]) -> str | None:
    """Why a US native manifest must not serve, or None when it may.

    The historical gates (failed parity, diagnostic, serving_eligible false) block as before.
    A `score_equivalent` manifest additionally needs a hash-pinned evidence file whose
    rule equals the manifest's financial_selection_rule and whose numbers pass the gate.
    """
    status = native.get("raw_feature_parity_status")
    if (status == "failed" or native.get("diagnostic_only") is True
            or native.get("serving_eligible") is False):
        return "diagnostic or failed parity"
    if status != US_PARITY_SCORE_EQUIVALENT:
        return None
    if native.get("serving_eligible") is not True or native.get("diagnostic_only") is not False:
        return "score_equivalent requires serving_eligible=true and diagnostic_only=false"
    rule = native.get("financial_selection_rule")
    if rule != US_FIN_SERVING_RULE:
        return f"financial_selection_rule {rule!r} is not {US_FIN_SERVING_RULE!r}"
    evidence_path = native.get("raw_feature_parity_evidence")
    pinned = native.get("raw_feature_parity_evidence_sha256")
    if not evidence_path or not pinned:
        return "score_equivalent evidence path or sha256 is not pinned"
    path = Path(evidence_path)
    if not path.is_file() or path.is_symlink():
        return "score_equivalent evidence file is missing"
    if _sha256(path) != pinned:
        return "score_equivalent evidence sha256 does not match the manifest"
    try:
        evidence = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return "score_equivalent evidence is not valid JSON"
    if not isinstance(evidence, dict):
        return "score_equivalent evidence is not an object"
    if evidence.get("status") != US_PARITY_SCORE_EQUIVALENT:
        return "evidence status is not score_equivalent"
    if evidence.get("rule") != rule:
        return "evidence rule differs from the manifest financial_selection_rule"
    fin_hash = native.get("financial_code_hash")
    if not fin_hash or evidence.get("financial_code_hash") != fin_hash:
        return "evidence financial_code_hash differs from the manifest"
    failures = us_parity_failures(evidence)
    if failures:
        return "evidence does not meet the gate: " + ", ".join(failures[:5])
    return None
