"""시장·섹터 입력을 select 단계에서 고정하고, 채점 직전에 다시 확인한다. 표준 라이브러리만 쓴다.

종목 순위 입력은 D 09:30 select가 고정한다(`serving.daily_inputs`). 시장·섹터도 같은 규칙을 따른다
(계획 03 §6.1, 리뷰 7).

* 규칙: 입력이 **D 09:30 KST 이전에 끝난** snapshot만 고른다. 그 뒤에 끝난 snapshot은 그날 쓰지
  않고, 그 전의 가장 최근 snapshot을 쓴다.
* 고른 입력은 `ms-selection.json`(``market-sector-selection.v1``)에 파일마다 sha256으로 적는다.
* `score_daily`는 이 selection이 가리키는 파일만 열고, 열기 전에 sha256과 파일 목록을 다시 확인한다.
  레이크의 "최신 snapshot"을 스스로 찾지 않는다.

입력 두 갈래

* KR raw snapshot(`<kr>/raw/raw_postgres/snapshot_date=S/source=sj2_remote/`): 끝난 시각은
  export가 남기는 `_manifests/_SUCCESS.json`의 `finished_at`이다.
  `krx_index_daily`·`common_feature_observation_raw` 두 표의 table manifest sha256, 두 표의
  parquet 파일 sha256, `pg_snapshot_id`를 적는다.
* US 레이크(`<us>/derived/snapshots/<표>/snapshot_date=S/*.parquet`): `prices_daily`·`corp_actions`·
  `macro_series`·`trading_calendar`. 완료 마커가 없어 **그 snapshot 파일들의 가장 늦은 mtime**을
  끝난 시각으로 본다. 파일 sha256을 적는다. KR 피쳐도 US `macro_series`를 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SEOUL = ZoneInfo("Asia/Seoul")
SCHEMA = "market-sector-selection.v1"
CUTOFF_TIME = time(9, 30)
KR_SOURCE = "sj2_remote"
KR_TABLES = ("krx_index_daily", "common_feature_observation_raw")
US_TABLES = ("prices_daily", "corp_actions", "macro_series", "trading_calendar")
#: 시장마다 필요한 US 표. KR은 피쳐용 거시만 쓴다.
US_TABLES_FOR = {"US": US_TABLES, "KR": ("macro_series",)}
SELECTION_MODES = ("scheduled", "run_fallback")
NO_SNAPSHOT_REASON = "no_snapshot_completed_before_cutoff"
_SNAPSHOT_DIR = re.compile(r"snapshot_date=(\d{4}-\d{2}-\d{2})\Z")


class InputPinError(ValueError):
    """고를 수 있는 입력이 없거나 형식이 맞지 않는다."""


class InputChangedError(InputPinError):
    """selection이 고정한 뒤에 파일이 바뀌었거나 목록이 달라졌다."""


def input_cutoff(report_date: date) -> datetime:
    return datetime.combine(report_date, CUTOFF_TIME, SEOUL)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _aware(raw: Any, label: str) -> datetime:
    try:
        value = datetime.fromisoformat(str(raw))
    except ValueError as exc:
        raise InputPinError(f"{label}: 시각 형식이 맞지 않습니다") from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise InputPinError(f"{label}: 시각에 시간대가 없습니다")
    return value


def _records(directory: Path, *, recursive: bool) -> dict[str, dict[str, Any]]:
    """디렉터리 안 parquet 파일의 ``상대경로 -> {sha256, bytes}``. 심볼릭 링크 파일은 거부한다."""
    if not directory.is_dir():
        raise InputPinError(f"입력 디렉터리가 없습니다: {directory.name}")
    found = sorted(directory.rglob("*.parquet") if recursive else directory.glob("*.parquet"))
    out: dict[str, dict[str, Any]] = {}
    for path in found:
        if path.is_symlink() or not path.is_file():
            raise InputPinError(f"입력 파일이 일반 파일이 아닙니다: {path.name}")
        out[path.relative_to(directory).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    if not out:
        raise InputPinError(f"입력 parquet 파일이 없습니다: {directory.name}")
    return out


def _snapshot_dirs(parent: Path) -> list[tuple[str, Path]]:
    if not parent.is_dir():
        return []
    items = []
    for child in parent.iterdir():
        match = _SNAPSHOT_DIR.fullmatch(child.name)
        if match and child.is_dir() and not child.is_symlink():
            items.append((match.group(1), child))
    return sorted(items, reverse=True)


# --------------------------------------------------------------------------- KR
def kr_snapshot_dir(kr_root: Path, snapshot_date: str) -> Path:
    snapshot = f"snapshot_date={snapshot_date}"
    return kr_root / "raw" / "raw_postgres" / snapshot / f"source={KR_SOURCE}"


def kr_marker(source_dir: Path) -> dict[str, Any]:
    """`_SUCCESS.json`을 읽는다(파일 해시는 하지 않는다). 끝난 시각과 표 목록을 확인한다."""
    marker = source_dir / "_manifests" / "_SUCCESS.json"
    if marker.is_symlink() or not marker.is_file():
        raise InputPinError("export 완료 marker(_SUCCESS.json)가 없습니다")
    try:
        body = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputPinError("export 완료 marker를 읽을 수 없습니다") from exc
    if not isinstance(body, dict) or not isinstance(body.get("tables"), dict):
        raise InputPinError("export 완료 marker 형식이 맞지 않습니다")
    missing = [t for t in KR_TABLES if t not in body["tables"]]
    if missing:
        raise InputPinError(f"export 완료 marker에 표가 없습니다: {missing[0]}")
    return {
        "finished_at": _aware(body.get("finished_at"), "export finished_at"),
        "pg_snapshot_id": body.get("pg_snapshot_id"),
        "marker_path": marker,
    }


def describe_kr_snapshot(kr_root: Path, snapshot_date: str) -> dict[str, Any]:
    """KR raw snapshot 하나를 selection 기록으로 만든다(파일 sha256 포함)."""
    source_dir = kr_snapshot_dir(kr_root, snapshot_date)
    marker = kr_marker(source_dir)
    record: dict[str, Any] = {
        "snapshot_date": snapshot_date,
        "source": KR_SOURCE,
        "export_finished_at": marker["finished_at"].isoformat(),
        "pg_snapshot_id": marker["pg_snapshot_id"],
        "success_marker_sha256": sha256_file(marker["marker_path"]),
        "tables": {},
    }
    for table in KR_TABLES:
        manifest = source_dir / "_manifests" / "table_manifests" / f"{table}.json"
        if manifest.is_symlink() or not manifest.is_file():
            raise InputPinError(f"{table}: table manifest가 없습니다")
        try:
            body = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise InputPinError(f"{table}: table manifest를 읽을 수 없습니다") from exc
        source = body.get("source") if isinstance(body, dict) else None
        table_pg = source.get("pg_snapshot_id") if isinstance(source, dict) else None
        if marker["pg_snapshot_id"] and table_pg and table_pg != marker["pg_snapshot_id"]:
            raise InputPinError(f"{table}: pg_snapshot_id가 export marker와 다릅니다")
        record["tables"][table] = {
            "manifest_sha256": sha256_file(manifest),
            "pg_snapshot_id": table_pg,
            "files": _records(source_dir / table, recursive=True),
        }
    return record


def select_kr(kr_root: Path, cutoff: datetime) -> dict[str, Any]:
    """D 09:30 이전에 끝난 가장 최근 KR raw snapshot. 없으면 ``unavailable``."""
    parent = kr_root / "raw" / "raw_postgres"
    skipped: list[dict[str, Any]] = []
    for snapshot_date, path in _snapshot_dirs(parent):
        source_dir = path / f"source={KR_SOURCE}"
        if not source_dir.is_dir():
            continue
        try:
            marker = kr_marker(source_dir)
        except InputPinError as exc:
            skipped.append(
                {"snapshot_date": snapshot_date, "reason": "incomplete", "detail": str(exc)})
            continue
        if marker["finished_at"] > cutoff:
            skipped.append({"snapshot_date": snapshot_date, "reason": "completed_after_cutoff",
                            "completed_at": marker["finished_at"].isoformat()})
            continue
        try:
            record = describe_kr_snapshot(kr_root, snapshot_date)
        except InputPinError as exc:
            skipped.append(
                {"snapshot_date": snapshot_date, "reason": "invalid", "detail": str(exc)})
            continue
        return {"status": "selected", **record, "skipped": skipped}
    return {"status": "unavailable", "reason": NO_SNAPSHOT_REASON, "skipped": skipped}


# --------------------------------------------------------------------------- US
def us_table_dir(us_root: Path, table: str, snapshot_date: str) -> Path:
    return us_root / "derived" / "snapshots" / table / f"snapshot_date={snapshot_date}"


def _completed_at(directory: Path) -> datetime:
    """snapshot 파일들의 가장 늦은 mtime. 레이크에는 완료 marker가 없다."""
    mtimes = [p.stat().st_mtime for p in directory.glob("*.parquet") if p.is_file()]
    if not mtimes:
        raise InputPinError("입력 parquet 파일이 없습니다")
    return datetime.fromtimestamp(max(mtimes), tz=SEOUL)


def describe_us_table(us_root: Path, table: str, snapshot_date: str) -> dict[str, Any]:
    directory = us_table_dir(us_root, table, snapshot_date)
    files = _records(directory, recursive=False)
    return {
        "snapshot_date": snapshot_date,
        "completed_at": _completed_at(directory).isoformat(),
        "completed_at_basis": "latest_file_mtime",
        "files": files,
    }


def select_us_table(us_root: Path, table: str, cutoff: datetime) -> dict[str, Any]:
    skipped: list[dict[str, Any]] = []
    parent = us_root / "derived" / "snapshots" / table
    for snapshot_date, path in _snapshot_dirs(parent):
        try:
            completed = _completed_at(path)
        except InputPinError:
            skipped.append({"snapshot_date": snapshot_date, "reason": "no_parquet"})
            continue
        if completed > cutoff:
            skipped.append({"snapshot_date": snapshot_date, "reason": "completed_after_cutoff",
                            "completed_at": completed.isoformat()})
            continue
        try:
            record = describe_us_table(us_root, table, snapshot_date)
        except InputPinError as exc:
            skipped.append(
                {"snapshot_date": snapshot_date, "reason": "invalid", "detail": str(exc)})
            continue
        return {"status": "selected", **record, "skipped": skipped}
    return {"status": "unavailable", "reason": NO_SNAPSHOT_REASON, "skipped": skipped}


# --------------------------------------------------------------------------- selection
def select_market_sector_inputs(
    *,
    report_date: date,
    selected_at: datetime,
    mode: str,
    kr_root: Path,
    us_root: Path,
    limits: dict[str, date | None],
    bundle_path: Path,
    bundle_sha256: str,
) -> dict[str, Any]:
    """시장·섹터 입력 selection(``market-sector-selection.v1``). 파일을 쓰지는 않는다.

    ``limits``: 시장별 기준 상한 날짜(KR은 직전 KR 세션 K, US는 직전 완료 US 세션). 계산 달력의
    세션으로 맞추는 일은 채점 쪽이 한다. ``None``이면 그 시장은 채점하지 못한다.
    """
    if mode not in SELECTION_MODES:
        raise InputPinError("selection mode must be scheduled or run_fallback")
    cutoff = input_cutoff(report_date)
    if selected_at.tzinfo is None or selected_at < cutoff:
        raise InputPinError("selection은 D 09:30 KST 이후여야 합니다")
    tables = {t: select_us_table(us_root, t, cutoff) for t in US_TABLES}
    present = [t for t, rec in tables.items() if rec["status"] == "selected"]
    return {
        "schema_version": SCHEMA,
        "report_date": report_date.isoformat(),
        "selected_at": selected_at.isoformat(),
        "selection_mode": mode,
        "input_cutoff": cutoff.isoformat(),
        "bundle": {"path": str(bundle_path), "sha256": bundle_sha256},
        "roots": {"kr": str(kr_root), "us": str(us_root)},
        "limits": {m: (v.isoformat() if v else None) for m, v in sorted(limits.items())},
        "kr": select_kr(kr_root, cutoff),
        "us": {
            "status": "selected" if len(present) == len(US_TABLES)
            else ("partial" if present else "unavailable"),
            "tables": tables,
        },
    }


def market_status(selection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """시장마다 채점할 수 있는지(``selected``/``unavailable``)와 사유. state 요약용."""
    out: dict[str, dict[str, Any]] = {}
    for market in ("KR", "US"):
        reason = None
        if not selection["limits"].get(market):
            reason = "limit_session_unknown"
        elif market == "KR" and selection["kr"].get("status") != "selected":
            reason = selection["kr"].get("reason") or "kr_snapshot_unavailable"
        else:
            for table in US_TABLES_FOR[market]:
                rec = selection["us"]["tables"].get(table, {})
                if rec.get("status") != "selected":
                    reason = f"us_{table}_unavailable"
                    break
        block: dict[str, Any] = {"status": "selected" if reason is None else "unavailable"}
        if reason:
            block["reason"] = reason
        else:
            block["limit_session"] = selection["limits"][market]
        out[market] = block
    return out


# --------------------------------------------------------------------------- 채점 직전 확인
def _same_files(label: str, recorded: dict[str, Any], live: dict[str, Any]) -> None:
    if set(recorded) != set(live):
        raise InputChangedError(f"{label}: 파일 목록이 selection 뒤에 달라졌습니다")
    for name, rec in recorded.items():
        if live[name]["sha256"] != rec["sha256"]:
            raise InputChangedError(f"{label}: 파일이 selection 뒤에 바뀌었습니다 ({name})")


def verify_pins(selection: dict[str, Any], market: str) -> None:
    """``market`` 채점에 필요한 입력이 selection이 고정한 그대로인지 확인한다. 아니면 예외.

    sha256을 다시 계산한다. 기록한 완료 시각이 cutoff 이후인 입력도 거부한다.
    """
    if selection.get("schema_version") != SCHEMA:
        raise InputPinError("selection 스키마가 맞지 않습니다")
    if market not in US_TABLES_FOR:
        raise InputPinError("market must be KR or US")
    cutoff = _aware(selection["input_cutoff"], "input_cutoff")
    if cutoff != input_cutoff(date.fromisoformat(selection["report_date"])):
        raise InputPinError("input_cutoff이 D 09:30 KST가 아닙니다")
    roots = selection["roots"]
    if market == "KR":
        kr = selection["kr"]
        if kr.get("status") != "selected":
            raise InputPinError("KR snapshot이 selection에 없습니다")
        if _aware(kr["export_finished_at"], "export_finished_at") > cutoff:
            raise InputPinError("KR export가 D 09:30 이후에 끝났습니다")
        live = describe_kr_snapshot(Path(roots["kr"]), kr["snapshot_date"])
        if live["success_marker_sha256"] != kr["success_marker_sha256"]:
            raise InputChangedError("KR export 완료 marker가 selection 뒤에 바뀌었습니다")
        for table in KR_TABLES:
            if live["tables"][table]["manifest_sha256"] != kr["tables"][table]["manifest_sha256"]:
                raise InputChangedError(f"KR {table}: table manifest가 selection 뒤에 바뀌었습니다")
            _same_files(f"KR {table}", kr["tables"][table]["files"], live["tables"][table]["files"])
    for table in US_TABLES_FOR[market]:
        rec = selection["us"]["tables"].get(table, {})
        if rec.get("status") != "selected":
            raise InputPinError(f"US {table} snapshot이 selection에 없습니다")
        if _aware(rec["completed_at"], "completed_at") > cutoff:
            raise InputPinError(f"US {table}가 D 09:30 이후에 끝났습니다")
        live_us = describe_us_table(Path(roots["us"]), table, rec["snapshot_date"])
        _same_files(f"US {table}", rec["files"], live_us["files"])


def atomic_write_json(path: Path, body: dict[str, Any]) -> str:
    """임시 파일에 쓰고 rename한다. 쓴 바이트의 sha256을 돌려준다."""
    raw = (json.dumps(body, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()
