"""CLI entry point for scheduled inference and publication-only retries."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

from .orchestration import (
    InferenceJob, code_inventory_sha256, retry_publication, run_daily, selection_summary,
)
from .calendars import SessionCalendar

SEOUL = ZoneInfo("Asia/Seoul")


def _callable(reference: str) -> Callable[..., Any]:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("entrypoint must use module:function syntax")
    function: Any = importlib.import_module(module_name)
    for part in attribute.split("."):
        function = getattr(function, part)
    if not callable(function):
        raise ValueError("entrypoint is not callable")
    return function


def _load_jobs(config_path: Path, prepared_root: Path) -> list[InferenceJob]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    jobs = []
    for raw in config.get("jobs", []):
        infer = _callable(raw["entrypoint"])
        input_path = Path(raw["prepared_input"])
        manifest_path = Path(raw["prepared_manifest"])
        native_manifest_path = Path(raw["native_manifest"])
        if not input_path.is_absolute():
            input_path = prepared_root / input_path
        if not manifest_path.is_absolute():
            manifest_path = prepared_root / manifest_path
        if not native_manifest_path.is_absolute():
            native_manifest_path = prepared_root / native_manifest_path
        code_files = []
        for item in raw.get("code_files", []):
            path = Path(item["path"])
            if not path.is_absolute():
                path = config_path.parent / path
            code_files.append((path, item["sha256"]))
        code_path = Path(raw["code_path"])
        if not code_path.is_absolute():
            code_path = config_path.parent / code_path
        code_hash = raw["code_sha256"]
        loaded_source = inspect.getsourcefile(infer)
        if loaded_source is None or Path(loaded_source).resolve(strict=True) != code_path.resolve(strict=True):
            raise ValueError("loaded inference callable is not the pinned release source file")
        if code_files and code_inventory_sha256(code_files) != code_hash:
            raise ValueError("configured runtime source inventory hash does not match")
        jobs.append(InferenceJob(
            market=raw["market"], model_id=raw["model_id"], model_version=raw["model_version"],
            prepared_input=input_path, input_sha256=raw["input_sha256"],
            prepared_manifest=manifest_path, prepared_manifest_sha256=raw["prepared_manifest_sha256"],
            native_manifest=native_manifest_path, native_manifest_sha256=raw["native_manifest_sha256"],
            bundle_path=Path(raw["bundle_path"]), bundle_sha256=raw["bundle_sha256"],
            code_path=code_path, code_sha256=code_hash,
            code_files=tuple(code_files), infer=infer))
    return jobs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="modeler-serving")
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser("infer", help="run pinned daily model inference")
    infer.add_argument("--report-date", required=True)
    infer.add_argument("--prepared-root", type=Path, required=True)
    infer.add_argument("--run-root", type=Path, required=True)
    infer.add_argument("--jobs-config", type=Path, required=True)
    infer.add_argument("--opening-json", type=Path,
                       help="D-specific opening observations with explicit publication evidence")
    infer.add_argument("--kr-calendar-json", type=Path,
                       help="pinned KR calendar independent of KR model-input availability")
    infer.add_argument("--kr-calendar-sha256",
                       help="SHA-256 of the pinned KR calendar file")
    infer.add_argument("--selection-state", type=Path,
                       help="the D selection-state.json; its per-market select status and reason "
                            "go into the envelope's selection_summary")
    infer.add_argument("--selection-state-sha256",
                       help="SHA-256 of the selection-state.json file")
    infer.add_argument("--historical-replay", action="store_true")
    infer.add_argument("--fixture-mode", action="store_true")
    infer.add_argument("--invocation-id", help="date-scoped coordinator invocation nonce")
    retry = sub.add_parser("retry-publication", help="retry delivery from a saved date report")
    retry.add_argument("--report", type=Path, required=True)
    retry.add_argument("--publisher", required=True, help="module:function consuming a saved report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "retry-publication":
            state = retry_publication(report_path=args.report, publish=_callable(args.publisher))
            print(json.dumps(state, sort_keys=True))
            return 0 if state["status"] == "ok" else 1
        report_day = date.fromisoformat(args.report_date)
        decision_at = datetime.combine(report_day, time(10), SEOUL)
        jobs = _load_jobs(args.jobs_config, args.prepared_root)
        opening = (json.loads(args.opening_json.read_text(encoding="utf-8"))
                   if args.opening_json else None)
        opening_calendar = None
        if args.kr_calendar_json or args.kr_calendar_sha256:
            if not args.kr_calendar_json or not args.kr_calendar_sha256:
                raise ValueError("KR calendar path and SHA-256 must both be present")
            if args.kr_calendar_json.is_symlink() or hashlib.sha256(
                    args.kr_calendar_json.read_bytes()).hexdigest() != args.kr_calendar_sha256:
                raise ValueError("pinned KR calendar hash mismatch")
            opening_calendar = SessionCalendar.from_manifest(json.loads(
                args.kr_calendar_json.read_text(encoding="utf-8")))
        summary = None
        if args.selection_state or args.selection_state_sha256:
            if not args.selection_state or not args.selection_state_sha256:
                raise ValueError("selection state path and SHA-256 must both be present")
            if args.selection_state.is_symlink() or hashlib.sha256(
                    args.selection_state.read_bytes()).hexdigest() != args.selection_state_sha256:
                raise ValueError("pinned selection state hash mismatch")
            summary = selection_summary(
                json.loads(args.selection_state.read_text(encoding="utf-8")))
        envelope = run_daily(report_date=report_day, decision_at=decision_at,
            prepared_root=args.prepared_root, run_root=args.run_root, jobs=jobs,
            opening=opening, opening_calendar=opening_calendar,
            historical_replay=args.historical_replay, fixture_mode=args.fixture_mode,
            invocation_id=args.invocation_id, selection_summary=summary)
        print(json.dumps({"report_date": report_day.isoformat(), "status": envelope["status"],
                          "market_count": len(envelope["markets"]),
                          "failure_count": len(envelope["failures"]),
                          "synthetic_fixture": envelope["synthetic_fixture"]}, sort_keys=True))
        return 0 if envelope["status"] in {"ok", "partial"} else 1
    except Exception as exc:
        # Exception text can expose local paths or input details; report only its class.
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
