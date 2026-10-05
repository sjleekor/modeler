"""Small Cronicle entrypoint for date-scoped briefing stages."""
from __future__ import annotations

import argparse
import json
import signal
import sys
from datetime import date, datetime
from pathlib import Path

from .daily_coordinator import (
    SEOUL, _absolute, _config, _day_dir, _lock, _read,
    ensure_selection, infer_stage, monitor_stage, publish_stage, render_stage, select_stage,
)
from .daily_opening import prepare_opening


def _end_on_signal(signum: int, _frame: object) -> None:
    """Turn TERM/INT/HUP into an exception so ``run_group`` ends the child's whole group.

    Python's default for TERM is to die at once, which leaves the runner and the publisher
    running without this process (2026-10-06).
    """
    raise SystemExit(128 + signum)


def install_signal_handlers() -> None:
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _end_on_signal)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modeler-daily-wrapper")
    parser.add_argument("stage", choices=("select", "run", "monitor"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-date", type=date.fromisoformat)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--fixture-now", type=datetime.fromisoformat,
                        help="synthetic release tests only")
    args = parser.parse_args(argv)
    try:
        config = _config(args.config)
        release = _read(_absolute(config, "release_manifest"))
        if args.fixture_now and release.get("synthetic_fixture") is not True:
            raise ValueError("fixture clock is forbidden for a real release")
        now = args.fixture_now or datetime.now(SEOUL)
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("wrapper clock must have a timezone")
        day = args.report_date or now.astimezone(SEOUL).date()
        with _lock(_day_dir(config, day) / ".coordinator.lock"):
            if args.stage == "select":
                result = select_stage(config, day, now)
                states = {"selection": result["status"]}
                failed = result["status"] not in {"selected", "partial", "holiday"}
            elif args.stage == "run":
                # select가 안 돈 날은 run이 같은 규칙으로 직접 고릅니다(변경 5). 못 고르면
                # infer_stage가 같은 사유로 실패 report를 만들어 단위는 나옵니다(변경 6).
                try:
                    selection = ensure_selection(config, day, now)
                except Exception:
                    selection = None
                if selection is not None and selection.get("status") == "holiday":
                    states = {"opening": "holiday"}
                else:
                    opening = prepare_opening(config, day)
                    states = {"opening": opening["status"]}
                inference = infer_stage(config, day, now)
                states["inference"] = inference["status"]
                failed = inference["status"] == "inference_failed"
                if inference["status"] == "holiday":
                    states["render"] = "holiday"
                    states["publication"] = "holiday"
                elif inference.get("report_ready", inference.get("this_invocation_completed")):
                    # Even zero successful model jobs, a runner timeout or an exception produce a
                    # current-D status unit: infer_stage wrote this invocation's own report.
                    states["render"] = render_stage(config, day)["status"]
                    states["publication"] = publish_stage(config, day)["status"]
                    failed = failed or states["publication"] == "publisher_failed"
                else:
                    states["render"] = "skipped"
                    states["publication"] = "skipped"
            else:
                result = monitor_stage(config, day, attempt=args.attempt)
                states = {"monitor": result["status"]}
                failed = result["status"] not in {"verified", "publication_withheld", "holiday_skipped"}
        print(json.dumps({"report_date": day.isoformat(), "stages": states}, sort_keys=True))
        return int(failed)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    install_signal_handlers()
    raise SystemExit(main())
