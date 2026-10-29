"""Run generated fixtures, or explicitly gated future KR endpoint read plans."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from modeler.feedback.core import canonical_bytes, sha256_bytes

from .access import load_bound_input, load_real_plan
from .contracts import OutcomeError, stamp
from .output import write_run
from .pipeline import evaluate, selected_sessions
from .selection import Calendar, select_cohorts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("synthetic", "real"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--read-plan", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            if args.read_plan is not None:
                raise OutcomeError(
                    "synthetic mode generates its own data and accepts no real input path"
                )
            from .synthetic import fixture

            data = fixture()
            result = evaluate(
                data["calendar"],
                data["events"],
                data["extract"],
                cutoff=data["cutoff"],
                synthetic=True,
            )
            inputs = {
                "source": data["data_origin"],
                "synthetic_fixture": True,
                "fixture_sha256": sha256_bytes(canonical_bytes(data)),
            }
            output = write_run(args.output_root, result, inputs=inputs)
        else:
            permit = load_real_plan(args.read_plan, now=datetime.now(UTC))
            source_roots = [Path(permit.request["source_root"])]
            calendar, cal_ref = load_bound_input(permit, "calendar", now=datetime.now(UTC))
            publications, pub_ref = load_bound_input(permit, "publications", now=datetime.now(UTC))
            from .metadata import normalize_publications

            cutoff = permit.request["cutoff"]
            events = normalize_publications(publications, observed_at=cutoff, synthetic=False)
            # The existing producer has no machine-readable withdrawal schema.
            # Do not read actual prices while that source contract is unresolved.
            if any(e.get("explicit_withdrawal_adapter") != "source_bound_v1" for e in events):
                raise OutcomeError("source-bound withdrawal evidence adapter is not ready")
            calendar["calendar_sha256"] = cal_ref["sha256"]
            selected = select_cohorts(events, Calendar.from_manifest(calendar))
            sessions = selected_sessions(selected["cohorts"], cutoff)
            extract, extract_ref = None, None
            if sessions:
                scope = permit.request["inputs"]["extract"]
                if scope["min_session"] != sessions[0] or scope["max_session"] != sessions[-1]:
                    raise OutcomeError("price read scope differs from mature E/X sessions")
                if any(
                    stamp(c["exit_close_at"]) >= stamp(cutoff)
                    for c in selected["cohorts"]
                    if c.get("exit_session") in sessions
                ):
                    raise OutcomeError("price scope includes an unfinished exit session")
                extract, extract_ref = load_bound_input(permit, "extract", now=datetime.now(UTC))
            result = evaluate(calendar, events, extract, cutoff=cutoff, now=datetime.now(UTC))
            inputs = {
                "request_sha256": permit.request_sha256,
                "read_plan": permit.request,
                "calendar": cal_ref,
                "publications": pub_ref,
                "extract": extract_ref,
            }
            output = write_run(args.output_root, result, inputs=inputs, source_roots=source_roots)
        print(json.dumps(output, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        print("feedback-outcomes: input or access contract failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
