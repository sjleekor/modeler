"""Child process of the daily ``macro_context`` stage: one ``macro-<D>.json`` from the D selection.

Plan: ``my/milestones/common/20261031_briefing_macro_context`` (§5.5 계산과 게시 경로, §9 M2-c).

The coordinator starts this module in its own process group under a time limit.  It reads the D
market-sector selection document (``market-sector-selection.v1``, ``ms-selection.json``, pinned at
09:30 select) whose SHA-256 the coordinator hands over,
calls ``modeler.scores.macro_context.build_document`` and writes ``document_bytes(body)`` to
``--output`` in one rename.  The section is display-only: no model, score, rank or report byte is
read or changed here.  Any failure ends with a nonzero exit and no output; the coordinator then
writes the ``failed`` document itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path


def _install(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".macro-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def _run(args: argparse.Namespace) -> int:
    selection_path = Path(args.selection)
    if selection_path.is_symlink() or not selection_path.is_file():
        print(json.dumps({"status": "failed", "reason": "selection_missing"}))
        return 2
    raw = selection_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.selection_sha256:
        print(json.dumps({"status": "failed", "reason": "selection_changed"}))
        return 2
    selection = json.loads(raw)
    if not isinstance(selection, dict):
        print(json.dumps({"status": "failed", "reason": "selection_invalid"}))
        return 2
    from modeler.scores.macro_context import (
        build_document,
        document_bytes,
    )  # the stage's only import

    body = build_document(
        selection,
        unit=args.report_date,
        invocation_id=args.invocation_id,
        release=args.release,
        selection_sha256=args.selection_sha256,
    )
    if not isinstance(body, dict):
        raise TypeError("macro document must be a JSON object")
    _install(Path(args.output), document_bytes(body))
    print(json.dumps({"status": body.get("state")}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="modeler-macro-context")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="compute macro-D.json into --output")
    run.add_argument("--report-date", required=True)
    run.add_argument("--selection", type=Path, required=True)
    run.add_argument("--selection-sha256", required=True)
    run.add_argument("--invocation-id", required=True)
    run.add_argument("--release", required=True)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.report_date):
        return 2
    try:
        return _run(args)
    except Exception as exc:  # only the class goes out; paths may be in the message
        print(json.dumps({"status": "failed", "error_class": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
