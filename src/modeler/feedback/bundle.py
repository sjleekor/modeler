"""Pinned calendar metadata and honest source inventories for feedback bundles."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from datetime import date, time
from pathlib import Path

from .core import FeedbackError, canonical_bytes, require_hash, sha256_bytes, source_path

MAX_BUNDLE_METADATA_BYTES = 1024 * 1024
_INTEGRATION_FILES = (
    "src/modeler/__init__.py",
    "src/modeler/reporting/__init__.py",
    "src/modeler/reporting/markdown.py",
    "src/modeler/reporting/security_names.py",
    "deploy/feedback/feedback-stage.sh",
    "deploy/reports/publish_reports.py",
    "deploy/reports/validate_reports.py",
)


def _metadata_bytes(path):
    path = Path(path).absolute()
    source_path(path, root=path.parent)
    if any(parent.is_symlink() for parent in path.parents):
        raise FeedbackError("bundle metadata cannot traverse symlinks")
    if not path.is_file() or path.stat().st_size > MAX_BUNDLE_METADATA_BYTES:
        raise FeedbackError("bundle metadata is missing or oversized")
    with path.open("rb") as stream:
        raw = stream.read(MAX_BUNDLE_METADATA_BYTES + 1)
    if len(raw) > MAX_BUNDLE_METADATA_BYTES:
        raise FeedbackError("bundle metadata is oversized")
    return raw


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise FeedbackError("bundle metadata contains duplicate fields")
        value[key] = item
    return value


def _metadata(path, digest):
    require_hash(digest)
    raw = _metadata_bytes(path)
    if sha256_bytes(raw) != digest:
        raise FeedbackError("bundle metadata pin mismatch")
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise FeedbackError("bundle metadata must be an object")
    return value


def _day(value):
    if not isinstance(value, str):
        raise FeedbackError("calendar dates must be canonical strings")
    try:
        day = date.fromisoformat(value)
    except ValueError as exc:
        raise FeedbackError("calendar date is invalid") from exc
    if day.isoformat() != value:
        raise FeedbackError("calendar dates must be canonical strings")
    return day


def _days(value):
    if not isinstance(value, list):
        raise FeedbackError("calendar date collection must be a list")
    days = [_day(item) for item in value]
    if len(days) != len(set(days)):
        raise FeedbackError("calendar dates must be unique")
    return set(days)


def _hours(open_at, close_at):
    hours = []
    for value in (open_at, close_at):
        if not isinstance(value, str) or not re.fullmatch(r"\d{2}:\d{2}(?::\d{2})?", value):
            raise FeedbackError("calendar hours must be explicit local wall times")
        try:
            hours.append(time.fromisoformat(value))
        except ValueError as exc:
            raise FeedbackError("calendar hours are invalid") from exc
    if hours[0] >= hours[1]:
        raise FeedbackError("calendar hours must be ordered local wall times")


def pinned_sessions(calendar_file, sha256):
    """Return only confirmed KR sessions from the exact pinned calendar bytes.

    This mirrors the prepared-input calendar contract without importing serving,
    models, price readers, or any optional data dependency.
    """
    calendar = _metadata(calendar_file, sha256)
    if calendar.get("market") != "KR" or calendar.get("timezone") != "Asia/Seoul":
        raise FeedbackError("feedback requires a KR calendar in Asia/Seoul")
    start, end = _day(calendar.get("coverage_start")), _day(calendar.get("coverage_end"))
    if start > end:
        raise FeedbackError("calendar coverage is reversed")
    sessions = _days(calendar.get("sessions"))
    if any(not start <= day <= end for day in sessions):
        raise FeedbackError("calendar sessions fall outside coverage")
    _hours(calendar.get("default_open_at"), calendar.get("default_close_at"))
    unconfirmed = _days(calendar.get("unconfirmed_dates", []))
    if not unconfirmed <= sessions:
        raise FeedbackError("unconfirmed dates must belong to declared sessions")
    override_rows = calendar.get("overrides", [])
    if not isinstance(override_rows, list):
        raise FeedbackError("calendar overrides must be a list")
    seen = set()
    for row in override_rows:
        if not isinstance(row, dict):
            raise FeedbackError("calendar override must be an object")
        day = _day(row.get("date"))
        if day not in sessions or day in seen:
            raise FeedbackError("calendar overrides must name unique declared sessions")
        seen.add(day)
        _hours(row.get("open_at"), row.get("close_at"))
        confirmed = row.get("confirmed", False)
        if type(confirmed) is not bool:
            raise FeedbackError("calendar override confirmation must be boolean")
        if confirmed and day in unconfirmed:
            raise FeedbackError("confirmed override conflicts with unconfirmed dates")
        if not confirmed:
            unconfirmed.add(day)
    return [day.isoformat() for day in sorted(sessions - unconfirmed)]


def source_inventory(code_root):
    """List the narrow source closure; never recurse through a repo, venv or .git."""
    root = Path(code_root).absolute()
    feedback = root / "src/modeler/feedback"
    if any(parent.is_symlink() for parent in (feedback, *feedback.parents)):
        raise FeedbackError("bundle source cannot traverse symlinks")
    files = set(_INTEGRATION_FILES)
    files.update(path.relative_to(root).as_posix() for path in feedback.glob("*.py"))
    if "src/modeler/feedback/runtime.py" not in files:
        raise FeedbackError("bundle source is missing the feedback runtime")
    return sorted(files)


def _code_file(name):
    if not isinstance(name, str) or (
        name not in _INTEGRATION_FILES
        and not re.fullmatch(r"src/modeler/feedback/[a-z_][a-z_0-9]*\.py", name)
    ):
        raise FeedbackError("bundle source is outside the explicit code allowlist")
    return name


def _git(root, *args):
    result = subprocess.run(
        ["git", "--no-pager", "--literal-pathspecs", "-C", str(root), *args],
        capture_output=True,
        check=False,
        timeout=10,
    )
    return result.returncode, result.stdout


def _imports(raw):
    names = set()
    for node in ast.walk(ast.parse(raw)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return sorted(names)


def build_manifest(code_root, *, expected_commit, files=None):
    """Hash explicit source bytes and prove their relationship to a local Git HEAD.

    A dirty or untracked byte is recorded as such rather than described as the
    committed source. This does not copy files, create an archive, run publisher,
    or claim that optional shadow/name generation paths use only stdlib.
    """
    root = Path(code_root).absolute()
    expected_commit = require_hash(expected_commit, length=40)
    status, head = _git(root, "rev-parse", "--verify", "HEAD")
    if status or head.decode("ascii").strip() != expected_commit:
        raise FeedbackError("bundle source commit cannot be verified")
    names = source_inventory(root) if files is None else [_code_file(name) for name in files]
    if not names or len(names) != len(set(names)):
        raise FeedbackError("bundle source inventory must be nonempty and unique")
    rows = {}
    for name in sorted(names):
        _code_file(name)
        raw = _metadata_bytes(root / name)
        digest = sha256_bytes(raw)
        status, committed = _git(root, "show", f"{expected_commit}:{name}")
        committed_digest = None if status else sha256_bytes(committed)
        rows[name] = {
            "sha256": digest,
            "bytes": len(raw),
            "committed_sha256": committed_digest,
            "matches_commit": digest == committed_digest,
            "imports": _imports(raw) if name.endswith(".py") else [],
        }
    changed = [name for name, row in rows.items() if not row["matches_commit"]]
    return {
        "schema": "prediction-feedback-bundle-source.v1",
        "modeler_commit": expected_commit,
        "source_state": "working_tree" if changed else "commit",
        "changed_or_untracked_files": changed,
        "source_inventory_sha256": sha256_bytes(canonical_bytes(rows)),
        "files": rows,
        "dependency_notes": {
            "stdlib_entrypoints": [
                "modeler.feedback.runtime (briefing only)",
                "modeler.feedback.publisher_capture",
            ],
            "external_commands": ["git", "curl"],
            "publisher_dependencies": [
                "deploy/reports/validate_reports.py",
                "src/modeler/reporting/markdown.py",
                "src/modeler/reporting/security_names.py",
            ],
            "optional_external_modules": {
                "feedback.shadow": ["polars", "modeler.models._03_kr_next.shadow"],
                "reporting.security_names (name input generation)": ["polars"],
            },
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar-file", type=Path)
    parser.add_argument("--calendar-sha256")
    parser.add_argument("--code-root", type=Path)
    parser.add_argument("--expected-commit")
    args = parser.parse_args(argv)
    try:
        if args.code_root is not None:
            if args.calendar_file is not None or args.calendar_sha256 is not None:
                raise FeedbackError("select either source inventory or calendar verification")
            result = build_manifest(args.code_root, expected_commit=args.expected_commit)
        else:
            result = {"sessions": pinned_sessions(args.calendar_file, args.calendar_sha256)}
        print(canonical_bytes(result).decode("utf-8"))
        return 0
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        print("feedback-bundle: metadata verification failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
