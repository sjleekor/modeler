"""Verify and restore only a synthetic outcome pointer; never operate Cronicle.

This rehearsal preserves immutable runs and uses the outcome writer's lock.
Production rollback remains a separately reviewed deployment operation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from modeler.feedback.core import canonical_bytes, require_hash, sha256_bytes
from modeler.feedback.store import _atomic, _writer, check_output_root
from modeler.feedback_operations.credential_launcher import _read_regular
from modeler.feedback_outcomes.access import explicit_path
from modeler.feedback_outcomes.contracts import RULE_ID, OutcomeError, rule_pin

FILES = frozenset({"rows.parquet", "cohorts.parquet", "summary.md", "decisions.json"})
MAX_BYTES = 64 * 1024 * 1024


def _root(value):
    root = explicit_path(value)
    if root.name != RULE_ID or root.parent.name != "synthetic" or "stock_data" in root.parts:
        raise OutcomeError("rollback rehearsal requires a synthetic outcome root")
    if not root.is_dir():
        raise OutcomeError("synthetic outcome root is missing")
    return check_output_root(root, [])


def _json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise OutcomeError("duplicate rollback metadata field")
            value[key] = item
        return value

    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise OutcomeError("invalid rollback metadata")
    return value


def _current(root):
    path = root / "current.json"
    if not path.exists():
        return None
    current = _json(_read_regular(path, 1024 * 1024))
    if set(current) != {"run_id", "manifest_sha256"}:
        raise OutcomeError("invalid current pointer")
    require_hash(current["run_id"])
    if current["manifest_sha256"] != current["run_id"]:
        raise OutcomeError("current pointer identity mismatch")
    return current


def _verified_run(root, run_id):
    target = root / "runs" / require_hash(run_id)
    raw = _read_regular(target / "run_manifest.json", 4 * 1024 * 1024)
    if sha256_bytes(raw) != run_id:
        raise OutcomeError("rollback manifest hash mismatch")
    manifest = _json(raw)
    if (
        manifest.get("schema") != "kr-v1-outcome-run.v1"
        or manifest.get("synthetic_fixture") is not True
        or manifest.get("rule") != rule_pin()
    ):
        raise OutcomeError("rollback target must be a frozen-rule synthetic run")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != FILES:
        raise OutcomeError("rollback target inventory mismatch")
    for name, digest in files.items():
        if sha256_bytes(_read_regular(target / name, MAX_BYTES)) != require_hash(digest):
            raise OutcomeError("rollback artifact hash mismatch")
    return {"run_id": run_id, "manifest_sha256": run_id}


def restore_synthetic_current(root, run_id, *, expected_current):
    """Compare current, fully verify the target, then atomically repoint current.

    expected_current=None explicitly requires a missing pointer; it never means
    'accept anything'. The returned previous pointer can be saved by the caller.
    No source bytes, prior runs, configs, credentials or events are changed.
    """
    root = _root(root)
    if expected_current is not None:
        require_hash(expected_current)
    with _writer(root):
        previous = _current(root)
        observed = previous["run_id"] if previous is not None else None
        if observed != expected_current:
            raise OutcomeError("current changed since rollback review")
        # A damaged current run is a reason to restore a known healthy target.
        # Do not open its payload or require it to be healthy to permit recovery.
        target = _verified_run(root, run_id)
        if previous != target:
            _atomic(root / "current.json", canonical_bytes(target))
        return {
            "status": "unchanged" if previous == target else "restored",
            "synthetic_fixture": True,
            "previous": previous,
            "current": target,
            "code_sha256": sha256_bytes(Path(__file__).read_bytes()),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--target-run", required=True)
    expected = parser.add_mutually_exclusive_group(required=True)
    expected.add_argument("--expected-current")
    expected.add_argument("--expect-missing", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = restore_synthetic_current(
            args.synthetic_root, args.target_run, expected_current=args.expected_current
        )
        print(canonical_bytes(result).decode())
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        print("feedback-rollback: rehearsal verification failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
