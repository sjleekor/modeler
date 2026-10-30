import json
from pathlib import Path

import pytest

from modeler.feedback.core import sha256_bytes
from modeler.feedback_operations import rollback
from modeler.feedback_outcomes.output import write_run
from modeler.feedback_outcomes.pipeline import evaluate
from modeler.feedback_outcomes.synthetic import fixture


def runs(tmp_path):
    data = fixture()
    result = evaluate(
        data["calendar"], data["events"], data["extract"], cutoff=data["cutoff"], synthetic=True
    )
    first = write_run(tmp_path, result, inputs={"revision": "synthetic-first"})
    second = write_run(tmp_path, result, inputs={"revision": "synthetic-second"})
    root = Path(first["path"]).parents[1]
    return root, first["run_id"], second["run_id"]


def fingerprints(root):
    return {
        p.relative_to(root).as_posix(): sha256_bytes(p.read_bytes())
        for p in root.rglob("*")
        if p.is_file() and p.name != "current.json"
    }


def test_verified_restore_preserves_runs_and_repeated_restore_is_noop(tmp_path):
    root, first, second = runs(tmp_path)
    before = fingerprints(root)
    result = rollback.restore_synthetic_current(root, first, expected_current=second)
    assert result["status"] == "restored" and result["previous"]["run_id"] == second
    assert json.loads((root / "current.json").read_bytes())["run_id"] == first
    assert fingerprints(root) == before
    repeated = rollback.restore_synthetic_current(root, first, expected_current=first)
    assert repeated["status"] == "unchanged"


def test_stale_review_and_corruption_cannot_replace_current(tmp_path):
    root, first, second = runs(tmp_path)
    before = (root / "current.json").read_bytes()
    with pytest.raises(ValueError, match="current changed"):
        rollback.restore_synthetic_current(root, first, expected_current=first)
    (root / "runs" / first / "summary.md").write_bytes(b"generated corruption")
    with pytest.raises(ValueError, match="artifact hash"):
        rollback.restore_synthetic_current(root, first, expected_current=second)
    assert (root / "current.json").read_bytes() == before


def test_missing_pointer_can_be_recovered_only_when_explicitly_expected(tmp_path):
    root, first, second = runs(tmp_path)
    (root / "current.json").unlink()
    with pytest.raises(ValueError, match="current changed"):
        rollback.restore_synthetic_current(root, first, expected_current=second)
    result = rollback.restore_synthetic_current(root, first, expected_current=None)
    assert result["previous"] is None and result["current"]["run_id"] == first


def test_damaged_current_payload_does_not_prevent_healthy_target_restore(tmp_path):
    root, first, second = runs(tmp_path)
    damaged = root / "runs" / second / "summary.md"
    damaged.write_bytes(b"generated damaged current")
    result = rollback.restore_synthetic_current(root, first, expected_current=second)
    assert result["status"] == "restored"
    assert damaged.read_bytes() == b"generated damaged current"
    assert json.loads((root / "current.json").read_bytes())["run_id"] == first


def test_atomic_failure_keeps_previous_pointer_and_rerun_recovers(tmp_path, monkeypatch):
    root, first, second = runs(tmp_path)
    before = (root / "current.json").read_bytes()
    original = rollback._atomic

    def interrupted(*args):
        raise OSError("generated interrupted replace")

    monkeypatch.setattr(rollback, "_atomic", interrupted)
    with pytest.raises(OSError):
        rollback.restore_synthetic_current(root, first, expected_current=second)
    assert (root / "current.json").read_bytes() == before
    monkeypatch.setattr(rollback, "_atomic", original)
    assert (
        rollback.restore_synthetic_current(root, first, expected_current=second)["status"]
        == "restored"
    )


@pytest.mark.parametrize("kind", ["real-root", "symlink", "malformed-id"])
def test_scope_failures_never_restore(tmp_path, kind):
    root, first, second = runs(tmp_path)
    if kind == "real-root":
        root = tmp_path / "stock_data/kr/output/prediction_feedback_outcomes" / root.name
        root.mkdir(parents=True)
    elif kind == "symlink":
        target = root / "runs" / first / "summary.md"
        actual = tmp_path / "generated-other.txt"
        actual.write_bytes(target.read_bytes())
        target.unlink()
        target.symlink_to(actual)
    else:
        first = "../invalid"
    with pytest.raises(ValueError):
        rollback.restore_synthetic_current(root, first, expected_current=second)
