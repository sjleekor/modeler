import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from modeler.feedback.core import canonical_bytes, sha256_bytes
from modeler.feedback_outcomes import access
from modeler.feedback_outcomes.contracts import OutcomeError, require_real_review, rule_pin

NOW = datetime(2026, 10, 30, 0, tzinfo=UTC)
REVIEW = datetime(2026, 12, 4, 5, tzinfo=UTC)


@pytest.mark.parametrize(
    "path", ["/absent/US/labels/eval.json", "/absent/kr/forward/checks/x.json", None]
)
def test_before_review_rejects_before_any_file_open(monkeypatch, path):
    monkeypatch.setattr(Path, "open", lambda *a, **kw: pytest.fail("opened a sealed input"))
    with pytest.raises(OutcomeError, match="registered Friday"):
        access.load_real_plan(path, now=NOW)


@pytest.mark.parametrize(
    "clock", ["2026-12-04T12:59:59+09:00", "2026-12-05T14:00:00+09:00", "2027-01-01T12:00:00+09:00"]
)
def test_registered_review_time_is_required(clock):
    with pytest.raises(OutcomeError):
        require_real_review(datetime.fromisoformat(clock))


def plan_case(tmp_path):
    root = tmp_path / "explicit-kr-endpoints"
    root.mkdir()
    inputs = {}
    for role in ("calendar", "publications", "extract"):
        path = root / (role + ".json")
        path.write_text(
            json.dumps(
                {"synthetic_fixture": False, "marker": "invented policy test, not real data"}
            )
        )
        inputs[role] = {"role": role, "path": str(path), "sha256": sha256_bytes(path.read_bytes())}
    inputs["extract"].update(min_session="2026-11-04", max_session="2026-12-02")
    pin = rule_pin()
    plan = {
        **{
            k: pin[k]
            for k in (
                "rule_id",
                "rule_sha256",
                "model_id",
                "model_version",
                "report_start",
                "report_end",
            )
        },
        "schema": access.READ_PLAN_SCHEMA,
        "market": "KR",
        "bundle_sha256": pin["bundle_sha256"],
        "synthetic_fixture": False,
        "source_scope": "explicit-kr-v1-endpoints",
        "cutoff": "2026-12-04T13:00:00+09:00",
        "max_session": "2026-12-02",
        "source_root": str(root),
        "inputs": inputs,
        "adapter_code_sha256": sha256_bytes(
            Path(access.__file__).with_name("prices.py").read_bytes()
        ),
    }
    path = tmp_path / "test-read-plan.json"
    path.write_text(json.dumps(plan))
    return path, plan


@pytest.mark.parametrize(
    "change", ["us", "p6", "bundle", "rule", "future", "past", "scope", "glob"]
)
def test_scope_failures_stop_before_any_bound_file_open(tmp_path, monkeypatch, change):
    path, plan = plan_case(tmp_path)
    if change == "us":
        plan["market"] = "US"
    elif change == "p6":
        plan["model_id"] = "kr-next"
    elif change == "bundle":
        plan["bundle_sha256"] = "a" * 64
    elif change == "rule":
        plan["rule_sha256"] = "b" * 64
    elif change == "future":
        plan["cutoff"] = "2026-12-11T13:00:00+09:00"
    elif change == "past":
        plan["inputs"]["extract"]["min_session"] = "2026-10-29"
    elif change == "scope":
        plan["inputs"]["extract"]["path"] = str(tmp_path / "US/labels.json")
    elif change == "glob":
        plan["inputs"]["extract"]["path"] = str(tmp_path / "*.json")
    path.write_text(json.dumps(plan))
    actual = access._read

    def request_only(candidate, *args, **kwargs):
        assert candidate == path
        return actual(candidate, *args, **kwargs)

    monkeypatch.setattr(access, "_read", request_only)
    with pytest.raises(OutcomeError):
        access.load_real_plan(path, now=REVIEW)


def test_bound_input_rechecks_time_hash_and_cannot_mutate_permit(tmp_path):
    path, plan = plan_case(tmp_path)
    permit = access.load_real_plan(path, now=REVIEW)
    value, binding = access.load_bound_input(permit, "extract", now=REVIEW)
    assert value["synthetic_fixture"] is False
    assert binding["sha256"] == plan["inputs"]["extract"]["sha256"]
    changed = permit.request
    changed["inputs"]["extract"]["path"] = "/unknown/source.json"
    assert permit.request["inputs"]["extract"]["path"] == plan["inputs"]["extract"]["path"]
    with pytest.raises(OutcomeError, match="registered Friday"):
        access.load_bound_input(permit, "extract", now=NOW)
    Path(plan["inputs"]["extract"]["path"]).write_bytes(b"changed")
    with pytest.raises(OutcomeError, match="hash mismatch"):
        access.load_bound_input(permit, "extract", now=REVIEW)


def test_duplicate_json_keys_and_symlinks_are_rejected(tmp_path):
    path, plan = plan_case(tmp_path)
    path.write_text('{"market":"KR","market":"US"}')
    with pytest.raises(OutcomeError, match="duplicate"):
        access.load_real_plan(path, now=REVIEW)
    path.write_text(json.dumps(plan))
    link = tmp_path / "linked-read-plan.json"
    link.symlink_to(path)
    with pytest.raises(OutcomeError, match="symlink"):
        access.load_real_plan(link, now=REVIEW)


@pytest.mark.parametrize("change", ["invalid_hash", "invalid_scope", "future_auth"])
def test_manually_constructed_permit_cannot_bypass_fixed_scope(tmp_path, monkeypatch, change):
    path, plan = plan_case(tmp_path)
    digest, authorized = "a" * 64, REVIEW.isoformat()
    if change == "invalid_hash":
        digest = "invalid-hash"
    elif change == "future_auth":
        authorized = "2026-12-11T14:00:00+09:00"
    else:
        plan = {"cutoff": "2000-01-01T00:00:00+09:00", "inputs": plan["inputs"]}
    permit = access.ReadPermit(canonical_bytes(plan), digest, authorized)
    monkeypatch.setattr(
        access, "_read", lambda *a, **kw: pytest.fail("unregistered permit read input")
    )
    with pytest.raises(ValueError):
        access.load_bound_input(permit, "extract", now=REVIEW)


def test_symlink_replacement_before_open_never_reads_other_file(tmp_path, monkeypatch):
    path, plan = plan_case(tmp_path)
    permit = access.load_real_plan(path, now=REVIEW)
    bound = Path(plan["inputs"]["extract"]["path"])
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text('{"outside":"generated unrelated input"}')
    original = os.open

    def replaced(name, flags, *args, **kwargs):
        if name == bound.name:
            bound.unlink()
            bound.symlink_to(unrelated)
        return original(name, flags, *args, **kwargs)

    monkeypatch.setattr(access.os, "open", replaced)
    monkeypatch.setattr(access.os, "read", lambda *a: pytest.fail("read replaced input bytes"))
    with pytest.raises((OSError, OutcomeError)):
        access.load_bound_input(permit, "extract", now=REVIEW)
