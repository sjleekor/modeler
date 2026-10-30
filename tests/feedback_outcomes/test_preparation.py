"""Invented metadata only: these tests never create or inspect source inputs."""

import copy
import io
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from modeler.feedback.core import sha256_bytes
from modeler.feedback_outcomes import access, preparation
from modeler.feedback_outcomes.contracts import OutcomeError

ROOT = "/invented-nonexistent/explicit-kr-endpoints"


def metadata():
    bindings = {
        role: {"role": role, "path": f"{ROOT}/{role}.json", "sha256": "a" * 64}
        for role in ("calendar", "publications", "extract")
    }
    bindings["extract"].update(min_session="2026-11-04", max_session="2026-12-02")
    return {
        "source_root": ROOT,
        "bindings": bindings,
        "cutoff": "2026-12-04T13:00:00+09:00",
        "max_session": "2026-12-02",
        # This is an invented supplied hash, not a claim to have verified bytes.
        "adapter_code_sha256": "b" * 64,
    }


def fail_source_io(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("metadata preparation attempted filesystem I/O")

    for method in ("open", "read_bytes", "read_text", "exists", "stat", "is_symlink", "resolve"):
        monkeypatch.setattr(Path, method, fail)
    monkeypatch.setattr(access, "_read", fail)


def test_template_is_incomplete_and_generation_is_pure_metadata(monkeypatch):
    fail_source_io(monkeypatch)
    template = preparation.template()
    assert template["prepared_valid"] is False
    assert template["authorized"] is template["real_ready"] is False
    assert "inputs.extract.sha256" in template["missing"]
    assert "adapter_code_sha256" in template["missing"]
    assert template["plan"]["inputs"]["extract"]["sha256"] is None
    supplied = metadata()
    plan = preparation.prepare_read_plan(**supplied)
    diagnostic = preparation.validate_preparation(plan)
    assert diagnostic["prepared_valid"] is True
    assert diagnostic["authorized"] is diagnostic["real_ready"] is False
    assert diagnostic["missing"] == []
    assert "unverified" in diagnostic["validation_scope"]
    assert len(diagnostic["remaining"]) == 4
    assert set(plan) == set(preparation.PINS) | preparation.VARIABLE_FIELDS
    supplied["bindings"]["extract"]["sha256"] = "c" * 64
    assert plan["inputs"]["extract"]["sha256"] == "a" * 64


def test_real_guard_remains_closed_even_after_preparing_future_plan(monkeypatch):
    plan = preparation.prepare_read_plan(**metadata())
    fail_source_io(monkeypatch)
    assert preparation.validate_preparation(plan)["prepared_valid"]
    with pytest.raises(OutcomeError, match="registered Friday"):
        access.load_real_plan(
            "/invented/future-read-plan.json", now=datetime(2026, 10, 30, tzinfo=UTC)
        )
    # Preparation never has an observed_at/now override that could forge a permit.
    with pytest.raises(TypeError):
        preparation.prepare_read_plan(**metadata(), observed_at="2026-12-04T14:00:00+09:00")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("market", "US"),
        ("model_id", "p6prime"),
        ("bundle_sha256", "c" * 64),
        ("report_end", "2027-01-30"),
        ("cutoff", "2026-12-04T12:59:59+09:00"),
        ("cutoff", "2026-12-04T13:00:00"),
        ("max_session", "2026-12-05"),
        ("max_session", "2026-12-2"),
        ("adapter_code_sha256", None),
        ("adapter_code_sha256", "not-a-digest"),
        ("synthetic_fixture", 0),
        ("authorized", True),
    ],
)
def test_unregistered_or_malformed_metadata_is_rejected(field, value, monkeypatch):
    plan = preparation.prepare_read_plan(**metadata())
    plan[field] = value
    fail_source_io(monkeypatch)
    assert preparation.validate_preparation(plan)["prepared_valid"] is False


@pytest.mark.parametrize(
    "path",
    [
        "relative/input.json",
        "/",
        "/invented/../input.json",
        "/invented/US/input.json",
        "/invented/P6/input.json",
        "/invented/raw/input.json",
        "/invented/.env/input.json",
        "/invented/*.json",
        "/invented/a\n.json",
        "/invented/nested/extract.json",
        ROOT + "/extract.parquet",
    ],
)
def test_path_checks_are_lexical_without_filesystem_probes(path, monkeypatch):
    supplied = metadata()
    supplied["bindings"]["extract"]["path"] = path
    fail_source_io(monkeypatch)
    with pytest.raises(OutcomeError):
        preparation.prepare_read_plan(**supplied)


@pytest.mark.parametrize("change", ["role", "duplicate", "min", "max", "extra", "missing"])
def test_input_bindings_are_exact(change, monkeypatch):
    supplied = metadata()
    entry = supplied["bindings"]["extract"]
    if change == "role":
        entry["role"] = "calendar"
    elif change == "duplicate":
        entry["path"] = supplied["bindings"]["calendar"]["path"]
    elif change == "min":
        entry["min_session"] = "2026-11-03"
    elif change == "max":
        entry["max_session"] = "2026-12-01"
    elif change == "extra":
        entry["observed_at"] = "2026-12-04T14:00:00+09:00"
    else:
        del supplied["bindings"]["publications"]
    fail_source_io(monkeypatch)
    with pytest.raises(OutcomeError):
        preparation.prepare_read_plan(**supplied)


def test_metadata_schema_matches_real_validator_without_opening_inputs(monkeypatch):
    supplied = metadata()
    # Reading our own implementation is permitted. No source input is read.
    supplied["adapter_code_sha256"] = sha256_bytes(
        Path(access.__file__).with_name("prices.py").read_bytes()
    )
    plan = preparation.prepare_read_plan(**supplied)
    monkeypatch.setattr(Path, "is_symlink", lambda *args, **kwargs: False)
    local = datetime.fromisoformat(plan["cutoff"])
    access._validate_plan(plan, local)
    for field, value in (("market", "US"), ("source_scope", "other"), ("max_session", "20261202")):
        changed = copy.deepcopy(plan)
        changed[field] = value
        assert preparation.validate_preparation(changed)["prepared_valid"] is False
        with pytest.raises((ValueError, TypeError)):
            access._validate_plan(changed, local)


def test_cli_stdin_metadata_stdout_only(monkeypatch, capsys):
    fail_source_io(monkeypatch)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(metadata())))
    assert preparation.main(["create"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["prepared_valid"] is True
    assert created["authorized"] is created["real_ready"] is False
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(created["plan"])))
    assert preparation.main(["validate"]) == 0
    assert json.loads(capsys.readouterr().out)["prepared_valid"]
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"source_root":"x","source_root":"y"}'))
    assert preparation.main(["create"]) == 2
    assert "duplicate" in json.loads(capsys.readouterr().out)["errors"][0]
    with pytest.raises(SystemExit):
        preparation.main(["validate", "/arbitrary/existing/performance.json"])


def test_real_module_cli_template():
    result = subprocess.run(
        [sys.executable, "-m", "modeler.feedback_outcomes.preparation", "template"],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(result.stdout)
    assert document["prepared_valid"] is False
    assert document["authorized"] is document["real_ready"] is False
    assert document["plan"]["inputs"]["calendar"]["path"] is None


def test_real_module_cli_create_and_validate():
    created = subprocess.run(
        [sys.executable, "-m", "modeler.feedback_outcomes.preparation", "create"],
        input=json.dumps(metadata()),
        check=True,
        capture_output=True,
        text=True,
    )
    envelope = json.loads(created.stdout)
    result = subprocess.run(
        [sys.executable, "-m", "modeler.feedback_outcomes.preparation", "validate"],
        input=json.dumps(envelope["plan"]),
        check=True,
        capture_output=True,
        text=True,
    )
    diagnostic = json.loads(result.stdout)
    assert diagnostic["prepared_valid"] is True
    assert diagnostic["authorized"] is diagnostic["real_ready"] is False
