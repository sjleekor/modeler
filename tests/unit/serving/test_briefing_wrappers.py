"""Contract tests for deploy/prod/bin/briefing-stage.sh and us-prepare.sh (fake python, no lake)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

BIN = Path(__file__).resolve().parents[3] / "deploy" / "prod" / "bin"
STAGE = BIN / "briefing-stage.sh"
PREPARE = BIN / "us-prepare.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash required")

FAKE_PYTHON = r"""#!/usr/bin/env bash
echo "argv=$*|PYTHONPATH=${PYTHONPATH:-}|DONTWRITE=${PYTHONDONTWRITEBYTECODE:-}|STOCK=${STOCK_DATA_ROOT:-}|POLARS=${POLARS_MAX_THREADS:-}|PWD=$PWD" >> "$FAKE_LOG"
if [ "$1" = -m ]; then
  case "$2" in
    modeler.serving.us_expected) echo "${FAKE_A:-2026-09-29}"; exit "${FAKE_E_RC:-0}";;
    modeler.serving.us_daily)
      [ "${FAKE_PREP_RC:-0}" -eq 0 ] || exit "$FAKE_PREP_RC"
      touch "$FAKE_STATE/prepared"; echo "prepared: /x/manifest.json"; exit 0;;
    modeler.serving.daily_wrapper) exit "${FAKE_RC:-0}";;
  esac
elif [ "$1" = -c ]; then
  case "$3" in
    prepared-ok) if [ -n "${FAKE_ALREADY:-}" ] || [ -f "$FAKE_STATE/prepared" ]; then echo /x/manifest.json; exit 0; fi; exit 1;;
    data-ready) echo "  prices_daily: fake" >&2; exit "${FAKE_DATA_RC:-0}";;
  esac
fi
exit 99
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def serving(tmp_path: Path) -> dict:
    root = tmp_path / "serving"
    release = root / "releases" / "r1"
    (release / "src").mkdir(parents=True)
    (release / "release.json").write_text("{}")
    fake = tmp_path / "fakebin" / "python"
    fake.parent.mkdir()
    fake.write_text(FAKE_PYTHON)
    fake.chmod(0o755)
    for name, body in (("timeout", 'echo "timeout $1" >> "$FAKE_LOG"; shift; exec "$@"'),
                       ("taskset", 'echo "taskset $1 $2" >> "$FAKE_LOG"; shift 2; exec "$@"')):
        shim = tmp_path / "shims" / name
        shim.parent.mkdir(exist_ok=True)
        shim.write_text("#!/usr/bin/env bash\n" + body + "\n")
        shim.chmod(0o755)
    config = root / "config"
    config.mkdir()
    evidence = config / "evidence.json"
    evidence.write_text('{"status": "score_equivalent"}')
    (config / "ops.json").write_text(json.dumps({"release_manifest": str(release / "release.json"),
                                                 "python": str(fake)}))
    (config / "pins.json").write_text(json.dumps(
        {"parity_evidence": {"path": str(evidence), "sha256": _sha(evidence)}}))
    state = tmp_path / "state"
    state.mkdir()
    log = tmp_path / "calls.log"
    env = {"PATH": f"{tmp_path / 'shims'}:{os.environ['PATH']}", "SERVING_ROOT": str(root),
           "BOOTSTRAP_PYTHON": sys.executable, "FAKE_LOG": str(log), "FAKE_STATE": str(state)}
    return {"root": root, "release": release, "log": log, "env": env, "evidence": evidence,
            "config": config}


def _run(script: Path, args: list[str], env: dict, extra: dict | None = None):
    return subprocess.run(["bash", str(script), *args], env={**env, **(extra or {})},
                          capture_output=True, text=True, timeout=60)


def _calls(serving: dict) -> list[str]:
    return serving["log"].read_text().splitlines() if serving["log"].exists() else []


@pytest.mark.parametrize("stage,extra,expected", [
    ("select", [], "daily_wrapper select --config"),
    ("run", [], "daily_wrapper run --config"),
    ("monitor", ["--attempt", "2"], "daily_wrapper monitor --config"),
])
def test_stage_contract(serving, stage, extra, expected):
    done = _run(STAGE, [stage, *extra], serving["env"])
    assert done.returncode == 0, done.stderr
    (line,) = _calls(serving)
    ops = serving["root"] / "config" / "ops.json"
    assert f"argv=-m modeler.serving.{expected} {ops}" in line
    assert f"PYTHONPATH={serving['release']}/src" in line and "DONTWRITE=1" in line
    assert f"PWD={serving['release'].resolve()}" in line or f"PWD={serving['release']}" in line
    assert ("--attempt 2" in line) == bool(extra)


def test_stage_propagates_failure_code(serving):
    assert _run(STAGE, ["select"], serving["env"], {"FAKE_RC": "1"}).returncode == 1


@pytest.mark.parametrize("args", [[], ["nope"], ["select", "--attempt", "1"], ["monitor", "--attempt"],
                                  ["monitor", "--attempt", "4"], ["monitor", "--attempt", "x"],
                                  ["run", "extra"], ["monitor", "--bogus", "1"]])
def test_stage_rejects_bad_args(serving, args):
    done = _run(STAGE, args, serving["env"])
    assert done.returncode == 2 and _calls(serving) == []


def test_stage_missing_ops_is_10(serving):
    (serving["config"] / "ops.json").unlink()
    done = _run(STAGE, ["select"], serving["env"])
    assert done.returncode == 10 and "ops.json not found" in done.stderr


def test_prepare_happy_path(serving):
    done = _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"])
    assert done.returncode == 0, done.stderr
    calls = _calls(serving)
    assert any("us_expected session --report-date 2026-10-01" in c for c in calls)
    (prepare,) = [c for c in calls if "modeler.serving.us_daily" in c]
    assert (f"argv=-m modeler.serving.us_daily prepare --as-of 2026-09-29 "
            f"--raw-feature-parity-status score_equivalent "
            f"--raw-feature-parity-evidence {serving['evidence']}|") in prepare
    assert f"PYTHONPATH={serving['release']}/src" in prepare
    assert f"STOCK={serving['root']}/stock_data" in prepare and "POLARS=2" in prepare
    assert "timeout 1800" in calls and "taskset -c 0,1" in calls


def test_prepare_data_not_ready(serving):
    done = _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"], {"FAKE_DATA_RC": "3"})
    assert done.returncode == 20
    assert "data not ready" in done.stderr and "A=2026-09-29" in done.stderr
    assert not [c for c in _calls(serving) if "modeler.serving.us_daily" in c]


def test_prepare_already_prepared_skips(serving):
    done = _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"], {"FAKE_ALREADY": "1"})
    assert done.returncode == 0 and "skipping" in done.stdout
    assert not [c for c in _calls(serving) if "modeler.serving.us_daily" in c]


def test_prepare_evidence_sha_mismatch(serving):
    serving["evidence"].write_text('{"status": "changed"}')
    done = _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"])
    assert done.returncode == 11 and "differs from pins.json" in done.stderr
    assert _calls(serving) == []


def test_prepare_expected_session_failure_is_12(serving):
    assert _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"], {"FAKE_E_RC": "1"}).returncode == 12


def test_prepare_failure_code_passes_through(serving):
    assert _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"], {"FAKE_PREP_RC": "124"}).returncode == 124


def test_prepare_bad_args(serving):
    assert _run(PREPARE, ["--run-date", "tomorrow"], serving["env"]).returncode == 2
    assert _run(PREPARE, ["x"], serving["env"]).returncode == 2
