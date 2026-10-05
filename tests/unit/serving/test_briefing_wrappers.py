"""Contract tests for deploy/prod/bin/briefing-stage.sh, us-prepare.sh and kr-prepare.sh (fake python, no lake)."""
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
KR_PREPARE = BIN / "kr-prepare.sh"

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
    modeler.serving.kr_live_prepare) exit "${FAKE_LIVE_RC:-0}";;
    modeler.serving.kr_prepare)
      [ "${FAKE_KR_PREP_RC:-0}" -eq 0 ] || exit "$FAKE_KR_PREP_RC"
      while [ $# -gt 0 ]; do [ "$1" = --output-dir ] && out=$2; shift; done
      mkdir -p "$out" && touch "$out/completion.json"; exit 0;;
  esac
elif [ "$1" = -c ]; then
  case "$3" in
    prepared-ok) if [ -n "${FAKE_ALREADY:-}" ] || [ -f "$FAKE_STATE/prepared" ]; then echo /x/manifest.json; exit 0; fi; exit 1;;
    data-ready) echo "  prices_daily: fake" >&2; exit "${FAKE_DATA_RC:-0}";;
    kr-dates) [ "${FAKE_CAL_RC:-0}" -eq 0 ] || exit "$FAKE_CAL_RC"
      echo "${FAKE_KR_SESSION:-session}"; echo "${FAKE_K:-2026-10-02}"; exit 0;;
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
                       ("taskset", 'echo "taskset $1 $2" >> "$FAKE_LOG"; shift 2; exec "$@"'),
                       ("nice", 'echo "nice $1 $2" >> "$FAKE_LOG"; shift 2; exec "$@"')):
        shim = tmp_path / "shims" / name
        shim.parent.mkdir(exist_ok=True)
        shim.write_text("#!/usr/bin/env bash\n" + body + "\n")
        shim.chmod(0o755)
    config = root / "config"
    config.mkdir()
    evidence = config / "evidence.json"
    evidence.write_text('{"status": "score_equivalent"}')
    calendar = config / "calendar-KR.json"
    calendar.write_text("{}")
    (config / "ops.json").write_text(json.dumps({"release_manifest": str(release / "release.json"),
                                                 "python": str(fake), "kr_calendar": str(calendar),
                                                 "prepared_root": str(root / "prepared")}))
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


FAKE_SDC = {
    "kr-export-wait-ready.sh": 'echo "gate $*" >> "$FAKE_LOG"; exit "${FAKE_GATE_RC:-0}"',
    "kr-raw-parquet-export.sh": (
        'echo "export $*" >> "$FAKE_LOG"; [ "${FAKE_EXPORT_RC:-0}" -eq 0 ] || exit "$FAKE_EXPORT_RC"\n'
        'm="$KR_STOCK_DATA_ROOT/kr/raw/raw_postgres/snapshot_date=$2/source=sj2_remote/_manifests"\n'
        'mkdir -p "$m" && touch "$m/_SUCCESS.json"'),
}


@pytest.fixture()
def kr(serving, tmp_path: Path) -> dict:
    sdc = tmp_path / "sdc_bin"
    sdc.mkdir()
    for name, body in FAKE_SDC.items():
        (sdc / name).write_text("#!/usr/bin/env bash\n" + body + "\n")
        (sdc / name).chmod(0o755)
    lake = tmp_path / "lake"
    env = {**serving["env"], "SDC_BIN": str(sdc), "KR_STOCK_DATA_ROOT": str(lake),
           "KR_PREPARE_GATE_UNTIL": "23:59"}
    out = serving["root"] / "prepared" / "kr" / "score_date=2026-10-02" / "prep_id=2026-10-05"
    return {**serving, "env": env, "lake": lake, "out": out}


def _marker(lake: Path, kind: str, snap: str = "2026-10-05") -> Path:
    sub = {"raw": "raw/raw_postgres", "feature": "derived/feature"}[kind]
    path = lake / "kr" / sub / f"snapshot_date={snap}" / "source=sj2_remote" / "_manifests" / "_SUCCESS.json"
    path.parent.mkdir(parents=True)
    path.touch()
    return path


def test_kr_prepare_happy_path(kr):
    done = _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"])
    assert done.returncode == 0, done.stderr
    calls = _calls(kr)
    (gate,) = [c for c in calls if c.startswith("gate ")]
    assert gate.startswith("gate --feature-asof-date 2026-10-02 --deadline-seconds ")
    assert "export --snapshot-date 2026-10-05 --consistent-snapshot" in calls
    cutoff = "--input-cutoff 2026-10-05T09:30:00+09:00"
    (live,) = [c for c in calls if "kr_live_prepare" in c]
    assert (f"argv=-m modeler.serving.kr_live_prepare --snapshot-date 2026-10-05 --feature-asof-date 2026-10-02 "
            f"{cutoff} --stock-data-root {kr['lake']} --profile full --max-temp-size 30GB|") in live
    (prep,) = [c for c in calls if "modeler.serving.kr_prepare" in c]
    assert f"--output-dir {kr['out']} --stock-data-root {kr['lake']}|" in prep
    assert f"PYTHONPATH={kr['release']}/src" in prep and "DONTWRITE=1" in prep
    assert "/logs/kr-prepare/D=2026-10-05" in prep
    assert "timeout 5400" in calls and "timeout 3600" in calls and "timeout 1800" in calls
    assert "taskset -c 0,1" in calls and "nice -n 10" in calls
    assert (kr["out"] / "completion.json").is_file()


def test_kr_prepare_closed_day_skips(kr):
    done = _run(KR_PREPARE, ["--report-date", "2026-10-04"], kr["env"], {"FAKE_KR_SESSION": "closed"})
    assert done.returncode == 0 and "not a KR session" in done.stdout
    assert not [c for c in _calls(kr) if c.startswith(("gate ", "export ")) or "argv=-m" in c]


def test_kr_prepare_already_prepared_skips(kr):
    kr["out"].mkdir(parents=True)
    (kr["out"] / "completion.json").touch()
    done = _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"])
    assert done.returncode == 0 and "already prepared" in done.stdout
    assert not [c for c in _calls(kr) if c.startswith(("gate ", "export ")) or "argv=-m" in c]


def test_kr_prepare_resumes_after_sealed_steps(kr):
    _marker(kr["lake"], "raw")
    _marker(kr["lake"], "feature")
    done = _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"])
    assert done.returncode == 0, done.stderr
    calls = _calls(kr)
    assert not [c for c in calls if c.startswith(("gate ", "export ")) or "kr_live_prepare" in c]
    assert [c for c in calls if "modeler.serving.kr_prepare" in c]


@pytest.mark.parametrize("gate_rc,code", [(75, 30), (1, 31), (5, 5)])
def test_kr_prepare_gate_failure_stops_before_export(kr, gate_rc, code):
    done = _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"], {"FAKE_GATE_RC": str(gate_rc)})
    assert done.returncode == code
    assert not [c for c in _calls(kr) if c.startswith("export ") or "argv=-m" in c]


def test_kr_prepare_consistent_snapshot_can_be_turned_off(kr):
    done = _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"], {"KR_PREPARE_CONSISTENT_SNAPSHOT": "0"})
    assert done.returncode == 0, done.stderr
    assert "export --snapshot-date 2026-10-05" in _calls(kr)


@pytest.mark.parametrize("extra,code", [({"FAKE_EXPORT_RC": "3"}, 3), ({"FAKE_LIVE_RC": "124"}, 124),
                                        ({"FAKE_KR_PREP_RC": "1"}, 1)])
def test_kr_prepare_step_failure_code_passes_through(kr, extra, code):
    assert _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"], extra).returncode == code


def test_kr_prepare_calendar_gap_is_12(kr):
    assert _run(KR_PREPARE, ["--report-date", "2027-01-04"], kr["env"], {"FAKE_CAL_RC": "3"}).returncode == 12


def test_kr_prepare_bad_args(kr):
    assert _run(KR_PREPARE, ["--report-date", "today"], kr["env"]).returncode == 2
    assert _run(KR_PREPARE, ["x"], kr["env"]).returncode == 2
    assert _run(KR_PREPARE, [], kr["env"], {"KR_PREPARE_CONSISTENT_SNAPSHOT": "yes"}).returncode == 2


# ---- step 5: retention ----

def _snap(lake: Path, kind: str, snap: str, *, files: bool = True) -> Path:
    """An old snapshot: sealed manifests plus a parquet file."""
    sub = {"raw": "raw/raw_postgres", "feature": "derived/feature"}[kind]
    base = lake / "kr" / sub / f"snapshot_date={snap}" / "source=sj2_remote"
    (base / "_manifests" / "table_manifests").mkdir(parents=True)
    (base / "_manifests" / "_SUCCESS.json").write_text(f'{{"snap": "{snap}"}}')
    (base / "_manifests" / "table_manifests" / "t.json").write_text("{}")
    if files:
        (base / "daily_ohlcv").mkdir()
        (base / "daily_ohlcv" / "part-0.parquet").write_text("x" * 100)
    return base.parent


def _archive(lake: Path, kind: str, snap: str) -> Path:
    sub = {"raw": "raw/_manifest_archive", "feature": "derived/_manifest_archive/feature"}[kind]
    return lake / "kr" / sub / f"snapshot_date={snap}" / "source=sj2_remote" / "_manifests"


def _kr_run(kr, extra: dict | None = None):
    return _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"], extra)


def test_retention_deletes_older_and_archives_manifests(kr):
    lake = kr["lake"]
    old = {k: _snap(lake, k, "2026-10-01") for k in ("raw", "feature")}
    cur = {k: _snap(lake, k, "2026-10-05") for k in ("raw", "feature")}
    newer = {k: _snap(lake, k, "2026-10-06") for k in ("raw", "feature")}
    done = _kr_run(kr)
    assert done.returncode == 0, done.stderr
    for kind in ("raw", "feature"):
        assert not old[kind].exists()
        assert cur[kind].is_dir() and newer[kind].is_dir()
        arch = _archive(lake, kind, "2026-10-01")
        assert (arch / "_SUCCESS.json").read_text() == '{"snap": "2026-10-01"}'
        assert (arch / "table_manifests" / "t.json").is_file()
        assert not _archive(lake, kind, "2026-10-05").parent.parent.exists()
    assert done.stdout.count("retention: deleted") == 2 and "archived ->" in done.stdout
    assert (kr["out"] / "completion.json").is_file()


def test_retention_deletes_older_metric_snapshots(kr):
    metric = kr["lake"] / "kr" / "derived" / "metric"
    for snap in ("2026-10-01", "2026-10-05"):
        fact = metric / f"snapshot_date={snap}" / "source=sj2_remote" / "stock_metric_fact"
        fact.mkdir(parents=True)
        (fact / "part-000000.parquet").write_text("x")
        (fact / "_plan.json").write_text("{}")
    done = _kr_run(kr)
    assert done.returncode == 0, done.stderr
    assert not (metric / "snapshot_date=2026-10-01").exists()
    assert (metric / "snapshot_date=2026-10-05" / "source=sj2_remote" / "stock_metric_fact" / "_plan.json").is_file()
    assert "retention: deleted metric snapshot_date=2026-10-01" in done.stdout
    assert "WARNING" not in done.stderr


def test_retention_keep_two_keeps_one_older(kr):
    lake = kr["lake"]
    for snap in ("2026-09-29", "2026-10-01"):
        _snap(lake, "raw", snap)
    done = _kr_run(kr, {"KR_PREPARE_KEEP_SNAPSHOTS": "2"})
    assert done.returncode == 0, done.stderr
    raw = lake / "kr" / "raw" / "raw_postgres"
    assert not (raw / "snapshot_date=2026-09-29").exists()
    assert (raw / "snapshot_date=2026-10-01").is_dir() and (raw / "snapshot_date=2026-10-05").is_dir()
    assert _archive(lake, "raw", "2026-09-29").is_dir() and not _archive(lake, "raw", "2026-10-01").exists()


def test_retention_disabled_keeps_everything(kr):
    old = _snap(kr["lake"], "raw", "2026-10-01")
    done = _kr_run(kr, {"KR_PREPARE_RETENTION": "0"})
    assert done.returncode == 0 and old.is_dir() and "retention disabled" in done.stdout
    assert not (kr["lake"] / "kr" / "raw" / "_manifest_archive").exists()


@pytest.mark.parametrize("extra", [{"KR_PREPARE_KEEP_SNAPSHOTS": "0"}, {"KR_PREPARE_KEEP_SNAPSHOTS": "x"},
                                   {"KR_PREPARE_KEEP_SNAPSHOTS": "-1"}, {"KR_PREPARE_RETENTION": "2"}])
def test_retention_invalid_knob_is_2(kr, extra):
    assert _kr_run(kr, extra).returncode == 2
    assert _calls(kr) == []


def test_retention_skipped_when_step4_fails(kr):
    old = _snap(kr["lake"], "raw", "2026-10-01")
    assert _kr_run(kr, {"FAKE_KR_PREP_RC": "1"}).returncode == 1
    assert old.is_dir() and not (kr["lake"] / "kr" / "raw" / "_manifest_archive").exists()


def test_retention_runs_when_already_prepared(kr):
    old = _snap(kr["lake"], "feature", "2026-10-01")
    _snap(kr["lake"], "feature", "2026-10-05")  # with no SNAP dir the older one is the newest and stays
    kr["out"].mkdir(parents=True)
    (kr["out"] / "completion.json").touch()
    done = _kr_run(kr)
    assert done.returncode == 0 and "already prepared" in done.stdout
    assert not old.exists() and _archive(kr["lake"], "feature", "2026-10-01").is_dir()
    assert not [c for c in _calls(kr) if c.startswith(("gate ", "export ")) or "argv=-m" in c]
    again = _kr_run(kr)  # a second rerun changes nothing
    assert again.returncode == 0 and "retention: deleted" not in again.stdout


def test_retention_ignores_other_names(kr):
    raw = kr["lake"] / "kr" / "raw" / "raw_postgres"
    for name in ("snapshot_date=2026-10-01.bak", "snapshot_date=latest", "other"):
        (raw / name).mkdir(parents=True)
    _kr_run(kr)
    assert all((raw / n).is_dir() for n in ("snapshot_date=2026-10-01.bak", "snapshot_date=latest", "other"))


def test_retention_failure_warns_and_exits_0(kr):
    old = _snap(kr["lake"], "raw", "2026-10-01")
    fine = _snap(kr["lake"], "feature", "2026-10-01")
    _snap(kr["lake"], "feature", "2026-10-05")
    locked = old / "source=sj2_remote" / "daily_ohlcv"
    locked.chmod(0o555)  # non-writable dir: its parquet cannot be unlinked
    try:
        if os.access(locked, os.W_OK):
            pytest.skip("running as a user that ignores directory permissions")
        done = _kr_run(kr)
        assert done.returncode == 0, done.stderr
        assert "WARNING retention incomplete" in done.stderr and "snapshot_date=2026-10-01" in done.stderr
        assert (kr["out"] / "completion.json").is_file()
        assert not fine.exists()  # the other parent is still cleaned
        # manifests went first: no leftover _SUCCESS.json for parquet that remains
        assert not list(old.rglob("_SUCCESS.json"))
        assert _archive(kr["lake"], "raw", "2026-10-01").is_dir()
    finally:
        locked.chmod(0o755)


def test_retention_refuses_symlinked_target(kr, tmp_path):
    raw = kr["lake"] / "kr" / "raw" / "raw_postgres"
    real = _snap(tmp_path / "elsewhere", "raw", "2026-09-01")
    raw.mkdir(parents=True)
    (raw / "snapshot_date=2026-10-01").symlink_to(real)
    done = _kr_run(kr)
    assert done.returncode == 0 and "symlink" in done.stderr
    assert real.is_dir() and (real / "source=sj2_remote" / "daily_ohlcv" / "part-0.parquet").is_file()


def test_retention_refuses_symlinked_parent(kr, tmp_path):
    real = _snap(tmp_path / "elsewhere", "raw", "2026-09-01").parent
    (kr["lake"] / "kr" / "raw").mkdir(parents=True)
    (kr["lake"] / "kr" / "raw" / "raw_postgres").symlink_to(real)
    done = _kr_run(kr)
    assert done.returncode == 0 and "symlink" in done.stderr
    assert (real / "snapshot_date=2026-09-01").is_dir()


def test_retention_keeps_conflicting_archive(kr):
    old = _snap(kr["lake"], "raw", "2026-10-01")
    arch = _archive(kr["lake"], "raw", "2026-10-01")
    arch.mkdir(parents=True)
    (arch / "_SUCCESS.json").write_text("different")
    done = _kr_run(kr)
    assert done.returncode == 0 and "not deleted" in done.stderr
    assert old.is_dir() and (arch / "_SUCCESS.json").read_text() == "different"


def test_retention_cleans_old_spill_dirs_only(kr):
    tmp = kr["lake"] / "kr" / "derived" / "_duckdb_tmp"
    old, same, odd = (tmp / "2026-10-01_20261001T041000Z_123", tmp / "2026-10-05_20261005T041000Z_9", tmp / "notes")
    for d in (old, same, odd):
        d.mkdir(parents=True)
    (old / "build_profile.partial.json").write_text("{}")
    (old / "spill.tmp").write_text("x")
    done = _kr_run(kr)
    assert done.returncode == 0, done.stderr
    assert not old.exists() and same.is_dir() and odd.is_dir()
    assert (kr["lake"] / "kr/derived/_manifest_archive/_duckdb_tmp" / old.name / "build_profile.partial.json").is_file()
