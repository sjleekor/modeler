"""Contract tests for deploy/prod/bin/briefing-stage.sh, us-prepare.sh and kr-prepare.sh (fake python, no lake).

The wrappers need util-linux ``flock`` (the servers have it, a Mac does not): a shim built on
``fcntl.flock`` stands in, locking the inherited file descriptor exactly like the real command.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
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
    modeler.serving.kr_reference)
      while [ $# -gt 0 ]; do [ "$1" = --output ] && ev=$2; [ "$1" = --k ] && k=$2; shift; done
      if [ "${FAKE_REF_RC:-0}" -eq 32 ]; then
        mkdir -p "$(dirname "$ev")"; echo '{"verdict": "none"}' > "$ev"; exit 32
      fi
      [ "${FAKE_REF_RC:-0}" -eq 0 ] || exit "$FAKE_REF_RC"
      ref=${FAKE_REF:-$k}
      mkdir -p "$(dirname "$ev")"
      echo "{\"verdict\": \"complete\", \"reference_date\": \"$ref\", \"cut_required\": ${FAKE_CUT_JSON:-false}}" > "$ev"
      echo "$ref"; echo "${FAKE_CUT:-nocut}"; exit 0;;
    modeler.serving.kr_prepare)
      [ "${FAKE_KR_PREP_RC:-0}" -eq 0 ] || exit "$FAKE_KR_PREP_RC"
      while [ $# -gt 0 ]; do [ "$1" = --output-dir ] && out=$2; shift; done
      mkdir -p "$out" && touch "$out/completion.json"; exit 0;;
  esac
elif [ "$1" = -c ]; then
  case "$3" in
    prepared-ok) if [ -n "${FAKE_ALREADY:-}" ] || [ -f "$FAKE_STATE/prepared" ]; then echo /x/manifest.json; exit 0; fi; exit 1;;
    resolve) echo "  resolve: fake" >&2; [ "${FAKE_DATA_RC:-0}" -eq 0 ] || exit "$FAKE_DATA_RC"
      echo "${FAKE_RESOLVED:-$4}"; exit 0;;
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
    flock_shim = tmp_path / "shims" / "flock"
    flock_shim.write_text(
        "#!/usr/bin/env bash\n"
        '[ "$1" = -n ] || exit 2\n'
        f'exec {sys.executable} -c "import fcntl,sys; fcntl.flock(int(sys.argv[1]), fcntl.LOCK_EX | fcntl.LOCK_NB)" "$2"\n')
    flock_shim.chmod(0o755)
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
    assert not any("WARNING" in line for line in done.stdout.splitlines())
    (prepare,) = [c for c in calls if "modeler.serving.us_daily" in c]
    assert (f"argv=-m modeler.serving.us_daily prepare --as-of 2026-09-29 "
            f"--raw-feature-parity-status score_equivalent "
            f"--raw-feature-parity-evidence {serving['evidence']}|") in prepare
    assert f"PYTHONPATH={serving['release']}/src" in prepare
    assert f"STOCK={serving['root']}/stock_data" in prepare and "POLARS=2" in prepare
    assert "timeout 1800" in calls and "taskset -c 0,1" in calls


def test_prepare_uses_the_newest_covered_session_when_the_lake_lacks_a(serving):
    """2026-10-05 change 4: no more exit 20 for a missing A; A' (< A) is prepared and announced."""
    done = _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"], {"FAKE_RESOLVED": "2026-09-26"})
    assert done.returncode == 0, done.stderr
    assert "WARNING the lake does not cover A=2026-09-29 yet; preparing A'=2026-09-26" in done.stdout
    (prepare,) = [c for c in _calls(serving) if "modeler.serving.us_daily" in c]
    assert "prepare --as-of 2026-09-26 --raw-feature-parity-status score_equivalent" in prepare


def test_prepare_exit_20_only_when_the_lake_covers_no_session_at_all(serving):
    done = _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"], {"FAKE_DATA_RC": "3"})
    assert done.returncode == 20
    assert "covers no completed US session" in done.stderr
    assert not [c for c in _calls(serving) if "modeler.serving.us_daily" in c]
    unreadable = _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"], {"FAKE_DATA_RC": "9"})
    assert unreadable.returncode == 20 and "could not read the lake" in unreadable.stderr


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
        # A slow export with a grandchild (the shape of docker/collector under the wrapper).
        'if [ -n "${FAKE_EXPORT_SLEEP:-}" ]; then\n'
        '  sleep "$FAKE_EXPORT_SLEEP" & echo "$!" >> "$FAKE_STATE/grandchildren"\n'
        '  echo "$$" >> "$FAKE_STATE/grandchildren"; touch "$FAKE_STATE/export-running"; wait\n'
        'fi\n'
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
    env = {**serving["env"], "SDC_BIN": str(sdc), "KR_STOCK_DATA_ROOT": str(lake)}
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
    # One look, never a wait: the deadline is 0 seconds.
    assert gate == "gate --feature-asof-date 2026-10-02 --deadline-seconds 0"
    assert "export --snapshot-date 2026-10-05 --consistent-snapshot" in calls
    assert calls.index(gate) < calls.index("export --snapshot-date 2026-10-05 --consistent-snapshot")
    evidence = kr["root"] / "logs" / "kr-prepare" / "D=2026-10-05" / "reference-selection.json"
    (ref,) = [c for c in calls if "modeler.serving.kr_reference" in c]
    assert (f"argv=-m modeler.serving.kr_reference --snapshot-date 2026-10-05 --report-date 2026-10-05 "
            f"--k 2026-10-02 --calendar {kr['config'] / 'calendar-KR.json'} --stock-data-root {kr['lake']} "
            f"--output {evidence} --gate-exit 0 --gate-evidence ") in ref
    cutoff = "--input-cutoff 2026-10-05T09:30:00+09:00"
    (live,) = [c for c in calls if "kr_live_prepare" in c]
    assert (f"argv=-m modeler.serving.kr_live_prepare --snapshot-date 2026-10-05 --feature-asof-date 2026-10-02 "
            f"{cutoff} --stock-data-root {kr['lake']} --profile full --max-temp-size 30GB|") in live
    (prep,) = [c for c in calls if "modeler.serving.kr_prepare" in c]
    assert (f"--output-dir {kr['out']} --stock-data-root {kr['lake']} --reference-evidence {evidence}|") in prep
    assert f"PYTHONPATH={kr['release']}/src" in prep and "DONTWRITE=1" in prep
    assert "/logs/kr-prepare/D=2026-10-05" in prep
    assert "timeout 5400" in calls and "timeout 3600" in calls and "timeout 1800" in calls
    assert "taskset -c 0,1" in calls and "nice -n 10" in calls
    assert (kr["out"] / "completion.json").is_file()


def test_kr_prepare_builds_for_the_fallback_session_and_cuts_the_marts(kr):
    """K=10-02 is incomplete in the export: K'=09-30 is prepared, the marts are cut at K'."""
    done = _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"],
                {"FAKE_REF": "2026-09-30", "FAKE_CUT": "cut", "FAKE_CUT_JSON": "true"})
    assert done.returncode == 0, done.stderr
    assert "WARNING reference: fallback to K'=2026-09-30" in done.stdout
    calls = _calls(kr)
    (live,) = [c for c in calls if "kr_live_prepare" in c]
    assert "--feature-asof-date 2026-09-30 " in live and "--profile full --max-temp-size 30GB --cut-to-asof|" in live
    (prep,) = [c for c in calls if "modeler.serving.kr_prepare" in c]
    out = kr["root"] / "prepared" / "kr" / "score_date=2026-09-30" / "prep_id=2026-10-05"
    assert f"--feature-asof-date 2026-09-30 " in prep and f"--output-dir {out} " in prep
    assert (out / "completion.json").is_file() and not (kr["out"] / "completion.json").exists()


def test_kr_prepare_with_no_complete_session_exits_32_and_prepares_nothing(kr):
    old = _snap(kr["lake"], "raw", "2026-10-01")
    done = _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"], {"FAKE_REF_RC": "32"})
    assert done.returncode == 32
    assert "no complete reference session" in done.stderr
    calls = _calls(kr)
    assert not [c for c in calls if "kr_live_prepare" in c or "modeler.serving.kr_prepare" in c]
    assert not old.exists()  # retention still runs: the export itself succeeded
    # A rerun on the same sealed snapshot repeats the verdict without exporting again.
    again = _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"])
    assert again.returncode == 32
    assert len([c for c in _calls(kr) if c.startswith("export ")]) == 1


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


def test_kr_prepare_a_fallback_session_counts_as_already_prepared(kr):
    """The fallback event finds the K' unit of the chain run and ends at once (prep_id=SNAP)."""
    earlier = kr["root"] / "prepared" / "kr" / "score_date=2026-09-30" / "prep_id=2026-10-05"
    earlier.mkdir(parents=True)
    (earlier / "completion.json").touch()
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


@pytest.mark.parametrize("gate_rc", [75, 1, 5])
def test_kr_prepare_gate_verdict_is_advisory_and_never_stops_the_run(kr, gate_rc):
    """Not ready, blocked and a gate error all continue: the exported snapshot decides."""
    done = _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"], {"FAKE_GATE_RC": str(gate_rc)})
    assert done.returncode == 0, done.stderr
    calls = _calls(kr)
    assert [c for c in calls if c.startswith("export ")] and [c for c in calls if "kr_reference" in c]
    (ref,) = [c for c in calls if "kr_reference" in c]
    assert f"--gate-exit {gate_rc} --gate-evidence " in ref
    assert "WARNING" in done.stdout and "export gate" in done.stdout
    assert (kr["out"] / "completion.json").is_file()


def test_kr_prepare_gate_can_be_switched_off(kr):
    done = _run(KR_PREPARE, ["--report-date", "2026-10-05"], kr["env"], {"KR_PREPARE_GATE": "off"})
    assert done.returncode == 0, done.stderr
    calls = _calls(kr)
    assert not [c for c in calls if c.startswith("gate ")]
    assert "--gate-exit" not in [c for c in calls if "kr_reference" in c][0]
    assert _run(KR_PREPARE, [], kr["env"], {"KR_PREPARE_GATE": "wait"}).returncode == 2


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


# ---- the shared lock and signal handling (2026-10-05 change 2) ----

def _hold_lock(path: Path) -> subprocess.Popen:
    """A process that holds an exclusive flock on ``path`` until it is killed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    holder = subprocess.Popen([sys.executable, "-c", (
        "import fcntl,sys,time\n"
        "f=open(sys.argv[1],'a+'); fcntl.flock(f, fcntl.LOCK_EX); print('held', flush=True); time.sleep(600)"),
        str(path)], stdout=subprocess.PIPE, text=True)
    assert holder.stdout.readline().strip() == "held"
    return holder


def test_kr_prepare_skips_with_exit_0_when_another_run_holds_the_lock(kr):
    holder = _hold_lock(kr["root"] / "locks" / "kr-prepare.lock")
    try:
        done = _kr_run(kr)
        assert done.returncode == 0, done.stderr
        assert "locked: another kr-prepare run holds" in done.stdout
        assert _calls(kr) == []  # not even the calendar helper or the gate ran
    finally:
        holder.kill()
        holder.wait()


def test_kr_prepare_lock_is_released_when_the_holder_dies(kr):
    holder = _hold_lock(kr["root"] / "locks" / "kr-prepare.lock")
    holder.kill()
    holder.wait()  # flock ends with the process: no stale lock
    done = _kr_run(kr)
    assert done.returncode == 0, done.stderr
    assert "locked" not in done.stdout and (kr["out"] / "completion.json").is_file()


def _wait_for(path: Path, run: subprocess.Popen) -> None:
    deadline = time.time() + 20
    while not path.exists():
        assert time.time() < deadline and run.poll() is None, f"{path.name} never appeared"
        time.sleep(0.05)


def test_kr_prepare_two_overlapping_runs_only_one_proceeds(kr):
    """The chain event and the fallback event start the same script: one exports, the other yields."""
    env = {**os.environ, **kr["env"], "FAKE_EXPORT_SLEEP": "3"}
    first = subprocess.Popen(["bash", str(KR_PREPARE), "--report-date", "2026-10-05"], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _wait_for(kr["log"].parent / "state" / "export-running", first)
    second = _kr_run(kr)
    assert second.returncode == 0 and "locked" in second.stdout
    _, err = first.communicate(timeout=60)
    assert first.returncode == 0, err
    assert len([c for c in _calls(kr) if c.startswith("export ")]) == 1
    assert (kr["out"] / "completion.json").is_file()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _all_dead(pids: list[int]) -> bool:
    deadline = time.time() + 10
    while any(_alive(pid) for pid in pids) and time.time() < deadline:
        time.sleep(0.1)
    return not any(_alive(pid) for pid in pids)


def test_kr_prepare_term_ends_the_whole_process_group_and_frees_the_lock(kr):
    """Cronicle's abort (TERM) must not leave the export's descendants running (10-06 incident)."""
    env = {**os.environ, **kr["env"], "FAKE_EXPORT_SLEEP": "120"}
    run = subprocess.Popen(["bash", str(KR_PREPARE), "--report-date", "2026-10-05"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    state = kr["log"].parent / "state"
    _wait_for(state / "export-running", run)
    pids = [int(x) for x in (state / "grandchildren").read_text().split()]
    assert pids and all(_alive(pid) for pid in pids)
    run.send_signal(signal.SIGTERM)
    out, err = run.communicate(timeout=30)
    assert run.returncode == 143, (out, err)
    assert "received TERM" in err
    assert _all_dead(pids), "a descendant of the aborted step is still running"
    # Nothing was prepared, and the lock is free again for the next run.
    assert not (kr["out"] / "completion.json").exists()
    again = _kr_run(kr)
    assert again.returncode == 0 and "locked" not in again.stdout


US_PREPARE_BRANCH = """    modeler.serving.us_daily)
      [ "${FAKE_PREP_RC:-0}" -eq 0 ] || exit "$FAKE_PREP_RC"
      touch "$FAKE_STATE/prepared"; echo "prepared: /x/manifest.json"; exit 0;;"""
US_SLOW_BRANCH = """    modeler.serving.us_daily)
      sleep 120 & echo "$!" >> "$FAKE_STATE/grandchildren"; echo "$$" >> "$FAKE_STATE/grandchildren"
      touch "$FAKE_STATE/prepare-running"; wait;;"""


def test_us_prepare_term_ends_the_prepare_group_and_the_lock_serializes_runs(serving, tmp_path):
    fake = tmp_path / "fakebin" / "python"
    assert US_PREPARE_BRANCH in FAKE_PYTHON
    fake.write_text(FAKE_PYTHON.replace(US_PREPARE_BRANCH, US_SLOW_BRANCH))
    env = {**os.environ, **serving["env"]}
    run = subprocess.Popen(["bash", str(PREPARE), "--run-date", "2026-09-30"], env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    state = serving["log"].parent / "state"
    _wait_for(state / "prepare-running", run)
    second = _run(PREPARE, ["--run-date", "2026-09-30"], serving["env"])
    assert second.returncode == 0 and "locked" in second.stdout
    pids = [int(x) for x in (state / "grandchildren").read_text().split()]
    run.send_signal(signal.SIGTERM)
    out, err = run.communicate(timeout=30)
    assert run.returncode == 143, (out, err)
    assert _all_dead(pids)


def test_briefing_stage_execs_the_python_wrapper_without_a_shell_layer():
    """No shell stays between Cronicle and Python: TERM reaches the Python wrapper itself, which
    ends the runner's process group (``daily_coordinator.run_group``)."""
    assert '\nexec "$python" -m modeler.serving.daily_wrapper' in STAGE.read_text()
