#!/usr/bin/env bash
# Daily native US prepare for the briefing (17:30 KST, and again the next morning).
#
#   us-prepare.sh [--run-date YYYY-MM-DD]     (default: today in Asia/Seoul)
#   SERVING_ROOT  serving root (default /home/whi/apps/market-briefing/serving)
#   BOOTSTRAP_PYTHON  date/hash helper only (default python3)
#
# A = expected US session of run date + 1 day (us_expected session), never guessed from data.
# When the lake does not cover A yet (the 2026-10-01 case), the script no longer stops: it prepares
# A' = the newest XNYS session <= A that prices_daily and universe_daily both cover, and says so.
# The selector then marks that input stale with its lag; five sessions or more behind, the report
# shows no ranking (04_run_without_waiting, change 4).  A later run, after the lake caught up,
# prepares A itself; the older A' native stays as it is.
#
# One flock (locks/us-prepare.lock) covers the run; a second run finds it taken and exits 0.  The
# prepare runs as the leader of its own process group, and TERM/INT/HUP end that whole group.
# Exit codes:
#   0  prepared now (A or A'), A/A' already prepared and serving-eligible (skipped), or locked
#   2  usage   10 serving root/config unreadable or flock missing
#   11 parity evidence sha256 differs from pins.json
#   12 A could not be computed   20 the lake covers no completed session at all (or cannot be read)
#   21 prepare finished but no serving-eligible native for A'   other: prepare's own code (124 = timeout)
set -euo pipefail
umask 027

run_date=""
if [ $# -gt 0 ]; then
  if [ $# -eq 2 ] && [ "$1" = "--run-date" ] && [[ "$2" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    run_date=$2
  else
    echo "usage: us-prepare.sh [--run-date YYYY-MM-DD]" >&2; exit 2
  fi
fi

SERVING_ROOT=${SERVING_ROOT:-/home/whi/apps/market-briefing/serving}
boot=${BOOTSTRAP_PYTHON:-python3}
ops="$SERVING_ROOT/config/ops.json"
pins="$SERVING_ROOT/config/pins.json"
log() { echo "us-prepare: $*"; }
fail() { local code=$1; shift; echo "us-prepare: $*" >&2; exit "$code"; }

# ---- process groups and signals (same block in kr-prepare.sh) ------------------------------------
child_pid=""
run_step() {
  set -m
  "$@" &
  child_pid=$!
  set +m
  local rc=0
  wait "$child_pid" || rc=$?
  child_pid=""
  return "$rc"
}
stop_child_group() {
  [ -n "$child_pid" ] || return 0
  kill -TERM -- "-$child_pid" 2>/dev/null || true
  local waited=0
  while kill -0 -- "-$child_pid" 2>/dev/null && [ "$waited" -lt 20 ]; do
    sleep 0.5; waited=$((waited + 1))
  done
  kill -KILL -- "-$child_pid" 2>/dev/null || true
  child_pid=""
}
on_signal() {
  trap '' TERM INT HUP
  echo "us-prepare: received $1, ending the running step's process group" >&2
  stop_child_group
  exit "$2"
}
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
trap 'on_signal HUP 129' HUP

[ -f "$ops" ] && [ -f "$pins" ] || fail 10 "ops.json or pins.json missing under $SERVING_ROOT/config"
fields=$("$boot" - "$ops" "$pins" "$run_date" <<'PY'
import json, sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
ops = json.load(open(sys.argv[1])); pins = json.load(open(sys.argv[2]))
today = date.fromisoformat(sys.argv[3]) if sys.argv[3] else datetime.now(ZoneInfo("Asia/Seoul")).date()
print(Path(ops["release_manifest"]).parent)
print(ops["python"])
print(pins["parity_evidence"]["path"])
print(pins["parity_evidence"]["sha256"])
print(today.isoformat())
print((today + timedelta(days=1)).isoformat())
PY
) || fail 10 "cannot read ops.json/pins.json"
release_root=$(printf '%s\n' "$fields" | sed -n 1p)
python=$(printf '%s\n' "$fields" | sed -n 2p)
evidence=$(printf '%s\n' "$fields" | sed -n 3p)
evidence_sha=$(printf '%s\n' "$fields" | sed -n 4p)
today=$(printf '%s\n' "$fields" | sed -n 5p)
report_date=$(printf '%s\n' "$fields" | sed -n 6p)
[ -x "$python" ] && [ -d "$release_root/src" ] || fail 10 "python or release src missing"

command -v flock >/dev/null 2>&1 || fail 10 "flock not found"
lock_file="$SERVING_ROOT/locks/us-prepare.lock"
mkdir -p "$SERVING_ROOT/locks"
exec 9>"$lock_file"
if ! flock -n 9; then
  log "locked: another us-prepare run holds $lock_file; this run does nothing"
  exit 0
fi

actual_sha=$("$boot" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$evidence") \
  || fail 11 "cannot read parity evidence: $evidence"
[ "$actual_sha" = "$evidence_sha" ] || fail 11 "parity evidence sha256 $actual_sha differs from pins.json $evidence_sha"

cd "$release_root"
export PYTHONPATH="$release_root/src"
export PYTHONDONTWRITEBYTECODE=1
export POLARS_MAX_THREADS=2
cpus=${TASKSET_CPUS:-0,1}

A=$("$python" -m modeler.serving.us_expected session --report-date "$report_date") \
  || fail 12 "cannot compute A for report date $report_date (see error above)"
[[ "$A" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || fail 12 "unexpected A value: $A"
log "run_date=$today report_date=$report_date A=$A"

helper=$(cat <<'PY'
import json, sys
from datetime import date
mode, a = sys.argv[1], date.fromisoformat(sys.argv[2])
if mode == "resolve":
    # A' = the newest session <= A that the pinned prices and universe snapshots both cover.
    from modeler.serving import us_daily as u
    found = u.latest_complete_session(a)
    print(f"  requested A={found['requested']} prices max {found['prices_max']} "
          f"universe max {found['universe_max']} -> resolved {found['resolved']}", file=sys.stderr)
    if found["resolved"] is None:
        sys.exit(3)
    print(found["resolved"])
    sys.exit(0)
if mode == "prepared-ok":
    from modeler.serving import us_daily as u
    from modeler.serving.orchestration import us_native_block_reason
    root = u.DataRoot.resolve(market="us")
    base = root.output / u.SCORING_VERSION / "prepared" / f"score_date={a}"
    code_hash = u.code_tree_hash()
    for manifest in sorted(base.glob("prep_id=*/manifest.json")):
        try:
            native = json.loads(manifest.read_text())
            u._read_native_completion(manifest)
        except (OSError, ValueError, KeyError):
            continue
        if (native.get("market") == "US" and native.get("feature_asof_date") == a.isoformat()
                and native.get("raw_feature_parity_status") == "score_equivalent"
                and native.get("serving_eligible") is True and native.get("diagnostic_only") is False
                and native.get("code_hash") == code_hash
                and native.get("raw_feature_parity_evidence_sha256") == sys.argv[3]
                and us_native_block_reason(native) is None):
            print(manifest)
            sys.exit(0)
    sys.exit(1)
sys.exit(2)
PY
)

export STOCK_DATA_ROOT="$SERVING_ROOT/stock_data"

log "checking what the lake covers (requested A=$A)"
rc=0
A_prime=$("$python" -c "$helper" resolve "$A") || rc=$?
if [ "$rc" -eq 3 ]; then
  fail 20 "the lake covers no completed US session (prices_daily or universe_daily is empty; see line above)"
elif [ "$rc" -ne 0 ]; then
  fail 20 "could not read the lake under $STOCK_DATA_ROOT/us (exit $rc)"
fi
[[ "$A_prime" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || fail 20 "unexpected resolved session: $A_prime"
if [ "$A_prime" != "$A" ]; then
  log "WARNING the lake does not cover A=$A yet; preparing A'=$A_prime instead (the selector marks it stale with its lag)"
fi

if ready=$("$python" -c "$helper" prepared-ok "$A_prime" "$evidence_sha"); then
  log "A'=$A_prime already prepared and serving-eligible, skipping: $ready"
  exit 0
fi

log "preparing A'=$A_prime (timeout 1800s, cpus $cpus)"
rc=0
out_file=$(mktemp "${TMPDIR:-/tmp}/us-prepare.XXXXXX")
run_step timeout 1800 taskset -c "$cpus" "$python" -m modeler.serving.us_daily prepare \
  --as-of "$A_prime" --raw-feature-parity-status score_equivalent \
  --raw-feature-parity-evidence "$evidence" > "$out_file" || rc=$?
out=$(cat "$out_file"); rm -f "$out_file"
[ -z "$out" ] || echo "$out"
[ "$rc" -eq 0 ] || fail "$rc" "prepare failed for A'=$A_prime (exit $rc)"

ready=$("$python" -c "$helper" prepared-ok "$A_prime" "$evidence_sha") \
  || fail 21 "prepare finished but no serving-eligible native input exists for A'=$A_prime"
log "prepared: $ready"
