#!/usr/bin/env bash
# Daily 17:30 KST native US prepare for the briefing.
#
#   us-prepare.sh [--run-date YYYY-MM-DD]     (default: today in Asia/Seoul)
#   SERVING_ROOT  serving root (default /home/whi/apps/market-briefing/serving)
#   BOOTSTRAP_PYTHON  date/hash helper only (default python3)
#
# A = expected US session of run date + 1 day (us_expected session), never guessed
# from data.  Exit codes:
#   0  prepared now, or A already prepared and serving-eligible (skipped)
#   2  usage   10 serving root/config unreadable   11 parity evidence sha256 differs from pins.json
#   12 A could not be computed   20 A is not in the lake yet (prices_daily/universe_daily max date < A)
#   21 prepare finished but no serving-eligible native for A   other: prepare's own code (124 = timeout)
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
from datetime import date, datetime
mode, a = sys.argv[1], date.fromisoformat(sys.argv[2])
if mode == "data-ready":
    import polars as pl
    from modeler.us.lake import UsLake
    lake = UsLake.resolve()
    short = False
    for table in ("prices_daily", "universe_daily"):
        snap = lake.latest_snapshot(table)
        latest = lake.scan_raw(table, snap).select(pl.col("date").max()).collect().item()
        if isinstance(latest, datetime):
            latest = latest.date()
        behind = latest is None or latest < a
        print(f"  {table}: latest snapshot {snap}, max date {latest}" + (f"  <-- behind A={a}" if behind else ""),
              file=sys.stderr)
        short = short or behind
    sys.exit(3 if short else 0)
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

if ready=$("$python" -c "$helper" prepared-ok "$A" "$evidence_sha"); then
  log "A=$A already prepared and serving-eligible, skipping: $ready"
  exit 0
fi

log "checking that the lake has A=$A"
rc=0
"$python" -c "$helper" data-ready "$A" || rc=$?
if [ "$rc" -eq 3 ]; then
  fail 20 "data not ready: the lake has no A=$A session yet (prices_daily or universe_daily max date < A; see lines above). Run after sdc_daily_us, derive and universe_incremental finished."
elif [ "$rc" -ne 0 ]; then
  fail 20 "could not read the lake under $STOCK_DATA_ROOT/us (exit $rc)"
fi

log "preparing A=$A (timeout 1800s, cpus $cpus)"
rc=0
out=$(timeout 1800 taskset -c "$cpus" "$python" -m modeler.serving.us_daily prepare \
  --as-of "$A" --raw-feature-parity-status score_equivalent \
  --raw-feature-parity-evidence "$evidence") || rc=$?
[ -z "$out" ] || echo "$out"
[ "$rc" -eq 0 ] || fail "$rc" "prepare failed for A=$A (exit $rc)"

ready=$("$python" -c "$helper" prepared-ok "$A" "$evidence_sha") \
  || fail 21 "prepare finished but no serving-eligible native input exists for A=$A"
log "prepared: $ready"
