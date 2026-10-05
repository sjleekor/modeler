#!/usr/bin/env bash
# Nightly KR native prepare for the briefing of report date D (04:10 KST).
#
#   kr-prepare.sh [--report-date YYYY-MM-DD]   (default: today in Asia/Seoul)
#   SERVING_ROOT        serving root (default /home/whi/apps/market-briefing/serving)
#   KR_STOCK_DATA_ROOT  lake holding kr/raw and kr/derived (default /home/whi/data/stock_data)
#   SDC_BIN             collector wrappers (default /home/whi/apps/sdc/bin)
#   KR_PREPARE_CONSISTENT_SNAPSHOT  1 (default) exports every table from one PostgreSQL snapshot
#   KR_PREPARE_GATE_UNTIL           HH:MM KST the export gate waits until (default 07:30)
#   KR_PREPARE_MAX_TEMP_SIZE        DuckDB spill cap of kr_live_prepare (default 30GB)
#   BOOTSTRAP_PYTHON    reads ops.json only (default python3)
#
# K = the KR session before D, SNAP = D, input cutoff = D 09:30 KST.  Steps, each skipped
# when its completion marker already exists, so a rerun picks up where the last one stopped:
#   1. export gate   $SDC_BIN/kr-export-wait-ready.sh (collection for K finished)
#   2. raw export    $SDC_BIN/kr-raw-parquet-export.sh -> kr/raw/raw_postgres/snapshot_date=SNAP
#   3. live marts    modeler.serving.kr_live_prepare (full profile) -> kr/derived/feature/snapshot_date=SNAP
#   4. native        modeler.serving.kr_prepare -> prepared/kr/score_date=K/prep_id=SNAP
# Exit codes:
#   0  prepared now, already prepared, or D is not a KR session (skipped)
#   2  usage   10 serving root/config unreadable   12 D or K outside the KR calendar
#   30 export gate: input for K not ready by the deadline   31 export gate: collection blocked
#   other: the failing step's own code (124 = timeout)
set -euo pipefail
umask 027

report_date=""
if [ $# -gt 0 ]; then
  if [ $# -eq 2 ] && [ "$1" = "--report-date" ] && [[ "$2" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    report_date=$2
  else
    echo "usage: kr-prepare.sh [--report-date YYYY-MM-DD]" >&2; exit 2
  fi
fi

SERVING_ROOT=${SERVING_ROOT:-/home/whi/apps/market-briefing/serving}
ROOT=${KR_STOCK_DATA_ROOT:-/home/whi/data/stock_data}
SDC_BIN=${SDC_BIN:-/home/whi/apps/sdc/bin}
consistent=${KR_PREPARE_CONSISTENT_SNAPSHOT:-1}
gate_until=${KR_PREPARE_GATE_UNTIL:-07:30}
max_temp=${KR_PREPARE_MAX_TEMP_SIZE:-30GB}
boot=${BOOTSTRAP_PYTHON:-python3}
ops="$SERVING_ROOT/config/ops.json"
log() { echo "kr-prepare: [$(TZ=Asia/Seoul date '+%F %T')] $*"; }
fail() { local code=$1; shift; echo "kr-prepare: $*" >&2; exit "$code"; }

case "$consistent" in 0|1) ;; *) fail 2 "KR_PREPARE_CONSISTENT_SNAPSHOT must be 0 or 1";; esac
[[ "$gate_until" =~ ^[0-9]{2}:[0-9]{2}$ ]] || fail 2 "KR_PREPARE_GATE_UNTIL must be HH:MM"

[ -f "$ops" ] || fail 10 "ops.json not found: $ops"
fields=$("$boot" - "$ops" "$report_date" "$gate_until" <<'PY'
import json, sys
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo
seoul = ZoneInfo("Asia/Seoul")
ops = json.load(open(sys.argv[1]))
now = datetime.now(seoul)
d = date.fromisoformat(sys.argv[2]) if sys.argv[2] else now.date()
until = datetime.combine(d, time.fromisoformat(sys.argv[3]), seoul)
print(Path(ops["release_manifest"]).parent)
print(ops["python"])
print(ops["kr_calendar"])
print(ops["prepared_root"])
print(d.isoformat())
print(max(0, int((until - now).total_seconds())))
PY
) || fail 10 "cannot read $ops"
release_root=$(printf '%s\n' "$fields" | sed -n 1p)
python=$(printf '%s\n' "$fields" | sed -n 2p)
calendar=$(printf '%s\n' "$fields" | sed -n 3p)
prepared_root=$(printf '%s\n' "$fields" | sed -n 4p)
D=$(printf '%s\n' "$fields" | sed -n 5p)
deadline=$(printf '%s\n' "$fields" | sed -n 6p)
[ -x "$python" ] && [ -d "$release_root/src" ] || fail 10 "python or release src missing"
[ -f "$calendar" ] || fail 10 "KR calendar missing: $calendar"

export PYTHONPATH="$release_root/src"
export PYTHONDONTWRITEBYTECODE=1
cpus=${TASKSET_CPUS:-0,1}

# Same calendar object the selector uses: K = calendars["KR"].previous_session(D).
calendar_helper=$(cat <<'PY'
import json, sys
from datetime import date
from modeler.serving.calendars import SessionCalendar
cal = SessionCalendar.from_manifest(json.loads(open(sys.argv[2]).read()))
d = date.fromisoformat(sys.argv[3])
session, k = cal.is_session(d), cal.previous_session(d)
if session is None or k is None:
    sys.exit(3)
print("session" if session else "closed")
print(k.isoformat())
PY
)
rc=0
dates=$("$python" -c "$calendar_helper" kr-dates "$calendar" "$D") || rc=$?
[ "$rc" -eq 0 ] || fail 12 "D=$D or its previous session is outside the KR calendar $calendar (exit $rc)"
if [ "$(printf '%s\n' "$dates" | sed -n 1p)" != session ]; then
  log "D=$D is not a KR session; nothing to prepare"
  exit 0
fi
K=$(printf '%s\n' "$dates" | sed -n 2p)
[[ "$K" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || fail 12 "unexpected K value: $K"
SNAP=$D
CUTOFF="${D}T09:30:00+09:00"
OUT="$prepared_root/kr/score_date=$K/prep_id=$SNAP"
raw_marker="$ROOT/kr/raw/raw_postgres/snapshot_date=$SNAP/source=sj2_remote/_manifests/_SUCCESS.json"
feature_marker="$ROOT/kr/derived/feature/snapshot_date=$SNAP/source=sj2_remote/_manifests/_SUCCESS.json"
log "D=$D K=$K SNAP=$SNAP cutoff=$CUTOFF release=$release_root"

if [ -f "$OUT/completion.json" ]; then
  log "already prepared: $OUT"
  exit 0
fi

# DuckDB spills next to the working directory for in-memory connections; the release is read-only.
work="$SERVING_ROOT/logs/kr-prepare/D=$D"
mkdir -p "$work"
cd "$work"

if [ -f "$raw_marker" ]; then
  log "1-2/4 raw export already sealed, skipping gate and export: $raw_marker"
else
  log "1/4 export gate for K=$K (waits ${deadline}s, until $D $gate_until KST)"
  rc=0
  "$SDC_BIN/kr-export-wait-ready.sh" --feature-asof-date "$K" --deadline-seconds "$deadline" || rc=$?
  case "$rc" in
    0) ;;
    75) fail 30 "input for K=$K not ready by $D $gate_until KST; collection chains have not finished (see gate evidence)";;
    1) fail 31 "collection for K=$K is blocked (a run failed or the input moved past K); rerun the failed collection, then this event";;
    *) fail "$rc" "export gate errored (exit $rc)";;
  esac
  export_args=(--snapshot-date "$SNAP")
  [ "$consistent" = 0 ] || export_args+=(--consistent-snapshot)
  log "2/4 raw export ${export_args[*]}"
  rc=0
  timeout 5400 "$SDC_BIN/kr-raw-parquet-export.sh" "${export_args[@]}" || rc=$?
  [ "$rc" -eq 0 ] || fail "$rc" "raw export failed for SNAP=$SNAP (exit $rc)"
  [ -f "$raw_marker" ] || fail 1 "raw export finished but left no marker: $raw_marker"
fi

if [ -f "$feature_marker" ]; then
  log "3/4 live marts already sealed, skipping: $feature_marker"
else
  log "3/4 kr_live_prepare (full profile, temp cap $max_temp, cpus $cpus)"
  rc=0
  timeout 3600 nice -n 10 taskset -c "$cpus" "$python" -m modeler.serving.kr_live_prepare \
    --snapshot-date "$SNAP" --feature-asof-date "$K" --input-cutoff "$CUTOFF" \
    --stock-data-root "$ROOT" --profile full --max-temp-size "$max_temp" || rc=$?
  [ "$rc" -eq 0 ] || fail "$rc" "kr_live_prepare failed for SNAP=$SNAP K=$K (exit $rc)"
fi

log "4/4 kr_prepare -> $OUT"
rc=0
timeout 1800 nice -n 10 taskset -c "$cpus" "$python" -m modeler.serving.kr_prepare \
  --snapshot-date "$SNAP" --feature-asof-date "$K" --input-cutoff "$CUTOFF" \
  --output-dir "$OUT" --stock-data-root "$ROOT" || rc=$?
[ "$rc" -eq 0 ] || fail "$rc" "kr_prepare failed for SNAP=$SNAP K=$K (exit $rc)"
[ -f "$OUT/completion.json" ] || fail 1 "kr_prepare finished but left no completion.json in $OUT"
log "prepared: $OUT"
