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
#   KR_PREPARE_RETENTION        1 (default) runs step 5; 0 skips it entirely
#   KR_PREPARE_KEEP_SNAPSHOTS   positive integer (default 1): the N newest raw and live-mart
#                               snapshots stay, SNAP included; 0 is invalid (exit 2)
#   BOOTSTRAP_PYTHON    reads ops.json only (default python3)
#
# K = the KR session before D, SNAP = D, input cutoff = D 09:30 KST.  Steps, each skipped
# when its completion marker already exists, so a rerun picks up where the last one stopped:
#   1. export gate   $SDC_BIN/kr-export-wait-ready.sh (collection for K finished)
#   2. raw export    $SDC_BIN/kr-raw-parquet-export.sh -> kr/raw/raw_postgres/snapshot_date=SNAP
#   3. live marts    modeler.serving.kr_live_prepare (full profile) -> kr/derived/feature/snapshot_date=SNAP
#   4. native        modeler.serving.kr_prepare -> prepared/kr/score_date=K/prep_id=SNAP
#   5. retention     only after step 4 left completion.json, and also on "already prepared" (a
#                    rerun finishes a retention that failed earlier). For every
#                    kr/raw/raw_postgres/, kr/derived/feature/ and kr/derived/metric/ snapshot_date=X older than SNAP
#                    (beyond the KEEP newest): copy source=*/_manifests to
#                    kr/raw/_manifest_archive/ or kr/derived/_manifest_archive/feature/, then
#                    delete the snapshot dir (its _manifests first, so no _SUCCESS.json outlives
#                    the parquet). Also removes per-run kr/derived/_duckdb_tmp dirs of days
#                    before SNAP (a failed run keeps its dir; its partial profile is archived).
#                    Never touches dates >= SNAP, never follows symlinks, and a failure here
#                    (e.g. a root-owned old snapshot) only logs a warning: it never changes the
#                    exit code.
# Exit codes:
#   0  prepared now, already prepared, or D is not a KR session (skipped); retention trouble
#      does not count
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
retention=${KR_PREPARE_RETENTION:-1}
keep=${KR_PREPARE_KEEP_SNAPSHOTS:-1}
boot=${BOOTSTRAP_PYTHON:-python3}
ops="$SERVING_ROOT/config/ops.json"
log() { echo "kr-prepare: [$(TZ=Asia/Seoul date '+%F %T')] $*"; }
fail() { local code=$1; shift; echo "kr-prepare: $*" >&2; exit "$code"; }

case "$consistent" in 0|1) ;; *) fail 2 "KR_PREPARE_CONSISTENT_SNAPSHOT must be 0 or 1";; esac
[[ "$gate_until" =~ ^[0-9]{2}:[0-9]{2}$ ]] || fail 2 "KR_PREPARE_GATE_UNTIL must be HH:MM"
case "$retention" in 0|1) ;; *) fail 2 "KR_PREPARE_RETENTION must be 0 or 1";; esac
[[ "$keep" =~ ^[1-9][0-9]*$ ]] || fail 2 "KR_PREPARE_KEEP_SNAPSHOTS must be a positive integer"

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

# ---- step 5: retention ------------------------------------------------------------------
# Every failure is collected in $retention_left and reported once; nothing here exits nonzero.
retention_left=()

# True when no path component from $ROOT down to $1 is a symlink.
no_symlink_path() {
  local p=$1
  while [ "$p" != "$ROOT" ] && [ "$p" != / ] && [ -n "$p" ]; do
    [ ! -L "$p" ] || return 1
    p=$(dirname "$p")
  done
  return 0
}

# Copies every source=*/_manifests of snapshot dir $1 below $2 unless an identical copy is there.
# Returns 1 on any problem (an archive that differs is never overwritten).
archive_manifests() {
  local snap=$1 dst=$2 src m target
  for src in "$snap"/source=*; do
    [ -e "$src" ] || continue
    { [ -d "$src" ] && [ ! -L "$src" ]; } || return 1
    m=$src/_manifests
    [ -e "$m" ] || continue
    { [ -d "$m" ] && [ ! -L "$m" ]; } || return 1
    [ -z "$(find "$m" -type l -print -quit)" ] || return 1
    target=$dst/${src##*/}/_manifests
    if [ -e "$target" ]; then
      diff -rq "$m" "$target" >/dev/null 2>&1 && continue
      return 1
    fi
    mkdir -p "$dst/${src##*/}" || return 1
    rm -rf "$target.tmp"
    if cp -RP "$m" "$target.tmp" && diff -rq "$m" "$target.tmp" >/dev/null 2>&1 \
        && mv "$target.tmp" "$target"; then
      :
    else
      rm -rf "$target.tmp"
      return 1
    fi
  done
  return 0
}

# retain_parent LABEL PARENT ARCHIVE_PARENT: deletes the snapshot_date=X dirs of PARENT older than
# SNAP and beyond the $keep newest, after archiving their manifests under ARCHIVE_PARENT.
retain_parent() {
  local label=$1 parent=$2 archive=$3 entry name d i size
  [ -e "$parent" ] || return 0
  if ! { [ -d "$parent" ] && no_symlink_path "$parent"; }; then
    retention_left+=("$label: $parent is a symlink or not a directory, left alone"); return 0
  fi
  if ! no_symlink_path "$archive"; then
    retention_left+=("$label: archive path $archive passes through a symlink, left alone"); return 0
  fi
  local dates=()
  for entry in "$parent"/snapshot_date=*; do
    [ -e "$entry" ] || [ -L "$entry" ] || continue
    name=${entry##*/}
    [[ "$name" =~ ^snapshot_date=([0-9]{4}-[0-9]{2}-[0-9]{2})$ ]] && dates+=("${BASH_REMATCH[1]}")
  done
  # The glob is sorted ascending: position from the end is the rank among the newest.
  for ((i = 0; i < ${#dates[@]}; i++)); do
    d=${dates[i]}
    [[ "$d" < "$SNAP" ]] || continue
    (( ${#dates[@]} - i <= keep )) && continue
    entry=$parent/snapshot_date=$d
    if [ -L "$entry" ] || [ ! -d "$entry" ]; then
      retention_left+=("$label snapshot_date=$d: symlink or not a directory, left alone"); continue
    fi
    if ! archive_manifests "$entry" "$archive/snapshot_date=$d"; then
      retention_left+=("$label snapshot_date=$d: manifests could not be archived to $archive/snapshot_date=$d (missing, different or unreadable), not deleted")
      continue
    fi
    size=$(du -sh "$entry" 2>/dev/null | cut -f1) || size=
    # _manifests first: a half-deleted dir must not keep a _SUCCESS.json for parquet that is gone.
    if rm -rf "$entry"/source=*/_manifests 2>/dev/null && rm -rf "$entry" 2>/dev/null; then
      log "retention: deleted $label snapshot_date=$d (${size:-size unknown}); manifests archived -> $archive/snapshot_date=$d"
    else
      retention_left+=("$label snapshot_date=$d: delete failed (permissions?), partly or fully remains in $entry")
    fi
  done
  return 0
}

# Per-run DuckDB spill dirs of days before SNAP (a failed run keeps its dir for inspection).
retain_duckdb_tmp() {
  local parent=$ROOT/kr/derived/_duckdb_tmp archive=$ROOT/kr/derived/_manifest_archive/_duckdb_tmp entry name size
  [ -e "$parent" ] || return 0
  if ! { [ -d "$parent" ] && no_symlink_path "$parent" && no_symlink_path "$archive"; }; then
    retention_left+=("spill: $parent or its archive path is a symlink, left alone"); return 0
  fi
  for entry in "$parent"/*; do
    [ -e "$entry" ] || [ -L "$entry" ] || continue
    name=${entry##*/}
    [[ "$name" =~ ^([0-9]{4}-[0-9]{2}-[0-9]{2})_[0-9]{8}T[0-9]{6}Z_[0-9]+$ ]] || continue
    [[ "${BASH_REMATCH[1]}" < "$SNAP" ]] || continue
    if [ -L "$entry" ] || [ ! -d "$entry" ]; then
      retention_left+=("spill $name: symlink or not a directory, left alone"); continue
    fi
    if [ -f "$entry/build_profile.partial.json" ] && ! { [ ! -L "$entry/build_profile.partial.json" ] \
        && mkdir -p "$archive/$name" && cp -P "$entry/build_profile.partial.json" "$archive/$name/"; }; then
      retention_left+=("spill $name: partial profile could not be archived, not deleted"); continue
    fi
    size=$(du -sh "$entry" 2>/dev/null | cut -f1) || size=
    if rm -rf "$entry" 2>/dev/null; then
      log "retention: deleted leftover spill dir $name (${size:-size unknown})"
    else
      retention_left+=("spill $name: delete failed, remains in $entry")
    fi
  done
  return 0
}

run_retention() {
  if [ "$retention" = 0 ]; then
    log "5/5 retention disabled (KR_PREPARE_RETENTION=0)"
    return 0
  fi
  log "5/5 retention: keep newest $keep snapshot(s), SNAP=$SNAP included"
  retention_left=()
  retain_parent raw "$ROOT/kr/raw/raw_postgres" "$ROOT/kr/raw/_manifest_archive"
  retain_parent feature "$ROOT/kr/derived/feature" "$ROOT/kr/derived/_manifest_archive/feature"
  # stock_metric_fact persisted by kr_live_prepare (register_derived_marts); no _manifests to keep.
  retain_parent metric "$ROOT/kr/derived/metric" "$ROOT/kr/derived/_manifest_archive/metric"
  retain_duckdb_tmp
  if [ "${#retention_left[@]}" -gt 0 ]; then
    echo "kr-prepare: WARNING retention incomplete, ${#retention_left[@]} item(s) remain (prepare itself succeeded):" >&2
    printf 'kr-prepare:   - %s\n' "${retention_left[@]}" >&2
  fi
  return 0
}

if [ -f "$OUT/completion.json" ]; then
  log "already prepared: $OUT"
  # Idempotent: a rerun finishes a retention that failed earlier (or predates this step).
  run_retention || true
  exit 0
fi

# DuckDB spills next to the working directory for in-memory connections; the release is read-only.
work="$SERVING_ROOT/logs/kr-prepare/D=$D"
mkdir -p "$work"
cd "$work"

if [ -f "$raw_marker" ]; then
  log "1-2/5 raw export already sealed, skipping gate and export: $raw_marker"
else
  log "1/5 export gate for K=$K (waits ${deadline}s, until $D $gate_until KST)"
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
  log "2/5 raw export ${export_args[*]}"
  rc=0
  timeout 5400 "$SDC_BIN/kr-raw-parquet-export.sh" "${export_args[@]}" || rc=$?
  [ "$rc" -eq 0 ] || fail "$rc" "raw export failed for SNAP=$SNAP (exit $rc)"
  [ -f "$raw_marker" ] || fail 1 "raw export finished but left no marker: $raw_marker"
fi

if [ -f "$feature_marker" ]; then
  log "3/5 live marts already sealed, skipping: $feature_marker"
else
  log "3/5 kr_live_prepare (full profile, temp cap $max_temp, cpus $cpus)"
  rc=0
  timeout 3600 nice -n 10 taskset -c "$cpus" "$python" -m modeler.serving.kr_live_prepare \
    --snapshot-date "$SNAP" --feature-asof-date "$K" --input-cutoff "$CUTOFF" \
    --stock-data-root "$ROOT" --profile full --max-temp-size "$max_temp" || rc=$?
  [ "$rc" -eq 0 ] || fail "$rc" "kr_live_prepare failed for SNAP=$SNAP K=$K (exit $rc)"
fi

log "4/5 kr_prepare -> $OUT"
rc=0
timeout 1800 nice -n 10 taskset -c "$cpus" "$python" -m modeler.serving.kr_prepare \
  --snapshot-date "$SNAP" --feature-asof-date "$K" --input-cutoff "$CUTOFF" \
  --output-dir "$OUT" --stock-data-root "$ROOT" || rc=$?
[ "$rc" -eq 0 ] || fail "$rc" "kr_prepare failed for SNAP=$SNAP K=$K (exit $rc)"
[ -f "$OUT/completion.json" ] || fail 1 "kr_prepare finished but left no completion.json in $OUT"
log "prepared: $OUT"
run_retention || true
