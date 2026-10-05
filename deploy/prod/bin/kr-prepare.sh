#!/usr/bin/env bash
# Daily KR native prepare for the briefing of report date D.  It does not wait for anything.
#
#   kr-prepare.sh [--report-date YYYY-MM-DD]   (default: today in Asia/Seoul)
#   SERVING_ROOT        serving root (default /home/whi/apps/market-briefing/serving)
#   KR_STOCK_DATA_ROOT  lake holding kr/raw and kr/derived (default /home/whi/data/stock_data)
#   SDC_BIN             collector wrappers (default /home/whi/apps/sdc/bin)
#   KR_PREPARE_CONSISTENT_SNAPSHOT  1 (default) exports every table from one PostgreSQL snapshot
#   KR_PREPARE_GATE                 judge (default) asks the collector gate once and only records its
#                                   verdict; off skips that call.  Neither ever waits or blocks.
#   KR_PREPARE_MAX_TEMP_SIZE        DuckDB spill cap of kr_live_prepare (default 30GB)
#   KR_PREPARE_RETENTION        1 (default) runs step 6; 0 skips it entirely
#   KR_PREPARE_KEEP_SNAPSHOTS   positive integer (default 1): the N newest raw and live-mart
#                               snapshots stay, SNAP included; 0 is invalid (exit 2)
#   BOOTSTRAP_PYTHON    reads ops.json only (default python3)
#
# K = the KR session before D, SNAP = D, input cutoff = D 09:30 KST.  R = the reference session the
# briefing is built for: K, or an earlier K' when K is not complete in the export (04_run_without_
# waiting, change 1).  Steps, each skipped when its completion marker already exists, so a rerun
# picks up where the last one stopped:
#   0. lock          one shared flock for the whole run, from here to the end of step 6.  The chain
#                    event and the fallback event both run this script; the second one finds the lock
#                    taken, logs "locked" and exits 0.  The lock is a file under the serving root
#                    (locks/kr-prepare.lock) and ends with the process, however it ends.
#   1. gate          $SDC_BIN/kr-export-wait-ready.sh --deadline-seconds 0: one look, advisory.  Its
#                    verdict and the DART run records go into the evidence; nothing is waited for.
#   2. raw export    $SDC_BIN/kr-raw-parquet-export.sh -> kr/raw/raw_postgres/snapshot_date=SNAP
#                    (whatever the database holds now, one consistent snapshot)
#   3. reference     modeler.serving.kr_reference: the completeness check runs on the exported snapshot
#                    itself (price count >= 97% of the previous session, every flow group reaches the
#                    session) and picks complete_K, fallback_K_prime or none.  Evidence:
#                    $SERVING_ROOT/logs/kr-prepare/D=D/reference-selection.json
#   4. live marts    modeler.serving.kr_live_prepare (full profile) for R, with --cut-to-asof when the
#                    export holds rows after R -> kr/derived/feature/snapshot_date=SNAP
#   5. native        modeler.serving.kr_prepare -> prepared/kr/score_date=R/prep_id=SNAP
#   6. retention     only after step 5 left completion.json, and also on "already prepared" (a
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
#
# Every step runs as the leader of its own process group.  TERM, INT or HUP (Cronicle aborting the
# job) ends that whole group, then the script; nothing keeps running below a dead wrapper.
# Exit codes:
#   0  prepared now (K or K'), already prepared, locked by another run, or D is not a KR session
#      (skipped); retention trouble does not count
#   2  usage   10 serving root/config unreadable or flock missing   12 D or K outside the KR calendar
#   32 no session within five sessions below K is complete: nothing was prepared, the selector
#      serves the newest older prepared input as stale
#   143/130/129  stopped by TERM/INT/HUP
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
gate_mode=${KR_PREPARE_GATE:-judge}
max_temp=${KR_PREPARE_MAX_TEMP_SIZE:-30GB}
retention=${KR_PREPARE_RETENTION:-1}
keep=${KR_PREPARE_KEEP_SNAPSHOTS:-1}
boot=${BOOTSTRAP_PYTHON:-python3}
ops="$SERVING_ROOT/config/ops.json"
log() { echo "kr-prepare: [$(TZ=Asia/Seoul date '+%F %T')] $*"; }
fail() { local code=$1; shift; echo "kr-prepare: $*" >&2; exit "$code"; }

# ---- process groups and signals (same block in us-prepare.sh) ----------------------------------
# Job control (set -m) makes each background job the leader of its own process group, so
# `kill -- -PID` reaches timeout, nice, taskset and the Python below them.  The step runs in the
# background and the script `wait`s, because bash runs a trap only between commands: a foreground
# step would postpone the handler until the step ended.
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
  echo "kr-prepare: [$(TZ=Asia/Seoul date '+%F %T')] received $1, ending the running step's process group" >&2
  stop_child_group
  exit "$2"
}
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT
trap 'on_signal HUP 129' HUP

case "$consistent" in 0|1) ;; *) fail 2 "KR_PREPARE_CONSISTENT_SNAPSHOT must be 0 or 1";; esac
case "$gate_mode" in judge|off) ;; *) fail 2 "KR_PREPARE_GATE must be judge or off";; esac
case "$retention" in 0|1) ;; *) fail 2 "KR_PREPARE_RETENTION must be 0 or 1";; esac
[[ "$keep" =~ ^[1-9][0-9]*$ ]] || fail 2 "KR_PREPARE_KEEP_SNAPSHOTS must be a positive integer"

[ -f "$ops" ] || fail 10 "ops.json not found: $ops"
fields=$("$boot" - "$ops" "$report_date" <<'PY'
import json, sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ops = json.load(open(sys.argv[1]))
d = date.fromisoformat(sys.argv[2]) if sys.argv[2] else datetime.now(ZoneInfo("Asia/Seoul")).date()
print(Path(ops["release_manifest"]).parent)
print(ops["python"])
print(ops["kr_calendar"])
print(ops["prepared_root"])
print(d.isoformat())
PY
) || fail 10 "cannot read $ops"
release_root=$(printf '%s\n' "$fields" | sed -n 1p)
python=$(printf '%s\n' "$fields" | sed -n 2p)
calendar=$(printf '%s\n' "$fields" | sed -n 3p)
prepared_root=$(printf '%s\n' "$fields" | sed -n 4p)
D=$(printf '%s\n' "$fields" | sed -n 5p)
[ -x "$python" ] && [ -d "$release_root/src" ] || fail 10 "python or release src missing"
[ -f "$calendar" ] || fail 10 "KR calendar missing: $calendar"

# ---- step 0: the shared lock ---------------------------------------------------------------
# One fixed file (not per date: retention deletes other dates' directories).  fd 9 stays open in
# every child, so a child that outlived this script would still hold the lock.
command -v flock >/dev/null 2>&1 || fail 10 "flock not found"
lock_file="$SERVING_ROOT/locks/kr-prepare.lock"
mkdir -p "$SERVING_ROOT/locks"
exec 9>"$lock_file"
if ! flock -n 9; then
  log "locked: another kr-prepare run holds $lock_file; this run does nothing (D=$D)"
  exit 0
fi

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
raw_marker="$ROOT/kr/raw/raw_postgres/snapshot_date=$SNAP/source=sj2_remote/_manifests/_SUCCESS.json"
feature_marker="$ROOT/kr/derived/feature/snapshot_date=$SNAP/source=sj2_remote/_manifests/_SUCCESS.json"
log "D=$D K=$K SNAP=$SNAP cutoff=$CUTOFF release=$release_root"

# ---- step 6: retention ------------------------------------------------------------------
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
    log "6/6 retention disabled (KR_PREPARE_RETENTION=0)"
    return 0
  fi
  log "6/6 retention: keep newest $keep snapshot(s), SNAP=$SNAP included"
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

# An earlier run (any R) already finished this day: prep_id=SNAP exists under some score_date.
shopt -s nullglob
done_marks=("$prepared_root"/kr/score_date=*/prep_id="$SNAP"/completion.json)
shopt -u nullglob
if [ "${#done_marks[@]}" -gt 0 ]; then
  log "already prepared: ${done_marks[0]%/completion.json}"
  # Idempotent: a rerun finishes a retention that failed earlier (or predates this step).
  run_retention || true
  exit 0
fi

# DuckDB spills next to the working directory for in-memory connections; the release is read-only.
work="$SERVING_ROOT/logs/kr-prepare/D=$D"
mkdir -p "$work"
cd "$work"
evidence="$work/reference-selection.json"

gate_exit=""
gate_file=""
if [ -f "$raw_marker" ]; then
  log "1-2/6 raw export already sealed, skipping gate and export: $raw_marker"
else
  if [ "$gate_mode" = judge ]; then
    gate_file="${KR_EXPORT_READY_EVIDENCE_DIR:-${STOCK_DATA_HOST_DIR:-/home/whi/data/stock_data}/kr/output/export_readiness}/K=$K.json"
    log "1/6 export gate for K=$K: one look, advisory (never waits, never blocks)"
    gate_exit=0
    run_step "$SDC_BIN/kr-export-wait-ready.sh" --feature-asof-date "$K" --deadline-seconds 0 || gate_exit=$?
    case "$gate_exit" in
      0) log "export gate: ready";;
      75) log "WARNING export gate: not ready (collection for K has not finished); the exported snapshot decides what is served";;
      1) log "WARNING export gate: blocked (a collection run failed or the input moved past K); the exported snapshot decides what is served";;
      *) log "WARNING export gate errored (exit $gate_exit); continuing without its verdict";;
    esac
  fi
  # A new export starts a new decision: an evidence file of an earlier (deleted) export is void.
  rm -f "$evidence" "$work/reference.out"
  export_args=(--snapshot-date "$SNAP")
  [ "$consistent" = 0 ] || export_args+=(--consistent-snapshot)
  log "2/6 raw export ${export_args[*]}"
  rc=0
  run_step timeout 5400 "$SDC_BIN/kr-raw-parquet-export.sh" "${export_args[@]}" || rc=$?
  [ "$rc" -eq 0 ] || fail "$rc" "raw export failed for SNAP=$SNAP (exit $rc)"
  [ -f "$raw_marker" ] || fail 1 "raw export finished but left no marker: $raw_marker"
fi

# ---- step 3: which session can this snapshot serve ----------------------------------------------
if [ ! -f "$evidence" ]; then
  log "3/6 reference session: completeness check on the exported snapshot (K=$K, up to five sessions below)"
  ref_args=(--snapshot-date "$SNAP" --report-date "$D" --k "$K" --calendar "$calendar"
            --stock-data-root "$ROOT" --output "$evidence")
  [ -z "$gate_exit" ] || ref_args+=(--gate-exit "$gate_exit" --gate-evidence "$gate_file")
  rc=0
  run_step timeout 600 nice -n 10 taskset -c "$cpus" "$python" -m modeler.serving.kr_reference "${ref_args[@]}" \
    > "$work/reference.out" || rc=$?
  case "$rc" in
    0) ;;
    32) ;;
    *) fail "$rc" "reference selection failed for SNAP=$SNAP K=$K (exit $rc)";;
  esac
else
  log "3/6 reference selection already made: $evidence"
  rc=0
  "$boot" -c 'import json,sys
e=json.load(open(sys.argv[1]))
if e["verdict"]=="none": sys.exit(32)
print(e["reference_date"]); print("cut" if e.get("cut_required") else "nocut")' "$evidence" > "$work/reference.out" || rc=$?
fi
if [ "$rc" -eq 32 ]; then
  log "WARNING no session from K=$K down to five sessions below is complete in the export; nothing prepared (evidence: $evidence)"
  run_retention || true
  echo "kr-prepare: no complete reference session; the selector serves the newest older prepared input as stale" >&2
  exit 32
fi
R=$(sed -n 1p "$work/reference.out")
cut_mode=$(sed -n 2p "$work/reference.out")
[[ "$R" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || fail 1 "unexpected reference date: $R"
case "$cut_mode" in cut|nocut) ;; *) fail 1 "unexpected cut mode: $cut_mode";; esac
OUT="$prepared_root/kr/score_date=$R/prep_id=$SNAP"
if [ "$R" = "$K" ]; then
  log "reference: complete_K R=$R"
else
  log "WARNING reference: fallback to K'=$R (K=$K is not complete in the export); the briefing says so"
fi

if [ -f "$feature_marker" ]; then
  log "4/6 live marts already sealed, skipping: $feature_marker"
else
  live_args=(--snapshot-date "$SNAP" --feature-asof-date "$R" --input-cutoff "$CUTOFF"
             --stock-data-root "$ROOT" --profile full --max-temp-size "$max_temp")
  [ "$cut_mode" = nocut ] || live_args+=(--cut-to-asof)
  log "4/6 kr_live_prepare R=$R ($cut_mode, full profile, temp cap $max_temp, cpus $cpus)"
  rc=0
  run_step timeout 3600 nice -n 10 taskset -c "$cpus" "$python" -m modeler.serving.kr_live_prepare "${live_args[@]}" || rc=$?
  [ "$rc" -eq 0 ] || fail "$rc" "kr_live_prepare failed for SNAP=$SNAP R=$R (exit $rc)"
fi

log "5/6 kr_prepare -> $OUT"
rc=0
run_step timeout 1800 nice -n 10 taskset -c "$cpus" "$python" -m modeler.serving.kr_prepare \
  --snapshot-date "$SNAP" --feature-asof-date "$R" --input-cutoff "$CUTOFF" \
  --output-dir "$OUT" --stock-data-root "$ROOT" --reference-evidence "$evidence" || rc=$?
[ "$rc" -eq 0 ] || fail "$rc" "kr_prepare failed for SNAP=$SNAP R=$R (exit $rc)"
[ -f "$OUT/completion.json" ] || fail 1 "kr_prepare finished but left no completion.json in $OUT"
log "prepared: $OUT"
run_retention || true
