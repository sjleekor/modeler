#!/usr/bin/env bash
# DRAFT (not deployed, no Cronicle event exists): daily shadow scoring of the next KR model.
#
#   kr-next-shadow.sh [--report-date YYYY-MM-DD]   (default: today in Asia/Seoul)
#   SHADOW_RELEASE      shadow release dir (release.json, src/, bundle/); required
#   SHADOW_ROOT         shadow root (default /home/whi/apps/market-briefing/shadow):
#                         records/        out_root of the daily records + records/ledger.jsonl
#                         locks/          kr-next-shadow.lock (this script's own lock)
#                         panels_in/D=<D>/panel.parquet   serving-mode panel of the signal day t
#                                         (+ panel_manifest.json), written by the shadow prepare
#                                         step (modeler.models._03_kr_next.shadow_prepare)
#                         skips/D=<D>.json  days left out on purpose (exit 33); outside records/ so that
#                                         `shadow verify` (which reads records/ only) never sees them
#   SERVING_ROOT        briefing serving root, read only for config/ops.json (python, kr_calendar,
#                       prepared_root)  (default /home/whi/apps/market-briefing/serving)
#   SHADOW_WAIT_UNTIL   HH:MM KST, stop waiting for the kr-prepare marker (default 09:20)
#   SHADOW_POLL_SECONDS poll interval (default 30)
#   SHADOW_PREPARE_CMD  optional command that builds panels_in/D=<D>/panel.parquet (gets D as $1);
#                       when unset the default is `python -m modeler.models._03_kr_next.shadow_prepare run`
#                       (derived views of the day's raw snapshot + the serving panel of t), which uses
#   SHADOW_RAW_LAKE     raw_postgres dir (default ${KR_STOCK_DATA_ROOT:-/home/whi/data/stock_data}/kr/raw/raw_postgres)
#   SHADOW_RAW_SOURCE   raw source name (default sj2_remote)
#   SHADOW_PREPARE_MEMORY_LIMIT  DuckDB memory_limit (default 2GB)   SHADOW_PREPARE_THREADS (default 4)
#   BOOTSTRAP_PYTHON    reads ops.json only (default python3)
#
# It waits for the kr-prepare completion marker prepared/kr/score_date=*/prep_id=<D>/completion.json
# (the end of the 04:00 chain / 06:30 fallback run) by polling the file.  ANY score_date counts: the
# marker only says "the raw export of D is finished".  Whether t is complete is not asked of
# kr-prepare (its rule also needs the flow groups and may change with the briefing); the prepare
# step checks it on this model's own input (price rows of t >= 0.97 x the session before t, in
# modeler.models._03_kr_next.shadow_prepare, frozen with the release).  It NEVER waits on, takes
# or inspects kr-prepare.lock: a held lock only means kr-prepare is still running, and the marker
# decides.  Own lock: locks/kr-next-shadow.lock, non-blocking; a second run logs "locked", exits 0.
# The last step is `exec`, so TERM/INT/HUP from Cronicle reach Python, and fd 9 (the lock) stays
# open in it until it exits.
#
# Exit codes: 0 written and valid (or locked / not a session),
#   2 usage, 10 config unreadable, 12 D or its previous day outside the KR calendar,
#   30 no kr-prepare marker before SHADOW_WAIT_UNTIL, 31 panel missing / prepare failed, 3 record
#   written after the 15:00 deadline (invalid, kept for the record),
#   33 the prepare step found the price rows of t incomplete: the day is left out ("missing day",
#   nothing substituted), skips/D=<D>.json is written once (by shadow_prepare; by this script with
#   fewer fields when a SHADOW_PREPARE_CMD hook returned 33 without writing it), no record,
#   other: the Python entrypoint's own code.
# t = the session right before D in the KR calendar (same as shadow.previous_session).
set -euo pipefail
umask 027

report_date=""
if [ $# -gt 0 ]; then
  if [ $# -eq 2 ] && [ "$1" = "--report-date" ] && [[ "$2" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]]; then
    report_date=$2
  else
    echo "usage: kr-next-shadow.sh [--report-date YYYY-MM-DD]" >&2; exit 2
  fi
fi

SERVING_ROOT=${SERVING_ROOT:-/home/whi/apps/market-briefing/serving}
SHADOW_ROOT=${SHADOW_ROOT:-/home/whi/apps/market-briefing/shadow}
wait_until=${SHADOW_WAIT_UNTIL:-09:20}
poll=${SHADOW_POLL_SECONDS:-30}
boot=${BOOTSTRAP_PYTHON:-python3}
ops="$SERVING_ROOT/config/ops.json"
log() { echo "kr-next-shadow: [$(TZ=Asia/Seoul date '+%F %T')] $*"; }
fail() { local code=$1; shift; echo "kr-next-shadow: $*" >&2; exit "$code"; }

[[ "$wait_until" =~ ^([01][0-9]|2[0-3]):[0-5][0-9]$ ]] || fail 2 "SHADOW_WAIT_UNTIL must be HH:MM"
[[ "$poll" =~ ^[1-9][0-9]*$ ]] || fail 2 "SHADOW_POLL_SECONDS must be a positive integer"
[ -n "${SHADOW_RELEASE:-}" ] && [ -f "$SHADOW_RELEASE/release.json" ] || fail 10 "SHADOW_RELEASE missing or has no release.json"
[ -f "$ops" ] || fail 10 "ops.json not found: $ops"

fields=$("$boot" - "$ops" "$report_date" <<'PY'
import json, sys
from datetime import date, datetime
from zoneinfo import ZoneInfo
ops = json.load(open(sys.argv[1]))
d = date.fromisoformat(sys.argv[2]) if sys.argv[2] else datetime.now(ZoneInfo("Asia/Seoul")).date()
print(ops["python"])
print(ops["kr_calendar"])
print(ops["prepared_root"])
print(d.isoformat())
PY
) || fail 10 "cannot read $ops"
python=$(printf '%s\n' "$fields" | sed -n 1p)
calendar=$(printf '%s\n' "$fields" | sed -n 2p)
prepared_root=$(printf '%s\n' "$fields" | sed -n 3p)
D=$(printf '%s\n' "$fields" | sed -n 4p)
[ -x "$python" ] && [ -d "$SHADOW_RELEASE/src" ] || fail 10 "python or release src missing"
[ -f "$calendar" ] || fail 10 "KR calendar missing: $calendar"

# ---- own lock (never kr-prepare.lock) ------------------------------------------------------
command -v flock >/dev/null 2>&1 || fail 10 "flock not found"
mkdir -p "$SHADOW_ROOT/locks" "$SHADOW_ROOT/records"
exec 9>"$SHADOW_ROOT/locks/kr-next-shadow.lock"
if ! flock -n 9; then
  log "locked: another kr-next-shadow run holds the lock; nothing to do (D=$D)"
  exit 0
fi

export PYTHONPATH="$SHADOW_RELEASE/src"
export PYTHONDONTWRITEBYTECODE=1

# Verify the release against its own release.json before anything runs.
"$python" -m modeler.models._03_kr_next.shadow_release verify --release "$SHADOW_RELEASE" >/dev/null \
  || fail 10 "release verification failed: $SHADOW_RELEASE"

# ---- is D a KR session ----------------------------------------------------------------------
rc=0
kind=$("$python" - "$calendar" "$D" <<'PY'
import json, sys
from datetime import date, timedelta
cal = json.load(open(sys.argv[1]))
days = {date.fromisoformat(x) for x in cal["sessions"]}
d = date.fromisoformat(sys.argv[2])
lo, hi = date.fromisoformat(cal["coverage_start"]), date.fromisoformat(cal["coverage_end"])
if not (lo < d <= hi):
    sys.exit(3)
if d not in days:
    print("closed")
else:
    prev = [x for x in days if x < d]
    if not prev:
        sys.exit(3)
    print("session")
    print(max(prev).isoformat())   # t, same definition as shadow.previous_session
PY
) || rc=$?
[ "$rc" -eq 0 ] || fail 12 "D=$D is outside the KR calendar $calendar"
t=$(printf '%s\n' "$kind" | sed -n 2p)
kind=$(printf '%s\n' "$kind" | sed -n 1p)
if [ "$kind" != session ]; then log "D=$D is not a KR session; nothing to score"; exit 0; fi

# ---- wait for the kr-prepare completion marker (poll the file only) --------------------------
end_s=$(TZ=Asia/Seoul date -d "$D ${wait_until}:00" +%s 2>/dev/null || date -j -f '%Y-%m-%d %H:%M:%S' "$D ${wait_until}:00" +%s)
marker=""
while :; do
  for m in "$prepared_root"/kr/score_date=*/prep_id="$D"/completion.json; do
    [ -f "$m" ] || continue
    marker=$m; break   # any score_date: the marker only means "the export of D is finished"
  done
  [ -z "$marker" ] || break
  [ "$(TZ=Asia/Seoul date +%s)" -lt "$end_s" ] || fail 30 "no kr-prepare marker for D=$D before $wait_until KST"
  log "waiting for kr-prepare marker (poll ${poll}s, until $wait_until KST)"
  sleep "$poll"
done
log "kr-prepare marker: $marker"

# prepare exit 33 = t incomplete, day left out.  shadow_prepare has written skips/D=<D>.json itself;
# a hook may not have, so write a short one (never over an existing file).
skip_or_fail() {
  local rc=$1 what=$2
  [ "$rc" -eq 33 ] || fail 31 "$what failed (exit $rc)"
  local skip="$SHADOW_ROOT/skips/D=$D.json"
  if [ ! -e "$skip" ]; then
    mkdir -p "$SHADOW_ROOT/skips"
    local tmp="$skip.tmp.$$"
    "$boot" - "$D" "$t" "$what" >"$tmp" <<'PY'
import json, sys
print(json.dumps({"D": sys.argv[1], "t": sys.argv[2], "reason": "t_price_incomplete",
                  "written_by": sys.argv[3]}, indent=2))
PY
    ln "$tmp" "$skip" 2>/dev/null || true   # hard link: never overwrites an existing file
    rm -f "$tmp"
  fi
  fail 33 "$what: price rows of t=$t incomplete; day D=$D left out, see $skip"
}

# A day once left out stays left out (draft §10.2: a missing point is final, late data does not
# bring it back), so an existing skips/D=<D>.json ends the run before anything is re-checked.
[ ! -e "$SHADOW_ROOT/skips/D=$D.json" ] || fail 33 "D=$D was left out before, see $SHADOW_ROOT/skips/D=$D.json"

# ---- panel of the signal day ----------------------------------------------------------------
panel_dir="$SHADOW_ROOT/panels_in/D=$D"
panel="$panel_dir/panel.parquet"
if [ ! -f "$panel" ]; then
  mkdir -p "$panel_dir"
  if [ -n "${SHADOW_PREPARE_CMD:-}" ]; then
    log "building the panel: $SHADOW_PREPARE_CMD $D"
    prc=0
    $SHADOW_PREPARE_CMD "$D" || prc=$?
    [ "$prc" -eq 0 ] || skip_or_fail "$prc" "SHADOW_PREPARE_CMD"
  else
    # default: shadow_prepare builds the derived views of the day's raw snapshot (once, under
    # $SHADOW_ROOT/views/D-<D>) and the serving panel of t.  Write-once; an existing panel is
    # never reached here.
    raw_lake=${SHADOW_RAW_LAKE:-${KR_STOCK_DATA_ROOT:-/home/whi/data/stock_data}/kr/raw/raw_postgres}
    mkdir -p "$SHADOW_ROOT/tmp"
    log "building the panel: shadow_prepare run (raw lake $raw_lake, memory ${SHADOW_PREPARE_MEMORY_LIMIT:-2GB}, threads ${SHADOW_PREPARE_THREADS:-4})"
    "$python" -m modeler.models._03_kr_next.shadow_prepare run --date "$D" \
      --raw-lake "$raw_lake" --source "${SHADOW_RAW_SOURCE:-sj2_remote}" \
      --out-root "$SHADOW_ROOT" --sessions "$calendar" \
      --memory-limit "${SHADOW_PREPARE_MEMORY_LIMIT:-2GB}" \
      --threads "${SHADOW_PREPARE_THREADS:-4}" || skip_or_fail "$?" "shadow_prepare"
  fi
fi
[ -f "$panel" ] || fail 31 "panel missing: $panel (the shadow prepare step has not written it)"
manifest_args=()
[ -f "$panel_dir/panel_manifest.json" ] && manifest_args=(--panel-manifest "$panel_dir/panel_manifest.json")

log "scoring D=$D release=$SHADOW_RELEASE"
exec "$python" -m modeler.models._03_kr_next.shadow score \
  --date "$D" --panel "$panel" ${manifest_args[@]+"${manifest_args[@]}"} \
  --bundle-dir "$SHADOW_RELEASE/bundle" \
  --out-root "$SHADOW_ROOT/records" --ledger "$SHADOW_ROOT/records/ledger.jsonl" \
  --sessions "$calendar"
