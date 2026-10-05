#!/usr/bin/env bash
# Cronicle entrypoint for one daily-briefing stage: select | run | monitor [--attempt N].
#
#   SERVING_ROOT  serving root (default /home/whi/apps/market-briefing/serving)
#   BOOTSTRAP_PYTHON  only used to read ops.json (default python3)
#
# Release root and Python come from config/ops.json; the frozen release's own
# src/ is put on PYTHONPATH.  Output goes to stdout/stderr for Cronicle; the exit
# code is the wrapper's (0 ok, 1 stage failed).  Exit 2 is a usage error, 10 a
# bad serving root.
#
# The script ends in `exec`: no shell stays between Cronicle and the Python wrapper, so TERM, INT
# and HUP reach the wrapper itself.  The wrapper turns them into an exit and ends the process
# group of the runner, opening and publisher children it started (`daily_coordinator.run_group`).
# (kr-prepare.sh and us-prepare.sh keep a shell and do the same with `set -m` and a trap.)
set -euo pipefail
umask 027

usage() { echo "usage: briefing-stage.sh <select|run|monitor> [--attempt N]" >&2; exit 2; }

[ $# -ge 1 ] || usage
stage=$1; shift
case "$stage" in select|run|monitor) ;; *) usage ;; esac
attempt=""
if [ $# -gt 0 ]; then
  [ "$stage" = monitor ] && [ $# -eq 2 ] && [ "$1" = "--attempt" ] || usage
  case "$2" in 0|1|2|3) attempt=$2 ;; *) echo "attempt must be 0..3" >&2; exit 2 ;; esac
fi

SERVING_ROOT=${SERVING_ROOT:-/home/whi/apps/market-briefing/serving}
ops="$SERVING_ROOT/config/ops.json"
[ -f "$ops" ] || { echo "briefing-stage: ops.json not found: $ops" >&2; exit 10; }

fields=$("${BOOTSTRAP_PYTHON:-python3}" - "$ops" <<'PY'
import json, sys
from pathlib import Path
config = json.load(open(sys.argv[1]))
release = Path(config["release_manifest"]).parent
print(release)
print(config["python"])
PY
) || { echo "briefing-stage: cannot read $ops" >&2; exit 10; }
release_root=$(printf '%s\n' "$fields" | sed -n 1p)
python=$(printf '%s\n' "$fields" | sed -n 2p)
[ -d "$release_root/src" ] || { echo "briefing-stage: release src missing: $release_root/src" >&2; exit 10; }
[ -x "$python" ] || { echo "briefing-stage: python is not executable: $python" >&2; exit 10; }

cd "$release_root"
export PYTHONPATH="$release_root/src"
export PYTHONDONTWRITEBYTECODE=1
args=("$stage" --config "$ops")
[ -z "$attempt" ] || args+=(--attempt "$attempt")
exec "$python" -m modeler.serving.daily_wrapper "${args[@]}"
