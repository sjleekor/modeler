#!/bin/bash
# Standalone feedback entrypoint. No serving release or shell environment is sourced.
# All three FEEDBACK_* paths must be supplied explicitly. Python may be a venv
# symlink; execute its original path so that Python keeps the venv's site-packages.
# Exit 10 means invalid configuration, 2 means unexpected command-line arguments.
# Once exec succeeds, Python owns this PID and its signal/exit handling.
set -euo pipefail
umask 027

fail() { printf 'feedback-stage: %s\n' "$*" >&2; exit 10; }

if [ "$#" -ne 0 ]; then
  printf 'usage: FEEDBACK_CONFIG=... FEEDBACK_PYTHON=... FEEDBACK_CODE_ROOT=... feedback-stage.sh\n' >&2
  exit 2
fi

# Check each component, not just the final entry. Do not resolve a symlink and
# silently accept a different config or code tree. Only Python's final component
# may be a symlink (the normal venv/bin/python layout).
check_path() {
  local name=$1 path=$2 allow_leaf_link=${3:-false} component rest current=""
  case "$path" in
    /*) ;;
    *) fail "$name must be an explicit absolute path" ;;
  esac
  case "$path" in
    /|*/|*//*|*/./*|*/../*|*/.|*/..|*$'\n'*|*$'\r'*)
      fail "$name must be a canonical absolute path" ;;
  esac
  rest=${path#/}
  while [ -n "$rest" ]; do
    component=${rest%%/*}
    current="$current/$component"
    if [ "$rest" = "$component" ]; then
      rest=""
    else
      rest=${rest#*/}
    fi
    if [ -L "$current" ] && { [ -n "$rest" ] || [ "$allow_leaf_link" != true ]; }; then
      fail "$name must not contain symlinks"
    fi
  done
}

config=${FEEDBACK_CONFIG:-}
python=${FEEDBACK_PYTHON:-}
code_root=${FEEDBACK_CODE_ROOT:-}
check_path FEEDBACK_CONFIG "$config"
check_path FEEDBACK_PYTHON "$python" true
check_path FEEDBACK_CODE_ROOT "$code_root"
[ -f "$config" ] && [ -r "$config" ] || fail "FEEDBACK_CONFIG must be a readable regular file"
[ -f "$python" ] && [ -x "$python" ] || fail "FEEDBACK_PYTHON must resolve to an executable regular file"
[ -d "$code_root" ] && [ -r "$code_root" ] && [ -x "$code_root" ] \
  || fail "FEEDBACK_CODE_ROOT must be a readable directory"
for entry in src src/modeler src/modeler/__init__.py src/modeler/feedback \
  src/modeler/feedback/__init__.py src/modeler/feedback/runtime.py; do
  check_path "FEEDBACK_CODE_ROOT/$entry" "$code_root/$entry"
done
[ -d "$code_root/src/modeler/feedback" ] \
  && [ -r "$code_root/src/modeler/__init__.py" ] \
  && [ -f "$code_root/src/modeler/__init__.py" ] \
  && [ -r "$code_root/src/modeler/feedback/__init__.py" ] \
  && [ -f "$code_root/src/modeler/feedback/__init__.py" ] \
  && [ -r "$code_root/src/modeler/feedback/runtime.py" ] \
  && [ -f "$code_root/src/modeler/feedback/runtime.py" ] \
  || fail "FEEDBACK_CODE_ROOT must contain the feedback runtime package"

cd -- "$code_root"
# Reset Python-specific inherited startup/installation settings in this process
# only. Keep the rest of the inherited environment, including managed proxies.
for variable in "${!PYTHON@}"; do
  unset "$variable"
done
export PYTHONPATH="$code_root/src"
export PYTHONDONTWRITEBYTECODE=1
# Cronicle lib/job.js sets JOB_ID=job.id. Runtime still verifies this exact record
# and its expected event through the API before excluding only its own job.
export FEEDBACK_JOB_ID="${FEEDBACK_JOB_ID:-${JOB_ID:-}}"
args=(-s -P -m modeler.feedback.runtime --config "$config")
exec "$python" "${args[@]}"
