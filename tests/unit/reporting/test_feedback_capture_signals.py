"""Real observer process groups are reaped on timeout and repeated termination."""

import json
import os
import signal
import subprocess
import sys
import time

import pytest
from reports_world import PROJECT


@pytest.mark.parametrize("mode", ["timeout", "repeated_term"])
def test_observer_cleanup_reaps_signal_ignoring_descendant(tmp_path, mode):
    pids = tmp_path / "pids.json"
    observer = tmp_path / "observer.py"
    observer.write_text(
        "import json,os,signal,subprocess,sys,time\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "child=subprocess.Popen([sys.executable,'-c',"
        "'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])\n"
        f"open({str(pids)!r},'w').write(json.dumps([os.getpid(),child.pid]))\n"
        "time.sleep(60)\n"
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import os,subprocess,sys\n"
        f"sys.path.insert(0,{str(PROJECT / 'deploy/reports')!r})\n"
        "import publish_reports as pr\n"
        "real_popen=subprocess.Popen\n"
        "def spawn(args,**kw):\n"
        f" return real_popen([sys.executable,{str(observer)!r}],**kw)\n"
        "pr.subprocess.Popen=spawn\n"
        "pr.rev=lambda *args:'a'*40\n"
        "os.environ['FEEDBACK_CAPTURE_SETTINGS']='/synthetic/settings.json'\n"
        "pr._feedback_event('prepare',{'checkout_dir':'/synthetic/checkout',"
        "'expected_remote_url':'synthetic','remote_name':'origin','branch':'main'},"
        " {}, '2026-10-29')\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT / "src")
    process = subprocess.Popen(
        [sys.executable, str(parent)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    children = []
    try:
        deadline = time.monotonic() + 8
        while not pids.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        children = json.loads(pids.read_text())
        if mode == "repeated_term":
            process.send_signal(signal.SIGTERM)
            for _ in range(5):
                time.sleep(0.01)
                try:
                    process.send_signal(signal.SIGTERM)
                except ProcessLookupError:
                    break
        _, stderr = process.communicate(timeout=15)
        assert process.returncode == (143 if mode == "repeated_term" else 0)
        if mode == "timeout":
            assert b"coverage awaits review" in stderr
        deadline = time.monotonic() + 3
        while True:
            live = []
            for pid in children:
                state = subprocess.run(
                    ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
                ).stdout.strip()
                if state and not state.startswith("Z"):
                    live.append(pid)
            if not live or time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        assert live == []
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate()
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
