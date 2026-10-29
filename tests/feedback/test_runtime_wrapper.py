from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[2] / "deploy/feedback/feedback-stage.sh"


@pytest.fixture
def wrapper_case(tmp_path):
    root = tmp_path.resolve()
    code = root / "code with spaces"
    package = code / "src/modeler/feedback"
    package.mkdir(parents=True)
    for file in (package.parent / "__init__.py", package / "__init__.py", package / "runtime.py"):
        file.write_text("# synthetic runtime\n")
    config = root / "config with spaces.json"
    config.write_text("{}\n")
    result = root / "called.json"
    python = root / "fake python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, signal, sys, time\n"
        "from pathlib import Path\n"
        f"result = Path({str(result)!r})\n"
        "result.write_text(json.dumps({'argv': sys.argv[1:], 'pid': os.getpid(), "
        "'cwd': os.getcwd(), 'pythonpath': os.environ.get('PYTHONPATH'), "
        "'bytecode': os.environ.get('PYTHONDONTWRITEBYTECODE')}))\n"
        "print('synthetic runtime stdout', flush=True)\n"
        "if os.environ.get('WRAPPER_TEST_WAIT') == '1':\n"
        "    while True:\n"
        "        time.sleep(0.1)\n"
        "sys.exit(int(os.environ.get('WRAPPER_TEST_EXIT', '0')))\n"
    )
    python.chmod(0o700)
    env = os.environ | {
        "FEEDBACK_CONFIG": str(config),
        "FEEDBACK_PYTHON": str(python),
        "FEEDBACK_CODE_ROOT": str(code),
        "PYTHONPATH": str(root / "untrusted inherited modules"),
    }
    return root, env, result


def run_wrapper(env, *args):
    return subprocess.run(
        ["/bin/bash", str(WRAPPER), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def test_exec_arguments_environment_and_exit_status(wrapper_case):
    root, env, result = wrapper_case
    env["WRAPPER_TEST_EXIT"] = "23"
    env["PYTHONHOME"] = str(root / "nonexistent-python-install")
    env["PYTHONINSPECT"] = "1"
    env["PYTHONHASHSEED"] = "invalid inherited seed"
    completed = run_wrapper(env)
    assert completed.returncode == 23
    assert completed.stdout == "synthetic runtime stdout\n"
    assert completed.stderr == ""
    called = json.loads(result.read_text())
    assert called["argv"] == [
        "-s",
        "-P",
        "-m",
        "modeler.feedback.runtime",
        "--config",
        env["FEEDBACK_CONFIG"],
    ]
    assert called["cwd"] == env["FEEDBACK_CODE_ROOT"]
    assert called["pythonpath"] == env["FEEDBACK_CODE_ROOT"] + "/src"
    assert called["bytecode"] == "1"


@pytest.mark.parametrize("name", ["FEEDBACK_CONFIG", "FEEDBACK_PYTHON", "FEEDBACK_CODE_ROOT"])
def test_requires_explicit_paths(wrapper_case, name):
    _, env, result = wrapper_case
    env.pop(name)
    completed = run_wrapper(env)
    assert completed.returncode == 10
    assert name in completed.stderr
    assert not result.exists()


@pytest.mark.parametrize("kind", ["missing", "relative", "parent", "symlink", "parent_symlink"])
def test_rejects_unsafe_config_paths(wrapper_case, kind):
    root, env, result = wrapper_case
    config = Path(env["FEEDBACK_CONFIG"])
    if kind == "missing":
        config.unlink()
    elif kind == "relative":
        env["FEEDBACK_CONFIG"] = config.name
    elif kind == "parent":
        env["FEEDBACK_CONFIG"] = str(root) + "/unused/../" + config.name
    elif kind == "symlink":
        link = root / "config-link.json"
        link.symlink_to(config)
        env["FEEDBACK_CONFIG"] = str(link)
    else:
        link = root / "root-link"
        link.symlink_to(root, target_is_directory=True)
        env["FEEDBACK_CONFIG"] = str(link / config.name)
    completed = run_wrapper(env)
    assert completed.returncode == 10
    assert "FEEDBACK_CONFIG" in completed.stderr
    assert not result.exists()


@pytest.mark.parametrize("kind", ["missing", "not_executable", "directory", "relative", "dangling"])
def test_rejects_invalid_interpreters(wrapper_case, kind):
    root, env, result = wrapper_case
    python = Path(env["FEEDBACK_PYTHON"])
    if kind == "missing":
        python.unlink()
    elif kind == "not_executable":
        python.chmod(0o600)
    elif kind == "directory":
        env["FEEDBACK_PYTHON"] = str(root)
    elif kind == "relative":
        env["FEEDBACK_PYTHON"] = python.name
    else:
        link = root / "missing-python"
        link.symlink_to(root / "absent")
        env["FEEDBACK_PYTHON"] = str(link)
    completed = run_wrapper(env)
    assert completed.returncode == 10
    assert "FEEDBACK_PYTHON" in completed.stderr
    assert not result.exists()


def test_preserves_venv_interpreter_symlink_path(wrapper_case):
    root, env, result = wrapper_case
    python = Path(env["FEEDBACK_PYTHON"])
    venv_python = root / "venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(python)
    # argv[0] identifies the path actually passed to exec, even for this fake Python.
    python.write_text(python.read_text().replace("'argv': sys.argv[1:]", "'argv': sys.argv"))
    env["FEEDBACK_PYTHON"] = str(venv_python)
    assert run_wrapper(env).returncode == 0
    assert json.loads(result.read_text())["argv"][0] == str(venv_python)


@pytest.mark.parametrize(
    "kind", ["missing", "relative", "symlink", "src_symlink", "runtime_symlink"]
)
def test_rejects_unsafe_code_roots(wrapper_case, kind):
    root, env, result = wrapper_case
    code = Path(env["FEEDBACK_CODE_ROOT"])
    if kind == "missing":
        (code / "src/modeler/feedback/runtime.py").unlink()
    elif kind == "relative":
        env["FEEDBACK_CODE_ROOT"] = code.name
    elif kind == "symlink":
        link = root / "code-link"
        link.symlink_to(code, target_is_directory=True)
        env["FEEDBACK_CODE_ROOT"] = str(link)
    elif kind == "src_symlink":
        src = code / "src"
        moved = root / "other-src"
        src.rename(moved)
        src.symlink_to(moved, target_is_directory=True)
    else:
        runtime = code / "src/modeler/feedback/runtime.py"
        moved = root / "other-runtime.py"
        runtime.rename(moved)
        runtime.symlink_to(moved)
    completed = run_wrapper(env)
    assert completed.returncode == 10
    assert "FEEDBACK_CODE_ROOT" in completed.stderr
    assert not result.exists()


def test_rejects_extra_arguments_without_running_python(wrapper_case):
    _, env, result = wrapper_case
    completed = run_wrapper(env, "--config", "/other/config.json")
    assert completed.returncode == 2
    assert "usage:" in completed.stderr
    assert not result.exists()


def test_term_reaches_exec_process_and_leaves_no_wrapper(wrapper_case):
    _, env, result = wrapper_case
    env["WRAPPER_TEST_WAIT"] = "1"
    process = subprocess.Popen(
        ["/bin/bash", str(WRAPPER)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not result.exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.01)
        assert result.exists(), "fake runtime never started"
        assert json.loads(result.read_text())["pid"] == process.pid
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == -signal.SIGTERM
        assert stdout == "synthetic runtime stdout\n"
        assert stderr == ""
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)
