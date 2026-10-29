from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from modeler.feedback_operations import credential_launcher as launcher

REPO = Path(__file__).resolve().parents[2]
WRAPPER = REPO / "deploy/feedback/feedback-stage.sh"
NOW = datetime(2026, 11, 9, 2, 0, tzinfo=UTC)  # 11:00 KST
PROXY_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


@pytest.fixture
def case(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    code = root / "code with spaces"
    package = code / "src/modeler"
    files = (
        "__init__.py",
        "reporting/__init__.py",
        "reporting/markdown.py",
        "feedback/__init__.py",
        "feedback/runtime.py",
    )
    pins = {}
    for name in files:
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# synthetic runtime\n")
        pins[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    python = root / "fake python"
    python.write_text(f"#!{sys.executable}\n")
    python.chmod(0o700)
    config = root / "config.json"
    settings = {
        "schema": "prediction-feedback-runtime.v1",
        "enabled": True,
        "mode": "incremental",
        "not_before_date": "2026-11-07",
        "code_files": pins,
    }
    config.write_text(json.dumps(settings))
    key = root / "dedicated-key"
    key.write_bytes(b"synthetic-dedicated-key\r\n")
    key.chmod(0o600)
    for name, value in (
        ("FEEDBACK_CONFIG", config),
        ("FEEDBACK_PYTHON", python),
        ("FEEDBACK_CODE_ROOT", code),
    ):
        monkeypatch.setenv(name, str(value))
    monkeypatch.setenv("JOB_ID", "synthetic-current-job")
    monkeypatch.delenv("FEEDBACK_JOB_ID", raising=False)
    return root, code, python, config, settings, key


def save_config(case):
    case[3].write_text(json.dumps(case[4]))


def never_exec(*_args):
    pytest.fail("validation must finish before exec")


@pytest.mark.parametrize("change", ["missing", "different"])
def test_job_context_is_required_before_key_read(case, monkeypatch, change):
    if change == "missing":
        monkeypatch.delenv("JOB_ID")
    else:
        monkeypatch.setenv("FEEDBACK_JOB_ID", "another-job")
    monkeypatch.setattr(launcher, "_read_key", lambda _: pytest.fail("unverified context read key"))
    with pytest.raises(launcher.LauncherError, match="job context"):
        launcher.launch(case[5], WRAPPER, now=NOW, execve=never_exec)


def test_disabled_never_opens_key_and_removes_stale_credentials(case, monkeypatch):
    case[4]["enabled"] = False
    save_config(case)
    case[5].unlink()
    monkeypatch.setenv("FEEDBACK_CRONICLE_API_KEY", "synthetic-stale-value")
    monkeypatch.setenv("FEEDBACK_CRONICLE_KEY_FD", "999")
    monkeypatch.setattr(launcher, "_read_key", lambda _: pytest.fail("disabled key read"))
    inherited = os.environ.copy()

    def execute(binary, args, environment):
        assert binary == "/bin/bash"
        assert args == ["/bin/bash", str(WRAPPER)]
        assert all(name not in environment for name in launcher._CREDENTIAL_ENV)
        proxy_preserved = all(environment.get(name) == inherited.get(name) for name in PROXY_NAMES)
        assert proxy_preserved
        return 23

    assert launcher.launch(case[5], WRAPPER, execve=execute) == 23
    parent_unchanged = dict(os.environ) == inherited
    assert parent_unchanged


def test_enabled_passes_only_read_descriptor_and_closes_it_when_exec_returns(case):
    inherited = os.environ.copy()
    captured = []

    def execute(binary, args, environment):
        assert binary == "/bin/bash"
        assert args == ["/bin/bash", str(WRAPPER)]
        assert "FEEDBACK_CRONICLE_API_KEY" not in environment
        fd = int(environment["FEEDBACK_CRONICLE_KEY_FD"])
        assert os.get_inheritable(fd)
        raw = os.read(fd, 4097)
        assert (
            hashlib.sha256(raw).hexdigest()
            == hashlib.sha256(b"synthetic-dedicated-key").hexdigest()
        )
        assert os.read(fd, 1) == b""
        proxy_preserved = all(environment.get(name) == inherited.get(name) for name in PROXY_NAMES)
        assert proxy_preserved
        captured.append(fd)
        return 7

    assert launcher.launch(case[5], WRAPPER, now=NOW, execve=execute) == 7
    with pytest.raises(OSError):
        os.fstat(captured[0])
    parent_unchanged = dict(os.environ) == inherited
    assert parent_unchanged


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("target", ["key", "wrapper", "config", "code", "python"])
@pytest.mark.parametrize("kind", ["relative", "parent", "parent_symlink"])
def test_guards_all_paths_before_config_read(case, monkeypatch, enabled, target, kind):
    root, code, python, config, settings, key = case
    settings["enabled"] = enabled
    save_config(case)
    wrapper = WRAPPER
    original = {"key": key, "wrapper": WRAPPER, "config": config, "code": code, "python": python}[
        target
    ]
    if kind == "relative":
        unsafe = original.name
    elif kind == "parent":
        unsafe = str(original.parent) + "/unused/../" + original.name
    else:
        link = root / "parent-link"
        link.symlink_to(original.parent, target_is_directory=True)
        unsafe = str(link / original.name)
    if target in ("config", "code", "python"):
        name = {
            "config": "FEEDBACK_CONFIG",
            "code": "FEEDBACK_CODE_ROOT",
            "python": "FEEDBACK_PYTHON",
        }[target]
        monkeypatch.setenv(name, unsafe)
    elif target == "wrapper":
        wrapper = unsafe
    else:
        key = unsafe
    monkeypatch.setattr(launcher, "_read_regular", lambda *_a, **_k: pytest.fail("early file read"))
    with pytest.raises((launcher.LauncherError, OSError)):
        launcher.launch(key, wrapper, now=NOW, execve=never_exec)


@pytest.mark.parametrize(
    "mutation", ["flag", "mode", "early_date", "bad_date", "future", "pin", "inventory", "wrapper"]
)
def test_rejects_ineligible_configuration_before_key_read(case, monkeypatch, mutation):
    settings = case[4]
    wrapper = WRAPPER
    if mutation == "flag":
        settings["enabled"] = 1
    elif mutation == "mode":
        settings["mode"] = "isolated"
    elif mutation == "early_date":
        settings["not_before_date"] = "2026-11-06"
    elif mutation == "bad_date":
        settings["not_before_date"] = "2026-11-7"
    elif mutation == "future":
        settings["not_before_date"] = "2026-11-10"
    elif mutation == "pin":
        settings["code_files"]["feedback/runtime.py"] = "0" * 64
    elif mutation == "inventory":
        settings["code_files"]["../outside.py"] = "0" * 64
    else:
        wrapper = case[0] / "other-wrapper.sh"
        wrapper.write_text("#!/bin/bash\nexit 0\n")
    save_config(case)
    monkeypatch.setattr(launcher, "_read_key", lambda _: pytest.fail("ineligible key read"))
    with pytest.raises((launcher.LauncherError, ValueError)):
        launcher.launch(case[5], wrapper, now=NOW, execve=never_exec)


@pytest.mark.parametrize(
    "clock",
    [
        datetime(2026, 11, 6, 2, 0, tzinfo=UTC),
        datetime(2026, 11, 9, 1, 34, 59, tzinfo=UTC),
        datetime(2026, 11, 9, 13, 45, tzinfo=UTC),
        datetime(2026, 11, 9, 2, 0),
    ],
)
def test_rejects_clock_before_key_read(case, monkeypatch, clock):
    monkeypatch.setattr(launcher, "_read_key", lambda _: pytest.fail("outside-window key read"))
    with pytest.raises(launcher.LauncherError):
        launcher.launch(case[5], WRAPPER, now=clock, execve=never_exec)


@pytest.mark.parametrize(
    "content", [b"", b"x" * 4097, b"a\nb", b"a\n\n", b"a\x00b", b"a\tb", b"\xff", b"\r\n"]
)
def test_rejects_malformed_key(case, content):
    case[5].write_bytes(content)
    with pytest.raises((launcher.LauncherError, UnicodeError)):
        launcher.launch(case[5], WRAPPER, now=NOW, execve=never_exec)


@pytest.mark.parametrize("mode", [0o400, 0o640, 0o644, 0o700])
def test_requires_exact_key_permissions(case, mode):
    case[5].chmod(mode)
    with pytest.raises(launcher.LauncherError):
        launcher.launch(case[5], WRAPPER, now=NOW, execve=never_exec)


@pytest.mark.parametrize(
    "kind", ["symlink", "hardlink", "directory", "fifo", "wrong_owner", "replaced"]
)
def test_rejects_unsafe_key_file(case, monkeypatch, kind):
    key = case[5]
    if kind == "symlink":
        other = key.with_suffix(".original")
        key.rename(other)
        key.symlink_to(other)
    elif kind == "hardlink":
        os.link(key, key.with_suffix(".link"))
    elif kind == "directory":
        key.unlink()
        key.mkdir()
    elif kind == "fifo":
        key.unlink()
        os.mkfifo(key, 0o600)
    elif kind == "wrong_owner":
        monkeypatch.setattr(launcher.os, "getuid", lambda: -1)
    else:
        original = launcher.os.open

        def replace(path, flags, *args, **kwargs):
            if path == key.name:
                other = key.with_suffix(".replacement")
                other.write_bytes(b"synthetic-replacement")
                other.chmod(0o600)
                other.replace(key)
            return original(path, flags, *args, **kwargs)

        monkeypatch.setattr(launcher.os, "open", replace)
    with pytest.raises((launcher.LauncherError, OSError)):
        launcher.launch(key, WRAPPER, now=NOW, execve=never_exec)


def test_exec_failure_closes_both_pipe_descriptors(case, monkeypatch):
    pipe = launcher.os.pipe
    created = []

    def record_pipe():
        descriptors = pipe()
        created.extend(descriptors)
        return descriptors

    def fail_exec(*_args):
        raise OSError("synthetic exec failure")

    monkeypatch.setattr(launcher.os, "pipe", record_pipe)
    with pytest.raises(OSError):
        launcher.launch(case[5], WRAPPER, now=NOW, execve=fail_exec)
    for fd in created:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_cli_failure_is_generic_and_does_not_expose_inputs(case, capsys):
    assert launcher.main(["--key-file", "sensitive-synthetic-path", "--wrapper", str(WRAPPER)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == "feedback-credential-launcher: execution failed\n"


def test_real_exec_pipe_survives_reviewed_bash_wrapper_without_key_environment(case):
    root, _, python, _, _, key = case
    result = root / "result.json"
    python.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, json, os\n"
        "from pathlib import Path\n"
        "fd = int(os.environ['FEEDBACK_CRONICLE_KEY_FD'])\n"
        "raw = os.read(fd, 4097)\n"
        f"Path({str(result)!r}).write_text(json.dumps({{"
        "'sha256': hashlib.sha256(raw).hexdigest(), 'pid': os.getpid(), "
        "'environment_key_absent': 'FEEDBACK_CRONICLE_API_KEY' not in os.environ, "
        "'eof': os.read(fd, 1) == b''}))\n"
    )
    script = (
        "from datetime import UTC, datetime; "
        "from modeler.feedback_operations.credential_launcher import launch; "
        f"launch({str(key)!r}, {str(WRAPPER)!r}, now=datetime(2026,11,9,2,tzinfo=UTC))"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO / "src")
    process = subprocess.Popen(
        [sys.executable, "-s", "-P", "-c", script],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == 0
    assert stdout == b""
    assert stderr == b""
    data = json.loads(result.read_text())
    assert data == {
        "sha256": hashlib.sha256(b"synthetic-dedicated-key").hexdigest(),
        "pid": process.pid,
        "environment_key_absent": True,
        "eof": True,
    }
