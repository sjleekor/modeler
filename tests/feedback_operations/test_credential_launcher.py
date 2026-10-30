from __future__ import annotations

import hashlib
import json
import os
import stat
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
        override_metadata(monkeypatch, key, st_uid=os.geteuid() + 1)
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


class AlteredStat:
    def __init__(self, original, **changes):
        self.original = original
        self.__dict__.update(changes)

    def __getattr__(self, name):
        return getattr(self.original, name)


def override_metadata(monkeypatch, path, **changes):
    identity = (path.stat().st_dev, path.stat().st_ino)
    original_stat = launcher.os.stat
    original_fstat = launcher.os.fstat

    def alter(info):
        if (info.st_dev, info.st_ino) == identity:
            return AlteredStat(info, **changes)
        return info

    monkeypatch.setattr(launcher.os, "stat", lambda *a, **k: alter(original_stat(*a, **k)))
    monkeypatch.setattr(launcher.os, "fstat", lambda *a, **k: alter(original_fstat(*a, **k)))


def forbid_key_open(monkeypatch, key):
    original = launcher.os.open

    def guarded(path, *args, **kwargs):
        assert os.fspath(path) not in (key.name, str(key)), "key opened before validation"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(launcher.os, "open", guarded)


@pytest.mark.parametrize("mode", [0o770, 0o707, 0o1777])
@pytest.mark.parametrize("target", ["key", "wrapper", "config", "code", "python"])
def test_untrusted_ancestor_rejected_before_key_open(case, monkeypatch, mode, target):
    root, code, python, config, _, key = case
    container = root / "untrusted"
    container.mkdir()
    destination = container / "deployment"
    destination.mkdir()
    original = {"key": key, "wrapper": WRAPPER, "config": config, "code": code, "python": python}[
        target
    ]
    moved = destination / original.name
    if target == "wrapper":
        moved.write_bytes(original.read_bytes())
    else:
        original.rename(moved)
    wrapper = moved if target == "wrapper" else WRAPPER
    if target == "key":
        key = moved
    elif target in ("config", "code", "python"):
        monkeypatch.setenv(
            {
                "config": "FEEDBACK_CONFIG",
                "code": "FEEDBACK_CODE_ROOT",
                "python": "FEEDBACK_PYTHON",
            }[target],
            str(moved),
        )
    container.chmod(mode)
    forbid_key_open(monkeypatch, key)
    with pytest.raises(launcher.LauncherError, match="directory protection"):
        launcher.launch(key, wrapper, now=NOW, execve=never_exec)


def test_wrong_owner_ancestor_rejected_before_key_open(case, monkeypatch):
    override_metadata(monkeypatch, case[0], st_uid=os.geteuid() + 1)
    forbid_key_open(monkeypatch, case[5])
    with pytest.raises(launcher.LauncherError, match="directory protection"):
        launcher.launch(case[5], WRAPPER, now=NOW, execve=never_exec)


@pytest.mark.parametrize("target", ["wrapper", "config", "code", "python"])
@pytest.mark.parametrize("mutation", ["group_write", "other_write", "wrong_owner"])
def test_untrusted_runtime_leaf_rejected_before_key_open(case, monkeypatch, target, mutation):
    wrapper = case[0] / "wrapper.sh"
    wrapper.write_bytes(WRAPPER.read_bytes())
    path = {
        "wrapper": wrapper,
        "config": case[3],
        "code": case[1] / "src/modeler/reporting/markdown.py",
        "python": case[2],
    }[target]
    if mutation == "wrong_owner":
        override_metadata(monkeypatch, path, st_uid=os.geteuid() + 1)
    else:
        path.chmod(
            stat.S_IMODE(path.stat().st_mode) | (0o020 if mutation == "group_write" else 0o002)
        )
    forbid_key_open(monkeypatch, case[5])
    with pytest.raises(launcher.LauncherError, match="file protection"):
        launcher.launch(case[5], wrapper, now=NOW, execve=never_exec)


def test_root_sticky_container_allowed_only_above_deployment_root(tmp_path, monkeypatch):
    container = tmp_path.resolve() / "container"
    deployment = container / "deployment"
    deployment.mkdir(parents=True)
    payload = deployment / "payload"
    payload.write_bytes(b"synthetic")
    override_metadata(monkeypatch, container, st_uid=0, st_mode=stat.S_IFDIR | 0o1777)
    assert launcher._read_regular(payload, 20, trusted=True, trust_root=deployment) == b"synthetic"
    with pytest.raises(launcher.LauncherError, match="directory protection"):
        launcher._inspect_path(container, directory=True)
    with pytest.raises(launcher.LauncherError, match="directory protection"):
        launcher._read_regular(payload, 20, trusted=True, trust_root=container)


def test_key_parent_must_be_protected_even_when_root_sticky(case, monkeypatch):
    parent = case[0] / "dedicated-key-parent"
    parent.mkdir()
    key = parent / case[5].name
    case[5].rename(key)
    override_metadata(monkeypatch, parent, st_uid=0, st_mode=stat.S_IFDIR | 0o1777)
    forbid_key_open(monkeypatch, key)
    with pytest.raises(launcher.LauncherError, match="directory protection"):
        launcher.launch(key, WRAPPER, now=NOW, execve=never_exec)


@pytest.mark.parametrize("mode", [0o770, 0o707, 0o1777])
def test_code_root_must_not_be_writable_even_if_root_owned(case, monkeypatch, mode):
    override_metadata(monkeypatch, case[1], st_uid=0, st_mode=stat.S_IFDIR | mode)
    forbid_key_open(monkeypatch, case[5])
    with pytest.raises(launcher.LauncherError, match="directory protection"):
        launcher.launch(case[5], WRAPPER, now=NOW, execve=never_exec)


@pytest.mark.parametrize("mutation", ["ancestor", "leaf_mode", "leaf_owner"])
def test_python_symlink_target_must_be_trusted(case, monkeypatch, mutation):
    target_parent = case[0] / "interpreter-target"
    target_parent.mkdir()
    target = target_parent / "python"
    case[2].rename(target)
    case[2].symlink_to(target)
    if mutation == "ancestor":
        target_parent.chmod(0o777)
    elif mutation == "leaf_mode":
        target.chmod(0o777)
    else:
        override_metadata(monkeypatch, target, st_uid=os.geteuid() + 1)
    forbid_key_open(monkeypatch, case[5])
    with pytest.raises(launcher.LauncherError, match="protection"):
        launcher.launch(case[5], WRAPPER, now=NOW, execve=never_exec)


def test_python_symlink_preserves_original_runtime_path(case):
    target = case[0] / "actual-python"
    case[2].rename(target)
    case[2].symlink_to(target)

    def execute(_binary, _args, environment):
        assert environment["FEEDBACK_PYTHON"] == str(case[2])
        return 19

    assert launcher.launch(case[5], WRAPPER, now=NOW, execve=execute) == 19


def test_python_target_link_route_does_not_hide_untrusted_ancestor(case, monkeypatch):
    target = case[0] / "actual-python"
    case[2].rename(target)
    unsafe = case[0] / "unsafe-link-route"
    unsafe.mkdir()
    (unsafe / "redirect").symlink_to(case[0], target_is_directory=True)
    case[2].symlink_to(unsafe / "redirect" / target.name)
    unsafe.chmod(0o777)
    forbid_key_open(monkeypatch, case[5])
    with pytest.raises(launcher.LauncherError, match="directory protection"):
        launcher.launch(case[5], WRAPPER, now=NOW, execve=never_exec)


def test_python_target_trusted_directory_link_is_supported(case):
    target = case[0] / "actual-python"
    case[2].rename(target)
    link = case[0] / "trusted-directory-link"
    link.symlink_to(case[0], target_is_directory=True)
    case[2].symlink_to(link / target.name)
    assert launcher.launch(case[5], WRAPPER, now=NOW, execve=lambda *_args: 29) == 29


def test_python_target_parent_segments_follow_symlink_semantics(case, monkeypatch):
    trusted = case[0] / "actual-python"
    case[2].rename(trusted)
    unsafe = case[0] / "unsafe-target"
    (unsafe / "child").mkdir(parents=True)
    (unsafe / trusted.name).write_bytes(trusted.read_bytes())
    (unsafe / trusted.name).chmod(0o700)
    redirect = case[0] / "redirect"
    redirect.symlink_to(unsafe / "child", target_is_directory=True)
    case[2].symlink_to(str(redirect) + "/../" + trusted.name)
    unsafe.chmod(0o777)
    forbid_key_open(monkeypatch, case[5])
    with pytest.raises(launcher.LauncherError, match="directory protection"):
        launcher.launch(case[5], WRAPPER, now=NOW, execve=never_exec)


def test_disabled_does_not_open_key_with_untrusted_parent(case, monkeypatch):
    case[4]["enabled"] = False
    save_config(case)
    parent = case[0] / "untrusted-key-parent"
    parent.mkdir(mode=0o700)
    key = parent / case[5].name
    case[5].rename(key)
    parent.chmod(0o777)
    forbid_key_open(monkeypatch, key)
    assert launcher.launch(key, WRAPPER, execve=lambda *_args: 13) == 13


def test_untrusted_payload_reads_remain_available_for_rollback(tmp_path):
    container = tmp_path.resolve() / "ordinary-payload"
    container.mkdir(mode=0o700)
    payload = container / "payload"
    payload.write_bytes(b"synthetic")
    container.chmod(0o777)
    assert launcher._read_regular(payload, 20) == b"synthetic"
