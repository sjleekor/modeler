"""Pass a dedicated protected Cronicle key to the reviewed wrapper through a pipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import unicodedata
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

WRAPPER_SHA256 = "b7dda874e353dc79918b12d5460ae2488f2f9a0bcb81473aa2b20fa7e21c93f3"
MAX_KEY_BYTES = 4096
SEOUL = ZoneInfo("Asia/Seoul")
_CREDENTIAL_ENV = ("FEEDBACK_CRONICLE_API_KEY", "FEEDBACK_CRONICLE_KEY_FD")


class LauncherError(ValueError):
    """An input failed validation; messages deliberately contain no input values."""


def _path(value, *, python=False):
    value = os.fspath(value)
    if (
        not isinstance(value, str)
        or not value.startswith("/")
        or value == "/"
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
        or any(part in ("", ".", "..") for part in value.split("/")[1:])
    ):
        raise LauncherError("invalid explicit path")
    path = Path(value)
    for part in path.parents:
        if part.is_symlink():
            raise LauncherError("unsafe path")
    if path.is_symlink() and not python:
        raise LauncherError("unsafe path")
    return path


def _identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _trusted_owner(info):
    return info.st_uid in (0, os.geteuid())


def _check_directory(info, *, traversal):
    writable = info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    sticky_container = traversal and info.st_uid == 0 and info.st_mode & stat.S_ISVTX
    if (
        not stat.S_ISDIR(info.st_mode)
        or not _trusted_owner(info)
        or (writable and not sticky_container)
    ):
        raise LauncherError("untrusted directory protection")


def _check_leaf(info):
    if not _trusted_owner(info) or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise LauncherError("untrusted file protection")


def _open_directory(path, *, trusted=False, trust_root=None):
    # Walk directories with openat/O_NOFOLLOW as well as checking path spelling.
    # A parent replaced with a symlink between validation and open cannot redirect us.
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        if trusted:
            _check_directory(os.fstat(directory), traversal=path != Path("/"))
        current = Path("/")
        for component in path.parts[1:]:
            next_directory = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = next_directory
            current /= component
            if trusted:
                _check_directory(
                    os.fstat(directory),
                    traversal=trust_root is None or not current.is_relative_to(trust_root),
                )
        return directory
    except BaseException:
        os.close(directory)
        raise


def _inspect_path(path, *, directory=False, trust_root=None):
    if directory:
        descriptor = _open_directory(path, trusted=True, trust_root=trust_root or path)
        os.close(descriptor)
        return
    parent = _open_directory(path.parent, trusted=True, trust_root=trust_root)
    descriptor = None
    try:
        before = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode):
            raise LauncherError("invalid runtime paths")
        _check_leaf(before)
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        if _identity(before) != _identity(os.fstat(descriptor)):
            raise LauncherError("file changed during open")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _inspect_python(path):
    # Keep the original argv/environment path: resolving a venv symlink for exec
    # would change Python's virtual-environment discovery.
    # Resolve links one at a time so a writable directory on the link's route
    # cannot disappear from validation when its target is canonicalized.
    candidate = path
    for _ in range(40):
        parts = candidate.parts
        for index, component in enumerate(parts[1:], start=1):
            parent_path = Path(*parts[:index])
            if component == "..":
                # Resolve earlier symlinks before interpreting '..'; normpath
                # could otherwise validate a different executable than exec.
                candidate = parent_path.parent / Path(*parts[index + 1 :])
                break
            parent = _open_directory(parent_path, trusted=True)
            try:
                before = os.stat(component, dir_fd=parent, follow_symlinks=False)
                if stat.S_ISLNK(before.st_mode):
                    if not _trusted_owner(before):
                        raise LauncherError("untrusted interpreter link")
                    target = os.readlink(component, dir_fd=parent)
                    if _identity(before) != _identity(
                        os.stat(component, dir_fd=parent, follow_symlinks=False)
                    ):
                        raise LauncherError("interpreter changed during validation")
                    candidate = parent_path / target / Path(*parts[index + 1 :])
                    break
                if index < len(parts) - 1:
                    _check_directory(before, traversal=index < len(parts) - 2)
                else:
                    _inspect_path(candidate, trust_root=candidate.parent)
                    if not os.access(path, os.X_OK):
                        raise LauncherError("invalid runtime paths")
                    return
            finally:
                os.close(parent)
        else:
            raise LauncherError("invalid runtime paths")
    raise LauncherError("interpreter link limit exceeded")


def _read_regular(path, maximum, *, protected=False, trusted=False, trust_root=None):
    # Rollback also uses this reader for ordinary payloads. Only launcher inputs
    # opt into trust enforcement; protected credential reads always enforce it.
    if protected and trust_root is None:
        trust_root = path.parent
    directory = _open_directory(path.parent, trusted=trusted or protected, trust_root=trust_root)
    descriptor = None
    try:
        before = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum:
            raise LauncherError("invalid file")
        if protected and (
            before.st_uid != os.geteuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
        ):
            raise LauncherError("invalid credential file protection")
        if trusted:
            _check_leaf(before)
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        opened = os.fstat(descriptor)
        if _identity(before) != _identity(opened):
            raise LauncherError("file changed during open")
        chunks = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if (
            len(content) != before.st_size
            or _identity(after) != _identity(opened)
            or _identity(current) != _identity(opened)
        ):
            raise LauncherError("file changed during read")
        return content
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def _read_key(path):
    raw = _read_regular(path, MAX_KEY_BYTES, protected=True)
    if raw.endswith(b"\r\n"):
        raw = raw[:-2]
    elif raw.endswith((b"\n", b"\r")):
        raw = raw[:-1]
    key = raw.decode("utf-8")
    if not key or any(unicodedata.category(char) == "Cc" for char in key):
        raise LauncherError("invalid credential content")
    return raw


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise LauncherError("duplicate configuration field")
        result[key] = value
    return result


def _code_pin(config, root):
    package = root / "src/modeler"
    feedback = _path(package / "feedback")
    _inspect_path(feedback, directory=True, trust_root=root)
    required = {"__init__.py", "reporting/__init__.py", "reporting/markdown.py"}
    required.update(path.relative_to(package).as_posix() for path in feedback.glob("*.py"))
    pins = config.get("code_files")
    if not isinstance(pins, dict) or set(pins) != required or "feedback/runtime.py" not in pins:
        raise LauncherError("invalid code inventory")
    for name in sorted(required):
        digest = pins[name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
            or hashlib.sha256(
                _read_regular(_path(package / name), 4 * 1024 * 1024, trusted=True, trust_root=root)
            ).hexdigest()
            != digest
        ):
            raise LauncherError("code pin mismatch")


def _activation(config, now):
    if config.get("mode") != "incremental":
        raise LauncherError("invalid activation mode")
    start = config.get("not_before_date")
    if not isinstance(start, str):
        raise LauncherError("invalid activation date")
    day = date.fromisoformat(start)
    if day.isoformat() != start or day < date(2026, 11, 7):
        raise LauncherError("invalid activation date")
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise LauncherError("invalid clock")
    local = now.astimezone(SEOUL)
    if local.date() < day or not time(10, 35) <= local.time().replace(tzinfo=None) < time(22, 45):
        raise LauncherError("activation window has not arrived")


def launch(key_file, wrapper, *, now=None, execve=os.execve):
    """Validate before reading a key, then replace this PID with the reviewed wrapper.

    The caller must provision a dedicated current-UID-owned 0600 key file separately.
    No network, service installation, environment sourcing, or activation is performed here.
    Trust checks cover POSIX ownership and modes, not ACLs. A malicious process
    running as the same UID can still replace trusted files after validation.
    """
    environment = os.environ.copy()
    for name in _CREDENTIAL_ENV:
        environment.pop(name, None)
    # All explicit path guards run before opening even the configuration.
    key_path = _path(key_file)
    wrapper_path = _path(wrapper)
    config_path = _path(environment.get("FEEDBACK_CONFIG", ""))
    python_path = _path(environment.get("FEEDBACK_PYTHON", ""), python=True)
    code_root = _path(environment.get("FEEDBACK_CODE_ROOT", ""))
    _inspect_python(python_path)
    _inspect_path(code_root, directory=True)
    for name in (
        "src/modeler/__init__.py",
        "src/modeler/feedback/__init__.py",
        "src/modeler/feedback/runtime.py",
    ):
        _inspect_path(_path(code_root / name), trust_root=code_root)
    wrapper_bytes = _read_regular(
        wrapper_path, 64 * 1024, trusted=True, trust_root=wrapper_path.parent
    )
    if hashlib.sha256(wrapper_bytes).hexdigest() != WRAPPER_SHA256:
        raise LauncherError("unreviewed wrapper")
    config = json.loads(
        _read_regular(config_path, 1024 * 1024, trusted=True, trust_root=config_path.parent),
        object_pairs_hook=_unique_object,
    )
    if (
        not isinstance(config, dict)
        or config.get("schema") != "prediction-feedback-runtime.v1"
        or type(config.get("enabled")) is not bool
    ):
        raise LauncherError("invalid runtime configuration")
    if not config["enabled"]:
        return execve("/bin/bash", ["/bin/bash", str(wrapper_path)], environment)
    _activation(config, datetime.now(UTC) if now is None else now)
    _code_pin(config, code_root)
    job_id = environment.get("JOB_ID")
    if not job_id or (environment.get("FEEDBACK_JOB_ID") not in (None, "", job_id)):
        raise LauncherError("verified Cronicle job context is required")
    environment["FEEDBACK_JOB_ID"] = job_id
    key = _read_key(key_path)
    read_fd, write_fd = os.pipe()
    try:
        os.set_inheritable(read_fd, True)
        # The bounded credential fits in a pipe without a reader running yet.
        if len(key) > os.fpathconf(write_fd, "PC_PIPE_BUF") or os.write(write_fd, key) != len(key):
            raise LauncherError("credential pipe failed")
        os.close(write_fd)
        write_fd = None
        environment["FEEDBACK_CRONICLE_KEY_FD"] = str(read_fd)
        return execve("/bin/bash", ["/bin/bash", str(wrapper_path)], environment)
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--wrapper", required=True)
    args = parser.parse_args(argv)
    try:
        launch(args.key_file, args.wrapper)
    except (OSError, ValueError, TypeError):
        print("feedback-credential-launcher: execution failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
