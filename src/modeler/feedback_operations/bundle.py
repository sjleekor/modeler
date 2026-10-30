"""Build an inert, narrowly inventoried operations/outcomes release for local review.

No credentials, runtime data, venv, installation, activation or network are involved.
The manifest hashes every source payload; the returned receipt hashes the manifest
and archive (neither can recursively contain its own digest).
"""

from __future__ import annotations

import argparse
import ast
import gzip
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

from modeler.feedback.core import FeedbackError, canonical_bytes, require_hash, sha256_bytes

WRAPPER_SHA256 = "b7dda874e353dc79918b12d5460ae2488f2f9a0bcb81473aa2b20fa7e21c93f3"
MAX_FILE_BYTES = 4 * 1024 * 1024
# Exact disabled template pins also prevent added credential fields or flag drift.
_TEMPLATE_SHA256 = {
    "deploy/feedback/runtime.disabled.json": (
        "6d52cbb0cb8f64ceea65c90dc0cdfd724c85e646e059931a51afabc02edc31cc"
    ),
    "deploy/feedback/capture.disabled.json": (
        "af3a79641f84720f09151a9e1611de08d470b56ccdafb9d7982d8676f2515265"
    ),
    "deploy/feedback/operations-event.disabled.json": (
        "792e04044bff2eb7bf10f59fb39b46d5ada18b7104de0610712eae23a04bf00d"
    ),
}
_PACKAGES = ("feedback", "feedback_operations", "feedback_outcomes")
_FIXED_FILES = (
    "src/modeler/__init__.py",
    "src/modeler/reporting/__init__.py",
    "src/modeler/reporting/markdown.py",
    "src/modeler/reporting/security_names.py",
    "src/modeler/serving/__init__.py",
    "src/modeler/serving/calendars.py",
    "src/modeler/serving/freshness.py",
    "src/modeler/serving/orchestration.py",
    "src/modeler/serving/schema.py",
    "src/modeler/serving/opening_validation.py",
    "deploy/feedback/feedback-stage.sh",
    "deploy/feedback/runtime.disabled.json",
    "deploy/feedback/capture.disabled.json",
    "deploy/feedback/operations-event.disabled.json",
    "deploy/reports/publish_reports.py",
    "deploy/reports/validate_reports.py",
)
_REQUIRED = {
    "src/modeler/feedback/__init__.py",
    "src/modeler/feedback/runtime.py",
    "src/modeler/feedback_operations/__init__.py",
    "src/modeler/feedback_operations/credential_launcher.py",
    "src/modeler/feedback_operations/bundle.py",
    "src/modeler/feedback_outcomes/__init__.py",
    "src/modeler/feedback_outcomes/__main__.py",
}


def _absolute(value):
    spelling = os.fspath(value)
    if not isinstance(spelling, str) or not spelling.startswith("/"):
        raise FeedbackError("bundle paths must be explicit absolute paths")
    if any(part in ("", ".", "..") for part in spelling.split("/")[1:]) or any(
        ord(char) < 32 or ord(char) == 127 for char in spelling
    ):
        raise FeedbackError("bundle path spelling is unsafe")
    path = Path(spelling)
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise FeedbackError("bundle paths cannot traverse symlinks")
    return path


def _allowed(name):
    if not isinstance(name, str) or (
        name not in _FIXED_FILES
        and not re.fullmatch(
            r"src/modeler/(?:feedback|feedback_operations|feedback_outcomes)/[a-z_][a-z_0-9]*\.py",
            name,
        )
    ):
        raise FeedbackError("bundle file is outside the explicit allowlist")
    return name


def _open_directory(path):
    directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:]:
            next_directory = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = next_directory
        return directory
    except BaseException:
        os.close(directory)
        raise


def _read_source(root, name):
    _allowed(name)  # Reject traversal and unknown names before any source open.
    path = _absolute(root / name)
    directory = _open_directory(path.parent)
    descriptor = None
    try:
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > MAX_FILE_BYTES
        ):
            raise FeedbackError("bundle source must be a bounded regular file")
        chunks = []
        remaining = MAX_FILE_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        current = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        identity = lambda info: (  # noqa: E731
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if (
            len(raw) != before.st_size
            or identity(before) != identity(after)
            or identity(before) != identity(current)
        ):
            raise FeedbackError("bundle source changed while reading")
        return raw
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def source_inventory(code_root):
    """Enumerate only the three flat Python packages and fixed integration files."""
    root = _absolute(code_root)
    names = set(_FIXED_FILES)
    for package in _PACKAGES:
        path = _absolute(root / "src/modeler" / package)
        directory = _open_directory(path)
        try:
            entries = os.listdir(directory)
        finally:
            os.close(directory)
        for entry in entries:
            if entry.endswith(".py"):
                names.add(_allowed(f"src/modeler/{package}/{entry}"))
    if not _REQUIRED <= names:
        raise FeedbackError("bundle source is missing required entrypoints")
    return sorted(names)


def _git(root, *args):
    environment = os.environ.copy()
    environment["GIT_NO_LAZY_FETCH"] = "1"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(
        ["git", "--no-pager", "--literal-pathspecs", "-C", str(root), *args],
        capture_output=True,
        check=False,
        timeout=10,
        env=environment,
    )
    return result.returncode, result.stdout


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise FeedbackError("duplicate template field")
        result[key] = value
    return result


def _json(raw):
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise FeedbackError("bundle template must be an object")
    return value


def _inert_templates(payload):
    for name, digest in _TEMPLATE_SHA256.items():
        if sha256_bytes(payload[name]) != digest:
            raise FeedbackError("bundle disabled template pin mismatch")
    if sha256_bytes(payload["deploy/feedback/feedback-stage.sh"]) != WRAPPER_SHA256:
        raise FeedbackError("bundle wrapper is not the reviewed wrapper")
    runtime = _json(payload["deploy/feedback/runtime.disabled.json"])
    capture = _json(payload["deploy/feedback/capture.disabled.json"])
    proposal = _json(payload["deploy/feedback/operations-event.disabled.json"])
    if (
        runtime.get("schema") != "prediction-feedback-runtime.v1"
        or runtime.get("enabled") is not False
        or runtime.get("mode") != "incremental"
        or runtime.get("source_mode") != "publisher-capture"
        or runtime.get("not_before_date") != "2026-11-07"
        or runtime.get("timeout_seconds") != 150
        or runtime.get("memory_limit_bytes") != 536870912
        or runtime.get("require_calendar_pin") is not True
        or runtime.get("require_preflight_provider") is not True
        or capture.get("schema") != "prediction-feedback-capture-settings.v1"
        or capture.get("enabled") is not False
        or capture.get("producer_scope")
        != {
            "reviewed": False,
            "registry_complete": False,
            "controlled_epoch": False,
            "registry_sha256": None,
            "producers": [],
        }
    ):
        raise FeedbackError("bundle templates must retain the disabled F3 safety contract")
    events = proposal.get("events")
    if (
        proposal.get("schema") != "prediction-feedback-operations-proposal.v1"
        or proposal.get("enabled") is not False
        or proposal.get("activation_not_before") != "2026-11-07"
        or proposal.get("requires_fresh_server_review") is not True
        or proposal.get("requires_live_category_target_plugin_review") is not True
        or proposal.get("is_cronicle_create_payload") is not False
        or not isinstance(events, list)
        or len(events) != 1
        or type(events[0].get("enabled")) is not int
        or events[0]["enabled"] != 0
        or events[0].get("timeout") != 180
        or events[0].get("memory_limit") != 536870912
        or any("KEY" in name or "FD" in name for name in events[0].get("environment", {}))
    ):
        raise FeedbackError("bundle event must remain an inert review proposal")


def _imports(raw):
    names = set()
    for node in ast.walk(ast.parse(raw)):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add("." * node.level + (node.module or ""))
    return sorted(names)


def _snapshot(code_root, expected_commit, files):
    root = _absolute(code_root)
    expected_commit = require_hash(expected_commit, length=40)
    status, head = _git(root, "rev-parse", "--verify", "HEAD")
    if status or head.decode("ascii").strip() != expected_commit:
        raise FeedbackError("bundle source Git HEAD does not match the expected commit")
    names = source_inventory(root)
    if files is not None:
        supplied = [_allowed(name) for name in files]
        if len(supplied) != len(set(supplied)) or set(supplied) != set(names):
            raise FeedbackError("bundle must include the complete narrow source inventory")
    payload = {name: _read_source(root, name) for name in names}
    _inert_templates(payload)
    rows = {}
    for name, raw in payload.items():
        digest = sha256_bytes(raw)
        status, committed = _git(root, "show", f"{expected_commit}:{name}")
        committed_digest = None if status else sha256_bytes(committed)
        rows[name] = {
            "sha256": digest,
            "bytes": len(raw),
            "committed_sha256": committed_digest,
            "matches_commit": digest == committed_digest,
            "imports": _imports(raw) if name.endswith(".py") else [],
        }
    status, end_head = _git(root, "rev-parse", "--verify", "HEAD")
    if status or end_head != head or source_inventory(root) != names:
        raise FeedbackError("bundle source inventory or Git HEAD changed during snapshot")
    changed = [name for name, row in rows.items() if not row["matches_commit"]]
    manifest = {
        "schema": "prediction-feedback-operations-bundle.v1",
        "source_git_commit": expected_commit,
        "source_state": "working_tree" if changed else "commit",
        "source_state_scope": "included source bytes only; unrelated Git dirt is not inspected",
        "changed_or_untracked_files": changed,
        "source_inventory_sha256": sha256_bytes(canonical_bytes(rows)),
        "files": rows,
        "archive_payload_prefix": "code/",
        "activation_performed": False,
        "dependency_notes": {
            "operational_runtime_and_launcher": "Python >=3.12 standard library; git and curl",
            "outcomes_writer": "pyarrow required in separately verified host venv",
            "venv_copied_or_dependencies_installed": False,
            "unsupported_optional_paths": [
                "feedback.shadow: polars and full modeler.models package are not bundled",
                "reporting.security_names name-input generation: polars is not bundled",
            ],
            "bootstrap": "protected PYTHONPATH=<release>/code/src before -s -P -m launcher",
            "credential": "dedicated protected key file provisioned separately; no key in bundle",
        },
    }
    return manifest, payload


def build_manifest(code_root, *, expected_commit, files=None):
    """Hash all allowlisted payload bytes with honest per-file Git provenance."""
    return _snapshot(code_root, expected_commit, files)[0]


def _archive_bytes(manifest_raw, payload):
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            members = {f"code/{name}": raw for name, raw in payload.items()}
            members["manifest.json"] = manifest_raw
            for name, raw in sorted(members.items()):
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                info.mode = 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(raw))
    return output.getvalue()


def build_bundle(code_root, destination, *, expected_commit):
    """Create only a new /private/tmp directory; never install or replace a release."""
    destination = _absolute(destination)
    if not destination.is_relative_to(Path("/private/tmp")) or destination == Path("/private/tmp"):
        raise FeedbackError("bundle destination must be a new directory under /private/tmp")
    if destination.exists():
        raise FeedbackError("bundle destination must not already exist")
    manifest, payload = _snapshot(code_root, expected_commit, None)
    manifest_raw = canonical_bytes(manifest) + b"\n"
    archive_raw = _archive_bytes(manifest_raw, payload)
    # mkdir is exclusive; all content comes from validated in-memory source bytes.
    destination.mkdir(mode=0o700)
    for name, raw in payload.items():
        path = destination / "code" / name
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
        path.chmod(0o644)
    for name, raw in (("manifest.json", manifest_raw), ("bundle.tar.gz", archive_raw)):
        with (destination / name).open("xb") as stream:
            stream.write(raw)
        (destination / name).chmod(0o600)
    return {
        "destination": str(destination),
        "manifest_sha256": sha256_bytes(manifest_raw),
        "archive_sha256": sha256_bytes(archive_raw),
        "source_inventory_sha256": manifest["source_inventory_sha256"],
        "source_git_commit": manifest["source_git_commit"],
        "source_state": manifest["source_state"],
        "payload_file_count": len(payload),
        "activation_performed": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.destination is None:
            result = build_manifest(args.code_root, expected_commit=args.expected_commit)
        else:
            result = build_bundle(
                args.code_root, args.destination, expected_commit=args.expected_commit
            )
        print(canonical_bytes(result).decode("utf-8"))
        return 0
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        print("feedback-operations-bundle: verification failed", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
