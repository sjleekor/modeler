#!/usr/bin/env python3
"""Validate and fast-forward publish an already-built SiteBuilder projection."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import fcntl

from validate_public_site import validate_site

EXPECTED_REPOSITORY = "sjleekor/market-briefing"
EXPECTED_BRANCH = "site"
CONFIG_KEYS = {
    "repository_confirmed", "checkout_dir", "projection_dir", "remote_name",
    "expected_remote_url", "branch", "base_path",
}
PUBLISH_STATE_VERSION = 1


def fail(message: str) -> None:
    raise ValueError(message)


def read_config(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        fail("config must be an explicit regular file")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read config: {exc}")
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        fail("config fields must exactly match config.example.json")
    if config["repository_confirmed"] is not True:
        fail("repository identity and Pages settings are not confirmed")
    if config["branch"] != EXPECTED_BRANCH:
        fail("only the dedicated site branch is allowed")
    if not isinstance(config["remote_name"], str) or not re.fullmatch(r"[A-Za-z0-9._-]+", config["remote_name"]):
        fail("invalid remote name")
    base_path = config["base_path"]
    if (not isinstance(base_path, str) or not re.fullmatch(r"/[A-Za-z0-9._/-]+/", base_path)
            or any(part in {".", ".."} for part in Path(base_path).parts)):
        fail("base_path must be the explicit absolute Pages project path")
    remote = config["expected_remote_url"]
    allowed_ssh = f"git@github.com:{EXPECTED_REPOSITORY}.git"
    allowed_https = f"https://github.com/{EXPECTED_REPOSITORY}.git"
    if remote not in {allowed_ssh, allowed_https}:
        fail("expected_remote_url must identify the reviewed dedicated repository without credentials")
    return config


def absolute_directory(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        fail(f"{label} must be an explicit absolute path")
    path = Path(value)
    if path.is_symlink() or not path.is_dir():
        fail(f"{label} must be an existing real directory")
    return path.resolve(strict=True)


def git(checkout: Path, *args: str, capture: bool = True) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args], check=False,
        text=True, stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if result.returncode:
        detail = (result.stderr or "").strip() if capture else ""
        fail(f"git {' '.join(args)} failed (exit {result.returncode}): {detail}")
    return (result.stdout or "").strip() if capture else ""


def git_status_code(checkout: Path, *args: str) -> int:
    result = subprocess.run(["git", "-C", str(checkout), *args], check=False,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode not in {0, 1}:
        fail(f"git {' '.join(args)} failed (exit {result.returncode}): {(result.stderr or '').strip()}")
    return result.returncode


def _state_path(checkout: Path) -> Path:
    return checkout.parent / f".{checkout.name}.pages-publish-state.json"


def _write_state(path: Path, state: dict[str, object]) -> None:
    if path.is_symlink() or path.exists():
        fail("Pages publish journal already exists; inspect it before continuing")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(state, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_state(path: Path) -> dict[str, object] | None:
    if path.is_symlink():
        fail("Pages publish journal cannot be a symlink")
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read Pages publish journal: {exc}")
    expected = {"version", "remote_url", "branch", "parent", "commit", "commit_tree",
                "public_tree", "allow_synthetic"}
    if not isinstance(state, dict) or set(state) != expected or state.get("version") != PUBLISH_STATE_VERSION:
        fail("Pages publish journal has an unknown format")
    if not isinstance(state.get("allow_synthetic"), bool):
        fail("Pages publish journal has an invalid fixture marker")
    return state


def _commit_state(checkout: Path, *, remote_url: str, allow_synthetic: bool) -> dict[str, object]:
    commit = git(checkout, "rev-parse", "HEAD")
    parents = git(checkout, "rev-list", "--parents", "-n", "1", commit).split()
    if len(parents) != 2:
        fail("publisher retry commit must have exactly one parent")
    paths = git(checkout, "diff-tree", "--no-commit-id", "--name-only", "-r", commit).splitlines()
    if not paths or any(path != "public" and not path.startswith("public/") for path in paths):
        fail("publisher commit contains changes outside public/")
    return {
        "version": PUBLISH_STATE_VERSION,
        "remote_url": remote_url,
        "branch": EXPECTED_BRANCH,
        "parent": parents[1],
        "commit": commit,
        "commit_tree": git(checkout, "rev-parse", f"{commit}^{{tree}}"),
        "public_tree": git(checkout, "rev-parse", f"{commit}:public"),
        "allow_synthetic": allow_synthetic,
    }


def _verify_retry_state(
    checkout: Path, state: dict[str, object], *, remote_url: str,
    remote_head: str, base_path: str, allow_synthetic: bool,
) -> None:
    current = _commit_state(checkout, remote_url=remote_url, allow_synthetic=bool(state["allow_synthetic"]))
    if current != state or git(checkout, "rev-parse", "HEAD") != state["commit"]:
        fail("local ahead commit does not match the publisher's recorded public-only commit")
    if state["remote_url"] != remote_url or state["branch"] != EXPECTED_BRANCH:
        fail("Pages publish journal targets a different remote or branch")
    if state["parent"] != remote_head:
        fail("remote site branch moved since the publisher commit; manual review is required")
    if state["allow_synthetic"] and not allow_synthetic:
        fail("retrying this synthetic site requires explicit --allow-synthetic")
    validate_site(checkout / "public", checkout / "public", base_path=base_path,
                  allow_synthetic=allow_synthetic)


def _remove_state(path: Path) -> None:
    if path.is_symlink():
        fail("Pages publish journal cannot be a symlink")
    path.unlink(missing_ok=True)


def _replace_public(source: Path, checkout: Path) -> None:
    destination = checkout / "public"
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        fail("checkout public path must be a real directory or absent")
    staged = Path(tempfile.mkdtemp(prefix=".public-stage-", dir=checkout))
    backup = checkout / ".public-previous-stage"
    try:
        shutil.rmtree(staged)
        shutil.copytree(source, staged, symlinks=False)
        if backup.exists() or backup.is_symlink():
            fail("stale public staging backup exists; inspect checkout before retrying")
        if destination.exists():
            os.replace(destination, backup)
        try:
            os.replace(staged, destination)
        except BaseException:
            if backup.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def publish(config: dict[str, object], *, allow_synthetic: bool) -> None:
    checkout = absolute_directory(config["checkout_dir"], "checkout_dir")
    projection = absolute_directory(config["projection_dir"], "projection_dir")
    if checkout == projection or checkout in projection.parents or projection in checkout.parents:
        fail("projection_dir and checkout_dir must be separate trees")
    lock_path = checkout.parent / f".{checkout.name}.pages-publish.lock"
    if lock_path.is_symlink():
        fail("publish lock path cannot be a symlink")
    with lock_path.open("a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        _publish_locked(config, checkout, projection, allow_synthetic=allow_synthetic)


def _publish_locked(config: dict[str, object], checkout: Path, projection: Path, *, allow_synthetic: bool) -> None:
    if checkout.name == "" or (checkout / ".git").is_symlink():
        fail("checkout must be a normal Git worktree")
    if git(checkout, "rev-parse", "--show-toplevel") != str(checkout):
        fail("checkout_dir is not the Git worktree root")
    if git(checkout, "status", "--porcelain"):
        fail("checkout must be clean before publishing")
    if git(checkout, "branch", "--show-current") != EXPECTED_BRANCH:
        fail("checkout must already be on the site branch")
    remote_name = str(config["remote_name"])
    expected_remote = str(config["expected_remote_url"])
    fetch_urls = git(checkout, "remote", "get-url", "--all", remote_name).splitlines()
    push_urls = git(checkout, "remote", "get-url", "--push", "--all", remote_name).splitlines()
    if fetch_urls != [expected_remote] or push_urls != [expected_remote]:
        fail("configured remote URL does not match the explicit repository target")
    state_path = _state_path(checkout)
    if state_path.is_symlink() or checkout in state_path.resolve(strict=False).parents or projection in state_path.resolve(strict=False).parents:
        fail("Pages publish journal must stay outside the checkout and projection")
    git(checkout, "fetch", remote_name, EXPECTED_BRANCH, capture=False)
    remote_head = git(checkout, "rev-parse", "FETCH_HEAD")
    state = _read_state(state_path)
    counts = git(checkout, "rev-list", "--left-right", "--count", f"HEAD...{remote_head}").split()
    if len(counts) != 2:
        fail("could not compare the local and remote site branches")
    if counts[0] != "0":
        if state is None:
            fail("local site branch contains an unjournaled commit; refusing to publish it")
        _verify_retry_state(checkout, state, remote_url=expected_remote, remote_head=remote_head,
                            base_path=str(config["base_path"]), allow_synthetic=allow_synthetic)
        git(checkout, "push", remote_name, f"{EXPECTED_BRANCH}:{EXPECTED_BRANCH}", capture=False)
        _remove_state(state_path)
        return
    if state is not None:
        if state.get("commit") != git(checkout, "rev-parse", "HEAD") or remote_head != state.get("commit"):
            fail("stale Pages publish journal does not match the current remote head")
        _verify_retry_state(checkout, state, remote_url=expected_remote, remote_head=state["parent"],
                            base_path=str(config["base_path"]), allow_synthetic=allow_synthetic)
        _remove_state(state_path)
    git(checkout, "merge", "--ff-only", "FETCH_HEAD", capture=False)
    if git(checkout, "status", "--porcelain"):
        fail("checkout changed unexpectedly after fast-forward")
    validate_site(projection, checkout / "public", base_path=str(config["base_path"]),
                  allow_synthetic=allow_synthetic)
    _replace_public(projection, checkout)
    git(checkout, "add", "--all", "--", "public", capture=False)
    diff_status = git_status_code(checkout, "diff", "--cached", "--quiet")
    if diff_status == 1:
        git(checkout, "commit", "-m", "Publish validated market briefing", capture=False)
        state = _commit_state(checkout, remote_url=expected_remote, allow_synthetic=allow_synthetic)
        _write_state(state_path, state)
        _verify_retry_state(checkout, state, remote_url=expected_remote,
                            remote_head=str(state["parent"]), base_path=str(config["base_path"]),
                            allow_synthetic=allow_synthetic)
        git(checkout, "push", remote_name, f"{EXPECTED_BRANCH}:{EXPECTED_BRANCH}", capture=False)
        _remove_state(state_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--publish", action="store_true", help="perform fetch, commit, and fast-forward push")
    parser.add_argument("--allow-synthetic", action="store_true",
                        help="explicitly permit a labeled synthetic fixture site")
    args = parser.parse_args()
    try:
        config = read_config(args.config)
        if not args.publish:
            fail("explicit --publish is required; no preview mode can publish")
        publish(config, allow_synthetic=args.allow_synthetic)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Pages publish stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
