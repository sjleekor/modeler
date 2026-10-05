#!/usr/bin/env python3
"""일일 브리핑 단위를 markdown으로 렌더해 private 저장소 `stock_reports`의 main에 올립니다.

절차는 두 단계입니다(02 문서 §4, 리뷰 3·4). 원격에 닿지 않는 날에도 단위가 남게 하려는 것입니다.

로컬 단계 (`local`) — 원격에 접속하지 않습니다.
    L1 이번 실행의 내부 report인지 확인합니다(sha256, invocation id).
    L2 `runs/D/markdown/unit/`에 단위 파일을 렌더합니다.
    L3 단위 수준 검증을 합니다(front matter, 크기, 단위 안 링크, 서버 경로·비밀값).
    L4 journal에 "동기화 대기"를 적습니다.

동기화 단계 (`sync`) — 10:15·10:30 재시도와 다음 날 실행도 이 단계만 다시 돕니다.
    S1 flock으로 동시 실행을 막습니다.        S2 checkout을 확인합니다.
    S3 fetch합니다. 닿지 않으면 journal을 그대로 두고 끝냅니다.
    S4 journal의 push 대기 커밋을 먼저 처리합니다(밀린 날 -> 오늘 순서).
    S5 원격 최신 위에 자기 단위만 얹습니다. 같은 내용이면 새 커밋만 만들지 않습니다.
    S6 인덱스를 다시 만들고 트리를 검증합니다.   S7 커밋합니다.
    S8 force 없이 push합니다. non-fast-forward면 S3부터 다시, 최대 3회.

`run`은 두 단계를 차례로 돕니다. 정정은 `sync --correct <단위> --reason <사유>`로만 합니다.

release의 `modeler` 패키지가 `PYTHONPATH`에 있어야 합니다(coordinator가 맞춰 줍니다).
이 스크립트와 `validate_reports.py`는 release 밖에 두고 sha256을 고정합니다.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import validate_reports as vr

from modeler.reporting import markdown as md

EXPECTED_REPOSITORY = "sjleekor/stock_reports"
EXPECTED_REMOTE_URL = f"git@github.com:{EXPECTED_REPOSITORY}.git"
EXPECTED_BRANCH = "main"
AUDIENCE = "owner_only"
JOURNAL_VERSION = 1
MAX_PUSH_ATTEMPTS = 3
GIT_TIMEOUT = 120
LOCAL_LOCK_WAIT_SECONDS = 60.0
KST = timezone(timedelta(hours=9))

REQUIRED_KEYS = {"audience", "checkout_dir", "remote_name", "expected_remote_url", "branch"}
LOCAL_KEYS = {"release", "run_dir", "report_date", "report_sha256", "invocation_id"}
OPTIONAL_KEYS = {
    "model_cards_path",
    "top_n",
    "generated_at",
    "market_sector_input",
    "report_date",
}
ALLOWED_KEYS = REQUIRED_KEYS | LOCAL_KEYS | OPTIONAL_KEYS

# 단위 상태. journal에는 앞의 넷만 남습니다.
SYNC_PENDING = "sync_pending"
PUSH_PENDING = "push_pending"
REJECTED = "rejected"
CORRECTION_REQUIRED = "correction_required"
STATES = {SYNC_PENDING, PUSH_PENDING, REJECTED, CORRECTION_REQUIRED}
EXIT_CODES = {
    "published": 0,
    "unchanged": 0,
    "nothing_to_do": 0,
    "local_ready": 0,
    "failed": 1,
    "rejected": 10,
    "sync_pending": 20,
    "push_pending": 21,
    "correction_required": 30,
    "locked": 75,
}
# 한 번에 여러 단위를 처리했을 때 전체 상태를 정하는 우선순위(앞이 더 나쁩니다).
WORST_FIRST = ("failed", "rejected", "correction_required", "push_pending", "sync_pending")


class PublishError(Exception):
    """멈춰야 하는 오류입니다. 상태는 결과의 status가 됩니다."""

    def __init__(self, message: str, status: str = "failed") -> None:
        super().__init__(message)
        self.status = status


class TransportError(Exception):
    """원격에 닿지 않았거나 push가 거절됐습니다."""


class NonFastForward(TransportError):
    """원격 main이 앞서 나가 push가 거절됐습니다."""


def log(message: str) -> None:
    print(message, file=sys.stderr)


def now_kst() -> str:
    return datetime.now(KST).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
def read_config(path: Path, *, need_local: bool, strict_remote: bool = True) -> dict:
    """설정 파일을 읽어 검증합니다. 알 수 없는 키는 받지 않습니다."""
    if path.is_symlink() or not path.is_file():
        raise PublishError("config must be an explicit regular file")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PublishError(f"cannot read config: {type(exc).__name__}") from None
    if not isinstance(config, dict):
        raise PublishError("config must be a JSON object")
    check_config(config, need_local=need_local, strict_remote=strict_remote)
    return config


def check_config(config: dict, *, need_local: bool, strict_remote: bool = True) -> None:
    unknown = set(config) - ALLOWED_KEYS
    if unknown:
        raise PublishError(f"unknown config keys: {sorted(unknown)}")
    missing = (REQUIRED_KEYS | (LOCAL_KEYS if need_local else set())) - set(config)
    if missing:
        raise PublishError(f"missing config keys: {sorted(missing)}")
    if config["audience"] != AUDIENCE:
        raise PublishError("audience must be owner_only; review the repository settings first")
    if config["branch"] != EXPECTED_BRANCH:
        raise PublishError("only the main branch is allowed")
    if not isinstance(config["remote_name"], str) or not re.fullmatch(
        r"[A-Za-z0-9._-]+", config["remote_name"]
    ):
        raise PublishError("invalid remote name")
    if strict_remote and config["expected_remote_url"] != EXPECTED_REMOTE_URL:
        raise PublishError(f"expected_remote_url must be exactly {EXPECTED_REMOTE_URL}")
    if not isinstance(config["expected_remote_url"], str) or not config["expected_remote_url"]:
        raise PublishError("expected_remote_url is required")
    _absolute(config["checkout_dir"], "checkout_dir")
    if need_local:
        _absolute(config["run_dir"], "run_dir")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", str(config["release"])):
            raise PublishError("release may only contain letters, digits, '.', '_' and '-'")
        if not re.fullmatch(r"[0-9a-f]{64}", str(config["report_sha256"])):
            raise PublishError("report_sha256 must be a 64-hex SHA-256")
        if not isinstance(config["invocation_id"], str) or not config["invocation_id"]:
            raise PublishError("invocation_id is required")
        if md.good_date(config["report_date"]) is None:
            raise PublishError("report_date must use YYYY-MM-DD")
    top_n = config.get("top_n", md.DEFAULT_TOP_N)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or not 1 <= top_n <= 500:
        raise PublishError("top_n must be 1..500")


def _absolute(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/"):
        raise PublishError(f"{label} must be an explicit absolute path")
    return Path(value)


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------
def _git_env() -> dict:
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def git(
    checkout: Path, *args: str, check: bool = True, timeout: int = GIT_TIMEOUT
) -> subprocess.CompletedProcess:
    try:
        done = subprocess.run(
            ["git", "-C", str(checkout), *args],
            check=False,
            capture_output=True,
            text=True,
            env=_git_env(),
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise TransportError(f"git {args[0]} timed out") from None
    except OSError as exc:
        raise PublishError(f"git is not runnable: {type(exc).__name__}") from None
    if check and done.returncode:
        raise PublishError(
            f"git {' '.join(args[:2])} failed (exit {done.returncode}): {done.stderr.strip()[:300]}"
        )
    return done


def git_out(checkout: Path, *args: str) -> str:
    return git(checkout, *args).stdout.strip()


def fetch(checkout: Path, remote: str, branch: str) -> None:
    """S3. 원격 branch를 받아 `refs/remotes/<remote>/<branch>`를 갱신합니다."""
    spec = f"+refs/heads/{branch}:refs/remotes/{remote}/{branch}"
    done = git(checkout, "fetch", "--no-tags", remote, spec, check=False)
    if done.returncode:
        raise TransportError(f"fetch failed (exit {done.returncode}): {done.stderr.strip()[:300]}")


def push(checkout: Path, remote: str, branch: str) -> None:
    """S8. force 없이 push합니다."""
    done = git(checkout, "push", remote, f"{branch}:refs/heads/{branch}", check=False)
    if done.returncode == 0:
        return
    detail = done.stderr.strip()
    if "non-fast-forward" in detail or "fetch first" in detail:
        raise NonFastForward("remote branch moved; push was not a fast-forward")
    raise TransportError(f"push failed (exit {done.returncode}): {detail[:300]}")


def rev(checkout: Path, ref: str) -> str:
    return git_out(checkout, "rev-parse", "--verify", f"{ref}^{{commit}}")


def is_ancestor(checkout: Path, ancestor: str, descendant: str) -> bool:
    done = git(checkout, "merge-base", "--is-ancestor", ancestor, descendant, check=False)
    if done.returncode not in (0, 1):
        raise PublishError(f"git merge-base failed: {done.stderr.strip()[:200]}")
    return done.returncode == 0


def object_exists(checkout: Path, sha: str) -> bool:
    return git(checkout, "cat-file", "-e", f"{sha}^{{commit}}", check=False).returncode == 0


def rev_list(checkout: Path, spec: str) -> list[str]:
    return git_out(checkout, "rev-list", "--reverse", spec).split()


# ---------------------------------------------------------------------------
# 잠금과 journal
# ---------------------------------------------------------------------------
def lock_path(checkout: Path) -> Path:
    return checkout.parent / f".{checkout.name}.reports-publish.lock"


def journal_path(checkout: Path) -> Path:
    return checkout.parent / f".{checkout.name}.reports-publish-journal.json"


@contextmanager
def publisher_lock(checkout: Path, *, wait_seconds: float) -> Iterator[bool]:
    """S1. 잠금을 잡으면 True를 돌려줍니다. wait_seconds 안에 못 잡으면 False입니다."""
    path = lock_path(checkout)
    if path.is_symlink():
        raise PublishError("publish lock path cannot be a symlink")
    if not path.parent.is_dir():
        raise PublishError("checkout parent directory is missing")
    with path.open("a", encoding="utf-8") as handle:
        deadline = time.monotonic() + wait_seconds
        acquired = False
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.2)
        try:
            yield acquired
        finally:
            if acquired:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_journal(checkout: Path, remote_url: str, branch: str) -> dict:
    path = journal_path(checkout)
    if path.is_symlink():
        raise PublishError("publish journal cannot be a symlink")
    if not path.exists():
        return {"version": JOURNAL_VERSION, "remote_url": remote_url, "branch": branch, "units": {}}
    try:
        journal = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PublishError(f"cannot read publish journal: {type(exc).__name__}") from None
    if (
        not isinstance(journal, dict)
        or journal.get("version") != JOURNAL_VERSION
        or not isinstance(journal.get("units"), dict)
    ):
        raise PublishError("publish journal has an unknown format")
    if journal.get("remote_url") != remote_url or journal.get("branch") != branch:
        raise PublishError("publish journal targets a different remote or branch")
    for unit, entry in journal["units"].items():
        if (
            md.good_date(unit) is None
            or not isinstance(entry, dict)
            or entry.get("state") not in STATES
        ):
            raise PublishError("publish journal has an invalid unit entry")
    return journal


def write_journal(checkout: Path, journal: dict) -> None:
    """원자적으로 씁니다. 처리할 단위가 없으면 파일을 지웁니다."""
    path = journal_path(checkout)
    if path.is_symlink():
        raise PublishError("publish journal cannot be a symlink")
    if not journal["units"]:
        path.unlink(missing_ok=True)
        return
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(journal, stream, sort_keys=True, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def result(status: str, **fields: object) -> dict:
    return {"status": status, **fields}


# ---------------------------------------------------------------------------
# 로컬 단계 (L1~L4)
# ---------------------------------------------------------------------------
def _regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise PublishError(f"{label} is not a regular file")
    return path


def _unit_dir_local(markdown_dir: Path) -> Path:
    return markdown_dir / "unit"


def _read_unit_files(directory: Path) -> dict | None:
    if directory.is_symlink() or not directory.is_dir():
        return None
    files = {}
    for name in md.SECTION_FILES.values():
        path = directory / name
        if path.is_symlink() or not path.is_file():
            return None
        files[name] = path.read_text(encoding="utf-8")
    return files


def _write_atomic(path: Path, data: bytes) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _store_unit(markdown_dir: Path, files: dict) -> None:
    """단위 파일을 임시 디렉터리에 다 쓴 뒤 `unit/`으로 바꿉니다."""
    stage = Path(tempfile.mkdtemp(prefix=".unit-stage-", dir=markdown_dir))
    try:
        for name, text in files.items():
            (stage / name).write_bytes(text.encode("utf-8"))
        target = _unit_dir_local(markdown_dir)
        if target.exists() or target.is_symlink():
            shutil.rmtree(target)
        os.replace(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def local_step(config: dict, *, allow_synthetic: bool = False) -> dict:
    """로컬 단계. 원격에 접속하지 않습니다."""
    check_config(config, need_local=True, strict_remote=False)
    checkout = Path(config["checkout_dir"])
    with publisher_lock(checkout, wait_seconds=LOCAL_LOCK_WAIT_SECONDS) as acquired:
        if not acquired:
            return result("locked", unit=config["report_date"], local_done=False)
        try:
            return _local_locked(config, checkout, allow_synthetic=allow_synthetic)
        except PublishError as exc:
            return result("failed", unit=config["report_date"], detail=str(exc), local_done=False)


def _reject_local(config: dict, checkout: Path, markdown_dir: Path | None, problems: list) -> dict:
    """L1·L3 실패. 단위를 rejected로 남깁니다."""
    unit = config["report_date"]
    detail = [str(p)[:300] for p in problems[:20]]
    journal = read_journal(checkout, config["expected_remote_url"], config["branch"])
    entry = journal["units"].get(unit)
    if entry is not None and entry.get("commit"):
        # 앞서 검증을 통과해 커밋한 판이 있습니다. 그 판은 그대로 push하고, 새 판만 거절합니다.
        return result(
            REJECTED, unit=unit, detail=detail, local_done=True, kept_commit=entry["commit"]
        )
    journal["units"][unit] = {
        "state": REJECTED,
        "detail": detail,
        "updated_at": now_kst(),
        "run_dir": config["run_dir"],
    }
    write_journal(checkout, journal)
    if markdown_dir is not None and markdown_dir.is_dir():
        _write_atomic(
            markdown_dir / "rejected.json",
            (
                json.dumps({"unit": unit, "problems": detail}, ensure_ascii=False, indent=2) + "\n"
            ).encode(),
        )
    return result(REJECTED, unit=unit, detail=detail, local_done=True)


def _local_locked(config: dict, checkout: Path, *, allow_synthetic: bool) -> dict:
    unit = config["report_date"]
    run_dir = Path(config["run_dir"])
    if run_dir.is_symlink() or not run_dir.is_dir():
        raise PublishError("run_dir is not a real directory")
    markdown_dir = run_dir / "markdown"
    if markdown_dir.is_symlink():
        raise PublishError("markdown directory cannot be a symlink")
    markdown_dir.mkdir(exist_ok=True)

    # L1. 이번 실행의 report인지 확인합니다.
    problems = []
    report_path = _regular_file(run_dir / f"report-{unit}.json", "report")
    report_bytes = report_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != config["report_sha256"]:
        problems.append("report sha256가 coordinator가 고정한 값과 다릅니다")
    try:
        run_state = json.loads((run_dir / "run-state.json").read_text(encoding="utf-8"))
        run_invocation = run_state.get("invocation_id") if isinstance(run_state, dict) else None
    except (OSError, ValueError):
        run_invocation = None
    if run_invocation != config["invocation_id"]:
        problems.append("run-state.json의 invocation id가 이번 실행과 다릅니다")
    try:
        envelope = json.loads(report_bytes.decode("utf-8"))
    except ValueError:
        envelope = None
    if not isinstance(envelope, dict) or envelope.get("report_date") != unit:
        problems.append("report의 report_date가 단위 날짜와 다릅니다")
        envelope = {}
    if "invocation_id" in envelope and envelope["invocation_id"] != config["invocation_id"]:
        problems.append("report에 적힌 invocation id가 이번 실행과 다릅니다")
    if envelope.get("synthetic_fixture") is True and not allow_synthetic:
        problems.append("합성 fixture report는 --allow-synthetic이 있어야 올립니다")
    if problems:
        return _reject_local(config, checkout, markdown_dir, problems)
    ms_path = config.get("market_sector_input")
    ms_file = Path(ms_path) if ms_path else run_dir / f"market-sector-{unit}.json"
    ms_bytes = None
    if ms_file.is_file() and not ms_file.is_symlink():
        ms_bytes = ms_file.read_bytes()
    else:
        ms_file = None

    # L2. 렌더합니다. 같은 입력을 다시 돌려 내용이 같으면 파일을 그대로 둡니다(generated_at 고정).
    md.reset_scrub_hits()
    try:
        ctx = md.build_context(
            checkout,
            report_bytes,
            ms_bytes,
            config["release"],
            config.get("generated_at"),
            config.get("top_n", md.DEFAULT_TOP_N),
            log,
        )
        files = md.render_unit(ctx)
    except md.InputError as exc:
        return _reject_local(config, checkout, markdown_dir, [f"렌더하지 못했습니다: {exc}"])
    if ctx["unit"] != unit:
        return _reject_local(
            config, checkout, markdown_dir, ["렌더한 단위 날짜가 report_date와 다릅니다"]
        )
    previous = _read_unit_files(_unit_dir_local(markdown_dir))
    if previous is not None and md.same_unit_content(previous, files):
        files = previous
    else:
        _store_unit(markdown_dir, files)
    (markdown_dir / "rejected.json").unlink(missing_ok=True)

    # L3. 단위 수준 검증.
    problems = md.validate_unit_files(unit, files, allow_synthetic=allow_synthetic)
    if problems:
        return _reject_local(config, checkout, markdown_dir, problems)

    # L4. journal에 동기화 대기를 적습니다.
    unit_hash = md.unit_sha256(files)
    manifest = {
        "unit": unit,
        "release": config["release"],
        "report_sha256": config["report_sha256"],
        "invocation_id": config["invocation_id"],
        "top_n": config.get("top_n", md.DEFAULT_TOP_N),
        "market_sector_path": str(ms_file) if ms_file else None,
        "market_sector_sha256": (
            hashlib.sha256(ms_bytes).hexdigest() if ms_bytes is not None else None
        ),
        "unit_sha256": unit_hash,
        "scrub_hits": len(md.SCRUB_HITS),
    }
    _write_atomic(
        markdown_dir / "manifest.json",
        (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    journal = read_journal(checkout, config["expected_remote_url"], config["branch"])
    old = journal["units"].get(unit)
    entry = {
        "state": SYNC_PENDING,
        "run_dir": str(run_dir),
        "markdown_dir": str(markdown_dir),
        "unit_sha256": unit_hash,
        "report_sha256": config["report_sha256"],
        "invocation_id": config["invocation_id"],
        "release": config["release"],
        "top_n": manifest["top_n"],
        "market_sector_path": manifest["market_sector_path"],
        "market_sector_sha256": manifest["market_sector_sha256"],
        "synthetic_fixture": envelope.get("synthetic_fixture") is True,
        "updated_at": now_kst(),
    }
    if old and old.get("commit"):
        # 커밋은 했지만 push하지 못한 단위입니다. 같은 내용이면 그대로 두고, 다르면 동기화 단계가
        # 그 커밋을 버리고 새 내용으로 다시 만듭니다.
        entry["commit"] = old["commit"]
        entry["committed_source_sha256"] = old.get("committed_source_sha256")
        entry["commit_kind"] = old.get("commit_kind", "create")
        entry["state"] = PUSH_PENDING
    journal["units"][unit] = entry
    write_journal(checkout, journal)
    return result(
        "local_ready",
        unit=unit,
        unit_sha256=unit_hash,
        markdown_dir=str(markdown_dir),
        state=entry["state"],
        local_done=True,
    )


# ---------------------------------------------------------------------------
# 동기화 단계 (S1~S8)
# ---------------------------------------------------------------------------
def check_checkout(config: dict, checkout: Path) -> None:
    """S2. 깨끗한 checkout, branch main, remote URL이 정확히 같은지 봅니다."""
    if checkout.is_symlink() or not checkout.is_dir():
        raise PublishError("checkout_dir must be an existing real directory")
    if (checkout / ".git").is_symlink():
        raise PublishError("checkout must be a normal Git worktree")
    if Path(git_out(checkout, "rev-parse", "--show-toplevel")).resolve() != checkout.resolve():
        raise PublishError("checkout_dir is not the Git worktree root")
    if git_out(checkout, "status", "--porcelain"):
        raise PublishError("checkout must be clean before publishing")
    if git_out(checkout, "branch", "--show-current") != config["branch"]:
        raise PublishError("checkout must already be on the main branch")
    remote, expected = config["remote_name"], config["expected_remote_url"]
    fetch_urls = git_out(checkout, "remote", "get-url", "--all", remote).splitlines()
    push_urls = git_out(checkout, "remote", "get-url", "--push", "--all", remote).splitlines()
    if fetch_urls != [expected] or push_urls != [expected]:
        raise PublishError("configured remote URL does not match the explicit repository target")
    for key in ("user.name", "user.email"):
        # 작성자는 checkout 자체의 git 설정입니다(stock-reports-bot). 전역 설정에 기대지 않습니다.
        if not git(checkout, "config", "--local", "--get", key, check=False).stdout.strip():
            raise PublishError(f"checkout git config {key} is not set in the repository")


def _clean_reset(checkout: Path, target: str) -> None:
    git(checkout, "reset", "--hard", target)
    git(checkout, "clean", "-fd")


def _commit_subject(unit: str, revision: int, reason: str | None) -> str:
    if revision <= 1 or not reason:
        return f"{md.FAMILY} {unit}"
    one_line = " ".join(reason.split())[:200]
    return f"{md.FAMILY} {unit} r{revision}: {one_line}"


def _touch(entry: dict, **fields: object) -> None:
    entry.update(fields)
    entry["updated_at"] = now_kst()


def _apply_unit(
    config: dict,
    checkout: Path,
    unit: str,
    entry: dict,
    *,
    cards: dict,
    allow_synthetic: bool,
    correct: str | None,
    reason: str | None,
) -> str:
    """S5~S7. 단위 하나를 원격 최신 트리 위에 얹고 커밋합니다. 단위 상태 문자열을 돌려줍니다.

    `committed`면 entry에 커밋 정보를 적었고, `unchanged`면 entry를 지워도 됩니다.
    """
    markdown_dir = Path(entry["markdown_dir"])
    run_dir = Path(entry["run_dir"])
    if markdown_dir.parent != run_dir or markdown_dir.is_symlink():
        _touch(entry, state=REJECTED, detail=["markdown 디렉터리가 run 디렉터리 밖에 있습니다"])
        return REJECTED
    files = _read_unit_files(_unit_dir_local(markdown_dir))
    if files is None or md.unit_sha256(files) != entry["unit_sha256"]:
        _touch(entry, state=REJECTED, detail=["로컬 단계 뒤에 단위 파일이 바뀌었거나 없습니다"])
        return REJECTED
    udir = md.unit_dir(checkout, unit)
    existing = md.read_existing(udir)
    corrected = False
    revision = 1
    if existing is not None and "README.md" in existing:
        if _same_as_existing(config, entry, unit, checkout, existing, files):
            return "unchanged"
        if correct != unit:
            _touch(
                entry,
                state=CORRECTION_REQUIRED,
                detail=[
                    f"원격에 같은 단위({unit})가 다른 내용으로 있습니다. "
                    "고치려면 --correct가 필요합니다"
                ],
            )
            return CORRECTION_REQUIRED
        if not reason:
            _touch(entry, state=REJECTED, detail=["--correct에는 --reason이 필요합니다"])
            return REJECTED
        previous = git_out(checkout, "log", "-1", "--format=%h", "--", md.unit_rel(unit))
        files = _render_correction(config, entry, unit, checkout, existing, reason, previous)
        corrected, revision = True, md.existing_revision(files)
    elif correct == unit:
        _touch(entry, state=REJECTED, detail=["고칠 단위가 저장소에 없습니다"])
        return REJECTED
    problems = md.validate_unit_files(unit, files, allow_synthetic=allow_synthetic)
    if not problems:
        udir.mkdir(parents=True, exist_ok=True)
        for name, text in files.items():
            md.write_if_changed(udir / name, text)
        md.ensure_model_cards(checkout, cards, None)
        md.reindex(checkout, lambda message: None)
        problems = vr.validate_working_tree(
            checkout,
            new_units=[] if corrected else [unit],
            corrected_units=[unit] if corrected else [],
            own_models=vr.own_models_of(files),
            allow_synthetic=allow_synthetic,
        )
    if problems:
        _clean_reset(checkout, "HEAD")
        _touch(entry, state=REJECTED, detail=[str(p)[:300] for p in problems[:20]])
        return REJECTED
    paths = [path for _, path, _ in vr.working_tree_changes(checkout)]
    if not paths:
        return "unchanged"
    git(checkout, "add", "--", *paths)
    message = _commit_subject(unit, revision, reason if corrected else None)
    git(checkout, "-c", "commit.gpgsign=false", "commit", "-q", "-m", message)
    commit = rev(checkout, "HEAD")
    _touch(
        entry,
        state=PUSH_PENDING,
        commit=commit,
        committed_source_sha256=entry["unit_sha256"],
        commit_kind="correct" if corrected else "create",
        revision=revision,
        detail=None,
    )
    return "committed"


def _entry_context(config: dict, entry: dict, unit: str, checkout: Path, stamp: str) -> dict:
    """journal 항목에 적힌 입력(report, 시장·섹터)으로 렌더 입력을 다시 만듭니다."""
    run_dir = Path(entry["run_dir"])
    report_path = _regular_file(run_dir / f"report-{unit}.json", "report")
    report_bytes = report_path.read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != entry["report_sha256"]:
        raise PublishError("report changed since the local step")
    ms_bytes = None
    if entry.get("market_sector_path"):
        ms_bytes = Path(entry["market_sector_path"]).read_bytes()
        if hashlib.sha256(ms_bytes).hexdigest() != entry["market_sector_sha256"]:
            raise PublishError("market sector input changed since the local step")
    try:
        return md.build_context(
            checkout, report_bytes, ms_bytes, entry["release"], stamp, entry["top_n"], log
        )
    except md.InputError as exc:
        raise PublishError(f"cannot render from the journaled inputs: {exc}") from None


def _same_as_existing(
    config: dict, entry: dict, unit: str, checkout: Path, existing: dict, files: dict
) -> bool:
    """원격에 이미 있는 단위와 내용이 같은지 봅니다(generated_at·revision·정정 줄은 뺍니다).

    정정판은 요약에 "정정판입니다" 경고가 더 붙으므로, 같은 정정 줄로 다시 렌더해 비교합니다.
    """
    if md.same_unit_content(existing, files):
        return True
    old_corrections = md.existing_corrections(existing["README.md"])
    if not old_corrections:
        return False
    ctx = _entry_context(config, entry, unit, checkout, config.get("generated_at") or now_kst())
    ctx["revision"] = md.existing_revision(existing)
    ctx["corrections"] = old_corrections
    return md.same_unit_content(existing, md.render_unit(ctx))


def _render_correction(
    config: dict,
    entry: dict,
    unit: str,
    checkout: Path,
    existing: dict,
    reason: str,
    previous: str,
) -> dict:
    """정정: 같은 입력으로 revision을 올리고 요약 맨 위에 정정 줄을 넣어 다시 렌더합니다."""
    ctx = _entry_context(config, entry, unit, checkout, config.get("generated_at") or now_kst())
    return md.correct_unit_files(ctx, existing, reason, previous or None)


def _load_cards(config: dict) -> dict:
    path = config.get("model_cards_path")
    if not path:
        return {}
    try:
        cards = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PublishError(f"cannot read model cards: {type(exc).__name__}") from None
    if not isinstance(cards, dict) or not all(isinstance(v, dict) for v in cards.values()):
        raise PublishError("model cards must be a JSON object of objects")
    return cards


def _reconcile_local_commits(checkout: Path, remote: str, branch: str, journal: dict) -> None:
    """S4. 로컬이 원격보다 앞선 커밋을 journal과 맞춰 보고, 다시 얹어야 하면 되돌립니다."""
    origin_ref = f"refs/remotes/{remote}/{branch}"
    origin = rev(checkout, origin_ref)
    units = journal["units"]
    by_commit = {e["commit"]: u for u, e in units.items() if e.get("commit")}
    ahead = rev_list(checkout, f"{origin}..HEAD")
    if not ahead:
        for unit, entry in units.items():
            commit = entry.get("commit")
            if not commit:
                continue
            if object_exists(checkout, commit) and is_ancestor(checkout, commit, origin):
                entry["_pushed"] = True  # push는 됐는데 journal을 못 지운 경우입니다.
            else:
                _touch(entry, state=SYNC_PENDING, commit=None, committed_source_sha256=None)
        if rev(checkout, "HEAD") != origin and is_ancestor(checkout, "HEAD", origin):
            git(checkout, "merge", "--ff-only", origin_ref)
        return
    unknown = [c for c in ahead if c not in by_commit]
    if unknown:
        raise PublishError(
            "local branch contains a commit that the journal does not know; refusing"
        )
    for commit in ahead:
        unit = by_commit[commit]
        entry = units[unit]
        problems = vr.validate_commit(
            checkout,
            commit,
            new_units=[unit] if entry.get("commit_kind") != "correct" else [],
            corrected_units=[unit] if entry.get("commit_kind") == "correct" else [],
            own_models=_models_of_commit(checkout, commit, unit),
        )
        if problems:
            raise PublishError(f"journaled commit is not a unit-only change: {problems[0]}")
    stale = any(
        units[by_commit[c]].get("committed_source_sha256") != units[by_commit[c]]["unit_sha256"]
        for c in ahead
    )
    diverged = not is_ancestor(checkout, origin, "HEAD")
    if stale or diverged:
        _clean_reset(checkout, origin_ref)
        for commit in ahead:
            _touch(
                units[by_commit[commit]],
                state=SYNC_PENDING,
                commit=None,
                committed_source_sha256=None,
            )


def _models_of_commit(checkout: Path, commit: str, unit: str) -> set:
    files = {}
    for name in md.SECTION_FILES.values():
        text = vr.read_at(checkout, commit, f"{md.unit_rel(unit)}/{name}")
        if text is not None:
            files[name] = text
    return vr.own_models_of(files)


def _sync_round(
    config: dict,
    checkout: Path,
    journal: dict,
    summary: dict,
    *,
    cards: dict,
    allow_synthetic: bool,
    correct: str | None,
    reason: str | None,
) -> str:
    """한 번 돕니다. "done" · "retry"(non-ff) · "push_failed"를 돌려줍니다."""
    remote, branch = config["remote_name"], config["branch"]
    units = journal["units"]
    _reconcile_local_commits(checkout, remote, branch, journal)
    for unit in [u for u, e in units.items() if e.pop("_pushed", False)]:
        summary["units"][unit] = "published"
        del units[unit]
    write_journal(checkout, journal)
    for unit in sorted(units):
        entry = units[unit]
        if entry["state"] in (SYNC_PENDING, CORRECTION_REQUIRED):
            outcome = _apply_unit(
                config,
                checkout,
                unit,
                entry,
                cards=cards,
                allow_synthetic=allow_synthetic,
                correct=correct,
                reason=reason,
            )
            if outcome == "unchanged":
                summary["units"][unit] = "unchanged"
                del units[unit]
            elif outcome in (REJECTED, CORRECTION_REQUIRED):
                summary["units"][unit] = outcome
            write_journal(checkout, journal)
    origin_ref = f"refs/remotes/{remote}/{branch}"
    committed = {u: e for u, e in units.items() if e["state"] == PUSH_PENDING and e.get("commit")}
    if not rev_list(checkout, f"{origin_ref}..HEAD"):
        return "done"
    try:
        push(checkout, remote, branch)
    except NonFastForward:
        return "retry"
    except TransportError as exc:
        summary["detail"] = str(exc)
        for unit in committed:
            summary["units"][unit] = PUSH_PENDING
        return "push_failed"
    git(checkout, "update-ref", origin_ref, "HEAD")  # 방금 push했으므로 원격 main과 같습니다.
    for unit in committed:
        summary["units"][unit] = "published"
        del units[unit]
    write_journal(checkout, journal)
    return "done"


def sync_step(
    config: dict,
    *,
    allow_synthetic: bool = False,
    correct: str | None = None,
    reason: str | None = None,
) -> dict:
    """동기화 단계. 10:15·10:30 재시도와 다음 날 실행도 이 단계만 다시 돕니다."""
    check_config(config, need_local=False, strict_remote=False)
    checkout = Path(config["checkout_dir"])
    with publisher_lock(checkout, wait_seconds=0) as acquired:
        if not acquired:
            return result("locked", detail="another publisher run holds the lock")
        try:
            return _sync_locked(
                config, checkout, allow_synthetic=allow_synthetic, correct=correct, reason=reason
            )
        except PublishError as exc:
            return result(exc.status, detail=str(exc))


def _sync_locked(
    config: dict, checkout: Path, *, allow_synthetic: bool, correct: str | None, reason: str | None
) -> dict:
    remote, branch = config["remote_name"], config["branch"]
    if (correct is None) != (reason is None) or (
        correct is not None and md.good_date(correct) is None
    ):
        raise PublishError("--correct <YYYY-MM-DD>와 --reason은 함께 줘야 합니다")
    journal = read_journal(checkout, config["expected_remote_url"], branch)
    focus = config.get("report_date")
    if correct is not None:
        target = journal["units"].get(correct)
        if target is None or target["state"] not in (SYNC_PENDING, CORRECTION_REQUIRED):
            raise PublishError(
                f"{correct}: 고칠 단위가 대기 중이 아닙니다. local 단계를 먼저 돌리십시오"
            )
    if not journal["units"]:
        return result("nothing_to_do", unit=focus)
    check_checkout(config, checkout)
    cards = _load_cards(config)
    summary: dict = {"units": {}}
    outcome = "retry"
    for _attempt in range(MAX_PUSH_ATTEMPTS):
        try:
            fetch(checkout, remote, branch)  # S3
        except TransportError as exc:
            states = {u: e["state"] for u, e in journal["units"].items()}
            summary["units"] = {**states, **summary["units"]}
            return _finish(
                focus, summary, detail=str(exc), checkout=None, remote=remote, branch=branch
            )
        outcome = _sync_round(
            config,
            checkout,
            journal,
            summary,
            cards=cards,
            allow_synthetic=allow_synthetic,
            correct=correct,
            reason=reason,
        )
        if outcome != "retry":
            break
    detail = summary.get("detail")
    if outcome == "retry":
        detail = "push was rejected as non-fast-forward three times; try again later"
        for unit, entry in journal["units"].items():
            if entry["state"] == PUSH_PENDING:
                summary["units"][unit] = PUSH_PENDING
    for unit, entry in journal["units"].items():
        summary["units"].setdefault(unit, entry["state"])
    return _finish(focus, summary, detail=detail, checkout=checkout, remote=remote, branch=branch)


def _finish(
    focus: str | None,
    summary: dict,
    *,
    detail: str | None,
    checkout: Path | None,
    remote: str,
    branch: str,
) -> dict:
    """단위별 상태를 한 결과로 합칩니다. focus 단위가 있으면 그 상태가 전체 상태입니다."""
    units = summary["units"]
    if focus in units:
        status = units[focus]
    else:
        status = "nothing_to_do"
        for candidate in WORST_FIRST:
            if candidate in units.values():
                status = candidate
                break
        else:
            if units:
                status = "published" if "published" in units.values() else "unchanged"
    out = result(status, unit=focus, units=units, detail=detail)
    if checkout is not None and status in ("published", "unchanged"):
        out["commit"] = rev(checkout, f"refs/remotes/{remote}/{branch}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def run_step(config: dict, *, allow_synthetic: bool = False) -> dict:
    """로컬 단계를 돌고 이어서 동기화 단계를 돕니다. 로컬 단계가 막혀도 밀린 단위는 동기화합니다."""
    local = local_step(config, allow_synthetic=allow_synthetic)
    if local["status"] in ("locked", "failed"):
        return local
    synced = sync_step(config, allow_synthetic=allow_synthetic)
    synced["local_status"] = local["status"]
    synced["local_done"] = local.get("local_done", False)
    if local["status"] == REJECTED:
        synced["sync_status"] = synced["status"]
        synced["status"] = REJECTED
        synced["detail"] = local.get("detail")
    return synced


def main(argv: list[str] | None = None, *, strict_remote: bool = True) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=("local", "sync", "run"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--allow-synthetic", action="store_true", help="합성 fixture 단위를 허용")
    parser.add_argument("--correct", metavar="UNIT", help="이미 올린 단위를 고칩니다(sync만)")
    parser.add_argument("--reason", help="정정 사유(--correct와 함께)")
    args = parser.parse_args(argv)
    try:
        if args.correct and args.mode != "sync":
            raise PublishError("--correct는 sync에서만 쓸 수 있습니다")
        config = read_config(
            args.config, need_local=args.mode != "sync", strict_remote=strict_remote
        )
        if args.mode == "local":
            outcome = local_step(config, allow_synthetic=args.allow_synthetic)
        elif args.mode == "sync":
            outcome = sync_step(
                config,
                allow_synthetic=args.allow_synthetic,
                correct=args.correct,
                reason=args.reason,
            )
        else:
            outcome = run_step(config, allow_synthetic=args.allow_synthetic)
    except PublishError as exc:
        outcome = result(exc.status, detail=str(exc))
    print(json.dumps(outcome, sort_keys=True, ensure_ascii=False))
    return EXIT_CODES.get(outcome["status"], 1)


if __name__ == "__main__":
    raise SystemExit(main())
