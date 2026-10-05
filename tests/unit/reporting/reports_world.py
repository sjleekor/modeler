"""reports publisher 시험이 함께 쓰는 로컬 bare remote, checkout, run 디렉터리 도우미.

원격은 로컬 디렉터리의 bare 저장소입니다. GitHub에는 접속하지 않습니다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import md_fixtures as mf

from modeler.reporting import markdown as md

PROJECT = Path(__file__).resolve().parents[3]
REPORTS_DIR = PROJECT / "deploy" / "reports"
CARDS = PROJECT / "deploy" / "prod" / "model-cards.json"
RELEASE = "r20261005"
if str(REPORTS_DIR) not in sys.path:
    sys.path.insert(0, str(REPORTS_DIR))

import publish_reports as pr  # noqa: E402
import validate_reports as vr  # noqa: E402

__all__ = [
    "CARDS",
    "PROJECT",
    "REPORTS_DIR",
    "RELEASE",
    "World",
    "git",
    "no_log",
    "pr",
    "sha",
    "vr",
]


def git(*args: str, cwd: Path | None = None) -> str:
    done = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_log(_message: str) -> None:
    return None


class World:
    """bare remote + 뼈대 커밋 + publisher checkout."""

    def __init__(self, tmp_path: Path) -> None:
        self.tmp = tmp_path
        self.bare = tmp_path / "stock_reports.git"
        self.checkout = tmp_path / "reports-checkout"
        self.runs = tmp_path / "runs"
        self.other = tmp_path / "user-clone"
        git("init", "--bare", "--initial-branch=main", str(self.bare))
        seed = tmp_path / "seed"
        seed.mkdir()
        git("init", "--initial-branch=main", str(seed))
        git("-C", str(seed), "config", "user.name", "Owner")
        git("-C", str(seed), "config", "user.email", "owner@example.invalid")
        md.init_repo(seed, CARDS, no_log)
        git("-C", str(seed), "add", "README.md", "CONVENTIONS.md", "reference", "reports")
        git("-C", str(seed), "commit", "-q", "-m", "뼈대")
        git("-C", str(seed), "remote", "add", "origin", str(self.bare))
        git("-C", str(seed), "push", "-q", "origin", "main:main")
        git("clone", "-q", "--branch", "main", str(self.bare), str(self.checkout))
        git("-C", str(self.checkout), "config", "user.name", "stock-reports-bot")
        git("-C", str(self.checkout), "config", "user.email", "bot@example.invalid")
        self.hidden = tmp_path / "stock_reports.git.away"

    # --- 원격 -------------------------------------------------------------------------
    def remote_head(self) -> str:
        return git("--git-dir", str(self.bare), "rev-parse", "refs/heads/main")

    def remote_log(self) -> list[str]:
        out = git("--git-dir", str(self.bare), "log", "--format=%s", "refs/heads/main")
        return out.splitlines()

    def remote_file(self, path: str) -> str:
        return git("--git-dir", str(self.bare), "show", f"refs/heads/main:{path}")

    def remote_has(self, path: str) -> bool:
        done = subprocess.run(
            ["git", "--git-dir", str(self.bare), "cat-file", "-e", f"refs/heads/main:{path}"],
            capture_output=True,
        )
        return done.returncode == 0

    def user_push(self, name: str = "reference/notes.md", text: str = "사용자 메모\n") -> str:
        """사용자가 다른 clone에서 직접 올린 커밋입니다(원격이 앞서 나가는 경우)."""
        if not self.other.exists():
            git("clone", "-q", "--branch", "main", str(self.bare), str(self.other))
            git("-C", str(self.other), "config", "user.name", "Owner")
            git("-C", str(self.other), "config", "user.email", "owner@example.invalid")
        else:
            git("-C", str(self.other), "pull", "-q", "--ff-only", "origin", "main")
        target = self.other / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        git("-C", str(self.other), "add", name)
        git("-C", str(self.other), "commit", "-q", "-m", f"사용자 수정 {name}")
        git("-C", str(self.other), "push", "-q", "origin", "main:main")
        return git("-C", str(self.other), "rev-parse", "HEAD")

    def go_offline(self) -> None:
        self.bare.rename(self.hidden)

    def come_online(self) -> None:
        self.hidden.rename(self.bare)

    def block_pushes(self) -> Path:
        """pre-receive 훅이 push를 거절하게 합니다. 파일을 지우면 다시 받습니다."""
        flag = self.tmp / "reject-pushes"
        flag.write_text("x")
        hook = self.bare / "hooks" / "pre-receive"
        hook.write_text(
            f'#!/bin/sh\n[ -e "{flag}" ] && echo "rejected by test hook" >&2 && exit 1\nexit 0\n'
        )
        hook.chmod(0o755)
        return flag

    # --- run 디렉터리와 설정 ------------------------------------------------------------
    def make_run(
        self,
        day: str,
        *,
        prev: str | None = None,
        score: float | None = None,
        synthetic: bool = False,
        market_sector: bool = True,
    ) -> Path:
        """그날의 내부 report와 run-state를 만듭니다. 내용은 합성 fixture입니다."""
        prev = prev or "2026-10-06"
        reports = [
            mf.kr_report(day, prev),
            mf.us_report(day, mf.LGB_ID, prev, prev, prev),
            mf.us_report(day, mf.RDG_ID, prev, prev, prev),
        ]
        if score is not None:
            reports[0]["rankings"][0]["score"] = score
        envelope = mf.envelope(day, reports, [])
        envelope["synthetic_fixture"] = synthetic
        run_dir = self.runs / day
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / f"report-{day}.json").write_text(
            json.dumps(envelope, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_dir / "run-state.json").write_text(
            json.dumps({"invocation_id": f"inv-{day}"}) + "\n", encoding="utf-8"
        )
        ms = run_dir / f"market-sector-{day}.json"
        if market_sector:
            ms.write_text(
                json.dumps(mf.ms_input(day, kr_asof=prev, us_asof=prev, macro_asof="2026-10-04")),
                encoding="utf-8",
            )
        else:
            ms.unlink(missing_ok=True)
        return run_dir

    def config(self, day: str, **overrides: object) -> dict:
        run_dir = self.runs / day
        config = {
            "audience": "owner_only",
            "checkout_dir": str(self.checkout),
            "remote_name": "origin",
            "expected_remote_url": str(self.bare),
            "branch": "main",
            "release": RELEASE,
            "run_dir": str(run_dir),
            "report_date": day,
            "report_sha256": sha(run_dir / f"report-{day}.json"),
            "invocation_id": f"inv-{day}",
            "model_cards_path": str(CARDS),
            "generated_at": f"{day}T10:03:12+09:00",
        }
        config.update(overrides)
        return config

    def journal(self) -> dict | None:
        path = pr.journal_path(self.checkout)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def local_unit(self, day: str) -> Path:
        return self.runs / day / "markdown" / "unit"

    def clone_for_checks(self, name: str = "verify-clone") -> Path:
        """원격 main을 새로 받아 트리를 검사할 수 있게 합니다."""
        target = self.tmp / name
        if target.exists():
            shutil.rmtree(target)
        git("clone", "-q", "--branch", "main", str(self.bare), str(target))
        return target
