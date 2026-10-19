"""동결 태그 가드 (사전등록 §0.3·§10 Q9·F10).

실제 자료로 전체 실행을 하는 진입점은 읽기 전에 :func:`require_frozen_tag`를 부른다.
통과 조건:

1. modeler 저장소(이 코드가 든 곳)의 HEAD에 태그 ``pension-alloc-frozen``이 붙어 있다.
2. modeler 작업 트리에 미커밋 변경이 없다(코드 sha가 결과를 만든 코드와 같다).
3. ``my`` 저장소의 HEAD에도 같은 이름의 태그가 붙어 있다(사전등록 문서가 동결된 상태).

통과하면 두 태그의 sha와 코드 sha를 돌려준다(실행 manifest에 적는다).
합성 자료 시험과 ``--schema-only``(값을 읽지 않는 점검)는 이 가드와 무관하다.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import modeler

FROZEN_TAG = "pension-alloc-frozen"
MY_REPO_ENV = "PENSION_MY_REPO"


class NotFrozenError(RuntimeError):
    """동결 태그 조건을 못 채웠다."""


def _repo_root() -> Path:
    return Path(modeler.__file__).resolve().parents[2]


def _default_my_root() -> Path:
    env = os.environ.get(MY_REPO_ENV)
    if env:
        return Path(env)
    # .../wss_p/modeler[/.claude/worktrees/x] -> .../wss_p/my
    for p in _repo_root().parents:
        if (p / "my" / ".git").exists():
            return p / "my"
    return Path("/Users/whishaw/wss_p/my")


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise NotFrozenError(f"git {' '.join(args)} 실패: {out.stderr.strip()}")
    return out.stdout.strip()


def _tag_at_head(root: Path, name: str) -> dict:
    tags = _git(root, "tag", "--points-at", "HEAD").split()
    if FROZEN_TAG not in tags:
        raise NotFrozenError(
            f"{name} 저장소 HEAD에 태그 {FROZEN_TAG}가 없습니다. "
            "동결 전에는 실제 자료로 전체 실행을 하지 않습니다."
        )
    return {
        "tag_object": _git(root, "rev-parse", FROZEN_TAG),
        "commit": _git(root, "rev-parse", "HEAD"),
    }


def require_frozen_tag(root: Path | None = None, my_root: Path | None = None) -> dict:
    """통과하면 ``{"code_sha", "modeler_tag", "my_tag"}``를 돌려준다. 아니면 예외."""
    root = root or _repo_root()
    my_root = my_root or _default_my_root()
    mod = _tag_at_head(root, "modeler")
    if _git(root, "status", "--porcelain"):
        raise NotFrozenError(
            "modeler 작업 트리에 미커밋 변경이 있습니다. 동결한 코드 그대로만 돌립니다."
        )
    my = _tag_at_head(my_root, "my")
    return {"code_sha": mod["commit"], "modeler_tag": mod, "my_tag": my}
