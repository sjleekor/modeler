"""동결 태그 가드 (사전등록 §0.3·§10 Q9).

실제 자료를 읽는 모든 진입점은 읽기 전에 :func:`require_frozen_tag`를 부른다.
통과 조건: (1) 이 코드가 든 저장소의 HEAD에 태그 ``pension-alloc-frozen``이 붙어 있다.
(2) 작업 트리에 미커밋 변경이 없다(코드 sha가 결과를 만든 코드와 같다).
합성 자료 시험은 이 가드와 무관하다(실제 파일을 열지 않는다).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modeler

FROZEN_TAG = "pension-alloc-frozen"


class NotFrozenError(RuntimeError):
    """동결 태그 조건을 못 채웠다."""


def _repo_root() -> Path:
    return Path(modeler.__file__).resolve().parents[2]


def _git(root: Path, *args: str) -> str:
    out = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if out.returncode != 0:
        raise NotFrozenError(f"git {' '.join(args)} 실패: {out.stderr.strip()}")
    return out.stdout.strip()


def require_frozen_tag(root: Path | None = None) -> str:
    """통과하면 HEAD sha를 돌려준다. 아니면 :class:`NotFrozenError`."""
    root = root or _repo_root()
    tags = _git(root, "tag", "--points-at", "HEAD").split()
    if FROZEN_TAG not in tags:
        raise NotFrozenError(
            f"HEAD에 태그 {FROZEN_TAG}가 없습니다. 동결 전에는 실제 자료를 읽지 않습니다."
        )
    if _git(root, "status", "--porcelain"):
        raise NotFrozenError("작업 트리에 미커밋 변경이 있습니다. 동결한 코드 그대로만 돌립니다.")
    return _git(root, "rev-parse", "HEAD")
