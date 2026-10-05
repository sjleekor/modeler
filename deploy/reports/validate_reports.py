#!/usr/bin/env python3
"""stock_reports 저장소에 올리려는 변경을 검증합니다 (02 문서 §5의 검사 6개와 symlink 거부).

| # | 검사 | 어디서 |
|---|---|---|
| 1 | 경로 allowlist: 자기 종류 폴더, 루트 README의 자동 구간, 자기 모델 카드 | check_change_set |
| 2 | front matter schema | modeler.reporting.markdown.validate |
| 3 | 파일 크기 1MB | modeler.reporting.markdown.validate |
| 4 | 상대 링크 | modeler.reporting.markdown.validate |
| 5 | 과거 단위 보호: 단위의 변경·삭제 거부(--correct한 단위만 예외) | check_change_set |
| 6 | 서버 경로·호스트명·키 모양 문자열 | modeler.reporting.markdown.validate |

publisher는 동기화 단계에서 이 모듈의 `validate_working_tree`를 부릅니다. 수동으로는
`--repo`만 주면 저장소 전체를, `--range BASE`를 주면 BASE..HEAD가 바꾼 파일만 봅니다.
release의 `modeler` 패키지가 `PYTHONPATH`에 있어야 합니다. 원격에는 접속하지 않습니다.

종료 코드: 0 통과, 1 위반, 2 사용법·git 오류.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

from modeler.reporting import markdown as md

GIT_TIMEOUT = 60
AUTO_ZONE = re.compile(re.escape(md.AUTO_BEGIN) + r".*?" + re.escape(md.AUTO_END), re.DOTALL)
FAMILY_README = f"reports/{md.FAMILY}/README.md"
MONTH_README = re.compile(rf"^reports/{re.escape(md.FAMILY)}/\d{{4}}/\d{{2}}/README\.md$")
SYMLINK_MODES = {"120000", "160000"}  # symlink, submodule

# (상태 A·M·D, 경로, 새 파일 모드). 작업 트리 변경은 모드를 알 수 없어 빈 문자열입니다.
Change = tuple[str, str, str]


class GitError(Exception):
    """git 명령이 실패했습니다."""


def _git(checkout: Path, *args: str) -> bytes:
    try:
        done = subprocess.run(
            ["git", "-C", str(checkout), *args],
            check=False,
            capture_output=True,
            timeout=GIT_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GitError(f"git {' '.join(args[:2])} failed: {type(exc).__name__}") from None
    if done.returncode:
        detail = done.stderr.decode("utf-8", "replace").strip()[:200]
        raise GitError(f"git {' '.join(args[:2])} failed (exit {done.returncode}): {detail}")
    return done.stdout


def working_tree_changes(checkout: Path) -> list[Change]:
    """HEAD와 작업 트리의 차이를 돌려줍니다. 추적하지 않는 새 파일은 A입니다."""
    raw = _git(checkout, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    out: list[Change] = []
    records = raw.decode("utf-8", "surrogateescape").split("\0")
    index = 0
    while index < len(records):
        record = records[index]
        index += 1
        if not record:
            continue
        status, path = record[:2], record[3:]
        if status[0] in "RC":  # 이름 바꾸기·복사는 원래 경로가 한 항목 더 옵니다.
            index += 1
            out.append(("A", path, ""))
        elif "D" in status:
            out.append(("D", path, ""))
        elif status == "??" or "A" in status:
            out.append(("A", path, ""))
        else:
            out.append(("M", path, ""))
    return sorted(out, key=lambda item: item[1])


def commit_changes(checkout: Path, commit: str) -> list[Change]:
    """커밋 하나(부모 하나)가 바꾼 파일을 돌려줍니다."""
    parents = _git(checkout, "rev-list", "--parents", "-n", "1", commit).decode().split()
    if len(parents) != 2:
        raise GitError("commit must have exactly one parent")
    raw = _git(checkout, "diff-tree", "-r", "-z", "--no-renames", "--raw", "--no-commit-id", commit)
    parts = raw.decode("utf-8", "surrogateescape").split("\0")
    out: list[Change] = []
    for index in range(0, len(parts) - 1, 2):
        header, path = parts[index], parts[index + 1]
        fields = header.lstrip(":").split()
        out.append((fields[4][0], path, fields[1]))
    return sorted(out, key=lambda item: item[1])


def read_at(checkout: Path, revision: str, path: str) -> str | None:
    """revision의 파일 내용을 돌려줍니다. 없으면 None입니다."""
    try:
        return _git(checkout, "show", f"{revision}:{path}").decode("utf-8")
    except (GitError, UnicodeDecodeError):
        return None


def outside_auto(text: str) -> str:
    """자동 구간을 뺀 나머지입니다. 앞뒤 공백은 무시합니다."""
    return AUTO_ZONE.sub("", text).strip()


def check_change_set(
    checkout: Path,
    changes: Iterable[Change],
    *,
    new_units: Iterable[str],
    corrected_units: Iterable[str] = (),
    own_models: Iterable[str] = (),
    base_revision: str = "HEAD",
    head_revision: str | None = None,
) -> list[str]:
    """바꾸는 파일 목록이 허용 범위 안인지 봅니다 (검사 1과 5).

    `new_units`는 이번에 새로 만드는 단위, `corrected_units`는 고치도록 허가한 단위입니다.
    `head_revision`이 없으면 작업 트리의 파일을 새 내용으로 읽습니다.
    """
    new, corrected, models = set(new_units), set(corrected_units), set(own_models)
    problems: list[str] = []
    for code, path, mode in changes:
        if mode in SYMLINK_MODES:
            problems.append(f"{path}: symlink는 허용하지 않습니다")
            continue
        if code == "D":
            problems.append(f"{path}: 삭제는 허용하지 않습니다 (단위와 문서는 지우지 않습니다)")
            continue
        if code not in ("A", "M"):
            problems.append(f"{path}: 지원하지 않는 변경 종류입니다 ({code})")
            continue
        unit_match = md.UNIT_PATH_RE.match(path) or md.ASSET_PATH_RE.match(path)
        if unit_match:
            unit = unit_match.group(3)
            if code == "M" and unit not in corrected:
                problems.append(
                    f"{path}: 이미 올린 단위({unit})는 바꿀 수 없습니다. "
                    "고치려면 --correct <단위> --reason을 쓰십시오"
                )
            elif code == "A" and unit not in new and unit not in corrected:
                problems.append(f"{path}: 이번에 올리기로 한 단위가 아닙니다 ({unit})")
            continue
        if path == "README.md" or path == FAMILY_README or MONTH_README.match(path):
            if code == "M" and not _only_auto_zone_changed(
                checkout, path, base_revision, head_revision
            ):
                problems.append(
                    f"{path}: 자동 구간 밖을 바꿨습니다 (사람이 쓴 부분은 건드리지 않습니다)"
                )
            continue
        if path.startswith("reference/models/") and path.count("/") == 2 and path.endswith(".md"):
            model_id = path.rsplit("/", 1)[1][: -len(".md")]
            if not (code == "A" and model_id in models):
                problems.append(f"{path}: 자기 모델의 새 카드만 만들 수 있습니다")
            continue
        problems.append(
            f"{path}: 허용 경로 밖입니다 (자기 종류 폴더, 루트 README 자동 구간, 모델 카드만)"
        )
    return problems


def _only_auto_zone_changed(
    checkout: Path, path: str, base_revision: str, head_revision: str | None
) -> bool:
    old = read_at(checkout, base_revision, path)
    if head_revision is None:
        try:
            new = (checkout / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return False
    else:
        new = read_at(checkout, head_revision, path)
    if old is None or new is None:
        return False
    return outside_auto(old) == outside_auto(new)


def own_models_of(unit_files: dict) -> set[str]:
    """단위 파일의 front matter `models`에 적힌 모델 id입니다."""
    found: set[str] = set()
    for text in unit_files.values():
        try:
            meta, _ = md.parse_front_matter(text)
        except ValueError:
            continue
        models = (meta or {}).get("models")
        if isinstance(models, list):
            found.update(m for m in models if isinstance(m, str))
    return found


def validate_working_tree(
    checkout: Path,
    *,
    new_units: Iterable[str],
    corrected_units: Iterable[str] = (),
    own_models: Iterable[str] = (),
    allow_synthetic: bool = False,
) -> list[str]:
    """커밋 직전의 작업 트리를 HEAD와 비교해 검증합니다 (동기화 단계 S6)."""
    changes = working_tree_changes(checkout)
    problems = check_change_set(
        checkout,
        changes,
        new_units=new_units,
        corrected_units=corrected_units,
        own_models=own_models,
    )
    paths = [path for code, path, _ in changes if code != "D"]
    problems += md.validate(checkout, paths=paths, allow_synthetic=allow_synthetic)
    return problems


def validate_commit(
    checkout: Path,
    commit: str,
    *,
    new_units: Iterable[str],
    corrected_units: Iterable[str] = (),
    own_models: Iterable[str] = (),
) -> list[str]:
    """이미 만든 커밋 하나가 허용 범위 안의 변경만 담았는지 봅니다 (push 재시도 전)."""
    return check_change_set(
        checkout,
        commit_changes(checkout, commit),
        new_units=new_units,
        corrected_units=corrected_units,
        own_models=own_models,
        base_revision=f"{commit}^",
        head_revision=commit,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True, type=Path, help="stock_reports checkout")
    parser.add_argument(
        "--range", dest="base", help="BASE..HEAD가 바꾼 파일만 검사합니다. 없으면 전체 트리"
    )
    parser.add_argument("--new-unit", action="append", default=[], help="새로 올리는 단위")
    parser.add_argument("--correct", action="append", default=[], help="고치도록 허가한 단위")
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.base:
            raw = _git(args.repo, "diff", "--name-status", "-z", "--no-renames", args.base, "HEAD")
            tokens = raw.decode("utf-8", "surrogateescape").split("\0")
            changes = [(tokens[i][0], tokens[i + 1], "") for i in range(0, len(tokens) - 1, 2)]
            problems = check_change_set(
                args.repo,
                changes,
                new_units=args.new_unit,
                corrected_units=args.correct,
                own_models=[
                    p.rsplit("/", 1)[1][:-3] for _, p, _ in changes if "reference/models/" in p
                ],
                base_revision=args.base,
                head_revision="HEAD",
            )
            problems += md.validate(
                args.repo,
                paths=[path for code, path, _ in changes if code != "D"],
                allow_synthetic=args.allow_synthetic,
            )
        else:
            problems = md.validate(args.repo, allow_synthetic=args.allow_synthetic)
    except GitError as exc:
        print(f"validate_reports: {exc}", file=sys.stderr)
        return 2
    if problems:
        print(f"검증 위반 {len(problems)}건", file=sys.stderr)
        for item in problems:
            print(f" - {item}", file=sys.stderr)
        return 1
    print("검증 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
