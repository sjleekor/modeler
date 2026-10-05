"""deploy/reports/validate_reports.py: 검사 6개와 symlink 거부, CLI 종료 코드."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from reports_world import PROJECT, REPORTS_DIR, World, git, pr, vr

from modeler.reporting import markdown as md

D1, D2 = "2026-10-07", "2026-10-08"


def unit_path(day: str) -> str:
    return f"reports/daily-briefing/{day[:4]}/{day[5:7]}/{day}"


@pytest.fixture()
def published(tmp_path: Path) -> World:
    world = World(tmp_path)
    world.make_run(D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    return world


def run_cli(*args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(PROJECT / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, str(REPORTS_DIR / "validate_reports.py"), *args],
        capture_output=True,
        text=True,
        env=env,
    )


def commit_all(world: World, message: str) -> str:
    git("-C", str(world.checkout), "add", "-A")
    git("-C", str(world.checkout), "commit", "-q", "-m", message)
    return git("-C", str(world.checkout), "rev-parse", "HEAD")


def test_cli_passes_a_published_tree_and_reports_each_violation_class(published: World) -> None:
    ok = run_cli("--repo", str(published.checkout))
    assert ok.returncode == 0 and "검증 통과" in ok.stdout
    root = published.checkout
    (root / "reference" / "leak.md").write_text(
        "서버 /home/whi 와 sj2 와 ghp_abc\n", encoding="utf-8"
    )
    (root / "reference" / "big.md").write_text("가" * (400 * 1024), encoding="utf-8")
    (root / "reference" / "img.png").write_bytes(b"x")
    (root / "data.csv").write_text("a,b\n", encoding="utf-8")
    (root / "reference" / "link.md").symlink_to(root / "README.md")
    bad = run_cli("--repo", str(root))
    assert bad.returncode == 1
    for needle in ("금지 문자열", "1MB", ".md만", "허용 경로 밖", "symlink"):
        assert needle in bad.stderr, needle


def test_cli_range_checks_only_the_changed_files(published: World) -> None:
    root = published.checkout
    git("-C", str(root), "checkout", "-q", "-b", "scratch")
    (root / "reference" / "glossary.md").write_text("서버 /home/whi\n", encoding="utf-8")
    broken = commit_all(published, "글로서리를 망가뜨림")
    # 전체 트리 검사는 걸리지만, 단위 커밋 하나만 보는 범위 검사는 그 파일을 보지 않습니다.
    assert run_cli("--repo", str(root)).returncode == 1
    base = git("-C", str(root), "rev-parse", f"{broken}~1")
    result = run_cli("--repo", str(root), "--range", base)
    assert result.returncode == 1 and "reference/glossary.md" in result.stderr
    clean = run_cli("--repo", str(root), "--range", broken)
    assert clean.returncode == 0


def test_range_check_needs_the_unit_to_be_announced(published: World) -> None:
    root = published.checkout
    parent = git("-C", str(root), "rev-parse", "HEAD~1")
    unannounced = run_cli("--repo", str(root), "--range", parent)
    assert (
        unannounced.returncode == 1 and "이번에 올리기로 한 단위가 아닙니다" in unannounced.stderr
    )
    announced = run_cli("--repo", str(root), "--range", parent, "--new-unit", D1)
    assert announced.returncode == 0, announced.stderr
    changed = run_cli("--repo", str(root), "--range", "HEAD", "--new-unit", D1)
    assert changed.returncode == 0  # HEAD..HEAD는 바뀐 것이 없습니다.


def test_commit_check_rejects_unit_edits_deletions_and_symlinks(published: World) -> None:
    root = published.checkout
    kr = root / unit_path(D1) / "kr-stocks.md"
    kr.write_text(kr.read_text(encoding="utf-8") + "\n수정\n", encoding="utf-8")
    edit = commit_all(published, "과거 단위 수정")
    problems = vr.validate_commit(root, edit, new_units=[D2])
    assert any("이미 올린 단위" in p and "--correct" in p for p in problems)
    assert vr.validate_commit(root, edit, new_units=[], corrected_units=[D1]) == []
    (root / unit_path(D1) / "us-stocks.md").unlink()
    deletion = commit_all(published, "삭제")
    assert any(
        "삭제는 허용하지 않습니다" in p for p in vr.validate_commit(root, deletion, new_units=[D2])
    )
    (root / "reference" / "link.md").symlink_to("../README.md")
    link = commit_all(published, "symlink")
    assert any("symlink" in p for p in vr.validate_commit(root, link, new_units=[D2]))


def test_model_card_rule_allows_only_new_cards_of_own_models(published: World) -> None:
    root = published.checkout
    card = root / "reference" / "models" / "kr_daily_h20_v1.md"
    card.write_text("고친 카드\n", encoding="utf-8")
    problems = vr.validate_working_tree(root, new_units=[D2], own_models=["kr_daily_h20_v1"])
    assert any("자기 모델의 새 카드만" in p for p in problems)
    git("-C", str(root), "checkout", "--", ".")
    other = root / "reference" / "models" / "someone_elses_model.md"
    other.write_text("---\nschema: stock-reports.v1\n---\n새 카드\n", encoding="utf-8")
    assert any("자기 모델의 새 카드만" in p for p in vr.validate_working_tree(root, new_units=[D2]))
    assert not [
        p
        for p in vr.validate_working_tree(root, new_units=[D2], own_models=["someone_elses_model"])
        if "자기 모델" in p
    ]


def test_working_tree_changes_classify_added_modified_and_deleted(published: World) -> None:
    root = published.checkout
    (root / "reference" / "new.md").write_text("x\n", encoding="utf-8")
    (root / "reference" / "glossary.md").write_text("y\n", encoding="utf-8")
    (root / "CONVENTIONS.md").unlink()
    changes = {path: code for code, path, _ in vr.working_tree_changes(root)}
    assert changes == {
        "reference/new.md": "A",
        "reference/glossary.md": "M",
        "CONVENTIONS.md": "D",
    }
    assert md.AUTO_BEGIN in (root / "README.md").read_text(encoding="utf-8")
