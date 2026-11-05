"""publisher가 근거 입력(explain-D.json)과 그림(assets/*.svg)을 나르는 경로를 시험합니다.

원격은 로컬 bare 저장소입니다.

계획 P2 완료 조건 ③④: 같은 단위 재실행은 "내용 같음", 정정 재렌더·네트워크 실패 뒤 재시도·
커밋 뒤 push 실패 재시도에서 그림이 유지됩니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import explain_fixtures as ef
import pytest
from reports_world import World, git, pr

from modeler.reporting import markdown as md

D1 = "2026-10-07"
FIVE = tuple(md.SECTION_FILES.values())
ASSETS = ("assets/kr-top20.svg", "assets/us-lightgbm-top20.svg", "assets/us-ridge-top20.svg")


@pytest.fixture()
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def unit_path(day: str) -> str:
    return f"reports/daily-briefing/{day[:4]}/{day[5:7]}/{day}"


def write_explain(world: World, day: str, *, invocation: str | None = None, edit=None) -> Path:
    run_dir = world.runs / day
    doc = ef.explain_doc((run_dir / f"report-{day}.json").read_bytes(), invocation or f"inv-{day}")
    if edit is not None:
        edit(doc)
    path = run_dir / f"explain-{day}.json"
    path.write_bytes(ef.dump(doc))
    return path


def remote_assets(world: World, day: str) -> list:
    out = git(
        "--git-dir",
        str(world.bare),
        "ls-tree",
        "--name-only",
        "refs/heads/main",
        f"{unit_path(day)}/assets/",
    )
    return sorted(line.rsplit("/", 1)[1] for line in out.splitlines())


def local_names(world: World, day: str) -> list:
    root = world.local_unit(day)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_publish_carries_svgs_and_records_the_explain_input(world: World) -> None:
    world.make_run(D1)
    explain = write_explain(world, D1)
    result = pr.run_step(world.config(D1))
    assert result["status"] == "published"
    assert remote_assets(world, D1) == sorted(a.split("/")[1] for a in ASSETS)
    assert local_names(world, D1) == sorted(FIVE + ASSETS)
    remote_svg = git(
        "--git-dir", str(world.bare), "show", f"refs/heads/main:{unit_path(D1)}/{ASSETS[0]}"
    )
    assert remote_svg.encode() + b"\n" == (world.local_unit(D1) / ASSETS[0]).read_bytes()
    manifest = json.loads((world.runs / D1 / "markdown" / "manifest.json").read_text())
    assert manifest["explain_path"] == str(explain)
    assert manifest["explain_sha256"] == md.sha256_bytes(explain.read_bytes())
    assert manifest["explain_state"] == "ok"
    kr = world.remote_file(f"{unit_path(D1)}/kr-stocks.md")
    assert "](assets/kr-top20.svg)" in kr and "- 근거·그래프: 정상" in kr
    assert md.validate(world.clone_for_checks()) == []


def test_same_unit_rerun_with_svgs_is_unchanged(world: World) -> None:
    world.make_run(D1)
    write_explain(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    head = world.remote_head()
    again = pr.run_step(world.config(D1, generated_at=f"{D1}T10:40:00+09:00"))
    assert again["status"] == "unchanged" and world.remote_head() == head
    assert world.journal() is None


def test_disabled_or_absent_explain_publishes_todays_unit(world: World) -> None:
    world.make_run(D1)
    plain = pr.local_step(world.config(D1))
    write_explain(world, D1)
    off = pr.local_step(world.config(D1, explain_enabled=False))
    assert off["unit_sha256"] == plain["unit_sha256"]
    assert local_names(world, D1) == sorted(FIVE)
    manifest = json.loads((world.runs / D1 / "markdown" / "manifest.json").read_text())
    assert "explain_path" not in manifest and "explain_path" not in world.journal()["units"][D1]
    on = pr.local_step(world.config(D1))
    assert on["unit_sha256"] != plain["unit_sha256"]
    with pytest.raises(pr.PublishError, match="explain_enabled"):
        pr.check_config(
            world.config(D1, explain_enabled="no"), need_local=True, strict_remote=False
        )


def test_stale_explain_of_another_invocation_renders_failed_line(world: World) -> None:
    world.make_run(D1)
    write_explain(world, D1, invocation="inv-earlier")
    assert pr.run_step(world.config(D1))["status"] == "published"
    assert remote_assets(world, D1) == []
    kr = world.remote_file(f"{unit_path(D1)}/kr-stocks.md")
    assert "- 근거·그래프: 실패 — 입력 불일치. 표는 근거 없이 냈습니다." in kr


def test_symlinked_explain_is_not_followed(world: World, tmp_path: Path) -> None:
    world.make_run(D1)
    target = tmp_path / "elsewhere.json"
    target.write_bytes(
        ef.dump(ef.explain_doc((world.runs / D1 / f"report-{D1}.json").read_bytes(), f"inv-{D1}"))
    )
    (world.runs / D1 / f"explain-{D1}.json").symlink_to(target)
    assert pr.run_step(world.config(D1))["status"] == "published"
    assert "실패 — 근거 파일 형식 오류" in world.remote_file(f"{unit_path(D1)}/kr-stocks.md")


def test_retry_after_network_failure_keeps_svgs(world: World) -> None:
    world.make_run(D1)
    write_explain(world, D1)
    world.go_offline()
    assert pr.run_step(world.config(D1))["status"] == "sync_pending"
    assert local_names(world, D1) == sorted(FIVE + ASSETS)
    world.come_online()
    assert pr.sync_step(world.config(D1))["status"] == "published"
    assert len(remote_assets(world, D1)) == 3


def test_push_failure_after_commit_then_retry_keeps_svgs(world: World) -> None:
    world.make_run(D1)
    write_explain(world, D1)
    flag = world.block_pushes()
    assert pr.run_step(world.config(D1))["status"] == "push_pending"
    commit = world.journal()["units"][D1]["commit"]
    shown = git("-C", str(world.checkout), "show", "--name-only", "--format=", commit).splitlines()
    assert sum(1 for line in shown if "/assets/" in line) == 3
    flag.unlink()
    again = pr.run_step(world.config(D1))
    assert again["status"] == "published" and world.remote_head() == commit
    assert len(remote_assets(world, D1)) == 3


def test_correction_rerenders_with_the_same_explain_and_keeps_svgs(world: World) -> None:
    world.make_run(D1)
    write_explain(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    before = world.remote_file(f"{unit_path(D1)}/{ASSETS[0]}")
    world.make_run(D1, score=0.123456)  # report가 바뀌면 근거 입력도 새 report에 맞춰 다시 씁니다
    write_explain(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "correction_required"
    fixed = pr.sync_step(world.config(D1), correct=D1, reason="KR 점수 재계산")
    assert fixed["status"] == "published"
    assert len(remote_assets(world, D1)) == 3
    assert world.remote_file(f"{unit_path(D1)}/{ASSETS[0]}") == before
    kr = world.remote_file(f"{unit_path(D1)}/kr-stocks.md")
    assert "0.1235" in kr and "](assets/kr-top20.svg)" in kr
    meta, _ = md.parse_front_matter(kr)
    assert meta["revision"] == 2
    assert pr.run_step(world.config(D1))["status"] == "unchanged"


def test_correction_without_explain_removes_old_svgs(world: World) -> None:
    world.make_run(D1)
    write_explain(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    assert pr.run_step(world.config(D1, explain_enabled=False))["status"] == "correction_required"
    fixed = pr.sync_step(world.config(D1, explain_enabled=False), correct=D1, reason="근거 표시 끔")
    assert fixed["status"] == "published"
    assert remote_assets(world, D1) == []
    assert "60세션 흐름" not in world.remote_file(f"{unit_path(D1)}/kr-stocks.md")
    assert md.validate(world.clone_for_checks()) == []


def test_explain_changed_after_the_local_step_blocks_the_correction(world: World) -> None:
    world.make_run(D1)
    write_explain(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    world.make_run(D1, score=0.123456)
    path = write_explain(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "correction_required"
    path.write_bytes(path.read_bytes().replace(b'"status": "ok"', b'"status": "partial"', 1))
    blocked = pr.sync_step(world.config(D1), correct=D1, reason="KR 점수 재계산")
    assert blocked["status"] == "failed" and "explain input changed" in blocked["detail"]


def test_local_unit_with_a_stray_or_symlinked_asset_is_rejected_at_sync(world: World) -> None:
    world.make_run(D1)
    write_explain(world, D1)
    world.go_offline()
    assert pr.run_step(world.config(D1))["status"] == "sync_pending"
    svg = world.local_unit(D1) / ASSETS[0]
    data = svg.read_bytes()
    svg.unlink()
    svg.symlink_to(world.tmp / "nowhere.svg")
    (world.tmp / "nowhere.svg").write_bytes(data)
    world.come_online()
    result = pr.sync_step(world.config(D1))
    assert result["status"] == "rejected"
    assert world.remote_log() == ["뼈대"]
