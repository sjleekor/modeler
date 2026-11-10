"""publisher가 설정 `summary_v2_enabled`로 첫 화면 v2를 켜고 끄는 경로를 시험합니다.

원격은 로컬 bare 저장소입니다. 끄면(키가 없거나 false) 단위·manifest·journal이 이 기능 전과 바이트까지
같고, 켜면 README와 순위 섹션만 바뀌며 파일 집합·report hash·순위 표는 그대로입니다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from reports_world import World, pr
from test_summary_v2 import rank_table_lines

from modeler.reporting import markdown as md

D1 = "2026-10-07"
D2 = "2026-10-08"
FIVE = sorted(md.SECTION_FILES.values())


def unit_path(day: str) -> str:
    return f"reports/daily-briefing/{day[:4]}/{day[5:7]}/{day}"


def local_files(world: World, day: str) -> dict:
    root = world.local_unit(day)
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(root.rglob("*.md"))
    }


def manifest_bytes(world: World, day: str) -> bytes:
    return (world.runs / day / "markdown" / "manifest.json").read_bytes()


@pytest.fixture()
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def test_off_or_absent_is_byte_identical_and_on_changes_only_the_display(world: World) -> None:
    world.make_run(D1)
    plain = pr.local_step(world.config(D1))
    plain_files = local_files(world, D1)
    plain_manifest = manifest_bytes(world, D1)
    off = pr.local_step(world.config(D1, summary_v2_enabled=False))
    assert off["unit_sha256"] == plain["unit_sha256"]
    assert manifest_bytes(world, D1) == plain_manifest
    assert "summary_v2_enabled" not in world.journal()["units"][D1]
    on = pr.local_step(world.config(D1, summary_v2_enabled=True))
    assert on["status"] == "local_ready" and on["unit_sha256"] != plain["unit_sha256"]
    files = local_files(world, D1)
    assert sorted(files) == FIVE
    assert "## 종목 순위 — 어떤 목록인가" in files["README.md"]
    assert files["kr-stocks.md"].count("### 목록 요약") == 1
    for name in ("market-sector.md", "data-status.md"):
        assert files[name] == plain_files[name]
    for name in ("kr-stocks.md", "us-stocks.md"):
        assert rank_table_lines(files[name]) == rank_table_lines(plain_files[name])
    manifest = json.loads(manifest_bytes(world, D1))
    assert manifest["summary_v2_enabled"] is True
    assert manifest["report_sha256"] == json.loads(plain_manifest)["report_sha256"]
    assert world.journal()["units"][D1]["summary_v2_enabled"] is True
    # 다시 끄면 그날 단위는 전과 같은 바이트로 돌아갑니다.
    back = pr.local_step(world.config(D1, summary_v2_enabled=False))
    assert back["unit_sha256"] == plain["unit_sha256"]
    assert manifest_bytes(world, D1) == plain_manifest


def test_config_value_must_be_a_bool(world: World) -> None:
    world.make_run(D1)
    for bad in ("yes", 1, None):
        with pytest.raises(pr.PublishError, match="summary_v2_enabled"):
            pr.check_config(
                world.config(D1, summary_v2_enabled=bad), need_local=True, strict_remote=False
            )


def test_published_unit_marks_the_same_base_session_of_the_previous_unit(world: World) -> None:
    world.make_run(D1)
    assert pr.run_step(world.config(D1, summary_v2_enabled=True))["status"] == "published"
    world.make_run(D2)  # 합성 fixture는 두 날 모두 기준일이 2026-10-06입니다
    assert pr.run_step(world.config(D2, summary_v2_enabled=True))["status"] == "published"
    readme = world.remote_file(f"{unit_path(D2)}/README.md")
    assert f"(**전 단위 {D1}와 같은 기준 세션**)" in readme
    us = world.remote_file(f"{unit_path(D2)}/us-stocks.md")
    assert f"전 단위({D1})와 같습니다." in us
    assert md.validate(world.clone_for_checks()) == []


def test_offline_sync_and_rerun_keep_the_v2_unit(world: World) -> None:
    world.make_run(D1)
    world.go_offline()
    assert pr.run_step(world.config(D1, summary_v2_enabled=True))["status"] == "sync_pending"
    world.come_online()
    assert pr.sync_step(world.config(D1, summary_v2_enabled=True))["status"] == "published"
    readme = world.remote_file(f"{unit_path(D1)}/README.md")
    assert "## 종목 순위 — 어떤 목록인가" in readme
    head = world.remote_head()
    again = pr.run_step(
        world.config(D1, summary_v2_enabled=True, generated_at=f"{D1}T10:40:00+09:00")
    )
    assert again["status"] == "unchanged" and world.remote_head() == head


def test_correction_rerenders_with_the_journaled_switch(world: World) -> None:
    world.make_run(D1)
    world.go_offline()
    assert pr.run_step(world.config(D1, summary_v2_enabled=True))["status"] == "sync_pending"
    entry = world.journal()["units"][D1]
    ctx = pr._entry_context(world.config(D1), entry, D1, world.checkout, f"{D1}T11:00:00+09:00")
    assert ctx.get("summary_v2") is True  # 그날 설정에 키가 없어도 journal 값을 따릅니다
    entry.pop("summary_v2_enabled")
    ctx = pr._entry_context(world.config(D1), entry, D1, world.checkout, f"{D1}T11:00:00+09:00")
    assert "summary_v2" not in ctx
