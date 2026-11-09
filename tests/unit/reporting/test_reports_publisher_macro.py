"""publisher가 거시 흐름 입력(macro-D.json)과 macro.md·assets/macro-*.svg를 나르는 경로를
시험합니다.

원격은 로컬 bare 저장소입니다. 계산·렌더 모듈은 `macro_fakes`의 가짜입니다.
계획 M2-c 완료 조건: 실패해도 단위가 안 막힘, 끄면 바이트 동일, 재실행은 "내용 같음",
정정 재렌더·재시도에서 그림 유지.
"""

from __future__ import annotations

import json
from pathlib import Path

import macro_fakes as mk
import pytest
from reports_world import World, git, pr

from modeler.reporting import markdown as md

D1 = "2026-10-07"
FIVE = tuple(md.SECTION_FILES.values())
PAGE = md.MACRO_FILE
SVG = "assets/macro-rates.svg"


class MacroWorld(World):
    """Daily publisher configs with the section switched on (the coordinator adds the key only
    when it is on; a missing key means off)."""

    def config(self, day: str, **overrides: object) -> dict:
        return super().config(day, **{"macro_context_enabled": True, **overrides})


@pytest.fixture()
def world(tmp_path: Path, monkeypatch) -> World:
    mk.install(monkeypatch)
    return MacroWorld(tmp_path)


def unit_path(day: str) -> str:
    return f"reports/daily-briefing/{day[:4]}/{day[5:7]}/{day}"


def write_macro(world: World, day: str, *, invocation: str | None = None, **kw) -> Path:
    path = world.runs / day / f"macro-{day}.json"
    path.write_bytes(mk.dump(mk.doc(day, invocation or f"inv-{day}", **kw)))
    return path


def local_names(world: World, day: str) -> list:
    root = world.local_unit(day)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def remote_names(world: World, day: str) -> list:
    out = git(
        "--git-dir",
        str(world.bare),
        "ls-tree",
        "-r",
        "--name-only",
        "refs/heads/main",
        f"{unit_path(day)}/",
    )
    return sorted(line[len(unit_path(day)) + 1 :] for line in out.splitlines())


def test_publish_carries_the_page_and_svg_and_records_the_input(world: World) -> None:
    world.make_run(D1)
    macro = write_macro(world, D1)
    result = pr.run_step(world.config(D1))
    assert result["status"] == "published"
    assert remote_names(world, D1) == sorted((*FIVE, PAGE, SVG))
    assert local_names(world, D1) == sorted((*FIVE, PAGE, SVG))
    manifest = json.loads((world.runs / D1 / "markdown" / "manifest.json").read_text())
    assert manifest["macro_path"] == str(macro)
    assert manifest["macro_sha256"] == md.sha256_bytes(macro.read_bytes())
    assert manifest["macro_state"] == "ok"
    assert "| [거시 흐름](macro.md) | 정상 |" in world.remote_file(f"{unit_path(D1)}/README.md")
    assert "거시 흐름은 [macro.md](macro.md)" in world.remote_file(
        f"{unit_path(D1)}/market-sector.md"
    )
    assert md.validate(world.clone_for_checks()) == []


def test_journal_keeps_the_macro_input_until_sync(world: World) -> None:
    world.make_run(D1)
    macro = write_macro(world, D1)
    world.go_offline()
    assert pr.run_step(world.config(D1))["status"] == "sync_pending"
    entry = world.journal()["units"][D1]
    assert entry["macro_path"] == str(macro) and entry["macro_state"] == "ok"
    assert entry["macro_sha256"] == md.sha256_bytes(macro.read_bytes())
    world.come_online()
    assert pr.sync_step(world.config(D1))["status"] == "published"
    assert SVG in remote_names(world, D1)


def test_same_unit_rerun_is_unchanged(world: World) -> None:
    world.make_run(D1)
    write_macro(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    head = world.remote_head()
    again = pr.run_step(world.config(D1, generated_at=f"{D1}T10:40:00+09:00"))
    assert again["status"] == "unchanged" and world.remote_head() == head
    assert world.journal() is None


def test_disabled_or_absent_macro_publishes_todays_unit_byte_for_byte(world: World) -> None:
    world.make_run(D1)
    plain = pr.local_step(world.config(D1))
    plain_manifest = (world.runs / D1 / "markdown" / "manifest.json").read_bytes()
    write_macro(world, D1)
    off = pr.local_step(world.config(D1, macro_context_enabled=False))
    assert off["unit_sha256"] == plain["unit_sha256"]
    assert local_names(world, D1) == sorted(FIVE)
    assert (world.runs / D1 / "markdown" / "manifest.json").read_bytes() == plain_manifest
    assert "macro_path" not in world.journal()["units"][D1]
    absent = pr.local_step(World.config(world, D1))  # no key at all = off
    assert absent["unit_sha256"] == plain["unit_sha256"]
    assert local_names(world, D1) == sorted(FIVE)
    assert (world.runs / D1 / "markdown" / "manifest.json").read_bytes() == plain_manifest
    on = pr.local_step(world.config(D1))
    assert on["unit_sha256"] != plain["unit_sha256"]
    with pytest.raises(pr.PublishError, match="macro_context_enabled"):
        pr.check_config(
            world.config(D1, macro_context_enabled="no"), need_local=True, strict_remote=False
        )


def test_failed_document_still_publishes_the_unit_with_a_notice(world: World) -> None:
    world.make_run(D1)
    write_macro(world, D1, state="failed", reason="timeout")
    assert pr.run_step(world.config(D1))["status"] == "published"
    page = world.remote_file(f"{unit_path(D1)}/{PAGE}")
    assert "거시 흐름을 내지 못했습니다: timeout." in page
    assert SVG not in remote_names(world, D1)
    manifest = json.loads((world.runs / D1 / "markdown" / "manifest.json").read_text())
    assert manifest["macro_state"] == "failed"


def test_stale_document_of_another_invocation_is_a_notice(world: World) -> None:
    world.make_run(D1)
    write_macro(world, D1, invocation="inv-earlier")
    assert pr.run_step(world.config(D1))["status"] == "published"
    assert "input_mismatch" in world.remote_file(f"{unit_path(D1)}/{PAGE}")
    assert SVG not in remote_names(world, D1)


def test_symlinked_macro_is_not_followed(world: World, tmp_path: Path) -> None:
    world.make_run(D1)
    target = tmp_path / "elsewhere.json"
    target.write_bytes(mk.dump(mk.doc(D1, f"inv-{D1}")))
    (world.runs / D1 / f"macro-{D1}.json").symlink_to(target)
    assert pr.run_step(world.config(D1))["status"] == "published"
    assert "format_error" in world.remote_file(f"{unit_path(D1)}/{PAGE}")


def test_bad_svg_from_the_renderer_does_not_block_the_unit(world: World, monkeypatch) -> None:
    import sys

    render_module = sys.modules[mk.RENDER_MODULE]
    good = render_module.render

    def bad(body):
        if body.get("state") == "failed":
            return good(body)
        return "![x](assets/macro-big.svg)\n", {
            "macro-big.svg": mk.SVG.replace("</svg>", "<script>1</script></svg>")
        }

    monkeypatch.setattr(render_module, "render", bad)
    world.make_run(D1)
    write_macro(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    assert "render_rejected" in world.remote_file(f"{unit_path(D1)}/{PAGE}")
    assert not any(name.startswith("assets/") for name in remote_names(world, D1))


def test_local_unit_with_a_tampered_svg_is_rejected_at_sync(world: World) -> None:
    world.make_run(D1)
    write_macro(world, D1)
    world.go_offline()
    assert pr.run_step(world.config(D1))["status"] == "sync_pending"
    svg = world.local_unit(D1) / SVG
    svg.write_bytes(svg.read_bytes().replace(b"</svg>", b"<script>1</script></svg>"))
    world.come_online()
    assert pr.sync_step(world.config(D1))["status"] == "rejected"
    assert world.remote_log() == ["뼈대"]


def test_correction_rerenders_with_the_same_macro_and_keeps_the_svg(world: World) -> None:
    world.make_run(D1)
    write_macro(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    before = world.remote_file(f"{unit_path(D1)}/{SVG}")
    world.make_run(D1, score=0.123456)  # report만 바뀝니다. 거시 흐름 입력은 그대로입니다
    assert pr.run_step(world.config(D1))["status"] == "correction_required"
    fixed = pr.sync_step(world.config(D1), correct=D1, reason="KR 점수 재계산")
    assert fixed["status"] == "published"
    assert world.remote_file(f"{unit_path(D1)}/{SVG}") == before
    meta, _ = md.parse_front_matter(world.remote_file(f"{unit_path(D1)}/{PAGE}"))
    assert meta["revision"] == 2
    assert pr.run_step(world.config(D1))["status"] == "unchanged"
    assert md.validate(world.clone_for_checks()) == []


def test_correction_without_macro_removes_the_page_and_svg(world: World) -> None:
    world.make_run(D1)
    write_macro(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    off = world.config(D1, macro_context_enabled=False)
    assert pr.run_step(off)["status"] == "correction_required"
    assert pr.sync_step(off, correct=D1, reason="거시 흐름 끔")["status"] == "published"
    assert remote_names(world, D1) == sorted(FIVE)
    assert "macro.md" not in world.remote_file(f"{unit_path(D1)}/README.md")
    assert md.validate(world.clone_for_checks()) == []


def test_macro_changed_after_the_local_step_blocks_the_correction(world: World) -> None:
    world.make_run(D1)
    write_macro(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    world.make_run(D1, score=0.123456)
    path = write_macro(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "correction_required"
    path.write_bytes(path.read_bytes().replace(b'"state":"ok"', b'"state":"failed"', 1))
    blocked = pr.sync_step(world.config(D1), correct=D1, reason="KR 점수 재계산")
    assert blocked["status"] == "failed" and "macro input changed" in blocked["detail"]
