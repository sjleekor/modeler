"""deploy/reports/publish_reports.py: 로컬 bare remote로 두 단계 publisher를 시험합니다."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from reports_world import (
    CARDS,
    World,
    git,
    no_log,
    pr,
    vr,
)

from modeler.reporting import markdown as md

D1, D2, D3, D4, D5 = "2026-10-07", "2026-10-08", "2026-10-12", "2026-10-13", "2026-10-14"
FIVE = tuple(md.SECTION_FILES.values())


def unit_path(day: str) -> str:
    return f"reports/daily-briefing/{day[:4]}/{day[5:7]}/{day}"


@pytest.fixture()
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def status_of(world: World) -> str:
    return git("-C", str(world.checkout), "status", "--porcelain")


def head_of(world: World) -> str:
    return git("-C", str(world.checkout), "rev-parse", "HEAD")


# --- 정상 push -------------------------------------------------------------------------
def test_normal_publish_pushes_one_commit_with_the_unit_and_indexes(world: World) -> None:
    world.make_run(D1)
    base = world.remote_log()
    result = pr.run_step(world.config(D1))
    assert result["status"] == "published" and result["local_done"] is True
    assert pr.EXIT_CODES[result["status"]] == 0
    assert result["commit"] == world.remote_head()
    assert world.remote_log() == [f"daily-briefing {D1}", *base]
    for name in FIVE:
        assert world.remote_has(f"{unit_path(D1)}/{name}")
    # 인덱스(자동 구간)가 같은 커밋에 들어갑니다.
    assert D1 in world.remote_file("README.md")
    assert D1 in world.remote_file("reports/daily-briefing/README.md")
    assert D1 in world.remote_file("reports/daily-briefing/2026/10/README.md")
    # journal이 지워지고 checkout은 깨끗하며 원격과 같습니다. 작성자는 checkout git 설정입니다.
    assert world.journal() is None
    assert status_of(world) == "" and head_of(world) == world.remote_head()
    assert git("--git-dir", str(world.bare), "log", "-1", "--format=%an") == "stock-reports-bot"
    assert md.validate(world.clone_for_checks()) == []
    # 사람이 쓴 부분은 그대로입니다.
    seed_root = world.remote_file("README.md").split(md.AUTO_BEGIN)[0]
    assert "본인 전용" in seed_root
    # 로컬 단계 산출물은 실행 디렉터리에 남습니다.
    assert sorted(p.name for p in world.local_unit(D1).iterdir()) == sorted(FIVE)
    assert (world.runs / D1 / "markdown" / "manifest.json").is_file()


def test_same_day_rerun_makes_no_new_commit_even_with_another_timestamp(world: World) -> None:
    world.make_run(D1)
    pr.run_step(world.config(D1))
    head = world.remote_head()
    count = len(world.remote_log())
    first_readme = (world.local_unit(D1) / "README.md").read_text(encoding="utf-8")
    again = pr.run_step(world.config(D1, generated_at=f"{D1}T10:40:00+09:00"))
    assert again["status"] == "unchanged" and again["commit"] == head
    assert world.remote_head() == head and len(world.remote_log()) == count
    assert world.journal() is None and status_of(world) == ""
    # 내용이 같으면 로컬 단위의 generated_at도 바꾸지 않습니다(해시가 흔들리지 않게).
    assert (world.local_unit(D1) / "README.md").read_text(encoding="utf-8") == first_readme


# --- 정정 ------------------------------------------------------------------------------
def test_different_content_needs_an_explicit_correction_then_raises_revision(world: World) -> None:
    world.make_run(D1)
    pr.run_step(world.config(D1))
    head = world.remote_head()
    world.make_run(D1, score=0.123456)
    blocked = pr.run_step(world.config(D1))
    assert blocked["status"] == "correction_required"
    assert pr.EXIT_CODES[blocked["status"]] == 30
    assert world.remote_head() == head and status_of(world) == ""
    assert world.journal()["units"][D1]["state"] == "correction_required"
    fixed = pr.sync_step(world.config(D1), correct=D1, reason="KR 점수 재계산")
    assert fixed["status"] == "published"
    assert world.remote_log()[0] == f"daily-briefing {D1} r2: KR 점수 재계산"
    for name in FIVE:
        meta, _ = md.parse_front_matter(world.remote_file(f"{unit_path(D1)}/{name}"))
        assert meta["revision"] == 2
    summary = world.remote_file(f"{unit_path(D1)}/README.md")
    line = re.search(
        r"> 정정 r2 \(2026-10-07T10:03:12\+09:00\): KR 점수 재계산\. 이전 판: ([0-9a-f]+)", summary
    )
    assert line and head.startswith(line.group(1))
    assert summary.index("정정 r2") < summary.index("## 섹션 상태")
    assert "0.1235" in world.remote_file(f"{unit_path(D1)}/kr-stocks.md")
    # 같은 입력으로 또 돌리면 이제는 바뀌는 것이 없습니다.
    final = pr.run_step(world.config(D1))
    assert final["status"] == "unchanged" and world.journal() is None


def test_correction_arguments_are_checked(world: World) -> None:
    world.make_run(D1)
    pr.local_step(world.config(D1))
    assert pr.sync_step(world.config(D1), correct=D1)["status"] == "failed"
    assert pr.sync_step(world.config(D1), reason="사유만")["status"] == "failed"
    # 대기 중이 아닌 단위는 고칠 수 없습니다.
    assert pr.sync_step(world.config(D1), correct=D2, reason="없는 단위")["status"] == "failed"
    # 저장소에 없는 단위를 --correct로 올릴 수는 없습니다.
    missing = pr.sync_step(world.config(D1), correct=D1, reason="처음 올리는 단위")
    assert missing["status"] == "rejected" and not world.remote_has(f"{unit_path(D1)}/README.md")
    assert world.journal()["units"][D1]["state"] == "rejected"


# --- non-fast-forward ------------------------------------------------------------------
def test_non_fast_forward_is_retried_on_top_of_the_new_remote_head(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.make_run(D1)
    real_fetch, calls = pr.fetch, {"n": 0}

    def fetch_then_race(checkout: Path, remote: str, branch: str) -> None:
        real_fetch(checkout, remote, branch)
        calls["n"] += 1
        if calls["n"] == 1:
            world.user_push("reference/a.md")

    monkeypatch.setattr(pr, "fetch", fetch_then_race)
    result = pr.run_step(world.config(D1))
    assert result["status"] == "published" and calls["n"] == 2
    assert world.remote_log()[:2] == [f"daily-briefing {D1}", "사용자 수정 reference/a.md"]
    assert world.remote_has("reference/a.md") and world.remote_has(f"{unit_path(D1)}/README.md")
    assert git("--git-dir", str(world.bare), "rev-list", "--merges", "--count", "main") == "0"
    assert world.journal() is None and status_of(world) == ""


def test_non_fast_forward_gives_up_after_three_rounds_and_keeps_the_commit(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.make_run(D1)
    real_fetch, calls = pr.fetch, {"n": 0}

    def always_race(checkout: Path, remote: str, branch: str) -> None:
        real_fetch(checkout, remote, branch)
        calls["n"] += 1
        world.user_push(f"reference/race{calls['n']}.md")

    monkeypatch.setattr(pr, "fetch", always_race)
    result = pr.run_step(world.config(D1))
    assert result["status"] == "push_pending" and calls["n"] == pr.MAX_PUSH_ATTEMPTS
    assert not world.remote_has(f"{unit_path(D1)}/README.md")
    assert world.journal()["units"][D1]["state"] == "push_pending"
    # 경합이 끝나면 다음 재시도가 원격 최신 위에 다시 얹어 push합니다.
    monkeypatch.setattr(pr, "fetch", real_fetch)
    done = pr.sync_step(world.config(D1))
    assert done["status"] == "published" and world.journal() is None
    assert world.remote_has("reference/race3.md") and world.remote_has(f"{unit_path(D1)}/README.md")


# --- 원격에 접속할 수 없을 때 ----------------------------------------------------------------
def test_local_step_never_touches_git_or_the_network(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    world.make_run(D1)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("local step must not run git")

    monkeypatch.setattr(pr, "git", forbidden)
    result = pr.local_step(world.config(D1))
    assert result["status"] == "local_ready" and result["state"] == "sync_pending"
    entry = world.journal()["units"][D1]
    assert entry["state"] == "sync_pending" and entry["unit_sha256"] == result["unit_sha256"]


def test_unreachable_remote_leaves_the_unit_and_journal_then_sync_publishes(world: World) -> None:
    world.make_run(D1)
    world.go_offline()
    result = pr.run_step(world.config(D1))
    assert result["status"] == "sync_pending" and pr.EXIT_CODES["sync_pending"] == 20
    assert result["local_done"] is True and result["units"] == {D1: "sync_pending"}
    assert sorted(p.name for p in world.local_unit(D1).iterdir()) == sorted(FIVE)
    entry = world.journal()["units"][D1]
    assert entry["state"] == "sync_pending" and len(entry["unit_sha256"]) == 64
    assert status_of(world) == ""
    # 10:15 재시도는 동기화 단계만 다시 돕니다. 아직 안 닿으면 그대로입니다.
    assert pr.sync_step(world.config(D1))["status"] == "sync_pending"
    assert world.journal()["units"][D1]["unit_sha256"] == entry["unit_sha256"]
    world.come_online()
    done = pr.sync_step(world.config(D1))
    assert done["status"] == "published" and world.journal() is None
    assert world.remote_log()[0] == f"daily-briefing {D1}"


def test_push_failure_after_commit_then_same_content_retry_pushes_the_same_commit(
    world: World,
) -> None:
    world.make_run(D1)
    flag = world.block_pushes()
    base = world.remote_head()
    result = pr.run_step(world.config(D1))
    assert result["status"] == "push_pending" and pr.EXIT_CODES["push_pending"] == 21
    entry = world.journal()["units"][D1]
    assert entry["state"] == "push_pending" and entry["commit"] == head_of(world)
    assert world.remote_head() == base
    local_commit = entry["commit"]
    flag.unlink()
    # 같은 내용으로 다시 돌립니다. "내용이 같다"는 이유로 끝내지 않고 밀린 push를 끝냅니다.
    again = pr.run_step(world.config(D1))
    assert again["status"] == "published"
    assert world.remote_head() == local_commit
    assert world.remote_log() == [f"daily-briefing {D1}", "뼈대"]
    assert world.journal() is None


def test_push_failure_then_sync_only_retry(world: World) -> None:
    world.make_run(D1)
    flag = world.block_pushes()
    assert pr.run_step(world.config(D1))["status"] == "push_pending"
    assert pr.sync_step(world.config(D1))["status"] == "push_pending"
    flag.unlink()
    assert pr.sync_step(world.config(D1))["status"] == "published"


def test_unpushed_commit_is_replaced_when_the_same_day_is_rendered_differently(
    world: World,
) -> None:
    world.make_run(D1)
    flag = world.block_pushes()
    assert pr.run_step(world.config(D1))["status"] == "push_pending"
    old_commit = world.journal()["units"][D1]["commit"]
    flag.unlink()
    world.make_run(D1, score=0.123456)
    done = pr.run_step(world.config(D1))
    # 아직 원격에 나간 적 없으므로 r2가 아니라 새 내용의 첫 판입니다.
    assert done["status"] == "published"
    assert world.remote_log() == [f"daily-briefing {D1}", "뼈대"]
    assert world.remote_head() != old_commit
    assert "0.1235" in world.remote_file(f"{unit_path(D1)}/kr-stocks.md")
    meta, _ = md.parse_front_matter(world.remote_file(f"{unit_path(D1)}/README.md"))
    assert meta["revision"] == 1


# --- 여러 날 -------------------------------------------------------------------------------
def test_multiday_publishes_one_commit_per_day_and_updates_indexes(world: World) -> None:
    world.make_run(D1)
    world.make_run(D2, prev=D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    assert pr.run_step(world.config(D2))["status"] == "published"
    assert world.remote_log()[:2] == [f"daily-briefing {D2}", f"daily-briefing {D1}"]
    family = world.remote_file("reports/daily-briefing/README.md")
    assert family.index(D2) < family.index(D1)
    month = world.remote_file("reports/daily-briefing/2026/10/README.md")
    assert D1 in month and D2 in month
    assert f"[{D2}](reports/daily-briefing/2026/10/{D2}/README.md)" in world.remote_file(
        "README.md"
    )
    # 이전 날 파일은 두 번째 날 커밋에서 바뀌지 않습니다.
    changed = git(
        "--git-dir", str(world.bare), "diff", "--name-only", "HEAD~1", "HEAD"
    ).splitlines()
    assert not [p for p in changed if f"/{D1}/" in p]
    assert md.validate(world.clone_for_checks()) == []


def test_backlog_is_pushed_oldest_first_before_today(world: World) -> None:
    world.make_run(D1)
    world.make_run(D2, prev=D1)
    world.make_run(D3, prev=D2)
    world.go_offline()
    assert pr.run_step(world.config(D1))["status"] == "sync_pending"
    assert pr.run_step(world.config(D2))["status"] == "sync_pending"
    assert set(world.journal()["units"]) == {D1, D2}
    world.come_online()
    result = pr.run_step(world.config(D3))
    assert result["status"] == "published"
    assert result["units"] == {D1: "published", D2: "published", D3: "published"}
    assert world.remote_log()[:3] == [
        f"daily-briefing {D3}",
        f"daily-briefing {D2}",
        f"daily-briefing {D1}",
    ]
    assert world.journal() is None and status_of(world) == ""
    assert md.validate(world.clone_for_checks()) == []


def test_new_unit_older_than_published_ones_is_allowed_but_never_changes_them(world: World) -> None:
    world.make_run(D2, prev=D1)
    world.make_run(D1)
    assert pr.run_step(world.config(D2))["status"] == "published"
    before = world.remote_file(f"{unit_path(D2)}/README.md")
    assert pr.run_step(world.config(D1))["status"] == "published"
    assert world.remote_file(f"{unit_path(D2)}/README.md") == before


# --- 과거 단위 보호 ------------------------------------------------------------------------
def test_validator_rejects_changing_or_deleting_a_published_unit(world: World) -> None:
    world.make_run(D1)
    pr.run_step(world.config(D1))
    target = world.checkout / unit_path(D1) / "kr-stocks.md"
    target.write_text(target.read_text(encoding="utf-8") + "\n덧붙인 줄\n", encoding="utf-8")
    problems = vr.validate_working_tree(world.checkout, new_units=[D2])
    assert any("이미 올린 단위(2026-10-07)" in p for p in problems)
    assert not vr.validate_working_tree(world.checkout, new_units=[D2], corrected_units=[D1])
    git("-C", str(world.checkout), "checkout", "--", ".")
    (world.checkout / unit_path(D1) / "us-stocks.md").unlink()
    problems = vr.validate_working_tree(world.checkout, new_units=[D2], corrected_units=[D1])
    assert any("삭제는 허용하지 않습니다" in p for p in problems)
    git("-C", str(world.checkout), "checkout", "--", ".")
    assert status_of(world) == ""


def test_validator_rejects_paths_outside_the_allowlist_and_human_text_edits(world: World) -> None:
    world.make_run(D1)
    pr.run_step(world.config(D1))
    (world.checkout / "reports" / "other-family").mkdir()
    (world.checkout / "reports" / "other-family" / "README.md").write_text("x\n", encoding="utf-8")
    (world.checkout / "CONVENTIONS.md").write_text("바꾼 규칙\n", encoding="utf-8")
    (world.checkout / "reference" / "glossary.md").write_text("바꾼 용어\n", encoding="utf-8")
    root = world.checkout / "README.md"
    root.write_text(
        "사람이 쓴 글을 지웁니다\n" + root.read_text(encoding="utf-8").split("\n", 2)[2],
        encoding="utf-8",
    )
    problems = vr.validate_working_tree(world.checkout, new_units=[D2])
    text = "\n".join(problems)
    assert "reports/other-family/README.md: 허용 경로 밖" in text
    assert "CONVENTIONS.md: 허용 경로 밖" in text
    assert "reference/glossary.md: 허용 경로 밖" in text
    assert "README.md: 자동 구간 밖을 바꿨습니다" in text


def test_unjournaled_local_commit_is_refused_and_nothing_is_pushed(world: World) -> None:
    world.make_run(D1)
    pr.local_step(world.config(D1))
    (world.checkout / "reference" / "stray.md").write_text("x\n", encoding="utf-8")
    git("-C", str(world.checkout), "add", "reference/stray.md")
    git("-C", str(world.checkout), "commit", "-q", "-m", "journal에 없는 커밋")
    base = world.remote_head()
    result = pr.sync_step(world.config(D1))
    assert result["status"] == "failed" and "journal does not know" in result["detail"]
    assert world.remote_head() == base and world.journal()["units"][D1]["state"] == "sync_pending"


# --- 서버 경로·비밀값, 합성 표시 -----------------------------------------------------------------
def test_server_path_or_host_name_is_rejected_in_the_local_step(world: World) -> None:
    world.make_run(D1)
    base = world.remote_head()
    result = pr.run_step(world.config(D1, release="sj2-release"))
    assert result["status"] == "rejected" and pr.EXIT_CODES["rejected"] == 10
    assert any("금지 문자열" in line for line in result["detail"])
    assert world.remote_head() == base
    entry = world.journal()["units"][D1]
    assert entry["state"] == "rejected" and "금지 문자열" in "".join(entry["detail"])
    assert (world.runs / D1 / "markdown" / "rejected.json").is_file()
    # 거절된 단위는 다시 돌려도 올라가지 않습니다.
    assert pr.sync_step(world.config(D1))["status"] == "rejected"
    assert not world.remote_has(f"{unit_path(D1)}/README.md")


def test_data_strings_with_server_paths_are_scrubbed_not_published(world: World) -> None:
    run_dir = world.make_run(D1)
    report = json.loads((run_dir / f"report-{D1}.json").read_text(encoding="utf-8"))
    report["markets"][0]["rankings"][0]["name"] = "누수/home/whi/secret/key.pem"
    report["markets"][0]["rankings"][1]["name"] = "host sj2-server ghp_abcdef api_key=zzz"
    (run_dir / f"report-{D1}.json").write_text(json.dumps(report), encoding="utf-8")
    assert pr.run_step(world.config(D1))["status"] == "published"
    text = world.remote_file(f"{unit_path(D1)}/kr-stocks.md")
    for needle in ("/home/", "sj2", "ghp_", "api_key"):
        assert needle not in text


def test_unit_changed_after_the_local_step_is_rejected_at_sync(world: World) -> None:
    world.make_run(D1)
    pr.local_step(world.config(D1))
    target = world.local_unit(D1) / "kr-stocks.md"
    target.write_text(target.read_text(encoding="utf-8") + "\n/home/whi/secret\n", encoding="utf-8")
    base = world.remote_head()
    result = pr.sync_step(world.config(D1))
    assert result["status"] == "rejected" and world.remote_head() == base
    assert status_of(world) == ""


def test_tree_validation_failure_rolls_the_working_tree_back(world: World, tmp_path: Path) -> None:
    world.make_run(D1)
    # 사용자가 카드를 지웠고 publisher는 카드 원본을 갖고 있지 않습니다 -> 링크가 깨집니다.
    world.user_push("reference/notes.md")
    git("-C", str(world.other), "rm", "-q", "reference/models/kr_daily_h20_v1.md")
    git("-C", str(world.other), "commit", "-q", "-m", "카드 삭제")
    git("-C", str(world.other), "push", "-q", "origin", "main:main")
    empty = tmp_path / "empty-cards.json"
    empty.write_text("{}", encoding="utf-8")
    base = world.remote_head()
    result = pr.run_step(world.config(D1, model_cards_path=str(empty)))
    assert result["status"] == "rejected"
    assert any("링크 대상이 없습니다" in line for line in world.journal()["units"][D1]["detail"])
    assert world.remote_head() == base
    assert status_of(world) == "" and head_of(world) == base
    assert not (world.checkout / unit_path(D1)).exists()


def test_synthetic_report_needs_the_explicit_flag(world: World) -> None:
    world.make_run(D1, synthetic=True)
    refused = pr.local_step(world.config(D1))
    assert refused["status"] == "rejected"
    assert any("--allow-synthetic" in line for line in refused["detail"])
    ok = pr.run_step(world.config(D1), allow_synthetic=True)
    assert ok["status"] == "published"
    meta, body = md.parse_front_matter(world.remote_file(f"{unit_path(D1)}/README.md"))
    assert meta["synthetic_fixture"] is True and md.SYNTHETIC_NOTICE in body


def test_report_must_be_the_one_of_this_invocation(world: World) -> None:
    world.make_run(D1)
    bad_sha = pr.local_step(world.config(D1, report_sha256="0" * 64))
    assert bad_sha["status"] == "rejected" and "sha256" in bad_sha["detail"][0]
    bad_invocation = pr.local_step(world.config(D1, invocation_id="another-invocation"))
    assert bad_invocation["status"] == "rejected" and "invocation id" in bad_invocation["detail"][0]
    run_dir = world.runs / D1
    report = json.loads((run_dir / f"report-{D1}.json").read_text(encoding="utf-8"))
    report["invocation_id"] = "inv-from-an-older-run"
    (run_dir / f"report-{D1}.json").write_text(json.dumps(report), encoding="utf-8")
    stale = pr.local_step(world.config(D1))
    assert stale["status"] == "rejected" and "invocation id" in stale["detail"][0]
    assert not world.remote_has(f"{unit_path(D1)}/README.md")


# --- checkout 확인, 잠금, 설정 --------------------------------------------------------------------
def test_checkout_must_be_clean_main_with_the_exact_remote_url(world: World) -> None:
    world.make_run(D1)
    pr.local_step(world.config(D1))
    base = world.remote_head()
    stray = world.checkout / "stray.txt"
    stray.write_text("x", encoding="utf-8")
    dirty = pr.sync_step(world.config(D1))
    assert dirty["status"] == "failed" and "clean" in dirty["detail"]
    stray.unlink()
    # journal은 다른 원격을 가리키는 설정으로는 읽지 않습니다.
    other_target = pr.sync_step(world.config(D1, expected_remote_url=str(world.bare) + "/"))
    assert other_target["status"] == "failed" and "different remote" in other_target["detail"]
    # checkout의 push URL이 설정과 정확히 같지 않으면 멈춥니다.
    git("-C", str(world.checkout), "remote", "set-url", "--push", "origin", str(world.bare) + "/")
    wrong_url = pr.sync_step(world.config(D1))
    assert wrong_url["status"] == "failed" and "remote URL" in wrong_url["detail"]
    git("-C", str(world.checkout), "remote", "set-url", "--push", "origin", str(world.bare))
    git("-C", str(world.checkout), "checkout", "-q", "-b", "other")
    wrong_branch = pr.sync_step(world.config(D1))
    assert wrong_branch["status"] == "failed" and "main branch" in wrong_branch["detail"]
    git("-C", str(world.checkout), "checkout", "-q", "main")
    git("-C", str(world.checkout), "config", "--unset", "user.name")
    no_author = pr.sync_step(world.config(D1))
    assert no_author["status"] == "failed" and "user.name" in no_author["detail"]
    assert world.remote_head() == base and world.journal()["units"][D1]["state"] == "sync_pending"


def test_a_second_publisher_run_is_refused_while_the_lock_is_held(world: World) -> None:
    world.make_run(D1)
    pr.local_step(world.config(D1))
    with pr.publisher_lock(world.checkout, wait_seconds=0) as acquired:
        assert acquired is True
        busy = pr.sync_step(world.config(D1))
        assert busy["status"] == "locked" and pr.EXIT_CODES["locked"] == 75
    assert pr.sync_step(world.config(D1))["status"] == "published"


def test_config_is_strict(world: World, tmp_path: Path) -> None:
    world.make_run(D1)
    good = world.config(D1, expected_remote_url=pr.EXPECTED_REMOTE_URL)
    path = tmp_path / "cfg.json"

    def read(config: dict, **kw: object) -> dict:
        path.write_text(json.dumps(config), encoding="utf-8")
        return pr.read_config(path, need_local=True, **kw)

    assert read(good)["expected_remote_url"] == pr.EXPECTED_REMOTE_URL
    for key, value, message in (
        ("expected_remote_url", "https://github.com/sjleekor/stock_reports.git", "exactly"),
        ("expected_remote_url", str(world.bare), "exactly"),
        ("audience", "public", "owner_only"),
        ("branch", "site", "main"),
        ("remote_name", "or igin", "remote name"),
        ("top_n", 0, "top_n"),
        ("report_sha256", "abc", "SHA-256"),
        ("run_dir", "relative/run", "absolute"),
    ):
        with pytest.raises(pr.PublishError, match=message):
            read({**good, key: value})
    with pytest.raises(pr.PublishError, match="unknown config keys"):
        read({**good, "projection_dir": "/x"})
    with pytest.raises(pr.PublishError, match="missing config keys"):
        read({k: v for k, v in good.items() if k != "invocation_id"})
    # 시험 지름길: 로컬 bare 경로는 strict_remote=False에서만 받습니다.
    assert read({**good, "expected_remote_url": str(world.bare)}, strict_remote=False)
    # sync 전용 설정은 로컬 키가 없어도 됩니다.
    sync_only = {k: good[k] for k in pr.REQUIRED_KEYS}
    path.write_text(json.dumps(sync_only), encoding="utf-8")
    assert pr.read_config(path, need_local=False)["branch"] == "main"


def test_cli_modes_print_one_json_line_and_return_the_status_exit_code(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    world.make_run(D1)
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(world.config(D1, expected_remote_url=pr.EXPECTED_REMOTE_URL)))
    assert pr.main(["local", "--config", str(path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "local_ready" and out["unit"] == D1
    assert pr.main(["sync", "--config", str(path), "--correct", D1], strict_remote=True) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"
    assert pr.main(["run", "--config", str(path), "--correct", D1, "--reason", "x"]) == 1
    capsys.readouterr()
    path.write_text(
        json.dumps(
            world.config(D1, release="sj2-release", expected_remote_url=pr.EXPECTED_REMOTE_URL)
        )
    )
    assert pr.main(["local", "--config", str(path)]) == 10
    assert json.loads(capsys.readouterr().out)["status"] == "rejected"
    path.write_text(json.dumps(world.config(D1, audience="public")))
    assert pr.main(["local", "--config", str(path)]) == 1
    capsys.readouterr()


def test_markdown_templates_are_python_strings_only() -> None:
    """release가 .py만 복사하므로 템플릿이 별도 데이터 파일에 있으면 안 됩니다."""
    source = Path(md.__file__).read_text(encoding="utf-8")
    assert "open(" not in source and "importlib.resources" not in source
    assert CARDS.is_file()  # 모델 카드는 release의 model-cards.json에서 읽는 별도 입력입니다.
    assert no_log("x") is None
