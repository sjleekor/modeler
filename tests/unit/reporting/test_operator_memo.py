"""운영자 메모: 실패·stale 섹션에 사람이 남긴 설명을 붙이고, 게시 규칙을 입력에서 막는다.

메모가 없으면 출력은 기능이 없던 때와 바이트까지 같고, 이미 올린 단위에 메모를 더하는 일은
`sync --correct`로만 올라간다.
"""
from __future__ import annotations

import json
from pathlib import Path

import md_fixtures as mf
import pytest
from reports_world import World, pr

from modeler.reporting import markdown as md

NO_REPO = Path("/nonexistent-stock-reports")
DAY = "2026-10-06"
STAMP = "2026-10-06T10:03:00+09:00"
FIVE = tuple(md.SECTION_FILES.values())


def memo_bytes(memos: dict, day: str = DAY, **extra) -> bytes:
    body = {"schema": md.OPERATOR_MEMO_SCHEMA, "report_date": day, "memos": memos, **extra}
    return json.dumps(body, ensure_ascii=False).encode()


def render(env: dict, ms: dict | None, memo: bytes | None = None, log=lambda m: None) -> dict:
    ms_bytes = None if ms is None else json.dumps(ms).encode()
    ctx = md.build_context(NO_REPO, json.dumps(env).encode(), ms_bytes, "r1", STAMP, 100, log,
                           operator_memo=memo)
    return md.render_unit(ctx)


def us_unavailable_day() -> tuple[dict, dict]:
    reports = [mf.kr_report(DAY, "2026-10-02")]
    failures = [{"market": "US", "model_id": mf.LGB_ID, "error_class": "MissingInference"},
                {"market": "US", "model_id": mf.RDG_ID, "error_class": "MissingInference"}]
    summary = {"US": {"status": "unavailable", "reason": "producer_completion_missing"}}
    env = mf.envelope(DAY, reports, failures, extra={"selection_summary": summary})
    ms = mf.ms_input(DAY, kr_asof="2026-10-02", us_asof="2026-10-02", macro_asof="2026-10-01")
    return env, ms


def test_failed_us_section_shows_the_memo_beside_cause_class_and_select_reason():
    env, ms = us_unavailable_day()
    text = "US universe revision이 맞지 않아 prepare가 입력을 만들지 못했습니다."
    files = render(env, ms, memo_bytes({"US": text}))
    us = files["us-stocks.md"]
    assert "원인 클래스: `MissingInference`" in us
    assert "select 사유:" in us
    line = "운영자 메모: " + text
    assert us.count(line) == 2  # US 모델 둘
    assert line in files["data-status.md"] and line in files["README.md"]
    assert "운영자 메모" not in files["kr-stocks.md"]
    assert "운영자 메모" not in files["market-sector.md"]


def test_without_a_memo_output_is_byte_identical():
    env, ms = us_unavailable_day()
    plain = render(env, ms)
    assert render(env, ms, None) == plain
    assert render(env, ms, memo_bytes({})) == plain
    assert all("운영자 메모" not in t for t in plain.values())
    env2, ms2 = mf.scenario_normal()
    assert render(env2, ms2) == render(env2, ms2, memo_bytes({}, env2["report_date"]))


def test_memo_on_a_healthy_section_is_not_shown_and_is_logged():
    env, ms = mf.scenario_normal()
    seen: list[str] = []
    memo = memo_bytes({"KR": "정상인데 적은 메모", "US": "x", "MS": "y"}, env["report_date"])
    files = render(env, ms, memo, seen.append)
    assert all("운영자 메모" not in t for t in files.values())
    assert len(seen) == 3 and all("쓰지 않았습니다" in m for m in seen)
    assert files == render(env, ms)


def test_stale_section_gets_the_memo_after_the_select_line():
    day = "2026-10-08"
    reports = [
        mf.kr_report(day, "2026-10-02", lag=3, status="stale", fresh_status="stale"),
        mf.us_report(day, mf.LGB_ID, "2026-10-02", "2026-10-07", "2026-10-07", lag=3,
                     status="stale"),
        mf.us_report(day, mf.RDG_ID, "2026-10-02", "2026-10-07", "2026-10-07", lag=3,
                     status="stale"),
    ]
    summary = {"US": {"status": "stale", "feature_asof_date": "2026-10-02", "lag_sessions": 3}}
    env = mf.envelope(day, reports, [], extra={"selection_summary": summary})
    ms = mf.ms_input(day, kr_asof="2026-10-06", us_asof="2026-10-07", macro_asof="2026-10-04")
    memo = memo_bytes({"US": "도착 지연 확인 중", "KR": "K 재계산 대기"}, day)
    files = render(env, ms, memo)
    assert "> select 사유: 상태 `stale`(자료 지연), 기준일 2026-10-02, 지연 3세션. " \
           "운영자 메모: 도착 지연 확인 중" in files["us-stocks.md"]
    assert "> 운영자 메모: K 재계산 대기" in files["kr-stocks.md"]
    assert "운영자 메모: 도착 지연 확인 중" in files["data-status.md"]
    assert "운영자 메모: K 재계산 대기" in files["README.md"]


def test_market_sector_failed_section_gets_the_memo():
    env, _ = us_unavailable_day()
    files = render(env, None, memo_bytes({"MS": "시장·섹터 입력은 오후에 다시 만듭니다."}))
    assert "운영자 메모: 시장·섹터 입력은 오후에 다시 만듭니다." in files["market-sector.md"]
    assert "운영자 메모: 시장·섹터 입력은 오후에 다시 만듭니다." in files["data-status.md"]


def test_memo_text_is_escaped_not_trusted():
    env, ms = us_unavailable_day()
    us = render(env, ms, memo_bytes({"US": "a|b [x](y) <b>z</b>"}))["us-stocks.md"]
    assert "a\\|b \\[x\\](y) \\<b\\>z\\</b\\>" in us and "<b>" not in us


@pytest.mark.parametrize(
    "text",
    [
        "서버 /home/whi/data 에서 실패",
        "/Users/whishaw/x",
        "sj2-server 점검 중",
        "ghp_abcdef123456",
        "token: abc",
        "password=hunter2",
        "x" * (md.OPERATOR_MEMO_MAX_CHARS + 1),
        "두 줄\n메모",
        "   ",
    ],
)
def test_rule_breaking_memos_are_refused(text):
    env, ms = us_unavailable_day()
    with pytest.raises(md.InputError):
        render(env, ms, memo_bytes({"US": text}))


def test_the_longest_allowed_memo_passes():
    env, ms = us_unavailable_day()
    text = "가" * md.OPERATOR_MEMO_MAX_CHARS
    assert "운영자 메모: " + text in render(env, ms, memo_bytes({"US": text}))["us-stocks.md"]


@pytest.mark.parametrize(
    "raw",
    [
        b"not json",
        b"[]",
        json.dumps({"schema": "x", "report_date": DAY, "memos": {}}).encode(),
        memo_bytes({}, "2026-10-05"),
        memo_bytes({"EU": "x"}),
        memo_bytes({"US": 5}),
        memo_bytes({}, unknown=1),
        json.dumps({"schema": md.OPERATOR_MEMO_SCHEMA, "report_date": DAY, "memos": []}).encode(),
    ],
)
def test_malformed_memo_files_are_refused(raw):
    env, ms = us_unavailable_day()
    with pytest.raises(md.InputError):
        render(env, ms, raw)


# --- publisher: 로컬 bare remote ------------------------------------------------------------
D1 = "2026-10-07"


def write_memo(world: World, day: str, memos: dict) -> Path:
    path = world.runs / day / f"operator-memo-{day}.json"
    path.write_bytes(memo_bytes(memos, day))
    return path


def make_failed_run(world: World, day: str) -> None:
    """KR만 순위가 있고 US 둘이 MissingInference인 날."""
    world.make_run(day)
    run_dir = world.runs / day
    env = mf.envelope(day, [mf.kr_report(day, "2026-10-06")],
                      [{"market": "US", "model_id": mf.LGB_ID, "error_class": "MissingInference"},
                       {"market": "US", "model_id": mf.RDG_ID, "error_class": "MissingInference"}])
    (run_dir / f"report-{day}.json").write_text(json.dumps(env, sort_keys=True) + "\n",
                                                 encoding="utf-8")


def test_publisher_without_a_memo_file_leaves_manifest_and_journal_as_before(world_factory):
    world = world_factory
    make_failed_run(world, D1)
    pr.local_step(world.config(D1))
    manifest = json.loads((world.runs / D1 / "markdown" / "manifest.json").read_text())
    assert not any(key.startswith("operator_memo") for key in manifest)
    assert not any(key.startswith("operator_memo") for key in world.journal()["units"][D1])


def test_memo_file_reaches_the_published_unit(world_factory):
    world = world_factory
    make_failed_run(world, D1)
    write_memo(world, D1, {"US": "prepare 입력이 늦어 오전에 만들지 못했습니다."})
    result = pr.run_step(world.config(D1))
    assert result["status"] == "published"
    published = world.remote_file(f"reports/daily-briefing/2026/10/{D1}/us-stocks.md")
    assert "운영자 메모: prepare 입력이 늦어 오전에 만들지 못했습니다." in published
    manifest = json.loads((world.runs / D1 / "markdown" / "manifest.json").read_text())
    assert manifest["operator_memo_sha256"]


def test_a_memo_added_after_publication_needs_correct(world_factory):
    world = world_factory
    make_failed_run(world, D1)
    assert pr.run_step(world.config(D1))["status"] == "published"
    head = world.remote_head()
    write_memo(world, D1, {"US": "universe revision이 맞지 않았습니다."})
    blocked = pr.run_step(world.config(D1))
    assert blocked["status"] == "correction_required" and world.remote_head() == head
    fixed = pr.sync_step(world.config(D1), correct=D1, reason="운영자 메모 추가")
    assert fixed["status"] == "published"
    assert world.remote_log()[0] == f"daily-briefing {D1} r2: 운영자 메모 추가"
    assert "운영자 메모: universe revision이 맞지 않았습니다." in world.remote_file(
        f"reports/daily-briefing/2026/10/{D1}/us-stocks.md")
    assert md.validate(world.clone_for_checks()) == []


def test_a_rule_breaking_memo_rejects_the_unit_and_publishes_nothing(world_factory):
    world = world_factory
    make_failed_run(world, D1)
    write_memo(world, D1, {"US": "서버 /home/whi/run 에서 실패"})
    head = world.remote_head()
    result = pr.run_step(world.config(D1))
    assert result["status"] == "rejected" and world.remote_head() == head
    assert "운영자 메모" in json.dumps(result["detail"], ensure_ascii=False)


def test_a_memo_changed_between_the_steps_does_not_change_the_stored_unit(world_factory):
    world = world_factory
    make_failed_run(world, D1)
    path = write_memo(world, D1, {"US": "처음 메모"})
    assert pr.local_step(world.config(D1))["status"] == "local_ready"
    path.write_bytes(memo_bytes({"US": "바뀐 메모"}, D1))
    # 올라가는 것은 로컬 단계가 해시로 고정한 단위입니다. 바뀐 메모는 다음 local 단계부터입니다.
    assert pr.sync_step(world.config(D1))["status"] == "published"
    published = world.remote_file(f"reports/daily-briefing/2026/10/{D1}/us-stocks.md")
    assert "처음 메모" in published and "바뀐 메모" not in published


@pytest.fixture()
def world_factory(tmp_path: Path) -> World:
    return World(tmp_path)
