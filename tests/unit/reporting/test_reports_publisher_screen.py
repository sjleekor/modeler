"""publisher·렌더러의 거르기(`rank_screen_enabled`, `screen-D.json`) 경로.

끄면(키가 없거나 false) 단위·manifest·journal이 바이트까지 같습니다. 켜면 순위 섹션에 선별 표만 더해지고
파일 집합·report hash·원순위 표·`- 모델:` 줄·피드백 게시 행 집합은 그대로입니다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from reports_world import World, pr
from test_summary_v2 import rank_table_lines

from modeler.feedback.briefing import _exposed_predictions
from modeler.feedback.core import Batch
from modeler.reporting import markdown as md
from modeler.serving import screen as sc

D1 = "2026-10-07"
KEY = "rank_screen_enabled"


@pytest.fixture()
def world(tmp_path: Path) -> World:
    return World(tmp_path)


def _screen_doc(world: World, day: str, *, every: int = 3, break_rank: bool = False) -> bytes:
    run = world.runs / day
    raw = (run / f"report-{day}.json").read_bytes()
    report = json.loads(raw)
    models = {}
    for m in report["markets"]:
        rows = []
        for r in m.get("rankings") or []:
            ok = r["rank"] % every == 0
            reasons = [] if ok else (["kind_preferred", "size_liquidity"] if m["market"] == "KR" else ["liquidity"])
            rows.append({"rank": r["rank"], "symbol": r["symbol"], "pass": ok, "reasons": reasons})
        screened = sc.screen_order(rows, [x["pass"] for x in rows])
        if break_rank and screened:
            screened[0] = {**screened[0], "symbol": "NOPE"}
        models[m["model_id"]] = {
            "market": m["market"], "status": "ok", "reason": None, "asof": m["feature_asof_date"],
            "dates": {"filing_basis_date": "2026-10-03", "risk_market_cap_date": "2026-10-02",
                      "judgement_session": "2026-10-01", "size_market_cap_date": "2026-10-01"},
            "summary": {"total": len(rows), "passed": sum(x["pass"] for x in rows), "top_n_kept": 3,
                        "filings_unchecked": 2},
            "screened": screened, "rows": rows}
    doc = sc.build_document(unit=day, invocation_id=f"inv-{day}", report_sha256=hashlib.sha256(raw).hexdigest(),
                            release="r", models=models)
    body = sc.document_bytes(doc)
    (run / f"screen-{day}.json").write_bytes(body)
    return body


def _files(world: World, day: str) -> dict:
    root = world.local_unit(day)
    return {p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.md"))}


def _exposed(world: World, day: str, files: dict) -> set:
    report = json.loads((world.runs / day / f"report-{day}.json").read_bytes())
    preds = [{"market": m["market"], "model_id": m["model_id"], "rank": r["rank"], "symbol": r["symbol"],
              "prediction_ref": f"{m['model_id']}:{r['rank']}"} for m in report["markets"]
             for r in m.get("rankings") or []]
    batch = Batch(predictions=preds)
    got = _exposed_predictions({k: files[k].encode() for k in ("kr-stocks.md", "us-stocks.md")}, batch)
    assert batch.errors == []
    return got


def test_off_or_absent_is_byte_identical_and_on_adds_only_the_screen_table(world: World) -> None:
    world.make_run(D1)
    _screen_doc(world, D1)  # a document on disk is not read while off
    manifest = world.runs / D1 / "markdown" / "manifest.json"
    plain = pr.local_step(world.config(D1))
    plain_files, plain_manifest = _files(world, D1), manifest.read_bytes()
    off = pr.local_step(world.config(D1, **{KEY: False}))
    assert off["unit_sha256"] == plain["unit_sha256"] and manifest.read_bytes() == plain_manifest
    assert KEY not in world.journal()["units"][D1]
    on = pr.local_step(world.config(D1, **{KEY: True}))
    assert on["status"] == "local_ready" and on["unit_sha256"] != plain["unit_sha256"]
    files = _files(world, D1)
    assert sorted(files) == sorted(plain_files)
    for name in ("README.md", "market-sector.md", "data-status.md"):
        assert files[name] == plain_files[name]  # v1 첫 화면은 그대로입니다
    for name in ("kr-stocks.md", "us-stocks.md"):
        assert rank_table_lines(files[name]) == rank_table_lines(plain_files[name])
        assert [x for x in files[name].splitlines() if x.startswith("- 모델:")] == \
            [x for x in plain_files[name].splitlines() if x.startswith("- 모델:")]
        assert "| 선별 | 원순위 | 코드 |" in files[name]
        assert "선별 상위 30 — 살 만한 종목 안에서의 순위" in files[name]
    assert _exposed(world, D1, files) == _exposed(world, D1, plain_files)
    assert "공시 확인 못 함 2" in files["us-stocks.md"] and "시총 기준일 2026-10-02" in files["kr-stocks.md"]
    body = json.loads(manifest.read_bytes())
    assert body[KEY] is True and body["screen_state"] == "ok"
    assert body["screen_sha256"] == hashlib.sha256((world.runs / D1 / f"screen-{D1}.json").read_bytes()).hexdigest()
    assert body["report_sha256"] == json.loads(plain_manifest)["report_sha256"]
    back = pr.local_step(world.config(D1, **{KEY: False}))
    assert back["unit_sha256"] == plain["unit_sha256"] and manifest.read_bytes() == plain_manifest


def test_on_with_a_missing_or_mismatched_document_says_so_and_keeps_the_tables(world: World) -> None:
    world.make_run(D1)
    plain = pr.local_step(world.config(D1))
    plain_files = _files(world, D1)
    on = pr.local_step(world.config(D1, **{KEY: True}))  # no screen file yet
    assert on["status"] == "local_ready" and on["unit_sha256"] != plain["unit_sha256"]
    files = _files(world, D1)
    assert "선별 목록: 실패 — 입력 형식 오류" in files["kr-stocks.md"]
    _screen_doc(world, D1, break_rank=True)
    pr.local_step(world.config(D1, **{KEY: True}))
    files = _files(world, D1)
    assert "선별 목록: 실패 — 입력 불일치(순위)" in files["kr-stocks.md"]
    for name in ("kr-stocks.md", "us-stocks.md"):
        assert rank_table_lines(files[name]) == rank_table_lines(plain_files[name])


def test_v2_front_page_lists_the_screened_top_10(world: World) -> None:
    world.make_run(D1)
    _screen_doc(world, D1)
    pr.local_step(world.config(D1, summary_v2_enabled=True))
    v2 = _files(world, D1)["README.md"]
    pr.local_step(world.config(D1, summary_v2_enabled=True, **{KEY: True}))
    both = _files(world, D1)["README.md"]
    assert "선별 상위 10: " in both and "선별 상위 10: " not in v2
    assert len(both.splitlines()) == len(v2.splitlines())  # 같은 줄을 바꿀 뿐 늘리지 않습니다


def test_config_value_must_be_a_bool_and_correction_follows_the_journal(world: World) -> None:
    world.make_run(D1)
    for bad in ("yes", 1, None):
        with pytest.raises(pr.PublishError, match=KEY):
            pr.check_config(world.config(D1, **{KEY: bad}), need_local=True, strict_remote=False)
    _screen_doc(world, D1)
    world.go_offline()
    assert pr.run_step(world.config(D1, **{KEY: True}))["status"] == "sync_pending"
    entry = world.journal()["units"][D1]
    ctx = pr._entry_context(world.config(D1), entry, D1, world.checkout, f"{D1}T11:00:00+09:00")
    assert ctx["screen"]["state"] == "ok"  # 그날 설정에 키가 없어도 journal 값을 따릅니다
    (world.runs / D1 / f"screen-{D1}.json").write_bytes(b"{}\n")
    with pytest.raises(pr.PublishError, match="screen input changed"):
        pr._entry_context(world.config(D1), entry, D1, world.checkout, f"{D1}T11:00:00+09:00")
    entry = {k: v for k, v in entry.items() if k != KEY}
    ctx = pr._entry_context(world.config(D1), entry, D1, world.checkout, f"{D1}T11:00:00+09:00")
    assert "screen" not in ctx


def test_validate_accepts_a_published_screen_unit(world: World) -> None:
    world.make_run(D1)
    _screen_doc(world, D1)
    assert pr.run_step(world.config(D1, **{KEY: True}))["status"] == "published"
    assert md.validate(world.clone_for_checks()) == []
