"""reporting/macro.py: 거시 흐름 body -> macro.md 본문과 SVG. 합성 body만 쓴다(실제 데이터 없음)."""

from __future__ import annotations

import copy
import re
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from tests.scores.macro_context import _world as W  # noqa: E402

from modeler.reporting import macro  # noqa: E402
from modeler.scores import macro_context as mc  # noqa: E402

KW = dict(unit=W.D.isoformat(), invocation_id="inv-1", release="v1.2.3", selection_sha256="ab" * 32)
FORBIDDEN_SVG = re.compile(r"<script|\bon[a-z]+\s*=|href\s*=|foreignObject|xlink:", re.I)


@pytest.fixture(scope="module")
def body(tmp_path_factory) -> dict:
    world = W.build_world(tmp_path_factory.mktemp("macro_render"))
    return mc.build_document(W.make_selection(world), **KW)


@pytest.fixture(scope="module")
def rendered(body) -> tuple[str, dict[str, str]]:
    return macro.render(body)


def test_deterministic_and_input_untouched(body):
    before = copy.deepcopy(body)
    first = macro.render(body)
    second = macro.render(copy.deepcopy(body))
    assert first == second
    assert body == before
    assert first[0].encode() == second[0].encode()


def test_failed_document():
    failed = mc.failure_document(
        unit="2026-10-28",
        invocation_id="i",
        release="r",
        selection_sha256=None,
        reason="input_missing",
    )
    md, svgs = macro.render(failed)
    assert svgs == {}
    assert md.startswith("# 거시 흐름 · 2026-10-28\n")
    assert "점수·순위에 쓰지 않습니다. 매매 신호가 아닙니다." in md
    assert "거시 흐름을 이번 실행에서 만들지 못했습니다(사유 코드: input_missing)." in md
    assert "<" not in md
    failed["reason"] = "<b>x</b>\n| y"
    assert "<" not in macro.render(failed)[0]


def test_head_and_no_review_banner(rendered, body):
    md, _ = rendered
    lines = md.split("\n")
    assert lines[0] == f"# 거시 흐름 · {body['unit']}"
    assert lines[2] == "점수·순위에 쓰지 않습니다. 매매 신호가 아닙니다."
    assert lines[4].startswith("일간 지표 기준일 ")
    assert "리뷰용 시제품" not in md
    for h in (
        "## 이번에 눈여겨볼 변화",
        "## 지난 보고서 대비",
        "## 영역별 상세",
        "## 부록",
        "### 출처",
        "### 범례",
    ):
        assert h in md


def test_hide_level_is_dash(rendered, body):
    md, _ = rendered
    credit = next(o for o in body["indicators"] if o["id"] == "credit_baa")
    assert credit["display"]["level"] == "hide"
    row = next(r for r in md.split("\n") if r.startswith("| " + credit["display_name"]))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2] == "—"
    assert credit["value"]["display"] not in row

    shown = copy.deepcopy(body)
    for o in shown["indicators"]:
        if o["id"] == "us_3m":
            o["display"]["level"] = "hide"
    md2, _ = macro.render(shown)
    row = next(r for r in md2.split("\n") if r.startswith("| 미 3개월물 금리"))
    assert [c.strip() for c in row.strip("|").split("|")][2] == "—"


def test_trading_value_is_ratio_only(rendered):
    md, _ = rendered
    for line in md.split("\n"):
        if line.startswith("| 거래대금") or "거래대금" in line.split("|")[1:2].__str__():
            assert "조원" not in line, line
    assert "| 거래대금 비율 KOSPI" in md and "배 |" in md


def test_markdown_hygiene(rendered):
    md, _ = rendered
    assert "전환" not in md
    assert "<" not in md  # HTML 태그 없음
    no_code = re.sub(r"`[^`]*`", "", md)
    assert not re.search(r"(?<!\\)~", no_code), "이스케이프 안 된 물결표"
    assert "\\~" in md
    assert md.endswith("\n") and not md.endswith("\n\n")
    assert not re.search(r"(?<![\w\-])-\d", md), "ASCII 마이너스로 쓴 음수"


def test_figures_one_to_one_with_images(rendered, body):
    md, svgs = rendered
    names = {f["name"] for f in body["figures"]}
    assert set(svgs) == {f"macro-{n}.svg" for n in names}
    images = re.findall(r"!\[[^\]]*\]\(([^)]*)\)", md)
    assert sorted(images) == sorted(f"assets/macro-{n}.svg" for n in names)
    assert len(images) == len(set(images)) == 21


def test_svg_rules(rendered):
    _, svgs = rendered
    for name, svg in svgs.items():
        assert svg.startswith("<svg ") and svg.rstrip().endswith("</svg>"), name
        assert not FORBIDDEN_SVG.search(svg), name
        assert len(svg.encode("utf-8")) <= 200 * 1024, name
        # 값 문구의 음수는 U+2212
        for text in re.findall(r">([^<]*)</text>", svg):
            assert not re.match(r"^-\d", text), (name, text)


def test_svg_end_label_and_shade(rendered, body):
    _, svgs = rendered
    fig = next(f for f in body["figures"] if f["name"] == "fx_usdkrw")
    svg = svgs["macro-fx_usdkrw.svg"]
    last = fig["lines"][0]["last"]
    assert f"{last['value']:,.1f}원 · {last['date'][5:7]}/{last['date'][8:10]}" in svg
    assert "최근 3개월" in svg
    monthly = svgs["macro-macro_fedfunds.svg"]
    assert "최근 3개월" not in monthly
    n_points = len(fig["lines"][0]["points"])
    assert max(len(p.split()) for p in re.findall(r'points="([^"]*)"', svg)) == n_points


def test_monthly_gap_breaks_line(body):
    fig = copy.deepcopy(next(f for f in body["figures"] if f["name"] == "macro_fedfunds"))
    ln = fig["lines"][0]
    pts = ln["points"]
    assert len(pts) >= 10
    del pts[5]  # 결측 달 하나
    ln["segments"] = [[0, 5], [5, len(pts)]]
    one = copy.deepcopy(body)
    one["figures"] = [fig]
    svg = macro.render(one)[1]["macro-macro_fedfunds.svg"]
    counts = sorted(len(p.split()) for p in re.findall(r'points="([^"]*)"', svg))
    assert counts == [5, len(pts) - 5]


def test_fmt_template_whitelist(body):
    bad = copy.deepcopy(body)
    bad["figures"] = [bad["figures"][0]]
    bad["figures"][0]["lines"][0]["fmt"] = "{v.__class__}"
    with pytest.raises(ValueError):
        macro.render(bad)


def test_sources_use_snapshot_date_from_inputs(rendered, body):
    md, _ = rendered
    kd = body["inputs"]["kr"]["snapshot_date"]
    y, m, d = (int(x) for x in kd.split("-"))
    assert f"{y}.{m}.{d}. 스냅샷 기준." in md
    assert (
        f"KR 스냅샷 {kd}" in md
        and f"US macro_series 스냅샷 {body['inputs']['us']['snapshot_date']}" in md
    )
