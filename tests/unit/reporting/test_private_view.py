from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

from modeler.reporting.site import PrivateViewBuilder, SiteBuilder
from test_site import D, _all_text, _opening, _reports

PAGES = Path(__file__).resolve().parents[3] / "deploy" / "pages"
sys.path.insert(0, str(PAGES))
import validate_public_site  # noqa: E402


def _ranked(n: int = 150, name: str = "종목"):
    reports = _reports("unresolved")
    for report in reports:
        report["rankings"] = [{"rank": i + 1, "symbol": f"S{i:03d}", "name": f"{name}{i}", "score": 1 / (i + 3)}
                              for i in range(n)]
    reports[0]["rankings"][0].update(quality_review=True, quality_reasons=["K_halt_or_unknown"])
    reports[0]["rankings"][1].update(quality_review=False, quality_reasons=[])
    return reports


def _render(**kw):
    return PrivateViewBuilder(synthetic_fixture=True).render(report_date=D, reports=_ranked(), opening=_opening(), **kw)


def test_private_view_has_rankings_scores_and_public_projection_does_not():
    reports = _ranked()
    private = _render()
    page = private[f"reports/{D.isoformat()}/index.html"].decode()
    assert "전체 150개 중 상위 100개" in page and "S099" in page and "S100" not in page
    assert "0.3333" in page and "순위용 점수 — 확률 아님" in page
    assert "품질 보류: K 기준 거래정지 또는 상태 미확인" in page and "이상 없음" in page
    public = SiteBuilder(synthetic_fixture=True).render(report_date=D, reports=reports, opening=_opening())
    text = _all_text(public)
    assert "S000" not in text and "0.3333" not in text and "비공개 — 게시 금지" not in text
    assert "PRIVATE_DO_NOT_PUBLISH" not in "".join(public)


def test_private_pages_banner_noindex_synthetic_relative_links_and_manifest():
    files = _render()
    for path, body in files.items():
        if path.endswith(".html"):
            text = body.decode()
            assert "비공개 — 게시 금지" in text and '<meta name="robots" content="noindex,nofollow">' in text
            assert "합성 fixture 예시 전용" in text
            for href in re.findall(r'href="([^"]+)"', text):
                assert not href.startswith(("/", "http")), (path, href)
                assert (Path(path).parent / href).as_posix() is not None
                target = Path(path).parent.joinpath(href)
                resolved = Path(*[p for p in _norm(target.parts)])
                assert resolved.as_posix() in files, (path, href)
    manifest = json.loads(files["private-manifest.json"])
    assert manifest["private"] is True and manifest["latest_report_date"] == D.isoformat()
    assert "PRIVATE" in files["PRIVATE_DO_NOT_PUBLISH.txt"].decode()
    assert b"0.3333" not in files["private-manifest.json"]


def _norm(parts):
    out = []
    for part in parts:
        if part == "..":
            out.pop()
        else:
            out.append(part)
    return out


def test_private_view_escapes_html_and_opening_time_rule():
    opening = _opening()
    opening["indices"].append({"publication": {"status": "allowed"}, "code": "NOTIME", "name": "<b>x</b>",
                               "last": 1.0, "change_pct": 0.1})
    files = PrivateViewBuilder().render(report_date=D, reports=_ranked(3, "<script>alert(1)</script>"),
                                        opening=opening)
    page = files[f"reports/{D.isoformat()}/index.html"].decode()
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert "<b>x</b>" not in page and "&lt;b&gt;x&lt;/b&gt;" in page
    assert "2026-09-29 09:25:00 KST" in page and "원천 시각 미확인" in page
    assert "합성 fixture" not in page
    unavailable = PrivateViewBuilder().render(report_date=D, reports=_ranked(3), opening={"status": "unavailable"})
    assert "장중 관측으로 확인되지 않았습니다" in unavailable[f"reports/{D.isoformat()}/index.html"].decode()


def test_public_validator_rejects_private_output(tmp_path: Path):
    target = tmp_path / "private"
    builder = PrivateViewBuilder(synthetic_fixture=True)
    builder.write_atomic(target, _render())
    with pytest.raises(ValueError):
        validate_public_site.validate_site(target, allow_synthetic=True)
    # a public tree polluted with the private marker is rejected too
    public = SiteBuilder(synthetic_fixture=True).render(report_date=D, reports=_ranked(), opening=_opening())
    public["PRIVATE_DO_NOT_PUBLISH.txt"] = b"x"
    SiteBuilder(synthetic_fixture=True).write_atomic(tmp_path / "polluted", public)
    with pytest.raises(ValueError):
        validate_public_site.validate_site(tmp_path / "polluted", allow_synthetic=True)


def test_private_view_keeps_previous_dates_and_rejects_mode_mismatch():
    first = _render()
    from datetime import date
    second = PrivateViewBuilder(synthetic_fixture=True).render(
        report_date=date(2026, 9, 30), reports=[{**r, "report_date": "2026-09-30", "decision_at": "2026-09-30T10:00:00+09:00"} for r in _ranked()],
        opening=_opening(), previous_files=first)
    archive = second["archive/index.html"].decode()
    assert "2026-09-29" in archive and "2026-09-30" in archive
    assert f"reports/{D.isoformat()}/index.html" in second
    with pytest.raises(ValueError):
        PrivateViewBuilder(synthetic_fixture=False).render(
            report_date=date(2026, 9, 30), reports=[{**r, "report_date": "2026-09-30", "decision_at": "2026-09-30T10:00:00+09:00"} for r in _ranked()],
            previous_files=first)


def test_private_css_wraps_korean_cells_and_tables_scroll():
    files = _render()
    css = files["assets/private.css"].decode()
    assert "word-break:keep-all" in css and "overflow-x:auto" in css and "min-width" in css
    page = files[f"reports/{D.isoformat()}/index.html"].decode()
    assert '<div class="table-scroll"><table>' in page
    public = SiteBuilder().render(report_date=D, reports=_reports("unresolved"), opening=_opening())
    assert all(b"table-scroll" not in v for v in public.values())
