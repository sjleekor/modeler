"""PrivateViewBuilder: the owner-only HTML view that outlived the public Pages projection."""

from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from modeler.reporting.site import (
    PRIVATE_MARKER_NAME,
    PRIVATE_MARKER_TEXT,
    PrivateViewBuilder,
    _safe_path,
)
from modeler.serving.schema import report_template

SEOUL = ZoneInfo("Asia/Seoul")
D = date(2026, 9, 29)


def _report(market: str, model_id: str) -> dict:
    report = report_template(
        market=market,
        report_date=D.isoformat(),
        decision_at=datetime(2026, 9, 29, 10, tzinfo=SEOUL),
        feature_asof_date="2026-09-28",
        model_id=model_id,
        model_version="1",
    )
    report.update(status="ok", synthetic_fixture=True)
    return report


def _opening() -> dict:
    return {
        "status": "ok",
        "publication": {"status": "unresolved", "evidence": [{"id": "opening-source"}]},
        "observed_at": "2026-09-29T09:25:00+09:00",
        "session_state": "regular",
        "indices": [
            {
                "publication": {"status": "allowed"},
                "code": "KOSPI",
                "name": "KOSPI",
                "open": 100.0,
                "previous_close": 99.0,
                "last": 101.0,
                "change_pct": 2.02,
                "observed_at": "2026-09-29T09:25:00+09:00",
            }
        ],
        "industries": [],
    }


def _ranked(n: int = 150, name: str = "종목") -> list[dict]:
    reports = [
        _report("KR", "kr_daily_h20_v1"),
        _report("US", "us_exploratory_20260929_r1_lightgbm"),
        _report("US", "us_exploratory_20260929_r1_ridge"),
    ]
    for report in reports:
        report["rankings"] = [
            {"rank": i + 1, "symbol": f"S{i:03d}", "name": f"{name}{i}", "score": 1 / (i + 3)}
            for i in range(n)
        ]
    reports[0]["rankings"][0].update(quality_review=True, quality_reasons=["K_halt_or_unknown"])
    reports[0]["rankings"][1].update(quality_review=False, quality_reasons=[])
    return reports


def _render(**kw):
    return PrivateViewBuilder(synthetic_fixture=True).render(
        report_date=D, reports=_ranked(), opening=_opening(), **kw
    )


def test_private_view_has_rankings_scores_and_quality_holds() -> None:
    page = _render()[f"reports/{D.isoformat()}/index.html"].decode()
    assert "전체 150개 중 상위 100개" in page and "S099" in page and "S100" not in page
    assert "0.3333" in page and "순위용 점수 — 확률 아님" in page
    assert "품질 보류: K 기준 거래정지 또는 상태 미확인" in page and "이상 없음" in page


def _norm(parts):
    out = []
    for part in parts:
        if part == "..":
            out.pop()
        else:
            out.append(part)
    return out


def test_private_pages_banner_noindex_synthetic_relative_links_and_manifest() -> None:
    files = _render()
    for path, body in files.items():
        if path.endswith(".html"):
            text = body.decode()
            assert "비공개 — 게시 금지" in text
            assert '<meta name="robots" content="noindex,nofollow">' in text
            assert "합성 fixture 예시 전용" in text
            for href in re.findall(r'href="([^"]+)"', text):
                assert not href.startswith(("/", "http")), (path, href)
                target = Path(path).parent.joinpath(href)
                resolved = Path(*_norm(target.parts))
                assert resolved.as_posix() in files, (path, href)
    manifest = json.loads(files["private-manifest.json"])
    assert manifest["private"] is True and manifest["latest_report_date"] == D.isoformat()
    assert "PRIVATE" in files[PRIVATE_MARKER_NAME].decode()
    assert b"0.3333" not in files["private-manifest.json"]


def test_marker_forbids_copying_this_directory_but_not_the_markdown_units() -> None:
    # "저장소에 올리지 않는다"는 private 저장소 push와 충돌했습니다.
    # 금지 대상은 이 화면 디렉터리입니다.
    assert "stock_reports" in PRIVATE_MARKER_TEXT and "markdown 단위" in PRIVATE_MARKER_TEXT
    assert "이 디렉터리를 어떤 저장소" in PRIVATE_MARKER_TEXT
    assert "GitHub Pages·저장소·공개 저장소로 복사하거나 올리지 않습니다" not in PRIVATE_MARKER_TEXT
    assert PRIVATE_MARKER_TEXT.startswith("PRIVATE - DO NOT PUBLISH\n")


def test_private_view_escapes_html_and_opening_time_rule() -> None:
    opening = _opening()
    opening["indices"].append(
        {
            "publication": {"status": "allowed"},
            "code": "NOTIME",
            "name": "<b>x</b>",
            "last": 1.0,
            "change_pct": 0.1,
        }
    )
    files = PrivateViewBuilder().render(
        report_date=D, reports=_ranked(3, "<script>alert(1)</script>"), opening=opening
    )
    page = files[f"reports/{D.isoformat()}/index.html"].decode()
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page
    assert "<b>x</b>" not in page and "&lt;b&gt;x&lt;/b&gt;" in page
    assert "2026-09-29 09:25:00 KST" in page and "원천 시각 미확인" in page
    assert "합성 fixture" not in page
    unavailable = PrivateViewBuilder().render(
        report_date=D, reports=_ranked(3), opening={"status": "unavailable"}
    )
    page = unavailable[f"reports/{D.isoformat()}/index.html"].decode()
    assert "장중 관측으로 확인되지 않았습니다" in page


def test_private_view_keeps_previous_dates_and_rejects_mode_mismatch() -> None:
    first = _render()

    def later(reports: list[dict]) -> list[dict]:
        return [
            {**r, "report_date": "2026-09-30", "decision_at": "2026-09-30T10:00:00+09:00"}
            for r in reports
        ]

    second = PrivateViewBuilder(synthetic_fixture=True).render(
        report_date=date(2026, 9, 30),
        reports=later(_ranked()),
        opening=_opening(),
        previous_files=first,
    )
    archive = second["archive/index.html"].decode()
    assert "2026-09-29" in archive and "2026-09-30" in archive
    assert f"reports/{D.isoformat()}/index.html" in second
    with pytest.raises(ValueError):
        PrivateViewBuilder(synthetic_fixture=False).render(
            report_date=date(2026, 9, 30), reports=later(_ranked()), previous_files=first
        )


def test_private_css_wraps_korean_cells_and_tables_scroll() -> None:
    files = _render()
    css = files["assets/private.css"].decode()
    assert "word-break:keep-all" in css and "overflow-x:auto" in css and "min-width" in css
    page = files[f"reports/{D.isoformat()}/index.html"].decode()
    assert '<div class="table-scroll"><table>' in page


def test_private_render_is_deterministic_and_files_are_written_idempotently(tmp_path: Path) -> None:
    builder = PrivateViewBuilder(synthetic_fixture=True)
    files = builder.render(report_date=D, reports=_ranked(), opening=_opening())
    assert files == builder.render(report_date=D, reports=_ranked(), opening=_opening())
    target = tmp_path / "private"
    builder.write_atomic(target, files)
    builder.write_atomic(target, files)
    with pytest.raises(FileExistsError):
        builder.write_atomic(target, {**files, "index.html": b"different"})
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        builder.write_atomic(link, files)


def test_path_traversal_is_rejected() -> None:
    for bad in ("reports/../../private.json", "/abs/path", "a\\b", ""):
        with pytest.raises(ValueError):
            _safe_path(bad)
    assert _safe_path("reports/2026-09-29/index.html") == "reports/2026-09-29/index.html"
