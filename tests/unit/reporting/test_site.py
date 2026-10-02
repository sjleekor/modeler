from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from modeler.reporting.site import SiteBuilder, _safe_path
from modeler.serving.schema import report_template

SEOUL = ZoneInfo("Asia/Seoul")
D = date(2026, 9, 29)


def _report(market: str, model_id: str, *, publication="unresolved", symbol="SYN001", name="합성종목"):
    report = report_template(market=market, report_date=D.isoformat(),
        decision_at=datetime(2026, 9, 29, 10, tzinfo=SEOUL),
        feature_asof_date="2026-09-28", model_id=model_id, model_version="1")
    report.update(status="ok", synthetic_fixture=True,
        provenance={"private_source_path": "/secret/source", "token": "TOKEN_SENTINEL"},
        quality={"raw_response": "RAW_SENTINEL", "private_count": 99},
        publication={"status": publication,
                     "evidence": [{"id": "source-policy-2026", "token": "EVIDENCE_TOKEN"}]})
    report["rankings"] = [{"rank": 1, "symbol": symbol, "name": name, "score": 0.987654321}]
    return report


def _reports(publication="unresolved", name="합성종목"):
    return [
        _report("KR", "kr_daily_h20_v1", publication=publication, name=name),
        _report("US", "us_exploratory_20260929_r1_lightgbm", publication=publication, name=name),
        _report("US", "us_exploratory_20260929_r1_ridge", publication=publication, name=name),
    ]


def _opening(publication="unresolved"):
    return {"status": "ok", "publication": {"status": publication, "evidence": [{"id": "opening-source"}]},
        "observed_at": "2026-09-29T09:25:00+09:00", "session_state": "regular",
        "indices": [{"publication": {"status": "allowed"}, "code": "KOSPI", "name": "KOSPI",
                     "open": 100.0, "previous_close": 99.0, "last": 101.0, "change_pct": 2.02,
                     "observed_at": "2026-09-29T09:25:00+09:00", "raw_response": "RAW_SENTINEL",
                     "token": "TOKEN_SENTINEL", "internal_path": "/secret/path"}],
        "industries": [{"publication": {"status": "allowed"}, "code": "I001", "name": "합성업종",
                        "rank": 1, "change_pct": 1.2, "observed_at": "2026-09-29T09:25:00+09:00",
                        "raw_response": "RAW_SENTINEL", "internal_path": "/secret/path"}]}


def _all_text(files: dict[str, bytes]) -> str:
    return "\n".join(value.decode("utf-8", errors="replace") for value in files.values())


def test_unresolved_and_withheld_projection_removes_scores_rankings_and_private_fields():
    for state in ("unresolved", "withheld"):
        files = SiteBuilder(synthetic_fixture=True).render(report_date=D, reports=_reports(state),
            opening=_opening(state), model_cards={"kr_daily_h20_v1": {
                "publication": {"status": state}, "summary": "MODEL_CARD_SECRET", "token": "TOKEN_SENTINEL"}})
        text = _all_text(files)
        report_json = json.loads(files[f"reports/{D.isoformat()}/report.json"])
        assert all(item["rankings"] == [] for item in report_json["markets"])
        assert "0.987654321" not in text
        assert "SYN001" not in text
        assert "RAW_SENTINEL" not in text
        assert "TOKEN_SENTINEL" not in text
        assert "EVIDENCE_TOKEN" not in text
        assert "MODEL_CARD_SECRET" not in text
        assert "/secret/path" not in text
        assert "synthetic_fixture" in text
        assert "실제 시장 데이터가 아닙니다" in text


def test_allowed_projection_still_hides_score_and_uses_allowlisted_opening_fields():
    files = SiteBuilder(synthetic_fixture=True).render(report_date=D,
        reports=_reports("allowed", name="</td><script>alert(1)</script>"),
        opening=_opening("allowed"))
    text = _all_text(files)
    report_json = json.loads(files[f"reports/{D.isoformat()}/report.json"])
    assert report_json["status"] == "ok"
    assert len(report_json["markets"]) == 3
    assert all("score" not in row for market in report_json["markets"] for row in market["rankings"])
    assert "0.987654321" not in text
    assert "RAW_SENTINEL" not in text and "TOKEN_SENTINEL" not in text
    assert "/secret/path" not in text
    page = files[f"reports/{D.isoformat()}/index.html"].decode()
    assert "<script>alert(1)</script>" not in page
    assert "&lt;/td&gt;&lt;script&gt;" in page
    assert '"internal_path"' not in files[f"reports/{D.isoformat()}/report.json"].decode()
    assert '"name":"KOSPI"' in files[f"reports/{D.isoformat()}/report.json"].decode()
    assert "/market-briefing/assets/site-" in page
    home = files["index.html"].decode()
    assert "최신 리포트 상태" in home
    assert "오늘 자료 상태" not in home


def test_model_card_page_uses_readable_sections_and_escapes_allowed_text():
    model_id = "kr_daily_h20_v1"
    files = SiteBuilder(synthetic_fixture=True).render(
        report_date=D, reports=_reports(), opening=_opening(), model_cards={model_id: {
            "title": "한국 종목 탐색 순위",
            "summary": "긴 한국어 설명 " * 30 + "<script>leak()</script>",
            "target": "동일 시장 대비 상대 순위",
            "scope": "학습 경계와 적용 범위",
            "limitations": "실제 수익률 아님",
            "publication": {"status": "allowed", "evidence": [{"token": "CARD_TOKEN"}]},
            "private_path": "/secret/model",
        }})
    page_path = next(path for path in files if path.startswith(f"models/{model_id}/")
                     and path.endswith("/index.html"))
    page = files[page_path].decode()
    card = json.loads(files[page_path.removesuffix("index.html") + "model-card.json"])
    css = next(value.decode() for path, value in files.items() if path.endswith(".css"))
    assert '<article class="model-card">' in page
    assert f"<small>{model_id}</small>" in page
    assert "<h2>한국 종목 탐색 순위" in page
    for label in ("소개", "예측 대상", "학습·적용 범위", "주의할 점"):
        assert f"<h3>{label}</h3><p>" in page
    assert "<pre>" not in page
    assert "publication:" not in page
    assert "CARD_TOKEN" not in _all_text(files)
    assert "/secret/model" not in _all_text(files)
    assert "<script>leak()</script>" not in page
    assert "&lt;script&gt;leak()&lt;/script&gt;" in page
    assert '.model-card p{overflow-wrap:anywhere;word-break:keep-all}' in css
    assert set(card) == {"model_id", "publication", "title", "summary", "target", "scope", "limitations"}


def test_model_card_page_withheld_gate_hides_text_and_internal_publication_dict():
    files = SiteBuilder(synthetic_fixture=True).render(
        report_date=D, reports=_reports(), opening=_opening(), model_cards={"kr_daily_h20_v1": {
            "title": "PRIVATE_TITLE", "summary": "PRIVATE_SUMMARY",
            "publication": {"status": "withheld", "evidence": [{"token": "CARD_TOKEN"}]},
        }})
    path = next(path for path in files if path.startswith("models/kr_daily_h20_v1/")
                and path.endswith("/index.html"))
    page = files[path].decode()
    assert "모델 설명 공개를 보류합니다." in page
    assert "PRIVATE_TITLE" not in _all_text(files)
    assert "PRIVATE_SUMMARY" not in _all_text(files)
    assert "CARD_TOKEN" not in _all_text(files)
    assert "publication:" not in page


def test_KR_quality_reason_survives_public_projection_without_raw_price_or_score():
    reports = _reports("allowed")
    reports[0]["rankings"] = [
        {"rank": 1, "symbol": "SYN001", "name": "합성종목", "score": 0.99,
         "market": "KOSPI", "K_simple_return": 0.35, "quality_review": True,
         "quality_reasons": ["K_price_jump_or_unknown", "PRIVATE_REASON"]},
        {"rank": 2, "symbol": "SYN002", "name": "두 번째", "score": 0.01,
         "market": "KOSPI", "quality_review": False, "quality_reasons": []},
    ]
    reports[0]["inference_started_at"] = "2026-09-29T01:00:01.123456+00:00"
    files = SiteBuilder(synthetic_fixture=True).render(
        report_date=D, reports=reports, opening=_opening("allowed"),
    )
    public = json.loads(files[f"reports/{D.isoformat()}/report.json"])
    kr = next(market for market in public["markets"] if market["market"] == "KR")
    assert [row["rank"] for row in kr["rankings"]] == [1, 2]
    assert kr["rankings"][0]["quality_review"] is True
    assert kr["rankings"][0]["quality_reasons"] == ["K 기준 가격 급변 또는 확인 불가"]
    assert kr["rankings"][1]["quality_review"] is False
    assert kr["quality"]["top100_quality_review_rows"] == 1
    text = _all_text(files)
    assert "PRIVATE_REASON" not in text
    assert "K_simple_return" not in text
    assert "0.35" not in text
    assert '"score"' not in text
    page = files[f"reports/{D.isoformat()}/index.html"].decode()
    assert "현재 관리종목·거래정지 상태는 확인되지 않았습니다" in page
    assert "2026-09-29 10:00:01 KST" in page
    assert "2026-09-29T01:00:01.123456+00:00" not in page
    assert "미국 종목 탐색 순위 · LightGBM" in page
    assert "상태: 정상</p>" in page


def test_public_projection_caps_rankings_at_one_hundred():
    reports = _reports("allowed")
    reports[0]["rankings"] = [
        {"rank": rank, "symbol": f"SYN{rank:03d}", "name": f"종목 {rank}", "score": 0.5}
        for rank in range(1, 102)
    ]
    files = SiteBuilder(synthetic_fixture=True).render(report_date=D,
        reports=reports, opening=_opening("allowed"))
    text = _all_text(files)
    report_json = json.loads(files[f"reports/{D.isoformat()}/report.json"])
    kr_market = next(row for row in report_json["markets"] if row["market"] == "KR")
    assert len(kr_market["rankings"]) == 100
    assert kr_market["rankings"][-1]["rank"] == 100
    assert "SYN101" not in text
    assert "종목 101" not in text


def test_revision_is_immutable_and_replay_does_not_move_latest_date_backwards():
    builder = SiteBuilder(synthetic_fixture=True)
    day1_files = builder.render(report_date=D, reports=_reports("allowed"), opening=_opening("allowed"))
    manifest = json.loads(day1_files["site-manifest.json"])
    day1_revision = manifest["reports"][0]["revision"]
    day1_payload_path = f"reports/{D.isoformat()}/revisions/{day1_revision}/report.json"
    old_immutable_payload = day1_files[day1_payload_path]
    later = date(2026, 9, 30)
    later_reports = [dict(report, report_date=later.isoformat(),
                          decision_at="2026-09-30T10:00:00+09:00") for report in _reports("allowed")]
    later_files = builder.render(report_date=later, reports=later_reports,
        opening={**_opening("allowed")}, previous_files=day1_files)
    before_replay = dict(later_files)
    corrected = _reports("allowed", name="고친 합성종목")
    replay_files = builder.render(report_date=D, reports=corrected, opening=_opening("allowed"),
        previous_files=later_files, historical_replay=True,
        inference_started_at=datetime(2026, 9, 30, 10, 2, tzinfo=SEOUL))
    manifest = json.loads(replay_files["site-manifest.json"])
    assert manifest["latest_report_date"] == later.isoformat()
    assert replay_files[day1_payload_path] == old_immutable_payload
    assert replay_files[f"reports/{later.isoformat()}/report.json"] == before_replay[
        f"reports/{later.isoformat()}/report.json"]
    assert "historical_replay" in replay_files[f"reports/{D.isoformat()}/index.html"].decode()
    assert "historical_replay" in replay_files[f"reports/{D.isoformat()}/report.json"].decode()
    revisions = [name for name in replay_files if name.startswith(f"reports/{D.isoformat()}/revisions/")]
    assert any(day1_revision in name for name in revisions)
    assert len({name.split("/")[3] for name in revisions}) == 2


def test_site_render_is_deterministic_and_files_are_written_idempotently(tmp_path):
    builder = SiteBuilder(synthetic_fixture=True)
    files = builder.render(report_date=D, reports=_reports(), opening=_opening())
    assert files == builder.render(report_date=D, reports=_reports(), opening=_opening())
    target = tmp_path / "site"
    builder.write_atomic(target, files)
    builder.write_atomic(target, files)
    with pytest.raises(FileExistsError):
        builder.write_atomic(target, {**files, "index.html": b"different"})


def test_path_traversal_and_missing_base_path_are_rejected():
    with pytest.raises(ValueError):
        _safe_path("reports/../../private.json")
    with pytest.raises(ValueError):
        SiteBuilder(base_path="/market-briefing/../private/")


def test_revision_hash_covers_public_payload_html_and_versioned_css():
    files = SiteBuilder(synthetic_fixture=True).render(report_date=D,
        reports=_reports("allowed"), opening=_opening("allowed"))
    manifest = json.loads(files["site-manifest.json"])
    revision = manifest["reports"][0]["revision"]
    html_bytes = files[f"reports/{D.isoformat()}/revisions/{revision}/index.html"]
    payload = files[f"reports/{D.isoformat()}/revisions/{revision}/report.json"]
    css_ref = next(line.split('href="', 1)[1].split('"', 1)[0] for line in html_bytes.decode().splitlines()
                   if 'stylesheet' in line)
    css_path = css_ref.removeprefix("/market-briefing/")
    digest = hashlib.sha256(payload + html_bytes + files[css_path]).hexdigest()
    assert digest.startswith(revision)


CARDS = {
    "kr_daily_h20_v1": ("한국 종목 탐색 순위 · H20", "kr"),
    "us_exploratory_20260929_r1_lightgbm": ("미국 탐색 <b>LightGBM</b>", "lgb"),
    "us_exploratory_20260929_r1_ridge": ("미국 탐색 Ridge", "ridge"),
}


def _cards():
    return {mid: {"title": title, "summary": "요약", "publication": {"status": "allowed"}}
            for mid, (title, _) in CARDS.items()}


def _site(**kwargs):
    return SiteBuilder(synthetic_fixture=True).render(
        report_date=D, reports=_reports("allowed"), opening=_opening("allowed"),
        model_cards=_cards(), **kwargs)


def _hrefs(page: str) -> list[str]:
    return re.findall(r'href="([^"]+)"', page)


def _resolve(files, href):
    assert href.startswith("/market-briefing/") and "://" not in href
    path = href.removeprefix("/market-briefing/")
    if path == "" or path.endswith("/"):
        path += "index.html"
    return path


def _html_pages(files):
    return {path: value.decode() for path, value in files.items() if path.endswith(".html")}


def test_every_page_has_home_archive_and_models_links():
    files = _site()
    pages = _html_pages(files)
    assert "index.html" in pages and "archive/index.html" in pages and "models/index.html" in pages
    for path, page in pages.items():
        hrefs = _hrefs(page)
        for target in ("index.html", "archive/index.html", "models/index.html"):
            assert f"/market-briefing/{target}" in hrefs, (path, target)


def test_model_and_month_pages_link_back_and_title_is_escaped():
    files = _site()
    month = files["archive/2026-09/index.html"].decode()
    assert "/market-briefing/index.html" in _hrefs(month)
    model_pages = {p: v.decode() for p, v in files.items() if p.startswith("models/kr_") and p.endswith("index.html")}
    assert len(model_pages) == 1
    page = next(iter(model_pages.values()))
    assert "/market-briefing/index.html" in _hrefs(page) and "/market-briefing/archive/index.html" in _hrefs(page)
    assert "<title>한국 종목 탐색 순위 · H20 · 모델 설명</title>" in page
    lgb = next(v.decode() for p, v in files.items()
               if p.startswith("models/us_exploratory_20260929_r1_lightgbm/") and p.endswith("index.html"))
    assert "<b>LightGBM</b>" not in lgb
    assert "<title>미국 탐색 &lt;b&gt;LightGBM&lt;/b&gt; · 모델 설명</title>" in lgb


def test_home_and_model_list_show_card_titles_with_id_and_fallback():
    files = _site()
    for page in (files["index.html"].decode(), files["models/index.html"].decode()):
        assert ">한국 종목 탐색 순위 · H20</a><small>kr_daily_h20_v1</small>" in page
        assert "&lt;b&gt;LightGBM&lt;/b&gt;" in page and "<b>LightGBM</b>" not in page
    fallback = SiteBuilder(synthetic_fixture=True).render(
        report_date=D, reports=_reports(), opening=_opening(),
        model_cards={"kr_daily_h20_v1": {"title": "SECRET_TITLE", "publication": {"status": "withheld"}}})
    home = fallback["index.html"].decode()
    assert ">kr_daily_h20_v1</a>" in home and "SECRET_TITLE" not in _all_text(fallback)


def test_date_page_sections_link_to_their_model_page():
    files = _site()
    page = files[f"reports/{D.isoformat()}/index.html"].decode()
    hrefs = _hrefs(page)
    for model_id in CARDS:
        model_dir = next(p.removesuffix("index.html") for p in files
                         if p.startswith(f"models/{model_id}/") and p.endswith("index.html"))
        assert "/market-briefing/" + model_dir in hrefs
    assert page.count("모델 설명 보기") == 3


def test_month_and_archive_lists_show_status_and_synthetic_mark():
    files = _site()
    for path in ("archive/index.html", "archive/2026-09/index.html"):
        page = files[path].decode()
        assert "2026-09-29</a> · " in page and "합성" in page
    real = SiteBuilder().render(report_date=D, reports=_reports("allowed")[:1], opening=_opening("allowed"))
    month = real["archive/2026-09/index.html"].decode()
    assert "부분 완료" in month and "합성" not in month


def test_all_internal_links_resolve_to_files_and_no_external_links():
    first = _site()
    later = date(2026, 9, 30)
    later_reports = [dict(r, report_date=later.isoformat(), decision_at="2026-09-30T10:00:00+09:00")
                     for r in _reports("allowed")]
    files = SiteBuilder(synthetic_fixture=True).render(
        report_date=later, reports=later_reports, opening=_opening("allowed"),
        model_cards=_cards(), previous_files=first)
    for path, page in _html_pages(files).items():
        for href in _hrefs(page):
            assert "://" not in href and not href.startswith("//"), (path, href)
            assert _resolve(files, href) in files, (path, href)
