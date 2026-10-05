"""Owner-only private HTML view of the daily reports.

The public projection (``SiteBuilder``) and its Pages delivery were removed when publication moved
to the private ``stock_reports`` repository (see ``modeler.reporting.markdown`` and
``deploy/reports/publish_reports.py``).  The private view stays: it is a server-side screen for the
owner and is never copied into any repository.
"""
from __future__ import annotations

import html
import json
import math
import os
import re
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from collections.abc import Iterable
from zoneinfo import ZoneInfo

from modeler.serving.schema import validate_report

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EXPECTED_MARKETS = {
    ("KR", "kr_daily_h20_v1"),
    ("US", "us_exploratory_20260929_r1_lightgbm"),
    ("US", "us_exploratory_20260929_r1_ridge"),
}
SEOUL = ZoneInfo("Asia/Seoul")
MODEL_DISPLAY_NAMES = {
    "kr_daily_h20_v1": "한국 종목 탐색 순위",
    "us_exploratory_20260929_r1_lightgbm": "미국 종목 탐색 순위 · LightGBM",
    "us_exploratory_20260929_r1_ridge": "미국 종목 탐색 순위 · Ridge",
}
KR_QUALITY_REASON_TEXT = {
    "K_halt_or_unknown": "K 기준 거래정지 또는 상태 미확인",
    "K_price_jump_or_unknown": "K 기준 가격 급변 또는 확인 불가",
    "K_share_change_or_unknown": "K 기준 상장주식수 변화 또는 확인 불가",
    "K_price_rule_unknown": "K 기준 가격 점검 규칙 적용 불가",
    "K_price_quality_unavailable": "K 기준 가격 품질 정보 없음",
}


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _safe_path(value: str) -> str:
    path = PurePosixPath(value)
    if (path.is_absolute() or not path.parts or "\\" in value or "\x00" in value or
            any(part in {".", ".."} for part in path.parts)):
        raise ValueError("site paths must be relative and cannot traverse directories")
    return path.as_posix()


def _status_text(status: str) -> str:
    return {"ok": "정상", "partial": "부분 완료", "stale": "자료 지연",
            "withheld": "공개 보류", "failed": "실패", "unavailable": "자료 없음"}.get(status, status)


def _css() -> bytes:
    return (b"body{font:16px/1.6 system-ui,sans-serif;max-width:70rem;margin:auto;padding:1rem;color:#18212b}"
            b"h2,small{overflow-wrap:anywhere}h2 small{display:block;font-size:.75rem;font-weight:normal;color:#526273}"
            b".model-card p{overflow-wrap:anywhere;word-break:keep-all}"
            b"table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #aab4bf;"
            b"padding:.4rem .7rem;text-align:left}.fixture{font-weight:bold;color:#8b2600}"
            b"nav{display:flex;flex-wrap:wrap;gap:.3rem 1.2rem;padding-bottom:.6rem;margin-bottom:1rem;"
            b"border-bottom:1px solid #aab4bf}h1,li{overflow-wrap:anywhere}"
            b"li small{color:#526273;font-size:.75rem;margin-left:.4rem}")


def _date_status(markets: list[dict[str, Any]], opening: dict[str, Any]) -> str:
    if not markets:
        return "failed"
    identities = {(row["market"], row["model_id"]) for row in markets}
    if not EXPECTED_MARKETS <= identities:
        return "partial"
    if any(row["status"] != "ok" or row["publication"]["status"] != "allowed" for row in markets):
        if all(row["status"] in {"failed", "stale", "unavailable"} for row in markets):
            return "failed"
        return "partial"
    if opening.get("status") != "ok" or opening.get("publication") != "allowed":
        return "partial"
    return "ok"


def write_tree_atomic(target: Path, files: dict[str, bytes]) -> None:
    """Write a complete generated tree, then atomically rename into a new target."""
    if target.is_symlink():
        raise ValueError("site target cannot be a symlink")
    target = target.absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if _read_tree(target) == files:
            return
        raise FileExistsError("site target exists with different content")
    staging = Path(tempfile.mkdtemp(prefix=".site-stage-", dir=target.parent))
    try:
        for relative, content in files.items():
            path = staging.joinpath(*PurePosixPath(_safe_path(relative)).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        _reject_symlinks(staging)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _reject_symlinks(root: Path) -> None:
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("generated site cannot contain symlinks")


def _read_tree(root: Path) -> dict[str, bytes]:
    if root.is_symlink():
        raise ValueError("site target cannot be a symlink")
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("site target contains symlink")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = path.read_bytes()
    return result


# ---------------------------------------------------------------------------
# Private view: owner-only pages with rankings and scores. Never copied into any repository.
# ---------------------------------------------------------------------------
PRIVATE_SCHEMA = "private-1.0"
PRIVATE_MARKER_NAME = "PRIVATE_DO_NOT_PUBLISH.txt"
PRIVATE_MANIFEST_NAME = "private-manifest.json"
PRIVATE_CSS_PATH = "assets/private.css"
PRIVATE_BANNER = "비공개 — 게시 금지"
PRIVATE_MARKER_TEXT = ("PRIVATE - DO NOT PUBLISH\n"
                       "비공개 — 게시 금지. 이 디렉터리는 서버에만 두는 본인용 화면이고 순위·점수·시세가 들어 있습니다.\n"
                       "이 디렉터리를 어떤 저장소(stock_reports 포함)나 Pages로도 복사하거나 올리지 않습니다.\n"
                       "stock_reports에는 reports publisher가 렌더한 markdown 단위만 올라갑니다.\n")
PRIVATE_SCORE_NOTE = "순위용 점수 — 확률 아님"
PRIVATE_LIST_LIMIT = 100


def _private_css() -> bytes:
    return (_css().decode() +
            ".private-banner{background:#8b0000;color:#fff;font-weight:bold;padding:.6rem 1rem;"
            "margin:-1rem -1rem 1rem;text-align:center}"
            ".synthetic-banner{background:#8b2600;color:#fff;font-weight:bold;padding:.4rem 1rem;"
            "margin:0 -1rem 1rem;text-align:center}"
            ".hold{color:#8b2600}td.num{text-align:right;font-variant-numeric:tabular-nums}"
            ".table-scroll{overflow-x:auto;max-width:100%;-webkit-overflow-scrolling:touch}"
            ".table-scroll table{display:table;min-width:36rem;width:100%}"
            ".table-scroll td,.table-scroll th{word-break:keep-all;overflow-wrap:normal;white-space:normal}"
            ".table-scroll td:nth-child(2),.table-scroll th:nth-child(2){white-space:nowrap;word-break:normal}").encode()


def _rel(depth: int, path: str) -> str:
    return html.escape("../" * depth + path, quote=True)


def _private_nav(depth: int) -> str:
    return ('<nav aria-label="비공개 리포트 이동">'
            f'<a href="{_rel(depth, "index.html")}">홈</a>'
            f'<a href="{_rel(depth, "archive/index.html")}">날짜별 목록</a></nav>')


def _private_page(title: str, body: str, depth: int, synthetic: bool) -> bytes:
    synthetic_banner = ('<div class="synthetic-banner">합성 fixture 예시 전용 — 실제 시장 데이터가 아닙니다.</div>'
                        if synthetic else "")
    return ("<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<meta name=\"robots\" content=\"noindex,nofollow\"><title>" + html.escape(PRIVATE_BANNER + " · " + title)
            + "</title><link rel=\"stylesheet\" href=\"" + _rel(depth, PRIVATE_CSS_PATH)
            + "\"></head><body><div class=\"private-banner\">" + html.escape(PRIVATE_BANNER)
            + "</div>" + synthetic_banner + _private_nav(depth) + "<main><h1>" + html.escape(title)
            + "</h1>" + body + "</main></body></html>").encode()


def _instant_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(SEOUL).strftime("%Y-%m-%d %H:%M:%S KST")


def _number_text(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "-"
    return html.escape(str(value))


def _private_opening_html(opening: dict[str, Any] | None) -> tuple[str, str]:
    opening = opening if isinstance(opening, dict) else {}
    status = str(opening.get("status", "unavailable"))
    verified = status == "ok"
    rows = []
    unverified_rows = 0
    for key, label in (("indices", "대표지수"), ("industries", "업종")):
        group = opening.get(key)
        for row in group if isinstance(group, list) else []:
            if not isinstance(row, dict):
                continue
            observed = _instant_text(row.get("observed_at"))
            if observed is None or not verified:
                unverified_rows += 1
            rows.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(label), html.escape(str(row.get("code", ""))), html.escape(str(row.get("name", ""))),
                _number_text(row.get("last")), _number_text(row.get("change_pct")),
                html.escape(observed) if observed is not None else "원천 시각 미확인"))
    summary_time = _instant_text(opening.get("observed_at"))
    head = (f"<p>상태: {html.escape(_status_text(status))} · 출처 시각(observed_at): "
            f"{html.escape(summary_time) if summary_time else '확인되지 않음'}</p>")
    if not verified:
        head += "<p class=\"hold\">장중 관측으로 확인되지 않았습니다. 아래 값은 참고용이며 장중 확인 값이 아닙니다.</p>"
    elif unverified_rows:
        head += "<p class=\"hold\">원천 시각이 없는 행은 장중 확인 값으로 보지 않습니다.</p>"
    table = ""
    if rows:
        table = ("<div class=\"table-scroll\"><table><thead><tr><th>종류</th><th>코드</th><th>이름</th><th>현재값</th>"
                 "<th>등락률</th><th>출처 시각</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
    return head, table


def _private_market_html(market: dict[str, Any]) -> str:
    title = html.escape(MODEL_DISPLAY_NAMES.get(market["model_id"], market["market"] + " 탐색 순위"))
    rankings = market["rankings"]
    is_kr = market["market"] == "KR"
    out = [f'<section id="{html.escape(market["model_id"])}"><h2>{title}<small>{html.escape(market["model_id"])}</small></h2>',
           "<p>상태: {} · 피쳐 기준일: {} · 추론 시작: {} · 공개 게이트: {}</p>".format(
               html.escape(_status_text(market["status"])),
               html.escape(market["feature_asof_date"] or "확인되지 않음"),
               html.escape(_instant_text(market.get("inference_started_at")) or "미기록"),
               html.escape(str(market["publication"]["status"])))]
    if not rankings:
        out.append("<p>이번 판에 순위가 없습니다 (전체 0개).</p></section>")
        return "".join(out)
    shown = rankings[:PRIVATE_LIST_LIMIT]
    out.append(f"<p>전체 {len(rankings)}개 중 상위 {len(shown)}개 · 점수는 {PRIVATE_SCORE_NOTE}입니다.</p>")
    body = []
    held = 0
    for row in shown:
        cells = [f"<td>{row['rank']}</td>", f"<td>{html.escape(row['symbol'])}</td>",
                 f"<td>{html.escape(row['name'])}</td>", f'<td class="num">{row["score"]:.4f}</td>']
        if is_kr:
            reasons = row.get("quality_reasons")
            review = row.get("quality_review")
            texts = [KR_QUALITY_REASON_TEXT.get(code, str(code)) for code in reasons] if isinstance(reasons, list) else []
            if review is True or texts:
                held += 1
                cells.append('<td class="hold">품질 보류: ' + html.escape(" / ".join(dict.fromkeys(texts)) or "사유 없음") + "</td>")
            elif review is False:
                cells.append("<td>이상 없음</td>")
            else:
                cells.append('<td class="hold">품질 정보 없음</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    headings = "<th>순위</th><th>코드</th><th>이름</th><th>점수</th>" + ("<th>품질</th>" if is_kr else "")
    if is_kr:
        out.append(f"<p>상위 {len(shown)}개 중 품질 보류 {held}개. 보류 행도 원래 순위에 남겼습니다.</p>")
    out.append("<div class=\"table-scroll\"><table><thead><tr>" + headings + "</tr></thead><tbody>"
               + "".join(body) + "</tbody></table></div></section>")
    return "".join(out)


class PrivateViewBuilder:
    """Render owner-only pages from full reports. Output is relative-linked and never public-safe."""

    write_atomic = staticmethod(write_tree_atomic)

    def __init__(self, *, synthetic_fixture: bool = False):
        self.synthetic_fixture = synthetic_fixture

    def render(self, *, report_date: date, reports: Iterable[dict[str, Any]],
               opening: dict[str, Any] | None = None,
               previous_files: dict[str, bytes] | None = None,
               historical_replay: bool = False) -> dict[str, bytes]:
        markets = []
        for report in reports:
            validate_report(report)
            if report["report_date"] != report_date.isoformat():
                raise ValueError("all market reports must match the requested date")
            markets.append(report)
        markets.sort(key=lambda item: (item["market"], item["model_id"]))
        if len({(x["market"], x["model_id"]) for x in markets}) != len(markets):
            raise ValueError("duplicate market/model report")
        date_str = report_date.isoformat()
        files = dict(previous_files or {})
        for existing in files:
            _safe_path(existing)
        prior = json.loads(files[PRIVATE_MANIFEST_NAME]) if PRIVATE_MANIFEST_NAME in files else {}
        if prior and (prior.get("private") is not True or
                      bool(prior.get("synthetic_fixture")) != self.synthetic_fixture):
            raise ValueError("previous private view differs in kind or fixture mode")
        rows_by_date: dict[str, dict[str, Any]] = {}
        for row in prior.get("reports", []):
            if DATE_PATTERN.fullmatch(str(row.get("report_date", ""))):
                rows_by_date[row["report_date"]] = row
        status = _date_status([{"market": m["market"], "model_id": m["model_id"], "status": m["status"],
                                "publication": {"status": "allowed"}} for m in markets],
                              {"status": "ok", "publication": "allowed"})
        rows_by_date[date_str] = {
            "report_date": date_str, "status": status,
            "synthetic_fixture": self.synthetic_fixture, "historical_replay": historical_replay,
            "markets": [{"market": m["market"], "model_id": m["model_id"], "status": m["status"],
                         "ranking_count": len(m["rankings"])} for m in markets]}
        all_dates = sorted(rows_by_date)
        latest = all_dates[-1]
        synthetic = self.synthetic_fixture
        opening_head, opening_table = _private_opening_html(opening)
        notes = ""
        if historical_replay:
            notes += "<p>과거 입력으로 다시 만든 historical_replay입니다.</p>"
        body = (f"<p>전체 상태: {html.escape(_status_text(status))}</p>{notes}"
                f'<section id="opening"><h2>한국장 관측</h2>{opening_head}{opening_table}</section>'
                + "".join(_private_market_html(m) for m in markets)
                + "<p>연구용 탐색 결과입니다. 비공개 화면이며 외부에 공유하지 않습니다.</p>")
        files[f"reports/{date_str}/index.html"] = _private_page(date_str + " 비공개 리포트", body, 2, synthetic)

        def item(day: str, depth: int) -> str:
            row = rows_by_date[day]
            mark = " · 합성" if row.get("synthetic_fixture") else ""
            return (f'<li><a href="{_rel(depth, "reports/" + day + "/index.html")}">{day}</a> · '
                    f'{html.escape(_status_text(str(row.get("status", ""))))}{mark}</li>')

        files["archive/index.html"] = _private_page(
            "날짜별 비공개 리포트", "<ul>" + "".join(item(d, 1) for d in reversed(all_dates)) + "</ul>", 1, synthetic)
        files["index.html"] = _private_page(
            "KR·US 비공개 리포트",
            f'<p>최근 날짜: <a href="{_rel(0, "reports/" + latest + "/index.html")}">{latest} 비공개 리포트</a></p>'
            f'<p>상태: {html.escape(_status_text(str(rows_by_date[latest]["status"])))}</p>'
            f'<p><a href="{_rel(0, "archive/index.html")}">날짜별 목록</a></p>', 0, synthetic)
        css = _private_css()
        files[PRIVATE_CSS_PATH] = css
        files[PRIVATE_MARKER_NAME] = PRIVATE_MARKER_TEXT.encode()
        files[PRIVATE_MANIFEST_NAME] = canonical_json({
            "schema_version": PRIVATE_SCHEMA, "private": True, "latest_report_date": latest,
            "reports": [rows_by_date[d] for d in all_dates], "synthetic_fixture": synthetic})
        return files
