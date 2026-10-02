"""Fail-closed public projection and deterministic static site generation."""
from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import shutil
import tempfile
from datetime import date, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from modeler.serving.schema import validate_report

SITE_SCHEMA = "1.0"
SAFE_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,100}$")
MODEL_CARD_FIELDS = ("title", "summary", "target", "scope", "limitations")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EXPECTED_MARKETS = {
    ("KR", "kr_daily_h20_v1"),
    ("US", "us_exploratory_20260929_r1_lightgbm"),
    ("US", "us_exploratory_20260929_r1_ridge"),
}
PUBLIC_RANKING_LIMIT = 100
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


def public_projection(report: dict[str, Any]) -> dict[str, Any]:
    """Expose ranking identities only after an explicit allowed publication decision."""
    validate_report(report)
    allowed = report["publication"]["status"] == "allowed"
    projected: dict[str, Any] = {
        "schema_version": report["schema_version"], "market": report["market"],
        "report_date": report["report_date"], "decision_at": report["decision_at"],
        "feature_asof_date": report["feature_asof_date"], "status": report["status"],
        "model_id": report["model_id"], "model_version": report["model_version"],
        "inference_started_at": report.get("inference_started_at"),
        "publication": {"status": report["publication"]["status"],
                        "evidence": _safe_evidence(report["publication"]["evidence"]) if allowed else []},
        "rankings": [], "quality": {"status": report["status"]},
        "provenance": {"model_id": report["model_id"], "model_version": report["model_version"]},
    }
    if allowed:
        projected["rankings"] = []
        for row in report["rankings"][:PUBLIC_RANKING_LIMIT]:
            public_row = {"rank": row["rank"], "symbol": row["symbol"], "name": row["name"]}
            if report["market"] == "KR":
                reasons = row.get("quality_reasons")
                safe_reasons = (
                    list(dict.fromkeys(KR_QUALITY_REASON_TEXT[code] for code in reasons
                                       if isinstance(code, str) and code in KR_QUALITY_REASON_TEXT))
                    if isinstance(reasons, list) else []
                )
                if not isinstance(reasons, list) or (row.get("quality_review") is not False and not safe_reasons):
                    safe_reasons = [KR_QUALITY_REASON_TEXT["K_price_quality_unavailable"]]
                public_row["quality_review"] = bool(safe_reasons) or row.get("quality_review") is not False
                public_row["quality_reasons"] = safe_reasons
            projected["rankings"].append(public_row)
        if report["market"] == "KR":
            projected["quality"] = {
                "status": report["status"],
                "D_management_and_halt_state": "현재 관리종목·거래정지 상태 미확인",
                "top100_quality_review_rows": sum(bool(row["quality_review"]) for row in projected["rankings"]),
            }
    else:
        # Do not publish input-level quality or provenance while the source gate is unresolved.
        projected["quality"] = {}
        projected["provenance"] = {}
    return projected


def _safe_evidence(items: list[Any]) -> list[dict[str, str]]:
    safe = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("id"), str) and SAFE_EVIDENCE_ID.fullmatch(item["id"]):
            safe.append({"id": item["id"]})
    return safe


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
    return ("body{font:16px/1.6 system-ui,sans-serif;max-width:70rem;margin:auto;padding:1rem;color:#18212b}"
            "h2,small{overflow-wrap:anywhere}h2 small{display:block;font-size:.75rem;font-weight:normal;color:#526273}"
            ".model-card p{overflow-wrap:anywhere;word-break:keep-all}"
            "table{border-collapse:collapse;display:block;overflow:auto}td,th{border:1px solid #aab4bf;"
            "padding:.4rem .7rem;text-align:left}.fixture{font-weight:bold;color:#8b2600}"
            "nav{display:flex;flex-wrap:wrap;gap:.3rem 1.2rem;padding-bottom:.6rem;margin-bottom:1rem;"
            "border-bottom:1px solid #aab4bf}h1,li{overflow-wrap:anywhere}"
            "li small{color:#526273;font-size:.75rem;margin-left:.4rem}").encode()


def _base(base_path: str, rel: str) -> str:
    return html.escape(base_path + rel, quote=True)


def _nav(base_path: str) -> str:
    return ('<nav aria-label="사이트 이동">'
            f'<a href="{_base(base_path, "index.html")}">홈</a>'
            f'<a href="{_base(base_path, "archive/index.html")}">날짜별 목록</a>'
            f'<a href="{_base(base_path, "models/index.html")}">모델 설명</a></nav>')


def _date_html(date_str: str, date_status: str, markets: list[dict[str, Any]],
               opening: dict[str, Any], base_path: str, css_path: str,
               historical_replay: bool,
               synthetic: bool,
               model_paths: dict[str, str] | None = None) -> bytes:
    model_paths = model_paths or {}
    sections = []
    seen_markets: set[str] = set()
    for market in markets:
        title = html.escape(MODEL_DISPLAY_NAMES.get(market["model_id"], market["market"] + " 탐색 순위"))
        model_identifier = html.escape(market["model_id"])
        status = html.escape(_status_text(market["status"]))
        asof = html.escape(market["feature_asof_date"] or "확인되지 않음")
        anchor = f' id="{market["market"].lower()}"' if market["market"] not in seen_markets else ""
        seen_markets.add(market["market"])
        inference = market.get("inference_started_at")
        inference_text = "미기록"
        if isinstance(inference, str):
            try:
                parsed = datetime.fromisoformat(inference)
                if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                    inference_text = parsed.astimezone(SEOUL).strftime("%Y-%m-%d %H:%M:%S KST")
            except ValueError:
                pass
        sections.append(f"<section{anchor}><h2>{title}<small>{model_identifier}</small></h2><p>상태: {status} · 피쳐 기준일: {asof} · "
                        f"추론 시작: {inference_text}</p>")
        if market["model_id"] in model_paths:
            sections.append(f'<p><a href="{_base(base_path, model_paths[market["model_id"]] + "/")}">모델 설명 보기</a></p>')
        if market["publication"]["status"] != "allowed":
            sections.append("<p>원천의 공개 조건을 확인하지 못해 이 시장의 종목 순위와 점수는 공개하지 않습니다.</p>")
        elif market["rankings"]:
            if market["market"] == "KR":
                sections.append("<p>현재 관리종목·거래정지 상태는 확인되지 않았습니다. 품질 검토가 필요한 행은 원래 순위에 그대로 남겼습니다.</p>")
                rows = "".join("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    row["rank"], html.escape(row["symbol"]), html.escape(row["name"]),
                    html.escape(" / ".join(row.get("quality_reasons", [])) if row.get("quality_review") else "K 가격 점검 이상 없음"))
                    for row in market["rankings"])
                headings = "<th>순위</th><th>코드</th><th>이름</th><th>품질 확인</th>"
            else:
                rows = "".join("<tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    row["rank"], html.escape(row["symbol"]), html.escape(row["name"]))
                    for row in market["rankings"])
                headings = "<th>순위</th><th>코드</th><th>이름</th>"
            sections.append("<p>기준 세션 종가로 만든 탐색 순위이며, 상위 100개까지만 공개합니다. 게시 시점에 그 종가로 거래할 수 있다는 뜻은 아닙니다.</p>"
                            "<table><thead><tr>" + headings + "</tr></thead><tbody>"
                            + rows + "</tbody></table>")
        else:
            sections.append("<p>이번 판에서 공개 가능한 순위가 없습니다.</p>")
        sections.append("</section>")
    opening_status = html.escape(_status_text(str(opening.get("status", "unavailable"))))
    fixture_text = '<p class="fixture">합성 fixture 예시 전용 — 실제 시장 데이터가 아닙니다.</p>' if synthetic else ""
    history_text = '<p>과거 입력으로 다시 만든 historical_replay입니다.</p>' if historical_replay else ""
    observation_rows = []
    for key, label in (("indices", "대표지수"), ("industries", "업종")):
        for row in opening.get(key, []):
            observation_rows.append("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                html.escape(label), html.escape(row.get("code", "")), html.escape(row.get("name", "")),
                html.escape(str(row.get("change_pct", "")))))
    observation_table = ("<table><thead><tr><th>종류</th><th>코드</th><th>이름</th><th>등락률</th></tr></thead><tbody>"
                         + "".join(observation_rows) + "</tbody></table>" if observation_rows else "")
    body = (f"<header>{_nav(base_path)}</header>"
            f"<main><h1>{html.escape(date_str)} 리포트</h1><p>전체 상태: {html.escape(_status_text(date_status))}</p>"
            f"{history_text}{fixture_text}<section id=\"opening\"><h2>한국장 관측</h2>"
            f"<p>상태: {opening_status}</p>{observation_table}</section>{''.join(sections)}"
            "<p>연구용 탐색 결과이며 투자 권유가 아닙니다. 실현 수익률은 표시하지 않습니다.</p></main>")
    return ("<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<meta name=\"robots\" content=\"noindex,nofollow\"><title>"
            + html.escape(date_str + " 시장 리포트") + "</title><link rel=\"stylesheet\" href=\""
            + _base(base_path, css_path) + "\"></head><body>" + body + "</body></html>").encode()


def _shell(title: str, body: str, base_path: str, css_path: str,
           heading: str | None = None) -> bytes:
    return ("<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<meta name=\"robots\" content=\"noindex,nofollow\"><title>" + html.escape(title)
            + "</title><link rel=\"stylesheet\" href=\"" + _base(base_path, css_path)
            + "\"></head><body>" + _nav(base_path) + "<main><h1>" + html.escape(heading or title) + "</h1>" + body
            + "</main></body></html>").encode()


def _public_opening(opening: dict[str, Any] | None) -> dict[str, Any]:
    if not opening:
        return {"status": "unavailable", "publication": "unresolved"}
    gate = opening.get("publication")
    gate_status = gate.get("status", "unresolved") if isinstance(gate, dict) else "unresolved"
    if gate_status not in {"unresolved", "allowed", "withheld"}:
        gate_status = "unresolved"
    opening_status = opening.get("status", "unavailable")
    if opening_status not in {"ok", "partial", "stale", "failed", "unavailable"}:
        opening_status = "unavailable"
    result: dict[str, Any] = {"status": opening_status, "publication": gate_status}
    if gate_status != "allowed":
        return result
    observed_at = opening.get("observed_at")
    if isinstance(observed_at, str):
        try:
            parsed = datetime.fromisoformat(observed_at)
            if parsed.tzinfo is not None and parsed.utcoffset() is not None:
                result["observed_at"] = parsed.isoformat()
        except ValueError:
            pass
    session_state = opening.get("session_state")
    if isinstance(session_state, str) and session_state in {"regular", "delayed_open", "closed", "halted", "pending_open", "unavailable"}:
        result["session_state"] = session_state
    for field in ("indices", "industries"):
        rows = opening.get(field, [])
        if isinstance(rows, list):
            result[field] = [_public_opening_row(field, row) for row in rows
                             if isinstance(row, dict) and isinstance(row.get("publication"), dict)
                             and row["publication"].get("status") == "allowed"]
    return result


def _public_opening_row(field: str, row: dict[str, Any]) -> dict[str, Any]:
    allowed = {"code", "name", "observed_at", "rank", "open", "previous_close", "last", "change_pct"}
    safe: dict[str, Any] = {}
    for key in allowed:
        value = row.get(key)
        if key in {"code", "name"} and isinstance(value, str):
            safe[key] = value
        elif key == "observed_at" and isinstance(value, str):
            try:
                instant = datetime.fromisoformat(value)
                if instant.tzinfo is not None and instant.utcoffset() is not None:
                    safe[key] = instant.isoformat()
            except ValueError:
                continue
        elif key == "rank" and field == "industries" and isinstance(value, int) and not isinstance(value, bool) and value > 0:
            safe[key] = value
        elif key in {"open", "previous_close", "last", "change_pct"} and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            safe[key] = value
    return safe


def _public_model_card(model_id: str, card: dict[str, Any]) -> dict[str, Any]:
    gate = card.get("publication", {})
    status = gate.get("status", "unresolved") if isinstance(gate, dict) else "unresolved"
    if status not in {"unresolved", "allowed", "withheld"}:
        status = "unresolved"
    public = {"model_id": model_id, "publication": {"status": status, "evidence": []}}
    if status == "allowed":
        public.update({key: card[key] for key in MODEL_CARD_FIELDS if isinstance(card.get(key), str)})
    return public


def _model_card_html(card: dict[str, Any]) -> str:
    model_id = html.escape(card["model_id"])
    if card["publication"]["status"] != "allowed":
        return (f'<article class="model-card"><h2>모델 설명<small>{model_id}</small></h2>'
                '<p>모델 설명 공개를 보류합니다.</p></article>')
    title = html.escape(card.get("title", "모델 설명"))
    sections = []
    for key, label in (("summary", "소개"), ("target", "예측 대상"),
                       ("scope", "학습·적용 범위"), ("limitations", "주의할 점")):
        if key in card:
            sections.append(f"<section><h3>{label}</h3><p>{html.escape(card[key])}</p></section>")
    return (f'<article class="model-card"><h2>{title}<small>{model_id}</small></h2>'
            + "".join(sections) + "</article>")


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


class SiteBuilder:
    def __init__(self, *, base_path: str = "/market-briefing/", synthetic_fixture: bool = False):
        if (not base_path.startswith("/") or base_path.startswith("//") or
                not base_path.endswith("/") or not re.fullmatch(r"/[A-Za-z0-9._/-]+/", base_path) or
                any(part in {".", ".."} for part in PurePosixPath(base_path).parts)):
            raise ValueError("base_path must be an absolute path without traversal")
        self.base_path = base_path
        self.synthetic_fixture = synthetic_fixture

    def render(self, *, report_date: date, reports: Iterable[dict[str, Any]],
               opening: dict[str, Any] | None = None,
               model_cards: dict[str, dict[str, Any]] | None = None,
               previous_files: dict[str, bytes] | None = None,
               historical_replay: bool = False,
               inference_started_at: datetime | None = None) -> dict[str, bytes]:
        reports = list(reports)
        public_markets = []
        for report in reports:
            validate_report(report)
            if report["report_date"] != report_date.isoformat():
                raise ValueError("all market reports must match the requested date")
            public_markets.append(public_projection(report))
        public_markets.sort(key=lambda item: (item["market"], item["model_id"]))
        if len({(x["market"], x["model_id"]) for x in public_markets}) != len(public_markets):
            raise ValueError("duplicate market/model report")
        if len({x["decision_at"] for x in public_markets}) > 1:
            raise ValueError("market reports have different decision cutoffs")
        public_opening = _public_opening(opening)
        status = _date_status(public_markets, public_opening)
        date_str = report_date.isoformat()
        if inference_started_at is not None and (inference_started_at.tzinfo is None or
                                                  inference_started_at.utcoffset() is None):
            raise ValueError("inference_started_at must include a timezone")
        css_bytes = _css()
        css_path = "assets/site-" + hashlib.sha256(css_bytes).hexdigest()[:16] + ".css"
        card_artifacts = {}
        for model_id, card in sorted((model_cards or {}).items()):
            if not re.fullmatch(r"[A-Za-z0-9._-]+", model_id):
                raise ValueError("unsafe model id")
            projected_card = _public_model_card(model_id, card)
            card_bytes = canonical_json(projected_card)
            card_body = _model_card_html(projected_card)
            page_title = "모델 설명"
            if isinstance(projected_card.get("title"), str) and projected_card["title"]:
                page_title = projected_card["title"] + " · 모델 설명"
            card_html = _shell(page_title, card_body, self.base_path, css_path, heading="모델 설명")
            card_hash = hashlib.sha256(card_bytes + card_html + css_bytes).hexdigest()[:16]
            card_artifacts[model_id] = {"hash": card_hash, "json": card_bytes, "html": card_html,
                                        "title": projected_card.get("title")}
        payload = {"schema_version": SITE_SCHEMA, "report_date": date_str,
                   "decision_at": public_markets[0]["decision_at"] if public_markets else None,
                   "status": status, "markets": public_markets, "opening": public_opening,
                   "model_cards": {model_id: item["hash"] for model_id, item in card_artifacts.items()},
                   "synthetic_fixture": self.synthetic_fixture,
                   "historical_replay": historical_replay,
                   "inference_started_at": inference_started_at.isoformat() if inference_started_at else None}
        payload_bytes = canonical_json(payload)
        model_paths = {mid: f"models/{mid}/{item['hash']}" for mid, item in card_artifacts.items()}
        html_bytes = _date_html(date_str, status, public_markets, public_opening, self.base_path,
                                css_path, historical_replay, self.synthetic_fixture, model_paths)
        digest = hashlib.sha256(payload_bytes + html_bytes + css_bytes).hexdigest()
        revision = digest[:16]
        files = dict(previous_files or {})
        for existing_path in files:
            _safe_path(existing_path)
        reports_by_date: dict[str, dict[str, Any]] = {}
        prior = json.loads(files["site-manifest.json"]) if "site-manifest.json" in files else {}
        for row in prior.get("reports", []):
            if DATE_PATTERN.fullmatch(row.get("report_date", "")):
                reports_by_date[row["report_date"]] = row
        records = [{"market": item["market"], "model_id": item["model_id"],
                    "status": item["status"], "publication": item["publication"]["status"]}
                   for item in public_markets]
        reports_by_date[date_str] = {"report_date": date_str, "status": status,
                                     "markets": records, "revision": revision,
                                     "url": self.base_path + f"reports/{date_str}/revisions/{revision}/",
                                     "synthetic_fixture": self.synthetic_fixture,
                                     "historical_replay": historical_replay}
        all_dates = sorted(reports_by_date)
        latest_date = all_dates[-1]
        base = f"reports/{date_str}"
        files[_safe_path(f"{base}/revisions/{revision}/report.json")] = payload_bytes
        files[_safe_path(f"{base}/revisions/{revision}/index.html")] = html_bytes
        files[_safe_path(f"{base}/report.json")] = payload_bytes
        files[_safe_path(f"{base}/index.html")] = html_bytes
        # Build the full archive from manifest rows; no historical body is rewritten.
        def day_item(day: str) -> str:
            row = reports_by_date[day]
            mark = " · 합성" if row.get("synthetic_fixture") else ""
            return (f'<li><a href="{_base(self.base_path, "reports/" + day + "/")}">{day}</a> · '
                    f'{html.escape(_status_text(str(row.get("status", ""))))}{mark}</li>')

        links = "".join(day_item(day) for day in reversed(all_dates))
        archive = _shell("날짜별 리포트", "<ul>" + links + "</ul>", self.base_path, css_path)
        files["archive/index.html"] = archive
        for month in sorted({day[:7] for day in all_dates}):
            month_links = "".join(day_item(day) for day in reversed(all_dates) if day[:7] == month)
            files[f"archive/{month}/index.html"] = _shell(month + " 리포트", "<ul>" + month_links + "</ul>", self.base_path, css_path)
        model_links = []
        for model_id, artifact in card_artifacts.items():
            card_hash = artifact["hash"]
            model_path = f"models/{model_id}/{card_hash}"
            files[f"{model_path}/model-card.json"] = artifact["json"]
            files[f"{model_path}/index.html"] = artifact["html"]
            card_title = artifact.get("title")
            if isinstance(card_title, str) and card_title:
                label = html.escape(card_title)
                extra = f"<small>{html.escape(model_id)}</small>"
            else:
                label, extra = html.escape(model_id), ""
            model_links.append(f'<li><a href="{_base(self.base_path, model_path + "/")}">{label}</a>{extra}</li>')
        manifest = {"schema_version": SITE_SCHEMA, "latest_report_date": latest_date,
                    "reports": [reports_by_date[day] for day in all_dates],
                    "synthetic_fixture": self.synthetic_fixture}
        files["site-manifest.json"] = canonical_json(manifest)
        latest_url = _base(self.base_path, f"reports/{latest_date}/")
        home_body = (f"<p>최신 발행일: <a href=\"{latest_url}\">{latest_date} 리포트</a></p>"
                     f"<p>최신 리포트 상태: {html.escape(_status_text(reports_by_date[latest_date]['status']))}</p>"
                     "<ul>"
                     + "".join(model_links) + "</ul>")
        if self.synthetic_fixture:
            home_body += '<p class="fixture">합성 fixture 예시 전용 — 실제 시장 데이터가 아닙니다.</p>'
        files["index.html"] = _shell("KR·US 시장 리포트", home_body, self.base_path, css_path)
        files["models/index.html"] = _shell("모델 설명", "<ul>" + "".join(model_links) + "</ul>", self.base_path, css_path)
        files[css_path] = css_bytes
        return files

    def write_atomic(self, target: Path, files: dict[str, bytes]) -> None:
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
# Private view: owner-only pages with rankings and scores. Never part of the public projection.
# ---------------------------------------------------------------------------
PRIVATE_SCHEMA = "private-1.0"
PRIVATE_MARKER_NAME = "PRIVATE_DO_NOT_PUBLISH.txt"
PRIVATE_MANIFEST_NAME = "private-manifest.json"
PRIVATE_CSS_PATH = "assets/private.css"
PRIVATE_BANNER = "비공개 — 게시 금지"
PRIVATE_MARKER_TEXT = ("PRIVATE - DO NOT PUBLISH\n"
                       "비공개 — 게시 금지. 이 디렉터리에는 순위·점수·시세가 들어 있습니다.\n"
                       "GitHub Pages·저장소·공개 저장소로 복사하거나 올리지 않습니다.\n")
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

    write_atomic = SiteBuilder.write_atomic

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
