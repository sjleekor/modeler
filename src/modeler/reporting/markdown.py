"""일일 브리핑 envelope를 `stock_reports` markdown 저장소 형식으로 바꿉니다.

내부 report(`runs/D/report-D.json`)를 단위 폴더(README.md와 섹션 파일 넷)로 렌더하고,
저장소 인덱스(자동 구간)를 다시 만들고, 저장소 트리를 검증합니다. 표준 라이브러리만 씁니다.
템플릿과 문구는 전부 이 파일의 문자열입니다. release가 `.py` 말고는 복사하지 않기 때문입니다.

원격에 접속하는 일은 여기 없습니다. 접속은 `deploy/reports/publish_reports.py`가 합니다.

수동 사용 예 (`python -m modeler.reporting.markdown`)
    # 저장소 뼈대(CONVENTIONS, reference/, 인덱스)를 만듭니다. 이미 있는 파일은 덮지 않습니다.
    --repo ./stock_reports --init --model-cards model-cards.json

    # 단위 하나를 만들고 인덱스를 다시 씁니다.
    --repo ./stock_reports --release r20261005 \
        --envelope report-2026-10-07.json --market-sector ms-2026-10-07.json

    # 검증만 합니다. 위반이 있으면 종료 코드 1입니다.
    --repo ./stock_reports --validate

종료 코드: 0 정상, 1 검증 위반, 2 입력·사용법 오류.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import posixpath
import re
import sys
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

KST = timezone(timedelta(hours=9))
SCHEMA = "stock-reports.v1"
FAMILY = "daily-briefing"
UNIT_FORMAT = "YYYY/MM/YYYY-MM-DD/"
LAG_SUPPRESS = 5  # 04 문서 Q5: 이 값을 넘으면 순위 표 대신 사유를 적습니다
LAG_WARN_US = 3  # 04 문서 변경 4: US는 3세션 이상이면 경고 배너
MAX_FILE_BYTES = 1024 * 1024
DEFAULT_TOP_N = 100

KR_MODEL = "kr_daily_h20_v1"
US_MODELS = ("us_exploratory_20260929_r1_lightgbm", "us_exploratory_20260929_r1_ridge")
MS_MODEL = "ms1_market_sector"

# site.py(MODEL_DISPLAY_NAMES, KR_QUALITY_REASON_TEXT)의 문구를 그대로 가져왔습니다.
MODEL_DISPLAY_NAMES = {
    "kr_daily_h20_v1": "한국 종목 탐색 순위",
    "us_exploratory_20260929_r1_lightgbm": "미국 종목 탐색 순위 · LightGBM",
    "us_exploratory_20260929_r1_ridge": "미국 종목 탐색 순위 · Ridge",
}
US_SHORT = {
    "us_exploratory_20260929_r1_lightgbm": "LightGBM",
    "us_exploratory_20260929_r1_ridge": "Ridge",
}
KR_QUALITY_REASON_TEXT = {
    "K_halt_or_unknown": "K 기준 거래정지 또는 상태 미확인",
    "K_price_jump_or_unknown": "K 기준 가격 급변 또는 확인 불가",
    "K_share_change_or_unknown": "K 기준 상장주식수 변화 또는 확인 불가",
    "K_price_rule_unknown": "K 기준 가격 점검 규칙 적용 불가",
    "K_price_quality_unavailable": "K 기준 가격 품질 정보 없음",
}
# 내부 report 상태 -> front matter status (01 문서 §6, 지시서의 매핑)
STATUS_MAP = {
    "ok": "ok",
    "partial": "partial",
    "stale": "stale",
    "withheld": "failed",
    "unavailable": "failed",
    "failed": "failed",
}
STATUS_KO = {"ok": "정상", "partial": "부분 완료", "stale": "자료 지연", "failed": "실패"}
INTERNAL_KO = {
    "ok": "정상",
    "partial": "부분 완료",
    "stale": "자료 지연",
    "withheld": "공개 보류",
    "failed": "실패",
    "unavailable": "자료 없음",
}
REASON_KO = {
    "verified input availability time is missing": "입력 가용 시각이 기록되지 않았습니다",
    "verified input availability is after cutoff; do not infer": (
        "입력이 D 09:30 cutoff 뒤에 준비되어 쓰지 않았습니다"
    ),
    "prepared-feature completion evidence is missing or unsupported": (
        "입력 완료 증거가 없거나 형식이 맞지 않습니다"
    ),
    "KR feature session does not match K": "KR 기준일이 K와 다릅니다",
    "K is not a confirmed, completed previous KR session": "K가 확정된 직전 KR 세션이 아닙니다",
    "Korean calendar coverage is unknown": "KR 달력 범위를 알 수 없습니다",
    "previous Korean session is outside calendar coverage": "직전 KR 세션이 달력 범위 밖입니다",
    "report date is not a Korean session": "리포트 날짜가 KR 휴장일입니다",
    "US U, E, or A session is missing": "미국 U·E·A 세션 중 빠진 값이 있습니다",
    "US calendar does not cover U, E, and A": "미국 달력이 U·E·A 세션을 덮지 못합니다",
    "US actual session exceeds market-lag ceiling": "미국 기준 세션이 지연 한도를 넘었습니다",
    "US delivery lag reached the stop limit": "미국 도착 지연이 중단 한도에 닿았습니다",
    "US data arrived one session after E": "미국 데이터가 기대 세션보다 한 세션 늦게 도착했습니다",
    "decision_at is not D 10:00 Asia/Seoul": "판단 시각이 D 10:00 KST가 아닙니다",
    "input_cutoff must be D 09:30 Asia/Seoul": "입력 cutoff가 D 09:30 KST가 아닙니다",
}

SECTION_FILES = {
    "summary": "README.md",
    "market-sector": "market-sector.md",
    "kr-stocks": "kr-stocks.md",
    "us-stocks": "us-stocks.md",
    "data-status": "data-status.md",
}
FILE_SECTIONS = {v: k for k, v in SECTION_FILES.items()}
UNIT_FM_FIELDS = (
    "schema",
    "family",
    "unit",
    "section",
    "title",
    "status",
    "markets",
    "decision_at",
    "generated_at",
    "data_asof",
    "models",
    "revision",
    "source",
)
AUTO_BEGIN = "<!-- auto:begin index -->"
AUTO_END = "<!-- auto:end index -->"
CORR_BEGIN = "<!-- correction:begin -->"
CORR_END = "<!-- correction:end -->"
REPLAY_NOTICE = (
    "재현 리포트입니다. D 09:30 뒤에 늦게 들어온 데이터가 섞일 수 있어 "
    "그날 실제로 냈을 결과와 다를 수 있습니다."
)
SYNTHETIC_NOTICE = "합성 fixture 예시 전용 — 실제 시장 데이터가 아닙니다."
DISCLAIMER = "연구용 탐색 결과입니다. 투자 권유가 아니며 실현 수익률은 표시하지 않습니다."

FORBIDDEN_PATTERNS = (
    ("서버 절대경로 /home/", re.compile(r"/home/")),
    ("서버 절대경로 /Users/", re.compile(r"/Users/")),
    ("서버 절대경로 /private/", re.compile(r"/private/")),
    ("호스트명 sj2", re.compile(r"sj2", re.IGNORECASE)),
    ("키 모양 문자열 gh*_", re.compile(r"gh[pousr]_")),
    ("키 모양 문자열 PEM", re.compile(r"-----BEGIN")),
    ("키 모양 문자열 api_key", re.compile(r"api_key", re.IGNORECASE)),
    ("키 모양 문자열 github_pat_", re.compile(r"github_pat_")),
    ("키 모양 문자열 AKIA", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("서버 절대경로 /srv/", re.compile(r"/srv/")),
    ("서버 절대경로 /tmp/", re.compile(r"/tmp/")),
)
SCRUB_PATH = re.compile(r"/(?:home|Users|private|srv|tmp)/\S*")
SCRUB_TOKENS = (
    re.compile(r"sj2\S*", re.IGNORECASE),
    re.compile(r"gh[pousr]_\w*"),
    re.compile(r"github_pat_\w*"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN[^\n]*"),
    re.compile(r"api_key\w*", re.IGNORECASE),
)


class InputError(Exception):
    """입력 형식이 잘못됐습니다. 종료 코드 2."""


# ---------------------------------------------------------------------------
# 문자열·숫자 도우미
# ---------------------------------------------------------------------------
SCRUB_HITS: list[str] = []


def reset_scrub_hits() -> None:
    """이번 렌더에서 지운 문자열 기록을 비웁니다."""
    SCRUB_HITS.clear()


def scrub(text: str) -> str:
    """서버 경로·호스트명·키 모양 문자열이 본문에 새지 않게 지웁니다."""
    out = SCRUB_PATH.sub("(경로 생략)", text)
    for pattern in SCRUB_TOKENS:
        out = pattern.sub("(생략)", out)
    if out != text:
        SCRUB_HITS.append(text[:40])
    return out


def c(value: object) -> str:
    """표 셀·본문에 넣는 데이터 문자열을 이스케이프합니다(`|`, 줄바꿈, 링크·HTML을 만드는 문자)."""
    text = scrub(str(value))
    text = text.replace("\\", "\\\\")
    text = re.sub(r"([|`*\[\]<>])", r"\\\1", text)
    text = re.sub(r"\r\n|\r|\n", "<br>", text).strip()
    return text if text else "-"


def code(value: object) -> str:
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9._:\-]+", text):
        return "`" + text + "`"
    return c(text)


def is_num(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def pct(value: object, digits: int = 2, signed: bool = True) -> str:
    """비율(0.0012)을 %로 적습니다."""
    if not is_num(value):
        return "-"
    spec = "{:+." + str(digits) + "f}%" if signed else "{:." + str(digits) + "f}%"
    return spec.format(float(value) * 100.0)


def pct_plain(value: object, digits: int = 2) -> str:
    """이미 %인 값을 적습니다."""
    if not is_num(value):
        return "-"
    return ("{:." + str(digits) + "f}%").format(float(value))


def num(value: object, digits: int = 1) -> str:
    if not is_num(value):
        return "-"
    return ("{:." + str(digits) + "f}").format(float(value))


def parse_dt(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def kst_text(value: object) -> str:
    parsed = parse_dt(value)
    return parsed.astimezone(KST).strftime("%Y-%m-%d %H:%M:%S KST") if parsed else "미기록"


def kst_minute(value: object) -> str:
    parsed = parse_dt(value)
    return parsed.astimezone(KST).strftime("%Y-%m-%d %H:%M KST") if parsed else "미기록"


ASOF_LABELS = (
    ("us_prices", "US 가격"),
    ("us_macro", "US 거시"),
    ("kr_index", "KR 지수"),
    ("kr_features", "KR 입력"),
    ("us_features", "US 입력"),
)


def asof_text(data_asof: dict, keys: tuple | None = None) -> str:
    parts = [
        f"{label} {data_asof[key]}"
        for key, label in ASOF_LABELS
        if key in data_asof and (keys is None or key in keys)
    ]
    return " · ".join(parts) if parts else "-"


def good_date(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        try:
            date.fromisoformat(value)
            return value
        except ValueError:
            return None
    return None


def md_table(headers: list, rows: list, right: tuple = ()) -> list:
    sep = ["---:" if i in right else "---" for i in range(len(headers))]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def escape_tilde(text: str) -> str:
    """범위·근사값 물결표를 `\\~`로 바꿉니다. 코드 블록과 인라인 코드는 건드리지 않습니다."""
    out = []
    in_fence = False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence:
            out.append(line)
            continue
        parts = re.split(r"(`+[^`]*`+)", line)
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(r"(?<!\\)~", r"\\~", parts[i])
        out.append("".join(parts))
    return "\n".join(out)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_hex64(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


# ---------------------------------------------------------------------------
# front matter (이 스크립트가 쓰는 YAML 부분집합만 읽고 씁니다)
# ---------------------------------------------------------------------------
def yaml_scalar(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if re.fullmatch(r"[A-Za-z0-9_./+:\-]+", text) and text.lower() not in (
        "null",
        "true",
        "false",
        "yes",
        "no",
        "~",
    ):
        return text
    return json.dumps(text, ensure_ascii=False)


def yaml_value(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(yaml_scalar(v) for v in value) + "]"
    return yaml_scalar(value)


def front_matter(meta: dict) -> str:
    lines = ["---"]
    for key, value in meta.items():
        if isinstance(value, dict):
            if not value:
                lines.append(key + ": {}")
            else:
                lines.append(key + ":")
                for sub_key, sub_value in value.items():
                    lines.append("  " + sub_key + ": " + yaml_value(sub_value))
        else:
            lines.append(key + ": " + yaml_value(value))
    lines.append("---")
    return "\n".join(lines) + "\n"


def _parse_scalar(raw: str) -> object:
    raw = raw.strip()
    if raw == "":
        return ""
    if raw.startswith('"'):
        return json.loads(raw)
    if raw.startswith("["):
        if not raw.endswith("]"):
            raise ValueError("list is not closed")
        inner = raw[1:-1].strip()
        if not inner:
            return []
        items, buf, quoted = [], "", False
        for ch in inner:
            if ch == '"':
                quoted = not quoted
            if ch == "," and not quoted:
                items.append(buf)
                buf = ""
            else:
                buf += ch
        items.append(buf)
        return [_parse_scalar(x) for x in items]
    if raw == "{}":
        return {}
    if raw in ("null", "~"):
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


def split_front_matter(text: str) -> tuple[str | None, str]:
    """(front matter 원문 | None, 본문)을 돌려줍니다."""
    if not text.startswith("---\n"):
        return None, text
    end = text.find("\n---\n", 3)
    if end == -1:
        if text.endswith("\n---"):
            return text[4:-4], ""
        return None, text
    return text[4:end], text[end + 5 :]


def parse_front_matter(text: str) -> tuple[dict | None, str]:
    raw, body = split_front_matter(text)
    if raw is None:
        return None, text
    meta: dict = {}
    current: str | None = None
    for line in raw.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError("front matter line without colon: " + stripped[:40])
        key, _, value = stripped.partition(":")
        if indent == 0:
            if value.strip() == "":
                meta[key] = {}
                current = key
            else:
                meta[key] = _parse_scalar(value)
                current = None
        else:
            if current is None or not isinstance(meta.get(current), dict):
                raise ValueError("unexpected indentation in front matter")
            meta[current][key] = _parse_scalar(value)
    return meta, body


def read_fm(path: Path) -> dict | None:
    try:
        meta, _ = parse_front_matter(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return meta


# ---------------------------------------------------------------------------
# 입력 읽기와 검증
# ---------------------------------------------------------------------------
REPORT_STATES = {"ok", "partial", "stale", "withheld", "failed", "unavailable"}


def validate_report_min(report: dict) -> None:
    """modeler.serving.schema.validate_report와 같은 검사를 표준 라이브러리만으로 합니다."""
    required = {
        "schema_version",
        "market",
        "report_date",
        "decision_at",
        "feature_asof_date",
        "status",
        "model_id",
        "model_version",
        "rankings",
        "quality",
        "provenance",
        "publication",
    }
    if not isinstance(report, dict):
        raise ValueError("report is not an object")
    missing = required - report.keys()
    if missing:
        raise ValueError("missing report fields: " + ", ".join(sorted(missing)))
    if report["schema_version"] != "1.0":
        raise ValueError("unsupported schema_version")
    if report["market"] not in {"KR", "US"}:
        raise ValueError("market must be KR or US")
    day = date.fromisoformat(report["report_date"])
    if day.isoformat() != report["report_date"]:
        raise ValueError("report_date must use YYYY-MM-DD")
    instant = parse_dt(report["decision_at"])
    if instant is None:
        raise ValueError("decision_at must include a timezone")
    local = instant.astimezone(KST)
    if local.date() != day or (local.hour, local.minute, local.second) != (10, 0, 0):
        raise ValueError("decision_at must be report_date at 10:00 Asia/Seoul")
    if report["feature_asof_date"] is not None and good_date(report["feature_asof_date"]) is None:
        raise ValueError("feature_asof_date must use YYYY-MM-DD")
    if report["status"] not in REPORT_STATES:
        raise ValueError("unknown report status")
    for field in ("model_id", "model_version"):
        if not isinstance(report[field], str) or not report[field]:
            raise ValueError(field + " is required")
    if not isinstance(report["rankings"], list):
        raise ValueError("rankings must be a list")
    last_rank, seen = 0, set()
    for row in report["rankings"]:
        if not isinstance(row, dict) or not {"rank", "symbol", "name", "score"} <= row.keys():
            raise ValueError("each ranking requires rank, symbol, name, and score")
        if (
            isinstance(row["rank"], bool)
            or not isinstance(row["rank"], int)
            or row["rank"] <= last_rank
        ):
            raise ValueError("ranking ranks must be positive and strictly increasing")
        if not isinstance(row["symbol"], str) or not row["symbol"] or row["symbol"] in seen:
            raise ValueError("ranking symbol is required and unique")
        if not isinstance(row["name"], str):
            raise ValueError("ranking name must be a string")
        if not is_num(row["score"]):
            raise ValueError("ranking score must be a finite number")
        seen.add(row["symbol"])
        last_rank = row["rank"]
    for field in ("quality", "provenance"):
        if not isinstance(report[field], dict):
            raise ValueError(field + " must be an object")
    publication = report["publication"]
    if not isinstance(publication, dict) or publication.get("status") not in {
        "unresolved",
        "allowed",
        "withheld",
    }:
        raise ValueError("publication.status must be unresolved, allowed, or withheld")
    if not isinstance(publication.get("evidence"), list):
        raise ValueError("publication.evidence must be a list")


def read_bytes(path: Path, what: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise InputError(
            f"{what}을(를) 읽지 못했습니다: {path.name} ({type(exc).__name__})"
        ) from None


def load_json_bytes(raw: bytes, what: str) -> dict:
    try:
        data = json.loads(raw.decode("utf-8"))
    except ValueError as exc:  # UnicodeDecodeError와 JSONDecodeError는 ValueError입니다.
        raise InputError(
            f"{what}을(를) 읽지 못했습니다: JSON이 아닙니다 ({type(exc).__name__})"
        ) from None
    if not isinstance(data, dict):
        raise InputError(f"{what}의 최상위가 객체가 아닙니다.")
    return data


def load_json_file(path: Path, what: str) -> dict:
    return load_json_bytes(read_bytes(path, what), what)


def _safe_str(value: object) -> str:
    return value if isinstance(value, str) else ""


def build_model_entry(
    market: str, model_id: str, report: dict | None, failures: list, top_n: int
) -> dict:
    entry = {
        "market": market,
        "model_id": model_id,
        "report": report,
        "reason": [],
        "rows": [],
        "total": 0,
        "lag": None,
        "asof": None,
        "suppressed": False,
        "fresh": {},
        "quality": {},
        "prov": {},
    }
    matched = [
        f
        for f in failures
        if f.get("market") == market and f.get("model_id") in (model_id, None, "")
    ]
    if report is None:
        entry["internal"] = "failed" if matched else "unavailable"
        entry["status"] = "failed"
        if matched:
            classes = sorted({_safe_str(f.get("error_class")) or "알 수 없음" for f in matched})
            entry["reason"].append(
                "추론이 실패했습니다. 원인 클래스: " + ", ".join(code(x) for x in classes)
            )
        else:
            entry["reason"].append("입력 없음: 이 모델의 report가 envelope에 없습니다.")
        return entry
    entry["internal"] = report["status"]
    entry["status"] = STATUS_MAP[report["status"]]
    entry["asof"] = report.get("feature_asof_date")
    entry["quality"] = report.get("quality") or {}
    entry["prov"] = report.get("provenance") or {}
    fresh = entry["prov"].get("freshness")
    entry["fresh"] = fresh if isinstance(fresh, dict) else {}
    entry["total"] = len(report["rankings"])
    entry["rows"] = report["rankings"][:top_n]
    q, f = entry["quality"], entry["fresh"]
    if market == "US":
        candidates = (
            q.get("market_lag_sessions"),
            f.get("market_lag"),
            f.get("market_lag_sessions"),
        )
    else:
        candidates = (f.get("delivery_lag"), f.get("market_lag"), q.get("lag_sessions"))
    lag = next((int(x) for x in candidates if is_num(x)), None)
    if lag is None and market == "KR" and report["status"] == "ok":
        lag = 0
    entry["lag"] = lag
    if entry["status"] == "failed":
        raw = _safe_str(f.get("reason"))
        if raw:
            entry["reason"].append(REASON_KO.get(raw, "원인 메시지 " + code(raw)))
        if matched:
            classes = sorted({_safe_str(x.get("error_class")) or "알 수 없음" for x in matched})
            entry["reason"].append("원인 클래스: " + ", ".join(code(x) for x in classes))
        if not entry["reason"]:
            entry["reason"].append(
                "내부 상태가 {}이라 순위를 내지 않습니다.".format(
                    INTERNAL_KO.get(report["status"], report["status"])
                )
            )
    elif lag is not None and lag > LAG_SUPPRESS:
        entry["suppressed"] = True
    return entry


def aggregate_status(statuses: list) -> str:
    if not statuses or all(s == "failed" for s in statuses):
        return "failed"
    if all(s == statuses[0] for s in statuses):
        return statuses[0]
    if any(s in ("failed", "partial") for s in statuses):
        return "partial"
    return "stale"


def build_market_sections(
    env: dict, reports: list, failures: list, top_n: int
) -> tuple[dict, dict]:
    kr_reports = {r["model_id"]: r for r in reports if r["market"] == "KR"}
    us_reports = {r["model_id"]: r for r in reports if r["market"] == "US"}
    kr_ids = [KR_MODEL] + sorted(set(kr_reports) - {KR_MODEL})
    us_ids = list(US_MODELS) + sorted(set(us_reports) - set(US_MODELS))
    kr_models = [build_model_entry("KR", m, kr_reports.get(m), failures, top_n) for m in kr_ids]
    us_models = [build_model_entry("US", m, us_reports.get(m), failures, top_n) for m in us_ids]
    # 기대 모델이 없고 다른 모델도 없을 때만 기대 모델 행을 남깁니다.
    # 있으면 그 행이 이미 들어 있습니다.
    kr = {"key": "kr-stocks", "models": kr_models}
    us = {"key": "us-stocks", "models": us_models}
    for sec in (kr, us):
        sec["status"] = aggregate_status([m["status"] for m in sec["models"]])
        first = next((m for m in sec["models"] if m["asof"]), None)
        sec["asof"] = good_date(first["asof"]) if first else None
    return kr, us


def build_ms_section(ms: dict | None, report_date: str) -> dict:
    sec = {
        "key": "market-sector",
        "status": "failed",
        "reason": [],
        "ms": None,
        "assets": [],
        "data_asof": {},
        "notes": [],
    }
    if ms is None:
        sec["reason"].append("입력 없음: 시장·섹터 입력(JSON)이 없습니다.")
        return sec
    assets_raw = ms.get("assets")
    if not isinstance(assets_raw, list):
        sec["reason"].append("입력 형식 오류: assets가 목록이 아닙니다.")
        return sec
    if ms.get("report_date") != report_date:
        sec["reason"].append(
            "입력 날짜 불일치: 입력 {}, 리포트 {}.".format(c(ms.get("report_date")), report_date)
        )
        return sec
    assets = [a for a in assets_raw if isinstance(a, dict) and a.get("market") in ("US", "KR")]
    if not assets:
        sec["reason"].append("자산 행이 없습니다.")
        return sec
    order = {"US": 0, "KR": 1}
    indexed = sorted(
        enumerate(assets),
        key=lambda t: (order[t[1]["market"]], 0 if t[1].get("group") == "market" else 1, t[0]),
    )
    assets = [a for _, a in indexed]
    core = ("asof_date", "ret_1", "ret_5", "ret_20", "ret_60", "rvol_20", "dd_252")
    problems = []
    for a in assets:
        missing = [
            k
            for k in core
            if (good_date(a.get(k)) is None if k == "asof_date" else not is_num(a.get(k)))
        ]
        if missing:
            problems.append(
                "{}: {} 값이 없습니다".format(
                    c(a.get("name") or a.get("asset_id")), ", ".join(missing)
                )
            )
    for market in ("US", "KR"):
        if not any(a["market"] == market for a in assets):
            problems.append(f"{market} 자산 행이 없습니다")
    status = "partial" if problems else "ok"
    if status == "ok" and ms.get("status") == "stale":
        status = "stale"
    sec.update(status=status, ms=ms, assets=assets, reason=problems)
    asof = ms.get("asof") if isinstance(ms.get("asof"), dict) else {}
    for key, label in (("KR", "kr_index"), ("US", "us_prices"), ("US_macro", "us_macro")):
        if good_date(asof.get(key)):
            sec["data_asof"][label] = asof[key]
    notes = ms.get("notes")
    sec["notes"] = [n for n in notes if isinstance(n, str)] if isinstance(notes, list) else []
    return sec


def unit_status(sec_statuses: list) -> str:
    """04 문서 변경 6: 전 섹션이 ok면 ok, 낸 섹션이 하나도 없으면 failed, 나머지는 partial."""
    if sec_statuses and all(s == "ok" for s in sec_statuses):
        return "ok"
    if all(s == "failed" for s in sec_statuses):
        return "failed"
    return "partial"


REPLAY_ALIASES = {
    "K": ("K", "k", "kr_session", "k_session"),
    "A": ("A", "a", "us_session", "a_session", "actual_us_session"),
    "snapshot": ("snapshot", "snapshots", "raw_snapshot", "kr_snapshot", "us_snapshot"),
    "executed_at": (
        "executed_at",
        "ran_at",
        "run_at",
        "actual_run_at",
        "started_at",
        "actual_executed_at",
    ),
    "skipped_checks": ("skipped_time_checks", "skipped_checks", "skipped_cutoff_checks"),
}


def replay_rows(replay: object) -> list:
    """replay 블록을 (이름, 값) 표 행으로 바꿉니다. 키 이름은 별칭을 받아 줍니다."""
    if not isinstance(replay, dict):
        return []
    used, rows = set(), []
    labels = {
        "K": "K (KR 기준 세션)",
        "A": "A (US 기준 세션)",
        "snapshot": "입력 snapshot",
        "executed_at": "실제 실행 시각",
        "skipped_checks": "건너뛴 시각 검사",
    }
    for canonical in ("K", "A", "snapshot", "executed_at", "skipped_checks"):
        found = next((k for k in REPLAY_ALIASES[canonical] if k in replay), None)
        if found is None:
            rows.append((labels[canonical], "기록 없음"))
            continue
        used.add(found)
        rows.append((labels[canonical], replay_value(replay[found], canonical == "executed_at")))
    for key in sorted(k for k in replay if k not in used and isinstance(k, str)):
        rows.append((code(key), replay_value(replay[key], False)))
    return rows


def replay_value(value: object, is_time: bool) -> str:
    if isinstance(value, dict):
        return "<br>".join(f"{c(k)}: {replay_value(value[k], False)}" for k in sorted(value))
    if isinstance(value, (list, tuple)):
        return "<br>".join(replay_value(v, False) for v in value) if value else "없음"
    if value is None:
        return "기록 없음"
    if is_time and parse_dt(value):
        return kst_text(value)
    return c(value)


# ---------------------------------------------------------------------------
# 저장소 경로와 트리 읽기
# ---------------------------------------------------------------------------
def unit_dir(repo: Path, unit: str) -> Path:
    return repo / "reports" / FAMILY / unit[0:4] / unit[5:7] / unit


def unit_rel(unit: str) -> str:
    return f"reports/{FAMILY}/{unit[0:4]}/{unit[5:7]}/{unit}"


def rel_from_unit(unit: str, target: str) -> str:
    return posixpath.relpath(target, unit_rel(unit))


def scan_units(repo: Path) -> list:
    base = repo / "reports" / FAMILY
    units = []
    if not base.is_dir():
        return units
    for ydir in sorted(base.iterdir()):
        if not (ydir.is_dir() and re.fullmatch(r"\d{4}", ydir.name)):
            continue
        for mdir in sorted(ydir.iterdir()):
            if not (mdir.is_dir() and re.fullmatch(r"\d{2}", mdir.name)):
                continue
            for udir in sorted(mdir.iterdir()):
                if not (udir.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", udir.name)):
                    continue
                meta = read_fm(udir / "README.md")
                if not meta:
                    sys.stderr.write(
                        f"경고: {udir.name}/README.md front matter를 읽지 못해 인덱스에서 뺍니다.\n"
                    )
                    continue
                sections = {}
                for key in ("market-sector", "kr-stocks", "us-stocks", "data-status"):
                    sm = read_fm(udir / SECTION_FILES[key])
                    sections[key] = sm.get("status") if sm else "-"
                units.append(
                    {
                        "unit": udir.name,
                        "year": ydir.name,
                        "month": mdir.name,
                        "status": meta.get("status"),
                        "sections": sections,
                        "replay": meta.get("historical_replay") is True,
                        "revision": meta.get("revision"),
                        "title": meta.get("title"),
                    }
                )
    return sorted(units, key=lambda u: u["unit"])


def last_normal_link(repo: Path, unit: str, section_file: str) -> str:
    """D보다 앞선 단위 중 그 섹션이 ok·partial인 가장 최근 단위로 가는 상대 링크.

    없으면 빈 문자열을 돌려줍니다.
    """
    for item in reversed(scan_units(repo)):
        if item["unit"] >= unit:
            continue
        meta = read_fm(unit_dir(repo, item["unit"]) / section_file)
        if meta and meta.get("status") in ("ok", "partial"):
            target = unit_rel(item["unit"]) + "/" + section_file
            link = rel_from_unit(unit, target)
            suffix = " (재현)" if item["replay"] else ""
            return "[{}]({}){}".format(item["unit"], link, suffix)
    return ""


# ---------------------------------------------------------------------------
# 본문 렌더링
# ---------------------------------------------------------------------------
def title_for(ctx: dict, base: str) -> str:
    return "{} — {}{}".format(base, ctx["unit"], " (재현)" if ctx["replay"] else "")


def head_lines(ctx: dict, title: str, summary: bool = False) -> list:
    lines = ["# " + title, ""]
    if summary and ctx["corrections"]:
        lines += [CORR_BEGIN] + ctx["corrections"] + [CORR_END, ""]
    if ctx["replay"]:
        lines += ["> " + REPLAY_NOTICE, ""]
    if ctx["synthetic"]:
        lines += ["> " + SYNTHETIC_NOTICE, ""]
    return lines


def model_link(ctx: dict, model_id: str) -> str:
    label = MODEL_DISPLAY_NAMES.get(model_id, model_id)
    path = rel_from_unit(ctx["unit"], f"reference/models/{model_id}.md")
    return f"[{c(label)}]({path}) ({code(model_id)})"


def status_with_internal(status: str, internal: str) -> str:
    text = STATUS_KO[status]
    if STATUS_MAP.get(internal) == status and internal == status:
        return text
    return f"{text} (내부 상태: {INTERNAL_KO.get(internal, internal)})"


def kr_basis_text(m: dict) -> str:
    asof = good_date(m["asof"])
    if asof is None:
        return "확인되지 않음"
    lag = m["lag"]
    if lag == 0:
        return f"{asof} (K와 같음)"
    if lag is None:
        return f"{asof} (K보다 이른 기준일 K′인지 확인되지 않음)"
    return f"{asof} (K′: K보다 {lag}세션 앞선 기준일)"


def stale_banner_kr(m: dict) -> str | None:
    if m["status"] == "failed" or m["suppressed"]:
        return None
    if m["status"] == "stale" or (m["lag"] or 0) > 0:
        lag = m["lag"]
        if lag is None:
            return "> 주의: 이 순위는 최신 K가 아니라 더 이른 기준일 입력으로 만들었습니다."
        asof = good_date(m["asof"]) or "확인되지 않음"
        return (
            f"> 주의: {lag}세션 전 기준 순위입니다. "
            f"K보다 {lag}세션 이른 기준일(K′ = {asof})의 입력으로 만들었습니다."
        )
    return None


def suppressed_lines(ctx: dict, m: dict, section_file: str, unit_market: str) -> list:
    lag = m["lag"]
    if unit_market == "KR":
        why = f"기준일이 K보다 {lag}세션 이르러(5세션 초과) 순위 표를 내지 않습니다."
    else:
        why = f"미국 입력이 {lag}세션 늦어(5세션 초과) 순위 표를 내지 않습니다."
    lines = ["> " + why, ""]
    return lines + last_normal_lines(ctx, section_file)


def last_normal_lines(ctx: dict, section_file: str) -> list:
    link = last_normal_link(ctx["repo"], ctx["unit"], section_file)
    if link:
        return ["마지막 정상 단위: " + link, ""]
    return ["마지막 정상 단위: 저장소에 아직 없습니다.", ""]


def render_kr(ctx: dict) -> tuple[list, dict]:
    sec = ctx["kr"]
    title = title_for(ctx, "KR 종목 순위")
    lines = head_lines(ctx, title)
    m = sec["models"][0]
    lines.append("- 상태: " + status_with_internal(sec["status"], m["internal"]))
    lines.append("- 기준일: " + kr_basis_text(m))
    lines.append("- 판단 시각: {}".format(kst_minute(ctx["decision_at"])))
    lines.append(
        "- 추론 시작: {}".format(kst_text((m["report"] or {}).get("inference_started_at")))
    )
    lines.append("- 모델: " + model_link(ctx, m["model_id"]))
    lines.append("")
    banner = stale_banner_kr(m)
    if banner:
        lines += [banner, ""]
    if m["status"] == "failed":
        lines += ["순위를 내지 못했습니다.", ""]
        lines += ["- " + r for r in m["reason"]] + [""]
        lines += last_normal_lines(ctx, "kr-stocks.md")
    elif m["suppressed"]:
        lines += suppressed_lines(ctx, m, "kr-stocks.md", "KR")
    else:
        rows = m["rows"]
        if m["status"] == "partial" and m["quality"].get("management_filter_available") is False:
            lines += [
                "KR 모델은 현재 관리종목·거래정지 상태를 확인하지 못해 늘 `부분 완료`로 "
                "표시합니다. "
                "품질 검토가 필요한 행도 원래 순위에 그대로 남겼습니다.",
                "",
            ]
        if not rows:
            lines += ["이번 판에 순위가 없습니다 (전체 0개).", ""]
        else:
            held, reasons, body = 0, Counter(), []
            for row in rows:
                reason_codes = row.get("quality_reasons")
                review = row.get("quality_review")
                texts = (
                    [KR_QUALITY_REASON_TEXT.get(x, str(x)) for x in reason_codes]
                    if isinstance(reason_codes, list)
                    else []
                )
                texts = list(dict.fromkeys(texts))
                if review is True or texts:
                    held += 1
                    for t in texts or ["사유 없음"]:
                        reasons[t] += 1
                    quality = "품질 보류: " + " / ".join(c(t) for t in (texts or ["사유 없음"]))
                elif review is False:
                    quality = "이상 없음"
                else:
                    quality = "품질 정보 없음"
                body.append(
                    [
                        str(row["rank"]),
                        c(row["symbol"]),
                        c(row["name"]),
                        "{:.4f}".format(row["score"]),
                        quality,
                    ]
                )
            lines.append(
                f"전체 {m['total']}개 중 상위 {len(rows)}개입니다. 그 밖의 순위는 올리지 않습니다."
            )
            lines.append("")
            lines.append("## 품질 보류 요약")
            lines.append("")
            lines.append(
                f"상위 {len(rows)}개 중 품질 보류 {held}개입니다. 보류 행도 원래 순위에 남겼습니다."
            )
            if reasons:
                lines.append("")
                lines += md_table(
                    ["보류 사유", "행 수"],
                    [
                        [c(k), str(v)]
                        for k, v in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
                    ],
                    right=(1,),
                )
            lines += ["", f"## 상위 {len(rows)}", ""]
            lines += md_table(
                ["순위", "코드", "이름", "점수 (순위용 점수 — 확률 아님)", "품질"],
                body,
                right=(0, 3),
            )
            lines.append("")
    lines += [DISCLAIMER, ""]
    meta_extra = {
        "data_asof": {"kr_features": sec["asof"]} if sec["asof"] else {},
        "models": [x["model_id"] for x in sec["models"]],
    }
    return lines, meta_extra


def render_us_model(ctx: dict, m: dict, show_name: bool) -> list:
    short = US_SHORT.get(m["model_id"], m["model_id"])
    lines = [f"## {c(short)}", ""]
    lines.append("- 상태: " + status_with_internal(m["status"], m["internal"]))
    f, q = m["fresh"], m["quality"]
    asof = good_date(m["asof"])
    if m["report"] is not None:
        lines.append("- 기준 세션 A: " + (asof or "확인되지 않음"))
        u = good_date(q.get("latest_us_session") or f.get("latest_us_session"))
        e = good_date(q.get("expected_us_session") or f.get("expected_us_session"))
        if u:
            lines.append(f"- 직전 완료 세션 U: {u}")
        if e:
            lines.append(f"- 기대 세션 E: {e}")
        if m["lag"] is not None:
            delivery = q.get("delivery_lag_sessions", f.get("delivery_lag"))
            extra = f" · 도착 지연(A와 E의 차이) {int(delivery)}세션" if is_num(delivery) else ""
            lines.append(f"- 지연 세션 수(A와 U의 차이): {m['lag']}{extra}")
        else:
            lines.append("- 지연 세션 수: 확인되지 않음")
    lines.append("- 모델: " + model_link(ctx, m["model_id"]))
    lines.append("")
    if (
        m["status"] != "failed"
        and m["lag"] is not None
        and m["lag"] >= LAG_WARN_US
        and not m["suppressed"]
    ):
        lines += [
            f"> 경고: 미국 입력이 {m['lag']}세션 늦었습니다. "
            f"기준 세션 A({asof or '확인되지 않음'})는 최신 세션이 아닙니다.",
            "",
        ]
    elif m["status"] == "stale" and not m["suppressed"]:
        lines += ["> 주의: 미국 입력이 최신 세션보다 늦습니다. 지연 세션 수를 확인하십시오.", ""]
    if m["status"] == "failed":
        lines += ["순위를 내지 못했습니다.", ""]
        lines += ["- " + r for r in m["reason"]] + [""]
        lines += last_normal_lines(ctx, "us-stocks.md")
        return lines
    if m["suppressed"]:
        return lines + suppressed_lines(ctx, m, "us-stocks.md", "US")
    rows = m["rows"]
    if not rows:
        return lines + ["이번 판에 순위가 없습니다 (전체 0개).", ""]
    lines.append(
        f"전체 {m['total']}개 중 상위 {len(rows)}개입니다. 그 밖의 순위는 올리지 않습니다."
    )
    lines.append("")
    headers = (
        ["순위", "코드"] + (["이름"] if show_name else []) + ["점수 (순위용 점수 — 확률 아님)"]
    )
    right = (0, len(headers) - 1)
    body = []
    for row in rows:
        cells = [str(row["rank"]), c(row["symbol"])]
        if show_name:
            cells.append(c(row["name"]))
        cells.append("{:.4f}".format(row["score"]))
        body.append(cells)
    lines += md_table(headers, body, right=right)
    lines.append("")
    return lines


def us_overlap(sec: dict, top_n: int) -> int | None:
    shown = [
        m for m in sec["models"] if m["status"] != "failed" and not m["suppressed"] and m["rows"]
    ]
    if len(shown) < 2:
        return None
    sets = [set(r["symbol"] for r in m["rows"][:top_n]) for m in shown[:2]]
    return len(sets[0] & sets[1])


def render_us(ctx: dict) -> tuple[list, dict]:
    sec = ctx["us"]
    title = title_for(ctx, "US 종목 순위")
    lines = head_lines(ctx, title)
    lines.append("- 상태: {}".format(STATUS_KO[sec["status"]]))
    lines.append("- 판단 시각: {}".format(kst_minute(ctx["decision_at"])))
    asofs = sorted({good_date(m["asof"]) for m in sec["models"] if good_date(m["asof"])})
    lines.append("- 기준 세션 A: " + (", ".join(asofs) if asofs else "확인되지 않음"))
    lag_values = sorted({m["lag"] for m in sec["models"] if m["lag"] is not None})
    if lag_values:
        lines.append("- 지연 세션 수: {}".format(", ".join(str(x) for x in lag_values)))
    overlap = us_overlap(sec, ctx["top_n"])
    if overlap is not None:
        lines.append(f"- 두 모델 상위 {ctx['top_n']}의 겹침: {overlap}개")
    lines.append("")
    shown_rows = [
        r
        for m in sec["models"]
        if m["status"] != "failed" and not m["suppressed"]
        for r in m["rows"]
    ]
    show_name = any(r["name"] != r["symbol"] for r in shown_rows)
    if shown_rows and not show_name:
        lines += [
            "회사명 원천이 없어 이름은 코드(symbol)로 대신 표시합니다. 같은 값이라 이름 열은 "
            "생략했습니다.",
            "",
        ]
    for m in sec["models"]:
        lines += render_us_model(ctx, m, show_name)
    lines += ["LightGBM과 Ridge의 점수는 서로 합치거나 비교하지 않습니다.", "", DISCLAIMER, ""]
    meta_extra = {
        "data_asof": {"us_features": asofs[0]} if asofs else {},
        "models": [x["model_id"] for x in sec["models"]],
    }
    return lines, meta_extra


def table_assets(assets: list, market: str) -> list:
    rows = []
    for a in assets:
        if a["market"] != market:
            continue
        basis = {"total_return": "배당 포함", "price_only": "배당 제외"}.get(
            a.get("return_basis"), c(a.get("return_basis"))
        )
        rows.append(
            [
                c(a.get("name") or a.get("asset_id")),
                basis,
                c(a.get("asof_date")),
                pct(a.get("ret_1")),
                pct(a.get("ret_5")),
                pct(a.get("ret_20")),
                pct(a.get("ret_60")),
                pct(a.get("rvol_20"), signed=False),
                pct(a.get("dd_252")),
            ]
        )
    return rows


def cash_line(assets: list, market: str, label: str) -> str:
    values = sorted(
        {
            round(a["cash_rate"], 6)
            for a in assets
            if a["market"] == market and is_num(a.get("cash_rate"))
        }
    )
    if not values:
        return f"- 현금 금리({label}): -"
    if len(values) == 1:
        return f"- 현금 금리({label}): {pct(values[0], signed=False)}"
    return "- 현금 금리({}): 자산마다 다릅니다 ({})".format(
        label, ", ".join(pct(v, signed=False) for v in values)
    )


def render_ms(ctx: dict) -> tuple[list, dict]:
    sec = ctx["ms"]
    title = title_for(ctx, "시장·섹터")
    lines = head_lines(ctx, title)
    lines.append("- 상태: {}".format(STATUS_KO[sec["status"]]))
    lines.append("- 판단 시각: {}".format(kst_minute(ctx["decision_at"])))
    lines.append(
        "- 모델 설명: [MS1 시장·섹터]({})".format(
            rel_from_unit(ctx["unit"], f"reference/models/{MS_MODEL}.md")
        )
    )
    lines.append("")
    meta_extra = {"data_asof": dict(sec["data_asof"]), "models": [MS_MODEL]}
    if sec["status"] == "failed":
        lines += ["시장·섹터를 내지 못했습니다.", ""]
        lines += ["- " + r for r in sec["reason"]] + [""]
        lines += [DISCLAIMER, ""]
        return lines, meta_extra
    ms, assets = sec["ms"], sec["assets"]
    if sec["status"] != "ok" and sec["reason"]:
        lines += ["> 주의: 일부 값이 비어 있습니다."] + ["> - " + r for r in sec["reason"]] + [""]
    asof = ms.get("asof") if isinstance(ms.get("asof"), dict) else {}
    lines += ["## 입력 기준일", ""]
    lines += md_table(
        ["입력", "기준일", "비고"],
        [
            ["US 가격·배당", c(asof.get("US") or "-"), "직전 US 세션 기준"],
            [
                "US 거시(DGS3MO 등)",
                c(asof.get("US_macro") or "-"),
                "주 1회 갱신이라 최대 7일 늦을 수 있습니다",
            ],
            ["KR 지수", c(asof.get("KR") or "-"), "T+1 공표라 K보다 한 세션 늦습니다"],
        ],
    )
    if sec["notes"]:
        lines.append("")
        lines += ["- 참고: " + c(n) for n in sec["notes"]]
    lines += [
        "",
        "## 시장 상태",
        "",
        "모델이 아니라 기술 통계입니다. 수익률은 세션 기준이고, 지수 종가 수준은 올리지 않습니다. "
        "변동성은 20세션 실현 변동성, 낙폭은 252세션 고점 대비입니다.",
        "",
    ]
    headers = [
        "자산",
        "수익 기준",
        "기준 종가일",
        "1세션",
        "5세션",
        "20세션",
        "60세션",
        "20세션 변동성",
        "252세션 낙폭",
    ]
    for market, label, cash_label in (("US", "US", "DGS3MO"), ("KR", "KR", "CD91")):
        rows = table_assets(assets, market)
        if not rows:
            continue
        lines += ["### " + label, ""]
        lines += md_table(headers, rows, right=tuple(range(3, 9)))
        lines += ["", cash_line(assets, market, cash_label), ""]
    lines += [
        "KR 수익률은 배당을 뺀 값(`price_only`)입니다. US 수익률은 배당을 포함한 "
        "값(`total_return`)입니다.",
        "",
    ]
    lines += [
        "## baseline",
        "",
        "MS1에서 연구용 모델과 비교하려고 둔 기준선입니다. 원래 단위 그대로 적고, 0\\~100 "
        "변환은 하지 않습니다. "
        "MS1 판정에서 채택할 모델이 없어 두 시장 모두 baseline을 유지합니다.",
        "",
    ]
    rows = [
        [
            c(a.get("name") or a.get("asset_id")),
            a["market"],
            pct_plain(a.get("b_opp_mean_pct")),
            pct_plain(a.get("b_stab_prob_pct")),
        ]
        for a in assets
    ]
    lines += md_table(
        [
            "자산",
            "시장",
            "60일 기대 초과수익 (b_opp_mean)",
            "8% 손실 사건 확률 (b_stab_logit_rvol)",
        ],
        rows,
        right=(2, 3),
    )
    lines += ["", DISCLAIMER, ""]
    lines += render_ms_details(ms, assets)
    return lines, meta_extra


def render_ms_details(ms: dict, assets: list) -> list:
    verdicts = ms.get("verdicts") if isinstance(ms.get("verdicts"), dict) else {}
    lines = [
        "<details>",
        "<summary>MS1 연구용 점수 — 판정 실패·채택 없음, 의사결정에 쓰지 않음</summary>",
        "",
    ]
    lines += [
        "> 경고: 아래 점수는 판정에서 채택되지 않은 연구용 값입니다. 의사결정에 쓰지 않습니다.",
        "> Stability 점수는 과신입니다. 확률로 읽지 않습니다.",
        "",
    ]
    counts = Counter()
    targets = (
        ("opportunity", "Opportunity (Ridge, alpha 10)"),
        ("sector_relative", "섹터 상대 선택 (Ridge)"),
        ("stability", "Stability (L2 Logistic, C 0.1)"),
    )
    for market in ("US", "KR"):
        block = verdicts.get(market) if isinstance(verdicts.get(market), dict) else {}
        for key, _ in targets:
            if isinstance(block.get(key), str):
                counts[block[key]] += 1
    lines.append("**목표별 판정**")
    lines.append("")
    if counts:
        total = sum(counts.values())
        lines.append(
            f"판정 {total}개 중 통과 {counts.get('통과', 0)}, 보류 {counts.get('보류', 0)}, "
            f"실패 {counts.get('실패', 0)}입니다."
        )
        lines.append("")
        rows = []
        for key, label in targets:
            cells = []
            for market in ("US", "KR"):
                block = verdicts.get(market) if isinstance(verdicts.get(market), dict) else {}
                cells.append(c(block.get(key) or "-"))
            rows.append([label] + cells)
        lines += md_table(["목표 (주 모델)", "US", "KR"], rows)
    else:
        lines.append("판정 정보가 입력에 없습니다.")
    lines += [
        "",
        "**연구용 점수**",
        "",
        "Opportunity는 모델 예측의 이전 fold OOF 백분위(0\\~100)이고, Stability는 `100 × (1 − "
        "p̂)`(0\\~100)입니다. "
        "판정에서 채택되지 않아 신호로 읽지 않습니다. 기준일은 자산마다 적었습니다.",
        "",
    ]
    rows = [
        [
            c(a.get("name") or a.get("asset_id")),
            a["market"],
            c(a.get("score_asof_date") or a.get("asof_date")),
            num(a.get("opportunity_score")),
            num(a.get("stability_score")),
        ]
        for a in assets
    ]
    if any(
        good_date(a.get("score_asof_date")) and a.get("score_asof_date") != a.get("asof_date")
        for a in assets
    ):
        lines += [
            "점수 기준일이 위 시장 상태 표의 기준 종가일과 다릅니다. 점수는 그 기준일 그대로 "
            "적었습니다.",
            "",
        ]
    lines += md_table(
        ["자산", "시장", "기준일", "Opportunity (0\\~100)", "Stability (0\\~100)"],
        rows,
        right=(3, 4),
    )
    lines += ["", "</details>", ""]
    return lines


FRESH_ROWS = (
    ("신선도 상태", "status"),
    ("입력 cutoff", "input_cutoff"),
    ("입력 가용 확인 시각", "verified_available_by"),
)
QUALITY_SCALARS = (
    "status",
    "management_filter_available",
    "management_state",
    "halted_rows_at_K",
    "price_jump_review",
    "top100_quality_review_rows",
    "D_management_and_halt_state",
    "eligible_rows_scored",
    "feature_count",
    "design_column_count",
    "eligible_rows",
    "monthly_membership_month",
    "freshness_status",
)
PROV_HASHES = (
    ("bundle_manifest_sha256", "bundle manifest"),
    ("prepared_manifest_sha256", "prepared manifest"),
    ("input_sha256", "입력 파일"),
    ("native_prepare_manifest_sha256", "native prepare manifest"),
    ("code_inventory_sha256", "코드 목록"),
)


def scalar_text(value: object) -> str:
    if isinstance(value, bool):
        return "예" if value else "아니오"
    if value is None:
        return "-"
    if is_num(value):
        return str(value)
    if isinstance(value, str):
        return c(value)
    return "-"


def internal_cell(sec: dict) -> str:
    models = sec["models"]
    if len(models) == 1:
        return INTERNAL_KO.get(models[0]["internal"], models[0]["internal"])
    return "<br>".join(
        "{} {}".format(
            c(US_SHORT.get(m["model_id"], m["model_id"])),
            INTERNAL_KO.get(m["internal"], m["internal"]),
        )
        for m in models
    )


def render_status(ctx: dict) -> tuple[list, dict]:
    title = title_for(ctx, "데이터 상태")
    lines = head_lines(ctx, title)
    kr, us, ms = ctx["kr"], ctx["us"], ctx["ms"]
    lines += ["## 섹션 상태", ""]

    def reason_text(sec: dict) -> str:
        if sec["key"] == "market-sector":
            return "<br>".join(sec["reason"]) if sec["reason"] else "-"
        parts = []
        for m in sec["models"]:
            if m["status"] == "failed" or m["suppressed"]:
                what = (
                    "; ".join(m["reason"])
                    if m["reason"]
                    else "지연 세션 수 {}로 순위 표를 내지 않았습니다".format(m["lag"])
                )
                parts.append(("{}: ".format(code(m["model_id"]))) + what)
        return "<br>".join(parts) if parts else "-"

    asof_ms = asof_text(ms["data_asof"])
    rows = [
        ["시장·섹터", STATUS_KO[ms["status"]], "-", c(asof_ms), reason_text(ms)],
        [
            "KR 종목 순위",
            STATUS_KO[kr["status"]],
            internal_cell(kr),
            c(kr["asof"] or "-"),
            reason_text(kr),
        ],
        [
            "US 종목 순위",
            STATUS_KO[us["status"]],
            internal_cell(us),
            c(us["asof"] or "-"),
            reason_text(us),
        ],
    ]
    lines += md_table(["섹션", "상태", "내부 상태", "기준일", "사유"], rows)
    lines += [
        "",
        "내부 report 전체 상태: {}. 단위 상태는 섹션 상태에서 다시 계산합니다 "
        "(전 섹션 정상이면 정상, 낸 섹션이 없으면 실패, 나머지는 부분 완료).".format(
            c(INTERNAL_KO.get(ctx["env_status"], ctx["env_status"]))
        ),
        "",
    ]
    lines += ["## 입력 기준일과 신선도", ""]
    fresh_rows = []
    for sec in (kr, us):
        for m in sec["models"]:
            if m["report"] is None:
                fresh_rows.append([m["market"], code(m["model_id"]), "-", "-", "-", "-", "-"])
                continue
            f = m["fresh"]
            fresh_rows.append(
                [
                    m["market"],
                    code(m["model_id"]),
                    c(m["asof"] or "-"),
                    c(f.get("status") or m["quality"].get("freshness_status") or "-"),
                    "-" if m["lag"] is None else str(m["lag"]),
                    (
                        kst_text(f.get("verified_available_by"))
                        if f.get("verified_available_by")
                        else "-"
                    ),
                    kst_text(m["report"].get("inference_started_at")),
                ]
            )
    lines += md_table(
        ["시장", "모델", "기준일", "신선도", "지연 세션 수", "입력 가용 확인", "추론 시작"],
        fresh_rows,
        right=(4,),
    )
    cutoffs = sorted(
        {
            kst_text(m["fresh"].get("input_cutoff"))
            for sec in (kr, us)
            for m in sec["models"]
            if m["fresh"].get("input_cutoff")
        }
    )
    lines += [
        "",
        "입력 cutoff: %s. 이 시각보다 늦게 끝난 입력은 쓰지 않습니다."
        % (", ".join(cutoffs) if cutoffs else "D 09:30 KST"),
    ]
    lines += ["", f"시장·섹터 입력 기준일: {c(asof_ms)}.", ""]
    lines += ["## 실패", ""]
    failure_rows = [
        [
            c(f.get("market") or "-"),
            code(f.get("model_id") or "-"),
            code(f.get("error_class") or "-"),
        ]
        for f in ctx["failures"]
    ]
    if failure_rows:
        lines += md_table(["시장", "모델", "원인 클래스"], failure_rows)
    else:
        lines.append("envelope에 기록된 추론 실패가 없습니다.")
    lines += ["", "## 게이트와 품질 값", ""]
    gate_rows = []
    pub_states: set = set()
    for sec in (kr, us):
        for m in sec["models"]:
            if m["report"] is None:
                continue
            q = m["quality"]
            for key in QUALITY_SCALARS:
                if key in q and not isinstance(q[key], (dict, list)):
                    gate_rows.append([code(m["model_id"]), code(key), scalar_text(q[key])])
            pub_states.add(str((m["report"].get("publication") or {}).get("status")))
    if gate_rows:
        lines += md_table(["모델", "항목", "값"], gate_rows)
    else:
        lines.append("기록된 값이 없습니다.")
    if pub_states:
        states = ", ".join(code(x) for x in sorted(pub_states))
        lines += [
            "",
            f"공개 게이트(`publication.status`): {states}. "
            "private 저장소를 본인만 보는 경로와는 별개인 값입니다.",
        ]
    lines += ["", "KIS 장중 관측(opening)은 이 리포트 범위 밖입니다.", ""]
    lines += ["## 출처", ""]
    prov_rows = [["release", code(ctx["release"])], ["내부 report sha256", code(ctx["sha"])]]
    for sec in (kr, us):
        for m in sec["models"]:
            if m["report"] is None:
                continue
            for key, label in PROV_HASHES:
                value = m["prov"].get(key)
                if is_hex64(value):
                    prov_rows.append([c("{} {}".format(m["model_id"], label)), code(value)])
            run_id = m["prov"].get("source_model_run_id")
            if isinstance(run_id, str) and re.fullmatch(r"[A-Za-z0-9._\-]+", run_id):
                prov_rows.append([c("{} 학습 run".format(m["model_id"])), code(run_id)])
    ms_raw = ms.get("ms") or {}
    ms_prov = ms_raw.get("provenance") if isinstance(ms_raw.get("provenance"), dict) else {}
    if isinstance(ms_prov.get("ms_runs"), list):
        prov_rows.append(
            [
                "시장·섹터 run",
                ", ".join(code(x) for x in ms_prov["ms_runs"] if isinstance(x, str)) or "-",
            ]
        )
    if isinstance(ms_prov.get("config_hash"), str):
        prov_rows.append(["시장·섹터 config hash", c(ms_prov["config_hash"])])
    if isinstance(ms_prov.get("calendar_basis"), dict):
        prov_rows.append(
            [
                "시장·섹터 계산 달력",
                "<br>".join(
                    f"{c(k)}: {c(v)}" for k, v in sorted(ms_prov["calendar_basis"].items())
                ),
            ]
        )
    lines += md_table(["항목", "값"], prov_rows)
    lines.append("")
    if ctx["replay"]:
        lines += ["## 재현 정보", ""]
        rows = replay_rows(ctx["replay_info"])
        if rows:
            lines += md_table(["항목", "값"], [[a, b] for a, b in rows])
        else:
            lines.append("envelope에 `replay` 블록이 없습니다.")
        lines.append("")
    return lines, {"data_asof": all_data_asof(ctx), "models": []}


def all_data_asof(ctx: dict) -> dict:
    merged = {}
    merged.update(ctx["ms"]["data_asof"])
    if ctx["kr"]["asof"]:
        merged["kr_features"] = ctx["kr"]["asof"]
    if ctx["us"]["asof"]:
        merged["us_features"] = ctx["us"]["asof"]
    return merged


def render_summary(ctx: dict) -> tuple[list, dict]:
    title = title_for(ctx, "일일 브리핑")
    lines = head_lines(ctx, title, summary=True)
    kr, us, ms = ctx["kr"], ctx["us"], ctx["ms"]
    lines.append(
        "단위 상태: {}. 판단 시각은 {}입니다.".format(
            STATUS_KO[ctx["status"]], kst_minute(ctx["decision_at"])
        )
    )
    lines += ["", "## 섹션 상태", ""]
    asof_ms = asof_text(ms["data_asof"])
    kr_m = kr["models"][0]
    rows = [
        ["[시장·섹터](market-sector.md)", STATUS_KO[ms["status"]], c(asof_ms)],
        [
            "[KR 종목 순위](kr-stocks.md)",
            STATUS_KO[kr["status"]],
            c(kr_basis_text(kr_m)) if kr_m["asof"] else "-",
        ],
        [
            "[US 종목 순위](us-stocks.md)",
            STATUS_KO[us["status"]],
            ("A " + us["asof"]) if us["asof"] else "-",
        ],
        [
            "[데이터 상태](data-status.md)",
            STATUS_KO[ctx["status"]],
            "입력별 기준일·신선도·실패·출처",
        ],
    ]
    lines += md_table(["섹션", "상태", "기준일"], rows)
    lines += ["", "## 경고", ""]
    warnings = []
    if ms["status"] == "failed":
        warnings.append("시장·섹터를 내지 못했습니다: " + " ".join(ms["reason"]))
    elif ms["status"] != "ok" and ms["reason"]:
        warnings.append("시장·섹터 일부 값이 비어 있습니다.")
    for sec, label in ((kr, "KR"), (us, "US")):
        groups: dict = {}
        for m in sec["models"]:
            if m["status"] == "failed":
                text = "순위를 내지 못했습니다: " + " ".join(m["reason"])
            elif m["suppressed"]:
                text = f"입력이 {m['lag']}세션 늦어(5세션 초과) 순위 표를 내지 않았습니다."
            elif m["status"] == "stale":
                text = f"입력이 {'?' if m['lag'] is None else m['lag']}세션 늦습니다."
            else:
                continue
            groups.setdefault(text, []).append(m)
        for text, models in groups.items():
            if sec is us:
                names = "·".join(US_SHORT.get(m["model_id"], m["model_id"]) for m in models)
                warnings.append(f"US {names}: {text}")
            else:
                warnings.append("KR: " + text)
    if kr_m["status"] == "partial" and kr_m["quality"].get("management_filter_available") is False:
        warnings.append("KR은 현재 관리종목·거래정지 상태를 확인하지 못해 부분 완료로 표시합니다.")
    if ctx["corrections"]:
        warnings.append("이 단위는 정정판입니다. 위 정정 줄을 확인하십시오.")
    if warnings:
        lines += ["- " + w for w in warnings]
    else:
        lines.append("특이 사항이 없습니다.")
    lines.append("")
    index_assets = [a for a in ms["assets"] if a.get("group") == "market"]
    lines += ["## 대표지수", ""]
    if ms["status"] != "failed" and index_assets:
        rows = [
            [
                c(a.get("name") or a.get("asset_id")),
                c(a.get("asof_date")),
                pct(a.get("ret_1")),
                pct(a.get("ret_20")),
                pct(a.get("dd_252")),
            ]
            for a in index_assets
        ]
        lines += md_table(
            ["지수", "기준 종가일", "1세션", "20세션", "252세션 낙폭"], rows, right=(2, 3, 4)
        )
        lines += ["", "지수 종가 수준은 올리지 않습니다.", ""]
    else:
        lines += ["대표지수 상태를 내지 못했습니다.", ""]
    lines += ["## KR 상위 10", ""]
    lines += top10_lines(kr_m, with_name=True)
    lines += ["## US 상위 10", ""]
    for m in us["models"]:
        lines += ["### " + c(US_SHORT.get(m["model_id"], m["model_id"])), ""]
        lines += top10_lines(m, with_name=False)
    lines += [
        f"상위 10개만 적었습니다. 상위 {ctx['top_n']}개와 점수(순위용 점수 — 확률 아님)는 "
        "각 섹션 파일에 있습니다.",
        "",
        DISCLAIMER,
        "",
    ]
    meta_extra = {
        "data_asof": all_data_asof(ctx),
        "models": list(
            dict.fromkeys(
                [m["model_id"] for m in kr["models"]]
                + [m["model_id"] for m in us["models"]]
                + [MS_MODEL]
            )
        ),
    }
    return lines, meta_extra


def top10_lines(m: dict, with_name: bool) -> list:
    if m["status"] == "failed":
        return ["순위를 내지 못했습니다.", ""]
    if m["suppressed"]:
        return [f"입력이 {m['lag']}세션 늦어(5세션 초과) 순위를 내지 않았습니다.", ""]
    rows = m["rows"][:10]
    if not rows:
        return ["이번 판에 순위가 없습니다.", ""]
    show_name = with_name or any(r["name"] != r["symbol"] for r in rows)
    headers = ["순위", "코드"] + (["이름"] if show_name else [])
    body = [[str(r["rank"]), c(r["symbol"])] + ([c(r["name"])] if show_name else []) for r in rows]
    return md_table(headers, body, right=(0,)) + [""]


SECTION_RENDERERS = (
    ("summary", "일일 브리핑", render_summary),
    ("market-sector", "시장·섹터", render_ms),
    ("kr-stocks", "KR 종목 순위", render_kr),
    ("us-stocks", "US 종목 순위", render_us),
    ("data-status", "데이터 상태", render_status),
)
SECTION_MARKETS = {
    "summary": ["KR", "US"],
    "market-sector": ["KR", "US"],
    "kr-stocks": ["KR"],
    "us-stocks": ["US"],
    "data-status": ["KR", "US"],
}


def render_unit(ctx: dict) -> dict:
    """단위 폴더의 파일 5개를 {파일 이름: 텍스트}로 만듭니다."""
    files = {}
    for section, base, renderer in SECTION_RENDERERS:
        lines, extra = renderer(ctx)
        status = (
            ctx["status"]
            if section in ("summary", "data-status")
            else ctx[{"market-sector": "ms", "kr-stocks": "kr", "us-stocks": "us"}[section]][
                "status"
            ]
        )
        meta = {
            "schema": SCHEMA,
            "family": FAMILY,
            "unit": ctx["unit"],
            "section": section,
            "title": title_for(ctx, base),
            "status": status,
            "markets": SECTION_MARKETS[section],
            "decision_at": ctx["decision_iso"],
            "generated_at": ctx["generated_at"],
            "data_asof": extra["data_asof"],
            "models": extra["models"],
            "revision": ctx["revision"],
            "source": {"release": ctx["release"], "report_sha256": ctx["sha"]},
        }
        if ctx["replay"]:
            meta["historical_replay"] = True
        if ctx["synthetic"]:
            meta["synthetic_fixture"] = True
        body = escape_tilde("\n".join(lines).rstrip("\n") + "\n")
        files[SECTION_FILES[section]] = front_matter(meta) + "\n" + body
    return files


# ---------------------------------------------------------------------------
# 단위 쓰기: 같은 내용이면 건드리지 않고, 다르면 정정 규칙을 따릅니다
# ---------------------------------------------------------------------------
def normalize(text: str) -> str:
    """비교용: generated_at·revision 줄과 정정 블록을 뺍니다."""
    raw, body = split_front_matter(text)
    if raw is None:
        return text
    kept = [x for x in raw.split("\n") if not re.match(r"(generated_at|revision):", x)]
    raw = "\n".join(kept)
    body = re.sub(
        re.escape(CORR_BEGIN) + r".*?" + re.escape(CORR_END) + r"\n?\n?", "", body, flags=re.DOTALL
    )
    return raw + "\n---\n" + body


def read_existing(udir: Path) -> dict | None:
    if not udir.is_dir():
        return None
    texts = {}
    for name in SECTION_FILES.values():
        path = udir / name
        if path.is_file():
            texts[name] = path.read_text(encoding="utf-8")
    return texts or None


def existing_corrections(readme_text: str) -> list:
    match = re.search(
        re.escape(CORR_BEGIN) + r"\n(.*?)\n" + re.escape(CORR_END), readme_text, flags=re.DOTALL
    )
    return match.group(1).split("\n") if match else []


def write_if_changed(path: Path, text: str) -> bool:
    data = text.encode("utf-8")
    if path.is_file() and path.read_bytes() == data:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(str(tmp), str(path))
    return True


def parse_envelope(env: dict) -> tuple[str, datetime]:
    """envelope의 report_date와 decision_at을 확인하고 (단위 날짜, 판단 시각)을 돌려줍니다."""
    for key in ("report_date", "decision_at", "markets"):
        if key not in env:
            raise InputError(f"envelope에 {key}가 없습니다.")
    unit = good_date(env["report_date"])
    if unit is None:
        raise InputError("report_date 형식이 YYYY-MM-DD가 아닙니다.")
    decision = parse_dt(env["decision_at"])
    if (
        decision is None
        or decision.astimezone(KST).strftime("%Y-%m-%dT%H:%M:%S") != unit + "T10:00:00"
    ):
        raise InputError("decision_at이 report_date 10:00 KST가 아닙니다.")
    if not isinstance(env["markets"], list):
        raise InputError("envelope.markets가 목록이 아닙니다.")
    return unit, decision


def build_context(
    repo: Path,
    env_bytes: bytes,
    ms_bytes: bytes | None,
    release: str,
    generated_at: str | None,
    top_n: int,
    log,
) -> dict:
    """envelope(JSON 바이트)와 시장·섹터 입력으로 렌더 입력(ctx)을 만듭니다. 파일은 쓰지 않습니다.

    `repo`는 "마지막 정상 단위" 링크를 찾으려고 읽기만 합니다. 없는 디렉터리여도 됩니다.
    """
    env = load_json_bytes(env_bytes, "envelope")
    unit, decision = parse_envelope(env)
    if not re.fullmatch(r"[A-Za-z0-9._\-]+", release or ""):
        raise InputError("--release는 영문·숫자·.·_·- 만 쓸 수 있습니다.")
    failures = []
    for f in env.get("failures") or []:
        if isinstance(f, dict):
            failures.append(
                {
                    "market": _safe_str(f.get("market")),
                    "model_id": _safe_str(f.get("model_id")),
                    "error_class": _safe_str(f.get("error_class")),
                }
            )
    reports = []
    for raw in env["markets"]:
        try:
            validate_report_min(raw)
            if raw["report_date"] != unit or parse_dt(raw["decision_at"]) != decision:
                raise ValueError("report date/decision mismatch")
            reports.append(raw)
        except (ValueError, KeyError, TypeError):
            failures.append(
                {
                    "market": _safe_str(raw.get("market")) if isinstance(raw, dict) else "",
                    "model_id": _safe_str(raw.get("model_id")) if isinstance(raw, dict) else "",
                    "error_class": "InvalidReport",
                }
            )
            log("경고: 형식이 맞지 않는 report를 실패로 처리했습니다.")
    ms = load_json_bytes(ms_bytes, "시장·섹터 입력") if ms_bytes is not None else None
    replay = env.get("historical_replay") is True
    if ms is not None and (ms.get("historical_replay") is True) != replay:
        log(
            "경고: envelope와 시장·섹터 입력의 historical_replay 값이 다릅니다. "
            "envelope 값을 따릅니다."
        )
    kr, us = build_market_sections(env, reports, failures, top_n)
    ms_sec = build_ms_section(ms, unit)
    stamp = generated_at
    if stamp:
        parsed = parse_dt(stamp)
        if parsed is None:
            raise InputError("--generated-at은 시간대가 있는 ISO 시각이어야 합니다.")
        stamp = parsed.astimezone(KST).isoformat()
    else:
        stamp = datetime.now(KST).replace(microsecond=0).isoformat()
    ctx = {
        "repo": repo,
        "unit": unit,
        "decision_at": env["decision_at"],
        "decision_iso": decision.astimezone(KST).isoformat(),
        "generated_at": stamp,
        "revision": 1,
        "release": release,
        "sha": sha256_bytes(env_bytes),
        "replay": replay,
        "replay_info": env.get("replay"),
        "synthetic": env.get("synthetic_fixture") is True,
        "corrections": [],
        "top_n": top_n,
        "kr": kr,
        "us": us,
        "ms": ms_sec,
        "failures": failures,
        "env_status": env.get("status") if isinstance(env.get("status"), str) else "failed",
    }
    ctx["status"] = unit_status([kr["status"], us["status"], ms_sec["status"]])
    return ctx


def unit_sha256(files: dict) -> str:
    """단위 파일 묶음의 해시입니다. 파일 이름과 내용 해시를 이름 순서로 이어 계산합니다."""
    digest = hashlib.sha256()
    for name in sorted(files):
        data = files[name].encode("utf-8") if isinstance(files[name], str) else files[name]
        digest.update(f"{name}\0{sha256_bytes(data)}\n".encode())
    return digest.hexdigest()


def same_unit_content(existing: dict, files: dict) -> bool:
    """generated_at·revision·정정 줄을 빼고 두 단위의 내용이 같은지 봅니다."""
    return set(existing) == set(files) and all(
        normalize(files[name]) == normalize(existing[name]) for name in files
    )


def existing_revision(existing: dict) -> int:
    try:
        meta, _ = parse_front_matter(existing["README.md"])
    except (KeyError, ValueError):
        meta = None
    if meta and isinstance(meta.get("revision"), int) and meta["revision"] >= 1:
        return meta["revision"]
    return 1


def correct_unit_files(ctx: dict, existing: dict, reason: str, previous_commit: str | None) -> dict:
    """기존 단위의 revision을 하나 올리고, 요약 맨 위에 정정 줄을 넣어 다시 렌더합니다."""
    old_rev = existing_revision(existing)
    old_corr = existing_corrections(existing["README.md"]) if "README.md" in existing else []
    ctx["revision"] = old_rev + 1
    prev = c(previous_commit) if previous_commit else "git 이력에 있습니다."
    line = (
        f"> 정정 r{ctx['revision']} ({ctx['generated_at']}): "
        f"{c(reason).rstrip('.')}. 이전 판: {prev}"
    )
    ctx["corrections"] = [line] + old_corr
    return render_unit(ctx)


def generate(
    repo: Path,
    env_path: Path,
    ms_path: Path | None,
    release: str,
    generated_at: str | None,
    top_n: int,
    reason: str | None,
    previous_commit: str | None,
    overwrite: bool,
    log,
) -> str:
    """단위 하나를 저장소 트리에 쓰고 인덱스를 다시 만듭니다. 수동·시험용입니다.

    publisher는 이 함수를 쓰지 않습니다. 원격에 맞춘 트리 위에 직접 얹습니다.
    """
    if not repo.is_dir():
        raise InputError("--repo 디렉터리가 없습니다. 먼저 --init으로 만드십시오.")
    env_bytes = read_bytes(env_path, "envelope")
    ms_bytes = read_bytes(ms_path, "시장·섹터 입력") if ms_path else None
    ctx = build_context(repo, env_bytes, ms_bytes, release, generated_at, top_n, log)
    unit = ctx["unit"]
    udir = unit_dir(repo, unit)
    existing = read_existing(udir)
    if existing is None or "README.md" not in existing:
        files = render_unit(ctx)
        action = "created"
    else:
        ctx["revision"] = existing_revision(existing)
        ctx["corrections"] = existing_corrections(existing["README.md"])
        files = render_unit(ctx)
        if same_unit_content(existing, files):
            log(f"{unit}: unchanged (내용이 같아 파일을 건드리지 않았습니다)")
            reindex(repo, log)
            return "unchanged"
        if reason:
            files = correct_unit_files(ctx, existing, reason, previous_commit)
            action = "corrected"
        elif overwrite:
            action = "overwritten"
        else:
            raise InputError(
                f"{unit} 단위가 이미 있고 내용이 다릅니다. --reason으로 정정하거나"
                "(revision을 올립니다), 같은 revision으로 덮으려면 --overwrite를 쓰십시오."
            )
    changed = [name for name, text in files.items() if write_if_changed(udir / name, text)]
    log(f"{unit}: {action} ({len(changed)}개 파일 쓰기)")
    reindex(repo, log)
    return action


# ---------------------------------------------------------------------------
# 인덱스 재생성 (자동 구간만 다시 씁니다)
# ---------------------------------------------------------------------------
def replace_auto(text: str, block_lines: list) -> str:
    block = AUTO_BEGIN + "\n" + "\n".join(block_lines).strip("\n") + "\n" + AUTO_END
    begin, end = text.find(AUTO_BEGIN), text.find(AUTO_END)
    if begin != -1 and end != -1 and end > begin:
        return text[:begin] + block + text[end + len(AUTO_END) :]
    return text.rstrip("\n") + "\n\n" + block + "\n"


def update_index_file(repo: Path, path: Path, default_text: str, block_lines: list, log) -> None:
    """자동 구간만 다시 씁니다. 사람이 쓴 부분은 그대로 둡니다(물결표도 손대지 않습니다)."""
    block = [escape_tilde(x) for x in block_lines]
    if path.is_file():
        text = path.read_text(encoding="utf-8")
    else:
        raw, body = split_front_matter(default_text)
        text = (
            escape_tilde(default_text)
            if raw is None
            else "---\n" + raw + "\n---\n" + escape_tilde(body)
        )
    if write_if_changed(path, replace_auto(text, block)):
        log("인덱스 갱신: " + path.relative_to(repo).as_posix())


def unit_table(units: list, link_prefix: callable) -> list:
    rows = []
    for u in units:
        s = u["sections"]
        notes = []
        if u["replay"]:
            notes.append("재현")
        if isinstance(u["revision"], int) and u["revision"] > 1:
            notes.append(f"r{u['revision']}")
        rows.append(
            [
                "[{}]({})".format(u["unit"], link_prefix(u)),
                STATUS_KO.get(u["status"], c(u["status"])),
                STATUS_KO.get(s["market-sector"], "-"),
                STATUS_KO.get(s["kr-stocks"], "-"),
                STATUS_KO.get(s["us-stocks"], "-"),
                ", ".join(notes) if notes else "-",
            ]
        )
    return md_table(["단위", "상태", "시장·섹터", "KR", "US", "비고"], rows)


FAMILY_DEFAULT = (
    "---\n"
    "schema: stock-reports.v1\n"
    "family: daily-briefing\n"
    'title: "일일 브리핑"\n'
    "cadence: daily\n"
    "owner: modeler.reporting.markdown\n"
    "unit_id_format: YYYY/MM/YYYY-MM-DD/\n"
    "---\n"
    "\n"
    "# 일일 브리핑\n"
    "\n"
    "거래일 D의 10:00(KST) 판단 시각에 맞춰 낸 리포트입니다. 하루가 폴더 하나이고 하루 한 "
    "커밋입니다.\n"
    "KR·US 종목 순위, 시장·섹터, 입력 상태를 한 단위에 모읍니다.\n"
    "\n"
    "## 한 단위의 구성\n"
    "\n"
    "| 파일 | 내용 |\n"
    "|---|---|\n"
    "| `README.md` | 단위 상태, 섹션 상태 표, 경고, 대표지수, KR·US 상위 10 |\n"
    "| `market-sector.md` | 시장 상태 표, baseline, 접어 둔 MS1 연구용 점수 |\n"
    "| `kr-stocks.md` | KR 상위 100, 기준일 K, 품질 보류 요약 |\n"
    "| `us-stocks.md` | US LightGBM·Ridge 각 상위 100, 기준 세션 A, 지연 세션 수 |\n"
    "| `data-status.md` | 입력별 기준일·신선도, 실패, 출처 |\n"
    "\n"
    "## 읽는 법\n"
    "\n"
    "- 상태는 `정상`, `부분 완료`, `자료 지연`, `실패` 넷입니다. 규칙은 "
    "[CONVENTIONS.md](../../CONVENTIONS.md)에 있습니다.\n"
    "- 점수는 순위용 점수이며 확률이 아닙니다.\n"
    "- 제목 끝에 `(재현)`이 붙은 단위는 지난 날짜를 나중에 다시 계산한 리포트입니다. 그날 "
    "실제로 냈을 결과와 다를 수 있습니다.\n"
    "- 5세션을 넘게 늦은 입력은 순위 표 대신 사유와 마지막 정상 단위 링크를 적습니다.\n"
    "\n"
    "## 목록\n"
    "\n"
)

MONTH_DEFAULT = (
    "# 일일 브리핑 {year}-{month}\n"
    "\n"
    "이 달에 낸 단위 목록입니다. 아래 표는 코드가 다시 씁니다.\n"
    "\n"
)

ROOT_DEFAULT = (
    "# stock_reports\n"
    "\n"
    "**이 저장소는 본인 전용입니다** (`audience: owner_only`). private이고 본인만 봅니다.\n"
    "협업자를 추가하거나 공개로 바꾸면 여기 올린 내용의 권리 판단이 무효가 됩니다. 그때는 올린 "
    "내용을 먼저 다시 검토하십시오.\n"
    "전체 순위, 원시 피쳐, raw 수치, 서버 경로, 비밀값은 올리지 않습니다.\n"
    "\n"
    "계층, 이름, front matter, 정정 규칙은 [CONVENTIONS.md](CONVENTIONS.md)에 있습니다.\n"
    "용어와 모델 설명은 [reference/](reference/README.md)에 있습니다.\n"
    "\n"
)


def family_infos(repo: Path) -> list:
    infos = []
    base = repo / "reports"
    if not base.is_dir():
        return infos
    for d in sorted(base.iterdir()):
        meta = read_fm(d / "README.md") if d.is_dir() else None
        if meta and meta.get("family") == d.name:
            infos.append({"dir": d.name, "meta": meta})
    return infos


def reindex(repo: Path, log) -> None:
    units = scan_units(repo)
    fam_dir = repo / "reports" / FAMILY
    if not fam_dir.is_dir() and not units:
        fam_dir.mkdir(parents=True, exist_ok=True)
    fam_readme = fam_dir / "README.md"
    recent = list(reversed(units))[:20]
    block = ["### 최근 20개", ""]
    block += (
        unit_table(recent, lambda u: "{}/{}/{}/README.md".format(u["year"], u["month"], u["unit"]))
        if recent
        else ["아직 단위가 없습니다."]
    )
    months = sorted({(u["year"], u["month"]) for u in units}, reverse=True)
    if months:
        block += ["", "### 월별 목록", ""]
        for year, month in months:
            count = sum(1 for u in units if (u["year"], u["month"]) == (year, month))
            block.append(f"- [{year}-{month}]({year}/{month}/README.md) — {count}개")
    update_index_file(repo, fam_readme, FAMILY_DEFAULT, block, log)
    for year, month in months:
        month_units = [u for u in reversed(units) if (u["year"], u["month"]) == (year, month)]
        mblock = unit_table(month_units, lambda u: "{}/README.md".format(u["unit"]))
        update_index_file(
            repo,
            fam_dir / year / month / "README.md",
            MONTH_DEFAULT.format(year=year, month=month),
            mblock,
            log,
        )
    root_block = ["### 리포트 종류", ""]
    rows = []
    latest_lines = []
    for info in family_infos(repo):
        meta = info["meta"]
        if info["dir"] == FAMILY:
            fam_units = units
        else:
            fam_units = []
        latest = fam_units[-1] if fam_units else None
        latest_cell = "-"
        if latest:
            path = "reports/{}/{}/{}/{}/README.md".format(
                info["dir"], latest["year"], latest["month"], latest["unit"]
            )
            latest_cell = "[{}]({})".format(latest["unit"], path)
            latest_lines.append(
                "- {}: {} — {}{}".format(
                    c(meta.get("title") or info["dir"]),
                    latest_cell,
                    STATUS_KO.get(latest["status"], c(latest["status"])),
                    " (재현)" if latest["replay"] else "",
                )
            )
        rows.append(
            [
                "[{}](reports/{}/README.md)".format(
                    c(meta.get("title") or info["dir"]), info["dir"]
                ),
                c(meta.get("cadence") or "-"),
                c(meta.get("owner") or "-"),
                str(len(fam_units)) if info["dir"] == FAMILY else "-",
                latest_cell,
            ]
        )
    root_block += (
        md_table(["종류", "주기", "만드는 코드", "단위 수", "최신 단위"], rows)
        if rows
        else ["아직 리포트 종류가 없습니다."]
    )
    root_block += ["", "### 최신 링크", ""] + (latest_lines or ["아직 단위가 없습니다."])
    update_index_file(repo, repo / "README.md", ROOT_DEFAULT, root_block, log)


# ---------------------------------------------------------------------------
# --init: CONVENTIONS, reference/
# ---------------------------------------------------------------------------
TREE_TEXT = (
    "stock_reports/\n"
    "├── README.md                    저장소 안내, 리포트 종류 표, 최신 링크(자동 구간)\n"
    "├── CONVENTIONS.md               계층·이름·front matter·정정 규칙 (사람이 씀)\n"
    "├── reports/                     날짜가 있는 발행물\n"
    "│   └── daily-briefing/          종류(family)\n"
    "│       ├── README.md            종류 설명 + 최근 20개 (자동 구간)\n"
    "│       └── 2026/\n"
    "│           └── 10/\n"
    "│               ├── README.md    그 달 목록 (자동)\n"
    "│               └── 2026-10-07/  발행 단위(unit) = 폴더\n"
    "│                   ├── README.md          요약 (폴더를 열면 GitHub가 바로 보여 줌)\n"
    "│                   ├── market-sector.md   시장·섹터\n"
    "│                   ├── kr-stocks.md       KR 종목 순위\n"
    "│                   ├── us-stocks.md       US 종목 순위 (모델 2개)\n"
    "│                   └── data-status.md     입력 기준일·신선도·실패·provenance\n"
    "└── reference/                   계속 고쳐 쓰는 참고 문서\n"
    "    ├── README.md\n"
    "    ├── models/                  모델 설명, 파일 하나 = model_id 하나\n"
    "    │   ├── kr_daily_h20_v1.md\n"
    "    │   ├── us_exploratory_20260929_r1_lightgbm.md\n"
    "    │   ├── us_exploratory_20260929_r1_ridge.md\n"
    "    │   └── ms1_market_sector.md\n"
    "    └── glossary.md              용어 (K, A, stale, 순위용 점수 등)"
)

FM_EXAMPLE = (
    "---\n"
    "schema: stock-reports.v1\n"
    "family: daily-briefing\n"
    "unit: 2026-10-07\n"
    "section: kr-stocks          # 요약 파일은 summary\n"
    "title: KR 종목 순위 — 2026-10-07\n"
    "status: partial             # ok | partial | stale | failed\n"
    "markets: [KR]\n"
    "decision_at: 2026-10-07T10:00:00+09:00\n"
    "generated_at: 2026-10-07T10:03:12+09:00\n"
    "data_asof:\n"
    "  kr_features: 2026-10-06\n"
    "models: [kr_daily_h20_v1]\n"
    "revision: 1\n"
    "source:\n"
    "  release: r2026xxxx\n"
    "  report_sha256: <64자리>\n"
    "---"
)

CONVENTIONS_TEXT = (
    "# CONVENTIONS\n"
    "\n"
    "> 이 저장소는 본인 전용입니다 (`audience: owner_only`).\n"
    "\n"
    "이 문서는 저장소 규칙입니다. 규칙이 바뀔 때만 고칩니다. 리포트 종류 목록은 여기에 두지 "
    "않고 루트 `README.md`의 자동 구간에 있습니다.\n"
    "\n"
    "## 이 저장소는 본인 전용입니다\n"
    "\n"
    "이 저장소는 private이고 본인만 봅니다(`audience: owner_only`). 협업자를 추가하거나 공개로 "
    "바꾸면 여기 올린 내용의 권리 판단이 무효가 됩니다. 그때는 올린 내용을 먼저 다시 "
    "검토하십시오. 전체 순위, 원시 피쳐, raw 수치, 서버 경로, 비밀값은 올리지 않습니다.\n"
    "\n"
    "## 구조\n"
    "\n"
    "- `reports/`는 날짜가 있는 발행물입니다. 발행한 뒤 고치지 않습니다.\n"
    "- `reports/<family>/`는 리포트 종류입니다. 이름은 영문 소문자 kebab-case로, 주제를 "
    "나타냅니다.\n"
    "- `reports/<family>/` 아래는 시간으로 나눕니다. 단위 id 형식은 종류 README의 "
    "`unit_id_format`에 적혀 있습니다.\n"
    "- 발행 단위는 폴더입니다. 진입 파일은 `README.md`이고, 폴더 이름에 완전한 날짜가 "
    "들어갑니다.\n"
    "- `reference/`는 계속 고쳐 쓰는 문서입니다. `reference/models/`에는 `model_id` 하나당 "
    "파일 하나를 둡니다.\n"
    "- 1단계 폴더는 `reports/`와 `reference/` 둘뿐입니다. 새 최상위 폴더를 만들지 않습니다.\n"
    "\n"
    "```text\n"
    "{tree}\n"
    "```\n"
    "\n"
    "## 단위 id 형식\n"
    "\n"
    "- 일간은 `YYYY/MM/YYYY-MM-DD/`입니다.\n"
    "- 주간은 `YYYY/YYYY-Www/`입니다.\n"
    "- 월간은 `YYYY/YYYY-MM/`입니다.\n"
    "- 수시는 `YYYY/MM/YYYY-MM-DD-<slug>/`입니다.\n"
    "\n"
    "## 파일 이름\n"
    "\n"
    "- 섹션 파일은 `<범위>-<주제>.md`입니다. 범위는 `kr`, `us`, `market`, `global` 중 "
    "하나입니다.\n"
    "- 범위가 없는 공통 섹션은 `data-status.md`입니다.\n"
    "- 한 파일은 1MB를 넘기지 않습니다.\n"
    "\n"
    "## 넣는 것과 넣지 않는 것\n"
    "\n"
    "- markdown 표와 글만 넣습니다. parquet, CSV, JSON 같은 데이터 파일은 넣지 않습니다.\n"
    "- 그림은 필요해질 때까지 쓰지 않습니다. 쓰게 되면 단위 폴더 안 `assets/`에 SVG만, 개당 "
    "200KB 이하로 둡니다.\n"
    "- 링크는 상대 경로만 씁니다.\n"
    "- 서버 절대경로, 호스트명, 비밀값은 넣지 않습니다. 검증기가 막습니다.\n"
    "- 범위나 근사값의 물결표는 `\\~`로 씁니다(예: `50KB\\~5MB`). 렌더러에 따라 물결표 두 "
    "개 사이가 취소선으로 그려지기 때문입니다.\n"
    "\n"
    "## front matter\n"
    "\n"
    "단위 폴더의 모든 파일은 YAML front matter로 시작합니다. 아래 필드를 모두 적습니다.\n"
    "\n"
    "```yaml\n"
    "{fm}\n"
    "```\n"
    "\n"
    "| 필드 | 뜻 |\n"
    "|---|---|\n"
    "| `schema` | 형식 버전. 지금은 `stock-reports.v1` |\n"
    "| `family`, `unit` | 리포트 종류와 발행 단위 id. 폴더 이름과 같습니다 |\n"
    "| `section` | 섹션 이름. 요약 파일(`README.md`)은 `summary` |\n"
    "| `title` | 화면에 보이는 제목 |\n"
    "| `status` | `ok`, `partial`, `stale`, `failed` 중 하나 |\n"
    "| `markets` | 이 파일이 다루는 시장 |\n"
    "| `decision_at`, `generated_at` | 판단 기준 시각, 파일을 만든 시각 |\n"
    "| `data_asof` | 입력별 기준일 |\n"
    "| `models` | 쓴 `model_id` 목록 |\n"
    "| `revision` | 같은 단위를 고쳐 다시 낸 횟수. 처음은 1 |\n"
    "| `source` | 만든 release와 정본 내부 report의 sha256 |\n"
    "| `historical_replay` | 지난 날짜를 나중에 다시 계산한 리포트일 때만 `true`로 적습니다. "
    "모든 파일에 같이 적고, 제목 끝에 `(재현)`을 붙입니다 |\n"
    "\n"
    "종류 README의 front matter에는 위 필드 대신 `schema`, `family`, `cadence`, `owner`, "
    "`unit_id_format`을 적습니다.\n"
    "\n"
    "## 상태 값\n"
    "\n"
    "내부 report의 상태는 여섯 가지이고, front matter의 `status`는 네 가지입니다. 아래 표로 "
    "바꿉니다.\n"
    "\n"
    "| 내부 상태 | front matter `status` | 뜻 |\n"
    "|---|---|---|\n"
    "| `ok` | `ok` | 정상 |\n"
    "| `partial` | `partial` | 일부만 확인됨. KR은 관리종목·거래정지 상태를 확인하지 못해 늘 "
    "이 값입니다 |\n"
    "| `stale` | `stale` | 입력이 늦음. 기준일과 지연 세션 수를 적습니다 |\n"
    "| `withheld` | `failed` | 공개 보류. 사유를 적습니다 |\n"
    "| `unavailable` | `failed` | 자료 없음. 사유를 적습니다 |\n"
    "| `failed` | `failed` | 실패. 사유를 적습니다 |\n"
    "\n"
    "- 단위 상태(`README.md`, `data-status.md`)는 섹션 상태에서 다시 계산합니다. 시장·섹터, "
    "KR, US가 모두 `ok`면 `ok`, 모두 `failed`면 `failed`, 나머지는 `partial`입니다.\n"
    "- 한 섹션 안에 모델이 여럿이면(US) 모두 같으면 그 값, 모두 `failed`면 `failed`, 하나라도 "
    "`failed`·`partial`이 섞이면 `partial`, 정상과 지연만 섞이면 `stale`입니다.\n"
    "- 입력이 5세션을 넘게 늦으면 순위 표를 내지 않습니다. 섹션 상태는 `stale`로 두고 사유와 "
    "마지막 정상 단위 링크를 적습니다.\n"
    "- 섹션이 실패해도 단위는 만듭니다. 실패한 섹션 파일에 사유를 적습니다.\n"
    "\n"
    "## 자동 구간\n"
    "\n"
    "- `<!-- auto:begin index -->`와 `<!-- auto:end index -->` 사이는 코드가 다시 씁니다. "
    "손으로 고치지 않습니다.\n"
    "- 이 구간은 저장소의 front matter만 읽어 만들므로 같은 저장소 상태에서는 같은 결과가 "
    "나옵니다.\n"
    "\n"
    "## 정정과 재실행\n"
    "\n"
    "- 같은 단위를 다시 만들었는데 내용이 같으면 커밋하지 않습니다.\n"
    "- 내용이 다르면 `revision`을 올리고, 요약 맨 위에 정정 줄(이유와 이전 판 커밋)을 "
    "넣습니다. 커밋 메시지는 `<family> <unit> r<revision>: <사유>` 형식입니다.\n"
    "- 일일 실행은 오늘 단위보다 이전 단위를 건드리지 않습니다. 이전 단위를 고칠 때는 수동으로 "
    "`--correct <unit> --reason`을 씁니다.\n"
    "- 단위를 지우지 않습니다.\n"
    "\n"
    "## 쓰는 쪽의 경계\n"
    "\n"
    "- 리포트를 만드는 코드는 자기 종류 폴더 아래와 공유 자동 구간, 자기 모델의 카드만 "
    "씁니다.\n"
    "- 다른 종류 폴더와 사람이 쓰는 문서(`CONVENTIONS.md`, `reference/glossary.md`)는 코드가 "
    "고치지 않습니다.\n"
    "\n"
    "## 새 종류를 더할 때\n"
    "\n"
    "- `reports/<새 family>/README.md`를 만들고 front matter에 `cadence`, `owner`, "
    "`unit_id_format`을 적습니다.\n"
    "- 만드는 코드와 검증기의 경로 allowlist에 종류 이름을 더합니다.\n"
    "- 이 문서는 규칙이 바뀔 때만 고칩니다.\n"
)

GLOSSARY_ROWS = (
    ("D", "리포트 날짜입니다. 판단 시각은 D 10:00 KST입니다."),
    (
        "cutoff",
        "입력을 쓸 수 있는 마지막 시각입니다. D 09:30 KST입니다. 이 시각보다 늦게 끝난 입력은 "
        "쓰지 않습니다.",
    ),
    ("K", "KR 직전 거래일입니다. KR 입력의 기준일이 K와 같으면 최신입니다."),
    (
        "K′",
        "K보다 이른 KR 기준일입니다. K 입력이 완결되지 않았거나 KR 준비가 실패했을 때 가장 "
        "최근의 유효한 기준일을 씁니다.",
    ),
    ("U", "미국의 마지막 완료 세션입니다."),
    ("E", "그날 쓰려고 기대한 미국 기준 세션입니다."),
    (
        "A",
        "미국 입력의 기준 세션입니다. 리포트가 실제로 쓴 세션입니다. 기대한 세션보다 이르면 "
        "A′(더 이른 대체 기준 세션)를 쓴 것이고, 지연 세션 수로 차이를 보입니다.",
    ),
    (
        "지연 세션 수",
        "기준일이 최신 세션보다 몇 세션 이른지 셉니다. 미국은 A와 U의 차이입니다. 3세션 "
        "이상이면 경고를 붙이고, 5세션을 넘으면 순위 표를 내지 않습니다.",
    ),
    (
        "stale",
        "입력이 늦었다는 상태입니다. 값이 틀린 것이 아니라 최신이 아닙니다. 기준일과 지연 세션 "
        "수를 함께 적습니다.",
    ),
    (
        "partial",
        "일부만 확인된 상태입니다. KR은 현재 관리종목·거래정지 상태를 확인하지 못해 늘 이 "
        "값입니다.",
    ),
    ("failed", "순위나 표를 내지 못한 상태입니다. 사유를 적습니다."),
    (
        "순위용 점수",
        "종목을 줄 세우는 데만 쓰는 점수입니다. 확률도 신뢰도도 아닙니다. 같은 날 같은 모델 "
        "안에서만 비교하고, 모델이 다르면 비교하지 않습니다.",
    ),
    ("상위 N", "전체 순위 가운데 위에서 N개입니다. 기본 100입니다. 나머지 순위는 올리지 않습니다."),
    (
        "품질 보류",
        "KR 상위 순위에서 거래정지·가격 급변·주식수 변화 등이 의심되어 검토가 필요한 행입니다. "
        "원래 순위는 바꾸지 않습니다.",
    ),
    (
        "historical_replay",
        "지난 날짜를 나중에 다시 계산한 리포트입니다. 제목 끝에 `(재현)`이 붙습니다. D 09:30 "
        "뒤에 들어온 데이터가 섞일 수 있어 그날 실제 결과와 다를 수 있습니다.",
    ),
    ("단위(unit)", "발행 하나입니다. 일일 브리핑은 하루가 폴더 하나입니다."),
    ("revision", "같은 단위를 고쳐 다시 낸 횟수입니다. 처음은 1입니다."),
    (
        "baseline",
        "MS1에서 연구용 모델과 비교하려고 둔 기준선입니다. 시장 상태 표 옆에 원래 단위로 적습니다.",
    ),
    (
        "MS1",
        "시장·섹터 연구용 모델입니다. 판정 6개 중 통과 0이라 채택하지 않았고 의사결정에 쓰지 "
        "않습니다. 설명은 [ms1_market_sector](models/ms1_market_sector.md)에 있습니다.",
    ),
    (
        "total_return / price_only",
        "수익률의 기준입니다. total_return은 배당을 포함하고(US), price_only는 배당을 뺍니다(KR).",
    ),
    ("PIT", "그 시점에 알 수 있던 정보만 쓴다는 원칙입니다."),
)


def init_repo(repo: Path, model_cards_path: Path, log) -> None:
    repo.mkdir(parents=True, exist_ok=True)

    def create(path: Path, text: str) -> None:
        if path.exists():
            log("이미 있어 건너뜁니다: " + str(path.relative_to(repo)))
            return
        write_if_changed(path, text)
        log("만들었습니다: " + str(path.relative_to(repo)))

    create(
        repo / "CONVENTIONS.md",
        escape_tilde(CONVENTIONS_TEXT.replace("{tree}", TREE_TEXT).replace("{fm}", FM_EXAMPLE)),
    )
    create(
        repo / "reference" / "README.md",
        escape_tilde(
            "# 참고 문서\n\n계속 고쳐 쓰는 문서입니다. 리포트 단위와 달리 발행 뒤에도 고칩니다.\n\n"
            "- [용어](glossary.md)\n- 모델 설명 (파일 하나가 `model_id` 하나입니다)\n"
            "  - [kr_daily_h20_v1](models/kr_daily_h20_v1.md)\n"
            "  - "
            "[us_exploratory_20260929_r1_lightgbm](models/us_exploratory_20260929_r1_lightgbm.md)\n"
            "  - [us_exploratory_20260929_r1_ridge](models/us_exploratory_20260929_r1_ridge.md)\n"
            "  - [ms1_market_sector](models/ms1_market_sector.md)\n"
        ),
    )
    glossary = ["# 용어", "", "리포트에 나오는 용어입니다.", "", "| 용어 | 뜻 |", "| --- | --- |"]
    glossary += ["| {} | {} |".format(c(k), v.replace("|", "\\|")) for k, v in GLOSSARY_ROWS]
    create(repo / "reference" / "glossary.md", escape_tilde("\n".join(glossary) + "\n"))
    cards = {}
    if model_cards_path.is_file():
        cards = json.loads(model_cards_path.read_text(encoding="utf-8"))
    else:
        log("경고: model-cards.json이 없어 모델 카드를 건너뜁니다: " + str(model_cards_path.name))
    ensure_model_cards(repo, cards, log)
    reindex(repo, log)


def ensure_model_cards(repo: Path, cards: dict, log) -> list:
    """없는 모델 카드만 만듭니다. 이미 있는 카드는 사람이 고쳤을 수 있어 건드리지 않습니다.

    만든 파일의 저장소 상대 경로 목록을 돌려줍니다. 코드가 쓸 수 있는 카드는 자기 모델의 것뿐입니다.
    """
    created = []
    wanted = [(model_id, model_card_text(model_id, cards[model_id])) for model_id in sorted(cards)]
    wanted.append((MS_MODEL, ms1_card_text()))
    for model_id, body in wanted:
        if not re.fullmatch(r"[A-Za-z0-9._\-]+", model_id):
            raise InputError("모델 id에 쓸 수 없는 문자가 있습니다.")
        path = repo / "reference" / "models" / (model_id + ".md")
        if path.exists() or path.is_symlink():
            if log:
                log("이미 있어 건너뜁니다: " + path.relative_to(repo).as_posix())
            continue
        write_if_changed(path, body)
        created.append(path.relative_to(repo).as_posix())
        if log:
            log("만들었습니다: " + path.relative_to(repo).as_posix())
    return created


def model_card_text(model_id: str, card: dict) -> str:
    meta = {
        "schema": SCHEMA,
        "kind": "model-card",
        "model_id": model_id,
        "title": card.get("title", model_id),
    }
    lines = ["# " + c(card.get("title", model_id)), "", code(model_id), ""]
    for key, label in (
        ("summary", "소개"),
        ("target", "예측 대상"),
        ("scope", "학습·적용 범위"),
        ("limitations", "주의할 점"),
    ):
        if isinstance(card.get(key), str):
            lines += ["## " + label, "", scrub(card[key]), ""]
    lines += [
        "## 점수 읽는 법",
        "",
        "리포트의 점수는 순위용 점수 — 확률 아님입니다.",
        "",
        "출처: 운영 모델 카드(`model-cards.json`)를 옮겼습니다.",
        "",
    ]
    return front_matter(meta) + "\n" + escape_tilde("\n".join(lines))


def ms1_card_text() -> str:
    meta = {
        "schema": SCHEMA,
        "kind": "model-card",
        "model_id": MS_MODEL,
        "title": "MS1 시장·섹터 (연구용)",
    }
    lines = [
        "# MS1 시장·섹터 (연구용)",
        "",
        code(MS_MODEL),
        "",
        "> 판정은 6개 중 통과 0, 보류 1, 실패 5입니다. 채택할 모델이 없고 두 시장 모두 "
        "baseline을 유지합니다 (2026-09-30 결과). "
        "의사결정에 쓰지 않습니다.",
        "",
        "## 대상",
        "",
        "자산 14개입니다. US 7개(SPY, QQQ, XLF, XLV, XLI, XLE, XLK)는 배당을 포함한 "
        "수익(`total_return`)을, "
        "KR 7개(코스피, 코스닥, KRX 은행, KRX 헬스케어, KRX 기계장비, KRX 에너지화학, KRX "
        "반도체)는 배당을 뺀 수익(`price_only`)을 씁니다.",
        "",
        "## 판정 요약",
        "",
        "| 대상 (주 모델) | US | KR (`price_only`) |",
        "| --- | --- | --- |",
        "| Opportunity (Ridge, alpha 10) | 실패 — skill -0.053 | 실패 — skill -0.091 |",
        "| 섹터 상대 선택 (Ridge) | 보류 — IC +0.100. baseline +0.093과 거의 같음 | 실패 |",
        "| Stability (L2 Logistic, C 0.1) | 실패 — Brier skill -0.255 | 실패 — Brier skill "
        "-0.118 |",
        "",
        "## 읽는 법",
        "",
        "- Opportunity의 skill은 과거 평균 baseline 대비 값입니다. 0보다 작으면 baseline보다 "
        "나쁩니다.",
        "- Stability는 예측이 과신입니다. US는 예측이 0.02\\~0.70에 퍼졌는데 실제 사건률은 "
        "0.13\\~0.28이었습니다. 확률로 읽지 않습니다.",
        "- 비교 후보 LightGBM도 통과한 것이 없습니다.",
        "- 모델 점수는 `validation_status=research`이고 의사결정에 쓰지 않는 것으로 정해져 "
        "있습니다.",
        "- Ridge alpha, Logistic C, 피쳐를 바꾸거나 새 변환을 만들려면 새 사전등록 문서가 "
        "있어야 합니다.",
        "",
        "## 일일 리포트에서",
        "",
        "- 본문에는 시장 상태 표와 baseline 두 값만 둡니다. 연구용 모델 점수는 "
        "`market-sector.md` 맨 아래에 접어 둡니다.",
        "- Opportunity 점수는 0\\~100입니다. 모델 예측의 이전 fold OOF 백분위입니다.",
        "- Stability 점수는 0\\~100입니다. `100 × (1 − p̂)`입니다.",
        "- baseline은 원래 단위(%)로 냅니다. 0\\~100 변환은 정의가 없어 만들지 않았습니다.",
        "- 지수 종가 수준은 올리지 않습니다. 파생값만 올립니다.",
        "",
    ]
    return front_matter(meta) + "\n" + escape_tilde("\n".join(lines))


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------
LINK_RE = re.compile(r"(?<![!\\])\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
UNIT_PATH_RE = re.compile(
    rf"^reports/{re.escape(FAMILY)}/(\d{{4}})/(\d{{2}})/(\d{{4}}-\d{{2}}-\d{{2}})/([^/]+)$"
)
ASSET_PATH_RE = re.compile(
    rf"^reports/{re.escape(FAMILY)}/(\d{{4}})/(\d{{2}})/(\d{{4}}-\d{{2}}-\d{{2}})"
    r"/assets/([A-Za-z0-9._\-]+\.svg)$"
)
FAMILY_FM_FIELDS = ("schema", "family", "cadence", "owner", "unit_id_format")
ALLOWED_FAMILIES = (FAMILY,)  # 새 종류를 더할 때 여기에 이름 한 줄을 더합니다(01 문서 §10).
SVG_MAX_BYTES = 200 * 1024
SVG_FORBIDDEN = re.compile(r"<script|javascript:|\son\w+\s*=", re.IGNORECASE)


def strip_code(text: str) -> str:
    out, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(re.sub(r"`+[^`]*`+", "", line))
    return "\n".join(out)


def allowed_path(rel: str) -> bool:
    return (
        rel in ("README.md", "CONVENTIONS.md")
        or rel.startswith("reference/")
        or any(rel.startswith(f"reports/{family}/") for family in ALLOWED_FAMILIES)
    )


def check_unit_fm(rel: str, meta: dict, match: re.Match) -> list:
    out = []
    year, month, unit, name = match.groups()
    if name not in FILE_SECTIONS:
        return [f"{rel}: 알 수 없는 섹션 파일 이름입니다"]
    for field in UNIT_FM_FIELDS:
        if field not in meta:
            out.append(f"{rel}: front matter 필드 없음: {field}")
    if out:
        return out
    if meta["schema"] != SCHEMA:
        out.append(f"{rel}: schema가 {SCHEMA}가 아닙니다")
    if meta["family"] != FAMILY:
        out.append(f"{rel}: family가 경로({FAMILY})와 다릅니다")
    if meta["unit"] != unit:
        out.append(f"{rel}: unit이 경로({unit})와 다릅니다")
    if unit[0:4] != year or unit[5:7] != month:
        out.append(f"{rel}: 단위 폴더가 연·월 폴더와 맞지 않습니다")
    if meta["section"] != FILE_SECTIONS[name]:
        out.append(f"{rel}: section이 파일 이름({FILE_SECTIONS[name]})과 다릅니다")
    if meta["status"] not in STATUS_KO:
        out.append(f"{rel}: status가 ok·partial·stale·failed가 아닙니다")
    if not isinstance(meta["title"], str) or not meta["title"]:
        out.append(f"{rel}: title이 비어 있습니다")
    markets = meta["markets"]
    if not isinstance(markets, list) or not markets or not set(markets) <= {"KR", "US"}:
        out.append(f"{rel}: markets는 KR·US의 목록이어야 합니다")
    stamp = parse_dt(meta["decision_at"]) if isinstance(meta["decision_at"], str) else None
    if stamp is None or stamp.astimezone(KST).strftime("%Y-%m-%dT%H:%M:%S") != unit + "T10:00:00":
        out.append(f"{rel}: decision_at이 단위 날짜 10:00 KST가 아닙니다")
    if not (isinstance(meta["generated_at"], str) and parse_dt(meta["generated_at"])):
        out.append(f"{rel}: generated_at이 시간대가 있는 시각이 아닙니다")
    if not isinstance(meta["data_asof"], dict) or any(
        good_date(v) is None for v in meta["data_asof"].values()
    ):
        out.append(f"{rel}: data_asof 값이 날짜가 아닙니다")
    if not isinstance(meta["models"], list):
        out.append(f"{rel}: models가 목록이 아닙니다")
    if not isinstance(meta["revision"], int) or meta["revision"] < 1:
        out.append(f"{rel}: revision은 1 이상의 정수여야 합니다")
    source = meta["source"]
    if (
        not isinstance(source, dict)
        or not source.get("release")
        or not is_hex64(source.get("report_sha256"))
    ):
        out.append(f"{rel}: source.release와 64자리 source.report_sha256이 있어야 합니다")
    return out


def iter_repo_files(repo: Path) -> list:
    """저장소의 파일과 symlink의 상대 경로를 이름 순서로 돌려줍니다. .git은 건너뜁니다.

    symlink는 디렉터리를 가리켜도 항목 하나로 돌려주고 따라 들어가지 않습니다.
    """
    found = []
    for current, dirs, names in os.walk(repo, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (current == str(repo) and d == ".git"))
        for name in list(dirs):
            if (Path(current) / name).is_symlink():
                found.append((Path(current) / name).relative_to(repo).as_posix())
                dirs.remove(name)
        for name in names:
            found.append((Path(current) / name).relative_to(repo).as_posix())
    return sorted(found)


def link_violations(rel: str, text: str, exists) -> list:
    """상대 링크를 검사합니다. `exists(resolved)`가 가리키는 파일이 있는지 알려줍니다."""
    out = []
    for target in LINK_RE.findall(strip_code(text)):
        if target.startswith("#"):
            continue
        if re.match(r"^[A-Za-z][A-Za-z0-9+.\-]*:", target):
            continue  # 외부 링크는 위반으로 세지 않습니다(CONVENTIONS는 상대 경로를 권합니다)
        path_part = target.split("#", 1)[0].split("?", 1)[0]
        if path_part.startswith("/"):
            out.append(f"{rel}: 절대 경로 링크입니다: {target}")
            continue
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(rel), path_part))
        if resolved.startswith("..") or not exists(resolved):
            out.append(f"{rel}: 링크 대상이 없습니다: {target}")
    return out


def md_file_violations(rel: str, text: str, exists, *, allow_synthetic: bool = False) -> tuple:
    """markdown 파일 하나의 내용을 검사합니다. (위반 목록, front matter | None)을 돌려줍니다."""
    violations = []
    for label, pattern in FORBIDDEN_PATTERNS:
        if pattern.search(text):
            violations.append(f"{rel}: 금지 문자열이 있습니다 ({label})")
    meta = None
    match = UNIT_PATH_RE.match(rel)
    if match:
        err = None
        try:
            meta, _ = parse_front_matter(text)
        except ValueError as exc:
            err = str(exc)
        if err:
            violations.append(f"{rel}: front matter를 읽지 못했습니다 ({err})")
        elif meta is None:
            violations.append(f"{rel}: front matter가 없습니다")
        else:
            violations += check_unit_fm(rel, meta, match)
            if meta.get("synthetic_fixture") is True and not allow_synthetic:
                violations.append(f"{rel}: 합성 fixture 단위는 명시 허용이 있어야 올립니다")
    elif rel == f"reports/{FAMILY}/README.md":
        try:
            meta, _ = parse_front_matter(text)
        except ValueError:
            meta = None
        if meta is None:
            violations.append(f"{rel}: 종류 README에 front matter가 없습니다")
        else:
            for field in FAMILY_FM_FIELDS:
                if field not in meta or meta[field] in ("", None):
                    violations.append(f"{rel}: front matter 필드 없음: {field}")
            if meta.get("schema") != SCHEMA:
                violations.append(f"{rel}: schema가 {SCHEMA}가 아닙니다")
            if meta.get("family") != FAMILY:
                violations.append(f"{rel}: family가 폴더 이름과 다릅니다")
    violations += link_violations(rel, text, exists)
    return violations, meta


def unit_set_violations(unit: str, metas: dict) -> list:
    """단위 하나의 파일 다섯 개가 다 있고 historical_replay 표시가 같은지 봅니다."""
    out = []
    missing = [n for n in FILE_SECTIONS if n not in metas]
    if missing:
        out.append(f"{unit_rel(unit)}: 단위에 파일이 빠졌습니다: {', '.join(missing)}")
    replays = {bool((m or {}).get("historical_replay")) for m in metas.values()}
    if len(replays) > 1:
        out.append(f"{unit} 단위: historical_replay가 파일마다 다릅니다")
    return out


def validate(repo: Path | str, paths=None, *, allow_synthetic: bool = False) -> list:
    """저장소 트리를 검사해 위반 목록(빈 목록이면 통과)을 돌려줍니다.

    `paths`(저장소 상대 경로 목록)를 주면 그 파일만 검사합니다. 이번 커밋이 바꾸는 파일만
    보려는 publisher가 씁니다. 사람이 고친 다른 문서 때문에 일일 게시가 막히지 않게 하려는 것입니다.
    단위를 건드렸으면 그 단위의 파일 구성은 디스크 기준으로 전부 봅니다.
    """
    repo = Path(repo)
    violations: list = []
    if not repo.is_dir():
        return ["저장소 디렉터리가 없습니다"]
    names = iter_repo_files(repo) if paths is None else sorted(set(paths))
    touched_units: set = set()

    def exists(resolved: str) -> bool:
        return (repo / resolved).exists()

    for rel in names:
        if rel == ".git" or rel.startswith(".git/"):
            continue
        path = repo / rel
        if path.is_symlink():
            violations.append(f"{rel}: symlink는 허용하지 않습니다")
            continue
        if not path.exists() or path.is_dir():
            continue
        if not allowed_path(rel):
            violations.append(f"{rel}: 허용 경로 밖의 파일입니다")
            continue
        asset = ASSET_PATH_RE.match(rel)
        if asset:
            touched_units.add(asset.group(3))
            raw = path.read_bytes()
            if len(raw) > SVG_MAX_BYTES:
                violations.append(f"{rel}: SVG가 200KB를 넘습니다")
            elif SVG_FORBIDDEN.search(raw.decode("utf-8", "replace")):
                violations.append(f"{rel}: SVG에 스크립트나 이벤트 속성이 있습니다")
            continue
        if not rel.endswith(".md"):
            violations.append(f"{rel}: .md만 올릴 수 있습니다")
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            violations.append(f"{rel}: 파일이 1MB를 넘습니다")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"{rel}: UTF-8이 아닙니다")
            continue
        found, _ = md_file_violations(rel, text, exists, allow_synthetic=allow_synthetic)
        violations += found
        match = UNIT_PATH_RE.match(rel)
        if match:
            touched_units.add(match.group(3))
    for unit in sorted(touched_units):
        metas = {}
        for name in FILE_SECTIONS:
            file_path = unit_dir(repo, unit) / name
            if file_path.is_file() and not file_path.is_symlink():
                metas[name] = read_fm(file_path) or {}
        violations += unit_set_violations(unit, metas)
    return violations


def validate_unit_files(unit: str, files: dict, *, allow_synthetic: bool = False) -> list:
    """단위 하나를 저장소 없이 검사합니다(로컬 단계 L3). 단위 밖으로 나가는 링크는 모양만 봅니다.

    front matter, 파일 크기, 단위 안 상대 링크, 서버 경로·비밀값을 봅니다. 단위 밖 링크의 대상은
    저장소 트리가 있어야 알 수 있으므로 동기화 단계의 트리 검증이 맡습니다.
    """
    prefix = unit_rel(unit)
    inside = {f"{prefix}/{name}" for name in files}
    violations: list = []
    metas = {}

    def exists(resolved: str) -> bool:
        if resolved.startswith(prefix + "/"):
            return resolved in inside
        return allowed_path(resolved)

    for name in sorted(files):
        rel = f"{prefix}/{name}"
        text = files[name]
        if len(text.encode("utf-8")) > MAX_FILE_BYTES:
            violations.append(f"{rel}: 파일이 1MB를 넘습니다")
            continue
        found, meta = md_file_violations(rel, text, exists, allow_synthetic=allow_synthetic)
        violations += found
        metas[name] = meta or {}
    violations += unit_set_violations(unit, metas)
    return violations


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="일일 브리핑 envelope를 stock_reports markdown으로 바꾸고 검증합니다 (수동용)."
    )
    parser.add_argument("--repo", required=True, type=Path, help="stock_reports checkout 경로")
    parser.add_argument("--envelope", type=Path, help="report-D.json 또는 replay envelope.json")
    parser.add_argument(
        "--market-sector", type=Path, help="시장·섹터 JSON (없으면 섹션을 failed로 씁니다)"
    )
    parser.add_argument("--release", help="서빙 release id")
    parser.add_argument("--generated-at", help="ISO 시각(시간대 포함). 없으면 지금")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--init",
        action="store_true",
        help="CONVENTIONS.md, reference/, 인덱스를 만듭니다(있으면 덮지 않음)",
    )
    parser.add_argument("--model-cards", type=Path, default=None, help="model-cards.json 경로")
    parser.add_argument(
        "--reason", help="같은 단위를 다른 내용으로 다시 낼 때 정정 사유(revision을 올립니다)"
    )
    parser.add_argument("--previous-commit", help="정정 줄에 적을 이전 판 커밋")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="내용이 달라도 같은 revision으로 덮습니다(첫 push 전 손볼 때만)",
    )
    parser.add_argument("--reindex", action="store_true", help="인덱스 자동 구간만 다시 씁니다")
    parser.add_argument("--validate", action="store_true", help="검증만 합니다")
    parser.add_argument("--no-validate", action="store_true", help="쓴 뒤 검증을 건너뜁니다")
    parser.add_argument(
        "--allow-synthetic", action="store_true", help="합성 fixture 단위를 검증에서 허용합니다"
    )
    args = parser.parse_args(argv)

    def log(message: str) -> None:
        print(message)

    reset_scrub_hits()
    try:
        if not args.validate:
            if args.init:
                if args.model_cards is None:
                    raise InputError("--init에는 --model-cards가 필요합니다.")
                init_repo(args.repo, args.model_cards, log)
            if args.reindex:
                reindex(args.repo, log)
            if args.envelope:
                if not args.release:
                    raise InputError("--envelope에는 --release가 필요합니다.")
                if args.top_n < 1:
                    raise InputError("--top-n은 1 이상이어야 합니다.")
                generate(
                    args.repo,
                    args.envelope,
                    args.market_sector,
                    args.release,
                    args.generated_at,
                    args.top_n,
                    args.reason,
                    args.previous_commit,
                    args.overwrite,
                    log,
                )
            elif not (args.init or args.reindex):
                raise InputError("--envelope, --init, --reindex, --validate 중 하나가 필요합니다.")
    except InputError as exc:
        sys.stderr.write(f"오류: {exc}\n")
        return 2
    if SCRUB_HITS:
        sys.stderr.write(
            f"경고: 서버 경로·호스트명·키 모양 문자열 {len(SCRUB_HITS)}건을 본문에서 지웠습니다.\n"
        )
    if args.no_validate and not args.validate:
        return 0
    problems = validate(args.repo, allow_synthetic=args.allow_synthetic)
    if problems:
        sys.stderr.write(f"검증 위반 {len(problems)}건\n")
        for item in problems:
            sys.stderr.write(f" - {item}\n")
        return 1
    print("검증 통과")
    return 0


if __name__ == "__main__":
    sys.exit(main())
