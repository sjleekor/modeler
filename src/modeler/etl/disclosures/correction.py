"""정정 공시(c) 원문 파서: zip 바이트 또는 본문 문자열 → dict (계획 13 §3).

정정 공시 맨 앞에는 정정 표(정정일자·정정관련 공시서류·제출일·사유·정정항목 전/후)가 있고,
그 뒤에 정정된 본문 전체가 다시 실려 있다. 이 모듈은 앞의 표와 맨 아래 ``※ 관련공시`` 링크만
읽는다. 값은 칸에 적힌 그대로 두고(``*_raw``) 해석은 최소로 한다. 텍스트 모델·LLM은 쓰지 않는다.

입출력이 분리된 순수 함수다. 파일·네트워크 접근이 없다.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import re

from modeler.etl.disclosures.docparse import key, norm, read_zip_bytes, rows_of

CORRECTION_PARSER_VERSION = "disclosure_correction_1.0.0"

_DATE = re.compile(r"(?<!\d)((?:19|20)\d{2})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})(?!\d)")
_DATE8 = re.compile(r"(?<!\d)((?:19|20)\d{2})(\d{2})(\d{2})(?!\d)")
_DATE_YY = re.compile(r"^\s*(\d{2})\s*\.\s*(\d{1,2})\s*\.\s*(\d{1,2})\s*\.?\s*$")
_DATE_Y4MD = re.compile(r"^\s*(20\d{2})\s*\.\s*(\d{2})(\d{2})\s*$")
_NUM = re.compile(r"^[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?$|^[-+]?\d+(?:\.\d+)?$")
_ANCHOR = re.compile(r"<a\b[^>]*?href\s*=\s*[\"']([^\"']*)[\"'][^>]*>(.*?)</a>", re.I | re.S)
_RCPNO = re.compile(r"rcpno=(\d{14})")
_TAG = re.compile(r"<[^>]+>")


def _valid(y: int, m: int, d: int) -> str | None:
    try:
        return dt.date(y, m, d).isoformat() if 1990 <= y <= 2100 else None
    except ValueError:
        return None


def parse_dates(s: str | None) -> list[str]:
    """글자 안의 날짜를 나온 순서대로 ISO로. 구분자 날짜가 있으면 그것만, 없으면 8자리 날짜."""
    s = s or ""
    out = []
    for m in _DATE.finditer(s):
        v = _valid(*map(int, m.groups()))
        if v:
            out.append(v)
    if out:
        return out
    for m in _DATE8.finditer(s):
        v = _valid(*map(int, m.groups()))
        if v:
            out.append(v)
    if out:
        return out
    # 칸 전체가 짧은 날짜일 때만(문장 속 숫자는 안 읽음): "22.07.14" → 2022-07-14, "2025.0214"
    m = _DATE_YY.match(s) or _DATE_Y4MD.match(s)
    if m:
        y, mo, d = map(int, m.groups())
        v = _valid(y + 2000 if y < 100 else y, mo, d)
        if v:
            out.append(v)
    return out


def to_num(s: str | None) -> float | None:
    """칸이 숫자만일 때만 숫자. 문장 안 값은 None."""
    if s is None:
        return None
    t = s.strip().replace(" ", "")
    if not t or not _NUM.match(t):
        return None
    return float(t.replace(",", ""))


def classify_item(label: str | None) -> tuple[str, bool]:
    """정정항목 이름 → (item_class, multi_field). 키워드 사전, 위에서부터 먼저 맞는 것.

    ``multi_field``는 한 항목에 둘 이상 종류(예: 금액과 매출액 대비)가 같이 적힌 경우다.
    """
    k = key(label or "")
    kinds = []
    if "유보" in k:
        kinds.append("hold")
    if "금액" in k:
        kinds.append("amount")
    if "매출액대비" in k or "매출액비" in k or "비율" in k:
        kinds.append("ratio")
    if "최근매출" in k and "대비" not in k:
        kinds.append("recent_sales")
    has_start, has_end = "시작일" in k, "종료일" in k
    if has_start and has_end:
        kinds.append("period")
    elif has_end:
        kinds.append("period_end")
    elif has_start:
        kinds.append("period_start")
    elif "계약기간" in k or "계약(수주)기간" in k:
        kinds.append("period")
    if "계약상대" in k or "상대방" in k:
        kinds.append("counterparty")
    if "기타" in k and ("투자판단" in k or "참고" in k or "중요사항" in k):
        kinds.append("other_text")
    # "2. 계약내역"은 절 이름이다. 다른 종류가 없을 때만 금액으로 본다.
    # 11-01 대조: 절 이름 때문에 "계약내역 - 매출액대비(%)"·"계약내역 - 최근매출액"이
    # 금액으로 분류되던 것을 고침(계획 13 §13.2).
    if not kinds and "계약내역" in k:
        kinds.append("amount")
    if not kinds:
        return "unmapped", False
    # 대표 분류: 유보 > 금액 > 기간 > 상대방 > 비율 > 최근 매출액 > 기타
    order = (
        "hold",
        "amount",
        "period_end",
        "period_start",
        "period",
        "counterparty",
        "ratio",
        "recent_sales",
    )
    for c in order:
        if c in kinds:
            return c, len(set(kinds)) > 1
    return kinds[0], len(set(kinds)) > 1


def title_family(report_nm: str | None) -> str:
    """접수 목록 제목 → ``original`` · ``correction`` · ``attach`` · ``cancel`` · ``other``.

    체결 계열은 original·correction뿐이다. 해지와 ``[첨부…]``는 체결 계열이 아니다.
    """
    t = key(report_nm or "")
    if "해지" in t:
        return "cancel"
    if "체결" not in t or "공급계약" not in t:
        return "other"
    if t.startswith("[기재정정]"):
        return "correction"
    if t.startswith("[첨부"):
        return "attach"
    if t.startswith("["):
        return "other"
    return "original"


def extract_links(text: str) -> list[tuple[str, str]]:
    """맨 아래 ``※ 관련공시`` 이후의 링크 → [(KIND 접수번호 14자리, 글자)]. 중복은 그대로 둔다."""
    marks = [m.start() for m in re.finditer(r"관련\s*공시", text)]
    if not marks:
        return []
    out = []
    for m in _ANCHOR.finditer(text, marks[-1]):
        h = _RCPNO.search(html.unescape(m.group(1)))
        if h:
            out.append((h.group(1), norm(_TAG.sub(" ", m.group(2)))))
    return out


def related_text(rows: list[list[str]]) -> str | None:
    for r in reversed(rows):
        if r and "관련공시" in key(r[0]):
            return " ".join(r[1:]) if len(r) > 1 else ""
    return None


def parse_correction_text(text: str, report_nm: str = "") -> dict:
    """본문 문자열 → 필드 dict. ``items``는 (label, before, after) 목록."""
    rows = rows_of(text)
    f: dict = {
        "corr_kind": None,
        "corr_date": None,
        "corr_date_raw": None,
        "ref_doc_name_raw": None,
        "ref_submit_raw": None,
        "ref_submit_date": None,
        "ref_submit_n_dates": 0,
        "ref_latest_corr_date": None,
        "reason_raw": None,
        "n_items": 0,
        "note_raw": None,
        "items": [],
        "rel_links": extract_links(text),
        "rel_text_raw": related_text(rows),
        "parse_status": "ok",
        "parse_err": None,
    }
    f["n_rel_links"] = len(f["rel_links"])
    f["rel_links_raw"] = json.dumps(f["rel_links"], ensure_ascii=False)
    errs: list[str] = []
    i0 = next((i for i, r in enumerate(rows) if r and key(r[0]).startswith("정정일자")), None)
    rn = key(report_nm)
    if i0 is None:
        if rn.startswith("[첨부추가]"):
            f["corr_kind"] = "attach_only"
        elif rn.startswith("[첨부정정]"):
            f["corr_kind"] = "attach_correction"
        else:
            f["corr_kind"] = "no_block"
            f["parse_status"] = "partial"
            errs.append("no_correction_block")
        f["parse_err"] = ",".join(errs) or None
        return f
    f["corr_kind"] = "value_correction"
    hdr = next(
        (i for i in range(i0 + 1, len(rows)) if rows[i] and key(rows[i][0]).startswith("정정항목")),
        None,
    )
    r0 = rows[i0]
    f["corr_date_raw"] = r0[1] if len(r0) > 1 else None
    ds = parse_dates(f["corr_date_raw"])
    f["corr_date"] = ds[0] if ds else None
    if hdr is None:
        f["parse_status"] = "fail"
        f["parse_err"] = "no_item_header"
        return f
    for r in rows[i0 + 1 : hdr]:
        k = key(r[0])
        v = r[1] if len(r) > 1 else None
        if "제출일" in k:
            f["ref_submit_raw"] = v
        elif "정정관련공시서류" in k:
            f["ref_doc_name_raw"] = v
        elif "정정사유" in k:
            f["reason_raw"] = v
    sd = parse_dates(f["ref_submit_raw"])
    f["ref_submit_n_dates"] = len(sd)
    if sd:
        f["ref_submit_date"] = sd[0]
        if len(sd) > 1 and "최근" in (f["ref_submit_raw"] or ""):
            f["ref_latest_corr_date"] = sd[1]
    j = hdr + 1
    while j < len(rows) and len(rows[j]) == 3:
        f["items"].append(tuple(rows[j]))
        j += 1
    f["n_items"] = len(f["items"])
    if j < len(rows) and len(rows[j]) == 1:
        f["note_raw"] = rows[j][0]
    if f["corr_date"] is None:
        errs.append("corr_date")
    if f["ref_submit_date"] is None:
        errs.append("ref_submit_date")
    if f["n_items"] == 0:
        errs.append("no_items")
    if errs:
        f["parse_status"] = "partial"
    f["parse_err"] = ",".join(errs) or None
    return f


def parse_correction_zip(data: bytes, rcept_no: str, report_nm: str = "") -> dict:
    """zip 바이트 → 필드 dict. 인코딩 정보를 같이 돌려준다."""
    doc = read_zip_bytes(data, rcept_no)
    f = parse_correction_text(doc.text, report_nm)
    f.update(
        src_member=doc.member,
        src_encoding=doc.encoding,
        src_declared_encoding=doc.declared_encoding,
        src_encoding_mismatch=doc.encoding_mismatch,
    )
    if doc.encoding == "replace":
        f["parse_err"] = ",".join(filter(None, [f["parse_err"], "decode_replace"]))
    if "�" in doc.text:
        f["parse_err"] = ",".join(filter(None, [f["parse_err"], "source_replacement_chars"]))
    return f
