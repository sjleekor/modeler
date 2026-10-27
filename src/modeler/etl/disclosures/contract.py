"""단일판매ㆍ공급계약 체결·해지 공시 표 파서 (정규식·표 구조만, LLM 없음).

``parse_contract_html(text)`` 가 원문 HTML 문자열을 받아 필드 dict를 돌려주는 순수 함수다.
값 열 ``x`` 와 원문 문자열 ``x_raw`` 를 같이 둔다. ``-``는 값 열을 NULL로 두고 raw에 남긴다.
"""

from __future__ import annotations

import datetime as dt
import re

from modeler.etl.disclosures.docparse import key, rows_of

_BLANK = {"", "-", "해당없음", "해당 없음", "N/A", "미정", "미해당", "해당사항없음"}

_NUM = re.compile(r"\(?\s*(-?)\s*([\d,]+(?:\.\d+)?)\s*(?:원|%|%p)?\s*\)?")


def parse_number(v: str | None) -> tuple[float | None, bool]:
    """(값, 오류 여부). 빈 값·``-``는 (None, False), 형식이 안 맞으면 (None, True)."""
    if v is None:
        return None, False
    s = v.strip().replace("△", "-")
    if s in _BLANK:
        return None, False
    m = _NUM.fullmatch(s)
    if not m:
        return None, True
    x = float(m.group(2).replace(",", ""))
    return (-x if m.group(1) else x), False


_DATE = re.compile(r"(\d{4})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})")


def parse_date(v: str | None) -> tuple[str | None, bool]:
    """(YYYY-MM-DD, 오류 여부). 달력에 없는 날짜·1990 이전·2100 이후는 오류."""
    if v is None:
        return None, False
    s = v.strip()
    if s in ("", "-"):
        return None, False
    m = _DATE.search(s)
    if not m:
        return None, True
    y, mo, d = (int(x) for x in m.groups())
    try:
        dt.date(y, mo, d)
    except ValueError:
        return None, True
    if not (1990 <= y <= 2100):
        return None, True
    return f"{y:04d}-{mo:02d}-{d:02d}", False


_UNIT_SCALE = {"원": 1, "천원": 1_000, "백만원": 1_000_000, "억원": 100_000_000}
_UNIT_IN_LABEL = re.compile(r"\((백만원|천원|억원|원|USD|EUR|JPY|CNY|달러)\)")


def amount_unit(label: str | None) -> str | None:
    if not label:
        return None
    m = _UNIT_IN_LABEL.search(label)
    return m.group(1) if m else None


def split_related(val: str | None) -> list[tuple[str, str]]:
    """``※관련공시`` 값을 (날짜, 제목 원문) 쌍으로 나눈다.

    날짜 표기가 ``2018-02-09`` ``2012.12.31`` ``[2012.01.13]`` ``2012년 1월 4일`` 로 갈린다.
    """
    if not val or val.strip() in ("", "-"):
        return []
    pat = re.compile(r"\[?\s*(\d{4})\s*[-./년]\s*(\d{1,2})\s*[-./월]\s*(\d{1,2})\s*일?\s*\]?")
    ms = list(pat.finditer(val))
    out = []
    for i, m in enumerate(ms):
        y, mo, d = (int(x) for x in m.groups())
        end = ms[i + 1].start() if i + 1 < len(ms) else len(val)
        title = val[m.end() : end].strip(" -:·\t")
        try:
            dt.date(y, mo, d)
        except ValueError:
            continue
        out.append((f"{y:04d}-{mo:02d}-{d:02d}", title))
    return out


_CCY = re.compile(r"(USD|EUR|JPY|CNY|RMB|GBP|AUD|SGD|HKD|CAD|달러|유로|엔화|위안)")
_FX_AMT = re.compile(
    r"((?:USD|EUR|JPY|CNY|RMB|GBP|AUD|SGD|HKD|CAD)\s*[\d,]+(?:\.\d+)?"
    r"|[\d,]+(?:\.\d+)?\s*(?:달러|유로|엔화|위안|천달러|백만달러))"
)

# 라벨(번호 제거 뒤) → 필드 이름. 값은 cells[-1].
_AMOUNT_LABEL = re.compile(r"^(계약금액|해지금액)(총액)?(\([^)]*\))?$")


def parse_contract_html(text: str) -> dict:
    """원문 HTML에서 1차·2차 필드를 읽는다. 예외는 호출자가 ``fail``로 처리한다."""
    rows = rows_of(text)
    f: dict = {"n_rows": len(rows)}
    labels: set[str] = set()
    err: list[str] = []
    other: list[str] = []
    related_parts: list[str] = []
    in_other = False
    in_related = False

    def put_num(name: str, val: str) -> None:
        f[f"{name}_raw"] = val
        x, e = parse_number(val)
        f[name] = x
        if e:
            err.append(name)

    def put_date(name: str, val: str) -> None:
        f[f"{name}_raw"] = val
        x, e = parse_date(val)
        f[name] = x
        if e:
            err.append(name)

    for cells in rows:
        c0 = cells[0]
        k0 = key(c0)
        if c0.startswith("자회사인") and len(cells) >= 2:
            f["subsidiary_notice"] = cells[1]
            continue
        if c0.startswith("※") or k0 == "관련공시":
            in_other = False
            in_related = True
            body = cells[-1] if len(cells) > 1 else re.sub(r"^※\s*관련\s*공시\s*", "", c0)
            related_parts.append(body)
            continue
        if in_related and not re.match(r"^\d+\.", k0):
            related_parts.append(" ".join(cells))
            continue
        if len(cells) == 1:
            if in_other:
                other.append(c0)
            if re.match(r"^\d+\.기타", k0):
                in_other = True
            continue
        lab = key(cells[-2])
        val = cells[-1]
        if re.match(r"^\d+\.기타투자판단", k0) or re.match(r"^\d+\.기타투자판단", lab):
            in_other = True
            other.append(val)
            continue
        if in_other and not re.match(r"^\d+\.", k0):
            other.append(val)
            continue
        in_other = False
        L = re.sub(r"^\d+\.", "", lab)  # 번호를 뗀 라벨
        labels.add(L)
        if L == "판매ㆍ공급계약구분" or L == "판매ㆍ공급계약해지구분":
            f["contract_kind"] = val
        elif L in ("판매ㆍ공급계약내용", "판매ㆍ공급계약해지내용", "-체결계약명", "-해지계약명"):
            f["contract_name"] = val
        elif L == "조건부계약여부":
            f["is_conditional"] = val
        elif L == "확정계약금액":
            put_num("amount_fixed", val)
        elif L == "조건부계약금액":
            put_num("amount_cond", val)
        elif _AMOUNT_LABEL.match(L):
            name = "cancel_amount" if L.startswith("해지금액") else "amount"
            put_num(name, val)
            f[f"{name}_label"] = L
            f[f"{name}_unit"] = amount_unit(L)
        elif L.startswith("최근매출액"):
            put_num("recent_sales", val)
        elif L.startswith("-최근매출액"):
            put_num("cp_recent_sales", val)
        elif L.startswith("매출액대비"):
            put_num("ratio_pct", val)
        elif L == "대규모법인여부":
            f["large_corp"] = val
        elif L.startswith("계약상대"):
            f["counterparty_raw"] = val
        elif L == "-주요사업":
            f["cp_business"] = val
        elif L == "-회사와의관계":
            f["cp_relation"] = val
        elif L.startswith("-회사와최근3년"):
            f["cp_prior_3y"] = val
        elif L.startswith("판매ㆍ공급지역"):
            f["region"] = val
        elif L == "시작일":
            put_date("period_start", val)
        elif L == "종료일":
            put_date("period_end", val)
        elif L in ("주요계약조건", "주요해지사유", "해지주요사유"):
            f["cancel_reason" if "해지" in L else "terms"] = val
        elif L == "계약금ㆍ선급금유무":
            f["advance_flag"] = val
        elif L == "대금지급조건등":
            f["payment_terms"] = val
        elif L == "자체생산":
            f["mode_inhouse"] = val
        elif L == "외주생산":
            f["mode_outsourced"] = val
        elif L.startswith("계약(수주)일"):
            put_date("contract_date", val)
        elif L.startswith("해지일자"):
            put_date("cancel_date", val)
        elif L == "유보사유":
            f["hold_reason"] = val
        elif L == "유보기한":
            put_date("hold_until", val)
        elif L in ("-세부내용", "-세부물건"):
            f["detail"] = val
        elif L == "제목":
            f["title_in_doc"] = val
        elif L == "주요내용":
            f["main_text_len"] = len(val)
        elif L.startswith("결정(확인)일자"):
            put_date("decision_date", val)

    if not f.get("contract_name") and f.get("detail") not in (None, "", "-"):
        f["contract_name"] = f["detail"]
        f["contract_name_src"] = "detail"
    f.pop("detail", None)

    related_raw = " ".join(p for p in related_parts if p).strip()
    f["related_raw"] = related_raw or None
    f["related_pairs"] = split_related(related_raw)
    other_text = " ".join(other).strip()
    f["other_text"] = other_text or None

    ctx = f"{other_text} {f.get('terms') or ''} {f.get('cancel_reason') or ''}"
    ccys = sorted(set(_CCY.findall(ctx)))
    f["fx_hint"] = ",".join(ccys) if ccys else None
    m = _FX_AMT.search(ctx)
    f["fx_amount_raw"] = m.group(1) if m else None
    if re.search(r"부가(가치)?세\s*포함|VAT\s*포함|V\.A\.T\s*포함", ctx):
        f["vat_hint"] = "포함"
    elif re.search(r"부가(가치)?세\s*(별도|제외)|VAT\s*(별도|제외)", ctx):
        f["vat_hint"] = "별도"
    else:
        f["vat_hint"] = None

    # 단위 환산: 라벨 단위가 원이 아니면 원으로 곱한다. 단위가 라벨에 없으면 원으로 가정하고 표시.
    for name in ("amount", "cancel_amount"):
        if f.get(name) is not None:
            u = f.get(f"{name}_unit")
            f[f"{name}_unit_assumed"] = u is None
            if u is None or u in _UNIT_SCALE:
                f[name] = f[name] * _UNIT_SCALE.get(u or "원", 1)
            else:  # 외화 단위 라벨은 환산하지 않는다
                f[name] = None
                err.append(f"{name}_currency")

    a = f.get("amount")
    if a is None:
        a = f.get("cancel_amount")
    rs, ra = f.get("recent_sales"), f.get("ratio_pct")
    if a is not None and rs and ra is not None and rs > 0:
        diff = round(a / rs * 100 - ra, 2)
        f["ratio_check_diff"] = diff
        f["ratio_check_ok"] = abs(diff) <= 0.5
    else:
        f["ratio_check_diff"] = None
        f["ratio_check_ok"] = None

    f["parse_err"] = ",".join(sorted(set(err))) or None
    f["form_variant"] = detect_form(labels)
    f["_labels"] = labels
    return f


def detect_form(labels: set[str]) -> str:
    """라벨 묶음으로 서식을 가른다. v3 > v2 > v1. 연도로는 못 가른다."""
    if labels & {"조건부계약여부", "확정계약금액", "조건부계약금액", "계약금액총액(원)"}:
        return "v3"
    if labels & {
        "판매ㆍ공급계약구분",
        "판매ㆍ공급계약해지구분",
        "대규모법인여부",
        "-체결계약명",
        "-해지계약명",
    }:
        return "v2"
    if labels & {"판매ㆍ공급계약내용", "판매ㆍ공급계약해지내용", "계약금액(원)", "해지금액(원)"}:
        return "v1"
    return "unknown"


def core_status(f: dict, is_cancel: bool) -> str:
    """핵심 필드가 다 인식됐으면 ok, 일부 빠졌으면 partial, 하나도 없으면 fail."""
    if f["form_variant"] == "unknown" or f["n_rows"] == 0:
        return "fail"
    if is_cancel:
        need = ("contract_name", "cancel_amount_raw", "cancel_date_raw")
    else:
        need = ("contract_name", "amount_raw", "recent_sales_raw", "contract_date_raw")
    return "ok" if all(f.get(k) is not None for k in need) and not f["parse_err"] else "partial"
