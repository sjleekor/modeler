"""사업보고서(b) 해석 표 — 셀 표(``kr_bizrpt_table_cell``) 한 개를 값 한 줄로 바꾸는 규칙.

입력은 승격된 셀 표의 셀 dict 목록 하나(표 하나)뿐이다. 원문 zip은 읽지 않는다. 결정·근거는
``my/milestones/common/research/20261027_investment_research_agenda/12_plan_b_stage2_interpretation.md``
(§2 공통 규칙 · §3 서식 군 · §4 해석 표 · 맨 아래 "결정" 절). 수치 규칙은 전부 정규식과 표 구조이고
LLM·퍼지 매칭은 쓰지 않는다.

해석 표 (이번 범위)
-------------------
* ``customer``      ``kr_bizrpt_customer_row``  — 주석 ``major_customer_note_consol/_sep``만
* ``orderbook``     ``kr_bizrpt_orderbook_row`` — ``orders``·``sales``·``main_products`` 절의 수주 표
* ``segment_sales`` ``kr_bizrpt_segment_sales_row`` — ``sales``·``main_products`` 절의 매출 표
* 원재료 가격 표는 만들지 않는다(결정 5). 본문 고객(``biz_sales``)·가격 표는 탐지만 하고
  ``adapter = not_in_scope``로 둔다(대상 목록에서 빼는 표지).

어댑터 (코드 상수 ``ADAPTERS``·``JUDGED``에 고정, 열 역할 문자열은 왼쪽부터)
---------------------------------------------------------------------------
열 역할: ``N`` 이름 열(고객 주석의 첫 열) · ``c`` 글자 열 · ``A`` 금액 · ``P`` 비율 · ``q`` 수량 ·
``M`` 글자 칸 안의 ``금액(비율%)`` · ``e`` 빈 열(무시) · ``x`` 머리 없는 숫자 열(표는 ``partial``).

고객 주석(첫 열 N, 이어서 글자 열 ``lead-1``개, 뒤쪽 글자 열은 무시)

* ``C-A*2``  = ``N A A``       — 두 열 모두 기간 머리, 기간 서로 다름 (판정)
* ``C-AP*2`` = ``N (A P){2}``   — 쌍마다 기간 머리, 서로 다름 (판정)
* ``C-A``    = ``N A``         (판정)
* ``C-AP``   = ``N A P`` (글자 열 1개 이상) — 판정 밖
* ``C-P*n``  = ``N P{n}``      — 판정 밖
* ``C-A*n``  = 그 밖의 ``N A{n}`` (n>=3) — 판정 밖
* 글자 열이 2개 이상이면 어댑터 이름 끝에 ``+L``이 붙고 판정 밖이다(``C-AP``는 원래 판정 밖).

수주잔고(수주총액·기납품액·수주잔고 묶음 × 수량·금액 머리)

* ``O-QA3`` = 글자 열 + ``(수주총액 수량·금액)(기납품액 수량·금액)(수주잔고 수량·금액)`` (판정)
* ``O-A3``  = 금액만 세 묶음 (판정)
* ``O-TA``  = 수주총액·수주잔고 금액만(간이형) — 판정 밖

사업부문·품목 매출(글자 열 1개 이상 + 숫자 열, 뒤쪽 글자 열은 설명 열)

* ``S-A*n``       = ``A{n}`` n>=2, 모든 열에 기간 머리, 서로 다름 (판정)
* ``S-AP``        = ``A P`` 기간 머리 0~1개 (판정)
* ``S-AP*n``      = ``(A P){n}`` n>=2, 쌍마다 기간 머리, 서로 다름 (판정)
* ``S-A*nP*n``    = ``A{n} P{n}`` n>=2 — 판정 밖
* ``S-AP-in-cell`` = ``M{n}`` — 판정 밖

어디에도 안 맞는 표는 ``no_adapter``(해석 행을 만들지 않는다).
"""

from __future__ import annotations

import collections
import re
import unicodedata
from datetime import date

INTERP_VERSION = "disclosure_bizinterp_1.0.0"

# ------------------------------------------------------------------ 상수 (결과 전 고정)

TARGETS = ("customer", "orderbook", "segment_sales")
NOT_IN_SCOPE = "not_in_scope"
NO_ADAPTER = "no_adapter"

ADAPTERS: dict[str, tuple[str, ...]] = {
    "customer": ("C-A*2", "C-AP*2", "C-A", "C-AP", "C-P*n", "C-A*n"),
    "orderbook": ("O-QA3", "O-A3", "O-TA"),
    "segment_sales": ("S-A*n", "S-AP", "S-AP*n", "S-A*nP*n", "S-AP-in-cell"),
}
JUDGED = frozenset(
    {
        ("customer", "C-A*2"),
        ("customer", "C-AP*2"),
        ("customer", "C-A"),
        ("orderbook", "O-QA3"),
        ("orderbook", "O-A3"),
        ("segment_sales", "S-A*n"),
        ("segment_sales", "S-AP"),
        ("segment_sales", "S-AP*n"),
    }
)

SRC_NOTE = {"major_customer_note_consol": "note_consol", "major_customer_note_sep": "note_sep"}
BODY_SECTIONS = ("sales", "main_products", "orders")
PRICE_SECTIONS = ("raw_materials", "main_products")

# 정규식 (정규화한 글자에 적용: ``norm``)
RE_CUST = re.compile(r"(매출처|거래처|고객|판매처|납품처|상대방|거래선)")
RE_AMT = re.compile(
    r"(매출액|매출금액|금액|수주액|판매액|매출|당기|전기|전전기|제\d+기|(?<!\d)(19|20)\d\d(?!\d)|\d{2}년)"
)
RE_PCT = re.compile(r"(비율|비중|구성비|%)")
RE_QTY = re.compile(r"(수량|물량|중량|수주량)")
RE_CAT = re.compile(
    r"(사업부문|부문|품목|제품|상품|구분|매출유형|사업|제품군|서비스|품명|부분|유형|종류)"
)
RE_PRICE = re.compile(r"(가격|단가|시세)")
RE_SEG_EXCL = re.compile(r"매입|취득|감가|기초|기말|상각")
RE_SEG_WORD = re.compile(r"매출|금액|판매|수출|내수")
RE_REGION = re.compile(r"(지역|국가|권역)")
RE_ENTITY = re.compile(r"(회사명|법인|종속)")
RE_DESC = re.compile(r"(용도|상표|판매처|비고|설명|내용|특징|주요)")

_TOTAL = {"합계", "총계", "총합계", "합계금액", "total", "매출액합계", "합계매출액"}
_SUBTOTAL = {"소계", "계", "subtotal", "sub-total", "소계및합계"}
_ADJUST = {
    "연결조정",
    "조정",
    "내부거래제거",
    "연결조정후",
    "제거",
    "연결조정및제거",
    "내부매출",
    "내부거래",
}

_ANON = [
    re.compile(p)
    for p in (
        r"^[a-z]$",
        r"^[a-z](사|기업|업체|고객|고객사|거래처|회사|법인|그룹)$",
        r"^(고객|거래처|매출처|주요고객|고객사|회사|업체|기업)[a-z0-9]{1,2}$",
        r"^[가-하]사$",
        r"^(주요)?(고객|거래처|매출처)\d$",
        r"^customer[a-z0-9]?$",
        r"^[a-z][0-9]$",
        r"^(상위)?\d+(대|위)(고객|거래처)?$",
    )
]

MONEY = {
    "조원": 1e12,
    "십억원": 1e9,
    "억원": 1e8,
    "백만원": 1e6,
    "천만원": 1e7,
    "만원": 1e4,
    "천원": 1e3,
    "원": 1.0,
}
_FX = re.compile(
    r"(\$|usd|천불|백만불|달러|불|k\$|m\$|천u\$|u\$|엔|jpy|cny|위안|eur|유로|rmb|천달러|백만달러)",
    re.I,
)
_MONEY_TOK = re.compile(r"조원|십억원|억원|백만원|천만원|만원|천원|(?<![백천만억조십])원")

_PERIOD_TOK = re.compile(
    r"전전기|당기|전기|\(?당\)?기|\(?전\)?기|\(전전\)|제\d+\(?[당전]*\)?기|(?<![\d.])\d{1,3}기"
    r"|(?<!\d)(?:19|20)\d\d(?!\d)|'\d\d년|(?<!\d)\d\d년"
)


# ------------------------------------------------------------------ 글자 도구


def norm(t: str | None) -> str:
    """NFKC, 공백·괄호·``·ㆍ,.:*※`` 제거, 소문자 (서명·열 이름 판별용)."""
    t = unicodedata.normalize("NFKC", t or "")
    return re.sub(r"[\s()\[\]{}·ㆍ,.:*※]", "", t).lower()


def clean_text(t: str | None) -> str:
    t = unicodedata.normalize("NFKC", t or "")
    return re.sub(r"\s+", " ", t).strip()


def _txt(c: dict) -> str:
    return (c.get("text_raw") or "").strip()


_PLACEHOLDER = {
    "-",
    "–",
    "—",
    "－",
    "n/a",
    "na",
    "해당없음",
    "해당사항없음",
    "없음",
    "x",
    "ㅡ",
    ",",
    ".",
    "·",
}
_RE_NUMHDR = re.compile(
    r"수주총액|수주액|수주잔고|수주잔액|기납품|납품액|금액|수량|중량|물량|매출|비율|비중|구성비|당기|전기|제\d+기|(?<!\d)(?:19|20)\d\d(?!\d)"
)


_RE_DATEHDR = re.compile(
    r"일자|납기|계약일|수주일|기한|시점|개시일|종료일|완료일|인도일|기간|납품일"
)
_RE_MEASURE = re.compile(r"금액|수량|총액|잔고|잔액|납품액|수주액|매출|비율|비중")


def is_placeholder(t: str) -> bool:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", t or "")).lower() in _PLACEHOLDER


# ------------------------------------------------------------------ §2.2 행 종류


def row_kind_of(texts: list[str]) -> str:
    """글자 칸들의 공백·괄호를 지운 값으로 total·subtotal·adjust·item을 가른다(왼쪽 우선)."""
    for t in texts:
        n = norm(t)
        if not n:
            continue
        if n in _TOTAL:
            return "total"
        if n in _SUBTOTAL:
            return "subtotal"
        if n in _ADJUST:
            return "adjust"
    return "item"


# ------------------------------------------------------------------ §2.4 단위


def unit_parse(u: str | None) -> tuple[str, float | None]:
    """``(unit_status, 원 환산 배수)``. status: krw · foreign · conflict · nomoney · missing."""
    if u is None or not u.strip():
        return "missing", None
    t = re.sub(r"\s", "", unicodedata.normalize("NFKC", u))
    t = re.sub(r"^\(?(금액)?단위[:：]?", "", t).rstrip(")")
    fx = bool(_FX.search(t))
    toks = set(_MONEY_TOK.findall(t))
    if re.search(r"\d{4}년", t):
        return "conflict", None
    if len(toks) > 1:
        return "conflict", None
    if len(toks) == 1:
        if fx:
            return "conflict", None
        return "krw", MONEY[toks.pop()]
    if fx:
        return "foreign", None
    return "nomoney", None


# ------------------------------------------------------------------ §2.3 금액(비율) 한 칸

_RE_AP_CELL = re.compile(r"^([\d,.\-()△▲]+)\s*\(\s*(\d+(?:\.\d+)?)\s*%?\s*\)$")


def parse_plain_num(s: str) -> float | None:
    """쉼표·괄호 음수·△를 읽는 숫자 파서(``bizparse.parse_num`` 래퍼)."""
    from modeler.etl.disclosures.bizparse import parse_num

    v, _pct, _err = parse_num(s)
    return v


def split_amount_ratio(text: str) -> tuple[float | None, float | None] | None:
    """``1,000 (3.4%)`` → ``(1000.0, 3.4)``. 모양이 아니면 None."""
    t = unicodedata.normalize("NFKC", text or "").strip()
    m = _RE_AP_CELL.match(t)
    if not m:
        return None
    return parse_plain_num(m.group(1)), float(m.group(2))


# ------------------------------------------------------------------ §2.6 이름·익명


def split_name_note(name: str) -> tuple[str, str | None]:
    """뒤의 주석 표시 ``(*1)``·``(주1)``를 분리한다."""
    n = clean_text(name)
    m = re.search(r"\s*[\(（]\s*(\*+\s*\d*|주\s*\d+|\d+\))\s*[\)）]\s*$", n)
    if m:
        return n[: m.start()].strip(), m.group(1).strip()
    m = re.search(r"\s*(\*+\d*)$", n)
    if m:
        return n[: m.start()].strip(), m.group(1)
    return n, None


def anon_info(name: str | None) -> tuple[bool, str | None]:
    """이름의 형태만 본다. ``(anonymized, anon_label)``."""
    if not name:
        return False, None
    n = norm(name)
    n = re.sub(r"\(\*?\d*\)$", "", n)
    if any(r.match(n) for r in _ANON):
        label = re.sub(
            r"(사|기업|업체|고객사|고객|거래처|매출처|회사|법인|그룹|주요|customer)", "", n
        )
        return True, label or n
    return False, None


# ------------------------------------------------------------------ §2.5 기간 읽기

_RE_FY0 = re.compile(r"\(\s*(\d{4})\s*\.\s*(\d{1,2})\s*\)")


def fy0_of(report_nm: str | None, rcept_dt: str | None) -> tuple[int | None, int | None, str]:
    """``(FY0, 결산월, 출처)``. ``report_nm``의 ``(YYYY.MM)``이 없으면 접수 연도-1."""
    m = _RE_FY0.search(report_nm or "")
    if m:
        return int(m.group(1)), int(m.group(2)), "report_nm"
    if rcept_dt and len(rcept_dt) >= 4 and rcept_dt[:4].isdigit():
        return int(rcept_dt[:4]) - 1, None, "rcept_dt"
    return None, None, "none"


def period_component(path: str | None) -> str | None:
    """머리 경로(``|`` 구분)에서 기간 표현이 든 첫 성분. 없으면 None."""
    if not path:
        return None
    for comp in path.split("|"):
        if _PERIOD_TOK.search(re.sub(r"\s+", "", unicodedata.normalize("NFKC", comp))):
            return comp
    return None


def parse_period_label(label: str) -> dict:
    """한 머리 글자에서 ``n``(제N기), ``rel``, ``year``, ``irregular``를 읽는다(§2.5 2·3)."""
    t = re.sub(r"\s+", "", unicodedata.normalize("NFKC", label))
    out = {"n": None, "rel": None, "year": None, "irregular": False}
    # 날짜 범위
    dates = []
    for m in re.finditer(r"(?<!\d)(\d{4}|\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})(?!\d)", t):
        y = int(m.group(1))
        y = y + 2000 if y < 100 else y
        dates.append((y, int(m.group(2)), int(m.group(3))))
    for m in re.finditer(r"(\d{4})년(\d{1,2})월(\d{1,2})일", t):
        dates.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    if len(dates) >= 2:
        try:
            d0, d1 = date(*dates[0]), date(*dates[-1])
            span = (d1 - d0).days + 1
            out["year"] = d1.year
            out["irregular"] = not (350 <= span <= 380)
        except ValueError:
            pass
    if out["year"] is None:
        m = re.search(r"(?<!\d)((?:19|20)\d\d)년?(\d{1,2})월(말?)(?!\d*일)", t)
        if m:
            out["year"] = int(m.group(1))
            out["irregular"] = not (
                m.group(2) == "12" and m.group(3)
            )  # ``2025년 12월말``은 사업연도 끝
        elif re.search(r"[1-4]분기|반기|Q[1-4]", t, re.I):
            m4 = re.search(r"(?<!\d)((?:19|20)\d\d)", t)
            out["year"] = int(m4.group(1)) if m4 else None
            out["irregular"] = True
        else:
            m4 = re.search(r"(?<!\d)((?:19|20)\d\d)(?!\d)", t)
            if m4:
                out["year"] = int(m4.group(1))
            else:
                m2 = re.search(r"(?<!\d)'?(\d{2})년", t)
                if m2:
                    out["year"] = 2000 + int(m2.group(1))
    mn = re.search(r"제(\d{1,3})(?:\([당전]+\))?기", t) or re.search(r"(?<![\d.])(\d{1,3})기", t)
    if mn:
        out["n"] = int(mn.group(1))
    t2 = t.replace("전전", "@@")
    if "@@" in t2:
        out["rel"] = -2
    elif re.search(r"당\)?기|\(당\)", t2):
        out["rel"] = 0
    elif re.search(r"전\)?기|\(전\)", t2):
        out["rel"] = -1
    return out


def resolve_periods(labels: list[str | None], fy0: int | None) -> list[dict]:
    """열마다 기간을 정한다. ``labels``는 ``period_component`` 결과(없으면 None).

    제N기는 표 안의 가장 큰 N을 FY0로 본다. 연도를 우선하고 서로 안 맞으면 ``period_conflict``.
    머리에 기간이 전혀 없으면(표 전체가 한 기간) 당기로 두고 ``period_inferred = True``.
    """
    parsed = [parse_period_label(lb) if lb else None for lb in labels]
    ns = [p["n"] for p in parsed if p and p["n"] is not None]
    nmax = max(ns) if ns else None
    res = []
    for lb, p in zip(labels, parsed):
        r = {
            "period_label_raw": clean_text(lb) if lb else None,
            "period_rel": None,
            "fiscal_year": None,
            "period_irregular": False,
            "period_conflict": False,
            "period_inferred": False,
        }
        if p is None:
            if len([x for x in labels if x]) == 0 and fy0 is not None:
                r.update(period_rel=0, fiscal_year=fy0, period_inferred=True)
            res.append(r)
            continue
        fy = p["year"]
        r["period_irregular"] = bool(p["irregular"])
        cands = []
        if p["rel"] is not None and fy0 is not None:
            cands.append(fy0 + p["rel"])
        if p["n"] is not None and nmax is not None and fy0 is not None:
            cands.append(fy0 - (nmax - p["n"]))
        if fy is None and cands:
            fy = cands[0]
        elif fy is not None and any(c != fy for c in cands):
            r["period_conflict"] = True
        r["fiscal_year"] = fy
        if fy is not None and fy0 is not None:
            r["period_rel"] = fy - fy0
        res.append(r)
    return res


# ------------------------------------------------------------------ 표 프로필


class Prof:
    """표 하나의 본문 격자와 열 역할."""

    def __init__(self, section_key: str, cells: list[dict]):
        self.section_key = section_key
        self.cells = cells
        self.body = [c for c in cells if not c["is_header"]]
        self.rows: dict[int, dict[int, dict]] = collections.defaultdict(dict)
        for c in self.body:
            self.rows[c["row_idx"]][c["col_idx"]] = c
        self.cols = sorted({c["col_idx"] for c in self.body})
        self.path: dict[int, str] = {}
        for c in self.body:
            p = c.get("header_path_raw")
            if p and c["col_idx"] not in self.path:
                self.path[c["col_idx"]] = p
        stats: dict[int, list] = collections.defaultdict(
            lambda: [0, 0, 0, 0]
        )  # 비어 있지 않음, 숫자, AP칸, 자리표시
        lens: dict[int, list[int]] = collections.defaultdict(list)
        for c in self.body:
            if (c.get("colspan") or 1) > 1:
                continue
            t = _txt(c)
            if not t:
                continue
            s = stats[c["col_idx"]]
            s[0] += 1
            lens[c["col_idx"]].append(len(t))
            if c.get("num_value") is not None:
                s[1] += 1
            elif split_amount_ratio(t) is not None:
                s[2] += 1
            elif is_placeholder(t):
                s[3] += 1
        self.nonblank = {k: v[0] for k, v in stats.items()}
        # ``-``·``해당없음`` 같은 자리표시 칸은 숫자 열 안의 빈 값으로 본다. 자리표시뿐인 열은
        # 머리에 금액·수량·비율·기간 낱말이 있을 때만 숫자 열이다.
        self.numfrac = {}
        for k, v in stats.items():
            if not v[0]:
                continue
            if v[1] == 0 and v[2] == 0:
                hdr = norm(self.path.get(k) or "")
                self.numfrac[k] = 1.0 if v[3] == v[0] and _RE_NUMHDR.search(hdr) else 0.0
            else:
                self.numfrac[k] = (v[1] + v[3]) / v[0]
        self.apfrac = {k: v[2] / v[0] for k, v in stats.items() if v[0]}
        self.medlen = {k: sorted(v)[len(v) // 2] for k, v in lens.items()}
        self.n_body_rows = len(self.rows)
        self.roles: dict[int, str] = {}
        for k in self.cols:
            self.roles[k] = self._role(k)

    def _role(self, k: int) -> str:
        p = self.path.get(k)
        full = norm(p or "")
        numhdr = bool(p) and bool(_RE_NUMHDR.search(full))
        if not self.nonblank.get(k):
            # 값이 하나도 없는 열: 머리가 금액·수량·비율·기간이면 숫자 열 자리를 지킨다(열 배치가 어긋나지 않게)
            if not numhdr:
                return "e"
        elif _RE_DATEHDR.search(full) and not _RE_MEASURE.search(full):
            return "c"  # ``2018.09``처럼 숫자로 읽히는 날짜 열
        elif self.numfrac.get(k, 0) < 0.5:
            return "M" if self.apfrac.get(k, 0) >= 0.5 else "c"
        if not p:
            return "x"
        last = ""
        for comp in reversed(p.split("|")):  # 기간뿐인 성분은 건너뛰고 그 위 성분으로 역할을 정한다
            cn = norm(comp)
            if re.sub(_PERIOD_TOK, "", cn).strip("()~-년도기분"):
                last = cn
                break
        if RE_PCT.search(last):
            return "P"
        if RE_QTY.search(last):
            return "q"
        return "A"

    def header_norm(self, k: int) -> str:
        return norm(self.path.get(k) or "")

    def all_header_norm(self) -> str:
        return norm("|".join(self.path.values()))

    def partial(self) -> bool:
        return any(r == "x" for r in self.roles.values())


# ------------------------------------------------------------------ 서명 (감사용, §3.1)


def _sig_comp(x: str) -> str:
    x = norm(x)
    x = re.sub(r"제?\d+\(?[당전]?\)?기(연간|말)?", "{P}", x)
    x = re.sub(r"[']?(19|20)?\d\d년도?(\{P\})?|(19|20)\d\d(?!\d)", "{P}", x)
    x = re.sub(r"(전전기|당기|전기)", "{P}", x)
    return re.sub(r"(\{P\})+", "{P}", x)


def signature(prof: Prof) -> str:
    out = [">".join(_sig_comp(c) for c in prof.path[k].split("|")) for k in sorted(prof.path)]
    res: list[list] = []
    for o in out:
        if res and res[-1][0] == o:
            res[-1][1] += 1
        else:
            res.append([o, 1])
    return "¦".join(o if n == 1 else f"{o}×{n}" for o, n in res)


# ------------------------------------------------------------------ 대상 탐지 (결과 전 고정)


def detect_target(section_key: str, prof: Prof) -> str | None:
    """``customer`` · ``orderbook`` · ``customer_biz`` · ``rawmat_price`` · ``segment_sales`` · None."""
    if not prof.cols:
        return None
    nrows = prof.n_body_rows
    num_cols = [k for k in prof.cols if prof.roles[k] in ("A", "P", "q", "x")]
    if section_key in SRC_NOTE:
        first = prof.cols[0]
        if (
            nrows >= 1
            and len([k for k in prof.cols if prof.roles[k] != "e"]) >= 2
            and num_cols
            and prof.roles[first] in ("c", "M", "e")
            and prof.numfrac.get(first, 0) < 0.5
        ):
            return "customer"
        return None
    allh = prof.all_header_norm()
    if section_key in BODY_SECTIONS:
        if (
            "수주" in allh
            and re.search("수주잔고|수주잔액|잔고|잔액|수주총액|수주액", allh)
            and nrows >= 1
        ):
            return "orderbook"
        custcols = [
            k
            for k in prof.cols
            if prof.path.get(k)
            and RE_CUST.search(prof.header_norm(k))
            and prof.numfrac.get(k, 0) < 0.5
            and prof.medlen.get(k, 0) <= 20
            and section_key != "raw_materials"
        ]
        if custcols:
            n_all = n_lst = 0
            for c in prof.body:
                if c["col_idx"] in custcols:
                    n_all += 1
                    if re.search(r"[,、]|등$|외$", c.get("text_raw") or ""):
                        n_lst += 1
            if n_all and n_lst / n_all >= 0.5:
                custcols = []
        if custcols and any(prof.roles[k] in ("A", "P") for k in prof.cols) and nrows >= 2:
            return "customer_biz"
    if section_key in PRICE_SECTIONS:
        first = min(prof.path) if prof.path else 0
        price_hdr = any(RE_PRICE.search(prof.header_norm(k)) for k in prof.cols if prof.path.get(k))
        price_first = any(
            RE_PRICE.search(norm(c.get("text_raw"))) for c in prof.body if c["col_idx"] == first
        )
        if (price_hdr or price_first) and num_cols and nrows >= 1:
            return "rawmat_price"
    if section_key in ("sales", "main_products"):
        amt = [
            k
            for k in prof.cols
            if prof.roles[k] in ("A", "x", "M")
            and prof.path.get(k)
            and RE_AMT.search(prof.header_norm(k))
        ]
        cat = [
            k
            for k in prof.cols
            if prof.roles[k] in ("c", "e")
            and prof.path.get(k)
            and RE_CAT.search(prof.header_norm(k))
        ]
        if amt and cat and RE_SEG_WORD.search(allh) and not RE_SEG_EXCL.search(allh) and nrows >= 2:
            return "segment_sales"
    return None


# ------------------------------------------------------------------ 어댑터 계획


def _period_key(path: str | None) -> str | None:
    comp = period_component(path)
    return norm(comp) if comp else None


def _distinct_periods(keys: list[str | None], need_all: bool = True) -> bool:
    if need_all and any(k is None for k in keys):
        return False
    ks = [k for k in keys if k is not None]
    return len(set(ks)) == len(ks)


def _split_layout(prof: Prof, first_is_name: bool) -> tuple[list[int], list[int], str]:
    """``(lead 열, 숫자 영역 열, 숫자 영역 역할 문자열)``. 비어 있는 열은 건너뛴다."""
    cols = [k for k in prof.cols if prof.roles[k] != "e"]
    lead: list[int] = []
    i = 0
    while i < len(cols) and prof.roles[cols[i]] == "c":
        lead.append(cols[i])
        i += 1
    if first_is_name and (not cols or prof.roles[cols[0]] not in ("c", "M")):
        return [], [], "?"
    rest = cols[i:]
    # 뒤쪽 글자 열은 설명 열로 보고 숫자 영역에서 뺀다
    while rest and prof.roles[rest[-1]] == "c":
        rest = rest[:-1]
    return lead, rest, "".join(prof.roles[k] for k in rest)


def plan_customer(prof: Prof) -> dict:
    first = prof.cols[0]
    cols = [k for k in prof.cols if prof.roles[k] != "e"]
    if not cols or cols[0] != first or prof.roles[first] != "c":
        return {"adapter": NO_ADAPTER}
    # 첫 열은 이름 열(N). 이어지는 글자 열은 segment/sales_type.
    others = cols[1:]
    lead_extra: list[int] = []
    i = 0
    while i < len(others) and prof.roles[others[i]] == "c":
        lead_extra.append(others[i])
        i += 1
    rest = others[i:]
    while rest and prof.roles[rest[-1]] == "c":
        rest = rest[:-1]
    s = "".join(prof.roles[k] for k in rest)
    nlead = 1 + len(lead_extra)
    suffix = "+L" if nlead > 1 else ""
    pk = lambda ks: [_period_key(prof.path.get(k)) for k in ks]  # noqa: E731
    base = None
    units: list[dict] = []
    if s == "A":
        base = "C-A"
        units = [{"a": rest[0], "p": None}]
        if not _distinct_periods(pk(rest), need_all=False):
            base = None
    elif s == "AA":
        if _distinct_periods(pk(rest)):
            base = "C-A*2"
            units = [{"a": k, "p": None} for k in rest]
    elif s == "AP":
        base = "C-AP"
        units = [{"a": rest[0], "p": rest[1]}]
        if len({k for k in pk(rest) if k}) > 1:
            base = None
    elif s == "APAP":
        pairs = [(rest[0], rest[1]), (rest[2], rest[3])]
        keys = [_period_key(prof.path.get(a)) or _period_key(prof.path.get(p)) for a, p in pairs]
        if _distinct_periods(keys):
            base = "C-AP*2"
            units = [{"a": a, "p": p} for a, p in pairs]
    elif re.fullmatch(r"P+", s) and s:
        base = "C-P*n"
        units = [{"a": None, "p": k} for k in rest]
    elif re.fullmatch(r"A{3,}", s):
        if _distinct_periods(pk(rest)):
            base = "C-A*n"
            units = [{"a": k, "p": None} for k in rest]
    if base is None:
        return {"adapter": NO_ADAPTER}
    adapter = base + suffix
    judged = (("customer", base) in JUDGED) and not suffix
    return {
        "adapter": adapter,
        "judged": judged,
        "name_col": first,
        "lead": lead_extra,
        "units": units,
    }


def plan_segment(prof: Prof) -> dict:
    lead, rest, s = _split_layout(prof, first_is_name=False)
    if not lead or not rest:
        return {"adapter": NO_ADAPTER}
    pk = lambda ks: [_period_key(prof.path.get(k)) for k in ks]  # noqa: E731
    adapter = None
    units: list[dict] = []
    if re.fullmatch(r"A{2,}", s):
        if _distinct_periods(pk(rest)):
            adapter = "S-A*n"
            units = [{"a": k, "p": None} for k in rest]
    elif s == "AP":
        if len({k for k in pk(rest) if k}) <= 1:
            adapter = "S-AP"
            units = [{"a": rest[0], "p": rest[1]}]
    elif re.fullmatch(r"(AP){2,}", s):
        pairs = [(rest[i], rest[i + 1]) for i in range(0, len(rest), 2)]
        keys = [_period_key(prof.path.get(a)) or _period_key(prof.path.get(p)) for a, p in pairs]
        if _distinct_periods(keys):
            adapter = "S-AP*n"
            units = [{"a": a, "p": p} for a, p in pairs]
    elif re.fullmatch(r"A{2,}P{2,}", s):
        n_a = s.count("A")
        if n_a == s.count("P"):
            a_cols, p_cols = rest[:n_a], rest[n_a:]
            if _distinct_periods(pk(a_cols)):
                adapter = "S-A*nP*n"
                units = [{"a": a, "p": p} for a, p in zip(a_cols, p_cols)]
    elif re.fullmatch(r"M+", s):
        keys = pk(rest)
        if (
            len(rest) == 1
            and len({k for k in keys if k}) <= 1
            or (len(rest) > 1 and _distinct_periods(keys))
        ):
            adapter = "S-AP-in-cell"
            units = [{"a": None, "p": None, "m": k} for k in rest]
    if adapter is None:
        return {"adapter": NO_ADAPTER}
    tail = [k for k in prof.cols if k > rest[-1] and prof.roles[k] == "c"]
    return {
        "adapter": adapter,
        "judged": ("segment_sales", adapter) in JUDGED,
        "lead": lead,
        "tail": tail,
        "units": units,
    }


_ORD_GROUPS = (
    ("total", re.compile(r"수주총액|총수주|수주액|수주금액|수주총")),
    ("delivered", re.compile(r"기납품|납품액|납품|기성")),
    ("backlog", re.compile(r"수주잔고|수주잔액|잔고|잔액")),
)


def _order_gm(prof: Prof, k: int) -> tuple[str, str] | None:
    p = prof.path.get(k)
    if not p:
        return None
    full = norm(p)
    group = None
    for g, rx in _ORD_GROUPS:
        if g == "total" and re.search(r"수주잔고|수주잔액|기납품", full):
            continue
        if rx.search(full):
            group = g
            break
    if group is None:
        return None
    measure = "qty" if prof.roles[k] == "q" else "amt"
    return group, measure


def plan_orderbook(prof: Prof) -> dict:
    lead, rest, s = _split_layout(prof, first_is_name=False)
    if not lead or not rest or not set(s) <= {"A", "q"}:
        return {"adapter": NO_ADAPTER}
    gm = [_order_gm(prof, k) for k in rest]
    if any(x is None for x in gm):
        return {"adapter": NO_ADAPTER}
    t, d, b = "total", "delivered", "backlog"
    q, a = "qty", "amt"
    seqs = {
        "O-QA3": [(t, q), (t, a), (d, q), (d, a), (b, q), (b, a)],
        "O-A3": [(t, a), (d, a), (b, a)],
        "O-TA": [(t, a), (b, a)],
    }
    for ad, seq in seqs.items():
        if gm == seq:
            return {
                "adapter": ad,
                "judged": ("orderbook", ad) in JUDGED,
                "lead": lead,
                "cols": dict(zip(gm, rest)),
                "units": [],
            }
    return {"adapter": NO_ADAPTER}


# ------------------------------------------------------------------ 표 배정 (대상·어댑터 목록, 값 없음)


def assign_table(section_key: str, cells: list[dict]) -> dict | None:
    """표 하나의 ``target``·``adapter``. 대상이 아니면 None. 해석 값은 만들지 않는다."""
    prof = Prof(section_key, cells)
    if not prof.body:
        return None
    target = detect_target(section_key, prof)
    if target is None:
        return None
    ustat, _ = unit_parse(cells[0].get("unit_raw"))
    info = {
        "target": target,
        "adapter": NOT_IN_SCOPE,
        "judged": False,
        "tmpl_sig": signature(prof),
        "n_body_rows": prof.n_body_rows,
        "n_periods": 1,
        "unit_status": ustat,
        "parse_status": "partial" if prof.partial() else "ok",
        "n_lead": None,
    }
    plan: dict = {"adapter": NOT_IN_SCOPE}
    if target in ("customer_biz", "rawmat_price"):
        return {**info, "_plan": plan, "_prof": prof}
    if prof.partial():
        plan = {"adapter": NO_ADAPTER}
    elif target == "customer":
        plan = plan_customer(prof)
    elif target == "orderbook":
        plan = plan_orderbook(prof)
    else:
        plan = plan_segment(prof)
    info["adapter"] = plan["adapter"]
    info["judged"] = bool(plan.get("judged"))
    if plan.get("units"):
        info["n_periods"] = len(plan["units"])
    if target == "customer" and plan.get("adapter") != NO_ADAPTER:
        info["n_lead"] = 1 + len(plan["lead"])
    elif plan.get("lead") is not None:
        info["n_lead"] = len(plan["lead"])
    return {**info, "_plan": plan, "_prof": prof}


# ------------------------------------------------------------------ 해석 (행 만들기)

COMMON_COLS = [
    "rcept_no",
    "rcept_dt",
    "corp_code",
    "stock_code",
    "corp_cls",
    "report_nm",
    "src_zip_sha256",
    "parser_version",
    "parse_status",
    "parsed_at",
    "available_from",
    "section_key",
    "table_idx",
    "row_idx",
    "period_idx",
    "tmpl_sig",
    "tmpl_group",
    "unit_raw",
    "unit_src",
    "unit_status",
    "unit_factor",
    "row_kind",
    "value_rowspan",
    "hdr_rule",
    "n_src_cells",
    "fy0",
    "fy0_src",
    "fy_end_month",
]

CUSTOMER_COLS = [
    "source",
    "scope",
    "name_raw",
    "name_note_raw",
    "anonymized",
    "anon_label",
    "segment_raw",
    "sales_type_raw",
    "col_header_raw",
    "period_label_raw",
    "period_rel",
    "fiscal_year",
    "period_irregular",
    "period_conflict",
    "period_inferred",
    "amount_num",
    "amount_raw",
    "amount_krw",
    "ratio_pct",
    "ratio_raw",
    "dup_group_id",
    "dup_of_consol",
]
SEGMENT_COLS = [
    "cat1_raw",
    "cat2_raw",
    "cat3_raw",
    "cat4_raw",
    "cat_path_raw",
    "desc_raw",
    "region_raw",
    "entity_raw",
    "col_header_raw",
    "period_label_raw",
    "period_rel",
    "fiscal_year",
    "period_irregular",
    "period_conflict",
    "period_inferred",
    "amount_num",
    "amount_raw",
    "amount_krw",
    "ratio_pct",
    "ratio_raw",
    "is_adjust_row",
    "dup_group_id",
]
ORDERBOOK_COLS = [
    "item_raw",
    "orderer_raw",
    "project_raw",
    "extra_text_raw",
    "order_date_raw",
    "order_date",
    "due_date_raw",
    "due_date",
    "total_qty",
    "total_qty_raw",
    "total_amt_num",
    "total_amt_raw",
    "total_amt_krw",
    "delivered_qty",
    "delivered_qty_raw",
    "delivered_amt_num",
    "delivered_amt_raw",
    "delivered_amt_krw",
    "backlog_qty",
    "backlog_qty_raw",
    "backlog_amt_num",
    "backlog_amt_raw",
    "backlog_amt_krw",
    "qty_unit_raw",
    "currency_raw",
    "fiscal_year",
    "dup_group_id",
]
TABLE_COLS = {
    "customer": COMMON_COLS + CUSTOMER_COLS,
    "orderbook": COMMON_COLS + ORDERBOOK_COLS,
    "segment_sales": COMMON_COLS + SEGMENT_COLS,
}
BOOL_COLS = {
    "anonymized",
    "period_irregular",
    "period_conflict",
    "period_inferred",
    "dup_of_consol",
    "is_adjust_row",
}
INT_COLS = {
    "table_idx",
    "row_idx",
    "period_idx",
    "period_rel",
    "fiscal_year",
    "value_rowspan",
    "n_src_cells",
    "fy0",
    "fy_end_month",
}
FLOAT_COLS = {
    "unit_factor",
    "amount_num",
    "amount_krw",
    "ratio_pct",
    "total_qty",
    "total_amt_num",
    "total_amt_krw",
    "delivered_qty",
    "delivered_amt_num",
    "delivered_amt_krw",
    "backlog_qty",
    "backlog_amt_num",
    "backlog_amt_krw",
}


def _carry_texts(prof: Prof, text_cols: list[int]):
    """행마다 ``{열: 글자}``를 낸다. rowspan으로 이어진 글자 칸은 아래 행에 같은 값을 둔다(§2.1)."""
    carry: dict[int, tuple[str, int]] = {}
    for ri in sorted(prof.rows):
        rowc = prof.rows[ri]
        covered: set[int] = set()
        for col, c in rowc.items():
            cs = c.get("colspan") or 1
            if cs > 1:
                covered.update(range(col + 1, col + cs))
        texts: dict[int, str] = {}
        for col in text_cols:
            if col in rowc:
                c = rowc[col]
                t = clean_text(_txt(c))
                rs = c.get("rowspan") or 1
                if rs > 1:
                    carry[col] = (t, ri + rs - 1)
                else:
                    carry.pop(col, None)
                texts[col] = t
            elif col in carry and col not in covered and ri <= carry[col][1]:
                texts[col] = carry[col][0]
            else:
                if col in carry and ri > carry[col][1]:
                    carry.pop(col, None)
        yield ri, rowc, texts


def _base_row(meta: dict, prof: Prof, key: tuple, ctx: dict) -> dict:
    r = dict.fromkeys(COMMON_COLS)
    for k in (
        "rcept_no",
        "rcept_dt",
        "corp_code",
        "stock_code",
        "corp_cls",
        "report_nm",
        "src_zip_sha256",
        "parsed_at",
        "available_from",
    ):
        r[k] = meta.get(k)
    r.update(
        parser_version=INTERP_VERSION,
        parse_status=ctx["parse_status"],
        section_key=key[0],
        table_idx=key[1],
        tmpl_sig=ctx["tmpl_sig"],
        tmpl_group=ctx["tmpl_group"],
        unit_raw=ctx["unit_raw"],
        unit_src=ctx["unit_src"],
        unit_status=ctx["unit_status"],
        unit_factor=ctx["unit_factor"],
        hdr_rule=ctx["hdr_rule"],
        fy0=ctx["fy0"],
        fy0_src=ctx["fy0_src"],
        fy_end_month=ctx["fy_end_month"],
    )
    return r


def _ctx(meta: dict, cells: list[dict], assign: dict, target: str) -> dict:
    ustat, fac = unit_parse(cells[0].get("unit_raw"))
    fy0, endm, src = fy0_of(meta.get("report_nm"), meta.get("rcept_dt"))
    return {
        "parse_status": assign["parse_status"],
        "tmpl_sig": assign["tmpl_sig"],
        "tmpl_group": f"{target}:{assign['adapter']}",
        "unit_raw": cells[0].get("unit_raw"),
        "unit_src": cells[0].get("unit_src"),
        "unit_status": ustat,
        "unit_factor": fac,
        "hdr_rule": cells[0].get("tbl_header_rule"),
        "fy0": fy0,
        "fy_end_month": endm,
        "fy0_src": src,
    }


def _krw(num: float | None, ctx: dict) -> float | None:
    if num is None or ctx["unit_status"] != "krw" or ctx["unit_factor"] is None:
        return None
    return num * ctx["unit_factor"]


def _cell_text(c: dict | None) -> str | None:
    if c is None:
        return None
    t = _txt(c)
    return t if t else None


def _read_unit(unit: dict, rowc: dict[int, dict]) -> dict | None:
    """행에서 한 기간 단위(금액·비율 또는 금액(비율) 칸)를 읽는다. 값 칸이 전부 비면 None."""
    out = {
        "amount_num": None,
        "amount_raw": None,
        "ratio_pct": None,
        "ratio_raw": None,
        "rs": 1,
        "n": 0,
    }
    a = rowc.get(unit["a"]) if unit.get("a") is not None else None
    p = rowc.get(unit["p"]) if unit.get("p") is not None else None
    m = rowc.get(unit["m"]) if unit.get("m") is not None else None
    any_val = False
    if a is not None and _cell_text(a):
        any_val = True
        out["amount_raw"] = _txt(a)
        out["amount_num"] = a.get("num_value")
        out["rs"] = a.get("rowspan") or 1
        out["n"] += 1
    if p is not None and _cell_text(p):
        any_val = True
        out["ratio_raw"] = _txt(p)
        out["ratio_pct"] = p.get("num_value")
        if out["n"] == 0:
            out["rs"] = p.get("rowspan") or 1
        out["n"] += 1
    if m is not None and _cell_text(m):
        any_val = True
        raw = _txt(m)
        sp = split_amount_ratio(raw)
        out["amount_raw"] = raw
        out["rs"] = m.get("rowspan") or 1
        out["n"] += 1
        if sp:
            out["amount_num"], out["ratio_pct"] = sp
            out["ratio_raw"] = raw
        else:
            out["amount_num"] = m.get("num_value")
    return out if any_val else None


def interp_customer(
    meta: dict, section_key: str, table_idx: int, cells: list[dict], assign: dict
) -> list[dict]:
    plan, prof = assign["_plan"], assign["_prof"]
    ctx = _ctx(meta, cells, assign, "customer")
    units = plan["units"]
    pers = resolve_periods(
        [period_component(prof.path.get(u["a"] if u["a"] is not None else u["p"])) for u in units],
        ctx["fy0"],
    )
    text_cols = [plan["name_col"], *plan["lead"]]
    src = SRC_NOTE[section_key]
    scope = "consol" if src == "note_consol" else "sep"
    rows = []
    for ri, rowc, texts in _carry_texts(prof, text_cols):
        name_t = texts.get(plan["name_col"])
        kind = row_kind_of([texts.get(k, "") for k in text_cols])
        for pi, (u, per) in enumerate(zip(units, pers)):
            v = _read_unit(u, rowc)
            if v is None:
                continue
            nm, note = split_name_note(name_t) if name_t else (None, None)
            anon, label = anon_info(nm)
            r = _base_row(meta, prof, (section_key, table_idx), ctx)
            r.update(
                row_idx=ri, period_idx=pi, row_kind=kind, value_rowspan=v["rs"], n_src_cells=v["n"]
            )
            r.update(dict.fromkeys(CUSTOMER_COLS))
            r.update(
                source=src,
                scope=scope,
                name_raw=nm or None,
                name_note_raw=note,
                anonymized=anon,
                anon_label=label,
                segment_raw=(texts.get(plan["lead"][0]) or None) if plan["lead"] else None,
                sales_type_raw=(texts.get(plan["lead"][1]) or None)
                if len(plan["lead"]) > 1
                else None,
                col_header_raw=prof.path.get(u["a"] if u["a"] is not None else u["p"]),
                period_label_raw=per["period_label_raw"],
                period_rel=per["period_rel"],
                fiscal_year=per["fiscal_year"],
                period_irregular=per["period_irregular"],
                period_conflict=per["period_conflict"],
                period_inferred=per["period_inferred"],
                amount_num=v["amount_num"],
                amount_raw=v["amount_raw"],
                amount_krw=_krw(v["amount_num"], ctx),
                ratio_pct=v["ratio_pct"],
                ratio_raw=v["ratio_raw"],
                dup_of_consol=False,
            )
            rows.append(r)
    return rows


def interp_segment(
    meta: dict, section_key: str, table_idx: int, cells: list[dict], assign: dict
) -> list[dict]:
    plan, prof = assign["_plan"], assign["_prof"]
    ctx = _ctx(meta, cells, assign, "segment_sales")
    units = plan["units"]
    pers = resolve_periods(
        [
            period_component(prof.path.get(u["m"] if u.get("m") is not None else u["a"]))
            or (period_component(prof.path.get(u["p"])) if u.get("p") is not None else None)
            for u in units
        ],
        ctx["fy0"],
    )
    cat_cols, desc_cols, region_cols, entity_cols = [], [], [], []
    for k in plan["lead"]:
        h = prof.header_norm(k)
        if RE_REGION.search(h):
            region_cols.append(k)
        elif RE_ENTITY.search(h):
            entity_cols.append(k)
        elif RE_DESC.search(h):
            desc_cols.append(k)
        else:
            cat_cols.append(k)
    desc_cols += plan["tail"]
    text_cols = plan["lead"] + plan["tail"]
    rows = []
    for ri, rowc, texts in _carry_texts(prof, text_cols):
        kind = row_kind_of([texts.get(k, "") for k in plan["lead"]])
        cats = [texts.get(k) or None for k in cat_cols]
        path = [x for x in cats if x]
        dedup = [x for i, x in enumerate(path) if i == 0 or x != path[i - 1]]
        for pi, (u, per) in enumerate(zip(units, pers)):
            v = _read_unit(u, rowc)
            if v is None:
                continue
            r = _base_row(meta, prof, (section_key, table_idx), ctx)
            r.update(
                row_idx=ri, period_idx=pi, row_kind=kind, value_rowspan=v["rs"], n_src_cells=v["n"]
            )
            r.update(dict.fromkeys(SEGMENT_COLS))
            r.update(
                cat1_raw=cats[0] if len(cats) > 0 else None,
                cat2_raw=cats[1] if len(cats) > 1 else None,
                cat3_raw=cats[2] if len(cats) > 2 else None,
                cat4_raw=cats[3] if len(cats) > 3 else None,
                cat_path_raw=" > ".join(dedup) or None,
                desc_raw=" | ".join(x for x in (texts.get(k) for k in desc_cols) if x) or None,
                region_raw=" | ".join(x for x in (texts.get(k) for k in region_cols) if x) or None,
                entity_raw=" | ".join(x for x in (texts.get(k) for k in entity_cols) if x) or None,
                col_header_raw=prof.path.get(u["m"] if u.get("m") is not None else u["a"]),
                period_label_raw=per["period_label_raw"],
                period_rel=per["period_rel"],
                fiscal_year=per["fiscal_year"],
                period_irregular=per["period_irregular"],
                period_conflict=per["period_conflict"],
                period_inferred=per["period_inferred"],
                amount_num=v["amount_num"],
                amount_raw=v["amount_raw"],
                amount_krw=_krw(v["amount_num"], ctx),
                ratio_pct=v["ratio_pct"],
                ratio_raw=v["ratio_raw"],
                is_adjust_row=kind == "adjust",
            )
            rows.append(r)
    return rows


_RE_DATE = re.compile(r"(\d{4})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})")


def parse_date(text: str | None) -> str | None:
    if not text:
        return None
    m = _RE_DATE.search(unicodedata.normalize("NFKC", text))
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3))).isoformat()
    except ValueError:
        return None


def interp_orderbook(
    meta: dict, section_key: str, table_idx: int, cells: list[dict], assign: dict
) -> list[dict]:
    plan, prof = assign["_plan"], assign["_prof"]
    ctx = _ctx(meta, cells, assign, "orderbook")
    roles: dict[int, str] = {}
    item_set = False
    for k in plan["lead"]:
        h = prof.header_norm(k)
        if re.search("수주일|계약일|일자", h):
            roles[k] = "order_date"
        elif re.search("납기|완료|인도|종료", h):
            roles[k] = "due_date"
        elif "통화" in h:
            roles[k] = "currency"
        elif re.search("발주처|수주처|고객|거래처", h):
            roles[k] = "orderer"
        elif re.search("공사명|사업명|프로젝트|계약명|건명", h):
            roles[k] = "project"
        elif not item_set:
            roles[k] = "item"
            item_set = True
        else:
            roles[k] = "extra"
    colmap = plan["cols"]
    qty_cols = [c for (g, m), c in colmap.items() if m == "qty"]
    rows = []
    for ri, rowc, texts in _carry_texts(prof, plan["lead"]):
        vals = {gm: rowc.get(c) for gm, c in colmap.items()}
        if not any(c is not None and _cell_text(c) for c in vals.values()):
            continue
        kind = row_kind_of(
            [
                texts.get(k, "")
                for k in plan["lead"]
                if roles[k] in ("item", "orderer", "project", "extra")
            ]
        )
        r = _base_row(meta, prof, (section_key, table_idx), ctx)
        r.update(dict.fromkeys(ORDERBOOK_COLS))
        r.update(row_idx=ri, period_idx=0, row_kind=kind, fiscal_year=ctx["fy0"])
        r["n_src_cells"] = sum(1 for c in vals.values() if c is not None and _cell_text(c))
        r["value_rowspan"] = max((c.get("rowspan") or 1) for c in vals.values() if c is not None)
        extras = []
        for k, role in roles.items():
            t = texts.get(k) or None
            if role == "item":
                r["item_raw"] = t
            elif role == "orderer":
                r["orderer_raw"] = t
            elif role == "project":
                r["project_raw"] = t
            elif role == "order_date":
                r["order_date_raw"] = t
                r["order_date"] = parse_date(t)
            elif role == "due_date":
                r["due_date_raw"] = t
                r["due_date"] = parse_date(t)
            elif role == "currency":
                r["currency_raw"] = t
            elif t:
                extras.append(t)
        r["extra_text_raw"] = " | ".join(extras) or None
        for (g, m), c in colmap.items():
            cell = rowc.get(c)
            num = cell.get("num_value") if cell is not None else None
            raw = _cell_text(cell)
            if m == "qty":
                r[f"{g}_qty"] = num
                r[f"{g}_qty_raw"] = raw
            else:
                r[f"{g}_amt_raw"] = raw
                r[f"{g}_amt_num"] = num
                r[f"{g}_amt_krw"] = _krw(num, ctx)
        if qty_cols:
            r["qty_unit_raw"] = ctx["unit_raw"]
        if r["currency_raw"] is None and ctx["unit_status"] == "foreign":
            r["currency_raw"] = ctx["unit_raw"]
        rows.append(r)
    return rows


INTERPRETERS = {
    "customer": interp_customer,
    "orderbook": interp_orderbook,
    "segment_sales": interp_segment,
}


def interp_table(
    meta: dict, section_key: str, table_idx: int, cells: list[dict], assign: dict | None = None
) -> list[dict]:
    """표 하나를 해석 행으로 바꾼다. 대상이 아니거나 어댑터가 없으면 빈 목록."""
    a = assign or assign_table(section_key, cells)
    if a is None or a["target"] not in INTERPRETERS or a["adapter"] in (NO_ADAPTER, NOT_IN_SCOPE):
        return []
    return INTERPRETERS[a["target"]](meta, section_key, table_idx, cells, a)


# ------------------------------------------------------------------ 연결·별도 / 중복 표시


def mark_customer_dups(rows: list[dict]) -> None:
    """같은 문서의 별도 주석 표가 연결 주석 표와 글자까지 같으면 별도 쪽 행에 표시한다(§4.1, 결정 3)."""
    by_tbl: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_tbl[(r["rcept_no"], r["section_key"], r["table_idx"])].append(r)

    def sig(rs: list[dict]):
        return tuple(
            sorted(
                (
                    x["row_idx"],
                    x["period_idx"],
                    x["name_raw"] or "",
                    x["row_kind"],
                    x["amount_raw"] or "",
                    x["ratio_raw"] or "",
                )
                for x in rs
            )
        )

    consol: dict[tuple, tuple] = {}
    for key, rs in by_tbl.items():
        if rs[0]["source"] == "note_consol":
            consol.setdefault((key[0], sig(rs)), key)
    for key, rs in by_tbl.items():
        if rs[0]["source"] == "note_sep":
            ck = consol.get((key[0], sig(rs)))
            if ck is not None:
                gid = f"{key[0]}:{ck[2]}:{key[2]}"
                for x in rs:
                    x["dup_of_consol"] = True
                    x["dup_group_id"] = gid
                for x in by_tbl[ck]:
                    x["dup_group_id"] = gid


def mark_table_dups(rows: list[dict], cols: tuple[str, ...]) -> None:
    """같은 문서에서 내용이 같은 표 쌍(``cols`` 값 집합이 같음)에 ``dup_group_id``를 단다(합치지 않음)."""
    by_tbl: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_tbl[(r["rcept_no"], r["section_key"], r["table_idx"])].append(r)
    groups: dict[tuple, list[tuple]] = collections.defaultdict(list)
    for key, rs in by_tbl.items():
        sg = tuple(sorted(tuple(str(x.get(c)) for c in cols) for x in rs))
        groups[(key[0], sg)].append(key)
    for (rn, _), keys in groups.items():
        if len(keys) > 1:
            gid = f"{rn}:" + ",".join(f"{k[1]}.{k[2]}" for k in sorted(keys))
            for k in keys:
                for x in by_tbl[k]:
                    x["dup_group_id"] = gid


# ------------------------------------------------------------------ 문서 단위 상태 (§2.7)


def build_interp_doc(
    docs: list[str],
    assignments: list[dict],
    rows_by_table: dict[tuple, int],
    passed_groups: set[str] | None = None,
) -> list[dict]:
    """``kr_bizrpt_interp_doc`` — (rcept_no, target)별 한 행.

    ``assignments``: ``rcept_no``·``section_key``·``table_idx``·``target``·``adapter``·``unit_status``·
    ``parse_status``·``judged``. ``rows_by_table``: 표 키 → 낸 행 수. ``passed_groups``: 판정을 통과한
    ``tmpl_group``(``target:adapter``) 집합. None이면 판정 전이라 판정 대상(judged) 군을 통과로 본다.
    ``status``: ok · no_candidate · not_passed_template · unit_failed · parse_partial.
    """
    by_key: dict[tuple, list[dict]] = collections.defaultdict(list)
    for a in assignments:
        if a["target"] in TARGETS:
            by_key[(a["rcept_no"], a["target"])].append(a)
    out = []
    for rn in docs:
        for tg in TARGETS:
            lst = by_key.get((rn, tg), [])
            used: list[dict] = []
            n_rows = 0
            for a in lst:
                grp = f"{tg}:{a['adapter']}"
                ok_grp = (
                    (grp in passed_groups) if passed_groups is not None else bool(a.get("judged"))
                )
                if a["adapter"] not in (NO_ADAPTER, NOT_IN_SCOPE) and ok_grp:
                    used.append(a)
                    n_rows += rows_by_table.get((rn, a["section_key"], a["table_idx"]), 0)
            if not lst:
                status = "no_candidate"
            elif not used:
                status = "not_passed_template"
            elif any(a.get("parse_status") == "partial" for a in used):
                status = "parse_partial"
            elif all(a["unit_status"] != "krw" for a in used):
                status = "unit_failed"
            else:
                status = "ok"
            out.append(
                {
                    "rcept_no": rn,
                    "target": tg,
                    "n_candidate_tables": len(lst),
                    "n_interpreted_tables": len(used),
                    "n_rows": n_rows,
                    "status": status,
                    "tmpl_groups": ",".join(sorted({f"{tg}:{a['adapter']}" for a in used})),
                    "parser_version": INTERP_VERSION,
                }
            )
    return out
