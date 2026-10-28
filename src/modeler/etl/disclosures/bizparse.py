"""사업보고서(b) 절 위치 표와 표 셀 표 (1차, 원문 보존).

입력은 사업보고서 원문 본문(HTML/XML 텍스트), 출력은 dict다(순수 함수). 해석은 하지 않는다:
절 위치·길이·표 수와 표 셀(좌표·span·머리 경로·원문·숫자 변환·단위 줄)만 만든다.
**절 본문 텍스트는 저장하지 않는다**(계획 07 §3 ⑦). 정규식과 표 구조만 쓴다.

절 키(``SECTION_KEYS``)와 판별 규칙
-----------------------------------
II. 사업의 내용(``SECTION-1`` 제목에 ``사업의 내용``) 안에서 찾는다. 제목 비교는 번호·공백·
``(제조서비스업)`` 같은 업종 꼬리표를 뗀 글자로 한다(``사 업의 개요`` 같은 글자 사이 공백 무시).
문단(``<P>``)은 ``&cr;``·``<BR>``에서 줄로 나눠 줄 하나를 블록 하나로 본다.

방법(``method``)
* ``section2``: 하위 ``SECTION-2`` 제목이 아래 규칙에 맞으면 그 절 전체(하위 ``SECTION-3`` 포함).
* ``p-split``: ``SECTION-2``에서 못 찾으면 ``4. 매출에 관한 사항`` 꼴 번호 줄을 경계로 자른다.
  아는 제목(사업의 개요·주요 제품·원재료·매출·수주·위험관리 등)은 번호가 줄지 않을 때만,
  모르는 제목은 대상 절 바로 다음 번호일 때만 그 절을 닫는 경계로 쓴다. 제목 뒤에 본문이
  붙은 긴 줄도 앞머리로 판별한다.
* ``p-split-marker``: 위에서 못 찾은 키만, 번호·``가.``·``(1)``·``1)`` 어떤 표지든 붙은 줄 중
  제목 규칙에 맞는 첫 줄. 끝은 같은 표지 종류이거나 II 안에서 먼저 나온(더 높은) 종류의 다음 줄.
* ``note_heading``: 재무제표 주석(아래).

| 키 | 제목 규칙(번호·공백 제거 뒤) |
|---|---|
| ``main_products`` | ``^(주요)?(제품|서비스)`` |
| ``raw_materials`` | ``원재료`` 포함(긴 줄은 ``^(주요)?원재료``) |
| ``sales`` | ``^(사업부문별|부문별)?매출``. 안에 ``수주상황`` 소제목이 있으면 그 앞까지 |
| ``orders`` | 최상위 ``^수주``. 없으면 ``sales`` 안 ``수주상황·현황·내역·실적`` 등 소제목 |

제목 하나가 두 키에 맞으면(``주요 제품 및 원재료``) 두 키가 같은 범위를 가진다. 다른 키의
시작이
범위 안에 있으면 거기서 끊는다. ``orders``가 줄 중간 소제목(``…(5) 수주상황``)이면 그 줄 전체가
``orders``에 들어가고 ``char_start``는 줄 안 위치(근사)다.

재무제표 주석의 ``주요 고객에 대한 정보``는 ``III. 재무에 관한 사항`` 안(없으면 II 밖 전체)에서
번호 뒤 ``주요 (외부)(고객|거래처|매출처)(에 대한) 정보·내역·현황·별`` 로 시작하는 줄, 또는
번호로 시작하고 ``10%``와 ``고객·거래처·매출처``가 들어 있는 200자 이하 문장(제목 없는 항목)을
찾는다. 범위는 그 줄부터 다음 번호 붙은 줄·다음 ``SECTION``까지다. 가장 가까운 상위 절 제목에
``연결``이 있으면 ``major_customer_note_consol``, 아니면 ``major_customer_note_sep``이고, 각각
첫 번째 줄만 쓴다. ``의존도`` 같은 위험 서술은 제외한다.

``char_start``·``char_end``: 절의 시작·다음 경계 블록의 원문 텍스트 위치(디코딩한 텍스트 기준).
``SECTION``·문단·표는 태그 시작이고, 한 문단 안 둘째 줄부터는 엔티티 길이만큼 어긋날 수 있다(근사).

셀 표 정의
----------
* 표는 최상위 ``TABLE``만이다. 표 안의 표는 바깥 셀의 텍스트에 붙는다(``tbl_has_nested``).
* 셀 = ``TD·TH·TU·TE`` 하나. ``row_idx``·``col_idx``는 rowspan·colspan을 펼친 격자에서
  셀의 왼쪽 위 좌표(0부터)이고 ``rowspan``·``colspan``을 같이 둔다.
* ``is_header``: ① 태그가 ``TH``이거나 ``THEAD`` 안이면 머리. ② 0번 행에 ①의 머리가 하나도
  없고 표가 2행 이상이며 0번 행에 숫자로 읽히는 셀이 없고 글자가 있는 셀이 있으면 0번 행을
  머리로 본다. ``tbl_header_rule``은 ``tag``(0번 행이 ①)·``first_row``(②)·``none``.
* ``header_path_raw``(본문 셀만): 그 셀의 ``col_idx`` 열을 0번 행부터 내려가며 **연속한 머리
  셀**의 원문을 ``|``로 이은 것. 같은 원본 셀(rowspan)은 한 번만, 빈 글자는 건너뜀.
* ``row_label_raw``(본문 셀만, ``col_idx>0``): 같은 행 0번 열 격자 셀의 원문.
* ``text_raw``: 공백 정리만 한 원문(``&cr;``·nbsp·연속 공백을 한 칸으로). 빈 셀은 빈 문자열.
* ``num_value``: ``text_raw`` 전체가 숫자일 때만 값(``1,234``·``12.5``·``12.5%``).
  음수는 앞의 ``-``·``－``·``−``·``△``·``▲`` 또는 괄호 ``(1,234)``. ``-`` 하나나 ``–``는 숫자
  아님(NULL). 단위 글자가 붙은 값(``1,234원``)은 NULL. ``num_has_pct``는 ``%``가 붙었는지.
* ``unit_raw``: ① 표 안 처음 두 행에 ``(단위 : …)``가 든 짧은 셀이 있으면 그 부분
  (``unit_src=in_table``), ② 없으면 표 바로 앞 블록(문단 또는 셀 1\\~2개짜리 표)을 최대 4개
  거슬러 올라가 처음 나온 ``(단위 : …)``(``p_before``·``tbl_before``). 다른 표·절 경계·200자
  넘는 문단을 만나면 멈춘다. 없으면 NULL이다(앞 표의 단위를 이어받지 않는다).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

BIZ_PARSER_VERSION = "disclosure_bizrpt_1.0.0"

SECTION_KEYS = (
    "main_products",
    "raw_materials",
    "sales",
    "orders",
    "major_customer_note_consol",
    "major_customer_note_sep",
)

_WS = re.compile(r"\s+")


def norm_text(s: str) -> str:
    """공백 정리만. ``&cr;``·nbsp·제어 공백을 한 칸으로 줄인다(엔티티 재해석은 하지 않는다)."""
    s = s.replace("&cr;", " ").replace("\xa0", " ").replace("​", "")
    return _WS.sub(" ", s).strip()


def nospace(s: str) -> str:
    return _WS.sub("", s.replace("&cr;", " ").replace("\xa0", " "))


# ---------------------------------------------------------------- 숫자

_NEG_CHARS = "-−－△▲"  # - − － △ ▲
_NUM_RE = re.compile(r"^(?P<sign>[+\-−－△▲]?)\s*(?P<body>\d[\d,]*(?:\.\d+)?|\.\d+)\s*(?P<pct>%?)$")


def parse_num(text: str) -> tuple[float | None, bool, bool]:
    """(값, % 여부, 변환 오류 여부). 숫자가 아니면 (None, False, False)."""
    t = text.strip()
    if not t:
        return None, False, False
    paren = False
    lead = ""
    if len(t) >= 3 and t[0] in _NEG_CHARS and t[1:].lstrip()[:1] in ("(", "（"):
        lead, t = t[0], t[1:].lstrip()  # ``△(1,234)``
    if len(t) >= 2 and t[0] in "(（" and t[-1] in ")）":
        paren = True
        t = t[1:-1].strip()
    if lead:
        t = lead + t
    m = _NUM_RE.match(t)
    if not m:
        return None, False, False
    body = m.group("body")
    if "," in body:
        # 천 단위 쉼표 모양이 아니면(1,2 · 12,34) 숫자로 보지 않는다
        head, *rest = body.split(".")[0].split(",")
        if not (1 <= len(head) <= 3 and all(len(x) == 3 for x in rest)):
            return None, False, False
    try:
        v = float(body.replace(",", ""))
    except ValueError:
        return None, False, True
    if m.group("sign") and m.group("sign") in _NEG_CHARS:
        v = -v
    if paren:
        v = -v if v > 0 else v
    return v, bool(m.group("pct")), False


# ---------------------------------------------------------------- 블록 파서


@dataclass
class RawCell:
    tag: str
    text: str
    rowspan: int
    colspan: int
    in_thead: bool


@dataclass
class RawTable:
    rows: list[list[RawCell]] = field(default_factory=list)
    has_nested: bool = False
    unclosed: bool = False


@dataclass
class Sec:
    level: int
    title: str = ""
    parent: int = -1


class _Scan(HTMLParser):
    """문서를 순서대로 ('sec'|'p'|'t', ...) 블록 목록으로 만든다."""

    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self.line_starts = [0]
        for m in re.finditer("\n", text):
            self.line_starts.append(m.end())
        self.secs: list[Sec] = [Sec(0, "ROOT", -1)]
        self.stack = [0]
        # 블록: (kind, payload, pos, sec_id, is_title)
        self.blocks: list[tuple] = []
        self.title: list[str] | None = None
        self.p: list[str] | None = None
        self.p_lines: list[tuple[int, str]] = []
        self.line_pos: int | None = None
        self.p_pos = 0
        self.tdepth = 0
        self.table: RawTable | None = None
        self.table_pos = 0
        self.row: list[RawCell] | None = None
        self.cell: list[str] | None = None
        self.cell_attrs: tuple = ("td", 1, 1, False)
        self.in_thead = False
        self.sec_block: dict[int, int] = {}

    def _pos(self) -> int:
        ln, off = self.getpos()
        return self.line_starts[ln - 1] + off

    # -- 셀·행 닫기
    def _close_cell(self) -> None:
        if self.cell is not None and self.row is not None:
            tag, rs, cs, th = self.cell_attrs
            self.row.append(RawCell(tag, norm_text("".join(self.cell)), rs, cs, th))
        self.cell = None

    def _close_row(self) -> None:
        self._close_cell()
        if self.row is not None and self.table is not None and self.row:
            self.table.rows.append(self.row)
        self.row = None

    def _end_line(self) -> None:
        """문단 안에서 줄(``&cr;``·``<BR>``)이 끝났다. 글자가 있으면 한 줄을 모은다."""
        if self.p is not None:
            tx = norm_text("".join(self.p))
            if tx:
                pos = self.line_pos if self.line_pos is not None else self.p_pos
                self.p_lines.append((pos, tx))
            self.p = []
            self.line_pos = None

    def _flush_p(self) -> None:
        if self.p is not None:
            self._end_line()
            for pos, tx in self.p_lines:
                self.blocks.append(("p", tx, pos, self.stack[-1], False))
            self.p = None
            self.p_lines = []

    @staticmethod
    def _span(attrs, name: str) -> int:
        for k, v in attrs:
            if k == name and v:
                try:
                    n = int(v.strip())
                except ValueError:
                    return 1
                return n if 1 <= n <= 1000 else 1
        return 1

    def handle_starttag(self, tag, attrs):
        m = re.fullmatch(r"section-(\d)", tag)
        if m:
            self._flush_p()
            lv = int(m.group(1))
            while len(self.stack) > 1 and self.secs[self.stack[-1]].level >= lv:
                self.stack.pop()
            self.secs.append(Sec(lv, "", self.stack[-1]))
            sid = len(self.secs) - 1
            self.stack.append(sid)
            self.sec_block[sid] = len(self.blocks)
            self.blocks.append(("sec", sid, self._pos(), sid, False))
            return
        if self.tdepth == 0:
            if tag == "title":
                self._flush_p()
                self.title = []
            elif tag == "table":
                self._flush_p()
                self.tdepth = 1
                self.table = RawTable()
                self.table_pos = self._pos()
                self.in_thead = False
            elif tag == "p" and self.title is None:
                self._flush_p()
                self.p = []
                self.p_lines = []
                self.line_pos = None
                self.p_pos = self._pos()
            elif tag == "br" and self.p is not None:
                self._end_line()
            return
        # 표 안
        if tag == "table":
            self.tdepth += 1
            if self.table is not None:
                self.table.has_nested = True
            if self.cell is not None:
                self.cell.append(" ")
            return
        if self.tdepth > 1:
            if tag in ("br", "p", "tr", "td", "th", "tu", "te") and self.cell is not None:
                self.cell.append(" ")
            return
        if tag == "thead":
            self.in_thead = True
        elif tag in ("tbody", "tfoot"):
            self.in_thead = False
        elif tag == "tr":
            self._close_row()
            self.row = []
        elif tag in ("td", "th", "tu", "te"):
            self._close_cell()
            if self.row is None:
                self.row = []
            self.cell = []
            self.cell_attrs = (
                tag,
                self._span(attrs, "rowspan"),
                self._span(attrs, "colspan"),
                tag == "th" or self.in_thead,
            )
        elif tag in ("br", "p") and self.cell is not None:
            self.cell.append(" ")

    def handle_endtag(self, tag):
        if re.fullmatch(r"section-\d", tag):
            self._flush_p()
            return
        if self.tdepth == 0:
            if tag == "title" and self.title is not None:
                tt = norm_text("".join(self.title))
                self.title = None
                cur = self.secs[self.stack[-1]]
                if not cur.title and self.stack[-1] != 0:
                    cur.title = tt
                elif tt:
                    self.blocks.append(("p", tt, self._pos(), self.stack[-1], True))
            elif tag == "p":
                self._flush_p()
            return
        if tag == "table":
            self.tdepth -= 1
            if self.tdepth == 0:
                self._close_row()
                if self.table is not None:
                    self.blocks.append(("t", self.table, self.table_pos, self.stack[-1], False))
                self.table = None
            elif self.cell is not None:
                self.cell.append(" ")
            return
        if self.tdepth > 1:
            if tag in ("p", "tr", "td", "th", "tu", "te") and self.cell is not None:
                self.cell.append(" ")
            return
        if tag in ("td", "th", "tu", "te"):
            self._close_cell()
        elif tag == "tr":
            self._close_row()
        elif tag == "p" and self.cell is not None:
            self.cell.append(" ")

    def handle_data(self, d):
        if self.tdepth > 0:
            if self.cell is not None:
                self.cell.append(d)
        elif self.title is not None:
            self.title.append(d)
        elif self.p is not None:
            base = self._pos()
            off = 0
            for i, piece in enumerate(d.split("&cr;")):
                if i > 0:
                    self._end_line()
                    off += 4
                if piece.strip() and self.line_pos is None:
                    self.line_pos = base + off  # 엔티티 길이 차이만큼 근사
                self.p.append(piece)
                off += len(piece)

    def close(self):
        super().close()
        if self.tdepth > 0 and self.table is not None:
            self.table.unclosed = True
            self._close_row()
            self.blocks.append(("t", self.table, self.table_pos, self.stack[-1], False))
            self.table = None
            self.tdepth = 0
        self._flush_p()


def scan_blocks(text: str) -> _Scan:
    s = _Scan(text)
    s.feed(text)
    s.close()
    return s


# ---------------------------------------------------------------- 제목 규칙

_PREFIX = re.compile(
    r"^(?:[\(\[]?(?:\d{1,2}(?:-\d{1,2})?|[ivxIVX]{1,4}|[가-하]|[①-⑩])[\.\)\]]|[-ㆍ·○●▶])\s*"
)
_NUM_HEAD = re.compile(r"^(\d{1,2})\.")
_SUB_NUM = re.compile(r"^\d{1,2}\.\d")
_INDUSTRY_TAG = re.compile(
    r"^\([가-힣]{2,8}업\)"
)  # 2025 서식: ``(제조서비스업)주요 제품 및 서비스``


def heading_core(s: str) -> str:
    k = _PREFIX.sub("", nospace(s), count=1)
    return _INDUSTRY_TAG.sub("", k, count=1)


_RULES = {
    "main_products": re.compile(r"^(주요)?(제품|서비스)"),
    "raw_materials": re.compile(r"원재료"),
    "sales": re.compile(r"^(사업부문별|부문별)?매출"),
    "orders": re.compile(r"^수주"),
}
# 제목 뒤에 본문이 붙은 긴 줄(60자 넘음)은 앞머리만 본다
_RULES_HEAD = {
    "main_products": re.compile(r"^(주요)?(제품|서비스)"),
    "raw_materials": re.compile(r"^(주요)?원재료"),
    "sales": re.compile(r"^(사업부문별|부문별)?매출"),
    "orders": re.compile(r"^수주"),
}
_ORDERS_WORDS = r"수주\s*(?:상황|상항|현황|내역|실적|잔고|사항|계약|에\s*관한)"
_ORDERS_SUB = re.compile(r"^수주(상황|상항|현황|내역|실적|잔고|사항|계약|에관한)")
_ORDERS_INLINE = re.compile(
    r"(?:\(\s*\d{1,2}\s*\)|\d{1,2}\s*\)|[가-하]\s*[\.\)]|\d{1,2}(?:-\d{1,2})?\s*\.|[-ㆍ·○●▶])\s*"
    + _ORDERS_WORDS
)
_KNOWN_TOP = re.compile(
    r"^(사업의개요|사업의내용|주요제품|제품및서비스|제품|원재료|주요원재료|매출|사업부문별매출|수주|위험관리|"
    r"주요계약|경영상의주요계약|연구개발|기타참고|그밖에|기타투자|시장위험|파생상품|영업의현황|영업설비|"
    r"영업의개황|생산및설비|생산설비|생산에관한|생산능력|회사의현황|업계의현황|판매)"
)
_SENTENCE_END = re.compile(r"(합니다|입니다|있습니다|습니다)\.?$|[.。]$")


def _keys_for_title(title: str) -> list[str]:
    core = heading_core(title)
    if len(core) > 60:
        return [k for k, rx in _RULES_HEAD.items() if rx.search(core[:16])]
    return [k for k, rx in _RULES.items() if rx.search(core)]


# ---------------------------------------------------------------- 단위 줄

_UNIT_PAREN = re.compile(r"[\(\[（]\s*단\s*위\s*[:：]?[^\)\]）]{0,40}[\)\]）]")
_UNIT_BARE = re.compile(r"^\W{0,3}단\s*위\s*[:：]\s*\S")


def _unit_in_text(text: str, short_bare: bool = True) -> str | None:
    ms = _UNIT_PAREN.findall(text)
    if ms:
        return norm_text(ms[-1])
    if short_bare and len(text) <= 60 and _UNIT_BARE.match(text):
        return text
    return None


def unit_for_table(blocks: list[tuple], bi: int, lo: int) -> tuple[str | None, str | None]:
    """(unit_raw, unit_src). ``bi``는 표 블록 위치, ``lo``는 범위 시작."""
    tb: RawTable = blocks[bi][1]
    for row in tb.rows[:2]:
        for c in row:
            if c.text and len(c.text) <= 80:
                u = _unit_in_text(c.text)
                if u:
                    return u, "in_table"
    j, n = bi - 1, 0
    while j >= lo and n < 4:
        kind, payload = blocks[j][0], blocks[j][1]
        if kind == "sec":
            break
        if kind == "p":
            if len(payload) > 200:
                break
            u = _unit_in_text(payload)
            if u:
                return u, "p_before"
        else:  # 표
            cells = [c for r in payload.rows for c in r]
            if len(cells) <= 2:
                for c in cells:
                    u = _unit_in_text(c.text)
                    if u:
                        return u, "tbl_before"
            break  # 단위 줄이 아닌 표를 만나면 거기서 멈춘다
        n += 1
        j -= 1
    return None, None


# ---------------------------------------------------------------- 격자


def _grid(tb: RawTable) -> tuple[list[dict], int, int]:
    """셀마다 {r,c,rs,cs,cell}과 (n_rows, n_cols). 겹침은 먼저 놓은 셀이 이긴다."""
    occ: dict[tuple[int, int], int] = {}
    placed: list[dict] = []
    nrows = len(tb.rows)
    ncols = 0
    for r, row in enumerate(tb.rows):
        c = 0
        for cell in row:
            while (r, c) in occ:
                c += 1
            rs = min(cell.rowspan, nrows - r)
            cs = cell.colspan
            idx = len(placed)
            placed.append({"r": r, "c": c, "rs": rs, "cs": cs, "cell": cell})
            for dr in range(rs):
                for dc in range(cs):
                    occ.setdefault((r + dr, c + dc), idx)
            c += cs
            ncols = max(ncols, c)
    for p in placed:
        p["occ"] = occ
    return placed, nrows, ncols


def table_cells(tb: RawTable) -> dict:
    """표 하나 → 셀 dict 목록과 표 요약."""
    placed, nrows, ncols = _grid(tb)
    if not placed:
        return {
            "cells": [],
            "n_rows": nrows,
            "n_cols": ncols,
            "rule": "none",
            "nested": tb.has_nested,
        }
    occ = placed[0]["occ"]
    # 머리 판정
    hdr = [p["cell"].in_thead for p in placed]
    row0 = [i for i, p in enumerate(placed) if p["r"] == 0]
    rule = "none"
    if any(hdr[i] for i in row0):
        rule = "tag"
    elif nrows >= 2 and row0:
        texts = [placed[i]["cell"].text for i in row0]
        if any(texts) and not any(parse_num(t)[0] is not None for t in texts):
            for i in row0:
                hdr[i] = True
            rule = "first_row"
    # 열별 머리 경로: 0번 행부터 연속한 머리 셀
    paths: dict[int, str] = {}

    def col_path(c: int) -> str:
        if c in paths:
            return paths[c]
        out: list[str] = []
        seen: set[int] = set()
        r = 0
        while r < nrows:
            i = occ.get((r, c))
            if i is None or not hdr[i]:
                break
            if i not in seen:
                seen.add(i)
                t = placed[i]["cell"].text
                if t and not (
                    len(t) <= 60 and _unit_in_text(t)
                ):  # ``(단위 : …)`` 칸은 경로에서 뺀다
                    out.append(t)
            r = placed[i]["r"] + placed[i]["rs"]
        paths[c] = "|".join(out)
        return paths[c]

    cells = []
    for i, p in enumerate(placed):
        cell: RawCell = p["cell"]
        v, pct, err = parse_num(cell.text)
        is_h = hdr[i]
        label = None
        if not is_h and p["c"] > 0:
            j = occ.get((p["r"], 0))
            if j is not None and j != i:
                label = placed[j]["cell"].text or None
        cells.append(
            {
                "row_idx": p["r"],
                "col_idx": p["c"],
                "rowspan": p["rs"],
                "colspan": p["cs"],
                "is_header": is_h,
                "header_path_raw": None if is_h else (col_path(p["c"]) or None),
                "row_label_raw": label,
                "text_raw": cell.text,
                "num_value": v,
                "num_has_pct": pct,
                "num_err": err,
            }
        )
    return {"cells": cells, "n_rows": nrows, "n_cols": ncols, "rule": rule, "nested": tb.has_nested}


# ---------------------------------------------------------------- 절 찾기


def _sec_end(
    blocks: list[tuple], secs: list[Sec], start: int, max_level: int, hard_end: int
) -> int:
    """``start`` 이후 처음 나오는 level <= max_level 절 블록 위치(없으면 hard_end)."""
    for i in range(start + 1, hard_end):
        if blocks[i][0] == "sec" and secs[blocks[i][1]].level <= max_level:
            return i
    return hard_end


def _title_of(b: tuple, secs: list[Sec]) -> str:
    return secs[b[1]].title if b[0] == "sec" else (b[1] if b[0] == "p" else "")


def _find_biz(blocks, secs):
    for i, b in enumerate(blocks):
        if b[0] == "sec" and secs[b[1]].level == 1 and "사업의내용" in nospace(secs[b[1]].title):
            return i
    return None


def _find_fin(blocks, secs):
    for i, b in enumerate(blocks):
        if (
            b[0] == "sec"
            and secs[b[1]].level == 1
            and "재무에관한사항" in nospace(secs[b[1]].title)
        ):
            return i
    return None


_MARKERS = (
    ("n.", re.compile(r"^\d{1,2}\s*\.(?!\d)")),
    ("k.", re.compile(r"^[가-하]\s*[\.\)]")),
    ("(n)", re.compile(r"^\(\s*\d{1,2}\s*\)")),
    ("n)", re.compile(r"^\d{1,2}\s*\)")),
    ("(k)", re.compile(r"^\([가-하]\)")),
    ("c", re.compile(r"^[①-⑩]")),
)


def _marker_class(txt: str) -> str | None:
    for name, rx in _MARKERS:
        if rx.match(txt):
            return name
    return None


def _marker_pass(blocks, secs, bi: int, end: int, found: dict) -> None:
    """번호·가나다 등 어떤 표지든 붙은 줄 중 제목 규칙에 맞는 것을 찾는다(아직 못 찾은 키만).

    끝은 같은 표지 종류이거나 II 안에서 먼저 나온 종류(더 높은 단계)의 다음 표지 줄이다."""
    rank: dict[str, int] = {}
    heads: list[tuple[int, str]] = []
    for i in range(bi + 1, end):
        b = blocks[i]
        if b[0] != "p" or len(b[1]) > 100:
            continue
        c = _marker_class(b[1].strip())
        if c is None:
            continue
        rank.setdefault(c, len(rank))
        heads.append((i, c))
    for n, (i, c) in enumerate(heads):
        txt = blocks[i][1]
        if _SENTENCE_END.search(txt) and len(txt) > 40:
            continue
        keys = [k for k in _keys_for_title(txt) if k not in found]
        if not keys:
            continue
        seg_end = end
        for j, c2 in heads[n + 1 :]:
            if rank[c2] <= rank[c]:
                seg_end = j
                break
        for k in keys:
            found[k] = {"b0": i, "b1": seg_end, "heading": txt, "method": "p-split-marker"}


_FIN_FORM = re.compile(
    r"^(\([가-힣]{2,8}업\))?(영업의현황|영업의종류별현황|영업종류별현황|영업설비|재무건전성)"
)


def _biz_form(blocks, secs, bi: int | None) -> str:
    """``financial``(영업의 현황·영업설비 서식) · ``general`` · ``no_biz``(II 절 없음)."""
    if bi is None:
        return "no_biz"
    end = _sec_end(blocks, secs, bi, 1, len(blocks))
    for i in range(bi + 1, end):
        b = blocks[i]
        if b[0] == "sec":
            t = secs[b[1]].title
        elif b[0] == "p" and len(b[1]) < 60 and _NUM_HEAD.match(nospace(b[1])):
            t = b[1]
        else:
            continue
        if _FIN_FORM.match(heading_core(t)):
            return "financial"
    return "general"


def _orders_in_sales(blocks, secs, found: dict) -> None:
    """``sales`` 절 안에 ``수주상황`` 소제목이 있으면 거기서부터를 ``orders``로 나눈다."""
    if "sales" in found and "orders" not in found:
        s = found["sales"]
        for i in range(s["b0"] + 1, s["b1"]):
            b = blocks[i]
            if b[0] == "sec":
                t = secs[b[1]].title
            elif b[0] == "p":
                t = b[1]
                m = _ORDERS_INLINE.search(t)  # 줄 안에 소제목이 끼어든 경우: ``…(5) 수주상황``
                if m:
                    t = t[m.start() :]
                elif len(t) > 40:
                    continue
            else:
                continue
            if _ORDERS_SUB.match(heading_core(t)):
                found["orders"] = {
                    "b0": i,
                    "b1": s["b1"],
                    "heading": t,
                    "method": s["method"],
                    "sub_of_sales": True,
                }
                s["b1"] = i
                break


def _locate_biz_sections(blocks, secs, bi: int) -> tuple[dict, str | None]:
    """II 절에서 4개 키의 (시작 블록, 끝 블록, 제목, 방법). 못 찾은 키는 없다."""
    end = _sec_end(blocks, secs, bi, 1, len(blocks))
    found: dict[str, dict] = {}
    # 방법 1: SECTION-2
    i = bi + 1
    while i < end:
        b = blocks[i]
        if b[0] == "sec" and secs[b[1]].level == 2:
            seg_end = _sec_end(blocks, secs, i, 2, end)
            title = secs[b[1]].title
            for k in _keys_for_title(title):
                if k not in found:
                    found[k] = {"b0": i, "b1": seg_end, "heading": title, "method": "section2"}
            i = seg_end
        else:
            i += 1
    method_tried = "section2" if found else "p-split"
    # 방법 2: 번호 붙은 P 문단
    if not any(k in found for k in ("main_products", "raw_materials", "sales", "orders")):
        cuts: list[int] = []
        last = 0
        last_target = False
        for i in range(bi + 1, end):
            b = blocks[i]
            if b[0] != "p":
                continue
            txt = b[1]
            k = nospace(txt)
            m = _NUM_HEAD.match(k)
            if not m or _SUB_NUM.match(k):
                continue
            num = int(m.group(1))
            core = heading_core(txt)
            known = bool(_KNOWN_TOP.match(core))
            # 아는 제목은 번호가 줄지 않을 때만, 모르는 제목은 대상 절 바로 다음 번호일 때만
            if (known and num >= last) or (
                not known
                and last_target
                and num == last + 1
                and len(txt) < 50
                and not _SENTENCE_END.search(txt)
            ):
                cuts.append(i)
                last = num
                last_target = bool(_keys_for_title(txt))
        for n, i in enumerate(cuts):
            seg_end = cuts[n + 1] if n + 1 < len(cuts) else end
            title = blocks[i][1]
            for k in _keys_for_title(title):
                if k not in found:
                    found[k] = {"b0": i, "b1": seg_end, "heading": title, "method": "p-split"}
        if found:
            method_tried = "p-split"
    _orders_in_sales(blocks, secs, found)
    # 번호 표지가 제각각인 서식: 못 찾은 키만 다시
    if any(k not in found for k in ("main_products", "raw_materials", "sales", "orders")):
        before = set(found)
        _marker_pass(blocks, secs, bi, end, found)
        if set(found) - before and method_tried == "section2":
            method_tried = "section2+marker"
        _orders_in_sales(blocks, secs, found)
    # 겹침 정리: 다른 키의 시작이 이 절 안에 있으면 거기서 끊는다(같은 시작은 그대로)
    for k, v in found.items():
        inner = [w["b0"] for kk, w in found.items() if kk != k and v["b0"] < w["b0"] < v["b1"]]
        if inner:
            v["b1"] = min(inner)
    return found, method_tried


_NOTE_HEAD_START = re.compile(
    r"^\s*(\(\s*\d{1,2}\s*\)|\d{1,2}\s*\)|\d{1,2}\s*\.(?!\d)|[가-하]\s*[\.\)]|\([가-하]\)|[①-⑩]|[■□○●◆▶▷])"
)
_NOTE_PREFIX = re.compile(
    r"^\s*(?:\(\s*\d{1,2}\s*\)|\d{1,2}(?:[-.]\d{1,2})*\s*[\.\)]?|[①-⑩]|[가-하]\s*[\.\)]|\([가-하]\))?\s*"
)
# A·B: 번호 + ``주요 고객에 대한 정보`` 꼴 제목(뒤에 본문이 붙어도 됨)
_NOTE_TITLE = re.compile(
    r"^주요\s*(외부\s*)?(고객|거래처|매출처)\s*(에\s*(대한|관한)\s*)?(정보|내역|현황|별)"
)
# C: 제목 없이 문장으로 시작하는 항목 ``(5) 당기 중 … 매출액의 10% 이상을 차지하는 주요 고객은 …``
_NOTE_SENT = re.compile(r"10\s*%")


def _is_note_heading(txt: str) -> bool:
    m = _NOTE_PREFIX.match(txt)
    rest = txt[m.end() :] if m else txt
    if _NOTE_TITLE.match(rest):
        return True
    if m and m.end() > 0 and len(txt) <= 200:  # 번호로 시작하는 문장
        return bool(_NOTE_SENT.search(rest) and re.search(r"고객|거래처|매출처", rest))
    return False


def _locate_notes(blocks, secs, fin: int | None, biz: int | None) -> dict[str, dict]:
    found: dict[str, dict] = {}
    n = len(blocks)
    if fin is not None:
        lo, hi = fin, _sec_end(blocks, secs, fin, 1, n)
    else:
        lo, hi = 0, n
    biz_end = _sec_end(blocks, secs, biz, 1, n) if biz is not None else -1
    for i in range(lo, hi):
        if biz is not None and biz <= i < biz_end:
            continue
        b = blocks[i]
        if b[0] == "sec":
            txt = secs[b[1]].title
        elif b[0] == "p":
            txt = b[1]
        else:
            continue
        if not _is_note_heading(txt):
            continue
        # 상위 절에 '연결'이 있으면 연결
        sid = b[3]
        consol = False
        while sid > 0:
            if "연결" in secs[sid].title:
                consol = True
                break
            sid = secs[sid].parent
        key = "major_customer_note_consol" if consol else "major_customer_note_sep"
        if key in found:
            continue
        end = hi
        for j in range(i + 1, hi):
            bj = blocks[j]
            if bj[0] == "sec":
                end = j
                break
            if bj[0] == "p" and _NOTE_HEAD_START.match(bj[1]):
                end = j
                break
        found[key] = {"b0": i, "b1": end, "heading": txt, "method": "note_heading"}
    return found


# ---------------------------------------------------------------- 진입점


def parse_biz_text(text: str) -> dict:
    """본문 텍스트 → ``{"sections": {key: dict}, "cells": [...], "flags": [...]}``.

    ``sections[key]``: found, method, miss_reason, heading_raw, char_start, char_end,
    char_len, n_tables, unit_raw, n_tables_with_unit. ``cells``: 셀 dict(section_key,
    table_idx 포함). 키마다 항목이 있다(못 찾아도).
    """
    sc = scan_blocks(text)
    blocks, secs = sc.blocks, sc.secs
    flags: list[str] = []
    bi = _find_biz(blocks, secs)
    fin = _find_fin(blocks, secs)
    located: dict[str, dict] = {}
    method_tried = None
    if bi is not None:
        located, method_tried = _locate_biz_sections(blocks, secs, bi)
    else:
        flags.append("no_biz_section")
    located.update(_locate_notes(blocks, secs, fin, bi))

    out_sec: dict[str, dict] = {}
    cells: list[dict] = []
    for key in SECTION_KEYS:
        loc = located.get(key)
        if loc is None:
            if key.startswith("major_customer_note"):
                reason = "no_fin_section" if fin is None else "no_heading"
            else:
                reason = "no_biz_section" if bi is None else "no_heading"
            out_sec[key] = {
                "found": False,
                "method": None,
                "miss_reason": reason,
                "heading_raw": None,
                "char_start": None,
                "char_end": None,
                "char_len": None,
                "n_tables": None,
                "unit_raw": None,
                "n_tables_with_unit": None,
            }
            continue
        b0, b1 = loc["b0"], loc["b1"]
        char_start = blocks[b0][2]
        char_end = blocks[b1][2] if b1 < len(blocks) else len(text)
        clen = 0
        tcount = 0
        units: list[str] = []
        n_unit = 0
        for i in range(b0, b1):
            b = blocks[i]
            if b[0] == "p":
                clen += len(b[1])
            elif b[0] == "t":
                tb = b[1]
                clen += sum(len(c.text) for r in tb.rows for c in r)
                info = table_cells(tb)
                unit, usrc = unit_for_table(blocks, i, b0)
                if unit:
                    n_unit += 1
                    if unit not in units:
                        units.append(unit)
                for c in info["cells"]:
                    c.update(
                        section_key=key,
                        table_idx=tcount,
                        unit_raw=unit,
                        unit_src=usrc,
                        tbl_n_rows=info["n_rows"],
                        tbl_n_cols=info["n_cols"],
                        tbl_header_rule=info["rule"],
                        tbl_has_nested=info["nested"],
                    )
                    cells.append(c)
                if tb.unclosed:
                    flags.append("unclosed_table")
                tcount += 1
        out_sec[key] = {
            "found": True,
            "method": loc["method"],
            "miss_reason": None,
            "heading_raw": loc["heading"][:120],
            "char_start": char_start,
            "char_end": char_end,
            "char_len": clen,
            "n_tables": tcount,
            "unit_raw": " | ".join(units) if units else None,
            "n_tables_with_unit": n_unit,
        }
    if any(c["num_err"] for c in cells):
        flags.append("num_conv_err")
    return {
        "sections": out_sec,
        "cells": cells,
        "flags": sorted(set(flags)),
        "n_blocks": len(blocks),
        "biz_form": _biz_form(blocks, secs, bi),
    }


def line_offset_check(text: str, pos: int) -> bool:
    """시험용: 위치가 태그 시작(<)을 가리키는가."""
    return 0 <= pos < len(text) and text[pos] == "<"


__all__ = [
    "BIZ_PARSER_VERSION",
    "SECTION_KEYS",
    "parse_biz_text",
    "parse_num",
    "norm_text",
    "scan_blocks",
]
