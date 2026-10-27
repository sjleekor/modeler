"""계약상대방 이름을 상장사·DART 법인과 잇는다.

**정확 일치와 정규화 일치만 쓴다.** 퍼지·별칭 사전은 쓰지 않는다(계획 ⑥). 사명이 바뀐 회사
(``에스케이하이닉스`` ↔ ``SK하이닉스``)는 못 맞춘 채로 둔다. 연결 실패를 "거래 관계 없음"으로
채우지 않는다. ``cp_class``에 이유를 남긴다.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_LEG = re.compile(
    r"\(주\)|㈜|주식회사|\(유\)|유한회사|\(합\)|co\.?,?\s*ltd\.?|ltd\.?|inc\.?|corp\.?|"
    r"corporation|limited|gmbh|llc|plc|\(.*?\)",
    re.I,
)
_ANON = re.compile(
    r"(소재|업체|제조사|제조업체|선사|발주처|고객사|국내\s|해외\s|국외|비공개|공개불가|기밀|"
    r"영업비밀|상대방|거래처|수요처|\s[가-힣]{1,6}사$|등$|외\s?\d+)"
)
_PUBLIC = re.compile(
    r"(공사$|공단$|청$|특별시|광역시|특별자치|[가-힣]{1,4}시\s|[가-힣]{1,4}도\s|정부|국방부|조달청|"
    r"공공|교육청|시청|군청|구청|진흥원|연구원$|협회$|재단)"
)
_FOREIGN_KW = re.compile(
    r"(inc|ltd|corp|co\.|gmbh|llc|plc|s\.a|b\.v|pte|limited|company|ag\b|s\.p\.a|pty|sdn|bhd|"
    r"k\.k|kabushiki|株式会社|有限公司|公司)",
    re.I,
)
# 쉼표 뒤가 법인 접미어면 이름 안의 쉼표(Co., Ltd.)라 나누지 않는다.
_SPLIT = re.compile(
    r"\s*[,;，、/]\s*(?!\s*(?:Ltd|Inc|Co|Corp|LTD|INC|LLC|Limited|CO)\b)|\s+및\s+|\n"
)


def n_exact(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s)).lower()


def n_norm(s: str) -> str:
    s = _LEG.sub("", unicodedata.normalize("NFKC", s))
    return re.sub(r"[\s.,\-·&]", "", s).lower()


def split_parts(raw: str | None) -> list[str]:
    """상대방 문자열을 회사 이름 조각으로 나눈다. 괄호 안 구분자는 나누지 않는다."""
    s = (raw or "").strip()
    if not s:
        return []
    out, depth = [], 0
    # 괄호 안을 지키면서 나눈다: 괄호 안 문자는 임시로 치환
    masked = []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        masked.append("\x00" if depth and ch in ",;/、，" else ch)
    parts = _SPLIT.split("".join(masked))
    for p in parts:
        p = p.replace("\x00", ",").strip()
        if p:
            out.append(p)
    return out or [s]


@dataclass
class MasterIndex:
    exact: dict[str, set[tuple[str, str | None, str | None]]] = field(default_factory=dict)
    norm: dict[str, set[tuple[str, str | None, str | None]]] = field(default_factory=dict)
    snapshot: str = ""

    def add(self, name, corp_code: str | None, ticker: str | None) -> None:
        if not isinstance(name, str) or not name.strip():
            return
        ent = (ticker or corp_code or "", corp_code, ticker)
        if not ent[0]:
            return
        for d, k in ((self.exact, n_exact(name)), (self.norm, n_norm(name))):
            if k:
                d.setdefault(k, set()).add(ent)


def build_index(dart_corp_master, stock_master, snapshot: str) -> MasterIndex:
    """``dart_corp_master``·``stock_master`` DataFrame으로 이름 색인을 만든다.

    상장 여부는 ``dart_corp_master.ticker`` 가 있거나 ``stock_master`` 에 있는 것이다.
    ``stock_master``만 있는 종목은 ``dart_corp_master.ticker`` 로 corp_code를 찾는다.
    """
    idx = MasterIndex(snapshot=snapshot)
    t2c: dict[str, str] = {}
    for r in dart_corp_master.itertuples():
        tk = (
            str(r.ticker).strip()
            if r.ticker not in (None, "") and str(r.ticker) != "None"
            else None
        )
        tk = tk or None
        if tk:
            t2c.setdefault(tk, r.corp_code)
        idx.add(r.corp_name, r.corp_code, tk)
        idx.add(r.stock_name, r.corp_code, tk)
    for r in stock_master.itertuples():
        tk = str(r.ticker).strip()
        idx.add(r.name, t2c.get(tk), tk)
    return idx


def classify_part(name: str, idx: MasterIndex) -> dict:
    """한 조각의 분류. 일치가 하나로 정해질 때만 ``matched_*``를 채운다."""
    s = (name or "").strip()
    base = dict(
        name_raw=s,
        name_norm=n_norm(s),
        cp_class="",
        matched_corp_code=None,
        matched_ticker=None,
        match_method=None,
        n_candidates=0,
    )
    if s in ("", "-", "해당없음", "해당 없음"):
        return {**base, "cp_class": "missing"}
    for k, d, method in ((n_exact(s), idx.exact, "exact"), (n_norm(s), idx.norm, "normalized")):
        cands = d.get(k) if k else None
        if cands:
            # 같은 법인(corp_code)을 가리키는 항목은 한 개로 센다.
            by_corp: dict[str, tuple] = {}
            for ent in cands:
                by_corp.setdefault(ent[1] or ent[0], ent)
            ents = list(by_corp.values())
            n = len(ents)
            if n == 1:
                _, cc, tk = ents[0]
                cls = "listed" if tk else "unlisted_dart"
                return {
                    **base,
                    "cp_class": f"{cls}_{method}",
                    "matched_corp_code": cc,
                    "matched_ticker": tk,
                    "match_method": method,
                    "n_candidates": 1,
                }
            return {**base, "cp_class": "ambiguous_name", "match_method": method, "n_candidates": n}
    latin = len(re.findall(r"[A-Za-z]", s))
    hang = len(re.findall(r"[가-힣]", s))
    if (latin > hang and (_FOREIGN_KW.search(s) or latin >= 4)) or re.search(
        r"유한공사|有限|株式", s
    ):
        return {**base, "cp_class": "foreign_or_latin_name"}
    if _PUBLIC.search(s):
        return {**base, "cp_class": "public_sector_unmatched"}
    if _ANON.search(s) and not re.search(r"\(주\)|㈜|주식회사", s):
        return {**base, "cp_class": "anonymous_descriptor"}
    return {**base, "cp_class": "unmatched_korean_name"}


def classify_counterparty(raw: str | None, idx: MasterIndex) -> list[dict]:
    """상대방 문자열 전체 → 조각별 분류. 통째로 정확 일치하면 나누지 않는다."""
    s = (raw or "").strip()
    whole = classify_part(s, idx)
    if whole["match_method"] and whole["n_candidates"] == 1:
        return [{**whole, "part_idx": 0}]
    parts = split_parts(s)
    if len(parts) <= 1:
        return [{**whole, "part_idx": 0}]
    return [{**classify_part(p, idx), "part_idx": i} for i, p in enumerate(parts)]
