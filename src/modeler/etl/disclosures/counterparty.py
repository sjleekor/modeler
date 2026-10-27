"""계약상대방 이름을 상장사·DART 법인과 잇는다.

**정확 일치와 정규화 일치만 쓴다.** 퍼지·별칭 사전은 쓰지 않는다(계획 ⑥). 사명이 바뀐 회사
(``에스케이하이닉스`` ↔ ``SK하이닉스``)는 못 맞춘 채로 둔다. 연결 실패를 "거래 관계 없음"으로
채우지 않는다. ``cp_class``에 이유를 남긴다.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_right
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
# 규칙 4: 괄호 안에 나라·지역·현지법인 표시가 있으면 모회사가 아니라 해외 법인이다.
# 영문 법인명 병기("(Air Busan Co., Ltd)")는 해당 아니다(법인 접미어가 있으면 건너뛴다).
FOREIGN_REGION_WORDS = (
    "중국|china|인도|india|베트남|vietnam|미국|usa|america|유럽|europe|일본|japan|법인|"
    "폴란드|헝가리|멕시코|브라질|태국|인도네시아|러시아|터키|체코|슬로바키아|독일|영국|캐나다|"
    "호주|싱가포르|홍콩|대만|"
    "상해|상하이|shanghai|북경|베이징|beijing|심천|선전|shenzhen|광저우|guangzhou|"
    "쑤저우|suzhou|칭다오|qingdao|천진|톈진|tianjin|우한|wuhan|충칭|chongqing|난징|nanjing|"
    "항저우|hangzhou|다롄|dalian|옌청|염성|시안|청두|chengdu|둥관|동관|dongguan|"
    "후이저우|huizhou|옌타이|yantai|웨이하이|위해|선양|심양|장가항|yancheng"
)
_PAREN = re.compile(r"[(（]([^()（）]*)[)）]")
_REGION_IN_PAREN = re.compile(FOREIGN_REGION_WORDS, re.I)
_LEGAL_SUFFIX_EN = re.compile(
    r"\b(co|ltd|inc|corp|corporation|company|limited|gmbh|llc|plc|s\.?a|ag|pte|sdn|bhd)\b\.?", re.I
)


def has_foreign_region_paren(s: str) -> bool:
    """괄호 안에 지역·현지법인 표시가 있으면 참. 영문 법인명 병기는 제외한다."""
    for m in _PAREN.finditer(unicodedata.normalize("NFKC", s or "")):
        g = m.group(1)
        if _LEGAL_SUFFIX_EN.search(g):
            continue
        if _REGION_IN_PAREN.search(g):
            return True
    return False


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


def _valid_name(name, ticker) -> str | None:
    """스냅샷 종목명이 비었거나 종목코드와 같으면 이름을 모르는 것으로 본다."""
    if not isinstance(name, str):
        return None
    n = name.strip()
    return n if n and n != str(ticker).strip() else None


@dataclass
class _Snap:
    as_of: str  # YYYYMMDD
    tickers: frozenset
    names: dict  # ticker -> 이름(유효한 것만)
    exact: dict = field(default_factory=dict)  # n_exact -> {ticker}
    norm: dict = field(default_factory=dict)


@dataclass
class PitIndex:
    """공시 접수일 시점의 종목 목록(``stock_master_snapshot``)과 지금 이름 색인.

    상장 여부는 접수일 이하 가장 늦은 스냅샷(첫 스냅샷 전이면 첫 스냅샷)에 종목이 있는지로 정한다.
    """

    now: MasterIndex
    snaps: list  # as_of 오름차순 _Snap
    cur_names: dict  # ticker -> 지금 이름들(stock_master.name·dart stock_name·corp_name)
    t2c: dict  # ticker -> corp_code
    snapshot: str = ""
    excluded: list = field(default_factory=list)  # [(as_of YYYYMMDD, 종목 수, 기준 종목 수)]
    _dates: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self._dates = [x.as_of for x in self.snaps]

    def snap_at(self, rcept_dt: str) -> _Snap:
        d = (rcept_dt or "").replace("-", "")
        i = bisect_right(self._dates, d) - 1
        return self.snaps[max(i, 0)]

    def name_at(self, ticker: str, rcept_dt: str) -> str | None:
        """접수일 이하 가장 늦은 스냅샷 중 이 종목에 이름이 있는 것의 이름. 없으면 None."""
        d = (rcept_dt or "").replace("-", "")
        i = bisect_right(self._dates, d) - 1
        if i < 0:  # 첫 스냅샷 전 접수일은 첫 스냅샷 쪽으로 본다
            i = 0
            for j in range(i, len(self.snaps)):
                n = self.snaps[j].names.get(ticker)
                if n:
                    return n
            return None
        for j in range(i, -1, -1):
            n = self.snaps[j].names.get(ticker)
            if n:
                return n
        return None


PARTIAL_SNAPSHOT_RATIO = 0.8


def drop_partial_snapshots(snaps: list, ratio: float = PARTIAL_SNAPSHOT_RATIO):
    """종목 수가 직전·직후 스냅샷 종목 수 중 작은 값의 ``ratio`` 미만이면 부분 스냅샷으로 뺀다.

    이웃은 원래 목록 기준이다(연쇄 제외 없음). 맨 앞·맨 뒤는 이웃이 하나다.
    """
    n = [len(x.tickers) for x in snaps]
    keep, out = [], []
    for i, sp in enumerate(snaps):
        nb = [n[j] for j in (i - 1, i + 1) if 0 <= j < len(n)]
        if nb and n[i] < ratio * min(nb):
            out.append((sp.as_of, n[i], min(nb)))
        else:
            keep.append(sp)
    return keep, out


def build_pit_index(
    dart_corp_master, stock_master, snapshots, items, snapshot: str = ""
) -> PitIndex:
    """``snapshots``(snapshot_id·as_of_date·fetched_at)·``items``(snapshot_id·ticker·name)로 만든다.

    같은 날짜 스냅샷이 여럿이면 ``fetched_at``이 가장 늦은 것 하나만 쓴다.
    """
    now = build_index(dart_corp_master, stock_master, snapshot)
    t2c: dict[str, str] = {}
    cur: dict[str, set] = {}
    for r in dart_corp_master.itertuples():
        tk = str(r.ticker).strip() if r.ticker not in (None, "") and str(r.ticker) != "None" else ""
        if tk:
            t2c.setdefault(tk, r.corp_code)
            for nm in (r.corp_name, r.stock_name):
                if isinstance(nm, str) and nm.strip():
                    cur.setdefault(tk, set()).add(nm.strip())
    for r in stock_master.itertuples():
        tk = str(r.ticker).strip()
        if isinstance(r.name, str) and r.name.strip():
            cur.setdefault(tk, set()).add(r.name.strip())
    sn = snapshots.copy()
    sn["d8"] = [str(x)[:10].replace("-", "") for x in sn["as_of_date"]]
    sn = sn.sort_values(["d8", "fetched_at"]).drop_duplicates("d8", keep="last")
    by_id = {k: g for k, g in items.groupby("snapshot_id")}
    snaps = []
    for r in sn.itertuples():
        g = by_id.get(r.snapshot_id)
        if g is None:
            continue
        names, tickers = {}, set()
        for tk, nm in zip(g["ticker"], g["name"]):
            tk = str(tk).strip()
            tickers.add(tk)
            v = _valid_name(nm, tk)
            if v:
                names[tk] = v
        sp = _Snap(as_of=r.d8, tickers=frozenset(tickers), names=names)
        for tk, nm in names.items():
            sp.exact.setdefault(n_exact(nm), set()).add(tk)
            k = n_norm(nm)
            if k:
                sp.norm.setdefault(k, set()).add(tk)
        snaps.append(sp)
    snaps, excluded = drop_partial_snapshots(snaps)
    return PitIndex(
        now=now, snaps=snaps, cur_names=cur, t2c=t2c, snapshot=snapshot, excluded=excluded
    )


def _contains_ok(a: str, b: str) -> bool:
    """정규화한 두 이름이 같거나 한쪽이 다른 쪽을 포함한다."""
    x, y = n_norm(a), n_norm(b)
    return bool(x) and bool(y) and (x == y or x in y or y in x)


def classify_part_pit(name: str, idx: PitIndex, rcept_dt: str) -> dict:
    """접수일 시점 기준 분류(결정 10_b 1~5번). 퍼지는 쓰지 않는다."""
    s = (name or "").strip()
    base = dict(
        name_raw=s,
        name_norm=n_norm(s),
        cp_class="",
        matched_corp_code=None,
        matched_ticker=None,
        match_method=None,
        n_candidates=0,
        ticker_ever=None,
        candidate_tickers=None,
        snapshot_as_of=None,
        name_at_date=None,
        name_basis=None,
    )
    if s in ("", "-", "해당없음", "해당 없음"):
        return {**base, "cp_class": "missing"}
    if has_foreign_region_paren(s):
        return {**base, "cp_class": "foreign_subsidiary"}
    snap = idx.snap_at(rcept_dt)
    base["snapshot_as_of"] = f"{snap.as_of[:4]}-{snap.as_of[4:6]}-{snap.as_of[6:]}"
    for k, dn, dnow, method in (
        (n_exact(s), snap.exact, idx.now.exact, "exact"),
        (n_norm(s), snap.norm, idx.now.norm, "normalized"),
    ):
        if not k:
            continue
        snap_t = set(dn.get(k, ()))  # 그때 이름으로 맞은 종목
        now_ents = dnow.get(k, set())  # 지금 이름으로 맞은 (키, corp, ticker)
        if not snap_t and not now_ents:
            continue
        # 후보를 정체(종목코드, 없으면 corp_code)로 묶는다.
        cands: dict[str, dict] = {}
        for tk in snap_t:
            cands.setdefault(tk, dict(ticker=tk, corp=idx.t2c.get(tk), via_snap=True))
        for _, cc, tk in now_ents:
            key = tk or cc
            c = cands.setdefault(key, dict(ticker=tk, corp=cc, via_snap=False))
            c["corp"] = c["corp"] or cc
        listed = [c for c in cands.values() if c["ticker"] and c["ticker"] in snap.tickers]
        if len(listed) > 1:
            return {
                **base,
                "cp_class": "ambiguous_name",
                "match_method": method,
                "n_candidates": len(listed),
                "candidate_tickers": ",".join(sorted(c["ticker"] for c in listed)),
            }
        if len(listed) == 1:
            c = listed[0]
            tk = c["ticker"]
            then = idx.name_at(tk, rcept_dt)
            basis = "snapshot_name"
            if not c["via_snap"]:
                if then is None:
                    basis = "current_name_then_unknown"
                elif any(_contains_ok(then, cn) for cn in idx.cur_names.get(tk, ())):
                    basis = "current_name_confirmed"
                else:
                    return {
                        **base,
                        "cp_class": "renamed_ticker_unverified",
                        "matched_corp_code": c["corp"],
                        "match_method": method,
                        "n_candidates": 1,
                        "candidate_tickers": tk,
                        "name_at_date": then,
                        "name_basis": "current_name_mismatch",
                    }
            return {
                **base,
                "cp_class": f"listed_pit_{method}",
                "matched_corp_code": c["corp"],
                "matched_ticker": tk,
                "match_method": method + ("+then_name_unknown" if then is None else ""),
                "n_candidates": 1,
                "name_at_date": then,
                "name_basis": basis,
            }
        # 후보 회사는 있는데 그때 상장 종목이 아니다.
        with_tk = [c for c in cands.values() if c["ticker"]]
        if with_tk:
            corps = {c["corp"] for c in with_tk if c["corp"]}
            return {
                **base,
                "cp_class": "not_listed_at_date",
                "matched_corp_code": next(iter(corps)) if len(corps) == 1 else None,
                "match_method": method,
                "n_candidates": len(corps) if corps else len(with_tk),
                "ticker_ever": ",".join(sorted(c["ticker"] for c in with_tk)),
            }
        corps = {c["corp"] for c in cands.values() if c["corp"]}
        if len(corps) == 1:
            return {
                **base,
                "cp_class": f"unlisted_dart_{method}",
                "matched_corp_code": next(iter(corps)),
                "match_method": method,
                "n_candidates": 1,
            }
        return {
            **base,
            "cp_class": "ambiguous_name",
            "match_method": method,
            "n_candidates": len(corps),
        }
    return _unmatched_class(s, base)


def _unmatched_class(s: str, base: dict) -> dict:
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


def classify_part(name: str, idx: MasterIndex) -> dict:
    """한 조각의 분류(1.0.0 규칙, 시점 무관). 일치가 하나로 정해질 때만 ``matched_*``를 채운다."""
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
    if has_foreign_region_paren(s):
        return {**base, "cp_class": "foreign_subsidiary"}
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
    return _unmatched_class(s, base)


def classify_counterparty(raw: str | None, idx, rcept_dt: str | None = None) -> list[dict]:
    """상대방 문자열 전체 → 조각별 분류. 통째로 정확 일치하면 나누지 않는다.

    ``idx``가 ``PitIndex``면 ``rcept_dt``(접수일) 시점 기준(1.1.0), ``MasterIndex``면 옛 규칙이다.
    """
    s = (raw or "").strip()
    if isinstance(idx, PitIndex):
        cls = lambda x: classify_part_pit(x, idx, rcept_dt or "")  # noqa: E731
    else:
        cls = lambda x: classify_part(x, idx)  # noqa: E731
    whole = cls(s)
    if whole["match_method"] and whole["n_candidates"] == 1:
        return [{**whole, "part_idx": 0}]
    parts = split_parts(s)
    if len(parts) <= 1:
        return [{**whole, "part_idx": 0}]
    return [{**cls(p), "part_idx": i} for i, p in enumerate(parts)]
