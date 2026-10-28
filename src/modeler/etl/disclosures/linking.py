"""해지 → 최초 체결 연결과 ``※관련공시`` 쌍 연결 (계획 §5).

후보가 하나일 때만 연결한다. 못 줄이면 ``n_candidates > 1``로 두고 연결하지 않는다.
후보가 없으면 ``unlinked_first_not_collected``다. 정정 공시는 수집하지 않았으므로 정정 쌍은
최초 계약 후보에서 뺀다.
"""

from __future__ import annotations

import re
from collections import defaultdict

_STRIP = re.compile(r"\[[^\]]*\]|\([^)]*\)|[\s·ㆍ,.\-]")


def compact_title(t: str | None) -> str:
    """제목 비교용. 괄호·공백·가운뎃점을 지운다."""
    return _STRIP.sub("", t or "")


def title_class(t: str | None) -> str:
    raw = t or ""
    c = compact_title(raw)
    correction = "정정" in raw
    if "해지" in c:
        return "cancel"
    if "체결" in c and "공급계약" in c:
        return "conclusion_correction" if correction else "conclusion"
    if "진행" in c:
        return "progress"
    return "other_correction" if correction else "other"


def _name_key(s: str | None) -> str:
    return re.sub(r"[\s·ㆍ,.\-()]", "", s or "").lower()


class Linker:
    """a(체결)·ga 문서 목록 위에서 후보를 찾는다."""

    def __init__(self, events: list[dict], all_docs: list[dict]) -> None:
        self.ev = defaultdict(list)  # (corp_code, rcept_dt) -> events
        for e in events:
            self.ev[(e["corp_code"], e["rcept_dt"])].append(e)
        self.docs = defaultdict(list)  # (corp_code, rcept_dt) -> 모든 a·ga 문서
        for d in all_docs:
            self.docs[(d["corp_code"], d["rcept_dt"])].append(d)

    def link_related_pair(self, corp_code: str, date_iso: str, title: str, self_rcept: str):
        """쌍 하나 → (related_rcept_no, n_candidates). 같은 법인·접수일·제목이 맞는 문서."""
        key = (corp_code, date_iso.replace("-", ""))
        ct = compact_title(title)
        cands = [
            d
            for d in self.docs.get(key, [])
            if d["rcept_no"] != self_rcept
            and ct
            and (ct in compact_title(d["report_nm"]) or compact_title(d["report_nm"]) in ct)
        ]
        return (cands[0]["rcept_no"] if len(cands) == 1 else None), len(cands)

    def link_cancel(self, cancel: dict, pairs: list[tuple[str, str]]) -> dict:
        res = dict(first_rcept_no=None, link_method=None, n_candidates=0, first_pair_date=None)
        if not pairs:
            res["link_method"] = "no_related"
            return res
        conc = [(d, t) for d, t in pairs if title_class(t) == "conclusion"]
        if not conc:
            only_corr = any(title_class(t) == "conclusion_correction" for _, t in pairs)
            res["link_method"] = "only_correction_pairs" if only_corr else "no_contract_pair"
            return res
        d0, _ = min(conc, key=lambda x: x[0])
        res["first_pair_date"] = d0
        cands = self.ev.get((cancel["corp_code"], d0.replace("-", "")), [])
        res["n_candidates"] = len(cands)
        if not cands:
            res["link_method"] = "unlinked_first_not_collected"
        elif len(cands) == 1:
            res.update(first_rcept_no=cands[0]["rcept_no"], link_method="unique_corp_date")
        else:
            nk = _name_key(cancel.get("contract_name"))
            by_name = [c for c in cands if nk and _name_key(c.get("contract_name")) == nk]
            if len(by_name) == 1:
                res.update(first_rcept_no=by_name[0]["rcept_no"], link_method="name_match")
                return res
            pool = by_name or cands
            amt = (
                cancel.get("amount")
                if cancel.get("amount") is not None
                else cancel.get("cancel_amount")
            )
            by_amt = [c for c in pool if amt is not None and c.get("amount") == amt]
            if len(by_amt) == 1:
                res.update(first_rcept_no=by_amt[0]["rcept_no"], link_method="amount_match")
            else:
                res["link_method"] = "ambiguous_multiple_candidates"
                res["n_candidates"] = len(by_amt or pool)
        return res
