"""요약 선정과 지난 보고서 대비. 문장은 규칙으로 만든 사실 템플릿만 쓴다(해석어 없음).

요약: 영역마다 먼저 1행 -> 남는 자리는 우선순위로, 영역당 최대 2행, 전체 4\\~6행.
우선순위: 표시 기간 중 급 > 중기·단기 방향 차이 > 10년 백분위 극단 > 월간 연속. 같으면 크기 순.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

from modeler.scores.macro_context.calc import Computed, r6
from modeler.scores.macro_context.spec import (
    ARROW,
    FLOW_PLABEL,
    GNAME,
    MINUS,
    ORDER,
    SHORT,
    SUMMARY_MAX,
    SUMMARY_MIN,
    SUMMARY_PER_GROUP,
    VS_PREV_MAX_COUNT,
)


def neg(s: str) -> str:
    return s.replace("-", MINUS)


def sgn(v: float | None, dec: int, suf: str = "", plus: bool = True) -> str:
    if v is None:
        return ""
    if round(v, dec) == 0:
        v = 0.0
    s = f"{v:+,.{dec}f}" if plus else f"{v:,.{dec}f}"
    return neg(s) + suf


def fmt_chg(v: float | None, unit: str, dec: int = 2) -> str:
    if unit == "천명":
        return sgn(v, 0, "천명")
    return sgn(v, dec, unit)


def prev_weekday(d: dt.date) -> dt.date:
    d = d - dt.timedelta(days=1)
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def plabel(o: dict[str, Any], lab: str) -> str:
    return FLOW_PLABEL.get(lab, lab) if o["kind"] == "sum" else lab


def dtext(o: dict[str, Any], x: dict[str, Any]) -> str:
    t = x["dlabel"]
    if o["kind"] == "sum":
        t += " (큰 쪽 10%)"
    return t


def obs_parts(o: dict[str, Any]) -> list[str]:
    c = o["conditions"]
    parts: list[str] = []
    if c["a"]:
        groups: dict[str, list[str]] = {}
        order: list[str] = []
        for x in c["a"]:
            t = dtext(o, x)
            if t not in groups:
                groups[t] = []
                order.append(t)
            groups[t].append(plabel(o, x["label"]))
        parts.append(" · ".join("·".join(groups[t]) + " " + t for t in order))
    if c["b"]:
        b = c["b"]
        parts.append(f"중기 {ARROW[b['long_dir']]} / 단기 {ARROW[b['short_dir']]}")
    if c["c"]:
        parts.append(f"10년 백분위 P{c['c']['p']:.0f}")
    if c["d"]:
        parts.append(c["d"]["text"])
    return parts


def obs_text(o: dict[str, Any]) -> str:
    return " · ".join(obs_parts(o))


def level_text(o: dict[str, Any]) -> str | None:
    """수준 문구. 신용 스프레드 수준은 싣지 않는다(R1)."""
    v = o["value"]
    if o["id"] == "credit_baa" or o["display"]["level"] == "hide":
        return None
    if o["id"] in ("us_10y2y", "krus_10y"):
        return neg(f"{v['level']:.2f}%p")
    return neg(v["display"])


def rep_text(o: dict[str, Any]) -> str:
    c = o["conditions"]
    if c["a"] or c["b"]:
        items: dict[str, tuple[int, float, str, int]] = {}
        for x in c["a"]:
            items[x["label"]] = (x.get("h", 0), x["change"], x["unit"], x["dec"])
        if c["b"]:
            b = c["b"]
            for lab, ch in ((b["short"], b["short_change"]), (b["long"], b["long_change"])):
                p = next(q for q in o["periods"] if q["label"] == lab)
                items[lab] = (p["h"], ch, b["unit"], 2)
        if o["kind"] == "monthly":
            seq = [(lab, items[lab]) for lab in [x["label"] for x in c["a"]]]
        else:
            seq = sorted(items.items(), key=lambda kv: kv[1][0])
        return " · ".join(
            f"{plabel(o, lab)} {fmt_chg(ch, un, dc)}" for lab, (_h, ch, un, dc) in seq
        )
    if c["c"]:
        lt = level_text(o)
        if lt:
            return lt
        p = [q for q in o["periods"] if q.get("change") is not None][-1]
        return f"{p['label']} {fmt_chg(p['change'], '%p')}"
    d = c["d"]
    return f"{d['measure']} {fmt_chg(d['change'], d['unit'])}"


# ------------------------------------------------------------------ 요약 선정
def build_summary(inds: list[dict[str, Any]], mons: list[dict[str, Any]]) -> dict[str, Any]:
    cands = []
    for o in inds + mons:
        c = o["conditions"]
        if not (c["a"] or c["b"] or c["c"] or c["d"]):
            continue
        if c["a"]:
            prio, mag = 0, max(x["mult"] for x in c["a"])
        elif c["b"]:
            prio, mag = 1, c["b"]["mag"]
        elif c["c"]:
            prio, mag = 2, abs(c["c"]["p"] - 50)
        else:
            prio, mag = 3, c["d"]["months"]
        cands.append(dict(o=o, prio=prio, mag=mag))

    def key(r: dict[str, Any]):
        return (r["prio"], -r["mag"], ORDER[r["o"]["id"]])

    cands.sort(key=key)
    # 1차: 후보가 있는 영역마다 대표 1행. 6행을 넘으면 우선순위로 6개
    reps = []
    seen_g: set[str] = set()
    for r in cands:
        g = r["o"]["group"]
        if g not in seen_g:
            seen_g.add(g)
            reps.append(r)
    chosen = reps[:SUMMARY_MAX]
    gcount: dict[str, int] = {}
    for r in chosen:
        gcount[r["o"]["group"]] = gcount.get(r["o"]["group"], 0) + 1
    # 2차: 남는 자리를 남은 후보 전체에서 같은 우선순위로 채운다(영역당 최대 2행)
    for r in cands:
        if len(chosen) >= SUMMARY_MAX:
            break
        if r in chosen or gcount.get(r["o"]["group"], 0) >= SUMMARY_PER_GROUP:
            continue
        chosen.append(r)
        gcount[r["o"]["group"]] = gcount.get(r["o"]["group"], 0) + 1
    if len(chosen) < SUMMARY_MIN:
        for r in cands:
            if len(chosen) >= SUMMARY_MIN:
                break
            if r not in chosen:
                chosen.append(r)
    # 표시 순서: 영역은 그 영역 대표 행의 우선순위·크기 순, 영역 안에서는 우선순위·크기 순
    grank: dict[str, int] = {}
    for r in cands:
        grank.setdefault(r["o"]["group"], len(grank))
    chosen.sort(key=lambda r: (grank[r["o"]["group"]], key(r)))
    rows = []
    for i, r in enumerate(chosen, 1):
        o = r["o"]
        rows.append(
            dict(
                rank=i,
                id=o["id"],
                group=o["group"],
                group_name=GNAME[o["group"]],
                name=SHORT[o["id"]],
                observation=f"{SHORT[o['id']]}: {obs_text(o)}",
                representative=rep_text(o),
                asof=o["asof"],
                stale=o["stale"],
                priority=r["prio"],
                magnitude=r6(r["mag"]),
            )
        )
    return dict(
        rows=rows,
        candidates=len(cands),
        not_selected=len(cands) - len(chosen),
        max_rows=SUMMARY_MAX,
        min_rows=SUMMARY_MIN,
        max_per_group=SUMMARY_PER_GROUP,
    )


# ------------------------------------------------------------------ 지난 보고서 대비
def states(res: Computed) -> dict[tuple, dict[str, Any]]:
    st: dict[tuple, dict[str, Any]] = {}
    for o in res.inds + res.mons:
        c = o["conditions"]
        for x in c["a"]:
            st[("a", o["id"], x["label"], x["dir"])] = dict(
                kind="a",
                id=o["id"],
                period=x["label"],
                dlabel=x["dlabel"],
                text=f"{SHORT[o['id']]} {plabel(o, x['label'])} {dtext(o, x)}",
                h=x.get("h", 0),
            )
        if c["b"]:
            b = c["b"]
            st[("b", o["id"])] = dict(
                kind="b",
                id=o["id"],
                period=f"{b['long']}/{b['short']}",
                dlabel="방향 차이",
                h=0,
                text=f"{SHORT[o['id']]} {plabel(o, b['long'])} {ARROW[b['long_dir']]} / "
                f"{plabel(o, b['short'])} {ARROW[b['short_dir']]} 방향 차이",
            )
    return st


def vs_prev(D: dt.date, res: Computed, compute_at: Callable[[dt.date], Computed]) -> dict[str, Any]:
    """직전 평일 P를 같은 입력의 P 가용분으로 다시 계산해 비교한다. 이어짐은 상한 20회."""
    P = prev_weekday(D)
    sD = states(res)
    cache: dict[dt.date, dict[tuple, dict[str, Any]]] = {}

    def st_at(d: dt.date) -> dict[tuple, dict[str, Any]]:
        if d not in cache:
            cache[d] = states(compute_at(d))
        return cache[d]

    sP = st_at(P)

    def skey(k: tuple, s: dict[str, Any]):
        return (ORDER[s["id"]], s["h"], k[0])

    new = [(k, s) for k, s in sD.items() if k[0] == "a" and k not in sP]
    dnew = [(k, s) for k, s in sD.items() if k[0] == "b" and k not in sP]
    cont = [k for k in sD if k in sP]
    gone = [(k, s) for k, s in sP.items() if k not in sD]
    streak = {k: 2 for k in cont}
    alive = set(cont)
    cur, n = P, 2
    ncomp = 1
    while alive and n < VS_PREV_MAX_COUNT:
        cur = prev_weekday(cur)
        s = st_at(cur)
        ncomp += 1
        n += 1
        for k in sorted(alive):
            if k in s:
                streak[k] = n
            else:
                alive.discard(k)

    def ent(k: tuple, s: dict[str, Any], extra: int | None = None) -> dict[str, Any]:
        e = dict(id=s["id"], kind=k[0], period=s["period"], condition=s["dlabel"], text=s["text"])
        if extra is not None:
            e["streak"] = extra
        return e

    return dict(
        previous_report_date=P.isoformat(),
        max_count=VS_PREV_MAX_COUNT,
        recomputed_dates=ncomp,
        new=[ent(k, s) for k, s in sorted(new, key=lambda t: skey(*t))],
        direction_diff_new=[ent(k, s) for k, s in sorted(dnew, key=lambda t: skey(*t))],
        continued=[ent(k, sD[k], streak[k]) for k in sorted(cont, key=lambda k: skey(k, sD[k]))],
        resolved=[ent(k, s) for k, s in sorted(gone, key=lambda t: skey(*t))],
    )
