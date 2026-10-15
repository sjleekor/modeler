# ruff: noqa: E501
"""Gallery HTML: one page per feature plus an index. Inline SVG, no script, light and dark."""

from __future__ import annotations

import datetime as dt
import html
import json
import re
from pathlib import Path

import polars as pl

from . import config as C
from .config import FeatureSpec
from .dist import group_order

TEMPLATE = Path(__file__).with_name("template.html")
MINUS = "−"
GROUP_COLOR = {
    "good": "var(--s3)",
    "opp": "var(--s2)",
    "mid": "var(--muted)",
    "top": "var(--s1)",
    "bottom": "var(--s2)",
    "all": "var(--muted)",
}
MKT_LABEL = "시장 동일가중(근사)"


def esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def pct_str(x: float | None, digits: int = 1, sign: bool = True) -> str:
    if x is None or x != x:
        return "없음"
    v = x * 100
    s = f"{abs(v):.{digits}f}"
    if v < 0 and float(s) != 0:
        return MINUS + s + "%"
    return ("+" if sign and float(s) != 0 else "") + s + "%"


def val_str(v: float | None) -> str:
    if v is None or v != v:
        return "없음"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    s = f"{v:.4g}"
    return s.replace("-", MINUS)


def group_label(g: str, spec: FeatureSpec) -> str:
    if spec.direction == 0:
        return {"top": "상위 10%", "bottom": "하위 10%", "mid": "중간 80%", "all": "전체 유니버스"}[
            g
        ]
    return {
        "good": "좋은 쪽 10%",
        "opp": "반대쪽 10%",
        "mid": "중간 80%",
        "all": "전체 유니버스",
    }[g]


# ---------------------------------------------------------------------------------------------
# commentary: generic "읽는 법" boxes, per-feature "이 피쳐에서" boxes, per-case captions
# ---------------------------------------------------------------------------------------------
COMMENTARY_DIR = Path(__file__).with_name("commentary")
COMMENTARY_KEYS = ("dist_raw", "dist_fwd", "path", "random", "extremes")

HOWTO = {
    "dist_raw": (
        "상자는 Q1부터 Q3까지, 굵은 선은 중앙값, 속 빈 점은 평균이고 수염은 5\u201395 백분위입니다. "
        "무리는 그날 시장 안 백분위로 나눈 좋은 쪽 10%·중간 80%·반대쪽 10%(방향이 없는 피쳐는 상위·중간·하위)입니다. "
        "초과수익은 종목 20일 수익에서 그날 시장 동일가중 수익을 뺀 값이라, 초과&gt;0 비율은 시장을 이긴 종목의 몫입니다. "
        "시장을 이기는 종목이 절반을 넘기 어려운 경우가 많아 비율이 50%보다 낮은 것은 이 그림만의 특징이 아닙니다. 무리끼리 비교해서 보십시오."
    ),
    "dist_fwd": (
        "왼쪽 그림과 같은 상자 그림이되 시장을 빼지 않은 20일 수익입니다. "
        "두 그림의 차이가 그날 시장 전체의 움직임입니다. 세 무리가 함께 위아래로 움직이면 피쳐가 아니라 시장 효과입니다."
    ),
    "path": (
        "선은 무리별 누적 초과수익의 중앙값, 띠는 Q1\u2013Q3입니다. 0일은 formation 종가, 20일은 라벨 시점입니다. "
        "선이 갈라지는 시점과 띠가 겹치는 정도를 보십시오. 띠가 많이 겹치면 무리 사이 차이가 종목 간 흩어짐보다 작다는 뜻입니다. "
        "점선은 거래세 0.20% 참고선이며 비용을 뺀 값이 아닙니다."
    ),
    "random": (
        "파란 선은 formation 종가를 100으로 맞춘 종목 가격, 회색 선은 시장 동일가중 근사입니다. "
        "파란 음영은 피쳐 계산 창, 주황 음영은 이후 20거래일(라벨 창), 세로 점선이 formation 날짜입니다. "
        "아래 칩은 맞음·틀림(좋은 쪽만)과 거래정지·상하한가 근접·우선주 같은 주의 표시입니다. "
        "카드 한 장은 한 종목의 한 시점일 뿐이라, 여러 장에서 되풀이되는 모양만 보십시오."
    ),
    "extremes": (
        "무작위가 아니라 결과를 보고 고른 양 끝입니다. 성과 상위·하위 5개를 나란히 놓았으니 둘을 비교해 공통 특징(칩, 시장, 날짜)을 찾는 용도입니다. "
        "눈에 띄는 모양이 있어도 이 5개는 가장 극단이라 일반적인 모습이 아닙니다."
    ),
    "index": (
        "각 피쳐 페이지는 개발 구간 전체 분포, 무작위 사례, 극단 사례 순서입니다. "
        "분포 그림은 무리 사이 차이를, 카드는 그 차이를 이루는 종목 하나하나의 실제 가격을 보여 줍니다. "
        "모든 그림 밑에 \u201c읽는 법\u201d(공통)과 \u201c이 피쳐에서\u201d(피쳐별 해설)가 있습니다. 해설은 관찰이며 판정이 아닙니다."
    ),
}

_ALLOWED_TAG = re.compile(r"&lt;(/?)(b|code)&gt;")


def safe_html(text: str) -> str:
    """Escape everything, then restore only <b> and <code> tags."""
    return _ALLOWED_TAG.sub(r"<\1\2>", html.escape(str(text), quote=False))


def load_commentary(feature: str, directory: Path | None = None) -> dict:
    """Read commentary/<feature>.json. Missing or broken file gives an empty dict."""
    path = (directory or COMMENTARY_DIR) / f"{feature}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out = {k: data[k] for k in COMMENTARY_KEYS if isinstance(data.get(k), str) and data[k].strip()}
    dates = data.get("dates")
    if isinstance(dates, dict):
        out["dates"] = {
            str(k): v for k, v in list(dates.items())[:12] if isinstance(v, str) and v.strip()
        }
    return out


def howto_box(key: str) -> str:
    return f'<div class="howto"><b>읽는 법</b> {HOWTO[key]}</div>'


def feature_box(com: dict, key: str) -> str:
    t = com.get(key)
    return f'<div class="thisfeat"><b>이 피쳐에서</b> {safe_html(t)}</div>' if t else ""


def explain(com: dict, key: str) -> str:
    return howto_box(key) + feature_box(com, key)


def _first_sentence(text: str) -> str:
    return re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]


def _chg(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or a == 0:
        return None
    return b / a - 1


def case_caption(r: dict, s: dict | None, spec: FeatureSpec) -> str:
    """One deterministic line computed from the case data only."""
    parts = []
    if s:
        by = dict(zip(s["rel"], s["close_norm"], strict=True))
        pre = _chg(by.get(-C.PRE_DAYS), by.get(0))
        if pre is not None:
            parts.append(f"formation 전 {C.PRE_DAYS}일 {pct_str(pre, 0)}")
        w = spec.window
        if w and w <= C.PRE_DAYS:
            win = _chg(by.get(-w), by.get(0))
            if win is not None:
                parts.append(f"계산 창(최근 {w}일) {pct_str(win, 0)}")
    if r["label_missing"]:
        parts.append("이후 20일 라벨 없음")
    else:
        tail = (
            f"이후 20일 {pct_str(r['fwd_ret_20d'], 0)}(시장 대비 {pct_str(r['raw_label_20d'], 0)})"
        )
        if r["group"] == "good" and r["hit"] is not None:
            tail += ", 맞음" if r["hit"] else ", 틀림"
        parts.append(tail)
    words = []
    if r["halt_fwd"]:
        words.append(f"이후 거래정지 {r['halt_fwd']}일")
    if r["halt_prev20"]:
        words.append(f"직전 20일 거래정지 {r['halt_prev20']}일")
    if r["limit_hits"]:
        words.append(f"상하한가 근접 {r['limit_hits']}회")
    if r["jump_over_limit"]:
        words.append(f"±31% 초과 변동 {r['jump_over_limit']}회")
    if r["preferred"]:
        words.append("우선주")
    if r["neg_equity"]:
        words.append("자본잠식")
    if r["corp_action"]:
        words.append(f"기업행동 {r['corp_action']}")
    line = ", ".join(parts)
    if words:
        line += ". 표시: " + ", ".join(words)
    return line + "."


# ---------------------------------------------------------------------------------------------
# case card
# ---------------------------------------------------------------------------------------------
W, H = 260, 150
ML, MR, MT, MB = 6, 6, 10, 16


def _xy(rel: float, v: float, lo: float, hi: float) -> tuple[float, float]:
    x = ML + (rel + C.PRE_DAYS) / (C.PRE_DAYS + 20) * (W - ML - MR)
    y = MT + (hi - v) / (hi - lo) * (H - MT - MB)
    return x, y


def _poly(rel: list[int], vals: list[float | None], lo: float, hi: float, cls: str) -> str:
    segs, cur = [], []
    for r, v in zip(rel, vals, strict=True):
        if v is None:
            if len(cur) > 1:
                segs.append(cur)
            cur = []
            continue
        cur.append(_xy(r, v, lo, hi))
    if len(cur) > 1:
        segs.append(cur)
    out = []
    for s in segs:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in s)
        out.append(f'<polyline points="{pts}" fill="none" {cls}/>')
    return "".join(out)


def case_svg(s: dict, spec: FeatureSpec, title: str) -> str:
    rel = s["rel"]
    vals = [v for v in s["close_norm"] if v is not None] + s["mkt_norm"] + [100.0]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.06 or 1.0
    lo, hi = lo - pad, hi + pad
    parts = [
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}"><title>{esc(title)}</title>'
    ]
    x0, _ = _xy(0, 100, lo, hi)
    xf, _ = _xy(20, 100, lo, hi)
    top, bot = MT, H - MB
    if spec.window:
        a = max(-C.PRE_DAYS, -spec.window + 1)
        xa, _ = _xy(a, 100, lo, hi)
        cut = " (그림 왼쪽 밖까지 이어짐)" if spec.window - 1 > C.PRE_DAYS else ""
        parts.append(
            f'<rect class="win" x="{xa:.1f}" y="{top}" width="{x0 - xa:.1f}" height="{bot - top}">'
            f"<title>피쳐 계산 창 {spec.window}거래일{cut}</title></rect>"
        )
    parts.append(
        f'<rect class="fwdz" x="{x0:.1f}" y="{top}" width="{xf - x0:.1f}" height="{bot - top}">'
        "<title>이후 20거래일(라벨 창)</title></rect>"
    )
    _, y100 = _xy(0, 100, lo, hi)
    parts.append(
        f'<line x1="{ML}" x2="{W - MR}" y1="{y100:.1f}" y2="{y100:.1f}" style="stroke:var(--axis)" '
        'stroke-width="0.8" stroke-dasharray="3 3"/>'
    )
    parts.append(
        _poly(rel, s["mkt_norm"], lo, hi, 'style="stroke:var(--muted)" stroke-width="1.2"')
    )
    parts.append(_poly(rel, s["close_norm"], lo, hi, 'style="stroke:var(--s1)" stroke-width="1.7"'))
    parts.append(
        f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="{top}" y2="{bot}" style="stroke:var(--ink)" '
        'stroke-width="1" stroke-dasharray="2 2"><title>formation 날짜(종가 진입 가정)</title></line>'
    )
    for r, h in zip(rel, s["halt"], strict=True):
        if h:
            x, _ = _xy(r, 100, lo, hi)
            parts.append(
                f'<rect x="{x - 0.8:.1f}" y="{bot - 5}" width="1.6" height="5" style="fill:var(--warn)">'
                "<title>거래정지일(시가=고가=저가=0)</title></rect>"
            )
    for r in (-120, -60, 0, 20):
        x, _ = _xy(r, 100, lo, hi)
        lab = "0" if r == 0 else (f"+{r}" if r > 0 else f"{MINUS}{-r}")
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{H - 3}" text-anchor="middle">{lab}</text>'
        )
    parts.append(f'<text class="tick" x="{ML}" y="{MT - 1}">{hi:.0f}</text>')
    parts.append(f'<text class="tick" x="{ML}" y="{bot - 6}">{lo:.0f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def chips(r: dict, spec: FeatureSpec) -> str:
    out = []
    if r["label_missing"]:
        out.append(
            '<span class="chip na" title="20거래일 뒤 라벨이 비어 있음(상장폐지·거래 중단 등)">라벨 없음</span>'
        )
    elif r["group"] == "good":
        if r["hit"]:
            out.append(
                '<span class="chip hit" title="좋은 쪽 무리에서 20거래일 초과수익 > 0">맞음</span>'
            )
        else:
            out.append(
                '<span class="chip miss" title="좋은 쪽 무리에서 20거래일 초과수익 ≤ 0">틀림</span>'
            )
    else:
        pos = r["raw_label_20d"] > 0
        out.append(
            f'<span class="chip na" title="맞음·틀림은 좋은 쪽에만 붙임. 여기서는 부호만 표시">'
            f"초과수익 {'> 0' if pos else '≤ 0'}</span>"
        )
    if r["halt_fwd"]:
        out.append(
            f'<span class="chip flag" title="이후 20거래일 중 거래정지(시가=고가=저가=0)">정지 {r["halt_fwd"]}일(이후)</span>'
        )
    if r["halt_prev20"]:
        out.append(
            f'<span class="chip flag" title="formation 날짜까지 최근 20거래일 중 정지일 수. 고저 폭 피쳐에서는 low=0이 값을 부풀림">정지 {r["halt_prev20"]}일(직전 20)</span>'
        )
    if r["limit_hits"]:
        out.append(
            f'<span class="chip flag" title="이후 20거래일 중 일간 변동이 ±29.5~31%인 날(가격제한폭 ±30% 근접)">상·하한가 근접 {r["limit_hits"]}회</span>'
        )
    if r["jump_over_limit"]:
        out.append(
            f'<span class="chip flag" title="이후 20거래일 중 일간 변동이 31%를 넘은 날. 제한폭 밖이라 기업행동·신규 상장 등일 수 있음(원인은 확인 안 함)">±31% 초과 {r["jump_over_limit"]}회</span>'
        )
    if r["preferred"]:
        out.append(
            '<span class="chip flag" title="종목명이 우·우B·우C로 끝나거나 종목코드 끝자리가 0이 아님">우선주</span>'
        )
    if r["neg_equity"]:
        out.append(
            '<span class="chip flag" title="fin_is_negative_equity = 1. 자본잠식이면 ROE 값이 거꾸로(적자인데 순위 상위)">자본잠식</span>'
        )
    if r["path_gap"] is not None and r["path_gap"] > 0.01:
        out.append(
            f'<span class="chip flag" title="그림의 20일째 수익(일봉 종가 기준)이 패널 라벨 수익과 {r["path_gap"] * 100:.1f}%p 다름. 원인은 확인 안 함">그림-라벨 차이 {r["path_gap"] * 100:.0f}%p</span>'
        )
    if r["corp_action"]:
        out.append(
            f'<span class="chip flag" title="DART 자본변동 기록(그림 구간 −120~+20일): {esc(r["corp_action"])}">기업행동 {esc(r["corp_action"])}</span>'
        )
    return "".join(out)


def case_card(r: dict, series: dict, spec: FeatureSpec, extreme: bool = False) -> str:
    s = series.get((r["trade_date"], r["ticker"]))
    excess = r["raw_label_20d"]
    title = (
        f"{r['name']} {r['ticker']} {r['market']} · formation {r['trade_date']} · "
        f"{spec.name} {val_str(r['value'])}(백분위 {r['pct'] * 100:.1f}) · "
        f"20일 수익 {pct_str(r['fwd_ret_20d'])} · 초과 {pct_str(excess)}"
    )
    cls = f"x-{'top' if r['kind'] == 'extreme_top' else 'bottom'}" if extreme else f"g-{r['group']}"
    svg = case_svg(s, spec, title) if s else '<p class="cap">가격 자료 없음</p>'
    tag = ""
    if extreme:
        tag = f'<span class="mk">{r["trade_date"]}</span>'
    return (
        f'<article class="case {cls}" title="{esc(title)}">'
        f'<div class="ch"><b>{esc(r["name"])}</b><span class="mono">{esc(r["ticker"])}</span>'
        f'<span class="mk">{esc(r["market"])}</span>{tag}</div>{svg}'
        '<dl class="kv">'
        f"<dt>피쳐 값</dt><dd>{val_str(r['value'])}</dd>"
        f'<dt title="그날 같은 시장 안 순위(평균 순위/종목 수). 100에 가까울수록 값이 큼">시장 내 백분위</dt><dd>{r["pct"] * 100:.1f}</dd>'
        f"<dt>20일 수익</dt><dd>{pct_str(r['fwd_ret_20d'])}</dd>"
        f'<dt title="그날 시장 전 종목 동일가중 20일 수익을 뺀 값(라벨 raw_label_20d)">시장 대비 초과</dt><dd>{pct_str(excess)}</dd>'
        f'</dl><p class="caption">{esc(case_caption(r, s, spec))}</p><div class="chips">{chips(r, spec)}</div></article>'
    )


# ---------------------------------------------------------------------------------------------
# distribution charts
# ---------------------------------------------------------------------------------------------
def _stat(dist: pl.DataFrame, g: str, metric: str, stat: str, k: int | None = None) -> float | None:
    f = dist.filter(
        (pl.col("group") == g) & (pl.col("metric") == metric) & (pl.col("stat") == stat)
    )
    if k is not None:
        f = f.filter(pl.col("k") == k)
    return f["value"][0] if f.height else None


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = hi - lo
    raw = span / n
    import math

    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-12:
        out.append(round(t, 10))
        t += step
    return out


def box_chart(dist: pl.DataFrame, spec: FeatureSpec, metric: str, label: str) -> str:
    groups = group_order(spec.direction)
    Wd, Hd, lm, r, t, b = 420, 270, 52, 10, 14, 58
    stats = {
        g: {
            s: _stat(dist, g, metric, s)
            for s in ("q05", "q25", "q50", "q75", "q95", "mean", "hit_rate", "n")
        }
        for g in groups
    }
    lows = [v["q05"] for v in stats.values() if v["q05"] is not None] + [0.0, -C.TAX_REF]
    highs = [v["q95"] for v in stats.values() if v["q95"] is not None] + [0.0, C.TAX_REF]
    lo, hi = min(lows), max(highs)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    def Y(v: float) -> float:
        return t + (hi - v) / (hi - lo) * (Hd - t - b)

    p = [
        f'<svg viewBox="0 0 {Wd} {Hd}" role="img" aria-label="{esc(label)} 상자 그림"><title>{esc(label)} 상자 그림: 수염은 5~95 백분위</title>'
    ]
    for tv in _ticks(lo, hi):
        p.append(
            f'<line x1="{lm}" x2="{Wd - r}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" style="stroke:var(--grid)" stroke-width="0.8"/>'
        )
        p.append(
            f'<text x="{lm - 5}" y="{Y(tv) + 3:.1f}" text-anchor="end" font-size="10">{pct_str(tv, 0 if abs(hi - lo) > 0.2 else 1, sign=False)}</text>'
        )
    p.append(
        f'<line x1="{lm}" x2="{Wd - r}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" style="stroke:var(--ink2)" stroke-width="1.2"><title>0</title></line>'
    )
    p.append(
        f'<line x1="{lm}" x2="{Wd - r}" y1="{Y(C.TAX_REF):.1f}" y2="{Y(C.TAX_REF):.1f}" style="stroke:var(--warn)" stroke-width="1" stroke-dasharray="4 3"><title>참고선: 거래세 0.20%. 이 그림은 비용 전 값</title></line>'
    )
    slot = (Wd - lm - r) / len(groups)
    for i, g in enumerate(groups):
        s = stats[g]
        cx = lm + slot * (i + 0.5)
        col = GROUP_COLOR[g]
        if s["q50"] is None:
            continue
        bw = min(70, slot * 0.5)
        tip = (
            f"{group_label(g, spec)} · n={int(s['n'])} · 중앙값 {pct_str(s['q50'], 2)} · 평균 {pct_str(s['mean'], 2)} · "
            f"Q1 {pct_str(s['q25'], 2)} · Q3 {pct_str(s['q75'], 2)} · 5% {pct_str(s['q05'], 2)} · 95% {pct_str(s['q95'], 2)} · "
            f"초과 > 0 비율 {s['hit_rate'] * 100:.1f}%"
        )
        p.append(f"<g><title>{esc(tip)}</title>")
        p.append(
            f'<line x1="{cx:.1f}" x2="{cx:.1f}" y1="{Y(s["q95"]):.1f}" y2="{Y(s["q05"]):.1f}" style="stroke:{col}" stroke-width="1.6"/>'
        )
        for q in ("q05", "q95"):
            p.append(
                f'<line x1="{cx - bw / 4:.1f}" x2="{cx + bw / 4:.1f}" y1="{Y(s[q]):.1f}" y2="{Y(s[q]):.1f}" style="stroke:{col}" stroke-width="1.6"/>'
            )
        p.append(
            f'<rect x="{cx - bw / 2:.1f}" y="{Y(s["q75"]):.1f}" width="{bw:.1f}" height="{max(1.0, Y(s["q25"]) - Y(s["q75"])):.1f}" style="fill:{col};fill-opacity:.22;stroke:{col}" stroke-width="1.6"/>'
        )
        p.append(
            f'<line x1="{cx - bw / 2:.1f}" x2="{cx + bw / 2:.1f}" y1="{Y(s["q50"]):.1f}" y2="{Y(s["q50"]):.1f}" style="stroke:{col}" stroke-width="3"/>'
        )
        p.append(
            f'<circle cx="{cx:.1f}" cy="{Y(s["mean"]):.1f}" r="3.2" style="fill:var(--surface);stroke:{col}" stroke-width="1.6"/>'
        )
        p.append("</g>")
        p.append(
            f'<text x="{cx:.1f}" y="{Hd - 36}" text-anchor="middle" font-size="11.5" class="strong" style="fill:var(--ink)">{esc(group_label(g, spec))}</text>'
        )
        p.append(
            f'<text x="{cx:.1f}" y="{Hd - 21}" text-anchor="middle" font-size="11">초과&gt;0 {s["hit_rate"] * 100:.1f}%</text>'
        )
        p.append(
            f'<text x="{cx:.1f}" y="{Hd - 7}" text-anchor="middle" font-size="10.5">n={int(s["n"]):,}</text>'
        )
    p.append("</svg>")
    return "".join(p)


def path_chart(dist: pl.DataFrame, spec: FeatureSpec) -> str:
    groups = group_order(spec.direction)
    Wd, Hd, lm, r, t, b = 460, 270, 52, 10, 14, 34
    data = {}
    for g in groups:
        f = dist.filter((pl.col("group") == g) & (pl.col("metric") == "cum_excess_path"))
        d = {}
        for stat in ("q25", "q50", "q75"):
            ff = f.filter(pl.col("stat") == stat).sort("k")
            d[stat] = dict(zip(ff["k"].to_list(), ff["value"].to_list(), strict=True))
        data[g] = d
    allv = [v for d in data.values() for s in ("q25", "q75") for v in d[s].values()] + [
        0.0,
        C.TAX_REF,
        -C.TAX_REF,
    ]
    lo, hi = min(allv), max(allv)
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    def Y(v: float) -> float:
        return t + (hi - v) / (hi - lo) * (Hd - t - b)

    def X(k: float) -> float:
        return lm + k / 20 * (Wd - lm - r)

    p = [
        f'<svg viewBox="0 0 {Wd} {Hd}" role="img" aria-label="누적 초과수익 경로"><title>누적 초과수익 경로: 선은 중앙값, 띠는 사분위</title>'
    ]
    for tv in _ticks(lo, hi):
        p.append(
            f'<line x1="{lm}" x2="{Wd - r}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" style="stroke:var(--grid)" stroke-width="0.8"/>'
        )
        p.append(
            f'<text x="{lm - 5}" y="{Y(tv) + 3:.1f}" text-anchor="end" font-size="10">{pct_str(tv, 1, sign=False)}</text>'
        )
    for k in (0, 5, 10, 15, 20):
        p.append(
            f'<text x="{X(k):.1f}" y="{Hd - 16}" text-anchor="middle" font-size="10.5">{k}</text>'
        )
    p.append(
        f'<text x="{(lm + Wd - r) / 2:.1f}" y="{Hd - 3}" text-anchor="middle" font-size="10.5">formation 이후 거래일</text>'
    )
    p.append(
        f'<line x1="{lm}" x2="{Wd - r}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" style="stroke:var(--ink2)" stroke-width="1.2"/>'
    )
    p.append(
        f'<line x1="{lm}" x2="{Wd - r}" y1="{Y(C.TAX_REF):.1f}" y2="{Y(C.TAX_REF):.1f}" style="stroke:var(--warn)" stroke-width="1" stroke-dasharray="4 3"><title>참고선: 거래세 0.20%. 이 그림은 비용 전 값</title></line>'
    )
    for g in groups:
        d = data[g]
        ks = sorted(d["q50"])
        if not ks:
            continue
        col = GROUP_COLOR[g]
        up = " ".join(f"{X(k):.1f},{Y(d['q75'][k]):.1f}" for k in ks)
        dn = " ".join(f"{X(k):.1f},{Y(d['q25'][k]):.1f}" for k in reversed(ks))
        p.append(f'<polygon points="{up} {dn}" style="fill:{col};fill-opacity:.15"/>')
        pts = " ".join(f"{X(k):.1f},{Y(d['q50'][k]):.1f}" for k in ks)
        end = d["q50"][ks[-1]]
        p.append(
            f'<polyline points="{pts}" fill="none" style="stroke:{col}" stroke-width="2.2"><title>{esc(group_label(g, spec))} 중앙값 20일째 {pct_str(end, 2)}</title></polyline>'
        )
    p.append("</svg>")
    return "".join(p)


def stats_table(dist: pl.DataFrame, spec: FeatureSpec) -> str:
    cols = [
        ("n", "n"),
        ("n_missing", "라벨 없음"),
        ("mean", "평균"),
        ("q05", "5%"),
        ("q25", "Q1"),
        ("q50", "중앙값"),
        ("q75", "Q3"),
        ("q95", "95%"),
        ("hit_rate", "초과>0 비율"),
    ]
    rows = []
    for metric, mname in (
        ("raw_label_20d", "시장 대비 초과(raw_label_20d)"),
        ("fwd_ret_20d", "20일 수익(fwd_ret_20d)"),
    ):
        for g in group_order(spec.direction):
            cells = []
            for st, _ in cols:
                v = _stat(dist, g, metric, st)
                if v is None:
                    cells.append("<td>-</td>")
                elif st in ("n", "n_missing"):
                    cells.append(f"<td>{int(v):,}</td>")
                elif st == "hit_rate":
                    cells.append(f"<td>{v * 100:.1f}%</td>")
                else:
                    cells.append(f"<td>{pct_str(v, 2)}</td>")
            rows.append(
                f'<tr><td class="l">{esc(mname)}</td><td class="l">{esc(group_label(g, spec))}</td>{"".join(cells)}</tr>'
            )
    head = "".join(f"<th>{h}</th>" for _, h in cols)
    return f'<div class="tw"><table><thead><tr><th class="l">지표</th><th class="l">무리</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def path_table(dist: pl.DataFrame, spec: FeatureSpec) -> str:
    groups = group_order(spec.direction)
    head = "".join(f'<th colspan="3">{esc(group_label(g, spec))}</th>' for g in groups)
    sub = "".join("<th>Q1</th><th>중앙값</th><th>Q3</th>" for _ in groups)
    rows = []
    for k in range(21):
        cells = []
        for g in groups:
            for st in ("q25", "q50", "q75"):
                v = _stat(dist, g, "cum_excess_path", st, k)
                cells.append(f"<td>{pct_str(v, 2)}</td>")
        rows.append(f"<tr><td>{k}</td>{''.join(cells)}</tr>")
    return f'<div class="tw"><table><thead><tr><th rowspan="2">일</th>{head}</tr><tr>{sub}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def legend(spec: FeatureSpec) -> str:
    return (
        '<div class="legend">'
        + "".join(
            f'<span><b class="sw" style="background:{GROUP_COLOR[g]}"></b>{esc(group_label(g, spec))}</span>'
            for g in group_order(spec.direction)
        )
        + '<span><i class="dash" style="border-color:var(--warn)"></i>거래세 0.20% 참고선(비용 전 값)</span></div>'
    )


# ---------------------------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------------------------
COMMON_CAVEATS = (
    "신호일(formation 날짜) 종가에 산다는 가정입니다. 체결 지연·호가·슬리피지는 반영하지 않았습니다.",
    "비용 전 값입니다. 거래세 0.20% 등은 빼지 않았고, 그림에 참고선만 그렸습니다.",
    "개발 구간(2015-01–2025-07)을 다시 본 것입니다. 이 구간은 이미 여러 번 열람했습니다. holdout·2026-09-17 이후는 읽지 않았습니다.",
    "사례는 결과를 보기 전에 고정한 규칙과 시드로 뽑았습니다. 극단 사례는 별도 절에 따로 두었고 대표가 아닙니다.",
)


def render_feature_page(
    spec: FeatureSpec,
    cases: pl.DataFrame,
    dist: pl.DataFrame,
    series: dict,
    dates: list[dt.date],
    n_grid: int,
    run_id: str,
    seed: int,
    commentary_dir: Path | None = None,
) -> str:
    com = load_commentary(spec.name, commentary_dir)
    rnd = cases.filter(pl.col("kind") == "random")
    ext = cases.filter(pl.col("kind") != "random")
    first, second = ("top", "bottom") if spec.direction == 0 else ("good", "opp")
    gname = {g: group_label(g, spec) for g in (first, second, "all")}

    b: list[str] = []
    b.append('<div class="top"><div class="kicker">KR 피쳐 사례 갤러리 · 시범 · 2026-10-15</div>')
    b.append(f"<h1><code>{esc(spec.name)}</code></h1>")
    b.append(
        f'<p class="lede">좋은 쪽: {esc(spec.good_label)}. 방향 근거: {esc(spec.basis)}. '
        f"날짜 {len(dates)}개 × 사례 8개 = {len(rnd)}개와 극단 {len(ext)}개를 실제 가격 그림으로 보여 줍니다.</p>"
    )
    b.append(
        '<div class="notice"><b>가설 탐색이지 판정이 아닙니다.</b> 합격·불합격, 피쳐 채택·폐기, 모델 수정은 이 그림으로 정하지 않습니다.<ul>'
        + "".join(f"<li>{esc(c)}</li>" for c in COMMON_CAVEATS)
        + "</ul></div>"
    )
    b.append(
        '<nav class="toc" aria-label="목차"><a href="index.html">갤러리 목록</a><a href="#card">카드</a><a href="#dist">① 분포</a><a href="#cases">② 무작위 사례</a><a href="#extreme">③ 극단 사례</a><a href="#memo">④ 관찰 메모</a></nav></div>'
    )

    # card
    b.append(
        '<section class="block" id="card"><h2>카드와 층 3</h2><dl class="cardinfo">'
        f'<dt>카드</dt><dd><a href="../{esc(spec.card)}.html">{esc(spec.card)}</a> (같은 폴더의 <code>{esc(spec.card)}.md</code>)</dd>'
        f"<dt>한 줄 직관</dt><dd>{esc(spec.short)}</dd>"
        f"<dt>기대 부호</dt><dd>{esc(spec.expected)}</dd>"
        f'<dt>층 3 모델 반응</dt><dd>{esc(spec.layer3)} <span class="cap">(<code>09_kr_result.md</code> §3)</span></dd>'
        f"<dt>단변량 검정</dt><dd>{esc(spec.test)}</dd></dl></section>"
    )

    # distribution
    b.append('<section class="block" id="dist"><h2>① 분포 먼저 — 개발 구간 전체 격자</h2>')
    b.append(
        f"<p>2016-01-04부터 20거래일 간격 격자 {n_grid}개 날짜(20일 창이 서로 겹치지 않음)의 모든 종목을 "
        "그날 시장(KOSPI·KOSDAQ)별 백분위로 나눈 세 무리입니다. 같은 날짜의 종목끼리는 시장 분위기를 공유하므로 서로 독립이 아닙니다. 비용 전 값입니다.</p>"
    )
    b.append(legend(spec))
    b.append(
        '<div class="two"><div class="chart"><h4>20일 시장 대비 초과수익 (raw_label_20d)</h4>'
        + box_chart(dist, spec, "raw_label_20d", "시장 대비 초과수익")
        + '<p class="cap">상자 Q1–Q3, 굵은 선 중앙값, 속 빈 점 평균, 수염 5–95 백분위.</p>'
        + explain(com, "dist_raw")
        + "</div>"
        '<div class="chart"><h4>20일 수익 (fwd_ret_20d)</h4>'
        + box_chart(dist, spec, "fwd_ret_20d", "20일 수익")
        + '<p class="cap">같은 그림을 시장을 빼기 전 수익으로 그렸습니다.</p>'
        + explain(com, "dist_fwd")
        + "</div></div>"
    )
    b.append(
        '<div class="chart" style="max-width:640px"><h4>formation 이후 누적 초과수익 경로 (0–20거래일)</h4>'
        + path_chart(dist, spec)
        + f'<p class="cap">선은 중앙값, 띠는 Q1–Q3. 종목 누적수익에서 {MKT_LABEL} 누적수익을 뺐습니다. 중간에 가격이 끊긴 종목은 그 날부터 빠집니다.</p>'
        + explain(com, "path")
        + "</div>"
    )
    b.append(
        '<details class="tbl"><summary>숫자 표 보기 — 상자 통계·적중률</summary>'
        + stats_table(dist, spec)
        + "</details>"
    )
    b.append(
        '<details class="tbl"><summary>숫자 표 보기 — 누적 경로</summary>'
        + path_table(dist, spec)
        + "</details></section>"
    )

    # random cases
    b.append('<section class="block" id="cases"><h2>② 무작위 사례</h2>')
    b.append(
        "<p>날짜마다 좋은 쪽(또는 상위) 3개 · 반대쪽(또는 하위) 3개 · 전체 2개를 시드 "
        f"{seed}로 뽑았습니다. 성과를 보고 고르지 않았습니다.</p>"
        '<div class="legend"><span><i style="border-color:var(--s1)"></i>종목 가격(formation 종가=100)</span>'
        f'<span><i style="border-color:var(--muted);border-top-width:2px"></i>{MKT_LABEL}</span>'
        '<span><b class="sw" style="background:var(--s1);opacity:.25"></b>피쳐 계산 창</span>'
        '<span><b class="sw" style="background:var(--s2);opacity:.3"></b>이후 20거래일</span>'
        '<span><b class="sw" style="background:var(--warn)"></b>거래정지일</span></div>'
        '<p class="cap">가격은 일봉 종가 그대로이며 수정주가 여부는 확인하지 않았습니다. 기업행동은 DART 자본변동(주식분할·무상증자·무상감자) 기록이 있는 것만 칩으로 표시하고, 기록이 빠진 것은 표시되지 않습니다. 그림 가로축은 formation 기준 거래일(−120…+20)입니다.</p>'
        + explain(com, "random")
    )
    for d in dates:
        day = rnd.filter(pl.col("trade_date") == d)
        end = d.isoformat()
        b.append(
            f'<div class="dategrp"><h3>{end}</h3><p class="sub">formation {end} · 사례 {day.height}개</p>'
        )
        dtext = com.get("dates", {}).get(end)
        if dtext:
            b.append(f'<div class="thisfeat"><b>이 날짜</b> {safe_html(dtext)}</div>')
        for g in (first, second, "all"):
            sub = day.filter(pl.col("group") == g)
            if sub.height == 0:
                continue
            b.append(
                f'<div class="gh"><i style="background:{GROUP_COLOR[g]}"></i>{esc(gname[g])} · {sub.height}개</div><div class="cases">'
            )
            b.extend(case_card(r, series, spec) for r in sub.iter_rows(named=True))
            b.append("</div>")
        b.append("</div>")
    b.append(
        '<details class="tbl"><summary>숫자 표 보기 — 무작위 사례 전체</summary>'
        + cases_table(rnd)
        + "</details></section>"
    )

    # extremes
    pool_text = "좋은 쪽 10% 무리" if spec.direction else "상위 10%와 하위 10% 무리를 합친 것"
    b.append(
        '<section class="block" id="extreme"><h2>③ 극단 사례</h2>'
        '<div class="banner"><b>대표 사례가 아닙니다.</b> 아래는 무작위가 아니라 결과로 고른 양 끝입니다. '
        f"고른 날짜 {len(dates)}개의 {pool_text} 전체 중 시장 대비 초과수익이 가장 큰 5개와 가장 작은 5개입니다.</div>"
        + explain(com, "extremes")
    )
    for kind, ttl in (
        ("extreme_top", "초과수익 상위 5개"),
        ("extreme_bottom", "초과수익 하위 5개"),
    ):
        sub = ext.filter(pl.col("kind") == kind).sort("ext_rank")
        b.append(
            f'<div class="gh"><i style="background:var(--warn)"></i>{ttl} · 대표 아님</div><div class="cases">'
        )
        b.extend(case_card(r, series, spec, extreme=True) for r in sub.iter_rows(named=True))
        b.append("</div>")
    b.append(
        '<details class="tbl"><summary>숫자 표 보기 — 극단 사례</summary>'
        + cases_table(ext)
        + "</details></section>"
    )

    b.append(
        '<section class="block" id="memo"><h2>④ 관찰 메모</h2>'
        '<p class="cap">그림을 본 사람이 직접 적는 칸입니다. 이 페이지는 저장하지 않으므로 적은 내용은 <code>12_case_observations.md</code>에 옮겨 적습니다. 비어 있어야 정상입니다.</p>'
        '<textarea class="memo" aria-label="관찰 메모" placeholder=""></textarea></section>'
    )
    b.append(
        '<footer class="foot"><ul>'
        f"<li>실행 {esc(run_id)} · 시드 {seed} · KR 개발 구간 사례 열람(가설 탐색), 라벨은 개발 구간만 사용.</li>"
        "<li>순위는 서빙과 같이 그날 시장(KOSPI·KOSDAQ)별 백분위입니다. 가격 그림의 시장선은 패널 px_ret_1d의 시장별 동일가중 평균을 이어 붙인 근사입니다.</li>"
        "<li>상·하한가 근접은 거래정지가 아닌 날 일간 변동 29.5~31%, ±31% 초과는 제한폭 밖(기업행동·신규 상장 등 원인 미확인)입니다.</li>"
        "</ul></footer>"
    )
    return render_shell(
        f"{spec.name} 사례 갤러리", "".join(b), f"{spec.name}의 개발 구간 사례 갤러리(가설 탐색)."
    )


def cases_table(df: pl.DataFrame) -> str:
    head = [
        "날짜",
        "무리",
        "종목",
        "시장",
        "값",
        "백분위",
        "20일 수익",
        "초과",
        "맞음",
        "정지(이후/직전)",
        "상하한가",
        "±31%",
        "우선주",
        "잠식",
        "기업행동",
    ]
    rows = []
    for r in df.sort(["trade_date", "group", "ticker"]).iter_rows(named=True):
        hit = "" if r["hit"] is None else ("맞음" if r["hit"] else "틀림")
        rows.append(
            "<tr>"
            + "".join(
                f"<td{' class="l"' if i in (1, 2, 3, 14) else ''}>{esc(c)}</td>"
                for i, c in enumerate(
                    [
                        r["trade_date"],
                        r["group"] if r["kind"] == "random" else r["kind"],
                        f"{r['name']} {r['ticker']}",
                        r["market"],
                        val_str(r["value"]),
                        f"{r['pct'] * 100:.1f}",
                        pct_str(r["fwd_ret_20d"]),
                        pct_str(r["raw_label_20d"]),
                        hit,
                        f"{r['halt_fwd']}/{r['halt_prev20']}",
                        r["limit_hits"],
                        r["jump_over_limit"],
                        "예" if r["preferred"] else "",
                        "예" if r["neg_equity"] else "",
                        r["corp_action"],
                    ]
                )
            )
            + "</tr>"
        )
    return (
        '<div class="tw"><table><thead><tr>'
        + "".join(f"<th>{h}</th>" for h in head)
        + f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_shell(title: str, body: str, desc: str) -> str:
    t = TEMPLATE.read_text(encoding="utf-8")
    return (
        t.replace("__TITLE__", esc(title)).replace("__DESC__", esc(desc)).replace("__BODY__", body)
    )


def render_index(
    specs: list[FeatureSpec],
    counts: dict,
    dates: list[dt.date],
    n_grid: int,
    run_id: str,
    seed: int,
    commentary_dir: Path | None = None,
) -> str:
    coms = {sp.name: load_commentary(sp.name, commentary_dir) for sp in specs}
    b = [
        '<div class="top"><div class="kicker">KR 피쳐 사례 갤러리 · 시범 · 2026-10-15</div><h1>피쳐 사례 갤러리</h1>'
        f'<p class="lede">피쳐 값이 극단인 종목이 그 뒤 20거래일 동안 실제로 어떻게 움직였는지 보는 그림판입니다. 개발 구간(formation 2016-01-04–2025-06-30)만 씁니다. 날짜 {len(dates)}개는 모든 피쳐가 같습니다.</p>'
        '<div class="notice"><b>가설 탐색이지 판정이 아닙니다.</b><ul>'
        + "".join(f"<li>{esc(c)}</li>" for c in COMMON_CAVEATS)
        + "</ul></div></div>"
    ]
    b.append('<section class="block"><h2>피쳐 9개</h2><div class="idx">')
    for s in specs:
        c = counts[s.name]
        b.append(
            '<div class="idxrow">'
            f'<h3><a href="{esc(s.name)}.html"><code>{esc(s.name)}</code></a></h3>'
            f"<p>{esc(s.short)}</p>"
            f"<p>좋은 쪽: {esc(s.good_label)} · 사례 {c['random']}개 + 극단 {c['extreme']}개 · 격자 {n_grid}개 날짜 · "
            f'카드 <a href="../{esc(s.card)}.html">{esc(s.card)}</a></p></div>'
        )
    b.append("</div>" + howto_box("index") + "</section>")
    if specs and all(coms[sp.name].get("dist_raw") for sp in specs):
        items = "".join(
            f'<li><a href="{esc(sp.name)}.html"><code>{esc(sp.name)}</code></a> \u2014 '
            f"{safe_html(_first_sentence(coms[sp.name]['dist_raw']))}</li>"
            for sp in specs
        )
        b.append(
            f'<section class="block"><h2>피쳐별 한 줄 요약</h2><ul class="sumlist">{items}</ul>'
            '<p class="cap">각 피쳐 페이지의 \u201c이 피쳐에서\u201d 해설 첫 문장입니다.</p></section>'
        )
    b.append(
        f'<section class="block"><h2>고른 날짜</h2><p class="mono">{" · ".join(d.isoformat() for d in dates)}</p>'
        f'<p class="cap">시드 {seed} · 실행 {esc(run_id)}. 해마다 1개(2016–2025) + 나머지 2개는 전체에서 무작위.</p></section>'
    )
    return render_shell("피쳐 사례 갤러리", "".join(b), "KR 피쳐 사례 갤러리 시범 9개의 목록.")
