"""일일 브리핑 상위 20 그래프(SVG)를 만듭니다. 표준 라이브러리만 씁니다.

한 장에 종목 20개를 4×5 작은 그림으로 모읍니다. 그림마다 그 종목(굵은 선)과 비교 대상 평균
(옅은 선)을 시작일=100 지수로 그립니다. 가격 수준은 받지도 그리지도 않습니다.

- 같은 입력이면 같은 바이트가 나옵니다(좌표는 소수 1자리 고정, 사전 순회 없음, 시각 없음).
- 스크립트, 이벤트 속성(`on*=`), 외부 참조(`href`), `foreignObject`를 쓰지 않습니다.
- 값이 `None`인 날은 선을 끊습니다. 가격 의심일(`breaks`)은 빨간 점선으로 표시하고 선을 끊습니다.

글자는 호출하는 쪽(`markdown.py`)이 이미 서버 경로·키 모양 문자열을 지운 것을 넘깁니다.
여기서는 XML 이스케이프만 합니다.
"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

NCOL = 4
PANEL_W = 236
PANEL_H = 112
GAP = 12
TOP = 54
STOCK_COLOR = "#1d4ed8"
BENCH_COLOR = "#6b7280"
BREAK_COLOR = "#dc2626"


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _segments(values: list, breaks: set) -> list:
    """끊는 날(`None`)과 의심일(`breaks`)에서 선을 나눕니다. [(t, v), ...]의 목록입니다."""
    segments, current = [], []
    for t, raw in enumerate(values):
        v = _finite(raw)
        if v is None or (t in breaks and current):
            if current:
                segments.append(current)
            current = []
        if v is not None:
            current.append((t, v))
    if current:
        segments.append(current)
    return segments


def _text(x: float | int, y: float | int, body: str, **attrs: str) -> str:
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<text x="{x}" y="{y}"{extra}>{escape(body)}</text>'


def small_multiples(title: str, subtitle: str, panels: list) -> bytes:
    """작은 그림 묶음 SVG를 UTF-8 바이트로 돌려줍니다.

    `panels`의 항목은 dict입니다.
        label  왼쪽 위 글자 (예: "#1 조아제약")
        sub    오른쪽 위 글자 (예: "종목 55.3 / 평균 106.7")
        stock  종목 지수 목록 (시작일=100, 끊는 날은 None)
        bench  비교 대상 평균 지수 목록 (같은 길이)
        breaks 가격 의심일의 위치(인덱스) 목록
        note   선이 하나도 없을 때 가운데 적을 글자 (선택)
    """
    nrow = max(1, math.ceil(len(panels) / NCOL))
    width = NCOL * PANEL_W + (NCOL + 1) * GAP
    height = TOP + nrow * (PANEL_H + GAP) + 4
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" font-family="sans-serif">',
        f"<title>{escape(title)}</title>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        _text(GAP, 22, title, font_size="14", font_weight="bold", fill="#1f2937"),
        _text(GAP, 40, subtitle, font_size="11", fill="#4b5563"),
    ]
    for k, panel in enumerate(panels):
        out += _panel(k, panel)
    out.append("</svg>")
    return ("\n".join(out) + "\n").encode("utf-8")


def _panel(k: int, panel: dict) -> list:
    r, col = divmod(k, NCOL)
    x0 = GAP + col * (PANEL_W + GAP)
    y0 = TOP + r * (PANEL_H + GAP)
    out = [
        f'<g transform="translate({x0},{y0})">',
        f'<rect width="{PANEL_W}" height="{PANEL_H}" rx="4" fill="#f9fafb" stroke="#d1d5db"/>',
        _text(8, 15, panel["label"], font_size="11", font_weight="bold", fill="#111827"),
        _text(PANEL_W - 8, 15, panel["sub"], font_size="10", text_anchor="end", fill="#4b5563"),
    ]
    stock = list(panel.get("stock") or [])
    bench = list(panel.get("bench") or [])
    breaks = {int(t) for t in panel.get("breaks") or []}
    finite = [v for v in (_finite(x) for x in stock + bench) if v is not None]
    length = max(len(stock), len(bench))
    if not finite or length < 2:
        note = panel.get("note") or "가격 경로 없음"
        out.append(
            _text(
                PANEL_W // 2,
                PANEL_H // 2 + 8,
                note,
                font_size="11",
                text_anchor="middle",
                fill="#6b7280",
            )
        )
        out.append("</g>")
        return out
    px0, px1, py0, py1 = 8, PANEL_W - 8, 24, PANEL_H - 8
    lo, hi = min(finite + [100.0]), max(finite + [100.0])
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = (hi - lo) * 0.06
    lo, hi = lo - pad, hi + pad

    def xpos(t: int) -> str:
        return f"{px0 + (px1 - px0) * t / (length - 1):.1f}"

    def ypos(v: float) -> str:
        return f"{py1 - (py1 - py0) * (v - lo) / (hi - lo):.1f}"

    base = ypos(100.0)
    out.append(
        f'<line x1="{px0}" y1="{base}" x2="{px1}" y2="{base}" stroke="#9ca3af" '
        'stroke-width="0.6" stroke-dasharray="3 3"/>'
    )
    for values, brk, color, stroke, opacity in (
        (bench, set(), BENCH_COLOR, "1.0", "0.55"),
        (stock, breaks, STOCK_COLOR, "2.0", "1.0"),
    ):
        for seg in _segments(values, brk):
            if len(seg) == 1:
                t, v = seg[0]
                out.append(
                    f'<circle cx="{xpos(t)}" cy="{ypos(v)}" r="1.6" fill="{color}" '
                    f'fill-opacity="{opacity}"/>'
                )
            else:
                points = " ".join(f"{xpos(t)},{ypos(v)}" for t, v in seg)
                out.append(
                    f'<polyline fill="none" stroke="{color}" stroke-opacity="{opacity}" '
                    f'stroke-width="{stroke}" stroke-linejoin="round" stroke-linecap="round" '
                    f'points="{points}"/>'
                )
    for t in sorted(breaks):
        if 0 <= t < length:
            out.append(
                f'<line x1="{xpos(t)}" y1="{py0}" x2="{xpos(t)}" y2="{py1}" '
                f'stroke="{BREAK_COLOR}" stroke-width="0.8" stroke-dasharray="2 2"/>'
            )
    out.append("</g>")
    return out
