"""그림 데이터. 렌더러가 body만으로 그림을 그릴 수 있게 점·구간·기준선·음영 시작을 담는다.

그림 21개(일간 13, 월간 8. 시제품 ``make_figs``와 같은 구성). 일간은 가장 늦은 마지막 관측일 기준
365일 창(약 250점), 월간은 60개월 창이다. 값은 소수 4자리로 반올림한다. 신용 스프레드는 1년 창
시작 대비 변화(R1), 거래대금은 비율(R2)을 싣는다. 월간 결측 달은 점을 빼고 ``segments``로 선을 끊는다.
"""

from __future__ import annotations

import calendar
import datetime as dt
from typing import Any

import numpy as np

from modeler.scores.macro_context.calc import Computed, first_of_month

DAILY_WINDOW_DAYS = 365
MONTHLY_WINDOW = 60
ROUND = 4

# 값 문구 템플릿(파이썬 str.format, 변수 v). 음수 기호는 렌더러가 −로 바꾼다.
F_PCT2 = "{v:.2f}%"
F_PP2 = "{v:.2f}%p"
F_PP2S = "{v:+.2f}%p"
F_PCT2S = "{v:+.2f}%"
F_FLOW = "{v:+.2f}조원"
F_RATIO = "{v:.2f}배"


def _daily_line(label: str, dates, vals, fmt: str) -> dict[str, Any]:
    return dict(label=label, fmt=fmt, dates=list(dates), vals=np.asarray(vals, dtype=float))


def _roll_sum(dates, x, h: int):
    c = np.concatenate([[0.0], np.cumsum(x)])
    s = (c[h:] - c[:-h]) / 1e4  # 조원
    return dates[h - 1 :], s


def _ratio_series(dates, x):
    ma20 = np.convolve(x, np.ones(20) / 20, mode="valid")
    ma250 = np.convolve(x, np.ones(250) / 250, mode="valid")
    r = np.round(ma20[249 - 19 :] / ma250, 8)
    return dates[249:], r


def _daily_figure(name, group, title, unit, lines, ref=None) -> dict[str, Any]:
    xmax = max(d.toordinal() for ln in lines for d in ln["dates"])
    x0 = xmax - DAILY_WINDOW_DAYS
    out_lines = []
    for ln in lines:
        pts = [
            [d.isoformat(), round(float(v), ROUND)]
            for d, v in zip(ln["dates"], ln["vals"], strict=True)
            if d.toordinal() >= x0 and not np.isnan(v)
        ]
        out_lines.append(
            dict(
                label=ln["label"],
                fmt=ln["fmt"],
                points=pts,
                segments=[[0, len(pts)]],
                last=dict(date=pts[-1][0], value=pts[-1][1]),
            )
        )
    s3 = max(minus_years_months(dt.date.fromordinal(xmax), 3).toordinal(), x0)
    return dict(
        name=name,
        group=group,
        title=title,
        kind="daily",
        unit=unit,
        ref=ref,
        shade_from=dt.date.fromordinal(s3).isoformat(),
        x_min=dt.date.fromordinal(x0).isoformat(),
        x_max=dt.date.fromordinal(xmax).isoformat(),
        lines=out_lines,
    )


def minus_years_months(d: dt.date, n: int) -> dt.date:
    """n개월 전 같은 날(말일을 넘으면 그 달 말일)."""
    y, m = divmod(d.year * 12 + d.month - 1 - n, 12)
    m += 1
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def _monthly_figure(name, group, title, unit, lines, ref=None) -> dict[str, Any]:
    xmax = max(
        int(k) for ln in lines for k, v in zip(ln["ks"], ln["vals"], strict=True) if not np.isnan(v)
    )
    x0 = xmax - (MONTHLY_WINDOW - 1)
    out_lines = []
    for ln in lines:
        ks = [
            int(k)
            for k, v in zip(ln["ks"], ln["vals"], strict=True)
            if int(k) >= x0 and not np.isnan(v)
        ]
        vs = [
            float(v)
            for k, v in zip(ln["ks"], ln["vals"], strict=True)
            if int(k) >= x0 and not np.isnan(v)
        ]
        pts = [
            [first_of_month(k).isoformat(), round(v, ROUND)] for k, v in zip(ks, vs, strict=True)
        ]
        segs, start = [], 0
        for i in range(1, len(ks) + 1):
            if i == len(ks) or ks[i] - ks[i - 1] > 1:
                segs.append([start, i])
                start = i
        out_lines.append(
            dict(
                label=ln["label"],
                fmt=ln["fmt"],
                points=pts,
                segments=segs,
                last=dict(date=pts[-1][0], value=pts[-1][1]),
            )
        )
    return dict(
        name=name,
        group=group,
        title=title,
        kind="monthly",
        unit=unit,
        ref=ref,
        shade_from=None,
        x_min=first_of_month(x0).isoformat(),
        x_max=first_of_month(xmax).isoformat(),
        lines=out_lines,
    )


def make_figures(res: Computed) -> list[dict[str, Any]]:
    s = res.series

    def dl(label, iid, fmt):
        dates, x = s[iid]
        return _daily_line(label, dates, x, fmt)

    def ml(label, iid, fmt):
        ks, vm = s[iid]
        return dict(label=label, fmt=fmt, ks=ks, vals=vm)

    F: list[dict[str, Any]] = []

    def D_(*a, **k):
        F.append(_daily_figure(*a, **k))

    def M_(*a, **k):
        F.append(_monthly_figure(*a, **k))

    D_(
        "rates_us_nominal",
        "rates",
        "미 명목금리 (%) · 최근 1년",
        "%",
        [
            dl("미 3개월물", "us_3m", F_PCT2),
            dl("미 2년물", "us_2y", F_PCT2),
            dl("미 10년물", "us_10y", F_PCT2),
        ],
    )
    D_(
        "rates_kr",
        "rates",
        "한 금리 (%) · 최근 1년",
        "%",
        [
            dl("한 CD91", "kr_cd91", F_PCT2),
            dl("한 국고 3년", "kr_gov3y", F_PCT2),
            dl("한 국고 10년", "kr_gov10y", F_PCT2),
        ],
    )
    D_(
        "rates_us_real",
        "rates",
        "미 10년 실질금리 (%) · 최근 1년",
        "%",
        [dl("미 10년 실질", "us_10y_real", F_PCT2)],
    )
    D_(
        "spread_us_10y2y",
        "rates",
        "미 10년−2년 금리차 (%p) · 최근 1년",
        "%p",
        [dl("10년−2년", "us_10y2y", F_PP2)],
        ref=0,
    )
    D_(
        "spread_krus_10y",
        "rates",
        "한미 10년 금리차 (미 − 한, %p) · 최근 1년",
        "%p",
        [dl("한미 10년 금리차", "krus_10y", F_PP2)],
        ref=0,
    )
    D_("fx_usdkrw", "fx", "원/달러 (원) · 최근 1년", "원", [dl("원/달러", "usdkrw", "{v:,.1f}원")])
    D_(
        "fx_usd_index",
        "fx",
        "달러지수 (광의) · 최근 1년",
        "지수",
        [dl("달러지수", "usd_index", "{v:,.2f}")],
    )
    D_("risk_vix", "risk", "VIX · 최근 1년", "지수", [dl("VIX", "vix", "{v:.2f}")])
    # 신용 스프레드: 1년 창 시작 대비 변화(수준은 싣지 않는다, R1)
    dates, x = s["credit_baa"]
    last = dates[-1].toordinal()
    i0 = next(i for i, t in enumerate(dates) if t.toordinal() >= last - DAILY_WINDOW_DAYS)
    D_(
        "risk_credit",
        "risk",
        "신용 스프레드 (BAA − 10년) · 1년 창 시작 대비 변화 (%p)",
        "%p",
        [
            _daily_line(
                "신용 스프레드 변화", dates, np.asarray(x, dtype=float) - float(x[i0]), F_PP2S
            )
        ],
        ref=0,
    )
    D_(
        "commod_oil",
        "commod",
        "WTI·브렌트 ($/배럴) · 최근 1년",
        "$/배럴",
        [dl("WTI", "wti", "${v:,.2f}"), dl("브렌트유", "brent", "${v:,.2f}")],
    )
    d_, v_ = _roll_sum(*s["fn_kospi"], 20)
    D_(
        "flow_foreign_kospi",
        "flow",
        "외국인 순매수 KOSPI · 20세션 합 (조원) · 최근 1년",
        "조원",
        [_daily_line("외국인 KOSPI", d_, v_, F_FLOW)],
        ref=0,
    )
    d_, v_ = _roll_sum(*s["fn_kosdaq"], 20)
    D_(
        "flow_foreign_kosdaq",
        "flow",
        "외국인 순매수 KOSDAQ · 20세션 합 (조원) · 최근 1년",
        "조원",
        [_daily_line("외국인 KOSDAQ", d_, v_, F_FLOW)],
        ref=0,
    )
    d1, v1 = _ratio_series(*s["tv_kospi"])
    d2, v2 = _ratio_series(*s["tv_kosdaq"])
    D_(
        "flow_trdval_ratio",
        "flow",
        "거래대금 비율 (20세션 평균 ÷ 250세션 평균, 배) · 최근 1년",
        "배",
        [_daily_line("KOSPI", d1, v1, F_RATIO), _daily_line("KOSDAQ", d2, v2, F_RATIO)],
        ref=1,
    )
    M_(
        "macro_prices_us",
        "macro",
        "미 물가 (CPI·PPI 전년비, %) · 최근 5년",
        "%",
        [ml("미 CPI", "us_cpi", F_PCT2S), ml("미 PPI", "us_ppi", F_PCT2S)],
    )
    M_(
        "macro_prices_kr",
        "macro",
        "한 물가 (CPI·PPI 전년비, %) · 최근 5년",
        "%",
        [ml("한 CPI", "kr_cpi", F_PCT2S), ml("한 PPI", "kr_ppi", F_PCT2S)],
    )
    M_(
        "macro_m2",
        "macro",
        "통화량 (미·한 M2 전년비, %) · 최근 5년",
        "%",
        [ml("미 M2", "us_m2", F_PCT2S), ml("한 M2", "kr_m2", F_PCT2S)],
    )
    M_(
        "macro_unrate_us",
        "macro",
        "미 실업률 (수준, %) · 최근 5년",
        "%",
        [ml("미 실업률", "us_unrate", F_PCT2)],
    )
    M_(
        "macro_unrate_kr",
        "macro",
        "한 실업률·고용률 (전년 동월 대비, %p) · 최근 5년",
        "%p",
        [ml("한 실업률", "kr_unrate", F_PP2S), ml("한 고용률", "kr_emprate", F_PP2S)],
        ref=0,
    )
    M_(
        "macro_payems",
        "macro",
        "미 비농업 고용 (월 증감, 천명) · 최근 5년",
        "천명",
        [ml("미 비농업 고용", "us_payems", "{v:+,.0f}천명")],
        ref=0,
    )
    M_(
        "macro_fedfunds",
        "macro",
        "미 기준금리 (FEDFUNDS, %) · 최근 5년",
        "%",
        [ml("미 기준금리", "us_fedfunds", F_PCT2)],
    )
    M_(
        "macro_sentiment",
        "macro",
        "한 소비자심리 (수준, 100 기준) · 최근 5년",
        "p",
        [ml("한 소비자심리", "kr_sent", "{v:.1f}")],
        ref=100,
    )
    return F
