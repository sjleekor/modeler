"""지표 계산: 방향 문턱·급·백분위·월간 날짜 맞춤·사본 이음. numpy만 쓴다.

입력은 날짜 규칙이 이미 적용된 보기(view)다.

* ``kr[series_id][observation_date] = (value, available_from_date)`` — ``available_from_date <= D`` 행만,
  같은 관측일이 여러 행이면 가용일이 가장 늦은 행.
* ``us[series_id][date] = (value, realtime_start)`` — 빈 값을 뺀 뒤 ``realtime_start <= D`` 중 가장 늦은 vintage.
* ``kr_first`` / ``us_first``: 관측일마다 처음 값이 생긴 가용일(발표일).

계획 §5.3(방향 문턱), §5.4(기간표), §10.2 R1\\~R7.
"""

from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from modeler.scores.macro_context.spec import (
    DAILY,
    HISTORY_YEARS,
    MIN_HISTORY_OBS,
    MONTHLY,
    PERIODS,
    Q_FLAT,
    Q_STRONG,
    STALE_DAILY_SESSIONS,
)


class MacroInputError(RuntimeError):
    """입력에 계산에 필요한 series·관측이 없다."""


def d2o(d: dt.date) -> int:
    return d.toordinal()


def r6(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        if np.isnan(x) or np.isinf(x):
            return None
        return round(float(x), 6)
    return x


def minus_years(d: dt.date, n: int) -> dt.date:
    """n년 전 같은 날. 2월 29일이 없는 해는 2월 28일."""
    y = d.year - n
    return dt.date(y, d.month, min(d.day, calendar.monthrange(y, d.month)[1]))


def month_idx(d: dt.date) -> int:
    return d.year * 12 + d.month - 1


def first_of_month(k: int) -> dt.date:
    return dt.date(k // 12, k % 12 + 1, 1)


def ym(k: int) -> str:
    return f"{k // 12:04d}-{k % 12 + 1:02d}"


def busdays_after(obs: dt.date, d: dt.date) -> int:
    """기준일 다음 날부터 D 전날까지 평일 수(공휴일은 반영하지 않는다)."""
    a = obs + dt.timedelta(days=1)
    if a >= d:
        return 0
    return int(np.busday_count(a, d))


# ------------------------------------------------------------------ 분류
def classify(delta: float | None, q30: float, q90: float, spread: bool = False):
    """(방향, 급 여부, 라벨, 화살표). 변화가 0이거나 |Δ| < q30이면 보합, |Δ| >= q90이면 급."""
    if delta is None:
        return None
    if delta == 0 or abs(delta) < q30:
        return ("flat", False, "보합", "→")
    strong = abs(delta) >= q90
    up = delta > 0
    if spread:
        lab = ("급확대" if strong else "확대") if up else ("급축소" if strong else "축소")
    else:
        lab = ("급등" if strong else "상승") if up else ("급락" if strong else "하락")
    arrow = ("▲▲" if strong else "▲") if up else ("▼▼" if strong else "▼")
    return ("up" if up else "down", strong, lab, arrow)


def classify_flow(s: float, q30: float, q90: float):
    d = "up" if s >= 0 else "down"
    a = abs(s)
    grade = "보합" if a < q30 else ("급" if a >= q90 else "보통")
    lab = "순매수" if d == "up" else "순매도"
    return (d, grade == "급", lab, grade)


def thresholds(hist: np.ndarray):
    """(보합 문턱, 급 문턱, 표본 수). 표본이 MIN_HISTORY_OBS보다 적으면 None."""
    a = np.abs(hist[~np.isnan(hist)])
    a = np.round(a, 8)
    if len(a) < MIN_HISTORY_OBS:
        return None
    return (
        float(np.round(np.percentile(a, Q_FLAT), 8)),
        float(np.round(np.percentile(a, Q_STRONG), 8)),
        int(len(a)),
    )


def pctile(vals: np.ndarray, cur: float):
    v = vals[~np.isnan(vals)]
    return float(100.0 * np.sum(v <= cur + 1e-9) / len(v)), int(len(v))


def fmt_level(fmtk: str, v: float) -> str:
    if fmtk == "pctlvl":
        return f"{v:.2f}%"
    if fmtk == "krw":
        return f"{v:,.1f}원"
    if fmtk == "idx":
        return f"{v:,.2f}"
    if fmtk == "usd":
        return f"${v:,.2f}"
    return f"{v}"


# ------------------------------------------------------------------ 일간 지표 하나
def _need(view: dict[str, Any], sid: str) -> dict[Any, Any]:
    rows = view.get(sid)
    if not rows:
        raise MacroInputError(f"series_missing:{sid}")
    return rows


def build_daily(
    spec: dict[str, Any], D: dt.date, us, kr, series: dict[str, tuple[list, np.ndarray]]
):
    """지표 한 개의 계산 결과(dict)와 값 시리즈 ``(dates, x)``."""
    start10 = minus_years(D, HISTORY_YEARS)
    s_ord = d2o(start10)
    splice: dict[str, Any] | None = None
    src_desc: list[str] = []
    if spec.get("derived") == "krus":
        usp_dates, usp_x = series["us_10y"]
        kr_dates, kr_x = series["kr_gov10y"]
        ud = dict(zip(usp_dates, usp_x, strict=True))
        kd = dict(zip(kr_dates, kr_x, strict=True))
        common = sorted(set(ud) & set(kd))
        dates = common
        x = np.round(np.array([ud[d] - kd[d] for d in common]), 8)
        src_desc = [
            "미 10년(us_10y 이음 시리즈) - 한 국고 10년(rate_kr_gov10y), 두 시리즈 모두 값이 있는 날만"
        ]
        splice = {"note": "미 10년 쪽 이음 정보는 us_10y 참조", "common_dates": len(common)}
    elif "us" in spec:
        m = _need(us, spec["us"])
        mdates = sorted(m)
        last = mdates[-1]
        dd = {d: m[d][0] for d in mdates}
        src_desc.append(
            f"FRED {spec['us']} (macro_series, 날짜별 realtime_start <= D 중 최신 vintage, 빈 값 제외)"
        )
        splice = {"macro_series_last_date": last.isoformat(), "macro_series_rows": len(mdates)}
        if "copy" in spec:
            c = kr.get(spec["copy"], {})
            cd = sorted(d for d in c if d > last)
            for d in cd:
                v = c[d][0]
                if spec.get("copy_round") is not None:
                    v = round(v, spec["copy_round"])
                dd[d] = v
            splice.update(
                {
                    "copy_series_id": spec["copy"],
                    "copy_rows": len(cd),
                    "copy_first_date": cd[0].isoformat() if cd else None,
                    "copy_last_date": cd[-1].isoformat() if cd else None,
                    "copy_rounding": spec.get("copy_round"),
                }
            )
            src_desc.append(f"{spec['copy']} 사본 ({last.isoformat()} 다음 날부터 이음)")
            if cd:
                splice["splice_point"] = (
                    f"{last.isoformat()} 까지 macro_series, {cd[0].isoformat()} 부터 {spec['copy']}"
                )
            else:
                splice["splice_point"] = "이을 사본 행 없음 (macro_series만)"
        else:
            splice["splice_point"] = "사본 없음 (macro_series만)"
        dates = sorted(dd)
        x = np.array([dd[d] for d in dates])
    else:
        c = _need(kr, spec["kr"])
        dates = sorted(c)
        x = np.array([c[d][0] for d in dates])
        src_desc.append(f"{spec['kr']} (한국 공통 피쳐 표, available_from_date <= D)")
    n = len(x)
    do = np.array([d2o(d) for d in dates])
    asof = dates[-1]
    out: dict[str, Any] = dict(
        id=spec["id"],
        name=spec["name"],
        group=spec["group"],
        kind=spec["kind"],
        source=src_desc,
        asof=asof.isoformat(),
        asof_lag_sessions=busdays_after(asof, D),
        splice=splice,
        n_obs=n,
        first_obs=dates[0].isoformat(),
    )
    kind = spec["kind"]
    spread = spec.get("spread", False)
    first_ok = dates[0] <= start10
    periods: list[dict[str, Any]] = []
    pos_series = pos_dates = None
    cur_val = None
    plist = list(spec["per"])
    if spec.get("ref_per") and spec["ref_per"] not in plist:
        plist.append(spec["ref_per"])
    if kind in ("diff", "pct"):
        out["value"] = dict(level=r6(x[-1]), display=fmt_level(spec["fmt"], x[-1]))
        pos_series, pos_dates, cur_val = x, do, x[-1]
        for lab in spec["per"]:
            h = PERIODS[lab]
            if n <= h:
                periods.append(dict(label=lab, h=h, change=None, reason="관측 부족"))
                continue
            if kind == "diff":
                hist = np.round(x[h:] - x[:-h], 8)
                cur = float(np.round(x[-1] - x[-1 - h], 8))
            else:
                a, b = x[:-h], x[h:]
                with np.errstate(divide="ignore", invalid="ignore"):
                    hist = np.where((a > 0) & (b > 0), (b / a - 1) * 100, np.nan)
                hist = np.round(hist, 8)
                cur = (
                    float(np.round((x[-1] / x[-1 - h] - 1) * 100, 8))
                    if x[-1 - h] > 0 and x[-1] > 0
                    else None
                )
            mask = do[h:] >= s_ord
            th = thresholds(hist[mask])
            p = dict(
                label=lab,
                h=h,
                start_date=dates[n - 1 - h].isoformat(),
                change=r6(cur),
                change_unit="%p" if kind == "diff" else "%",
            )
            if th and cur is not None:
                cl = classify(cur, th[0], th[1], spread)
                p.update(
                    direction=cl[0],
                    strong=cl[1],
                    direction_label=cl[2],
                    arrow=cl[3],
                    threshold=dict(flat_below=th[0], strong_from=th[1], n=th[2]),
                )
            periods.append(p)
        w = x[-250:]
        hi, lo = float(w.max()), float(w.min())
        if kind == "diff":
            rng = dict(from_high=r6(x[-1] - hi), from_low=r6(x[-1] - lo), unit="%p")
        else:
            rng = dict(
                from_high=r6((x[-1] / hi - 1) * 100), from_low=r6((x[-1] / lo - 1) * 100), unit="%"
            )
        rng.update(
            high=r6(hi), low=r6(lo), n=int(len(w)), window_start=dates[max(0, n - 250)].isoformat()
        )
        out["range_1y"] = rng
        if spec.get("vs_median"):
            med = float(np.median(w))
            out["vs_1y_median_pct"] = r6((x[-1] / med - 1) * 100)
            out["median_1y"] = r6(med)
    elif kind == "sum":
        c = np.concatenate([[0.0], np.cumsum(x)])
        out["value"] = None
        for lab in plist:
            h = PERIODS[lab]
            if n < h:
                periods.append(dict(label=lab, h=h, change=None, reason="관측 부족"))
                continue
            s = np.round((c[h:] - c[:-h]) / 1e4, 8)  # 조원
            end_do = do[h - 1 :]
            cur = float(s[-1])
            th = thresholds(s[end_do >= s_ord])
            p = dict(
                label=lab,
                h=h,
                start_date=dates[n - h].isoformat(),
                change=r6(cur),
                change_unit="조원",
                reference_only=(lab == spec.get("ref_per") and lab not in spec["per"]),
            )
            if th:
                cl = classify_flow(cur, th[0], th[1])
                p.update(
                    direction=cl[0],
                    strong=cl[1],
                    direction_label=cl[2],
                    grade=cl[3],
                    threshold=dict(flat_below=th[0], strong_from=th[1], n=th[2]),
                )
            periods.append(p)
            if lab == "1개월":
                pos_series, pos_dates, cur_val = s, end_do, cur
                out["value"] = dict(
                    level=r6(cur), display=f"{cur:+.2f}조원", label="20세션 순매수 합"
                )
    else:  # ratio — 거래대금은 비율만 싣는다(R2). 평균 조원 값은 body에 넣지 않는다.
        if n < 250:
            raise MacroInputError(f"series_too_short:{spec['kr']}")
        ma20 = np.convolve(x, np.ones(20) / 20, mode="valid")
        ma250 = np.convolve(x, np.ones(250) / 250, mode="valid")
        r = np.round(ma20[249 - 19 :] / ma250, 8)
        end_do = do[249:]
        cur = float(r[-1])
        dev = np.round(r - 1, 8)
        th = thresholds(dev[end_do >= s_ord])
        out["value"] = dict(level=r6(cur), display=f"{cur:.2f}배")
        p = dict(
            label="20세션 평균 ÷ 250세션 평균",
            h=20,
            change=r6((cur - 1) * 100),
            change_unit="%(1년 평균 대비)",
            ratio=r6(cur),
        )
        if th:
            cl = classify(cur - 1, th[0], th[1], False)
            p.update(
                direction=cl[0],
                strong=cl[1],
                direction_label=cl[2],
                arrow=cl[3],
                threshold=dict(
                    flat_below=r6(th[0] * 100), strong_from=r6(th[1] * 100), n=th[2], unit="%"
                ),
            )
        periods.append(p)
        pos_series, pos_dates, cur_val = r, end_do, cur
    out["periods"] = periods
    if pos_series is not None and cur_val is not None:
        if not first_ok:
            out["percentile"] = dict(
                value=None,
                reason=f"10년 이력 없음 (첫 관측 {dates[0].isoformat()} > {start10.isoformat()})",
            )
        else:
            pv, pn = pctile(pos_series[pos_dates >= s_ord], cur_val)
            what = (
                "수준"
                if kind in ("diff", "pct")
                else ("20세션 합" if kind == "sum" else "20/250 평균비")
            )
            out["percentile"] = dict(
                value=r6(pv), n=pn, basis=what, window_start=start10.isoformat()
            )
    out["display"] = {"level": "hide" if spec.get("level_hidden") else "show"}
    return out, (dates, x)


# ------------------------------------------------------------------ 월간 지표 하나
def _shift(a: np.ndarray, n: int) -> np.ndarray:
    o = np.full(len(a), np.nan)
    o[n:] = a[:-n]
    return o


def _measure(arr: np.ndarray, key: str) -> np.ndarray:
    a = arr
    with np.errstate(divide="ignore", invalid="ignore"):
        if key == "lvl":
            return a
        if key == "mom":
            return (a / _shift(a, 1) - 1) * 100
        if key == "ann3":
            return ((a / _shift(a, 3)) ** 4 - 1) * 100
        if key == "yoy":
            return (a / _shift(a, 12) - 1) * 100
        if key == "yd":
            return a - _shift(a, 12)
        if key == "d1":
            return a - _shift(a, 1)
        if key == "d3":
            return a - _shift(a, 3)
        if key == "d12":
            return a - _shift(a, 12)
        if key == "d3avg":
            ma = np.full(len(a), np.nan)
            for t in range(2, len(a)):
                ma[t] = a[t - 2 : t + 1].mean()
            return ma - _shift(ma, 3)
        if key == "low12":
            o = np.full(len(a), np.nan)
            for t in range(11, len(a)):
                o[t] = a[t] - np.nanmin(a[t - 11 : t + 1])
            return o
        if key == "avg3d1":
            d1 = a - _shift(a, 1)
            o = np.full(len(a), np.nan)
            for t in range(3, len(a)):
                o[t] = d1[t - 2 : t + 1].mean()
            return o
        if key == "yoy_d3":
            y = (a / _shift(a, 12) - 1) * 100
            return y - _shift(y, 3)
    raise KeyError(key)


def build_monthly(spec: dict[str, Any], D: dt.date, us, kr):
    """월간 지표 한 개의 계산 결과(dict)와 값 시리즈 ``(월 순번 배열, 값)``. 월은 날짜로 맞춘다(R6)."""
    start10 = minus_years(D, HISTORY_YEARS)
    s_m = month_idx(start10)
    if spec["src"] == "us":
        m = _need(us, spec["sid"])
        srcd = f"FRED {spec['sid']} (macro_series, realtime_start <= D 중 최신 vintage)"
    else:
        m = _need(kr, spec["sid"])
        srcd = f"{spec['sid']} (한국 공통 피쳐 표, available_from_date <= D)"
    pub = {month_idx(d): (v, rs) for d, (v, rs) in m.items()}
    ks = sorted(pub)
    k0, k1 = ks[0], ks[-1]
    miss = [k for k in range(k0, k1 + 1) if k not in pub]
    arr = np.full(k1 - k0 + 1, np.nan)
    for k in ks:
        arr[k - k0] = pub[k][0]
    asof = ym(k1)
    pubdate = pub[k1][1]
    lag = month_idx(D) - k1
    kidx = np.arange(k0, k1 + 1)
    hist_mask = kidx >= s_m
    out: dict[str, Any] = dict(
        id=spec["id"],
        name=spec["name"],
        group="macro",
        kind="monthly",
        source=[srcd],
        asof=asof,
        asof_lag_months=lag,
        last_publication=pubdate.isoformat(),
        n_obs=len(ks),
        first_obs=ym(k0),
        missing_months_inside=len(miss),
        splice=None,
    )
    vm = _measure(arr, spec["value"])
    vu = {
        "yoy": "%",
        "lvl": {"us_unrate": "%", "us_fedfunds": "%", "kr_sent": "p"}.get(spec["id"], ""),
        "d1": "천명",
        "yd": "%p",
    }[spec["value"]]
    vv = float(vm[-1])
    vok = not np.isnan(vv)  # 12개월 전 달이 비었으면 전년비를 못 낸다
    vdisp = None
    if vok:
        vdisp = {
            "%": f"{vv:+.2f}%",
            "%p": f"{vv:+.2f}%p",
            "천명": f"{vv:+,.0f}천명",
            "p": f"{vv:.1f}",
        }.get(vu)
        if spec["value"] == "lvl" and vu == "%":
            vdisp = f"{vv:.2f}%"
    out["value"] = dict(label=spec["vlabel"], level=r6(vv), unit=vu, display=vdisp)
    vh = vm[hist_mask]
    nvalid = int(np.sum(~np.isnan(vh)))
    nmiss = int(len(vh) - nvalid)
    if vok and k0 <= s_m and nvalid >= 100:
        pv, pn = pctile(vh, vv)
        out["percentile"] = dict(
            value=r6(pv),
            n=pn,
            missing_months=nmiss,
            basis=spec["vlabel"],
            window_start=f"{start10.year}-{start10.month:02d}",
        )
    elif not vok:
        out["percentile"] = dict(value=None, reason="현재 값 없음 (비교할 달이 비어 있음)")
    else:
        out["percentile"] = dict(
            value=None, reason=f"10년 이력 부족 (첫 관측 {ym(k0)}, 유효 {nvalid}개월)"
        )
    if spec["id"] in ("kr_unrate", "kr_emprate"):
        out["level_note"] = f"수준 {float(arr[-1]):.1f}%"
    measures: list[dict[str, Any]] = []
    primary = None
    for lab, key, unit, hasdir, prim in spec["m"]:
        a = _measure(arr, key)
        cur = float(np.round(a[-1], 8))
        md: dict[str, Any] = dict(label=lab, key=key, unit=unit, change=r6(cur), primary=bool(prim))
        if hasdir and not np.isnan(cur):
            th = thresholds(np.round(a[hist_mask], 8))
            if th:
                cl = classify(cur, th[0], th[1], False)
                md.update(
                    direction=cl[0],
                    strong=cl[1],
                    direction_label=cl[2],
                    arrow=cl[3],
                    threshold=dict(flat_below=th[0], strong_from=th[1], n=th[2]),
                )
                if prim:
                    ar = np.round(a, 8)

                    def dr(v, flat=th[0]):
                        if np.isnan(v) or v == 0 or abs(v) < flat:
                            return 0
                        return 1 if v > 0 else -1

                    lastd = dr(ar[-1])
                    cnt = 0
                    if lastd != 0:
                        t = len(ar) - 1
                        while t >= 0 and dr(ar[t]) == lastd:
                            cnt += 1
                            t -= 1
                    md["streak"] = dict(
                        months=cnt, direction=cl[0] if lastd != 0 else "flat", text=None
                    )
                    if cnt >= 2:
                        md["streak"]["text"] = f"{cnt}개월 연속 {'상승' if lastd > 0 else '하락'}"
                    primary = md
        measures.append(md)
    out["measures"] = measures
    out["primary_measure"] = primary["label"] if primary else None
    out["streak"] = primary.get("streak") if primary else None
    out["display"] = {"level": "show"}
    return out, (kidx, vm)


def annotate_daily(o: dict[str, Any]) -> None:
    o["stale"] = bool(o["asof_lag_sessions"] > STALE_DAILY_SESSIONS)


def annotate_monthly(
    o: dict[str, Any],
    spec: dict[str, Any],
    D: dt.date,
    us_first,
    kr_first,
    series: tuple[np.ndarray, np.ndarray],
) -> None:
    """발표일·갱신 상태·직전 발표값·원천 결측 달."""
    sid = spec["sid"]
    first = us_first[sid] if spec["src"] == "us" else kr_first[sid]
    fr: dict[int, dt.date] = {}
    for d, rd in first.items():
        fr[month_idx(d)] = rd
    k1 = max(fr)
    o["release_date"] = fr[k1].isoformat()
    lo = minus_years(D, 3)
    delays = sorted((rd - first_of_month(k)).days for k, rd in fr.items() if lo <= rd <= D)
    if delays:
        med = float(np.median(delays))
        k = month_idx(D)
        while (D - first_of_month(k)).days < med:
            k -= 1
        late = k - k1
        o["status"] = "정상" if late <= 0 else f"지연 {late}개월"
        o["status_ok"] = bool(late <= 0)
        o["status_months_late"] = int(max(late, 0))
        o["expected_last_month"] = ym(k)
        o["delay_median_days"] = r6(med)
        o["delay_sample_n"] = len(delays)
    else:
        o["status"] = "판정 불가"
        o["status_ok"] = True
        o["status_months_late"] = 0
        o["expected_last_month"] = None
        o["delay_median_days"] = None
        o["delay_sample_n"] = 0
    o["stale"] = bool(not o["status_ok"])
    _, vm = series
    pv = float(vm[-2]) if len(vm) >= 2 and not np.isnan(vm[-2]) else None
    o["prev_value"] = dict(asof=ym(k1 - 1), level=r6(pv))
    s_m = month_idx(minus_years(D, HISTORY_YEARS))
    o["missing_months_10y"] = [ym(k) for k in range(max(s_m, min(fr)), k1 + 1) if k not in fr]


# ------------------------------------------------------------------ 조건 (a)(b)(c)(d)
def conds(o: dict[str, Any]) -> dict[str, Any]:
    """a: 표시 기간 중 급, b: 중기·단기 방향 차이, c: 10년 백분위 극단, d: 월간 3개월 이상 연속."""
    c: dict[str, Any] = {"a": [], "b": None, "c": None, "d": None}
    if o["kind"] == "monthly":
        for m in o["measures"]:
            if m.get("strong"):
                c["a"].append(
                    dict(
                        label=m["label"],
                        dir=m["direction"],
                        dlabel=m["direction_label"],
                        change=m["change"],
                        unit=m["unit"],
                        dec=2,
                        mult=r6(abs(m["change"]) / m["threshold"]["strong_from"]),
                    )
                )
        st = o.get("streak")
        if st and st.get("months", 0) >= 3:
            pm = next(m for m in o["measures"] if m["primary"])
            c["d"] = dict(
                months=st["months"],
                text=st["text"],
                measure=pm["label"],
                change=pm["change"],
                unit=pm["unit"],
            )
        return c
    kind = o["kind"]
    unit = {"diff": "%p", "pct": "%", "sum": "조원", "ratio": "%"}[kind]
    dec = 1 if kind == "ratio" else 2
    shown = [
        p
        for p in o["periods"]
        if p.get("change") is not None and "direction" in p and not p.get("reference_only")
    ]
    shown.sort(key=lambda p: p["h"])
    for p in shown:
        if p.get("strong"):
            lab = "1년 평균 대비" if kind == "ratio" else p["label"]
            c["a"].append(
                dict(
                    label=lab,
                    dir=p["direction"],
                    dlabel=p["direction_label"],
                    change=p["change"],
                    unit=unit,
                    dec=dec,
                    mult=r6(abs(p["change"]) / p["threshold"]["strong_from"]),
                    h=p["h"],
                )
            )
    if kind == "sum":
        nonflat = [p for p in shown if p.get("grade") != "보합"]
    else:
        nonflat = [p for p in shown if p["direction"] != "flat"]
    if len(nonflat) >= 2:
        sp, lp = nonflat[0], nonflat[-1]
        if sp["direction"] != lp["direction"]:
            mag = min(
                abs(sp["change"]) / max(sp["threshold"]["flat_below"], 1e-12),
                abs(lp["change"]) / max(lp["threshold"]["flat_below"], 1e-12),
            )
            c["b"] = dict(
                short=sp["label"],
                short_dir=sp["direction"],
                long=lp["label"],
                long_dir=lp["direction"],
                short_change=sp["change"],
                long_change=lp["change"],
                unit=unit,
                mag=r6(mag),
            )
    pc = o.get("percentile") or {}
    if (
        kind in ("diff", "pct")
        and pc.get("value") is not None
        and (pc["value"] >= 95 or pc["value"] <= 5)
    ):
        c["c"] = dict(p=pc["value"])
    return c


# ------------------------------------------------------------------ 상태 계산 (D 하나)
@dataclass
class Computed:
    D: dt.date
    inds: list[dict[str, Any]]
    mons: list[dict[str, Any]]
    series: dict[str, tuple[Any, np.ndarray]] = field(default_factory=dict)


def compute(D: dt.date, kr, us, kr_first, us_first) -> Computed:
    """보기(view) 하나로 지표 31개를 계산한다. 사본·이음·방향·백분위·갱신 상태·조건까지."""
    series: dict[str, tuple[Any, np.ndarray]] = {}
    inds = []
    for spec in DAILY:
        o, s = build_daily(spec, D, us, kr, series)
        series[spec["id"]] = s
        inds.append(o)
    mons = []
    for spec in MONTHLY:
        o, s = build_monthly(spec, D, us, kr)
        series[spec["id"]] = s
        mons.append(o)
    for o in inds:
        annotate_daily(o)
    for o, spec in zip(mons, MONTHLY, strict=True):
        annotate_monthly(o, spec, D, us_first, kr_first, series[spec["id"]])
    for o in inds + mons:
        o["conditions"] = conds(o)
    return Computed(D, inds, mons, series)
