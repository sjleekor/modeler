"""일일 리포트 "거시 흐름" 섹션 렌더러. 계산 모듈이 만든 body(dict)로 ``macro.md`` 본문과 SVG를 만듭니다.

``render(body)`` 하나가 입구입니다. 파일·시각·난수를 쓰지 않는 순수 함수라, 같은 body는 같은 바이트를 냅니다.

- markdown(GFM) 요소만 씁니다: 제목, 문단, 인용, 목록, 표, 굵게, 이미지, 링크. HTML 태그는 쓰지 않습니다.
- 범위 물결표는 ``\\~``로 이스케이프합니다.
- 그림은 ``assets/macro-<figure name>.svg``로 가리킵니다. SVG에는 스크립트·이벤트 속성·외부 참조·
  ``foreignObject``를 쓰지 않습니다(``svg.py``와 같은 규칙).
- 해석·전망·매매 문구는 없습니다. 계산 모듈이 규칙으로 만든 사실 문구만 옮깁니다.
"""

from __future__ import annotations

import datetime as dt
import math
import re
from xml.sax.saxutils import escape

from modeler.scores.macro_context.spec import ARROW, GROUPS, MINUS, PERIODS
from modeler.scores.macro_context.summary import fmt_chg, level_text, neg, obs_text, sgn

WD = "월화수목금토일"
TITLE = "거시 흐름"
NOT_SIGNAL = "점수·순위에 쓰지 않습니다. 매매 신호가 아닙니다."
ASSET_PREFIX = "assets/macro-"
SVG_MAX_BYTES = 200 * 1024

#: 입력 기준일 표의 묶음(시제품과 같다). 사본 이음 대상은 한국 표에 사본이 있는 네 지표다.
KR_DAILY_IDS = (
    "kr_cd91",
    "kr_gov3y",
    "kr_gov10y",
    "usdkrw",
    "fn_kospi",
    "fn_kosdaq",
    "tv_kospi",
    "tv_kosdaq",
)
#: 입력 기준일 표·사본 이음 문구에 쓰는 줄임 이름. 없으면 짧은 이름(`short_name`)을 쓴다.
ABBR = {
    "us_2y": "미 2년",
    "us_10y": "미 10년",
    "vix": "VIX",
    "wti": "WTI",
    "us_3m": "3개월물",
    "us_10y_real": "실질금리",
    "us_10y2y": "10년−2년",
    "credit_baa": "BAA",
    "usd_index": "달러지수",
    "brent": "브렌트",
}
FLOW_COLS = (("1주", "5세션"), ("1개월", "20세션"), ("3개월", "60세션"), ("6개월", "120세션(참고)"))


# ------------------------------------------------------------------ 입구
def render(body: dict) -> tuple[str, dict[str, str]]:
    """(macro.md 본문, {"macro-<figure name>.svg": svg 텍스트})를 돌려줍니다."""
    if body.get("state") == "failed":
        return _failed_md(body), {}
    return _Doc(body).md(), _svgs(body)


def _safe_code(value: object, default: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_.\-]", "_", str(value)) if value not in (None, "") else ""
    return text or default


def _failed_md(body: dict) -> str:
    unit = _safe_code(body.get("unit"), "-")
    reason = _safe_code(body.get("reason"), "unknown")
    lines = [
        f"# {TITLE} · {unit}",
        "",
        NOT_SIGNAL,
        "",
        f"거시 흐름을 이번 실행에서 만들지 못했습니다(사유 코드: {reason}).",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ markdown
def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _table(head: list[str], rows: list[list[str]], align: list[str] | None = None) -> list[str]:
    align = align or ["---"] * len(head)
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join(align) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return lines


def _hmd(iso: str) -> str:
    return f"{iso[5:7]}/{iso[8:10]}"


def _names(objs: list[dict], sep: str) -> str:
    """줄임 이름을 `ABBR` 순서대로 `sep`로 잇는다. 앞뒤가 모두 "미 "로 시작하면 뒤의 "미 "를 떼고 `·`로 붙인다."""
    out = ""
    prev = ""
    rank = {k: i for i, k in enumerate(ABBR)}
    for o in sorted(objs, key=lambda x: rank.get(x["id"], len(rank))):
        name = ABBR.get(o["id"], o["short_name"])
        if not out:
            out = name
        elif prev.startswith("미 ") and name.startswith("미 "):
            out += "·" + name[2:]
        else:
            out += sep + name
        prev = name
    return out


def _us_daily(inds: list[dict]) -> tuple[list[dict], list[dict]]:
    """미국 원천 일간 지표를 (사본으로 이은 것, macro_series만 쓴 것)으로 나눈다.

    미국 원천은 splice에 `macro_series_last_date`가 있는 지표이고, 이은 것은 `copy_rows > 0`이다.
    """
    us = [o for o in inds if (o.get("splice") or {}).get("macro_series_last_date")]
    spliced = [o for o in us if o["splice"].get("copy_rows", 0) > 0]
    ids = {o["id"] for o in spliced}
    return spliced, [o for o in us if o["id"] not in ids]


def _tilde(text: str) -> str:
    return text.replace("~", "\\~")


class _Doc:
    def __init__(self, body: dict) -> None:
        self.b = body
        self.D = dt.date.fromisoformat(body["unit"])
        inds = body["indicators"]
        self.inds = [o for o in inds if o["kind"] != "monthly"]
        self.mons = [o for o in inds if o["kind"] == "monthly"]
        self.by_id = {o["id"]: o for o in inds}
        self.L: list[str] = []

    def a(self, text: str = "") -> None:
        self.L.append(text)

    # ---- 지표 한 개의 표시 조각
    def short(self, iid: str) -> str:
        return self.by_id[iid]["short_name"]

    def pcell(self, o: dict, p: dict | None) -> str:
        if p is None or p.get("change") is None:
            return ""
        kind = o["kind"]
        unit = {"diff": "%p", "pct": "%", "sum": "조원", "ratio": "%"}[kind]
        dec = 1 if kind == "ratio" else 2
        txt = fmt_chg(p["change"], unit, dec)
        if "direction" not in p:
            return txt
        arr = ARROW[p["direction"]]
        if kind == "sum" and p.get("grade") == "보합":
            arr = ARROW["flat"]
        txt = f"{arr} {txt}"
        if p.get("strong") and not p.get("reference_only"):
            txt = f"**{txt}**"
        return txt

    @staticmethod
    def pctl(o: dict) -> str:
        p = o.get("percentile") or {}
        if p.get("value") is None:
            return "—"
        return f"P{p['value']:.0f}"

    @staticmethod
    def level_cell(o: dict) -> str:
        text = level_text(o)
        return "—" if text is None else text

    @staticmethod
    def note_cell(o: dict) -> str:
        notes = []
        c = o["conditions"]
        if c["b"]:
            notes.append(f"중기 {ARROW[c['b']['long_dir']]} / 단기 {ARROW[c['b']['short_dir']]}")
        if o["stale"]:
            notes.append(f"갱신 지연 {o['asof_lag_sessions']}세션")
        if o["id"] == "usdkrw":
            notes.append(f"1년 고점 대비 {sgn(o['range_1y']['from_high'], 2, '%')}")
        if o["id"] == "vix":
            notes.append(f"1년 중앙값 대비 {sgn(o['vs_1y_median_pct'], 1, '%')}")
        return "; ".join(notes)

    @staticmethod
    def mcell(o: dict) -> str:
        m = next(x for x in o["measures"] if x["primary"])
        # 비교할 달이 원천에 비었으면 계산 모듈이 변화·방향을 비워 둔다: 값만, 값도 없으면 "—".
        txt = fmt_chg(m.get("change"), m["unit"]) or "—"
        if m.get("direction") in ARROW:
            txt = f"{ARROW[m['direction']]} {txt}"
            if m.get("strong"):
                txt = f"**{txt}**"
        return f"{m['label']} {txt}"

    @staticmethod
    def mvalue(o: dict, v: float | None) -> str:
        if v is None:
            return "—"
        u = o["value"]["unit"]
        if u == "%" and o["value"]["label"] == "수준":
            return f"{v:.2f}%"
        if u == "p":
            return f"{v:.1f}"
        return fmt_chg(v, u)

    def group_line(self, objs: list[dict]) -> str:
        parts = []
        for o in objs:
            t = obs_text(o)
            if t:
                parts.append(f"{o['short_name']}: {t}")
        return "관찰: " + ("; ".join(parts) if parts else "조건에 해당하는 지표 없음")

    # ---- 표
    def daily_table(self, group: str) -> list[str]:
        sel = [o for o in self.inds if o["group"] == group and o["kind"] in ("diff", "pct")]
        labs = [lab for lab in PERIODS if any(p["label"] == lab for o in sel for p in o["periods"])]
        rows = []
        for o in sel:
            pm = {p["label"]: p for p in o["periods"]}
            rows.append(
                [o["display_name"], o["asof"], self.level_cell(o)]
                + [self.pcell(o, pm.get(lab)) for lab in labs]
                + [self.pctl(o), self.note_cell(o)]
            )
        return _table(["지표", "기준일", "최근값"] + labs + ["10년 백분위", "비고"], rows)

    def flow_tables(self) -> list[str]:
        lines: list[str] = []
        rows = []
        for o in (x for x in self.inds if x["kind"] == "sum"):
            pm = {p["label"]: p for p in o["periods"]}
            rows.append(
                [o["display_name"], o["asof"]]
                + [self.pcell(o, pm.get(k)) for k, _ in FLOW_COLS]
                + [self.pctl(o), self.note_cell(o)]
            )
        lines += _table(
            ["지표", "기준일"] + [v for _, v in FLOW_COLS] + ["10년 백분위", "비고"], rows
        )
        lines.append("")
        lines.append(
            "순매수는 기간 합(조원)입니다. ↑는 순매수, ↓는 순매도입니다. 120세션 합은 참고용이라 굵게 "
            "표시하거나 방향 판정에 쓰지 않습니다. 10년 백분위는 20세션 합 기준입니다."
        )
        lines.append("")
        rows = []
        for o in (x for x in self.inds if x["kind"] == "ratio"):
            rows.append(
                [
                    o["display_name"],
                    o["asof"],
                    f"{o['value']['level']:.2f}배",
                    self.pcell(o, o["periods"][0]),
                    self.pctl(o),
                    self.note_cell(o),
                ]
            )
        lines += _table(["지표", "기준일", "비율", "1년 평균 대비", "10년 백분위", "비고"], rows)
        lines.append("")
        lines.append(
            "비율 1배는 20세션 평균이 250세션 평균과 같다는 뜻입니다. 10년 백분위는 비율 기준입니다."
        )
        return lines

    def monthly_table(self) -> list[str]:
        rows = []
        for o in self.mons:
            st = (o.get("streak") or {}).get("text") or "—"
            rows.append(
                [
                    o["display_name"],
                    o["asof"],
                    o["release_date"],
                    o["status"],
                    "—" if o["value"].get("display") is None else neg(o["value"]["display"]),
                    self.mvalue(o, o["prev_value"]["level"]),
                    self.mcell(o),
                    st,
                    self.pctl(o),
                ]
            )
        return _table(
            [
                "지표(계산)",
                "관측월",
                "발표일",
                "갱신 상태",
                "최근값",
                "이전 발표값",
                "추세",
                "연속",
                "10년 백분위",
            ],
            rows,
        )

    # ---- 절
    def head(self) -> None:
        a = self.a
        a(f"# {TITLE} · {self.D.isoformat()}")
        a()
        a(NOT_SIGNAL)
        a()
        hd = self.b["header"]
        stale_d = hd["stale_daily"]
        us_weekly = hd["stale_daily_us_weekly"]
        if not stale_d:
            sd = "갱신 지연 0개"
        elif len(us_weekly) == len(stale_d):
            sd = f"갱신 지연 {len(stale_d)}개(미국 주간 수집분)"
        else:
            sd = (
                f"갱신 지연 {len(stale_d)}개(미국 주간 수집분 {len(us_weekly)}개, "
                f"그 밖 {len(stale_d) - len(us_weekly)}개)"
            )
        line = (
            f"일간 지표 기준일 {_hmd(hd['daily_asof_min'])}\\~{_hmd(hd['daily_asof_max'])} · {sd}"
        )
        if hd["stale_monthly"]:
            line += f" · 월간 갱신 지연 {len(hd['stale_monthly'])}개"
        a(line)
        a()

    def summary(self) -> None:
        a = self.a
        sm = self.b["summary"]
        a("## 이번에 눈여겨볼 변화")
        a()
        rows = [
            [
                r["group_name"],
                r["observation"],
                r["representative"],
                r["asof"] + (" (갱신 지연)" if r["stale"] else ""),
            ]
            for r in sm["rows"]
        ]
        self.L += _table(["영역", "관찰", "대표 수치", "기준일"], rows)
        a()
        a(
            "선정 규칙: 후보가 있는 영역마다 먼저 1행(그 영역에서 우선순위가 가장 높은 지표)을 뽑고, "
            "남는 자리는 우선순위로 채웁니다. 우선순위는 표시 기간 중 급등·급락 > 중기와 단기의 방향 차이 > "
            "10년 백분위 P5 이하·P95 이상 > 월간 3개월 이상 연속이고, 같으면 크기 순입니다. "
            f"지표마다 한 행, 영역당 최대 {sm['max_per_group']}행, "
            f"전체 {sm['min_rows']}\\~{sm['max_rows']}행입니다. "
            f"나머지 후보 {sm['not_selected']}개는 영역별 상세에 있습니다."
        )
        a()

    def vs_prev(self) -> None:
        a = self.a
        vsp = self.b["vs_prev"]
        mx = vsp["max_count"]
        pd_ = dt.date.fromisoformat(vsp["previous_report_date"])
        a("## 지난 보고서 대비")
        a()
        a(f"이전 보고서 날짜는 {pd_.isoformat()} ({WD[pd_.weekday()]})입니다.")
        a()
        for title, key in (
            ("새로 나타남", "new"),
            ("방향 차이 새로 생김", "direction_diff_new"),
            (f"이어짐 (이번 보고서 포함, {mx}회에 닿으면 {mx}회 이상으로 표시)", "continued"),
            ("해제", "resolved"),
        ):
            a(f"**{title}**")
            a()
            items = vsp[key]
            if not items:
                a("- 없음")
            byid: dict[str, list[dict]] = {}
            for e in items:
                byid.setdefault(e["id"], []).append(e)
            for iid, es in byid.items():
                parts = []
                for e in es:
                    part = e["text"][len(self.short(iid)) + 1 :]
                    if key == "continued":
                        n = e["streak"]
                        part += f" {mx}회 이상" if n >= mx else f" {n}회"
                    parts.append(part)
                a(f"- {self.short(iid)}: " + " · ".join(parts))
            a()
        a(
            "이전 상태는 저장된 보고서가 아니라 같은 입력을 그날 가용분으로 다시 계산한 값입니다. "
            "미국 자료는 realtime_start가 그날보다 앞선 것만 씁니다."
        )
        a()

    def detail(self) -> None:
        a = self.a
        a("## 영역별 상세")
        a()
        figs_by: dict[str, list[dict]] = {}
        for f in self.b["figures"]:
            figs_by.setdefault(f["group"], []).append(f)
        for g, gname in GROUPS:
            objs = self.mons if g == "macro" else [x for x in self.inds if x["group"] == g]
            a(f"### {gname}")
            a()
            a(self.group_line(objs))
            a()
            if g == "flow":
                self.L += self.flow_tables()
            elif g == "macro":
                self.L += self.monthly_table()
                a()
                miss: dict[tuple, list[str]] = {}
                for o in self.mons:
                    if o["missing_months_10y"]:
                        miss.setdefault(tuple(o["missing_months_10y"]), []).append(o["short_name"])
                for months, names in miss.items():
                    a(f"{'·'.join(names)}: 원천 결측 {len(months)}개월({', '.join(months)})")
                    a()
                kru = [o for o in self.mons if o["id"] in ("kr_unrate", "kr_emprate")]
                levels = ", ".join(
                    f"{o['short_name']} {o['level_note'].replace('수준 ', '')}" for o in kru
                )
                a(f"한 실업률·고용률 수준(원계열): {levels}. 표의 값은 전년 동월 대비입니다.")
            else:
                self.L += self.daily_table(g)
                if g in ("rates", "risk"):
                    a()
                    a("금리차·스프레드의 ↑는 확대, ↓는 축소입니다.")
                if g == "risk":
                    a()
                    a("신용 스프레드는 원천 약관상 수준을 싣지 않고 변화·백분위만 적습니다.")
            a()
            for f in figs_by.get(g, []):
                a(f"![{f['title']}]({ASSET_PREFIX}{f['name']}.svg)")
                a()

    def appendix(self) -> None:
        a = self.a
        a("## 부록")
        a()
        a("### 산식과 방향 문턱")
        a()
        for text in APPENDIX_RULES:
            a(text)
        a()
        a("### 기간표 요약")
        a()
        rows = []
        for r in self.b["rules"]["periods"]:
            if r["kind"] == "ratio":
                per = "1년 평균 대비 (20세션 평균 ÷ 250세션 평균)"
            elif r["kind"] == "sum":
                per = ", ".join(r["periods"])
                if r.get("reference_period"):
                    per += f" (참고 {r['reference_period']})"
                per += " 합"
            else:
                per = ", ".join(r["periods"])
            rows.append([r["name"], per])
        self.L += _table(["지표", "표시 기간·변화"], rows)
        a()
        self.inputs_section()
        self.splice_section()
        self.checks_section()
        self.source_section()
        self.legend_section()

    def inputs_section(self) -> None:
        a = self.a
        D = self.D
        kr_snap = self.b["inputs"]["kr"]["snapshot_date"]
        us_snap = self.b["inputs"]["us"]["snapshot_date"]
        a("### 입력 기준일")
        a()
        a(
            f"리포트 날짜 {D.isoformat()} ({WD[D.weekday()]}) · KR 스냅샷 {kr_snap} · "
            f"US macro_series 스냅샷 {us_snap}"
        )
        a()
        a("| 입력 | 지표 수 | 기준일 | 지연 |")
        a("| --- | ---: | --- | --- |")

        def grp(label: str, sel: list[dict]) -> None:
            if not sel:
                return
            ds = [o["asof"] for o in sel]
            lg = [o["asof_lag_sessions"] for o in sel]
            a(
                _tilde(
                    f"| {label} | {len(sel)} | {min(ds)} ~ {max(ds)} | {min(lg)} ~ {max(lg)}세션 |"
                )
            )

        def pick(ids: tuple[str, ...]) -> list[dict]:
            return [o for o in self.inds if o["id"] in ids]

        grp("KR 일간 (ECOS 금리·환율·증시 자금)", pick(KR_DAILY_IDS))
        spliced, weekly = _us_daily(self.inds)
        if spliced:
            grp(f"US 일간, 한국 표 사본으로 이음 ({_names(spliced, ', ')})", spliced)
        if weekly:
            grp(f"US 일간, macro_series만 ({_names(weekly, '·')})", weekly)
        grp("파생 (한미 10년 금리차)", pick(("krus_10y",)))
        for lab, src in (("US 월간 (macro_series)", "us_"), ("KR 월간 (공통 피쳐 표)", "kr_")):
            sel = [o for o in self.mons if o["id"].startswith(src)]
            if not sel:
                continue
            ds = [o["asof"] for o in sel]
            lg = [o["asof_lag_months"] for o in sel]
            a(_tilde(f"| {lab} | {len(sel)} | {min(ds)} ~ {max(ds)} | {min(lg)} ~ {max(lg)}개월 |"))
        a()

    def splice_section(self) -> None:
        a = self.a
        a("### 사본 이음")
        a()
        spliced, _ = _us_daily(self.inds)
        if spliced:
            a(
                f"{_names(spliced, '·')}는 macro_series 마지막 날 다음 날부터 "
                "한국 공통 피쳐 표의 사본으로 이었습니다."
            )
            a()
        for o in spliced:
            s = o["splice"]
            if s.get("copy_series_id"):
                rounded = (
                    ", 사본 소수 둘째 자리 반올림" if s.get("copy_rounding") is not None else ""
                )
                a(f"- {o['short_name']}: {s['splice_point']} ({s['copy_rows']}행{rounded})")
        a(
            "- 한미 10년 금리차는 이음을 거친 미 10년 시리즈에서 한 국고 10년을 뺍니다. 두 시리즈 모두 값이 있는 날만 씁니다."
        )
        a(
            "- WTI와 KOSDAQ 거래대금은 지속률을 따로 재지 않아 각각 브렌트, KOSPI와 같은 기간으로 정했습니다."
        )
        a()

    def checks_section(self) -> None:
        a = self.a
        ck = self.b["checks"]
        a("### 입력 검사")
        a()
        a(
            f"모든 기준일 ≤ {self.D.isoformat()}, 한국 available_from_date 최대 "
            f"{ck['kr_max_available_from_used']}, 미국 realtime_start 최대 "
            f"{ck['us_max_realtime_start_used']} → {'통과' if ck['overall_pass'] else '실패'}"
        )
        a()

    def source_section(self) -> None:
        a = self.a
        kd = dt.date.fromisoformat(self.b["inputs"]["kr"]["snapshot_date"])
        a("### 출처")
        a()
        a(
            "- FRED: 미 금리(3개월·2년·10년·실질 10년)·달러지수·기준금리 — Board of Governors of the Federal "
            "Reserve System (US) via FRED. 미 CPI·PPI·실업률·비농업 고용 — U.S. Bureau of Labor Statistics via "
            "FRED. 미 M2 — Board of Governors via FRED. WTI·브렌트 — U.S. Energy Information Administration "
            "via FRED. VIX — Chicago Board Options Exchange via FRED. 10년−2년 금리차 — Federal Reserve Bank "
            "of St. Louis via FRED. 신용 스프레드 — Moody's via FRED."
        )
        a(
            "- ECOS: 출처 : ECOS(한국은행, 시장금리(일별)) · ECOS(한국은행, 소비자동향조사) · "
            "ECOS(서울외국환중개, 주요국 통화의 대원화환율) · ECOS(한국거래소, 주식시장(일)) · "
            "ECOS(국가데이터처, 소비자물가지수·경제활동인구) · ECOS(생산자물가지수·M2, 작성기관 표시 없음), "
            f"{kd.year}.{kd.month}.{kd.day}. 스냅샷 기준."
        )
        a(
            "- 수준이 아닌 값(변화폭·방향·10년 백분위·합계·비율)은 ECOS·FRED 원자료를 가공한 값입니다."
        )
        spliced, _ = _us_daily(self.inds)
        if spliced:
            vix = (
                "(VIX 사본 원천은 FinanceDataReader)"
                if any(o["id"] == "vix" for o in spliced)
                else ""
            )
            a(
                f"- {_names(spliced, '·')}의 최근 며칠은 한국 공통 피쳐 표의 사본으로 이었습니다{vix}."
            )
        a(
            "- 신용 스프레드 수준과 거래대금 수준은 원천 약관과 이용 조건 때문에 싣지 않았습니다. 변화·백분위·비율만 적었습니다."
        )
        a()

    def legend_section(self) -> None:
        a = self.a
        a("### 범례")
        a()
        a(
            "- ↑ 상승(확대, 순매수), ↓ 하락(축소, 순매도), → 보합. 굵게는 급등·급락(급확대·급축소, "
            "순매수·순매도 큰 쪽 10%)입니다. 마이너스는 −로 적었습니다."
        )
        a("- 금리차·스프레드의 ↑는 확대, ↓는 축소입니다.")
        a(
            '- 비고의 "중기 ↑ / 단기 ↓"는 표시 기간 중 가장 긴 기간과 가장 짧은 기간의 방향이 다르다는 뜻입니다.'
        )
        a(
            "- 그림의 옅은 영역은 최근 3개월, 점선은 기준선(0 또는 1배, 100)입니다. 선은 그 지표의 마지막 관측일에서 멈춥니다."
        )
        a()

    def md(self) -> str:
        self.head()
        self.summary()
        self.vs_prev()
        self.detail()
        self.appendix()
        return "\n".join(self.L)


APPENDIX_RULES = (
    "- 방향은 지표마다, 기간마다 지난 10년의 같은 기간 변화 크기(|Δ|)와 비교해 정합니다. 작은 쪽 30% 안은 "
    "보합(→), 큰 쪽 10% 안은 급등·급락(굵게)입니다. 금리차·스프레드는 확대·축소·급확대·급축소로 적습니다. "
    "변화가 정확히 0이면 보합입니다.",
    "- 문턱은 고정값이 아니라 그날 입력의 지난 10년으로 계산합니다. 분위수는 `numpy.percentile` 선형 보간입니다.",
    "- 기간은 관측 행 기준입니다. 1주 = 5세션, 1개월 = 20세션, 3개월 = 60세션, 6개월 = 120세션, "
    "1년 = 250세션입니다.",
    "- 변화: 금리·금리차·스프레드는 차이(%p), 그 밖의 가격·지수는 변화율(%)입니다. 외국인 순매수는 기간 "
    "합(조원)이고 방향은 부호(↑ 순매수, ↓ 순매도), 급은 큰 쪽 10%입니다. 거래대금 비율은 20세션 평균 ÷ "
    "250세션 평균이며 비율 − 1의 크기로 방향을 정합니다.",
    "- 중기·단기 방향 차이: 표시 기간 중 보합이 아닌 가장 짧은 기간과 가장 긴 기간의 방향이 다를 때입니다. "
    "일간 지표에만 적용하고 120세션 참고 합은 쓰지 않습니다.",
    "- 10년 백분위: 지난 10년 관측 중 현재 값 이하인 비율입니다. 외국인 순매수는 20세션 합, 거래대금은 "
    "비율, 월간은 표의 값 기준입니다. 이력이 10년에 못 미치면 —로 비웁니다.",
    '- 월간 계산은 행 순번이 아니라 날짜로 맞춥니다. 값이 빈 달은 건너뛰고 "n개월 연속"은 거기서 끊습니다.',
    "- 갱신 상태(월간): 그 series의 지난 3년 발표 지연 중앙값(관측월 1일부터 첫 발표까지 일수)으로 리포트 "
    "날짜까지 나왔어야 할 마지막 관측월을 구하고, 최근 관측월이 그 이상이면 정상, 모자라면 지연 n개월입니다. "
    "일간은 기준일 지연이 3평일을 넘으면 갱신 지연입니다.",
    "- 발표일: 미국은 그 관측월 값의 첫 vintage `realtime_start`, 한국은 `available_from_date`입니다.",
)


# ------------------------------------------------------------------ SVG
COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#6b7280")
FONT = "Apple SD Gothic Neo, Noto Sans CJK KR, Malgun Gothic, sans-serif"
W, ML, MR, PH, MB = 820, 62, 236, 180, 30
LABEL_GAP = 27

#: ``fmt`` 템플릿은 ``{v:<서식>}`` 하나만 허용한다(속성 접근 등 다른 필드는 거부).
_FMT_RE = re.compile(r"^[^{}]*\{v:[+,]*(?:\.\d+)?f\}[^{}]*$")


def _fmt(template: str, v: float) -> str:
    if not _FMT_RE.match(template):
        raise ValueError(f"unsupported figure fmt: {template!r}")
    return template.format(v=v).replace("-", MINUS)


def _esc(s: str) -> str:
    return escape(s, {'"': "&quot;"})


def _nice_ticks(lo: float, hi: float, n: int = 5) -> tuple[list[float], int]:
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = mag * 10
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    t = math.ceil(lo / step - 1e-9) * step
    ticks = []
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    dec = max(0, math.ceil(-math.log10(step) - 1e-9))
    return ticks, dec


def _month_index(iso: str) -> int:
    return int(iso[:4]) * 12 + int(iso[5:7]) - 1


def _x(iso: str, monthly: bool) -> int:
    return _month_index(iso) if monthly else dt.date.fromisoformat(iso).toordinal()


def _svgs(body: dict) -> dict[str, str]:
    return {f"macro-{f['name']}.svg": _chart(f) for f in body["figures"]}


def _chart(fig: dict) -> str:  # noqa: C901 - 시제품 `svg_chart`를 그대로 옮긴 것이라 한 함수로 둔다
    title = fig["title"]
    monthly = fig["kind"] == "monthly"
    lines = fig["lines"]
    ref = fig.get("ref")
    legend = len(lines) >= 3
    PT = 64 if legend else 44
    H = PT + PH + MB
    pw = W - ML - MR
    pts_all = [[(_x(d, monthly), v) for d, v in ln["points"]] for ln in lines]
    xmax = max(x for pts in pts_all for x, _ in pts)
    x0 = xmax - 59 if monthly else xmax - 365
    win = [[(x, y) for x, y in pts if x >= x0] for pts in pts_all]
    yv = [y for pts in win for _, y in pts]
    lo, hi = min(yv), max(yv)
    if ref is not None:
        lo, hi = min(lo, ref), max(hi, ref)
    if hi == lo:
        hi = lo + 1
    pad = (hi - lo) * 0.08
    lo -= pad
    hi += pad

    def X(v: float) -> float:
        return ML + (v - x0) / (xmax - x0) * pw

    def Y(v: float) -> float:
        return PT + PH - (v - lo) / (hi - lo) * PH

    P = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'role="img" aria-label="{_esc(title)}">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="12" y="20" font-size="14" font-weight="bold" fill="#111827" '
        f'font-family="{FONT}">{escape(title)}</text>',
    ]
    if legend:
        lx = 12
        for i, ln in enumerate(lines):
            col = COLORS[i % len(COLORS)]
            P.append(
                f'<rect x="{lx}" y="33" width="14" height="3" fill="{col}"/>'
                f'<text x="{lx + 18}" y="38" font-size="10" fill="#374151" '
                f'font-family="{FONT}">{escape(ln["label"])}</text>'
            )
            lx += 18 + 11 * len(ln["label"]) + 14
    if not monthly and fig.get("shade_from"):
        s3 = max(_x(fig["shade_from"], False), x0)
        P.append(
            f'<rect x="{X(s3):.1f}" y="{PT}" width="{X(xmax) - X(s3):.1f}" height="{PH}" fill="#eef2f7"/>'
        )
        P.append(
            f'<text x="{X(s3) + 4:.1f}" y="{PT + 11}" font-size="9" fill="#6b7280" '
            f'font-family="{FONT}">최근 3개월</text>'
        )
    ticks, dec = _nice_ticks(lo, hi)
    for t in ticks:
        yy = Y(t)
        P.append(
            f'<line x1="{ML}" y1="{yy:.1f}" x2="{ML + pw}" y2="{yy:.1f}" stroke="#e5e7eb" stroke-width="1"/>'
        )
        P.append(
            f'<text x="{ML - 6}" y="{yy + 3:.1f}" font-size="10" text-anchor="end" fill="#6b7280" '
            f'font-family="{FONT}">{f"{t:,.{dec}f}".replace("-", MINUS)}</text>'
        )
    if monthly:
        for k in range(int(x0), int(xmax) + 1):
            if k % 12 == 0:
                xx = X(k)
                P.append(
                    f'<line x1="{xx:.1f}" y1="{PT}" x2="{xx:.1f}" y2="{PT + PH}" '
                    f'stroke="#f1f5f9" stroke-width="1"/>'
                )
                P.append(
                    f'<text x="{xx:.1f}" y="{PT + PH + 15}" font-size="10" text-anchor="middle" '
                    f'fill="#6b7280" font-family="{FONT}">{k // 12}</text>'
                )
    else:
        # 달의 첫 관측일마다 눈금. 창 시작이 든 달은 그 달 첫 관측일이 창 밖이므로(1일이 창 시작일일 때만 예외) 뺀다.
        x0_date = dt.date.fromordinal(x0)
        firsts: dict[tuple[int, int], int] = {}
        for pts in win:
            for x, _ in pts:
                d = dt.date.fromordinal(x)
                key = (d.year, d.month)
                if key not in firsts or x < firsts[key]:
                    firsts[key] = x
        for (yy_, mm_), x in sorted(firsts.items()):
            if x < x0 + 2:
                continue
            if (yy_, mm_) == (x0_date.year, x0_date.month) and x0_date.day != 1:
                continue
            xx = X(x)
            lab = f"{yy_}-01" if mm_ == 1 else f"{mm_}월"
            P.append(
                f'<line x1="{xx:.1f}" y1="{PT}" x2="{xx:.1f}" y2="{PT + PH}" '
                f'stroke="#f1f5f9" stroke-width="1"/>'
            )
            P.append(
                f'<text x="{xx:.1f}" y="{PT + PH + 15}" font-size="9.5" text-anchor="middle" '
                f'fill="#6b7280" font-family="{FONT}">{lab}</text>'
            )
    if ref is not None:
        P.append(
            f'<line x1="{ML}" y1="{Y(ref):.1f}" x2="{ML + pw}" y2="{Y(ref):.1f}" stroke="#6b7280" '
            f'stroke-width="1" stroke-dasharray="4 3"/>'
        )
        P.append(
            f'<text x="{ML + 4}" y="{Y(ref) - 3:.1f}" font-size="9" fill="#6b7280" '
            f'font-family="{FONT}">기준 {ref:g}</text>'
        )
    labels = []
    for i, (ln, pts_all_i) in enumerate(zip(lines, pts_all, strict=True)):
        col = COLORS[i % len(COLORS)]
        # 끊는 구간은 body의 segments(점 번호 [시작, 끝))를 따른다. 창 안 점만 그린다.
        segs = []
        for s, e in ln["segments"]:
            sg = [(x, y) for x, y in pts_all_i[s:e] if x >= x0]
            if sg:
                segs.append(sg)
        for sg in segs:
            if len(sg) == 1:
                P.append(
                    f'<circle cx="{X(sg[0][0]):.1f}" cy="{Y(sg[0][1]):.1f}" r="2" fill="{col}"/>'
                )
            else:
                pts_s = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in sg)
                P.append(
                    f'<polyline points="{pts_s}" fill="none" stroke="{col}" stroke-width="1.6" '
                    f'stroke-linejoin="round"/>'
                )
        ex, ey = segs[-1][-1]
        P.append(f'<circle cx="{X(ex):.1f}" cy="{Y(ey):.1f}" r="2.6" fill="{col}"/>')
        if monthly:
            dtxt = f"{int(ex) // 12:04d}-{int(ex) % 12 + 1:02d}"
        else:
            dd = dt.date.fromordinal(int(ex))
            dtxt = f"{dd.month:02d}/{dd.day:02d}"
        labels.append(
            dict(
                col=col, name=ln["label"], txt=f"{_fmt(ln['fmt'], ey)} · {dtxt}", ex=X(ex), ey=Y(ey)
            )
        )
    order = sorted(labels, key=lambda r: r["ey"])
    pos = [r["ey"] for r in order]
    for j in range(1, len(pos)):
        pos[j] = max(pos[j], pos[j - 1] + LABEL_GAP)
    over = pos[-1] - (PT + PH - 6)
    if over > 0:
        pos = [p - over for p in pos]
        for j in range(len(pos) - 2, -1, -1):
            pos[j] = min(pos[j], pos[j + 1] - LABEL_GAP)
    lx = ML + pw + 12
    for r, ly in zip(order, pos, strict=True):
        P.append(
            f'<line x1="{r["ex"]:.1f}" y1="{r["ey"]:.1f}" x2="{lx - 3}" y2="{ly:.1f}" '
            f'stroke="{r["col"]}" stroke-width="0.8" stroke-dasharray="2 2"/>'
        )
        P.append(
            f'<text x="{lx}" y="{ly - 1:.1f}" font-size="11" font-weight="bold" fill="{r["col"]}" '
            f'font-family="{FONT}">{escape(r["name"])}</text>'
        )
        P.append(
            f'<text x="{lx}" y="{ly + 11:.1f}" font-size="10" fill="#374151" '
            f'font-family="{FONT}">{escape(r["txt"])}</text>'
        )
    P.append("</svg>")
    return "\n".join(P) + "\n"
