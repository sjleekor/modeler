"""summary 선정 규칙(영역 1행 먼저·영역당 2·4~6)과 vs_prev(연속 상한·새로 나타남·해제)."""

from __future__ import annotations

import datetime as dt

from modeler.scores.macro_context import summary as S
from modeler.scores.macro_context.calc import Computed
from modeler.scores.macro_context.spec import DAILY_BY_ID, MONTHLY_BY_ID, VS_PREV_MAX_COUNT


def a_cond(label="1개월", dir_="down", dlabel="급락", change=-1.0, unit="%", mult=1.5, h=20):
    return dict(
        label=label, dir=dir_, dlabel=dlabel, change=change, unit=unit, dec=2, mult=mult, h=h
    )


def ind(iid, *, a=(), b=None, c=None, d=None, asof="2026-10-27", stale=False):
    spec = DAILY_BY_ID.get(iid) or MONTHLY_BY_ID[iid]
    kind = spec["kind"] if iid in DAILY_BY_ID else "monthly"
    return dict(
        id=iid,
        group=spec["group"] if iid in DAILY_BY_ID else "macro",
        kind=kind,
        asof=asof,
        stale=stale,
        conditions=dict(a=list(a), b=b, c=c, d=d),
        periods=[dict(label="1개월", h=20, change=-1.0), dict(label="3개월", h=60, change=2.0)],
        display=dict(level="show"),
        value=dict(level=1.0, display="1.00%"),
    )


def cond_b(mag=2.0):
    return dict(
        short="1개월",
        short_dir="down",
        long="3개월",
        long_dir="up",
        short_change=-1.0,
        long_change=2.0,
        unit="%",
        mag=mag,
    )


def ids(summary):
    return [r["id"] for r in summary["rows"]]


def test_each_group_gets_first_row_before_second_rows_and_caps_at_two_per_group():
    # 같은 영역(원자재·한국 증시 자금)에 강한 후보가 몰려 있어도 다른 영역의 대표가 먼저 들어간다
    inds = [
        ind("fn_kospi", a=[a_cond("1주", mult=9.0, unit="조원", h=5, dlabel="순매도")]),
        ind("fn_kosdaq", a=[a_cond("1개월", mult=8.0, unit="조원", dlabel="순매도")]),
        ind("tv_kospi", a=[a_cond("1년 평균 대비", mult=7.0, dlabel="하락")]),
        ind("wti", a=[a_cond(mult=6.0, dlabel="급등", dir_="up")]),
        ind("brent", a=[a_cond(mult=5.0, dlabel="급등", dir_="up")]),
        ind("usdkrw", b=cond_b(3.0)),
        ind("vix", c=dict(p=97.0)),
    ]
    s = S.build_summary(
        inds,
        [
            ind(
                "us_m2",
                d=dict(
                    months=7, text="7개월 연속 상승", measure="3개월 연율", change=1.0, unit="%"
                ),
            )
        ],
    )
    got = ids(s)
    assert 4 <= len(got) <= 6 and len(got) == 6
    groups = [r["group"] for r in s["rows"]]
    assert {"flow", "commod", "fx", "risk", "macro"} <= set(groups)  # 영역 대표가 먼저
    assert all(groups.count(g) <= 2 for g in set(groups))
    # 영역 대표 5행(외국인 KOSPI·WTI·원/달러·VIX·미 M2) 뒤 남은 한 자리는 우선순위 순(외국인 KOSDAQ)
    assert got == ["fn_kospi", "fn_kosdaq", "wti", "usdkrw", "vix", "us_m2"]
    assert s["candidates"] == 8 and s["not_selected"] == 2
    assert [r["rank"] for r in s["rows"]] == [1, 2, 3, 4, 5, 6]


def test_group_representative_is_highest_priority_in_group_and_second_slot_by_priority():
    inds = [
        ind("kr_cd91", c=dict(p=5.0)),  # 우선순위 2
        ind("kr_gov3y", a=[a_cond(mult=1.2)]),  # 우선순위 0 -> 금리 대표
        ind("kr_gov10y", b=cond_b(4.0)),  # 우선순위 1
        ind("usdkrw", a=[a_cond("3개월", mult=2.0, dlabel="급락", h=60)]),
        ind("vix", c=dict(p=99.0)),
    ]
    s = S.build_summary(inds, [])
    rows = {r["id"]: r for r in s["rows"]}
    assert rows["kr_gov3y"]["priority"] == 0 and rows["kr_gov10y"]["priority"] == 1
    # 금리 영역에 3후보라도 최대 2행: 대표(gov3y) + 다음 우선순위(gov10y), cd91은 영역 상한에 걸려 빠진다
    assert "kr_cd91" not in rows and len(rows) == 4
    assert [r["priority"] for r in s["rows"] if r["group"] == "rates"] == [0, 1]


def test_minimum_four_rows_fill_even_over_group_cap():
    inds = [
        ind("kr_cd91", c=dict(p=5.0)),
        ind("kr_gov3y", c=dict(p=96.0)),
        ind("kr_gov10y", c=dict(p=97.0)),
        ind("us_2y", c=dict(p=98.0)),
    ]
    s = S.build_summary(inds, [])
    assert len(s["rows"]) == 4 and all(
        r["group"] == "rates" for r in s["rows"]
    )  # 4행 하한이 영역 상한보다 앞선다
    assert S.build_summary([], [])["rows"] == [] and S.build_summary([], [])["candidates"] == 0


def test_text_templates_use_facts_only():
    o = ind(
        "usdkrw",
        a=[a_cond("1개월", mult=2.0, h=20), a_cond("3개월", mult=1.0, h=60, change=-9.42)],
        b=cond_b(),
        c=dict(p=99.0),
    )
    assert S.obs_text(o) == "1개월·3개월 급락 · 중기 ↑ / 단기 ↓ · 10년 백분위 P99"
    # 방향 차이가 있으면 그 두 기간은 b의 변화폭을 쓴다(3개월 +2.00%)
    assert S.rep_text(o) == "1개월 −1.00% · 3개월 +2.00%"
    only_a = ind(
        "usdkrw", a=[a_cond("1개월", mult=2.0, h=20), a_cond("3개월", mult=1.0, h=60, change=-9.42)]
    )
    assert S.rep_text(only_a) == "1개월 −1.00% · 3개월 −9.42%"
    flow = ind("fn_kospi", a=[a_cond("1주", unit="조원", change=-13.57, dlabel="순매도", h=5)])
    assert S.obs_text(flow) == "5세션 순매도 (큰 쪽 10%)" and S.rep_text(flow) == "5세션 −13.57조원"
    cr = ind("credit_baa", c=dict(p=97.0))
    cr["display"] = dict(level="hide")
    assert S.level_text(cr) is None


# --------------------------------------------------------------------------- vs_prev
def comp(*inds):
    return Computed(dt.date(2026, 10, 28), list(inds), [])


def test_prev_weekday_skips_weekend():
    assert S.prev_weekday(dt.date(2026, 10, 28)) == dt.date(2026, 10, 27)
    assert S.prev_weekday(dt.date(2026, 10, 26)) == dt.date(2026, 10, 23)  # 월 -> 금
    assert S.prev_weekday(dt.date(2026, 10, 25)) == dt.date(2026, 10, 23)  # 일 -> 금


def test_vs_prev_new_direction_diff_continued_resolved_and_cap():
    D = dt.date(2026, 10, 28)
    calls = []
    steady = ind("usdkrw", a=[a_cond("3개월", mult=2.0, h=60)])
    today = comp(
        steady,
        ind("brent", a=[a_cond("1주", mult=2.0, dlabel="급등", dir_="up", h=5)]),
        ind("us_2y", b=cond_b()),
    )

    def compute_at(d):
        calls.append(d)
        past = [ind("usdkrw", a=[a_cond("3개월", mult=2.0, h=60)])]  # 전 기간 이어짐
        if d == dt.date(2026, 10, 27):
            past.append(
                ind("wti", a=[a_cond("1주", mult=2.0, dlabel="급등", dir_="up", h=5)])
            )  # 어제만 있었다
        return comp(*past)

    out = S.vs_prev(D, today, compute_at)
    assert out["previous_report_date"] == "2026-10-27"
    assert [e["id"] for e in out["new"]] == ["brent"]
    assert [e["id"] for e in out["direction_diff_new"]] == ["us_2y"]
    assert [e["id"] for e in out["resolved"]] == ["wti"]
    cont = out["continued"]
    assert [e["id"] for e in cont] == ["usdkrw"]
    assert cont[0]["streak"] == VS_PREV_MAX_COUNT  # 상한 20
    assert out["max_count"] == 20 and out["recomputed_dates"] == len(calls)
    assert len(calls) == len(set(calls)) == 19  # 날짜마다 한 번만 다시 계산, 20회에 닿으면 멈춤
    assert max(calls) == dt.date(2026, 10, 27) and all(c.weekday() < 5 for c in calls)


def test_vs_prev_streak_counts_consecutive_days_and_stops_at_first_gap():
    D = dt.date(2026, 10, 28)  # 수
    today = comp(ind("usdkrw", a=[a_cond("3개월", mult=2.0, h=60)]))
    have = {dt.date(2026, 10, 27), dt.date(2026, 10, 26), dt.date(2026, 10, 23)}  # 10-22는 없음

    def compute_at(d):
        return comp(*([ind("usdkrw", a=[a_cond("3개월", mult=2.0, h=60)])] if d in have else []))

    out = S.vs_prev(D, today, compute_at)
    # 오늘(28) + 27 + 26 + 23(주말 24·25는 건너뜀) = 4회. 22일에 없어서 거기서 끊긴다
    assert out["continued"][0]["streak"] == 4
