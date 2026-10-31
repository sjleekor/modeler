"""거시 흐름 문서 ``briefing-macro.v1`` 만들기. 같은 selection·인자는 같은 바이트다(시각·난수 없음)."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np

from modeler.scores.macro_context.calc import Computed, compute
from modeler.scores.macro_context.figures import make_figures
from modeler.scores.macro_context.load import MacroData, load_macro_data, view
from modeler.scores.macro_context.spec import (
    DAILY,
    DISP,
    FLOW_PLABEL,
    MONTHLY,
    RULES,
    SCHEMA,
    SHORT,
    SPLICE_MAP,
    STALE_DAILY_SESSIONS,
    SUMMARY_MAX,
    SUMMARY_MIN,
    SUMMARY_PER_GROUP,
    US_SOURCE_IDS,
    VS_PREV_MAX_COUNT,
)
from modeler.scores.macro_context.summary import build_summary, vs_prev


def jsonable(o: Any) -> Any:
    """numpy·날짜·튜플을 JSON 값으로. NaN·inf는 ``None``."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (dt.date, dt.datetime)):
        return o.isoformat()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return None if f != f or f in (float("inf"), float("-inf")) else f
    return o


def compute_at(data: MacroData, d: dt.date, *, us_before: bool = False) -> Computed:
    """입력의 ``d`` 가용분으로 지표 전체를 계산한다.

    ``us_before``이면 미국 행은 ``realtime_start < d``만 쓴다. 지난 보고서 대비로 앞 날짜 d를 다시
    계산할 때다: 미국 날짜 d에 나온 vintage는 d 09:30 KST 보고서 때 아직 없었다.
    """
    kr, kr_first = view(data.kr_rows, d)
    us, us_first = view(data.us_rows, d, before=us_before)
    return compute(d, kr, us, kr_first, us_first)


def input_checks(data: MacroData, res: Computed) -> dict[str, Any]:
    """입력 기준일 검사. 쓴 값의 가용일·관측일이 D 이하이고, 모든 지표 기준일이 D 이하인지."""
    D = res.D
    kr, _ = view(data.kr_rows, D)
    us, _ = view(data.us_rows, D)
    kr_afs = [af for sid in kr for (_v, af) in kr[sid].values()]
    kr_obs = [od for sid in kr for od in kr[sid]]
    us_rs = [rs for sid in us for (_v, rs) in us[sid].values()]
    us_obs = [d for sid in us for d in us[sid]]
    snap_us = dt.date.fromisoformat(data.us_snapshot)
    viol: list[str] = []
    for o in res.inds:
        if dt.date.fromisoformat(o["asof"]) > D:
            viol.append(o["id"])
    for o in res.mons:
        y, m = o["asof"].split("-")
        if (int(y), int(m)) > (D.year, D.month):
            viol.append(o["id"])
    c: dict[str, Any] = {
        "kr_max_available_from_used": max(kr_afs).isoformat(),
        "kr_max_observation_used": max(kr_obs).isoformat(),
        "kr_rows_excluded_by_available_from_gt_D": int(data.kr_excluded),
        "us_max_realtime_start_used": max(us_rs).isoformat(),
        "us_max_observation_used": max(us_obs).isoformat(),
        "all_available_from_le_D": bool(max(kr_afs) <= D),
        "all_realtime_start_le_D": bool(max(us_rs) <= D),
        "all_realtime_start_le_us_snapshot": bool(max(us_rs) <= snap_us),
        "kr_max_observation_le_D": bool(max(kr_obs) <= D),
        "us_max_observation_le_D": bool(max(us_obs) <= D),
        "all_indicator_asof_le_D": not viol,
        "violations": viol,
        # 지난 보고서 대비로 다시 계산하는 앞 날짜 d의 미국 행 규칙(통과 판정에는 안 씀)
        "vs_prev_us_rule": "realtime_start < d",
    }
    c["overall_pass"] = bool(
        all(v for k, v in c.items() if k.startswith("all_") or k.endswith("_le_D"))
    )
    return c


def period_table() -> list[dict[str, Any]]:
    """부록 기간표 요약(지표마다 표시 기간·변화)."""
    rows: list[dict[str, Any]] = []
    for s in DAILY:
        if s["kind"] == "ratio":
            per = ["20세션 평균 ÷ 250세션 평균 (1년 평균 대비)"]
        elif s["kind"] == "sum":
            per = [FLOW_PLABEL[p] for p in s["per"]]
        else:
            per = list(s["per"])
        row = dict(id=s["id"], name=SHORT[s["id"]], kind=s["kind"], periods=per)
        if s.get("ref_per"):
            row["reference_period"] = FLOW_PLABEL[s["ref_per"]]
        rows.append(row)
    for s in MONTHLY:
        rows.append(
            dict(id=s["id"], name=SHORT[s["id"]], kind="monthly", periods=[m[0] for m in s["m"]])
        )
    return rows


def rules_block() -> dict[str, Any]:
    return {
        **RULES,
        "stale_daily_sessions": STALE_DAILY_SESSIONS,
        "stale_monthly": "지난 3년 발표 지연 중앙값으로 구한 나왔어야 할 마지막 관측월보다 모자라면 지연",
        "summary": {
            "min_rows": SUMMARY_MIN,
            "max_rows": SUMMARY_MAX,
            "max_per_group": SUMMARY_PER_GROUP,
            "priority": [
                "표시 기간 중 급",
                "중기·단기 방향 차이",
                "10년 백분위 P5 이하·P95 이상",
                "월간 3개월 이상 연속",
            ],
            "tie_break": "크기 순, 같으면 지표 정의 순서",
        },
        "vs_prev": {
            "max_count": VS_PREV_MAX_COUNT,
            "method": "직전 평일을 같은 입력의 그날 가용분으로 다시 계산",
            "availability": "다시 계산하는 날 d에서 한국은 available_from_date <= d, 미국은 "
            "realtime_start < d(미국 날짜 d에 나온 자료는 d 09:30 KST 보고서 뒤에 나옴). "
            "이번 보고서 D는 realtime_start <= D 그대로",
        },
        "splice": {
            "map": SPLICE_MAP,
            "rule": "macro_series 마지막 날 다음 날부터 한국 표 사본으로 잇는다. 표에 series가 없으면 macro_series만",
            "rounded": {"global_vix": 2},
        },
        "display_policy": {
            "credit_baa": "수준 표시 금지(R1) — 변화·백분위·1년 창 시작 대비 변화만",
            "trading_value": "비율만 표시(R2) — 평균 조원 값은 body에 없음",
        },
        "periods": period_table(),
    }


def build_document(
    selection: dict[str, Any], *, unit: str, invocation_id: str, release: str, selection_sha256: str
) -> dict[str, Any]:
    """selection이 고정한 입력만 열어 거시 흐름 문서를 만든다. 실패하면 예외를 올린다."""
    D = dt.date.fromisoformat(unit)
    data = load_macro_data(selection, D)
    res = compute_at(data, D)
    for o in res.inds + res.mons:
        o["short_name"] = SHORT[o["id"]]
        o["display_name"] = DISP[o["id"]]
    summary = build_summary(res.inds, res.mons)
    vsp = vs_prev(D, res, lambda d: compute_at(data, d, us_before=True))
    stale_d = [o["id"] for o in res.inds if o["stale"]]
    header = dict(
        daily_asof_min=min(o["asof"] for o in res.inds),
        daily_asof_max=max(o["asof"] for o in res.inds),
        stale_daily=stale_d,
        stale_daily_us_weekly=[i for i in stale_d if i in US_SOURCE_IDS],
        stale_monthly=[o["id"] for o in res.mons if o["stale"]],
        stale_daily_threshold_sessions=STALE_DAILY_SESSIONS,
    )
    body = {
        "schema": SCHEMA,
        "state": "ok",
        "unit": unit,
        "invocation_id": invocation_id,
        "release": release,
        "selection_sha256": selection_sha256,
        "inputs": data.inputs,
        "header": header,
        "indicators": res.inds + res.mons,
        "summary": summary,
        "vs_prev": vsp,
        "figures": make_figures(res),
        "checks": input_checks(data, res),
        "rules": rules_block(),
    }
    return jsonable(body)
