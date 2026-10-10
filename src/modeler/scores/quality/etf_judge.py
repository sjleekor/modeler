"""ETF 품질 점수 부분 E 판정 결합 — Holm 결합·게이트·등급·기록용 결과·manifest
(사전등록 20261010_quality_score §6·§9·§10·§11.3·§11.4·§12.2, 구현 해석 표 v1 I02·I09·I13·I18·I25·I26).

E1(``etf_e1``)과 E2(``etf_e2``) 통계 모듈을 묶는다. 판정 가설은 둘(E 가족, Holm m = 2, 단측)이다.

- H_E1: E1 국내형, 선행 6개월. p = ``etf_e1.p_value``(귀무 적중률 0.30).
- H_E2: E2 괴리, lenient, 그룹 크기 5 이상. p = ``etf_e2.p_value``(ρ ≤ 0).

**정정 E-1 재분류(2026-10-10 16:01 사용자).** 판정 변경 부분(E1 국내형을 판정에서 빼고 E2 하나로 판정)은
철회했다. 판정 규칙을 바꾸는 항목은 정정 블록으로 못 하고 다음 사전등록으로 넘긴다(02_rules A4·00_plan P3).
그래서 동결 규칙대로 판정한다. 룩어헤드 기록과 'E1 국내형 순자산 단독 전체 풀'(``records.e1_domestic_netasst_full``)은
결과 전 기록용(P3 범위)이다. E1 국내형은 기초지수 종가 룩어헤드가 있는 입력으로 계산된 판정이다(02 문서 §5).

    STOCK_DATA_ROOT=../stock_data PYTHONPATH=src python -m modeler.scores.quality.etf_judge --period dev

경로는 환경변수 ``STOCK_DATA_ROOT`` 로 받는다(기본 ``../stock_data``). 개발 구간 산출물은
``kr/output/quality_score_etf_dev_20261010/judge/``, 판정 구간은
``kr/output/quality_score_etf_judgment_20261010_frozen/`` 이다.

**판정 구간 보호.** ``run("judgment")`` 는 환경변수 ``QUALITY_E_JUDGMENT_CONFIRMED`` 가 비어 있지 않을 때만
돈다. 그 값은 manifest에 그대로 적는다.

**Holm 기각 판정(구현 선택).** 사전등록 §11.4 G2는 "Holm 단측 하한"으로 기각을 정의한다. 그래서 단계 α의
부트스트랩 분위수가 문턱(E1 ≥ 0.30, E2 > 0)을 넘으면 그 단계에서 기각한다. ``p < α`` 판정은 참고로 같이 적는다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import unicodedata
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from modeler.scores.quality import etf_e1 as e1
from modeler.scores.quality import etf_e2 as e2
from modeler.scores.quality import etf_panel as ep

JUDGE_VERSION = "quality-score-v0/etf_judge/3-frozen"
CORRECTION_ID = "E-1R"
JUDGMENT_FAMILY = ("H_E1", "H_E2")  # 동결 규칙: 판정 가설 둘, Holm m = 2
CORRECTION_NOTE = (
    "정정 E-1 재분류(10-10 16:01 사용자): 판정 변경 부분 철회, 동결 규칙으로 판정. "
    "룩어헤드 기록과 'E1 국내형 순자산 단독 전체 풀'은 결과 전 기록용(P3)"
)
LOOKAHEAD_NOTE = "룩어헤드 주의: E1 국내형은 기초지수 종가 룩어헤드가 있는 입력으로 계산된 판정이다(02 문서 §5 참조)"
JUDGMENT_ENV = e1.JUDGMENT_ENV

DEFAULT_OUT_REL_DEV = "kr/output/quality_score_etf_dev_20261010/judge"
DEFAULT_OUT_REL_JUDGMENT = "kr/output/quality_score_etf_judgment_20261010_frozen"
DEFAULT_KIND_REL = "kr/output/quality_score_etf_inputs_20261010/kind/kind_etf_delisting.csv"
DEFAULT_KIND_WINDOWS_REL = "kr/output/quality_score_etf_inputs_20261010/kind/kind_query_windows.csv"
DEFAULT_INTERP_TABLE = ep.INTERP_TABLE_REL  # 상대 경로. 실제 경로는 ep.default_interp_table()
INTERP_TABLE_SHA256 = "29c89969b8f5d467f2c998110eb8de00afb237de240fe2aad76abc7d07d728b1"
DEFAULT_PREREG = ep.PREREG_REL  # 상대 경로. 실제 경로는 ep.default_prereg()
UV_LOCK = str(ep.repo_root() / "uv.lock")

# ---- 처음 고른 숫자(§11.4)와 해석 표 값
HOLM_ALPHAS = (0.025, 0.05)  # I18: p가 작은 가설부터 단계 α
E1_G1_MIN = 30  # 점수 있는 사건 30 미만 표본 부족
E1_G1_FULL = 50  # 30~49 탐색 판정, 50 이상 판정
E1_G2_RATE = 0.50  # 적중률 점추정 ≥ 50%
E1_G2_LOWER = 0.30  # Holm 하한 ≥ 30%
E1_G3_RATE = 0.30  # 전반·후반 점추정 ≥ 30%
E2_G1_MIN = 150  # 풀링 ETF-연 150 미만 표본 부족
E2_G2_RHO = 0.30  # 가중평균 ρ ≥ 0.30
E2_G2_LOWER = 0.0  # Holm 하한 > 0
E2_G3_SHARE = (2, 3)  # 연도별 ρ > 0인 해가 2/3 이상(I16)
UNSCORED_CAP = e1.UNSCORED_SHARE_CAP  # 점수 없는 사건 30% 초과면 등급 C 위로 안 올림
STOP_TEXT = "중단 조건 충족 — 사용자 확인 대기"  # §10

# KIND 일치(기록용). 조회 창 공백과 그 앞뒤 30일은 "창 밖".
KIND_GAP = (date(2025, 1, 1), date(2025, 10, 8))
KIND_GAP_MARGIN_DAYS = 30

H_E1 = "H_E1"
H_E2 = "H_E2"


# ---------------------------------------------------------------- 보호
def _guard(period: str) -> None:
    if period not in ("dev", "judgment"):
        raise ValueError(f"period must be 'dev' or 'judgment', got {period!r}")
    if period == "judgment" and not os.environ.get(JUDGMENT_ENV, "").strip():
        raise PermissionError(
            f"판정 구간은 환경변수 {JUDGMENT_ENV} 가 비어 있지 않을 때만 돌 수 있다(사전등록 §12.4)."
        )


# ---------------------------------------------------------------- Holm 결합(I18)
def holm_combine(items: list[dict], alphas: tuple[float, ...] = HOLM_ALPHAS) -> dict[str, dict]:
    """Holm 단계 결합. ``items`` 원소: ``name``, ``p``, ``bound_fn(alpha) -> float``, ``bound_ok(bound) -> bool``.

    p가 작은 가설부터(동률이면 목록 순서) 단계 α를 ``alphas`` 순서로 준다. 단계 k의 가설은 앞 단계를 모두
    기각했을 때만 닿는다. 닿으면 하한 = 그 단계 α의 분위수이고 ``bound_ok`` 를 넘으면 기각이다. 못 닿으면
    하한은 첫 단계 α로 적고 ``status`` 를 "Holm 앞 단계 탈락"으로 둔다(기각 안 됨).
    ``p_rule_reject`` 는 참고용 ``p < 단계 α`` 판정이다.
    """
    order = sorted(range(len(items)), key=lambda i: (items[i]["p"], i))
    out: dict[str, dict] = {}
    all_rejected_so_far = True
    for rank, i in enumerate(order):
        it = items[i]
        stage_alpha = alphas[rank] if rank < len(alphas) else alphas[-1]
        if all_rejected_so_far:
            bound = it["bound_fn"](stage_alpha)
            ok = bool(it["bound_ok"](bound)) if bound is not None and not math.isnan(bound) else False
            out[it["name"]] = {
                "p": it["p"],
                "stage": rank + 1,
                "stage_alpha": stage_alpha,
                "reached": True,
                "holm_lower_bound": bound,
                "bound_alpha": stage_alpha,
                "rejected": ok,
                "status": f"단계 {rank + 1} 기각" if ok else f"단계 {rank + 1} 기각 못 함",
                "p_rule_reject": bool(it["p"] < stage_alpha),
            }
            all_rejected_so_far = ok
        else:
            a0 = alphas[0]
            out[it["name"]] = {
                "p": it["p"],
                "stage": rank + 1,
                "stage_alpha": stage_alpha,
                "reached": False,
                "holm_lower_bound": it["bound_fn"](a0),
                "bound_alpha": a0,
                "rejected": False,
                "status": "Holm 앞 단계 탈락",
                "p_rule_reject": False,
            }
    return out


# ---------------------------------------------------------------- 게이트·등급(§11.4)
def assign_grade(g1: bool, g2: bool, g3: bool, unscored_cap_exceeded: bool = False) -> str:
    """D = G1 또는 G2 탈락 · C = G2 통과 G3 탈락 · A = 전부 통과. 점수 없는 사건 > 30%면 C 위로 안 올린다(A → C)."""
    if not (g1 and g2):
        return "D"
    if not g3:
        return "C"
    return "C" if unscored_cap_exceeded else "A"


def _ge(x: float | None, thr: float) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and x >= thr


def e1_gates(stats: dict, holm_res: dict) -> dict:
    """E1 국내형 게이트와 등급. ``stats`` 는 ``etf_e1.outcome_stats`` 의 유형별 dict(``n_scored``, ``hit_rate``,
    ``unscored_share``, ``halves``), ``holm_res`` 는 ``holm_combine`` 의 H_E1 항목."""
    n = int(stats["n_scored"])
    if n < E1_G1_MIN:
        g1_status = "표본 부족"
    elif n < E1_G1_FULL:
        g1_status = "탐색 판정"
    else:
        g1_status = "판정"
    g1 = n >= E1_G1_MIN
    hr = stats.get("hit_rate")
    point_ok = _ge(hr, E1_G2_RATE)
    g2 = bool(point_ok and holm_res["rejected"])
    h1 = stats["halves"]["first"]["hit_rate"]
    h2 = stats["halves"]["second"]["hit_rate"]
    g3 = bool(_ge(h1, E1_G3_RATE) and _ge(h2, E1_G3_RATE))
    cap = bool((stats.get("unscored_share") or 0.0) > UNSCORED_CAP)
    rule = assign_grade(g1, g2, g3, cap)
    return {
        "g1": {"n_scored": n, "status": g1_status, "pass": g1},
        "g2": {
            "hit_rate": hr,
            "point_threshold": E1_G2_RATE,
            "point_ok": point_ok,
            "holm_lower_bound": holm_res["holm_lower_bound"],
            "holm_lower_threshold": E1_G2_LOWER,
            "holm_status": holm_res["status"],
            "holm_rejected": holm_res["rejected"],
            "pass": g2,
        },
        "g3": {"first_half_hit_rate": h1, "second_half_hit_rate": h2, "threshold": E1_G3_RATE, "pass": g3},
        "unscored_share": stats.get("unscored_share"),
        "unscored_cap": UNSCORED_CAP,
        "unscored_cap_exceeded": cap,
        "grade_rule": rule,
        # 표본 부족이면 등급 없음, 목적을 "감쇠 폭 추정"으로 낮춘다(§6). 규칙을 그대로 적용한 결과는 grade_rule.
        "grade": None if g1_status == "표본 부족" else rule,
        "purpose": "감쇠 폭 추정" if g1_status == "표본 부족" else "판정",
    }


def e2_gates(stats: dict, holm_res: dict) -> dict:
    """E2 괴리 게이트와 등급. ``stats`` 는 ``etf_e2.e2_stats`` 결과의 ``pooled_etf_years``, ``weighted_rho``,
    ``n_years``, ``n_pos_years``."""
    n = int(stats["pooled_etf_years"])
    g1 = n >= E2_G1_MIN
    rho = stats.get("weighted_rho")
    point_ok = _ge(rho, E2_G2_RHO)
    g2 = bool(point_ok and holm_res["rejected"])
    ny, npos = int(stats["n_years"]), int(stats["n_pos_years"])
    g3 = bool(ny > 0 and npos * E2_G3_SHARE[1] >= ny * E2_G3_SHARE[0])  # 정수 비교: npos/ny ≥ 2/3
    rule = assign_grade(g1, g2, g3, False)
    return {
        "g1": {"pooled_etf_years": n, "status": "표본 부족" if not g1 else "판정", "pass": g1},
        "g2": {
            "weighted_rho": rho,
            "point_threshold": E2_G2_RHO,
            "point_ok": point_ok,
            "holm_lower_bound": holm_res["holm_lower_bound"],
            "holm_lower_threshold": E2_G2_LOWER,
            "holm_status": holm_res["status"],
            "holm_rejected": holm_res["rejected"],
            "pass": g2,
        },
        "g3": {"n_years": ny, "n_pos_years": npos, "share": (npos / ny) if ny else None, "pass": g3},
        "grade_rule": rule,
        "grade": None if not g1 else rule,
        "purpose": "감쇠 폭 추정" if not g1 else "판정",
    }


def stop_clause(e1_gate: dict, e2_gate: dict) -> dict:
    """§10: E1 국내형 또는 E2 괴리가 D이면 "중단 조건 충족 — 사용자 확인 대기"를 적는다(스스로 멈추지 않음).
    D는 규칙을 그대로 적용한 값(``grade_rule``)이다. 표본 부족으로 등급이 없어도 규칙상 D면 해당한다."""
    which = []
    if e1_gate["grade_rule"] == "D":
        which.append("E1 국내형")
    if e2_gate["grade_rule"] == "D":
        which.append("E2 괴리")
    return {
        "fired": bool(which),
        "text": STOP_TEXT if which else None,
        "which": which,
        "note": "표본 부족(등급 없음)으로 규칙상 D인 경우 포함" if any(
            g["grade"] is None and g["grade_rule"] == "D" for g in (e1_gate, e2_gate)
        ) else None,
    }


# ---------------------------------------------------------------- KIND 일치
def norm_name(s: str | None) -> str:
    """KIND 쪽 ``name_norm`` 과 같은 규칙: NFKC 뒤 공백(모든 공백 문자) 제거."""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def load_kind(path: str | Path) -> pl.DataFrame:
    """KIND 거래소 폐지 공시(``event_type == "exchange_delisting"``)만. ``notice_date`` 를 날짜로 바꾼다."""
    d = pl.read_csv(str(path), infer_schema_length=0)
    d = d.filter(pl.col("event_type") == "exchange_delisting")
    return d.with_columns(pl.col("notice_date").str.strptime(pl.Date, "%Y-%m-%d")).select(
        "notice_date", "kind_name_raw", "name_norm", "kind_code5", "acptno", "maturity_name"
    )


def kind_window_of(last_date: date) -> str:
    """마지막 거래일이 KIND 조회 창 공백(과 앞뒤 30일) 안이면 "out", 아니면 "in"."""
    lo = KIND_GAP[0] - timedelta(days=KIND_GAP_MARGIN_DAYS)
    hi = KIND_GAP[1] + timedelta(days=KIND_GAP_MARGIN_DAYS)
    return "out" if lo <= last_date <= hi else "in"


def kind_match(events: pl.DataFrame, kind: pl.DataFrame) -> pl.DataFrame:
    """사건마다 KIND 거래소 폐지 공시와 맞춘다. 1순위 ``isu_cd[:5] == kind_code5``, 2순위 정규화 이름
    (우리 쪽은 마지막 거래일 ``isu_nm``). 후보가 여럿이면 공시일이 마지막 거래일에 가장 가까운 것.
    ``events`` 칸: ``isu_cd``, ``isu_nm``, ``last_date``. 더하는 칸: ``kind_method``(code|name|null),
    ``kind_notice_date``, ``kind_days``(공시일 − 마지막 거래일), ``kind_window``(in|out)."""
    by_code: dict[str, list[date]] = {}
    by_name: dict[str, list[date]] = {}
    for r in kind.iter_rows(named=True):
        by_code.setdefault(r["kind_code5"], []).append(r["notice_date"])
        by_name.setdefault(r["name_norm"], []).append(r["notice_date"])

    def nearest(c: list[date], L: date) -> date:
        return min(c, key=lambda d: (abs((d - L).days), d))

    method, notice, days, win = [], [], [], []
    for r in events.iter_rows(named=True):
        L = r["last_date"]
        c = by_code.get((r["isu_cd"] or "")[:5])
        m = "code"
        if not c:
            c = by_name.get(norm_name(r.get("isu_nm"))) or None
            m = "name"
        if c:
            d = nearest(c, L)
            method.append(m)
            notice.append(d)
            days.append((d - L).days)
        else:
            method.append(None)
            notice.append(None)
            days.append(None)
        win.append(kind_window_of(L))
    return events.with_columns(
        pl.Series("kind_method", method, dtype=pl.String),
        pl.Series("kind_notice_date", notice, dtype=pl.Date),
        pl.Series("kind_days", days, dtype=pl.Int64),
        pl.Series("kind_window", win, dtype=pl.String),
    )


def _days_dist(v: list[int]) -> dict:
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=float)
    vals, cnt = np.unique(a.astype(int), return_counts=True)
    return {
        "n": len(v),
        "min": int(a.min()),
        "p25": float(np.percentile(a, 25)),
        "median": float(np.median(a)),
        "p75": float(np.percentile(a, 75)),
        "max": int(a.max()),
        "counts_by_day": {str(int(k)): int(c) for k, c in zip(vals, cnt)},
    }


def kind_summary(m: pl.DataFrame) -> dict:
    """``kind_match`` 결과의 일치율. 일치율은 창 안 사건으로만 계산하고 창 밖은 따로 센다."""

    def one(df: pl.DataFrame) -> dict:
        inn = df.filter(pl.col("kind_window") == "in")
        n = inn.height
        mt = inn.filter(pl.col("kind_method").is_not_null())
        return {
            "n_events": df.height,
            "n_out_of_window": df.height - n,
            "n_in_window": n,
            "n_matched": mt.height,
            "n_matched_code": int((mt["kind_method"] == "code").sum()),
            "n_matched_name": int((mt["kind_method"] == "name").sum()),
            "n_unmatched": n - mt.height,
            "match_rate": (mt.height / n) if n else None,
            "notice_minus_last_days": _days_dist([x for x in mt["kind_days"].to_list() if x is not None]),
        }

    out = {"all": one(m)}
    for reg in e1.REGIONS:
        out[reg] = one(m.filter(pl.col("region") == reg))
    return out


# ---------------------------------------------------------------- 보조
def _sha256(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _git(*args: str) -> str:
    here = Path(__file__).resolve().parent
    try:
        return subprocess.check_output(["git", "-C", str(here), *args], text=True).strip()
    except Exception:
        return ""


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (date, datetime)):
        return str(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _boot_sum(a: np.ndarray, null_fn=None) -> dict:
    """부트스트랩 배열 요약(nan 회차 수 포함)."""
    f = a[np.isfinite(a)]
    res: dict = {"b": int(len(a)), "n_nan": int(len(a) - len(f))}
    if len(f) == 0:
        return res
    res.update(
        mean=float(f.mean()),
        ci95=[float(np.quantile(f, 0.025)), float(np.quantile(f, 0.975))],
        q05=float(np.quantile(f, 0.05)),
        q025=float(np.quantile(f, 0.025)),
    )
    return res


def _e2_lower(boot: np.ndarray, alpha: float) -> float:
    return e2.lower_bound(boot, alpha) if np.isfinite(boot).any() else float("nan")


def _e1_lower(boot: np.ndarray, alpha: float) -> float:
    return e1.lower_bound(boot, alpha) if np.isfinite(boot).any() else float("nan")


def _wmean_by_year_excl(gy: pl.DataFrame, drop_year: int | None) -> float:
    v = gy.filter(pl.col("rho").is_not_null() & pl.col("rho").is_not_nan())
    if drop_year is not None:
        v = v.filter(pl.col("year") != drop_year)
    return e2._wmean(v["rho"].to_numpy(), v["size"].to_numpy())


def e1_leave_one_year_out(events: pl.DataFrame, region: str = "domestic") -> dict:
    """F 연도 하나씩 뺀 적중률(점수 있는 사건 기준)의 목록과 최소·최대."""
    e = events.filter((pl.col("region") == region) & pl.col("scored"))
    years = sorted(e["formation_month_end"].dt.year().unique().to_list())
    rows = []
    for y in years:
        g = e.filter(pl.col("formation_month_end").dt.year() != y)
        n = g.height
        h = int(g["alert"].sum() or 0)
        rows.append({"left_out_year": y, "n_scored": n, "hits": h, "hit_rate": (h / n) if n else None})
    rates = [r["hit_rate"] for r in rows if r["hit_rate"] is not None]
    return {
        "by_year": rows,
        "min": min(rates) if rates else None,
        "max": max(rates) if rates else None,
    }


def e2_leave_one_year_out(gy: pl.DataFrame) -> dict:
    years = sorted(gy["year"].unique().to_list())
    rows = [{"left_out_year": y, "weighted_rho": _wmean_by_year_excl(gy, y)} for y in years]
    vals = [r["weighted_rho"] for r in rows if not math.isnan(r["weighted_rho"])]
    return {"by_year": rows, "min": min(vals) if vals else None, "max": max(vals) if vals else None}


# ---------------------------------------------------------------- E1 쪽
def run_e1(
    panel: ep.Panel, life: pl.DataFrame, period: str, b: int, seed: int
) -> dict:
    """E1 전체: 선행 6·3·12개월 통계와 부트스트랩, 만기형 포함 민감도, 한 해씩 뺀 범위. DataFrame 몇 개를 같이 돌려준다."""
    scores = e1.monthly_scores(panel, life)
    scores_m = e1.monthly_scores(panel, life, include_maturity=True)
    horizons: dict = {}
    boots: dict = {}
    events6 = None
    for h in e1.LEAD_HORIZONS:
        st = e1.outcome_stats(scores, life, period, h, panel=panel)
        bt = e1.bootstrap(scores, life, period, h, panel=panel, b=b, seed=seed)
        boots[h] = bt
        horizons[str(h)] = {
            "stats": st["by_region"],
            "bootstrap": {reg: e1._boot_summary(br) for reg, br in bt.items()},
        }
        if h == 6:
            events6 = st["events"]
    scores_full = e1.monthly_scores_netasst_full(panel, life)
    full_h: dict = {}
    full_events6 = None
    for h in e1.LEAD_HORIZONS:
        st = e1.outcome_stats(scores_full, life, period, h, panel=panel)
        bt = e1.bootstrap(scores_full, life, period, h, panel=panel, b=b, seed=seed)
        full_h[str(h)] = {
            "stats": {"domestic": st["by_region"]["domestic"]},
            "bootstrap": {"domestic": e1._boot_summary(bt["domestic"])},
            "boot_arrays": bt["domestic"],
        }
        if h == 6:
            full_events6 = st["events"].filter(pl.col("region") == "domestic")
    stm = e1.outcome_stats(scores_m, life, period, 6, panel=panel, include_maturity=True)
    btm = e1.bootstrap(scores_m, life, period, 6, panel=panel, b=b, seed=seed, include_maturity=True)
    maturity_incl = {
        "stats": stm["by_region"],
        "bootstrap": {reg: e1._boot_summary(br) for reg, br in btm.items()},
    }
    return {
        "scores": scores,
        "horizons": horizons,
        "boots": boots,
        "events6": events6,
        "netasst_full_horizons": {k: {kk: vv for kk, vv in v.items() if kk != "boot_arrays"} for k, v in full_h.items()},
        "netasst_full_events6": full_events6,
        "netasst_full_pool_sizes": e1.pool_size_summary(
            scores_full.filter(pl.col("region") == "domestic"), *e1.PERIODS[period]
        ),
        "netasst_full_loyo": e1_leave_one_year_out(full_events6, "domestic"),
        "events_incl_maturity": stm["events"],
        "maturity_included": maturity_incl,
        "loyo_domestic": e1_leave_one_year_out(events6, "domestic"),
        "loyo_foreign": e1_leave_one_year_out(events6, "foreign"),
        "pool_sizes": e1.pool_size_summary(scores, *e1.PERIODS[period]),
    }


# ---------------------------------------------------------------- E2 쪽
E2_VARIANTS = {
    "main": dict(min_group=e2.MIN_GROUP, value="gap", key="lenient"),
    "trdval": dict(min_group=e2.MIN_GROUP, value="trdval", key="lenient"),
    "min3": dict(min_group=e2.MIN_GROUP_RECORD, value="gap", key="lenient"),
    "strict": dict(min_group=e2.MIN_GROUP, value="gap", key="strict"),
}


def run_e2(panel: ep.Panel, life: pl.DataFrame, period: str, n_boot: int, seed: int) -> dict:
    lo, hi = e2.PERIODS[period]
    ey = e2.etf_years(panel, life, list(range(lo, hi + 1)), with_values=True)
    excl = e2.exclusion_counts(ey)
    stats = {n: e2.e2_stats(ey, period, **kw) for n, kw in E2_VARIANTS.items()}
    boots = {n: e2.bootstrap(ey, period, **kw, n_boot=n_boot, seed=seed) for n, kw in E2_VARIANTS.items()}
    netassets = ep.month_end_netassets(panel, life).filter(pl.col("month_end").dt.year().is_between(lo, hi))
    base = e2.e2_baseline(ey, netassets, period, min_group=e2.MIN_GROUP, key="lenient")
    base_gy = base.pop("group_years")
    variants = {}
    for n in E2_VARIANTS:
        variants[n] = {
            **e2._stats_summary(stats[n]),
            "bootstrap": {**_boot_sum(boots[n]), "p_value": e2.p_value(boots[n]) if np.isfinite(boots[n]).any() else None},
            "leave_one_year_out": e2_leave_one_year_out(stats[n]["group_years"]),
        }
    gy = pl.concat(
        [stats[n]["group_years"].with_columns(pl.lit(n).alias("variant")) for n in E2_VARIANTS]
    ).select("variant", "year", "group_key", "size", "rho")
    return {
        "stats": stats,
        "boots": boots,
        "variants": variants,
        "baseline_I19": base,
        "baseline_group_years": base_gy,
        "exclusions_by_year": excl,
        "exclusions_total": {c: int(excl[c].sum()) for c in excl.columns if c != "year"},
        "n_removed_delisted_y1_total": int(excl["n_removed_delisted_y1"].sum()),
        "group_years": gy,
        "formation_years": [lo, hi],
    }


# ---------------------------------------------------------------- 전체
def run(
    period: str,
    *,
    input_path: str | None = None,
    out_dir: str | None = None,
    kind_csv: str | None = None,
    interp_table: str | None = None,
    prereg: str | None = None,
    b: int = e1.BOOTSTRAP_B,
    seed: int = e1.BOOTSTRAP_SEED,
) -> dict:
    """판정 결합 전체를 돌려 JSON·CSV로 쓴다. 판정 구간은 보호 변수가 있어야 한다."""
    _guard(period)
    root = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
    input_path = input_path or str(root / ep.DEFAULT_INPUT_REL)
    kind_csv = kind_csv or str(root / DEFAULT_KIND_REL)
    interp_table = interp_table or ep.default_interp_table()
    prereg = prereg or ep.default_prereg()
    if out_dir is None:
        rel = DEFAULT_OUT_REL_DEV if period == "dev" else DEFAULT_OUT_REL_JUDGMENT
        out_dir = str(root / rel)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    panel = ep.read_panel(input_path)
    life = ep.lifecycle(panel)
    r1 = run_e1(panel, life, period, b, seed)
    r2 = run_e2(panel, life, period, b, seed)

    # ---- 판정 가설 둘과 Holm
    hit6 = r1["boots"][6]["domestic"].arrays["hit"]
    gap_boot = r2["boots"]["main"]
    items = [
        {
            "name": H_E1,
            "p": e1.p_value(hit6),
            "bound_fn": lambda a: _e1_lower(hit6, a),
            "bound_ok": lambda x: x >= E1_G2_LOWER,
        },
        {
            "name": H_E2,
            "p": e2.p_value(gap_boot) if np.isfinite(gap_boot).any() else 1.0,
            "bound_fn": lambda a: _e2_lower(gap_boot, a),
            "bound_ok": lambda x: x > E2_G2_LOWER,
        },
    ]
    hres = holm_combine(items)
    hres[H_E1]["null"] = e1.HIT_NULL
    hres[H_E1]["n_nan_rounds"] = int(np.isnan(hit6).sum())
    hres[H_E2]["null"] = 0.0
    hres[H_E2]["n_nan_rounds"] = int(np.sum(~np.isfinite(gap_boot)))

    st_dom = r1["horizons"]["6"]["stats"]["domestic"]
    g_e1 = e1_gates(st_dom, hres[H_E1])
    g_e2 = e2_gates(r2["stats"]["main"], hres[H_E2])
    stop = stop_clause(g_e1, g_e2)

    # ---- KIND 일치
    kind = load_kind(kind_csv)
    life_flags = life.select("isu_cd", "exclude_maturity")
    ev_main = r1["events6"].with_columns(pl.lit(False).alias("exclude_maturity"))
    ev_incl = r1["events_incl_maturity"].join(life_flags, on="isu_cd", how="left").with_columns(
        pl.col("exclude_maturity").fill_null(False)
    )
    ev_mat_only = ev_incl.filter(pl.col("exclude_maturity"))
    km_main = kind_match(ev_main, kind)
    km_mat = kind_match(ev_mat_only, kind)
    km_incl = kind_match(ev_incl, kind)
    kind_rec = {
        "kind_rows_exchange_delisting": kind.height,
        "gap_window": [str(KIND_GAP[0]), str(KIND_GAP[1])],
        "gap_margin_days": KIND_GAP_MARGIN_DAYS,
        "main_excl_maturity": kind_summary(km_main),
        "maturity_only": kind_summary(km_mat),
        "incl_maturity": kind_summary(km_incl),
    }

    # ---- 사건 CSV: 주(만기형 제외) + 만기형 추가분
    ev_csv = pl.concat(
        [
            km_main.with_columns(pl.lit(True).alias("in_main")),
            km_mat.with_columns(pl.lit(False).alias("in_main")),
        ],
        how="diagonal",
    ).sort(["in_main", "region", "last_date", "isu_cd"], descending=[True, False, False, False])
    ev_csv.write_csv(out / "e1_events.csv")
    r2["group_years"].write_csv(out / "e2_group_years.csv")
    r1["netasst_full_events6"].write_csv(out / "e1_netasst_full_events.csv")
    r2["baseline_group_years"].write_csv(out / "e2_baseline_group_years.csv")
    r2["exclusions_by_year"].write_csv(out / "e2_exclusions_by_year.csv")

    created = datetime.now().isoformat(timespec="seconds")
    summary = {
        "version": JUDGE_VERSION,
        "correction": {"id": CORRECTION_ID, "note": CORRECTION_NOTE, "lookahead_note": LOOKAHEAD_NOTE, "judgment_family": list(JUDGMENT_FAMILY)},
        "period": period,
        "created_at": created,
        "e1_formation_range": [str(d) for d in e1.PERIODS[period]],
        "e2_formation_years": r2["formation_years"],
        "bootstrap": {"b": b, "seed": seed},
        "holm": {"alphas": list(HOLM_ALPHAS), "m": 2, "rule": "하한(단계 α 분위수)이 문턱을 넘으면 기각, p 작은 쪽부터"},
        "hypotheses": hres,
        "e1_domestic": g_e1,
        "e2_gap": g_e2,
        "stop_clause": stop,
        "records": {
            "e1_domestic_netasst_full": {
                "label": "기록용(정정 E-1R, P3): E1 국내형 순자산 단독, 전체 풀(상관 유무 무관, 지수 종가 안 씀). 순자산 백분위 ≤ 10 경보",
                "horizons": r1["netasst_full_horizons"],
                "leave_one_formation_year_out": r1["netasst_full_loyo"],
                "pool_sizes": r1["netasst_full_pool_sizes"],
            },
            "e1": {
                "horizons": r1["horizons"],
                "maturity_included_h6": r1["maturity_included"],
                "leave_one_formation_year_out_domestic": r1["loyo_domestic"],
                "leave_one_formation_year_out_foreign": r1["loyo_foreign"],
                "pool_sizes": r1["pool_sizes"],
            },
            "e2": {
                "variants": r2["variants"],
                "baseline_I19": r2["baseline_I19"],
                "exclusions_total": r2["exclusions_total"],
                "n_removed_delisted_y1_total": r2["n_removed_delisted_y1_total"],
            },
            "kind": kind_rec,
        },
    }
    (out / "judge_summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2))
    (out / "judge_summary.md").write_text(render_md(_jsonable(summary)))

    # ---- manifest(§12.2)
    here = Path(__file__).resolve().parent
    status = _git("status", "--porcelain")
    inputs = {
        "etf_csv": input_path,
        "kind_csv": kind_csv,
        "kind_query_windows_csv": str(Path(kind_csv).parent / "kind_query_windows.csv"),
        "interp_table": interp_table,
        "preregistration": prereg,
    }
    manifest = {
        "created_at": created,
        "purpose": f"quality-score v0 부분 E 판정 결합({period} 구간)",
        "period": period,
        "inputs": {k: {"path": v, "sha256": _sha256(v) if Path(v).exists() else None} for k, v in inputs.items()},
        "code": {
            "version": JUDGE_VERSION,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "head": _git("rev-parse", "HEAD"),
            "git_status_porcelain": status.splitlines(),
            "modules": {
                n: {"path": str(here / n), "sha256": _sha256(here / n)}
                for n in ("etf_judge.py", "etf_e1.py", "etf_e2.py", "etf_panel.py", "etf_classify.py")
            },
        },
        "seed": seed,
        "bootstrap_b": b,
        "uv_lock": {"path": UV_LOCK, "sha256": _sha256(UV_LOCK) if Path(UV_LOCK).exists() else None},
        "runtime": {"python": sys.version.split()[0], "polars": pl.__version__, "numpy": np.__version__},
        "judgment_env": {"name": JUDGMENT_ENV, "value": os.environ.get(JUDGMENT_ENV)},
        "correction": {
            "id": CORRECTION_ID,
            "note": CORRECTION_NOTE,
            "judgment_family": list(JUDGMENT_FAMILY),
            "lookahead_note": LOOKAHEAD_NOTE,
            "interp_table_expected_sha256": INTERP_TABLE_SHA256,
            "interp_table_sha256_matches": bool(
                Path(interp_table).exists() and _sha256(interp_table) == INTERP_TABLE_SHA256
            ),
        },
        "outputs": sorted(p.name for p in out.glob("*") if p.name != "manifest.json") + ["manifest.json"],
    }
    (out / "manifest.json").write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2))
    return {"summary": summary, "out_dir": str(out)}


# ---------------------------------------------------------------- 사람이 읽는 표(숫자만)
def _f(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _table(head: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_f(c) for c in r) + " |")
    return "\n".join(lines) + "\n"


def render_md(s: dict) -> str:
    """judge_summary.json(이미 JSON 가능한 dict)에서 표만 뽑는다. 해석 문장은 쓰지 않는다."""
    o = [
        f"# 부분 E 판정 결합 ({s['period']})\n",
        f"{s['created_at']} · {s['version']} · B {s['bootstrap']['b']} · 시드 {s['bootstrap']['seed']}\n",
        f"**{s['correction']['note']}** (정정 {s['correction']['id']})\n",
        f"{s['correction']['lookahead_note']}\n",
    ]
    h = s["hypotheses"]
    o.append("## 가설·Holm\n")
    o.append(_table(
        ["가설", "p", "Holm 단계", "단계 α", "하한 α", "Holm 하한", "상태", "기각", "nan 회차"],
        [[k, v["p"], v["stage"], v["stage_alpha"], v["bound_alpha"], v["holm_lower_bound"], v["status"], v["rejected"], v["n_nan_rounds"]] for k, v in h.items()],
    ))
    g1, g2 = s["e1_domestic"], s["e2_gap"]
    o.append("## 게이트·등급\n")
    o.append(_table(
        ["부분", "G1 값", "G1", "G2 점추정", "G2 Holm 하한", "G2", "G3", "점수 없는 비율", "규칙 등급", "등급", "목적"],
        [
            ["E1 국내형", g1["g1"]["n_scored"], g1["g1"]["status"], g1["g2"]["hit_rate"], g1["g2"]["holm_lower_bound"], g1["g2"]["pass"],
             f"{_f(g1['g3']['first_half_hit_rate'])} / {_f(g1['g3']['second_half_hit_rate'])} → {g1['g3']['pass']}", g1["unscored_share"], g1["grade_rule"], g1["grade"], g1["purpose"]],
            ["E2 괴리", g2["g1"]["pooled_etf_years"], g2["g1"]["status"], g2["g2"]["weighted_rho"], g2["g2"]["holm_lower_bound"], g2["g2"]["pass"],
             f"{g2['g3']['n_pos_years']}/{g2['g3']['n_years']} → {g2['g3']['pass']}", "-", g2["grade_rule"], g2["grade"], g2["purpose"]],
        ],
    ))
    sc = s["stop_clause"]
    o.append(f"중단 조항: {sc['text'] or '해당 없음'} {sc['which']}\n")
    nf = s["records"]["e1_domestic_netasst_full"]
    o.append("## E1 국내형 순자산 단독, 전체 풀 — 기록용(정정 E-1R, P3)\n")
    rows = []
    for hz, v in nf["horizons"].items():
        st = v["stats"]["domestic"]
        bs = v["bootstrap"]["domestic"]
        ci = bs.get("hit", {}).get("ci95") or ["-", "-"]
        rows.append([hz, st["n_events"], st["n_scored"], st["n_unscored"], st["unscored_share"], st["hits"], st["hit_rate"], ci[0], ci[1],
                     st["hit_rate_unscored_as_miss"], st["far"]["far"], bs["p_hit_le_0.30"]])
    o.append(_table(["개월", "사건", "점수 있음", "점수 없음", "점수 없는 비율", "적중", "적중률", "CI 하", "CI 상", "점수 없음=미적중", "FAR", "p(≤0.30)"], rows))
    rows = []
    for hz, v in nf["horizons"].items():
        ld = v["stats"]["domestic"]["lead"]
        rows.append([hz, ld.get("n"), ld.get("mean"), ld.get("median"), ld.get("p25"), ld.get("p75"), ld.get("max"), ld.get("n_zero"), ld.get("n_1_2"), ld.get("n_3_5"), ld.get("n_6_11"), ld.get("n_ge_12")])
    o.append("선행 분포\n")
    o.append(_table(["개월", "n", "평균", "중앙", "p25", "p75", "최대", "0", "1-2", "3-5", "6-11", "12+"], rows))
    o.append("## E1 선행 기간별 통계\n")
    rows = []
    for hz, v in s["records"]["e1"]["horizons"].items():
        for reg, st in v["stats"].items():
            bs = v["bootstrap"][reg]
            ci = bs.get("hit", {}).get("ci95") or ["-", "-"]
            dci = bs.get("diff", {}).get("ci95") or ["-", "-"]
            rows.append([hz, reg, st["n_events"], st["n_scored"], st["hits"], st["hit_rate"], ci[0], ci[1], st["hit_rate_unscored_as_miss"],
                         st["baseline_hit_rate"], st["hit_minus_baseline"], dci[0], dci[1], st["far"]["far"], st["baseline_far"]["far"], bs["p_hit_le_0.30"]])
    o.append(_table(["개월", "유형", "사건", "점수 있음", "적중", "적중률", "CI 하", "CI 상", "점수 없음=미적중", "기준선", "차이", "차이 CI 하", "차이 CI 상", "FAR", "기준선 FAR", "p(≤0.30)"], rows))
    o.append("## E1 선행 분포 (6개월)\n")
    rows = []
    for reg, st in s["records"]["e1"]["horizons"]["6"]["stats"].items():
        ld = st["lead"]
        rows.append([reg, ld.get("n"), ld.get("mean"), ld.get("median"), ld.get("p25"), ld.get("p75"), ld.get("max"), ld.get("n_zero"), ld.get("n_1_2"), ld.get("n_3_5"), ld.get("n_6_11"), ld.get("n_ge_12")])
    o.append(_table(["유형", "n", "평균", "중앙", "p25", "p75", "최대", "0", "1-2", "3-5", "6-11", "12+"], rows))
    mi = s["records"]["e1"]["maturity_included_h6"]
    o.append("## E1 만기형 포함 (6개월)\n")
    o.append(_table(["유형", "사건", "점수 있음", "적중", "적중률", "p"], [[r, st["n_events"], st["n_scored"], st["hits"], st["hit_rate"], mi["bootstrap"][r]["p_hit_le_0.30"]] for r, st in mi["stats"].items()]))
    lo = s["records"]["e1"]["leave_one_formation_year_out_domestic"]
    o.append(f"E1 국내형 한 해씩 뺀 적중률: 최소 {_f(lo['min'])} · 최대 {_f(lo['max'])}\n")
    e2v = s["records"]["e2"]["variants"]
    o.append("## E2 변형\n")
    rows = []
    for n, v in e2v.items():
        bs = v["bootstrap"]
        rows.append([n, v["value"], v["min_group"], v["key"], v["pooled_etf_years"], v["n_group_years"], v["weighted_rho"], (bs.get("ci95") or ["-", "-"])[0], (bs.get("ci95") or ["-", "-"])[1],
                     v["n_pos_years"], v["n_years"], v["leave_one_year_out"]["min"], v["leave_one_year_out"]["max"], bs.get("p_value")])
    o.append(_table(["변형", "값", "크기", "규칙", "ETF-연", "그룹-연도", "가중 ρ", "CI 하", "CI 상", "ρ>0 해", "해", "뺀 최소", "뺀 최대", "p"], rows))
    bl = s["records"]["e2"]["baseline_I19"]
    o.append("## E2 기준선(I19)·제외\n")
    o.append(_table(["칸", "값"], [[k, v] for k, v in bl.items()]))
    o.append(_table(["제외 칸", "합계"], [[k, v] for k, v in s["records"]["e2"]["exclusions_total"].items()]))
    k = s["records"]["kind"]
    o.append("## KIND 일치\n")
    rows = []
    for grp in ("main_excl_maturity", "maturity_only", "incl_maturity"):
        for reg in ("all", "domestic", "foreign"):
            x = k[grp][reg]
            d = x["notice_minus_last_days"]
            rows.append([grp, reg, x["n_events"], x["n_out_of_window"], x["n_in_window"], x["n_matched"], x["n_matched_code"], x["n_matched_name"], x["match_rate"], d.get("min"), d.get("median"), d.get("max")])
    o.append(_table(["집합", "유형", "사건", "창 밖", "창 안", "일치", "코드", "이름", "일치율", "공시−마지막 최소", "중앙", "최대"], rows))
    return "\n".join(o)


# ---------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--period", choices=["dev", "judgment"], required=True)
    ap.add_argument("--input")
    ap.add_argument("--out-dir")
    ap.add_argument("--kind-csv")
    ap.add_argument("--interp-table")
    ap.add_argument("--prereg")
    a = ap.parse_args(argv)
    res = run(
        a.period,
        input_path=a.input,
        out_dir=a.out_dir,
        kind_csv=a.kind_csv,
        interp_table=a.interp_table,
        prereg=a.prereg,
    )
    print((Path(res["out_dir"]) / "judge_summary.md").read_text())
    print("out:", res["out_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
