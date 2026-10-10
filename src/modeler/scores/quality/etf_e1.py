"""ETF 품질 점수 E1 상장 유지 위험 — 월말 점수, 결과 통계, 클러스터 부트스트랩
(사전등록 20261010_quality_score §11.1~11.4, §7.1, 구현 해석 표 v1 I01~I24).

세 덩어리로 나뉜다.

1. ``monthly_scores`` — 월말마다 ETF-월 한 행. **결과(사건)를 보지 않는다.** 어느 월말에나 부를 수 있다.
2. ``outcome_stats`` — 사건(만기형 제외 상장폐지)과 점수를 잇는 통계. 구간 보호가 걸려 있다.
3. ``bootstrap`` — ETF 단위 복원 추출. p값과 단측 하한은 배열에서 따로 뽑는다(Holm 결합은 다른 모듈).

    STOCK_DATA_ROOT=../stock_data PYTHONPATH=src python -m modeler.scores.quality.etf_e1 --period dev

경로는 환경변수 ``STOCK_DATA_ROOT`` 로 받는다(기본 ``../stock_data``). 산출물은
``kr/output/quality_score_etf_dev_20261010/e1/`` 에만 쓴다.

**판정 구간 보호.** ``outcome_stats``·``bootstrap`` 에 ``period="judgment"`` 를 주면 환경변수
``QUALITY_E_JUDGMENT_CONFIRMED`` 가 비어 있지 않을 때만 돈다. 아니면 예외다. 개발 구간으로 돌 때도
점수는 형성 월말 2014-12까지만 사건과 맞춘다(선행 분포의 연속 경보 사슬 포함).
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl

from modeler.scores.quality import etf_panel as ep

E1_VERSION = "quality-score-v0/etf_e1/1"

DEFAULT_OUT_REL = "kr/output/quality_score_etf_dev_20261010/e1"
DEFAULT_INTERP_TABLE = (
    "/private/tmp/claude-501/-Users-whishaw-wss-p-my/b1cc46eb-1957-4b1b-b4b5-6208b7ec737c/"
    "scratchpad/interp_table_v1.md"
)

# ---- 처음 고른 값(사전등록 §11.2~11.4)과 해석 표 값. 코드에 이름을 붙여 둔다.
NETASST_FLOOR_WON = 5e9  # §11.2: 순자산 거리 log(월말 순자산 ÷ 50억 원)
CORR_WINDOW_SESSIONS = 252  # §11.2·I05: 창 = M까지 시장 거래일 252일
CORR_MIN_PAIRS = 200  # §11.2: 창 안 유효 쌍 200 이상이어야 계산(199면 결측)
RULE_CORR_PASSIVE = 0.9  # §11.2: 상장폐지 요건 상관 0.9(패시브)
RULE_CORR_ACTIVE = 0.7  # 액티브 0.7
ALERT_PCT = 10.0  # §11.3: 경보 = E1 백분위 하위 10% (I11), 기준선도 같은 10% (I12)
LAST_MONTH_END = date(2026, 9, 30)  # I21: 자료 끝 달(2026-10)은 월말로 안 쓴다
FAR_WINDOW_MONTHS = 12  # I09: 다음 12개월
HIT_NULL = 0.30  # I02: 귀무 적중률 30%
BOOTSTRAP_B = 2000  # I14
BOOTSTRAP_SEED = 20261010  # I14
LEAD_HORIZONS = (6, 3, 12)  # §11.3: 주 6개월, 선행 분포용 3·12개월
UNSCORED_SHARE_CAP = 0.30  # §11.3: 점수 없는 사건 30% 초과면 등급 C 이하
JUDGMENT_ENV = "QUALITY_E_JUDGMENT_CONFIRMED"

# 형성 월말(F, 또는 FAR의 M)의 구간(§7.1). 양 끝 포함.
PERIODS = {
    "dev": (date(2011, 1, 1), date(2014, 12, 31)),
    "judgment": (date(2015, 1, 1), date(2026, 3, 31)),
}
# I08: 판정 구간의 전반·후반(F의 연도). 개발 구간에는 규정이 없어 같은 폭으로 둘로 가른 기록용 값을 쓴다.
HALVES = {
    "judgment": {"first": (2015, 2020), "second": (2021, 2026)},
    "dev": {"first": (2011, 2012), "second": (2013, 2014)},  # 규정 아님, 기록용
}
REGIONS = ("domestic", "foreign")
# I26: 선행 분포의 경보 사슬이 쓰는 점수의 끝 월말. 개발은 개발 구간 끝, 판정은 자료 끝 달 직전 월말.
CHAIN_END = {"dev": PERIODS["dev"][1], "judgment": LAST_MONTH_END}


# ---------------------------------------------------------------- 구간 보호
def _guard(period: str) -> tuple[date, date]:
    """period 확인. judgment는 환경변수가 있어야 한다. 구간 (처음, 끝)을 돌려준다."""
    if period not in PERIODS:
        raise ValueError(f"period must be 'dev' or 'judgment', got {period!r}")
    if period == "judgment" and not os.environ.get(JUDGMENT_ENV, "").strip():
        raise PermissionError(
            f"판정 구간은 환경변수 {JUDGMENT_ENV} 가 비어 있지 않을 때만 돌 수 있다(사전등록 §12.4)."
        )
    return PERIODS[period]


# ---------------------------------------------------------------- 백분위(I03)
def percentile_expr(col: str, over: list[str]) -> pl.Expr:
    """(평균 순위 − 1) ÷ (n − 1) × 100. 높을수록 안전. n = 1이면 50. null은 n과 순위에서 빠진다."""
    n = pl.col(col).count().over(over)
    r = pl.col(col).rank("average").over(over)
    return (
        pl.when(pl.col(col).is_null())
        .then(None)
        .when(n == 1)
        .then(pl.lit(50.0))
        .otherwise((r - 1.0) / (n - 1.0) * 100.0)
    )


# ---------------------------------------------------------------- 상관계수(I05)
def window_corr(
    grid: pl.DataFrame,
    pairs: pl.DataFrame,
    window: int = CORR_WINDOW_SESSIONS,
    min_pairs: int = CORR_MIN_PAIRS,
) -> pl.DataFrame:
    """grid(``isu_cd``, ``day_idx``)의 행마다 M까지 시장 거래일 ``window`` 일 창 안 ``return_pairs`` 의
    피어슨 상관계수. 창 = day_idx ∈ [M−window+1, M]. 쌍이 min_pairs 미만이거나 한쪽이 상수면 null.
    칸: ``n_pairs``, ``corr``."""
    by = {
        k[0]: (g["day_idx"].to_numpy(), g["nav_ret"].to_numpy(), g["idx_ret"].to_numpy())
        for k, g in pairs.sort(["isu_cd", "day_idx"]).partition_by(
            "isu_cd", as_dict=True, maintain_order=True
        ).items()
    }
    n_out: list[int] = []
    c_out: list[float | None] = []
    for isu, m in zip(grid["isu_cd"].to_list(), grid["day_idx"].to_list()):
        if isu not in by:
            n_out.append(0)
            c_out.append(None)
            continue
        di, x, y = by[isu]
        lo = int(np.searchsorted(di, m - window + 1, side="left"))
        hi = int(np.searchsorted(di, m, side="right"))
        n = hi - lo
        n_out.append(n)
        if n < min_pairs:
            c_out.append(None)
            continue
        xs, ys = x[lo:hi], y[lo:hi]
        if xs.max() == xs.min() or ys.max() == ys.min():  # 한쪽이 상수면 상관 없음
            c_out.append(None)
            continue
        xd, yd = xs - xs.mean(), ys - ys.mean()
        sx, sy = float(np.sqrt((xd * xd).sum())), float(np.sqrt((yd * yd).sum()))
        if sx == 0.0 or sy == 0.0:
            c_out.append(None)
        else:
            c_out.append(float((xd * yd).sum() / (sx * sy)))
    return grid.with_columns(
        pl.Series("n_pairs", n_out, dtype=pl.Int64),
        pl.Series("corr", c_out, dtype=pl.Float64),
    )


# ---------------------------------------------------------------- 월말 점수
def monthly_scores(
    panel: ep.Panel,
    life: pl.DataFrame,
    last_month_end: date = LAST_MONTH_END,
    include_maturity: bool = False,
) -> pl.DataFrame:
    """월말 E1 점수. 대상 ETF-월마다 한 행(결과를 보지 않는다).

    대상(I23·I04·I10): 월말에 상장 중(첫 거래일 ≤ 월말 ≤ 마지막 거래일), 상장 1년(``is_listed_one_year``),
    ``exclude_maturity`` 아님, ``pension_ineligible_candidate`` 아님, region ∈ {domestic, foreign}.
    월말은 ``last_month_end`` 까지(I21).

    성분: ``log_netasst_dist`` = ln(월말 순자산 ÷ 5e9)(I24 순자산), ``corr``(국내형만, I05·I22).
    풀(I11): 그 월말·그 유형에서 성분이 다 있는 ETF(``in_pool``). ``pct_netasst`` 는 풀 전체,
    ``pct_corr`` 는 풀 안 패시브·액티브 따로. ``e1`` = 국내형 두 백분위 평균 / 해외형 ``pct_netasst``.
    ``e1_pct`` = e1을 같은 풀에서 다시 백분위로. ``alert`` = ``e1_pct`` ≤ 10(I11),
    ``baseline_alert`` = ``pct_netasst`` ≤ 10(I12). ``corr_gap`` = 상관 − 0.9(액티브 0.7), 기록용.
    풀 밖 행은 점수 칸이 null이고 ``alert`` 도 null이다(점수 없음).

    ``include_maturity=True``(기록용 민감도, 사전등록 §9)이면 만기형 ETF도 대상과 풀에 넣는다. 기본값에서는
    결과가 이 인자가 없던 때와 같다.
    """
    mn = ep.month_end_netassets(panel, life)
    meta = life.select(
        "isu_cd",
        "first_date",
        pl.col("exclude_maturity").fill_null(False),
        pl.col("pension_ineligible_candidate").fill_null(False),
        "region",
        pl.col("active").fill_null(False),
    )
    grid = (
        mn.join(meta, on="isu_cd")
        .filter(
            (pl.col("month_end") <= last_month_end)
            & ep.listed_one_year_expr()
            & (True if include_maturity else ~pl.col("exclude_maturity"))
            & ~pl.col("pension_ineligible_candidate")
            & pl.col("region").is_in(list(REGIONS))
        )
        .select("isu_cd", "month_end", "day_idx", "region", "active", "netasst")
    )
    dom = grid.filter(pl.col("region") == "domestic").select("isu_cd", "day_idx")
    pairs = ep.return_pairs(panel).join(dom.select("isu_cd").unique(), on="isu_cd")
    cr = window_corr(dom, pairs)
    g = grid.join(cr, on=["isu_cd", "day_idx"], how="left")
    g = g.with_columns(
        pl.when(pl.col("netasst").is_not_null())
        .then((pl.col("netasst") / NETASST_FLOOR_WON).log())
        .otherwise(None)
        .alias("log_netasst_dist"),
        pl.when(pl.col("region") == "domestic")
        .then(
            pl.col("corr")
            - pl.when(pl.col("active")).then(RULE_CORR_ACTIVE).otherwise(RULE_CORR_PASSIVE)
        )
        .otherwise(None)
        .alias("corr_gap"),
        (
            pl.col("netasst").is_not_null()
            & ((pl.col("region") == "foreign") | pl.col("corr").is_not_null())
        ).alias("in_pool"),
    )
    pool = g.filter(pl.col("in_pool")).with_columns(
        pl.when(pl.col("region") == "domestic").then(pl.col("corr")).otherwise(None).alias("_c")
    )
    pool = pool.with_columns(
        percentile_expr("netasst", ["month_end", "region"]).alias("pct_netasst"),
        percentile_expr("_c", ["month_end", "region", "active"]).alias("pct_corr"),
        pl.len().over(["month_end", "region"]).alias("pool_size"),
    ).with_columns(
        pl.when(pl.col("region") == "domestic")
        .then((pl.col("pct_netasst") + pl.col("pct_corr")) / 2.0)
        .otherwise(pl.col("pct_netasst"))
        .alias("e1")
    )
    pool = pool.with_columns(percentile_expr("e1", ["month_end", "region"]).alias("e1_pct"))
    pool = pool.with_columns(
        (pl.col("e1_pct") <= ALERT_PCT).alias("alert"),
        (pl.col("pct_netasst") <= ALERT_PCT).alias("baseline_alert"),
    ).select(
        "isu_cd", "month_end", "pct_netasst", "pct_corr", "pool_size", "e1", "e1_pct",
        "alert", "baseline_alert",
    )
    out = g.join(pool, on=["isu_cd", "month_end"], how="left").select(
        "isu_cd", "month_end", "day_idx", "region", "active", "netasst", "log_netasst_dist",
        "n_pairs", "corr", "corr_gap", "in_pool", "pool_size", "pct_netasst", "pct_corr", "e1",
        "e1_pct", "alert", "baseline_alert",
    )
    return out.sort(["month_end", "isu_cd"])


def monthly_scores_netasst_full(
    panel: ep.Panel,
    life: pl.DataFrame,
    last_month_end: date = LAST_MONTH_END,
    include_maturity: bool = False,
) -> pl.DataFrame:
    """정정 E-1(2026-10-10 승인) 기록용 점수: "E1 국내형 순자산 단독, 전체 풀". 국내형만 낸다.

    사유: 기초지수 종가(``OBJ_STKPRC_IDX``)가 사라진 ETF에서 생애 내내 비어 있어(룩어헤드) 상관 결측 여부로
    정해지는 풀이 미래 폐지에 따라 달라진다. 그래서 풀 = 월말에 순자산이 있는 국내형 대상 ETF 전부다
    (상관계수 유무와 무관, 지수 종가를 쓰지 않는다). 대상 조건(상장 1년·만기형·부적격·region)은
    ``monthly_scores`` 와 같다. ``pct_netasst`` 는 그 풀 전체의 백분위, ``e1 = e1_pct = pct_netasst``,
    ``alert = baseline_alert = pct_netasst <= 10``. 상관 관련 칸(``n_pairs``·``corr``·``corr_gap``·``pct_corr``)은
    null. 칸 구조는 ``monthly_scores`` 와 같아 ``outcome_stats``·``bootstrap`` 에 그대로 넣을 수 있다."""
    mn = ep.month_end_netassets(panel, life)
    meta = life.select(
        "isu_cd",
        "first_date",
        pl.col("exclude_maturity").fill_null(False),
        pl.col("pension_ineligible_candidate").fill_null(False),
        "region",
        pl.col("active").fill_null(False),
    )
    g = (
        mn.join(meta, on="isu_cd")
        .filter(
            (pl.col("month_end") <= last_month_end)
            & ep.listed_one_year_expr()
            & (True if include_maturity else ~pl.col("exclude_maturity"))
            & ~pl.col("pension_ineligible_candidate")
            & (pl.col("region") == "domestic")
        )
        .select("isu_cd", "month_end", "day_idx", "region", "active", "netasst")
    )
    g = g.with_columns(
        pl.when(pl.col("netasst").is_not_null())
        .then((pl.col("netasst") / NETASST_FLOOR_WON).log())
        .otherwise(None)
        .alias("log_netasst_dist"),
        pl.lit(None, dtype=pl.Int64).alias("n_pairs"),
        pl.lit(None, dtype=pl.Float64).alias("corr"),
        pl.lit(None, dtype=pl.Float64).alias("corr_gap"),
        pl.col("netasst").is_not_null().alias("in_pool"),
    )
    pool = g.filter(pl.col("in_pool")).with_columns(
        percentile_expr("netasst", ["month_end"]).alias("pct_netasst"),
        pl.len().over("month_end").alias("pool_size"),
    )
    pool = pool.with_columns(
        pl.col("pct_netasst").alias("e1"),
        pl.col("pct_netasst").alias("e1_pct"),
        (pl.col("pct_netasst") <= ALERT_PCT).alias("alert"),
        (pl.col("pct_netasst") <= ALERT_PCT).alias("baseline_alert"),
        pl.lit(None, dtype=pl.Float64).alias("pct_corr"),
    ).select(
        "isu_cd", "month_end", "pct_netasst", "pct_corr", "pool_size", "e1", "e1_pct", "alert", "baseline_alert"
    )
    out = g.join(pool, on=["isu_cd", "month_end"], how="left").select(
        "isu_cd", "month_end", "day_idx", "region", "active", "netasst", "log_netasst_dist",
        "n_pairs", "corr", "corr_gap", "in_pool", "pool_size", "pct_netasst", "pct_corr", "e1",
        "e1_pct", "alert", "baseline_alert",
    )
    return out.sort(["month_end", "isu_cd"])


# ---------------------------------------------------------------- 사건(I13)·형성 월말(I01)
def formation_month_end(last_date: date, month_end_dates: list[date], horizon_months: int) -> date | None:
    """I01: L에서 달력 ``horizon_months`` 개월을 뺀 날 이하의 가장 늦은 월말 거래일. 없으면 None.
    ``month_end_dates`` 는 오름차순."""
    target = ep.add_months(last_date, -horizon_months)
    i = bisect.bisect_right(month_end_dates, target)
    return month_end_dates[i - 1] if i > 0 else None


def event_etfs(life: pl.DataFrame, include_maturity: bool = False) -> pl.DataFrame:
    """I13 사건 후보: ``status == "disappeared"`` 이고 만기형·연금 부적격·region unknown 아님.
    ``include_maturity=True`` 이면 만기형도 넣는다(기록용 민감도)."""
    return life.filter(
        (pl.col("status") == "disappeared")
        & (True if include_maturity else ~pl.col("exclude_maturity").fill_null(False))
        & ~pl.col("pension_ineligible_candidate").fill_null(False)
        & pl.col("region").is_in(list(REGIONS))
    )


def _month_diff(later: date, earlier: date) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def _events_scored(
    scores: pl.DataFrame,
    life: pl.DataFrame,
    panel: ep.Panel,
    period: str,
    horizon_months: int,
    include_maturity: bool = False,
    chain_end: date | None = None,
) -> pl.DataFrame:
    """구간 안 사건마다 한 행: L, F, 점수 유무, 경보, 기준선 경보, 선행 개월 수(I20).
    사건의 F·점수는 구간(``PERIODS[period]``) 안만 쓴다. 선행 분포의 경보 사슬은 ``chain_end``(기본
    ``CHAIN_END[period]``, I26)까지의 월말 점수를 쓴다."""
    lo, hi = PERIODS[period]
    chain_hi = CHAIN_END[period] if chain_end is None else chain_end
    me = panel.month_ends["date"].to_list()
    me_idx = {d: i for i, d in enumerate(me)}
    ev = event_etfs(life, include_maturity)
    rows = []
    s = scores.filter((pl.col("month_end") >= lo) & (pl.col("month_end") <= chain_hi))
    s_ev = s.filter(pl.col("isu_cd").is_in(ev["isu_cd"].to_list()))
    sdict: dict[tuple[str, date], tuple[bool, bool | None, bool | None, float | None]] = {
        (r[0], r[1]): (r[2], r[3], r[4], r[5])
        for r in s_ev.select("isu_cd", "month_end", "in_pool", "alert", "baseline_alert", "e1_pct").iter_rows()
    }
    for r in ev.iter_rows(named=True):
        L = r["last_date"]
        F = formation_month_end(L, me, horizon_months)
        if F is None or not (lo <= F <= hi):
            continue
        key = (r["isu_cd"], F)
        in_pool, alert, base, e1p = sdict.get(key, (False, None, None, None))
        # I20: L 전(L보다 앞선) 마지막 점수 월말에서 거꾸로 이어지는 연속 경보
        prior = [d for d in me if d < L]
        last_true_before_L = prior[-1] if prior else None
        usable = [d for d in prior if d <= chain_hi]
        lead = None
        cens = bool(last_true_before_L is not None and last_true_before_L > chain_hi)
        j = None
        for k in range(len(usable) - 1, -1, -1):
            if sdict_get_pool(sdict, r["isu_cd"], usable[k]):
                j = k
                break
        if j is not None:
            lead = 0
            start = usable[j]
            if sdict[(r["isu_cd"], start)][1]:
                k = me_idx[start]
                while True:
                    k -= 1
                    if k < 0:
                        break
                    prev = me[k]
                    key2 = (r["isu_cd"], prev)
                    if key2 in sdict and sdict[key2][0] and sdict[key2][1]:
                        start = prev
                    else:
                        break
                lead = _month_diff(L, start)
        rows.append(
            {
                "isu_cd": r["isu_cd"],
                "isu_nm": r.get("isu_nm"),
                "region": r["region"],
                "active": r["active"],
                "last_date": L,
                "formation_month_end": F,
                "horizon_months": horizon_months,
                "scored": bool(in_pool),
                "alert": alert if in_pool else None,
                "baseline_alert": base if in_pool else None,
                "e1_pct": e1p if in_pool else None,
                "lead_months": lead,
                "lead_censored_by_period": cens,
            }
        )
    schema = {
        "isu_cd": pl.String, "isu_nm": pl.String, "region": pl.String, "active": pl.Boolean,
        "last_date": pl.Date, "formation_month_end": pl.Date, "horizon_months": pl.Int64,
        "scored": pl.Boolean, "alert": pl.Boolean, "baseline_alert": pl.Boolean,
        "e1_pct": pl.Float64, "lead_months": pl.Int64, "lead_censored_by_period": pl.Boolean,
    }
    return pl.DataFrame(rows, schema=schema).sort(["region", "last_date", "isu_cd"])


def sdict_get_pool(sdict: dict, isu: str, d: date) -> bool:
    v = sdict.get((isu, d))
    return bool(v and v[0])


# ---------------------------------------------------------------- FAR(I09)
def far_months(
    scores: pl.DataFrame, life: pl.DataFrame, panel: ep.Panel, period: str
) -> pl.DataFrame:
    """FAR 분모 행(I09): 구간 안 월말 M의 점수 있는 ETF-월 중 M+12개월 ≤ 자료 끝이고,
    [M, M+12개월]에 확정 사라짐이 없는 것. 그 창에 pending ETF의 마지막 거래일이 있으면 제외.

    창의 앞 끝은 M 포함이다(마지막 거래일 L = M이면 그 달 뒤로 곧바로 사라진 ETF이므로 "다음 12개월 안
    소멸"로 센다. 해석 표 I09의 "(M, M+12개월]" 와 L = M인 경우만 다르다 — 보고에 적음).
    칸: ``isu_cd``, ``region``, ``month_end``, ``alert``, ``baseline_alert``."""
    lo, hi = PERIODS[period]
    s = scores.filter(
        pl.col("in_pool") & (pl.col("month_end") >= lo) & (pl.col("month_end") <= hi)
    ).select("isu_cd", "month_end", "region", "alert", "baseline_alert")
    j = s.join(life.select("isu_cd", "last_date", "status"), on="isu_cd")
    # 월말 거래일은 달의 마지막 날이 아닐 수 있어 파이썬 add_months로 M+12개월을 구한다.
    mp = [ep.add_months(d, FAR_WINDOW_MONTHS) for d in j["month_end"].to_list()]
    j = j.with_columns(pl.Series("m_plus", mp, dtype=pl.Date))
    in_win = (pl.col("last_date") >= pl.col("month_end")) & (pl.col("last_date") <= pl.col("m_plus"))
    keep = (
        (pl.col("m_plus") <= panel.end_date)
        & ~(in_win & pl.col("status").is_in(["disappeared", "pending"]))
    )
    return j.filter(keep).select("isu_cd", "region", "month_end", "alert", "baseline_alert")


# ---------------------------------------------------------------- 통계
def _rate(num: int, den: int) -> float | None:
    return num / den if den else None


def _lead_dist(s: pl.Series) -> dict:
    v = s.drop_nulls()
    if v.len() == 0:
        return {"n": 0}
    a = v.to_numpy()
    return {
        "n": int(len(a)),
        "mean": round(float(a.mean()), 3),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "max": int(a.max()),
        "n_zero": int((a == 0).sum()),
        "n_1_2": int(((a >= 1) & (a <= 2)).sum()),
        "n_3_5": int(((a >= 3) & (a <= 5)).sum()),
        "n_6_11": int(((a >= 6) & (a <= 11)).sum()),
        "n_ge_12": int((a >= 12).sum()),
    }


def _hit_stats(ev: pl.DataFrame) -> dict:
    n = ev.height
    sc = ev.filter(pl.col("scored"))
    k = sc.height
    hit = int(sc["alert"].sum() or 0)
    base = int(sc["baseline_alert"].sum() or 0)
    return {
        "n_events": n,
        "n_scored": k,
        "n_unscored": n - k,
        "unscored_share": _rate(n - k, n),
        "hits": hit,
        "hit_rate": _rate(hit, k),
        "hit_rate_unscored_as_miss": _rate(hit, n),
        "baseline_hits": base,
        "baseline_hit_rate": _rate(base, k),
        "hit_minus_baseline": (hit - base) / k if k else None,
    }


def outcome_stats(
    scores: pl.DataFrame,
    life: pl.DataFrame,
    period: str,
    horizon_months: int = 6,
    *,
    panel: ep.Panel,
    include_maturity: bool = False,
    chain_end: date | None = None,
) -> dict:
    """사건과 점수를 이어 유형별(domestic·foreign) 통계를 낸다(§11.3). 구간 보호가 걸려 있다.

    돌려주는 dict: ``events``(사건 표 DataFrame), ``by_region``(유형별 통계 dict),
    ``n_events_out_of_period``. 통계 칸: 사건 수·점수 있음/없음·비율·적중률·점수 없는 사건을 미적중으로
    센 적중률·기준선 적중률·차이·FAR(점수·기준선)·선행 분포(I20)·전반/후반(I08).
    ``include_maturity``(기록용)면 만기형도 사건에 넣는다(점수도 ``include_maturity=True`` 로 만든 것을 넘길 것).
    ``chain_end``(I26)는 선행 분포의 경보 사슬이 쓰는 점수의 끝 월말, 기본 ``CHAIN_END[period]``.
    """
    lo, hi = _guard(period)
    events = _events_scored(scores, life, panel, period, horizon_months, include_maturity, chain_end)
    far = far_months(scores, life, panel, period)
    me = panel.month_ends["date"].to_list()
    all_ev = event_etfs(life, include_maturity)
    n_all = {
        r: int((all_ev["region"] == r).sum()) for r in REGIONS
    }
    by_region: dict = {}
    for reg in REGIONS:
        e = events.filter(pl.col("region") == reg)
        st = _hit_stats(e)
        st["n_events_out_of_period"] = n_all[reg] - e.height
        f = far.filter(pl.col("region") == reg)
        nf = f.height
        a = int(f["alert"].sum() or 0)
        b = int(f["baseline_alert"].sum() or 0)
        st["far"] = {"n_etf_months": nf, "n_etfs": f["isu_cd"].n_unique(), "n_alert": a, "far": _rate(a, nf)}
        st["baseline_far"] = {"n_alert": b, "far": _rate(b, nf)}
        st["lead"] = _lead_dist(e["lead_months"])
        st["lead_censored_by_period_n"] = int(e["lead_censored_by_period"].sum() or 0)
        halves = {}
        for name, (y0, y1) in HALVES[period].items():
            h = e.filter(pl.col("formation_month_end").dt.year().is_between(y0, y1))
            hs = _hit_stats(h)
            halves[name] = {
                "years": [y0, y1],
                "n_events": hs["n_events"],
                "n_scored": hs["n_scored"],
                "hits": hs["hits"],
                "hit_rate": hs["hit_rate"],
            }
        st["halves"] = halves
        st["halves_note"] = "규정(I08)" if period == "judgment" else "개발 구간에는 규정 없음, 기록용"
        st["g1_scored_ge_30"] = st["n_scored"] >= 30
        st["unscored_gt_cap"] = (st["unscored_share"] or 0.0) > UNSCORED_SHARE_CAP
        by_region[reg] = st
    return {
        "period": period,
        "horizon_months": horizon_months,
        "formation_range": [str(lo), str(hi)],
        "events": events,
        "far_rows": far,
        "by_region": by_region,
    }


# ---------------------------------------------------------------- 부트스트랩(I14)
@dataclass
class BootResult:
    """반복마다의 통계 배열. 키: ``hit``, ``far``, ``base_hit``, ``base_far``, ``diff``."""

    region: str
    b: int
    seed: int
    arrays: dict[str, np.ndarray]


def p_value(boot_hit: np.ndarray, null: float = HIT_NULL) -> float:
    """I02·I25: (부트스트랩 적중률 ≤ null 개수 + 1) ÷ (B + 1). nan 회차(점수 있는 사건이 하나도 안 뽑힌 회차)는
    분자와 B에서 모두 뺀다(E2의 ``p_value`` 와 같은 방식)."""
    a = np.asarray(boot_hit, dtype=float)
    a = a[~np.isnan(a)]
    return float(((a <= null).sum() + 1) / (len(a) + 1))


def lower_bound(arr: np.ndarray, alpha: float) -> float:
    """단측 하한: 부트스트랩 분포의 α 분위수(nan 제외)."""
    return float(np.nanquantile(np.asarray(arr, dtype=float), alpha))


def bootstrap(
    scores: pl.DataFrame,
    life: pl.DataFrame,
    period: str,
    horizon_months: int = 6,
    *,
    panel: ep.Panel,
    b: int = BOOTSTRAP_B,
    seed: int = BOOTSTRAP_SEED,
    include_maturity: bool = False,
) -> dict[str, BootResult]:
    """ETF 복원 추출 b회(I14). 유형별로 따로, 같은 난수 흐름(domestic → foreign)으로 돌린다.
    ETF 집합 = 그 유형의 점수 있는 사건 ETF ∪ FAR 분모에 한 달이라도 있는 ETF.
    적중률은 뽑힌 사건 ETF로, FAR는 뽑힌 ETF의 ETF-월로 센다. 판정 구간 보호가 걸려 있다."""
    _guard(period)
    events = _events_scored(scores, life, panel, period, horizon_months, include_maturity)
    far = far_months(scores, life, panel, period)
    rng = np.random.default_rng(seed)
    out: dict[str, BootResult] = {}
    for reg in REGIONS:
        e = events.filter((pl.col("region") == reg) & pl.col("scored"))
        f = far.filter(pl.col("region") == reg)
        ids = sorted(set(e["isu_cd"].to_list()) | set(f["isu_cd"].to_list()))
        n = len(ids)
        arrays = {k: np.full(b, np.nan) for k in ("hit", "far", "base_hit", "base_far", "diff")}
        if n == 0:
            out[reg] = BootResult(reg, b, seed, arrays)
            continue
        pos = {i: k for k, i in enumerate(ids)}
        sc = np.zeros(n)
        hit = np.zeros(n)
        bh = np.zeros(n)
        for r in e.iter_rows(named=True):
            k = pos[r["isu_cd"]]
            sc[k] += 1
            hit[k] += 1 if r["alert"] else 0
            bh[k] += 1 if r["baseline_alert"] else 0
        fa = f.group_by("isu_cd").agg(
            pl.len().alias("n"), pl.col("alert").sum().alias("a"), pl.col("baseline_alert").sum().alias("b")
        )
        fn = np.zeros(n)
        fal = np.zeros(n)
        fb = np.zeros(n)
        for r in fa.iter_rows(named=True):
            k = pos[r["isu_cd"]]
            fn[k], fal[k], fb[k] = r["n"], r["a"], r["b"]
        draws = rng.integers(0, n, size=(b, n))
        with np.errstate(invalid="ignore", divide="ignore"):
            for it in range(b):
                w = np.bincount(draws[it], minlength=n).astype(float)
                d_sc = (w * sc).sum()
                d_fn = (w * fn).sum()
                if d_sc > 0:
                    arrays["hit"][it] = (w * hit).sum() / d_sc
                    arrays["base_hit"][it] = (w * bh).sum() / d_sc
                if d_fn > 0:
                    arrays["far"][it] = (w * fal).sum() / d_fn
                    arrays["base_far"][it] = (w * fb).sum() / d_fn
        arrays["diff"] = arrays["hit"] - arrays["base_hit"]
        out[reg] = BootResult(reg, b, seed, arrays)
    return out


def _boot_summary(br: BootResult) -> dict:
    res: dict = {"b": br.b, "seed": br.seed}
    for k, a in br.arrays.items():
        nn = int(np.isnan(a).sum())
        if nn == len(a):
            res[k] = {"n_nan": nn}
            continue
        res[k] = {
            "n_nan": nn,
            "mean": float(np.nanmean(a)),
            "ci95": [lower_bound(a, 0.025), float(np.nanquantile(a, 0.975))],
            "lower_alpha_0.05": lower_bound(a, 0.05),
            "lower_alpha_0.025": lower_bound(a, 0.025),
        }
    res["p_hit_le_0.30"] = p_value(br.arrays["hit"])
    res["p_n_nan_rounds"] = int(np.isnan(br.arrays["hit"]).sum())  # I25: p값에서 빠진 회차 수
    res["p_b_used"] = int((~np.isnan(br.arrays["hit"])).sum())
    return res


# ---------------------------------------------------------------- CLI
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


def pool_size_summary(scores: pl.DataFrame, lo: date, hi: date) -> list[dict]:
    """월말별 풀 크기를 연·유형별로 요약(최소·중앙·최대)."""
    p = (
        scores.filter(pl.col("in_pool") & (pl.col("month_end") >= lo) & (pl.col("month_end") <= hi))
        .group_by("month_end", "region")
        .agg(pl.len().alias("pool"), pl.col("active").sum().alias("n_active"))
        .with_columns(pl.col("month_end").dt.year().alias("year"))
    )
    return (
        p.group_by("year", "region")
        .agg(
            pl.len().alias("n_months"),
            pl.col("pool").min().alias("min"),
            pl.col("pool").median().alias("median"),
            pl.col("pool").max().alias("max"),
            pl.col("n_active").max().alias("max_active"),
        )
        .sort(["region", "year"])
        .to_dicts()
    )


def _json_ready(o):
    if isinstance(o, dict):
        return {str(k): _json_ready(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_ready(v) for v in o]
    if isinstance(o, (date, datetime)):
        return str(o)
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    root = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
    ap.add_argument("--period", choices=["dev"], required=True, help="이 CLI는 개발 구간만 돈다")
    ap.add_argument("--input", default=str(root / ep.DEFAULT_INPUT_REL))
    ap.add_argument("--out-dir", default=str(root / DEFAULT_OUT_REL))
    ap.add_argument("--interp-table", default=os.environ.get("QUALITY_E_INTERP_TABLE", DEFAULT_INTERP_TABLE))
    a = ap.parse_args(argv)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel = ep.read_panel(a.input)
    life = ep.lifecycle(panel)
    scores = monthly_scores(panel, life)
    lo, hi = PERIODS["dev"]
    dev_scores = scores.filter((pl.col("month_end") >= lo) & (pl.col("month_end") <= hi))
    dev_scores.write_parquet(out / "scores_dev.parquet")

    summary: dict = {
        "period": "dev",
        "formation_range": [str(lo), str(hi)],
        "last_month_end": str(LAST_MONTH_END),
        "pool_sizes_dev": pool_size_summary(scores, lo, hi),
        "horizons": {},
        "bootstrap": {},
    }
    ev_all = []
    for h in LEAD_HORIZONS:
        st = outcome_stats(dev_scores, life, "dev", h, panel=panel)
        ev_all.append(st["events"])
        summary["horizons"][str(h)] = {
            "by_region": st["by_region"],
        }
        bt = bootstrap(dev_scores, life, "dev", h, panel=panel)
        summary["bootstrap"][str(h)] = {reg: _boot_summary(br) for reg, br in bt.items()}
    pl.concat(ev_all).write_csv(out / "events_dev.csv")
    (out / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2)
    )

    src = Path(__file__)
    status = _git("status", "--porcelain")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": "quality-score v0 E1 개발 구간 점수·결과 통계(사전등록 §11, 해석 표 v1). 판정 구간은 돌리지 않음",
        "input": {"path": str(a.input), "sha256": _sha256(a.input)},
        "interp_table": {"path": str(a.interp_table), "sha256": _sha256(a.interp_table)},
        "module": {
            "version": E1_VERSION,
            "etf_e1_sha256": _sha256(src),
            "etf_panel_sha256": _sha256(src.parent / "etf_panel.py"),
        },
        "code": {
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "HEAD"),
            "git_status_porcelain": status.splitlines(),
        },
        "constants": {
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_b": BOOTSTRAP_B,
            "alert_pct": ALERT_PCT,
            "hit_null": HIT_NULL,
            "corr_window": CORR_WINDOW_SESSIONS,
            "corr_min_pairs": CORR_MIN_PAIRS,
        },
        "runtime": {"python": sys.version.split()[0], "polars": pl.__version__},
        "judgment_env_set": bool(os.environ.get(JUDGMENT_ENV, "").strip()),
        "outputs": sorted(p.name for p in out.glob("*") if p.name != "manifest.json"),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps(_json_ready(summary["horizons"]["6"]["by_region"]), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
