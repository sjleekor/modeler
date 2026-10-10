"""MRS 장부·지표·placebo·게이트·등급 (사전등록 §5.3~§5.5, §0-2, §0-3, §6).

실제 데이터를 읽지 않는다. 평범한 배열(날짜, 가격, MRS, 하위 점수, sigma, 현금 수익)을 받는다.

**장부** (격자 인덱스 ``i``, 결정일 ``g_i``)::

    w_i     = 비중 규칙(MRS_i)             (null -> 1.0)
    r_bh_i  = P_{i+1}/P_i - 1              (g_i 종가 -> g_{i+1} 종가, 비용 없음)
    r_cash_i= 현금 계정의 같은 구간 수익   (불가면 NaN)
    a_i     = w_{i-1}                      (g_{i-1}에 정해 g_i 종가 체결; a_0 = 1.0, MI33)
    r_i     = a_i*r_bh_i + (1-a_i)*r_cash_i - |a_i - a_{i-1}| * RT/2/1e4

비용 ``|a_i - a_{i-1}|``는 체결이 일어나는 ``i``에서 낸다 (§5.3).
변동성 관리·바닥 없는 기준선도 같은 식이다.

**구간** (MI14): ``t0`` = ``start`` 이상 격자일 중 MRS가 처음 non-null인 인덱스,
``S = {i : i >= t0+1, g_{i+1} <= end}`` 에서 ``r_bh``·``r_cash``가 NaN인 세션(그리고 호출자가 준
``availability_ok`` 거짓 세션, MI34)을 **모든 규칙에서 같이** 뺀다 (§0-3.1). 비용은 전체 격자에서
식대로 낸 뒤 세션을 빼므로 뺀 세션의 비용도 같이 빠진다 (MI15).

**HAC t** (MI18): ``metrics.newey_west_ols`` (gap-aware Bartlett, lag 20, 격자 인덱스 간격 기준)의
``t_delta``. lag 0이면 White(HC0) 샌드위치와 같다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

from modeler.etl.metrics import newey_west_ols
from modeler.scores.mrs import config as C
from modeler.scores.mrs import weights as W

RULES = ("mrs", "vm", "vm_nofloor")
#: 순서대로 처음 실패한 게이트의 등급 문자 (§5.5).
GATE_LETTERS = ("D", "C", "R", "S", "X")
#: 점수 구간(Q) 길이: t+1 종가 진입 -> t+21 종가 (MI18).
Q_PATH_LEN = C.FORWARD_HORIZON + 1

RESULT_COLUMNS = (
    "rule",
    "MDD_ratio",
    "RET_ratio",
    "CAGR_gap",
    "Q1_Q5_gap",
    "t_gap",
    "Q1_MDD",
    "Q5_MDD",
    "Calmar_ratio",
    "turnover",
    "sub_gap_T",
    "sub_gap_V",
    "sub_gap_L",
    "placebo_p",
    "g1",
    "g2",
    "g3",
    "g4",
    "g5",
    "g5_vs_nofloor",
    "gate_class",
    "grade",
    "w_fix_mean",
    "MDD_ratio_fix",
    "RET_ratio_fix",
    "Calmar_ratio_fix",
    "beats_fix",
    "MDD",
    "CAGR",
    "Calmar",
    "MDD_denom",
    "CAGR_denom",
    "Calmar_denom",
    "MDD_fix",
    "CAGR_fix",
    "n_sessions",
    "n_excluded",
    "window_start",
    "window_end",
    "t0_date",
    "first_session",
    "last_session",
)


# --------------------------------------------------------------------------- 작은 도구
def _nn(x: Any) -> Any:
    """NaN/inf -> None, numpy 스칼라 -> 파이썬 값."""
    if x is None:
        return None
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, float, np.integer, int)):
        f = float(x)
        return f if math.isfinite(f) else None
    return x


def applied_weights(w: np.ndarray, a0: float = 1.0) -> np.ndarray:
    """``a_i = w_{i-1}``, ``a_0 = a0`` (MI33 — 첫 세션 이전에는 보유 100%, IRP는 상한 값)."""
    a = np.empty_like(np.asarray(w, dtype=float))
    a[0] = a0
    a[1:] = w[:-1]
    return a


def rule_returns(
    a: np.ndarray, r_bh: np.ndarray, r_cash: np.ndarray, cost_oneway: float
) -> np.ndarray:
    """``a_i*r_bh_i + (1-a_i)*r_cash_i - |a_i - a_{i-1}|*cost`` (인덱스 0은 비용 0)."""
    step = np.zeros_like(a)
    step[1:] = np.abs(a[1:] - a[:-1])
    return a * r_bh + (1.0 - a) * r_cash - step * cost_oneway


def cost_oneway(rt_bp: float) -> float:
    """왕복 bp -> 편도 비율 (§5.3)."""
    return rt_bp / 2.0 / 1e4


def forward_returns(price: np.ndarray) -> np.ndarray:
    """``r_bh_i = P_{i+1}/P_i - 1``. 마지막 원소와 가격 NaN인 곳은 NaN."""
    p = np.asarray(price, dtype=float)
    out = np.full(p.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:-1] = p[1:] / p[:-1] - 1.0
    return out


def cash_returns(account: Any, n: int) -> np.ndarray:
    """``CashAccount`` -> 격자 구간 현금 수익 배열(마지막 NaN). ``interval_return`` 재사용."""
    out = np.full(n, np.nan)
    if n > 1:
        ret, _ = account.interval_return(np.arange(n - 1), np.arange(1, n))
        out[:-1] = ret
    return out


def zero_cash(n: int) -> np.ndarray:
    """``sens_cash0``: 모든 규칙의 ``r_cash = 0`` (마지막 구간은 가격이 없어 어차피 빠진다)."""
    return np.zeros(n)


def placebo_shifts() -> list[int]:
    """시프트 50개: 시드 ``PLACEBO_SEED``, [20, 1000]에서 비복원 추출 후 정렬 (MI21)."""
    rng = np.random.default_rng(C.PLACEBO_SEED)
    pool = np.arange(C.PLACEBO_SHIFT_MIN, C.PLACEBO_SHIFT_MAX + 1)
    return sorted(int(k) for k in rng.choice(pool, size=C.PLACEBO_N, replace=False))


# --------------------------------------------------------------------------- 장부
@dataclass(frozen=True)
class Ledger:
    """격자 전체의 규칙별 비중·수익 (장부 parquet 입력). 배열 길이는 모두 격자 길이다."""

    dates: list[date]
    r_bh: np.ndarray
    r_cash: np.ndarray
    r_den: np.ndarray  # 분모 수익: r_bh, IRP protocol이면 r_bh_cap
    w: dict[str, np.ndarray]  # 규칙별 결정 비중 (상한 적용 후)
    a: dict[str, np.ndarray]  # 규칙별 적용 비중
    r: dict[str, np.ndarray]  # 규칙별 수익 (비용 반영)
    cost_oneway: float

    def to_frame(self) -> pl.DataFrame:
        cols: dict[str, Any] = {"date": self.dates, "r_bh": self.r_bh, "r_cash": self.r_cash}
        for k in RULES:
            cols[f"w_{k}"] = self.w[k]
            cols[f"a_{k}"] = self.a[k]
            cols[f"r_{k}"] = self.r[k]
        cols["r_den"] = self.r_den
        return pl.DataFrame(cols, nan_to_null=True)


def build_ledger(
    *,
    dates: Sequence[date],
    price: np.ndarray,
    mrs: np.ndarray,
    sigma20: np.ndarray,
    sigma_target: np.ndarray,
    r_cash: np.ndarray,
    cost_rt_bp: float,
    irp_cap: float | None = None,
) -> Ledger:
    """규칙 셋(mrs / vm / vm_nofloor)의 장부를 만든다.

    ``irp_cap``이 있으면 (§0-3.2) 세 비중 모두 ``min(w, cap)``이고 분모는
    ``r_bh_cap = cap*r_bh + (1-cap)*r_cash``다.
    """
    r_bh = forward_returns(price)
    rc = np.asarray(r_cash, dtype=float)
    w = {
        "mrs": W.mrs_weight(mrs),
        "vm": W.vm_weight(sigma20, sigma_target, floor=True),
        "vm_nofloor": W.vm_weight(sigma20, sigma_target, floor=False),
    }
    if irp_cap is not None:
        w = {k: W.cap(v, irp_cap) for k, v in w.items()}
        r_den = irp_cap * r_bh + (1.0 - irp_cap) * rc
    else:
        r_den = r_bh
    c1 = cost_oneway(cost_rt_bp)
    a0 = 1.0 if irp_cap is None else min(1.0, irp_cap)
    a = {k: applied_weights(v, a0) for k, v in w.items()}
    r = {k: rule_returns(a[k], r_bh, rc, c1) for k in RULES}
    return Ledger(list(dates), r_bh, rc, r_den, w, a, r, c1)


# --------------------------------------------------------------------------- 구간·경로 지표
def window_session_set(
    dates: Sequence[date], mrs: np.ndarray, start: date, end: date
) -> tuple[int | None, np.ndarray]:
    """``(t0, S)``. ``S = {i : i >= t0+1, i <= n-2, g_{i+1} <= end}`` (MI14)."""
    n = len(dates)
    m = np.asarray(mrs, dtype=float)
    t0 = next((i for i in range(n) if dates[i] >= start and np.isfinite(m[i])), None)
    if t0 is None:
        return None, np.empty(0, dtype=int)
    idx = np.array([i for i in range(t0 + 1, n - 1) if dates[i + 1] <= end], dtype=int)
    return t0, idx


def path_metrics(r: np.ndarray, first_date: date, last_next_date: date) -> dict[str, float | None]:
    """수익 열 ``r``(이미 세션을 뺀 것)의 ``MDD``·``CAGR``·``Calmar`` (§0-3.1, MI17).

    ``V_0 = 1``이 러닝 최댓값에 들어간다. ``CAGR = V_end^(1/Y) - 1``,
    ``Y = (last_next_date - first_date).days / 365.25`` (달력 기준).
    """
    rr = np.asarray(r, dtype=float)
    if rr.size == 0:
        return {"MDD": None, "CAGR": None, "Calmar": None}
    v = np.concatenate([[1.0], np.cumprod(1.0 + rr)])
    v = np.maximum(v, 0.0)
    peak = np.maximum.accumulate(v)
    mdd = float(np.max(1.0 - v / peak))
    years = (last_next_date - first_date).days / 365.25
    cagr = float(v[-1] ** (1.0 / years) - 1.0) if years > 0 else None
    calmar = cagr / mdd if (cagr is not None and mdd > 0) else None
    return {"MDD": mdd, "CAGR": cagr, "Calmar": calmar}


def _ratios(m: Mapping[str, Any], den: Mapping[str, Any]) -> dict[str, float | None]:
    """전략 대 분모 비율. 분모 CAGR <= 0이면 ``RET_ratio`` null (G2는 CAGR 직접 비교)."""
    mdd_ratio = m["MDD"] / den["MDD"] if m["MDD"] is not None and den["MDD"] else None
    ret_ratio = (
        m["CAGR"] / den["CAGR"]
        if m["CAGR"] is not None and den["CAGR"] is not None and den["CAGR"] > 0
        else None
    )
    calmar_ratio = (
        m["Calmar"] / den["Calmar"]
        if m["Calmar"] is not None and den["Calmar"] is not None and den["Calmar"] > 0
        else None
    )
    return {"MDD_ratio": mdd_ratio, "RET_ratio": ret_ratio, "Calmar_ratio": calmar_ratio}


# --------------------------------------------------------------------------- Q 통계
def _hac_t(f: np.ndarray, d: np.ndarray, idx: np.ndarray, lag: int = C.HAC_LAG) -> float | None:
    """``f = alpha + beta*d`` OLS의 beta HAC t (Bartlett, gap-aware). NaN -> None."""
    out = newey_west_ols(f, d, idx, lag)
    return _nn(out["t_delta"])


def _max_drawdown_rows(paths: np.ndarray) -> np.ndarray:
    """행마다 경로(시작 포함)의 최대 낙폭."""
    peak = np.maximum.accumulate(paths, axis=1)
    return np.max(1.0 - paths / peak, axis=1)


def quintile_stats(
    price: np.ndarray,
    score: np.ndarray,
    dates: Sequence[date],
    t0: int,
    end: date,
    *,
    with_mdd: bool = False,
) -> dict[str, float | None]:
    """``Q1_Q5_gap``·``t_gap`` (그리고 ``Q1_MDD``·``Q5_MDD``). 사양 4.2 (MI18, MI19, MI20).

    결정일 t: ``t >= t0``, ``g_{t+21} <= end``, 점수 non-null,
    ``P_{t+1..t+21}`` 모두 유한·양수 (MI36).
    ``f_t = P_{t+21}/P_{t+1} - 1``. 분위는 이 표본의 ``np.quantile`` 20%·80%(linear)이고
    ``Q1 = 점수 <= q20``, ``Q5 = 점수 >= q80``. ``q20 >= q80``이거나 한쪽이 비면 null (MI20).
    """
    out: dict[str, float | None] = {
        "gap": None,
        "t": None,
        "Q1_MDD": None,
        "Q5_MDD": None,
        "n_q1": 0,
        "n_q5": 0,
    }
    p = np.asarray(price, dtype=float)
    s = np.asarray(score, dtype=float)
    n = len(p)
    if n < Q_PATH_LEN + 1:
        return out
    wins = sliding_window_view(p, Q_PATH_LEN)  # wins[j] = P[j..j+20]
    # 결정일 t의 경로 = P[t+1..t+21] = wins[t+1]
    cand = np.arange(t0, n - Q_PATH_LEN)
    cand = cand[[dates[t + Q_PATH_LEN] <= end for t in cand]] if cand.size else cand
    if cand.size == 0:
        return out
    paths = wins[cand + 1]
    # Q 표본은 경로 전체가 유한하고 양수여야 한다 (MI36)
    ok = np.isfinite(s[cand]) & np.all(np.isfinite(paths), axis=1) & np.all(paths > 0, axis=1)
    cand, paths = cand[ok], paths[ok]
    if cand.size < 2:
        return out
    sc = s[cand]
    f = paths[:, -1] / paths[:, 0] - 1.0
    q20, q80 = np.quantile(sc, [C.QUINTILE, 1.0 - C.QUINTILE])
    if not q20 < q80:
        return out
    q1, q5 = sc <= q20, sc >= q80
    out["n_q1"], out["n_q5"] = int(q1.sum()), int(q5.sum())
    if not (q1.any() and q5.any()):
        return out
    out["gap"] = float(f[q1].mean() - f[q5].mean())
    sel = q1 | q5
    out["t"] = _hac_t(f[sel], q1[sel].astype(float), cand[sel])
    if with_mdd:
        mdd = _max_drawdown_rows(paths)
        out["Q1_MDD"] = float(mdd[q1].mean())
        out["Q5_MDD"] = float(mdd[q5].mean())
    return out


# --------------------------------------------------------------------------- placebo
def placebo_mdds(
    a_s: np.ndarray,
    r_bh_s: np.ndarray,
    r_cash_s: np.ndarray,
    cost1: float,
    shifts: Sequence[int],
) -> np.ndarray:
    """세션열 ``S`` 안에서 적용 비중 수열을 원형으로 ``k``칸 민 경로들의 MDD.

    ``a'_j = a_{(j-k) mod |S|}``. 수익·현금은 그대로, 비용은 ``|a'_j - a'_{j-1}|``로 다시 낸다
    (``j=0``의 앞은 ``a'_{|S|-1}``, MI21).
    """
    m = len(a_s)
    out = np.empty(len(shifts))
    for n_k, k in enumerate(shifts):
        ap = np.roll(a_s, k % m)
        step = np.abs(ap - np.roll(ap, 1))
        r = ap * r_bh_s + (1.0 - ap) * r_cash_s - step * cost1
        v = np.concatenate([[1.0], np.cumprod(1.0 + r)])
        v = np.maximum(v, 0.0)
        out[n_k] = np.max(1.0 - v / np.maximum.accumulate(v))
    return out


def placebo_p(mdd_actual: float, mdd_placebo: np.ndarray) -> float:
    """``(1 + #{placebo <= 실제}) / (n + 1)`` — 낮을수록 좋은 통계라 ``<=`` (MI21)."""
    return (1.0 + float(np.count_nonzero(mdd_placebo <= mdd_actual))) / (len(mdd_placebo) + 1.0)


# --------------------------------------------------------------------------- 게이트·등급
def _ret_better(a: Mapping[str, Any], b: Mapping[str, Any], *, strict: bool) -> bool | None:
    """RET 비교. 분모 CAGR > 0이면 ``RET_ratio``, 아니면 CAGR 직접 비교 (§5.4 G5)."""
    if a.get("RET_ratio") is not None and b.get("RET_ratio") is not None:
        x, y = a["RET_ratio"], b["RET_ratio"]
    elif a.get("CAGR_denom") is not None and a["CAGR_denom"] <= 0:
        x, y = a.get("CAGR"), b.get("CAGR")
    else:
        return None
    if x is None or y is None:
        return None
    return x > y if strict else x >= y


def beats(a: Mapping[str, Any], b: Mapping[str, Any], *, suffix_b: str = "") -> bool:
    """``a``가 ``b``보다 ``MDD_ratio`` 작고 **동시에** ``RET`` 큰가 (G5, ``beats_fix`` 공용).

    ``suffix_b="_fix"``면 ``b``의 칸 이름에 ``_fix``를 붙여 읽는다.
    """
    mdd_a = a.get("MDD_ratio")
    mdd_b = b.get("MDD_ratio" + suffix_b)
    if mdd_a is None or mdd_b is None:
        return False
    bb = dict(b)
    if suffix_b:
        bb = {
            "RET_ratio": b.get("RET_ratio" + suffix_b),
            "CAGR": b.get("CAGR" + suffix_b),
            "CAGR_denom": b.get("CAGR_denom"),
        }
    r = _ret_better(a, bb, strict=True)
    return bool(mdd_a < mdd_b and r)


def g1(mdd_ratio: float | None) -> bool:
    return mdd_ratio is not None and mdd_ratio <= C.G1_MDD_RATIO_MAX


def g2(row: Mapping[str, Any]) -> bool:
    """RET_ratio >= 0.90. 분모 CAGR <= 0이면 ``CAGR(전략) >= CAGR(분모)``."""
    if row.get("CAGR_denom") is not None and row["CAGR_denom"] <= 0:
        return row.get("CAGR") is not None and row["CAGR"] >= row["CAGR_denom"]
    return row.get("RET_ratio") is not None and row["RET_ratio"] >= C.G2_RET_RATIO_MIN


def g3(p: float | None) -> bool:
    return p is not None and p < C.G3_P_MAX


def g4(sub_gaps: Sequence[float | None]) -> bool:
    """살아 있는(null 아닌) 하위 점수 중 ``sub_gap < 0``인 것이 2개 이상."""
    return sum(1 for g in sub_gaps if g is not None and g < 0) >= C.G4_MIN_CORRECT_SIGN


def gate_class(gates: Sequence[bool | None]) -> str:
    """처음 실패한 게이트: g1 D, g2 C, g3 R, g4 S, g5 X. 다 통과하면 ``PASS`` (§5.5)."""
    for ok, letter in zip(gates, GATE_LETTERS, strict=True):
        if not ok:
            return letter
    return "PASS"


def official_grade(
    kr_main: Mapping[str, Any] | None,
    us_explore: Mapping[str, Any] | None,
    kr_explore: Mapping[str, Any] | None,
) -> str | None:
    """공식 등급 (§5.5, MI22). 세 인자는 각각 해당 구간 KOSPI/SPY ``main_close_cash``의 ``mrs`` 행.

    한국 주 판정 ``gate_class``가 PASS가 아니면 그 문자(D/C/R/S/X).
    PASS면 미국 탐색 g1·g2가 둘 다 참 **그리고** 한국 2010~ 탐색 ``MDD_ratio < 1``이면 ``A``,
    아니면 ``B``. 행이 없으면 미달로 본다 (MI35).
    """
    if kr_main is None or kr_main.get("gate_class") is None:
        return None
    cls = kr_main["gate_class"]
    if cls != "PASS":
        return cls
    us_ok = bool(us_explore and us_explore.get("g1") and us_explore.get("g2"))
    mdd = kr_explore.get("MDD_ratio") if kr_explore else None
    kr_ok = mdd is not None and mdd < 1.0
    return "A" if us_ok and kr_ok else "B"


# --------------------------------------------------------------------------- 창 평가
def evaluate_window(
    *,
    dates: Sequence[date],
    price: np.ndarray,
    mrs: np.ndarray,
    sub_scores: Mapping[str, np.ndarray],
    sigma20: np.ndarray,
    sigma_target: np.ndarray,
    r_cash: np.ndarray,
    start: date,
    end: date,
    cost_rt_bp: float,
    irp_cap: float | None = None,
    with_gates: bool = True,
    availability_ok: np.ndarray | None = None,
    meta: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """한 (구간, 자산, protocol)의 규칙별 행 3개(``mrs``, ``vm``, ``vm_nofloor``)를 낸다.

    * ``r_cash``: 격자 구간 현금 수익 배열. ``sens_cash0``는 :func:`zero_cash`, 합성 국고채
      protocol은 ``synth_ktb.synth_returns`` 결과를 그대로 넘긴다.
    * ``with_gates=False``: ``placebo_p``·``g1..g5``·``gate_class``·``grade``를 null로 둔다
      (``ktb_synth_cash_sens``, MI25).
    * ``availability_ok``: 격자 인덱스별 불리언. 거짓인 세션은 검정에서 뺀다 (§11, MI34).
    * ``meta``: 행마다 그대로 붙일 식별 칸(``market, asset, protocol`` 등).
    * 결과 행 칸은 :data:`RESULT_COLUMNS`. ``vm`` 계열 행에는 ``g4``·``g5``·Q 통계·고정 비중 칸이
      null이다 (MI23).
      ``grade``는 항상 null이다 — 공식 등급은 :func:`official_grade`로 호출자가 ``mrs`` 행에 채운다.
    """
    price = np.asarray(price, dtype=float)
    led = build_ledger(
        dates=dates,
        price=price,
        mrs=mrs,
        sigma20=sigma20,
        sigma_target=sigma_target,
        r_cash=r_cash,
        cost_rt_bp=cost_rt_bp,
        irp_cap=irp_cap,
    )
    t0, idx = window_session_set(dates, mrs, start, end)
    if t0 is None or idx.size == 0:
        raise ValueError("구간에 점수가 있는 세션이 없다")
    good = np.isfinite(led.r_bh[idx]) & np.isfinite(led.r_cash[idx])
    if availability_ok is not None:
        good &= np.asarray(availability_ok, dtype=bool)[idx]
    s_idx = idx[good]
    n_excluded = int(idx.size - s_idx.size)
    if s_idx.size == 0:
        raise ValueError("뺀 뒤 남은 세션이 없다")
    first_d, last_next = dates[int(s_idx[0])], dates[int(s_idx[-1]) + 1]

    den_m = path_metrics(led.r_den[s_idx], first_d, last_next)
    base = {
        "n_sessions": int(s_idx.size),
        "n_excluded": n_excluded,
        "window_start": start,
        "window_end": end,
        "t0_date": dates[t0],
        "first_session": first_d,
        "last_session": dates[int(s_idx[-1])],
        "MDD_denom": den_m["MDD"],
        "CAGR_denom": den_m["CAGR"],
        "Calmar_denom": den_m["Calmar"],
    }

    shifts = placebo_shifts()
    do_placebo = with_gates and s_idx.size > C.PLACEBO_SHIFT_MAX
    rows: dict[str, dict[str, Any]] = {}
    for rule in RULES:
        m = path_metrics(led.r[rule][s_idx], first_d, last_next)
        a_s = led.a[rule][s_idx]
        step = np.abs(led.a[rule][s_idx] - led.a[rule][s_idx - 1])
        row: dict[str, Any] = {k: None for k in RESULT_COLUMNS}
        row.update(base)
        row["rule"] = rule
        row.update({"MDD": m["MDD"], "CAGR": m["CAGR"], "Calmar": m["Calmar"]})
        row.update(_ratios(m, den_m))
        row["CAGR_gap"] = (
            den_m["CAGR"] - m["CAGR"]
            if den_m["CAGR"] is not None and m["CAGR"] is not None
            else None
        )
        row["turnover"] = float(step.mean() * C.TURNOVER_ANNUALIZE)
        if with_gates:
            if do_placebo and m["MDD"] is not None:
                pm = placebo_mdds(a_s, led.r_bh[s_idx], led.r_cash[s_idx], led.cost_oneway, shifts)
                row["placebo_p"] = placebo_p(m["MDD"], pm)
            row["g1"] = g1(row["MDD_ratio"])
            row["g2"] = g2(row)
            row["g3"] = g3(row["placebo_p"])
        rows[rule] = row

    # --- mrs 행 전용: Q 통계, 고정 비중, G4·G5
    mr = rows["mrs"]
    q = quintile_stats(price, mrs, dates, t0, end, with_mdd=True)
    mr.update(
        {"Q1_Q5_gap": q["gap"], "t_gap": q["t"], "Q1_MDD": q["Q1_MDD"], "Q5_MDD": q["Q5_MDD"]}
    )
    subs: list[float | None] = []
    for key in C.SUB_SCORES:
        arr = sub_scores.get(key)
        g = quintile_stats(price, arr, dates, t0, end)["gap"] if arr is not None else None
        mr[f"sub_gap_{key}"] = g
        subs.append(g)

    w_fix = float(led.a["mrs"][s_idx].mean())
    r_fix = w_fix * led.r_bh[s_idx] + (1.0 - w_fix) * led.r_cash[s_idx]  # 비용 0 (§0-2)
    fx = path_metrics(r_fix, first_d, last_next)
    fr = _ratios(fx, den_m)
    mr.update(
        {
            "w_fix_mean": w_fix,
            "MDD_ratio_fix": fr["MDD_ratio"],
            "RET_ratio_fix": fr["RET_ratio"],
            "Calmar_ratio_fix": fr["Calmar_ratio"],
            "MDD_fix": fx["MDD"],
            "CAGR_fix": fx["CAGR"],
        }
    )
    if mr["MDD_ratio"] is not None and fr["MDD_ratio"] is not None:
        mr["beats_fix"] = beats(mr, mr, suffix_b="_fix")

    if with_gates:
        mr["g4"] = g4(subs)
        mr["g5"] = beats(mr, rows["vm"])
        mr["g5_vs_nofloor"] = beats(mr, rows["vm_nofloor"])
        mr["gate_class"] = gate_class([mr["g1"], mr["g2"], mr["g3"], mr["g4"], mr["g5"]])

    out = []
    for rule in RULES:
        row = {k: _nn(v) for k, v in rows[rule].items()}
        if meta:
            row.update(meta)
        out.append(row)
    return out
