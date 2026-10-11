"""ETF E3 — 분배락일 규칙 (사양 §4, D3).

- KRX 기준가 조정일: ``prev_ref_t = TDD_CLSPRC_t − CMPPREVDD_PRC_t`` 가 직전 거래일 종가와 다른 날(둘 다 있을 때).
- 실효 기준일 E = ``RGT_STD_DT`` 가 거래일이면 그날, 아니면 그 직전 거래일.
  후보 R−1(E의 직전 거래일) · R0(E 당일) · R+1(E의 다음 거래일).
- 개발 구간(RGT_STD_DT 2010-01-01~2015-12-31, 금액 > 0, 청산분배 제외)에서 후보마다
  "규칙 날짜가 그 ETF의 KRX 조정일인 분배의 비율"을 센다. 가장 높은 후보를 채택하되 90% 이상이어야 한다.
- NAV 차감일 점검(개발 구간만): 분배금/직전 NAV ≥ 0.3%이고 그 ETF의 개발 구간 NAV 하루 변화 표준편차 < 0.2%인
  분배에서, 규칙 날짜 d와 d−1·d+1 가운데 NAV 하루 변화가 −분배금/직전 NAV 에 가장 가까운 날의 분포.
- 판정 구간에서는 ETF-연 정합성 비율(분배 중 규칙 날짜 = KRX 조정일 비율)만 센다. 값 크기 계산 없음.

**동결할 상수**는 ``FROZEN_RULE`` 와 ``FROZEN_NAV_SHIFT`` 다. 개발 구간 점검(``dev_check``)이 정한 값을
동결 전에 여기 적는다(결과를 보고 규칙을 고르는 일이 아니라, 미리 정한 선택 절차의 결과를 옮겨 적는 것이다).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from modeler.scores.quality.etf_e3_dense import Dense

RULE_SHIFT = {"R-1": -1, "R0": 0, "R+1": 1}
DEV_FROM = np.datetime64("2010-01-01")
DEV_TO = np.datetime64("2015-12-31")
MATCH_MIN = 0.90  # 후보 채택 문턱(처음 고른 값)
NAV_DROP_MIN = 0.003  # 분배금/직전 NAV ≥ 0.3%
NAV_SD_MAX = 0.002  # 개발 구간 NAV 하루 변화 표준편차 < 0.2%
CONSISTENCY_MIN = 0.90  # ETF-연 정합성 문턱(제외 단계 5)

# 개발 구간 점검(2026-10-11 run (a), prefreeze 산출물 참조)으로 채운 동결 값.
FROZEN_RULE = "R-1"
FROZEN_NAV_SHIFT = -1  # NAV 차감일 = 규칙 날짜(R−1)의 하루 앞(개발 구간 164건 중 164건)


def krx_adjust_days(d: Dense) -> np.ndarray:
    """bool (T, N): KRX 기준가 조정일. t=0 은 직전 거래일이 없어 False."""
    adj = np.zeros_like(d.has_row)
    pr = d.prev_ref[1:]
    pc = d.close[:-1]
    ok = np.isfinite(pr) & np.isfinite(pc)
    adj[1:] = ok & (np.abs(pr - pc) > 1e-9)
    return adj


def effective_idx(d: Dense, rgt: np.ndarray) -> np.ndarray:
    """RGT_STD_DT(datetime64[D]) → 실효 기준일 E의 거래일 번호. 달력 앞·뒤 바깥이면 −1."""
    e = np.searchsorted(d.dates, rgt, side="right") - 1
    e = np.where((rgt < d.dates[0]) | (rgt > d.dates[-1]), -1, e)
    return e.astype(np.int64)


def rule_idx(d: Dense, rgt: np.ndarray, rule: str) -> np.ndarray:
    """규칙 날짜의 거래일 번호. 범위를 벗어나면 −1."""
    e = effective_idx(d, rgt)
    r = e + RULE_SHIFT[rule]
    return np.where((e < 0) | (r < 0) | (r >= d.n_days), -1, r)


def prepare_dist(d: Dense, dist: pl.DataFrame) -> pl.DataFrame:
    """분배 표(isu_cd, rgt_std_dt, amount)에 ETF 열 번호(``j``, 없으면 −1)와 RGT 날짜 배열 칸을 붙인다."""
    j = np.array([d.col.get(c, -1) for c in dist["isu_cd"].to_list()], dtype=np.int64)
    return dist.with_columns(pl.Series("j", j))


def _rgt(dist: pl.DataFrame) -> np.ndarray:
    return dist["rgt_std_dt"].to_numpy().astype("datetime64[D]")


def dev_distributions(d: Dense, dist: pl.DataFrame) -> pl.DataFrame:
    """개발 구간 분배: RGT_STD_DT 2010-01-01~2015-12-31. (``dist`` 는 청산분배 제외·금액 > 0 이어야 한다.)"""
    r = _rgt(dist)
    keep = (r >= DEV_FROM) & (r <= DEV_TO)
    return prepare_dist(d, dist.filter(pl.Series(keep)))


def candidate_match_rates(d: Dense, adj: np.ndarray, dev: pl.DataFrame) -> dict:
    """후보 규칙마다 개발 구간 일치 비율. 분모 둘: KRX에 있는 ETF의 분배 / 그중 규칙 날짜가 달력 안인 분배."""
    j = dev["j"].to_numpy()
    r = _rgt(dev)
    known = j >= 0
    out = {"n_dev_distributions": int(len(j)), "n_with_krx_etf": int(known.sum()), "candidates": {}}
    for name in RULE_SHIFT:
        ri = rule_idx(d, r, name)
        inside = known & (ri >= 0)
        match = np.zeros(len(j), dtype=bool)
        match[inside] = adj[ri[inside], j[inside]]
        out["candidates"][name] = {
            "n_match": int(match.sum()),
            "n_inside_calendar": int(inside.sum()),
            "rate_over_krx_etf": float(match.sum() / known.sum()) if known.sum() else float("nan"),
            "rate_over_inside": float(match.sum() / inside.sum()) if inside.sum() else float("nan"),
        }
    return out


def choose_rule(rates: dict) -> dict:
    """가장 높은 후보(``rate_over_krx_etf`` 기준)를 채택, 90% 미만이면 '판정 대기'."""
    best = max(RULE_SHIFT, key=lambda k: rates["candidates"][k]["rate_over_krx_etf"])
    rate = rates["candidates"][best]["rate_over_krx_etf"]
    return {
        "chosen": best if rate >= MATCH_MIN else None,
        "best": best,
        "best_rate": rate,
        "threshold": MATCH_MIN,
    }


def nav_alignment(d: Dense, dev: pl.DataFrame, rule: str) -> dict:
    """NAV 차감일 점검. 저변동 ETF의 개발 구간 분배(분배금/직전 NAV ≥ 0.3%)마다 d−1·d·d+1 중 NAV 하루 변화가
    −분배금/직전 NAV 에 가장 가까운 날을 찾아 d 기준 오프셋 분포를 센다.

    ETF의 NAV 하루 변화 표준편차는 개발 구간(2010~2015) 전체에서 구하되, 그 ETF 분배의 d−1~d+1 이웃 날은 뺀다
    (분배 낙폭이 표준편차를 부풀리지 않게).
    """
    j = dev["j"].to_numpy()
    amt = dev["amount"].to_numpy().astype(float)
    r = _rgt(dev)
    ri = rule_idx(d, r, rule)
    T = d.n_days
    dev_days = (d.dates >= DEV_FROM) & (d.dates <= DEV_TO)
    chg = np.full_like(d.nav, np.nan)
    chg[1:] = d.nav[1:] / d.nav[:-1] - 1.0
    # 이웃 날 마스크
    mask = np.zeros_like(d.has_row)
    for jj, rr in zip(j, ri):
        if jj >= 0 and rr >= 0:
            mask[max(rr - 1, 0) : min(rr + 2, T), jj] = True
    use = np.where(dev_days[:, None] & ~mask, chg, np.nan)
    with np.errstate(all="ignore"):
        sd = np.nanstd(use, axis=0)
        nuse = np.sum(np.isfinite(use), axis=0)
    counts = {-1: 0, 0: 0, 1: 0}
    n_low_vol = n_qual = n_undef = n_tie = 0
    for jj, rr, a in zip(j, ri, amt):
        if jj < 0 or rr < 1 or rr + 1 >= T:
            continue
        prev = d.nav[rr - 1, jj]
        if not np.isfinite(prev) or prev <= 0 or a / prev < NAV_DROP_MIN:
            continue
        n_qual += 1
        if not (nuse[jj] >= 20 and sd[jj] < NAV_SD_MAX):
            continue
        n_low_vol += 1
        best, bd = None, np.inf
        ties = False
        for off in (-1, 0, 1):
            t = rr + off
            c = chg[t, jj]
            base = d.nav[t - 1, jj]
            if not np.isfinite(c) or not np.isfinite(base) or base <= 0:
                continue
            dist_ = abs(c - (-a / base))
            if dist_ < bd - 1e-15:
                best, bd, ties = off, dist_, False
            elif abs(dist_ - bd) <= 1e-15:
                ties = True
        if best is None:
            n_undef += 1
        elif ties:
            n_tie += 1
        else:
            counts[best] += 1
    n_decided = sum(counts.values())
    shares = {k: (v / n_decided if n_decided else float("nan")) for k, v in counts.items()}
    majority = next((k for k, v in shares.items() if v > 0.5), None)
    return {
        "rule": rule,
        "n_distributions_ge_0.3pct": n_qual,
        "n_low_vol_etf": n_low_vol,
        "n_undefined": n_undef,
        "n_tie": n_tie,
        "n_decided": n_decided,
        "offset_counts": {str(k): v for k, v in counts.items()},
        "offset_shares": {str(k): v for k, v in shares.items()},
        "majority_offset": majority,
        "decision": (
            "no_majority(판정 대기)"
            if majority is None
            else ("rule_date" if majority == 0 else f"shift:{majority:+d}")
        ),
    }


# ---------------------------------------------------------------- 판정 구간: 정합성·D 행렬
def dist_rule_idx(d: Dense, dist: pl.DataFrame, rule: str = FROZEN_RULE) -> pl.DataFrame:
    """분배마다 ETF 열 번호 j, 규칙 날짜 번호 ``ri``, KRX 조정일 일치 ``match`` 를 붙인다."""
    p = prepare_dist(d, dist)
    ri = rule_idx(d, _rgt(p), rule)
    adj = krx_adjust_days(d)
    j = p["j"].to_numpy()
    ok = (j >= 0) & (ri >= 0)
    match = np.zeros(len(j), dtype=bool)
    match[ok] = adj[ri[ok], j[ok]]
    return p.with_columns(pl.Series("ri", ri), pl.Series("match", match))


def exdate_matrix(
    d: Dense, p: pl.DataFrame, nav_shift: int = FROZEN_NAV_SHIFT
) -> tuple[np.ndarray, dict]:
    """NAV 분배락일의 주당 분배금 행렬 D (T, N). ``p`` 는 ``dist_rule_idx`` 결과. 달력 밖은 건너뛰고 개수를 센다."""
    D = np.zeros_like(d.nav)
    j = p["j"].to_numpy()
    t = p["ri"].to_numpy() + nav_shift
    a = p["amount"].to_numpy().astype(float)
    ok = (j >= 0) & (p["ri"].to_numpy() >= 0) & (t >= 0) & (t < d.n_days)
    np.add.at(D, (t[ok], j[ok]), a[ok])
    return D, {"n_used": int(ok.sum()), "n_skipped": int((~ok).sum())}


def etf_year_consistency(
    d: Dense, p: pl.DataFrame, years: list[int], min_ratio: float = CONSISTENCY_MIN
) -> pl.DataFrame:
    """(ETF, Y)마다 Y와 Y+1 창 안(규칙 날짜 기준) 분배 수·일치 수·비율. 분배가 없으면 ratio = null.

    판정 구간에서 이 표는 **개수와 비율만** 담는다(값 크기 없음).
    """
    j = p["j"].to_numpy()
    ri = p["ri"].to_numpy()
    m = p["match"].to_numpy()
    rows = []
    for y in years:
        s0, _ = d.year_range(y)
        _, e1 = d.year_range(y + 1)
        if s0 < 0 or e1 < 0:
            continue
        sel = (j >= 0) & (ri >= s0) & (ri <= e1)
        if not sel.any():
            continue
        jj = j[sel]
        mm = m[sel]
        tot = np.bincount(jj, minlength=len(d.isu))
        hit = np.bincount(jj, weights=mm.astype(float), minlength=len(d.isu))
        for k in np.flatnonzero(tot):
            rows.append(
                {
                    "year": y,
                    "isu_cd": d.isu[k],
                    "n_dist": int(tot[k]),
                    "n_match": int(hit[k]),
                    "ratio": float(hit[k] / tot[k]),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "year": pl.Int64,
            "isu_cd": pl.String,
            "n_dist": pl.Int64,
            "n_match": pl.Int64,
            "ratio": pl.Float64,
        },
    )
