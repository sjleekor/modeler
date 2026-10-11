"""ETF E3 — 기록용 칸(사양 §6 "기록용", m에 안 넣음). **판정 실행에서만 부른다.**

값(수익률·보수)을 읽으므로 동결 전에는 판정 구간에 돌리지 않는다. 합성 자료 시험만 돈다.

- ``r1_unexplained``  R1 설명 안 되는 차이: 그룹-연도 고정효과 회귀 ``RTR_i(Y+1) ~ −A_i/100`` 의 그룹-연도 안
  잔차 표준편차(bp)와 R².
- ``r7_general_account``  R7 일반 계좌: 분배금 D에 0.846(= 1 − 15.4%)을 곱한 총수익으로 G·H를 다시 만든 C.
- ``r8_close_based``  R8 종가 기준: NAV 대신 ``TDD_CLSPRC``, 분배금은 KRX 기준가 조정일(규칙 날짜 R−1, NAV 이동 없음).
- ``r11_delisted_partial``  R11 폐지 ETF 부분 창: Y+1 중 폐지된 ETF-연을 마지막 NAV 날까지의 창으로 넣은 C와 표본 구성 수.
- ``r12_baseline``  R12 기준선: 그룹-연도마다 (가) 형성 자기 창 G_own 최고 ETF vs (나) 순자산 최대 패시브 ETF의
  H_own 차이(bp).
- ``delisted_by_f_band``  폐지·합병 ETF-연 수를 F 구간(<0.10, 0.10~0.30, 0.30~0.50, ≥0.50 %)별로.
- ``r9_index_tracking``  R9 지수 TR 절대 추적차이(S&P 500 TR 자료가 있을 때만, 없으면 "자료 대기").
- ``r13_forward``  R13 전진 확인(형성 2025 → 결과 2026): KRX 자료가 2026-12-30까지 있을 때만.
"""

from __future__ import annotations

import dataclasses
import os
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from modeler.scores.quality import etf_e2 as e2
from modeler.scores.quality import etf_e3_exdate as ex
from modeler.scores.quality import etf_e3_fees as fees
from modeler.scores.quality import etf_e3_sample as sm
from modeler.scores.quality import etf_e3_stat as st

GENERAL_ACCOUNT_FACTOR = 0.846  # 일반 계좌 분배금 세후 = 1 − 15.4%
F_BANDS = ((None, 0.10), (0.10, 0.30), (0.30, 0.50), (0.50, None))  # % (처음 고른 값)
F_BAND_LABELS = ("F<0.10", "0.10<=F<0.30", "0.30<=F<0.50", "F>=0.50")
FORWARD_FORM_YEAR = 2025
FORWARD_MIN_END = date(2026, 12, 30)
SP500_TR_ETFS = ("379800", "448290")
SP500_ENV = "E3_SP500_TR_CSV"


def _run():
    from modeler.scores.quality import etf_e3_run as run  # 순환 import 피하려고 늦게 부른다

    return run


# ---------------------------------------------------------------- R1
def r1_unexplained(xs: list[np.ndarray], ys: list[np.ndarray]) -> dict:
    """그룹-연도 고정효과 회귀(y = β x)의 그룹-연도 안 잔차 표준편차(bp)와 R²(그룹-연도 안 분산 중 설명 몫)."""
    sxy = sxx = syy = 0.0
    parts = []
    n_obs = 0
    n_gy = 0
    for x, y in zip(xs, ys):
        if len(x) < 2:
            continue
        xd, yd = x - x.mean(), y - y.mean()
        parts.append((xd, yd))
        sxy += float((xd * yd).sum())
        sxx += float((xd * xd).sum())
        syy += float((yd * yd).sum())
        n_obs += len(x)
        n_gy += 1
    if sxx <= 0 or n_obs - n_gy - 1 <= 0:
        return {"n_group_years": n_gy, "n_etf_years": n_obs, "beta": None}
    beta = sxy / sxx
    ssr = sum(float(((yd - beta * xd) ** 2).sum()) for xd, yd in parts)
    return {
        "n_group_years": n_gy,
        "n_etf_years": n_obs,
        "beta": beta,
        "resid_sd_bp": float(np.sqrt(ssr / (n_obs - n_gy - 1)) * 1e4),
        "within_sd_bp_before": float(np.sqrt(syy / (n_obs - n_gy)) * 1e4),
        "r2_within": float(1 - ssr / syy) if syy > 0 else None,
    }


# ---------------------------------------------------------------- R7 · R8
def r7_general_account(ctx: sm.Ctx, sample: sm.Sample, n_boot: int, seed: int) -> dict:
    """분배금 D × 0.846 으로 G·H를 다시 만든다(분배락일 위치·표본은 그대로)."""
    ctx2 = dataclasses.replace(ctx, D=ctx.D * GENERAL_ACCOUNT_FACTOR)
    inp = _run().build_inputs(ctx2, sample)
    return {"factor": GENERAL_ACCOUNT_FACTOR, **st.c_only(inp["sd_b"], n_boot, seed)}


def close_based_ctx(ctx: sm.Ctx) -> sm.Ctx:
    """NAV 자리에 종가를 넣고 분배금은 KRX 조정일(규칙 날짜, NAV 이동 0)에 넣은 ``Ctx``."""
    d = ctx.dense
    d2 = dataclasses.replace(d, nav=d.close.copy())
    D2, _ = ex.exdate_matrix(d2, ctx.dist, 0)
    return dataclasses.replace(
        ctx, dense=d2, D=D2, rfin=sm.return_defined(d2), exmask=D2 > 0, nav_shift=0
    )


def r8_close_based(ctx: sm.Ctx, sample: sm.Sample, n_boot: int, seed: int) -> dict:
    """종가 기준 총수익. 표본(생존 ETF-연)은 그대로, 쌍 규칙(공통일·분배락일)은 종가 기준으로 다시 본다."""
    ctx2 = close_based_ctx(ctx)
    gys2 = []
    for gy in sample.gys:
        n = len(gy.cols)
        ok = np.zeros((n, n), dtype=bool)
        for a in range(n):
            for b in range(a + 1, n):
                ca, cb = int(gy.cols[a]), int(gy.cols[b])
                ok[a, b] = ok[b, a] = sm.pair_window_ok(
                    ctx2, ca, cb, gy.year
                ) and sm.pair_window_ok(ctx2, ca, cb, gy.year + 1)
        gys2.append(dataclasses.replace(gy, pair_ok=ok))
    inp = _run().build_inputs(ctx2, dataclasses.replace(sample, gys=gys2))
    return {
        "n_pairs_exdate_ok_close": int(
            sum(g.pair_ok[np.triu_indices(len(g.cols), 1)].sum() for g in gys2)
        ),
        "n_pairs_exdate_ok_nav": int(
            sum(g.pair_ok[np.triu_indices(len(g.cols), 1)].sum() for g in sample.gys)
        ),
        **st.c_only(inp["sd_b"], n_boot, seed),
    }


# ---------------------------------------------------------------- R11
def r11_delisted_partial(ctx: sm.Ctx, years, n_boot: int, seed: int) -> dict:
    """Y+1 중 폐지된 ETF-연을 Y+1 마지막 NAV 날까지의 창으로 넣는다(결과 보수 A는 있는 달 1개 이상이면 됨)."""
    s2 = sm.build_sample(ctx, years, include_delisted_y1=True)
    if not s2.gys:
        return {"note": "표본 없음"}
    inp = _run().build_inputs(ctx, s2)
    n_del = int(sum(int(v.delisted_y1.sum()) for v in inp["vals"]))
    n_fin = int(sum(len(g.cols) for g in s2.gys))
    return {
        "n_final_etf_years": n_fin,
        "n_delisted_partial_etf_years_in_final": n_del,
        "sample_counts": s2.counts.to_dicts(),
        **st.c_only(inp["sd_b"], n_boot, seed),
    }


# ---------------------------------------------------------------- R12
def _netassets(ctx: sm.Ctx, y: int, isu: list[str]) -> dict[str, float]:
    """T(= Y 마지막 거래일)의 순자산. 0·없음이면 직전 5거래일 안에서 가장 가까운 0 아닌 값(etf_panel 규칙)."""
    t = ctx.dense.year_range(y)[1]
    r = ctx.panel.rows.filter(
        pl.col("day_idx").is_between(t - 5, t)
        & pl.col("ISU_CD").is_in(isu)
        & (pl.col("INVSTASST_NETASST_TOTAMT") > 0)
    )
    r = r.sort("day_idx").group_by("ISU_CD").agg(pl.col("INVSTASST_NETASST_TOTAMT").last())
    return dict(zip(r["ISU_CD"].to_list(), r["INVSTASST_NETASST_TOTAMT"].to_list()))


def r12_baseline(ctx: sm.Ctx, sample: sm.Sample, vals) -> dict:
    """(가) 자기 창 G_own 최고 ETF − (나) 순자산 최대 패시브 ETF 의 H_own 차이(bp). 구간 없음."""
    active = dict(zip(ctx.life["isu_cd"].to_list(), ctx.life["active"].to_list()))
    diffs, ws = [], []
    n_same = n_no_passive = n_tie = n_invalid = 0
    for gy, v in zip(sample.gys, vals):
        n = len(gy.cols)
        valid = (
            ~v.own_y_lost
            & ~v.own_lost
            & np.isfinite(v.own_y)
            & np.isfinite(v.own_y1)
            & np.isfinite(v.F)
            & np.isfinite(v.A)
        )
        if valid.sum() < 2:
            n_invalid += 1
            continue
        g_own = np.where(valid, v.own_y + v.F / 100.0, -np.inf)
        a = int(np.argmax(g_own))
        if int((g_own == g_own[a]).sum()) > 1:
            n_tie += 1
            continue
        net = _netassets(ctx, gy.year, gy.isu)
        cand = [
            i
            for i in range(n)
            if valid[i] and not active.get(gy.isu[i], False) and gy.isu[i] in net
        ]
        if not cand:
            n_no_passive += 1
            continue
        b = max(cand, key=lambda i: net[gy.isu[i]])
        if a == b:
            n_same += 1
            continue
        h_own = v.own_y1 + v.A / 100.0
        diffs.append((h_own[a] - h_own[b]) * 1e4)
        ws.append(n)
    out = {
        "n_group_years": len(diffs),
        "n_same_etf": n_same,
        "n_no_passive": n_no_passive,
        "n_tie_top_g": n_tie,
        "n_invalid": n_invalid,
    }
    if diffs:
        d, w = np.array(diffs), np.array(ws, dtype=float)
        out.update(
            {
                "diff_bp_weighted_mean": float((d * w).sum() / w.sum()),
                "share_positive": float((d > 0).mean()),
                "diff_bp_median": float(np.median(d)),
            }
        )
    return out


# ---------------------------------------------------------------- 폐지·합병 × F 구간
def delisted_by_f_band(ctx: sm.Ctx, years) -> dict:
    """시작 표본 조건에서 '상장 유지'만 뺀 ETF-연 중 Y+1 안에 사라진 수를 F 구간별로(F 값을 읽는다 — 판정 실행에서만)."""
    d = ctx.dense
    ey = e2.etf_years(ctx.panel, ctx.life, list(years), with_values=False)
    ey = ey.filter(
        ~(
            pl.col("fail_days_y")
            | pl.col("fail_days_y1")
            | pl.col("fail_maturity")
            | pl.col("fail_pension")
            | pl.col("fail_hold")
        )
    )
    sz = ey.group_by("year", "group_key").len().rename({"len": "gsize"})
    ey = ey.join(sz, on=["year", "group_key"]).filter(pl.col("gsize") >= sm.MIN_GROUP)
    counts = {lab: {"n_etf_years": 0, "n_delisted_y1": 0} for lab in (*F_BAND_LABELS, "F 없음")}
    for isu, y, dl in zip(ey["isu_cd"], ey["year"], ey["fail_delisted_y1"]):
        t = d.dates[d.year_range(int(y))[1]]
        f = fees.formation_value(ctx.fee, ctx.code_map.get(isu), t)
        lab = "F 없음"
        if np.isfinite(f):
            for (lo, hi), name in zip(F_BANDS, F_BAND_LABELS):
                if (lo is None or f >= lo) and (hi is None or f < hi):
                    lab = name
                    break
        counts[lab]["n_etf_years"] += 1
        counts[lab]["n_delisted_y1"] += int(dl)
    return {"bands_pct": counts, "n_etf_years": int(ey.height)}


# ---------------------------------------------------------------- R9
def r9_index_tracking(ctx: sm.Ctx, path: str | Path | None = None) -> dict:
    """S&P 500 TR 지수(CSV: ``date``(YYYY-MM-DD)·``value``)와 379800·448290 의 연 절대 추적차이.

    KRX 거래일 t 의 NAV는 미국 t−1 종가를 반영하므로 지수는 t 직전(엄밀히 이전) 지수 날짜 값에 맞춘다.
    코스피200 TR(KRX drvprod)은 백필 전이라 없다. 경로는 인자 또는 환경변수 ``E3_SP500_TR_CSV``.
    """
    path = path or os.environ.get(SP500_ENV)
    if not path or not Path(path).exists():
        return {
            "status": "자료 대기",
            "note": "S&P 500 TR(2021-04-09~2025-05-23) 입력 없음. 코스피200 TR은 10-23 백필 전",
        }
    idx = pl.read_csv(str(path), try_parse_dates=True).sort("date")
    dates = idx["date"].to_numpy().astype("datetime64[D]")
    vals = idx["value"].to_numpy().astype(float)
    d = ctx.dense
    out: dict = {"status": "계산", "etfs": {}}
    for code in SP500_TR_ETFS:
        if code not in d.col:
            out["etfs"][code] = {"note": "KRX 자료에 없음"}
            continue
        j = d.col[code]
        pos = np.searchsorted(dates, d.dates, side="left") - 1  # t 보다 이전인 마지막 지수 날짜
        ok = (pos >= 0) & ctx.rfin[:, j]
        iv = np.where(pos >= 0, vals[np.maximum(pos, 0)], np.nan)
        per_year = {}
        for y in sorted(set(d.years[ok].tolist())):
            t = np.flatnonzero(ok & (d.years == y))
            if len(t) < 100:
                continue
            f = (d.nav[t, j] + ctx.D[t, j]) / d.nav[t - 1, j]
            etf = float(np.prod(f) - 1)
            index = float(iv[t[-1]] / iv[t[0] - 1] - 1)
            per_year[int(y)] = {
                "etf_tr": etf,
                "index_tr": index,
                "abs_diff_bp": abs(etf - index) * 1e4,
            }
        out["etfs"][code] = per_year
    return out


# ---------------------------------------------------------------- R13
def r13_forward(
    ctx: sm.Ctx,
    n_boot: int,
    seed: int,
    form_year: int = FORWARD_FORM_YEAR,
    min_end: date = FORWARD_MIN_END,
) -> dict:
    """전진 확인: 형성 ``form_year`` → 결과 ``form_year``+1. KRX 자료가 ``min_end`` 까지 없으면 돌지 않는다."""
    last = ctx.dense.dates[-1].astype("datetime64[D]").astype(object)
    if last < min_end:
        return {
            "status": "2027-01 이후",
            "note": f"KRX 자료 끝 {last} < {min_end}. 형성 {form_year}→결과 {form_year + 1} 은 자료가 닫힌 뒤",
        }
    s = sm.build_sample(ctx, (form_year,), close_year_end=True)
    if not s.gys:
        return {"status": "표본 없음", "sample_counts": s.counts.to_dicts()}
    inp = _run().build_inputs(ctx, s)
    return {
        "status": "계산",
        "sample_counts": s.counts.to_dicts(),
        **st.c_only(inp["sd_b"], n_boot, seed),
    }
