"""E3 테스트용 합성 패널. 거래일 = 평일. NAV는 종목마다 연 알파(드리프트)와 약한 잡음을 갖는다.

분배는 분기 말 기준일 E(마지막 거래일)에 주고, KRX 기준가 조정일 = E의 직전 거래일(R−1),
NAV 차감일 = R−1 의 하루 앞(E−2)으로 만든다(실제 개발 구간 점검 결과와 같은 모양).
"""

import csv
from datetime import date, timedelta

import numpy as np
import polars as pl

from modeler.scores.quality import etf_e3_sample as sm
from modeler.scores.quality import etf_panel as ep

COLS = [
    "BAS_DD",
    "ISU_CD",
    "ISU_NM",
    "TDD_CLSPRC",
    "CMPPREVDD_PRC",
    "FLUC_RT",
    "NAV",
    "TDD_OPNPRC",
    "TDD_HGPRC",
    "TDD_LWPRC",
    "ACC_TRDVOL",
    "ACC_TRDVAL",
    "MKTCAP",
    "INVSTASST_NETASST_TOTAMT",
    "LIST_SHRS",
    "IDX_IND_NM",
    "OBJ_STKPRC_IDX",
    "CMPPREVDD_IDX",
    "FLUC_RT_IDX",
]


def weekdays(start, end):
    out, d = [], start
    while d <= end:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


DAYS = weekdays(date(2013, 1, 1), date(2018, 12, 31))


def quarter_end_idx(days):
    """분기 말 기준일 E = 그 분기의 마지막 거래일 번호(마지막 분기는 자료 끝 직전까지)."""
    out = []
    for i in range(len(days) - 1):
        a, b = days[i], days[i + 1]
        if a.month in (3, 6, 9, 12) and b.month != a.month:
            out.append(i)
    return out


def build_panel(tmp_path, etfs, days=DAYS, seed=1, pay=True, quarterly_amt=0.01):
    """etfs: {isu: dict(name, idx, alpha(연), first=None, last=None, shares=1e6, nav_noise=0.004)}.

    반환 (panel, dist_df). dist_df 칸: isu_cd, rgt_std_dt, amount, kind, liquidation.
    """
    rng = np.random.default_rng(seed)
    qe = quarter_end_idx(days)
    ex_idx = {e - 2 for e in qe}  # NAV 차감일
    adj_idx = {e - 1 for e in qe}  # KRX 기준가 조정일
    p = tmp_path / "x.csv"
    dist_rows = []
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for isu, spec in etfs.items():
            first = spec.get("first") or days[0]
            last = spec.get("last") or days[-1]
            nav = 10000.0
            prev_close = None
            for i, d in enumerate(days):
                if d < first or d > last:
                    continue
                shock = spec.get("nav_noise", 0.004) * rng.standard_normal()
                idx_nm = (
                    spec["idx_before"]
                    if spec.get("idx_before") and d < spec["idx_change_at"]
                    else spec["idx"]
                )
                r = spec["alpha"] / 250.0 + shock
                drop = 0.0
                if pay and i in ex_idx:
                    drop = quarterly_amt * nav
                    r -= quarterly_amt
                nav = nav * (1 + r) if prev_close is not None else nav
                shares = spec.get("shares", 1_000_000)
                if spec.get("split_at") and d >= spec["split_at"]:
                    shares = shares * 2
                    if d == spec["split_at"]:
                        nav = nav / 2
                if spec.get("outlier_at") == d:
                    nav = nav * 0.6
                close = round(nav)
                prev_ref = prev_close
                if pay and i in adj_idx and prev_close is not None:
                    prev_ref = prev_close - round(quarterly_amt * nav)
                cmp_ = "" if prev_ref is None else str(close - prev_ref)
                row = {c: "" for c in COLS}
                row.update(
                    BAS_DD=d.strftime("%Y%m%d"),
                    ISU_CD=isu,
                    ISU_NM=spec["name"],
                    TDD_CLSPRC=str(close),
                    CMPPREVDD_PRC=cmp_,
                    NAV="" if d in spec.get("nav_missing", ()) else f"{nav:.2f}",
                    LIST_SHRS=str(shares),
                    IDX_IND_NM=idx_nm,
                    ACC_TRDVAL="1000000",
                    INVSTASST_NETASST_TOTAMT=str(spec.get("netasst", 100000000)),
                    MKTCAP="100000000",
                )
                w.writerow(row)
                prev_close = close
                if pay and i in ex_idx:
                    e = i + 2
                    dist_rows.append(
                        {
                            "isu_cd": isu,
                            "rgt_std_dt": days[e],
                            "amount": float(round(drop, 2)),
                            "kind": "이익분배",
                            "liquidation": False,
                        }
                    )
    panel = ep.read_panel(p)
    dist = pl.DataFrame(
        dist_rows,
        schema={
            "isu_cd": pl.String,
            "rgt_std_dt": pl.Date,
            "amount": pl.Float64,
            "kind": pl.String,
            "liquidation": pl.Boolean,
        },
    )
    return panel, dist


def fee_frame(ters, days=DAYS):
    """ters: {fund_code: ter(%)}. 월말(마지막 평일) 기준일마다 한 행."""
    ends = [days[i] for i in range(len(days) - 1) if days[i].month != days[i + 1].month]
    rows = []
    for fc, t in ters.items():
        for d in ends:
            rows.append(
                {
                    "fund_code": fc,
                    "operator": "OP",
                    "sum_fee": t + 0.1,
                    "ter": t,
                    "brokerage": 0.05,
                    "base_date": d,
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "fund_code": pl.String,
            "operator": pl.String,
            "sum_fee": pl.Float64,
            "ter": pl.Float64,
            "brokerage": pl.Float64,
            "base_date": pl.Date,
        },
    )


def code_map_frame(pairs):
    return pl.DataFrame(
        {
            "short_code": [a for a, _ in pairs],
            "fund_code": [b for _, b in pairs],
            "level": ["high"] * len(pairs),
        }
    )


def make_ctx(tmp_path, etfs, ters, **kw):
    panel, dist = build_panel(tmp_path, etfs, **kw)
    cm = code_map_frame([(k, f"F{k}") for k in etfs])
    fee = fee_frame({f"F{k}": ters[k] for k in etfs})
    return sm.make_ctx(panel, dist, cm, fee)
