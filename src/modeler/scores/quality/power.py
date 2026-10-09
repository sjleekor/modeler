"""검정력 재계산 (사전등록 20261010_quality_score §6, ST-15).

Hanley-McNeil 표준오차 + 정규 근사. 귀무 AUC 0.5의 표준오차로 임계값을 만들고,
설계효과(DE)는 분산에 곱한다(표준오차에는 √DE). 데이터를 읽지 않는다.

    python -m modeler.scores.quality.power [--out-dir DIR]

"80% 검출 최소 AUC"는 검정력이 0.80이 되는 AUC(이분법)다. §6 표의 소수 셋째 자리와
±0.001 안에서 맞는다(표가 반올림·올림 중 어느 쪽인지 문서에 없다).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

from scipy.stats import norm

# ---- §6 가정
N_TOTAL_O1 = 8500  # 판정 형성 FY2019~2024 넷 다 회사 수의 합
EVENTS_O1 = (50, 100, 160, 245)
AUC_GRID = (0.58, 0.65, 0.70)
DESIGN_EFFECTS = (1.0, 2.0)
ALPHA_O1 = 0.05  # 한쪽, 고정 순서 첫 단계 (처음 값은 0.0125 = 0.05/4)
ALPHA_O1_OLD = 0.0125
O2_EVENTS = (20, 30, 50, 75, 100)
O2_NEG_PER_EVENT = 20


def hm_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil(1982) AUC 표준오차."""
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    v = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / (
        n_pos * n_neg
    )
    return math.sqrt(v)


def auc_power(auc: float, n_pos: int, n_neg: int, alpha: float, de: float = 1.0) -> float:
    """귀무 AUC=0.5 대 대립 AUC=auc, 한쪽 검정의 검정력."""
    s0 = hm_se(0.5, n_pos, n_neg) * math.sqrt(de)
    s1 = hm_se(auc, n_pos, n_neg) * math.sqrt(de)
    crit = 0.5 + norm.ppf(1 - alpha) * s0
    return float(1 - norm.cdf((crit - auc) / s1))


def min_detectable_auc(
    n_pos: int, n_neg: int, alpha: float, de: float = 1.0, target: float = 0.8
) -> float:
    """검정력이 target이 되는 AUC(이분법 해). 보고할 때는 소수 셋째 자리로 반올림한다."""
    lo, hi = 0.5001, 0.999
    for _ in range(100):
        mid = (lo + hi) / 2
        if auc_power(mid, n_pos, n_neg, alpha, de) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def g2_joint_power(
    auc: float, n_pos: int, n_neg: int, alpha: float, de: float, point_gate: float
) -> float:
    """G2 = 점추정 ≥ point_gate 그리고 한쪽 p < alpha. 점추정이 정규라고 보고 두 조건의 합."""
    s0 = hm_se(0.5, n_pos, n_neg) * math.sqrt(de)
    s1 = hm_se(auc, n_pos, n_neg) * math.sqrt(de)
    thr = max(point_gate, 0.5 + norm.ppf(1 - alpha) * s0)
    return float(1 - norm.cdf((thr - auc) / s1))


def capture_pass_prob(true_c: float, threshold: float, n_events: int, de: float) -> float:
    """G3: 하위 20% 포착률 점추정 ≥ threshold. 이항 정규 근사, n_eff = n/DE."""
    ne = n_events / de
    s = math.sqrt(true_c * (1 - true_c) / ne)
    return float(1 - norm.cdf((threshold - true_c) / s))


def capture_sig_power(
    true_c: float, n_events: int, alpha: float, de: float, c0: float = 0.2, alt_se: bool = False
) -> float:
    """포착률이 우연(c0=0.2)보다 높은지 한쪽 검정의 검정력(유의성 기준)."""
    ne = n_events / de
    s0 = math.sqrt(c0 * (1 - c0) / ne)
    s1 = math.sqrt(true_c * (1 - true_c) / ne)
    if alt_se:
        s0 = s1
    return float(1 - norm.cdf((c0 + norm.ppf(1 - alpha) * s0 - true_c) / s1))


def e1_gate_power(
    true_hit: float,
    n_events: int,
    de: float = 1.5,
    z: float = 1.96,
    point_gate: float = 0.5,
    lower_gate: float = 0.3,
) -> float:
    """E1 적중률 게이트: 점추정 ≥ 0.5 그리고 단측 하한 ≥ 0.3 (Holm m=2 → z=1.96)."""
    ne = n_events / de
    s = math.sqrt(true_hit * (1 - true_hit) / ne)

    # 점추정 p가 정규. 하한 조건 p - z*sqrt(p(1-p)/ne) ≥ lower_gate 를 만족하는 최소 p.
    def lower(p: float) -> float:
        return p - z * math.sqrt(p * (1 - p) / ne)

    lo, hi = lower_gate, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if lower(mid) >= lower_gate:
            hi = mid
        else:
            lo = mid
    thr = max(point_gate, hi)
    return float(1 - norm.cdf((thr - true_hit) / s))


def e2_gate_power(
    true_rho: float, n_eff: int = 150, z: float = 1.96, point_gate: float = 0.30
) -> float:
    """E2 게이트: 가중평균 ρ 점추정 ≥ 0.30 그리고 Holm(m=2) 단측 하한 > 0.

    Fisher z, se = 1/√(n-3).
    """
    s = 1 / math.sqrt(n_eff - 3)
    thr = max(math.atanh(point_gate), z * s)
    return float(1 - norm.cdf((thr - math.atanh(true_rho)) / s))


# ---------------------------------------------------------------- 표와 §6 값 대조
# §6 O1 표(α=0.05): 사건 -> 칸 순서 [AUC .58 DE1/2, .65 DE1/2, .70 DE1/2, 최소 AUC DE1/2]
DOC_O1 = {
    50: ((0.62, 0.40), (0.97, 0.82), (1.00, 0.96), (0.603, 0.646)),
    100: ((0.86, 0.62), (1.00, 0.97), (1.00, 1.00), (0.573, 0.603)),
    160: ((0.96, 0.78), (1.00, 1.00), (1.00, 1.00), (0.558, 0.582)),
    245: ((0.99, 0.91), (1.00, 1.00), (1.00, 1.00), (0.547, 0.566)),
}
# §6 "처음 값"(α=0.0125): 100·160·245건의 최소 AUC(DE 1/2)와 AUC 0.58 검정력(DE 1/2)
DOC_O1_OLD_MIN = {100: (0.591, 0.628), 160: (0.572, 0.602), 245: (0.559, 0.583)}
DOC_O1_OLD_P58 = {100: (0.69, 0.39), 160: (0.88, 0.58), 245: (0.98, 0.78)}
DOC_O2 = {
    20: ((0.73, 0.48), (0.51, 0.27)),
    30: ((0.86, 0.62), (0.70, 0.39)),
    50: ((0.97, 0.81), (0.90, 0.61)),
    75: ((1.00, 0.92), (0.98, 0.80)),
    100: ((1.00, 0.97), (1.00, 0.90)),
}
DOC_E1 = {  # 진짜 적중률 -> 사건 100·150·190
    0.4: (0.05, 0.02, 0.01),
    0.5: (0.5, 0.5, 0.5),
    0.6: (0.95, 0.98, 0.99),
    0.7: (1.0, 1.0, 1.0),
}
DOC_E2 = {0.2: 0.10, 0.3: 0.5, 0.4: 0.92, 0.5: 1.0}
DOC_LIFT = {  # (lift, 사건) -> 값 (DE 2, 유의성)
    (2.0, 160): 0.90,
    (2.0, 100): 0.75,
}
DOC_G3 = {"G2_100_de2": 0.97, "G3_t30": 0.88, "G3_t40": 0.39, "prod_t30": 0.85, "prod_t40": 0.38}


def _r2(x: float) -> float:
    return round(x + 1e-12, 2)


def reproduce() -> list[dict]:
    rows: list[dict] = []

    def add(table, cell, doc, mine, tol):
        rows.append(
            {
                "table": table,
                "cell": cell,
                "doc": doc,
                "mine": round(mine, 4),
                "diff": round(mine - doc, 4),
                "match": abs(mine - doc) <= tol,
            }
        )

    for e, cells in DOC_O1.items():
        n_neg = N_TOTAL_O1 - e
        for ci, auc in enumerate(AUC_GRID):
            for di, de in enumerate(DESIGN_EFFECTS):
                add(
                    "O1(α=0.05)",
                    f"events={e} AUC={auc} DE={de:g}",
                    cells[ci][di],
                    auc_power(auc, e, n_neg, ALPHA_O1, de),
                    0.005,
                )
        for di, de in enumerate(DESIGN_EFFECTS):
            add(
                "O1(α=0.05)",
                f"events={e} 최소AUC DE={de:g}",
                cells[3][di],
                min_detectable_auc(e, n_neg, ALPHA_O1, de),
                0.0011,
            )
    for e, (m1, m2) in DOC_O1_OLD_MIN.items():
        n_neg = N_TOTAL_O1 - e
        for di, (de, d) in enumerate(zip(DESIGN_EFFECTS, (m1, m2))):
            add(
                "O1 처음값(α=0.0125)",
                f"events={e} 최소AUC DE={de:g}",
                d,
                min_detectable_auc(e, n_neg, ALPHA_O1_OLD, de),
                0.0011,
            )
        for de, d in zip(DESIGN_EFFECTS, DOC_O1_OLD_P58[e]):
            add(
                "O1 처음값(α=0.0125)",
                f"events={e} AUC=0.58 DE={de:g}",
                d,
                auc_power(0.58, e, n_neg, ALPHA_O1_OLD, de),
                0.005,
            )
    for e, (new, old) in DOC_O2.items():
        for de, d in zip(DESIGN_EFFECTS, new):
            add(
                "O2~O4(α=0.05)",
                f"events={e} DE={de:g}",
                d,
                auc_power(0.65, e, O2_NEG_PER_EVENT * e, ALPHA_O1, de),
                0.005,
            )
        for de, d in zip(DESIGN_EFFECTS, old):
            add(
                "O2~O4 처음값(α=0.0125)",
                f"events={e} DE={de:g}",
                d,
                auc_power(0.65, e, O2_NEG_PER_EVENT * e, ALPHA_O1_OLD, de),
                0.005,
            )
    for hit, vals in DOC_E1.items():
        for n, d in zip((100, 150, 190), vals):
            add("E1", f"hit={hit} n={n}", d, e1_gate_power(hit, n), 0.005)
    for rho, d in DOC_E2.items():
        add("E2", f"rho={rho}", d, e2_gate_power(rho), 0.005)
    # G2∧G3 (어림): 사건 100, DE 2, 기준 AUC 0.65, 포착률 0.38
    g2 = auc_power(0.65, 100, N_TOTAL_O1 - 100, ALPHA_O1, 2.0)
    g3_30 = capture_pass_prob(0.38, 0.30, 100, 2.0)
    g3_40 = capture_pass_prob(0.38, 0.40, 100, 2.0)
    add("G2∧G3 어림", "G2 (100건, DE2, AUC.65, 유의성만)", DOC_G3["G2_100_de2"], g2, 0.005)
    add("G2∧G3 어림", "G3 문턱 30%", DOC_G3["G3_t30"], g3_30, 0.005)
    add("G2∧G3 어림", "G3 문턱 40%", DOC_G3["G3_t40"], g3_40, 0.005)
    add("G2∧G3 어림", "곱 문턱 30%", DOC_G3["prod_t30"], g2 * g3_30, 0.01)
    add("G2∧G3 어림", "곱 문턱 40%", DOC_G3["prod_t40"], g2 * g3_40, 0.01)
    # 하위 20% lift 유의성 검정력(DE 2): 문서 0.90(160건)·0.75(100건).
    # 가정이 문서에 없어 후보 가정 둘로 계산한다.
    for (lift, n), d in DOC_LIFT.items():
        add(
            "lift 유의성(가정불명)",
            f"lift={lift} n={n} α=0.05 DE=2 H0 SE",
            d,
            capture_sig_power(0.2 * lift, n, 0.05, 2.0),
            0.01,
        )
        add(
            "lift 유의성(가정불명)",
            f"lift={lift} n={n} α=0.0125 DE=2 H1 SE",
            d,
            capture_sig_power(0.2 * lift, n, 0.0125, 2.0, alt_se=True),
            0.01,
        )
    return rows


def new_numbers() -> list[dict]:
    """사전등록이 "동결 전 재구현 때 다시 낸다"고 한 값(α=0.05에서 G2 합 조건 등)."""
    out = []
    for e in (100, 160):
        n_neg = N_TOTAL_O1 - e
        for auc in (0.62, 0.65):
            g2 = g2_joint_power(auc, e, n_neg, ALPHA_O1, 2.0, 0.60)
            out.append(
                {
                    "item": f"G2 합 조건(점추정≥0.60 ∧ p<0.05) events={e} AUC={auc} DE=2",
                    "value": round(g2, 4),
                }
            )
        for auc in (0.62, 0.65):
            g2 = g2_joint_power(auc, e, n_neg, ALPHA_O1_OLD, 2.0, 0.60)
            out.append(
                {
                    "item": f"(참고) 처음 α=0.0125 G2 합 조건 events={e} AUC={auc} DE=2",
                    "value": round(g2, 4),
                }
            )
    g2 = g2_joint_power(0.65, 100, N_TOTAL_O1 - 100, ALPHA_O1, 2.0, 0.60)
    for thr in (0.30, 0.40):
        g3 = capture_pass_prob(0.38, thr, 100, 2.0)
        out.append(
            {
                "item": f"G2(합 조건)×G3 곱, 100건 DE2 AUC.65 포착률.38 문턱{thr:.0%}",
                "value": round(g2 * g3, 4),
            }
        )
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    root = Path(os.environ.get("STOCK_DATA_ROOT", "../stock_data"))
    ap.add_argument("--out-dir", default=str(root / "kr/output/quality_score_prep_20261010"))
    a = ap.parse_args(argv)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = reproduce()
    with open(out / "power_reproduction.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    nn = new_numbers()
    with open(out / "power_new_numbers.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["item", "value"])
        w.writeheader()
        w.writerows(nn)
    bad = [r for r in rows if not r["match"]]
    print(f"cells={len(rows)} match={len(rows) - len(bad)} mismatch={len(bad)}")
    for r in bad:
        print("MISMATCH", r)
    for r in nn:
        print(r["item"], r["value"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
