"""ETF E3 — 검정력 모의 (사양 §7, 동결 전 modeler 재구현).

**실제 자료에서 설계효과를 역산하지 않는다.** 구조는 판정 표본의 그룹-연도 구조(동결 키, 그룹 수·그룹-연도 수·
크기 분포)를 **개수만** 가져온다. 값은 전부 모의로 만든다. 통계량·게이트는 ``etf_e3_stat`` 의 실제 코드를 쓴다
(모의에서는 순열 999회·부트스트랩 500회로 줄여 돌린다).

모의 모형(ETF-연 i, 그룹-연도 안):
- 보수 ``F_i = ter_sd · z`` (%), 결과 연도 보수 ``A_i = F_i``(수준은 그룹 안 차이만 의미가 있다).
- 총수익(보수 전) ``gross_{i,y} = 지속_i + 잡음_{i,y}``. 지속 몫 s = 지속 분산 / (지속 + 잡음 분산), 잡음 sd = σ.
  지속 성분은 형성 연도와 결과 연도에 같다.
- 운용사 공통: 운용사 번호를 ETF마다 임의로 붙이고(운용사 10개, 점유 지프식), 몫 ω만큼 ① 지속 성분의 일부가
  운용사 단위(그룹을 넘어 같은 운용사가 공유, 연도 불변)이고 ② 잡음의 일부가 운용사-연도 공통 충격(연도마다 운용사마다 하나, 그룹을 넘어 공유)이다.
- 전달률 β: 관측 총수익 ``RTR_i(y) = gross_{i,y} − β · A_i/100`` . 따라서
  ``G_i = RTR_i(Y) + F_i/100`` , ``H_i = RTR_i(Y+1) + A_i/100`` 이고 β = 1 이면 보수 전 수익만 남는다.
- 모의에서는 쌍마다의 공통 창 차이를 무시한다(H[a, b] = H_a). 같은 ETF가 여러 연도에 나오는 것도 무시한다(연도마다 독립).

게이트 G2 = C ≥ 0.55 ∧ 순열 p < 0.05 ∧ 부트스트랩 하한 > 0.5. 규칙 1-a는 모의 β̂(그룹-연도 고정효과)로 건다.
"""

from __future__ import annotations

import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from modeler.scores.quality import etf_e3_stat as st

SIM_PERM = 999
SIM_BOOT = 500
N_OPERATORS = 10
GRID = {
    "persist_share": (0.0, 0.1, 0.2, 0.3),
    "noise_bp": (5.0, 10.0, 20.0),
    "ter_sd_bp": (5.0, 10.0),
    "beta": (0.0, 0.5, 1.0),
    "operator_share": (0.0, 0.3),
}


@dataclass
class Skeleton:
    """판정 표본 구조(개수만): 그룹-연도 (그룹 번호, 연도, 크기)에서 만든 쌍 인덱스."""

    sizes: np.ndarray
    years: np.ndarray
    group: np.ndarray
    n_groups: int
    etf_off: np.ndarray
    pa: np.ndarray
    pb: np.ndarray
    pgy: np.ndarray
    pair_off: np.ndarray
    hoff: np.ndarray
    hmat_src: np.ndarray  # hmat_flat[k] = H[hmat_src[k]]
    ga: np.ndarray  # 쌍의 ETF 전역 번호
    gb: np.ndarray

    @property
    def E(self) -> int:
        return int(self.etf_off[-1])


def skeleton_from_structure(structure: list[tuple[int, int, int]]) -> Skeleton:
    """``structure`` = [(그룹 번호, 연도, 크기), ...]  (그룹 번호는 0부터 연속)."""
    group = np.array([s[0] for s in structure])
    years = np.array([s[1] for s in structure])
    sizes = np.array([s[2] for s in structure], dtype=np.int64)
    etf_off = np.concatenate([[0], np.cumsum(sizes)])
    pa, pb, pgy, pair_off = [], [], [], [0]
    hoff, src = [], []
    pos = 0
    for g, n in enumerate(sizes):
        hoff.append(pos)
        idx = etf_off[g] + np.repeat(np.arange(n), n)  # H[a, b] = H_a
        src.append(idx)
        pos += n * n
        for a in range(n):
            for b in range(a + 1, n):
                pa.append(a)
                pb.append(b)
                pgy.append(g)
        pair_off.append(len(pa))
    pa_a, pb_a, pgy_a = np.array(pa), np.array(pb), np.array(pgy)
    return Skeleton(
        sizes=sizes,
        years=years,
        group=group,
        n_groups=int(group.max()) + 1,
        etf_off=etf_off,
        pa=pa_a,
        pb=pb_a,
        pgy=pgy_a,
        pair_off=np.array(pair_off),
        hoff=np.array(hoff),
        hmat_src=np.concatenate(src),
        ga=etf_off[pgy_a] + pa_a,
        gb=etf_off[pgy_a] + pb_a,
    )


def structure_from_counts(ey) -> list[tuple[int, int, int]]:
    """polars ETF-연 표(``year``, ``gkey``)에서 그룹-연도 크기 구조만 뽑는다. 크기 2 이상만."""
    g = ey.group_by("year", "gkey").len().filter(ey.group_by("year", "gkey").len()["len"] >= 2)
    keys = {k: i for i, k in enumerate(sorted(set(g["gkey"].to_list())))}
    rows = sorted(
        zip(g["gkey"].to_list(), g["year"].to_list(), g["len"].to_list()),
        key=lambda r: (r[1], r[0]),
    )
    return [(keys[k], int(y), int(n)) for k, y, n in rows]


def _zipf_probs(k: int) -> np.ndarray:
    w = 1.0 / np.arange(1, k + 1)
    return w / w.sum()


def simulate(sk: Skeleton, sc: dict, rng: np.random.Generator) -> tuple[st.StatData, dict]:
    """한 번의 모의. 반환: ``StatData`` 와 β̂용 (x, y) 전역 배열."""
    E = sk.E
    s = sc["persist_share"]
    sigma = sc["noise_bp"] * 1e-4
    om = sc["operator_share"]
    tau = sigma * np.sqrt(s / (1.0 - s)) if s > 0 else 0.0
    ter_sd = sc["ter_sd_bp"] / 100.0  # % 단위 (5bp = 0.05%)
    F = ter_sd * rng.standard_normal(E)
    A = F.copy()
    op = rng.choice(N_OPERATORS, size=E, p=_zipf_probs(N_OPERATORS))
    u_op = rng.standard_normal(N_OPERATORS)
    persist = tau * (np.sqrt(1 - om) * rng.standard_normal(E) + np.sqrt(om) * u_op[op])
    # 운용사-연도 충격: 연도마다 운용사마다 하나. 형성 Y 는 v[Y], 결과 Y+1 은 v[Y+1] (연속한 두 연도가 겹쳐 쓴다)
    y_min = int(sk.years.min())
    v = rng.standard_normal((int(sk.years.max()) - y_min + 2, N_OPERATORS))
    yi = np.repeat(sk.years - y_min, sk.sizes)  # ETF-연마다 형성 연도 번호
    gross_y = persist + sigma * (np.sqrt(1 - om) * rng.standard_normal(E) + np.sqrt(om) * v[yi, op])
    gross_y1 = persist + sigma * (
        np.sqrt(1 - om) * rng.standard_normal(E) + np.sqrt(om) * v[yi + 1, op]
    )
    beta = sc["beta"]
    rtr_y = gross_y - beta * F / 100.0
    rtr_y1 = gross_y1 - beta * A / 100.0
    G = rtr_y + F / 100.0
    H = rtr_y1 + A / 100.0
    gd = G[sk.ga] - G[sk.gb]
    ok = np.abs(gd) >= st.SCORE_THR
    gsign = np.where(ok, np.sign(gd), 0).astype(np.int8)
    sd = st.StatData(
        year=sk.years,
        group=sk.group,
        group_keys=[str(i) for i in range(sk.n_groups)],
        n=sk.sizes,
        T=(sk.sizes * (sk.sizes - 1) // 2).astype(float),
        pair_off=sk.pair_off,
        pa=sk.pa,
        pb=sk.pb,
        pgy=sk.pgy,
        gsign=gsign,
        hmat_flat=H[sk.hmat_src],
        hoff=sk.hoff,
        etf_off=sk.etf_off,
    )
    return sd, {"x": -A / 100.0, "y": rtr_y1}


def one_rep(args: tuple) -> dict:
    sk, sc, seed = args
    rng = np.random.default_rng(seed)
    sd, xy = simulate(sk, sc, rng)
    r = st.judge_c(sd, n_perm=SIM_PERM, n_boot=SIM_BOOT, seed=int(rng.integers(1 << 31)))
    b = st.beta_hat_fast(
        xy["x"], xy["y"], sd.etf_off, sd.group, n_boot=SIM_BOOT, seed=int(rng.integers(1 << 31))
    )
    g = r["gates"]
    cap = b["rule_1a"]
    return {
        "C": r["C"],
        "G1": g["G1"],
        "G2": g["G2"],
        "G3": g["G3"],
        "cap_1a": bool(cap),
        "beta_hat": b["beta"],
        "grade_A": bool(g["G1"] and g["G2"] and g["G3"] and not cap),
        "pass_G2_after_1a": bool(g["G1"] and g["G2"] and not cap),
        "n_pairs_cond": r["n_pairs_cond"],
    }


def scenario_list(grid: dict | None = None) -> list[dict]:
    grid = grid or GRID
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]


def run_power(
    sk: Skeleton, scenarios: list[dict], n_rep: int = 200, seed: int = 20261012, workers: int = 4
) -> list[dict]:
    """시나리오마다 ``n_rep`` 번 모의. 결과: 시나리오 칸 + 비율들."""
    rows = []
    jobs = []
    for i, sc in enumerate(scenarios):
        for r in range(n_rep):
            jobs.append((sk, sc, seed + i * 100003 + r))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(one_rep, jobs, chunksize=max(1, n_rep // 4)))
    else:
        res = [one_rep(j) for j in jobs]
    for i, sc in enumerate(scenarios):
        rr = res[i * n_rep : (i + 1) * n_rep]
        rows.append(
            {
                **sc,
                "n_rep": n_rep,
                "mean_C": float(np.mean([x["C"] for x in rr])),
                "p_G1": float(np.mean([x["G1"] for x in rr])),
                "p_G2": float(np.mean([x["G1"] and x["G2"] for x in rr])),
                "p_G2_raw": float(np.mean([x["G2"] for x in rr])),
                "p_cap_1a": float(np.mean([x["cap_1a"] for x in rr])),
                "p_G2_after_1a": float(np.mean([x["pass_G2_after_1a"] for x in rr])),
                "p_grade_A": float(np.mean([x["grade_A"] for x in rr])),
                "mean_beta_hat": float(np.mean([x["beta_hat"] for x in rr])),
                "mean_pairs_cond": float(np.mean([x["n_pairs_cond"] for x in rr])),
            }
        )
    return rows


def write_rows(rows: list[dict], path: str | Path) -> None:
    import csv

    path = Path(path)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def dump_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str))
