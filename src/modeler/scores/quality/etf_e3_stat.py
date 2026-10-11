"""ETF E3 — 통계량·검정·등급 (사양 §6). 입력은 숫자 배열뿐이라 합성 자료와 실제 자료가 같은 코드를 지난다.

E3b′(판정, m = 1):
- 쌍 (i, j) 조건: ``|G_i − G_j| ≥ 0.0005`` (5bp), 결과 동률 ``|H_i − H_j| < 0.00005`` (0.5bp) 제외.
  ``c_p = 1`` ⇔ sign(G_i − G_j) = sign(H_i − H_j).
- 무게: 그룹-연도 (g, Y)의 무게 = 그 안 ETF-연 수 n, 조건 채운 쌍에 똑같이 나눔. 조건 채운 쌍이 그 그룹-연도 전체 쌍의
  절반 미만이면 무게에 (조건 채운 쌍 / 전체 쌍)을 곱한다.  ``C = Σ w c / Σ w`` , 귀무 0.5, 단측.
- 1차 p = 순열 검정(그룹-연도 안에서 ETF의 결과 쪽 신원을 무작위로 섞는다. 쌍 (i, j)의 결과를 (π(i), π(j)) 쌍의
  H 로 바꿔 같은 통계량을 다시 계산한다. 조건·무게도 바뀐 결과에 맞춰 다시 센다). B = 10,000,
  ``p = (1 + #{C* ≥ C}) / (B + 1)``.
- 구간 = 그룹 클러스터 부트스트랩 2,000회, 단측 95% 하한 = 5백분위, 상한 = 95백분위.
- G1: 조건 채운 쌍 ≥ 300 ∧ 쌍 있는 그룹 ≥ 30. G2: C ≥ 0.55 ∧ p < 0.05 ∧ 하한 > 0.50.
  G3: 전반(형성 2015~2019) C > 0.5 ∧ 후반(2020~2024) C > 0.5 ∧ 그룹 하나씩 뺀 C 최솟값 > 0.5.
- 등급: D = G1 또는 G2 탈락 / C = G2 통과·G3 탈락, 또는 규칙 1-a 상한, 또는 지연 > 92일 / A = 전부 통과.

E3a(기록용): β̂(그룹-연도 고정효과), 최저 TER 선택 이득, 일치율 C_a.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCORE_THR = 0.0005  # 5bp, G 차이 문턱 (처음 고른 값)
TIE_THR = 0.00005  # 0.5bp, 결과 동률 (처음 고른 값)
C_MIN = 0.55
P_MAX = 0.05
G1_PAIRS = 300
G1_GROUPS = 30
B_PERM = 10_000
B_BOOT = 2_000
SEED = 20261014  # 제안값. 동결 때 확정
SPLIT_YEAR = 2019  # 전반 ≤ 2019 < 후반
STOP_UPPER = 0.55  # D일 때 상한 < 0.55 → 접기 제안
BETA_LO = 0.7  # 규칙 1-a: β̂ 90% 구간 상한 < 0.7 또는 하한 > 1.3
BETA_HI = 1.3
F_THR_PCT = 0.05  # E3a 일치율: |F_i − F_j| ≥ 5bp = 0.05 %p


@dataclass
class StatData:
    """쌍 단위로 펼친 입력. 쌍은 그룹-연도 순서대로 붙어 있다(그룹-연도마다 쌍이 하나 이상)."""

    year: np.ndarray  # (n_gy,)
    group: np.ndarray  # (n_gy,) 그룹 번호
    group_keys: list[str]
    n: np.ndarray  # (n_gy,) 제외 뒤 ETF-연 수
    T: np.ndarray  # (n_gy,) 전체 쌍 수 C(n, 2)
    pair_off: np.ndarray  # (n_gy + 1,)
    pa: np.ndarray  # (P,) 쌍의 ETF 로컬 번호 a < b
    pb: np.ndarray
    pgy: np.ndarray  # (P,) 그룹-연도 번호
    gsign: np.ndarray  # (P,) 점수 차 부호(조건 미충족·규칙 실패면 0)
    hmat_flat: np.ndarray  # 그룹-연도별 n×n 결과 행렬을 이어 붙임(NaN = 정의 안 됨)
    hoff: np.ndarray  # (n_gy,) 각 그룹-연도 행렬의 시작
    etf_off: np.ndarray  # (n_gy + 1,) ETF-연 전역 번호 시작
    tie_thr: float = TIE_THR

    @property
    def n_gy(self) -> int:
        return len(self.n)

    @property
    def n_pairs(self) -> int:
        return len(self.pa)

    def hd_real(self) -> np.ndarray:
        """실제 결과 차 H[a,b] − H[b,a]."""
        n_p = self.n[self.pgy]
        off = self.hoff[self.pgy]
        return (
            self.hmat_flat[off + self.pa * n_p + self.pb]
            - self.hmat_flat[off + self.pb * n_p + self.pa]
        )


def make_stat_data(
    sizes: list[int],
    years: list[int],
    group_ids: list[int],
    group_keys: list[str],
    score_diff: list[np.ndarray],
    outcome: list[np.ndarray],
    pair_valid: list[np.ndarray],
    score_thr: float = SCORE_THR,
    tie_thr: float = TIE_THR,
) -> StatData:
    """그룹-연도별 행렬에서 ``StatData`` 를 만든다.

    ``score_diff[g][a, b]`` = a 쪽 점수 − b 쪽 점수(부호만 쓴다), ``outcome[g][a, b]`` = (a, b) 쌍에서 a 의 결과,
    ``pair_valid[g][a, b]`` = 쌍-연이 규칙을 통과했는가(실제 통계량에서 쓰는 쌍).
    """
    pa, pb, pgy, gsign = [], [], [], []
    hflat, hoff = [], []
    pair_off = [0]
    pos = 0
    for g, n in enumerate(sizes):
        hoff.append(pos)
        hflat.append(np.asarray(outcome[g], dtype=float).ravel())
        pos += n * n
        for a in range(n):
            for b in range(a + 1, n):
                sd = score_diff[g][a, b]
                ok = bool(pair_valid[g][a, b]) and np.isfinite(sd) and abs(sd) >= score_thr
                pa.append(a)
                pb.append(b)
                pgy.append(g)
                gsign.append(int(np.sign(sd)) if ok else 0)
        pair_off.append(len(pa))
    n_arr = np.array(sizes, dtype=np.int64)
    return StatData(
        year=np.array(years),
        group=np.array(group_ids),
        group_keys=list(group_keys),
        n=n_arr,
        T=(n_arr * (n_arr - 1) // 2).astype(float),
        pair_off=np.array(pair_off),
        pa=np.array(pa, dtype=np.int64),
        pb=np.array(pb, dtype=np.int64),
        pgy=np.array(pgy, dtype=np.int64),
        gsign=np.array(gsign, dtype=np.int8),
        hmat_flat=np.concatenate(hflat) if hflat else np.zeros(0),
        hoff=np.array(hoff, dtype=np.int64),
        etf_off=np.concatenate([[0], np.cumsum(n_arr)]),
        tie_thr=tie_thr,
    )


# ---------------------------------------------------------------- 통계량
def gy_sums(cond: np.ndarray, corr: np.ndarray, sd: StatData) -> tuple[np.ndarray, np.ndarray]:
    """(B, P) 조건·일치 → (B, n_gy) 그룹-연도별 (Σ w·c, w). 조건 쌍이 없는 그룹-연도는 0."""
    starts = sd.pair_off[:-1]
    k = np.add.reduceat(cond.astype(np.float64), starts, axis=1)
    c = np.add.reduceat((cond & corr).astype(np.float64), starts, axis=1)
    frac = k / sd.T
    w = sd.n * np.where(frac < 0.5, frac, 1.0)
    w = np.where(k > 0, w, 0.0)
    num = np.where(k > 0, w * c / np.maximum(k, 1.0), 0.0)
    return num, w


def _cond_corr(sd: StatData, hd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gs = sd.gsign[None, :] if hd.ndim == 2 else sd.gsign
    with np.errstate(invalid="ignore"):
        cond = (gs != 0) & np.isfinite(hd) & (np.abs(hd) >= sd.tie_thr)
        corr = np.sign(hd) == gs
    return cond, corr


def real_stat(sd: StatData) -> dict:
    """실제 C, 그룹-연도별 (num, w), 조건 쌍 수, 쌍이 있는 그룹 수."""
    hd = sd.hd_real()[None, :]
    cond, corr = _cond_corr(sd, hd)
    num, w = gy_sums(cond, corr, sd)
    num, w = num[0], w[0]
    k = np.add.reduceat(cond[0].astype(float), sd.pair_off[:-1])
    has = k > 0
    C = float(num.sum() / w.sum()) if w.sum() > 0 else float("nan")
    return {
        "C": C,
        "num": num,
        "w": w,
        "k": k,
        "n_pairs_cond": int(k.sum()),
        "n_groups_with_pairs": int(len(set(sd.group[has].tolist()))),
        "n_group_years_with_pairs": int(has.sum()),
    }


def perm_pvalue(
    sd: StatData, c_real: float, n_perm: int = B_PERM, seed: int = SEED, chunk: int = 500
) -> tuple[float, np.ndarray]:
    """그룹-연도 안 결과 쪽 신원 순열 p. 반환 (p, 순열 C* 배열)."""
    rng = np.random.default_rng(seed)
    E = int(sd.etf_off[-1])
    gy_of_etf = np.repeat(np.arange(sd.n_gy), sd.n)
    off_p = sd.etf_off[sd.pgy]  # 쌍의 그룹-연도 ETF 시작
    ga = off_p + sd.pa
    gb = off_p + sd.pb
    n_p = sd.n[sd.pgy]
    hoff_p = sd.hoff[sd.pgy]
    out = []
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        keys = rng.random((b, E)) + gy_of_etf[None, :]
        order = np.argsort(keys, axis=1)  # 그룹-연도 블록 안에서만 섞인다
        pi_a = order[:, ga] - off_p[None, :]
        pi_b = order[:, gb] - off_p[None, :]
        h_ab = sd.hmat_flat[hoff_p[None, :] + pi_a * n_p[None, :] + pi_b]
        h_ba = sd.hmat_flat[hoff_p[None, :] + pi_b * n_p[None, :] + pi_a]
        hd = h_ab - h_ba
        cond, corr = _cond_corr(sd, hd)
        num, w = gy_sums(cond, corr, sd)
        ws = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(ws > 0, num.sum(axis=1) / ws, np.nan))
        done += b
    cs = np.concatenate(out)
    ge = int(np.sum(cs >= c_real - 1e-12))
    return (1 + ge) / (n_perm + 1), cs


def group_sums(num: np.ndarray, w: np.ndarray, sd: StatData) -> tuple[np.ndarray, np.ndarray]:
    """그룹-연도 합 → 그룹 합."""
    ng = len(sd.group_keys)
    return np.bincount(sd.group, weights=num, minlength=ng), np.bincount(
        sd.group, weights=w, minlength=ng
    )


def cluster_boot(
    num_g: np.ndarray, w_g: np.ndarray, n_boot: int = B_BOOT, seed: int = SEED + 1
) -> np.ndarray:
    """그룹 복원추출 부트스트랩 C* 배열(그룹 수만큼 뽑는다)."""
    rng = np.random.default_rng(seed)
    ng = len(num_g)
    cnt = rng.multinomial(ng, np.full(ng, 1.0 / ng), size=n_boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = cnt @ w_g
        return np.where(d > 0, (cnt @ num_g) / d, np.nan)


def subset_c(num: np.ndarray, w: np.ndarray, mask: np.ndarray) -> float:
    d = w[mask].sum()
    return float(num[mask].sum() / d) if d > 0 else float("nan")


def leave_one_group_out(num_g: np.ndarray, w_g: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return (num_g.sum() - num_g) / (w_g.sum() - w_g)


# ---------------------------------------------------------------- 게이트·등급
def gates(sd: StatData, rs: dict, p: float, boot: np.ndarray) -> dict:
    """G1·G2·G3. 부트스트랩 하한 = 5백분위, 상한 = 95백분위(NaN 제외)."""
    b = boot[np.isfinite(boot)]
    lo = float(np.percentile(b, 5)) if len(b) else float("nan")
    hi = float(np.percentile(b, 95)) if len(b) else float("nan")
    C = rs["C"]
    g1 = rs["n_pairs_cond"] >= G1_PAIRS and rs["n_groups_with_pairs"] >= G1_GROUPS
    g2 = bool(np.isfinite(C) and C >= C_MIN and p < P_MAX and np.isfinite(lo) and lo > 0.5)
    early = subset_c(rs["num"], rs["w"], sd.year <= SPLIT_YEAR)
    late = subset_c(rs["num"], rs["w"], sd.year > SPLIT_YEAR)
    ng, wg = group_sums(rs["num"], rs["w"], sd)
    with np.errstate(invalid="ignore"):
        loo = leave_one_group_out(ng, wg)
    loo = loo[np.isfinite(loo)]
    loo_min = float(loo.min()) if len(loo) else float("nan")
    g3 = bool(early > 0.5 and late > 0.5 and loo_min > 0.5)
    return {
        "G1": bool(g1),
        "G2": g2,
        "G3": g3,
        "boot_lower_95": lo,
        "boot_upper_95": hi,
        "C_early": early,
        "C_late": late,
        "C_loo_min": loo_min,
    }


def rule_1a(beta_lo: float, beta_hi: float) -> bool:
    """β̂ 90% 구간 상한 < 0.7 또는 하한 > 1.3 이면 E3b′ 등급 상한 C."""
    return bool(
        (np.isfinite(beta_hi) and beta_hi < BETA_LO) or (np.isfinite(beta_lo) and beta_lo > BETA_HI)
    )


def grade(g: dict, cap_1a: bool = False, lag: dict | None = None) -> dict:
    """등급과 꼬리표. ``lag`` 는 ``etf_e3_fees.lag_summary`` 결과(없으면 잠정)."""
    lag = lag or {"cap": False, "status": "잠정(지연 실측 전)"}
    if not g["G1"]:
        base = "D"
        reason = "표본 부족(G1)"
    elif not g["G2"]:
        base = "D"
        reason = "G2 탈락"
    else:
        caps = []
        if not g["G3"]:
            caps.append("G3 탈락")
        if cap_1a:
            caps.append("규칙 1-a 상한(β̂ 구간이 0.7~1.3 밖)")
        if lag.get("cap"):
            caps.append("지연 규칙 부족(실측 최대 > 92일)")
        base = "C" if caps else "A"
        reason = ", ".join(caps) if caps else "G1·G2·G3 통과"
    labels = []
    if base in ("A", "C"):
        labels.append("(전진 확인 대기)")
        if lag.get("status") != "실측":
            labels.append("잠정(지연 실측 전)")
    stop = None
    if base == "D":
        hi = g["boot_upper_95"]
        stop = "접기 제안" if (np.isfinite(hi) and hi < STOP_UPPER) else "보류(감쇠 폭 추정)"
    return {
        "grade": base,
        "reason": reason,
        "labels": labels,
        "stop_proposal": stop,
        "cap_1a_active": bool(cap_1a and g["G1"] and g["G2"]),
    }


def judge_c(sd: StatData, n_perm: int = B_PERM, n_boot: int = B_BOOT, seed: int = SEED) -> dict:
    """C, p, 부트스트랩, G1~G3 를 한 번에."""
    rs = real_stat(sd)
    p, cs = perm_pvalue(sd, rs["C"], n_perm, seed)
    ng, wg = group_sums(rs["num"], rs["w"], sd)
    boot = cluster_boot(ng, wg, n_boot, seed + 1)
    g = gates(sd, rs, p, boot)
    return {
        "C": rs["C"],
        "p": p,
        "n_pairs_cond": rs["n_pairs_cond"],
        "n_groups_with_pairs": rs["n_groups_with_pairs"],
        "gates": g,
        "rs": rs,
        "boot": boot,
        "perm": cs,
    }


def c_only(sd: StatData, n_boot: int = B_BOOT, seed: int = SEED) -> dict:
    """기록용 변형: C 와 부트스트랩 구간만(순열 없음)."""
    rs = real_stat(sd)
    ng, wg = group_sums(rs["num"], rs["w"], sd)
    boot = cluster_boot(ng, wg, n_boot, seed + 1)
    b = boot[np.isfinite(boot)]
    return {
        "C": rs["C"],
        "n_pairs_cond": rs["n_pairs_cond"],
        "n_groups_with_pairs": rs["n_groups_with_pairs"],
        "boot_lower_95": float(np.percentile(b, 5)) if len(b) else float("nan"),
        "boot_upper_95": float(np.percentile(b, 95)) if len(b) else float("nan"),
    }


def c_by_year(sd: StatData, rs: dict) -> dict[int, float]:
    out = {}
    for y in sorted(set(sd.year.tolist())):
        out[int(y)] = subset_c(rs["num"], rs["w"], sd.year == y)
    return out


# ---------------------------------------------------------------- E3a (기록용)
def beta_hat(
    xs: list[np.ndarray],
    ys: list[np.ndarray],
    group: np.ndarray,
    n_boot: int = B_BOOT,
    seed: int = SEED + 2,
    q: tuple[float, float] = (5.0, 95.0),
) -> dict:
    """그룹-연도 고정효과 회귀 y = β x. ``xs[g]``·``ys[g]`` 는 그 그룹-연도의 (NaN 없는) 값. 그룹 클러스터 부트스트랩 90% 구간."""
    ng = int(group.max()) + 1 if len(group) else 0
    sxy = np.zeros(ng)
    sxx = np.zeros(ng)
    for k, (x, y) in enumerate(zip(xs, ys)):
        if len(x) < 2:
            continue
        xd = x - x.mean()
        yd = y - y.mean()
        sxy[group[k]] += float((xd * yd).sum())
        sxx[group[k]] += float((xd * xd).sum())
    beta = float(sxy.sum() / sxx.sum()) if sxx.sum() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(ng, np.full(ng, 1.0 / ng), size=n_boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bb = (cnt @ sxy) / (cnt @ sxx)
    b = bb[np.isfinite(bb)]
    lo, hi = (
        (float(np.percentile(b, q[0])), float(np.percentile(b, q[1])))
        if len(b)
        else (float("nan"), float("nan"))
    )
    return {
        "beta": beta,
        "ci_lo": lo,
        "ci_hi": hi,
        "rule_1a": rule_1a(lo, hi),
        "n_group_years": int(sum(len(x) >= 2 for x in xs)),
    }


def beta_hat_fast(
    x: np.ndarray,
    y: np.ndarray,
    etf_off: np.ndarray,
    group: np.ndarray,
    n_boot: int = B_BOOT,
    seed: int = SEED + 2,
) -> dict:
    """``beta_hat`` 의 전부-값-있음 판(모의용). x·y 는 ETF-연 전역 배열, ``etf_off`` 는 그룹-연도 시작."""
    starts = etf_off[:-1]
    n = np.diff(etf_off)
    mx = np.add.reduceat(x, starts) / n
    my = np.add.reduceat(y, starts) / n
    xd = x - np.repeat(mx, n)
    yd = y - np.repeat(my, n)
    sxy_gy = np.add.reduceat(xd * yd, starts)
    sxx_gy = np.add.reduceat(xd * xd, starts)
    ng = int(group.max()) + 1
    sxy = np.bincount(group, weights=sxy_gy, minlength=ng)
    sxx = np.bincount(group, weights=sxx_gy, minlength=ng)
    beta = float(sxy.sum() / sxx.sum()) if sxx.sum() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(ng, np.full(ng, 1.0 / ng), size=n_boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bb = (cnt @ sxy) / (cnt @ sxx)
    b = bb[np.isfinite(bb)]
    lo, hi = (
        (float(np.percentile(b, 5)), float(np.percentile(b, 95)))
        if len(b)
        else (float("nan"), float("nan"))
    )
    return {"beta": beta, "ci_lo": lo, "ci_hi": hi, "rule_1a": rule_1a(lo, hi)}


def lowest_fee_gain(
    F: list[np.ndarray],
    R: list[np.ndarray],
    group: np.ndarray,
    n_boot: int = B_BOOT,
    seed: int = SEED + 3,
) -> dict:
    """그룹-연도마다 F 최저 ETF의 RTR(Y+1) − 그 그룹-연도 RTR(Y+1) 중앙값(bp). 최저 동률이면 그 그룹-연도 뺌.

    그룹-연도 무게(= ETF-연 수) 평균과 그룹 클러스터 부트스트랩 구간(5·95백분위).
    """
    ng = int(group.max()) + 1 if len(group) else 0
    num = np.zeros(ng)
    den = np.zeros(ng)
    gains = []
    n_tie = 0
    for k, (f, r) in enumerate(zip(F, R)):
        if len(f) < 2:
            continue
        m = f.min()
        if int((f == m).sum()) != 1:
            n_tie += 1
            continue
        g = float((r[int(np.argmin(f))] - np.median(r)) * 1e4)
        gains.append(g)
        num[group[k]] += len(f) * g
        den[group[k]] += len(f)
    val = float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(ng, np.full(ng, 1.0 / ng), size=n_boot).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        bb = (cnt @ num) / (cnt @ den)
    b = bb[np.isfinite(bb)]
    return {
        "gain_bp_weighted_mean": val,
        "gain_bp_median_over_group_years": float(np.median(gains)) if gains else float("nan"),
        "ci_lo": float(np.percentile(b, 5)) if len(b) else float("nan"),
        "ci_hi": float(np.percentile(b, 95)) if len(b) else float("nan"),
        "n_group_years": len(gains),
        "n_dropped_tie": n_tie,
    }
