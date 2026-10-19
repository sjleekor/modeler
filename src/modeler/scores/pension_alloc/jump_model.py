"""R-JM: 두 상태 통계적 jump model (사전등록 §3.1·§4.2·§10 Q3~Q5·§10.2).

Shu·Yu·Mulvey(2024) 설정을 옮겼다. 우리와 다른 점(사전등록이 정한 것)은 세 가지다.
Θ를 월말 결정일마다 다시 맞춘다. 위험 상태의 위험 몫 배수는 ``m = 0.5``(안전 1.0)이다.
입력 계열은 KOSPI 가격지수 일수익이다.

원문이 비워 둔 값(§10.2 "동결 전 사용자 결정")은 모두 :class:`JMConfig`의 인자다.
``lambdas``(λ 격자)와 ``seeds``(k-means++ 시드)는 기본값이 없다. 나머지는 이 구현의 잠정
선택이고 모두 docstring에 적었다.

* 표준화: 학습창 안의 평균·표준편차로 z-점수(우리의 선택, 원문에 없음). 온라인 추정 창이
  학습창과 같으므로 같은 통계를 쓴다.
* 피쳐 예열: ``feature_warmup``일(기본 0)은 창에 넣지 않는다.
* "누적 초과수익"(상태 이름): 기본은 그 상태로 분류된 날의 일 초과수익 합(``"sum"``).
  ``"compound"``는 복리(Π(1+r)−1).
* λ 검증에서 Θ 재적합: 검증 월말마다 그 시점의 창으로 월말 재적합(결정일과 같은 방식).
* 결측 CD91은 0으로 채운다(호출자가 null 세션을 미리 걸러야 하면 걸러서 넘긴다).

결정일 t의 신호는 t까지의 자료만 쓴다(§10 Q1 ② 시험으로 확인).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

TD = 252


# --------------------------------------------------------------------------- 피쳐
def ewm_mean(x: np.ndarray, halflife: float) -> np.ndarray:
    """인과적 지수가중 평균(``y_t = a x_t + (1−a) y_{t−1}``, 첫 값은 ``x_0``)."""
    x = np.asarray(x, dtype=float)
    a = 1.0 - 0.5 ** (1.0 / halflife)
    b = 1.0 - a
    y, _ = lfilter([a], [1.0, -b], x, zi=np.array([b * x[0]]))
    return y


def downside_dev(r: np.ndarray, halflife: float) -> np.ndarray:
    """EWM 하방편차 ``sqrt(EWM[R²·1{R<0}])`` (사전등록 §3.1)."""
    r = np.asarray(r, dtype=float)
    return np.sqrt(ewm_mean(np.where(r < 0, r * r, 0.0), halflife))


def sortino(r: np.ndarray, halflife: float) -> np.ndarray:
    """EWM Sortino = EWM 평균 ÷ EWM 하방편차(같은 반감기). 하방편차 0이면 0."""
    dd = downside_dev(r, halflife)
    mu = ewm_mean(r, halflife)
    out = np.zeros_like(mu)
    np.divide(mu, dd, out=out, where=dd > 1e-12)
    return out


def jm_features(excess: np.ndarray) -> np.ndarray:
    """(T, 3): [하방편차(hl 10), Sortino(hl 20), Sortino(hl 60)]. t열은 t까지만 쓴다."""
    return np.column_stack([downside_dev(excess, 10), sortino(excess, 20), sortino(excess, 60)])


def standardize(x: np.ndarray) -> np.ndarray:
    """창 안의 평균·표준편차로 z-점수(우리의 선택)."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (x - mu) / sd


# --------------------------------------------------------------------------- JM 핵심
def _cost(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(T, K) ``½‖x_t − θ_k‖²``."""
    d = x[:, None, :] - theta[None, :, :]
    return 0.5 * (d * d).sum(axis=2)


def viterbi_jump(x: np.ndarray, theta: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Θ 고정, 목적함수 ``Σ½‖x−θ_s‖² + λ·#전환``을 최소화하는 상태열(동적계획).

    동점이면 전환하지 않는다. 반환: (상태열, 목적함수 값).
    """
    c = _cost(x, theta)
    T, K = c.shape
    if K != 2:
        raise ValueError("K=2만 지원합니다")
    c0 = c[:, 0].tolist()
    c1 = c[:, 1].tolist()
    from0 = [False] * T  # 상태 0에 오는 최선이 상태 1에서 옮긴 것인가
    from1 = [False] * T
    v0, v1 = c0[0], c1[0]
    for t in range(1, T):
        m0 = v1 + lam  # 1 -> 0
        m1 = v0 + lam  # 0 -> 1
        if m0 < v0:
            from0[t] = True
            n0 = m0
        else:
            n0 = v0
        if m1 < v1:
            from1[t] = True
            n1 = m1
        else:
            n1 = v1
        v0, v1 = c0[t] + n0, c1[t] + n1
    s = np.empty(T, dtype=np.int64)
    cur = 0 if v0 <= v1 else 1
    obj = min(v0, v1)
    s[-1] = cur
    for t in range(T - 1, 0, -1):
        if cur == 0:
            cur = 1 if from0[t] else 0
        else:
            cur = 0 if from1[t] else 1
        s[t - 1] = cur
    return s, float(obj)


def kmeanspp_centers(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ 초기 중심."""
    n = len(x)
    centers = [x[int(rng.integers(n))]]
    for _ in range(1, k):
        d2 = np.min([((x - c) ** 2).sum(axis=1) for c in centers], axis=0)
        tot = d2.sum()
        p = d2 / tot if tot > 0 else np.full(n, 1.0 / n)
        centers.append(x[int(rng.choice(n, p=p))])
    return np.array(centers)


def _fit_once(
    x: np.ndarray, lam: float, seed: int, max_iter: int, k: int = 2
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    theta = kmeanspp_centers(x, k, rng)
    s, obj = viterbi_jump(x, theta, lam)
    for _ in range(max_iter):
        new = theta.copy()
        for j in range(k):
            m = s == j
            if m.any():
                new[j] = x[m].mean(axis=0)
        s2, obj2 = viterbi_jump(x, new, lam)
        theta = new
        if np.array_equal(s2, s):
            obj = obj2
            break
        s, obj = s2, obj2
    return theta, s, obj


def fit_jm(
    x: np.ndarray, lam: float, seeds: tuple[int, ...], max_iter: int = 100
) -> tuple[np.ndarray, np.ndarray, float]:
    """``len(seeds)``번 k-means++ 초기화로 돌려 목적함수가 가장 낮은 적합을 남긴다.

    동점이면 앞선 시드. 반환: (Θ (2, d), 상태열, 목적함수).
    """
    best = None
    for sd in seeds:
        fit = _fit_once(x, lam, sd, max_iter)
        if best is None or fit[2] < best[2] - 1e-12:
            best = fit
    assert best is not None
    return best


def online_last_state(x: np.ndarray, theta: np.ndarray, lam: float) -> int:
    """Θ 고정, 주어진 창(마지막 L일)에서 동적계획으로 구한 상태열의 마지막 상태."""
    s, _ = viterbi_jump(x, theta, lam)
    return int(s[-1])


def risk_state(states: np.ndarray, excess: np.ndarray, rule: str = "sum") -> int:
    """위험(bear) 상태 번호. 학습창에서 상태별 누적 초과수익이 낮은 쪽(§3.1·§10.2).

    ``rule``: ``"sum"``(분류된 날의 합) 또는 ``"compound"``(복리). 빈 상태는 −∞로 둔다.
    """
    cum = []
    for j in (0, 1):
        r = excess[states == j]
        if len(r) == 0:
            cum.append(-np.inf)
        elif rule == "sum":
            cum.append(float(r.sum()))
        elif rule == "compound":
            cum.append(float(np.prod(1.0 + r) - 1.0))
        else:
            raise ValueError(rule)
    # 동점이면 상태 1을 위험으로 둔다.
    return 0 if cum[0] < cum[1] else 1


# --------------------------------------------------------------------------- 규칙
@dataclass(frozen=True)
class JMConfig:
    lambdas: tuple[float, ...]  # λ 격자 — 미정(§10.2 결정 1), 기본값 없음
    seeds: tuple[int, ...]  # k-means++ 시드 목록 — 미정(§10.2 결정 3), 기본값 없음
    window_len: int = 3000  # 원문 l = 3000 (§3.1)
    val_months: int = 96  # 8년 검증창
    max_iter: int = 100
    val_cost: float = 0.0005  # 편도 5bp (§10 Q5 ③)
    val_delay: int = 1  # 검증 전략 1세션 지연
    m_risk: float = 0.5  # 위험 상태 위험 몫 배수 (§10 Q4)
    m_safe: float = 1.0
    label_rule: str = "sum"
    feature_warmup: int = 0

    def __post_init__(self) -> None:
        if not self.lambdas or not self.seeds:
            raise ValueError("lambdas와 seeds는 비어 있을 수 없습니다")


class JumpModelRule:
    """R-JM. ``decide(t)``는 인덱스 t까지의 자료만 쓴다.

    ``kospi_ret``·``cash_ret``: 같은 달력의 일수익(길이 T). ``month_ends``: 월 마지막 거래일의
    인덱스(오름차순, t는 이 안에 있어야 한다).
    """

    def __init__(
        self,
        kospi_ret: np.ndarray,
        cash_ret: np.ndarray,
        month_ends: np.ndarray | list[int],
        cfg: JMConfig,
    ):
        self.kospi = np.asarray(kospi_ret, dtype=float)
        self.cash = np.nan_to_num(np.asarray(cash_ret, dtype=float), nan=0.0)
        self.excess = self.kospi - self.cash
        self.feat = jm_features(self.excess)  # 인과적이라 전체를 한 번에 만들어도 t 뒤를 안 본다
        self.me = np.asarray(sorted(month_ends), dtype=int)
        self.cfg = cfg
        self._cache: dict[tuple[int, float], bool] = {}

    # -- 월말 d, λ의 위험 여부 (창은 d까지만) -------------------------------------
    def is_risk(self, d: int, lam: float) -> bool:
        key = (d, float(lam))
        if key in self._cache:
            return self._cache[key]
        cfg = self.cfg
        lo = d - cfg.window_len + 1
        if lo < cfg.feature_warmup:
            raise ValueError(f"{d}: 창({cfg.window_len}일)+예열이 모자랍니다")
        x = standardize(self.feat[lo : d + 1])
        theta, states, _ = fit_jm(x, lam, cfg.seeds, cfg.max_iter)
        bear = risk_state(states, self.excess[lo : d + 1], cfg.label_rule)
        last = online_last_state(x, theta, lam)
        out = last == bear
        self._cache[key] = out
        return out

    # -- λ 선택 ---------------------------------------------------------------------
    def val_month_ends(self, t: int) -> np.ndarray:
        me = self.me[self.me <= t]
        if len(me) < self.cfg.val_months:
            raise ValueError("검증창 월 수가 모자랍니다")
        return me[-self.cfg.val_months :]

    def validation_sharpe(self, t: int, lam: float) -> float:
        """검증창 0/1 전략(위험 상태면 CD91)의 연환산 샤프. 수익은 t까지만."""
        cfg = self.cfg
        ve = self.val_month_ends(t)
        pos = np.array([0.0 if self.is_risk(int(d), lam) else 1.0 for d in ve])
        start = int(ve[0]) + 1 + cfg.val_delay  # 첫 신호의 첫 수익일
        days = np.arange(start, t + 1)
        if len(days) < 2:
            return float("-inf")
        # 날 i의 포지션: 체결일(d+val_delay) 종가 뒤, 즉 d+val_delay < i 인 마지막 d의 신호
        eff = ve + cfg.val_delay
        j = np.searchsorted(eff, days, side="left") - 1
        p = pos[j]
        r = p * self.kospi[days] + (1.0 - p) * self.cash[days]
        # 체결일 종가에 포지션 변화분 × 편도 비용
        for jj in range(1, len(ve)):
            day = int(eff[jj])
            if start <= day <= t:
                r[day - start] -= abs(pos[jj] - pos[jj - 1]) * cfg.val_cost
        ex = r - self.cash[days]
        sd = ex.std(ddof=1)
        if sd < 1e-15:
            return 0.0
        return float(ex.mean() / sd * np.sqrt(TD))

    def select_lambda(self, t: int) -> tuple[float, dict[float, float]]:
        """검증 샤프가 가장 높은 λ. 동점이면 더 큰 λ (§10 Q5)."""
        sh = {float(lam): self.validation_sharpe(t, lam) for lam in self.cfg.lambdas}
        best = None
        for lam in sorted(sh):  # 오름차순으로 훑으며 >= 로 갱신 → 동점은 큰 λ
            if best is None or sh[lam] >= sh[best] - 1e-12:
                best = lam
        assert best is not None
        return best, sh

    # -- 결정 -----------------------------------------------------------------------
    def decide(self, t: int) -> dict:
        lam, sh = self.select_lambda(t)
        risk = self.is_risk(t, lam)
        return {
            "t": t,
            "lambda": lam,
            "is_risk": risk,
            "m": self.cfg.m_risk if risk else self.cfg.m_safe,
            "val_sharpe": sh,
        }

    def m_path(self, decisions: list[int]) -> np.ndarray:
        return np.array([self.decide(t)["m"] for t in decisions])
