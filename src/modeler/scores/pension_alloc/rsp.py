"""R-SP: 샤프를 직접 학습하는 작은 softmax 정책 (사전등록 §3.3·§10 Q5·Q7).

여섯 슬롯(069500·143850·114260·148070·132030·현금)의 월 비중을 직접 낸다.

* 입력(슬롯 i): 직전 21거래일 누적수익, 63거래일 누적수익, 직전 21일 일간 σ(표본 sd).
* 표준화: 학습에 쓴 (월×슬롯) 표본 전체에서 입력 종류마다 평균·표준편차로 z-점수
  (우리의 선택, 사전등록은 "동결 때"로 남김). 통계는 그 학습 구간에서만 구한다.
* 점수 ``s_i = α·x_i + β_i`` (α 3개 공유 + β 6개 = 9개 모수), ``w = softmax(s)``.
* 손실 ``−Sharpe(CD91 초과 월수익) + γ‖θ‖²`` (F11). 월 초과수익은
  ``r_ex = (w·rr).sum − rr[:, cash]``(현금 슬롯 수익이 같은 기간 CD91)이고, 샤프는 연환산 없이
  ``mean(r_ex) / sqrt(var(r_ex) + ε²)``, ``ε = 0.005``(월 0.5%, 사전 고정 ``SHARPE_EPS``)다.
  무위험을 빼지 않으면 변동이 거의 없는 현금 비중이 무한히 좋아 "전부 현금"이 최적이 된다.
  부드러운 하한은 전부 현금(0/0)에서 기울기가 커지는 것을 막는다. 학습 때 월수익은
  체결일(d+1) 종가부터 다음 체결일 종가까지의 슬롯 수익을 비중으로 가중한 값이다
  (월중 비중 흐름·비용은 무시, 사후에 장부가 정확히 계산). 안쪽 γ 선택도 같은 통계량을 쓴다.
* 현금(F7): 슬롯 수익의 현금 열은 기준선 장부와 같은 계열이다. 비거나 오래된 날은 NaN이고
  채우지 않는다. 쌍의 수익 기간에 NaN이 낀 달은 학습에서 빼고 ``n_dropped``로 센다. 현금 슬롯
  입력(누적수익·σ)은 NaN 날을 건너뛰고 계산한다.
* 확장창, 학습 쌍이 60개월 이상일 때 시작, 12개월마다 재학습.
* γ는 재학습마다 학습창 마지막 24개월을 떼어 안쪽 시간순 분할로 고른다(앞부분 학습 →
  24개월 앙상블 샤프 → 최댓값, 동점은 더 큰 γ). 고른 γ로 전체 창에 다시 맞춘다.
* 시드 ``n_seed = 10``(20261018~20261027)의 정책 비중 평균이 결정 비중이다(시드 선택 없음).
  시드별 비중은 ``seed_weights``로 남긴다(시드 sd 판정용).
* IRP 구조(Q7): ``w = 0.7·softmax(s) + 0.3·u``, u는 (114260·148070·현금) 1/3씩.
  학습할 때부터 이 구조를 쓴다(사후 clip 없음).

결정일 d_k의 비중은 d_k까지의 자료만 쓴다. 학습 쌍 j의 실현 수익은 ``d_{j+1}+1``까지 닫혀야
하므로 ``j ≤ k−2``인 쌍만 학습에 들어간다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

SHARPE_EPS = 0.005  # F11: 부드러운 하한 sd = sqrt(var + ε²)
SLOTS = ("069500", "143850", "114260", "148070", "132030", "cash")
SAFE_SLOTS = (2, 3, 5)  # 114260·148070·현금
IRP_RISK_CAP = 0.7
DEFAULT_SEEDS = tuple(range(20261018, 20261028))  # §10 Q5 ①
N_FEAT = 3
N_THETA = N_FEAT + len(SLOTS)


# --------------------------------------------------------------------------- 입력·표본
def slot_inputs(slot_ret: np.ndarray, t: int) -> np.ndarray:
    """t일(포함)까지의 (6, 3) 입력: 21일 누적, 63일 누적, 21일 σ."""
    if t < 62:
        raise ValueError("입력에 63거래일이 필요합니다")
    r21 = slot_ret[t - 20 : t + 1]
    r63 = slot_ret[t - 62 : t + 1]
    # 현금 열의 NaN 날은 건너뛴다(채우지 않는다). ETF 열에 NaN이 있으면 결과가 NaN -> 예외.
    out = np.column_stack(
        [
            np.nanprod(1.0 + r21, axis=0) - 1.0,
            np.nanprod(1.0 + r63, axis=0) - 1.0,
            np.nanstd(r21, axis=0, ddof=1),
        ]
    )
    etf_nan = np.isnan(slot_ret[t - 62 : t + 1, :5]).any()
    if etf_nan or np.isnan(out).any():
        raise ValueError(f"{t}일 입력에 결측이 있습니다")
    return out


def pair_return(slot_ret: np.ndarray, d: int, d_next: int) -> np.ndarray:
    """결정일 d의 비중을 체결일 d+1 종가부터 d_next+1 종가까지 들고 간 슬롯 수익 (6,)."""
    return np.prod(1.0 + slot_ret[d + 2 : d_next + 2], axis=0) - 1.0


# --------------------------------------------------------------------------- 정책
def _softmax(s: np.ndarray) -> np.ndarray:
    z = s - s.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _unif_safe() -> np.ndarray:
    u = np.zeros(len(SLOTS))
    u[list(SAFE_SLOTS)] = 1.0 / 3.0
    return u


def policy_weights(theta: np.ndarray, xs: np.ndarray, structure: str = "main") -> np.ndarray:
    """``xs``: (…, 6, 3) 표준화된 입력 -> (…, 6) 비중."""
    s = xs @ theta[:N_FEAT] + theta[N_FEAT:]
    p = _softmax(s)
    if structure == "main":
        return p
    if structure == "irp":
        return IRP_RISK_CAP * p + (1.0 - IRP_RISK_CAP) * _unif_safe()
    raise ValueError(structure)


def excess_sharpe(r_ex: np.ndarray, eps: float = SHARPE_EPS) -> float:
    """``mean / sqrt(var + ε²)`` (F11, 연환산 없음)."""
    return float(r_ex.mean() / np.sqrt(r_ex.var() + eps * eps))


def loss_and_grad(
    theta: np.ndarray,
    xs: np.ndarray,
    rr: np.ndarray,
    gamma: float,
    structure: str = "main",
) -> tuple[float, np.ndarray]:
    """``−Sharpe(CD91 초과 월수익) + γ‖θ‖²``와 해석적 기울기. xs (n,6,3), rr (n,6; 마지막 열 현금).

    ``r_ex_j = Σ_i w_ji rr_ji − rr_j,cash``, ``S = μ/σ``, ``σ = sqrt(var + ε²)``.
    """
    n = len(rr)
    c = 1.0 if structure == "main" else IRP_RISK_CAP
    s = xs @ theta[:N_FEAT] + theta[N_FEAT:]
    p = _softmax(s)
    w = p if structure == "main" else IRP_RISK_CAP * p + (1.0 - IRP_RISK_CAP) * _unif_safe()
    r = (w * rr).sum(axis=1) - rr[:, -1]
    mu = r.mean()
    sd = np.sqrt(((r - mu) ** 2).mean() + SHARPE_EPS**2)
    sharpe = mu / sd
    dS_dr = (1.0 / sd - mu * (r - mu) / sd**3) / n  # (n,)
    pr = (p * rr).sum(axis=1, keepdims=True)
    ds = dS_dr[:, None] * c * p * (rr - pr)  # dS/ds_ji  (n,6)
    g = np.zeros_like(theta)
    g[:N_FEAT] = np.einsum("ji,jif->f", ds, xs)
    g[N_FEAT:] = ds.sum(axis=0)
    loss = -sharpe + gamma * float(theta @ theta)
    return float(loss), -g + 2.0 * gamma * theta


@dataclass
class Policy:
    theta: np.ndarray
    mean: np.ndarray  # (3,)
    std: np.ndarray  # (3,)
    structure: str

    def weights(self, x_raw: np.ndarray) -> np.ndarray:
        return policy_weights(self.theta, (x_raw - self.mean) / self.std, self.structure)


def _stats(x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = x_raw.reshape(-1, N_FEAT)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    return mean, np.where(std > 1e-12, std, 1.0)


def fit_policy(
    x_raw: np.ndarray, rr: np.ndarray, gamma: float, seed: int, structure: str = "main"
) -> Policy:
    mean, std = _stats(x_raw)
    xs = (x_raw - mean) / std
    theta0 = np.random.default_rng(seed).normal(0.0, 0.1, N_THETA)
    res = minimize(
        loss_and_grad, theta0, args=(xs, rr, gamma, structure), jac=True, method="L-BFGS-B"
    )
    return Policy(res.x, mean, std, structure)


def train_ensemble(
    x_raw: np.ndarray,
    rr: np.ndarray,
    gamma: float,
    seeds: tuple[int, ...],
    structure: str = "main",
) -> list[Policy]:
    return [fit_policy(x_raw, rr, gamma, s, structure) for s in seeds]


def ensemble_weights(policies: list[Policy], x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(시드별 비중 (S,6), 평균 비중 (6,)). 평균이 결정 비중이다."""
    sw = np.array([p.weights(x_raw) for p in policies])
    return sw, sw.mean(axis=0)


def inner_split(n: int, holdout: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """안쪽 시간순 분할: (앞부분 학습 인덱스, 마지막 ``holdout``개월 인덱스). 겹치지 않는다."""
    if n <= holdout:
        raise ValueError("학습 쌍이 안쪽 검증 길이 이하입니다")
    return np.arange(n - holdout), np.arange(n - holdout, n)


def monthly_sharpe(r_ex: np.ndarray) -> float:
    """안쪽 γ 선택용 통계량: CD91 초과 월수익의 부드러운 하한 샤프(F11)."""
    return excess_sharpe(r_ex)


def select_gamma(
    x_raw: np.ndarray,
    rr: np.ndarray,
    gammas: tuple[float, ...],
    seeds: tuple[int, ...],
    structure: str = "main",
    holdout: int = 24,
) -> tuple[float, dict[float, float]]:
    """안쪽 분할에서 앙상블 24개월 샤프가 가장 높은 γ. 동점이면 더 큰 γ."""
    tr, ho = inner_split(len(rr), holdout)
    score: dict[float, float] = {}
    for g in gammas:
        pol = train_ensemble(x_raw[tr], rr[tr], g, seeds, structure)
        w = np.array([ensemble_weights(pol, x_raw[j])[1] for j in ho])
        score[float(g)] = monthly_sharpe((w * rr[ho]).sum(axis=1) - rr[ho][:, -1])
    best = None
    for g in sorted(score):
        if best is None or score[g] >= score[best] - 1e-12:
            best = g
    assert best is not None
    return best, score


# --------------------------------------------------------------------------- 규칙
class SoftmaxPolicyRule:
    """R-SP. ``month_ends``는 데이터의 모든 월말 결정일(63거래일 이상 지난 것부터)이다.

    ``decide(k)``의 k는 ``month_ends`` 안의 위치다. 학습 쌍은 입력·수익에 NaN이 없는 것만
    쓴다(F7). 재학습은 유효한 쌍이 ``min_train``개 이상이 되는 첫 결정에서 시작해
    ``retrain_every``개월마다 한다.
    """

    def __init__(
        self,
        slot_ret: np.ndarray,
        month_ends: np.ndarray | list[int],
        gammas: tuple[float, ...],
        seeds: tuple[int, ...] = DEFAULT_SEEDS,
        structure: str = "main",
        min_train: int = 60,
        retrain_every: int = 12,
        holdout: int = 24,
    ):
        if not gammas:
            raise ValueError("gammas는 비어 있을 수 없습니다")
        self.ret = np.asarray(slot_ret, dtype=float)
        self.me = np.asarray(sorted(month_ends), dtype=int)
        if self.me[0] < 62:
            raise ValueError("첫 결정일에 63거래일 입력이 없습니다")
        self.gammas, self.seeds, self.structure = tuple(gammas), tuple(seeds), structure
        self.min_train, self.retrain_every, self.holdout = min_train, retrain_every, holdout
        self._pairs: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        self._first_k: int | None = None
        self._pol: dict[int, tuple[list[Policy], float, dict[float, float]]] = {}

    def pair(self, j: int) -> tuple[np.ndarray, np.ndarray] | None:
        """쌍 j의 (입력 (6,3), 월 수익 (6,)). NaN이 끼면 ``None``(학습에서 뺀다)."""
        if j not in self._pairs:
            try:
                x = slot_inputs(self.ret, int(self.me[j]))
                y = pair_return(self.ret, int(self.me[j]), int(self.me[j + 1]))
                ok = not np.isnan(y).any()
            except ValueError:
                ok = False
            self._pairs[j] = (x, y) if ok else None
        return self._pairs[j]

    @property
    def first_k(self) -> int:
        """유효한 쌍 j ≤ k−2가 ``min_train``개 이상이 되는 첫 결정 위치."""
        if self._first_k is None:
            n_ok = 0
            for j in range(len(self.me) - 1):
                n_ok += self.pair(j) is not None
                if n_ok >= self.min_train:
                    self._first_k = j + 2
                    break
            else:
                raise ValueError("학습 쌍이 모자랍니다")
        return self._first_k

    def retrain_k(self, k: int) -> int:
        if k < self.first_k:
            raise ValueError("학습 쌍이 모자랍니다")
        return self.first_k + self.retrain_every * ((k - self.first_k) // self.retrain_every)

    def training_set(self, r: int) -> tuple[np.ndarray, np.ndarray]:
        """재학습 결정 r의 학습 표본(쌍 0..r−2 중 유효한 것)."""
        got = [self.pair(j) for j in range(0, r - 1)]
        got = [g for g in got if g is not None]
        return np.array([g[0] for g in got]), np.array([g[1] for g in got])

    def n_dropped(self, r: int) -> int:
        """재학습 r의 학습 쌍 중 NaN 때문에 뺀 수."""
        return sum(self.pair(j) is None for j in range(0, r - 1))

    def _policies(self, r: int) -> tuple[list[Policy], float, dict[float, float]]:
        if r not in self._pol:
            x, y = self.training_set(r)
            g, score = select_gamma(x, y, self.gammas, self.seeds, self.structure, self.holdout)
            self._pol[r] = (train_ensemble(x, y, g, self.seeds, self.structure), g, score)
        return self._pol[r]

    def decide(self, k: int) -> dict:
        r = self.retrain_k(k)
        pols, g, score = self._policies(r)
        x = slot_inputs(self.ret, int(self.me[k]))
        sw, w = ensemble_weights(pols, x)
        return {
            "k": k,
            "t": int(self.me[k]),
            "weights": w,
            "seed_weights": sw,
            "gamma": g,
            "gamma_scores": score,
            "retrain_k": r,
            "n_dropped": self.n_dropped(r),
            "cash_weight": float(w[5]),
            "risk_exposure": float(w[[0, 1, 4]].sum()),
        }
