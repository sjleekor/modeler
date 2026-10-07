"""다중 검정 보정: Deflated Sharpe Ratio(DSR)와 Probability of Backtest Overfitting(PBO).

* DSR: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio".
* PBO: Bailey, Borwein, Lopez de Prado, Zhu (2017), CSCV.

``etl/metrics.py`` 와 달리 ``MODEL_CODE_FILES`` 에 들어 있지 않다. 여기를 고쳐도
기존 run 의 ``model_code_hash`` 가 바뀌지 않는다.

규칙
----
* 샤프는 **리밸런스 단위(per period)** ``mean/std(ddof=1)`` 다. 연환산은 ``annualize_sharpe`` 로 따로 한다.
* NaN·inf 가 입력에 있으면 ``ValueError`` 로 멈춘다. 조용히 빼지 않는다.
* 첨도는 초과 첨도가 아니라 4차 적률 비다 (정규분포 3).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import ndtri  # Phi^-1

EULER_GAMMA = 0.5772156649015329


def _as_clean_1d(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim != 1:
        raise ValueError(f"{name}: 1차원이어야 한다 (shape={a.shape})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name}: NaN 또는 inf 가 {int((~np.isfinite(a)).sum())}개 있다. 먼저 처리해야 한다")
    return a


def sharpe(returns) -> float:
    """per-period 샤프 ``mean/std(ddof=1)``.

    관측이 2개 미만이거나 표준편차가 0이면 ``nan`` 이다 (정의되지 않음).
    """
    a = _as_clean_1d(returns, "returns")
    if a.size < 2:
        return float("nan")
    if np.ptp(a) == 0.0:  # 상수열. std 가 부동소수 잔여값(1e-17)으로 남는 것을 막는다
        return float("nan")
    return float(a.mean() / a.std(ddof=1))


def annualize_sharpe(sr: float, horizon: int) -> float:
    """리밸런스 단위 샤프를 연환산한다. 연 250세션, 리밸런스 간격 ``horizon`` 세션."""
    return float(sr) * math.sqrt(250.0 / horizon)


def expected_max_sharpe(sr_var: float, n_trials: int) -> float:
    """N번 시험한 샤프의 기대 최댓값 SR0 (귀무: 진짜 SR = 0).

    SR0 = sqrt(V) * ((1-g) * Phi^-1(1-1/N) + g * Phi^-1(1-1/(N e))), g = 오일러-마스케로니 상수.
    N = 1 이면 Phi^-1(0) = -inf 라 식이 깨지므로 0 으로 정의한다 (보정할 시행이 없다).
    """
    if n_trials < 1:
        raise ValueError("n_trials >= 1 이어야 한다")
    if n_trials == 1:
        return 0.0
    if not np.isfinite(sr_var) or sr_var < 0:
        raise ValueError(f"sr_var 는 0 이상 유한값이어야 한다: {sr_var}")
    n = float(n_trials)
    z = (1.0 - EULER_GAMMA) * ndtri(1.0 - 1.0 / n) + EULER_GAMMA * ndtri(1.0 - 1.0 / (n * math.e))
    return float(math.sqrt(sr_var) * z)


def probabilistic_sharpe(sr_hat: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float) -> float:
    """PSR = Phi((SR^ - SR*) sqrt(T-1) / sqrt(1 - skew SR^ + (kurt-1)/4 SR^2)).

    ``kurtosis`` 는 4차 적률 비(정규 3). 분모의 제곱근 안이 0 이하면 ``ValueError``.
    """
    if n_obs < 2:
        raise ValueError("n_obs >= 2 이어야 한다")
    inner = 1.0 - skew * sr_hat + (kurtosis - 1.0) / 4.0 * sr_hat**2
    if not inner > 0:
        raise ValueError(f"PSR 분모의 근호 안이 0 이하다 ({inner}). skew·kurtosis·SR 조합을 확인해야 한다")
    z = (sr_hat - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(inner)
    return float(stats.norm.cdf(z))


@dataclass(frozen=True)
class DeflatedSharpe:
    sr_hat: float  # 선택 전략의 per-period 샤프
    sr0: float  # 기대 최대 샤프 (per-period)
    dsr: float  # PSR(SR0)
    p_value: float  # 1 - DSR
    n_obs: int
    n_trials: int
    skew: float
    kurtosis: float  # 4차 적률 비 (정규 3)
    sr_var: float  # 시행 샤프의 분산 (ddof 1)
    horizon: int
    sr_hat_annual: float
    sr0_annual: float

    def to_dict(self) -> dict:
        return asdict(self)


def deflated_sharpe(selected_returns, trial_sharpes, n_trials: int | None = None, horizon: int = 20) -> DeflatedSharpe:
    """선택 전략의 DSR.

    ``trial_sharpes``: 시행 전체(선택 전략 포함)의 per-period 샤프. 분산은 ddof 1.
    ``n_trials``: 기본값은 ``len(trial_sharpes)``.
    """
    r = _as_clean_1d(selected_returns, "selected_returns")
    ts = _as_clean_1d(trial_sharpes, "trial_sharpes")
    n = len(ts) if n_trials is None else int(n_trials)
    if r.size < 3:
        raise ValueError("selected_returns 는 3개 이상이어야 한다")
    sr_hat = sharpe(r)
    if not np.isfinite(sr_hat):
        raise ValueError("선택 전략의 샤프가 정의되지 않는다 (표준편차 0)")
    if ts.size >= 2:
        sr_var = float(ts.var(ddof=1))
    elif n == 1:
        sr_var = 0.0
    else:
        raise ValueError("trial_sharpes 가 1개인데 n_trials > 1 이다. 분산을 못 구한다")
    sr0 = expected_max_sharpe(sr_var, n)
    skew = float(stats.skew(r, bias=True))
    kurt = float(stats.kurtosis(r, fisher=False, bias=True))
    dsr = probabilistic_sharpe(sr_hat, sr0, r.size, skew, kurt)
    return DeflatedSharpe(
        sr_hat=sr_hat,
        sr0=sr0,
        dsr=dsr,
        p_value=1.0 - dsr,
        n_obs=int(r.size),
        n_trials=n,
        skew=skew,
        kurtosis=kurt,
        sr_var=sr_var,
        horizon=horizon,
        sr_hat_annual=annualize_sharpe(sr_hat, horizon),
        sr0_annual=annualize_sharpe(sr0, horizon),
    )


# --------------------------------------------------------------------------- PBO


def _block_sharpes(x: np.ndarray) -> np.ndarray:
    """열별 per-period 샤프. 표준편차 0 인 열은 0.0 (증거 없음)으로 둔다. CSCV 안에서만 쓴다."""
    sd = x.std(axis=0, ddof=1)
    mean = x.mean(axis=0)
    out = np.zeros(x.shape[1])
    ok = np.ptp(x, axis=0) > 0
    out[ok] = mean[ok] / sd[ok]
    return out


def _lambda_from_omega(omega: np.ndarray) -> np.ndarray:
    """lambda = ln(omega/(1-omega)). omega = 0 이면 -inf, 1 이면 +inf.

    ``r/(N+1)`` 정의에서는 0·1 이 나오지 않지만 (r 은 1..N), 방어용으로 둔다.
    PBO 는 ``lambda <= 0`` 비율이라 -inf 는 과적합, +inf 는 아님으로 센다.
    """
    with np.errstate(divide="ignore"):
        return np.where(omega <= 0, -np.inf, np.where(omega >= 1, np.inf, np.log(omega / (1.0 - omega))))


@dataclass
class CscvResult:
    pbo: float
    n_splits: int
    n_combinations: int
    n_strategies: int
    block_sizes: list[int]
    lambdas: np.ndarray  # (C,)
    table: pd.DataFrame  # 조합별: combo, is_blocks, best_col, best_name, oos_rank, oos_rank_from_top, omega, lam, is_sharpe, oos_sharpe
    is_sharpes: np.ndarray  # (C, N) 조합별 IS 샤프
    oos_sharpes: np.ndarray  # (C, N)
    columns: list


def pbo_cscv(matrix, n_splits: int = 8) -> CscvResult:
    """CSCV 로 PBO 를 구한다.

    ``matrix``: 행 = 시간 순서, 열 = 전략 (DataFrame 이면 열 이름을 쓴다).
    행은 ``np.array_split`` 로 ``n_splits`` 블록 (행 수가 안 나눠떨어지면 앞 블록이 1행 더 크다.
    65행, S=8 이면 9행 1개 + 8행 7개). 행을 버리지 않는다.
    조합 C(S, S/2) 마다 IS 블록 합집합에서 샤프 최고 열을 고르고 (동점은 앞쪽 열),
    OOS 에서 그 열의 순위 r (1 = 최악, N = 최선, 동점은 평균 순위) 을 구해
    omega = r/(N+1), lambda = ln(omega/(1-omega)). PBO = lambda <= 0 인 조합 비율.
    열이 전부 같아 동점이면 omega = 0.5, lambda = 0 이라 PBO = 1 이다.

    열이 2개 미만, NaN·inf, 블록당 2행 미만, 홀수 S 는 ``ValueError``.
    """
    if isinstance(matrix, pd.DataFrame):
        columns = list(matrix.columns)
        x = matrix.to_numpy(dtype=float)
    else:
        x = np.asarray(matrix, dtype=float)
        columns = list(range(x.shape[1])) if x.ndim == 2 else []
    if x.ndim != 2:
        raise ValueError("matrix 는 2차원이어야 한다")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"matrix: NaN 또는 inf 가 {int((~np.isfinite(x)).sum())}개 있다. 먼저 처리해야 한다")
    t, n = x.shape
    if n < 2:
        raise ValueError(f"전략(열)이 2개 이상이어야 한다 (지금 {n})")
    if n_splits < 2 or n_splits % 2:
        raise ValueError("n_splits 는 2 이상의 짝수여야 한다")
    if t < 2 * n_splits:
        raise ValueError(f"행 {t} 개로 {n_splits} 블록을 만들면 블록당 2행 미만이다")

    blocks = np.array_split(np.arange(t), n_splits)
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    c = len(combos)
    is_sh = np.empty((c, n))
    oos_sh = np.empty((c, n))
    rows = []
    omegas = np.empty(c)
    for i, comb in enumerate(combos):
        rest = [b for b in range(n_splits) if b not in comb]
        is_idx = np.concatenate([blocks[b] for b in comb])
        oos_idx = np.concatenate([blocks[b] for b in rest])
        is_sh[i] = _block_sharpes(x[is_idx])
        oos_sh[i] = _block_sharpes(x[oos_idx])
        best = int(np.argmax(is_sh[i]))
        rank = float(stats.rankdata(oos_sh[i], method="average")[best])
        omegas[i] = rank / (n + 1.0)
        rows.append((i, "".join(str(b) for b in comb), best, columns[best], rank, n - rank + 1.0))
    lam = _lambda_from_omega(omegas)
    table = pd.DataFrame(rows, columns=["combo", "is_blocks", "best_col", "best_name", "oos_rank", "oos_rank_from_top"])
    table["omega"] = omegas
    table["lam"] = lam
    table["is_sharpe"] = is_sh[np.arange(c), table["best_col"].to_numpy()]
    table["oos_sharpe"] = oos_sh[np.arange(c), table["best_col"].to_numpy()]
    return CscvResult(
        pbo=float(np.mean(lam <= 0)),
        n_splits=n_splits,
        n_combinations=c,
        n_strategies=n,
        block_sizes=[len(b) for b in blocks],
        lambdas=lam,
        table=table,
        is_sharpes=is_sh,
        oos_sharpes=oos_sh,
        columns=columns,
    )


def strategy_is_best_summary(result: CscvResult, column) -> dict:
    """특정 열이 IS 최선이었던 조합 수와 그때의 OOS 순위.

    ``column``: 열 이름 또는 위치. IS 1위 조합이 없으면 ``n_is_best = 0`` 이고 순위는 None 이다.
    이때를 위해 모든 조합에서의 IS·OOS 순위 분포(1 = 최악, N = 최선, 평균 순위)도 같이 돌려준다.
    """
    pos = result.columns.index(column) if column in result.columns else int(column)
    t = result.table
    sel = t[t["best_col"] == pos]
    n = result.n_strategies
    is_rank = np.array([stats.rankdata(r, method="average")[pos] for r in result.is_sharpes])
    oos_rank = np.array([stats.rankdata(r, method="average")[pos] for r in result.oos_sharpes])
    out = {
        "column": result.columns[pos],
        "n_is_best": int(len(sel)),
        "n_combinations": result.n_combinations,
        "oos_rank_median": float(sel["oos_rank"].median()) if len(sel) else None,
        "oos_rank_min": float(sel["oos_rank"].min()) if len(sel) else None,
        "oos_rank_max": float(sel["oos_rank"].max()) if len(sel) else None,
        "oos_rank_from_top_median": float(sel["oos_rank_from_top"].median()) if len(sel) else None,
        "oos_rank_from_top_best": float(sel["oos_rank_from_top"].min()) if len(sel) else None,
        "oos_rank_from_top_worst": float(sel["oos_rank_from_top"].max()) if len(sel) else None,
        "n_strategies": n,
        "is_rank_from_top_median_all": float(np.median(n - is_rank + 1.0)),
        "oos_rank_from_top_median_all": float(np.median(n - oos_rank + 1.0)),
        "is_rank_from_top_range_all": [float((n - is_rank + 1.0).min()), float((n - is_rank + 1.0).max())],
        "oos_rank_from_top_range_all": [float((n - oos_rank + 1.0).min()), float((n - oos_rank + 1.0).max())],
    }
    return out
