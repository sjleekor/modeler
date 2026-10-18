"""줄 정의: B1~B5, 참고 줄 R1·R1i·R2·R2i의 목표 비중과 B4 신호 (09b §0·§3·I7~I10)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

DOM, US_MAIN, US_U2 = "069500", "143850", "360750"
BOND3, BOND10, GOLD = "114260", "148070", "132030"

VM_LO, VM_HI = 0.5, 1.0
SIGMA_WINDOW = 20
SIGMA_TARGET_MIN_OBS = 252


# --------------------------------------------------------------------------- B4 신호
@dataclass
class RiskShareSignal:
    rs: np.ndarray  # 위험 몫 일간 수익 (069500·미국 50:50 매일 맞춤)
    sigma20: np.ndarray
    n_obs: np.ndarray  # 그날까지 σ_20 관측 수
    sigma_target: np.ndarray  # n_obs >= 252일 때만
    w_vm: np.ndarray  # 준비 안 됐으면 NaN


def risk_share_signal(r_dom: np.ndarray, r_us: np.ndarray) -> RiskShareSignal:
    """σ_20(t): 위험 몫 일간 수익 직전 20거래일(t 포함) 표준편차(표본, ddof=1).

    σ_target(t): 첫 σ_20부터 t까지 σ_20의 중앙값(확장창), 관측 252개 이상일 때만.
    ``w_vm(t) = clip(σ_target/σ_20, 0.5, 1.0)``. t까지의 값만 쓴다(미래를 보지 않는다).
    """
    T = len(r_dom)
    rs = 0.5 * np.asarray(r_dom, float) + 0.5 * np.asarray(r_us, float)
    sig = np.full(T, np.nan)
    for i in range(SIGMA_WINDOW - 1, T):
        w = rs[i - SIGMA_WINDOW + 1 : i + 1]
        if not np.isnan(w).any():
            sig[i] = w.std(ddof=1)
    n_obs = np.cumsum(~np.isnan(sig)).astype(int)
    tgt = np.full(T, np.nan)
    for i in range(T):
        if n_obs[i] >= SIGMA_TARGET_MIN_OBS:
            tgt[i] = np.nanmedian(sig[: i + 1])
    with np.errstate(invalid="ignore", divide="ignore"):
        w_vm = np.clip(tgt / sig, VM_LO, VM_HI)
    return RiskShareSignal(rs, sig, n_obs, tgt, w_vm)


# --------------------------------------------------------------------------- 결정일
def month_end_indices(days: list[date]) -> list[int]:
    """월 마지막 거래일의 인덱스(달력 마지막 날은 마지막 달이 끝나지 않았으면 제외하지 않는다)."""
    out = []
    for i in range(len(days) - 1):
        if (days[i].year, days[i].month) != (days[i + 1].year, days[i + 1].month):
            out.append(i)
    return out


def decision_schedule(days: list[date], first_decision: date, end_idx: int) -> list[int]:
    """첫 결정일(월 마지막 거래일이어야 함)부터 월말 결정일 목록. 체결(t+1)이 end_idx 안인 것만."""
    me = month_end_indices(days)
    pos = {days[i]: i for i in me}
    if first_decision not in pos:
        raise ValueError(f"{first_decision}은 월 마지막 거래일이 아닙니다")
    return [i for i in me if i >= pos[first_decision] and i + 1 <= end_idx]


# --------------------------------------------------------------------------- 줄
@dataclass(frozen=True)
class LineSpec:
    name: str
    assets: tuple[str, ...]  # ETF 코드 (현금은 따로, 마지막 열)
    kind: str  # static | vm | wfix | buyhold
    base: tuple[float, ...] = ()  # static/buyhold: assets별 비중(현금은 나머지)

    def weights(self, w_vm: float | None = None, w_fix: float | None = None) -> np.ndarray:
        k = len(self.assets)
        w = np.zeros(k + 1)
        if self.kind in ("static", "buyhold"):
            w[:k] = self.base
            w[k] = 1.0 - sum(self.base)
        elif self.kind == "vm":
            if w_vm is None or np.isnan(w_vm):
                raise ValueError("w_vm이 없습니다")
            # assets = (DOM, US, BOND10): 0.3·w, 0.3·w, 0.4, 현금 0.6·(1−w)
            w[:3] = [0.3 * w_vm, 0.3 * w_vm, 0.4]
            w[3] = 0.6 * (1.0 - w_vm)
        elif self.kind == "wfix":
            if w_fix is None:
                raise ValueError("w_fix가 없습니다")
            w[:3] = [w_fix / 2, w_fix / 2, 0.4]
            w[3] = 0.6 - w_fix
        else:
            raise ValueError(self.kind)
        if abs(w.sum() - 1.0) > 1e-9 or (w < -1e-12).any():
            raise ValueError(f"{self.name}: 비중이 잘못됐습니다 {w}")
        return w


def line_specs(us: str) -> dict[str, LineSpec]:
    """``us``: 미국 주식 ETF 코드(주 구간 143850, U2 360750)."""
    return {
        "B1": LineSpec("B1", (DOM, us, BOND3, BOND10, GOLD), "static", (0.2,) * 5),
        "B2": LineSpec("B2", (DOM, us, BOND10), "static", (0.3, 0.3, 0.4)),
        "B3": LineSpec("B3", (DOM, us, BOND3), "static", (0.15, 0.15, 0.7)),
        "B4": LineSpec("B4", (DOM, us, BOND10), "vm"),
        "B5": LineSpec("B5", (DOM, us, BOND10), "wfix"),
        "R1": LineSpec("R1", (DOM,), "buyhold", (1.0,)),
        "R1i": LineSpec("R1i", (DOM,), "static", (0.7,)),
        "R2": LineSpec("R2", (us,), "buyhold", (1.0,)),
        "R2i": LineSpec("R2i", (us,), "static", (0.7,)),
    }


def build_targets(
    spec: LineSpec,
    decisions: list[int],
    w_vm: np.ndarray | None = None,
    w_fix: float | None = None,
) -> dict[int, np.ndarray]:
    """체결일(결정일+1) -> 목표 비중. 첫 결정이 시작(비용 없음). R1·R2는 재조정 없음."""
    out: dict[int, np.ndarray] = {}
    for n, d in enumerate(decisions):
        if spec.kind == "buyhold" and n > 0:
            break
        out[d + 1] = spec.weights(None if w_vm is None else w_vm[d], w_fix)
    return out
