"""단일피쳐 검정 — ``04_feature_test_plan.md`` §4 그대로.

    uv run python -m modeler.us.scan

동결된 피쳐 44개(F1~F14, 태그 ``us-features-frozen``)를 h21 라벨(``L2``) 순위와
맞춰 매월 첫 거래일 횡단면 rank IC를 내고, Newey-West HAC t(lag 3) ·
top-bottom decile spread(``L0``) · h5·h63 감쇠 · placebo(circular shift 50개)를
계산해 등급(A~D·R)을 매긴다(``06_execution_steps.md`` M3).

**개발 구간만 읽는다.** ``DEV_END``(2025-06-30, ``05_validation_protocol.md``
§1) 뒤 날짜는 이 모듈이 읽는 모든 데이터셋에서 :func:`load_dev_frame`이 바로
잘라낸다 — holdout을 열 경로가 이 파일 안에 없다.

**피쳐를 더하거나 정의를 바꾸지 않는다** (R6 동결). 여기서 하는 일은 이미
``datasets/us_features_v1``에 있는 44개를 등급 매기는 것뿐이다.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from modeler.etl.config import DataRoot
from modeler.etl.metrics import benjamini_hochberg, newey_west_tstat, two_sided_normal_p

# --- 0. 사전등록 상수 (04 §1·§4·§5, 06 M3) ---------------------------------

#: 개발 구간 끝 — 이 뒤는 holdout이다(``05`` §1). **코드에 박는 날짜 벽.**
DEV_END: date = date(2025, 6, 30)

#: 개발 구간 시작 — 패널 자체가 여기서부터라 필터링 효과는 없지만, 어떤
#: 구간을 "개발 구간"으로 보는지 이 파일 안에서 명시적으로 남겨 둔다.
DEV_START: date = date(2018, 9, 7)

#: HAC(Newey-West) lag — ``04`` §1 사전등록값.
HAC_LAG = 3

#: BH 안에서 FDR 문턱 — ``04`` §1.
BH_Q_THRESHOLD = 0.10

#: A·B 등급 안 중복 제거 문턱(순위상관 절댓값) — ``04`` §1.
DEDUP_RHO_THRESHOLD = 0.8

#: placebo 개수 — ``04`` §1 "50개".
N_PLACEBO = 50

#: placebo shift 후보 범위(개월) — ``04`` §1 "6~76개월".
PLACEBO_SHIFT_MIN = 6
PLACEBO_SHIFT_MAX = 76

#: placebo 기각(R) 문턱 — ``04`` §1 "실제 |t|의 순열 p >= 0.05면 R".
PLACEBO_P_REJECT = 0.05

#: placebo shift 후보(6~76개월, 12의 배수 제외) 65개 중 50개를 고를 때 쓰는
#: 고정 시드. 실행 시각·순서와 무관하게 항상 같은 50개가 뽑히게 날짜로
#: 고정한다(이 계획을 실행한 날, 2026-09-21) — 재현성 그 자체가 목적이라 값이
#: 무엇이든 상관없고, 한 번 정하면 다시 바꾸지 않는다.
PLACEBO_SAMPLE_SEED = 20260921

#: 월별 횡단면 IC·spread를 낼 때 그 달을 쓰려면 최소 이만큼의 비결측 이름이
#: 있어야 한다. ``04``에 숫자가 없어 여기서 정한다 — 코드베이스가 이미 쓰는
#: "소수 셀" 문턱(``modeler.us.labels.MIN_SIC2_GROUP_SIZE``)과 같은 20을
#: 쓴다. 20 미만이면 top/bottom 10% decile이 사실상 종목 한둘이 된다.
MIN_NAMES = 20

#: top/bottom decile 컷 — 각 10%.
DECILE_FRACTION = 0.10

UNIVERSES: tuple[str, ...] = ("all", "price_ge_5")


@dataclass(frozen=True)
class FeatureSpec:
    """피쳐 하나의 등록 정보 — ``04`` §3 표 그대로. 부호가 없는 것(``None``)은
    "부호 없음 — 상호작용/분해축"이라 명시적으로 적힌 셋뿐이다."""

    feature: str
    family: str
    expected_sign: Literal["+", "-"] | None


#: 검정 대상 44개 — ``04`` §3 F1~F14. 순서·family 이름은
#: ``modeler.us.features.FAMILY_ORDER``와 맞춘다. F15(시장 수준)·F16(캘린더)
#: 다섯은 여기 없다 — 단독 검정을 하지 않는다(``04`` §3, §4 "F15·F16 다섯은
#: 검정하지 않는다").
FEATURE_REGISTRY: tuple[FeatureSpec, ...] = (
    # F1 모멘텀
    FeatureSpec("mom_12_1", "F1_momentum", "+"),
    FeatureSpec("mom_6_1", "F1_momentum", "+"),
    FeatureSpec("mom_1m", "F1_momentum", "-"),
    # F2 단기 반전
    FeatureSpec("rev_1w", "F2_reversal", "-"),
    FeatureSpec("max_ret_1m", "F2_reversal", "-"),
    # F3 변동성
    FeatureSpec("rv_20", "F3_volatility", "-"),
    FeatureSpec("rv_60", "F3_volatility", "-"),
    FeatureSpec("idio_vol_60", "F3_volatility", "-"),
    FeatureSpec("beta_252", "F3_volatility", "-"),
    # F4 유동성·규모
    FeatureSpec("log_dvol_20", "F4_liquidity", "-"),
    FeatureSpec("amihud_20", "F4_liquidity", "+"),
    FeatureSpec("turnover_rank", "F4_liquidity", "-"),
    FeatureSpec("mcap_rank", "F4_liquidity", "-"),
    # F5 밸류
    FeatureSpec("bm", "F5_valuation", "+"),
    FeatureSpec("ep_ttm", "F5_valuation", "+"),
    FeatureSpec("cfp_ttm", "F5_valuation", "+"),
    FeatureSpec("sp_ttm", "F5_valuation", "+"),
    # F6 수익성
    FeatureSpec("roa_ttm", "F6_profitability", "+"),
    FeatureSpec("roe_ttm", "F6_profitability", "+"),
    FeatureSpec("gpa", "F6_profitability", "+"),
    FeatureSpec("opm_ttm", "F6_profitability", "+"),
    # F7 투자·발생액
    FeatureSpec("asset_growth", "F7_investment", "-"),
    FeatureSpec("accruals", "F7_investment", "-"),
    FeatureSpec("net_issuance", "F7_investment", "-"),
    # F8 배당·자사주
    FeatureSpec("div_yield", "F8_payout", "+"),
    FeatureSpec("buyback_yield", "F8_payout", "+"),
    # F9 실적 이벤트
    FeatureSpec("sue_last", "F9_earnings", "+"),
    FeatureSpec("days_since_earn", "F9_earnings", None),
    FeatureSpec("days_to_earn", "F9_earnings", None),
    FeatureSpec("n_estimates", "F9_earnings", "-"),
    # F10 내부자
    FeatureSpec("ins_netbuy_90", "F10_insider", "+"),
    FeatureSpec("ins_cluster_90", "F10_insider", "+"),
    FeatureSpec("ins_officer_buy_90", "F10_insider", "+"),
    # F11 공매도
    FeatureSpec("si_ratio", "F11_short", "-"),
    FeatureSpec("dtc", "F11_short", "-"),
    FeatureSpec("si_chg", "F11_short", "-"),
    FeatureSpec("sv_share_20", "F11_short", "-"),
    # F12 공시 활동
    FeatureSpec("n_8k_90", "F12_filing_activity", "-"),
    FeatureSpec("filing_lag", "F12_filing_activity", "-"),
    # F13 옵션 IV
    FeatureSpec("iv_rank", "F13_options_iv", "-"),
    FeatureSpec("iv_hv_spread", "F13_options_iv", "-"),
    FeatureSpec("iv_isna", "F13_options_iv", None),
    # F14 지수 편입
    FeatureSpec("sp500_member", "F14_index_membership", None),
    FeatureSpec("sp500_days_since_add", "F14_index_membership", "-"),
)

assert len(FEATURE_REGISTRY) == 44, f"검정 대상은 44개여야 한다 (실제 {len(FEATURE_REGISTRY)})"

FEATURE_COLUMNS: tuple[str, ...] = tuple(spec.feature for spec in FEATURE_REGISTRY)
#: 결측률 > 50% — ``06`` M2 결과에서 이미 알려진 여덟. 표에 표시만 하고
#: (``04`` §5) 실제 결측률은 매번 다시 잰다 — 이 목록은 검증용 참고치다.
KNOWN_HIGH_MISSING: dict[str, float] = {
    "sp500_days_since_add": 0.975,
    "gpa": 0.704,
    "buyback_yield": 0.692,
    "iv_rank": 0.680,
    "iv_hv_spread": 0.671,
    "ins_netbuy_90": 0.580,
    "opm_ttm": 0.579,
    "sp_ttm": 0.555,
}


# --- 1. holdout 벽 -----------------------------------------------------------


def enforce_dev_window(df: pl.DataFrame, *, dev_end: date = DEV_END) -> pl.DataFrame:
    """``date`` 컬럼이 ``dev_end``를 넘는 행을 버린다 — 이 함수가 holdout 벽이다.

    이 모듈이 읽는 프레임은 전부 (직접, 또는 :func:`load_dev_frame`을 통해)
    이 함수를 거친다. 이 함수를 우회해 만든 프레임을 나중 계산에 섞으면
    holdout이 샌다.
    """
    if "date" not in df.columns:
        raise ValueError("enforce_dev_window: 'date' 컬럼이 없습니다")
    return df.filter(pl.col("date") <= dev_end)


def assert_dev_window(df: pl.DataFrame, *, dev_end: date = DEV_END) -> None:
    """``df``에 ``dev_end`` 뒤 날짜가 있으면 예외. holdout이 새지 않았는지
    마지막에 다시 확인하는 방어선이다(``main``이 매 프레임마다 부른다)."""
    if df.height and df["date"].max() > dev_end:
        raise ValueError(
            f"holdout 날짜가 섞였습니다: max(date)={df['date'].max()} > dev_end={dev_end}"
        )


def load_dev_frame(root: DataRoot, name: str, *, dev_end: date = DEV_END) -> pl.DataFrame:
    """``root.datasets/<name>/part.parquet``를 읽고 개발 구간만 남긴다."""
    path = root.datasets / name / "part.parquet"
    df = pl.read_parquet(path)
    dev = enforce_dev_window(df, dev_end=dev_end)
    assert_dev_window(dev, dev_end=dev_end)
    return dev


# --- 2. 월별 순위 IC · HAC t --------------------------------------------------


def monthly_rank_ic(
    df: pl.DataFrame,
    *,
    x_col: str,
    y_col: str,
    group_col: str = "month_idx",
    min_names: int = MIN_NAMES,
) -> pl.DataFrame:
    """``group_col``(매월 첫 거래일)별 스피어만 순위상관.

    같은 달 안에서 ``x_col``·``y_col``을 각각 순위로 바꾼 뒤 피어슨 상관을
    내는 것이 스피어만 상관의 정의다(동순위는 평균 순위,
    ``pl.Expr.rank()`` 기본값). ``min_names``(그 달의 비결측 이름 수) 미만인
    달은 뺀다 — 미만이면 top/bottom 10% decile이 종목 한둘이 되어 spread가
    의미를 잃는다(``MIN_NAMES`` 참고).

    반환: ``group_col, ic, n`` (``group_col``으로 정렬됨). 비어 있으면
    빈 프레임을 스키마만 맞춰 돌려준다.
    """
    ranked = df.with_columns(
        pl.col(x_col).rank().over(group_col).alias("_xr"),
        pl.col(y_col).rank().over(group_col).alias("_yr"),
        pl.len().over(group_col).alias("_n"),
    ).filter(pl.col("_n") >= min_names)
    if ranked.height == 0:
        return pl.DataFrame(
            schema={group_col: df.schema[group_col], "ic": pl.Float64, "n": pl.Int64}
        )
    return (
        ranked.group_by(group_col, maintain_order=True)
        .agg(pl.corr("_xr", "_yr").alias("ic"), pl.first("_n").cast(pl.Int64).alias("n"))
        .sort(group_col)
    )


def ic_and_t(
    ic_table: pl.DataFrame, *, group_col: str = "month_idx", lag: int = HAC_LAG
) -> tuple[float, float, int]:
    """``ic_table``(:func:`monthly_rank_ic` 출력)에서 평균 IC와 HAC t.

    Newey-West HAC 분산(Bartlett 커널, gap-aware, ``04`` §1 사전등록 lag=3)::

        IC_bar   = mean_t(IC_t)
        gamma_0  = mean_t((IC_t - IC_bar)^2)
        gamma_k  = mean_t((IC_t - IC_bar)(IC_{t-k} - IC_bar))   (t, t-k가 둘 다
                   있는 쌍만 — 달이 비어도 잘못된 간격을 안 만든다)
        longrun  = gamma_0 + 2 * sum_{k=1}^{lag} (1 - k/(lag+1)) * gamma_k
        se(IC)   = sqrt(longrun / n)
        t_nw     = IC_bar / se(IC)

    ``modeler.etl.metrics.newey_west_tstat``이 이 식을 구현한다 — 한국
    horizon scan이 이미 쓰고 회귀 검증한 것과 같은 함수를 그대로 불러
    다시 구현하지 않는다(``06`` M0 판단과 같은 이유: 검증된 것을 재사용).
    ``ic_table``의 ``ic``가 NaN(그 달 횡단면이 상수라 상관이 정의 안 됨)이면
    ``newey_west_tstat``가 유한값만 걸러 쓴다.
    """
    if ic_table.height == 0:
        return float("nan"), float("nan"), 0
    values = ic_table["ic"].to_numpy()
    idx = ic_table[group_col].to_numpy()
    finite = np.isfinite(values)
    ic_mean = float(values[finite].mean()) if finite.any() else float("nan")
    t_nw = newey_west_tstat(values, idx, lag)
    return ic_mean, t_nw, int(finite.sum())


def decile_spread(
    df: pl.DataFrame,
    *,
    feature_col: str,
    value_col: str = "L0",
    group_col: str = "month_idx",
    min_names: int = MIN_NAMES,
    fraction: float = DECILE_FRACTION,
) -> tuple[float, int]:
    """``group_col``별 top/bottom ``fraction`` 분위 ``value_col`` 평균 차,
    그 뒤 시계열 평균 — ``04`` §4 ``spread = mean(top decile L0) - mean(bottom decile L0)``.

    decile은 ``feature_col``의 그 달 순위(백분위, [0,1])로 가른다. **``L0``는
    원수익률이다** — 중립화 전 값으로 부호를 확인한다(``04`` §4 주석).

    순위는 ``method="ordinal"``로 매긴다(동순위를 원래 행 순서로 갈라
    1..n을 유일하게 배정) — ``div_yield``·``ins_cluster_90``처럼 값 대부분이
    0으로 묶인 피쳐는 ``method="average"``를 쓰면 그 묶음 전체가 같은
    평균순위(백분위 구간 한가운데)를 받아 top/bottom 10% 어느 쪽에도
    걸리지 않고, 그러면 그 달의 top 또는 bottom 그룹이 통째로 비어 spread가
    NaN이 된다(실측 — 2026-09-21 스캔에서 처음 드러났다). ordinal은 그 묶음
    안 개별 종목을 임의로(그러나 결정적으로) 흩어 배정해 이 문제를 없앤다 —
    spread는 부호 확인용이라(``04`` §4) 동순위 안에서 어느 특정 종목이
    "10분위"로 뽑히는지는 중요하지 않다.
    """
    base = df.with_columns(
        pl.col(feature_col).rank(method="ordinal").over(group_col).alias("_r"),
        pl.len().over(group_col).alias("_n"),
    ).filter(pl.col("_n") >= min_names)
    if base.height == 0:
        return float("nan"), 0
    base = base.with_columns(((pl.col("_r") - 1) / (pl.col("_n") - 1)).alias("_pct"))
    top = (
        base.filter(pl.col("_pct") >= 1 - fraction)
        .group_by(group_col)
        .agg(pl.col(value_col).mean().alias("top"))
    )
    bottom = (
        base.filter(pl.col("_pct") <= fraction)
        .group_by(group_col)
        .agg(pl.col(value_col).mean().alias("bottom"))
    )
    merged = top.join(bottom, on=group_col, how="inner").with_columns(
        (pl.col("top") - pl.col("bottom")).alias("spread")
    )
    if merged.height == 0:
        return float("nan"), 0
    spread_values = merged["spread"].drop_nulls()
    if spread_values.len() == 0:
        return float("nan"), 0
    return float(spread_values.mean()), merged.height


def spread_sign_matches(spread: float, expected_sign: Literal["+", "-"] | None) -> bool:
    """``04`` §1 등급 A의 "spread 부호 일치" 조건.

    예상 부호가 등록되지 않은 피쳐(``days_since_earn``·``days_to_earn``·
    ``sp500_member``·``iv_isna`` — ``04`` §3이 "부호 없음"이라 명시한 넷)는
    방향이 맞는지 판단할 근거가 아예 없으므로, 이 조건을 통과한 것으로 둔다
    — A 등급 판정이 |t|만으로 갈리게 하는 명시적 설계 판단이다(보고서에
    남긴다).
    """
    if expected_sign is None:
        return True
    if spread is None or not math.isfinite(spread) or spread == 0:
        return False
    observed = "+" if spread > 0 else "-"
    return observed == expected_sign


# --- 3. placebo (circular shift) --------------------------------------------


def placebo_shift_candidates(
    *, min_shift: int = PLACEBO_SHIFT_MIN, max_shift: int = PLACEBO_SHIFT_MAX
) -> list[int]:
    """``[min_shift, max_shift]`` 중 12의 배수를 뺀 후보 — ``04`` §1 "12의
    배수 아닌 것 위주"(12개월 배수만 쓰면 계절성이 placebo에 그대로 남는다,
    ``08`` V9)."""
    return [s for s in range(min_shift, max_shift + 1) if s % 12 != 0]


def select_placebo_shifts(*, n: int = N_PLACEBO, seed: int = PLACEBO_SAMPLE_SEED) -> list[int]:
    """후보 중 ``n``개를 고정 시드로 뽑는다 — 실행할 때마다 같은 50개가 나온다."""
    candidates = placebo_shift_candidates()
    if n > len(candidates):
        raise ValueError(f"placebo shift 후보({len(candidates)})가 요청한 {n}개보다 적습니다")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(np.array(candidates), size=n, replace=False)
    return sorted(int(x) for x in chosen)


def _shift_month_index_expr(*, shift: int, total_months: int) -> pl.Expr:
    """라벨의 ``month_idx``를 순환 이동한 값으로 relabel하는 식.

    원래 ``month_idx=j``였던 라벨 행은 이 식을 거치면
    ``month_idx = ((j - 1 - shift) % total_months) + 1``가 된다. 이후 이 값을
    피쳐의 ``month_idx``와 조인하면, 피쳐 달 ``i``에 라벨 달
    ``j = ((i - 1 + shift) % total_months) + 1``이 물린다 — ``04`` §4
    placebo("y를 shift")의 정의 그대로, 같은 shift를 모든 종목에 공동 적용한다.
    """
    if total_months <= 0:
        raise ValueError("total_months는 양수여야 합니다")
    return ((pl.col("month_idx").cast(pl.Int64) - 1 - shift) % total_months) + 1


def placebo_p_value(real_abs_t: float, placebo_abs_t: list[float]) -> float:
    """``04`` §1: 실제 |t|의 순열 p = (1 + #{placebo >= real}) / (repeats + 1).

    (한국 ``horizon_scan_permutation.temporal_placebo_p``와 같은 관례 —
    0이 되지 않는 보수적 추정.)
    """
    finite_placebo = [t for t in placebo_abs_t if math.isfinite(t)]
    if not math.isfinite(real_abs_t) or not finite_placebo:
        return float("nan")
    at_least = sum(1 for t in finite_placebo if t >= real_abs_t)
    return (1 + at_least) / (len(finite_placebo) + 1)


def compute_placebo_abs_t(
    feature_side: pl.DataFrame,
    labels_only: pl.DataFrame,
    *,
    total_months: int,
    shifts: list[int],
    min_names: int = MIN_NAMES,
    lag: int = HAC_LAG,
) -> list[float]:
    """``shifts``마다 |t_placebo|를 낸다.

    ``feature_side``는 ``month_idx, symbol, value``(이미 유니버스·비결측
    필터가 걸린 실제 검정 대상), ``labels_only``는 ``month_idx, symbol, L2``
    (유니버스 필터 없이 전체 — 라벨 쪽은 shift로 다른 달 값을 끌어오므로
    피쳐 달의 유니버스 자격과 무관하게 그 종목의 그 시점 라벨이면 된다).
    """
    abs_ts: list[float] = []
    for shift in shifts:
        shifted_labels = labels_only.with_columns(
            _shift_month_index_expr(shift=shift, total_months=total_months).alias("month_idx")
        )
        joined = feature_side.join(shifted_labels, on=["month_idx", "symbol"], how="inner")
        ic_table = monthly_rank_ic(joined, x_col="value", y_col="L2", min_names=min_names)
        _, t_nw, _ = ic_and_t(ic_table, lag=lag)
        abs_ts.append(abs(t_nw) if math.isfinite(t_nw) else float("nan"))
    return abs_ts


# --- 4. 등급 ------------------------------------------------------------------

Grade = Literal["A", "B", "C", "D", "R"]


def grade_feature(*, t_nw: float, spread_sign_ok: bool, placebo_p: float | None) -> Grade:
    """``04`` §1 등급표.

    평가 순서: |t|가 아예 정의되지 않으면(달 수 부족 등) D, 그다음 placebo
    탈락(R, 실제 |t|의 순열 p >= 0.05)을 |t| 구간보다 먼저 본다 — 한국
    horizon scan의 evidence_grade 평가 순서(R을 A~D보다 먼저 본다,
    ``horizon_scan_config.py``)와 같은 관례다. R이 아니면 |t| 구간으로
    A/B/C/D를 가른다.
    """
    if not math.isfinite(t_nw):
        return "D"
    abs_t = abs(t_nw)
    if placebo_p is not None and math.isfinite(placebo_p) and placebo_p >= PLACEBO_P_REJECT:
        return "R"
    if abs_t >= 3.0 and spread_sign_ok:
        return "A"
    if abs_t >= 2.0:
        return "B"
    if abs_t >= 1.0:
        return "C"
    return "D"


# --- 5. 결과 행 ---------------------------------------------------------------


@dataclass
class FeatureScanRow:
    feature: str
    family: str
    universe: str
    expected_sign: str | None
    n_dates: int
    ic_mean: float
    t_nw: float
    p_two_sided: float
    spread: float
    spread_sign_match: bool
    placebo_max_abs_t: float
    placebo_p: float
    grade: Grade
    h5_ic: float
    h63_ic: float
    missing_rate: float
    high_missing: bool
    bh_q: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "family": self.family,
            "universe": self.universe,
            "expected_sign": self.expected_sign,
            "n_dates": self.n_dates,
            "ic_mean": self.ic_mean,
            "t_nw": self.t_nw,
            "p_two_sided": self.p_two_sided,
            "spread": self.spread,
            "spread_sign_match": self.spread_sign_match,
            "placebo_max_abs_t": self.placebo_max_abs_t,
            "placebo_p": self.placebo_p,
            "grade": self.grade,
            "h5_ic": self.h5_ic,
            "h63_ic": self.h63_ic,
            "missing_rate": self.missing_rate,
            "high_missing": self.high_missing,
            "bh_q": self.bh_q,
        }


def apply_bh_within_family(rows: list[FeatureScanRow]) -> None:
    """(family, universe) 안에서 BH q값을 매겨 ``row.bh_q``를 채운다.

    ``04`` §4 검정 절차의 "BH(FDR 10%) family 안에서" 단계다 — 문턱
    ``BH_Q_THRESHOLD``는 이 q값을 "discovery"로 읽을 때(보고 단계)의 기준이고,
    q값 계산 자체에는 필요 없다. **등급 자체는 이 q값을 참조하지 않는다** —
    ``04`` §1 등급표가 |t|·spread 부호·placebo만으로 A~D·R을 정의하기
    때문이다. 여기서 낸 ``bh_q``는 다중비교 진단으로 표에 같이 싣는다
    (보고서에 이 판단을 명시한다).
    """
    by_key: dict[tuple[str, str], list[FeatureScanRow]] = {}
    for row in rows:
        by_key.setdefault((row.family, row.universe), []).append(row)
    for group_rows in by_key.values():
        pvals = [row.p_two_sided for row in group_rows]
        qvals = benjamini_hochberg(pvals)
        for row, q in zip(group_rows, qvals, strict=True):
            row.bh_q = float(q) if math.isfinite(q) else None


# --- 6. 중복 제거 -------------------------------------------------------------


def pairwise_avg_rank_corr(
    core: pl.DataFrame, feature_a: str, feature_b: str, *, min_names: int = MIN_NAMES
) -> float:
    """두 피쳐의 월별 횡단면 순위상관 평균 — ``04`` §1 중복 제거 판정값.

    ``core``는 ``month_idx``와 두 피쳐 컬럼을 담은 프레임(해당 유니버스로
    이미 필터된 것)이어야 한다. 둘 다 비결측인 행만 쓴다.
    """
    sub = core.select("month_idx", feature_a, feature_b).drop_nulls()
    ic_table = monthly_rank_ic(sub, x_col=feature_a, y_col=feature_b, min_names=min_names)
    if ic_table.height == 0:
        return float("nan")
    values = ic_table["ic"].drop_nulls()
    if values.len() == 0:
        return float("nan")
    return float(values.mean())


def dedup_ab_features(
    rows: list[FeatureScanRow],
    core: pl.DataFrame,
    *,
    universe: str,
    rho_threshold: float = DEDUP_RHO_THRESHOLD,
) -> tuple[list[str], list[dict[str, object]]]:
    """``04`` §1: A·B 등급 안에서 |순위상관| > ``rho_threshold``면 |t|가 큰
    쪽만 남긴다. ``core``는 이미 ``universe``로 필터된 프레임이어야 한다.

    반환: (남는 모델 입력 피쳐 id, ``t`` 내림차순), (뺀 피쳐와 사유 목록).
    """
    ab = [r for r in rows if r.universe == universe and r.grade in {"A", "B"}]
    ab_sorted = sorted(
        ab, key=lambda r: abs(r.t_nw) if math.isfinite(r.t_nw) else -1.0, reverse=True
    )
    kept: list[FeatureScanRow] = []
    dropped: list[dict[str, object]] = []
    for row in ab_sorted:
        collision: tuple[str, float] | None = None
        for kept_row in kept:
            rho = pairwise_avg_rank_corr(core, row.feature, kept_row.feature)
            if math.isfinite(rho) and abs(rho) > rho_threshold:
                collision = (kept_row.feature, rho)
                break
        if collision is None:
            kept.append(row)
        else:
            dropped.append(
                {"feature": row.feature, "collides_with": collision[0], "rho": collision[1]}
            )
    return [r.feature for r in kept], dropped


# --- 7. 입력 조립 -------------------------------------------------------------


@dataclass
class ScanInputs:
    features_dev: pl.DataFrame
    core21: pl.DataFrame  # date, month_idx, symbol, price_ge_5, <feature*44>, L0, L2
    core5: pl.DataFrame  # date, month_idx, symbol, price_ge_5, <feature*44>, L2
    core63: pl.DataFrame  # 〃
    labels_only: pl.DataFrame  # month_idx, symbol, L2 (유니버스 필터 없음, placebo용)
    total_months: int  # h21 개발 구간 리밸런스 수 (04 §1 완료 판정 "약 82")


def _with_month_idx(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(pl.col("date").rank(method="dense").cast(pl.Int64).alias("month_idx"))


def build_scan_inputs(root: DataRoot) -> ScanInputs:
    """레이크가 아니라 이미 만들어진 데이터셋(``us_features_v1``·
    ``us_labels_v1``·``us_labels_h5_v1``·``us_labels_h63_v1``)을 읽는다 —
    ``06`` §1 M1·M2가 이미 끝냈다. 여기서 다시 만들지 않는다.
    """
    features_dev = load_dev_frame(root, "us_features_v1")
    labels21_dev = load_dev_frame(root, "us_labels_v1")
    labels5_dev = load_dev_frame(root, "us_labels_h5_v1")
    labels63_dev = load_dev_frame(root, "us_labels_h63_v1")

    base_cols = ["date", "symbol", "price_ge_5", *FEATURE_COLUMNS]
    core21 = _with_month_idx(
        features_dev.select(base_cols).join(
            labels21_dev.select("date", "symbol", "L0", "L2"), on=["date", "symbol"], how="inner"
        )
    )
    core5 = _with_month_idx(
        features_dev.select(base_cols).join(
            labels5_dev.select("date", "symbol", "L2"), on=["date", "symbol"], how="inner"
        )
    )
    core63 = _with_month_idx(
        features_dev.select(base_cols).join(
            labels63_dev.select("date", "symbol", "L2"), on=["date", "symbol"], how="inner"
        )
    )
    for frame in (core21, core5, core63):
        assert_dev_window(frame)

    labels_only = core21.select("month_idx", "symbol", "L2").unique()
    total_months = int(core21["month_idx"].max()) if core21.height else 0
    return ScanInputs(
        features_dev=features_dev,
        core21=core21,
        core5=core5,
        core63=core63,
        labels_only=labels_only,
        total_months=total_months,
    )


def _missing_rate(features_dev: pl.DataFrame, feature_col: str, *, universe: str) -> float:
    scoped = features_dev if universe == "all" else features_dev.filter(pl.col("price_ge_5"))
    if scoped.height == 0:
        return float("nan")
    return float(scoped[feature_col].is_null().mean())


# --- 8. 피쳐 하나 검정 ---------------------------------------------------------


def scan_one(
    spec: FeatureSpec,
    *,
    universe: str,
    inputs: ScanInputs,
    placebo_shifts: list[int],
) -> FeatureScanRow:
    feature_col = spec.feature

    def _scope(core: pl.DataFrame) -> pl.DataFrame:
        return core if universe == "all" else core.filter(pl.col("price_ge_5"))

    core21 = _scope(inputs.core21)
    feat21 = core21.select("month_idx", "symbol", feature_col, "L0", "L2").filter(
        pl.col(feature_col).is_not_null()
    )

    ic_table = monthly_rank_ic(feat21, x_col=feature_col, y_col="L2")
    ic_mean, t_nw, n_dates = ic_and_t(ic_table)
    p_two_sided = two_sided_normal_p(t_nw)

    spread, _n_spread_dates = decile_spread(feat21, feature_col=feature_col, value_col="L0")
    spread_ok = spread_sign_matches(spread, spec.expected_sign)

    feature_side = feat21.select("month_idx", "symbol", pl.col(feature_col).alias("value"))
    real_abs_t = abs(t_nw) if math.isfinite(t_nw) else float("nan")
    placebo_abs_t = compute_placebo_abs_t(
        feature_side,
        inputs.labels_only,
        total_months=inputs.total_months,
        shifts=placebo_shifts,
    )
    placebo_max = max((t for t in placebo_abs_t if math.isfinite(t)), default=float("nan"))
    placebo_p = placebo_p_value(real_abs_t, placebo_abs_t)

    grade = grade_feature(t_nw=t_nw, spread_sign_ok=spread_ok, placebo_p=placebo_p)

    def _decay_ic(core: pl.DataFrame) -> float:
        scoped = (
            _scope(core)
            .select("month_idx", feature_col, "L2")
            .filter(pl.col(feature_col).is_not_null())
        )
        table = monthly_rank_ic(scoped, x_col=feature_col, y_col="L2")
        values = table["ic"].drop_nulls() if table.height else table["ic"]
        return float(values.mean()) if values.len() else float("nan")

    h5_ic = _decay_ic(inputs.core5)
    h63_ic = _decay_ic(inputs.core63)

    missing_rate = _missing_rate(inputs.features_dev, feature_col, universe=universe)

    return FeatureScanRow(
        feature=feature_col,
        family=spec.family,
        universe=universe,
        expected_sign=spec.expected_sign,
        n_dates=n_dates,
        ic_mean=ic_mean,
        t_nw=t_nw,
        p_two_sided=p_two_sided,
        spread=spread,
        spread_sign_match=spread_ok,
        placebo_max_abs_t=placebo_max,
        placebo_p=placebo_p,
        grade=grade,
        h5_ic=h5_ic,
        h63_ic=h63_ic,
        missing_rate=missing_rate,
        high_missing=bool(math.isfinite(missing_rate) and missing_rate > 0.5),
    )


def run_scan(
    inputs: ScanInputs, *, placebo_shifts: list[int] | None = None
) -> list[FeatureScanRow]:
    shifts = placebo_shifts if placebo_shifts is not None else select_placebo_shifts()
    rows = [
        scan_one(spec, universe=universe, inputs=inputs, placebo_shifts=shifts)
        for universe in UNIVERSES
        for spec in FEATURE_REGISTRY
    ]
    apply_bh_within_family(rows)
    return rows


# --- 9. CLI -------------------------------------------------------------------


def _git_commit(repo: Path) -> str:
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}-dirty" if dirty else head


def _grade_distribution(rows: list[FeatureScanRow], *, universe: str) -> dict[str, int]:
    counts = {g: 0 for g in ("A", "B", "C", "D", "R")}
    for row in rows:
        if row.universe == universe:
            counts[row.grade] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="output 하위 디렉터리 이름 (기본: 오늘 날짜, snapshot_date=<날짜>)",
    )
    args = parser.parse_args(argv)

    root = DataRoot.resolve(market="us")
    inputs = build_scan_inputs(root)
    shifts = select_placebo_shifts()
    rows = run_scan(inputs, placebo_shifts=shifts)

    kept_all, dropped_all = dedup_ab_features(rows, inputs.core21, universe="all")
    kept_p5, dropped_p5 = dedup_ab_features(
        rows, inputs.core21.filter(pl.col("price_ge_5")), universe="price_ge_5"
    )

    table = pl.DataFrame([row.as_dict() for row in rows])

    snapshot_date = args.snapshot_date or date.today().isoformat()
    out_dir = root.output / "feature_scan" / f"snapshot_date={snapshot_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out_dir / "feature_scan.parquet")
    table.write_csv(out_dir / "feature_scan.csv")

    modeler_repo = Path(__file__).resolve().parents[3]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "modeler_git_commit": _git_commit(modeler_repo),
        "dev_start": DEV_START.isoformat(),
        "dev_end": DEV_END.isoformat(),
        "rebalance_dates_dev_h21": inputs.total_months,
        "hac_lag": HAC_LAG,
        "bh_q_threshold": BH_Q_THRESHOLD,
        "dedup_rho_threshold": DEDUP_RHO_THRESHOLD,
        "min_names": MIN_NAMES,
        "n_placebo": N_PLACEBO,
        "placebo_shift_range": [PLACEBO_SHIFT_MIN, PLACEBO_SHIFT_MAX],
        "placebo_sample_seed": PLACEBO_SAMPLE_SEED,
        "placebo_shifts": shifts,
        "n_features_tested": len(FEATURE_REGISTRY),
        "grade_distribution": {u: _grade_distribution(rows, universe=u) for u in UNIVERSES},
        "model_input_features": {"all": kept_all, "price_ge_5": kept_p5},
        "dedup_dropped": {"all": dropped_all, "price_ge_5": dropped_p5},
        "row_count": table.height,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    )

    print(f"{out_dir}  {table.height}행")
    for universe in UNIVERSES:
        dist = _grade_distribution(rows, universe=universe)
        print(
            f"  [{universe}] A={dist['A']} B={dist['B']} C={dist['C']} "
            f"D={dist['D']} R={dist['R']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
