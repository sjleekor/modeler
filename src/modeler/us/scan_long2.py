"""N1 — 2차 사전등록(롱 쪽 검정) 등급표 코드. **실행은 N2다.**

정본: ``my/milestones/us/plan/20260921_long_side/00_candidate_plan/
01_preregistration.md``(태그 ``us2-features-frozen``). 이 문서 §3~§6을 그대로
구현한다 — 해석이 갈리면 그 문서를 따른다.

    uv run --offline python -m modeler.us.scan_long2

**``modeler.us.scan``(M3)을 본떴고, ``modeler.us.scan_long``(1차 후속 조사)의
방향 결정·롱/숏 분해·placebo 골격을 그대로 가져다 쓴다.** 이 모듈이 새로
더하는 것은 셋뿐이다:

1. **top-100을 판정 통계량 자체로 쓴다** — ``scan_long``에서는 top-100이
   "참고"였고 상위 10%(decile)가 본 통계량이었다. 여기서는 반대다
   (``01`` §1 "판정 통계량").
2. **게이트 G1(비용, ``cost.py``)·G2(중립화 생존, L2)** — ``scan``·
   ``scan_long`` 둘 다 없던 것이다. G3(placebo)는 그대로 있지만 이제
   "게이트 순서"의 세 번째로 명시적으로 자리 잡는다.
3. **부호 미등록 넷을 양방향 2행씩 낸다** — ``scan_long``은 개발 구간
   in-sample IC로 하나를 골랐다(``01`` §2가 지적한 바로 그 문제). 여기서는
   고르지 않는다.

**개발 구간만 읽는다.** ``scan.DEV_END``·``scan.assert_dev_window``와 같은
벽을 그대로 쓰되, ``--dev-end``로 바꿔 받을 수 있게 인자로 뺐다(N2가
2026-06-30으로 넓힐 것이다, ``01`` 문서 밖 지시).

**피쳐를 더하거나 정의를 바꾸지 않는다**(``us-features-frozen`` 동결,
``scan.FEATURE_REGISTRY`` 44개 그대로). ``scan``·``scan_long``·``cost``·
``metrics``의 기존 함수·동작도 바꾸지 않는다 — 전부 그대로 불러 쓴다.

**N1은 이 코드를 실제 레이크(``us_features_v1``·``us_labels_v1``)에 돌리지
않는다.** ``build_scan_inputs``는 CLI가 쓸 조립 함수로 존재하지만, 단위테스트는
전부 합성 데이터만 쓴다. 실제 레이크로 등급표를 만드는 것은 N2다.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import polars as pl

from modeler.etl.config import DataRoot
from modeler.etl.metrics import benjamini_hochberg, two_sided_normal_p
from modeler.us import cost as cost_mod
from modeler.us import scan, scan_long
from modeler.us.lake import UsLake

# --- 0. 사전등록 상수 (01 §1·§3·§4·§6) --------------------------------------

#: 개발 구간 끝 기본값 — ``scan.DEV_END``와 같은 값(2025-06-30). CLI에서
#: ``--dev-end``로 바꿔 받는다(N2가 2026-06-30으로 넓힌다).
DEV_END: date = scan.DEV_END

#: top-100 바스켓 크기 — ``01`` §1 "실제 포트폴리오". ``scan_long.TOP_K``와
#: 같은 값(100)이라 새로 정의하지 않고 그대로 재노출한다.
TOP_K: int = scan_long.TOP_K

#: G1(실행 가능성)에 쓰는 비용 모델 파라미터 — ``01`` §3 G1, ``cost.py``의
#: 판정값(``cost.DEFAULT_Q_DOLLAR``·``cost.DEFAULT_K``)을 그대로 쓴다. 새로
#: 고른 숫자가 아니다(``01`` §3 G1 "임의 문턱을 쓰지 않는다").
G1_Q_DOLLAR: float = cost_mod.DEFAULT_Q_DOLLAR
G1_K: float = cost_mod.DEFAULT_K

#: G2(중립화 생존) |t| 문턱 — ``01`` §3 G2 "이미 등록돼 있던 B 등급 문턱을
#: 그대로 쓴 것" = ``scan_long.LONG_GRADE_B_ABS_T``(2.0). 새로 고르지 않는다.
G2_T_THRESHOLD: float = scan_long.LONG_GRADE_B_ABS_T

#: 등급 |t| 경계 — ``01`` §4 표. ``scan_long``이 이미 등록한 것과 같은 값
#: (A=3.0·B=2.0·C=1.0)이라 새로 정의하지 않는다.
GRADE_A_ABS_T: float = scan_long.LONG_GRADE_A_ABS_T
GRADE_B_ABS_T: float = scan_long.LONG_GRADE_B_ABS_T
GRADE_C_ABS_T: float = scan_long.LONG_GRADE_C_ABS_T

#: 판정 유니버스 — ``01`` §1 "기본 유니버스: 주가 >= $5(N-D4)". 하한 없는
#: ``"all"``도 같이 내지만(``scan.UNIVERSES``) 판정은 이쪽이다. 이 모듈은
#: 이 상수로 어느 쪽을 계산할지 가르지 않는다 — 두 유니버스 다 같은 코드
#: 경로로 낸다(``scan.py``와 같은 관례). 기록용으로만 남긴다.
PRIMARY_UNIVERSE: str = "price_ge_5"

#: CLI 출력 디렉터리 이름.
OUTPUT_DIR_NAME = "feature_scan_long2"

Grade = Literal["A", "B", "C", "D", "R", "X"]
Direction = Literal["registered", "both_a", "both_b"]


# --- 1. 방향(direction) 결정 — 01 §2 -----------------------------------------


@dataclass(frozen=True)
class DirectionSpec:
    """피쳐 하나 × 검정할 방향 하나. ``scan.FeatureSpec``에서 파생된다."""

    feature: str
    family: str
    direction: Direction
    sign: Literal["+", "-"]
    expected_sign: Literal["+", "-"] | None


def direction_specs_for(spec: scan.FeatureSpec) -> list[DirectionSpec]:
    """``01`` §2: 부호가 등록된 피쳐는 그 부호로 한 행, **부호 미등록 넷은
    양방향 2행씩** — 하나를 고르지 않는다.

    부호 미등록 넷(``days_since_earn``·``days_to_earn``·``sp500_member``·
    ``iv_isna``, ``spec.expected_sign is None``)의 두 방향을 어느 것도
    "선택"하지 않기 위해, in-sample 통계(IC 등)를 보지 않고 **고정된
    이름**으로 부호를 매긴다: ``both_a`` = ``"+"``(값이 클수록 상위),
    ``both_b`` = ``"-"``(값이 작을수록 상위). 이 매핑 자체는 임의이지만
    (``01``에 이 이름의 뜻이 적혀 있지 않다 — 판단 근거는 보고서 참고),
    **데이터를 보고 고른 것이 아니라는 점**이 핵심이다: 두 행 다 항상 나오고
    표에 항상 남는다.
    """
    if spec.expected_sign is not None:
        return [
            DirectionSpec(
                feature=spec.feature,
                family=spec.family,
                direction="registered",
                sign=spec.expected_sign,
                expected_sign=spec.expected_sign,
            )
        ]
    return [
        DirectionSpec(
            feature=spec.feature,
            family=spec.family,
            direction="both_a",
            sign="+",
            expected_sign=None,
        ),
        DirectionSpec(
            feature=spec.feature,
            family=spec.family,
            direction="both_b",
            sign="-",
            expected_sign=None,
        ),
    ]


def all_direction_specs() -> list[DirectionSpec]:
    """``scan.FEATURE_REGISTRY`` 44개 전부의 방향 행 — registered 40 + both_a/
    both_b 4×2 = 48개."""
    out: list[DirectionSpec] = []
    for spec in scan.FEATURE_REGISTRY:
        out.extend(direction_specs_for(spec))
    return out


# --- 2. 월별 통계량 -----------------------------------------------------------
#
# LONG(top-100)·LONG_d10(상위10%)·LONG_L2(top-100, L2)는 scan_long.
# long_short_monthly를 그대로(값을 바꾸지 않고) 두 번 부른다(value_col="L0",
# "L2") — 그 함수의 "top100" 필드가 이미 "top-100 평균 - 유니버스 평균"이고
# "long" 필드가 이미 "상위10% 평균 - 유니버스 평균"이다(§01 §5 LONG_d10).
# 이 모듈에서 새로 필요한 것은 둘뿐이다: bottom-100(고정 개수) SHORT — scan_long
# 의 "short"는 하위10%(decile)라 못 쓴다 — 와 바스켓 진단(ADV·종가·L2·비용).


def _ranked(
    df: pl.DataFrame,
    *,
    feature_col: str,
    sign: Literal["+", "-"],
    min_names: int = scan.MIN_NAMES,
    group_col: str = "month_idx",
) -> pl.DataFrame:
    """``scan_long.scored_column``으로 부호를 맞추고 그 달 순위(``_r``)·이름
    수(``_n``)를 붙인다 — ``min_names`` 미만인 달은 뺀다. ``monthly_bottom100_short``·
    ``monthly_basket_diagnostics``가 공유하는 준비 단계다(``scan_long.
    long_short_monthly``는 자기 안에서 같은 준비를 따로 한다 — 그 함수는
    바꾸지 않는다는 지시라 고치지 않는다)."""
    scored = df.with_columns(scan_long.scored_column(feature_col, sign).alias("_score"))
    return scored.with_columns(
        pl.col("_score").rank(method="ordinal").over(group_col).alias("_r"),
        pl.len().over(group_col).cast(pl.Int64).alias("_n"),
    ).filter(pl.col("_n") >= min_names)


def monthly_bottom100_short(
    df: pl.DataFrame,
    *,
    feature_col: str,
    sign: Literal["+", "-"],
    value_col: str = "L0",
    min_names: int = scan.MIN_NAMES,
    top_k: int = TOP_K,
    group_col: str = "month_idx",
) -> pl.DataFrame:
    """``01`` §5 ``SHORT_t = mean(L0 | 유니버스) - mean(L0 | 하위 100)`` (기록).

    ``유니버스가 top_k 미만인 달``은 하위 ``top_k``개 마스크가 사실상 유니버스
    전체가 돼 ``short``가 0 근처로 나온다 — ``scan_long.long_short_monthly``의
    top-100(``top100``) 필드가 같은 상황에서 보이는 것과 같은 동작이다(그
    함수를 고치지 않고 그대로 두는 것과 같은 이유로 여기서도 특별 취급하지
    않는다 — 보고서에 명시).
    """
    ranked = _ranked(
        df, feature_col=feature_col, sign=sign, min_names=min_names, group_col=group_col
    )
    schema = {group_col: df.schema[group_col], "n": pl.Int64, "short": pl.Float64}
    if ranked.height == 0:
        return pl.DataFrame(schema=schema)
    per_month = ranked.group_by(group_col, maintain_order=True).agg(
        pl.first("_n").cast(pl.Int64).alias("n"),
        pl.col(value_col).mean().alias("universe_mean"),
        pl.col(value_col).filter(pl.col("_r") <= top_k).mean().alias("bottom100_mean"),
    )
    return per_month.select(
        group_col, "n", (pl.col("universe_mean") - pl.col("bottom100_mean")).alias("short")
    ).sort(group_col)


def monthly_basket_diagnostics(
    df: pl.DataFrame,
    *,
    feature_col: str,
    sign: Literal["+", "-"],
    min_names: int = scan.MIN_NAMES,
    top_k: int = TOP_K,
    q_dollar: float = G1_Q_DOLLAR,
    k: float = G1_K,
    group_col: str = "month_idx",
) -> pl.DataFrame:
    """top-100 바스켓의 월별 진단 — ADV 중앙값·유니버스 ADV 중앙값·종가
    중앙값·``price_ge_5`` 비율·L2 평균·왕복비용 평균(``cost.cost_roundtrip``,
    G1이 쓰는 바로 그 값).

    ``df``는 ``close``·``adv_20d``·``sigma_daily``·``price_ge_5``·``L2``
    컬럼이 있어야 한다. 비용은 ``cost.cost_roundtrip``을 **그대로** 부른다
    (``01`` §3 G1 "비용은 ... cost.py를 그대로 쓴다") — spread·impact를 여기서
    다시 조립하지 않는다.
    """
    ranked = _ranked(
        df, feature_col=feature_col, sign=sign, min_names=min_names, group_col=group_col
    )
    schema = {
        group_col: df.schema[group_col],
        "n": pl.Int64,
        "basket_adv_median": pl.Float64,
        "universe_adv_median": pl.Float64,
        "basket_price_median": pl.Float64,
        "basket_pct_ge5": pl.Float64,
        "basket_l2_mean": pl.Float64,
        "basket_cost_roundtrip": pl.Float64,
    }
    if ranked.height == 0:
        return pl.DataFrame(schema=schema)
    ranked = ranked.with_columns(
        cost_mod.cost_roundtrip(
            pl.col("close"), pl.col("sigma_daily"), pl.col("adv_20d"), q_dollar=q_dollar, k=k
        ).alias("_cost_roundtrip")
    )
    top100_mask = pl.col("_r") > (pl.col("_n") - top_k)
    per_month = ranked.group_by(group_col, maintain_order=True).agg(
        pl.first("_n").cast(pl.Int64).alias("n"),
        pl.col("adv_20d").filter(top100_mask).median().alias("basket_adv_median"),
        pl.col("adv_20d").median().alias("universe_adv_median"),
        pl.col("close").filter(top100_mask).median().alias("basket_price_median"),
        pl.col("price_ge_5").filter(top100_mask).cast(pl.Float64).mean().alias("basket_pct_ge5"),
        pl.col("L2").filter(top100_mask).mean().alias("basket_l2_mean"),
        (
            pl.col("_cost_roundtrip")
            .filter(top100_mask & pl.col("_cost_roundtrip").is_finite())
            .mean()
            .alias("basket_cost_roundtrip")
        ),
    )
    return per_month.sort(group_col)


def _safe_mean(table: pl.DataFrame, col: str) -> float:
    if table.height == 0:
        return float("nan")
    values = table[col].drop_nulls()
    return float(values.mean()) if values.len() else float("nan")


# --- 3. placebo (circular shift) — LONG(top-100) 통계량에 적용 ----------------


def compute_placebo_abs_t_long2(
    feature_side: pl.DataFrame,  # month_idx, symbol, value
    labels_l0_only: pl.DataFrame,  # month_idx, symbol, L0 (유니버스 필터 없음)
    *,
    sign: Literal["+", "-"],
    total_months: int,
    shifts: list[int],
    min_names: int = scan.MIN_NAMES,
    top_k: int = TOP_K,
    lag: int = scan.HAC_LAG,
) -> list[float]:
    """``shifts``마다 ``|t_LONG|``(top-100) placebo.

    ``scan_long.compute_placebo_abs_t_long``과 같은 골격이다 — 다른 것은
    ``long_short_monthly``에서 읽는 필드뿐이다(``"long"``/decile 대신
    ``"top100"``). shift 자체(``scan._shift_month_index_expr``)와 p값
    (``scan.placebo_p_value``)은 그대로 가져다 쓴다 — **1차와 같은 50개
    shift·같은 시드**를 보장하는 지점이다.
    """
    abs_ts: list[float] = []
    for shift in shifts:
        shifted_labels = labels_l0_only.with_columns(
            scan._shift_month_index_expr(shift=shift, total_months=total_months).alias("month_idx")
        )
        joined = feature_side.join(shifted_labels, on=["month_idx", "symbol"], how="inner")
        monthly = scan_long.long_short_monthly(
            joined, feature_col="value", sign=sign, value_col="L0", min_names=min_names, top_k=top_k
        )
        _, t_long, _ = scan_long.mean_and_hac_t(monthly, value_col="top100", lag=lag)
        abs_ts.append(abs(t_long) if math.isfinite(t_long) else float("nan"))
    return abs_ts


# --- 4. 게이트 · 등급 — 01 §3·§4 ----------------------------------------------


def _sign_label(x: float) -> str | None:
    if not math.isfinite(x) or x == 0:
        return None
    return "+" if x > 0 else "-"


def check_g1(long_mean: float, cost_mean: float) -> bool:
    """G1 — ``01`` §3: ``LONG > 그 바스켓의 왕복 비용``. ``cost_mean``은
    호출부가 :func:`monthly_basket_diagnostics`의 ``basket_cost_roundtrip``
    시계열 평균으로 채운다(``cost.cost_roundtrip`` 산출값, G1이 이걸로 채운다)."""
    return math.isfinite(long_mean) and math.isfinite(cost_mean) and long_mean > cost_mean


def check_g2(
    t_long_l2: float, long_l2_mean: float, long_mean: float, *, threshold: float = G2_T_THRESHOLD
) -> bool:
    """G2 — ``01`` §3: ``|t(LONG on L2)| >= 2.0`` **그리고** 부호가 L0쪽
    (``long_mean``)과 같아야 한다. 부호가 0이거나(``_sign_label`` None) 값이
    비유한이면 판정 불가로 보고 실패시킨다(안전 쪽으로 닫는다)."""
    if not math.isfinite(t_long_l2):
        return False
    s_l2, s_long = _sign_label(long_l2_mean), _sign_label(long_mean)
    if s_l2 is None or s_long is None:
        return False
    return abs(t_long_l2) >= threshold and s_l2 == s_long


def check_g3(placebo_p: float) -> bool:
    """G3 — ``01`` §3: 순열 ``p >= 0.05``면 R(탈락). 통과 조건은 그 반대."""
    return math.isfinite(placebo_p) and placebo_p < scan.PLACEBO_P_REJECT


def grade_from_long_stats(t_long: float, long_mean: float) -> Grade:
    """G1·G2·G3를 **전부 통과한 행에만** 부른다.

    ``scan_long.grade_long``의 A~D 문턱(``01`` §4와 값이 같다: A=3.0·B=2.0·
    C=1.0)을 그대로 쓰되, ``placebo_p=None``을 넘겨 그 함수의 R 분기를 꺼서
    재사용한다 — G3(placebo)는 :func:`evaluate_gates`가 이 함수를 부르기
    전에 이미 따로 판정했으므로 여기서 다시 보면 안 된다.
    """
    return scan_long.grade_long(t_long=t_long, long_mean=long_mean, placebo_p=None)


def evaluate_gates(
    *,
    long_mean: float,
    cost_mean: float,
    t_long_l2: float,
    long_l2_mean: float,
    t_long: float,
    compute_placebo: Callable[[], tuple[float, float]],
) -> tuple[str, Grade, float | None, float | None]:
    """게이트를 **순서대로** 건다 — ``01`` §3 "앞에서 걸리면 뒤를 보지 않는다".

    ``compute_placebo``는 G1·G2를 **둘 다 통과했을 때만** 부른다 — placebo(50
    shift 재계산)가 이 검정에서 가장 비싼 단계라, G1·G2에서 이미 X로 떨어질
    행은 그 계산을 하지 않는다(호출 자체를 안 한다 — 지연 평가로 게이트
    순서를 강제한다).

    반환: ``(gate_failed, grade, placebo_max_abs_t, placebo_p)``.
    ``gate_failed``는 ``"G1"``/``"G2"``/``"G3"``/``"none"``.
    """
    if not check_g1(long_mean, cost_mean):
        return "G1", "X", None, None
    if not check_g2(t_long_l2, long_l2_mean, long_mean):
        return "G2", "X", None, None
    placebo_max, placebo_p = compute_placebo()
    if not check_g3(placebo_p):
        return "G3", "R", placebo_max, placebo_p
    return "none", grade_from_long_stats(t_long, long_mean), placebo_max, placebo_p


# --- 5. 결과 행 ---------------------------------------------------------------


@dataclass
class LongScanRow2:
    feature: str
    family: str
    universe: str
    direction: Direction
    expected_sign: str | None
    n_dates: int
    LONG: float
    t_LONG: float
    LONG_d10: float
    t_LONG_d10: float
    basket_cost_roundtrip: float
    LONG_L2: float
    t_LONG_L2: float
    basket_l2_mean: float
    basket_adv_median: float
    adv_ratio: float
    basket_price_median: float
    basket_pct_ge5: float
    SHORT: float
    t_SHORT: float
    placebo_max_abs_t: float | None
    placebo_p: float | None
    h63_LONG: float
    missing_rate: float
    high_missing: bool
    gate_failed: str
    grade: Grade
    bh_q: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "family": self.family,
            "universe": self.universe,
            "direction": self.direction,
            "expected_sign": self.expected_sign,
            "n_dates": self.n_dates,
            "LONG": self.LONG,
            "t_LONG": self.t_LONG,
            "LONG_d10": self.LONG_d10,
            "t_LONG_d10": self.t_LONG_d10,
            "basket_cost_roundtrip": self.basket_cost_roundtrip,
            "LONG_L2": self.LONG_L2,
            "t_LONG_L2": self.t_LONG_L2,
            "basket_l2_mean": self.basket_l2_mean,
            "basket_adv_median": self.basket_adv_median,
            "adv_ratio": self.adv_ratio,
            "basket_price_median": self.basket_price_median,
            "basket_pct_ge5": self.basket_pct_ge5,
            "SHORT": self.SHORT,
            "t_SHORT": self.t_SHORT,
            "placebo_max_abs_t": self.placebo_max_abs_t,
            "placebo_p": self.placebo_p,
            "h63_LONG": self.h63_LONG,
            "missing_rate": self.missing_rate,
            "high_missing": self.high_missing,
            "bh_q": self.bh_q,
            "gate_failed": self.gate_failed,
            "grade": self.grade,
        }


def apply_bh_within_family(rows: list[LongScanRow2]) -> None:
    """(family, universe) 안에서 BH q값을 매겨 ``row.bh_q``를 채운다 —
    ``scan.apply_bh_within_family``와 같은 그룹핑, p값은 ``t_LONG``에서 낸다
    (``01`` §1 "판정 통계량"이 ``LONG``이므로). ``etl.metrics.
    benjamini_hochberg``를 그대로 쓴다(다시 구현하지 않는다). 등급 자체는
    이 값을 참조하지 않는다(``01`` §1 "등급에는 반영하지 않고 bh_q로 기록").
    """
    by_key: dict[tuple[str, str], list[LongScanRow2]] = {}
    for row in rows:
        by_key.setdefault((row.family, row.universe), []).append(row)
    for group_rows in by_key.values():
        pvals = [two_sided_normal_p(row.t_LONG) for row in group_rows]
        qvals = benjamini_hochberg(pvals)
        for row, q in zip(group_rows, qvals, strict=True):
            row.bh_q = float(q) if math.isfinite(q) else None


# --- 6. 입력 조립 -------------------------------------------------------------


@dataclass
class ScanInputs2:
    features_dev: pl.DataFrame
    #: date, month_idx, symbol, price_ge_5, <feature*44>, L0, L2, close, adv_20d, sigma_daily
    core21: pl.DataFrame
    core63: pl.DataFrame  # date, month_idx, symbol, price_ge_5, <feature*44>, L0
    labels_l0_only: pl.DataFrame  # month_idx, symbol, L0 (유니버스 필터 없음, placebo용)
    total_months: int


def load_features_and_labels(
    root: DataRoot, *, dev_end: date = DEV_END
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """``us_features_v1``·``us_labels_v1``·``us_labels_h63_v1``을 개발 구간까지만
    읽는다(``scan.load_dev_frame`` 그대로 재사용) — ``sigma_daily``는 아직
    안 붙어 있다(레이크가 있어야 조인할 수 있다, :func:`build_scan_inputs`
    참고).

    :func:`build_scan_inputs`에서 이 부분만 뺀 이유: 디스크에 쓴 합성
    데이터셋으로 ``--dev-end`` 벽을 단위테스트하려면 ``UsLake``(레이크) 없이도
    부를 수 있는 진입점이 있어야 한다.
    """
    features_dev = scan.load_dev_frame(root, "us_features_v1", dev_end=dev_end)
    labels_dev = scan.load_dev_frame(root, "us_labels_v1", dev_end=dev_end)
    labels63_dev = scan.load_dev_frame(root, "us_labels_h63_v1", dev_end=dev_end)
    return features_dev, labels_dev, labels63_dev


def build_scan_inputs(root: DataRoot, lake: UsLake, *, dev_end: date = DEV_END) -> ScanInputs2:
    """레이크가 아니라 이미 만들어진 데이터셋을 읽는다(``scan.
    build_scan_inputs``와 같은 관례) — ``sigma_daily``만 예외로
    ``cost.daily_volatility(lake)``에서 조인한다(``m6_run.bucket_universe``와
    같은 이유: 라벨 데이터셋에 저장돼 있지 않다).

    **N1은 이 함수를 실행하지 않는다** — CLI(:func:`main`)가 쓸 조립 코드로
    존재할 뿐이다(N2가 실제로 부른다).
    """
    features_dev, labels_dev, labels63_dev = load_features_and_labels(root, dev_end=dev_end)
    sigma = cost_mod.daily_volatility(lake).select("date", "symbol", "sigma_daily").collect()
    labels_dev = labels_dev.join(sigma, on=["date", "symbol"], how="left")

    base_cols = ["date", "symbol", "price_ge_5", *scan.FEATURE_COLUMNS]
    core21 = scan._with_month_idx(
        features_dev.select(base_cols).join(
            labels_dev.select("date", "symbol", "L0", "L2", "close", "adv_20d", "sigma_daily"),
            on=["date", "symbol"],
            how="inner",
        )
    )
    core63 = scan._with_month_idx(
        features_dev.select(base_cols).join(
            labels63_dev.select("date", "symbol", "L0"), on=["date", "symbol"], how="inner"
        )
    )
    for frame in (core21, core63):
        scan.assert_dev_window(frame, dev_end=dev_end)

    labels_l0_only = core21.select("month_idx", "symbol", "L0").unique()
    total_months = int(core21["month_idx"].max()) if core21.height else 0
    return ScanInputs2(
        features_dev=features_dev,
        core21=core21,
        core63=core63,
        labels_l0_only=labels_l0_only,
        total_months=total_months,
    )


# --- 7. 피쳐 하나 검정 ---------------------------------------------------------


def scan_one(
    dspec: DirectionSpec,
    *,
    universe: str,
    inputs: ScanInputs2,
    placebo_shifts: list[int],
) -> LongScanRow2:
    feature_col = dspec.feature
    core21 = scan_long._scope(inputs.core21, universe=universe)
    feat = core21.select(
        "month_idx",
        "symbol",
        feature_col,
        "L0",
        "L2",
        "close",
        "adv_20d",
        "sigma_daily",
        "price_ge_5",
    ).filter(pl.col(feature_col).is_not_null())

    l0_table = scan_long.long_short_monthly(
        feat, feature_col=feature_col, sign=dspec.sign, value_col="L0", top_k=TOP_K
    )
    long_mean, t_long, n_dates = scan_long.mean_and_hac_t(l0_table, value_col="top100")
    long_d10_mean, t_long_d10, _ = scan_long.mean_and_hac_t(l0_table, value_col="long")

    l2_table = scan_long.long_short_monthly(
        feat, feature_col=feature_col, sign=dspec.sign, value_col="L2", top_k=TOP_K
    )
    long_l2_mean, t_long_l2, _ = scan_long.mean_and_hac_t(l2_table, value_col="top100")

    short_table = monthly_bottom100_short(
        feat, feature_col=feature_col, sign=dspec.sign, value_col="L0", top_k=TOP_K
    )
    short_mean, t_short, _ = scan_long.mean_and_hac_t(short_table, value_col="short")

    basket_table = monthly_basket_diagnostics(
        feat, feature_col=feature_col, sign=dspec.sign, top_k=TOP_K, q_dollar=G1_Q_DOLLAR, k=G1_K
    )
    basket_cost_roundtrip = _safe_mean(basket_table, "basket_cost_roundtrip")
    basket_l2_mean = _safe_mean(basket_table, "basket_l2_mean")
    basket_adv_median = _safe_mean(basket_table, "basket_adv_median")
    universe_adv_median = _safe_mean(basket_table, "universe_adv_median")
    adv_ratio_defined = (
        math.isfinite(basket_adv_median)
        and math.isfinite(universe_adv_median)
        and universe_adv_median != 0
    )
    adv_ratio = basket_adv_median / universe_adv_median if adv_ratio_defined else float("nan")
    basket_price_median = _safe_mean(basket_table, "basket_price_median")
    basket_pct_ge5 = _safe_mean(basket_table, "basket_pct_ge5")

    feature_side = feat.select("month_idx", "symbol", pl.col(feature_col).alias("value"))
    real_abs_t = abs(t_long) if math.isfinite(t_long) else float("nan")

    def _compute_placebo() -> tuple[float, float]:
        placebo_abs_t = compute_placebo_abs_t_long2(
            feature_side,
            inputs.labels_l0_only,
            sign=dspec.sign,
            total_months=inputs.total_months,
            shifts=placebo_shifts,
            top_k=TOP_K,
        )
        placebo_max = max((t for t in placebo_abs_t if math.isfinite(t)), default=float("nan"))
        placebo_p = scan.placebo_p_value(real_abs_t, placebo_abs_t)
        return placebo_max, placebo_p

    gate_failed, grade, placebo_max, placebo_p = evaluate_gates(
        long_mean=long_mean,
        cost_mean=basket_cost_roundtrip,
        t_long_l2=t_long_l2,
        long_l2_mean=long_l2_mean,
        t_long=t_long,
        compute_placebo=_compute_placebo,
    )

    core63 = scan_long._scope(inputs.core63, universe=universe)
    feat63 = core63.select("month_idx", "symbol", feature_col, "L0").filter(
        pl.col(feature_col).is_not_null()
    )
    h63_table = scan_long.long_short_monthly(
        feat63, feature_col=feature_col, sign=dspec.sign, value_col="L0", top_k=TOP_K
    )
    h63_long_mean, _, _ = scan_long.mean_and_hac_t(h63_table, value_col="top100")

    missing_rate = scan._missing_rate(inputs.features_dev, feature_col, universe=universe)

    return LongScanRow2(
        feature=feature_col,
        family=dspec.family,
        universe=universe,
        direction=dspec.direction,
        expected_sign=dspec.expected_sign,
        n_dates=n_dates,
        LONG=long_mean,
        t_LONG=t_long,
        LONG_d10=long_d10_mean,
        t_LONG_d10=t_long_d10,
        basket_cost_roundtrip=basket_cost_roundtrip,
        LONG_L2=long_l2_mean,
        t_LONG_L2=t_long_l2,
        basket_l2_mean=basket_l2_mean,
        basket_adv_median=basket_adv_median,
        adv_ratio=adv_ratio,
        basket_price_median=basket_price_median,
        basket_pct_ge5=basket_pct_ge5,
        SHORT=short_mean,
        t_SHORT=t_short,
        placebo_max_abs_t=placebo_max,
        placebo_p=placebo_p,
        h63_LONG=h63_long_mean,
        missing_rate=missing_rate,
        high_missing=bool(math.isfinite(missing_rate) and missing_rate > 0.5),
        gate_failed=gate_failed,
        grade=grade,
    )


def run_scan_long2(
    inputs: ScanInputs2, *, placebo_shifts: list[int] | None = None
) -> list[LongScanRow2]:
    """``01`` §6 검정 절차 전부 — 44개 피쳐 × 방향(registered 1개 또는
    both_a/both_b 2개) × 유니버스(``scan.UNIVERSES``) = 96행.
    """
    shifts = placebo_shifts if placebo_shifts is not None else scan.select_placebo_shifts()
    rows = [
        scan_one(dspec, universe=universe, inputs=inputs, placebo_shifts=shifts)
        for universe in scan.UNIVERSES
        for spec in scan.FEATURE_REGISTRY
        for dspec in direction_specs_for(spec)
    ]
    apply_bh_within_family(rows)
    return rows


# --- 8. CLI -------------------------------------------------------------------


def _grade_distribution(rows: list[LongScanRow2], *, universe: str) -> dict[str, int]:
    counts = {g: 0 for g in ("A", "B", "C", "D", "R", "X")}
    for row in rows:
        if row.universe == universe:
            counts[row.grade] += 1
    return counts


def _gate_failed_distribution(rows: list[LongScanRow2], *, universe: str) -> dict[str, int]:
    counts = {g: 0 for g in ("G1", "G2", "G3", "none")}
    for row in rows:
        if row.universe == universe:
            counts[row.gate_failed] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="output 하위 디렉터리 이름 (기본: 오늘 날짜, snapshot_date=<날짜>)",
    )
    parser.add_argument(
        "--dev-end",
        type=date.fromisoformat,
        default=DEV_END,
        help=f"개발 구간 끝 날짜 (기본: {DEV_END.isoformat()}). N2에서 2026-06-30으로 넓힌다",
    )
    args = parser.parse_args(argv)

    root = DataRoot.resolve(market="us")
    lake = UsLake.resolve()
    shifts = scan.select_placebo_shifts()
    inputs = build_scan_inputs(root, lake, dev_end=args.dev_end)
    rows = run_scan_long2(inputs, placebo_shifts=shifts)

    table = pl.DataFrame([row.as_dict() for row in rows])
    # N2 운영 지시("두 실행을 구분해서 남겨라 — dev_end 를 manifest 와 컬럼에") —
    # 사전등록(``01``)에는 없는 요구라 LongScanRow2(``as_dict`` 26개 필드, N1이
    # 이미 테스트해 둔 계약)는 건드리지 않고 CLI 출력 표에만 부가한다.
    table = table.with_columns(pl.lit(args.dev_end.isoformat()).alias("dev_end"))

    snapshot_date = args.snapshot_date or date.today().isoformat()
    out_dir = root.output / OUTPUT_DIR_NAME / f"snapshot_date={snapshot_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out_dir / "feature_scan_long2.parquet")
    table.write_csv(out_dir / "feature_scan_long2.csv")

    modeler_repo = Path(__file__).resolve().parents[3]
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "modeler_git_commit": scan._git_commit(modeler_repo),
        "preregistration": (
            "my/milestones/us/plan/20260921_long_side/00_candidate_plan/01_preregistration.md"
            " (태그 us2-features-frozen)"
        ),
        "dev_start": scan.DEV_START.isoformat(),
        "dev_end": args.dev_end.isoformat(),
        "primary_universe": PRIMARY_UNIVERSE,
        "top_k": TOP_K,
        "hac_lag": scan.HAC_LAG,
        "min_names": scan.MIN_NAMES,
        "n_placebo": scan.N_PLACEBO,
        "placebo_shift_range": [scan.PLACEBO_SHIFT_MIN, scan.PLACEBO_SHIFT_MAX],
        "placebo_sample_seed": scan.PLACEBO_SAMPLE_SEED,
        "placebo_shifts": shifts,
        "g1_cost_model": {
            "q_dollar": G1_Q_DOLLAR,
            "k": G1_K,
            "source": "modeler.us.cost.cost_roundtrip",
        },
        "g2_t_threshold": G2_T_THRESHOLD,
        "grade_thresholds": {
            "A": f"|t_LONG| >= {GRADE_A_ABS_T} and LONG > 0",
            "B": f"{GRADE_B_ABS_T} <= |t_LONG| < {GRADE_A_ABS_T} and LONG > 0",
            "C": f"{GRADE_C_ABS_T} <= |t_LONG| < {GRADE_B_ABS_T} (부호 무관)",
            "D": "그 아래, 또는 |t_LONG|이 커도 LONG<=0이라 A/B/C 밖",
            "R": "G3(placebo) 탈락",
            "X": "G1(비용) 또는 G2(중립화 생존) 탈락",
        },
        "direction_convention": {
            "registered": "01 §2 등록 부호를 그대로 쓴다",
            "both_a": "부호 미등록 넷 — '+' (값이 클수록 상위)",
            "both_b": "부호 미등록 넷 — '-' (값이 작을수록 상위). in-sample로 고르지 않는다",
        },
        "n_features_tested": len(scan.FEATURE_REGISTRY),
        "row_count": table.height,
        "grade_distribution": {u: _grade_distribution(rows, universe=u) for u in scan.UNIVERSES},
        "gate_failed_distribution": {
            u: _gate_failed_distribution(rows, universe=u) for u in scan.UNIVERSES
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    )

    print(f"{out_dir}  {table.height}행")
    for universe in scan.UNIVERSES:
        dist = _grade_distribution(rows, universe=universe)
        gates = _gate_failed_distribution(rows, universe=universe)
        print(
            f"  [{universe}] A={dist['A']} B={dist['B']} C={dist['C']} D={dist['D']} "
            f"R={dist['R']} X={dist['X']}  (G1={gates['G1']} G2={gates['G2']} G3={gates['G3']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
