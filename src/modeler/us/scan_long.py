"""단일피쳐 **롱 전용** 초과수익 등급표 — 조사다. 판정이 아니다.

``modeler.us.scan``(M3, 전 횡단면 rank IC)을 본떠 만든다. M3는 롱·숏을 합쳐
잰다 — 미국 1차 모델링 결과([``01_result.md`` §3.1](../../../../../my/
milestones/us/plan/20260920_modeling_first_touch/01_result.md))가 보여준
문제가 바로 이것이다. 전략은 long-only인데 ``S``(상위10%−하위10%)의 88%가
숏 쪽이었다 — 개발 구간에서도 롱 쪽 몫은 12%뿐이었고 holdout에서는 롱 쪽이
음수로 뒤집혔다. **M3 등급표(A~D·R)에 이 숏 신호가 그대로 얹혀 있었을 수
있다** — 이 모듈은 같은 44개 피쳐를 롱 쪽 초과수익 하나로 다시 잘라 A가
몇 개나 남는지 본다.

    uv run --offline python -m modeler.us.scan_long

**통계량**(``00_candidate_modeling/04_feature_test_plan.md`` §1·§4의 등급
문턱을 그대로 쓰되 통계량만 바꾼다)::

    매월 첫 거래일 t, 피쳐 f, "상위"는 f의 방향(``effective_direction``)에 맞춘 것:
      long_t  = mean(L0 | 상위 10%) − mean(L0 | 그날 유니버스 전체)
      short_t = mean(L0 | 그날 유니버스 전체) − mean(L0 | 하위 10%)
      top100_t(참고) = mean(L0 | 상위 100종목) − mean(L0 | 유니버스)
    LONG = mean(long_t), t_LONG = HAC(long_t, lag=3) — 이 값이 등급을 정한다.

**피쳐를 더하거나 정의를 바꾸지 않는다**(``us-features-frozen`` 동결).
``scan.FEATURE_REGISTRY`` 44개를 그대로 쓴다 — 다른 축으로 다시 자를 뿐이다.

**개발 구간과 holdout을 하나의 판정으로 섞지 않는다.** 개발 구간(``scan.
DEV_END`` 이전)만 등급·placebo를 매긴다. holdout(``m7_run.HOLDOUT_START``~
``HOLDOUT_END``)은 ``window="holdout_post_hoc"``로 따로 표시하고, **등급도
placebo도 매기지 않는다** — 방향(``effective_sign``)은 개발 구간에서 정한
것을 그대로 재사용한다(홀드아웃에서 다시 정하면 그 자체가 홀드아웃을 보고
고른 것이 된다). 참고용 숫자일 뿐이다.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import polars as pl

from modeler.etl.config import DataRoot
from modeler.etl.metrics import two_sided_normal_p
from modeler.us import scan
from modeler.us.m4_run import FEATURE_SCAN_DIR_NAME
from modeler.us.m7_run import (
    HOLDOUT_END,
    HOLDOUT_START,
    assert_holdout_window,
    load_holdout_frame,
)

# --- 0. 등급 문턱 — ``04`` §1과 같은 |t| 경계, 부호 조건만 LONG으로 바꾼다 ------

#: A: ``|t_LONG| >= 3.0`` **그리고** ``LONG > 0``.
LONG_GRADE_A_ABS_T = 3.0
#: B: ``2.0 <= |t_LONG| < 3.0`` **그리고** ``LONG > 0``.
LONG_GRADE_B_ABS_T = 2.0
#: C: ``1.0 <= |t_LONG| < 2.0`` — **부호를 보지 않는다**(표 그대로). 그래서
#: LONG이 음수라도 |t|가 이 구간이면 C가 나올 수 있다. |t|가 이보다 커서
#: A·B 문턱은 넘었는데 부호가 틀린 경우만 D로 떨어진다 — 그런 경우를
#: ``short_only``로 따로 표시한다.
LONG_GRADE_C_ABS_T = 1.0

#: "숏 전용" 표시 문턱 — LONG이 음수이면서 |t_LONG|이 이 값(B 등급 문턱과
#: 같다) 이상이면, 부호만 아니었으면 A·B였을 신호라는 뜻이라 따로 표시한다.
#: 지시("LONG이 음수인데 |t|가 큰 피쳐는 A가 아니다. 다만 표에 남기고
#: '숏 전용'으로 표시해라")를 구체적인 수로 정한 것 — ``04``에 이 숫자가
#: 없어 여기서 B 문턱을 그대로 재사용한다(등급 결과와 어긋나지 않게).
SHORT_ONLY_ABS_T = LONG_GRADE_B_ABS_T

#: "상위 100종목" 참고 통계량의 종목 수 — ``01_result.md``의 top-100 관례와
#: 맞춘다.
TOP_K = 100

Grade = Literal["A", "B", "C", "D", "R"]
Window = Literal["dev", "holdout_post_hoc"]


# --- 1. 방향(effective_sign) 결정 -------------------------------------------


def effective_direction(
    spec: scan.FeatureSpec, dev_ic_mean: float
) -> tuple[Literal["+", "-"], str]:
    """ "상위 10%"가 어느 쪽인지 정한다.

    ``04`` §3에 예상 부호가 등록된 피쳐는 그 부호를 그대로 쓴다
    (``sign_source="registered"``). 등록되지 않은 넷(``days_since_earn``·
    ``days_to_earn``·``sp500_member``·``iv_isna``)은 **개발 구간 IC(feature
    vs L2) 부호**로 정한다(``sign_source="in_sample_dev_ic"``) — 지시(§3)
    그대로다. **이건 in-sample로 방향을 정한 것이라 낙관 쪽으로 치우친다** —
    호출부가 ``sign_source``를 표에 남겨 드러낸다.
    """
    if spec.expected_sign is not None:
        return spec.expected_sign, "registered"
    if not math.isfinite(dev_ic_mean) or dev_ic_mean == 0:
        return "+", "in_sample_dev_ic_undefined"
    return ("+" if dev_ic_mean > 0 else "-"), "in_sample_dev_ic"


def scored_column(feature_col: str, sign: Literal["+", "-"]) -> pl.Expr:
    """``sign``에 맞춰 "값이 클수록 상위"가 되게 부호를 맞춘 식.

    ``Float64``로 캐스트한 뒤 뒤집는다 — ``n_8k_90``류 정수 카운트 피쳐가
    부호 없는 정수(``UInt32``)라 그대로 ``-col``을 하면 polars가
    ``neg not supported for dtype u32``로 죽는다(실측, 2026-09-21).
    """
    col = pl.col(feature_col).cast(pl.Float64)
    return col if sign == "+" else -col


# --- 2. 월별 long_t · short_t · top100_t ------------------------------------


def long_short_monthly(
    df: pl.DataFrame,
    *,
    feature_col: str,
    sign: Literal["+", "-"],
    value_col: str = "L0",
    group_col: str = "month_idx",
    min_names: int = scan.MIN_NAMES,
    fraction: float = scan.DECILE_FRACTION,
    top_k: int = TOP_K,
) -> pl.DataFrame:
    """``group_col``(매월 첫 거래일)별 ``long``·``short``·``top100``.

    순위는 ``decile_spread``와 같은 이유로 ``method="ordinal"``을 쓴다 — 값
    대부분이 0으로 묶인 피쳐(``div_yield``·``ins_cluster_90`` 류)가
    ``"average"``에서 top/bottom 어느 쪽에도 안 걸려 그 달이 NaN이 되는 문제를
    피한다(``scan.decile_spread`` 참고). "유니버스 전체"는 그 피쳐가
    결측이 아닌 그 달 population 전체다(``04`` §5 "피쳐 결측 → 검정에서는
    제외"와 같은 population).

    반환: ``group_col, n, long, short, top100`` (``group_col``으로 정렬됨).
    """
    scored = df.with_columns(scored_column(feature_col, sign).alias("_score"))
    ranked = scored.with_columns(
        pl.col("_score").rank(method="ordinal").over(group_col).alias("_r"),
        pl.len().over(group_col).cast(pl.Int64).alias("_n"),
    ).filter(pl.col("_n") >= min_names)
    empty_schema = {
        group_col: df.schema[group_col],
        "n": pl.Int64,
        "long": pl.Float64,
        "short": pl.Float64,
        "top100": pl.Float64,
    }
    if ranked.height == 0:
        return pl.DataFrame(schema=empty_schema)
    ranked = ranked.with_columns(((pl.col("_r") - 1) / (pl.col("_n") - 1)).alias("_pct"))
    per_month = ranked.group_by(group_col, maintain_order=True).agg(
        pl.first("_n").cast(pl.Int64).alias("n"),
        pl.col(value_col).mean().alias("universe_mean"),
        pl.col(value_col).filter(pl.col("_pct") >= 1 - fraction).mean().alias("top_mean"),
        pl.col(value_col).filter(pl.col("_pct") <= fraction).mean().alias("bottom_mean"),
        pl.col(value_col).filter(pl.col("_r") > pl.col("_n") - top_k).mean().alias("top100_mean"),
    )
    return per_month.select(
        group_col,
        "n",
        (pl.col("top_mean") - pl.col("universe_mean")).alias("long"),
        (pl.col("universe_mean") - pl.col("bottom_mean")).alias("short"),
        (pl.col("top100_mean") - pl.col("universe_mean")).alias("top100"),
    ).sort(group_col)


def mean_and_hac_t(
    table: pl.DataFrame,
    *,
    value_col: str,
    group_col: str = "month_idx",
    lag: int = scan.HAC_LAG,
) -> tuple[float, float, int]:
    """``table[value_col]``의 시계열 평균과 HAC t.

    ``scan.ic_and_t``를 그대로 불러 쓴다(컬럼을 ``"ic"``로 임시로 바꿔서) —
    "시계열 평균 + Newey-West HAC t"라는 계산 자체는 IC든 long_t든 같은
    식이라 다시 구현하지 않는다.
    """
    renamed = table.select(group_col, pl.col(value_col).alias("ic"))
    return scan.ic_and_t(renamed, group_col=group_col, lag=lag)


# --- 3. placebo (circular shift) — LONG 통계량에 적용 ------------------------


def compute_placebo_abs_t_long(
    feature_side: pl.DataFrame,  # month_idx, symbol, value
    labels_l0_only: pl.DataFrame,  # month_idx, symbol, L0 (유니버스 필터 없음)
    *,
    sign: Literal["+", "-"],
    total_months: int,
    shifts: list[int],
    min_names: int = scan.MIN_NAMES,
    lag: int = scan.HAC_LAG,
) -> list[float]:
    """``shifts``마다 ``|t_LONG|`` (placebo)을 낸다.

    ``scan.compute_placebo_abs_t``와 같은 shift(``scan._shift_month_index_expr``,
    같은 50개, 같은 시드)를 쓰되 통계량이 rank IC가 아니라 LONG이다. ``sign``은
    실제(비-shift) 데이터로 이미 정한 방향을 고정해 쓴다 — shift마다 다시
    정하면 방향 자체가 placebo 대상이 되어 실제 검정과 다른 것을 재는 셈이다.
    """
    abs_ts: list[float] = []
    for shift in shifts:
        shifted_labels = labels_l0_only.with_columns(
            scan._shift_month_index_expr(shift=shift, total_months=total_months).alias("month_idx")
        )
        joined = feature_side.join(shifted_labels, on=["month_idx", "symbol"], how="inner")
        monthly = long_short_monthly(
            joined, feature_col="value", sign=sign, value_col="L0", min_names=min_names
        )
        _, t_long, _ = mean_and_hac_t(monthly, value_col="long", lag=lag)
        abs_ts.append(abs(t_long) if math.isfinite(t_long) else float("nan"))
    return abs_ts


# --- 4. 등급 -----------------------------------------------------------------


def grade_long(*, t_long: float, long_mean: float, placebo_p: float | None) -> Grade:
    """롱 쪽 등급 — ``04`` §1 표 그대로, 부호 조건만 "spread 부호 일치"에서
    "LONG > 0"으로 바꿨다.

    **문자 그대로 구간을 본다**(캐스케이드 elif가 아니다): ``|t_LONG| >= 3.0``
    인데 ``LONG <= 0``이면 A도 아니고 B도 아니다. C의 구간(``[1.0, 2.0)``)에도
    안 들어가면(즉 |t|가 2.0 이상인데 부호가 틀리면) **D로 떨어진다** — "LONG이
    음수인데 |t|가 큰 피쳐는 A가 아니다. 롱으로 못 쓴다"는 지시를 그대로
    반영한 것이다. 그런 경우는 ``short_only``로 따로 표시한다(호출부).
    """
    if not math.isfinite(t_long):
        return "D"
    abs_t = abs(t_long)
    if placebo_p is not None and math.isfinite(placebo_p) and placebo_p >= scan.PLACEBO_P_REJECT:
        return "R"
    sign_ok = math.isfinite(long_mean) and long_mean > 0
    if abs_t >= LONG_GRADE_A_ABS_T and sign_ok:
        return "A"
    if LONG_GRADE_B_ABS_T <= abs_t < LONG_GRADE_A_ABS_T and sign_ok:
        return "B"
    if LONG_GRADE_C_ABS_T <= abs_t < LONG_GRADE_B_ABS_T:
        return "C"
    return "D"


def is_short_only(*, long_mean: float, t_long: float) -> bool:
    """LONG이 음수이면서 ``|t_LONG|``이 B 등급 문턱 이상 — 부호만 아니었으면
    A·B였을 신호. ``SHORT_ONLY_ABS_T`` 참고."""
    return (
        math.isfinite(long_mean)
        and long_mean < 0
        and math.isfinite(t_long)
        and abs(t_long) >= SHORT_ONLY_ABS_T
    )


# --- 5. 결과 행 ---------------------------------------------------------------


@dataclass
class LongScanRow:
    feature: str
    family: str
    universe: str
    window: Window
    expected_sign: str | None
    effective_sign: str
    sign_source: str
    n_dates: int
    long_mean: float
    long_t: float
    long_p: float
    short_mean: float
    short_t: float
    short_p: float
    top100_mean: float
    top100_t: float
    grade: Grade | None  # holdout 행은 None — 판정하지 않는다
    placebo_max_abs_t: float | None
    placebo_p: float | None
    short_only: bool | None
    dev_ic_mean: float | None
    missing_rate: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "family": self.family,
            "universe": self.universe,
            "window": self.window,
            "expected_sign": self.expected_sign,
            "effective_sign": self.effective_sign,
            "sign_source": self.sign_source,
            "n_dates": self.n_dates,
            "long_mean": self.long_mean,
            "long_t": self.long_t,
            "long_p": self.long_p,
            "short_mean": self.short_mean,
            "short_t": self.short_t,
            "short_p": self.short_p,
            "top100_mean": self.top100_mean,
            "top100_t": self.top100_t,
            "grade": self.grade,
            "placebo_max_abs_t": self.placebo_max_abs_t,
            "placebo_p": self.placebo_p,
            "short_only": self.short_only,
            "dev_ic_mean": self.dev_ic_mean,
            "missing_rate": self.missing_rate,
        }


# --- 6. 입력 조립 — 개발 구간 · holdout -----------------------------------------


@dataclass
class DevInputs:
    features_dev: pl.DataFrame
    core: pl.DataFrame  # date, month_idx, symbol, price_ge_5, <feature*44>, L0, L2
    labels_l0_only: pl.DataFrame  # month_idx, symbol, L0 (유니버스 필터 없음, placebo용)
    total_months: int


def build_dev_inputs(root: DataRoot) -> DevInputs:
    """``scan.build_scan_inputs``를 본떴다 — h5·h63 감쇠 라벨은 이 조사에
    쓰지 않아 읽지 않는다(``us_labels_h5_v1``·``us_labels_h63_v1`` 생략)."""
    features_dev = scan.load_dev_frame(root, "us_features_v1")
    labels_dev = scan.load_dev_frame(root, "us_labels_v1")
    base_cols = ["date", "symbol", "price_ge_5", *scan.FEATURE_COLUMNS]
    core = scan._with_month_idx(
        features_dev.select(base_cols).join(
            labels_dev.select("date", "symbol", "L0", "L2"), on=["date", "symbol"], how="inner"
        )
    )
    scan.assert_dev_window(core)
    labels_l0_only = core.select("month_idx", "symbol", "L0").unique()
    total_months = int(core["month_idx"].max()) if core.height else 0
    return DevInputs(
        features_dev=features_dev,
        core=core,
        labels_l0_only=labels_l0_only,
        total_months=total_months,
    )


@dataclass
class HoldoutInputs:
    core: pl.DataFrame  # date, month_idx, symbol, price_ge_5, <feature*44>, L0


def build_holdout_inputs(root: DataRoot) -> HoldoutInputs:
    """``m7_run.load_holdout_frame``으로 holdout 구간만 읽는다 — 개발 구간
    벽(``scan.DEV_END``)이 아니라 holdout 벽(``HOLDOUT_START``~``HOLDOUT_END``)을
    쓴다는 점이 ``build_dev_inputs``와 다르다."""
    features_hold = load_holdout_frame(root, "us_features_v1")
    labels_hold = load_holdout_frame(root, "us_labels_v1")
    base_cols = ["date", "symbol", "price_ge_5", *scan.FEATURE_COLUMNS]
    core = scan._with_month_idx(
        features_hold.select(base_cols).join(
            labels_hold.select("date", "symbol", "L0"), on=["date", "symbol"], how="inner"
        )
    )
    assert_holdout_window(core)
    return HoldoutInputs(core=core)


def _scope(core: pl.DataFrame, *, universe: str) -> pl.DataFrame:
    return core if universe == "all" else core.filter(pl.col("price_ge_5"))


# --- 7. 피쳐 하나 검정 ---------------------------------------------------------


def scan_one_long_dev(
    spec: scan.FeatureSpec,
    *,
    universe: str,
    inputs: DevInputs,
    placebo_shifts: list[int],
) -> LongScanRow:
    feature_col = spec.feature
    core = _scope(inputs.core, universe=universe)
    feat = core.select("month_idx", "symbol", feature_col, "L0", "L2").filter(
        pl.col(feature_col).is_not_null()
    )

    dev_ic_mean = float("nan")
    if spec.expected_sign is None:
        ic_table = scan.monthly_rank_ic(feat, x_col=feature_col, y_col="L2")
        dev_ic_mean, _, _ = scan.ic_and_t(ic_table)
    sign, sign_source = effective_direction(spec, dev_ic_mean)

    monthly = long_short_monthly(feat, feature_col=feature_col, sign=sign, value_col="L0")
    long_mean, long_t, n_dates = mean_and_hac_t(monthly, value_col="long")
    short_mean, short_t, _ = mean_and_hac_t(monthly, value_col="short")
    top100_mean, top100_t, _ = mean_and_hac_t(monthly, value_col="top100")
    long_p = two_sided_normal_p(long_t)
    short_p = two_sided_normal_p(short_t)

    feature_side = feat.select("month_idx", "symbol", pl.col(feature_col).alias("value"))
    real_abs_t = abs(long_t) if math.isfinite(long_t) else float("nan")
    placebo_abs_t = compute_placebo_abs_t_long(
        feature_side,
        inputs.labels_l0_only,
        sign=sign,
        total_months=inputs.total_months,
        shifts=placebo_shifts,
    )
    placebo_max = max((t for t in placebo_abs_t if math.isfinite(t)), default=float("nan"))
    placebo_p = scan.placebo_p_value(real_abs_t, placebo_abs_t)

    grade = grade_long(t_long=long_t, long_mean=long_mean, placebo_p=placebo_p)
    short_only = is_short_only(long_mean=long_mean, t_long=long_t)
    missing_rate = scan._missing_rate(inputs.features_dev, feature_col, universe=universe)

    return LongScanRow(
        feature=feature_col,
        family=spec.family,
        universe=universe,
        window="dev",
        expected_sign=spec.expected_sign,
        effective_sign=sign,
        sign_source=sign_source,
        n_dates=n_dates,
        long_mean=long_mean,
        long_t=long_t,
        long_p=long_p,
        short_mean=short_mean,
        short_t=short_t,
        short_p=short_p,
        top100_mean=top100_mean,
        top100_t=top100_t,
        grade=grade,
        placebo_max_abs_t=placebo_max,
        placebo_p=placebo_p,
        short_only=short_only,
        dev_ic_mean=dev_ic_mean if math.isfinite(dev_ic_mean) else None,
        missing_rate=missing_rate,
    )


def scan_one_long_holdout(
    spec: scan.FeatureSpec,
    *,
    universe: str,
    inputs: HoldoutInputs,
    sign: Literal["+", "-"],
    sign_source: str,
) -> LongScanRow:
    """holdout — **사후 참고용**. 등급도 placebo도 매기지 않는다. 방향은
    개발 구간에서 정한 것을 그대로 받는다(홀드아웃에서 다시 정하지 않는다)."""
    feature_col = spec.feature
    core = _scope(inputs.core, universe=universe)
    feat = core.select("month_idx", "symbol", feature_col, "L0").filter(
        pl.col(feature_col).is_not_null()
    )

    monthly = long_short_monthly(feat, feature_col=feature_col, sign=sign, value_col="L0")
    long_mean, long_t, n_dates = mean_and_hac_t(monthly, value_col="long")
    short_mean, short_t, _ = mean_and_hac_t(monthly, value_col="short")
    top100_mean, top100_t, _ = mean_and_hac_t(monthly, value_col="top100")
    long_p = two_sided_normal_p(long_t)
    short_p = two_sided_normal_p(short_t)

    return LongScanRow(
        feature=feature_col,
        family=spec.family,
        universe=universe,
        window="holdout_post_hoc",
        expected_sign=spec.expected_sign,
        effective_sign=sign,
        sign_source=sign_source,
        n_dates=n_dates,
        long_mean=long_mean,
        long_t=long_t,
        long_p=long_p,
        short_mean=short_mean,
        short_t=short_t,
        short_p=short_p,
        top100_mean=top100_mean,
        top100_t=top100_t,
        grade=None,
        placebo_max_abs_t=None,
        placebo_p=None,
        short_only=None,
        dev_ic_mean=None,
        missing_rate=None,
    )


def run_scan_long(
    dev_inputs: DevInputs,
    holdout_inputs: HoldoutInputs,
    *,
    placebo_shifts: list[int] | None = None,
) -> list[LongScanRow]:
    """개발 구간(등급·placebo 포함) + holdout(사후 참고) 전부.

    ``scan.run_scan(inputs, ...)``과 같은 모양이다 — 디스크(``DataRoot``)에서
    분리해 둬서 합성 ``DevInputs``/``HoldoutInputs``로 단위테스트할 수 있다.
    디스크에서 읽는 조립은 ``build_dev_inputs``/``build_holdout_inputs``(호출부,
    ``main``)가 한다.
    """
    shifts = placebo_shifts if placebo_shifts is not None else scan.select_placebo_shifts()

    dev_rows: list[LongScanRow] = []
    sign_by_key: dict[tuple[str, str], tuple[str, str]] = {}
    for universe in scan.UNIVERSES:
        for spec in scan.FEATURE_REGISTRY:
            row = scan_one_long_dev(
                spec, universe=universe, inputs=dev_inputs, placebo_shifts=shifts
            )
            dev_rows.append(row)
            sign_by_key[(spec.feature, universe)] = (row.effective_sign, row.sign_source)

    holdout_rows: list[LongScanRow] = []
    for universe in scan.UNIVERSES:
        for spec in scan.FEATURE_REGISTRY:
            sign, sign_source = sign_by_key[(spec.feature, universe)]
            holdout_rows.append(
                scan_one_long_holdout(
                    spec,
                    universe=universe,
                    inputs=holdout_inputs,
                    sign=sign,
                    sign_source=sign_source,
                )
            )
    return dev_rows + holdout_rows


# --- 8. M3(합산) 등급과 나란히 놓기 ---------------------------------------------


def load_m3_grades(
    root: DataRoot, *, snapshot_date: str | None = None
) -> tuple[pl.DataFrame, Path]:
    """M3(``scan.py``) 산출물에서 ``(feature, universe)``별 등급·t·IC·spread를
    읽는다 — 대조용. ``snapshot_date``를 안 주면 가장 최근 스냅샷이다."""
    base = root.output / FEATURE_SCAN_DIR_NAME
    if snapshot_date is not None:
        snapshot_dir = base / f"snapshot_date={snapshot_date}"
        if not snapshot_dir.is_dir():
            raise FileNotFoundError(f"M3 산출물이 없습니다: {snapshot_dir}")
    else:
        if not base.is_dir():
            raise FileNotFoundError(f"M3 산출물이 없습니다: {base}")
        snapshot_dirs = sorted(
            p for p in base.iterdir() if p.is_dir() and p.name.startswith("snapshot_date=")
        )
        if not snapshot_dirs:
            raise FileNotFoundError(f"{base} 아래 snapshot_date= 디렉터리가 없습니다")
        snapshot_dir = snapshot_dirs[-1]
    table = pl.read_csv(snapshot_dir / "feature_scan.csv").select(
        "feature",
        "universe",
        pl.col("grade").alias("m3_grade"),
        pl.col("t_nw").alias("m3_t_nw"),
        pl.col("ic_mean").alias("m3_ic_mean"),
        pl.col("spread").alias("m3_spread_l0"),
    )
    return table, snapshot_dir


# --- 9. CLI -------------------------------------------------------------------


def _grade_distribution(rows: list[LongScanRow], *, universe: str) -> dict[str, int]:
    counts = {g: 0 for g in ("A", "B", "C", "D", "R")}
    for row in rows:
        if row.window == "dev" and row.universe == universe and row.grade is not None:
            counts[row.grade] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="output 하위 디렉터리 이름 (기본: 오늘 날짜, snapshot_date=<날짜>)",
    )
    parser.add_argument(
        "--m3-snapshot-date",
        default=None,
        help="대조용 M3(scan.py) 스냅샷 날짜 (기본: 가장 최근)",
    )
    args = parser.parse_args(argv)

    root = DataRoot.resolve(market="us")
    shifts = scan.select_placebo_shifts()
    dev_inputs = build_dev_inputs(root)
    holdout_inputs = build_holdout_inputs(root)
    rows = run_scan_long(dev_inputs, holdout_inputs, placebo_shifts=shifts)

    m3_table, m3_snapshot_dir = load_m3_grades(root, snapshot_date=args.m3_snapshot_date)
    table = pl.DataFrame([row.as_dict() for row in rows]).join(
        m3_table, on=["feature", "universe"], how="left"
    )

    snapshot_date = args.snapshot_date or date.today().isoformat()
    out_dir = root.output / "feature_scan_long" / f"snapshot_date={snapshot_date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    table.write_parquet(out_dir / "feature_scan_long.parquet")
    table.write_csv(out_dir / "feature_scan_long.csv")

    modeler_repo = Path(__file__).resolve().parents[3]
    in_sample_direction_features = sorted(
        {spec.feature for spec in scan.FEATURE_REGISTRY if spec.expected_sign is None}
    )
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "modeler_git_commit": scan._git_commit(modeler_repo),
        "purpose": (
            "조사다. 판정이 아니다 — M3(전 횡단면 rank IC, 롱·숏 합산)를 "
            "롱 전용 초과수익 축으로 다시 잘랐다. M7 holdout에서 S의 88%가 "
            "숏 쪽이었던 것이 계기다."
        ),
        "not_a_verdict": True,
        "reference": [
            "my/milestones/us/plan/20260920_modeling_first_touch/01_result.md §3.1",
            "my/milestones/us/plan/20260920_modeling_first_touch/00_candidate_modeling/"
            "04_feature_test_plan.md §1·§4",
        ],
        "dev_start": scan.DEV_START.isoformat(),
        "dev_end": scan.DEV_END.isoformat(),
        "holdout_start": HOLDOUT_START.isoformat(),
        "holdout_end": HOLDOUT_END.isoformat(),
        "holdout_is_post_hoc_reference_only": True,
        "hac_lag": scan.HAC_LAG,
        "min_names": scan.MIN_NAMES,
        "decile_fraction": scan.DECILE_FRACTION,
        "top_k_reference": TOP_K,
        "n_placebo": scan.N_PLACEBO,
        "placebo_shift_range": [scan.PLACEBO_SHIFT_MIN, scan.PLACEBO_SHIFT_MAX],
        "placebo_sample_seed": scan.PLACEBO_SAMPLE_SEED,
        "placebo_shifts": shifts,
        "placebo_applies_to": "dev window LONG t-stat만. holdout은 placebo·등급이 없다",
        "grade_thresholds": {
            "A": f"|t_LONG| >= {LONG_GRADE_A_ABS_T} and LONG > 0",
            "B": f"{LONG_GRADE_B_ABS_T} <= |t_LONG| < {LONG_GRADE_A_ABS_T} and LONG > 0",
            "C": (
                f"{LONG_GRADE_C_ABS_T} <= |t_LONG| < {LONG_GRADE_B_ABS_T} "
                "(부호 무관, 04 §1 표 그대로)"
            ),
            "D": "그 위 나머지 — |t_LONG|이 커도 LONG<=0이면 A/B/C 범위를 벗어나 D로 떨어진다",
            "R": "placebo p >= 0.05 (04 §1과 동일 문턱, A~D보다 먼저 본다)",
        },
        "short_only_definition": f"LONG < 0 and |t_LONG| >= {SHORT_ONLY_ABS_T} (dev window에서만)",
        "direction_method": {
            "registered": "04_feature_test_plan.md §3에 등록된 부호를 그대로 쓴다",
            "in_sample_dev_ic": (
                "부호 미등록 피쳐 넷은 개발 구간 IC(feature vs L2) 부호로 방향을 정했다 "
                "— in-sample이라 낙관 쪽으로 치우칠 수 있다. holdout 행도 같은 방향을 "
                "재사용한다(다시 정하지 않는다)"
            ),
            "features_using_in_sample_direction": in_sample_direction_features,
        },
        "m3_comparison_snapshot": str(m3_snapshot_dir),
        "n_features_tested": len(scan.FEATURE_REGISTRY),
        "grade_distribution_long_dev": {
            u: _grade_distribution(rows, universe=u) for u in scan.UNIVERSES
        },
        "row_count": table.height,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    )

    print(f"{out_dir}  {table.height}행")
    for universe in scan.UNIVERSES:
        dist = _grade_distribution(rows, universe=universe)
        print(
            f"  [dev/{universe}] A={dist['A']} B={dist['B']} C={dist['C']} "
            f"D={dist['D']} R={dist['R']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
