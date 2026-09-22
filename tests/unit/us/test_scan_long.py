"""``modeler.us.scan_long`` 단위 테스트. 전부 합성 데이터다 — 실제 레이크를 읽지
않는다.

이 조사(20260921 롱 전용 등급표)가 요구하는 넷을 검사한다: 방향 결정
(``effective_direction``), 롱/숏 분해(``long_short_monthly``), placebo 재사용
(``compute_placebo_abs_t_long``이 ``scan._shift_month_index_expr``·
``scan.placebo_p_value``를 그대로 쓰는가), 개발/holdout 분리(``build_dev_inputs``·
``build_holdout_inputs``가 각자의 날짜 벽만 읽는가).
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us.dataset import write_dataset
from modeler.us.m7_run import HOLDOUT_END, HOLDOUT_START
from modeler.us.scan import DEV_END, FEATURE_REGISTRY, FeatureSpec
from modeler.us.scan_long import (
    LONG_GRADE_B_ABS_T,
    SHORT_ONLY_ABS_T,
    DevInputs,
    HoldoutInputs,
    build_dev_inputs,
    build_holdout_inputs,
    compute_placebo_abs_t_long,
    effective_direction,
    grade_long,
    is_short_only,
    long_short_monthly,
    mean_and_hac_t,
    run_scan_long,
    scan_one_long_dev,
    scan_one_long_holdout,
    scored_column,
)

# --- 1. 방향(effective_direction) 결정 ----------------------------------------


def test_effective_direction_uses_registered_sign_when_present() -> None:
    spec = FeatureSpec("mom_12_1", "F1_momentum", "+")
    sign, source = effective_direction(spec, float("nan"))
    assert sign == "+"
    assert source == "registered"

    spec_minus = FeatureSpec("rv_20", "F3_volatility", "-")
    sign, source = effective_direction(spec_minus, 0.5)  # dev_ic_mean 무시돼야 한다
    assert sign == "-"
    assert source == "registered"


def test_effective_direction_uses_dev_ic_sign_when_unregistered() -> None:
    spec = FeatureSpec("days_to_earn", "F9_earnings", None)

    sign, source = effective_direction(spec, 0.03)
    assert sign == "+"
    assert source == "in_sample_dev_ic"

    sign, source = effective_direction(spec, -0.03)
    assert sign == "-"
    assert source == "in_sample_dev_ic"


def test_effective_direction_handles_nan_or_zero_dev_ic() -> None:
    spec = FeatureSpec("sp500_member", "F14_index_membership", None)
    sign, source = effective_direction(spec, float("nan"))
    assert sign == "+"
    assert source == "in_sample_dev_ic_undefined"

    sign, source = effective_direction(spec, 0.0)
    assert source == "in_sample_dev_ic_undefined"


def test_scored_column_flips_sign_only_for_minus() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, -3.0]})
    plus = df.select(scored_column("x", "+").alias("s"))["s"].to_list()
    minus = df.select(scored_column("x", "-").alias("s"))["s"].to_list()
    assert plus == [1.0, 2.0, -3.0]
    assert minus == [-1.0, -2.0, 3.0]


# --- 2. 롱/숏/top100 분해 -------------------------------------------------------


def test_long_short_monthly_matches_hand_computed_values() -> None:
    # 10개 이름, x 오름차순 1..10, sign="+"라 값이 클수록 "상위".
    # top10% = {x=10}, bottom10% = {x=1}, 유니버스 평균 = mean(L0).
    df = pl.DataFrame(
        {
            "month_idx": [1] * 10,
            "symbol": [f"S{i:02d}" for i in range(10)],
            "x": list(range(1, 11)),
            "L0": [float(v) / 10 for v in range(1, 11)],  # 0.1..1.0
        }
    )
    table = long_short_monthly(df, feature_col="x", sign="+", value_col="L0", min_names=5)
    assert table.height == 1
    row = table.row(0, named=True)
    universe_mean = sum(v / 10 for v in range(1, 11)) / 10
    assert row["long"] == pytest.approx(1.0 - universe_mean)
    assert row["short"] == pytest.approx(universe_mean - 0.1)
    assert row["n"] == 10


def test_long_short_monthly_sign_flip_swaps_top_and_bottom() -> None:
    df = pl.DataFrame(
        {
            "month_idx": [1] * 10,
            "symbol": [f"S{i:02d}" for i in range(10)],
            "x": list(range(1, 11)),
            "L0": [float(v) / 10 for v in range(1, 11)],
        }
    )
    plus_row = long_short_monthly(df, feature_col="x", sign="+", value_col="L0", min_names=5).row(
        0, named=True
    )
    minus_row = long_short_monthly(df, feature_col="x", sign="-", value_col="L0", min_names=5).row(
        0, named=True
    )
    # sign="-"면 x=1(원래 bottom)이 "상위"가 된다. 두 쪽 다 같은 universe_mean을
    # 쓰므로 "top 그룹"과 "bottom 그룹"이 통째로 맞바뀐다:
    #   minus.long  = (원래 bottom 그룹 평균) − universe_mean = −plus.short
    #   minus.short = universe_mean − (원래 top 그룹 평균)    = −plus.long
    assert plus_row["long"] == pytest.approx(-minus_row["short"])
    assert plus_row["short"] == pytest.approx(-minus_row["long"])


def test_long_short_monthly_top100_falls_back_to_full_group_when_smaller() -> None:
    # n=10 < top_k=100 -> top100 그룹이 전체가 되어 top100 == universe_mean 대비 0.
    df = pl.DataFrame(
        {
            "month_idx": [1] * 10,
            "symbol": [f"S{i:02d}" for i in range(10)],
            "x": list(range(1, 11)),
            "L0": [float(v) for v in range(1, 11)],
        }
    )
    table = long_short_monthly(df, feature_col="x", sign="+", value_col="L0", min_names=5)
    assert table.row(0, named=True)["top100"] == pytest.approx(0.0, abs=1e-9)


def test_long_short_monthly_drops_months_below_min_names() -> None:
    df = pl.DataFrame(
        {
            "month_idx": [1, 1, 2, 2, 2, 2, 2],
            "symbol": [f"S{i:02d}" for i in range(7)],
            "x": [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "L0": [0.1, 0.2, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    table = long_short_monthly(df, feature_col="x", sign="+", value_col="L0", min_names=3)
    assert table["month_idx"].to_list() == [2]


def test_long_short_monthly_empty_when_no_month_qualifies() -> None:
    df = pl.DataFrame(
        {"month_idx": [1, 1], "symbol": ["S0", "S1"], "x": [1.0, 2.0], "L0": [0.1, 0.2]}
    )
    table = long_short_monthly(df, feature_col="x", sign="+", value_col="L0", min_names=10)
    assert table.height == 0
    assert set(table.columns) == {"month_idx", "n", "long", "short", "top100"}


def test_mean_and_hac_t_delegates_to_scan_ic_and_t() -> None:
    from modeler.us.scan import ic_and_t

    table = pl.DataFrame({"month_idx": [1, 2, 3, 4, 5], "long": [0.1, 0.2, -0.05, 0.15, 0.3]})
    mean_v, t_v, n_v = mean_and_hac_t(table, value_col="long", lag=3)

    expected_mean, expected_t, expected_n = ic_and_t(
        table.rename({"long": "ic"}), group_col="month_idx", lag=3
    )
    assert mean_v == pytest.approx(expected_mean)
    assert t_v == pytest.approx(expected_t)
    assert n_v == expected_n


# --- 3. 등급 --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t_long", "long_mean", "placebo_p", "expected"),
    [
        (3.5, 0.01, 0.01, "A"),
        (-3.5, 0.01, 0.01, "A"),  # 부호는 long_mean으로만 본다 — t_long 자체의 부호는 무관
        (2.5, 0.01, 0.01, "B"),
        (1.5, 0.01, 0.01, "C"),
        (1.5, -0.01, 0.01, "C"),  # C는 부호를 보지 않는다(04 §1 표 그대로)
        (0.5, 0.01, 0.01, "D"),
        (4.0, 0.01, 0.5, "R"),  # placebo가 |t| 구간보다 먼저다
        (float("nan"), 0.01, 0.5, "D"),
    ],
)
def test_grade_long_boundaries(t_long, long_mean, placebo_p, expected) -> None:
    assert grade_long(t_long=t_long, long_mean=long_mean, placebo_p=placebo_p) == expected


def test_grade_long_big_t_with_wrong_sign_falls_to_d_not_b() -> None:
    """LONG이 음수인데 |t|가 3.5(A 문턱을 넘음)면 A도 B도 아니고, C의 구간
    ([1,2))에도 안 들어가 D로 떨어져야 한다 — 캐스케이드 elif가 아니라 문자
    그대로의 구간 판정이라는 뜻이다."""
    grade = grade_long(t_long=3.5, long_mean=-0.02, placebo_p=0.01)
    assert grade == "D"


def test_grade_long_moderate_t_with_wrong_sign_still_gets_c() -> None:
    """|t|가 [1,2) 구간이면 부호가 틀려도 C다(표에 부호 조건이 없다)."""
    grade = grade_long(t_long=1.5, long_mean=-0.02, placebo_p=0.01)
    assert grade == "C"


def test_grade_long_r_boundary_is_ge_not_gt() -> None:
    assert grade_long(t_long=4.0, long_mean=0.01, placebo_p=0.05) == "R"
    assert grade_long(t_long=4.0, long_mean=0.01, placebo_p=0.0499) == "A"


def test_grade_long_nan_placebo_p_does_not_force_r() -> None:
    assert grade_long(t_long=4.0, long_mean=0.01, placebo_p=float("nan")) == "A"


# --- 4. "숏 전용" 표시 -----------------------------------------------------------


def test_is_short_only_requires_negative_long_and_large_t() -> None:
    assert is_short_only(long_mean=-0.02, t_long=-3.5) is True
    assert is_short_only(long_mean=-0.02, t_long=3.5) is True  # |t|만 본다
    assert is_short_only(long_mean=0.02, t_long=3.5) is False  # 부호가 맞으면 아니다
    assert is_short_only(long_mean=-0.02, t_long=1.0) is False  # |t|가 문턱 미만
    assert is_short_only(long_mean=-0.02, t_long=float("nan")) is False


def test_short_only_threshold_matches_b_grade_threshold() -> None:
    assert SHORT_ONLY_ABS_T == LONG_GRADE_B_ABS_T


# --- 5. placebo 재사용 -----------------------------------------------------------


def test_compute_placebo_abs_t_long_returns_one_value_per_shift() -> None:
    months = list(range(1, 13))
    symbols = [f"S{i:02d}" for i in range(30)]
    rng = np.random.default_rng(0)
    rows = []
    for m in months:
        perm = rng.permutation(30)
        for rank, sym_idx in enumerate(perm):
            rows.append({"month_idx": m, "symbol": symbols[sym_idx], "value": float(rank)})
    feature_side = pl.DataFrame(rows)
    labels_only = feature_side.rename({"value": "L0"})

    shifts = [3, 5, 7]
    abs_ts = compute_placebo_abs_t_long(
        feature_side, labels_only, sign="+", total_months=12, shifts=shifts, min_names=10
    )
    assert len(abs_ts) == len(shifts)
    assert all(math.isfinite(t) for t in abs_ts)


def test_compute_placebo_abs_t_long_real_association_beats_shifted() -> None:
    """진짜(shift=0에 준하는) 관계는 강하고, 멀리 shift한 것은 약해야 한다 —
    M3 placebo의 같은 논리(느린 추세가 아니면 shift가 관계를 깬다)."""
    n_months = 24
    n_symbols = 40
    rng = np.random.default_rng(11)
    real_rows, label_rows = [], []
    for m in range(1, n_months + 1):
        ranks = rng.permutation(n_symbols).astype(float)
        for s in range(n_symbols):
            real_rows.append({"month_idx": m, "symbol": f"S{s:03d}", "value": ranks[s]})
            # 라벨은 feature와 거의 같은 순서(진짜 신호), 아주 약간의 잡음만.
            label_rows.append(
                {
                    "month_idx": m,
                    "symbol": f"S{s:03d}",
                    "L0": float(ranks[s]) + rng.normal(scale=0.5),
                }
            )
    feature_side = pl.DataFrame(real_rows)
    labels_only = pl.DataFrame(label_rows)

    real_monthly = long_short_monthly(
        feature_side.join(labels_only, on=["month_idx", "symbol"]),
        feature_col="value",
        sign="+",
        value_col="L0",
        min_names=10,
    )
    _, real_t, _ = mean_and_hac_t(real_monthly, value_col="long")

    shifted_abs_t = compute_placebo_abs_t_long(
        feature_side,
        labels_only,
        sign="+",
        total_months=n_months,
        shifts=[11, 13],  # T=24라 12의 배수를 피한 중간값
        min_names=10,
    )
    assert abs(real_t) > max(shifted_abs_t)


# --- 6. 개발/holdout 분리 --------------------------------------------------------


def _write_synthetic_lake(root: DataRoot, *, dates: list[date], n_symbols: int = 30) -> None:
    """``us_features_v1``·``us_labels_v1``을 ``FEATURE_REGISTRY`` 44개 컬럼
    전부 채워 쓴다 — ``build_dev_inputs``/``build_holdout_inputs``가 그대로
    읽는 이름이다."""
    rng = np.random.default_rng(5)
    feature_names = [spec.feature for spec in FEATURE_REGISTRY]
    feature_rows = []
    label_rows = []
    for d in dates:
        l2 = rng.permutation(n_symbols).astype(float)
        for s in range(n_symbols):
            frow = {"date": d, "symbol": f"S{s:03d}", "price_ge_5": True}
            for name in feature_names:
                frow[name] = float(l2[s]) + rng.normal(scale=1.0)
            feature_rows.append(frow)
            label_rows.append(
                {
                    "date": d,
                    "symbol": f"S{s:03d}",
                    "L0": float(l2[s]) / n_symbols,
                    "L2": float(l2[s]),
                }
            )
    write_dataset(pl.DataFrame(feature_rows), root, "us_features_v1", manifest={})
    write_dataset(pl.DataFrame(label_rows), root, "us_labels_v1", manifest={})


def test_build_dev_inputs_only_sees_dev_window(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    dates = [
        date(2020, 1, 2),
        date(2020, 2, 3),
        DEV_END,
        date(2025, 7, 15),  # holdout — dev에 섞이면 안 된다
        date(2026, 1, 5),  # holdout 뒤 — dev에도 holdout에도 섞이면 안 된다
    ]
    _write_synthetic_lake(root, dates=dates)

    inputs = build_dev_inputs(root)

    assert isinstance(inputs, DevInputs)
    assert inputs.core["date"].max() <= DEV_END
    assert inputs.total_months == 3  # dates_in_dev 3개


def test_build_holdout_inputs_only_sees_holdout_window(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    dates = [
        date(2020, 1, 2),  # dev — holdout에 섞이면 안 된다
        HOLDOUT_START,
        date(2025, 12, 1),
        HOLDOUT_END,
        date(2026, 9, 1),  # holdout 뒤 — 섞이면 안 된다
    ]
    _write_synthetic_lake(root, dates=dates)

    inputs = build_holdout_inputs(root)

    assert isinstance(inputs, HoldoutInputs)
    assert inputs.core["date"].min() >= HOLDOUT_START
    assert inputs.core["date"].max() <= HOLDOUT_END
    assert set(inputs.core["date"].unique().to_list()) == {
        HOLDOUT_START,
        date(2025, 12, 1),
        HOLDOUT_END,
    }


def test_scan_one_long_holdout_never_grades_or_placebos() -> None:
    """holdout 행은 등급·placebo가 전부 None이어야 한다 — "판정에 쓰지
    마라"를 코드로 강제한 것."""
    df = pl.DataFrame(
        {
            "month_idx": [1] * 20,
            "date": [date(2025, 7, 1)] * 20,
            "symbol": [f"S{i:02d}" for i in range(20)],
            "price_ge_5": [True] * 20,
            "mom_12_1": [float(i) for i in range(20)],
            "L0": [float(i) / 20 for i in range(20)],
        }
    )
    inputs = HoldoutInputs(core=df)
    spec = FeatureSpec("mom_12_1", "F1_momentum", "+")
    row = scan_one_long_holdout(
        spec, universe="all", inputs=inputs, sign="+", sign_source="registered"
    )
    assert row.window == "holdout_post_hoc"
    assert row.grade is None
    assert row.placebo_max_abs_t is None
    assert row.placebo_p is None
    assert row.short_only is None
    assert math.isfinite(row.long_mean)


# --- 7. end-to-end (합성 DevInputs/HoldoutInputs, 44개 x 2 유니버스) -------------


def _synthetic_dev_inputs(*, n_months: int = 12, n_symbols: int = 40, seed: int = 3) -> DevInputs:
    rng = np.random.default_rng(seed)
    records = []
    for m in range(1, n_months + 1):
        l2 = rng.permutation(n_symbols).astype(float)
        feature_values = {
            spec.feature: l2 + rng.normal(scale=5.0, size=n_symbols) for spec in FEATURE_REGISTRY
        }
        for s in range(n_symbols):
            record = {
                "month_idx": m,
                "symbol": f"S{s:03d}",
                "price_ge_5": True,
                "L0": l2[s] / n_symbols,
                "L2": l2[s],
            }
            for spec in FEATURE_REGISTRY:
                record[spec.feature] = float(feature_values[spec.feature][s])
            records.append(record)
    core = pl.DataFrame(records)
    feature_cols = [spec.feature for spec in FEATURE_REGISTRY]
    labels_l0_only = core.select("month_idx", "symbol", "L0").unique()
    return DevInputs(
        features_dev=core.select("symbol", "price_ge_5", *feature_cols),
        core=core,
        labels_l0_only=labels_l0_only,
        total_months=n_months,
    )


def _synthetic_holdout_inputs(dev_inputs: DevInputs, *, n_months: int = 3) -> HoldoutInputs:
    # dev 몇 달을 그대로 재사용해 holdout 흉내를 낸다 — window 분리 테스트가
    # 아니라 파이프라인 전체(run_scan_long)가 44 x 2 x 2윈도우를 다 도는지만
    # 본다.
    core = dev_inputs.core.filter(pl.col("month_idx") <= n_months)
    return HoldoutInputs(core=core)


def test_run_scan_long_produces_dev_and_holdout_rows_for_all_features() -> None:
    dev_inputs = _synthetic_dev_inputs()
    holdout_inputs = _synthetic_holdout_inputs(dev_inputs)
    shifts = [11, 13, 17, 19, 23]  # T=12 — 단위테스트 전용, 좁힌 범위

    rows = run_scan_long(dev_inputs, holdout_inputs, placebo_shifts=shifts)

    dev_rows = [r for r in rows if r.window == "dev"]
    holdout_rows = [r for r in rows if r.window == "holdout_post_hoc"]
    assert len(dev_rows) == len(FEATURE_REGISTRY) * 2
    assert len(holdout_rows) == len(FEATURE_REGISTRY) * 2
    assert all(r.grade in {"A", "B", "C", "D", "R"} for r in dev_rows)
    assert all(r.grade is None for r in holdout_rows)
    assert all(r.placebo_p is None for r in holdout_rows)

    # holdout 행은 같은 (feature, universe)의 dev 행과 같은 방향을 쓴다 —
    # 홀드아웃에서 다시 정하지 않는다.
    dev_sign = {(r.feature, r.universe): r.effective_sign for r in dev_rows}
    for r in holdout_rows:
        assert r.effective_sign == dev_sign[(r.feature, r.universe)]


#: T=24용 placebo shift 풀 — ``test_scan.py``의 ``_TEST_PLACEBO_SHIFTS``와 같은
#: 이유다: shift가 너무 적으면(예: 5개) ``placebo_p_value``의 최솟값
#: ``1/(n+1)``이 이미 0.05를 넘어(5개면 1/6≈0.167) 실제 신호와 무관하게 항상
#: R이 나온다. 21개로 최솟값을 1/22≈0.045로 낮춰 A/B가 나올 여지를 둔다.
_TEST_PLACEBO_SHIFTS = list(range(1, 22))


def _single_feature_dev_inputs(
    *, n_months: int = 24, n_symbols: int = 30, seed: int = 7
) -> DevInputs:
    """``test_scan.py``의 ``_synthetic_scan_inputs``를 본떴다 — 피쳐 하나
    (``strong_signal``)만 실제 신호를 갖고, 나머지(``no_signal``)는 잡음이다."""
    rng = np.random.default_rng(seed)
    records = []
    for m in range(1, n_months + 1):
        l2 = rng.permutation(n_symbols).astype(float)
        noise = rng.normal(scale=3.0, size=n_symbols)
        for s in range(n_symbols):
            records.append(
                {
                    "month_idx": m,
                    "symbol": f"S{s:03d}",
                    "price_ge_5": True,
                    "strong_signal": l2[s] + noise[s],
                    "no_signal": rng.permutation(n_symbols).astype(float)[s],
                    "L0": l2[s] / n_symbols,
                    "L2": l2[s],
                }
            )
    core = pl.DataFrame(records)
    labels_l0_only = core.select("month_idx", "symbol", "L0").unique()
    return DevInputs(
        features_dev=core.select("symbol", "price_ge_5", "strong_signal", "no_signal"),
        core=core,
        labels_l0_only=labels_l0_only,
        total_months=n_months,
    )


def test_scan_one_long_dev_grades_a_strongly_correlated_feature() -> None:
    inputs = _single_feature_dev_inputs()
    spec = FeatureSpec("strong_signal", "F1_momentum", "+")
    row = scan_one_long_dev(
        spec, universe="all", inputs=inputs, placebo_shifts=_TEST_PLACEBO_SHIFTS
    )

    assert row.long_mean > 0
    assert abs(row.long_t) > 2.0
    assert row.grade in {"A", "B"}  # 진짜 신호라 placebo와 뚜렷이 구별돼야 한다


def test_scan_one_long_dev_grades_low_for_pure_noise_feature() -> None:
    inputs = _single_feature_dev_inputs()
    spec = FeatureSpec("no_signal", "F1_momentum", "+")
    row = scan_one_long_dev(
        spec, universe="all", inputs=inputs, placebo_shifts=_TEST_PLACEBO_SHIFTS
    )

    assert row.grade != "A"
    # 순수 잡음이면 |t| 구간상 C/D거나 placebo와 구별이 안 돼 R로 떨어진다 —
    # 셋 다 "롱 쪽에 실을 신호가 아니다"라는 같은 결론이다.
    assert row.grade in {"C", "D", "R"}
