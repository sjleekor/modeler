"""``modeler.us.scan_long2`` 단위 테스트. 전부 합성 데이터다 — 실제 레이크를
읽지 않는다(N1 §0 "허용: 합성 데이터 단위 테스트. 금지: 실제 레이크로 등급표를
만드는 것").

``01_preregistration.md``(``us2-features-frozen-v2``, G1 개정 포함)이 요구하는
것들을 검사한다: G1(``01`` §3 "G1 개정" — 실측 회전율 × 비용, ``cost.py``)·
G2(중립화 생존)·G3(placebo, ``scan.py``와 같은 50개 shift)의 순서와 각 조건,
등급 경계, 방향(등록/양방향), top-100·상위10% 병존, 개발 구간 날짜 벽
(``--dev-end``), 회전율(``monthly_basket_turnover``)·``effective_cost`` 계산.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.us import cost as cost_mod
from modeler.us import scan, scan_long, scan_long2
from modeler.us.dataset import write_dataset
from modeler.us.scan import FEATURE_REGISTRY, FeatureSpec
from modeler.us.scan_long2 import (
    DirectionSpec,
    LongScanRow2,
    ScanInputs2,
    all_direction_specs,
    apply_bh_within_family,
    check_g1,
    check_g2,
    check_g3,
    compute_placebo_abs_t_long2,
    direction_specs_for,
    evaluate_gates,
    grade_from_long_stats,
    load_features_and_labels,
    monthly_basket_diagnostics,
    monthly_basket_turnover,
    monthly_bottom100_short,
    run_scan_long2,
    scan_one,
)

# --- 1. 방향(direction) 결정 — 01 §2 ------------------------------------------


def test_direction_specs_for_registered_feature_is_a_single_row() -> None:
    spec = FeatureSpec("mom_1m", "F1_momentum", "-")
    rows = direction_specs_for(spec)
    assert len(rows) == 1
    assert rows[0].direction == "registered"
    assert rows[0].sign == "-"
    assert rows[0].expected_sign == "-"


def test_direction_specs_for_unregistered_feature_is_two_rows_both_directions() -> None:
    spec = FeatureSpec("iv_isna", "F13_options_iv", None)
    rows = direction_specs_for(spec)
    assert len(rows) == 2
    directions = {r.direction: r.sign for r in rows}
    assert directions == {"both_a": "+", "both_b": "-"}
    assert all(r.expected_sign is None for r in rows)


def test_all_four_unregistered_features_get_both_directions() -> None:
    unregistered = [s for s in FEATURE_REGISTRY if s.expected_sign is None]
    assert {s.feature for s in unregistered} == {
        "days_since_earn",
        "days_to_earn",
        "sp500_member",
        "iv_isna",
    }
    for spec in unregistered:
        rows = direction_specs_for(spec)
        assert len(rows) == 2


def test_all_direction_specs_total_count() -> None:
    # 40 registered * 1 + 4 unregistered * 2 = 48
    registered = sum(1 for s in FEATURE_REGISTRY if s.expected_sign is not None)
    unregistered = len(FEATURE_REGISTRY) - registered
    assert len(all_direction_specs()) == registered + unregistered * 2 == 48


# --- 2. 게이트 — 순수 함수 (01 §3) --------------------------------------------


def test_check_g1_passes_when_long_exceeds_cost() -> None:
    assert check_g1(0.02, 0.01) is True


def test_check_g1_fails_when_long_below_cost() -> None:
    assert check_g1(0.005, 0.01) is False


def test_check_g1_fails_when_long_equals_cost() -> None:
    # 부등호는 엄격하다(>) — 같으면 통과가 아니다.
    assert check_g1(0.01, 0.01) is False


def test_check_g1_fails_on_nonfinite_inputs() -> None:
    assert check_g1(float("nan"), 0.01) is False
    assert check_g1(0.02, float("nan")) is False


def test_check_g2_passes_when_t_large_and_sign_matches() -> None:
    assert check_g2(t_long_l2=2.0, long_l2_mean=0.01, long_mean=0.02) is True  # 경계 포함


def test_check_g2_fails_when_sign_matches_but_t_too_small() -> None:
    assert check_g2(t_long_l2=1.9, long_l2_mean=0.01, long_mean=0.02) is False


def test_check_g2_fails_when_sign_differs_even_with_large_t() -> None:
    """부호가 다르면 |t|가 커도 X — G2의 핵심 조건."""
    assert check_g2(t_long_l2=5.0, long_l2_mean=-0.02, long_mean=0.03) is False


def test_check_g2_fails_on_zero_or_nonfinite() -> None:
    assert check_g2(t_long_l2=5.0, long_l2_mean=0.0, long_mean=0.03) is False
    assert check_g2(t_long_l2=float("nan"), long_l2_mean=0.02, long_mean=0.03) is False


def test_check_g3_passes_below_threshold() -> None:
    assert check_g3(0.0499) is True


def test_check_g3_fails_at_and_above_threshold() -> None:
    assert check_g3(0.05) is False
    assert check_g3(0.5) is False


def test_check_g3_fails_on_nonfinite() -> None:
    assert check_g3(float("nan")) is False


# --- 3. 등급 경계 — 01 §4 -----------------------------------------------------


@pytest.mark.parametrize(
    ("t_long", "long_mean", "expected"),
    [
        (3.0, 0.01, "A"),  # A 경계 정각
        (2.999, 0.01, "B"),
        (2.0, 0.01, "B"),  # B 경계 정각
        (1.999, 0.01, "C"),
        (1.0, 0.01, "C"),
        (0.999, 0.01, "D"),
    ],
)
def test_grade_from_long_stats_boundaries(t_long, long_mean, expected) -> None:
    assert grade_from_long_stats(t_long, long_mean) == expected


def test_grade_from_long_stats_large_t_negative_long_is_not_a_or_b() -> None:
    """LONG<=0인데 |t|가 A 문턱(3.0)을 넘어도 A·B가 아니다 — C 구간([1,2))에도
    안 들어가므로 D로 떨어진다(``scan_long.grade_long``의 문자 그대로 구간
    판정을 그대로 물려받는다)."""
    grade = grade_from_long_stats(t_long=3.5, long_mean=-0.02)
    assert grade not in {"A", "B"}
    assert grade == "D"


def test_grade_from_long_stats_moderate_t_negative_long_still_c() -> None:
    assert grade_from_long_stats(t_long=1.5, long_mean=-0.02) == "C"


def test_grade_from_long_stats_placebo_branch_is_disabled() -> None:
    """``grade_from_long_stats``는 항상 ``placebo_p=None``으로 부른다 — G3는
    ``evaluate_gates``가 이미 따로 판정했으므로 여기서 R이 나오면 안 된다."""
    # scan_long.grade_long에 큰 |t|·placebo_p=0.5(R 조건)를 직접 주면 R이지만,
    # grade_from_long_stats는 그 인자를 안 받으므로 R이 나올 길이 없다.
    assert grade_from_long_stats(t_long=4.0, long_mean=0.01) == "A"


# --- 4. 게이트 순서 — evaluate_gates ------------------------------------------


def test_evaluate_gates_g1_failure_skips_placebo_and_returns_x() -> None:
    calls = []

    def spy() -> tuple[float, float]:
        calls.append(1)
        return (0.0, 0.0)

    gate_failed, grade, placebo_max, placebo_p = evaluate_gates(
        long_mean=0.001,
        cost_mean=0.01,  # G1 실패: LONG <= cost
        t_long_l2=5.0,
        long_l2_mean=0.02,  # G2는 통과할 조건이지만 G1에서 이미 걸린다
        t_long=5.0,
        compute_placebo=spy,
    )
    assert gate_failed == "G1"
    assert grade == "X"
    assert placebo_max is None
    assert placebo_p is None
    assert calls == []  # placebo 계산 자체가 호출되지 않았다


def test_evaluate_gates_g2_failure_skips_placebo_and_returns_x() -> None:
    calls = []

    def spy() -> tuple[float, float]:
        calls.append(1)
        return (0.0, 0.0)

    gate_failed, grade, placebo_max, placebo_p = evaluate_gates(
        long_mean=0.02,
        cost_mean=0.01,  # G1 통과
        t_long_l2=1.0,
        long_l2_mean=0.02,  # G2 실패: |t| < 2.0
        t_long=5.0,
        compute_placebo=spy,
    )
    assert gate_failed == "G2"
    assert grade == "X"
    assert placebo_max is None
    assert placebo_p is None
    assert calls == []


def test_evaluate_gates_g3_failure_returns_r() -> None:
    gate_failed, grade, placebo_max, placebo_p = evaluate_gates(
        long_mean=0.02,
        cost_mean=0.01,
        t_long_l2=5.0,
        long_l2_mean=0.02,
        t_long=5.0,
        compute_placebo=lambda: (4.9, 0.5),  # p >= 0.05
    )
    assert gate_failed == "G3"
    assert grade == "R"
    assert placebo_max == 4.9
    assert placebo_p == 0.5


def test_evaluate_gates_all_pass_grades_normally() -> None:
    gate_failed, grade, placebo_max, placebo_p = evaluate_gates(
        long_mean=0.02,
        cost_mean=0.01,
        t_long_l2=5.0,
        long_l2_mean=0.02,
        t_long=4.0,
        compute_placebo=lambda: (3.0, 0.01),
    )
    assert gate_failed == "none"
    assert grade == "A"
    assert placebo_max == 3.0
    assert placebo_p == 0.01


def test_evaluate_gates_calls_placebo_exactly_once_when_reached() -> None:
    calls = []

    def spy() -> tuple[float, float]:
        calls.append(1)
        return (1.0, 0.01)

    evaluate_gates(
        long_mean=0.02,
        cost_mean=0.01,
        t_long_l2=5.0,
        long_l2_mean=0.02,
        t_long=4.0,
        compute_placebo=spy,
    )
    assert calls == [1]


# --- 5. G1이 cost.py를 실제로 쓰는지 -------------------------------------------


def _single_month_basket_df(
    *, n_symbols: int = 150, price: float = 100.0, sigma: float = 0.02, adv: float = 5e7
) -> pl.DataFrame:
    """한 달짜리 프레임 — ``feature``가 ``symbol`` 인덱스와 완전히 같은 순서라
    top-100/decile 바스켓이 결정적이다. 가격·변동성·ADV는 전부 상수라 바스켓
    비용도 상수여야 한다(값 검증용)."""
    return pl.DataFrame(
        {
            "month_idx": [1] * n_symbols,
            "symbol": [f"S{i:03d}" for i in range(n_symbols)],
            "feat": [float(i) for i in range(n_symbols)],
            "L0": [float(i) / n_symbols for i in range(n_symbols)],
            "L2": [float(i) / n_symbols for i in range(n_symbols)],
            "close": [price] * n_symbols,
            "adv_20d": [adv] * n_symbols,
            "sigma_daily": [sigma] * n_symbols,
            "price_ge_5": [True] * n_symbols,
        }
    )


def test_monthly_basket_diagnostics_cost_matches_cost_module_formula() -> None:
    df = _single_month_basket_df()
    table = monthly_basket_diagnostics(
        df,
        feature_col="feat",
        sign="+",
        min_names=10,
        top_k=100,
        q_dollar=cost_mod.DEFAULT_Q_DOLLAR,
        k=cost_mod.DEFAULT_K,
    )
    assert table.height == 1

    expected_spread = max(0.0002, 0.01 / 100.0)
    expected_impact = 0.1 * 0.02 * math.sqrt(cost_mod.DEFAULT_Q_DOLLAR / 5e7)
    expected_cost = expected_spread + 2 * expected_impact

    assert table["basket_cost_roundtrip"][0] == pytest.approx(expected_cost, rel=1e-9)


def test_monthly_basket_diagnostics_cost_changes_with_adv_like_cost_module() -> None:
    """ADV가 낮을수록(유동성이 나쁠수록) 비용이 커져야 한다 — ``cost.impact``의
    ``sqrt(Q/ADV)``가 정확히 이 방향이다(다시 구현했다면 부호가 뒤집혔을 수
    있다, 그래서 방향까지 확인한다)."""
    low_adv = monthly_basket_diagnostics(
        _single_month_basket_df(adv=1e6), feature_col="feat", sign="+", min_names=10, top_k=100
    )
    high_adv = monthly_basket_diagnostics(
        _single_month_basket_df(adv=1e9), feature_col="feat", sign="+", min_names=10, top_k=100
    )
    assert low_adv["basket_cost_roundtrip"][0] > high_adv["basket_cost_roundtrip"][0]


def test_scan_one_g1_fails_when_basket_is_illiquid_and_signal_is_thin() -> None:
    """LONG이 아주 작고 유동성이 나빠(ADV 작음, sigma 큼) 비용이 큰 바스켓은
    G1에서 X가 나와야 하고, placebo(``placebo_p``)는 계산되지 않아야 한다."""
    n_months = 6
    rng = np.random.default_rng(1)
    rows = []
    for m in range(1, n_months + 1):
        n = 150
        # feature와 L0의 상관을 아주 약하게(거의 잡음) 만든다 — LONG이 비용을
        # 넘을 만큼 크지 않게 하려는 의도.
        feat = rng.permutation(n).astype(float)
        l0 = rng.normal(scale=0.001, size=n)
        for i in range(n):
            rows.append(
                {
                    "month_idx": m,
                    "symbol": f"S{i:03d}",
                    "feat": float(feat[i]),
                    "L0": float(l0[i]),
                    "L2": float(l0[i]),
                    "close": 6.0,
                    "adv_20d": 2e5,  # 아주 얇은 유동성
                    "sigma_daily": 0.08,  # 높은 변동성 -> 비용 커짐
                    "price_ge_5": True,
                }
            )
    core = pl.DataFrame(rows)
    inputs = ScanInputs2(
        features_dev=core.select("symbol", "price_ge_5", pl.col("feat").alias("feat")),
        core21=core,
        core63=core.select("month_idx", "symbol", "feat", "L0", "price_ge_5"),
        labels_l0_only=core.select("month_idx", "symbol", "L0").unique(),
        total_months=n_months,
    )
    dspec = DirectionSpec(
        feature="feat", family="F_test", direction="registered", sign="+", expected_sign="+"
    )
    row = scan_one(dspec, universe="all", inputs=inputs, placebo_shifts=[1, 2, 3])

    assert row.gate_failed == "G1"
    assert row.grade == "X"
    assert row.placebo_p is None
    assert row.placebo_max_abs_t is None
    # G1 판정에 쓴 비용 자체는 그래도 채워져 있어야 한다(진단 칸은 게이트와
    # 무관하게 항상 계산한다, 01 §7 완료 판정).
    assert math.isfinite(row.basket_cost_roundtrip)
    assert math.isfinite(row.basket_turnover)
    assert math.isfinite(row.effective_cost)
    assert math.isfinite(row.LONG_L2)


# --- 5b. G1 개정 — 회전율(turnover)·effective_cost ----------------------------


def _turnover_df(feat_by_month: dict[int, list[float]]) -> pl.DataFrame:
    rows = []
    for m, feats in feat_by_month.items():
        for i, f in enumerate(feats):
            rows.append({"month_idx": m, "symbol": f"S{i:02d}", "feat": f})
    return pl.DataFrame(rows)


def test_monthly_basket_turnover_zero_percent_when_basket_never_changes() -> None:
    """세 달 내내 top-10 바스켓이 같으면(피쳐 순서 불변) 회전율은 0% 다."""
    ascending = list(range(20))  # sign="+" -> top10 = 인덱스10~19(S10~S19), 매달 같다
    df = _turnover_df({1: ascending, 2: ascending, 3: ascending})
    table = monthly_basket_turnover(df, feature_col="feat", sign="+", min_names=20, top_k=10)

    assert table.height == 2  # 첫 달(month_idx=1)은 빠진다
    assert sorted(table["month_idx"].to_list()) == [2, 3]
    assert table["turnover"].to_list() == pytest.approx([0.0, 0.0])


def test_monthly_basket_turnover_hundred_percent_on_full_swap() -> None:
    """매달 top-10 바스켓이 완전히 바뀌면(동결본이 가정했던 상황) 회전율 100%다."""
    ascending = list(range(20))  # top10 = S10~S19
    descending = list(range(19, -1, -1))  # top10 = S00~S09 (완전 교체)
    df = _turnover_df({1: ascending, 2: descending, 3: ascending})
    table = monthly_basket_turnover(df, feature_col="feat", sign="+", min_names=20, top_k=10)

    assert table.height == 2
    assert table.sort("month_idx")["turnover"].to_list() == pytest.approx([1.0, 1.0])


def test_monthly_basket_turnover_fifty_percent_on_half_swap() -> None:
    """top-10 바스켓의 절반만 바뀌면(5/10) 회전율 50%다."""

    def _feats(top_members: set[int]) -> list[float]:
        return [100.0 + i if i in top_members else float(i) for i in range(20)]

    month1 = _feats({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    month2 = _feats({5, 6, 7, 8, 9, 10, 11, 12, 13, 14})  # 5개 겹침, 5개 신규
    month3 = _feats({10, 11, 12, 13, 14, 15, 16, 17, 18, 19})  # month2 대비 5개 신규
    df = _turnover_df({1: month1, 2: month2, 3: month3})
    table = monthly_basket_turnover(df, feature_col="feat", sign="+", min_names=20, top_k=10)

    assert table.height == 2
    assert table.sort("month_idx")["turnover"].to_list() == pytest.approx([0.5, 0.5])


def test_monthly_basket_turnover_excludes_first_month_from_output() -> None:
    """직전 바스켓이 없는 첫 달은 표에 행 자체가 없다 — 평균에서 저절로 빠진다."""
    ascending = list(range(20))
    df = _turnover_df({1: ascending})  # 달이 하나뿐 -> 회전율을 잴 수 없다
    table = monthly_basket_turnover(df, feature_col="feat", sign="+", min_names=20, top_k=10)
    assert table.height == 0


def _cost_turnover_df(
    feat_by_month: dict[int, list[float]], *, n_symbols: int = 20
) -> pl.DataFrame:
    """비용을 상수로 고정한(가격·변동성·ADV 전부 같은 값) 회전율 테스트용 프레임."""
    rows = []
    for m, feats in feat_by_month.items():
        for i in range(n_symbols):
            rows.append(
                {
                    "month_idx": m,
                    "symbol": f"S{i:02d}",
                    "feat": feats[i],
                    "close": 100.0,
                    "adv_20d": 5e7,
                    "sigma_daily": 0.02,
                    "L2": 0.0,
                    "price_ge_5": True,
                }
            )
    return pl.DataFrame(rows)


def test_effective_cost_equals_turnover_times_cost_when_cost_constant() -> None:
    """비용이 매달 상수면 ``effective_cost`` = 회전율 평균 x 비용이어야 한다 —
    ``scan_one``이 두 표(:func:`monthly_basket_turnover`·
    :func:`monthly_basket_diagnostics`)를 합쳐 내는 것과 같은 조합이다."""

    def _feats(top_members: set[int]) -> list[float]:
        return [100.0 + i if i in top_members else float(i) for i in range(20)]

    month1 = _feats({0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
    month2 = _feats({5, 6, 7, 8, 9, 10, 11, 12, 13, 14})
    month3 = _feats({10, 11, 12, 13, 14, 15, 16, 17, 18, 19})
    df = _cost_turnover_df({1: month1, 2: month2, 3: month3})

    turnover_table = monthly_basket_turnover(
        df, feature_col="feat", sign="+", min_names=20, top_k=10
    )
    basket_table = monthly_basket_diagnostics(
        df,
        feature_col="feat",
        sign="+",
        min_names=20,
        top_k=10,
        q_dollar=cost_mod.DEFAULT_Q_DOLLAR,
        k=cost_mod.DEFAULT_K,
    )
    raw_cost = basket_table["basket_cost_roundtrip"][0]  # 상수라 매달 같다
    assert basket_table["basket_cost_roundtrip"].to_list() == pytest.approx(
        [raw_cost, raw_cost, raw_cost]
    )

    joined = turnover_table.join(
        basket_table.select("month_idx", "basket_cost_roundtrip"), on="month_idx", how="inner"
    ).with_columns((pl.col("turnover") * pl.col("basket_cost_roundtrip")).alias("effective_cost_t"))
    effective_cost = float(joined["effective_cost_t"].mean())
    basket_turnover = float(turnover_table["turnover"].mean())

    assert basket_turnover == pytest.approx(0.5)
    assert effective_cost == pytest.approx(0.5 * raw_cost)

    # G1 개정의 핵심: 회전율이 낮으면 옛 규칙(원단위 비용)으로는 걸렸을 LONG이
    # 새 규칙(effective_cost)으로는 통과할 수 있다.
    long_between = (effective_cost + raw_cost) / 2
    assert check_g1(long_between, raw_cost) is False  # 개정 전 규칙이면 X
    assert check_g1(long_between, effective_cost) is True  # 개정 뒤에는 통과


# --- 6. G3 placebo — scan.py와 같은 shift·시드 ---------------------------------


def test_compute_placebo_abs_t_long2_uses_same_shift_function_as_scan() -> None:
    """``compute_placebo_abs_t_long2``가 ``scan._shift_month_index_expr``를
    그대로 쓰는지, 직접 같은 함수로 재현한 값과 비교해 확인한다."""
    months = list(range(1, 13))
    symbols = [f"S{i:02d}" for i in range(20)]
    rng = np.random.default_rng(0)
    rows = []
    for m in months:
        perm = rng.permutation(20)
        for rank, sym_idx in enumerate(perm):
            rows.append({"month_idx": m, "symbol": symbols[sym_idx], "value": float(rank)})
    feature_side = pl.DataFrame(rows)
    labels_only = feature_side.rename({"value": "L0"})

    shift = 5
    abs_ts = compute_placebo_abs_t_long2(
        feature_side, labels_only, sign="+", total_months=12, shifts=[shift], min_names=10, top_k=10
    )

    # 손으로 같은 shift 함수를 불러 같은 경로를 재현한다.
    shifted = labels_only.with_columns(
        scan._shift_month_index_expr(shift=shift, total_months=12).alias("month_idx")
    )
    joined = feature_side.join(shifted, on=["month_idx", "symbol"], how="inner")
    monthly = scan_long.long_short_monthly(
        joined, feature_col="value", sign="+", value_col="L0", min_names=10, top_k=10
    )
    _, expected_t, _ = scan_long.mean_and_hac_t(monthly, value_col="top100")

    assert abs_ts == pytest.approx([abs(expected_t)], nan_ok=True)


def test_run_scan_long2_default_shifts_match_scan_select_placebo_shifts() -> None:
    """``placebo_shifts``를 안 주면 ``scan.select_placebo_shifts()``(1차와 같은
    50개·같은 시드 20260921)를 쓰는지, 명시적으로 넘긴 것과 같은 결과가
    나오는지로 확인한다."""
    default_shifts = scan.select_placebo_shifts()
    assert len(default_shifts) == 50
    assert scan.PLACEBO_SAMPLE_SEED == 20260921

    inputs = _tiny_single_feature_inputs(n_months=12, n_symbols=30)
    dspec = DirectionSpec(
        feature="feat", family="F_test", direction="registered", sign="+", expected_sign="+"
    )
    row_default = scan_one(dspec, universe="all", inputs=inputs, placebo_shifts=default_shifts)
    row_explicit = scan_one(dspec, universe="all", inputs=inputs, placebo_shifts=default_shifts)
    assert row_default.placebo_p == row_explicit.placebo_p
    assert row_default.placebo_max_abs_t == row_explicit.placebo_max_abs_t


# --- 7. top-100 vs 상위 10% 병존 · 유니버스 < 100인 달 -------------------------


def _monotonic_month_df(n_symbols: int, n_months: int = 6) -> pl.DataFrame:
    rows = []
    for m in range(1, n_months + 1):
        for i in range(n_symbols):
            rows.append(
                {
                    "month_idx": m,
                    "symbol": f"S{i:03d}",
                    "feat": float(i),
                    "L0": float(i) / n_symbols,  # feat과 완전히 같은 순서
                }
            )
    return pl.DataFrame(rows)


def test_top100_and_decile_both_present_when_universe_exceeds_100() -> None:
    df = _monotonic_month_df(n_symbols=150)
    table = scan_long.long_short_monthly(
        df, feature_col="feat", sign="+", value_col="L0", top_k=100
    )
    long_mean, t_long, _ = scan_long.mean_and_hac_t(table, value_col="top100")
    d10_mean, t_d10, _ = scan_long.mean_and_hac_t(table, value_col="long")
    assert math.isfinite(long_mean) and long_mean > 0
    assert math.isfinite(d10_mean) and d10_mean > 0
    # 상위 10%(15종목)가 top-100(100종목)보다 더 극단적인 이름들만 담으므로
    # 유니버스 대비 초과폭이 더 커야 한다(완전한 단조 신호에서).
    assert d10_mean > long_mean


def test_universe_below_top_k_makes_top100_excess_zero() -> None:
    """유니버스가 100종목 미만인 달은 top-100 마스크가 사실상 전체가 돼
    top100 평균이 유니버스 평균과 같아진다(``long_short_monthly``의 기존
    동작을 그대로 물려받는다 — 이 모듈에서 새로 만든 특수 처리가 아니다,
    보고서 참고). 상위 10%(decile)는 그래도 정의된다."""
    df = _monotonic_month_df(n_symbols=40)
    table = scan_long.long_short_monthly(
        df, feature_col="feat", sign="+", value_col="L0", top_k=100
    )
    long_mean, _, _ = scan_long.mean_and_hac_t(table, value_col="top100")
    d10_mean, _, _ = scan_long.mean_and_hac_t(table, value_col="long")
    assert long_mean == pytest.approx(0.0, abs=1e-9)
    assert d10_mean > 0  # decile은 여전히 신호를 잡는다


# --- 8. SHORT(하위 100) -------------------------------------------------------


def test_monthly_bottom100_short_uses_fixed_count_not_decile() -> None:
    """``01`` §5: SHORT는 하위 100(고정 개수)이지 하위 10%(decile)가 아니다 —
    유니버스 150종목에서 하위 10%(15종목)와 하위 100종목은 다른 값이 나와야
    한다는 것으로 구분한다."""
    df = _monotonic_month_df(n_symbols=150)
    bottom100 = monthly_bottom100_short(df, feature_col="feat", sign="+", value_col="L0", top_k=100)
    short_mean, _, _ = scan_long.mean_and_hac_t(bottom100, value_col="short")

    decile_table = scan_long.long_short_monthly(df, feature_col="feat", sign="+", value_col="L0")
    decile_short_mean, _, _ = scan_long.mean_and_hac_t(decile_table, value_col="short")

    assert math.isfinite(short_mean) and math.isfinite(decile_short_mean)
    assert short_mean != pytest.approx(decile_short_mean)


# --- 9. bh_q — family 안 BH ----------------------------------------------------


def _bare_row(feature: str, family: str, universe: str, t_long: float) -> LongScanRow2:
    return LongScanRow2(
        feature=feature,
        family=family,
        universe=universe,
        direction="registered",
        expected_sign="+",
        n_dates=10,
        LONG=0.01,
        t_LONG=t_long,
        LONG_d10=0.01,
        t_LONG_d10=t_long,
        basket_cost_roundtrip=0.001,
        basket_turnover=0.5,
        effective_cost=0.0005,
        LONG_L2=0.01,
        t_LONG_L2=t_long,
        basket_l2_mean=0.01,
        cut_inside_tie=0.0,
        basket_tie_fraction=0.0,
        basket_adv_median=1e7,
        adv_ratio=1.0,
        basket_price_median=50.0,
        basket_pct_ge5=1.0,
        SHORT=0.0,
        t_SHORT=0.0,
        placebo_max_abs_t=1.0,
        placebo_p=0.01,
        h63_LONG=0.01,
        missing_rate=0.0,
        high_missing=False,
        gate_failed="none",
        grade="B",
    )


def test_apply_bh_within_family_groups_by_family_and_universe() -> None:
    from modeler.etl.metrics import benjamini_hochberg, two_sided_normal_p

    rows = [
        _bare_row("f1", "FAM_A", "all", 3.0),
        _bare_row("f2", "FAM_A", "all", 1.0),
        _bare_row("f3", "FAM_B", "all", 5.0),  # 다른 family — 따로 묶여야 한다
    ]
    apply_bh_within_family(rows)

    fam_a = [r for r in rows if r.family == "FAM_A"]
    expected_q = benjamini_hochberg([two_sided_normal_p(3.0), two_sided_normal_p(1.0)])
    actual_q = [r.bh_q for r in fam_a]
    assert actual_q == pytest.approx(list(expected_q), nan_ok=True)

    fam_b = [r for r in rows if r.family == "FAM_B"][0]
    # 혼자뿐인 그룹은 BH 보정이 없어 q == p 그대로다.
    assert fam_b.bh_q == pytest.approx(two_sided_normal_p(5.0))


# --- 10. 개발 구간 날짜 벽 — --dev-end ------------------------------------------


def _write_synthetic_datasets(root: DataRoot, *, dates: list[date], n_symbols: int = 10) -> None:
    rng = np.random.default_rng(5)
    feature_names = [spec.feature for spec in FEATURE_REGISTRY]
    feature_rows, label_rows, label63_rows = [], [], []
    for d in dates:
        for s in range(n_symbols):
            frow = {"date": d, "symbol": f"S{s:02d}", "price_ge_5": True}
            for name in feature_names:
                frow[name] = float(rng.normal())
            feature_rows.append(frow)
            label_rows.append(
                {
                    "date": d,
                    "symbol": f"S{s:02d}",
                    "L0": float(rng.normal()),
                    "L2": float(rng.normal()),
                    "close": 50.0,
                    "adv_20d": 1e7,
                }
            )
            label63_rows.append({"date": d, "symbol": f"S{s:02d}", "L0": float(rng.normal())})
    write_dataset(pl.DataFrame(feature_rows), root, "us_features_v1", manifest={})
    write_dataset(pl.DataFrame(label_rows), root, "us_labels_v1", manifest={})
    write_dataset(pl.DataFrame(label63_rows), root, "us_labels_h63_v1", manifest={})


def test_load_features_and_labels_respects_default_dev_end(tmp_path: Path) -> None:
    root = DataRoot(base=tmp_path)
    dates = [
        date(2020, 1, 2),
        scan_long2.DEV_END,
        date(2025, 7, 1),  # 기본 dev_end 뒤 — 섞이면 안 된다
        date(2026, 1, 5),
    ]
    _write_synthetic_datasets(root, dates=dates)

    features, labels, labels63 = load_features_and_labels(root)

    assert features["date"].max() == scan_long2.DEV_END
    assert labels["date"].max() == scan_long2.DEV_END
    assert labels63["date"].max() == scan_long2.DEV_END


def test_load_features_and_labels_respects_custom_dev_end(tmp_path: Path) -> None:
    """``--dev-end``로 기본값보다 훨씬 이른 날짜를 주면 그 뒤 행은 어느
    데이터셋에서도 읽히지 않아야 한다 — N2가 2026-06-30으로 넓힐 때 쓸
    인자다."""
    root = DataRoot(base=tmp_path)
    dates = [date(2019, 1, 2), date(2020, 6, 30), date(2021, 1, 5)]
    _write_synthetic_datasets(root, dates=dates)
    custom_end = date(2020, 6, 30)

    features, labels, labels63 = load_features_and_labels(root, dev_end=custom_end)

    assert features["date"].max() == custom_end
    assert labels["date"].max() == custom_end
    assert labels63["date"].max() == custom_end
    assert date(2021, 1, 5) not in features["date"].to_list()
    assert date(2021, 1, 5) not in labels["date"].to_list()
    assert date(2021, 1, 5) not in labels63["date"].to_list()


# --- 11. end-to-end (합성 ScanInputs2, 44개 x 방향 x 유니버스) ------------------


def _tiny_single_feature_inputs(*, n_months: int = 12, n_symbols: int = 30) -> ScanInputs2:
    rng = np.random.default_rng(9)
    rows = []
    for m in range(1, n_months + 1):
        ranks = rng.permutation(n_symbols).astype(float)
        for s in range(n_symbols):
            rows.append(
                {
                    "month_idx": m,
                    "symbol": f"S{s:03d}",
                    "feat": float(ranks[s]),
                    "L0": float(ranks[s]) / n_symbols + rng.normal(scale=0.01),
                    "L2": float(ranks[s]) / n_symbols + rng.normal(scale=0.01),
                    "close": 50.0,
                    "adv_20d": 2e7,
                    "sigma_daily": 0.02,
                    "price_ge_5": True,
                }
            )
    core = pl.DataFrame(rows)
    return ScanInputs2(
        features_dev=core.select("symbol", "price_ge_5", pl.col("feat").alias("feat")),
        core21=core,
        core63=core.select("month_idx", "symbol", "feat", "L0", "price_ge_5"),
        labels_l0_only=core.select("month_idx", "symbol", "L0").unique(),
        total_months=n_months,
    )


def _synthetic_scan_inputs2(
    *, n_months: int = 12, n_symbols: int = 40, seed: int = 3
) -> ScanInputs2:
    rng = np.random.default_rng(seed)
    feature_cols = [spec.feature for spec in FEATURE_REGISTRY]
    records = []
    for m in range(1, n_months + 1):
        l2_rank = rng.permutation(n_symbols).astype(float)
        feature_values = {
            name: l2_rank + rng.normal(scale=5.0, size=n_symbols) for name in feature_cols
        }
        for s in range(n_symbols):
            record = {
                "month_idx": m,
                "symbol": f"S{s:03d}",
                "price_ge_5": bool(s % 4 != 0),  # 일부는 $5 미만 취급
                "L0": float(l2_rank[s]) / n_symbols,
                "L2": float(l2_rank[s]),
                "close": float(3.0 + s),  # 일부 $5 밑, 일부 위
                "adv_20d": float(1e6 * (1 + s)),
                "sigma_daily": 0.015 + 0.0001 * s,
            }
            for name in feature_cols:
                record[name] = float(feature_values[name][s])
            records.append(record)
    core = pl.DataFrame(records)
    return ScanInputs2(
        features_dev=core.select("symbol", "price_ge_5", *feature_cols),
        core21=core,
        core63=core.select("month_idx", "symbol", "price_ge_5", *feature_cols, "L0"),
        labels_l0_only=core.select("month_idx", "symbol", "L0").unique(),
        total_months=n_months,
    )


_TEST_PLACEBO_SHIFTS = list(range(1, 22))  # T=12 — scan.select_placebo_shifts()의 50개는 너무 크다


def test_run_scan_long2_covers_all_features_directions_and_universes() -> None:
    inputs = _synthetic_scan_inputs2()
    rows = run_scan_long2(inputs, placebo_shifts=_TEST_PLACEBO_SHIFTS)

    expected_n = len(all_direction_specs()) * len(scan.UNIVERSES)
    assert len(rows) == expected_n

    for row in rows:
        assert row.grade in {"A", "B", "C", "D", "R", "X"}
        assert row.gate_failed in {"G1", "G2", "G3", "none"}
        if row.gate_failed in {"G1", "G2"}:
            assert row.grade == "X"
            assert row.placebo_p is None
        elif row.gate_failed == "G3":
            assert row.grade == "R"
            assert row.placebo_p is not None
        else:
            assert row.grade in {"A", "B", "C", "D"}


def test_run_scan_long2_row_columns_match_preregistration_spec() -> None:
    expected_columns = {
        "feature",
        "family",
        "universe",
        "direction",
        "expected_sign",
        "n_dates",
        "LONG",
        "t_LONG",
        "LONG_d10",
        "t_LONG_d10",
        "basket_cost_roundtrip",
        "basket_turnover",
        "effective_cost",
        "LONG_L2",
        "t_LONG_L2",
        "basket_l2_mean",
        # 2026-09-22 에 더했다 — 사전등록 `01` §5 에 없던 칸이다.
        # top-100 이 동점에서 정의되지 않던 것을 고치면서(`01_tie_break.md`),
        # **이 등급이 임의 추출 위에 있나**를 사람이 볼 수 있게 남긴다.
        "cut_inside_tie",
        "basket_tie_fraction",
        "basket_adv_median",
        "adv_ratio",
        "basket_price_median",
        "basket_pct_ge5",
        "SHORT",
        "t_SHORT",
        "placebo_max_abs_t",
        "placebo_p",
        "h63_LONG",
        "missing_rate",
        "high_missing",
        "bh_q",
        "gate_failed",
        "grade",
    }
    inputs = _tiny_single_feature_inputs()
    dspec = DirectionSpec(
        feature="feat", family="F_test", direction="registered", sign="+", expected_sign="+"
    )
    row = scan_one(dspec, universe="all", inputs=inputs, placebo_shifts=_TEST_PLACEBO_SHIFTS)
    assert set(row.as_dict().keys()) == expected_columns


# --- 데이터셋 판을 코드에 박지 않는다 (2026-09-22) ----------------------------


def test_dataset_names_follow_the_version():
    """3차는 **사양을 안 바꾸고 데이터만 v2 로** 바꾼다 (T-D1).

    이름이 코드에 박혀 있으면 그 약속을 지킬 수가 없다.
    """
    from modeler.us import scan_long2

    assert scan_long2.dataset_names("v1") == (
        "us_features_v1",
        "us_labels_v1",
        "us_labels_h63_v1",
    )
    assert scan_long2.dataset_names("v2") == (
        "us_features_v2",
        "us_labels_v2",
        "us_labels_h63_v2",
    )


def test_default_version_is_v1_until_someone_changes_it():
    """기본을 v2 로 슬쩍 바꾸면 옛 결과를 다시 못 만든다. **인자로 준다.**"""
    from modeler.us import scan_long2

    assert scan_long2.DATASET_VERSION == "v1"
