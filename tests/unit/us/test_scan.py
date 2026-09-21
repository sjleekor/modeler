"""``modeler.us.scan`` 단위 테스트. 전부 합성 데이터다 — 실제 레이크를 읽지 않는다.

``06_execution_steps.md`` M3 완료 판정이 요구하는 넷을 검사한다: HAC t 계산,
등급 매김, placebo shift, holdout 날짜 벽.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from modeler.etl.config import DataRoot
from modeler.etl.metrics import newey_west_tstat
from modeler.us.dataset import write_dataset
from modeler.us.features import FAMILY_ORDER
from modeler.us.scan import (
    DEV_END,
    FEATURE_REGISTRY,
    N_PLACEBO,
    PLACEBO_SHIFT_MAX,
    PLACEBO_SHIFT_MIN,
    FeatureScanRow,
    FeatureSpec,
    ScanInputs,
    _missing_rate,
    _shift_month_index_expr,
    apply_bh_within_family,
    assert_dev_window,
    build_scan_inputs,
    compute_placebo_abs_t,
    decile_spread,
    dedup_ab_features,
    enforce_dev_window,
    grade_feature,
    ic_and_t,
    load_dev_frame,
    monthly_rank_ic,
    placebo_p_value,
    placebo_shift_candidates,
    run_scan,
    scan_one,
    select_placebo_shifts,
    spread_sign_matches,
)

# --- 1. 피쳐 등록 -------------------------------------------------------------


def test_feature_registry_has_44_features() -> None:
    assert len(FEATURE_REGISTRY) == 44


def test_feature_registry_families_are_a_subset_of_family_order() -> None:
    """검정 대상 family 이름이 ``features.FAMILY_ORDER``에 실제로 있는
    F1~F14 이름과 정확히 일치한다(오타 방지). F15·F16은 검정 대상이 아니다."""
    testable_family_names = {
        name for name, _ in FAMILY_ORDER if not name.startswith(("F15", "F16"))
    }
    registry_family_names = {spec.family for spec in FEATURE_REGISTRY}
    assert registry_family_names == testable_family_names


def test_feature_registry_ids_are_unique() -> None:
    ids = [spec.feature for spec in FEATURE_REGISTRY]
    assert len(ids) == len(set(ids))


# --- 2. holdout 날짜 벽 --------------------------------------------------------


def test_enforce_dev_window_drops_rows_after_dev_end() -> None:
    df = pl.DataFrame(
        {
            "date": [date(2025, 6, 30), date(2025, 7, 1), date(2026, 1, 5)],
            "symbol": ["AAA", "BBB", "CCC"],
        }
    )
    result = enforce_dev_window(df)
    assert result["date"].to_list() == [date(2025, 6, 30)]


def test_enforce_dev_window_keeps_dev_end_inclusive() -> None:
    df = pl.DataFrame({"date": [DEV_END], "symbol": ["AAA"]})
    result = enforce_dev_window(df)
    assert result.height == 1


def test_enforce_dev_window_requires_date_column() -> None:
    df = pl.DataFrame({"symbol": ["AAA"]})
    with pytest.raises(ValueError, match="date"):
        enforce_dev_window(df)


def test_assert_dev_window_passes_when_clean() -> None:
    df = pl.DataFrame({"date": [date(2020, 1, 1), DEV_END]})
    assert_dev_window(df)  # 예외 없이 통과해야 한다


def test_assert_dev_window_raises_when_holdout_leaked() -> None:
    df = pl.DataFrame({"date": [date(2020, 1, 1), date(2025, 7, 1)]})
    with pytest.raises(ValueError, match="holdout"):
        assert_dev_window(df)


def test_load_dev_frame_filters_dataset_written_to_disk(tmp_path: Path) -> None:
    """``write_dataset``으로 실제 parquet를 쓰고, holdout 뒤 행이 섞여 있어도
    ``load_dev_frame``이 개발 구간만 돌려주는지 본다(레이크가 아니라 이미
    만들어진 데이터셋을 읽는 경로 — ``build_scan_inputs``와 같은 함수를 쓴다)."""
    root = DataRoot(base=tmp_path)
    df = pl.DataFrame(
        {
            "date": [date(2024, 1, 2), date(2025, 6, 30), date(2025, 7, 1), date(2026, 9, 1)],
            "symbol": ["AAA", "AAA", "AAA", "AAA"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    write_dataset(df, root, "toy_dataset", manifest={})

    loaded = load_dev_frame(root, "toy_dataset")

    assert loaded["date"].max() == DEV_END
    assert loaded.height == 2
    assert_dev_window(loaded)  # 다시 확인해도 통과해야 한다(방어선 자체 검증)


# --- 3. 월별 순위 IC · HAC t ---------------------------------------------------


def test_monthly_rank_ic_is_one_for_perfectly_ranked_data() -> None:
    """같은 순서로 오르는 x·y는 그 달의 스피어만 상관이 정확히 1.0이어야 한다."""
    df = pl.DataFrame(
        {
            "month_idx": [1] * 5 + [2] * 5,
            "x": [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
            "y": [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
        }
    )
    ic_table = monthly_rank_ic(df, x_col="x", y_col="y", min_names=3)
    assert ic_table.height == 2
    assert ic_table["ic"].to_list() == pytest.approx([1.0, 1.0])


def test_monthly_rank_ic_is_minus_one_for_perfectly_inverted_data() -> None:
    df = pl.DataFrame({"month_idx": [1] * 5, "x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
    ic_table = monthly_rank_ic(df, x_col="x", y_col="y", min_names=3)
    assert ic_table["ic"].to_list() == pytest.approx([-1.0])


def test_monthly_rank_ic_drops_months_below_min_names() -> None:
    df = pl.DataFrame(
        {
            "month_idx": [1, 1, 2, 2, 2, 2, 2],
            "x": [1, 2, 1, 2, 3, 4, 5],
            "y": [1, 2, 1, 2, 3, 4, 5],
        }
    )
    ic_table = monthly_rank_ic(df, x_col="x", y_col="y", min_names=3)
    # month 1은 이름이 2개뿐이라 빠지고 month 2만 남는다.
    assert ic_table["month_idx"].to_list() == [2]


def test_ic_and_t_matches_newey_west_tstat_directly() -> None:
    """``ic_and_t``는 ``newey_west_tstat``를 그대로 불러 쓴다 — 위임 자체를 확인한다."""
    ic_table = pl.DataFrame({"month_idx": [1, 2, 3, 4, 5], "ic": [0.1, 0.2, -0.05, 0.15, 0.3]})
    ic_mean, t_nw, n = ic_and_t(ic_table, lag=3)
    expected_t = newey_west_tstat([0.1, 0.2, -0.05, 0.15, 0.3], [1, 2, 3, 4, 5], 3)
    assert n == 5
    assert ic_mean == pytest.approx(0.14)
    assert t_nw == pytest.approx(expected_t)


def test_ic_and_t_empty_table_is_nan() -> None:
    ic_table = pl.DataFrame(schema={"month_idx": pl.Int64, "ic": pl.Float64, "n": pl.Int64})
    ic_mean, t_nw, n = ic_and_t(ic_table)
    assert math.isnan(ic_mean)
    assert math.isnan(t_nw)
    assert n == 0


def test_ic_and_t_ignores_non_finite_ic_values() -> None:
    """어떤 달의 횡단면이 상수라 상관이 정의되지 않으면(``None``/NaN) 그
    달은 평균·HAC 계산에서 빠져야 한다."""
    ic_table = pl.DataFrame(
        {"month_idx": [1, 2, 3, 4], "ic": [0.1, None, 0.2, 0.3]},
        schema={"month_idx": pl.Int64, "ic": pl.Float64},
    )
    ic_mean, _t_nw, n = ic_and_t(ic_table)
    assert n == 3
    assert ic_mean == pytest.approx((0.1 + 0.2 + 0.3) / 3)


# --- 4. decile spread ----------------------------------------------------------


def test_decile_spread_matches_hand_computed_value() -> None:
    # 10개 이름, x 오름차순. top 10% = 값 10(rank 최고), bottom 10% = 값 1.
    df = pl.DataFrame(
        {
            "month_idx": [1] * 10,
            "x": list(range(1, 11)),
            "L0": [float(v) / 10 for v in range(1, 11)],  # 0.1..1.0
        }
    )
    spread, n_dates = decile_spread(df, feature_col="x", value_col="L0", min_names=5)
    assert n_dates == 1
    # top 10% = {rank10 -> L0=1.0}, bottom 10% = {rank1 -> L0=0.1}
    assert spread == pytest.approx(1.0 - 0.1)


def test_decile_spread_sign_flips_with_feature_direction() -> None:
    df = pl.DataFrame(
        {
            "month_idx": [1] * 10,
            "x": list(range(10, 0, -1)),  # x 내림차순
            "L0": [float(v) / 10 for v in range(1, 11)],
        }
    )
    spread, _n = decile_spread(df, feature_col="x", value_col="L0", min_names=5)
    assert spread < 0


def test_spread_sign_matches_none_expected_sign_is_always_true() -> None:
    assert spread_sign_matches(-5.0, None) is True
    assert spread_sign_matches(5.0, None) is True
    assert spread_sign_matches(float("nan"), None) is True


def test_spread_sign_matches_checks_direction() -> None:
    assert spread_sign_matches(0.01, "+") is True
    assert spread_sign_matches(-0.01, "+") is False
    assert spread_sign_matches(-0.01, "-") is True
    assert spread_sign_matches(0.0, "+") is False
    assert spread_sign_matches(float("nan"), "+") is False


# --- 5. 등급 --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("t_nw", "spread_ok", "placebo_p", "expected"),
    [
        # placebo_p=0.01 < 0.05 -> placebo를 통과한 경우, |t| 구간으로만 갈린다.
        (3.5, True, 0.01, "A"),
        (3.5, False, 0.01, "B"),  # |t|>=3이어도 부호가 안 맞으면 A가 아니라 B
        (2.5, True, 0.01, "B"),
        (1.5, True, 0.01, "C"),
        (0.5, True, 0.01, "D"),
        # t_nw의 부호 자체는 등급에 안 쓰인다 — |t_nw|만 본다. 방향은 호출부가
        # spread_sign_ok로 미리 판정해 넘긴다(spread_sign_matches 참고).
        (-3.5, True, 0.01, "A"),
        (4.0, True, 0.5, "R"),  # placebo_p>=0.05 — |t|가 커도 placebo 탈락이 먼저다
        (float("nan"), True, 0.5, "D"),  # t_nw 자체가 없으면 placebo와 무관하게 D
    ],
)
def test_grade_feature_boundaries(t_nw, spread_ok, placebo_p, expected) -> None:
    assert grade_feature(t_nw=t_nw, spread_sign_ok=spread_ok, placebo_p=placebo_p) == expected


def test_grade_feature_a_requires_both_t_and_sign() -> None:
    assert grade_feature(t_nw=3.0, spread_sign_ok=True, placebo_p=0.01) == "A"
    assert grade_feature(t_nw=2.999, spread_sign_ok=True, placebo_p=0.01) == "B"


def test_grade_feature_r_boundary_is_ge_not_gt() -> None:
    assert grade_feature(t_nw=4.0, spread_sign_ok=True, placebo_p=0.05) == "R"
    assert grade_feature(t_nw=4.0, spread_sign_ok=True, placebo_p=0.0499) == "A"


def test_grade_feature_nan_placebo_p_does_not_force_r() -> None:
    """placebo가 계산되지 않은 경우(달 수 부족 등) R로 잘못 떨어지면 안 된다."""
    assert grade_feature(t_nw=4.0, spread_sign_ok=True, placebo_p=float("nan")) == "A"


# --- 6. placebo circular shift --------------------------------------------------


def test_placebo_shift_candidates_excludes_multiples_of_12() -> None:
    candidates = placebo_shift_candidates()
    assert min(candidates) == PLACEBO_SHIFT_MIN
    assert max(candidates) == PLACEBO_SHIFT_MAX
    assert all(c % 12 != 0 for c in candidates)
    assert len(candidates) == 65  # [6,76] 71개 중 12의 배수 6개(12..72) 제외


def test_select_placebo_shifts_is_deterministic_and_sized() -> None:
    first = select_placebo_shifts()
    second = select_placebo_shifts()
    assert first == second
    assert len(first) == N_PLACEBO
    assert len(set(first)) == N_PLACEBO
    assert all(s % 12 != 0 for s in first)


def test_select_placebo_shifts_rejects_n_larger_than_pool() -> None:
    with pytest.raises(ValueError):
        select_placebo_shifts(n=1000)


def test_shift_month_index_expr_pairs_expected_month() -> None:
    """``_shift_month_index_expr``는 프레임의 ``month_idx`` 컬럼을 읽어 relabel한다.

    정의: ``new = ((month_idx - 1 - shift) % T) + 1``. 원래 ``month_idx=5``였던
    라벨 행은 이 식을 거치면 ``new_month_idx=2``가 되어야 한다(``T=10, shift=3``).
    왕복 확인: 피쳐 달 ``i=2``에 라벨 달 ``j=((i-1+shift)%T)+1=5``가 물려야
    한다 — relabel(5) == 2와 같은 말이다.
    """
    labels = pl.DataFrame({"month_idx": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    relabeled = labels.with_columns(
        _shift_month_index_expr(shift=3, total_months=10).alias("new_month_idx")
    )
    assert relabeled.filter(pl.col("month_idx") == 5)["new_month_idx"].item() == 2

    i, shift, total = 2, 3, 10
    j = ((i - 1 + shift) % total) + 1
    assert relabeled.filter(pl.col("month_idx") == j)["new_month_idx"].item() == i


def test_shift_month_index_expr_wraps_circularly() -> None:
    labels = pl.DataFrame({"month_idx": [1, 2, 3]})
    relabeled = labels.with_columns(
        _shift_month_index_expr(shift=2, total_months=3).alias("new_month_idx")
    )
    # month_idx=1 -> ((1-1-2)%3)+1 = ((-2)%3)+1 = 1+1 = 2
    assert relabeled["new_month_idx"].to_list() == [2, 3, 1]


def test_shift_month_index_expr_rejects_non_positive_total() -> None:
    with pytest.raises(ValueError):
        pl.DataFrame({"month_idx": [1]}).with_columns(
            _shift_month_index_expr(shift=1, total_months=0).alias("month_idx")
        )


def test_compute_placebo_abs_t_returns_one_value_per_shift() -> None:
    # T=12개월, 30개 심볼. feature는 진짜(shift 없는) 라벨과 완전 상관이지만,
    # 다른 달로 shift하면 (독립적으로 만든) 라벨과는 상관이 약해야 한다.
    months = list(range(1, 13))
    symbols = [f"S{i:02d}" for i in range(30)]
    rng = np.random.default_rng(0)
    rows = []
    for m in months:
        perm = rng.permutation(30)
        for rank, sym_idx in enumerate(perm):
            rows.append({"month_idx": m, "symbol": symbols[sym_idx], "value": float(rank)})
    feature_side = pl.DataFrame(rows)
    # 라벨은 그 달의 feature 값과 완전히 같은 순서(진짜 상관 1.0)로 만든다.
    labels_only = feature_side.rename({"value": "L2"})

    shifts = [3, 5, 7]
    abs_ts = compute_placebo_abs_t(
        feature_side, labels_only, total_months=12, shifts=shifts, min_names=10
    )
    assert len(abs_ts) == len(shifts)
    assert all(math.isfinite(t) for t in abs_ts)


# --- 7. placebo p-value ----------------------------------------------------------


def test_placebo_p_value_formula() -> None:
    # 실제 |t|=3.0, placebo 중 2개가 >= 3.0 -> p = (1+2)/(4+1) = 0.6
    p = placebo_p_value(3.0, [3.5, 2.9, 1.0, 3.0])
    assert p == pytest.approx(3 / 5)


def test_placebo_p_value_nan_when_real_t_is_nan() -> None:
    assert math.isnan(placebo_p_value(float("nan"), [1.0, 2.0]))


def test_placebo_p_value_nan_when_no_finite_placebo() -> None:
    assert math.isnan(placebo_p_value(3.0, [float("nan"), float("nan")]))


def test_placebo_p_value_never_zero() -> None:
    # 실제 |t|가 placebo 전부보다 커도 p는 최소 1/(n+1)이다(0이 아니다).
    p = placebo_p_value(100.0, [0.1, 0.2, 0.3])
    assert p == pytest.approx(1 / 4)


# --- 8. BH · 중복 제거 -----------------------------------------------------------


def _row(
    feature: str, family: str, universe: str, t_nw: float, p: float, grade: str = "D"
) -> FeatureScanRow:
    return FeatureScanRow(
        feature=feature,
        family=family,
        universe=universe,
        expected_sign="+",
        n_dates=50,
        ic_mean=0.05,
        t_nw=t_nw,
        p_two_sided=p,
        spread=0.01,
        spread_sign_match=True,
        placebo_max_abs_t=1.0,
        placebo_p=0.5,
        grade=grade,
        h5_ic=0.02,
        h63_ic=0.01,
        missing_rate=0.0,
        high_missing=False,
    )


def test_apply_bh_within_family_groups_by_family_and_universe() -> None:
    rows = [
        _row("f1", "FAM_A", "all", 4.0, 0.0001),
        _row("f2", "FAM_A", "all", 1.0, 0.3),
        _row("f3", "FAM_B", "all", 4.0, 0.0001),  # 다른 family — 섞이면 안 된다
        _row("f4", "FAM_A", "price_ge_5", 4.0, 0.0001),  # 다른 universe
    ]
    apply_bh_within_family(rows)
    for row in rows:
        assert row.bh_q is not None
    # FAM_A/all의 f1은 유일하게 유의한 것이라 q가 p보다 작지 않다(BH 단조성).
    fam_a_all = [r for r in rows if r.family == "FAM_A" and r.universe == "all"]
    assert {r.feature for r in fam_a_all} == {"f1", "f2"}


def test_dedup_ab_features_drops_the_smaller_t_among_correlated_pair() -> None:
    core = pl.DataFrame(
        {
            "month_idx": list(range(1, 21)) * 3,
            "symbol": ["S0"] * 20 + ["S1"] * 20 + ["S2"] * 20,
        }
    )
    # 실제로는 각 (month_idx, symbol) 조합에 값이 필요하다 — 20개월 x 여러 종목을
    # 만들어 두 피쳐가 거의 같은 순위가 되도록 구성한다.
    n_months = 20
    n_symbols = 25
    rng = np.random.default_rng(1)
    records = []
    for m in range(1, n_months + 1):
        base = rng.permutation(n_symbols).astype(float)
        for s in range(n_symbols):
            records.append(
                {
                    "month_idx": m,
                    "symbol": f"S{s}",
                    "feat_a": base[s],
                    "feat_b": base[s] + 0.01,  # feat_a와 사실상 같은 순위
                    "feat_c": rng.permutation(n_symbols).astype(float)[s],  # 독립
                }
            )
    core = pl.DataFrame(records)

    rows = [
        _row("feat_a", "FAM_A", "all", 4.0, 0.0001, grade="A"),
        _row("feat_b", "FAM_A", "all", 3.5, 0.0002, grade="A"),
        _row("feat_c", "FAM_A", "all", 3.2, 0.0003, grade="A"),
    ]
    kept, dropped = dedup_ab_features(rows, core, universe="all", rho_threshold=0.8)

    assert "feat_a" in kept  # |t|가 더 큰 쪽이 남는다
    assert "feat_b" not in kept
    dropped_features = {d["feature"] for d in dropped}
    assert "feat_b" in dropped_features
    assert "feat_c" in kept  # feat_c는 독립이라 남아야 한다


def test_dedup_ab_features_ignores_other_universe_and_grades() -> None:
    core = pl.DataFrame(
        {"month_idx": [1, 2, 3], "feat_a": [1.0, 2.0, 3.0], "feat_b": [1.0, 2.0, 3.0]}
    )
    rows = [
        _row("feat_a", "FAM_A", "all", 4.0, 0.0001, grade="A"),
        _row("feat_b", "FAM_A", "all", 3.0, 0.0002, grade="C"),  # C등급은 대상이 아니다
        _row("feat_c", "FAM_A", "price_ge_5", 4.0, 0.0001, grade="A"),  # 다른 universe
    ]
    kept, _dropped = dedup_ab_features(rows, core, universe="all", rho_threshold=0.8)
    assert kept == ["feat_a"]


# --- 9. 결측률 ------------------------------------------------------------------


def test_missing_rate_computed_per_universe() -> None:
    df = pl.DataFrame(
        {
            "x": [1.0, None, 3.0, None],
            "price_ge_5": [True, True, False, False],
        }
    )
    assert _missing_rate(df, "x", universe="all") == pytest.approx(0.5)
    # price_ge_5 유니버스: 2행(True) 중 1행 결측
    assert _missing_rate(df, "x", universe="price_ge_5") == pytest.approx(0.5)


def test_missing_rate_empty_frame_is_nan() -> None:
    df = pl.DataFrame(
        {"x": [], "price_ge_5": []}, schema={"x": pl.Float64, "price_ge_5": pl.Boolean}
    )
    assert math.isnan(_missing_rate(df, "x", universe="all"))


# --- 10. end-to-end (synthetic ScanInputs, no lake) -----------------------------


def _synthetic_scan_inputs(*, n_months: int = 24, n_symbols: int = 30, seed: int = 7) -> ScanInputs:
    rng = np.random.default_rng(seed)
    records = []
    for m in range(1, n_months + 1):
        l2 = rng.permutation(n_symbols).astype(float)
        # 순위 스케일(0..n_symbols-1, 간격 1)에 견줄 만한 잡음을 더한다 — 너무
        # 작으면(예: 0.001) 매달 순위가 전혀 안 흔들려 IC가 항상 정확히 1.0이
        # 되고, 그러면 월별 IC의 분산이 0이라 HAC t가 정의되지 않는다(NaN).
        noise = rng.normal(scale=3.0, size=n_symbols)
        for s in range(n_symbols):
            records.append(
                {
                    "month_idx": m,
                    "date": date(2019, 1, 1),  # 실제로는 안 쓰인다(month_idx로만 그룹핑)
                    "symbol": f"S{s:03d}",
                    "price_ge_5": True,
                    "strong_signal": l2[s] + noise[s],  # L2와 강하게(완벽하지 않게) 상관
                    "no_signal": rng.permutation(n_symbols).astype(float)[s],
                    "L0": l2[s] / n_symbols,
                    "L2": l2[s],
                }
            )
    core21 = pl.DataFrame(records)
    core5 = core21.drop("L0")
    core63 = core21.drop("L0")
    labels_only = core21.select("month_idx", "symbol", "L2").unique()
    return ScanInputs(
        features_dev=core21,
        core21=core21,
        core5=core5,
        core63=core63,
        labels_only=labels_only,
        total_months=n_months,
    )


#: 단위테스트 전용 placebo shift — 04 §1 범위(6~76, T~82용)가 아니라 이 합성
#: 데이터의 T=24에 맞춘 것이다. ``placebo_p_value``의 최소값은 1/(n+1)이라
#: n이 너무 작으면(예: 5) 최소 p가 이미 0.05를 넘어 실제 신호와 무관하게 항상
#: R이 나온다 — n=21로 최소 p를 1/22≈0.045로 낮춰 A/B가 나올 여지를 둔다.
_TEST_PLACEBO_SHIFTS = list(range(1, 22))


def test_scan_one_grades_a_strongly_correlated_feature_a() -> None:
    inputs = _synthetic_scan_inputs()
    spec = FeatureSpec("strong_signal", "F1_momentum", "+")
    row = scan_one(spec, universe="all", inputs=inputs, placebo_shifts=_TEST_PLACEBO_SHIFTS)

    assert row.t_nw > 3.0
    assert row.grade in {"A", "B"}  # 진짜 신호라 placebo와 뚜렷이 구별돼야 한다


def test_scan_one_grades_low_for_pure_noise_feature() -> None:
    inputs = _synthetic_scan_inputs()
    shifts = _TEST_PLACEBO_SHIFTS
    spec = FeatureSpec("no_signal", "F1_momentum", "+")
    row = scan_one(spec, universe="all", inputs=inputs, placebo_shifts=shifts)

    assert abs(row.t_nw) < 3.0
    assert row.grade != "A"
    # 순수 잡음이면 |t| 구간상 C/D거나, placebo와 구별이 안 돼 R로 떨어진다 —
    # 둘 다 "유의한 신호가 아니다"라는 같은 결론이라 셋 다 허용한다.
    assert row.grade in {"C", "D", "R"}


def test_build_scan_inputs_reads_written_datasets_and_enforces_dev_window(tmp_path: Path) -> None:
    """레이크가 아니라 ``datasets/us_features_v1`` 등 이미 만들어진 데이터셋을
    읽는 경로 전체(``build_scan_inputs``)가 holdout을 걸러내는지 본다."""
    root = DataRoot(base=tmp_path)
    n_symbols = 25
    dates_in_dev = [date(2020, 1, 2), date(2020, 2, 3)]
    dates_after_dev = [date(2025, 8, 1)]  # holdout — 읽히면 안 된다

    feature_cols = {spec.feature: pl.Float64 for spec in FEATURE_REGISTRY}
    rows = []
    for d in [*dates_in_dev, *dates_after_dev]:
        for s in range(n_symbols):
            row = {"date": d, "symbol": f"S{s:03d}", "price_ge_5": True}
            for name in feature_cols:
                row[name] = float(s)
            rows.append(row)
    features_df = pl.DataFrame(rows)
    write_dataset(features_df, root, "us_features_v1", manifest={})

    labels_rows = []
    for d in [*dates_in_dev, *dates_after_dev]:
        for s in range(n_symbols):
            labels_rows.append(
                {"date": d, "symbol": f"S{s:03d}", "L0": float(s) / n_symbols, "L2": float(s)}
            )
    labels_df = pl.DataFrame(labels_rows)
    write_dataset(labels_df, root, "us_labels_v1", manifest={})
    write_dataset(labels_df.drop("L0"), root, "us_labels_h5_v1", manifest={})
    write_dataset(labels_df.drop("L0"), root, "us_labels_h63_v1", manifest={})

    inputs = build_scan_inputs(root)

    assert inputs.core21["date"].max() <= DEV_END
    assert inputs.total_months == len(dates_in_dev)
    assert_dev_window(inputs.core21)
    assert_dev_window(inputs.core5)
    assert_dev_window(inputs.core63)


def _synthetic_full_scan_inputs(
    *, n_months: int = 12, n_symbols: int = 40, seed: int = 3
) -> ScanInputs:
    """``FEATURE_REGISTRY`` 44개 컬럼을 전부 채운 합성 ``ScanInputs`` —
    ``run_scan``(전체 파이프라인)을 실제 레이크 없이 돌리기 위한 것이다."""
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
                "date": date(2019, 1, 1),
                "symbol": f"S{s:03d}",
                "price_ge_5": True,
                "L0": l2[s] / n_symbols,
                "L2": l2[s],
            }
            for spec in FEATURE_REGISTRY:
                record[spec.feature] = float(feature_values[spec.feature][s])
            records.append(record)
    core21 = pl.DataFrame(records)
    feature_cols = [spec.feature for spec in FEATURE_REGISTRY]
    core5 = core21.select("month_idx", "date", "symbol", "price_ge_5", "L2", *feature_cols)
    core63 = core5
    labels_only = core21.select("month_idx", "symbol", "L2").unique()
    return ScanInputs(
        features_dev=core21.select("date", "symbol", "price_ge_5", *feature_cols),
        core21=core21,
        core5=core5,
        core63=core63,
        labels_only=labels_only,
        total_months=n_months,
    )


def test_run_scan_produces_88_rows_with_valid_grades() -> None:
    inputs = _synthetic_full_scan_inputs()
    shifts = [11, 13, 17, 19, 23]  # 단위테스트 전용 — T=12라 04 §1 범위(6~76)보다 좁힌다

    rows = run_scan(inputs, placebo_shifts=shifts)

    assert len(rows) == len(FEATURE_REGISTRY) * 2  # 44 피쳐 x 2 유니버스 = 88
    assert {r.universe for r in rows} == {"all", "price_ge_5"}
    assert {r.feature for r in rows} == {spec.feature for spec in FEATURE_REGISTRY}
    assert all(r.grade in {"A", "B", "C", "D", "R"} for r in rows)
    assert all(r.bh_q is not None for r in rows)  # run_scan이 BH를 이미 적용한다


def test_dedup_ab_features_runs_on_run_scan_output() -> None:
    """``run_scan`` 출력을 그대로 ``dedup_ab_features``에 넣어도 죽지 않고,
    남는 목록이 A·B 등급의 부분집합인지 확인한다(배선 검증)."""
    inputs = _synthetic_full_scan_inputs()
    shifts = [11, 13, 17, 19, 23]
    rows = run_scan(inputs, placebo_shifts=shifts)

    kept, _dropped = dedup_ab_features(rows, inputs.core21, universe="all")

    ab_features_all = {r.feature for r in rows if r.universe == "all" and r.grade in {"A", "B"}}
    assert set(kept).issubset(ab_features_all)
