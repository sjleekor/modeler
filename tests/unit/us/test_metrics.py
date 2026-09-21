"""``modeler.us.metrics`` 단위 테스트 — 전부 합성 데이터다. 실제 레이크를
읽지 않는다(``06_execution_steps.md`` M6 §6 지시).

top-k 선택 · 회전율 · 비용 항력(Q·k·스프레드 배수, 오염 sigma 방어) ·
손익분기 자금규모 · Sharpe/DSR/PBO 계산을 검사한다.
"""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
import pytest

from modeler.us import cost as cost_mod
from modeler.us import metrics as m

D1, D2, D3 = date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1)


# --- 1. topk_rows --------------------------------------------------------------


def test_topk_rows_keeps_only_top_k_by_pred() -> None:
    df = pl.DataFrame({"date": [D1] * 5, "symbol": list("ABCDE"), "pred": [5.0, 4, 3, 2, 1]})
    picked = m.topk_rows(df, k=2)
    assert sorted(picked["symbol"].to_list()) == ["A", "B"]


def test_topk_rows_drops_null_and_nonfinite_pred() -> None:
    df = pl.DataFrame(
        {
            "date": [D1] * 4,
            "symbol": list("ABCD"),
            "pred": [1.0, None, float("nan"), float("inf")],
        }
    )
    picked = m.topk_rows(df, k=10)
    assert picked["symbol"].to_list() == ["A"]


def test_topk_rows_breaks_ties_deterministically() -> None:
    df = pl.DataFrame({"date": [D1] * 4, "symbol": list("ABCD"), "pred": [1.0, 1.0, 1.0, 1.0]})
    picked = m.topk_rows(df, k=2)
    assert picked.height == 2  # 동순위라도 정확히 k개만 뽑는다


# --- 2. turnover_by_date --------------------------------------------------------


def test_turnover_by_date_first_month_is_full_turnover() -> None:
    membership = {D1: {"A", "B"}, D2: {"A", "B"}}
    out = m.turnover_by_date(membership, [D1, D2])
    assert out[D1] == 1.0  # 첫 리밸런스 — 전량 신규 진입
    assert out[D2] == 0.0  # 이후 보유 종목이 그대로다


def test_turnover_by_date_matches_jaccard_complement() -> None:
    membership = {D1: {"A", "B", "C"}, D2: {"A", "B", "D"}}
    out = m.turnover_by_date(membership, [D1, D2])
    # |교집합|=2, max(|A|,|B|)=3 -> turnover = 1 - 2/3
    assert out[D2] == pytest.approx(1.0 / 3.0)


def test_turnover_by_date_zero_denominator_is_nan() -> None:
    membership = {D1: set(), D2: set()}
    out = m.turnover_by_date(membership, [D1, D2])
    assert math.isnan(out[D2])


# --- 3. monthly_cost_drag --------------------------------------------------------


def _picked_frame(*, sigma: list[float], price: list[float], adv: list[float]) -> pl.DataFrame:
    n = len(sigma)
    return pl.DataFrame(
        {
            "date": [D1] * n,
            "symbol": [f"S{i}" for i in range(n)],
            "close": price,
            "sigma_daily": sigma,
            "adv_20d": adv,
        }
    )


def test_monthly_cost_drag_scales_mean_cost_by_turnover() -> None:
    picked = _picked_frame(sigma=[0.02, 0.02], price=[50.0, 50.0], adv=[1e8, 1e8])
    out_full = m.monthly_cost_drag(
        picked, turnover={D1: 1.0}, q_dollar=cost_mod.DEFAULT_Q_DOLLAR, k=cost_mod.DEFAULT_K
    )
    out_half = m.monthly_cost_drag(
        picked, turnover={D1: 0.5}, q_dollar=cost_mod.DEFAULT_Q_DOLLAR, k=cost_mod.DEFAULT_K
    )
    assert out_half["cost_drag"][0] == pytest.approx(out_full["cost_drag"][0] * 0.5)


def test_monthly_cost_drag_matches_cost_roundtrip_building_blocks() -> None:
    picked = _picked_frame(sigma=[0.02], price=[50.0], adv=[1e8])
    out = m.monthly_cost_drag(
        picked, turnover={D1: 1.0}, q_dollar=cost_mod.DEFAULT_Q_DOLLAR, k=cost_mod.DEFAULT_K
    )
    expected = cost_mod.spread(pl.col("close")) + 2 * cost_mod.impact(
        pl.col("sigma_daily"),
        pl.col("adv_20d"),
        q_dollar=cost_mod.DEFAULT_Q_DOLLAR,
        k=cost_mod.DEFAULT_K,
    )
    expected_value = picked.select(expected.alias("c"))["c"][0]
    assert out["mean_cost_roundtrip"][0] == pytest.approx(expected_value)


def test_monthly_cost_drag_spread_multiplier_doubles_spread_component() -> None:
    picked = _picked_frame(sigma=[0.0], price=[50.0], adv=[1e8])  # sigma=0 -> impact=0
    out_1x = m.monthly_cost_drag(picked, turnover={D1: 1.0}, q_dollar=1e7, k=0.1)
    out_2x = m.monthly_cost_drag(
        picked, turnover={D1: 1.0}, q_dollar=1e7, k=0.1, spread_multiplier=2.0
    )
    assert out_2x["mean_cost_roundtrip"][0] == pytest.approx(out_1x["mean_cost_roundtrip"][0] * 2.0)


def test_monthly_cost_drag_excludes_implausible_sigma() -> None:
    """corp_actions 결함(Y8c)이 만드는 것 같은 오염된 sigma는 평균에서 빠진다."""
    picked = _picked_frame(
        sigma=[0.02, 50_000.0], price=[50.0, 50.0], adv=[1e8, 1e8]
    )  # 두 번째 행은 명백한 오염값
    out = m.monthly_cost_drag(picked, turnover={D1: 1.0}, q_dollar=1e7, k=0.1)
    clean_only = m.monthly_cost_drag(
        _picked_frame(sigma=[0.02], price=[50.0], adv=[1e8]),
        turnover={D1: 1.0},
        q_dollar=1e7,
        k=0.1,
    )
    assert out["mean_cost_roundtrip"][0] == pytest.approx(clean_only["mean_cost_roundtrip"][0])


def test_monthly_cost_drag_empty_input_returns_empty_frame() -> None:
    empty = pl.DataFrame(
        schema={
            "date": pl.Date,
            "close": pl.Float64,
            "sigma_daily": pl.Float64,
            "adv_20d": pl.Float64,
        }
    )
    out = m.monthly_cost_drag(empty, turnover={}, q_dollar=1e7, k=0.1)
    assert out.height == 0


# --- 4. portfolio_track ----------------------------------------------------------


def test_portfolio_track_net_return_is_gross_minus_cost_drag() -> None:
    df = pl.DataFrame(
        {
            "date": [D1] * 3 + [D2] * 3,
            "symbol": ["A", "B", "C"] * 2,
            "pred": [3.0, 2.0, 1.0] * 2,
            "L0": [0.05, 0.03, 0.01, 0.04, 0.02, 0.0],
            "close": [50.0] * 6,
            "sigma_daily": [0.02] * 6,
            "adv_20d": [1e8] * 6,
        }
    )
    track = m.portfolio_track(df, k=2)
    assert track.height == 2
    diff = (track["gross_return"] - track["cost_drag"].fill_null(0.0)) - track["net_return"]
    assert diff.abs().max() < 1e-12


# --- 5. excess_over · hit_rate · sensitivity -------------------------------------


def test_excess_over_computes_mean_difference() -> None:
    track = pl.DataFrame({"date": [D1, D2], "net_return": [0.02, 0.04]})
    bench = pl.DataFrame({"date": [D1, D2], "bench": [0.01, 0.01]})
    value, n = m.excess_over(track, bench, benchmark_col="bench")
    assert n == 2
    assert value == pytest.approx((0.01 + 0.03) / 2)


def test_excess_over_series_matches_excess_over_mean() -> None:
    """``m7_run``의 permutation 재료 — ``excess_over``의 평균과 어긋나면 안 된다."""
    track = pl.DataFrame({"date": [D1, D2, D3], "net_return": [0.02, 0.04, -0.01]})
    bench = pl.DataFrame({"date": [D1, D2, D3], "bench": [0.01, 0.01, 0.02]})
    series = m.excess_over_series(track, bench, benchmark_col="bench")
    mean_value, n = m.excess_over(track, bench, benchmark_col="bench")
    assert series["date"].to_list() == [D1, D2, D3]
    assert series["excess"].to_list() == pytest.approx([0.01, 0.03, -0.03])
    assert float(series["excess"].mean()) == pytest.approx(mean_value)
    assert series.height == n


def test_excess_over_series_empty_when_no_overlap() -> None:
    track = pl.DataFrame({"date": [D1], "net_return": [0.02]})
    bench = pl.DataFrame({"date": [D2], "bench": [0.01]})
    series = m.excess_over_series(track, bench, benchmark_col="bench")
    assert series.height == 0
    assert series.columns == ["date", "excess"]


def test_hit_rate_fraction_positive() -> None:
    picked = pl.DataFrame({"L2": [0.1, -0.1, 0.2, -0.2, 0.0]})
    assert m.hit_rate(picked) == pytest.approx(2 / 5)


# --- 6. breakeven_q_dollar -------------------------------------------------------


def test_breakeven_q_dollar_none_when_gross_alpha_nonpositive() -> None:
    picked = _picked_frame(sigma=[0.02] * 5, price=[50.0] * 5, adv=[1e8] * 5)
    assert m.breakeven_q_dollar(0.0, picked, {D1: 0.5}) is None
    assert m.breakeven_q_dollar(-0.01, picked, {D1: 0.5}) is None


def test_breakeven_q_dollar_finds_root_where_e_is_zero() -> None:
    picked = _picked_frame(sigma=[0.02] * 20, price=[50.0] * 20, adv=[1e7] * 20)
    turnover = {D1: 1.0}
    gross_alpha = 0.01
    q = m.breakeven_q_dollar(gross_alpha, picked, turnover)
    assert q is not None
    drag = m._mean_cost_drag_at(
        picked,
        turnover,
        q_dollar=q,
        k=cost_mod.DEFAULT_K,
        spread_multiplier=1.0,
        price_col="close",
        sigma_col="sigma_daily",
        adv_col="adv_20d",
        date_col="date",
    )
    assert gross_alpha - drag == pytest.approx(0.0, abs=1e-6)


def test_breakeven_q_dollar_none_when_alpha_survives_whole_range() -> None:
    picked = _picked_frame(sigma=[0.001], price=[500.0], adv=[1e12])  # 비용이 거의 0
    q = m.breakeven_q_dollar(1.0, picked, {D1: 1.0}, q_high=1e6)
    assert q is None


def test_sensitivity_grid_cost_increases_with_q_and_k() -> None:
    picked = _picked_frame(sigma=[0.02] * 10, price=[50.0] * 10, adv=[1e7] * 10)
    turnover = {D1: 1.0}
    gross = pl.DataFrame({"date": [D1], "gross_return": [0.02]})
    bench = pl.DataFrame({"date": [D1], "bench": [0.0]})
    grid = m.sensitivity_grid(
        picked, turnover, gross_return_by_date=gross, benchmark=bench, benchmark_col="bench"
    )
    assert grid.height == len(cost_mod.Q_GRID) * len(cost_mod.K_GRID)
    at_low_k = grid.filter(pl.col("k") == min(cost_mod.K_GRID)).sort("q_dollar")["E"].to_list()
    at_high_k = grid.filter(pl.col("k") == max(cost_mod.K_GRID)).sort("q_dollar")["E"].to_list()
    assert at_low_k == sorted(at_low_k, reverse=True)  # Q가 커질수록 비용이 늘어 E가 준다
    for lo, hi in zip(at_low_k, at_high_k):
        assert hi <= lo  # k가 커질수록 비용이 늘어 E가 준다(또는 같다)


# --- 7. Sharpe · DSR · PBO -------------------------------------------------------


def test_sharpe_ratio_annualizes_by_sqrt_periods() -> None:
    returns = [0.01, 0.02, -0.01, 0.03, 0.0]
    monthly = m.sharpe_ratio(returns, periods_per_year=None)
    annual = m.sharpe_ratio(returns, periods_per_year=12)
    assert annual == pytest.approx(monthly * math.sqrt(12))


def test_sharpe_ratio_nan_when_too_few_or_constant() -> None:
    assert math.isnan(m.sharpe_ratio([0.01]))
    assert math.isnan(m.sharpe_ratio([0.01, 0.01, 0.01]))


def test_expected_max_sharpe_increases_with_n_trials() -> None:
    small_n = m.expected_max_sharpe(5, 0.01)
    large_n = m.expected_max_sharpe(500, 0.01)
    assert large_n > small_n > 0


def test_deflated_sharpe_ratio_matches_documented_formula() -> None:
    """``z``가 docstring이 적은 식(Bailey·Lopez de Prado 2014 식 8·9) 그대로인지
    직접 계산해 대조한다 — 왜도=0·첨도=3(정규분포)이라도 분모의
    ``(kurtosis-1)/4*SR^2`` 항은 0이 아니다(SR=0일 때만 1로 준다)."""
    sr_hat, skewness, kurt, n_obs = 0.5, 0.0, 3.0, 100
    result = m.deflated_sharpe_ratio(
        sr_hat, n_trials=2, sr_var_across_trials=1e-6, skewness=skewness, kurtosis=kurt, n_obs=n_obs
    )
    assert result["sr0_expected_max_sharpe"] == pytest.approx(0.0, abs=1e-2)
    denom = math.sqrt(1 - skewness * sr_hat + (kurt - 1) / 4 * sr_hat**2)
    expected_z = (sr_hat - result["sr0_expected_max_sharpe"]) * math.sqrt(n_obs - 1) / denom
    assert result["z"] == pytest.approx(expected_z)
    assert 0.0 <= result["dsr"] <= 1.0


def test_deflated_sharpe_ratio_more_trials_lowers_dsr() -> None:
    few = m.deflated_sharpe_ratio(
        0.3, n_trials=2, sr_var_across_trials=0.01, skewness=0.1, kurtosis=3.0, n_obs=60
    )
    many = m.deflated_sharpe_ratio(
        0.3, n_trials=100, sr_var_across_trials=0.01, skewness=0.1, kurtosis=3.0, n_obs=60
    )
    assert many["dsr"] < few["dsr"]  # 시행이 많을수록 같은 SR_hat도 덜 특별해 보인다


def test_sample_skew_kurtosis_normal_like() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=5000)
    skewness, kurt = m.sample_skew_kurtosis(values.tolist())
    assert skewness == pytest.approx(0.0, abs=0.15)
    assert kurt == pytest.approx(3.0, abs=0.3)


def test_pbo_fraction_nonpositive() -> None:
    # <= 0인 값은 -0.1, -0.2, 0.0 셋 -> 5개 중 3개.
    assert m.pbo_fraction_nonpositive([0.1, -0.1, 0.2, -0.2, 0.0]) == pytest.approx(3 / 5)
    assert math.isnan(m.pbo_fraction_nonpositive([]))
    assert math.isnan(m.pbo_fraction_nonpositive([float("nan")]))


# --- 8. s_spread_long_short · s_long_short_summary (M7 추가) ---------------------


def _long_short_frame() -> pl.DataFrame:
    """symbol 10개, pred=1..10, L0=pred와 같은 방향(0.01×pred).

    fraction=0.2 -> top 2(pred 9,10) 평균 L0=0.095, bottom 2(pred 1,2)
    평균 L0=0.015, universe 평균=0.055 -> spread=0.08, long=0.04, short=0.04.
    """
    n = 10
    return pl.DataFrame(
        {
            "date": [D1] * n,
            "symbol": [f"S{i}" for i in range(n)],
            "pred": [float(i) for i in range(1, n + 1)],
            "L0": [0.01 * i for i in range(1, n + 1)],
        }
    )


def test_s_spread_long_short_matches_hand_computed_values() -> None:
    df = _long_short_frame()
    monthly = m.s_spread_long_short(df, min_names=5, fraction=0.2)
    assert monthly.height == 1
    row = monthly.row(0, named=True)
    assert row["top"] == pytest.approx(0.095)
    assert row["bottom"] == pytest.approx(0.015)
    assert row["universe"] == pytest.approx(0.055)
    assert row["spread"] == pytest.approx(0.08)
    assert row["long_excess"] == pytest.approx(0.04)
    assert row["short_excess"] == pytest.approx(0.04)
    assert row["long_excess"] + row["short_excess"] == pytest.approx(row["spread"])


def test_s_spread_long_short_drops_dates_below_min_names() -> None:
    df = _long_short_frame()
    monthly = m.s_spread_long_short(df, min_names=20, fraction=0.2)
    assert monthly.height == 0
    assert monthly.columns == [
        "date",
        "top",
        "universe",
        "bottom",
        "spread",
        "long_excess",
        "short_excess",
    ]


def test_s_long_short_summary_shares_sum_to_one() -> None:
    monthly = m.s_spread_long_short(_long_short_frame(), min_names=5, fraction=0.2)
    summary = m.s_long_short_summary(monthly)
    assert summary["S"] == pytest.approx(0.08)
    assert summary["long_share"] == pytest.approx(0.5)
    assert summary["short_share"] == pytest.approx(0.5)
    assert summary["long_share"] + summary["short_share"] == pytest.approx(1.0)
    assert summary["n_months"] == 1


def test_s_long_short_summary_empty_input() -> None:
    empty = pl.DataFrame(
        schema={
            "date": pl.Date,
            "top": pl.Float64,
            "universe": pl.Float64,
            "bottom": pl.Float64,
            "spread": pl.Float64,
            "long_excess": pl.Float64,
            "short_excess": pl.Float64,
        }
    )
    summary = m.s_long_short_summary(empty)
    assert summary["n_months"] == 0
    assert math.isnan(summary["S"])


# --- 9. sign_flip_permutation_probability (M7 추가 — M8 갈래 A 귀무 확률 재료) -----


def test_sign_flip_permutation_probability_all_zero_series_never_positive() -> None:
    """모든 값이 0이면 부호를 아무리 뒤섞어도 평균은 항상 0이다 — 절대 양수가 아니다."""
    series = {"E": [0.0] * 12, "E_ew": [0.0] * 12, "I": [0.0] * 12, "S": [0.0] * 12}
    result = m.sign_flip_permutation_probability(series, n_perm=200, seed=1)
    assert result["probability_all_positive"] == 0.0
    assert result["n_months"] == 12


def test_sign_flip_permutation_probability_reproducible_with_same_seed() -> None:
    series = {"E": [0.01, -0.02, 0.03], "E_ew": [0.02, 0.01, -0.01], "I": [0.0, 0.05, 0.02]}
    first = m.sign_flip_permutation_probability(series, n_perm=500, seed=42)
    second = m.sign_flip_permutation_probability(series, n_perm=500, seed=42)
    assert first["probability_all_positive"] == second["probability_all_positive"]


def test_sign_flip_permutation_probability_single_month_shared_sign_is_all_or_nothing() -> None:
    """한 시행 안에서는 같은 부호 벡터를 시리즈 전부에 곱한다 — 한 달짜리면 뒤집힐 때
    넷이 같이 뒤집혀 대략 절반은 전부 양수, 절반은 전부 음수가 된다."""
    series = {"E": [1.0], "E_ew": [1.0], "I": [1.0], "S": [1.0]}
    result = m.sign_flip_permutation_probability(series, n_perm=5000, seed=7)
    assert result["probability_all_positive"] == pytest.approx(0.5, abs=0.03)


def test_sign_flip_permutation_probability_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="길이가 다릅니다"):
        m.sign_flip_permutation_probability({"E": [0.1, 0.2], "S": [0.1]}, n_perm=10, seed=0)


def test_sign_flip_permutation_probability_empty_series_map_raises() -> None:
    with pytest.raises(ValueError, match="비어"):
        m.sign_flip_permutation_probability({}, n_perm=10, seed=0)


def test_sign_flip_permutation_probability_empty_series_returns_nan() -> None:
    result = m.sign_flip_permutation_probability({"E": [], "S": []}, n_perm=10, seed=0)
    assert result["n_months"] == 0
    assert math.isnan(result["probability_all_positive"])
