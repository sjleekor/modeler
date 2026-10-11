"""E3 보수 선택: F = T−92일 규칙(8월 말 사례), A = 12개 중 10개 이상 평균, 지연 판독."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from modeler.scores.quality import etf_e3_fees as fees


def frame(rows):
    return pl.DataFrame(
        rows,
        schema={
            "fund_code": pl.String,
            "operator": pl.String,
            "sum_fee": pl.Float64,
            "ter": pl.Float64,
            "brokerage": pl.Float64,
            "base_date": pl.Date,
        },
    )


def month_ends(y):
    import calendar

    return [date(y, m, calendar.monthrange(y, m)[1]) for m in range(1, 13)]


def make(ter_by_month=None, fund="F1", y0=2016, y1=2017):
    rows = []
    for y in (y0, y1):
        for i, d in enumerate(month_ends(y)):
            t = (ter_by_month or {}).get((y, i + 1), 0.1 * (i + 1) + (y - y0))
            rows.append(
                {
                    "fund_code": "OTHER",
                    "operator": "OP",
                    "sum_fee": 1.0,
                    "ter": 0.5,
                    "brokerage": 0.5,
                    "base_date": d,
                }
            )
            if t is not None:
                rows.append(
                    {
                        "fund_code": fund,
                        "operator": "OP",
                        "sum_fee": t + 1,
                        "ter": t,
                        "brokerage": 0.5,
                        "base_date": d,
                    }
                )
    return fees.build_fee_arrays(frame(rows))


@pytest.mark.parametrize("t", [date(2016, 12, 30), date(2016, 12, 29), date(2016, 12, 23)])
def test_formation_uses_august_end_for_december_t(t):
    fa = make()
    # 8월 말 값 = 0.8 (월 번호 × 0.1)
    v = fees.formation_value(fa, "F1", np.datetime64(t))
    assert v == pytest.approx(0.8)
    # 12월 말 T 에서 실효 지연 119~121일
    assert 114 <= (t - date(2016, 8, 31)).days <= 121


def test_formation_boundary_includes_exactly_92_days():
    fa = make()
    t = date(2016, 11, 30)
    # T−92 = 2016-08-30 이라 8월 말 파일(08-31)은 제외, 7월 말 값(0.7)
    assert fees.formation_value(fa, "F1", np.datetime64(t)) == pytest.approx(0.7)
    t = date(2016, 12, 1)  # T−92 = 08-31 → 8월 말 포함
    assert fees.formation_value(fa, "F1", np.datetime64(t)) == pytest.approx(0.8)


def test_formation_no_fallback_when_latest_file_lacks_fund():
    fa = make({(2016, 8): None})  # 8월 말에 이 펀드 행이 없다
    t = np.datetime64(date(2016, 12, 30))
    assert not fees.formation_available(fa, "F1", t)
    assert np.isnan(fees.formation_value(fa, "F1", t))


def test_formation_lag_variants():
    fa = make()
    t = np.datetime64(date(2016, 12, 30))
    assert fees.formation_value(fa, "F1", t, lag_days=0) == pytest.approx(1.1)
    assert fees.formation_value(fa, "F1", t, lag_days=61) == pytest.approx(
        0.9
    )  # T−61 = 10-30 이라 9월 말 파일


def test_formation_unknown_fund_and_too_early():
    fa = make()
    assert not fees.formation_available(fa, "ZZ", np.datetime64(date(2016, 12, 30)))
    assert not fees.formation_available(fa, "F1", np.datetime64(date(2016, 2, 1)))


def test_outcome_mean_and_min_months():
    fa = make()
    # 2017년 값 = 0.1 m + 1 → 평균 = 0.65 + 1
    assert fees.outcome_value(fa, "F1", 2017) == pytest.approx(1.65)
    assert fees.outcome_months(fa, "F1", 2017) == 12
    fa10 = make({(2017, 3): None, (2017, 4): None})
    assert fees.outcome_months(fa10, "F1", 2017) == 10
    assert np.isfinite(fees.outcome_value(fa10, "F1", 2017))
    fa9 = make({(2017, 3): None, (2017, 4): None, (2017, 5): None})
    assert np.isnan(fees.outcome_value(fa9, "F1", 2017))


def test_fee_field_variants():
    fa = make()
    t = np.datetime64(date(2016, 12, 30))
    assert fees.formation_value(fa, "F1", t, field="sum_fee") == pytest.approx(1.8)
    assert fees.formation_value(fa, "F1", t, field="ter+brokerage") == pytest.approx(1.3)
    with pytest.raises(ValueError):
        fees.formation_value(fa, "F1", t, field="bogus")


def test_lag_summary():
    s = fees.lag_summary([40, 41, 45])
    assert s["status"] == "잠정(지연 실측 전)" and s["cap"] is False and s["max"] == 45
    s = fees.lag_summary([40, 41, 45, 50, 60, 70])
    assert s["status"] == "실측" and not s["cap"] and s["median"] == 47.5
    s = fees.lag_summary([40, 41, 45, 50, 60, 93])
    assert s["cap"] is True


def test_lag_cap_on_any_single_observation_over_92():
    s = fees.lag_summary([95])
    assert s["status"] == "잠정(지연 실측 전)" and s["cap"] is True
    s = fees.lag_summary([], censored_over=93)
    assert s["cap"] is True
    s = fees.lag_summary([40], censored_over=30)
    assert s["cap"] is False


def test_aug_fixed_picks_august_file_even_when_rule_picks_september():
    fa = make()
    # 2016-12-30: 규칙은 8월 말(0.8). T = 2016-12-01 처럼 T−92 가 9월 말을 고르는 날을 만든다
    t = np.datetime64(date(2017, 1, 3))  # T−92 = 2016-10-03 → 규칙은 2016-09-30 파일(0.9)
    assert fees.formation_value(fa, "F1", t) == pytest.approx(0.9)
    # 같은 해(2017)의 8월 말 값 = 0.8 + 1
    assert fees.formation_value(fa, "F1", t, aug_fixed=True) == pytest.approx(1.8)
    assert fees.formation_available(fa, "F1", t, aug_fixed=True)
    assert not fees.formation_available(fa, "F1", np.datetime64(date(2019, 12, 30)), aug_fixed=True)
