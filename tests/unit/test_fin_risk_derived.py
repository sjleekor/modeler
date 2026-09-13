"""F-5.4 — the three families the new metrics made possible.

`fin_current_ratio`, `fin_borrowings_to_mcap` and `fin_altman_z` are built on
the five metrics F-5.1 measured and F-5.2 mapped. Each carries a decision the
measurement forced, and each of those is what these tests pin:

* `fin_borrowings_to_mcap` sums short and long term only. The current portion
  of long-term borrowings never reached 0.5 coverage, so it is not a required
  component -- and the omission is stated on the card (median 10.2%, p90 59.7%
  understatement) rather than hidden;
* `fin_altman_z` is the full five-term form, which retained earnings is what
  made possible, and it is NULL unless all five terms are present. A partial
  sum would put a company missing its sales line into the same distribution as
  one that genuinely scored low;
* the weights are Altman's, not re-fitted here, and they live in the SQL text
  so a changed weight changes the mart's cache key.
"""

from __future__ import annotations

import datetime as dt

import pytest

from modeler.etl.features import fin_risk

from .test_fin_risk import _fixture, _instant, _ttm

_ASOF = dt.date(2020, 3, 2)
_READ = dt.date(2020, 3, 3)


def _read(con, column: str, on: dt.date = _READ):
    return con.execute(
        f"SELECT {column} FROM feat_fin_risk WHERE trade_date = ? AND ticker = '005930'",
        [on],
    ).fetchone()[0]


def _balance_sheet(**values: float) -> list[tuple]:
    """Balance-sheet vintages, plus the `net_income` row the basis rule needs.

    `fs_basis_used` is decided by whether `net_income` has a consolidated value
    that day, and every other metric is then read on that one basis -- so a
    fixture without it reads NULL for everything, not just for net income.
    """
    rows = [_instant("net_income", 0, 5.0, _ASOF)]
    rows += [
        _instant(metric, i + 1, value, _ASOF)
        for i, (metric, value) in enumerate(values.items())
    ]
    return rows


# --------------------------------------------------------------------------
# fin_current_ratio
# --------------------------------------------------------------------------


def test_the_current_ratio_is_current_assets_over_current_liabilities() -> None:
    con = _fixture(_balance_sheet(current_assets=300.0, current_liabilities=200.0))

    assert _read(con, "fin_current_ratio") == pytest.approx(1.5)


def test_a_zero_current_liability_denominator_is_null_not_infinity() -> None:
    con = _fixture(_balance_sheet(current_assets=300.0, current_liabilities=0.0))

    assert _read(con, "fin_current_ratio") is None


def test_the_current_ratio_needs_both_legs() -> None:
    con = _fixture(_balance_sheet(current_liabilities=200.0))

    assert _read(con, "fin_current_ratio") is None


# --------------------------------------------------------------------------
# fin_borrowings_to_mcap
# --------------------------------------------------------------------------


def test_borrowings_are_the_sum_of_the_two_components() -> None:
    con = _fixture(
        _balance_sheet(borrowings_short_term=120.0, borrowings_long_term=80.0)
    )

    # market_cap_pit is 1,000,000 in the shared fixture.
    assert _read(con, "fin_borrowings_total") == pytest.approx(200.0)
    assert _read(con, "fin_borrowings_to_mcap") == pytest.approx(200.0 / 1_000_000)


def test_one_missing_component_makes_the_sum_null_rather_than_partial() -> None:
    # Treating an absent long-term line as zero would report a company with
    # only short-term disclosure as having less debt than it does, and that
    # error correlates with disclosure quality.
    con = _fixture(_balance_sheet(borrowings_short_term=120.0))

    assert _read(con, "fin_borrowings_total") is None
    assert _read(con, "fin_borrowings_to_mcap") is None


def test_the_current_portion_of_long_term_debt_is_not_a_component() -> None:
    # Rejected at a 0.466 coverage maximum (F-5.1). It is not registered as a
    # metric at all, so nothing downstream can quietly start using it.
    assert "borrowings_current_portion" not in fin_risk._METRICS


def test_the_raw_numerator_is_emitted_so_the_sum_can_be_audited() -> None:
    # The reconciliation F-5.1 §6.3 ran -- short + long never exceeding total
    # liabilities across 10,012 corp-years -- is a claim about this column.
    assert "fin_borrowings_total" in fin_risk.DIAGNOSTIC_COLUMNS


# --------------------------------------------------------------------------
# fin_altman_z
# --------------------------------------------------------------------------


def _altman_inputs() -> list[tuple]:
    """Five terms with values chosen so each contributes a distinct amount."""
    return [
        _instant("net_income", 0, 5.0, _ASOF),
        _instant("current_assets", 1, 400.0, _ASOF),
        _instant("current_liabilities", 2, 200.0, _ASOF),
        _instant("retained_earnings", 3, 300.0, _ASOF),
        _instant("total_assets", 4, 1000.0, _ASOF),
        _instant("total_liabilities", 5, 500.0, _ASOF),
        _ttm("operating_income", 6, 100.0, _ASOF),
        _ttm("revenue", 7, 2000.0, _ASOF),
    ]


def test_the_five_terms_are_each_emitted() -> None:
    con = _fixture(_altman_inputs())

    assert _read(con, "fin_altman_wc_to_assets") == pytest.approx(0.2)  # (400-200)/1000
    assert _read(con, "fin_altman_re_to_assets") == pytest.approx(0.3)
    assert _read(con, "fin_altman_ebit_to_assets") == pytest.approx(0.1)
    assert _read(con, "fin_altman_mve_to_liabilities") == pytest.approx(2000.0)
    assert _read(con, "fin_altman_sales_to_assets") == pytest.approx(2.0)


def test_the_score_is_altmans_weighted_sum_of_those_terms() -> None:
    con = _fixture(_altman_inputs())
    w = fin_risk.ALTMAN_WEIGHTS

    expected = (
        w["working_capital_to_assets"] * 0.2
        + w["retained_earnings_to_assets"] * 0.3
        + w["ebit_to_assets"] * 0.1
        + w["market_equity_to_liabilities"] * 2000.0
        + w["sales_to_assets"] * 2.0
    )

    assert _read(con, "fin_altman_z") == pytest.approx(expected)


def test_the_weights_are_the_papers() -> None:
    # Altman (1968) table 1. Not this repository's to re-fit, the same rule
    # LIFECYCLE_STAGES follows for Dickinson.
    assert fin_risk.ALTMAN_WEIGHTS == {
        "working_capital_to_assets": 1.2,
        "retained_earnings_to_assets": 1.4,
        "ebit_to_assets": 3.3,
        "market_equity_to_liabilities": 0.6,
        "sales_to_assets": 1.0,
    }


def test_the_weights_are_part_of_the_marts_cache_key() -> None:
    sql = fin_risk.build_fin_risk_sql()

    for weight in fin_risk.ALTMAN_WEIGHTS.values():
        assert str(weight) in sql
    assert fin_risk.FORMULA_VERSION == "fin_risk_v3"
    assert fin_risk.FORMULA_VERSION in sql


@pytest.mark.parametrize(
    "dropped",
    ["current_assets", "retained_earnings", "revenue"],
)
def test_one_missing_term_makes_the_whole_score_null(dropped: str) -> None:
    rows = [row for row in _altman_inputs() if row[1] != dropped]
    con = _fixture(rows)

    assert _read(con, "fin_altman_z") is None


def test_the_score_needs_a_usable_market_cap() -> None:
    # MVE/TL is the one price-scaled term, so it carries base_ok exactly as
    # fin_net_debt_to_mcap does. Without it the score would mix a tradable
    # price with a stale share count.
    rows = [row for row in _altman_inputs() if row[1] != "total_liabilities"]
    con = _fixture(rows)

    assert _read(con, "fin_altman_mve_to_liabilities") is None
    assert _read(con, "fin_altman_z") is None


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def test_the_three_families_are_primaries_with_lag1_variants() -> None:
    # Every registered mart carries a lag1 column per primary -- the gap F-HS-1
    # found in feat_relation_stat, where the config would have named a column
    # that did not exist.
    con = _fixture(_altman_inputs())
    columns = {row[0] for row in con.execute("DESCRIBE feat_fin_risk").fetchall()}

    for name in ("fin_current_ratio", "fin_borrowings_to_mcap", "fin_altman_z"):
        assert name in fin_risk.PRIMARY_COLUMNS
        assert name in columns
        assert f"{name}_lag1" in columns


def test_each_new_family_reports_its_own_availability_date() -> None:
    con = _fixture(_altman_inputs())
    columns = {row[0] for row in con.execute("DESCRIBE feat_fin_risk").fetchall()}

    for group in ("current_ratio", "borrowings", "altman"):
        assert group in fin_risk.AVAILABILITY_GROUPS
        assert f"{group}_available_from" in columns
        assert f"{group}_fin_age_days" in columns


def test_the_altman_availability_is_the_latest_of_all_its_inputs() -> None:
    # A score is only readable once its *last* input is, so the group date is a
    # max over seven metrics rather than any one of them.
    late = dt.date(2020, 3, 4)
    rows = [row for row in _altman_inputs() if row[1] != "revenue"]
    rows.append(_ttm("revenue", 8, 2000.0, late))
    con = _fixture(rows, sessions=[dt.date(2020, 3, d) for d in (2, 3, 4, 5, 6)])

    assert _read(con, "altman_available_from", dt.date(2020, 3, 5)) == late
    assert _read(con, "fin_altman_z", dt.date(2020, 3, 3)) is None
    assert _read(con, "fin_altman_z", dt.date(2020, 3, 5)) is not None


def test_the_nine_f4_families_still_come_first_in_output_order() -> None:
    # The v2 families are published and their formulas are untouched; the three
    # new ones are appended, not interleaved.
    assert fin_risk.PRIMARY_COLUMNS[:9] == (
        "fin_debt_to_assets",
        "fin_net_debt_to_mcap",
        "fin_interest_coverage",
        "fin_ext_finance_to_assets",
        "fin_lifecycle_stage",
        "fin_lifecycle_transition",
        "fin_profit_turn",
        "fin_dividend_initiation",
        "fin_negative_equity_exit",
    )
