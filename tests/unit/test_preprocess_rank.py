"""Unit tests for the ``rank`` preprocess profile (20260907_model_experiment `02` §4.1).

The profile's whole claim is that it has no fitted state: a percentile inside
one ``(trade_date, market)`` cross-section cannot borrow anything from another
fold. Two tests below assert exactly that, and the rest pin the percentile
definition the interaction builder shares (`02` §3.2).
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from modeler.etl import preprocess as pp

D1 = datetime.date(2020, 1, 1)
D2 = datetime.date(2020, 1, 2)


def _panel() -> pl.DataFrame:
    """Two dates x two markets, with a null, a tie, and three flag columns."""
    return pl.DataFrame(
        {
            "trade_date": [D1, D1, D1, D1, D2, D2],
            "ticker": ["A", "B", "C", "D", "A", "B"],
            "market": ["KOSPI", "KOSPI", "KOSPI", "KOSDAQ", "KOSPI", "KOSPI"],
            # KOSPI/D1: 0.1, 0.1, 0.3 (a tie at the bottom) ; KOSDAQ/D1: one name
            "px_ret_20d": [0.1, 0.1, 0.3, 9.9, 0.2, None],
            "px_turnover": [100.0, 200.0, 300.0, 400.0, 50.0, 150.0],
            "px_is_halted": [False, False, True, False, False, False],
            "rg_vix_high": [1, 1, 1, 1, 0, 0],
            "y_up_20d": [1, 0, 1, 0, 1, 0],
        }
    )


def _fit(panel: pl.DataFrame, **kwargs) -> pl.DataFrame:
    cfg = pp.PreprocessConfig(profile="rank", **kwargs)
    fitted = pp.fit(panel, cfg)
    return fitted.transform(panel)


def _cell(out: pl.DataFrame, ticker: str, day: datetime.date, col: str):
    return out.filter((pl.col("ticker") == ticker) & (pl.col("trade_date") == day))[col][0]


# --- the percentile definition ----------------------------------------------


def test_rank_spans_the_unit_interval_within_date_and_market() -> None:
    out = _fit(_panel())
    kospi_d1 = out.filter((pl.col("trade_date") == D1) & (pl.col("market") == "KOSPI"))
    values = sorted(kospi_d1["px_turnover"].to_list())
    assert values == [pytest.approx(0.0), pytest.approx(0.5), pytest.approx(1.0)]


def test_rank_is_taken_inside_the_market_not_across_it() -> None:
    """KOSDAQ's single name is 0.5, even though its raw value is the panel max."""
    out = _fit(_panel())
    assert _cell(out, "D", D1, "px_ret_20d") == pytest.approx(0.5)
    assert _cell(out, "D", D1, "px_turnover") == pytest.approx(0.5)


def test_ties_share_the_average_rank() -> None:
    out = _fit(_panel())
    # KOSPI/D1 px_ret_20d = {0.1, 0.1, 0.3}: average ranks 1.5, 1.5, 3
    # -> (r-1)/(n-1) = 0.25, 0.25, 1.0. Minimum ties would give 0.0 twice.
    assert _cell(out, "A", D1, "px_ret_20d") == pytest.approx(0.25)
    assert _cell(out, "B", D1, "px_ret_20d") == pytest.approx(0.25)
    assert _cell(out, "C", D1, "px_ret_20d") == pytest.approx(1.0)


def test_a_one_name_cross_section_is_neutral() -> None:
    single = _panel().filter((pl.col("trade_date") == D1) & (pl.col("market") == "KOSDAQ"))
    out = _fit(single)
    assert out["px_ret_20d"].to_list() == [pytest.approx(0.5)]


# --- missing values ---------------------------------------------------------


def test_missing_becomes_neutral_and_the_isna_flag_records_it() -> None:
    out = _fit(_panel())
    assert _cell(out, "B", D2, "px_ret_20d") == pytest.approx(0.5)
    assert _cell(out, "B", D2, "px_ret_20d_isna") == 1
    assert _cell(out, "A", D2, "px_ret_20d_isna") == 0


def test_missing_stays_null_for_a_tree() -> None:
    out = _fit(_panel(), rank_null_fill=None)
    assert _cell(out, "B", D2, "px_ret_20d") is None
    # and the one name left in that cross-section is neutral, not 0 or 1.
    assert _cell(out, "A", D2, "px_ret_20d") == pytest.approx(0.5)


# --- flags are not ranked ---------------------------------------------------


def test_flag_columns_keep_their_zero_one_values() -> None:
    out = _fit(_panel())
    # px_is_halted: ranking it would turn "halted" into "how many were halted".
    assert out["px_is_halted"].to_list() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    # rg_* arrives as Int8 from dim_regime_daily, so the name-based rule is what
    # keeps it out of the transform.
    assert out["rg_vix_high"].to_list() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert pp.is_flag_feature("rg_market_up", pl.Int8)
    assert pp.is_flag_feature("fin_has_fs", pl.Float64)
    assert pp.is_flag_feature("px_ret_20d_isna", pl.Int8)
    assert not pp.is_flag_feature("px_ret_20d", pl.Float64)


def test_rankable_columns_excludes_flags_and_keys() -> None:
    panel = _panel()
    feats = pp.feature_columns(panel)
    rankable = pp.rankable_columns(panel, feats)
    assert rankable == ["px_ret_20d", "px_turnover"]


# --- no fitted state --------------------------------------------------------


def test_rank_profile_fits_nothing() -> None:
    cfg = pp.PreprocessConfig(profile="rank")
    fitted = pp.fit(_panel(), cfg)
    assert fitted.bounds == {}


def test_output_does_not_depend_on_the_slice_it_was_fit_on() -> None:
    """The structural no-leak property: fit on one fold, get the same numbers."""
    panel = _panel()
    cfg = pp.PreprocessConfig(profile="rank")
    train_only = pp.fit(panel.filter(pl.col("trade_date") == D1), cfg).transform(panel)
    whole = pp.fit(panel, cfg).transform(panel)
    assert train_only.equals(whole)


def test_transform_is_row_order_independent() -> None:
    panel = _panel()
    cfg = pp.PreprocessConfig(profile="rank")
    straight = pp.fit(panel, cfg).transform(panel)
    shuffled = pp.fit(panel, cfg).transform(panel.reverse())
    assert shuffled.sort(["trade_date", "ticker"]).equals(straight.sort(["trade_date", "ticker"]))


# --- the other two profiles are untouched -----------------------------------


def test_linear_and_tree_profiles_still_behave_as_before() -> None:
    panel = _panel()
    linear = pp.fit(panel, pp.PreprocessConfig(profile="linear")).transform(panel)
    pp.assert_finite(linear, pp.feature_columns(panel))
    # per-date z-score: mean ~0 within each date.
    for mean in linear.group_by("trade_date").agg(pl.col("px_turnover").mean())["px_turnover"]:
        assert abs(mean) < 1e-9

    tree = pp.fit(panel, pp.PreprocessConfig(profile="tree")).transform(panel)
    assert tree["px_ret_20d"].null_count() == 1  # no impute for trees
    # winsorize bounds are still fit for both.
    assert pp.fit(panel, pp.PreprocessConfig(profile="tree")).bounds != {}


def test_flag_cast_does_not_change_flag_values_in_the_linear_profile() -> None:
    panel = _panel()
    out = pp.fit(panel, pp.PreprocessConfig(profile="tree")).transform(panel)
    # the cast widens Int8/Boolean to Float64; the 0/1 content is preserved.
    assert out["px_is_halted"].to_list() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert out["rg_vix_high"].to_list() == [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_rank_is_a_valid_profile_and_typos_are_not() -> None:
    pp.PreprocessConfig(profile="rank")
    with pytest.raises(ValueError):
        pp.PreprocessConfig(profile="ranked")
