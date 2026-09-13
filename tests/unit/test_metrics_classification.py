"""Unit tests for the probability metrics (20260907_model_experiment `03` §2.1, §2.3).

Everything that has an sklearn equivalent is checked against it to 1e-12 — the
point of writing them here is to avoid an sklearn dependency in the mart path,
not to invent different definitions. ECE, precision@k and the threshold
portfolio have no sklearn twin, so those are pinned by hand-computed examples.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from modeler.etl.metrics import (
    PROB_CLIP,
    binary_auc,
    binary_log_loss,
    brier_score,
    classification_report,
    expected_calibration_error,
    rebalance_grid,
    reliability_table,
    threshold_economic_report,
    threshold_membership,
)
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

RNG = np.random.default_rng(0)


def _frame(n_dates: int = 6, n_names: int = 40) -> pl.DataFrame:
    """A synthetic panel with a real (weak) relation between p and y."""
    dates, tickers, preds, ys = [], [], [], []
    for d in range(n_dates):
        p = RNG.uniform(0.05, 0.95, n_names)
        y = (RNG.uniform(size=n_names) < p).astype(int)
        dates += [d] * n_names
        tickers += [f"T{i:03d}" for i in range(n_names)]
        preds += p.tolist()
        ys += y.tolist()
    return pl.DataFrame({"trade_date": dates, "ticker": tickers, "p": preds, "y": ys})


# --- against sklearn ---------------------------------------------------------


def test_log_loss_matches_sklearn() -> None:
    df = _frame()
    p, y = df["p"].to_numpy(), df["y"].to_numpy()
    assert binary_log_loss(p, y) == pytest.approx(log_loss(y, p), abs=1e-12)


def test_brier_matches_sklearn() -> None:
    df = _frame()
    p, y = df["p"].to_numpy(), df["y"].to_numpy()
    assert brier_score(p, y) == pytest.approx(brier_score_loss(y, p), abs=1e-12)


def test_auc_matches_sklearn_including_ties() -> None:
    df = _frame()
    p, y = df["p"].to_numpy(), df["y"].to_numpy()
    assert binary_auc(p, y) == pytest.approx(roc_auc_score(y, p), abs=1e-12)

    # heavy ties: half the scores identical, so tie handling decides the value.
    tied = np.round(p * 4) / 4
    assert binary_auc(tied, y) == pytest.approx(roc_auc_score(y, tied), abs=1e-12)


def test_report_metrics_match_sklearn_on_the_pooled_slice() -> None:
    df = _frame()
    rep = classification_report(df, pred_col="p", y_col="y", k=10)
    p, y = df["p"].to_numpy(), df["y"].to_numpy()
    assert rep.log_loss == pytest.approx(log_loss(y, p), abs=1e-12)
    assert rep.brier == pytest.approx(brier_score_loss(y, p), abs=1e-12)
    assert rep.auc_pooled == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert rep.n_obs == df.height
    assert rep.n_dates == 6


# --- the clip ----------------------------------------------------------------


def test_log_loss_clip_bounds_a_confident_miss() -> None:
    p = np.array([0.0, 1.0])
    y = np.array([1.0, 0.0])
    assert binary_log_loss(p, y) == pytest.approx(-math.log(PROB_CLIP))


def test_uninformative_probability_scores_ln_two() -> None:
    p = np.full(1000, 0.5)
    y = np.array([1, 0] * 500, dtype=float)
    assert binary_log_loss(p, y) == pytest.approx(math.log(2), abs=1e-12)
    assert brier_score(p, y) == pytest.approx(0.25)


# --- AUC is undefined on a one-class day -------------------------------------


def test_auc_is_nan_when_a_class_is_absent() -> None:
    assert math.isnan(binary_auc(np.array([0.2, 0.8]), np.array([1.0, 1.0])))


def test_daily_auc_skips_one_class_dates_instead_of_scoring_them_half() -> None:
    df = pl.DataFrame(
        {
            "trade_date": [1, 1, 1, 1, 2, 2],
            "ticker": ["A", "B", "C", "D", "A", "B"],
            # date 1: perfectly ordered -> AUC 1.0 ; date 2: all positive -> NaN
            "p": [0.1, 0.2, 0.8, 0.9, 0.3, 0.7],
            "y": [0, 0, 1, 1, 1, 1],
        }
    )
    rep = classification_report(df, pred_col="p", y_col="y", k=2)
    assert rep.auc_daily_mean == pytest.approx(1.0)
    assert rep.n_dates_auc == 1
    assert rep.n_dates == 2


# --- ECE and the reliability table -------------------------------------------


def test_ece_hand_example() -> None:
    # two bins used: p=0.15 (bin 1) x4 with y mean 0.25 -> gap 0.10
    #                p=0.85 (bin 8) x6 with y mean 1.00 -> gap 0.15
    # ECE = 0.4*0.10 + 0.6*0.15 = 0.13
    p = np.array([0.15] * 4 + [0.85] * 6)
    y = np.array([1, 0, 0, 0] + [1] * 6, dtype=float)
    assert expected_calibration_error(p, y) == pytest.approx(0.13, abs=1e-12)


def test_perfect_calibration_has_zero_ece() -> None:
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0] + [1, 1, 1, 0], dtype=float)
    assert expected_calibration_error(p, y) == pytest.approx(0.0, abs=1e-12)


def test_reliability_table_omits_empty_bins_and_bins_one_into_the_last() -> None:
    df = pl.DataFrame(
        {
            "trade_date": [1] * 5,
            "p": [0.05, 0.05, 0.55, 1.0, 1.0],
            "y": [0, 1, 1, 1, 0],
        }
    )
    table = reliability_table(df, pred_col="p", y_col="y")
    assert table["bin"].to_list() == [0, 5, 9]  # 1.0 must not open an 11th bin
    row = table.filter(pl.col("bin") == 0)
    assert row["n"][0] == 2
    assert row["p_mean"][0] == pytest.approx(0.05)
    assert row["y_mean"][0] == pytest.approx(0.5)
    assert row["gap"][0] == pytest.approx(0.05 - 0.5)
    assert table.filter(pl.col("bin") == 9)["y_mean"][0] == pytest.approx(0.5)


# --- precision@k / lift@k -----------------------------------------------------


def test_precision_and_lift_at_k_are_per_date_then_averaged() -> None:
    df = pl.DataFrame(
        {
            "trade_date": [1, 1, 1, 1, 2, 2, 2, 2],
            "ticker": ["A", "B", "C", "D"] * 2,
            "p": [0.9, 0.8, 0.2, 0.1, 0.9, 0.8, 0.2, 0.1],
            # date 1: top-2 = {A,B} both positive -> 1.0 ; base rate 0.5
            # date 2: top-2 = {A,B} one positive  -> 0.5 ; base rate 0.25
            "y": [1, 1, 0, 0, 1, 0, 0, 0],
        }
    )
    rep = classification_report(df, pred_col="p", y_col="y", k=2)
    assert rep.precision_at_k == pytest.approx(0.75)
    assert rep.base_rate == pytest.approx(0.375)
    assert rep.lift_at_k == pytest.approx(0.75 / 0.375)


def test_k_larger_than_the_cross_section_takes_every_name() -> None:
    df = pl.DataFrame({"trade_date": [1, 1], "ticker": ["A", "B"], "p": [0.9, 0.1], "y": [1, 0]})
    rep = classification_report(df, pred_col="p", y_col="y", k=100)
    assert rep.precision_at_k == pytest.approx(0.5)
    assert rep.lift_at_k == pytest.approx(1.0)  # no selection, no lift


# --- input contract ----------------------------------------------------------


def test_a_rank_score_is_rejected_rather_than_scored() -> None:
    df = pl.DataFrame({"trade_date": [1, 1], "ticker": ["A", "B"], "p": [1.4, -0.2], "y": [1, 0]})
    with pytest.raises(ValueError, match="probability"):
        classification_report(df, pred_col="p", y_col="y")


def test_a_three_class_label_is_rejected() -> None:
    df = pl.DataFrame(
        {"trade_date": [1, 1, 1], "ticker": ["A", "B", "C"], "p": [0.2, 0.5, 0.8], "y": [-1, 0, 1]}
    )
    with pytest.raises(ValueError, match="binary"):
        classification_report(df, pred_col="p", y_col="y")


def test_unresolved_rows_drop_out_instead_of_counting_as_negatives() -> None:
    df = pl.DataFrame(
        {
            "trade_date": [1, 1, 1],
            "ticker": ["A", "B", "C"],
            "p": [0.9, 0.1, 0.5],
            "y": [1, 0, None],  # C's horizon has not closed yet
        }
    )
    rep = classification_report(df, pred_col="p", y_col="y", k=2)
    assert rep.n_obs == 2


# --- the tau portfolio (`03` §2.3) -------------------------------------------


def _tau_frame() -> pl.DataFrame:
    """4 sessions; horizon 2 -> grid = sessions 1 and 3."""
    rows = {
        # (date, ticker): (p, realized)
        (1, "A"): (0.9, 0.10),
        (1, "B"): (0.7, 0.02),
        (1, "C"): (0.3, -0.05),
        (2, "A"): (0.9, 0.00),
        (2, "B"): (0.1, 0.00),
        (2, "C"): (0.1, 0.00),
        (3, "A"): (0.65, 0.04),
        (3, "B"): (0.10, 0.30),
        (3, "C"): (0.55, 0.00),
        (4, "A"): (0.9, 0.00),
        (4, "B"): (0.9, 0.00),
        (4, "C"): (0.9, 0.00),
    }
    return pl.DataFrame(
        {
            "trade_date": [d for (d, _t) in rows],
            "ticker": [t for (_d, t) in rows],
            "p": [v[0] for v in rows.values()],
            "raw_label_2d": [v[1] for v in rows.values()],
        }
    )


def test_threshold_membership_takes_everything_at_or_above_tau() -> None:
    got = threshold_membership(_tau_frame(), pred_col="p", tau=0.6)
    assert got[1] == {"A", "B"}
    assert got[3] == {"A"}


def test_threshold_report_nets_turnover_cost_against_the_grid_return() -> None:
    df = _tau_frame()
    rep = threshold_economic_report(
        df, pred_col="p", realized_col="raw_label_2d", horizon=2, tau=0.6
    )
    assert rebalance_grid(df["trade_date"].to_list(), 2) == [1, 3]
    assert rep.n_rebalances == 2
    # grid 1 holds {A,B} -> mean(0.10, 0.02) = 0.06 ; grid 3 holds {A} -> 0.04
    assert rep.grid_mean_return == pytest.approx((0.06 + 0.04) / 2)
    # turnover between {A,B} and {A}: 1 - 1/2 = 0.5
    assert rep.turnover == pytest.approx(0.5)
    assert rep.cost_adjusted_return == pytest.approx(0.05 - 0.5 * 60 / 10_000)
    assert rep.mean_names_held == pytest.approx(1.5)
    assert rep.min_names_held == 1
    assert rep.n_rebalances_cash == 0


def test_a_grid_date_with_nothing_above_tau_is_held_in_cash_not_skipped() -> None:
    df = _tau_frame()
    rep = threshold_economic_report(
        df, pred_col="p", realized_col="raw_label_2d", horizon=2, tau=0.85
    )
    # grid 1: {A} -> 0.10 ; grid 3: nothing clears 0.85 -> cash, 0.0
    assert rep.n_rebalances_cash == 1
    assert rep.n_rebalances_held == 1
    assert rep.min_names_held == 0
    assert rep.grid_mean_return == pytest.approx(0.05)
    # a single snapshot on the grid leaves turnover undefined, and an undefined
    # turnover must not silently become a zero cost.
    assert math.isnan(rep.turnover)
    assert rep.cost_adjusted_return == pytest.approx(0.05)


def test_a_slice_with_no_resolved_labels_reports_nan_instead_of_crashing() -> None:
    """A fold whose labels have not closed must not take the runner down."""
    df = pl.DataFrame(
        {"trade_date": [1, 1], "ticker": ["A", "B"], "p": [0.5, 0.6], "y": [None, None]},
        schema_overrides={"y": pl.Int8},
    )
    rep = classification_report(df, pred_col="p", y_col="y")
    assert rep.n_obs == 0
    assert math.isnan(rep.log_loss)
    assert math.isnan(rep.auc_pooled)
    assert math.isnan(rep.precision_at_k)
    assert reliability_table(df, pred_col="p", y_col="y").height == 0
