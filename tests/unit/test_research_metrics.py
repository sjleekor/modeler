from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from modeler.etl.metrics import (
    CostModel,
    benjamini_hochberg,
    choose_nw_lag,
    daily_market_weighted_ic,
    daily_market_weighted_spread,
    decile_membership,
    drawdown_stats,
    economic_report,
    exact_binomial_sign_test_p,
    krx_tick_size,
    market_weight_means,
    n_hac_pairs,
    newey_west_tstat,
    newey_west_tstat_legacy,
    per_date_market_quantile_spread,
    per_date_market_rank_ic,
    per_name_cost_bps,
    portfolio_turnover,
    raw_vs_rank_quantile_spread,
    rebalance_grid,
    topk_economic_report,
    topk_hysteresis_report,
    topk_membership,
    topk_rebalance_series,
    two_sided_normal_p,
)


def test_market_ic_is_not_double_counted_by_date() -> None:
    df = pl.DataFrame(
        {
            "trade_date": [1, 1, 1, 1, 2, 2],
            "market": ["KOSPI", "KOSPI", "KOSDAQ", "KOSDAQ", "KOSPI", "KOSPI"],
            "pred": [1, 2, 1, 2, 2, 1],
            "realized": [1, 2, 2, 1, 1, 2],
        }
    )
    market = per_date_market_rank_ic(df, pred_col="pred", realized_col="realized")
    daily = daily_market_weighted_ic(market)
    assert market.height == 3
    assert daily.height == 2
    assert daily.filter(pl.col("trade_date") == 1).height == 1


def test_native_rank_ic_matches_legacy_across_contract_edge_cases() -> None:
    # Covers ties, a single-market date, NaN/inf filtering, the exact
    # min_names boundary, a constant group, and shuffled input order.
    frame = pl.DataFrame(
        {
            "trade_date": [1] * 4 + [2] * 3 + [3] * 3 + [4] + [5] * 4,
            "market": ["KOSPI"] * 4 + ["KOSDAQ"] * 3 + ["KOSPI"] * 3 + ["KOSPI"] + ["KOSPI"] * 4,
            "pred": [
                1.0,
                1.0,
                2.0,
                3.0,
                1.0,
                2.0,
                3.0,
                4.0,
                4.0,
                4.0,
                1.0,
                2.0,
                float("nan"),
                float("inf"),
                3.0,
            ],
            "realized": [
                1.0,
                2.0,
                2.0,
                3.0,
                3.0,
                2.0,
                1.0,
                4.0,
                4.0,
                4.0,
                3.0,
                2.0,
                1.0,
                2.0,
                float("nan"),
            ],
        }
    ).sample(fraction=1.0, shuffle=True, seed=17)
    legacy = per_date_market_rank_ic(
        frame, pred_col="pred", realized_col="realized", min_names=2, engine="legacy"
    ).sort(["trade_date", "market"])
    native = per_date_market_rank_ic(
        frame, pred_col="pred", realized_col="realized", min_names=2, engine="polars_native_v1"
    ).sort(["trade_date", "market"])

    assert native.select(["trade_date", "market", "n"]).equals(
        legacy.select(["trade_date", "market", "n"])
    )
    for left, right in zip(legacy["rank_ic"], native["rank_ic"]):
        if math.isnan(left):
            assert math.isnan(right)
        else:
            assert right == pytest.approx(left, abs=1e-12)


def test_native_rank_ic_nans_a_constant_cross_section_at_realistic_sizes() -> None:
    """I13: Polars' Spearman returns a spurious number for a constant column at
    some group sizes and NaN at others (measured: NaN at n=50, a number at
    n=745). A constant cross-section carries no rank information, so the
    correlation is undefined and the date has to drop out.

    The edge-case test above already had a constant group — with three rows,
    which is a size where Polars happens to agree. That is why the defect
    survived: the size that matters is the one a real market day has.
    """
    rng = np.random.default_rng(11)
    for n in (50, 300, 745, 1000, 1581):
        frame = pl.DataFrame(
            {
                "trade_date": [1] * n,
                "market": ["KOSPI"] * n,
                "pred": np.zeros(n),
                "realized": rng.normal(size=n),
            }
        )
        for engine in ("legacy", "polars_native_v1"):
            out = per_date_market_rank_ic(
                frame, pred_col="pred", realized_col="realized", min_names=20, engine=engine
            )
            assert out.height == 1, (engine, n)
            assert math.isnan(out["rank_ic"][0]), f"{engine} n={n} gave {out['rank_ic'][0]}"


def test_native_rank_ic_nans_a_constant_realized_cross_section() -> None:
    """The same guard on the other side: an all-tied label is equally
    undefined, and `_spearman` checks both."""
    rng = np.random.default_rng(12)
    n = 745
    frame = pl.DataFrame(
        {
            "trade_date": [1] * n,
            "market": ["KOSPI"] * n,
            "pred": rng.normal(size=n),
            "realized": np.zeros(n),
        }
    )
    for engine in ("legacy", "polars_native_v1"):
        out = per_date_market_rank_ic(
            frame, pred_col="pred", realized_col="realized", min_names=20, engine=engine
        )
        assert math.isnan(out["rank_ic"][0]), engine


def test_native_rank_ic_matches_legacy_on_randomized_market_days() -> None:
    """Randomized parity at market-day scale, including whole cross-sections
    that are constant — the shape a count feature has at the start of its
    history, which is where I13 actually bit."""
    rng = np.random.default_rng(2026)
    rows: list[dict] = []
    for date in range(1, 41):
        for market, size in (
            ("KOSPI", int(rng.integers(300, 900))),
            ("KOSDAQ", int(rng.integers(300, 900))),
        ):
            kind = date % 4
            if kind == 0:
                pred = np.zeros(size)  # 전부 동점
            elif kind == 1:
                pred = rng.integers(0, 3, size).astype(float)  # 심한 동점
            elif kind == 2:
                pred = np.zeros(size)
                pred[: max(1, size // 50)] = 1.0  # 거의 전부 동점
            else:
                pred = rng.normal(size=size)  # 동점 없음
            rows.extend(
                {"trade_date": date, "market": market, "pred": float(p), "realized": float(r)}
                for p, r in zip(pred, rng.normal(size=size), strict=True)
            )
    frame = pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=5)

    legacy = per_date_market_rank_ic(
        frame, pred_col="pred", realized_col="realized", min_names=20, engine="legacy"
    ).sort(["trade_date", "market"])
    native = per_date_market_rank_ic(
        frame, pred_col="pred", realized_col="realized", min_names=20, engine="polars_native_v1"
    ).sort(["trade_date", "market"])

    assert native.select(["trade_date", "market", "n"]).equals(
        legacy.select(["trade_date", "market", "n"])
    )
    n_nan = 0
    for left, right in zip(legacy["rank_ic"], native["rank_ic"]):
        if math.isnan(left):
            n_nan += 1
            assert math.isnan(right)
        else:
            assert right == pytest.approx(left, abs=1e-12)
    # The constant days must actually be in the sample, or this proves nothing.
    assert n_nan >= 20


def test_gap_aware_newey_west_native_matches_legacy_randomized() -> None:
    rng = np.random.default_rng(20260823)
    for n in (5, 11, 31):
        for lag in (0, 1, 7, 19):
            sessions = np.sort(rng.choice(np.arange(1, 160), size=n, replace=False))
            values = rng.normal(size=n)
            expected = newey_west_tstat_legacy(values, sessions, lag)
            actual = newey_west_tstat(values, sessions, lag)
            if math.isnan(expected):
                assert math.isnan(actual)
            else:
                assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_nw_uses_session_gap_not_compressed_array_position() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    dense = newey_west_tstat(values, [1, 2, 3, 4], lag=1)
    gapped = newey_west_tstat(values, [1, 3, 4, 6], lag=1)
    assert dense == pytest.approx(4.0)
    assert gapped == pytest.approx(4.5883146774)


def test_raw_and_rank_quantile_spreads_are_finite() -> None:
    df = pl.DataFrame(
        {
            "trade_date": [1] * 20,
            "raw": list(range(20)),
            "rank": [i / 19 for i in range(20)],
        }
    )
    result = raw_vs_rank_quantile_spread(df, rank_col="rank", raw_col="raw", min_names=20)
    assert result.height == 1
    assert result["raw_score_spread"][0] == pytest.approx(result["rank_score_spread"][0])


def test_lag_and_bh_contract() -> None:
    assert choose_nw_lag(scan_type="cum", horizon=20) == 19
    assert choose_nw_lag(scan_type="bucket", bucket_width=5) == 4
    q = benjamini_hochberg([0.001, 0.02, 0.5])
    assert q[0] <= q[1] <= q[2]
    assert q[0] == pytest.approx(0.003)


def test_exact_binomial_sign_test_matches_known_values() -> None:
    # cross-checked against scipy.stats.binomtest(..., alternative="greater")
    assert exact_binomial_sign_test_p(4, 4) == pytest.approx(0.0625)
    assert exact_binomial_sign_test_p(3, 4) == pytest.approx(0.3125)
    assert exact_binomial_sign_test_p(5, 10) == pytest.approx(0.623046875)


def test_exact_binomial_sign_test_edge_cases() -> None:
    assert exact_binomial_sign_test_p(0, 0) != exact_binomial_sign_test_p(0, 0)  # nan
    assert exact_binomial_sign_test_p(0, 5) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        exact_binomial_sign_test_p(6, 5)


def test_two_sided_normal_p_matches_known_value() -> None:
    assert two_sided_normal_p(1.96) == pytest.approx(0.05, abs=1e-4)
    assert two_sided_normal_p(0.0) == pytest.approx(1.0)
    assert math.isnan(two_sided_normal_p(float("nan")))


def test_n_hac_pairs_counts_within_lag_distance_not_array_position() -> None:
    # dense: sessions 1..4, lag=1 -> 3 adjacent pairs
    assert n_hac_pairs([1, 2, 3, 4], lag=1) == 3
    # a multi-year gap (session 3 -> 100) breaks the lag=1 adjacency there
    assert n_hac_pairs([1, 2, 3, 100, 101], lag=1) == 3
    assert n_hac_pairs([1, 2, 3, 4], lag=0) == 0


def test_market_weight_means_reflects_n_weighted_composition() -> None:
    df = pl.DataFrame(
        {
            "trade_date": [1, 1, 2],
            "market": ["KOSPI", "KOSDAQ", "KOSPI"],
            "pred": [1, 2, 1],
            "realized": [1, 2, 1],
        }
    )
    market_ic = per_date_market_rank_ic(df, pred_col="pred", realized_col="realized", min_names=1)
    weights = market_weight_means(market_ic)
    # date 1: KOSPI/KOSDAQ each 1 name -> kospi weight 0.5; date 2: KOSPI-only -> weight 1.0
    assert weights["kospi_weight_mean"] == pytest.approx(0.75)
    assert weights["kosdaq_weight_mean"] == pytest.approx(0.25)


def test_market_quantile_spread_matches_raw_vs_rank_identity_per_market() -> None:
    # Same date×market cross-section as the raw/rank identity test, but through
    # the market-aware (§4.3) path used by the Phase A scan.
    df = pl.DataFrame(
        {
            "trade_date": [1] * 20,
            "market": ["KOSPI"] * 20,
            "raw": [float(i) for i in range(20)],
        }
    )
    market_spread = per_date_market_quantile_spread(
        df, feature_col="raw", raw_label_col="raw", min_names=20
    )
    daily = daily_market_weighted_spread(market_spread)
    assert daily.height == 1
    # rank/20 >= 0.8 (>=16th of 20, ties-free) keeps {15..19}; <= 0.2 keeps {0..3}.
    assert daily["spread"][0] == pytest.approx(sum(range(15, 20)) / 5 - sum(range(0, 4)) / 4)


def test_market_quantile_spread_drops_thin_cross_sections() -> None:
    df = pl.DataFrame(
        {
            "trade_date": [1, 1, 1],
            "market": ["KOSPI"] * 3,
            "raw": [1.0, 2.0, 3.0],
        }
    )
    result = per_date_market_quantile_spread(
        df, feature_col="raw", raw_label_col="raw", min_names=50
    )
    assert result.is_empty()


def test_rebalance_grid_spacing() -> None:
    assert rebalance_grid(list(range(1, 11)), horizon=3) == [1, 4, 7, 10]
    with pytest.raises(ValueError):
        rebalance_grid([1, 2, 3], horizon=0)


def test_portfolio_turnover_zero_when_membership_identical() -> None:
    membership = {1: {"A", "B"}, 2: {"A", "B"}, 3: {"A", "B"}}
    assert portfolio_turnover(membership, [1, 2, 3]) == pytest.approx(0.0)


def test_portfolio_turnover_one_when_fully_disjoint() -> None:
    membership = {1: {"A", "B"}, 2: {"C", "D"}}
    assert portfolio_turnover(membership, [1, 2]) == pytest.approx(1.0)


def test_portfolio_turnover_skips_missing_snapshots() -> None:
    # date 2 never had enough names to form a top-decile membership entry.
    membership = {1: {"A"}, 3: {"A"}}
    assert portfolio_turnover(membership, [1, 2, 3]) == pytest.approx(0.0)


def test_decile_membership_takes_only_the_upper_tail() -> None:
    df = pl.DataFrame(
        {
            "trade_date": [1, 1, 1, 1],
            "ticker": ["A", "B", "C", "D"],
            "pred": [4.0, 3.0, 2.0, 1.0],
        }
    )
    membership = decile_membership(df, pred_col="pred", q=0.9)
    assert membership == {1: {"A"}}


def test_economic_report_nets_turnover_cost_against_grid_spread() -> None:
    # 6 dates, horizon=2 -> rebalance grid = [1, 3, 5]. Top-decile (q=0.9 of 4
    # names, so a single top name per date) alternates A/B/A across the grid ->
    # full turnover (1.0) between every consecutive rebalance. Dates 2/4/6 only
    # pad out the calendar so the grid spacing lands on 1/3/5; their pred/
    # realized values are irrelevant and set to a flat, non-tied baseline.
    # (ticker, pred) per date; the realized return of the top name is the only
    # non-zero realized value that date.
    preds_per_date = {
        1: {"A": 4.0, "B": 1.0, "C": 2.0, "D": 3.0},  # top = A
        2: {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},  # not on the grid
        3: {"A": 1.0, "B": 4.0, "C": 2.0, "D": 3.0},  # top = B
        4: {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},  # not on the grid
        5: {"A": 4.0, "B": 1.0, "C": 2.0, "D": 3.0},  # top = A
        6: {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},  # not on the grid
    }
    top_realized = {1: ("A", 0.10), 3: ("B", 0.05), 5: ("A", 0.08)}
    rows = []
    for d, preds in preds_per_date.items():
        top_name, top_ret = top_realized.get(d, (None, 0.0))
        for name, pred in preds.items():
            realized = top_ret if name == top_name else 0.0
            rows.append({"trade_date": d, "ticker": name, "pred": pred, "realized": realized})
    df = pl.DataFrame(rows)

    report = economic_report(
        df, pred_col="pred", realized_col="realized", horizon=2, cost_bps_roundtrip=100.0
    )

    assert report.n_rebalances == 3
    assert report.turnover == pytest.approx(1.0)
    assert report.grid_top_decile_spread == pytest.approx((0.10 + 0.05 + 0.08) / 3)
    assert report.cost_adjusted_spread == pytest.approx(report.grid_top_decile_spread - 0.01)


def test_economic_report_zero_turnover_when_top_decile_never_changes() -> None:
    rows = []
    for d in range(1, 5):
        for name, pred, realized in (("A", 4.0, 0.2), ("B", 1.0, 0.0)):
            rows.append({"trade_date": d, "ticker": name, "pred": pred, "realized": realized})
    df = pl.DataFrame(rows)

    report = economic_report(
        df, pred_col="pred", realized_col="realized", horizon=1, cost_bps_roundtrip=100.0
    )

    assert report.turnover == pytest.approx(0.0)
    assert report.cost_adjusted_spread == pytest.approx(report.grid_top_decile_spread)


def _topk_frame(preds_per_date: dict, realized_per_date: dict | None = None) -> pl.DataFrame:
    rows = []
    for d, preds in preds_per_date.items():
        realized = (realized_per_date or {}).get(d, {})
        for name, pred in preds.items():
            rows.append(
                {
                    "trade_date": d,
                    "ticker": name,
                    "pred": pred,
                    "realized": realized.get(name, 0.0),
                }
            )
    return pl.DataFrame(rows)


def test_topk_membership_takes_exactly_k_names_per_date() -> None:
    df = _topk_frame({1: {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}})
    assert topk_membership(df, pred_col="pred", k=2) == {1: {"A", "B"}}


def test_topk_membership_never_buys_an_unscored_name() -> None:
    df = pl.DataFrame(
        [
            {"trade_date": 1, "ticker": "A", "pred": 4.0, "realized": 0.0},
            {"trade_date": 1, "ticker": "B", "pred": None, "realized": 0.0},
            {"trade_date": 1, "ticker": "C", "pred": 2.0, "realized": 0.0},
        ]
    )
    assert topk_membership(df, pred_col="pred", k=2) == {1: {"A", "C"}}


def test_topk_economic_report_nets_turnover_cost_against_the_buy_list_return() -> None:
    # 4 dates, horizon=2 -> grid = [1, 3]. k=2, and the list swaps one of its two
    # names between rebalances -> turnover 0.5.
    df = _topk_frame(
        {
            1: {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},  # holds A, B
            2: {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},  # not on the grid
            3: {"A": 4.0, "B": 1.0, "C": 3.0, "D": 2.0},  # holds A, C
            4: {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},  # not on the grid
        },
        {
            1: {"A": 0.10, "B": 0.06},
            3: {"A": 0.04, "C": 0.02},
        },
    )

    report = topk_economic_report(
        df, pred_col="pred", realized_col="realized", horizon=2, k=2, cost_bps_roundtrip=100.0
    )

    assert report.k == 2
    assert report.n_rebalances == 2
    assert report.mean_names_held == pytest.approx(2.0)
    assert report.turnover == pytest.approx(0.5)
    assert report.grid_topk_mean_return == pytest.approx((0.08 + 0.03) / 2)
    assert report.cost_adjusted_return == pytest.approx(report.grid_topk_mean_return - 0.005)


def test_topk_rebalance_series_reproduces_the_report() -> None:
    # Same frame as the report test above: grid = [1, 3], k=2, one name swapped.
    # The series is the report's inputs before they were averaged, so its means
    # have to land on the report's floats exactly — this is what lets drawdown
    # and CSCV read the series instead of re-deriving the economics.
    df = _topk_frame(
        {
            1: {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
            2: {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},
            3: {"A": 4.0, "B": 1.0, "C": 3.0, "D": 2.0},
            4: {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},
        },
        {
            1: {"A": 0.10, "B": 0.06},
            3: {"A": 0.04, "C": 0.02},
        },
    )
    kwargs = dict(
        pred_col="pred", realized_col="realized", horizon=2, k=2, cost_bps_roundtrip=100.0
    )

    report = topk_economic_report(df, **kwargs)
    series = topk_rebalance_series(df, **kwargs)

    assert series.height == report.n_rebalances
    assert series["rebalance_date"].to_list() == [1, 3]
    assert series["n_held"].to_list() == [2, 2]
    # first row has no earlier list to trade against, so no turnover and no cost
    assert series["turnover"].to_list() == [None, pytest.approx(0.5)]
    assert series["net_return"][0] == pytest.approx(series["gross_return"][0])

    gross = series["gross_return"].drop_nulls().to_numpy()
    turns = series["turnover"].drop_nulls().to_numpy()
    assert float(gross.mean()) == pytest.approx(report.grid_topk_mean_return)
    assert float(turns.mean()) == pytest.approx(report.turnover)


def test_topk_rebalance_series_keeps_a_held_but_unscored_rebalance() -> None:
    # Date 3's holdings have no closed label yet. The row stays (it is held, and
    # it still costs turnover) but carries a null return rather than a zero,
    # which would otherwise drag a drawdown path toward flat.
    rows = [
        {"trade_date": 1, "ticker": "A", "pred": 4.0, "realized": 0.10},
        {"trade_date": 1, "ticker": "B", "pred": 3.0, "realized": 0.20},
        {"trade_date": 1, "ticker": "C", "pred": 1.0, "realized": 0.0},
        {"trade_date": 2, "ticker": "A", "pred": 1.0, "realized": 0.0},
        {"trade_date": 2, "ticker": "B", "pred": 2.0, "realized": 0.0},
        {"trade_date": 2, "ticker": "C", "pred": 4.0, "realized": 0.0},
        {"trade_date": 3, "ticker": "A", "pred": 4.0, "realized": None},
        {"trade_date": 3, "ticker": "B", "pred": 3.0, "realized": None},
        {"trade_date": 3, "ticker": "C", "pred": 1.0, "realized": None},
    ]
    series = topk_rebalance_series(
        pl.DataFrame(rows),
        pred_col="pred",
        realized_col="realized",
        horizon=2,
        k=2,
        cost_bps_roundtrip=100.0,
    )

    assert series["rebalance_date"].to_list() == [1, 3]
    assert series["n_held"].to_list() == [2, 2]
    assert series["n_scored"].to_list() == [2, 0]
    assert series["gross_return"][1] is None
    assert series["net_return"][1] is None
    assert series["turnover"][1] == pytest.approx(0.0)


def test_topk_report_holds_names_whose_label_has_not_closed_yet() -> None:
    # B is held on both rebalances but its label is still null on date 3, so it
    # counts toward turnover and mean_names_held, not toward the realized mean.
    rows = [
        {"trade_date": 1, "ticker": "A", "pred": 4.0, "realized": 0.10},
        {"trade_date": 1, "ticker": "B", "pred": 3.0, "realized": 0.20},
        {"trade_date": 1, "ticker": "C", "pred": 1.0, "realized": 0.0},
        {"trade_date": 2, "ticker": "A", "pred": 4.0, "realized": 0.0},
        {"trade_date": 2, "ticker": "B", "pred": 3.0, "realized": None},
        {"trade_date": 2, "ticker": "C", "pred": 1.0, "realized": 0.0},
    ]
    df = pl.DataFrame(rows, strict=False)

    report = topk_economic_report(
        df, pred_col="pred", realized_col="realized", horizon=1, k=2, cost_bps_roundtrip=0.0
    )

    assert report.turnover == pytest.approx(0.0)
    assert report.mean_names_held == pytest.approx(2.0)
    assert report.mean_names_scored == pytest.approx(1.5)
    assert report.grid_topk_mean_return == pytest.approx((0.15 + 0.0) / 2)


def test_topk_rejects_a_non_positive_k() -> None:
    df = _topk_frame({1: {"A": 1.0}})
    with pytest.raises(ValueError, match="k must be >= 1"):
        topk_membership(df, pred_col="pred", k=0)


# --- R3: drawdown of the rebalance path -------------------------------------


def test_drawdown_measures_the_fall_and_how_long_it_lasted() -> None:
    # +10%, -20%, -10%, +5%, +30%: the peak is step 1's 1.10 and the path never
    # gets back to it — 1.10*0.8*0.9*1.05*1.30 = 1.081. So every later step is
    # underwater, which is the case a "how deep" number alone would hide.
    stats = drawdown_stats([0.10, -0.20, -0.10, 0.05, 0.30])
    assert stats.max_drawdown == pytest.approx(0.792 / 1.10 - 1)  # trough at step 3
    assert stats.longest_underwater == 4


def test_drawdown_skips_a_rebalance_with_no_return() -> None:
    # A held-but-unscored rebalance is missing, not a flat step: reading it as
    # zero would end the drawdown clock early.
    assert drawdown_stats([0.10, None, -0.20]) == drawdown_stats([0.10, -0.20])


def test_drawdown_of_a_path_that_only_rises_is_zero() -> None:
    stats = drawdown_stats([0.01, 0.02, 0.03])
    assert stats.max_drawdown == 0.0
    assert stats.longest_underwater == 0


# --- T1-2: per-name round-trip cost -----------------------------------------


def test_krx_tick_size_follows_the_price_bands() -> None:
    # each band is [low, high): 1,000 is the first price that pays the 5 tick.
    close = np.array([999.0, 1_000.0, 4_999.0, 5_000.0, 49_999.0, 100_000.0, 600_000.0])
    assert krx_tick_size(close).tolist() == [1, 5, 5, 10, 50, 500, 1_000]


def test_per_name_cost_is_higher_for_the_thinner_name() -> None:
    # Same price and sigma; B trades a tenth of A's value, so the same order is
    # ten times the participation and sqrt(10) times the impact.
    df = pl.DataFrame(
        {
            "px_turnover_ma20": [1e10, 1e9],
            "px_vol_20d": [0.02, 0.02],
            "close": [10_000.0, 10_000.0],
        }
    )
    cost = per_name_cost_bps(df, model=CostModel(capital_krw=1e9, k_names=100))
    a, b = cost.to_list()
    # linear part: 1 tick at close 10,000 is 50 (the 10,000-50,000 band), so
    # 50/10,000 = 50bp of spread, + 2*1.5bp fee + 15bp tax.
    linear = 50.0 + 3.0 + 15.0
    assert a - linear == pytest.approx(2 * 1.0 * 0.02 * math.sqrt(1e7 / 1e10) * 10_000)
    assert (b - linear) / (a - linear) == pytest.approx(math.sqrt(10.0))


def test_per_name_cost_is_null_where_it_cannot_be_known() -> None:
    df = pl.DataFrame(
        {
            "px_turnover_ma20": [0.0, None, 1e9],
            "px_vol_20d": [0.02, 0.02, None],
            "close": [10_000.0, 10_000.0, 10_000.0],
        }
    )
    got = per_name_cost_bps(df, model=CostModel(capital_krw=1e9)).to_list()
    assert all(v is None or math.isnan(v) for v in got)


def test_cost_model_replaces_the_flat_number_in_the_report() -> None:
    df = _topk_frame(
        {1: {"A": 4.0, "B": 3.0, "C": 2.0}, 2: {"A": 4.0, "B": 3.0, "C": 2.0}},
        {1: {"A": 0.10, "B": 0.06}, 2: {"A": 0.04, "B": 0.02}},
    ).with_columns(
        pl.lit(1e10).alias("px_turnover_ma20"),
        pl.lit(0.02).alias("px_vol_20d"),
        pl.lit(10_000.0).alias("close"),
    )
    kwargs = dict(pred_col="pred", realized_col="realized", horizon=1, k=2)
    flat = topk_economic_report(df, **kwargs, cost_bps_roundtrip=60.0)
    modelled = topk_economic_report(df, **kwargs, cost_model=CostModel(capital_krw=1e9))

    assert flat.cost_bps_roundtrip == 60.0
    assert modelled.cost_bps_roundtrip != 60.0
    # same list, same returns — only the cost moved
    assert modelled.grid_topk_mean_return == pytest.approx(flat.grid_topk_mean_return)
    assert modelled.turnover == pytest.approx(flat.turnover)


# --- T1-6: hysteresis --------------------------------------------------------


def test_hysteresis_with_no_band_is_the_plain_topk_rule() -> None:
    """k_out == k_in must reproduce topk_economic_report exactly."""
    df = _topk_frame(
        {
            1: {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
            2: {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},
            3: {"A": 4.0, "B": 1.0, "C": 3.0, "D": 2.0},
            4: {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},
        },
        {1: {"A": 0.10, "B": 0.06}, 3: {"A": 0.04, "C": 0.02}},
    )
    kwargs = dict(pred_col="pred", realized_col="realized", horizon=2, cost_bps_roundtrip=100.0)
    plain = topk_economic_report(df, k=2, **kwargs)
    band = topk_hysteresis_report(df, k_in=2, k_out=2, **kwargs)

    assert band.turnover == pytest.approx(plain.turnover)
    assert band.grid_topk_mean_return == pytest.approx(plain.grid_topk_mean_return)
    assert band.cost_adjusted_return == pytest.approx(plain.cost_adjusted_return)
    assert band.n_rebalances == plain.n_rebalances


def test_hysteresis_holds_a_name_that_slipped_inside_the_band() -> None:
    # B is 2nd on date 1 and 3rd on date 3. The tight rule sells B and buys C;
    # the band buys C and *keeps* B, so the book widens to 3 instead of swapping
    # a name. Holding above k_in is the documented behaviour — the band trades
    # less, it does not trade nothing.
    df = _topk_frame(
        {
            1: {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0},
            2: {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},  # off the grid
            3: {"A": 4.0, "C": 3.5, "B": 3.0, "D": 1.0},
            4: {"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0},  # off the grid
        },
        {1: {"A": 0.10, "B": 0.06}, 3: {"A": 0.04, "B": 0.02}},
    )
    kwargs = dict(pred_col="pred", realized_col="realized", horizon=2, cost_bps_roundtrip=100.0)
    tight = topk_hysteresis_report(df, k_in=2, k_out=2, **kwargs)
    band = topk_hysteresis_report(df, k_in=2, k_out=3, **kwargs)

    assert tight.turnover == pytest.approx(0.5)  # B out, C in
    assert band.turnover == pytest.approx(1 / 3)  # C in, nothing sold
    assert band.mean_names_held > tight.mean_names_held
    # same names earn the same gross here, so the whole gap is the churn
    assert band.grid_topk_mean_return == pytest.approx(tight.grid_topk_mean_return)
    assert band.cost_adjusted_return > tight.cost_adjusted_return


def test_hysteresis_rejects_a_band_that_is_not_one() -> None:
    df = _topk_frame({1: {"A": 2.0, "B": 1.0}})
    with pytest.raises(ValueError):
        topk_hysteresis_report(
            df, pred_col="pred", realized_col="realized", horizon=1, k_in=100, k_out=50
        )
