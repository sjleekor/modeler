from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from modeler.etl.metrics import (
    benjamini_hochberg,
    choose_nw_lag,
    daily_market_weighted_ic,
    daily_market_weighted_spread,
    decile_membership,
    economic_report,
    exact_binomial_sign_test_p,
    market_weight_means,
    n_hac_pairs,
    newey_west_tstat,
    newey_west_tstat_legacy,
    per_date_market_quantile_spread,
    per_date_market_rank_ic,
    portfolio_turnover,
    raw_vs_rank_quantile_spread,
    rebalance_grid,
    topk_economic_report,
    topk_membership,
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
