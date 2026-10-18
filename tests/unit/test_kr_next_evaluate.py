"""Synthetic tests for the KR-next judgment computations (19 / spec G-3..G-9, G-18, C-5)."""

from __future__ import annotations

import hashlib
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy import stats

from modeler.models._03_kr_next import evaluate as ev


def _sessions(n: int, start: date = date(2027, 1, 4)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


# --- grid -------------------------------------------------------------------


def test_grid_dates_step_and_d0():
    s = _sessions(100)
    g = ev.grid_dates(s, s[3])
    assert g == [s[3], s[23], s[43], s[63], s[83]]
    assert ev.grid_dates(s, s[0], step=50) == [s[0], s[50]]


def test_grid_dates_d0_not_session():
    with pytest.raises(ValueError):
        ev.grid_dates(_sessions(10), date(2027, 1, 2))


def test_check_dates_plain_and_with_sessions():
    plain = ev.check_dates(2027, years=2)
    assert plain == [date(2027, 1, 15), date(2027, 7, 15), date(2028, 1, 15), date(2028, 7, 15)]
    s = [date(2027, 1, 18), date(2027, 7, 15), date(2028, 1, 14), date(2028, 1, 17)]
    # 2027-01-15 is a Friday but not in this synthetic session list -> next session
    assert ev.check_dates(2027, years=2, sessions=s) == [
        date(2027, 1, 18),
        date(2027, 7, 15),
        date(2028, 1, 17),
    ]


# --- weights ----------------------------------------------------------------


def test_cap_weights_water_filling():
    mcap = {f"t{i}": 1.0 for i in range(20)}
    mcap["big"] = 1000.0
    w = ev.cap_weights(mcap)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["big"] == pytest.approx(0.10)
    assert max(w.values()) <= 0.10 + 1e-12
    assert w["t0"] == pytest.approx(0.9 / 20)


def test_cap_weights_cascading_cap():
    # second pass: after capping the biggest, the next one now exceeds as well
    mcap = {"a": 100.0, "b": 50.0, **{f"x{i}": 1.0 for i in range(10)}}
    w = ev.cap_weights(mcap, cap=0.2)
    assert w["a"] == pytest.approx(0.2) and w["b"] == pytest.approx(0.2)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["x0"] == pytest.approx(0.6 / 10)


def test_cap_weights_infeasible_is_equal():
    w = ev.cap_weights({"a": 5.0, "b": 1.0, "c": 1.0})
    assert w == {k: pytest.approx(1 / 3) for k in "abc"}  # 3 * 10% < 1
    assert ev.cap_weights({}) == {}


def test_cap_weights_exactly_ten_names_all_at_cap():
    w = ev.cap_weights({f"t{i}": float(i + 1) for i in range(10)})
    assert all(v == pytest.approx(0.1) for v in w.values())


def test_cap_weights_null_and_zero_mcap():
    w = ev.cap_weights({**{f"t{i}": 1.0 for i in range(10)}, "n": None, "z": 0.0})
    assert w["n"] == 0.0 and w["z"] == 0.0
    assert sum(w.values()) == pytest.approx(1.0)
    allzero = ev.cap_weights({f"t{i}": 0.0 for i in range(12)})
    assert allzero["t0"] == pytest.approx(1 / 12)


def test_cap_weights_series_input():
    w = ev.cap_weights(pl.Series([1.0] * 12))
    assert len(w) == 12 and sum(w.values()) == pytest.approx(1.0)


# --- basket -----------------------------------------------------------------


def _cands(n: int, **over) -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "ticker": [f"T{i:03d}" for i in range(n)],
            "score": [float(n - i) for i in range(n)],  # T000 best
            "tv20": [1.0] * n,
            "mcap": [1.0] * n,
        }
    )
    return df


def test_build_basket_first_build_top_k():
    b = ev.build_basket(None, _cands(200))
    assert set(b) == {f"T{i:03d}" for i in range(30)}
    assert sum(b.values()) == pytest.approx(1.0)
    assert max(b.values()) <= 0.10 + 1e-12


def test_build_basket_buffer_keeps_rank_60_holding():
    prev = {"T059": 0.1, "T100": 0.1}  # rank 60 stays, rank 101 > 90 goes
    b = ev.build_basket(prev, _cands(200))
    assert "T059" in b and "T100" not in b
    assert len(b) == 30
    # 29 best non-held fill the rest: T000..T028
    assert set(b) == {f"T{i:03d}" for i in range(29)} | {"T059"}


def test_build_basket_rank_out_boundary():
    b = ev.build_basket({"T089": 1.0, "T090": 1.0}, _cands(200))
    assert "T089" in b and "T090" not in b  # rank 90 kept, rank 91 dropped


def test_build_basket_tie_breaks():
    df = pl.DataFrame(
        {
            "ticker": ["B", "A", "C", "D"],
            "score": [1.0, 1.0, 1.0, 2.0],
            "tv20": [5.0, 5.0, 9.0, 0.0],
            "mcap": [1.0] * 4,
        }
    )
    b = ev.build_basket(None, df, k=3)
    assert set(b) == {"D", "C", "A"}  # D top score, C higher tv20, A before B by ticker


def test_build_basket_fewer_than_k_holds_all_and_empty():
    b = ev.build_basket({"T000": 1.0, "GONE": 1.0}, _cands(12))
    assert set(b) == {f"T{i:03d}" for i in range(12)}
    assert ev.build_basket(None, _cands(0)) == {}


# --- drift / turnover -------------------------------------------------------


def test_drift_and_turnover():
    t = {"a": 0.5, "b": 0.5}
    d = ev.drift(t, {"a": 0.2, "b": -0.2})
    assert d["a"] == pytest.approx(0.6 / 1.0) and d["b"] == pytest.approx(0.4)
    assert ev.one_way_turnover(t, d) == pytest.approx(0.1)
    assert ev.one_way_turnover(t, None) == 1.0
    # union: a name only on one side
    assert ev.one_way_turnover({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)
    assert ev.one_way_turnover(t, t) == 0.0


def test_drift_missing_return_carried_flat():
    d = ev.drift({"a": 0.5, "b": 0.5}, {"a": None})
    assert d == {"a": pytest.approx(0.5), "b": pytest.approx(0.5)}


def test_turnover_warning():
    assert not ev.turnover_warning([])
    assert ev.turnover_warning([0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
    assert not ev.turnover_warning([0.5] * 6)  # not strictly greater
    assert ev.turnover_warning([0.9])  # fewer than window: mean of what exists
    assert not ev.turnover_warning([1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # old 1.0 out of window


# --- holding returns / G ----------------------------------------------------


def _closes() -> pl.DataFrame:
    s = _sessions(6)
    rows = []
    for i, d in enumerate(s):  # A: normal
        rows.append((d, "A", 100.0 + i * 10, False))
    for i, d in enumerate(s):  # H: halted from index 3 on (close carried)
        rows.append((d, "H", 100.0 + min(i, 2) * 10, i >= 3))
    for i, d in enumerate(s[:4]):  # D: delisted after index 3
        rows.append((d, "D", 100.0 - i * 10, i == 3))  # last row halted (liquidation)
    for i, d in enumerate(s[2:], start=2):  # U: not trading at start
        rows.append((d, "U", 50.0, False))
    return pl.DataFrame(rows, schema=["trade_date", "ticker", "close", "is_halted"], orient="row")


def test_holding_returns_cases():
    s = _sessions(6)
    r = ev.holding_returns(_closes(), s[0], s[5], tickers=["A", "H", "D", "U", "ZZ"])
    assert r["A"] == pytest.approx(150 / 100 - 1)
    assert r["H"] == pytest.approx(120 / 100 - 1)  # halted at end -> last non-halt close
    assert r["D"] == pytest.approx(80 / 100 - 1)  # delisted: last non-halt, then frozen
    assert r["U"] is None  # untradable at entry
    assert r["ZZ"] is None  # unknown


def test_holding_returns_ignores_after_end_and_default_tickers():
    s = _sessions(6)
    r = ev.holding_returns(_closes(), s[0], s[2])
    assert r["A"] == pytest.approx(120 / 100 - 1)
    assert set(r) == {"A", "H", "D", "U"}
    assert r["U"] is None


def test_halted_at_start_is_untradable():
    s = _sessions(3)
    df = pl.DataFrame(
        [(s[0], "X", 10.0, True), (s[1], "X", 11.0, False), (s[2], "X", 12.0, False)],
        schema=["trade_date", "ticker", "close", "is_halted"],
        orient="row",
    )
    assert ev.holding_returns(df, s[0], s[2])["X"] is None


def test_g_point_renormalizes_and_costs():
    target = {"a": 0.5, "b": 0.3, "c": 0.2}
    ret = {"a": 0.10, "b": 0.00, "c": None}
    p = ev.g_point(target, ret, bench_ret=0.02, turnover=0.4)
    assert p["gross"] == pytest.approx((0.5 * 0.10) / 0.8)
    assert p["excluded_weight"] == pytest.approx(0.2) and p["n_excluded"] == 1
    assert p["cost"] == pytest.approx(0.0065 * 0.4)
    assert p["g"] == pytest.approx(p["gross"] - 0.02 - 0.0065 * 0.4)


def test_g_point_first_build_cost_and_empty():
    p = ev.g_point({"a": 1.0}, {"a": 0.0}, 0.0, 1.0)
    assert p["g"] == pytest.approx(-0.0065)
    e = ev.g_point({"a": 1.0}, {"a": None}, 0.01, 1.0)
    assert e["gross"] is None and e["g"] is None and e["excluded_weight"] == pytest.approx(1.0)


# --- tail -------------------------------------------------------------------


def _tail_frame(n: int = 11, with_roa: bool = True) -> pl.DataFrame:
    d = date(2027, 2, 1)
    return pl.DataFrame(
        {
            "trade_date": [d] * n,
            "market": ["KOSPI"] * n,
            "ticker": [f"T{i}" for i in range(n)],
            "px_vol_20d_v2": [float(i) for i in range(n)],
            "px_idio_vol_60d": [0.0] * n,
            "fin_roa_v2": [float(i) if with_roa else None for i in range(n)],
            "fin_capital_impaired_v2": [False] * n,
        }
    )


def test_tail_flags_percent_rank_boundaries():
    # n = 11: pct = i/10 -> vol pct >= 0.9 only i = 9, 10; ROA pct <= 0.1 only i = 0, 1
    out = ev.tail_flags(_tail_frame())
    assert out.filter(pl.col("tail1"))["ticker"].to_list() == ["T9", "T10"]
    assert out.filter(pl.col("tail2"))["ticker"].to_list() == ["T0", "T1"]
    assert out["tail"].sum() == 4


def test_tail_flags_average_ties_idio_constant_not_tail():
    # idio vol constant: average rank gives pct 0.5 everywhere, so no name is tail through it
    out = ev.tail_flags(_tail_frame())
    assert out["px_idio_vol_60d_pct"].to_list() == [pytest.approx(0.5)] * 11


def test_tail_flags_nulls_and_impaired_and_small_group():
    df = _tail_frame().with_columns(
        pl.when(pl.col("ticker") == "T10")
        .then(None)
        .otherwise(pl.col("px_vol_20d_v2"))
        .alias("px_vol_20d_v2"),
        pl.when(pl.col("ticker") == "T5")
        .then(True)
        .when(pl.col("ticker") == "T6")
        .then(None)
        .otherwise(False)
        .alias("fin_capital_impaired_v2"),
    )
    out = ev.tail_flags(df)
    t = {r["ticker"]: r for r in out.to_dicts()}
    assert t["T10"]["px_vol_20d_v2_pct"] is None and not t["T10"]["tail1"]  # NULL not tail
    assert t["T9"]["tail1"] and t["T9"]["px_vol_20d_v2_pct"] == pytest.approx(1.0)  # n = 10 now
    assert t["T5"]["tail2"] and t["T5"]["tail"]  # impaired TRUE
    assert not t["T6"]["tail2"]  # NULL impaired is not tail
    assert out["tail1"].null_count() == 0 and out["tail"].null_count() == 0
    # all-NULL ROA: no tail2 at all
    assert not ev.tail_flags(_tail_frame(with_roa=False))["tail2"].any()
    # group of one: percent rank undefined -> not tail
    one = ev.tail_flags(_tail_frame(1))
    assert one["tail"].to_list() == [False]


def test_tail_flags_groups_are_date_and_market():
    a = _tail_frame()
    b = a.with_columns(pl.lit("KOSDAQ").alias("market"))
    out = ev.tail_flags(pl.concat([a, b]))
    assert out["tail"].sum() == 8  # ranks computed per market, not pooled


def test_tail_point_means_and_exclusions():
    df = pl.DataFrame(
        {
            "tail": [True, True, True, False, False, False, False],
            "fwd_ret_c1_20d": [-0.10, -0.20, 0.5, 0.01, 0.03, 9.0, None],
            "label_exit_kind": [
                "regular",
                "terminal",
                "censored",
                "late",
                "terminal_halted",
                "ca_masked",
                "regular",
            ],
        }
    )
    p = ev.tail_point(df)
    assert p["tail_mean"] == pytest.approx(-0.15) and p["n_tail"] == 2
    assert p["rest_mean"] == pytest.approx(0.02) and p["n_rest"] == 2
    assert p["diff"] == pytest.approx(-0.17)
    assert p["excluded_tail"] == {"censored": 1}
    assert p["excluded_rest"] == {"ca_masked": 1, "null_return": 1}


def test_tail_point_empty_group_and_null_kind():
    df = pl.DataFrame(
        {
            "tail": [True, False],
            "fwd_ret_c1_20d": [0.1, 0.2],
            "label_exit_kind": [None, "regular"],
        },
        schema={"tail": pl.Boolean, "fwd_ret_c1_20d": pl.Float64, "label_exit_kind": pl.String},
    )
    p = ev.tail_point(df)
    assert p["tail_mean"] is None and p["diff"] is None
    assert p["excluded_tail"] == {"null": 1}


# --- random basket ----------------------------------------------------------


def test_date_seed_matches_spec():
    d = date(2027, 3, 5)
    assert ev.date_seed(d) == int(hashlib.sha256(b"2027-03-05").hexdigest()[:8], 16)


def _rb_inputs(n=200):
    rng = np.random.default_rng(1)
    c = pl.DataFrame(
        {"ticker": [f"T{i}" for i in range(n)], "mcap": rng.uniform(1, 10, n).tolist()}
    )
    ret = {f"T{i}": float(x) for i, x in enumerate(rng.normal(0.01, 0.1, n))}
    return c, ret


def test_random_basket_deterministic_and_percentile():
    c, ret = _rb_inputs()
    a = ev.random_basket(c, date(2027, 1, 4), ret, 0.0, model_excess=0.5)
    b = ev.random_basket(c, date(2027, 1, 4), ret, 0.0, model_excess=0.5)
    other = ev.random_basket(c, date(2027, 1, 5), ret, 0.0)
    assert a["n_draws"] == 1000 and np.array_equal(a["excess"], b["excess"])
    assert not np.array_equal(a["excess"], other["excess"])
    assert a["percentile"] == 1.0  # model far above every draw
    low = ev.random_basket(c, date(2027, 1, 4), ret, 0.0, model_excess=-5.0)
    assert low["percentile"] == 0.0
    mid = ev.random_basket(
        c, date(2027, 1, 4), ret, 0.0, model_excess=float(np.median(a["excess"]))
    )
    assert 0.4 < mid["percentile"] < 0.6


def test_random_basket_small_universe_and_missing_returns():
    c, ret = _rb_inputs(20)
    r = ev.random_basket(c, date(2027, 1, 4), ret, 0.0, n_draws=50)
    assert r["n_draws"] == 50
    none = ev.random_basket(c, date(2027, 1, 4), {}, 0.0, n_draws=10)
    assert none["n_draws"] == 0 and none["mean"] is None
    empty = ev.random_basket(c.head(0), date(2027, 1, 4), ret, 0.0, n_draws=5)
    assert empty["n_draws"] == 0


# --- rank IC ----------------------------------------------------------------


def test_rank_ic_matches_scipy_with_ties_and_nulls():
    s = [1.0, 2.0, 2.0, 4.0, 5.0, None, 3.0]
    y = [0.3, 0.1, 0.2, 0.2, 0.9, 0.5, None]
    keep = [i for i in range(7) if s[i] is not None and y[i] is not None]
    ref = stats.spearmanr([s[i] for i in keep], [y[i] for i in keep]).statistic
    assert ev.rank_ic(s, y) == pytest.approx(ref)


def test_rank_ic_perfect_and_degenerate():
    assert ev.rank_ic([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ev.rank_ic([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert ev.rank_ic([1, 1, 1], [1, 2, 3]) is None
    assert ev.rank_ic([1, 2], [1, 2]) is None


# --- check decision ---------------------------------------------------------


def _points(g, ic, tail, start=date(2027, 1, 4)) -> pl.DataFrame:
    n = len(g)
    grid = [start + timedelta(days=28 * i) for i in range(n)]
    return pl.DataFrame(
        {
            "grid_date": grid,
            "matured_at": [d + timedelta(days=28) for d in grid],
            "g": g,
            "tail_diff": tail,
            "rank_ic": ic,
        }
    )


def test_check_insufficient_below_six():
    p = _points([-1.0] * 5, [-1.0] * 5, [1.0] * 5)
    r = ev.check_decision(p, date(2030, 1, 1))
    assert r["n_matured"] == 5 and r["model"] == ev.STATUS_INSUFFICIENT
    assert r["tail"] == ev.STATUS_INSUFFICIENT and r["stop_proposal"] is False


def test_check_counts_only_matured_points():
    p = _points([0.01] * 8, [0.1] * 8, [-0.01] * 8)
    cutoff = p["matured_at"][5]  # exactly 6 points matured on that day (<=)
    assert ev.check_decision(p, cutoff)["n_matured"] == 6
    assert ev.check_decision(p, cutoff - timedelta(days=1))["n_matured"] == 5


def test_check_model_three_way():
    d = date(2030, 1, 1)
    both = ev.check_decision(_points([-0.01] * 6, [-0.1] * 6, [-0.01] * 6), d)
    assert both["model"] == ev.STATUS_STOP_PROPOSAL and both["stop_proposal"] is True
    zero = ev.check_decision(_points([0.0] * 6, [0.0] * 6, [-0.01] * 6), d)
    assert zero["model"] == ev.STATUS_STOP_PROPOSAL  # literal "<= 0 and <= 0"
    g_only = ev.check_decision(_points([-0.01] * 6, [0.1] * 6, [-0.01] * 6), d)
    ic_only = ev.check_decision(_points([0.01] * 6, [-0.1] * 6, [-0.01] * 6), d)
    assert g_only["model"] == ev.STATUS_UNDECIDED == ic_only["model"]
    assert not g_only["stop_proposal"]
    ok = ev.check_decision(_points([0.01] * 6, [0.1] * 6, [-0.01] * 6), d)
    assert ok["model"] == ev.STATUS_CONTINUE


def test_check_tail_direction_never_stops():
    d = date(2030, 1, 1)
    bad = ev.check_decision(_points([0.01] * 6, [0.1] * 6, [0.02] * 6), d)
    assert bad["tail"] == ev.TAIL_NOT_CONFIRMED and bad["model"] == ev.STATUS_CONTINUE
    zero = ev.check_decision(_points([0.01] * 6, [0.1] * 6, [0.0] * 6), d)
    assert zero["tail"] == ev.TAIL_NOT_CONFIRMED  # >= 0
    good = ev.check_decision(_points([0.01] * 6, [0.1] * 6, [-0.02] * 6), d)
    assert good["tail"] == ev.TAIL_DIRECTION_OK


def test_check_significance_hold_until_matured():
    rng = np.random.default_rng(3)
    p = _points(
        rng.normal(0.02, 0.03, 40).tolist(), [0.1] * 40, rng.normal(-0.02, 0.03, 40).tolist()
    )
    early = ev.check_decision(p, p["matured_at"][22])  # only 23 matured
    assert early["significance_g"]["n24"] == {"status": ev.STATUS_HOLD}
    assert early["significance_tail"]["n36"] == {"status": ev.STATUS_HOLD}
    mid = ev.check_decision(p, p["matured_at"][23])
    assert mid["significance_g"]["n24"]["status"] == "검정"
    assert mid["significance_g"]["n36"]["status"] == ev.STATUS_HOLD
    full = ev.check_decision(p, p["matured_at"][39])
    assert full["significance_g"]["n36"]["n"] == 36
    # "first 24" stays the first 24 even when later points exist
    assert full["significance_g"]["n24"]["n"] == 24


@pytest.mark.parametrize("n", [24, 36])
def test_check_ttest_matches_scipy(n):
    rng = np.random.default_rng(n)
    g = rng.normal(0.01, 0.04, 40)
    tl = rng.normal(-0.01, 0.03, 40)
    p = _points(g.tolist(), [0.1] * 40, tl.tolist())
    r = ev.check_decision(p, p["matured_at"][39])
    sg, st = r["significance_g"][f"n{n}"], r["significance_tail"][f"n{n}"]
    ref_g = stats.ttest_1samp(g[:n], 0.0, alternative="greater")
    ref_t = stats.ttest_1samp(tl[:n], 0.0, alternative="less")
    assert sg["p"] == pytest.approx(ref_g.pvalue) and sg["t"] == pytest.approx(ref_g.statistic)
    assert st["p"] == pytest.approx(ref_t.pvalue) and st["t"] == pytest.approx(ref_t.statistic)
    assert sg["significant"] == (ref_g.pvalue < 0.025)


def test_check_ttest_alpha_boundary():
    # build a sample whose p-value is just below / above 0.025 by scaling the mean
    base = np.array([1.0, -1.0] * 12)  # n = 24, mean 0, sd ~1.021
    sd = base.std(ddof=1)
    crit = stats.t.isf(0.025, 23) * sd / np.sqrt(24)
    for shift, sig in ((crit * 1.001, True), (crit * 0.999, False)):
        g = (base + shift).tolist()
        p = _points(g + [0.0] * 6, [0.1] * 30, [-0.1] * 30)
        r = ev.check_decision(p, p["matured_at"][29])
        assert r["significance_g"]["n24"]["significant"] is sig


def test_check_ttest_constant_sample_and_nulls():
    p = _points([0.01] * 24, [0.1] * 24, [-0.01] * 24)
    r = ev.check_decision(p, p["matured_at"][23])
    assert r["significance_g"]["n24"]["p"] == pytest.approx(0.0)
    assert r["significance_g"]["n24"]["significant"] is True
    assert r["significance_tail"]["n24"]["significant"] is True
    # a NULL inside the first 24 counts as a missing point for that cell: the test waits for
    # the 25th point and then uses the first 24 valued points (n stays 24, never shortened)
    g = [0.01] * 24
    g[3] = None
    r2 = ev.check_decision(_points(g, [0.1] * 24, [-0.01] * 24), date(2030, 1, 1))
    assert r2["significance_g"]["n24"] == {"status": ev.STATUS_HOLD}
    assert r2["significance_tail"]["n24"]["status"] == "검정"  # the tail cell has no NULL
    assert r2["n_matured"] == 24 and r2["mean_g"] == pytest.approx(0.01)
    g25 = g + [0.02]
    r3 = ev.check_decision(_points(g25, [0.1] * 25, [-0.01] * 25), date(2030, 1, 1))
    s = r3["significance_g"]["n24"]
    assert s["status"] == "검정" and s["n"] == 24 and s["n_skipped_null"] == 1
    assert s["mean"] == pytest.approx((0.01 * 23 + 0.02) / 24)


def test_check_empty_points():
    p = _points([], [], [])
    r = ev.check_decision(
        p.cast({"g": pl.Float64, "tail_diff": pl.Float64, "rank_ic": pl.Float64}), date(2030, 1, 1)
    )
    assert r["n_matured"] == 0 and r["model"] == ev.STATUS_INSUFFICIENT
