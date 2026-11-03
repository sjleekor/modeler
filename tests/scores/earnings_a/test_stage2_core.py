"""Synthetic-only tests for the stage 2 core (calendar, mcap, baskets, returns, statistics)."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from modeler.scores.earnings_a import stage2 as S


def _weekdays(start: date, end: date, holidays: set[date] = frozenset()) -> list[date]:
    out, d = [], start
    while d <= end:
        if d.weekday() < 5 and d not in holidays:
            out.append(d)
        d += timedelta(days=1)
    return out


# ------------------------------------------------------------------ calendar


def test_signal_day_skips_weekend_and_holiday():
    hol = {date(2025, 7, 14)}  # Monday holiday
    ss = _weekdays(date(2025, 6, 1), date(2026, 8, 31), hol)
    # t + 14 = 2025-07-14 (holiday) -> last session on or before is Fri 2025-07-11
    assert S.signal_day(ss, date(2025, 6, 30)) == date(2025, 7, 11)
    # t + 14 = 2025-01-14 is a session in a plain calendar
    ss2 = _weekdays(date(2024, 12, 1), date(2025, 3, 1))
    assert S.signal_day(ss2, date(2024, 12, 31)) == date(2025, 1, 14)


def test_entry_is_fifth_session_after_signal_day():
    ss = _weekdays(date(2025, 6, 1), date(2026, 8, 31), {date(2025, 7, 16)})
    u = date(2025, 7, 14)
    # sessions after u: 15, 17 (16 is a holiday), 18, 21, 22 -> 5th = 07-22
    assert S.entry_day(ss, u) == date(2025, 7, 22)
    assert S.entry_day(ss, date(2025, 7, 13)) is None  # u must be a session
    assert S.entry_day(ss[:20], ss[17]) is None  # data ends before the 5th session


def test_entry_estimate_continues_with_weekdays():
    ss = _weekdays(date(2026, 9, 1), date(2026, 10, 21))
    e, est = S.entry_day_estimate(ss, date(2026, 10, 14))
    assert (e, est) == (date(2026, 10, 21), False)
    e, est = S.entry_day_estimate(ss, date(2026, 10, 20))
    assert est is True and e == date(2026, 10, 27)


def test_add_months_month_end_and_leap_day():
    assert S.add_months(date(2024, 2, 29), 12) == date(2025, 2, 28)
    assert S.add_months(date(2025, 1, 31), 12) == date(2026, 1, 31)
    assert S.add_months(date(2025, 7, 31), 1) == date(2025, 8, 31)
    assert S.add_months(date(2025, 8, 31), 6) == date(2026, 2, 28)


def test_exit_last_session_on_or_before_target_and_beyond_data():
    ss = _weekdays(date(2025, 1, 1), date(2027, 1, 31), {date(2026, 7, 21)})
    entry = date(2025, 7, 21)
    # target 2026-07-21 is a holiday -> last session before = 07-20
    assert S.exit_day(ss, entry) == date(2026, 7, 20)
    # weekend target: 2025-07-19 (Sat) + 12m -> 2026-07-19 (Sun) -> Fri 07-17
    assert S.exit_day(ss, date(2025, 7, 19)) == date(2026, 7, 17)
    # data ends before the target: beyond data
    assert S.exit_day(_weekdays(date(2025, 1, 1), date(2026, 3, 1)), entry) is None


def test_schedule_and_signal_known():
    ss = _weekdays(date(2026, 9, 1), date(2026, 10, 22))
    sch = S.schedule(ss, date(2026, 9, 30))
    assert sch.signal_day == date(2026, 10, 14) and sch.entry == date(2026, 10, 21)
    assert sch.exit is None
    assert S.signal_day_known(ss, date(2026, 9, 30))
    assert not S.signal_day_known(ss, date(2026, 10, 15))
    # data up to Friday; t + 14 is a Saturday -> still known
    fri = _weekdays(date(2026, 9, 1), date(2026, 10, 16))
    assert S.signal_day_known(fri, date(2026, 10, 3))


def test_late_flag():
    assert S.late_flag(date(2026, 11, 4), date(2026, 10, 21)) is True
    assert S.late_flag(date(2026, 10, 21), date(2026, 10, 21)) is True
    assert S.late_flag(date(2026, 10, 20), date(2026, 10, 21)) is False


# ------------------------------------------------------------------ company market cap


def test_company_mcap_sums_preferred_lines_by_first_five_characters():
    m = pd.DataFrame(
        {
            "ticker": ["005930", "005935", "005937", "000660", "123450", "123455", "999990"],
            "market_cap": [1000, 100, 50, 500, 200, 30, 10],
        }
    )
    co = S.company_mcap(m).set_index("ticker")
    assert co.loc["005930", "mcap_total"] == 1150 and co.loc["005930", "n_pref"] == 2
    assert co.loc["000660", "mcap_total"] == 500 and not co.loc["000660", "has_pref"]
    assert co.loc["123450", "pref_ratio"] == pytest.approx(0.15)
    assert int(co["has_pref"].sum()) == 2
    assert int((co["pref_ratio"] > 0.10).sum()) == 2  # 005930 (0.15) and 123450 (0.15)
    assert "005935" not in co.index  # only common tickers are returned


def test_company_mcap_missing_common_gives_nan_and_pref_only_company_is_absent():
    m = pd.DataFrame({"ticker": ["111110", "111115", "222225"], "market_cap": [np.nan, 40.0, 70.0]})
    co = S.company_mcap(m).set_index("ticker")
    assert np.isnan(co.loc["111110", "mcap_total"])
    assert "222220" not in co.index


# ------------------------------------------------------------------ universe and signals


def _feats(tickers, **cols):
    df = pd.DataFrame(
        {
            "ticker": tickers,
            "corp_code": [f"C{t}" for t in tickers],
            "input_eligible": True,
            "a_t": 1000.0,
            "oi_ttm_raw": 0.1,
            "div_ttm": 0.01,
            "ksic2": "26",
        }
    )
    for k, v in cols.items():
        df[k] = v
    return df


def _info(tickers, **over):
    df = pd.DataFrame(
        {
            "ticker": tickers,
            "halted_u": False,
            "p6_tradable_u": True,
            "liq_notice_u": False,
            "raw_close_u": 5000.0,
        }
    )
    for k, v in over.items():
        df[k] = v
    return df


def test_assemble_flags_follow_priority_and_do_not_need_labels():
    tk = ["000010", "000020", "000030", "000040", "000050", "000060"]
    f = _feats(tk)
    info = _info(tk[:5])  # 000060 is not listed on u
    info.loc[1, "halted_u"] = True
    info.loc[2, "p6_tradable_u"] = False
    info.loc[3, "liq_notice_u"] = True
    mc = pd.DataFrame({"ticker": tk[:4] + ["000060"], "market_cap": [1e9] * 4 + [1e9]})
    co = S.company_mcap(mc)
    uni = S.assemble(f, info, co, preds=None)
    got = dict(zip(uni["ticker"], uni["drop_reason"], strict=True))
    got = {k: (None if pd.isna(v) else v) for k, v in got.items()}
    assert got == {
        "000010": None,
        "000020": "halted",
        "000030": "p6_not_tradable",
        "000040": "liq_notice",
        "000050": "no_mcap",
        "000060": "not_listed",
    }
    assert uni["in_stage2"].sum() == 1


def test_ineligible_rows_are_not_in_the_universe():
    f = _feats(["000010", "000020"])
    f.loc[1, "input_eligible"] = False
    uni = S.assemble(
        f,
        _info(["000010", "000020"]),
        S.company_mcap(pd.DataFrame({"ticker": ["000010", "000020"], "market_cap": [1e9, 1e9]})),
    )
    assert list(uni["ticker"]) == ["000010"]


def test_signals_use_company_mcap_and_ev_rules():
    tk = ["000010", "000020"]
    f = _feats(
        tk, debt_st_amt=[np.nan, 100.0], debt_lt_amt=[np.nan, 50.0], cash_amt=[200.0, np.nan]
    )
    mc = pd.DataFrame(
        {"ticker": ["000010", "000015", "000020"], "market_cap": [800.0, 200.0, 1000.0]}
    )
    preds = pd.DataFrame({"ticker": tk, "y_hat_hgb": [0.2, 0.2], "y_hat_pers": [0.1, 0.1]})
    u = S.assemble(f, _info(tk), S.company_mcap(mc), preds)
    a = u.set_index("ticker")
    assert a.loc["000010", "mcap_u"] == 1000.0  # common 800 + preferred 200
    assert a.loc["000010", "M"] == pytest.approx(0.2 * 1000 / 1000)
    assert a.loc["000010", "B1"] == pytest.approx(0.1 * 1000 / 1000)
    assert a.loc["000010", "B1u"] == pytest.approx(0.1 * 1000 / 1000)
    # null borrowings -> 0: EV = 1000 + 0 + 0 - 200 = 800
    assert a.loc["000010", "M_EV"] == pytest.approx(0.2 * 1000 / 800)
    assert np.isnan(a.loc["000020", "M_EV"])  # null cash -> NaN
    assert a.loc["000010", "div_yield"] == pytest.approx(0.01 * 1000 / 1000)


def test_pick_top_ties_by_ticker_ascending_and_skips_nan():
    df = pd.DataFrame(
        {
            "ticker": ["C", "A", "B", "D", "E"],
            "M": [0.5, 0.5, 0.5, np.nan, 0.9],
        }
    )
    top = S.pick_top(df, "M", 3)
    assert list(top["ticker"]) == ["E", "A", "B"]
    assert list(S.pick_top(df, "M", 10)["ticker"]) == ["E", "A", "B", "C"]


def test_build_baskets_variants_sizes_and_weights():
    n = 60
    tk = [f"{i:06d}" for i in range(10, 10 + n * 10, 10)]
    df = pd.DataFrame(
        {
            "ticker": tk,
            "corp_code": tk,
            "ksic2": "26",
            "mcap_u": np.linspace(1e9, 6e9, n),
            "div_yield": 0.01,
            "M": np.linspace(0, 1, n),
            "B1": np.linspace(1, 0, n),
            "B1u": np.linspace(0, 1, n),
            "M_EV": np.linspace(0, 1, n),
            "B1_EV": np.linspace(1, 0, n),
        }
    )
    b = S.build_baskets(df)
    sizes = b.groupby("variant").size().to_dict()
    assert sizes["M"] == 30 and sizes["B1"] == 30 and sizes["M_N15"] == 15 and sizes["B1_N50"] == 50
    w = b[b["variant"] == "M"]["weight"]
    assert w.sum() == pytest.approx(1.0) and w.nunique() == 1
    cap = b[b["variant"] == "M_cap"]
    assert cap["weight"].sum() == pytest.approx(1.0) and cap["weight"].max() <= 0.10 + 1e-12
    m_top = b[b["variant"] == "M"].sort_values("rank")["ticker"].tolist()
    assert m_top[0] == tk[-1]  # highest M first
    # M and B1 baskets are disjoint here (opposite orders over 60 names, 30 each)
    assert len(set(m_top) & set(b[b["variant"] == "B1"]["ticker"])) == 0


def test_quintiles_and_exposure():
    uni = pd.DataFrame(
        {"ticker": [f"{i:06d}" for i in range(10)], "mcap_u": np.arange(1, 11) * 1.0}
    )
    q = S.size_quintile(uni["mcap_u"])
    assert list(q.astype(int)) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    base = pd.DataFrame(
        {
            "variant": ["M", "M", "B1", "B1"],
            "ticker": ["000000", "000001", "000001", "000009"],
            "ksic2": ["26", "62", "26", "26"],
            "mcap_u": [1.0, 2.0, 2.0, 10.0],
        }
    )
    exp = S.tranche_exposure(uni, base, lambda k: "C" if k == "26" else "J")
    assert exp["overlap_M_B1"] == 1
    assert exp["M_ksic_section"] == {"C": 1, "J": 1}
    assert exp["B1_size_quintile"] == {1: 1, 5: 1}


# ------------------------------------------------------------------ returns


def _closes(rows):
    """rows: (ticker, date, close, halted)."""
    df = pd.DataFrame(rows, columns=["ticker", "trade_date", "close", "is_halted"])
    return S.closes_frame(df)


def test_basket_return_mean_minus_cost_and_entry_halt_drop_renormalizes():
    e, x = date(2025, 1, 10), date(2026, 1, 9)
    rows = []
    for tk, c0, c1 in (("A", 100, 120), ("B", 100, 90)):
        rows += [(tk, e, c0, False), (tk, x, c1, False)]
    rows += [("C", e, 0, True), ("C", x, 50, False)]  # halted at entry -> dropped
    w = {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}
    r = S.basket_return(_closes(rows), e, x, w)
    assert r["gross"] == pytest.approx((0.2 - 0.1) / 2)  # no replacement, renormalized
    assert r["net"] == pytest.approx(0.05 - 0.0065)
    assert r["n_entry_dropped"] == 1 and r["n_usable"] == 2


def test_delisted_name_freezes_at_last_nonhalted_close():
    e, x = date(2025, 1, 10), date(2026, 1, 9)
    mid = date(2025, 6, 2)
    rows = [("A", e, 100, False), ("A", mid, 60, False), ("A", date(2025, 7, 1), 0, True)]
    rows += [("B", e, 100, False), ("B", x, 110, False)]
    r = S.basket_return(_closes(rows), e, x, {"A": 0.5, "B": 0.5})
    assert r["gross"] == pytest.approx((-0.4 + 0.1) / 2)
    assert r["n_delisted"] == 1


def test_halted_at_exit_uses_last_nonhalted_close_and_counts():
    e, x = date(2025, 1, 10), date(2026, 1, 9)
    rows = [("A", e, 100, False), ("A", date(2025, 12, 1), 80, False), ("A", x, 0, True)]
    r = S.basket_return(_closes(rows), e, x, {"A": 1.0}, cost=0.01)
    assert r["gross"] == pytest.approx(-0.2) and r["net"] == pytest.approx(-0.21)
    assert r["n_halted_at_exit"] == 1 and r["n_delisted"] == 0


def test_dividend_yield_added_with_same_weights():
    e, x = date(2025, 1, 10), date(2026, 1, 9)
    rows = [("A", e, 100, False), ("A", x, 110, False), ("B", e, 100, False), ("B", x, 100, False)]
    r = S.basket_return(_closes(rows), e, x, {"A": 0.5, "B": 0.5}, extra={"A": 0.04, "B": 0.0})
    assert r["div_yield"] == pytest.approx(0.02)


def test_max_drawdown_of_buy_and_hold_path():
    days = _weekdays(date(2025, 1, 6), date(2025, 1, 10))
    rows = []
    path_a = [100, 120, 60, 90, 90]
    path_b = [100, 100, 100, 100, 100]
    for d, a, b in zip(days, path_a, path_b, strict=True):
        rows += [("A", d, a, False), ("B", d, b, False)]
    mdd = S.basket_max_drawdown(_closes(rows), days, days[0], days[-1], {"A": 0.5, "B": 0.5})
    # value: 1, 1.1, 0.8, 0.95, 0.95 -> worst = 0.8 / 1.1 - 1
    assert mdd == pytest.approx(0.8 / 1.1 - 1)


def test_cost_variants_and_b0_index():
    e, x = date(2025, 1, 10), date(2026, 1, 9)
    rows = [("A", e, 100, False), ("A", x, 110, False)]
    base = pd.DataFrame(
        {
            "variant": ["M", "B1"],
            "ticker": ["A", "A"],
            "weight": [1.0, 1.0],
            "div_yield": [0.0, 0.0],
        }
    )
    idx = pd.Series([200.0, 230.0], index=pd.to_datetime([e, x]))
    ss = _weekdays(e, x)
    sc = S.score_baskets(base, _closes(rows), ss, e, x, index_close=idx)
    assert sc["M_net"] == pytest.approx(0.10 - 0.0065)
    assert sc["M_net100"] == pytest.approx(0.10 - 0.0100)
    assert sc["M_net130"] == pytest.approx(0.10 - 0.0130)
    assert sc["B0"] == pytest.approx(0.15)  # price index, no cost
    assert "B0_tr" not in sc


def test_index_return_uses_last_value_on_or_before():
    idx = pd.Series(
        [100.0, 110.0, 99.0], index=pd.to_datetime(["2025-01-03", "2025-01-08", "2026-01-02"])
    )
    assert S.index_return(idx, date(2025, 1, 5), date(2026, 1, 9)) == pytest.approx(99 / 100 - 1)


def test_etf_total_return_reinvests_distribution():
    close = pd.Series(
        [100.0, 100.0, 110.0], index=pd.to_datetime(["2025-01-10", "2025-06-30", "2026-01-09"])
    )
    dist = pd.DataFrame(
        {
            "rgt_std_date": pd.to_datetime(["2025-06-30", "2024-12-31", "2026-02-01"]),
            "per_share_amount": [2.0, 5.0, 3.0],
        }
    )
    r = S.etf_total_return(close, dist, date(2025, 1, 10), date(2026, 1, 9))
    assert r == pytest.approx(1.02 * 1.10 - 1)  # only the in-window record date counts


# ------------------------------------------------------------------ statistics


def test_block_bootstrap_is_deterministic_and_covers_the_mean():
    vals = np.array([0.02, 0.05, -0.01, 0.03, 0.04, 0.0, 0.06, 0.01, -0.02, 0.03])
    a = S.block_bootstrap_ci(vals)
    b = S.block_bootstrap_ci(vals)
    assert a == b
    assert a[0] < vals.mean() < a[1]
    long = np.random.default_rng(5).normal(0.02, 0.05, size=25)
    assert S.block_bootstrap_ci(long, seed=1) != S.block_bootstrap_ci(long)
    # blocks are consecutive groups of four: 4 + 4 + 2
    one_block = S.block_bootstrap_ci(np.array([0.1, 0.1, 0.1, 0.1]))
    assert one_block == (pytest.approx(0.1), pytest.approx(0.1))


def test_summarize_tranches_means_and_pairs():
    n = 8
    tab = pd.DataFrame(
        {
            "decision_date": pd.date_range("2020-03-31", periods=n, freq="QE"),
            "M_net": np.full(n, 0.10),
            "B1_net": np.full(n, 0.06),
            "B0": np.full(n, 0.04),
        }
    )
    s = S.summarize_tranches(tab)
    assert s["n_tranches"] == n
    assert s["M-B1"]["mean"] == pytest.approx(0.04)
    assert s["M-B0"]["mean"] == pytest.approx(0.06)
    assert s["B1-B0"]["mean"] == pytest.approx(0.02)
    assert s["seed"] == 20261103 and s["draws"] == 10000 and s["block"] == 4
    tab.loc[3, "B0"] = np.nan
    assert S.summarize_tranches(tab)["n_dropped_incomplete"] == 1


def test_decay_ratio():
    assert S.decay_ratio(0.02, 0.04) == pytest.approx(0.5)
    assert S.decay_ratio(-0.01, 0.04) == pytest.approx(-0.25)
    assert np.isnan(S.decay_ratio(0.01, 0.0))
