"""Synthetic tests for earnings_a.features. No real data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from modeler.scores.earnings_a import features as F
from modeler.scores.earnings_a.spec import BINARY_FEATURES, FEATURES

T = "2024-09-30"


def r(
    metric,
    pe,
    avail,
    st=None,
    ttm=None,
    ttm_avail=None,
    basis="CFS",
    ticker="000010",
    fye=None,
    conflict=False,
):
    pe = pd.Timestamp(pe)
    return {
        "ticker": ticker,
        "corp_code": "c_" + ticker,
        "metric_code": metric,
        "fs_basis": basis,
        "statement_period_end": pe,
        "available_from": pd.Timestamp(avail),
        "standalone_value": np.nan if st is None else float(st),
        "ttm_value": np.nan if ttm is None else float(ttm),
        "ttm_complete": ttm is not None,
        "ttm_available_from": pd.Timestamp(ttm_avail if ttm_avail else avail)
        if ttm is not None
        else pd.NaT,
        "ttm_includes_conflict": conflict,
        "fiscal_year_end": pd.Timestamp(fye) if fye else pd.Timestamp(f"{pe.year}-12-31"),
    }


def rows_of(*items):
    return F._prep_rows(pd.DataFrame(list(items)))


def uni(tickers=("000010",), dates=(T,), **kw):
    recs = []
    for d in dates:
        for tk in tickers:
            rec = dict(
                decision_date=pd.Timestamp(d),
                session_date=pd.Timestamp(d),
                ticker=tk,
                corp_code="c_" + tk,
                market="KOSPI",
                ksic2="26",
                induty_code="26000",
                is_halted=False,
                is_financial=False,
                is_holding=False,
                is_spac=False,
                liq_notice=False,
                is_non_krw=False,
                p6_tradable=True,
                only_preferred_listed=False,
                in_universe=True,
                exclusion_reason=None,
            )
            rec.update(kw)
            recs.append(rec)
    return pd.DataFrame(recs)


def full_set(basis="CFS", ticker="000010", avail="2024-08-14", pe="2024-06-30"):
    """A complete, consistent set of rows anchored at pe0=2024-06-30."""
    k = dict(basis=basis, ticker=ticker)
    out = [
        r("total_assets", pe, avail, st=1000, **k),
        r("total_assets", "2023-06-30", "2023-08-14", st=800, **k),
        r("operating_income", pe, avail, st=30, ttm=100, **k),
        r("operating_income", "2024-03-31", "2024-05-15", st=25, **k),
        r("operating_income", "2023-12-31", "2024-03-20", st=20, **k),
        r("operating_income", "2023-09-30", "2023-11-14", st=15, **k),
        r("operating_income", "2023-06-30", "2023-08-14", st=10, ttm=60, **k),
        r("revenue", pe, avail, st=300, ttm=1000, **k),
        r("net_income", pe, avail, st=20, ttm=80, **k),
        r("operating_cash_flow", pe, avail, st=40, ttm=120, **k),
        r("total_equity", pe, avail, st=500, **k),
        r("cash_and_cash_equivalents", pe, avail, st=100, **k),
        r("borrowings_short_term", pe, avail, st=50, **k),
        r("borrowings_long_term", pe, avail, st=70, **k),
        r("capex_ppe", pe, avail, st=-5, ttm=-30, **k),
        r("capex_intangible", pe, avail, st=1, ttm=10, **k),
        r("dividends_paid", pe, avail, st=-2, ttm=-8, **k),
    ]
    return out


def feats(items, universe=None):
    rows = rows_of(*items)
    u = universe if universe is not None else uni()
    anchor = F.build_anchor(rows, u)
    return rows, anchor, F.build_features(rows, anchor)


def test_values_and_cfs_first():
    items = full_set() + full_set(basis="OFS", avail="2024-08-14")
    _, anchor, out = feats(items)
    a = anchor.iloc[0]
    assert a.basis == "CFS" and a.a_t_status == "ok" and a.a_t == 1000.0
    assert a.a_t_dec == "1000.0000" and a.pe0 == pd.Timestamp("2024-06-30")
    assert a.fye_month == 12 and a.fye_on_grid
    o = out.iloc[0]
    assert o.oi_ttm == pytest.approx(0.1)
    assert [o.oi_q0, o.oi_q1, o.oi_q2, o.oi_q3] == pytest.approx([0.03, 0.025, 0.02, 0.015])
    assert o.oi_chg_4q == pytest.approx(0.04)
    assert o.asset_growth == pytest.approx(0.25)
    assert o.rev_ttm == pytest.approx(1.0) and o.margin_ttm == pytest.approx(0.1)
    assert o.ocf_ttm == pytest.approx(0.12) and o.ni_ttm == pytest.approx(0.08)
    assert o.accrual_ttm == pytest.approx(-0.04)
    assert o.equity_ratio == pytest.approx(0.5) and o.cash_ratio == pytest.approx(0.1)
    assert o.debt_ratio == pytest.approx(0.12)
    assert o.capex_ttm == pytest.approx(0.04) and o.div_ttm == pytest.approx(0.008)
    assert o.log_assets == pytest.approx(np.log(1000))
    assert o.loss_ttm == 0.0 and o.is_ofs == 0.0
    assert o.stale_days == (pd.Timestamp(T) - pd.Timestamp("2024-08-14")).days
    assert o.input_eligible and o.input_eligible_incl_holding
    assert list(FEATURES) == [c for c in out.columns if c in FEATURES]
    assert o.oi_ttm_raw == o.oi_ttm


def test_ofs_fallback_when_cfs_stale():
    items = [
        r("total_assets", "2023-03-31", "2023-05-15", st=900, basis="CFS"),
        r("total_assets", "2024-06-30", "2024-08-14", st=1000, basis="OFS"),
        r("operating_income", "2024-06-30", "2024-08-14", st=3, ttm=10, basis="OFS"),
    ]
    _, anchor, out = feats(items)
    a = anchor.iloc[0]
    assert a.basis == "OFS" and a.a_t_status == "ok" and a.a_t == 1000
    assert out.iloc[0].is_ofs == 1.0 and out.iloc[0].oi_ttm == pytest.approx(0.01)


def test_stale_13_months_and_boundary_12_ok():
    items = [r("total_assets", "2023-06-30", "2023-08-14", st=900)]
    _, anchor, out = feats(items)  # t=2024-09 -> gap 15
    assert anchor.iloc[0].a_t_status == "stale" and anchor.iloc[0].basis == "CFS"
    assert not out.iloc[0].input_eligible
    items = [r("total_assets", "2023-09-30", "2023-11-14", st=900)]
    _, anchor, _ = feats(items)  # gap exactly 12
    assert anchor.iloc[0].a_t_status == "ok"
    items = [r("total_assets", "2023-08-31", "2023-11-14", st=900)]
    _, anchor, _ = feats(items)  # gap 13
    assert anchor.iloc[0].a_t_status == "stale"


def test_nonpositive_and_absent():
    _, anchor, out = feats([r("total_assets", "2024-06-30", "2024-08-14", st=0)])
    assert anchor.iloc[0].a_t_status == "nonpositive" and not out.iloc[0].input_eligible
    assert np.isnan(out.iloc[0].log_assets)
    _, anchor, out = feats([r("total_assets", "2024-06-30", "2024-10-14", st=5)])  # after t
    a = anchor.iloc[0]
    assert a.a_t_status == "absent" and a.basis is None and np.isnan(a.a_t)
    assert not out.iloc[0].input_eligible


def test_pe0_is_latest_period_not_latest_available():
    items = [
        r("total_assets", "2024-03-31", "2024-09-20", st=700),  # older period, later available
        r("total_assets", "2024-06-30", "2024-08-14", st=1000),
    ]
    _, anchor, _ = feats(items)
    assert anchor.iloc[0].pe0 == pd.Timestamp("2024-06-30") and anchor.iloc[0].a_t == 1000


def test_availability_gates_quarter_and_ttm():
    items = full_set()
    # make the pe0-3m oi quarter available only after t
    items = [
        x
        if not (
            x["metric_code"] == "operating_income"
            and x["statement_period_end"] == pd.Timestamp("2024-03-31")
        )
        else {**x, "available_from": pd.Timestamp("2024-10-05")}
        for x in items
    ]
    _, _, out = feats(items)
    o = out.iloc[0]
    assert np.isnan(o.oi_q1) and not np.isnan(o.oi_q0) and not np.isnan(o.oi_q2)

    items = [
        x
        if not (
            x["metric_code"] == "operating_income"
            and x["statement_period_end"] == pd.Timestamp("2024-06-30")
        )
        else {**x, "ttm_available_from": pd.Timestamp("2024-10-05")}
        for x in full_set()
    ]
    _, _, out = feats(items)
    o = out.iloc[0]
    assert np.isnan(o.oi_ttm) and np.isnan(o.oi_chg_4q) and np.isnan(o.oi_ttm_raw)
    assert not o.input_eligible and np.isnan(o.loss_ttm)
    assert not o.oi_q0 != o.oi_q0  # standalone still present


def test_lag_rows_unavailable_give_nan():
    items = [x for x in full_set() if x["statement_period_end"] != pd.Timestamp("2023-06-30")]
    _, _, out = feats(items)
    o = out.iloc[0]
    assert np.isnan(o.asset_growth) and np.isnan(o.oi_chg_4q) and not np.isnan(o.oi_ttm)


def test_offgrid_fiscal_year():
    items = [
        r("total_assets", "2024-05-31", "2024-07-15", st=1000, fye="2024-11-30"),
        r("operating_income", "2024-05-31", "2024-07-15", st=3, ttm=10),
    ]
    _, anchor, out = feats(items)
    a = anchor.iloc[0]
    assert a.fye_month == 11 and not a.fye_on_grid and a.a_t_status == "ok"
    assert not out.iloc[0].input_eligible and not out.iloc[0].input_eligible_incl_holding


def test_capex_abs_and_missing_parts():
    items = full_set()
    _, _, out = feats(items)
    assert out.iloc[0].capex_ttm == pytest.approx(0.04)
    no_int = [x for x in items if x["metric_code"] != "capex_intangible"]
    _, _, out = feats(no_int)
    assert np.isnan(out.iloc[0].capex_ttm)
    no_lt = [x for x in items if x["metric_code"] != "borrowings_long_term"]
    _, _, out = feats(no_lt)
    assert np.isnan(out.iloc[0].debt_ratio)
    no_div = [x for x in items if x["metric_code"] != "dividends_paid"]
    _, _, out = feats(no_div)
    assert np.isnan(out.iloc[0].div_ttm)  # no zero fill


def test_margin_nan_when_revenue_nonpositive_and_loss_flag():
    items = [{**x, "ttm_value": -1.0} if x["metric_code"] == "revenue" else x for x in full_set()]
    items = [
        {**x, "ttm_value": -50.0}
        if x["metric_code"] == "operating_income"
        and x["ttm_complete"]
        and x["statement_period_end"] == pd.Timestamp("2024-06-30")
        else x
        for x in items
    ]
    _, _, out = feats(items)
    o = out.iloc[0]
    assert np.isnan(o.margin_ttm) and o.loss_ttm == 1.0 and o.oi_ttm < 0


def test_no_cross_basis_filling():
    items = [x for x in full_set() if x["metric_code"] != "total_equity"]
    items += [r("total_equity", "2024-06-30", "2024-08-14", st=999, basis="OFS")]
    _, _, out = feats(items)
    assert np.isnan(out.iloc[0].equity_ratio)


def test_basis_switched():
    items = [
        r("total_assets", "2023-03-31", "2023-05-15", st=900, basis="CFS"),
        r("total_assets", "2024-03-31", "2024-05-15", st=900, basis="OFS"),
        r("total_assets", "2024-03-31", "2024-05-15", st=900, basis="OFS", ticker="000020"),
        r("total_assets", "2024-06-30", "2024-08-14", st=900, basis="OFS", ticker="000020"),
    ]
    u = uni(tickers=("000010", "000020"), dates=("2024-03-31", "2024-06-30"))
    anchor = F.build_anchor(rows_of(*items), u)
    got = {
        (row.ticker, row.decision_date.strftime("%Y-%m-%d")): (row.basis, bool(row.basis_switched))
        for row in anchor.itertuples()
    }
    assert got[("000010", "2024-03-31")] == ("CFS", False)
    assert got[("000010", "2024-06-30")] == (
        "OFS",
        True,
    )  # OFS pe0 2024-03 is later than CFS pe0 2023-03
    assert got[("000020", "2024-06-30")] == ("OFS", False)


def test_preferred_rows_dropped_and_holding_eligibility():
    u = pd.concat(
        [
            uni(),
            uni(
                tickers=("000015",),
                only_preferred_listed=True,
                in_universe=False,
                exclusion_reason="only_preferred",
            ),
            uni(
                tickers=("000020",),
                in_universe=False,
                is_holding=True,
                is_financial=True,
                exclusion_reason="financial",
            ),
        ]
    )
    items = full_set() + full_set(ticker="000020")
    _, anchor, out = feats(items, u)
    assert set(out.ticker) == {"000010", "000020"}
    h = out[out.ticker == "000020"].iloc[0]
    assert not h.input_eligible and h.input_eligible_incl_holding


def test_clip_bounds_only_from_eligible_rows():
    n = 101
    df = pd.DataFrame(
        {
            "decision_date": pd.Timestamp(T),
            "oi_ttm": np.arange(n, dtype=float),
            "loss_ttm": np.r_[np.full(n - 1, 0.0), 1.0],
            "is_ofs": np.zeros(n),
            "input_eligible": True,
        }
    )
    ext = df.iloc[[0]].copy()
    ext["oi_ttm"] = 1e9
    ext["input_eligible"] = False
    df = pd.concat([df, ext], ignore_index=True)
    df["oi_ttm_raw"] = df["oi_ttm"]
    out = F.clip_features(df, cols=["oi_ttm"])
    # eligible values 0..100 -> q01=1, q99=99
    assert out["oi_ttm"].min() == pytest.approx(1.0) and out["oi_ttm"].max() == pytest.approx(99.0)
    assert out["oi_ttm_raw"].max() == 1e9
    assert out["oi_ttm"].iloc[-1] == pytest.approx(
        99.0
    )  # ineligible row clipped, did not move bound


def test_clip_per_date_and_binary_default_untouched():
    vals = np.arange(101, dtype=float)
    d1 = pd.DataFrame({"decision_date": pd.Timestamp("2024-06-30"), "oi_ttm": vals})
    d2 = pd.DataFrame({"decision_date": pd.Timestamp(T), "oi_ttm": vals * 10})
    df = pd.concat([d1, d2], ignore_index=True)
    df["loss_ttm"] = 1000.0  # binary: must stay
    df["is_ofs"] = -5.0
    df["input_eligible"] = True
    out = F.clip_features(
        df, cols=[c for c in FEATURES if c not in BINARY_FEATURES and c in df.columns]
    )
    assert out.loc[
        out.decision_date == pd.Timestamp("2024-06-30"), "oi_ttm"
    ].max() == pytest.approx(99.0)
    assert out.loc[out.decision_date == pd.Timestamp(T), "oi_ttm"].max() == pytest.approx(990.0)
    for c in BINARY_FEATURES:
        assert (out[c] == df[c]).all()
    assert np.isnan(F.clip_features(df.assign(oi_ttm=np.nan), cols=["oi_ttm"])["oi_ttm"]).all()


def _basis_of(items, t=T):
    a = F.build_anchor(rows_of(*items), uni(dates=(t,))).iloc[0]
    return a.basis, a.a_t_status, a.pe0


def test_basis_follows_latest_period():
    items = [
        r("total_assets", "2023-03-31", "2023-05-15", st=900, basis="CFS"),
        r("total_assets", "2023-06-30", "2023-08-14", st=950, basis="OFS"),
    ]
    assert _basis_of(items, "2023-09-30") == ("OFS", "ok", pd.Timestamp("2023-06-30"))


def test_same_latest_period_prefers_cfs():
    items = [
        r("total_assets", "2024-06-30", "2024-08-14", st=900, basis="CFS"),
        r("total_assets", "2024-06-30", "2024-08-14", st=950, basis="OFS"),
    ]
    assert _basis_of(items) == ("CFS", "ok", pd.Timestamp("2024-06-30"))


def test_latest_nonpositive_has_no_fallback():
    items = [
        r("total_assets", "2024-03-31", "2024-05-15", st=900, basis="CFS"),
        r("total_assets", "2024-06-30", "2024-08-14", st=0, basis="OFS"),
    ]
    assert _basis_of(items) == ("OFS", "nonpositive", pd.Timestamp("2024-06-30"))


def test_latest_stale_when_both_old():
    items = [
        r("total_assets", "2022-12-31", "2023-03-15", st=900, basis="CFS"),
        r("total_assets", "2023-03-31", "2023-05-15", st=900, basis="OFS"),
    ]
    assert _basis_of(items) == ("OFS", "stale", pd.Timestamp("2023-03-31"))
