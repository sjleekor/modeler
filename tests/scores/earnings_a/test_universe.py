"""Universe rules on synthetic duckdb views (no real data)."""

from __future__ import annotations

from datetime import date, timedelta

import duckdb
import pandas as pd
import pytest

from modeler.scores.earnings_a import universe as U


def _con(ohlcv, corp, master, receipts=(), fs=()):
    con = duckdb.connect(":memory:")
    o = pd.DataFrame(
        ohlcv, columns=["trade_date", "ticker", "market", "open", "high", "low", "close", "volume"]
    )
    o["trade_date"] = pd.to_datetime(o["trade_date"]).dt.date
    con.register("_o", o)
    con.execute("CREATE VIEW daily_ohlcv AS SELECT * FROM _o")
    mc = o[["trade_date", "ticker"]].copy()
    mc["market"] = "KOSPI"
    mc["source_close"] = o["close"]
    mc["volume"] = o["volume"]
    mc["trading_value"] = o["close"] * o["volume"]
    con.register("_mc", mc)
    con.execute("CREATE VIEW daily_market_cap AS SELECT * FROM _mc")
    con.register(
        "_c", pd.DataFrame(corp, columns=["ticker", "corp_code", "induty_code", "corp_name"])
    )
    con.execute("CREATE VIEW dart_corp_master AS SELECT * FROM _c")
    con.register("_m", pd.DataFrame(master, columns=["ticker", "market", "name"]))
    con.execute("CREATE VIEW stock_master AS SELECT * FROM _m")
    r = pd.DataFrame(receipts, columns=["ticker", "rcept_dt", "rcept_no", "report_nm"])
    r["rcept_dt"] = pd.to_datetime(r["rcept_dt"]).dt.date
    con.register("_r", r)
    con.execute(
        "CREATE VIEW dart_filing_receipt_raw AS SELECT CAST(ticker AS VARCHAR) ticker, "
        "CAST(rcept_dt AS DATE) rcept_dt, CAST(rcept_no AS VARCHAR) rcept_no, "
        "CAST(report_nm AS VARCHAR) report_nm FROM _r"
    )
    f = pd.DataFrame(fs, columns=["corp_code", "currency"])
    con.register("_f", f)
    con.execute(
        "CREATE VIEW dart_financial_statement_raw AS "
        "SELECT CAST(corp_code AS VARCHAR) corp_code, CAST(currency AS VARCHAR) currency FROM _f"
    )
    return con


def _days(start, n, tickers, halted=()):
    rows = []
    d = date.fromisoformat(start)
    for i in range(n):
        day = d + timedelta(days=i)
        if day.weekday() >= 5:
            continue
        for t in tickers:
            z = (t, day) in halted
            p = 0 if z else 5000
            rows.append((day, t, "KOSPI", p, p, p, 5000, 100000))
    return rows


TICKERS = {
    "000010": ("C1", "10000", "회사A"),
    "000015": ("C1", "10000", "회사A우"),  # preferred of C1
    "000020": ("C2", "64110", "은행B"),
    "000030": ("C3", "64992", "홀딩C"),
    "000040": ("C4", None, "회사D"),
    "000050": ("C5", "66100", "회사E"),
    "000060": ("C6", "10000", "OO스팩"),
    "000070": ("C7", "10000", "제7호기업인수목적"),
    "000080": ("C8", "10000", "회사H"),
    "000090": ("C9", "10000", "회사I"),
    "000105": ("C10", "10000", "회사J우"),  # preferred only (common 000100 not listed)
}
CORP = [(t, v[0], v[1], v[2]) for t, v in TICKERS.items() if t != "000015" and t != "000105"]
CORP.append(("000100", "C10", "10000", "회사J"))
MASTER = [(t, "KOSPI", v[2]) for t, v in TICKERS.items()]


def _base(**kw):
    ohlcv = kw.pop("ohlcv", None) or _days("2024-12-20", 20, list(TICKERS))
    return _con(ohlcv, CORP, MASTER, **kw)


D = date


def test_session_on_holiday_and_weekend():
    # 2024-12-31 Tue is a trading day here, 2024-12-28 Sat, 2025-01-01 Wed has no rows? all weekdays exist
    ohlcv = [r for r in _days("2024-12-20", 20, list(TICKERS)) if r[0] != D(2024, 12, 31)]
    con = _con(ohlcv, CORP, MASTER)
    u = U.build_universe(con, [D(2024, 12, 31), D(2024, 12, 28)], non_krw_corp_codes=frozenset())
    got = u.groupby("decision_date")["session_date"].first().to_dict()
    assert got[D(2024, 12, 31)] == D(2024, 12, 30)
    assert got[D(2024, 12, 28)] == D(2024, 12, 27)
    assert U.session_dates([D(2024, 1, 5)], [D(2024, 1, 1)]) == {D(2024, 1, 1): None}


def test_halted_kept_and_flagged():
    ohlcv = _days("2024-12-20", 20, list(TICKERS), halted={("000080", D(2024, 12, 31))})
    u = U.build_universe(
        _con(ohlcv, CORP, MASTER), [D(2024, 12, 31)], non_krw_corp_codes=frozenset()
    )
    row = u[u.ticker == "000080"].iloc[0]
    assert row.is_halted and row.in_universe
    assert not u[u.ticker == "000010"].iloc[0].is_halted


def test_preferred_rules_and_flags():
    u = U.build_universe(_base(), [D(2024, 12, 31)], non_krw_corp_codes=frozenset())
    assert (u.ticker == "000015").sum() == 0  # common listed -> common row only
    assert (u.ticker == "000010").sum() == 1
    pref = u[u.ticker == "000105"].iloc[0]
    assert pref.only_preferred_listed and not pref.in_universe
    assert pref.corp_code == "C10"
    assert pref.exclusion_reason == "only_preferred"
    assert not u[u.ticker == "000010"].iloc[0].only_preferred_listed


def test_exclusion_flags():
    u = U.build_universe(
        _base(), [D(2024, 12, 31)], non_krw_corp_codes=frozenset({"C9"})
    ).set_index("ticker")
    assert u.loc["000020", "is_financial"] and u.loc["000020", "exclusion_reason"] == "financial"
    assert u.loc["000030", "is_holding"] and u.loc["000030", "is_financial"]
    assert u.loc["000030", "ksic2"] == "64"
    assert not u.loc["000040", "is_financial"] and u.loc["000040", "in_universe"]
    assert u.loc["000040", "ksic2"] is None
    assert u.loc["000050", "is_financial"]
    assert u.loc["000060", "is_spac"] and u.loc["000060", "exclusion_reason"] == "spac"
    assert u.loc["000070", "is_spac"]
    assert u.loc["000090", "is_non_krw"] and u.loc["000090", "exclusion_reason"] == "non_krw"
    assert u.loc["000010", "in_universe"] and u.loc["000010", "exclusion_reason"] is None


def test_liq_notice_flag():
    receipts = [("000080", "2024-12-27", "20241227000001", "정리매매 개시 안내")]
    u = U.build_universe(
        _base(receipts=receipts), [D(2024, 12, 31)], non_krw_corp_codes=frozenset()
    ).set_index("ticker")
    assert u.loc["000080", "liq_notice"] and u.loc["000080", "exclusion_reason"] == "liq_notice"
    assert not u.loc["000010", "liq_notice"]


def _listing(rows):
    cols = ["ticker", "corp_code", "market", "is_halted", "liq_notice", "p6_tradable"]
    df = pd.DataFrame(rows, columns=cols)
    df.insert(0, "session_date", D(2024, 12, 31))
    df.insert(0, "decision_date", D(2024, 12, 31))
    return df


def test_reason_priority_and_p6_is_not_a_filter():
    rows = [
        ("000010", "C1", "KOSPI", False, True, True),
        ("000020", "C2", "KOSPI", False, True, None),
        ("000060", "C6", "KOSPI", False, True, False),
        ("000105", "C10", "KOSPI", False, True, True),
    ]
    attrs = pd.DataFrame(
        {
            "ticker": ["000010", "000020", "000060", "000105"],
            "induty_code": ["64110", "64110", "10000", "64110"],
            "is_spac": [True, True, True, True],
        }
    )
    u = U._apply_rules(
        _listing(rows), attrs, non_krw=frozenset({"C1", "C2", "C6", "C10"})
    ).set_index("ticker")
    assert u.loc["000010", "exclusion_reason"] == "financial"  # induty 64110 for all rows
    assert u.loc["000060", "exclusion_reason"] == "spac"
    assert u.loc["000105", "exclusion_reason"] == "only_preferred"
    # without financial: spac beats liq_notice beats non_krw
    attrs["induty_code"] = "10000"
    u = U._apply_rules(_listing(rows), attrs, non_krw=frozenset({"C1"})).set_index("ticker")
    assert u.loc["000010", "exclusion_reason"] == "spac"
    attrs["is_spac"] = False
    u = U._apply_rules(_listing(rows), attrs, non_krw=frozenset({"C1"})).set_index("ticker")
    assert u.loc["000010", "exclusion_reason"] == "liq_notice"
    rows = [(t, c, m, h, False, p) for t, c, m, h, _, p in rows]
    u = U._apply_rules(_listing(rows), attrs, non_krw=frozenset({"C1"})).set_index("ticker")
    assert u.loc["000010", "exclusion_reason"] == "non_krw"
    # p6_tradable never changes in_universe; None -> False
    assert not u.loc["000020", "p6_tradable"] and u.loc["000020", "in_universe"]
    assert u.loc["000060", "in_universe"] and not u.loc["000060", "p6_tradable"]


def test_non_krw_corp_codes_and_default():
    fs = [("C9", "KRW"), ("C9", "USD"), ("C8", "KRW"), ("C8", None)]
    con = _base(fs=fs)
    assert U.non_krw_corp_codes(con) == frozenset({"C9"})
    u = U.build_universe(con, [D(2024, 12, 31)]).set_index("ticker")  # default None -> computed
    assert u.loc["000090", "is_non_krw"] and not u.loc["000080", "is_non_krw"]
    u = U.build_universe(con, [D(2024, 12, 31)], non_krw_corp_codes=frozenset()).set_index("ticker")
    assert not u.loc["000090", "is_non_krw"]


def test_unmapped_preferred_groups_with_unmapped_common():
    rows = [
        ("00ABC0", None, "KOSPI", False, False, True),
        ("00ABC5", None, "KOSPI", False, False, True),
    ]
    attrs = pd.DataFrame({"ticker": ["00ABC0"], "induty_code": [None], "is_spac": [False]})
    u = U._apply_rules(_listing(rows), attrs)
    assert list(u.ticker) == ["00ABC0"] and u.in_universe.all()


def test_columns():
    u = U.build_universe(_base(), [D(2024, 12, 31)], non_krw_corp_codes=frozenset())
    assert list(u.columns) == U.COLUMNS
    with pytest.raises(KeyError):
        u["nothing"]
