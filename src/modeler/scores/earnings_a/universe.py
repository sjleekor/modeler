"""Earnings forecast A: the listed universe at each decision date (decisions A1-A4, A22, A23).

One row per (decision_date, company) where a company is represented by its common line
when that line is listed on the session ``s_t``, and by one preferred line otherwise.
``s_t`` is the latest ``daily_ohlcv`` trade date on or before the decision date, over all
tickers. Listed means a row on ``s_t``. A halted row (open = high = low = 0) is listed and
flagged, never dropped. No financial value is read here.

``in_universe`` is common and not preferred-only and not financial (holding companies are a
subset of financial) and not SPAC and not on a liquidation notice and not non-KRW.
``exclusion_reason`` is the first that applies, in this fixed priority:
only_preferred, financial, spac, liq_notice, non_krw, else None.
``p6_tradable`` is a column for the record and never a filter.
"""

from __future__ import annotations

from datetime import date

import duckdb
import numpy as np
import pandas as pd

from modeler.etl.universe import tradable_universe_filter
from modeler.etl.universe_raw_price import build_universe_raw_price_sql
from modeler.models._03_kr_next.liquidation_notice import build_liquidation_notice_sql
from modeler.scores.earnings_a.spec import FINANCIAL_KSIC2, HOLDING_KSIC, SPAC_PATTERNS

MARKETS = ("KOSPI", "KOSDAQ")
EXCLUSION_PRIORITY = ("only_preferred", "financial", "spac", "liq_notice", "non_krw")

COLUMNS = [
    "decision_date",
    "session_date",
    "ticker",
    "corp_code",
    "market",
    "ksic2",
    "induty_code",
    "is_halted",
    "is_financial",
    "is_holding",
    "is_spac",
    "liq_notice",
    "is_non_krw",
    "p6_tradable",
    "only_preferred_listed",
    "in_universe",
    "exclusion_reason",
]


def session_dates(trade_dates: list[date], decision_dates: list[date]) -> dict[date, date | None]:
    """Map each decision date to the latest trade date on or before it (None if none)."""
    ordered = sorted(set(trade_dates))
    out: dict[date, date | None] = {}
    for t in decision_dates:
        pos = int(
            np.searchsorted(np.array(ordered, dtype="datetime64[D]"), np.datetime64(t), "right")
        )
        out[t] = ordered[pos - 1] if pos > 0 else None
    return out


def _apply_rules(
    listing: pd.DataFrame,
    attrs: pd.DataFrame,
    *,
    non_krw: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Pure rules over the listed lines of every decision date.

    ``listing``: decision_date, session_date, ticker, corp_code (None allowed), market,
    is_halted, liq_notice, p6_tradable (None allowed -> False). One row per listed line.
    ``attrs``: ticker, induty_code (None allowed), is_spac; missing tickers get no attributes.
    Returns one row per (decision_date, company) with the module's columns.
    """
    df = listing.merge(attrs[["ticker", "induty_code", "is_spac"]], on="ticker", how="left")
    df["is_common"] = df["ticker"].str[-1] == "0"
    # Company key: the DART corp_code, else the first five characters of the ticker, so a
    # preferred line without a mapping still groups with its unmapped common sibling.
    df["_key"] = df["corp_code"].where(df["corp_code"].notna(), "T:" + df["ticker"].str[:5])
    df["_order"] = (~df["is_common"]).astype(int)
    df = df.sort_values(["decision_date", "_key", "_order", "ticker"], kind="stable")
    has_common = df.groupby(["decision_date", "_key"])["is_common"].transform("any")
    first = ~df.duplicated(["decision_date", "_key"], keep="first")
    df = df[first].copy()
    df["only_preferred_listed"] = ~df["is_common"]
    assert not (df["only_preferred_listed"] & has_common[first]).any()

    induty = df["induty_code"].astype("object").where(df["induty_code"].notna(), None)
    df["induty_code"] = induty
    df["ksic2"] = induty.map(lambda v: v[:2] if isinstance(v, str) and len(v) >= 2 else None)
    df["is_financial"] = df["ksic2"].isin(FINANCIAL_KSIC2)
    df["is_holding"] = df["induty_code"] == HOLDING_KSIC
    df["is_spac"] = df["is_spac"].astype("object").where(df["is_spac"].notna(), False).astype(bool)
    df["liq_notice"] = (
        df["liq_notice"].astype("object").where(df["liq_notice"].notna(), False).astype(bool)
    )
    df["p6_tradable"] = (
        df["p6_tradable"].astype("object").where(df["p6_tradable"].notna(), False).astype(bool)
    )
    df["is_halted"] = (
        df["is_halted"].astype("object").where(df["is_halted"].notna(), False).astype(bool)
    )
    df["is_non_krw"] = df["corp_code"].isin(non_krw)

    flags = {
        "only_preferred": df["only_preferred_listed"],
        "financial": df["is_financial"],
        "spac": df["is_spac"],
        "liq_notice": df["liq_notice"],
        "non_krw": df["is_non_krw"],
    }
    reason = pd.Series(None, index=df.index, dtype="object")
    for name in reversed(EXCLUSION_PRIORITY):
        reason = reason.where(~flags[name], name)
    df["in_universe"] = reason.isna()
    df["exclusion_reason"] = reason.astype("object").where(reason.notna(), None)
    df = df.sort_values(["decision_date", "ticker"]).reset_index(drop=True)
    return df[COLUMNS]


def _spac_sql() -> str:
    cond = " OR ".join(f"contains(name, '{p}')" for p in SPAC_PATTERNS)
    return f"""
        SELECT ticker, bool_or(COALESCE({cond}, FALSE)) AS is_spac
        FROM stock_master
        WHERE market IN ('KOSPI', 'KOSDAQ')
        GROUP BY ticker
    """


def non_krw_corp_codes(con: duckdb.DuckDBPyConnection) -> frozenset[str]:
    """Corp codes with any ``dart_financial_statement_raw`` row whose currency is not KRW (A4).

    A static firm-level set over the whole history (a currency-validity filter, not related
    to outcomes). Limitation: a firm that switched KRW to USD is excluded for every date.
    """
    rows = con.execute(
        "SELECT DISTINCT corp_code FROM dart_financial_statement_raw "
        "WHERE currency IS NOT NULL AND currency <> 'KRW' AND corp_code IS NOT NULL"
    ).fetchall()
    return frozenset(r[0] for r in rows)


_non_krw_set = non_krw_corp_codes


def build_universe(
    con: duckdb.DuckDBPyConnection,
    decision_dates: list[date],
    *,
    non_krw_corp_codes: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Build the universe table for ``decision_dates`` (see the module doc).

    ``non_krw_corp_codes=None`` computes the set with :func:`non_krw_corp_codes`; an
    explicit frozenset (also an empty one) overrides it.
    """
    decision_dates = list(decision_dates)
    non_krw = _non_krw_set(con) if non_krw_corp_codes is None else non_krw_corp_codes
    trade_dates = [
        r[0] for r in con.execute("SELECT DISTINCT trade_date FROM daily_ohlcv").fetchall()
    ]
    sess = session_dates(trade_dates, decision_dates)
    needed = sorted({s for s in sess.values() if s is not None})
    if not needed:
        return pd.DataFrame(columns=COLUMNS)

    con.execute("CREATE OR REPLACE TEMP TABLE _ea_sessions(trade_date DATE)")
    con.executemany("INSERT INTO _ea_sessions VALUES (?)", [(s,) for s in needed])

    flt = tradable_universe_filter()
    corp_sql = """
        SELECT ticker, MIN(corp_code) AS corp_code, MIN(induty_code) AS induty_code
        FROM dart_corp_master
        WHERE ticker IS NOT NULL AND ticker <> '' AND corp_code IS NOT NULL AND corp_code <> ''
        GROUP BY ticker
    """
    lines = con.execute(
        f"""
        WITH corp AS ({corp_sql}),
        o AS (
            SELECT trade_date, ticker, arg_min(market, market) AS market,
                   bool_or(open = 0 AND high = 0 AND low = 0) AS is_halted
            FROM daily_ohlcv
            WHERE trade_date IN (SELECT trade_date FROM _ea_sessions)
            GROUP BY trade_date, ticker
        ),
        liq AS (
            SELECT trade_date, ticker, bool_or(liq_notice_flag) AS liq_notice
            FROM ({build_liquidation_notice_sql()})
            WHERE trade_date IN (SELECT trade_date FROM _ea_sessions)
            GROUP BY trade_date, ticker
        ),
        p6 AS (
            SELECT trade_date, ticker, bool_or(tradable_ok) AS p6_tradable
            FROM ({build_universe_raw_price_sql(flt)})
            WHERE trade_date IN (SELECT trade_date FROM _ea_sessions)
            GROUP BY trade_date, ticker
        )
        SELECT o.trade_date AS session_date, o.ticker, o.market, o.is_halted,
               COALESCE(c1.corp_code, c2.corp_code) AS corp_code,
               COALESCE(c1.induty_code, c2.induty_code) AS induty_code,
               liq.liq_notice, p6.p6_tradable
        FROM o
        LEFT JOIN corp c1 ON c1.ticker = o.ticker
        LEFT JOIN corp c2
          ON RIGHT(o.ticker, 1) <> '0' AND c2.ticker = SUBSTR(o.ticker, 1, 5) || '0'
        LEFT JOIN liq ON liq.trade_date = o.trade_date AND liq.ticker = o.ticker
        LEFT JOIN p6 ON p6.trade_date = o.trade_date AND p6.ticker = o.ticker
        """  # noqa: S608
    ).df()
    spac = con.execute(_spac_sql()).df()
    con.execute("DROP TABLE IF EXISTS _ea_sessions")

    lines["session_date"] = pd.to_datetime(lines["session_date"]).dt.date
    parts = []
    for t in decision_dates:
        s = sess[t]
        if s is None:
            continue
        part = lines[lines["session_date"] == s].copy()
        part.insert(0, "decision_date", t)
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=COLUMNS)
    listing = pd.concat(parts, ignore_index=True)
    attrs = (
        lines[["ticker", "induty_code"]]
        .drop_duplicates("ticker")
        .merge(spac, on="ticker", how="left")
    )
    listing = listing.drop(columns=["induty_code"])
    listing["corp_code"] = (
        listing["corp_code"].astype("object").where(listing["corp_code"].notna(), None)
    )
    attrs["induty_code"] = (
        attrs["induty_code"].astype("object").where(attrs["induty_code"].notna(), None)
    )
    return _apply_rules(listing, attrs, non_krw=non_krw)
