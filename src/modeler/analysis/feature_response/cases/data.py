"""Readers for the development panel, daily prices, names and corporate actions.

Every read goes through ``guard_path`` and is cut at ``DATA_END``. The panel carries the 20-day
labels in the same file, so the frame is split right after loading: the selection side gets
``trade_date, ticker, market, value`` only.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from .config import (
    CORP_ACTIONS,
    DATA_END,
    DATASET_MANIFEST_REL,
    LIMIT_HI,
    LIMIT_LO,
    PANEL_REL,
    PRE_DAYS,
    RAW_REL,
)
from .guards import guard_path

LABEL_COLS = ["fwd_ret_20d", "raw_label_20d"]


def panel_path(stock_data: Path) -> Path:
    return guard_path(stock_data / PANEL_REL)


def load_panel(path: Path, features: list[str]) -> pl.DataFrame:
    """Panel rows up to ``DATA_END`` with the columns this tool needs."""
    p = guard_path(path)
    cols = ["trade_date", "ticker", "market", "px_ret_1d", "fin_is_negative_equity", *LABEL_COLS]
    cols += [f for f in features if f not in cols]
    df = pl.scan_parquet(p).filter(pl.col("trade_date") <= DATA_END).select(cols).collect()
    if df.height and df["trade_date"].max() > DATA_END:
        raise RuntimeError("panel rows after DATA_END survived the scan filter")
    return df


def calendar_of(panel: pl.DataFrame) -> list[dt.date]:
    return panel["trade_date"].unique().sort().to_list()


def feature_frame(
    panel: pl.DataFrame, feature: str, dates: list[dt.date] | None = None
) -> pl.DataFrame:
    """Selection input: no outcome columns."""
    df = panel
    if dates is not None:
        df = df.filter(pl.col("trade_date").is_in(dates))
    return df.select(
        "trade_date", "ticker", "market", pl.col(feature).cast(pl.Float64).alias("value")
    )


def market_returns(panel: pl.DataFrame, cal: list[dt.date]) -> dict[str, np.ndarray]:
    """Per market equal-weight mean of ``px_ret_1d`` on each calendar day (nan -> 0)."""
    g = panel.group_by(["market", "trade_date"]).agg(pl.col("px_ret_1d").mean().alias("m"))
    idx = {d: i for i, d in enumerate(cal)}
    out: dict[str, np.ndarray] = {}
    for mk in sorted(g["market"].unique().to_list()):
        arr = np.zeros(len(cal))
        sub = g.filter(pl.col("market") == mk)
        for d, m in zip(sub["trade_date"].to_list(), sub["m"].to_list(), strict=True):
            if m is not None and m == m:
                arr[idx[d]] = max(m, -0.99)
        out[mk] = arr
    return out


@dataclass
class Prices:
    cal: list[dt.date]
    tid: dict[str, int]
    close: np.ndarray  # (tickers, days), nan where there is no row
    halt: np.ndarray  # bool, open = high = low = 0
    logcum: dict[str, np.ndarray]  # per market cumulative log(1 + equal-weight daily mean)

    def row(self, ticker: str) -> int | None:
        return self.tid.get(ticker)


def load_prices(stock_data: Path, cal: list[dt.date], mret: dict[str, np.ndarray]) -> Prices:
    root = guard_path(stock_data / RAW_REL / "daily_ohlcv")
    o = (
        pl.scan_parquet(str(root / "**/*.parquet"), hive_partitioning=True)
        .filter((pl.col("trade_date") >= cal[0]) & (pl.col("trade_date") <= DATA_END))
        .select("trade_date", "ticker", "open", "high", "low", "close")
        .collect()
    )
    if o.height and o["trade_date"].max() > DATA_END:
        raise RuntimeError("price rows after DATA_END survived the scan filter")
    ci = pl.DataFrame({"trade_date": cal, "ci": list(range(len(cal)))}).with_columns(
        pl.col("ci").cast(pl.Int32)
    )
    tickers = sorted(o["ticker"].unique().to_list())
    ti = pl.DataFrame({"ticker": tickers, "ti": list(range(len(tickers)))}).with_columns(
        pl.col("ti").cast(pl.Int32)
    )
    o = o.join(ci, on="trade_date").join(ti, on="ticker")
    close = np.full((len(tickers), len(cal)), np.nan)
    halt = np.zeros((len(tickers), len(cal)), dtype=bool)
    t = o["ti"].to_numpy()
    c = o["ci"].to_numpy()
    close[t, c] = o["close"].cast(pl.Float64).to_numpy()
    halt[t, c] = ((o["open"] == 0) & (o["high"] == 0) & (o["low"] == 0)).to_numpy()
    return Prices(
        cal=cal,
        tid={k: i for i, k in enumerate(tickers)},
        close=close,
        halt=halt,
        logcum={k: np.cumsum(np.log1p(v)) for k, v in mret.items()},
    )


def load_names(stock_data: Path) -> dict[str, str]:
    p = guard_path(
        stock_data / RAW_REL / "stock_master" / "schema_version=1" / "part-000000.parquet"
    )
    df = pl.read_parquet(p, columns=["ticker", "name"])
    return dict(zip(df["ticker"].to_list(), df["name"].to_list(), strict=True))


def is_preferred(ticker: str, name: str | None) -> bool:
    """Name ends with 우 / 우B / 우C / 우(전환), or the ticker does not end in 0."""
    n = (name or "").strip()
    by_name = n.endswith(("우", "우B", "우C", "우(전환)"))
    return by_name or (len(ticker) == 6 and not ticker.endswith("0"))


def load_corp_actions(stock_data: Path) -> pl.DataFrame:
    """Split / bonus issue / capital reduction events from DART capital change rows."""
    root = guard_path(stock_data / RAW_REL / "dart_capital_change_raw")
    df = (
        pl.scan_parquet(str(root / "**/*.parquet"))
        .filter(pl.col("isu_dcrs_stle").is_in(list(CORP_ACTIONS)))
        .filter(pl.col("isu_dcrs_de").is_not_null() & (pl.col("isu_dcrs_de") <= DATA_END))
        .select(
            "ticker",
            pl.col("isu_dcrs_de").alias("event_date"),
            pl.col("isu_dcrs_stle").alias("kind"),
        )
        .unique()
        .collect()
    )
    return df.filter(pl.col("ticker").is_not_null())


def case_window(prices: Prices, ci0: int) -> tuple[int, int]:
    lo = ci0 - PRE_DAYS
    hi = ci0 + 20
    if lo < 0 or hi >= len(prices.cal):
        raise IndexError("case window leaves the calendar")
    return lo, hi


def case_series(prices: Prices, ticker: str, market: str, ci0: int) -> dict | None:
    """Normalised price (formation close = 100) and market line for day -120 .. +20."""
    r = prices.row(ticker)
    if r is None:
        return None
    lo, hi = case_window(prices, ci0)
    c = prices.close[r, lo : hi + 1].copy()
    base = prices.close[r, ci0]
    if not np.isfinite(base) or base <= 0:
        return None
    # carry the last close over missing days, leave leading / trailing gaps empty
    last = np.nan
    for i in range(len(c)):
        if np.isfinite(c[i]):
            last = c[i]
        elif np.isfinite(last) and i <= PRE_DAYS:
            c[i] = last
    L = prices.logcum[market]
    m = 100.0 * np.exp(L[lo : hi + 1] - L[ci0])
    return {
        "rel": list(range(-PRE_DAYS, 21)),
        "close_norm": [None if not np.isfinite(x) else float(x / base * 100.0) for x in c],
        "mkt_norm": [float(x) for x in m],
        "halt": [bool(x) for x in prices.halt[r, lo : hi + 1]],
    }


def case_flags(prices: Prices, ticker: str, ci0: int) -> dict:
    r = prices.row(ticker)
    out = {"halt_fwd": 0, "halt_prev20": 0, "limit_hits": 0, "jump_over_limit": 0}
    if r is None:
        return out
    out["halt_fwd"] = int(prices.halt[r, ci0 + 1 : ci0 + 21].sum())
    out["halt_prev20"] = int(prices.halt[r, max(0, ci0 - 19) : ci0 + 1].sum())
    seg = prices.close[r, ci0 : ci0 + 21]
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.abs(seg[1:] / seg[:-1] - 1.0)
    ok = np.isfinite(ret) & ~prices.halt[r, ci0 + 1 : ci0 + 21]
    out["limit_hits"] = int(((ret >= LIMIT_LO) & (ret <= LIMIT_HI) & ok).sum())
    out["jump_over_limit"] = int(((ret > LIMIT_HI) & ok).sum())
    return out


def corp_action_text(actions: pl.DataFrame, prices: Prices, ticker: str, ci0: int) -> str:
    lo, hi = case_window(prices, ci0)
    a, b = prices.cal[lo], prices.cal[hi]
    ev = actions.filter(
        (pl.col("ticker") == ticker) & (pl.col("event_date") >= a) & (pl.col("event_date") <= b)
    ).sort("event_date")
    return "; ".join(
        f"{k} {d.isoformat()}" for d, k in zip(ev["event_date"], ev["kind"], strict=True)
    )


def path_cum_excess(prices: Prices, tid: np.ndarray, ci0: np.ndarray, mk: list[str]) -> np.ndarray:
    """Stock cumulative return minus market equal-weight, day 0..20, shape (rows, 21)."""
    k = np.arange(21)
    idx = ci0[:, None] + k[None, :]
    c = prices.close[tid[:, None], idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        stock = c / c[:, [0]] - 1.0
    mkt = np.zeros_like(stock)
    mk_arr = np.array(mk)
    for name, L in prices.logcum.items():
        sel = mk_arr == name
        if sel.any():
            mkt[sel] = np.exp(L[idx[sel]] - L[ci0[sel]][:, None]) - 1.0
    return stock - mkt


def dataset_manifest_path(stock_data: Path) -> Path:
    return guard_path(stock_data / DATASET_MANIFEST_REL)
