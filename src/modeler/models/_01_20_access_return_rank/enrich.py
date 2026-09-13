"""enrich — display-only enrichment for the top-k buy list (model 01).

Attaches human-facing reference columns to a per-date top-k frame WITHOUT
touching the model / ETL pipeline:

  * ``name``            — Korean security name (``stock_master.name``, latest)
  * ``market_cap_eok``  — market cap in 억원 = ``daily_ohlcv.close`` × shares
  * ``close`` / ``shares``                — market-cap inputs (kept for audit)
  * ``revenue_eok`` / ``operating_income_eok`` — sales / operating income (억원)
  * ``operating_margin``                  — operating_income / revenue
  * ``net_income_eok`` / ``total_equity_eok`` — higher-coverage backups + ROE
  * ``roe``                               — net_income / total_equity

Point-in-time (PIT) safety: financial metrics from ``stock_metric_fact`` carry
NO disclosure date, so the SAME conservative lag as ``modeler.etl.features.
fin_pit`` is used (annual ``period_end+90d``, quarterly ``+45d``). For each row
``(trade_date, ticker)`` only reports with ``available_from <= trade_date`` are
used (``join_asof`` backward), so the holdout backtest carries NO look-ahead.
``issued_shares`` is taken the same as-of way, so market cap is also PIT-consistent.

Names are display-only (latest ``stock_master`` value), so no PIT concern.

Coverage caveat: ``revenue`` / ``operating_income`` exist for only ~200 names in
this DB, so most rows will be NULL; ``net_income`` / ``total_equity`` cover ~2,600.
"""

from __future__ import annotations

import os

import polars as pl

from modeler.etl.features.fin_pit import ANNUAL_LAG_DAYS, QUARTERLY_LAG_DAYS

# Amount metrics shown in 억원; ``issued_shares`` (count) drives market cap.
_AMOUNT_METRICS = ("revenue", "operating_income", "net_income", "total_equity")
_FUND_METRICS = (*_AMOUNT_METRICS, "issued_shares")

# 1 억원 = 1e8 KRW.
_EOK = 1e8


def _db_dsn() -> str:
    """Resolve the Postgres DSN from the environment (``.env`` fallback)."""
    dsn = os.environ.get("DB_DSN")
    if not dsn:
        try:
            from dotenv import dotenv_values

            dsn = dotenv_values(".env").get("DB_DSN")
        except Exception:  # pragma: no cover - dotenv optional
            dsn = None
    if not dsn:
        raise SystemExit("DB_DSN not set (env or .env) — cannot enrich top-k.")
    return dsn


def _fetch(dsn: str, sql: str, params: tuple | None = None) -> pl.DataFrame:
    """Run a query and return a polars frame (numerics already cast to float)."""
    import psycopg2

    with psycopg2.connect(dsn) as conn:
        cur = conn.cursor()
        cur.execute(sql, params or ())
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pl.DataFrame(rows, schema=cols, orient="row")


def _load_names(dsn: str) -> pl.DataFrame:
    return _fetch(dsn, "SELECT ticker, market, name FROM stock_master")


def _load_close(dsn: str, dates: list) -> pl.DataFrame:
    sql = (
        "SELECT trade_date, ticker, market, CAST(close AS DOUBLE PRECISION) AS close "
        "FROM daily_ohlcv WHERE trade_date = ANY(%s)"
    )
    return _fetch(dsn, sql, (dates,))


def _load_fundamentals(dsn: str) -> pl.DataFrame:
    """``stock_metric_fact`` deduped to latest period_end per disclosure lag.

    Mirrors ``fin_pit.build_available_sql`` (Postgres has no QUALIFY, so a
    ROW_NUMBER subquery is used). Returns (ticker, metric_code, value,
    available_from).
    """
    lag = (
        f"(CASE WHEN period_type = 'annual' "
        f"THEN period_end + INTERVAL '{ANNUAL_LAG_DAYS} days' "
        f"ELSE period_end + INTERVAL '{QUARTERLY_LAG_DAYS} days' END)::date"
    )
    sql = f"""
        SELECT ticker, metric_code, value, available_from
        FROM (
            SELECT ticker, metric_code,
                   CAST(value_numeric AS DOUBLE PRECISION) AS value,
                   {lag} AS available_from,
                   ROW_NUMBER() OVER (
                       PARTITION BY ticker, metric_code, {lag}
                       ORDER BY period_end DESC
                   ) AS rn
            FROM stock_metric_fact
            WHERE metric_code = ANY(%s)
        ) t
        WHERE rn = 1
    """
    return _fetch(dsn, sql, (list(_FUND_METRICS),))


def _asof_metric(topk: pl.DataFrame, avail: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Backward as-of join of a single metric onto ``topk`` by ticker.

    Adds one column named ``metric`` carrying the latest value disclosed on or
    before each row's ``trade_date``.
    """
    right = (
        avail.filter(pl.col("metric_code") == metric)
        .select("ticker", "available_from", pl.col("value").alias(metric))
        .sort("available_from")
    )
    if right.height == 0:
        return topk.with_columns(pl.lit(None, dtype=pl.Float64).alias(metric))
    return (
        topk.sort("trade_date")
        .join_asof(
            right,
            left_on="trade_date",
            right_on="available_from",
            by="ticker",
            strategy="backward",
        )
        .drop("available_from")
    )


def enrich_topk(topk: pl.DataFrame) -> pl.DataFrame:
    """Attach name / market cap / financials to a top-k frame (display-only).

    PIT-safe (see module docstring). Missing values stay NULL. Returns the frame
    re-sorted by (trade_date, rank) with the reference columns inserted after the
    identifier columns.
    """
    dsn = _db_dsn()
    dates = topk.get_column("trade_date").unique().to_list()

    names = _load_names(dsn)
    close = _load_close(dsn, dates)
    avail = _load_fundamentals(dsn)

    out = topk.join(names, on=["ticker", "market"], how="left")
    out = out.join(close, on=["trade_date", "ticker", "market"], how="left")
    for metric in _FUND_METRICS:
        out = _asof_metric(out, avail, metric)

    out = out.with_columns(
        (pl.col("close") * pl.col("issued_shares") / _EOK).alias("market_cap_eok"),
        pl.col("issued_shares").alias("shares"),
    )
    # Amount metrics -> 억원, plus derived ratios.
    out = out.with_columns(
        *[(pl.col(m) / _EOK).alias(f"{m}_eok") for m in _AMOUNT_METRICS],
        pl.when(pl.col("revenue") != 0)
        .then(pl.col("operating_income") / pl.col("revenue"))
        .otherwise(None)
        .alias("operating_margin"),
        pl.when(pl.col("total_equity") != 0)
        .then(pl.col("net_income") / pl.col("total_equity"))
        .otherwise(None)
        .alias("roe"),
    )

    ref_cols = [
        "name",
        "market_cap_eok",
        "close",
        "shares",
        "revenue_eok",
        "operating_income_eok",
        "operating_margin",
        "net_income_eok",
        "total_equity_eok",
        "roe",
    ]
    lead = [c for c in ("trade_date", "rank", "ticker", "market") if c in out.columns]
    rest = [c for c in out.columns if c not in lead and c not in ref_cols]
    # Drop the raw KRW helper columns now folded into *_eok (keep close/shares).
    drop_raw = [m for m in _AMOUNT_METRICS] + ["issued_shares"]
    rest = [c for c in rest if c not in drop_raw]
    ordered = lead + ref_cols + rest
    return out.select(ordered).sort(["trade_date", "rank"])
