# ruff: noqa: E501
"""Price-quality, corporate-action and short-selling availability helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view
from modeler.etl.trading_panel import build_full_panel_sql

QUALITY_TABLE = "dim_price_quality_daily"


@dataclass(frozen=True)
class ShortRegime:
    start: str
    end: str | None
    regime: str


SHORT_REGIMES: tuple[ShortRegime, ...] = (
    ShortRegime("2014-06-01", "2020-03-13", "allowed"),
    ShortRegime("2020-03-16", "2021-05-02", "banned"),
    ShortRegime("2021-05-03", "2023-11-03", "partial"),
    ShortRegime("2023-11-06", "2025-03-28", "banned"),
    ShortRegime("2025-03-31", None, "allowed"),
)


def short_regime_sql(date_expr: str = "trade_date") -> str:
    parts = [f"WHEN {date_expr} BETWEEN DATE '{r.start}' AND DATE '{r.end}' THEN '{r.regime}'"
             if r.end else f"WHEN {date_expr} >= DATE '{r.start}' THEN '{r.regime}'"
             for r in SHORT_REGIMES]
    return "CASE " + " ".join(parts) + " ELSE 'unknown' END"


def build_price_quality_sql(
    *,
    price_view: str = "daily_ohlcv",
    pit_view: str = "dim_stock_pit_daily",
    price_limit_multiplier: float = 1.10,
    share_change_threshold: float = 0.25,
    share_ratio_low: float = 0.8,
    share_ratio_high: float = 1.25,
    short_balance_lag_sessions: int = 2,
) -> str:
    """Create quality rows while retaining the original OHLCV grain."""
    panel = build_full_panel_sql(price_view)
    return f"""
        WITH panel AS ({panel}),
        joined AS (
            SELECT p.*,
                   s.issued_shares_pit,
                   LAG(s.issued_shares_pit) OVER (
                       PARTITION BY p.ticker, p.market ORDER BY p.trade_date
                   ) AS prev_issued_shares,
                   LAG(p.close_d) OVER (
                       PARTITION BY p.ticker, p.market ORDER BY p.trade_date
                   ) AS prev_close_d
            FROM panel p
            LEFT JOIN {pit_view} s USING (trade_date, ticker, market)
        ),
        flags AS (
            SELECT *,
                CASE
                    WHEN is_halted OR prev_close_d IS NULL OR prev_close_d <= 0 THEN FALSE
                    WHEN trade_date < DATE '2014-06-01' THEN NULL
                    WHEN ABS(simple_ret) >
                         (CASE WHEN trade_date < DATE '2015-06-15' THEN 0.15 ELSE 0.30 END)
                         * {price_limit_multiplier}
                        THEN TRUE
                    ELSE FALSE
                END AS ca_price_jump_suspect,
                CASE
                    WHEN prev_issued_shares IS NULL OR issued_shares_pit IS NULL
                         OR prev_issued_shares <= 0 OR issued_shares_pit <= 0
                         OR ABS(issued_shares_pit / prev_issued_shares - 1) < {share_change_threshold}
                        THEN FALSE
                    WHEN close_d / NULLIF(prev_close_d, 0)
                         * issued_shares_pit / prev_issued_shares
                         BETWEEN {share_ratio_low} AND {share_ratio_high}
                        THEN TRUE
                    ELSE FALSE
                END AS ca_share_change_confirmed
            FROM joined
        ),
        events AS (
            SELECT *,
                -- NULL means applicability is unknown; mask conservatively.
                (COALESCE(ca_price_jump_suspect, TRUE)
                 OR COALESCE(ca_share_change_confirmed, FALSE)) AS ca_event
            FROM flags
        ),
        cumulative AS (
            SELECT *,
                SUM(CASE WHEN ca_event THEN 1 ELSE 0 END) OVER (
                    PARTITION BY ticker, market ORDER BY trade_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS ca_event_cumulative
            FROM events
        ),
        calendar AS (
            SELECT DISTINCT trade_date,
                ROW_NUMBER() OVER (ORDER BY trade_date) AS session_idx
            FROM cumulative WHERE valid_session_idx IS NOT NULL
        ),
        cutoff AS (
            SELECT MIN(session_idx) AS base_idx FROM calendar
            WHERE trade_date >= DATE '2016-06-30'
        ),
        availability AS (
            SELECT MIN(c.trade_date) AS short_balance_available_from
            FROM calendar c, cutoff x
            WHERE c.session_idx >= x.base_idx + {short_balance_lag_sessions}
        )
        SELECT
            c.trade_date, c.ticker, c.market, c.valid_session_idx,
            c.is_halted, c.volume_zero, c.simple_ret, c.log_ret,
            c.ca_price_jump_suspect, c.ca_share_change_confirmed,
            (c.ca_event) AS ca_mask,
            (c.ca_price_jump_suspect IS NULL) AS ca_rule_applicability_unknown,
            c.ca_event_cumulative, c.ca_event,
            {short_regime_sql('c.trade_date')} AS short_regime,
            (c.trade_date >= a.short_balance_available_from) AS short_balance_is_available,
            a.short_balance_available_from
        FROM cumulative c CROSS JOIN availability a
    """


def materialize_price_quality(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    price_view: str = "daily_ohlcv",
    pit_view: str = "dim_stock_pit_daily",
    price_limit_multiplier: float = 1.10,
    share_change_threshold: float = 0.25,
    share_ratio_low: float = 0.8,
    share_ratio_high: float = 1.25,
    short_balance_lag_sessions: int = 2,
    force: bool = False,
) -> str:
    materialize(
        con,
        config,
        QUALITY_TABLE,
        build_price_quality_sql(
            price_view=price_view,
            pit_view=pit_view,
            price_limit_multiplier=price_limit_multiplier,
            share_change_threshold=share_change_threshold,
            share_ratio_low=share_ratio_low,
            share_ratio_high=share_ratio_high,
            short_balance_lag_sessions=short_balance_lag_sessions,
        ),
        force=force,
    )
    return register_mart_view(con, config, QUALITY_TABLE)


MAX_PLAUSIBLE_LAG_SESSIONS = 10

EVIDENCE_BASIS = (
    "src/collector/kr/adapters/flows_krx/parsers.py: short_selling_balance_quantity "
    "is keyed off the KRX row field RPT_DUTY_OCCR_DD, reporting-duty-occurrence date "
    "(bogo uimu balsaeng-il), which is a measurement date, not a disclosure date",
    "empirical: krx_security_flow_raw max(trade_date) for short_selling_balance_quantity "
    "trails max(trade_date) for short_selling_volume/short_selling_value from the same "
    "fetch by a fixed number of KRX trading sessions",
    "regulatory context: Financial Investment Services and Capital Markets Act "
    "Enforcement Decree Article 14-3 (net short position reporting) requires filing "
    "by the reporting-duty date plus 2 business days",
)


def diagnose_publication_lag(
    con: duckdb.DuckDBPyConnection,
    *,
    flow_raw_view: str = "krx_security_flow_raw",
    price_view: str = "daily_ohlcv",
    balance_metric: str = "short_selling_balance_quantity",
    reference_metrics: tuple[str, ...] = ("short_selling_volume", "short_selling_value"),
    max_plausible_lag_sessions: int = MAX_PLAUSIBLE_LAG_SESSIONS,
) -> dict:
    """Empirically diagnose the short-balance publication lag from the connected raw lake.

    Same-fetch flow metrics that are not subject to the T+2 filing deadline
    (short_selling_volume/value) expose the true gap: how many KRX trading
    sessions the balance metric's max observed trade_date trails behind them.
    """
    placeholders = ", ".join("?" for _ in reference_metrics)
    balance_max, reference_max = con.execute(
        f"""
        SELECT
            max(CASE WHEN metric_code = ? THEN trade_date END),
            max(CASE WHEN metric_code IN ({placeholders}) THEN trade_date END)
        FROM {flow_raw_view}
        """,
        [balance_metric, *reference_metrics],
    ).fetchone()
    if balance_max is None or reference_max is None or balance_max >= reference_max:
        return {
            "verified": False,
            "public_lag_sessions": None,
            "balance_max_trade_date": str(balance_max) if balance_max else None,
            "reference_max_trade_date": str(reference_max) if reference_max else None,
            "reason": "insufficient_raw_coverage_or_no_observed_gap",
        }
    lag_sessions = con.execute(
        f"""
        SELECT count(DISTINCT trade_date)
        FROM {price_view}
        WHERE trade_date > ? AND trade_date <= ?
        """,
        [balance_max, reference_max],
    ).fetchone()[0]
    plausible = 0 < lag_sessions <= max_plausible_lag_sessions
    return {
        "verified": plausible,
        "public_lag_sessions": lag_sessions if plausible else None,
        "balance_max_trade_date": str(balance_max),
        "reference_max_trade_date": str(reference_max),
        "reason": (
            "empirical_session_gap_between_balance_and_reference_flow_metrics"
            if plausible
            else f"observed_gap_{lag_sessions}_sessions_outside_plausible_range"
        ),
    }


def write_publication_lag_evidence(
    path: Path,
    con: duckdb.DuckDBPyConnection | None = None,
    *,
    flow_raw_view: str = "krx_security_flow_raw",
    price_view: str = "daily_ohlcv",
) -> Path:
    """Write the auditable lag decision; unresolved is the safe default.

    When a connection is supplied, the lag is diagnosed empirically against the
    connected snapshot rather than assumed, so evidence tracks the actual raw
    lake instead of going stale.
    """
    if con is not None:
        diagnosis = diagnose_publication_lag(con, flow_raw_view=flow_raw_view, price_view=price_view)
    else:
        diagnosis = {
            "verified": False,
            "public_lag_sessions": None,
            "reason": "no_connection_supplied",
        }
    payload = {
        "status": "verified" if diagnosis["verified"] else "unresolved",
        "measurement_date_field": (
            "trade_date (sourced from KRX RPT_DUTY_OCCR_DD)" if diagnosis["verified"] else None
        ),
        "public_lag_sessions": diagnosis["public_lag_sessions"],
        "balance_max_trade_date": diagnosis.get("balance_max_trade_date"),
        "reference_max_trade_date": diagnosis.get("reference_max_trade_date"),
        "diagnosis_reason": diagnosis["reason"],
        "evidence_basis": list(EVIDENCE_BASIS),
        "reason": "A0 requires source evidence before balance families become official",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path
