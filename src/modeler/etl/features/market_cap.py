# ruff: noqa: E501
"""``feat_market_cap`` — size from the exchange's own share count (`mcap_krx_log`).

The existing size feature is `fin_log_mcap` in `fin_scan`, built as
`close_d x issued_raw` where `issued_raw` is DART's issued-share count. That
pairs a price which reflects a corporate action immediately with a share count
that does not move until the next periodic report is **disclosed** — months, not
days. Nothing gates it: `base_ok` checks availability and validity, and
`shares_age_days` is computed in `stock_pit` and never read.

`daily_market_cap.listed_shares` is the exchange's number and changes on the
**listing** date. That collapses the mismatch from months to about three weeks
and removes the disclosure-lag error class entirely — the reason N1 was collected.

Because the definition changes, this lands on a **new id** rather than moving
`fin_log_mcap` (N1-7 decision 1: frozen ids keep their definition, new
definitions get new ids). `fin_log_mcap` stays as published, with its lag
recorded as a limitation.

The residual three-week window is **masked, not corrected** — see
`modeler.etl.corporate_actions` for what the window is and
`poc/n1_validation.md` §5.5 for why estimating the ratio was rejected. The raw
value is still emitted as `mcap_krx` so the masking can be audited; only the
modelling column `mcap_krx_log` goes NULL.
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.corporate_actions import (
    MCAP_DISTORTION_VIEW,
    register_mcap_distortion_view,
)
from modeler.etl.mart import materialize, register_mart_view

MARKET_CAP_TABLE = "feat_market_cap"
MARKET_CAP_FORMULA_VERSION = "market_cap_v2"


def build_market_cap_sql(
    market_cap_view: str = "daily_market_cap",
    *,
    distortion_view: str | None = MCAP_DISTORTION_VIEW,
) -> str:
    """SQL producing ``feat_market_cap`` at ``(trade_date, ticker, market)``.

    ``distortion_view`` may be ``None`` to emit the unmasked series. That is for
    measuring how much the mask removes, not for feeding a model — leaving it off
    reintroduces the swing the mask exists to remove.
    """
    if distortion_view:
        join = f"LEFT JOIN {distortion_view} d USING (trade_date, ticker)"
        unreliable = "COALESCE(d.mcap_unreliable, FALSE)"
    else:
        join = ""
        unreliable = "FALSE"
    return f"""
        WITH base AS (
        SELECT
            m.trade_date,
            m.ticker,
            m.market,
            m.market_cap AS mcap_krx,
            {unreliable} AS mcap_unreliable,
            -- Masked, not corrected. A wrong value concentrated on bonus-issue
            -- names is worse in a factor test than a missing one: it moves the
            -- cross-sectional rank of exactly the companies whose event is
            -- correlated with past returns and small size.
            CASE WHEN NOT {unreliable} AND m.market_cap > 0
                 THEN ln(CAST(m.market_cap AS DOUBLE)) END AS mcap_krx_log
        FROM {market_cap_view} m
        {join}
        )
        SELECT
            *,
            lag(mcap_krx_log) OVER (
                PARTITION BY ticker, market ORDER BY trade_date
            ) AS mcap_krx_log_lag1
        FROM base
    """


def materialize_market_cap(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    market_cap_view: str = "daily_market_cap",
    filing_receipt_view: str = "dart_filing_receipt_raw",
    force: bool = False,
) -> str:
    """Build + register the ``feat_market_cap`` mart view. Returns the view name.

    Registers the distortion view first — the mask is part of the feature, not an
    optional decoration, so the two are never built apart.

    Args:
        con: DuckDB connection with the source views registered.
        config: Lake configuration.
        market_cap_view: Raw ``daily_market_cap`` view name.
        filing_receipt_view: Raw receipt view name, source of the 권리락 dates.
        force: Rebuild even when the mart already exists.

    Returns:
        The registered view name.
    """
    register_mcap_distortion_view(
        con,
        receipt_view=filing_receipt_view,
        market_cap_view=market_cap_view,
    )
    materialize(
        con,
        config,
        MARKET_CAP_TABLE,
        build_market_cap_sql(market_cap_view),
        force=force,
    )
    return register_mart_view(con, config, MARKET_CAP_TABLE)
