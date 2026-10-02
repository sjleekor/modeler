"""Shared valid-session panel primitives used by features, labels and quality."""

from __future__ import annotations


def build_valid_session_sql(price_view: str = "daily_ohlcv") -> str:
    """Return one row per real ticker session with a stable valid-session index.

    Halt rows remain available in the source view for audit joins, but never
    participate in lag/lead windows. This is the only place where the session
    index and adjacent simple/log returns are defined.
    """
    return f"""
        SELECT
            trade_date, ticker, market,
            ROW_NUMBER() OVER (
                PARTITION BY ticker, market ORDER BY trade_date
            ) AS valid_session_idx,
            CAST(close AS DOUBLE) AS close_d,
            CAST(volume AS DOUBLE) AS volume_d,
            CAST(close AS DOUBLE) * CAST(volume AS DOUBLE) AS turnover,
            LAG(CAST(close AS DOUBLE)) OVER w AS prev_close_d,
            CAST(close AS DOUBLE) / NULLIF(LAG(CAST(close AS DOUBLE)) OVER w, 0) - 1
                AS simple_ret,
            LN(CAST(close AS DOUBLE) / NULLIF(LAG(CAST(close AS DOUBLE)) OVER w, 0))
                AS log_ret
        FROM {price_view}
        WHERE NOT (open = 0 AND high = 0 AND low = 0)
        WINDOW w AS (PARTITION BY ticker, market ORDER BY trade_date)
    """


def build_full_panel_sql(price_view: str = "daily_ohlcv") -> str:
    """Return all OHLCV keys with valid-session fields left joined."""
    valid = build_valid_session_sql(price_view)
    return f"""
        WITH valid AS ({valid})
        SELECT
            p.trade_date, p.ticker, p.market,
            (p.open = 0 AND p.high = 0 AND p.low = 0) AS is_halted,
            (CAST(p.volume AS DOUBLE) = 0) AS volume_zero,
            v.valid_session_idx, v.close_d, v.volume_d, v.turnover,
            v.prev_close_d, v.simple_ret, v.log_ret
        FROM {price_view} p
        LEFT JOIN valid v USING (trade_date, ticker, market)
    """


MARKET_SEMANTICS_V1 = "v1"
MARKET_SEMANTICS_V2 = "v2"
MARKET_SEMANTICS_VERSIONS = (MARKET_SEMANTICS_V1, MARKET_SEMANTICS_V2)


def build_market_model_sql(source_cte: str, semantics: str = MARKET_SEMANTICS_V1) -> str:
    """The ``market``/``modeled``/``residuals`` CTE trio, over ``source_cte``.

    One market-model definition, shared by every mart that needs a residual
    return: equal-weighted per-(date, market) market return, a 252-session
    market model fitted on the *prior* sessions only, and the residual it
    leaves — which is only produced once the window is completely full
    (``model_n_252 >= 252``), never from a partial fit.

    ``feat_price`` and ``feat_macro_exposure`` must agree on this exactly:
    ``px_idio_vol_60d``/``px_resid_mom_12_1`` and the ``macro_beta_*`` family
    are all statements about the same ``resid_ret``, and A0's input lineage is
    keyed on ``feat_price``'s SQL text (``mart._sql_hash``), so this returns
    that text verbatim rather than a tidied-up equivalent.

    ``semantics="v2"`` changes only how ``market_ret`` is summed. ``v1`` averages with a
    window aggregate whose input order depends on how the engine happens to partition the
    rows, so the float sum (and everything fitted on it: beta, alpha, ``resid_ret``) can
    move in the last bits from run to run. ``v2`` averages inside a ``GROUP BY`` with
    ``ORDER BY ticker``, a total order within (date, market) because ``(date, ticker,
    market)`` is the primary key, so the sum is taken in one fixed order. The value is the
    same mean up to rounding (it differs from ``v1`` at the 1e-16 relative level), which is
    why this is a semantics version and not a plan.
    """
    if semantics not in MARKET_SEMANTICS_VERSIONS:
        raise ValueError(
            f"unknown market model semantics {semantics!r}; expected {MARKET_SEMANTICS_VERSIONS}"
        )
    if semantics == MARKET_SEMANTICS_V2:
        market_cte = f"""market AS (
            SELECT s.*, m.market_ret
            FROM {source_cte} s
            LEFT JOIN (
                SELECT trade_date, market, AVG(log_ret ORDER BY ticker) AS market_ret
                FROM {source_cte}
                GROUP BY trade_date, market
            ) m USING (trade_date, market)
        ),"""
    else:
        market_cte = f"""market AS (
            SELECT *, AVG(log_ret) OVER (PARTITION BY trade_date, market) AS market_ret
            FROM {source_cte}
        ),"""
    return f"""{market_cte}
        modeled AS (
            SELECT *,
                REGR_SLOPE(log_ret, market_ret) OVER (
                    PARTITION BY ticker, market ORDER BY trade_date
                    ROWS BETWEEN 252 PRECEDING AND 1 PRECEDING
                ) AS beta_252,
                REGR_INTERCEPT(log_ret, market_ret) OVER (
                    PARTITION BY ticker, market ORDER BY trade_date
                    ROWS BETWEEN 252 PRECEDING AND 1 PRECEDING
                ) AS alpha_252,
                COUNT(CASE WHEN log_ret IS NOT NULL AND market_ret IS NOT NULL THEN 1 END)
                    OVER (
                        PARTITION BY ticker, market ORDER BY trade_date
                        ROWS BETWEEN 252 PRECEDING AND 1 PRECEDING
                    ) AS model_n_252
            FROM market
        ),
        residuals AS (
            SELECT *,
                   CASE WHEN model_n_252 >= 252
                        THEN log_ret - (alpha_252 + beta_252 * market_ret)
                   END AS resid_ret
            FROM modeled
        )"""
