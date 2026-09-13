# ruff: noqa: E501
"""feat_macro_exposure — per-ticker rolling macro exposure betas (Stage 1a).

Design: ``docs/dev/20260829_macro_features/01_design/02_stage1a_exposure_beta_families.md``.

Grain ``(trade_date, ticker, market)``, valid sessions only — the same three
key columns ``feat_price`` emits and nothing else shared, so the Phase B panel
can LEFT JOIN this in without shadowing a column it already has (the panel's
``valid_session_idx`` comes from ``dim_price_quality_daily``; a second one here
would be silently renamed rather than rejected). Six primary columns, each a
rolling slope of one stock's return on one macro factor:

    macro_beta_usdkrw / _wti / _kr10y / _sp500_lag / _vix   +   px_market_beta

Two things decide the shape of this mart.

**Residual, not raw.** The scan's label is an excess return over the
per-(date, market) equal-weighted mean — market-*average* neutral, not
market-*beta* neutral. A raw-return factor beta satisfies
``Cov(r_i, f) = beta_i,m * Cov(r_m, f) + Cov(e_i, f)``, so for a factor that
co-moves with the market (the won especially) it largely reproduces
``px_market_beta``. The primary columns therefore regress the market-model
residual; the raw-return version is kept beside it as ``macro_rawbeta_*``, and
the difference between the two is the diagnostic (§4).

**Two factor timings, so two windows.** Domestic series reach the fact one
session late (``next_krx_session``: the row on session t carries the t-1
observation), while foreign series are the previous New York close, already
known on the morning of session t (``same_krx_session_morning``). So a
domestic factor's same-session change ``g_tau`` is only complete on session
tau+1, and its beta on session t may use pairs up to tau <= t-1 — window B,
ending one session back. Foreign factors pair directly with the same session's
return — window A, through the current session. Neither window can see a value
that was not published yet. See §2.2/§2.3.

A factor is NULL for a session whose source observation did not update
(``asof_available_date`` unchanged): a stale carry-forward is not a zero
change, and ``global_sp500_ret_1d`` — used as-is — would otherwise contribute
the same return twice. ``REGR_*`` drops NULL pairs on its own.
"""

from __future__ import annotations

import duckdb

from modeler.etl.config import LakeConfig
from modeler.etl.mart import materialize, register_mart_view
from modeler.etl.trading_panel import build_market_model_sql, build_valid_session_sql

MACRO_EXPOSURE_TABLE = "feat_macro_exposure"
COMMON_FEATURE_FACT_VIEW = "common_feature_daily_fact"

# The five catalog feature codes this mart reads. `commodity_wti_spot_level`
# was added for it (PR-1a-1); the other four already existed.
REQUIRED_FEATURE_CODES: tuple[str, ...] = (
    "fx_usdkrw_level",
    "rate_kr_gov10y_level",
    "commodity_wti_spot_level",
    "global_sp500_ret_1d",
    "global_vix_level",
)

# 252-session windows, and at least half of that window's pairs actually
# usable. This is this design's choice, not price.py's: KRX and NY holidays
# both punch NULLs into a factor series, so demanding a completely full window
# (what `beta_252` does) would empty the column. The semibeta sees roughly half
# the sessions by construction, hence its own lower floor. §2.3.
BETA_WINDOW_SESSIONS = 252
BETA_MIN_PAIRS = 126
SEMIBETA_MIN_PAIRS = 60

# Primary + secondary columns, in the order they appear in the output. Every
# one of them also gets a `_lag1` twin (the preregistered variant axis).
_PRIMARY_COLUMNS: tuple[str, ...] = (
    "macro_beta_usdkrw",
    "macro_beta_wti",
    "macro_beta_kr10y",
    "macro_beta_sp500_lag",
    "macro_beta_vix",
    "px_market_beta",
)
_SECONDARY_COLUMNS: tuple[str, ...] = (
    "macro_rawbeta_usdkrw",
    "macro_rawbeta_wti",
    "macro_rawbeta_kr10y",
    "macro_rawbeta_sp500_lag",
    "macro_rawbeta_vix",
    "macro_semibeta_usdkrw_up",
)
FEATURE_COLUMNS: tuple[str, ...] = _PRIMARY_COLUMNS + _SECONDARY_COLUMNS

# REGR_COUNT companions — how many usable pairs each beta actually had. A
# foreign-holiday run or a thin factor history shows up here rather than as a
# silently confident slope. §4.
COUNT_COLUMNS: tuple[str, ...] = (
    "macro_beta_n_usdkrw",
    "macro_beta_n_wti",
    "macro_beta_n_kr10y",
    "macro_beta_n_sp500_lag",
    "macro_beta_n_vix",
    "px_market_beta_n",
    "macro_semibeta_n_usdkrw_up",
)

# (column, y, x, window) — window "A" pairs with the current session, "B" ends
# one session back. Kept as data so the SQL builder and the tests read the same
# table rather than two copies of the same list.
_BETA_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("macro_beta_usdkrw", "resid_ret", "g_usdkrw", "B"),
    ("macro_beta_wti", "resid_ret", "f_wti", "A"),
    ("macro_beta_kr10y", "resid_ret", "g_kr10y", "B"),
    ("macro_beta_sp500_lag", "resid_ret", "f_sp500_lag", "A"),
    ("macro_beta_vix", "resid_ret", "f_vix", "A"),
    ("px_market_beta", "log_ret", "market_ret", "A"),
    ("macro_rawbeta_usdkrw", "log_ret", "g_usdkrw", "B"),
    ("macro_rawbeta_wti", "log_ret", "f_wti", "A"),
    ("macro_rawbeta_kr10y", "log_ret", "g_kr10y", "B"),
    ("macro_rawbeta_sp500_lag", "log_ret", "f_sp500_lag", "A"),
    ("macro_rawbeta_vix", "log_ret", "f_vix", "A"),
)

# The `*_n` column each factor reports its pair count through. There is one
# per factor and it is the *primary* (residual) regression's count, which is
# the coverage number the family is judged on. The `macro_rawbeta_*` twin sees
# more pairs over the same window — `resid_ret` needs 252 prior sessions before
# it exists at all, `log_ret` does not — so this count is a floor for it, not
# its own value.
_COUNT_OF: dict[str, str] = {
    "macro_beta_usdkrw": "macro_beta_n_usdkrw",
    "macro_beta_wti": "macro_beta_n_wti",
    "macro_beta_kr10y": "macro_beta_n_kr10y",
    "macro_beta_sp500_lag": "macro_beta_n_sp500_lag",
    "macro_beta_vix": "macro_beta_n_vix",
    "px_market_beta": "px_market_beta_n",
}


def _factor_pivot_sql(fact_view: str) -> str:
    """One row per ``feature_date`` with the five factor levels and their asofs.

    ``common_feature_daily_fact`` is already as-of resolved — one row per
    (feature_date, feature_code) whose ``asof_available_date <= feature_date``
    — so this is a pivot, not a PIT filter.
    """
    codes = ", ".join(f"'{code}'" for code in REQUIRED_FEATURE_CODES)
    pairs = [
        ("fx_usdkrw_level", "fx_level", "fx_asof"),
        ("rate_kr_gov10y_level", "kr10y_level", "kr10y_asof"),
        ("commodity_wti_spot_level", "wti_level", "wti_asof"),
        ("global_sp500_ret_1d", "sp500_ret", "sp500_asof"),
        ("global_vix_level", "vix_level", "vix_asof"),
    ]
    columns = ",\n                   ".join(
        f"MAX(CASE WHEN feature_code = '{code}' THEN CAST(value_numeric AS DOUBLE) END) AS {value_alias},\n"
        f"                   MAX(CASE WHEN feature_code = '{code}' THEN asof_available_date END) AS {asof_alias}"
        for code, value_alias, asof_alias in pairs
    )
    return f"""
            SELECT feature_date,
                   {columns}
            FROM {fact_view}
            WHERE feature_code IN ({codes})
            GROUP BY feature_date
    """


def _factor_series_sql() -> str:
    """The six factor changes, on the KRX session grid.

    The grid matters. The fact's own ``feature_date`` axis is every weekday for
    2014-2023 (``src/collector/kr/infra/calendar/data/holidays_krx.csv`` only covers 2024-2026), so a KRX
    holiday sits in it carrying the previous session's value. Differencing on
    that axis would manufacture a zero-change session; differencing on the
    panel's own sessions — which is what this does — cannot.

    Foreign factors difference *into* session t (t-1 -> t). Domestic factors
    are placed on session tau as the change that completes on tau+1, which is
    what makes window B the correct one to read them with.

    A log change needs both ends strictly positive. That is not hypothetical
    for oil: WTI spot settled at -36.98 on 2020-04-20, which reaches the fact
    on the 2020-04-21 KRX session, so ``f_wti`` is undefined on that session
    and the next. Those two sessions are NULL — the same treatment a holiday
    gets, and the same reason: there is no factor value, not a zero one. The
    rate factor is a difference in percentage points and needs no such guard.
    """
    return """
            SELECT
                trade_date,
                CASE WHEN vix_asof IS DISTINCT FROM LAG(vix_asof) OVER w
                      AND vix_level > 0 AND LAG(vix_level) OVER w > 0
                     THEN LN(vix_level / LAG(vix_level) OVER w) END AS f_vix,
                CASE WHEN wti_asof IS DISTINCT FROM LAG(wti_asof) OVER w
                      AND wti_level > 0 AND LAG(wti_level) OVER w > 0
                     THEN LN(wti_level / LAG(wti_level) OVER w) END AS f_wti,
                CASE WHEN sp500_asof IS DISTINCT FROM LAG(sp500_asof) OVER w
                     THEN sp500_ret END AS f_sp500_lag,
                CASE WHEN LEAD(fx_asof) OVER w IS DISTINCT FROM fx_asof
                      AND fx_level > 0 AND LEAD(fx_level) OVER w > 0
                     THEN LN(LEAD(fx_level) OVER w / fx_level) END AS g_usdkrw,
                CASE WHEN LEAD(kr10y_asof) OVER w IS DISTINCT FROM kr10y_asof
                     THEN LEAD(kr10y_level) OVER w - kr10y_level END AS g_kr10y
            FROM session_factors
            WINDOW w AS (ORDER BY trade_date)
    """


def build_macro_factor_sql(
    price_view: str = "daily_ohlcv",
    *,
    fact_view: str = COMMON_FEATURE_FACT_VIEW,
) -> str:
    """Just the factor series, one row per KRX session.

    Exposed separately from the betas because it is the half worth inspecting
    on its own: which sessions a factor is NULL on, and why, is the whole
    content of §2.2. The beta SQL below builds the same CTEs.
    """
    return f"""
        WITH valid AS (
            {build_valid_session_sql(price_view)}
        ),
        factor_pivot AS (
            {_factor_pivot_sql(fact_view)}
        ),
        session_factors AS (
            SELECT s.trade_date, f.* EXCLUDE (feature_date)
            FROM (SELECT DISTINCT trade_date FROM valid) s
            LEFT JOIN factor_pivot f ON f.feature_date = s.trade_date
        )
        {_factor_series_sql()}
    """


def _beta_expression(column: str, y: str, x: str, window: str) -> str:
    count = f"REGR_COUNT({y}, {x}) OVER w{window}"
    return (
        f"CASE WHEN {count} >= {BETA_MIN_PAIRS}\n"
        f"                     THEN REGR_SLOPE({y}, {x}) OVER w{window} END AS {column}"
    )


def build_macro_exposure_sql(
    price_view: str = "daily_ohlcv",
    *,
    fact_view: str = COMMON_FEATURE_FACT_VIEW,
) -> str:
    """SQL producing ``feat_macro_exposure``.

    Both ``price_view`` and ``fact_view`` must already be registered on the
    connection; the fact must be the *same* one the readiness gate saw, which
    is why the Phase B run binds it from the snapshot's persisted derived mart
    rather than recomputing it (§3).
    """
    betas = ",\n                ".join(
        _beta_expression(column, y, x, window) for column, y, x, window in _BETA_SPECS
    )
    counts = ",\n                ".join(
        f"REGR_COUNT({y}, {x}) OVER w{window} AS {_COUNT_OF[column]}"
        for column, y, x, window in _BETA_SPECS
        if column in _COUNT_OF
    )
    lag1 = ",\n                ".join(
        f"LAG({column}) OVER (PARTITION BY ticker, market ORDER BY trade_date) AS {column}_lag1"
        for column in FEATURE_COLUMNS
    )
    outputs = ",\n            ".join(
        [*FEATURE_COLUMNS, *(f"{column}_lag1" for column in FEATURE_COLUMNS), *COUNT_COLUMNS]
    )
    # The semibeta keeps only the won-weakening half of the pairs (Chu 2022's
    # downside co-movement, read on the Korean cross-section). Masking both
    # sides of the regression is what restricts it to those rows.
    up_y = "CASE WHEN g_usdkrw > 0 THEN resid_ret END"
    up_x = "CASE WHEN g_usdkrw > 0 THEN g_usdkrw END"
    return f"""
        WITH valid AS (
            {build_valid_session_sql(price_view)}
        ),
        {build_market_model_sql("valid")},
        factor_pivot AS (
            {_factor_pivot_sql(fact_view)}
        ),
        session_factors AS (
            SELECT s.trade_date, f.* EXCLUDE (feature_date)
            FROM (SELECT DISTINCT trade_date FROM valid) s
            LEFT JOIN factor_pivot f ON f.feature_date = s.trade_date
        ),
        factors AS (
            {_factor_series_sql()}
        ),
        panel AS (
            SELECT r.*, k.f_vix, k.f_wti, k.f_sp500_lag, k.g_usdkrw, k.g_kr10y
            FROM residuals r
            LEFT JOIN factors k USING (trade_date)
        ),
        betas AS (
            SELECT
                trade_date, ticker, market,
                {betas},
                CASE WHEN REGR_COUNT({up_y}, {up_x}) OVER wB >= {SEMIBETA_MIN_PAIRS}
                     THEN REGR_SLOPE({up_y}, {up_x}) OVER wB END AS macro_semibeta_usdkrw_up,
                {counts},
                REGR_COUNT({up_y}, {up_x}) OVER wB AS macro_semibeta_n_usdkrw_up
            FROM panel
            WINDOW
                wA AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN {BETA_WINDOW_SESSIONS - 1} PRECEDING AND CURRENT ROW),
                wB AS (PARTITION BY ticker, market ORDER BY trade_date ROWS BETWEEN {BETA_WINDOW_SESSIONS} PRECEDING AND 1 PRECEDING)
        ),
        variants AS (
            SELECT *,
                {lag1}
            FROM betas
        )
        SELECT
            trade_date, ticker, market,
            {outputs}
        FROM variants
    """


def materialize_macro_exposure(
    con: duckdb.DuckDBPyConnection,
    config: LakeConfig,
    *,
    price_view: str = "daily_ohlcv",
    fact_view: str = COMMON_FEATURE_FACT_VIEW,
    force: bool = False,
) -> str:
    """Build + register the ``feat_macro_exposure`` mart view. Returns its name.

    Requires ``price_view`` and ``fact_view`` registered on ``con``. Raises
    ``duckdb.Error`` when the fact view is absent — which is how the Phase B
    orchestrator leaves the six macro families ``blocked_exploratory`` for a
    snapshot whose ``common_feature_daily_fact`` was never built.
    """
    materialize(
        con,
        config,
        MACRO_EXPOSURE_TABLE,
        build_macro_exposure_sql(price_view, fact_view=fact_view),
        force=force,
    )
    return register_mart_view(con, config, MACRO_EXPOSURE_TABLE)
