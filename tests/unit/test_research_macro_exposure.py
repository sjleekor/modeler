"""Unit tests for feat_macro_exposure (Stage 1a).

Design: ``docs/dev/20260829_macro_features/01_design/02_stage1a_exposure_beta_families.md``
§2.2 (factor timing + NULL rule), §2.3 (windows, minimum pairs), §5 (this list).

Every fixture is synthetic. Closes are DOUBLE rather than the lake's BIGINT so a
planted relationship survives exactly — ``build_valid_session_sql`` casts either
to DOUBLE, and rounding is not what these tests are about.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import duckdb
import pytest
from modeler.etl.features.macro_exposure import (
    BETA_MIN_PAIRS,
    COUNT_COLUMNS,
    FEATURE_COLUMNS,
    SEMIBETA_MIN_PAIRS,
    build_macro_exposure_sql,
    build_macro_factor_sql,
)

_FACT_CODES = (
    "fx_usdkrw_level",
    "rate_kr_gov10y_level",
    "commodity_wti_spot_level",
    "global_sp500_ret_1d",
    "global_vix_level",
)


def _weekdays(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE daily_ohlcv (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume BIGINT)"
    )
    con.execute(
        "CREATE TABLE common_feature_daily_fact (feature_date DATE, feature_code VARCHAR, "
        "value_numeric DOUBLE, asof_available_date DATE)"
    )
    return con


def _add_prices(
    con: duckdb.DuckDBPyConnection,
    days: list[date],
    closes: list[float],
    *,
    ticker: str = "A",
    market: str = "KOSPI",
) -> None:
    con.executemany(
        "INSERT INTO daily_ohlcv VALUES (?,?,?,?,?,?,?,?)",
        [(d, ticker, market, 1.0, 1.0, 1.0, c, 100) for d, c in zip(days, closes, strict=True)],
    )


def _add_fact(
    con: duckdb.DuckDBPyConnection,
    days: list[date],
    values_by_code: dict[str, list[float]],
    *,
    asof_by_code: dict[str, list[date]] | None = None,
) -> None:
    rows = []
    for code, values in values_by_code.items():
        asofs = (asof_by_code or {}).get(code, days)
        for d, value, asof in zip(days, values, asofs, strict=True):
            rows.append((d, code, value, asof))
    con.executemany("INSERT INTO common_feature_daily_fact VALUES (?,?,?,?)", rows)


def _flat_fact(days: list[date], **overrides: list[float]) -> dict[str, list[float]]:
    """A fact where every factor is constant unless overridden.

    A constant factor has zero variance, so its beta is NaN and never disturbs
    the one factor a test is actually looking at.
    """
    defaults = {
        "fx_usdkrw_level": 1300.0,
        "rate_kr_gov10y_level": 3.0,
        "commodity_wti_spot_level": 70.0,
        "global_sp500_ret_1d": 0.0,
        "global_vix_level": 20.0,
    }
    return {code: overrides.get(code, [defaults[code]] * len(days)) for code in _FACT_CODES}


def _factors(con: duckdb.DuckDBPyConnection) -> dict[date, dict[str, float | None]]:
    rows = con.execute(build_macro_factor_sql()).fetchall()
    names = [d[0] for d in con.execute(f"DESCRIBE {build_macro_factor_sql()}").fetchall()]
    return {r[0]: dict(zip(names[1:], r[1:], strict=True)) for r in rows}


# --- §2.2 factor definitions -------------------------------------------------


def test_foreign_factors_difference_into_the_current_session() -> None:
    con = _con()
    days = _weekdays(date(2024, 1, 1), 4)
    _add_prices(con, days, [100.0, 101.0, 102.0, 103.0])
    _add_fact(
        con,
        days,
        _flat_fact(
            days,
            global_vix_level=[20.0, 22.0, 22.0, 24.0],
            commodity_wti_spot_level=[70.0, 71.0, 70.0, 70.0],
            global_sp500_ret_1d=[0.01, 0.02, 0.03, 0.04],
        ),
    )
    factors = _factors(con)

    assert factors[days[0]]["f_vix"] is None  # no prior session to difference from
    assert factors[days[1]]["f_vix"] == pytest.approx(math.log(22.0 / 20.0))
    assert factors[days[2]]["f_vix"] == pytest.approx(0.0)  # updated, genuinely unchanged
    assert factors[days[1]]["f_wti"] == pytest.approx(math.log(71.0 / 70.0))
    # global_sp500_ret_1d is already a NY t-1 return; it is used as it stands.
    assert factors[days[2]]["f_sp500_lag"] == pytest.approx(0.03)


def test_domestic_factors_sit_on_the_session_whose_change_completes_next() -> None:
    """§2.2: ``g_tau = ln(fx_{tau+1} / fx_tau)`` is written onto row tau.

    The numerator only reaches the fact on tau+1, which is exactly why the
    domestic betas read window B (ending one session back) rather than A.
    """
    con = _con()
    days = _weekdays(date(2024, 1, 1), 4)
    _add_prices(con, days, [100.0, 101.0, 102.0, 103.0])
    _add_fact(
        con,
        days,
        _flat_fact(
            days,
            fx_usdkrw_level=[1300.0, 1310.0, 1320.0, 1320.0],
            rate_kr_gov10y_level=[3.0, 3.1, 3.05, 3.05],
        ),
    )
    factors = _factors(con)

    assert factors[days[0]]["g_usdkrw"] == pytest.approx(math.log(1310.0 / 1300.0))
    assert factors[days[1]]["g_kr10y"] == pytest.approx(3.05 - 3.1)
    assert factors[days[3]]["g_usdkrw"] is None  # nothing after the last session


def test_a_factor_is_null_when_its_source_observation_did_not_update() -> None:
    """§2.2: a stale carry-forward is not a zero change.

    ``global_sp500_ret_1d`` is the sharp case — used as-is, an unchanged asof
    would feed the identical return into the regression twice.
    """
    con = _con()
    days = _weekdays(date(2024, 1, 1), 4)
    _add_prices(con, days, [100.0, 101.0, 102.0, 103.0])
    stalled = [days[0], days[1], days[1], days[3]]  # session 3 repeats session 2's asof
    _add_fact(
        con,
        days,
        _flat_fact(
            days,
            global_vix_level=[20.0, 22.0, 22.0, 24.0],
            global_sp500_ret_1d=[0.01, 0.02, 0.02, 0.03],
            fx_usdkrw_level=[1300.0, 1310.0, 1310.0, 1320.0],
        ),
        asof_by_code={
            "global_vix_level": stalled,
            "global_sp500_ret_1d": stalled,
            "fx_usdkrw_level": stalled,
        },
    )
    factors = _factors(con)

    assert factors[days[2]]["f_vix"] is None
    assert factors[days[2]]["f_sp500_lag"] is None
    # Domestic: the pair (tau, tau+1) is stale looking forward from session 2.
    assert factors[days[1]]["g_usdkrw"] is None
    assert factors[days[2]]["g_usdkrw"] == pytest.approx(math.log(1320.0 / 1310.0))


def test_a_log_change_through_a_non_positive_level_is_null() -> None:
    """WTI spot settled at -36.98 on 2020-04-20. A log return has no value
    there, so the session it lands on and the next are NULL rather than an
    invented number — and the build must not raise."""
    con = _con()
    days = _weekdays(date(2024, 1, 1), 4)
    _add_prices(con, days, [100.0, 101.0, 102.0, 103.0])
    _add_fact(
        con,
        days,
        _flat_fact(days, commodity_wti_spot_level=[70.0, -36.98, 20.0, 21.0]),
    )
    factors = _factors(con)

    assert factors[days[1]]["f_wti"] is None
    assert factors[days[2]]["f_wti"] is None
    assert factors[days[3]]["f_wti"] == pytest.approx(math.log(21.0 / 20.0))


def test_factors_are_differenced_on_krx_sessions_not_the_facts_weekday_grid() -> None:
    """§2.1/§2.2: the fact's own axis is every weekday for 2014-2023, with KRX
    holidays carrying the previous value. Differencing there would invent a
    zero-change session; the panel's session grid is what this mart uses."""
    con = _con()
    weekdays = _weekdays(date(2024, 1, 1), 4)
    sessions = [weekdays[0], weekdays[1], weekdays[3]]  # weekdays[2] is a KRX holiday
    _add_prices(con, sessions, [100.0, 101.0, 102.0])
    _add_fact(
        con,
        weekdays,
        _flat_fact(weekdays, global_vix_level=[20.0, 22.0, 22.0, 24.0]),
    )
    factors = _factors(con)

    assert set(factors) == set(sessions)
    # Straight from session 2's 22.0 to session 4's 24.0 — the holiday's copied
    # 22.0 never becomes a zero-change row.
    assert factors[weekdays[3]]["f_vix"] == pytest.approx(math.log(24.0 / 22.0))


# --- §2.3 betas --------------------------------------------------------------


def _linear_scenario(slope: float, *, n: int = 700) -> tuple[duckdb.DuckDBPyConnection, list[date]]:
    """Ticker A's log return is exactly ``slope`` times the oil factor.

    Ticker Z rides along on an unrelated path purely so the market return is an
    average of two names rather than a copy of A's own return — with a single
    ticker the market model fits perfectly and ``resid_ret`` collapses to zero.
    """
    days = _weekdays(date(2020, 1, 1), n)
    wti = [70.0 * math.exp(0.01 * math.sin(i * 0.7)) for i in range(n)]
    con = _con()
    closes = [100.0]
    for i in range(1, n):
        closes.append(closes[-1] * math.exp(slope * math.log(wti[i] / wti[i - 1])))
    _add_prices(con, days, closes, ticker="A")
    _add_prices(
        con,
        days,
        [50.0 * math.exp(0.003 * math.cos(i * 0.41)) for i in range(n)],
        ticker="Z",
    )
    _add_fact(con, days, _flat_fact(days, commodity_wti_spot_level=wti))
    return con, days


def test_raw_beta_recovers_a_planted_linear_relationship() -> None:
    con, _days = _linear_scenario(2.5)
    rows = con.execute(
        f"SELECT macro_rawbeta_wti FROM ({build_macro_exposure_sql()}) "
        "WHERE ticker = 'A' AND macro_rawbeta_wti IS NOT NULL"
    ).fetchall()
    assert rows
    for (beta,) in rows:
        assert beta == pytest.approx(2.5, abs=1e-9)


def test_a_beta_needs_the_preregistered_minimum_of_usable_pairs() -> None:
    """§2.3: 252-session window, at least 126 usable pairs, else NULL."""
    con, _days = _linear_scenario(2.5)
    rows = con.execute(
        f"SELECT macro_beta_wti, macro_beta_n_wti FROM ({build_macro_exposure_sql()}) "
        "WHERE ticker = 'A' ORDER BY trade_date"
    ).fetchall()
    for beta, n in rows:
        assert (beta is not None) == (n >= BETA_MIN_PAIRS)
    assert any(n >= BETA_MIN_PAIRS for _b, n in rows)
    assert any(n < BETA_MIN_PAIRS for _b, n in rows)


def test_the_pair_count_column_reports_the_primary_residual_regression() -> None:
    """One ``*_n`` per factor, and it counts the residual regression's pairs.

    ``resid_ret`` needs a full 252-session market model before it exists, so
    the raw-return twin clears the 126 floor first — the count is a floor for
    it, never its own value.
    """
    con, _days = _linear_scenario(2.5)
    rows = con.execute(
        f"SELECT macro_beta_wti, macro_rawbeta_wti, macro_beta_n_wti "
        f"FROM ({build_macro_exposure_sql()}) WHERE ticker = 'A' ORDER BY trade_date"
    ).fetchall()
    assert any(beta is None and raw is not None for beta, raw, _n in rows)
    assert all(beta is None for beta, _raw, n in rows if n < BETA_MIN_PAIRS)


def test_a_domestic_beta_never_reads_a_factor_value_published_after_the_session() -> None:
    """§2.2/§2.3 look-ahead check.

    ``g_tau`` on row tau is built from the fx level of tau+1, so window B
    (ending at t-1) is what keeps session t's beta inside what session t knows.
    Changing every fx level strictly after session t must therefore leave
    session t's beta untouched.
    """
    days = _weekdays(date(2020, 1, 1), 700)
    fx = [1300.0 * math.exp(0.002 * math.sin(i * 0.6)) for i in range(len(days))]
    cut = 600

    def _beta_at(levels: list[float]) -> float:
        con = _con()
        closes = [100.0 * math.exp(0.001 * math.cos(i * 0.3)) for i in range(len(days))]
        _add_prices(con, days, closes)
        _add_fact(con, days, _flat_fact(days, fx_usdkrw_level=levels))
        (value,) = con.execute(
            f"SELECT macro_rawbeta_usdkrw FROM ({build_macro_exposure_sql()}) "
            f"WHERE trade_date = DATE '{days[cut].isoformat()}'"
        ).fetchone()
        return value

    tampered = [*fx[: cut + 1], *(level * 1.05 for level in fx[cut + 1 :])]
    assert _beta_at(fx) is not None
    assert _beta_at(tampered) == pytest.approx(_beta_at(fx), abs=1e-12)


def test_the_semibeta_uses_only_the_won_weakening_half_of_the_pairs() -> None:
    """§2.3: both sides of the regression are masked to ``g_usdkrw > 0``, so its
    pair count is the count of those rows — not of every row in the window."""
    days = _weekdays(date(2020, 1, 1), 700)
    fx = [1300.0 * math.exp(0.002 * math.sin(i * 0.6)) for i in range(len(days))]
    con = _con()
    _add_prices(con, days, [100.0 * math.exp(0.001 * math.cos(i * 0.3)) for i in range(len(days))])
    _add_fact(con, days, _flat_fact(days, fx_usdkrw_level=fx))
    rows = con.execute(
        f"SELECT macro_semibeta_usdkrw_up, macro_semibeta_n_usdkrw_up, macro_beta_n_usdkrw "
        f"FROM ({build_macro_exposure_sql()}) ORDER BY trade_date"
    ).fetchall()

    for beta, n_up, n_all in rows:
        assert n_up <= n_all
        assert (beta is not None) == (n_up >= SEMIBETA_MIN_PAIRS)
    # A sine factor is up roughly half the time, so the semibeta must clear its
    # own lower floor well before the full beta clears 126.
    assert max(n_up for _b, n_up, _n in rows) < max(n_all for _b, _n, n_all in rows)


def test_lag1_columns_are_the_previous_valid_session_value() -> None:
    con, days = _linear_scenario(2.5, n=400)
    rows = con.execute(
        f"SELECT trade_date, macro_rawbeta_wti, macro_rawbeta_wti_lag1 "
        f"FROM ({build_macro_exposure_sql()}) WHERE ticker = 'A' ORDER BY trade_date"
    ).fetchall()
    previous = None
    for _d, native, lag1 in rows:
        if previous is None:
            assert lag1 is None
        else:
            assert lag1 == previous
        previous = native


def test_px_market_beta_is_produced_without_any_macro_factor() -> None:
    """§2.6: the market factor is internal to the mart, so ``px_market_beta``
    starts at the panel's own history rather than the 2014-06 common-feature
    start every other family waits for."""
    days = _weekdays(date(2020, 1, 1), 400)
    con = _con()
    for i, (ticker, phase) in enumerate((("A", 0.0), ("B", 1.1), ("C", 2.2))):
        _add_prices(
            con,
            days,
            [
                100.0 * math.exp(0.004 * (i + 1) * math.sin(0.3 * j + phase))
                for j in range(len(days))
            ],
            ticker=ticker,
        )
    con.execute("DELETE FROM common_feature_daily_fact")
    rows = con.execute(
        f"SELECT count(px_market_beta), count(macro_beta_wti) FROM ({build_macro_exposure_sql()})"
    ).fetchone()
    assert rows[0] > 0
    assert rows[1] == 0


# --- shape / wiring ----------------------------------------------------------


def test_the_mart_emits_every_preregistered_column() -> None:
    con, _days = _linear_scenario(1.0, n=300)
    names = [r[0] for r in con.execute(f"DESCRIBE {build_macro_exposure_sql()}").fetchall()]
    assert names[:3] == ["trade_date", "ticker", "market"]
    assert set(names) == {
        "trade_date",
        "ticker",
        "market",
        *FEATURE_COLUMNS,
        *(f"{column}_lag1" for column in FEATURE_COLUMNS),
        *COUNT_COLUMNS,
    }


def test_the_grain_is_one_row_per_valid_session() -> None:
    days = _weekdays(date(2024, 1, 1), 6)
    con = _con()
    _add_prices(con, days, [100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    # A halt (OHL all zero) is not a valid session and must not appear.
    con.execute(
        f"UPDATE daily_ohlcv SET open = 0, high = 0, low = 0 "
        f"WHERE trade_date = DATE '{days[3].isoformat()}'"
    )
    _add_fact(con, days, _flat_fact(days))
    rows = con.execute(
        f"SELECT count(*), count(DISTINCT trade_date) FROM ({build_macro_exposure_sql()})"
    ).fetchone()
    assert rows == (5, 5)


def test_a_missing_common_feature_fact_view_fails_loudly() -> None:
    """The Phase B orchestrator turns this error into ``blocked_exploratory``
    for the six macro families — it must be an error, not an empty mart."""
    con = _con()
    con.execute("DROP TABLE common_feature_daily_fact")
    with pytest.raises(duckdb.Error, match="common_feature_daily_fact"):
        con.execute(f"SELECT * FROM ({build_macro_exposure_sql()})").fetchall()


def test_a_factor_code_missing_from_the_fact_nulls_only_its_own_beta() -> None:
    """A snapshot whose ``common_feature_daily_fact`` predates a catalog entry
    (``commodity_wti_spot_level`` was added in PR-1a-1) still builds: the oil
    beta is empty and its pair count is 0, every other factor is unaffected.
    Visible in the coverage diagnostics rather than a crash or a silent zero."""
    days = _weekdays(date(2020, 1, 1), 500)
    con = _con()
    _add_prices(con, days, [100.0 * math.exp(0.004 * math.sin(0.3 * i)) for i in range(len(days))])
    _add_prices(
        con,
        days,
        [50.0 * math.exp(0.003 * math.cos(0.41 * i)) for i in range(len(days))],
        ticker="Z",
    )
    fact = _flat_fact(
        days,
        global_vix_level=[20.0 * math.exp(0.02 * math.sin(0.5 * i)) for i in range(len(days))],
    )
    del fact["commodity_wti_spot_level"]
    _add_fact(con, days, fact)

    n_wti, n_vix, pairs_wti = con.execute(
        f"SELECT count(macro_beta_wti), count(macro_beta_vix), max(macro_beta_n_wti) "
        f"FROM ({build_macro_exposure_sql()})"
    ).fetchone()
    assert n_wti == 0
    assert pairs_wti == 0
    assert n_vix > 0
