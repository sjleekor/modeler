"""Unit tests for dim_regime_daily (20260907_model_experiment `02` §3.1).

The mart must be a *projection* of the Phase C series, not a second
implementation: the model's FS1 -> FS2 comparison only means something if the
regime it conditions on is the same partition Phase C measured the interactions
on. So the central test here re-derives both from one connection and demands
equality, column by column.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path

import duckdb
import polars as pl
import pytest

from modeler.analysis.horizon_scan_phase_c_regimes import (
    LONG_WINDOW,
    REGIME_IDS,
    SHORT_WINDOW,
    SOURCE_FEATURE_CODES,
    build_regime_series,
)
from modeler.etl.config import DataRoot, LakeConfig
from modeler.etl.features.regime import (
    EXPLORATORY_REGIME_COLUMNS,
    MODEL_REGIME_COLUMNS,
    REGIME_TABLE,
    REGIME_VALID_FROM,
    build_regime_mart_sql,
    materialize_regime,
)
from modeler.etl.mart import StaleMartContract, is_materialized

_DEFAULTS = {
    "global_vix_level": 20.0,
    "market_kospi_close": 2500.0,
    "market_kospi_turnover_value": 1.0e12,
    "market_kosdaq_turnover_value": 5.0e11,
    "rate_kr_term_spread_10y_3y": 0.5,
    "market_kosdaq_ret_1d": 0.0,
    "market_kospi_ret_1d": 0.0,
    "fx_usdkrw_level": 1300.0,
}


def _sessions(start: date, count: int) -> list[date]:
    days: list[date] = []
    current = start
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def _con(
    sessions: list[date],
    *,
    fact_dates: list[date] | None = None,
    **series: list[float],
) -> duckdb.DuckDBPyConnection:
    fact_dates = fact_dates or sessions
    con = duckdb.connect()
    con.execute("CREATE TABLE label_scan (trade_date DATE)")
    con.executemany("INSERT INTO label_scan VALUES (?)", [(d,) for d in sessions])
    con.execute(
        "CREATE TABLE common_feature_daily_fact (feature_date DATE, feature_code VARCHAR, "
        "value_numeric DOUBLE, asof_available_date DATE)"
    )
    rows = []
    for code in SOURCE_FEATURE_CODES:
        values = series.get(code, [_DEFAULTS[code]] * len(fact_dates))
        for d, value in zip(fact_dates, values, strict=True):
            rows.append((d, code, value, d))
    con.executemany("INSERT INTO common_feature_daily_fact VALUES (?,?,?,?)", rows)
    return con


def _mart(con: duckdb.DuckDBPyConnection, **kwargs) -> pl.DataFrame:
    """The mart, with warm-up gating off unless a test asks for it.

    The synthetic fixtures below sit in 2018-2020, so the real lake's
    ``REGIME_VALID_FROM`` would blank them wholesale; the gate has its own
    tests instead.
    """
    kwargs.setdefault("valid_from", "1900-01-01")
    return con.execute(build_regime_mart_sql(**kwargs)).pl().sort("trade_date")


# --- column contract ---------------------------------------------------------


def test_the_four_model_regimes_are_the_primary_ones() -> None:
    assert MODEL_REGIME_COLUMNS == ("rg_vix_up", "rg_vix_high", "rg_market_up", "rg_liq_high")
    assert EXPLORATORY_REGIME_COLUMNS == (
        "rg_term_steep",
        "rg_kosdaq_rel_up",
        "rg_krw_weak_20d",
    )


def test_mart_emits_one_binary_and_one_z_per_regime_and_nothing_else() -> None:
    frame = _mart(_con(_sessions(date(2020, 1, 1), 300)))
    expected = {"trade_date"}
    expected |= {f"rg_{rid}" for rid in REGIME_IDS}
    expected |= {f"rg_{rid}_z" for rid in REGIME_IDS}
    assert set(frame.columns) == expected
    # Phase C diagnostics stay in Phase C: the alternative cut is not a judged
    # partition, and a second session index would shadow the panel's own.
    assert not [c for c in frame.columns if c.startswith("alt_")]
    assert "session_idx" not in frame.columns


def test_the_binaries_are_boolean_and_the_z_columns_are_double() -> None:
    frame = _mart(_con(_sessions(date(2020, 1, 1), 300)))
    for rid in REGIME_IDS:
        assert frame.schema[f"rg_{rid}"] == pl.Boolean
        assert frame.schema[f"rg_{rid}_z"] == pl.Float64


# --- parity with Phase C -----------------------------------------------------


def test_every_column_equals_the_phase_c_series_it_renames() -> None:
    sessions = _sessions(date(2018, 1, 1), 600)
    n = len(sessions)
    # Periods chosen so all seven regimes actually flip inside the sample: a
    # 252-session comparison needs the underlying series to cycle slower than
    # its own window, a 20-session one faster.
    con = _con(
        sessions,
        global_vix_level=[20.0 + 6.0 * math.sin(2 * math.pi * i / 61) for i in range(n)],
        market_kospi_close=[
            2500.0 * (1.0 + 0.25 * math.sin(2 * math.pi * i / 311)) for i in range(n)
        ],
        market_kospi_turnover_value=[
            1.0e12 * (1.0 + 0.4 * math.sin(2 * math.pi * i / 97)) for i in range(n)
        ],
        rate_kr_term_spread_10y_3y=[0.5 + 0.2 * math.sin(2 * math.pi * i / 83) for i in range(n)],
        market_kosdaq_ret_1d=[0.001 * math.sin(2 * math.pi * i / 23) for i in range(n)],
        fx_usdkrw_level=[1300.0 + 40.0 * math.sin(2 * math.pi * i / 71) for i in range(n)],
    )
    phase_c = build_regime_series(con).sort("trade_date")
    mart = _mart(con)

    assert mart["trade_date"].to_list() == phase_c["trade_date"].to_list()
    for rid in REGIME_IDS:
        assert mart[f"rg_{rid}"].to_list() == phase_c[f"s_{rid}"].to_list()
        assert mart[f"rg_{rid}_z"].to_list() == phase_c[f"z_{rid}"].to_list()
    # the fixture has to actually exercise both sides of every partition, or the
    # equality above would be a comparison of two all-NULL columns.
    for column in (*MODEL_REGIME_COLUMNS, *EXPLORATORY_REGIME_COLUMNS):
        assert set(mart[column].drop_nulls().to_list()) == {True, False}


# --- the session grid --------------------------------------------------------


def test_the_grid_is_the_session_list_not_the_fact_calendar() -> None:
    """A KRX holiday sitting in the fact must not become a regime row.

    ``common_feature_daily_fact`` is indexed by every weekday for 2014-2023, so
    counting 252 rows on the fact's own axis would measure a different window
    before and after 2024 (Phase C's module docstring). The mart joins the
    scan's sessions instead.
    """
    sessions = _sessions(date(2020, 1, 1), 100)
    fact_dates = _sessions(date(2020, 1, 1), 140)  # 40 non-session weekdays
    frame = _mart(_con(sessions, fact_dates=fact_dates))
    assert frame["trade_date"].to_list() == sessions
    assert frame.height == len(sessions)


# --- warm-up -----------------------------------------------------------------


def test_a_regime_is_null_until_its_window_is_full() -> None:
    sessions = _sessions(date(2020, 1, 1), 300)
    frame = _mart(_con(sessions, global_vix_level=[20.0 + 0.1 * i for i in range(300)]))
    # a 20-session difference reaches back to t-20, so it needs 21 rows;
    # a 252-session rolling median needs 252.
    assert frame["rg_vix_up"][SHORT_WINDOW - 1] is None
    assert frame["rg_vix_up"][SHORT_WINDOW] is not None
    assert frame["rg_vix_high"][LONG_WINDOW - 2] is None
    assert frame["rg_vix_high"][LONG_WINDOW - 1] is not None
    assert frame["rg_market_up"][LONG_WINDOW - 1] is None  # a 252 *difference*
    assert frame["rg_market_up"][LONG_WINDOW] is not None


def test_every_regime_is_null_before_the_observation_window_is_full() -> None:
    """`02` §3.1: the fact's daily series start 2014-06-16, so a 252-session
    median before 2015-06-16 is taken over a window that is not full of
    observations. Phase C trims those sessions; the mart NULLs them and keeps
    the row, so the panel join sees an explicit unknown.
    """
    assert REGIME_VALID_FROM == "2015-06-16"
    sessions = _sessions(date(2015, 1, 1), 300)
    frame = _mart(_con(sessions), valid_from=REGIME_VALID_FROM)
    assert frame.height == len(sessions)  # rows kept
    early = frame.filter(pl.col("trade_date") < date(2015, 6, 16))
    late = frame.filter(pl.col("trade_date") >= date(2015, 6, 16))
    assert early.height > 0 and late.height > 0
    for rid in REGIME_IDS:
        assert early[f"rg_{rid}"].null_count() == early.height
        assert early[f"rg_{rid}_z"].null_count() == early.height
    # after the cut, the short-window regimes are populated again.
    assert late["rg_vix_up"].null_count() < late.height


def test_the_warm_up_gate_does_not_touch_values_after_the_cut() -> None:
    sessions = _sessions(date(2015, 1, 1), 400)
    con = _con(sessions, global_vix_level=[20.0 + 0.1 * i for i in range(400)])
    ungated = _mart(con)
    gated = _mart(con, valid_from=REGIME_VALID_FROM)
    after = pl.col("trade_date") >= date(2015, 6, 16)
    assert gated.filter(after).equals(ungated.filter(after))


# --- materialization ---------------------------------------------------------


def test_materialize_writes_and_registers_the_mart(tmp_path: Path) -> None:
    sessions = _sessions(date(2020, 1, 1), 300)  # past REGIME_VALID_FROM
    con = _con(sessions)
    config = LakeConfig(root=DataRoot(tmp_path), snapshot_date="2020-12-31", source="test")

    view = materialize_regime(con, config)

    assert view == REGIME_TABLE
    assert is_materialized(config, REGIME_TABLE)
    rows, first, last = con.execute(
        f"SELECT count(*), min(trade_date), max(trade_date) FROM {view}"
    ).fetchone()
    assert rows == len(sessions)
    assert (first, last) == (sessions[0], sessions[-1])
    # cached: a second call is a no-op that still hands back a usable view.
    assert materialize_regime(con, config) == REGIME_TABLE


def test_materialize_refuses_a_stale_contract(tmp_path: Path) -> None:
    """The mart is cache-keyed on its SQL, so a changed definition cannot be reused."""
    sessions = _sessions(date(2020, 1, 1), 300)
    con = _con(sessions)
    config = LakeConfig(root=DataRoot(tmp_path), snapshot_date="2020-12-31", source="test")
    materialize_regime(con, config)

    root = tmp_path / "derived" / "feature" / "snapshot_date=2020-12-31" / "source=test"
    path = root / REGIME_TABLE / "_cache_metadata.json"
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["sql_hash"] = "0" * 64  # as if the definition had changed under it
    path.write_text(json.dumps(stored, sort_keys=True), encoding="utf-8")
    with pytest.raises(StaleMartContract, match="contract"):
        materialize_regime(con, config)
    # ...and rebuilding on request is how the caller gets out of it.
    assert materialize_regime(con, config, force=True) == REGIME_TABLE
