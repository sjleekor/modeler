"""Unit tests for P8 — feat_common (broadcast/PIT) and feat_event (flags).

Synthetic canonical/raw views (no real lake). feat_common: PIT filter on
asof_available_date, one row per date, latest-available wins. feat_event:
se='합계' totals, +90d annual PIT lag, treasury/shares flags.
"""

from __future__ import annotations

import datetime

import duckdb
import pytest

from modeler.etl.features import common, event

# --- feat_common ------------------------------------------------------------


def _cfdf_con(rows: list[tuple]) -> duckdb.DuckDBPyConnection:
    """rows: (feature_date, feature_code, value_numeric, asof_available_date)."""
    con = duckdb.connect()
    vals = ",".join(f"(DATE '{fd}','{fc}',{v},DATE '{ad}')" for (fd, fc, v, ad) in rows)
    con.execute(
        "CREATE VIEW common_feature_daily_fact AS SELECT * FROM (VALUES "
        + vals
        + ") AS t(feature_date, feature_code, value_numeric, asof_available_date)"
    )
    return con


def _make_common_view(con: duckdb.DuckDBPyConnection, *codes: str) -> None:
    """Register feat_common over the given feature codes (test helper)."""
    con.execute(f"CREATE VIEW feat_common AS {common.build_common_sql(feature_codes=codes)}")


def test_common_one_row_per_date_broadcast_grain() -> None:
    rows = [
        ("2026-01-02", "market_kospi_ret_5d", 0.01, "2026-01-02"),
        ("2026-01-02", "global_vix_level", 15.0, "2026-01-02"),
        ("2026-01-03", "market_kospi_ret_5d", 0.02, "2026-01-03"),
    ]
    con = _cfdf_con(rows)
    codes = ("market_kospi_ret_5d", "global_vix_level")
    con.execute(f"CREATE VIEW feat_common AS {common.build_common_sql(feature_codes=codes)}")
    out = con.execute(
        "SELECT trade_date, cf_market_kospi_ret_5d, cf_global_vix_level "
        "FROM feat_common ORDER BY trade_date"
    ).fetchall()
    assert len(out) == 2  # one row per feature_date
    assert out[0][1] == pytest.approx(0.01)
    assert out[0][2] == pytest.approx(15.0)


def test_common_pit_excludes_unavailable_rows() -> None:
    # value known only on 2026-01-05 must not appear for feature_date 2026-01-02.
    rows = [("2026-01-02", "market_kospi_ret_5d", 0.99, "2026-01-05")]
    con = _cfdf_con(rows)
    _make_common_view(con, "market_kospi_ret_5d")
    n = con.execute("SELECT count(*) FROM feat_common").fetchone()[0]
    assert n == 0  # asof_available_date > feature_date -> filtered out


def test_common_latest_available_vintage_wins() -> None:
    # Two vintages for the same feature_date, BOTH available by then (asof <=
    # feature_date) -> the later asof (restatement known in time) wins.
    rows = [
        ("2026-01-05", "market_kospi_ret_5d", 0.01, "2026-01-02"),
        ("2026-01-05", "market_kospi_ret_5d", 0.05, "2026-01-04"),
    ]
    con = _cfdf_con(rows)
    _make_common_view(con, "market_kospi_ret_5d")
    (v,) = con.execute("SELECT cf_market_kospi_ret_5d FROM feat_common").fetchone()
    assert v == pytest.approx(0.05)  # later asof_available_date wins


def test_common_restatement_after_date_excluded() -> None:
    # A vintage that becomes known AFTER the feature_date must not be used (would
    # be look-ahead). Only the in-time value remains.
    rows = [
        ("2026-01-02", "market_kospi_ret_5d", 0.01, "2026-01-02"),
        ("2026-01-02", "market_kospi_ret_5d", 0.05, "2026-01-04"),  # too late
    ]
    con = _cfdf_con(rows)
    _make_common_view(con, "market_kospi_ret_5d")
    (v,) = con.execute("SELECT cf_market_kospi_ret_5d FROM feat_common").fetchone()
    assert v == pytest.approx(0.01)  # restatement excluded by PIT filter


# --- feat_event -------------------------------------------------------------


def _event_con(share_rows: list[tuple], universe_rows: list[tuple]) -> duckdb.DuckDBPyConnection:
    """share_rows: (ticker, bsns_year, se, istc_totqy, tesstk_co)."""
    con = duckdb.connect()
    svals = ",".join(f"('{tk}',{by},'{se}',{istc},{tes})" for (tk, by, se, istc, tes) in share_rows)
    con.execute(
        "CREATE VIEW dart_share_count_raw AS SELECT * FROM (VALUES "
        + svals
        + ") AS t(ticker, bsns_year, se, istc_totqy, tesstk_co)"
    )
    uvals = ",".join(f"(DATE '{d}','{tk}','{mk}',TRUE)" for (d, tk, mk) in universe_rows)
    con.execute(
        "CREATE VIEW dim_universe_daily AS SELECT * FROM (VALUES "
        + uvals
        + ") AS t(trade_date, ticker, market, in_universe)"
    )
    return con


def test_event_treasury_ratio_and_flag() -> None:
    # 2023 total: issued 1000, treasury 100 -> ratio 0.1, available 2024-03-31.
    con = _event_con(
        [("A", 2023, "합계", 1000, 100)],
        [("2024-06-01", "A", "KOSPI")],
    )
    con.execute(f"CREATE VIEW feat_event AS {event.build_event_sql()}")
    ratio, has = con.execute("SELECT ev_treasury_ratio, ev_has_treasury FROM feat_event").fetchone()
    assert ratio == pytest.approx(0.1)
    assert has is True


def test_event_pit_lag_blocks_lookahead() -> None:
    # 2023 bsns_year -> available 2023-12-31 + 90d = 2024-03-30 (2024 is a leap
    # year). A day before availability is excluded; the availability day itself
    # (and after) is included.
    con = _event_con(
        [("A", 2023, "합계", 1000, 0)],
        [
            ("2024-03-29", "A", "KOSPI"),  # before available_from
            ("2024-03-30", "A", "KOSPI"),  # == available_from
            ("2024-04-15", "A", "KOSPI"),  # after
        ],
    )
    con.execute(f"CREATE VIEW feat_event AS {event.build_event_sql()}")
    dates = [r[0] for r in con.execute("SELECT trade_date FROM feat_event").fetchall()]
    assert datetime.date(2024, 3, 29) not in dates
    assert datetime.date(2024, 3, 30) in dates
    assert datetime.date(2024, 4, 15) in dates


def test_event_shares_chg_yoy() -> None:
    # 2022 issued 1000, 2023 issued 1100 -> yoy +0.1 once 2023 is available.
    con = _event_con(
        [("A", 2022, "합계", 1000, 0), ("A", 2023, "합계", 1100, 0)],
        [("2024-06-01", "A", "KOSPI")],
    )
    con.execute(f"CREATE VIEW feat_event AS {event.build_event_sql()}")
    (chg,) = con.execute("SELECT ev_shares_chg_yoy FROM feat_event").fetchone()
    assert chg == pytest.approx(0.1)


def test_event_only_total_rows_used() -> None:
    # non-합계 rows (보통주/우선주) must be ignored.
    con = _event_con(
        [("A", 2023, "보통주", 800, 50), ("A", 2023, "합계", 1000, 100)],
        [("2024-06-01", "A", "KOSPI")],
    )
    con.execute(f"CREATE VIEW feat_event AS {event.build_event_sql()}")
    (ratio,) = con.execute("SELECT ev_treasury_ratio FROM feat_event").fetchone()
    assert ratio == pytest.approx(0.1)  # 100/1000 from the 합계 row, not 50/800
