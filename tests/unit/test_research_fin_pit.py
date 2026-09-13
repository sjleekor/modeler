"""Unit tests for P7 — feat_fin_pit (synthetic canonical lake, no real lake).

Verifies the etl_00 §3.3 / etl_01 §6 mechanics: PIT lag (period_end + 90/45d),
no look-ahead, all metric_codes preserved per (t, ticker) (NOT a 1-row ASOF),
derived ratios, and the negative-equity flag/clip.
"""

from __future__ import annotations

import datetime

import duckdb
import pytest
from modeler.etl.features import fin_pit


def _con_with_smf(smf_rows: list[tuple], universe_rows: list[tuple]) -> duckdb.DuckDBPyConnection:
    """Register synthetic stock_metric_fact + dim_universe_daily views.

    smf_rows: (ticker, market, metric_code, period_type, period_end, value)
    universe_rows: (trade_date, ticker, market)  [all in_universe=True]
    """
    con = duckdb.connect()
    smf_vals = ",".join(
        f"('{tk}','{mk}','{mc}','{pt}',DATE '{pe}',{v})" for (tk, mk, mc, pt, pe, v) in smf_rows
    )
    con.execute(
        "CREATE VIEW stock_metric_fact AS SELECT * FROM (VALUES "
        + smf_vals
        + ") AS t(ticker, market, metric_code, period_type, period_end, value_numeric)"
    )
    uni_vals = ",".join(f"(DATE '{d}','{tk}','{mk}',TRUE)" for (d, tk, mk) in universe_rows)
    con.execute(
        "CREATE VIEW dim_universe_daily AS SELECT * FROM (VALUES "
        + uni_vals
        + ") AS t(trade_date, ticker, market, in_universe)"
    )
    return con


def _smf_full(ticker: str, period_end: str, period_type: str, **metrics) -> list[tuple]:
    return [(ticker, "KOSPI", mc, period_type, period_end, v) for mc, v in metrics.items()]


def test_pit_lag_blocks_lookahead() -> None:
    # annual report period_end 2023-12-31 -> available 2024-03-30 (+90d).
    smf = _smf_full("A", "2023-12-31", "annual", total_assets=1000.0, total_equity=600.0)
    # t before availability: no row; t after: present.
    con = _con_with_smf(smf, [("2024-03-29", "A", "KOSPI"), ("2024-03-30", "A", "KOSPI")])
    con.execute(f"CREATE VIEW feat_fin_pit AS {fin_pit.build_fin_pit_sql()}")
    rows = dict(
        con.execute(
            "SELECT trade_date, fin_equity_ratio FROM feat_fin_pit ORDER BY trade_date"
        ).fetchall()
    )
    # 2024-03-29 is before available_from -> not in result at all.
    assert datetime.date(2024, 3, 29) not in rows
    assert rows[datetime.date(2024, 3, 30)] == pytest.approx(0.6)  # 600/1000


def test_all_metrics_preserved_not_single_asof() -> None:
    # 4 distinct metric_codes available -> wide row keeps all 4 (etl_01 §6).
    smf = _smf_full(
        "A",
        "2023-12-31",
        "annual",
        total_assets=1000.0,
        total_liabilities=400.0,
        total_equity=600.0,
        net_income=120.0,
    )
    con = _con_with_smf(smf, [("2024-06-01", "A", "KOSPI")])
    con.execute(f"CREATE VIEW feat_fin_pit AS {fin_pit.build_fin_pit_sql()}")
    row = con.execute(
        "SELECT fin_roa, fin_debt_to_equity, fin_equity_ratio FROM feat_fin_pit"
    ).fetchone()
    assert row[0] == pytest.approx(0.12)  # 120/1000
    assert row[1] == pytest.approx(400 / 600)  # debt/equity
    assert row[2] == pytest.approx(0.6)


def test_latest_available_report_wins() -> None:
    # Two reports available by 2024-08-01: annual (2023-12-31, +90=2024-03-30,
    # equity_ratio 0.5) and q1 (2024-03-31, +45=2024-05-15, equity_ratio 0.8).
    # The q1 interval starts later, so at t=2024-08-01 the q1 value must win.
    smf = _smf_full(
        "A", "2023-12-31", "annual", total_assets=1000.0, total_equity=500.0
    ) + _smf_full("A", "2024-03-31", "q1", total_assets=1000.0, total_equity=800.0)
    con = _con_with_smf(smf, [("2024-08-01", "A", "KOSPI")])
    con.execute(f"CREATE VIEW feat_fin_pit AS {fin_pit.build_fin_pit_sql()}")
    (eq_ratio,) = con.execute("SELECT fin_equity_ratio FROM feat_fin_pit").fetchone()
    assert eq_ratio == pytest.approx(0.8)  # q1 (later available_from) wins over annual


def test_negative_equity_flag_and_clip() -> None:
    smf = _smf_full(
        "A",
        "2023-12-31",
        "annual",
        total_assets=1000.0,
        total_liabilities=1200.0,
        total_equity=-200.0,
    )
    con = _con_with_smf(smf, [("2024-06-01", "A", "KOSPI")])
    con.execute(f"CREATE VIEW feat_fin_pit AS {fin_pit.build_fin_pit_sql()}")
    flag, dte = con.execute(
        "SELECT fin_is_negative_equity, fin_debt_to_equity FROM feat_fin_pit"
    ).fetchone()
    assert flag is True
    assert dte is None  # debt_to_equity clipped to NULL when equity <= 0


def test_quarterly_lag_is_45_days() -> None:
    # q1 period_end 2024-03-31 -> available 2024-05-15 (+45d), not +90.
    smf = _smf_full("A", "2024-03-31", "q1", total_assets=500.0, total_equity=300.0)
    con = _con_with_smf(smf, [("2024-05-14", "A", "KOSPI"), ("2024-05-15", "A", "KOSPI")])
    con.execute(f"CREATE VIEW feat_fin_pit AS {fin_pit.build_fin_pit_sql()}")
    dates = [r[0] for r in con.execute("SELECT trade_date FROM feat_fin_pit").fetchall()]
    assert datetime.date(2024, 5, 14) not in dates
    assert datetime.date(2024, 5, 15) in dates
