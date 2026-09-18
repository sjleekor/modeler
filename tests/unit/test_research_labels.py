"""Unit tests for P4 — make_label (synthetic fixtures, no lake).

Checks the etl_00 §2 mechanics: d_idx forward over non-halt sessions, eqw-market
excess (per-date mean ~0), percentile rank in [0,1], 3-class thresholds, and the
LabelSpec validation. Risk labels (vol/mdd) get a small monotonic-path check.
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import pytest

from modeler.etl import labels


def _ohlcv_view(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> None:
    values = ",".join(
        f"(DATE '{d}', '{tk}', '{mk}', {o}, {h}, {lo}, {c}, {v})"
        for (d, tk, mk, o, h, lo, c, v) in rows
    )
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + values
        + ") AS t(trade_date, ticker, market, open, high, low, close, volume)"
    )


# --- LabelSpec validation ---------------------------------------------------


def test_labelspec_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        labels.LabelSpec(kind="weird")
    with pytest.raises(ValueError):
        labels.LabelSpec(outputs=("rank", "nope"))
    with pytest.raises(ValueError):
        labels.LabelSpec(winsor=(0.9, 0.1))
    with pytest.raises(ValueError):
        labels.LabelSpec(horizons=())


def test_bench_rejects_the_unimplemented_index_swap() -> None:
    # ``index_bench=True`` adds a second benchmark; swapping the primary one is
    # not offered, because two label variants would share a dataset directory.
    with pytest.raises(ValueError):
        labels.LabelSpec(bench="index")


def _index_view(con: duckdb.DuckDBPyConnection, rows: list[tuple]) -> None:
    values = ",".join(f"(DATE '{d}', '{sid}', {v})" for (d, sid, v) in rows)
    con.execute(
        "CREATE VIEW common_feature_observation_raw AS SELECT * FROM (VALUES "
        + values
        + ") AS t(observation_date, series_id, value_numeric)"
    )


def test_index_bench_rides_beside_the_eqw_label(tmp_path: Path) -> None:
    """R5: ``raw_label_idx`` is added, and the eqw label is untouched."""
    con = duckdb.connect()
    # Two KOSPI names over 3 sessions. h=1.
    _ohlcv_view(
        con,
        [
            ("2020-01-01", "A", "KOSPI", 10, 11, 9, 100, 5),
            ("2020-01-02", "A", "KOSPI", 10, 11, 9, 120, 5),  # +20%
            ("2020-01-01", "B", "KOSPI", 10, 11, 9, 100, 5),
            ("2020-01-02", "B", "KOSPI", 10, 11, 9, 100, 5),  # 0%
        ],
    )
    # index rises 10% over the same session -> A beats it by 10pp, B trails 10pp.
    _index_view(
        con,
        [
            ("2020-01-01", "market_kospi_krx", 1000.0),
            ("2020-01-02", "market_kospi_krx", 1100.0),
        ],
    )
    spec = labels.LabelSpec(horizons=(1,), outputs=("rank",), index_bench=True)
    rows = con.execute(
        labels.build_label_sql(spec) + " ORDER BY ticker"
    ).fetchall()
    cols = [d[0] for d in con.execute(labels.build_label_sql(spec)).description]
    got = [dict(zip(cols, r)) for r in rows]

    assert [g["ticker"] for g in got] == ["A", "B"]
    # eqw bench is the 10% cross-sectional mean -> unchanged by index_bench
    assert got[0]["raw_label_1d"] == pytest.approx(0.10)
    assert got[1]["raw_label_1d"] == pytest.approx(-0.10)
    # index bench is the index's own 10% move
    assert got[0]["bench_ret_idx_1d"] == pytest.approx(0.10)
    assert got[0]["raw_label_idx_1d"] == pytest.approx(0.10)
    assert got[1]["raw_label_idx_1d"] == pytest.approx(-0.10)


def test_index_bench_keeps_rows_the_index_cannot_price(tmp_path: Path) -> None:
    """A market with no index series keeps its rows, with a NULL benchmark."""
    con = duckdb.connect()
    _ohlcv_view(
        con,
        [
            ("2020-01-01", "A", "KOSPI", 10, 11, 9, 100, 5),
            ("2020-01-02", "A", "KOSPI", 10, 11, 9, 120, 5),
            ("2020-01-01", "K", "KOSDAQ", 10, 11, 9, 100, 5),
            ("2020-01-02", "K", "KOSDAQ", 10, 11, 9, 150, 5),
        ],
    )
    _index_view(  # KOSPI only — KOSDAQ's index is absent
        con,
        [
            ("2020-01-01", "market_kospi_krx", 1000.0),
            ("2020-01-02", "market_kospi_krx", 1100.0),
        ],
    )
    spec = labels.LabelSpec(horizons=(1,), outputs=("rank",), index_bench=True)
    sql = labels.build_label_sql(spec)
    cols = [d[0] for d in con.execute(sql).description]
    got = [dict(zip(cols, r)) for r in con.execute(sql + " ORDER BY ticker").fetchall()]

    assert [g["ticker"] for g in got] == ["A", "K"]  # nothing dropped
    assert got[1]["market"] == "KOSDAQ"
    assert got[1]["bench_ret_idx_1d"] is None
    assert got[1]["raw_label_idx_1d"] is None
    assert got[1]["raw_label_1d"] is not None  # the eqw label still works


def test_index_bench_off_leaves_the_sql_alone() -> None:
    """Default builds must stay byte-identical — runs already judged on them."""
    plain = labels.build_label_sql(labels.LabelSpec())
    assert "idx" not in plain.replace("d_idx", "")
    assert "common_feature_observation_raw" not in plain


# --- forward join over non-halt sessions ------------------------------------


def test_forward_skips_halt_days(tmp_path: Path) -> None:
    """t+1 forward return must skip a halt session (etl_00 §2.1)."""
    con = duckdb.connect()
    # 4 sessions for ticker A; session 3 (2020-01-03) is halted.
    rows = [
        ("2020-01-01", "A", "KOSPI", 10, 11, 9, 100, 5),
        ("2020-01-02", "A", "KOSPI", 10, 11, 9, 110, 5),
        ("2020-01-03", "A", "KOSPI", 0, 0, 0, 110, 0),  # halted, excluded
        ("2020-01-04", "A", "KOSPI", 10, 11, 9, 121, 5),
    ]
    # second ticker B so the eqw benchmark has >1 name per date
    rows += [
        ("2020-01-01", "B", "KOSPI", 10, 11, 9, 200, 5),
        ("2020-01-02", "B", "KOSPI", 10, 11, 9, 200, 5),
        ("2020-01-04", "B", "KOSPI", 10, 11, 9, 200, 5),
    ]
    _ohlcv_view(con, rows)
    spec = labels.LabelSpec(horizons=(1,), outputs=("reg", "rank", "cls"))
    con.execute(f"CREATE VIEW label_daily AS {labels.build_label_sql(spec)}")

    # A on 2020-01-02: next non-halt session is 2020-01-04 (close 121), not the
    # halted 2020-01-03 -> fwd = 121/110 - 1.
    (fwd,) = con.execute(
        "SELECT fwd_ret_1d FROM label_daily WHERE ticker='A' AND trade_date=DATE '2020-01-02'"
    ).fetchone()
    assert fwd == pytest.approx(121 / 110 - 1)


# --- excess / rank / cls semantics ------------------------------------------


@pytest.fixture()
def ranked_con() -> duckdb.DuckDBPyConnection:
    """5 tickers on one date with distinct +1 forward returns -> clean ranks."""
    con = duckdb.connect()
    rows = []
    # day1 (entry) close = 100 for all; day2 close varies to set forward return.
    closes_day2 = {"A": 100, "B": 105, "C": 110, "D": 115, "E": 120}
    for tk, c2 in closes_day2.items():
        rows.append(("2020-01-01", tk, "KOSPI", 10, 11, 9, 100, 5))
        rows.append(("2020-01-02", tk, "KOSPI", 10, 11, 9, c2, 5))
    _ohlcv_view(con, rows)
    spec = labels.LabelSpec(horizons=(1,), outputs=("reg", "rank", "cls"))
    con.execute(f"CREATE VIEW label_daily AS {labels.build_label_sql(spec)}")
    return con


def test_eqw_excess_mean_zero_per_date(ranked_con) -> None:
    (avg_excess,) = ranked_con.execute(
        "SELECT AVG(raw_label_1d) FROM label_daily WHERE trade_date=DATE '2020-01-01'"
    ).fetchone()
    assert avg_excess == pytest.approx(0.0, abs=1e-12)


def test_rank_in_unit_interval_and_ordered(ranked_con) -> None:
    rows = ranked_con.execute(
        "SELECT ticker, y_rank_1d FROM label_daily "
        "WHERE trade_date=DATE '2020-01-01' ORDER BY y_rank_1d"
    ).fetchall()
    ranks = [r[1] for r in rows]
    assert ranks[0] == pytest.approx(0.0)
    assert ranks[-1] == pytest.approx(1.0)
    # A (flat) lowest, E (highest return) top.
    assert rows[0][0] == "A"
    assert rows[-1][0] == "E"


def test_cls_thresholds(ranked_con) -> None:
    rows = dict(
        ranked_con.execute(
            "SELECT ticker, y_cls_1d FROM label_daily WHERE trade_date=DATE '2020-01-01'"
        ).fetchall()
    )
    # 5 names, PERCENT_RANK: A=0, B=.25, C=.5, D=.75, E=1.0
    # cls_top=0.8 -> only E is +1 ; cls_bottom=0.2 -> only A is -1.
    assert rows["E"] == 1
    assert rows["A"] == -1
    assert rows["C"] == 0


# --- risk labels ------------------------------------------------------------


def test_risk_vol_and_mdd_on_path(tmp_path: Path) -> None:
    con = duckdb.connect()
    # monotonic up then down so MDD is clearly negative, vol > 0.
    closes = [100, 110, 121, 121 * 0.9, 121 * 0.9 * 1.05]
    rows = [
        (f"2020-01-0{i+1}", "A", "KOSPI", 10, 11, 9, round(c, 4), 5) for i, c in enumerate(closes)
    ]
    _ohlcv_view(con, rows)
    spec = labels.LabelSpec(risk_horizons=(4,))
    sql = labels.build_risk_label_sql(spec)
    vol, mdd = con.execute(
        f"WITH r AS ({sql}) SELECT y_vol_4d, y_mdd_4d FROM r "
        "WHERE ticker='A' AND trade_date=DATE '2020-01-01'"
    ).fetchone()
    assert vol is not None and vol > 0
    assert mdd is not None and mdd < 0  # the -10% leg creates a drawdown
    # the single largest daily drop is ln(0.9) ~ -0.105; mdd should be <= that.
    assert mdd <= math.log(0.9) + 1e-9


def test_secondary_horizon_no_forward_has_null_outputs(tmp_path: Path) -> None:
    """Regression: a horizon whose forward return is NULL must not get outputs.

    A 30-session single ticker has no 60-session forward, so raw_label_60d is
    all NULL; y_rank_60d / y_cls_60d / y_reg_60d must be NULL too.

    ``y_reg`` is the subtle one: PERCENT_RANK assigns a value to NULL keys, and
    ``LEAST(GREATEST(NULL, lo), hi)`` returns ``lo`` because DuckDB's
    GREATEST/LEAST skip NULL arguments. Both need an explicit guard.
    """
    con = duckdb.connect()
    rows = [(f"2020-01-{i + 1:02d}", "A", "KOSPI", 10, 11, 9, 100 + i, 5) for i in range(30)]
    _ohlcv_view(con, rows)
    con.execute(f"CREATE VIEW label_daily AS {labels.build_label_sql(labels.LabelSpec())}")
    raw60, rank60, cls60, reg60 = con.execute(
        "SELECT count(raw_label_60d), count(y_rank_60d), count(y_cls_60d), count(y_reg_60d) "
        "FROM label_daily"
    ).fetchone()
    assert raw60 == 0
    assert rank60 == 0  # not 20 — the bug would assign ranks to NULL-forward rows
    assert cls60 == 0
    assert reg60 == 0  # not 20 — the bug would clip NULL to the bottom quantile


def test_greatest_least_skip_nulls() -> None:
    """The DuckDB behaviour the y_reg guard exists for — pinned so it stays true.

    If a future DuckDB made GREATEST NULL-propagating this test would fail and
    the guard could be simplified; until then, every winsorize site needs it.
    """
    con = duckdb.connect()
    (clipped,) = con.execute(
        "SELECT LEAST(GREATEST(NULL, 0.1::DOUBLE), 0.9::DOUBLE)"
    ).fetchone()
    assert clipped == pytest.approx(0.1)  # not NULL


def test_risk_labels_emitted_only_when_included(tmp_path: Path) -> None:
    """Fix #4: y_vol/y_mdd land in label_daily iff include_risk=True."""
    con = duckdb.connect()
    rows = [(f"2020-01-{i + 1:02d}", "A", "KOSPI", 10, 11, 9, 100 + i, 5) for i in range(12)]
    _ohlcv_view(con, rows)

    on = labels.LabelSpec(horizons=(5,), risk_horizons=(5,), include_risk=True)
    con.execute(f"CREATE VIEW lab_on AS {labels.build_label_sql(on)}")
    cols_on = {c[0] for c in con.execute("DESCRIBE lab_on").fetchall()}
    assert "y_vol_5d" in cols_on and "y_mdd_5d" in cols_on

    off = labels.LabelSpec(horizons=(5,))
    con.execute(f"CREATE VIEW lab_off AS {labels.build_label_sql(off)}")
    cols_off = {c[0] for c in con.execute("DESCRIBE lab_off").fetchall()}
    assert "y_vol_5d" not in cols_off  # default off (expensive MDD path)
