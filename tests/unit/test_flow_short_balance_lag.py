"""KR 10월 계획 4.5 — 잔고 두 컬럼의 공표 지연(`short_balance_lag_sessions`)."""

from __future__ import annotations

import duckdb
import pytest

from modeler.etl.features.flow import build_flow_sql, materialize_flow
from modeler.models._02_updown_prob.experiments import flow_lag_v2
from modeler.models._02_updown_prob.features import FS0_FLOW_COLS

BALANCE = "flow_short_balance_qty"
CHG = "flow_short_balance_chg_20d"
N_DAYS = 30
SKIPPED = 15  # ticker B has no flow row on this session


def _day(i: int) -> str:
    return f"2024-03-{i:02d}"


def _con() -> duckdb.DuckDBPyConnection:
    """A: every session. B: same, but session SKIPPED has no flow row at all."""
    con = duckdb.connect()
    flow_rows, price_rows, pit_rows = [], [], []
    for ticker in ("A", "B"):
        for i in range(1, N_DAYS + 1):
            d = _day(i)
            price_rows.append(f"(DATE '{d}', '{ticker}', 'KOSPI', 100,100,100,100,{1000 + i})")
            pit_rows.append(f"(DATE '{d}', '{ticker}', 'KOSPI', 10000.0)")
            if ticker == "B" and i == SKIPPED:
                continue
            for code, value in (
                ("foreign_net_buy_volume", 1.0),
                ("short_selling_volume", 2.0),
                ("short_selling_balance_quantity", 100.0 * i + (7 if ticker == "B" else 0)),
            ):
                flow_rows.append(f"(DATE '{d}', '{ticker}', 'KOSPI', '{code}', {value}, 'KRX')")
    con.execute(
        "CREATE VIEW krx_security_flow_raw AS SELECT * FROM (VALUES "
        + ",".join(flow_rows)
        + ") t(trade_date,ticker,market,metric_code,value,source)"
    )
    con.execute(
        "CREATE VIEW daily_ohlcv AS SELECT * FROM (VALUES "
        + ",".join(price_rows)
        + ") t(trade_date,ticker,market,open,high,low,close,volume)"
    )
    con.execute(
        "CREATE VIEW dim_stock_pit_daily AS SELECT * FROM (VALUES "
        + ",".join(pit_rows)
        + ") t(trade_date,ticker,market,float_shares_pit)"
    )
    con.execute(
        "CREATE VIEW dim_price_quality_daily AS SELECT trade_date,ticker,market, "
        "TRUE AS short_balance_is_available, 'allowed' AS short_regime, "
        "ROW_NUMBER() OVER (PARTITION BY ticker, market ORDER BY trade_date) AS valid_session_idx "
        "FROM daily_ohlcv"
    )
    return con


def _sql(lag: int | None = None) -> str:
    kwargs = {} if lag is None else {"short_balance_lag_sessions": lag}
    return build_flow_sql(
        price_view="daily_ohlcv",
        pit_view="dim_stock_pit_daily",
        quality_view="dim_price_quality_daily",
        **kwargs,
    )


def _series(con, lag: int | None, ticker: str, column: str) -> dict[str, float | None]:
    con.execute(f"CREATE OR REPLACE VIEW feat_flow AS {_sql(lag)}")
    rows = con.execute(
        f"SELECT trade_date, {column} FROM feat_flow WHERE ticker = ? ORDER BY trade_date",
        [ticker],
    ).fetchall()
    return {str(d): v for d, v in rows}


def test_default_and_zero_lag_emit_the_same_text() -> None:
    assert _sql(0) == _sql()
    assert "balance_lag" not in _sql()
    assert "balance_lag" in _sql(2)


def test_negative_lag_is_refused() -> None:
    with pytest.raises(ValueError):
        build_flow_sql(short_balance_lag_sessions=-1)


def test_default_output_is_the_unshifted_balance() -> None:
    con = _con()
    base = _series(con, None, "A", BALANCE)
    days = range(1, N_DAYS + 1)
    assert [base[_day(i)] for i in days] == [100.0 * i for i in days]
    assert _series(con, 0, "A", BALANCE) == base


@pytest.mark.parametrize("lag", [1, 2, 3])
def test_balance_qty_moves_back_exactly_lag_sessions(lag: int) -> None:
    con = _con()
    base = _series(con, None, "A", BALANCE)
    shifted = _series(con, lag, "A", BALANCE)
    for i in range(1, N_DAYS + 1):
        d = _day(i)
        if i <= lag:
            assert shifted[d] is None, d
        else:
            # row t carries the balance measured at session t-lag
            assert shifted[d] == base[_day(i - lag)] == 100.0 * (i - lag), d


def test_balance_chg_20d_moves_with_the_same_shift() -> None:
    con = _con()
    base = _series(con, None, "A", CHG)
    shifted = _series(con, 2, "A", CHG)
    assert base[_day(21)] == pytest.approx(2000.0)  # 100*21 - 100*1
    assert base[_day(20)] is None
    assert shifted[_day(22)] is None  # its source session (20) has no 20d change yet
    assert shifted[_day(23)] == base[_day(21)] == pytest.approx(2000.0)
    assert shifted[_day(N_DAYS)] == base[_day(N_DAYS - 2)]


def test_a_missing_flow_row_gives_null_not_a_value_from_the_wrong_day() -> None:
    con = _con()
    base = _series(con, None, "B", BALANCE)
    shifted = _series(con, 2, "B", BALANCE)
    assert _day(SKIPPED) not in base
    # 2 sessions back from SKIPPED+2 is SKIPPED, which has no row -> NULL
    assert shifted[_day(SKIPPED + 2)] is None
    # SKIPPED+1 looks 2 sessions back = SKIPPED-1, a real row; a row-count LAG would hit SKIPPED-2
    assert shifted[_day(SKIPPED + 1)] == base[_day(SKIPPED - 1)]
    assert shifted[_day(SKIPPED + 3)] == base[_day(SKIPPED + 1)]


def test_every_other_column_and_the_row_set_are_unchanged() -> None:
    con = _con()
    con.execute(f"CREATE VIEW f0 AS {_sql()}")
    con.execute(f"CREATE VIEW f2 AS {_sql(2)}")
    cols0 = [r[0] for r in con.execute("DESCRIBE f0").fetchall()]
    cols2 = [r[0] for r in con.execute("DESCRIBE f2").fetchall()]
    assert cols0 == cols2
    other = ", ".join(c for c in cols0 if c not in (BALANCE, CHG))
    diff = con.execute(
        f"SELECT count(*) FROM ((SELECT {other} FROM f0 EXCEPT SELECT {other} FROM f2) "
        f"UNION ALL (SELECT {other} FROM f2 EXCEPT SELECT {other} FROM f0))"
    ).fetchone()[0]
    assert diff == 0


def test_the_two_columns_are_the_baseline_balance_pair() -> None:
    assert {BALANCE, CHG} <= set(FS0_FLOW_COLS)


def test_lag_cannot_overwrite_the_shared_mart_in_place() -> None:
    with pytest.raises(ValueError, match="separate lake root"):
        materialize_flow(None, None, short_balance_lag_sessions=2, force=True)  # type: ignore[arg-type]


def test_lagged_root_is_not_the_shared_root(monkeypatch, tmp_path) -> None:
    from modeler.models._02_updown_prob.experiments import isolated_lake as iso

    monkeypatch.setattr(iso, "_shared_root", lambda: type("R", (), {"derived": tmp_path})())
    flow_lag_v2.lagged_root.cache_clear()
    try:
        assert flow_lag_v2.lagged_root().base == tmp_path / flow_lag_v2.ROOT_NAME
        assert flow_lag_v2.LAG_SESSIONS == 2
    finally:
        flow_lag_v2.lagged_root.cache_clear()
