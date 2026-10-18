"""feat_price_v2: valid-session windows, corporate-action masks, limit-hit ratio."""

from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta

import duckdb

from modeler.etl.features.price import build_price_sql
from modeler.etl.features.price_v2 import (
    PRICE_V2_COLUMNS,
    PRICE_V2_TABLE,
    build_price_v2_sql,
)

# sha256 of build_price_sql() output at main aaa0f54 (default / with quality_view="q").
_LEGACY_SQL_SHA = "6886ddd45d33c6a13f0d1fd1653a8dd37518bfaa5189ae1227f853bb3b2a4bf3"
_LEGACY_SQL_Q_SHA = "9584c83551f58410dd86e8d5ee7f62e072373482e50cebfb2a131a465ea2fa0a"


def _days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _con(rows: list[tuple], ca_dates: tuple[date, ...] = ()) -> duckdb.DuckDBPyConnection:
    """rows: (date, open, high, low, close, volume)."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE ohlcv (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "open BIGINT, high BIGINT, low BIGINT, close BIGINT, volume BIGINT)"
    )
    con.executemany("INSERT INTO ohlcv VALUES (?, 'A', 'KOSPI', ?, ?, ?, ?, ?)", rows)
    con.execute(
        "CREATE TABLE qual (trade_date DATE, ticker VARCHAR, market VARCHAR, ca_event BOOLEAN)"
    )
    for d in ca_dates:
        con.execute("INSERT INTO qual VALUES (?, 'A', 'KOSPI', TRUE)", [d])
    return con


def _run(con, with_quality: bool = False) -> dict[date, dict]:
    sql = build_price_v2_sql("ohlcv", quality_view="qual" if with_quality else None)
    cur = con.execute(sql)
    names = [c[0] for c in cur.description]
    return {r[0]: dict(zip(names, r, strict=True)) for r in cur.fetchall()}


def _rising(n: int, start: date = date(2020, 1, 6)) -> list[tuple]:
    rows = []
    for i, d in enumerate(_days(start, n)):
        c = 1000 + 10 * i
        rows.append((d, c, c + 5, c - 5, c, 100))
    return rows


def test_columns_and_keys() -> None:
    res = _run(_con(_rising(5)))
    row = next(iter(res.values()))
    assert list(row) == ["trade_date", "ticker", "market", *PRICE_V2_COLUMNS]
    assert PRICE_V2_TABLE == "feat_price_v2"


def test_ret_60_20_positive_and_equals_hand_value() -> None:
    rows = _rising(80)
    days = [r[0] for r in rows]
    res = _run(_con(rows))
    t = 79
    want = math.log(rows[t - 20][4] / rows[t - 60][4])
    got = res[days[t]]["px_ret_60_20_v2"]
    assert got > 0
    assert math.isclose(got, want, rel_tol=1e-12)
    # Not enough sessions: index 59 has no t-60.
    assert res[days[59]]["px_ret_60_20_v2"] is None
    assert res[days[60]]["px_ret_60_20_v2"] is not None


def test_simple_returns_and_vol_hand_values() -> None:
    rows = _rising(70)
    days = [r[0] for r in rows]
    res = _run(_con(rows))
    t = 65
    assert math.isclose(res[days[t]]["px_ret_5d_v2"], math.log(rows[t][4] / rows[t - 5][4]))
    assert math.isclose(res[days[t]]["px_ret_20d_v2"], math.log(rows[t][4] / rows[t - 20][4]))
    assert math.isclose(res[days[t]]["px_ret_60d_v2"], math.log(rows[t][4] / rows[t - 60][4]))
    rets = [math.log(rows[i][4] / rows[i - 1][4]) for i in range(t - 19, t + 1)]
    mean = sum(rets) / 20
    sd = math.sqrt(sum((x - mean) ** 2 for x in rets) / 19)
    assert math.isclose(res[days[t]]["px_vol_20d_v2"], sd, rel_tol=1e-9)
    ma = sum(rows[i][4] for i in range(t - 19, t + 1)) / 20
    assert math.isclose(res[days[t]]["px_gap_vs_ma20_v2"], rows[t][4] / ma - 1)
    assert res[days[t]]["px_dist_52w_high_v2"] is None  # fewer than 252 sessions
    assert res[days[19]]["px_vol_20d_v2"] is None  # needs 20 returns
    assert res[days[20]]["px_vol_20d_v2"] is not None


def test_ret_1d_equals_hand_value() -> None:
    rows = _rising(10)
    days = [r[0] for r in rows]
    res = _run(_con(rows))
    t = 5
    want = math.log(rows[t][4] / rows[t - 1][4])
    got = res[days[t]]["px_ret_1d_v2"]
    assert got > 0
    assert math.isclose(got, want, rel_tol=1e-12)
    assert res[days[0]]["px_ret_1d_v2"] is None  # no previous session


def test_ret_1d_is_null_only_on_the_ca_event_day() -> None:
    rows = _rising(10)
    days = [r[0] for r in rows]
    res = _run(_con(rows, ca_dates=(days[5],)), with_quality=True)
    assert res[days[5]]["px_ret_1d_v2"] is None  # the corporate-action jump itself
    for t in (4, 6):  # neighbours keep their clean returns
        want = math.log(rows[t][4] / rows[t - 1][4])
        assert math.isclose(res[days[t]]["px_ret_1d_v2"], want, rel_tol=1e-12)
    # Without a quality view nothing is masked.
    res_nq = _run(_con(rows, ca_dates=(days[5],)))
    assert res_nq[days[5]]["px_ret_1d_v2"] is not None


def test_ret_1d_halt_row_is_null_and_next_day_skips_it() -> None:
    rows = _rising(10)
    halt_day = rows[4][0]
    rows[4] = (halt_day, 0, 0, 0, rows[3][4], 0)
    days = [r[0] for r in rows]
    res = _run(_con(rows))
    assert res[halt_day]["px_ret_1d_v2"] is None  # halt row carries no value
    # The day after a halt measures against the last valid session (t-2 here).
    want = math.log(rows[5][4] / rows[3][4])
    assert math.isclose(res[days[5]]["px_ret_1d_v2"], want, rel_tol=1e-12)


def test_halt_row_does_not_zero_the_range_min() -> None:
    rows = _rising(40)
    halt_day = rows[30][0]
    rows[30] = (halt_day, 0, 0, 0, rows[29][4], 0)
    days = [r[0] for r in rows]
    res = _run(_con(rows))
    t = 35
    window = [r for r in rows[: t + 1] if r[3] > 0][-20:]
    want = (max(r[2] for r in window) - min(r[3] for r in window)) / rows[t][4]
    got = res[days[t]]["px_high_low_range_20d_v2"]
    assert got is not None and math.isclose(got, want)
    assert res[halt_day]["px_high_low_range_20d_v2"] is None  # halt row itself carries no value


def test_fewer_than_20_valid_sessions_is_null() -> None:
    rows = _rising(24)
    # Four halts leave 20 valid sessions at the last day, but only 19 returns.
    for i in (5, 6, 7, 8):
        rows[i] = (rows[i][0], 0, 0, 0, rows[i][4], 0)
    days = [r[0] for r in rows]
    res = _run(_con(rows))
    last = res[days[-1]]
    assert last["px_high_low_range_20d_v2"] is not None
    assert last["px_vol_20d_v2"] is None
    assert last["px_amihud_20d_v2"] is None
    assert last["px_limit_hit_ratio_20d_v2"] is None
    # Plain 19-session history: range also NULL.
    res2 = _run(_con(_rising(19)))
    assert all(r["px_high_low_range_20d_v2"] is None for r in res2.values())


def test_ca_event_in_window_nulls_all_columns() -> None:
    rows = _rising(300)
    days = [r[0] for r in rows]
    ca = days[200]
    res = _run(_con(rows, ca_dates=(ca,)), with_quality=True)
    t = 205  # CA 5 sessions back: inside every window (the 1-day return has no window).
    assert all(res[days[t]][c] is None for c in PRICE_V2_COLUMNS if c != "px_ret_1d_v2")
    assert res[days[t]]["px_ret_1d_v2"] is not None
    t = 299  # CA 99 sessions back: only the 252-session window still sees it.
    assert res[days[t]]["px_ret_60d_v2"] is not None
    assert res[days[t]]["px_vol_60d_v2"] is not None
    assert res[days[t]]["px_dist_52w_high_v2"] is None
    # Without a quality view nothing is masked.
    res_nq = _run(_con(rows, ca_dates=(ca,)))
    assert res_nq[days[205]]["px_ret_5d_v2"] is not None


def test_limit_hit_ratio_threshold_changes_on_2015_06_15() -> None:
    def build(start: date) -> dict[date, dict]:
        days = _days(start, 25)
        rows, c = [], 1000
        for i, d in enumerate(days):
            if i == 22:
                c = int(c * 1.2)  # +20%: hits the 15% limit, not the 30% one
            elif i == 23:
                c = int(c * 1.3)  # +30%: hits both
            rows.append((d, c, c, c, c, 100))
        res = _run(_con(rows))
        return {days[24]: res[days[24]]}

    after = build(date(2016, 1, 4))
    before = build(date(2015, 3, 2))
    assert math.isclose(next(iter(after.values()))["px_limit_hit_ratio_20d_v2"], 1 / 20)
    assert math.isclose(next(iter(before.values()))["px_limit_hit_ratio_20d_v2"], 2 / 20)


def test_limit_threshold_is_limit_minus_one_point() -> None:
    days = _days(date(2016, 1, 4), 22)
    rows, c = [], 1000
    for i, d in enumerate(days):
        if i == 10:
            c = 1290  # +29%: counts
        elif i == 11:
            c = int(c * 1.285)  # +28.5%: does not
        rows.append((d, c, c, c, c, 100))
    res = _run(_con(rows))
    assert math.isclose(res[days[21]]["px_limit_hit_ratio_20d_v2"], 1 / 20)


def test_importing_price_v2_leaves_legacy_sql_unchanged() -> None:
    assert hashlib.sha256(build_price_sql().encode()).hexdigest() == _LEGACY_SQL_SHA
    sql_q = build_price_sql(quality_view="q")
    assert hashlib.sha256(sql_q.encode()).hexdigest() == _LEGACY_SQL_Q_SHA
    assert "_v2" not in build_price_sql()
