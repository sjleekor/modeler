"""``feat_flow_v2`` (15_quality_fix_plan §4.2-§4.3, §10.7; 17_c2_feature_set §8) and the frozen
``feat_flow`` text."""

from __future__ import annotations

import datetime as dt
import hashlib
import statistics

import duckdb
import pytest

from modeler.etl.features.flow import build_flow_sql
from modeler.etl.features.flow_v2 import FLOW_V2_COLUMNS, build_flow_v2_sql

FLOAT = 1000.0
START = dt.date(2019, 1, 2)  # inside the 'allowed' short regime

Krx = dict[dt.date, tuple[float | None, float | None]]  # date -> (raw volume, listed shares)
KrxRow = tuple[dt.date, float | None, float | None, str]  # date, volume, listed, market


def _sql_num(value: float | None) -> str:
    return "NULL" if value is None else repr(float(value))


def _krx_view(con, krx: list[KrxRow], ticker: str) -> None:
    """``daily_market_cap`` with only the columns the mart reads; a date absent from ``krx``
    has no row at all, a ``None`` is a NULL column value."""
    if krx:
        body = ",".join(
            f"(DATE '{d}', '{ticker}', '{market}', CAST({_sql_num(v)} AS BIGINT),"
            f" CAST({_sql_num(ls)} AS BIGINT))"
            for d, v, ls, market in krx
        )
    else:  # typed but empty
        body = "(DATE '2000-01-01', 'X', 'KOSPI', CAST(1 AS BIGINT), CAST(1 AS BIGINT))"
    where = "" if krx else " WHERE FALSE"
    con.execute(
        f"CREATE VIEW daily_market_cap AS SELECT * FROM (VALUES {body})"
        f" t(trade_date,ticker,market,volume,listed_shares){where}"
    )


def _con(
    rows: list[tuple[dt.date, dict[str, float | None]]],
    ticker: str = "A",
    krx: Krx | None = None,
    krx_rows: list[KrxRow] | None = None,
):
    """One ticker, one flow/pit/quality row per entry; a ``None`` value is a missing metric.

    Flow, pit and quality rows are all ``KOSPI`` (the ticker's current market). ``krx`` gives
    one ``KOSPI`` ``daily_market_cap`` row per date; ``krx_rows`` gives the rows verbatim, with
    their own market, and wins when both are passed.
    """
    con = duckdb.connect()
    flow = []
    for d, metrics in rows:
        for code, value in metrics.items():
            if value is not None:
                flow.append(f"(DATE '{d}', '{ticker}', 'KOSPI', '{code}', {value}, 'KRX')")
    con.execute(
        "CREATE VIEW krx_security_flow_raw AS SELECT * FROM (VALUES "
        + ",".join(flow)
        + ") t(trade_date,ticker,market,metric_code,value,source)"
    )
    pit = ",".join(f"(DATE '{d}', '{ticker}', 'KOSPI', {FLOAT})" for d, _ in rows)
    con.execute(
        "CREATE VIEW dim_stock_pit_daily AS SELECT * FROM (VALUES "
        + pit
        + ") t(trade_date,ticker,market,float_shares_pit)"
    )
    qual = ",".join(f"(DATE '{d}', '{ticker}', 'KOSPI', {i + 1})" for i, (d, _) in enumerate(rows))
    con.execute(
        "CREATE VIEW dim_price_quality_daily AS SELECT * FROM (VALUES "
        + qual
        + ") t(trade_date,ticker,market,valid_session_idx)"
    )
    if krx_rows is None:
        krx_rows = [(d, v, ls, "KOSPI") for d, (v, ls) in (krx or {}).items()]
    _krx_view(con, krx_rows, ticker)
    con.execute(f"CREATE VIEW feat_flow_v2 AS {build_flow_v2_sql()}")
    return con


LEGACY_V2_COLUMNS: tuple[str, ...] = (
    "flow_foreign_netbuy_to_float_20d_v2",
    "flow_inst_netbuy_to_float_20d_v2",
    "flow_indiv_netbuy_to_float_20d_v2",
    "flow_foreign_netbuy_z_20d_v2",
    "flow_inst_netbuy_z_20d_v2",
    "flow_short_avg_price_v2",
    "flow_short_selling_volume_v2",
    "flow_short_selling_value_v2",
    "flow_foreign_netbuy_to_float_5d_v2",
    "flow_inst_netbuy_to_float_5d_v2",
    "flow_indiv_netbuy_to_float_5d_v2",
)


def _series(con, column: str) -> list[float | None]:
    return [
        r[0]
        for r in con.execute(f"SELECT {column} FROM feat_flow_v2 ORDER BY trade_date").fetchall()
    ]


def _days(n: int) -> list[dt.date]:
    return [START + dt.timedelta(days=i) for i in range(n)]


def test_netbuy_to_float_needs_a_full_window() -> None:
    days = _days(25)
    rows = [
        (
            d,
            {
                "foreign_net_buy_volume": 10.0,
                "institution_net_buy_volume": 5.0,
                "individual_net_buy_volume": -15.0,
            },
        )
        for d in days
    ]
    con = _con(rows)
    for col, per_day in (
        ("flow_foreign_netbuy_to_float_20d_v2", 10.0),
        ("flow_inst_netbuy_to_float_20d_v2", 5.0),
        ("flow_indiv_netbuy_to_float_20d_v2", -15.0),
    ):
        s = _series(con, col)
        assert all(v is None for v in s[:19])
        assert s[19] == pytest.approx(per_day * 20 / FLOAT)
        assert s[24] == pytest.approx(per_day * 20 / FLOAT)


def test_netbuy_to_float_is_null_when_any_window_row_is_missing() -> None:
    days = _days(30)
    rows = [
        (d, {"foreign_net_buy_volume": None if i == 21 else 1.0, "institution_net_buy_volume": 1.0})
        for i, d in enumerate(days)
    ]
    s = _series(_con(rows), "flow_foreign_netbuy_to_float_20d_v2")
    assert s[19] == pytest.approx(20 / FLOAT)  # window 0..19, full
    assert s[20] == pytest.approx(20 / FLOAT)  # window 1..20, full
    assert all(v is None for v in s[21:41])  # windows 2..21 .. contain row 21
    assert s[29] is None


def test_netbuy_to_float_5d_needs_a_full_window() -> None:
    days = _days(8)
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    rows = [
        (
            d,
            {
                "foreign_net_buy_volume": v,
                "institution_net_buy_volume": v * 2,
                "individual_net_buy_volume": -v * 3,
            },
        )
        for d, v in zip(days, vals, strict=True)
    ]
    con = _con(rows)
    for col, scale in (
        ("flow_foreign_netbuy_to_float_5d_v2", 1),
        ("flow_inst_netbuy_to_float_5d_v2", 2),
        ("flow_indiv_netbuy_to_float_5d_v2", -3),
    ):
        s = _series(con, col)
        assert all(v is None for v in s[:4])  # fewer than 5 rows
        assert s[4] == pytest.approx(scale * sum(vals[0:5]) / FLOAT)  # rows 0..4
        assert s[7] == pytest.approx(scale * sum(vals[3:8]) / FLOAT)  # rows 3..7


def test_netbuy_to_float_5d_is_null_when_any_window_row_is_missing() -> None:
    days = _days(12)
    rows = [
        (d, {"foreign_net_buy_volume": None if i == 6 else 1.0, "institution_net_buy_volume": 1.0})
        for i, d in enumerate(days)
    ]
    s = _series(_con(rows), "flow_foreign_netbuy_to_float_5d_v2")
    assert s[4] == pytest.approx(5 / FLOAT)  # rows 0..4, full
    assert s[5] == pytest.approx(5 / FLOAT)  # rows 1..5, full
    assert all(v is None for v in s[6:11])  # windows ending 6..10 contain row 6
    assert s[11] == pytest.approx(5 / FLOAT)  # rows 7..11, full again


def test_z_is_null_until_twenty_observations() -> None:
    days = _days(22)
    vals = [float((i * 7) % 11) for i in range(22)]
    rows = [
        (d, {"foreign_net_buy_volume": v, "institution_net_buy_volume": v * 2})
        for d, v in zip(days, vals, strict=True)
    ]
    con = _con(rows)
    for col, scale in (("flow_foreign_netbuy_z_20d_v2", 1), ("flow_inst_netbuy_z_20d_v2", 2)):
        s = _series(con, col)
        assert all(v is None for v in s[:19])
        w = [v * scale for v in vals[:20]]
        expected = (w[-1] - statistics.mean(w)) / statistics.stdev(w)
        assert s[19] == pytest.approx(expected)
        w2 = [v * scale for v in vals[2:22]]
        assert s[21] == pytest.approx((w2[-1] - statistics.mean(w2)) / statistics.stdev(w2))


def test_z_is_null_with_a_missing_observation_in_the_window() -> None:
    days = _days(25)
    rows = [
        (
            d,
            {
                "foreign_net_buy_volume": None if i == 10 else float(i % 5),
                "institution_net_buy_volume": 1.0,
            },
        )
        for i, d in enumerate(days)
    ]
    s = _series(_con(rows), "flow_foreign_netbuy_z_20d_v2")
    assert all(v is None for v in s[:20])  # windows 1..20 etc. contain row 10 up to idx 29
    assert s[24] is None


# --- C2-7: individual net buy over the KRX raw volume ---------------------------

VOLUME_COLS = (
    ("flow_individual_netbuy_to_volume_5d_v2", 5),
    ("flow_individual_netbuy_to_volume_20d_v2", 20),
)


def _volume_fixture(n: int = 25):
    """Hand-checkable series: net buy ``-(i + 1)``, raw volume ``100 + 10 * i``."""
    days = _days(n)
    net = [-(i + 1.0) for i in range(n)]
    vol = [100.0 + 10 * i for i in range(n)]
    rows = [(d, {"individual_net_buy_volume": v}) for d, v in zip(days, net, strict=True)]
    krx: Krx = {d: (v, 5000.0) for d, v in zip(days, vol, strict=True)}
    return days, net, vol, rows, krx


def test_netbuy_to_volume_hand_values_and_full_window() -> None:
    _, net, vol, rows, krx = _volume_fixture()
    con = _con(rows, krx=krx)
    for col, w in VOLUME_COLS:
        s = _series(con, col)
        assert all(v is None for v in s[: w - 1])  # window not full yet
        for end in (w - 1, 24):
            lo = end - w + 1
            assert s[end] == pytest.approx(sum(net[lo : end + 1]) / sum(vol[lo : end + 1]))
    # spot check one value by hand: rows 0..4 -> -(1+2+3+4+5) / (100+110+120+130+140)
    assert _series(con, "flow_individual_netbuy_to_volume_5d_v2")[4] == pytest.approx(-15 / 600)


def test_netbuy_to_volume_is_null_when_raw_volume_is_missing_in_the_window() -> None:
    days, net, vol, rows, krx = _volume_fixture(12)
    krx[days[6]] = (None, 5000.0)  # NULL volume on row 6
    s = _series(_con(rows, krx=krx), "flow_individual_netbuy_to_volume_5d_v2")
    assert s[4] == pytest.approx(sum(net[0:5]) / sum(vol[0:5]))
    assert s[5] == pytest.approx(sum(net[1:6]) / sum(vol[1:6]))
    assert all(v is None for v in s[6:11])  # windows ending 6..10 contain row 6
    assert s[11] == pytest.approx(sum(net[7:12]) / sum(vol[7:12]))
    # a ticker-day with no daily_market_cap row at all behaves the same
    days, net, vol, rows, krx = _volume_fixture(12)
    del krx[days[6]]
    s = _series(_con(rows, krx=krx), "flow_individual_netbuy_to_volume_5d_v2")
    assert all(v is None for v in s[6:11])
    assert s[11] == pytest.approx(sum(net[7:12]) / sum(vol[7:12]))


def test_netbuy_to_volume_is_null_when_net_buy_is_missing_in_the_window() -> None:
    days, net, vol, rows, krx = _volume_fixture(12)
    # keep another metric so the flow row exists (a row with no metric at all is not a row)
    rows[6] = (days[6], {"individual_net_buy_volume": None, "foreign_net_buy_volume": 1.0})
    s = _series(_con(rows, krx=krx), "flow_individual_netbuy_to_volume_5d_v2")
    assert s[5] is not None
    assert all(v is None for v in s[6:11])
    assert s[11] == pytest.approx(sum(net[7:12]) / sum(vol[7:12]))


def test_netbuy_to_volume_is_null_when_the_volume_sum_is_not_positive() -> None:
    days, _, _, rows, _ = _volume_fixture(6)
    krx: Krx = {d: (0.0, 5000.0) for d in days}
    s = _series(_con(rows, krx=krx), "flow_individual_netbuy_to_volume_5d_v2")
    assert all(v is None for v in s)


# --- C2-8: net buy over the same-day KRX listed shares ---------------------------

LISTED_COLS = (
    ("flow_foreign_netbuy_to_listed_5d_v2", "foreign_net_buy_volume", 5),
    ("flow_foreign_netbuy_to_listed_20d_v2", "foreign_net_buy_volume", 20),
    ("flow_inst_netbuy_to_listed_5d_v2", "institution_net_buy_volume", 5),
    ("flow_inst_netbuy_to_listed_20d_v2", "institution_net_buy_volume", 20),
    ("flow_indiv_netbuy_to_listed_5d_v2", "individual_net_buy_volume", 5),
    ("flow_indiv_netbuy_to_listed_20d_v2", "individual_net_buy_volume", 20),
)


def _listed_fixture(n: int = 25):
    """Three investor series that differ, and listed shares that change from day to day."""
    days = _days(n)
    metrics = [
        {
            "foreign_net_buy_volume": 1.0 + i,
            "institution_net_buy_volume": 2.0 * i - 3,
            "individual_net_buy_volume": -(3.0 * i - 2),
        }
        for i in range(n)
    ]
    listed = [10_000.0 + 100 * i for i in range(n)]
    rows = list(zip(days, metrics, strict=True))
    krx: Krx = {d: (500.0, ls) for d, ls in zip(days, listed, strict=True)}
    return days, metrics, listed, rows, krx


def test_netbuy_to_listed_hand_values_use_the_same_day_listed_shares() -> None:
    _, metrics, listed, rows, krx = _listed_fixture()
    con = _con(rows, krx=krx)
    for col, code, w in LISTED_COLS:
        s = _series(con, col)
        assert all(v is None for v in s[: w - 1])
        for end in (w - 1, 24):
            lo = end - w + 1
            num = sum(m[code] for m in metrics[lo : end + 1])
            assert s[end] == pytest.approx(num / listed[end]), (col, end)
    # one by hand: foreign 5d at row 4 = (1+2+3+4+5) / 10_400
    assert _series(con, "flow_foreign_netbuy_to_listed_5d_v2")[4] == pytest.approx(15 / 10_400)


def test_netbuy_to_listed_is_null_when_the_window_is_not_full() -> None:
    days, metrics, _, rows, krx = _listed_fixture(12)
    rows[6] = (days[6], {**metrics[6], "foreign_net_buy_volume": None})
    con = _con(rows, krx=krx)
    s5 = _series(con, "flow_foreign_netbuy_to_listed_5d_v2")
    assert s5[5] is not None
    assert all(v is None for v in s5[6:11])
    assert s5[11] is not None
    # the other investors' columns are not affected by foreign's hole
    assert _series(con, "flow_inst_netbuy_to_listed_5d_v2")[8] is not None


def test_netbuy_to_listed_is_null_only_on_a_day_without_listed_shares() -> None:
    days, _, _, rows, krx = _listed_fixture(12)
    krx[days[8]] = (500.0, None)  # NULL listed shares on row 8
    del krx[days[9]]  # no daily_market_cap row on row 9
    krx[days[10]] = (500.0, 0.0)  # zero listed shares on row 10
    con = _con(rows, krx=krx)
    for col, _, w in LISTED_COLS:
        if w != 5:
            continue
        s = _series(con, col)
        assert s[7] is not None and s[11] is not None
        assert s[8] is None and s[9] is None and s[10] is None  # same-day denominator only


def test_missing_krx_rows_null_the_new_columns_but_not_the_old_ones() -> None:
    _, _, _, rows, _ = _listed_fixture(25)
    with_krx = _con(rows, krx=_listed_fixture(25)[4])
    without = _con(rows, krx=None)
    for col in FLOW_V2_COLUMNS[11:]:
        assert all(v is None for v in _series(without, col)), col
    assert any(v is not None for v in _series(with_krx, "flow_foreign_netbuy_to_listed_20d_v2"))
    old = ", ".join(LEGACY_V2_COLUMNS)
    q = f"SELECT trade_date, {old} FROM feat_flow_v2 ORDER BY trade_date"
    assert with_krx.execute(q).fetchall() == without.execute(q).fetchall()


def test_existing_columns_are_unchanged_by_the_new_source() -> None:
    """The six ``to_float`` columns keep their values and denominator (float shares PIT)."""
    days = _days(25)
    rows = [
        (
            d,
            {
                "foreign_net_buy_volume": 10.0,
                "institution_net_buy_volume": 5.0,
                "individual_net_buy_volume": -15.0,
            },
        )
        for d in days
    ]
    krx: Krx = {d: (777.0, 31_000.0) for d in days}  # unrelated to FLOAT
    con = _con(rows, krx=krx)
    for col, per_day in (
        ("flow_foreign_netbuy_to_float_20d_v2", 10.0),
        ("flow_inst_netbuy_to_float_20d_v2", 5.0),
        ("flow_indiv_netbuy_to_float_20d_v2", -15.0),
    ):
        assert _series(con, col)[24] == pytest.approx(per_day * 20 / FLOAT)
    for col, per_day in (
        ("flow_foreign_netbuy_to_float_5d_v2", 10.0),
        ("flow_inst_netbuy_to_float_5d_v2", 5.0),
        ("flow_indiv_netbuy_to_float_5d_v2", -15.0),
    ):
        assert _series(con, col)[24] == pytest.approx(per_day * 5 / FLOAT)


def test_reverse_split_raw_ratio_is_stable_while_adjusted_ratio_inflates() -> None:
    """15 §10.7: a later 5:1 reverse split rewrites Naver volume to 1/5 of the raw count.

    ``feat_flow``'s ``flow_individual_netbuy_to_volume_20d`` divides raw net buy by that
    adjusted volume and comes out 5x too large; the ``_v2`` column divides by KRX raw volume
    and keeps the value the day itself would have shown.
    """
    n, split = 25, 5.0
    days = _days(n)
    raw_net, raw_vol = -200.0, 1000.0
    rows = [(d, {"individual_net_buy_volume": raw_net}) for d in days]
    krx: Krx = {d: (raw_vol, 50_000.0) for d in days}
    con = _con(rows, krx=krx)
    v2_5 = _series(con, "flow_individual_netbuy_to_volume_5d_v2")[24]
    v2_20 = _series(con, "flow_individual_netbuy_to_volume_20d_v2")[24]
    assert v2_5 == pytest.approx(raw_net / raw_vol)  # -0.2
    assert v2_20 == pytest.approx(raw_net / raw_vol)

    # the legacy mart, fed the adjusted volume daily_ohlcv would carry after the split
    adjusted_vol = raw_vol / split
    flow = ",".join(
        f"(DATE '{d}', 'A', 'KOSPI', 'individual_net_buy_volume', {raw_net}, 'KRX')" for d in days
    )
    legacy = duckdb.connect()
    legacy.execute(
        "CREATE VIEW krx_security_flow_raw AS SELECT * FROM (VALUES "
        + flow
        + ") t(trade_date,ticker,market,metric_code,value,source)"
    )
    legacy.execute(
        "CREATE VIEW p AS SELECT * FROM (VALUES "
        + ",".join(f"(DATE '{d}', 'A', 'KOSPI', 10, 11, 9, {adjusted_vol})" for d in days)
        + ") t(trade_date,ticker,market,open,high,low,volume)"
    )
    legacy.execute(
        "CREATE VIEW s AS SELECT * FROM (VALUES "
        + ",".join(f"(DATE '{d}', 'A', 'KOSPI', {FLOAT})" for d in days)
        + ") t(trade_date,ticker,market,float_shares_pit)"
    )
    legacy.execute(
        "CREATE VIEW q AS SELECT * FROM (VALUES "
        + ",".join(
            f"(DATE '{d}', 'A', 'KOSPI', {i + 1}, TRUE, 'allowed')" for i, d in enumerate(days)
        )
        + ") t(trade_date,ticker,market,valid_session_idx,short_balance_is_available,short_regime)"
    )
    legacy.execute(
        f"CREATE VIEW feat_flow AS {build_flow_sql(price_view='p', pit_view='s', quality_view='q')}"
    )
    old_5 = legacy.execute(
        "SELECT flow_individual_netbuy_to_volume_5d FROM feat_flow ORDER BY trade_date"
    ).fetchall()[24][0]
    old_20 = legacy.execute(
        "SELECT flow_individual_netbuy_to_volume_20d FROM feat_flow ORDER BY trade_date"
    ).fetchall()[24][0]
    assert old_5 == pytest.approx(split * v2_5)  # -1.0, five times the raw ratio
    assert old_20 == pytest.approx(split * v2_20)


def test_a_market_transfer_ticker_keeps_its_krx_rows_before_the_transfer() -> None:
    """daily_market_cap carries the market of the day; flow/ohlcv carry the current market.

    A KOSDAQ -> KOSPI transfer after row 11: flow rows say KOSPI for all history, the KRX rows
    say KOSDAQ until the transfer. Joining on market would drop every KRX row before it.
    """
    days, net, vol, rows, _ = _volume_fixture(25)
    listed = [20_000.0 + 50 * i for i in range(25)]
    krx_rows: list[KrxRow] = [
        (d, v, ls, "KOSDAQ" if i < 12 else "KOSPI")
        for i, (d, v, ls) in enumerate(zip(days, vol, listed, strict=True))
    ]
    con = _con(rows, krx_rows=krx_rows)

    to_volume = _series(con, "flow_individual_netbuy_to_volume_5d_v2")
    assert to_volume[4] == pytest.approx(sum(net[0:5]) / sum(vol[0:5]))  # before the transfer
    assert to_volume[11] == pytest.approx(sum(net[7:12]) / sum(vol[7:12]))  # last KOSDAQ row
    assert to_volume[14] == pytest.approx(sum(net[10:15]) / sum(vol[10:15]))  # window spans it
    assert all(v is not None for v in to_volume[4:])

    to_volume_20 = _series(con, "flow_individual_netbuy_to_volume_20d_v2")
    assert to_volume_20[19] == pytest.approx(sum(net[0:20]) / sum(vol[0:20]))

    # to_listed uses the same-day listed shares, KOSDAQ days included
    for col in (
        "flow_indiv_netbuy_to_listed_5d_v2",
        "flow_indiv_netbuy_to_listed_20d_v2",
    ):
        s = _series(con, col)
        w = 5 if "5d" in col else 20
        assert all(v is not None for v in s[w - 1 :]), col
        assert s[w - 1] == pytest.approx(sum(net[0:w]) / listed[w - 1])
    # no extra rows from the market-free join
    assert con.execute("SELECT COUNT(*) FROM feat_flow_v2").fetchone()[0] == 25


def test_duplicate_krx_rows_for_one_ticker_day_resolve_deterministically() -> None:
    """Two markets for one ticker-day: take the flow row's market, else a fixed order."""
    days, net, vol, rows, _ = _volume_fixture(8)
    same, other = 1000.0, 3.0  # volume of the KOSPI row vs a stray KOSDAQ row
    krx_rows: list[KrxRow] = []
    for d in days:
        krx_rows.append((d, other, other, "KOSDAQ"))  # listed first on purpose
        krx_rows.append((d, same, 9_000.0, "KOSPI"))
    con = _con(rows, krx_rows=krx_rows)
    assert con.execute("SELECT COUNT(*) FROM feat_flow_v2").fetchone()[0] == 8  # no fan-out
    s = _series(con, "flow_individual_netbuy_to_volume_5d_v2")
    assert s[4] == pytest.approx(sum(net[0:5]) / (5 * same))  # the KOSPI rows, not the stray ones
    s = _series(con, "flow_indiv_netbuy_to_listed_5d_v2")
    assert s[4] == pytest.approx(sum(net[0:5]) / 9_000.0)

    # neither duplicate matches the flow market: the same one wins whatever the row order
    krx_rows = []
    for d in days:
        krx_rows.append((d, 11.0, 7_000.0, "KOSDAQ"))
        krx_rows.append((d, 22.0, 8_000.0, "KONEX"))
    forward = _series(_con(rows, krx_rows=krx_rows), "flow_indiv_netbuy_to_listed_5d_v2")
    backward = _series(
        _con(rows, krx_rows=list(reversed(krx_rows))), "flow_indiv_netbuy_to_listed_5d_v2"
    )
    assert forward == backward
    assert forward[4] == pytest.approx(sum(net[0:5]) / 8_000.0)  # KONEX < KOSDAQ


SHORT_COLS = (
    "flow_short_avg_price_v2",
    "flow_short_selling_volume_v2",
    "flow_short_selling_value_v2",
)


@pytest.mark.parametrize(
    ("day", "allowed"),
    [
        ("2024-01-02", False),  # banned (2023-11-06 .. 2025-03-28)
        ("2022-01-03", False),  # partial
        ("2013-01-02", False),  # unknown (before 2014-06-01)
        ("2020-04-01", False),  # banned (2020-03-16 .. 2021-05-02)
        ("2019-01-02", True),  # allowed
        ("2025-04-01", True),  # allowed again
    ],
)
def test_short_columns_follow_the_regime(day: str, allowed: bool) -> None:
    d = dt.date.fromisoformat(day)
    con = _con([(d, {"short_selling_volume": 4.0, "short_selling_value": 400.0})])
    row = con.execute(f"SELECT {', '.join(SHORT_COLS)} FROM feat_flow_v2").fetchone()
    if allowed:
        assert row == (100.0, 4.0, 400.0)
    else:
        assert row == (None, None, None)


def test_columns_are_exactly_the_planned_set() -> None:
    con = _con([(START, {"foreign_net_buy_volume": 1.0})])
    cols = [r[0] for r in con.execute("DESCRIBE feat_flow_v2").fetchall()]
    assert cols == ["trade_date", "ticker", "market", *FLOW_V2_COLUMNS]
    assert len(FLOW_V2_COLUMNS) == 19
    assert len(set(FLOW_V2_COLUMNS)) == 19


def test_new_columns_are_appended_after_the_original_eleven() -> None:
    assert FLOW_V2_COLUMNS[:11] == LEGACY_V2_COLUMNS
    assert FLOW_V2_COLUMNS[11:] == (
        "flow_individual_netbuy_to_volume_5d_v2",
        "flow_individual_netbuy_to_volume_20d_v2",
        "flow_foreign_netbuy_to_listed_5d_v2",
        "flow_foreign_netbuy_to_listed_20d_v2",
        "flow_inst_netbuy_to_listed_5d_v2",
        "flow_inst_netbuy_to_listed_20d_v2",
        "flow_indiv_netbuy_to_listed_5d_v2",
        "flow_indiv_netbuy_to_listed_20d_v2",
    )


def test_feat_flow_sql_text_is_unchanged() -> None:
    """Hashes of ``build_flow_sql`` as emitted at ``aaa0f54``."""

    def _h(sql: str) -> str:
        return hashlib.sha256(sql.encode()).hexdigest()

    assert _h(build_flow_sql()) == (
        "e6712309bd22ef4a87a3aff4b5dc900fa5a60cf5fad03bb43ec9ac3f4894a5b4"
    )
    assert _h(build_flow_sql(price_view="p", pit_view="s", quality_view="q")) == (
        "4d7f691269471ea72b3fee2fdab2e3fba163e3c91f2c4fbb50c38c96056bc92e"
    )
