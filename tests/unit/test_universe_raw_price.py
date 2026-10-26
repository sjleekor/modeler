"""Universe with the price floor on the KRX raw close (``etl/universe_raw_price.py``, C1-4a).

Synthetic ``daily_ohlcv`` (adjusted) and ``daily_market_cap`` (KRX raw) tables:
a later reverse split, a later split, rows with no KRX raw price, a market
transfer, and a halt. No lake data.
"""

from __future__ import annotations

import datetime
import hashlib

import duckdb
import pytest

from modeler.etl.universe import (
    UniverseFilter,
    broad_universe_filter,
    build_universe_sql,
    tradable_universe_filter,
)
from modeler.etl.universe_raw_price import (
    assert_raw_price_inputs_unique,
    build_raw_price_view_sql,
    build_universe_raw_price_sql,
)
from modeler.etl.universe_terminal import build_universe_v2_sql

# build_universe_sql(...) as of the commit before universe_raw_price.py existed.
_SQL_SHA256 = {
    "default": "cb5b7612f00bc1d520cef818815324fb6cfc643737d20eea2a8eceadc689b279",
    "tradable": "7ada71ce589b45f517adf051fe7f0acdadb0de03ee56c0200536d319cf46fedf",
    "broad": "37aff3145f241d03b70361854d4d2806c1d680ae143a15326dc3d8d0524ec8be",
}

N_DAYS = 5
# Small windows so five rows are enough: any non-halt row warms up, the average
# turnover is over the last two rows, floors are the production ones.
FLT = UniverseFilter(
    warmup_window=3,
    warmup_min_valid=1,
    liquidity_window=2,
    min_liquidity_krw=1e8,
    min_close_krw=1000.0,
    label_horizon=1,
    apply_liquidity_filter=True,
)


def _d(i: int) -> datetime.date:
    return datetime.date(2020, 1, 1) + datetime.timedelta(days=i)


def _con(ohlcv: list[tuple], market_cap: list[tuple]) -> duckdb.DuckDBPyConnection:
    """``ohlcv``: (date, ticker, market, o, h, l, close, volume);
    ``market_cap``: (date, ticker, market, source_close, volume)."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE daily_ohlcv (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "open BIGINT, high BIGINT, low BIGINT, close BIGINT, volume BIGINT)"
    )
    con.execute(
        "CREATE TABLE daily_market_cap (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "source_close BIGINT, volume BIGINT)"
    )
    if ohlcv:
        con.executemany("INSERT INTO daily_ohlcv VALUES (?,?,?,?,?,?,?,?)", ohlcv)
    if market_cap:
        con.executemany("INSERT INTO daily_market_cap VALUES (?,?,?,?,?)", market_cap)
    return con


def _bar(i: int, ticker: str, close: int, volume: int, market: str = "KOSPI") -> tuple:
    return (_d(i), ticker, market, close, close, close, close, volume)


def _fixture() -> duckdb.DuckDBPyConnection:
    ohlcv: list[tuple] = []
    cap: list[tuple] = []

    def add(ticker, adj_close, adj_vol, raw_close, raw_vol, market="KOSPI", cap_market=None):
        for i in range(N_DAYS):
            ohlcv.append(_bar(i, ticker, adj_close[i], adj_vol, market))
            cap.append((_d(i), ticker, cap_market or market, raw_close[i], raw_vol))

    # R: reverse split later, adjusted history is 5x the raw price (volume 1/5).
    #    Raw close 800 on days 0-2 (below the floor), 1,200 on days 3-4.
    add("R", [4000, 4000, 4000, 6000, 6000], 100_000, [800, 800, 800, 1200, 1200], 500_000)
    # S: split later, adjusted history is 1/4 of the raw price (volume 4x).
    #    Raw close 2,000 (above the floor), adjusted 500 (below).
    add("S", [500] * 5, 400_000, [2000] * 5, 100_000)
    # G: no corporate action, raw == adjusted, 5,000원, 1억 turnover each day.
    add("G", [5000] * 5, 20_000, [5000] * 5, 20_000)
    # T: transferred market, ohlcv labels KOSPI for all history, KRX said KOSDAQ then.
    add("T", [3000] * 5, 100_000, [3000] * 5, 100_000, market="KOSPI", cap_market="KOSDAQ")
    # H: halted on day 2 (open=high=low=0), prices carried, KRX volume 0 that day.
    for i in range(N_DAYS):
        halted = i == 2
        ohlcv.append(
            (_d(i), "H", "KOSPI", 0 if halted else 4000, 0 if halted else 4000,
             0 if halted else 4000, 4000, 0 if halted else 50_000)
        )  # fmt: skip
        cap.append((_d(i), "H", "KOSPI", 4000, 0 if halted else 50_000))
    # M: no KRX row on days 1-2, source_close NULL on day 3, source_close 0 on day 4;
    #    adjusted 2,500원 / 80,000주 = 2억.
    for i in range(N_DAYS):
        ohlcv.append(_bar(i, "M", 2500, 80_000))
    cap.append((_d(0), "M", "KOSPI", 2500, 80_000))
    cap.append((_d(3), "M", "KOSPI", None, 80_000))
    cap.append((_d(4), "M", "KOSPI", 0, 80_000))
    # V: KRX raw volume is NULL on day 2 although source_close is there -> pair falls back.
    for i in range(N_DAYS):
        ohlcv.append(_bar(i, "V", 2500, 80_000))
        cap.append((_d(i), "V", "KOSPI", 2400, None if i == 2 else 90_000))
    return _con(ohlcv, cap)


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> dict[tuple, dict]:
    cur = con.execute(sql)
    cols = [c[0] for c in cur.description]
    return {(r[0], r[1], r[2]): dict(zip(cols, r, strict=True)) for r in cur.fetchall()}


@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    return _fixture()


@pytest.fixture(scope="module")
def old(con) -> dict[tuple, dict]:
    return _rows(con, build_universe_sql(FLT))


@pytest.fixture(scope="module")
def new(con) -> dict[tuple, dict]:
    return _rows(con, build_universe_raw_price_sql(FLT))


def _key(i: int, ticker: str) -> tuple:
    return (_d(i), ticker, "KOSPI")


def test_reverse_split_row_admitted_by_old_is_excluded_by_new(old, new) -> None:
    # raw 800원 (adjusted 4,000원): the old floor lets it in, the raw floor does not.
    for i in (0, 1, 2):
        assert old[_key(i, "R")]["close_ok"] is True
        assert old[_key(i, "R")]["in_universe"] is True
        assert new[_key(i, "R")]["close_ok"] is False
        assert new[_key(i, "R")]["tradable_ok"] is False
        assert new[_key(i, "R")]["in_universe"] is False
    # raw 1,200원 on days 3-4: in under both. Same 4e8 turnover, so liquidity is fine.
    for i in (3, 4):
        assert old[_key(i, "R")]["in_universe"] is True
        assert new[_key(i, "R")]["in_universe"] is True


def test_later_split_row_wrongly_excluded_by_old_is_included_by_new(old, new) -> None:
    for i in range(N_DAYS):
        assert old[_key(i, "S")]["close_ok"] is False
        assert old[_key(i, "S")]["in_universe"] is False
        assert new[_key(i, "S")]["close_ok"] is True
        assert new[_key(i, "S")]["in_universe"] is True


def test_liquidity_is_unchanged_when_adjusted_and_raw_products_match(old, new) -> None:
    for ticker in ("R", "S", "G", "T", "H"):
        for i in range(N_DAYS):
            o, n = old[_key(i, ticker)], new[_key(i, ticker)]
            assert n["avg_turnover"] == o["avg_turnover"], (ticker, i)
            assert n["liquidity_ok"] == o["liquidity_ok"], (ticker, i)
            assert n["is_halted"] == o["is_halted"], (ticker, i)
            assert n["not_halted"] == o["not_halted"], (ticker, i)
            assert n["warmup_ok"] == o["warmup_ok"], (ticker, i)
    # R turnover is raw 800 x 500,000 = adjusted 4,000 x 100,000 = 4e8
    assert new[_key(0, "R")]["avg_turnover"] == 4e8


def test_everything_but_the_price_floor_is_identical(old, new) -> None:
    # V is left out: its KRX raw turnover is deliberately not the adjusted one.
    flag_cols = {"close_ok", "tradable_ok", "in_universe", "raw_price_missing"}
    assert set(new) == set(old)
    for k, o in old.items():
        if k[1] == "V":
            continue
        n = new[k]
        for c, v in o.items():
            if c not in flag_cols:
                assert n[c] == v, (k, c)


def test_missing_source_close_falls_back_to_adjusted_and_is_flagged(con, new) -> None:
    raw = _rows(con, build_raw_price_view_sql())
    # M: no KRX row (1, 2), NULL source_close (3), non-positive source_close (4)
    for i in (1, 2, 3, 4):
        r = raw[_key(i, "M")]
        assert r["raw_price_missing"] is True
        assert (r["close"], r["volume"]) == (2500, 80_000)
        assert new[_key(i, "M")]["raw_price_missing"] is True
    # M day 0 has the KRX row (the raw view carries the KRX numbers)
    assert raw[_key(0, "M")]["raw_price_missing"] is False
    # V: raw volume NULL on day 2 -> BOTH values come from daily_ohlcv, never a mixed pair
    r = raw[_key(2, "V")]
    assert r["raw_price_missing"] is True
    assert (r["close"], r["volume"]) == (2500, 80_000)
    assert (raw[_key(1, "V")]["close"], raw[_key(1, "V")]["volume"]) == (2400, 90_000)
    # fallback rows get the old (adjusted) floor: 2,500 >= 1,000, turnover 2억 -> in
    assert new[_key(2, "M")]["close_ok"] is True
    # rows with a KRX price are not flagged
    assert not any(
        v["raw_price_missing"] for k, v in new.items() if k[1] in ("R", "S", "G", "T", "H")
    )


def test_raw_view_has_universe_price_columns_with_krx_values(con) -> None:
    cur = con.execute(build_raw_price_view_sql())
    assert [c[0] for c in cur.description] == [
        "trade_date", "ticker", "market", "open", "high", "low", "close", "volume",
        "raw_price_missing",
    ]  # fmt: skip
    raw = _rows(con, build_raw_price_view_sql())
    assert (raw[_key(0, "R")]["close"], raw[_key(0, "R")]["volume"]) == (800, 500_000)
    assert raw[_key(0, "R")]["open"] == 4000  # open/high/low stay from daily_ohlcv
    assert raw[_key(2, "H")]["open"] == raw[_key(2, "H")]["high"] == raw[_key(2, "H")]["low"] == 0


def test_market_label_difference_between_sources_still_joins(new) -> None:
    # T: KRX says KOSDAQ, daily_ohlcv says KOSPI. The row is keyed by the ohlcv label
    # and carries its KRX raw price.
    for i in range(N_DAYS):
        assert new[_key(i, "T")]["raw_price_missing"] is False
    assert not any(k[1] == "T" and k[2] == "KOSDAQ" for k in new)


def test_halt_flag_comes_from_ohlcv_not_from_krx(new) -> None:
    assert new[_key(2, "H")]["is_halted"] is True
    assert new[_key(2, "H")]["in_universe"] is False
    assert new[_key(1, "H")]["is_halted"] is False


def test_output_keys_are_unique_and_one_per_price_row(con, new) -> None:
    n_price = con.execute("SELECT count(*) FROM daily_ohlcv").fetchone()[0]
    inner = build_universe_raw_price_sql(FLT)
    total, distinct = con.execute(
        f"SELECT count(*), count(DISTINCT (trade_date, ticker, market)) FROM ({inner})"
    ).fetchone()
    assert total == distinct == n_price == len(new)
    assert_raw_price_inputs_unique(con)


def test_columns_are_the_old_columns_plus_the_flag(con) -> None:
    old_cols = [c[0] for c in con.execute(build_universe_sql(FLT)).description]
    new_cols = [c[0] for c in con.execute(build_universe_raw_price_sql(FLT)).description]
    assert new_cols == [*old_cols, "raw_price_missing"]


def test_no_price_floor_means_same_universe_as_before(con) -> None:
    # min_close_krw = 0 (broad universe, model02 serving spec): raw prices change nothing.
    for flt in (broad_universe_filter(), UniverseFilter()):
        o = _rows(con, build_universe_sql(flt))
        n = _rows(con, build_universe_raw_price_sql(flt))
        assert {k: v["in_universe"] for k, v in o.items()} == {
            k: v["in_universe"] for k, v in n.items()
        }


def test_raw_equals_adjusted_means_same_universe_as_tradable_before() -> None:
    ohlcv = [_bar(i, "G", 5000, 20_000) for i in range(N_DAYS)]
    cap = [(_d(i), "G", "KOSPI", 5000, 20_000) for i in range(N_DAYS)]
    c = _con(ohlcv, cap)
    flt = tradable_universe_filter()
    o = _rows(c, build_universe_sql(flt))
    n = _rows(c, build_universe_raw_price_sql(flt))
    assert {k: v["in_universe"] for k, v in o.items()} == {
        k: v["in_universe"] for k, v in n.items()
    }
    assert not any(v["raw_price_missing"] for v in n.values())


def test_composes_with_the_terminal_tail_exclusion(con) -> None:
    con.execute(
        "CREATE OR REPLACE TABLE label_tail (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "in_terminal_tail BOOLEAN)"
    )
    con.execute("INSERT INTO label_tail VALUES (?, 'G', 'KOSPI', TRUE)", [_d(4)])
    sql = build_universe_v2_sql(build_universe_raw_price_sql(FLT), "label_tail")
    rows = _rows(con, sql)
    assert rows[_key(4, "G")]["in_universe"] is True
    assert rows[_key(4, "G")]["in_universe_v2"] is False
    assert rows[_key(3, "G")]["in_universe_v2"] is True
    assert rows[_key(0, "R")]["in_universe_v2"] is False  # price floor still applies
    assert "raw_price_missing" in next(iter(rows.values()))


def test_duplicate_inputs_fail_closed() -> None:
    dup_cap = _con(
        [_bar(0, "G", 5000, 20_000)],
        [(_d(0), "G", "KOSPI", 5000, 20_000), (_d(0), "G", "KOSDAQ", 5000, 20_000)],
    )
    with pytest.raises(AssertionError, match="daily_market_cap"):
        assert_raw_price_inputs_unique(dup_cap)
    dup_px = _con([_bar(0, "G", 5000, 20_000), _bar(0, "G", 5000, 20_000)], [])
    with pytest.raises(AssertionError, match="daily_ohlcv"):
        assert_raw_price_inputs_unique(dup_px)


def test_view_names_must_be_identifiers() -> None:
    with pytest.raises(ValueError, match="price_view"):
        build_raw_price_view_sql("daily_ohlcv; DROP TABLE x")
    with pytest.raises(ValueError, match="market_cap_view"):
        build_universe_raw_price_sql(FLT, market_cap_view="x y")


def test_custom_view_names_are_used() -> None:
    sql = build_universe_raw_price_sql(FLT, price_view="px.v", market_cap_view="mc.v")
    assert "px.v" in sql and "mc.v" in sql


def test_universe_py_output_is_unchanged() -> None:
    # model02 hashes universe.py; this module must not have changed what it emits.
    got = {
        "default": hashlib.sha256(build_universe_sql(UniverseFilter()).encode()).hexdigest(),
        "tradable": hashlib.sha256(
            build_universe_sql(tradable_universe_filter()).encode()
        ).hexdigest(),
        "broad": hashlib.sha256(build_universe_sql(broad_universe_filter()).encode()).hexdigest(),
    }
    assert got == _SQL_SHA256
