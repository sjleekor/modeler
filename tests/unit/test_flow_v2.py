"""``feat_flow_v2`` (15_quality_fix_plan §4.2-§4.3) and the frozen ``feat_flow`` text."""

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


def _con(rows: list[tuple[dt.date, dict[str, float | None]]], ticker: str = "A"):
    """One ticker, one flow/pit/quality row per entry; a ``None`` value is a missing metric."""
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
    con.execute(f"CREATE VIEW feat_flow_v2 AS {build_flow_v2_sql()}")
    return con


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
    assert not any("to_volume" in c for c in cols)


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
