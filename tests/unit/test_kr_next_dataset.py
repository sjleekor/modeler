"""Unit tests for the next KR model's dataset assembly (C stage 3, 18_c3_training_test.md C3-10).

Synthetic, tiny panels only. The six feature marts are replaced by registered views whose
every column holds ``day_index + 100 * ticker_number`` so a test can tell which day a value
was read from; ``daily_ohlcv`` and the label inputs are real small tables, so the universe,
the terminal-tail CTEs, the entry label and the one-per-company rule run for real.
"""

from __future__ import annotations

import datetime
import re

import duckdb
import polars as pl
import pytest

from modeler.etl.label_gate import DEFAULT_ALLOWED_KINDS
from modeler.etl.universe import UniverseFilter
from modeler.models._03_kr_next import dataset as ds
from modeler.models._03_kr_next import feature_set as fs

START = datetime.date(2020, 1, 1)
N_DAYS = 28
HALT_DAY = 5  # T2 is halted on this day index
SMALL_UNIVERSE = UniverseFilter(
    warmup_window=3,
    warmup_min_valid=1,
    liquidity_window=2,
    min_liquidity_krw=0.0,
    min_close_krw=0.0,
    apply_liquidity_filter=False,
)

# ticker -> (market, daily growth, corp_code)
TICKERS: dict[str, tuple[str, float, str]] = {
    "000010": ("KOSPI", 0.02, "C1"),
    "000015": ("KOSPI", 0.02, "C1"),  # preferred stock of C1: dropped by one-per-company
    "000020": ("KOSPI", 0.04, "C2"),
    "000030": ("KOSDAQ", 0.05, "C3"),
    "000040": ("KOSDAQ", 0.01, "C4"),
    "000050": ("KOSDAQ", 0.03, "C5"),
}
TNO = {t: i + 1 for i, t in enumerate(TICKERS)}
MART_VIEWS = {
    "feat_price_v2": "syn_price_v2",
    "feat_price": "syn_price",
    "feat_flow_v2": "syn_flow_v2",
    "feat_fin_v2": "syn_fin_v2",
    "feat_market_cap": "syn_mcap",
    "feat_filing_activity": "syn_filing",
}
OTHER_VIEWS = ("daily_ohlcv", "daily_market_cap", "dim_price_quality_daily", "stock_master")


def _date(i: int) -> datetime.date:
    return START + datetime.timedelta(days=i)


def _con() -> duckdb.DuckDBPyConnection:
    ohlcv, mcap, keys = [], [], []
    for t, (market, g, _) in TICKERS.items():
        for i in range(N_DAYS):
            close = 1000.0 * (1 + g) ** i
            halted = t == "000020" and i == HALT_DAY
            bar = (0.0, 0.0, 0.0, 0.0) if halted else (close - 0.5, close + 0.5, close - 1, close)
            ohlcv.append((_date(i), t, market, *bar, 1000.0))
            mcap.append((_date(i), t, market, close, 1000.0, close * 1000.0, close * 1e6, 1e6))
            keys.append((_date(i), t, market, float(i + 100 * TNO[t])))
    con = duckdb.connect()
    con.register(
        "daily_ohlcv",
        pl.DataFrame(
            ohlcv,
            schema=["trade_date", "ticker", "market", "open", "high", "low", "close", "volume"],
            orient="row",
        ),
    )
    con.register(
        "daily_market_cap",
        pl.DataFrame(
            mcap,
            schema=[
                "trade_date",
                "ticker",
                "market",
                "source_close",
                "volume",
                "trading_value",
                "market_cap",
                "listed_shares",
            ],
            orient="row",
        ),
    )
    keyed = pl.DataFrame(keys, schema=["trade_date", "ticker", "market", "v"], orient="row")
    con.register(
        "dim_price_quality_daily",
        keyed.select("trade_date", "ticker", "market", pl.lit(False).alias("ca_event")),
    )
    con.register(
        "stock_master",
        keyed.select("ticker", "market", pl.lit("ACTIVE").alias("status")).unique(),
    )
    con.register(
        "dart_corp_master",
        pl.DataFrame(
            {
                "ticker": [t for t in TICKERS if t != "000015"],
                "corp_code": [c for t, (_, _, c) in TICKERS.items() if t != "000015"],
            }
        ),
    )
    con.register(
        "krx_index_daily",
        pl.DataFrame(
            {
                "bas_dd": [_date(i) for i in range(N_DAYS)],
                "idx_clss": ["KOSPI"] * N_DAYS,
                "idx_nm": ["코스피 200"] * N_DAYS,
                "close_idx": [300.0 + i for i in range(N_DAYS)],
                "open_idx": [300.0 + i for i in range(N_DAYS)],
            }
        ),
    )
    for mart, view in MART_VIEWS.items():
        cols = fs.MART_COLUMNS[mart]
        frame = keyed.select(
            "trade_date", "ticker", "market", *[pl.col("v").alias(c) for c in cols]
        )
        con.register(view, frame)
    return con


def _panel(con: duckdb.DuckDBPyConnection, feature_set: str = "main", **kw: object) -> pl.DataFrame:
    kw.setdefault("universe_filter", SMALL_UNIVERSE)
    kw.setdefault("start_date", "2019-01-01")
    return ds.build_kr_next_panel(con, feature_set, mart_overrides=MART_VIEWS, **kw)  # type: ignore[arg-type]


def _value(frame: pl.DataFrame, ticker: str, day: int, col: str) -> float | None:
    row = frame.filter((pl.col("ticker") == ticker) & (pl.col("trade_date") == _date(day)))
    assert row.height == 1, (ticker, day, row.height)
    return row[col][0]


# --- constants -----------------------------------------------------------------------------


def test_lagged_columns_are_the_c3_4_decision() -> None:
    krx = {"mcap_krx_log", "px_turnover_v2", "px_turnover_ma20_v2", "px_amihud_20d_v2"}
    assert set(ds.LAGGED_COLS) == krx | set(fs.FLOW_COLS)
    assert len(ds.LAGGED_COLS) == len(set(ds.LAGGED_COLS)) == 4 + 13
    # price columns that do not read KRX raw values stay same-day
    for c in ("px_ret_1d_v2", "px_vol_20d_v2", "px_halt_ratio_20d", "px_idio_vol_60d"):
        assert c not in ds.LAGGED_COLS
    # the KRX-denominator flow columns named in C3-4 are inside the flow group
    assert set(ds.KRX_RAW_LAGGED_COLS) - krx <= set(fs.FLOW_COLS)
    assert set(ds.lagged_columns("noflow")) == krx
    assert set(ds.lagged_columns("main")) == krx | set(fs.FLOW_COLS)
    # fin and filing columns are same-day
    assert not set(fs.FIN_COLS) - {"mcap_krx_log"} & set(ds.LAGGED_COLS)
    assert not set(fs.FILING_COLS) & set(ds.LAGGED_COLS)


def test_feature_columns_and_unknown_set() -> None:
    assert ds.feature_columns("main") == fs.MAIN_COLS
    assert ds.feature_columns("noflow") == fs.NOFLOW_COLS
    with pytest.raises(ValueError):
        ds.feature_columns("flow_only")


# --- feature columns -----------------------------------------------------------------------


@pytest.mark.parametrize("feature_set", ["main", "noflow"])
def test_output_columns_are_exactly_the_set(feature_set: str) -> None:
    out = _panel(_con(), feature_set)
    expected = [*ds.KEYS, *ds.feature_columns(feature_set), *ds.UNIVERSE_FLAG_COLS]
    assert out.columns == expected
    assert len(ds.feature_columns("main")) == 43 and len(ds.feature_columns("noflow")) == 30


def test_noflow_does_not_build_or_read_flow() -> None:
    sql = ds.build_kr_next_panel_sql("noflow")
    assert "krx_security_flow_raw" not in sql and "kn_feat_flow_v2" not in sql
    main = ds.build_kr_next_panel_sql("main")
    assert "krx_security_flow_raw" in main and "kn_feat_flow_v2" in main


def test_default_sql_builds_every_mart_and_parses() -> None:
    sql = ds.build_kr_next_panel_sql("main", include_labels=True)
    assert len(duckdb.extract_statements(sql)) == 1
    for needle in (
        "fin_quarterly_metric_vintage_dlr",  # vintage with delisted names + R1 (18 §8.1)
        "dim_stock_pit_daily",  # PIT view
        "krx_security_flow_raw",
        "dart_filing_receipt_raw",
        "dim_mcap_distortion",
        "stock_master",
        "krx_index_daily",
        "코스피 200",
    ):
        assert needle in sql, needle
    custom = ds.KrNextViews(vintage_view="my_vintage", pit_view="my_pit", flow_raw="my_flow")
    sql2 = ds.build_kr_next_panel_sql("main", views=custom)
    assert "my_vintage" in sql2 and "my_pit" in sql2 and "my_flow" in sql2
    assert "fin_quarterly_metric_vintage_dlr" not in sql2


def test_bad_arguments() -> None:
    with pytest.raises(ValueError):
        ds.build_kr_next_panel_sql(start_date="2017/01/01")
    with pytest.raises(ValueError):
        ds.build_kr_next_panel_sql(allowed_kinds=("regular", "nope"))
    with pytest.raises(KeyError):
        ds.build_kr_next_panel_sql(mart_overrides={"feat_nothing": "x"})
    with pytest.raises(ValueError):
        ds.KrNextViews(daily_ohlcv="a; DROP TABLE x")


# --- input timing (C3-4) -------------------------------------------------------------------


def test_lag_shifts_krx_and_flow_columns_one_valid_session() -> None:
    out = _panel(_con(), "main")
    t = "000030"
    n = TNO[t] * 100
    for col in ds.LAGGED_COLS:
        assert _value(out, t, 0, col) is None  # no previous session
        assert _value(out, t, 3, col) == 2 + n
    for col in ("px_ret_1d_v2", "px_halt_ratio_20d", "fin_roa_v2", "ev_filing_burst_60d"):
        assert _value(out, t, 3, col) == 3 + n  # same day


def test_lag_skips_halt_rows() -> None:
    out = _panel(_con(), "main")
    t = "000020"
    n = TNO[t] * 100
    assert out.filter(
        (pl.col("ticker") == t) & (pl.col("trade_date") == _date(HALT_DAY))
    ).is_empty()
    # day 6 follows the halt day 5: the previous valid session is day 4, not the halt row
    for col in ("mcap_krx_log", "px_turnover_v2", "flow_foreign_netbuy_z_20d_v2"):
        assert _value(out, t, HALT_DAY + 1, col) == (HALT_DAY - 1) + n
        assert _value(out, t, HALT_DAY - 1, col) == (HALT_DAY - 2) + n
    assert _value(out, t, HALT_DAY + 1, "px_ret_5d_v2") == (HALT_DAY + 1) + n


def test_lag_runs_before_the_sample_filters() -> None:
    # the first kept day (start_date) must read the previous day, which is outside the sample
    out = _panel(_con(), "main", start_date=_date(3).isoformat())
    t = "000030"
    assert out.filter(pl.col("ticker") == t)["trade_date"].min() == _date(3)
    assert _value(out, t, 3, "mcap_krx_log") == 2 + TNO[t] * 100
    # a ticker's preferred line is not kept, but a kept row never reads another ticker
    assert _value(out, "000010", 3, "mcap_krx_log") == 2 + TNO["000010"] * 100


def test_missing_values_stay_missing() -> None:
    con = _con()
    con.register(
        "syn_price_nan",
        con.execute("SELECT * FROM syn_price_v2")
        .pl()
        .with_columns(
            pl.when(pl.col("trade_date") == _date(8))
            .then(None)
            .otherwise(pl.col("px_ret_5d_v2"))
            .alias("px_ret_5d_v2")
        ),
    )
    overrides = {**MART_VIEWS, "feat_price_v2": "syn_price_nan"}
    out = ds.build_kr_next_panel(
        con,
        "main",
        mart_overrides=overrides,
        universe_filter=SMALL_UNIVERSE,
        start_date="2019-01-01",
    )
    assert _value(out, "000030", 8, "px_ret_5d_v2") is None
    assert not any(c.endswith("_isna") for c in out.columns)


# --- universe and sample -------------------------------------------------------------------


def test_one_row_per_date_and_company() -> None:
    out = _panel(_con(), "noflow")
    assert out.group_by("trade_date", "corp_code").len()["len"].max() == 1
    assert "000015" not in set(out["ticker"])  # preferred line of C1
    assert out["keep_one_per_company"].all()
    assert out["in_universe_v2"].all()
    assert out["raw_price_missing"].sum() == 0


def test_start_date_and_end_date_filter() -> None:
    con = _con()
    out = _panel(con, "noflow", start_date=_date(10).isoformat(), end_date=_date(12).isoformat())
    assert out["trade_date"].min() == _date(10) and out["trade_date"].max() == _date(12)
    default = ds.build_kr_next_panel_sql("noflow")
    assert f"DATE '{ds.DEFAULT_START_DATE}'" in default and ds.DEFAULT_START_DATE == "2017-01-01"
    everything = _panel(con, "noflow", start_date="2019-01-01")
    assert everything["trade_date"].min() == _date(0)


def test_terminal_tail_rows_leave_the_universe() -> None:
    # T9 is delisted: 5 trading days, 30 halted days, then 7 tail days with prices
    con = _con()
    rows = []
    for i in range(N_DAYS):
        d = _date(i)
        if i < 5:
            c = 100.0 + i
            rows.append((d, "000090", "KOSPI", c - 0.5, c + 0.5, c - 1, c, 1000.0))
        elif i < 20:
            rows.append((d, "000090", "KOSPI", 0.0, 0.0, 0.0, 0.0, 0.0))
        elif i < 25:
            c = 50.0 - i
            rows.append((d, "000090", "KOSPI", c - 0.5, c + 0.5, c - 1, c, 1000.0))
    extra = pl.DataFrame(
        rows,
        schema=["trade_date", "ticker", "market", "open", "high", "low", "close", "volume"],
        orient="row",
    )
    con.register("daily_ohlcv", pl.concat([con.execute("SELECT * FROM daily_ohlcv").pl(), extra]))
    mc = extra.select(
        "trade_date", "ticker", "market",
        pl.col("close").alias("source_close"), "volume",
        (pl.col("close") * pl.col("volume")).alias("trading_value"),
        pl.lit(1e6).alias("market_cap"), pl.lit(1e6).alias("listed_shares"),
    )  # fmt: skip
    con.register(
        "daily_market_cap", pl.concat([con.execute("SELECT * FROM daily_market_cap").pl(), mc])
    )
    con.register(
        "dart_corp_master",
        pl.concat(
            [
                con.execute("SELECT * FROM dart_corp_master").pl(),
                pl.DataFrame({"ticker": ["000090"], "corp_code": ["C9"]}),
            ]
        ),
    )
    con.register(
        "stock_master",
        pl.concat(
            [
                con.execute("SELECT * FROM stock_master").pl(),
                pl.DataFrame({"ticker": ["000090"], "market": ["KOSPI"], "status": ["DELISTED"]}),
            ]
        ),
    )
    keyed = extra.select("trade_date", "ticker", "market", pl.lit(0.0).alias("v"))
    con.register(
        "dim_price_quality_daily",
        pl.concat(
            [
                con.execute("SELECT * FROM dim_price_quality_daily").pl(),
                keyed.select("trade_date", "ticker", "market", pl.lit(False).alias("ca_event")),
            ]
        ),
    )
    for mart, view in MART_VIEWS.items():
        cols = fs.MART_COLUMNS[mart]
        con.register(
            view,
            pl.concat(
                [
                    con.execute(f"SELECT * FROM {view}").pl(),
                    keyed.select(
                        "trade_date", "ticker", "market", *[pl.col("v").alias(c) for c in cols]
                    ),
                ]
            ),
        )
    serving = _panel(con, "noflow")
    labelled = _panel(con, "noflow", include_labels=True)
    tail_days = {_date(i) for i in range(20, 25)}
    for frame in (serving, labelled):
        got = set(frame.filter(pl.col("ticker") == "000090")["trade_date"])
        assert got <= {_date(i) for i in range(5)}  # warm-up ok days only, tail rows excluded
        assert not got & tail_days
    assert serving.shape[0] == labelled.shape[0]


# --- labels --------------------------------------------------------------------------------


def test_serving_mode_emits_no_label_or_return_columns() -> None:
    con = _con()
    out = _panel(con, "main")
    banned = re.compile(r"fwd_ret|raw_label|label_|bench_ret|^y$|trainable|in_terminal_tail")
    assert not [c for c in out.columns if banned.search(c)]
    sql = ds.build_kr_next_panel_sql("main")
    for needle in ("fwd_ret", "raw_label", "label_exit_kind", "PERCENT_RANK", "krx_index_daily"):
        assert needle not in sql, needle
    # the tail flag is still computed: serving and training samples are the same rows
    labelled = _panel(con, "main", include_labels=True)
    shared = out.columns
    assert out.equals(labelled.select(shared))


def test_label_columns_and_percentile_across_markets() -> None:
    out = _panel(_con(), "noflow", include_labels=True)
    lab = ds.label_columns()
    assert out.columns[-5:] == ["y", "trainable", lab["kind"], lab["end_date"], lab["exit_lag"]]
    day0 = out.filter(pl.col("trade_date") == _date(0)).sort("y")
    assert day0["ticker"].to_list() == ["000040", "000010", "000050", "000020", "000030"]
    assert day0["y"].to_list() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    # KOSPI and KOSDAQ names are one cross-section: a within-market rank would give 0 and 1 to
    # the KOSPI pair (000010, 000020)
    assert _value(day0, "000010", 0, "y") == pytest.approx(0.25)


def test_trainable_follows_label_gate() -> None:
    con = _con()
    out = _panel(con, "noflow", include_labels=True, keep_raw_label=True)
    lab = ds.label_columns()
    gated = pl.col(lab["raw"]).is_not_null() & pl.col(lab["kind"]).is_in(
        list(DEFAULT_ALLOWED_KINDS)
    )
    assert out.filter(pl.col("trainable") != gated.fill_null(False)).is_empty()
    # rows are kept even when not trainable, and y exists exactly for trainable rows
    assert out.filter(~pl.col("trainable")).height > 0
    assert out.filter(pl.col("trainable") != pl.col("y").is_not_null()).is_empty()
    assert set(out.filter(pl.col("trainable"))[lab["kind"]]) <= {"regular", "late"}
    # end of the panel: the exit session is not there yet
    last = out.filter(pl.col("trade_date") == _date(N_DAYS - 2))
    assert (last[lab["kind"]] == "censored").all() and not last["trainable"].any()
    # the entry day after a halt is not tradable
    row = out.filter((pl.col("ticker") == "000020") & (pl.col("trade_date") == _date(HALT_DAY - 1)))
    assert row[lab["kind"]][0] == "untradable" and not row["trainable"][0]
    # a halt inside the window moves the exit later: still trainable as 'late'
    t2 = out.filter((pl.col("ticker") == "000020") & (pl.col("trade_date") == _date(0)))
    assert t2[lab["kind"]][0] == "late" and t2["trainable"][0]
    assert t2[lab["exit_lag"]][0] == 1
    assert t2[lab["end_date"]][0] == _date(22)


def test_allowed_kinds_changes_gate_and_rank_population() -> None:
    out = _panel(
        _con(),
        "noflow",
        include_labels=True,
        allowed_kinds=("regular",),
        keep_raw_label=True,
    )
    day0 = out.filter(pl.col("trade_date") == _date(0))
    t2 = day0.filter(pl.col("ticker") == "000020")
    assert not t2["trainable"][0] and t2["y"][0] is None  # 'late' is not allowed here
    # the rank is over the four trainable rows only
    ranked = day0.filter(pl.col("trainable")).sort("y")
    assert ranked["y"].to_list() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
    assert ranked["ticker"].to_list() == ["000040", "000010", "000050", "000030"]


def test_keep_raw_label_adds_fwd_and_raw() -> None:
    out = _panel(_con(), "noflow", include_labels=True, keep_raw_label=True)
    lab = ds.label_columns()
    assert out.columns[-2:] == [lab["fwd"], lab["raw"]]
    # excess over KOSPI 200 (close 300 + i): entry day 1, exit day 21 for a regular row
    r = out.filter((pl.col("ticker") == "000030") & (pl.col("trade_date") == _date(0)))
    fwd = 1.05**20 - 1
    assert r[lab["fwd"]][0] == pytest.approx(fwd)
    assert r[lab["raw"]][0] == pytest.approx(fwd - (321.0 / 301.0 - 1))
