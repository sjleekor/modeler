"""정리매매 (terminal tail) exclusion from the formation universe
(``etl/universe_terminal.py``): the SQL and polars forms, then both on real
``universe.build_universe_sql`` + ``labels_entry`` output."""

from __future__ import annotations

import datetime

import duckdb
import polars as pl
import pytest

from modeler.etl.labels import LabelSpec
from modeler.etl.labels_entry import build_entry_label_sql
from modeler.etl.universe import UniverseFilter, build_universe_sql
from modeler.etl.universe_terminal import (
    build_universe_v2_sql,
    terminal_exclusion_counts,
    with_terminal_exclusion,
)

from ._entry_label_fixtures import delisted_after_long_halt, price_con

KEYS = ["trade_date", "ticker", "market"]


def _d(i: int) -> datetime.date:
    return datetime.date(2020, 1, 1) + datetime.timedelta(days=i)


def _universe() -> pl.DataFrame:
    rows = [
        # (date, ticker, in_universe)
        (_d(0), "A", True),  # no tail flag                         -> stays
        (_d(1), "A", True),  # in the tail                           -> out
        (_d(2), "A", True),  # tail flag NULL                        -> stays
        (_d(3), "A", False),  # not in the universe, tail             -> out (as before)
        (_d(4), "A", None),  # NULL in_universe                      -> out
        (_d(0), "B", True),  # no label row at all (halted day)      -> stays
        (_d(1), "B", True),  # duplicated label rows, flags disagree -> out (any)
    ]
    return pl.DataFrame(
        {
            "trade_date": [r[0] for r in rows],
            "ticker": [r[1] for r in rows],
            "market": ["KOSPI"] * len(rows),
            "in_universe": [r[2] for r in rows],
            "avg_turnover": [float(i) for i in range(len(rows))],
        }
    )


def _label() -> pl.DataFrame:
    rows = [
        (_d(0), "A", False),
        (_d(1), "A", True),
        (_d(2), "A", None),
        (_d(3), "A", True),
        (_d(4), "A", True),
        (_d(1), "B", False),
        (_d(1), "B", True),
        (_d(7), "Z", True),  # a label row with no universe row: ignored
    ]
    return pl.DataFrame(
        {
            "trade_date": [r[0] for r in rows],
            "ticker": [r[1] for r in rows],
            "market": ["KOSPI"] * len(rows),
            "in_terminal_tail": [r[2] for r in rows],
            "fwd_ret_c1_20d": [0.1] * len(rows),  # other label columns must not leak in
        }
    )


_EXPECTED_V2 = [True, False, True, False, False, True, False]
_EXPECTED_TAIL = [False, True, False, True, True, False, True]


def test_polars_form_excludes_tail_rows_and_never_multiplies_or_reorders() -> None:
    uni = _universe()
    out = with_terminal_exclusion(uni, _label())
    assert out.columns == [*uni.columns, "in_terminal_tail", "in_universe_v2"]
    assert out.select(KEYS).equals(uni.select(KEYS))  # same rows, same order
    assert out["in_terminal_tail"].to_list() == _EXPECTED_TAIL
    assert out["in_universe_v2"].to_list() == _EXPECTED_V2
    assert out["in_universe_v2"].null_count() == 0 and out["in_terminal_tail"].null_count() == 0
    # v2 is a subset of the original gate
    assert (out["in_universe_v2"] <= out["in_universe"].fill_null(False)).all()


def test_polars_form_with_the_flag_already_in_the_frame_and_lazy_roundtrip() -> None:
    uni = _universe().with_columns(pl.Series("in_terminal_tail", _EXPECTED_TAIL))
    out = with_terminal_exclusion(uni)
    assert out["in_universe_v2"].to_list() == _EXPECTED_V2
    lazy = with_terminal_exclusion(_universe().lazy(), _label().lazy())
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect()["in_universe_v2"].to_list() == _EXPECTED_V2


def test_polars_form_rejects_ambiguous_inputs() -> None:
    with pytest.raises(ValueError, match="no 'in_terminal_tail'"):
        with_terminal_exclusion(_universe())
    flagged = _universe().with_columns(pl.lit(False).alias("in_terminal_tail"))
    with pytest.raises(ValueError, match="label=None"):
        with_terminal_exclusion(flagged, _label())
    done = with_terminal_exclusion(_universe(), _label())
    with pytest.raises(ValueError, match="in_universe_v2"):
        with_terminal_exclusion(done.drop("in_terminal_tail"), _label())


def test_counts() -> None:
    out = with_terminal_exclusion(_universe(), _label())
    assert terminal_exclusion_counts(out) == {
        "n_in_universe": 5,  # in_universe TRUE
        "n_terminal_tail_dropped": 2,  # (A, d1) and (B, d1) were in and are tail rows
        "n_in_universe_v2": 3,
    }
    assert terminal_exclusion_counts(out.lazy()) == terminal_exclusion_counts(out)


# --- SQL form -------------------------------------------------------------------


def _sql_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("uni", _universe().to_arrow())
    con.register("lab", _label().to_arrow())
    return con


def _run(con: duckdb.DuckDBPyConnection, sql: str) -> pl.DataFrame:
    return pl.from_arrow(con.execute(sql).to_arrow_table()).sort(KEYS)  # type: ignore[return-value]


def test_sql_form_equals_polars_form() -> None:
    sql = build_universe_v2_sql("uni", "lab")
    got = _run(_sql_con(), sql)
    want = with_terminal_exclusion(_universe(), _label()).sort(KEYS)
    assert got.columns == want.columns
    assert got.equals(want)


def test_sql_form_wraps_a_statement_and_keeps_row_count() -> None:
    con = _sql_con()
    statement = "SELECT * FROM uni WHERE ticker = 'A'"
    got = _run(con, build_universe_v2_sql(statement, "lab"))
    assert got.height == 5 and set(got["ticker"]) == {"A"}
    label_sub = "WITH l AS (SELECT * FROM lab) SELECT * FROM l"
    both = _run(con, build_universe_v2_sql("uni", label_sub))
    assert both.height == _universe().height  # duplicated label rows did not multiply


def test_sql_form_rejects_non_identifiers() -> None:
    for bad in ("uni; DROP TABLE uni", "uni u", "1uni", "", "uni--x"):
        with pytest.raises(ValueError):
            build_universe_v2_sql(bad, "lab")
    with pytest.raises(ValueError):
        build_universe_v2_sql("uni", "lab", out_col="a b")
    with pytest.raises(ValueError):
        build_universe_v2_sql("uni", "lab", keys=())
    with pytest.raises(ValueError):
        build_universe_v2_sql("uni", "lab", keys=["trade_date; x"])
    assert "analytics.dim_universe_daily" in build_universe_v2_sql(
        "analytics.dim_universe_daily", "lab"
    )


# --- real universe + labels_entry output -----------------------------------------


def test_real_universe_and_entry_labels_both_forms_drop_the_seven_tail_rows() -> None:
    con = price_con(delisted_after_long_halt())
    flt = UniverseFilter(
        warmup_window=3,
        warmup_min_valid=1,
        liquidity_window=3,
        min_liquidity_krw=0.0,
        apply_liquidity_filter=False,
    )
    spec = LabelSpec(horizons=(5,), kind="abs")
    con.execute("CREATE VIEW label_entry AS " + build_entry_label_sql(spec, "c1"))
    con.execute("CREATE VIEW dim_universe AS " + build_universe_sql(flt))

    via_sql = _run(con, build_universe_v2_sql("dim_universe", "label_entry"))
    via_stmt = _run(con, build_universe_v2_sql(build_universe_sql(flt), "label_entry"))
    uni = pl.from_arrow(con.execute("SELECT * FROM dim_universe").to_arrow_table())
    lab = pl.from_arrow(con.execute("SELECT * FROM label_entry").to_arrow_table())
    via_polars = with_terminal_exclusion(uni, lab).sort(KEYS)  # type: ignore[arg-type]

    assert via_sql.equals(via_polars) and via_stmt.equals(via_polars)
    assert via_sql.height == uni.height  # one output row per universe row

    a_tail = via_sql.filter(pl.col("in_terminal_tail"))
    assert set(a_tail["ticker"]) == {"A"} and a_tail.height == 7
    assert a_tail["in_universe"].all()  # the old gate let all seven through
    assert not a_tail["in_universe_v2"].any()
    assert terminal_exclusion_counts(via_sql) == {
        "n_in_universe": via_sql["in_universe"].sum(),
        "n_terminal_tail_dropped": 7,
        "n_in_universe_v2": via_sql["in_universe"].sum() - 7,
    }
    # A's five trading days before the halt (2020-01-02..06) stay in; so do B's rows
    before = via_sql.filter(
        (pl.col("ticker") == "A") & (pl.col("trade_date") < datetime.date(2020, 1, 7))
    )
    assert before["in_universe_v2"].all() and before.height == 5
    assert via_sql.filter(pl.col("ticker") == "B")["in_universe_v2"].all()
