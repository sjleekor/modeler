"""Unit tests for the binary label outputs (20260907_model_experiment `01` §3).

``y_up`` (L-A, beat the market) and ``y_top`` (L-B, top 20%) are additions to
the shared label library, so these tests carry two burdens: the new columns are
correct, and model 01's label SQL is untouched by their existence.
"""

from __future__ import annotations

import hashlib

import duckdb
import pytest
from modeler.etl import labels

# The label SQL model 01 builds its dataset from. Frozen deliberately: a change
# here changes that model's dataset and breaks the chain back to its published
# gate numbers, so it must be an intentional decision with the golden parity
# re-measured — not a side effect of extending the library for a new model.
_MODEL_01_LABEL_SQL_SHA256 = "bd927b316d37c87b9cedec686fd8b5b4f3f3e6bfa06b1cf5bf210631161b41b0"


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


@pytest.fixture()
def ranked_con() -> duckdb.DuckDBPyConnection:
    """5 names on one date with distinct forward returns -> clean ranks.

    Every output is requested at once so ``y_top`` can be compared with the
    ``y_cls`` it is supposed to agree with.
    """
    con = duckdb.connect()
    rows = []
    closes_day2 = {"A": 100, "B": 105, "C": 110, "D": 115, "E": 120}
    for tk, c2 in closes_day2.items():
        rows.append(("2020-01-01", tk, "KOSPI", 10, 11, 9, 100, 5))
        rows.append(("2020-01-02", tk, "KOSPI", 10, 11, 9, c2, 5))
    _ohlcv_view(con, rows)
    spec = labels.LabelSpec(horizons=(1,), outputs=("reg", "rank", "cls", "up", "top"))
    con.execute(f"CREATE VIEW label_daily AS {labels.build_label_sql(spec)}")
    return con


# --- model 01 is unaffected --------------------------------------------------


def test_default_outputs_do_not_emit_the_binary_columns() -> None:
    sql = labels.build_label_sql(labels.LabelSpec())
    assert "y_up_" not in sql
    assert "y_top_" not in sql


def test_model_01_label_sql_is_byte_identical() -> None:
    sql = labels.build_label_sql(labels.LabelSpec(horizons=(20, 5, 60)))
    digest = hashlib.sha256(sql.encode()).hexdigest()
    assert digest == _MODEL_01_LABEL_SQL_SHA256


def test_up_and_top_are_accepted_and_typos_still_are_not() -> None:
    labels.LabelSpec(outputs=("rank", "up", "top"))
    with pytest.raises(ValueError):
        labels.LabelSpec(outputs=("rank", "upp"))


# --- L-A: y_up ---------------------------------------------------------------


def test_up_is_the_sign_of_the_excess_return(ranked_con) -> None:
    rows = ranked_con.execute(
        "SELECT ticker, raw_label_1d, y_up_1d FROM label_daily "
        "WHERE trade_date=DATE '2020-01-01' ORDER BY ticker"
    ).fetchall()
    for _ticker, raw, up in rows:
        assert up == int(raw > 0)
    got = {ticker: up for ticker, _raw, up in rows}
    # forward returns 0/5/10/15/20% -> eqw bench 10% -> A,B below, D,E above.
    # C sits on the benchmark to within float noise, so its sign is asserted by
    # the invariant above rather than pinned here.
    assert got["A"] == 0
    assert got["B"] == 0
    assert got["D"] == 1
    assert got["E"] == 1


def test_up_is_zero_when_the_excess_is_exactly_zero() -> None:
    """Two names moving identically are each their own benchmark -> raw == 0.

    ``y_up`` is 0 there, not 1: an excess return of exactly zero did not beat
    the market. This is the only construction where the tie is exact — with
    distinct returns the mean carries float noise.
    """
    con = duckdb.connect()
    rows = []
    for tk in ("A", "B"):
        rows.append(("2020-01-01", tk, "KOSPI", 10, 11, 9, 100, 5))
        rows.append(("2020-01-02", tk, "KOSPI", 10, 11, 9, 110, 5))
    _ohlcv_view(con, rows)
    spec = labels.LabelSpec(horizons=(1,), outputs=("up",))
    con.execute(f"CREATE VIEW label_daily AS {labels.build_label_sql(spec)}")
    got = con.execute(
        "SELECT raw_label_1d, y_up_1d FROM label_daily WHERE trade_date=DATE '2020-01-01'"
    ).fetchall()
    assert got == [(0.0, 0), (0.0, 0)]


def test_up_on_kind_abs_is_the_sign_of_the_forward_return() -> None:
    """``kind="abs"`` turns y_up into L-C — not used by E0~E5, but it must hold."""
    con = duckdb.connect()
    rows = [
        ("2020-01-01", "A", "KOSPI", 10, 11, 9, 100, 5),
        ("2020-01-02", "A", "KOSPI", 10, 11, 9, 90, 5),
        ("2020-01-01", "B", "KOSPI", 10, 11, 9, 100, 5),
        ("2020-01-02", "B", "KOSPI", 10, 11, 9, 110, 5),
    ]
    _ohlcv_view(con, rows)
    spec = labels.LabelSpec(horizons=(1,), kind="abs", outputs=("up",))
    con.execute(f"CREATE VIEW label_daily AS {labels.build_label_sql(spec)}")
    got = dict(
        con.execute(
            "SELECT ticker, y_up_1d FROM label_daily WHERE trade_date=DATE '2020-01-01'"
        ).fetchall()
    )
    # both fall vs the market in excess terms, but absolutely A is down, B up.
    assert got == {"A": 0, "B": 1}


# --- L-B: y_top --------------------------------------------------------------


def test_top_agrees_with_the_positive_class_of_y_cls(ranked_con) -> None:
    rows = ranked_con.execute(
        "SELECT ticker, y_top_1d, y_cls_1d, y_rank_1d FROM label_daily "
        "WHERE trade_date=DATE '2020-01-01' ORDER BY ticker"
    ).fetchall()
    for _ticker, top, cls, rank in rows:
        assert top == int(cls == 1)
        assert top == int(rank >= 0.8)
    # 5 names, PERCENT_RANK 0/.25/.5/.75/1 -> only E clears cls_top=0.8.
    assert [r[1] for r in rows] == [0, 0, 0, 0, 1]


def test_top_follows_cls_top_when_it_is_moved() -> None:
    con = duckdb.connect()
    rows = []
    for tk, c2 in {"A": 100, "B": 105, "C": 110, "D": 115, "E": 120}.items():
        rows.append(("2020-01-01", tk, "KOSPI", 10, 11, 9, 100, 5))
        rows.append(("2020-01-02", tk, "KOSPI", 10, 11, 9, c2, 5))
    _ohlcv_view(con, rows)
    spec = labels.LabelSpec(horizons=(1,), outputs=("top",), cls_top=0.5)
    con.execute(f"CREATE VIEW label_daily AS {labels.build_label_sql(spec)}")
    got = dict(
        con.execute(
            "SELECT ticker, y_top_1d FROM label_daily WHERE trade_date=DATE '2020-01-01'"
        ).fetchall()
    )
    assert got == {"A": 0, "B": 0, "C": 1, "D": 1, "E": 1}


# --- null guard, replicated from the cls/reg regression ----------------------


def test_secondary_horizon_without_forward_keeps_the_binaries_null() -> None:
    """The ``cls``/``reg`` guard regression, for ``up``/``top``.

    PERCENT_RANK assigns a value to rows whose ORDER BY key is NULL, and
    ``CAST(NULL > 0 AS TINYINT)`` is NULL only because the comparison is —
    ``y_top`` has no such luck and would score every unlabelled row.
    """
    con = duckdb.connect()
    rows = [(f"2020-01-{i + 1:02d}", "A", "KOSPI", 10, 11, 9, 100 + i, 5) for i in range(30)]
    rows += [(f"2020-01-{i + 1:02d}", "B", "KOSPI", 10, 11, 9, 200, 5) for i in range(30)]
    _ohlcv_view(con, rows)
    spec = labels.LabelSpec(horizons=(5, 60), outputs=("rank", "up", "top"))
    con.execute(f"CREATE VIEW label_daily AS {labels.build_label_sql(spec)}")
    raw60, up60, top60 = con.execute(
        "SELECT count(raw_label_60d), count(y_up_60d), count(y_top_60d) FROM label_daily"
    ).fetchone()
    assert raw60 == 0
    assert up60 == 0
    assert top60 == 0
    # the primary horizon is unaffected: every row with a forward return scores.
    raw5, up5, top5 = con.execute(
        "SELECT count(raw_label_5d), count(y_up_5d), count(y_top_5d) FROM label_daily"
    ).fetchone()
    assert raw5 > 0
    assert up5 == raw5
    assert top5 == raw5
