"""``feat_fin_v2.fin_roe_v2`` (15_quality_fix_plan §4.4) and the frozen fin_scan text."""

from __future__ import annotations

import datetime as dt
import hashlib

import duckdb
import pytest

from modeler.etl.features.fin_scan import build_fin_scan_daily_sql
from modeler.etl.features.fin_v2 import build_fin_v2_sql
from modeler.etl.features.fin_vintage import JOIN_PLAN_COALESCE, JOIN_PLAN_OR

FILED = dt.date(2020, 5, 1)
ASOF = dt.date(2020, 6, 1)


def _row(ticker, metric, value, rcept, *, basis="CFS", seq=1, filed=FILED):
    flow = metric in ("net_income", "controlling_net_income")
    return (
        ticker,
        metric,
        basis,
        seq,
        rcept,
        "direct_interim" if flow else "instant",
        value if flow else None,
        None if flow else value,
        filed if flow else None,
        None if flow else filed,
        None,
    )


def _con(rows, tickers):
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE fin_quarterly_metric_vintage (ticker VARCHAR, metric_code VARCHAR, "
        "fs_basis VARCHAR, seq_key BIGINT, rcept_no VARCHAR, metric_kind VARCHAR, "
        "ttm_value DOUBLE, standalone_value DOUBLE, ttm_available_from DATE, "
        "available_from DATE, value_lag_4q DOUBLE)"
    )
    con.executemany("INSERT INTO fin_quarterly_metric_vintage VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.execute(
        "CREATE TABLE dim_stock_pit_daily (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "market_cap_pit DOUBLE, issued_shares_pit DOUBLE, shares_is_available BOOLEAN, "
        "shares_invalid_flag BOOLEAN, shares_available_from DATE)"
    )
    con.execute(
        "CREATE TABLE dim_price_quality_daily (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "is_halted BOOLEAN, valid_session_idx BIGINT)"
    )
    for t in tickers:
        con.execute(
            "INSERT INTO dim_stock_pit_daily VALUES (?, ?, 'KOSPI', 1e6, 100, TRUE, FALSE, ?)",
            [ASOF, t, FILED],
        )
        con.execute(
            "INSERT INTO dim_price_quality_daily VALUES (?, ?, 'KOSPI', FALSE, 1)", [ASOF, t]
        )
    return con


def _roe(con) -> dict[str, float | None]:
    sql = build_fin_v2_sql()
    return {t: v for t, v in con.execute(f"SELECT ticker, fin_roe_v2 FROM ({sql})").fetchall()}


def test_ratio_and_equity_guard() -> None:
    rows = []
    for t, equity in (("POS", 200.0), ("NEG", -200.0), ("ZERO", 0.0)):
        rows += [
            _row(t, "net_income", 30.0, f"R{t}"),
            _row(t, "controlling_net_income", 20.0, f"R{t}"),
            _row(t, "total_equity", equity, f"R{t}"),
        ]
    rows += [
        _row("NOEQ", "net_income", 30.0, "RNOEQ"),
        _row("NOEQ", "controlling_net_income", 20.0, "RNOEQ"),
    ]
    out = _roe(_con(rows, ["POS", "NEG", "ZERO", "NOEQ"]))
    assert out["POS"] == pytest.approx(0.1)  # controlling 20 / 200
    assert out["NEG"] is None
    assert out["ZERO"] is None
    assert out["NOEQ"] is None


def test_falls_back_to_net_income_when_controlling_is_missing() -> None:
    rows = [_row("A", "net_income", 30.0, "R1"), _row("A", "total_equity", 300.0, "R1")]
    assert _roe(_con(rows, ["A"]))["A"] == pytest.approx(0.1)


def test_numerator_and_denominator_must_share_a_filing() -> None:
    rows = [
        _row("A", "net_income", 30.0, "R1"),
        _row("A", "controlling_net_income", 20.0, "R1"),
        _row("A", "total_equity", 200.0, "R2"),  # same quarter re-filed elsewhere
        _row("B", "net_income", 30.0, "R1"),
        _row("B", "controlling_net_income", 20.0, "R1"),
        _row("B", "total_equity", 200.0, "R1"),
    ]
    out = _roe(_con(rows, ["A", "B"]))
    assert out["A"] is None
    assert out["B"] == pytest.approx(0.1)


def test_one_basis_per_ticker_date() -> None:
    # CFS net income exists -> CFS is used even though an OFS equity exists with its own rcept.
    rows = [
        _row("A", "net_income", 30.0, "R1", basis="CFS"),
        _row("A", "total_equity", 300.0, "R1", basis="CFS"),
        _row("A", "net_income", 50.0, "R9", basis="OFS"),
        _row("A", "total_equity", 100.0, "R9", basis="OFS"),
    ]
    assert _roe(_con(rows, ["A"]))["A"] == pytest.approx(0.1)


def test_base_ok_gate() -> None:
    rows = [_row("A", "net_income", 30.0, "R1"), _row("A", "total_equity", 300.0, "R1")]
    con = _con(rows, ["A"])
    con.execute("UPDATE dim_price_quality_daily SET is_halted = TRUE")
    assert _roe(con)["A"] is None


def test_numerator_branches() -> None:
    rows = [
        # C: controlling shares the equity filing -> controlling (net income is older)
        _row("C", "controlling_net_income", 20.0, "R2"),
        _row("C", "net_income", 90.0, "R1"),
        _row("C", "total_equity", 200.0, "R2"),
        # N: controlling is older than the equity filing, net income shares it -> net income
        _row("N", "controlling_net_income", 20.0, "R1"),
        _row("N", "net_income", 30.0, "R2"),
        _row("N", "total_equity", 200.0, "R2"),
        # B: both shared -> controlling wins
        _row("B", "controlling_net_income", 20.0, "R2"),
        _row("B", "net_income", 30.0, "R2"),
        _row("B", "total_equity", 200.0, "R2"),
        # X: neither shares the equity filing -> NULL
        _row("X", "controlling_net_income", 20.0, "R1"),
        _row("X", "net_income", 30.0, "R1"),
        _row("X", "total_equity", 200.0, "R2"),
    ]
    out = _roe(_con(rows, ["C", "N", "B", "X"]))
    assert out["C"] == pytest.approx(0.1)
    assert out["N"] == pytest.approx(0.15)
    assert out["B"] == pytest.approx(0.1)
    assert out["X"] is None


def test_default_join_plan_is_coalesce() -> None:
    assert build_fin_v2_sql() == build_fin_v2_sql(join_plan=JOIN_PLAN_COALESCE)
    assert build_fin_v2_sql() != build_fin_v2_sql(join_plan=JOIN_PLAN_OR)


def test_both_join_plans_return_the_same_rows() -> None:
    d = dt.date
    rows = []
    for t in ("A", "B"):
        for seq, (filed, rcept, ni, eq) in enumerate(
            [
                (d(2020, 3, 2), "R1", 10.0, 100.0),
                (d(2020, 5, 1), "R2", 20.0, 150.0),
                (d(2020, 8, 3), "R3", 25.0, -50.0 if t == "B" else 160.0),
            ],
            start=1,
        ):
            rows += [
                _row(t, "net_income", ni, rcept, seq=seq, filed=filed),
                _row(t, "controlling_net_income", ni - 1, rcept, seq=seq, filed=filed),
                _row(t, "total_equity", eq, rcept, seq=seq, filed=filed),
                _row(t, "total_assets", 500.0 + seq, rcept, seq=seq, filed=filed),
                _row(t, "total_liabilities", 300.0 + seq, rcept, seq=seq, filed=filed),
                _row(t, "operating_income", 5.0 * seq, rcept, seq=seq, filed=filed),
                _row(t, "revenue", 100.0 * seq, rcept, seq=seq, filed=filed),
                _row(t, "operating_cash_flow", 7.0 * seq, rcept, seq=seq, filed=filed),
                _row(t, "cash_and_cash_equivalents", 20.0 * seq, rcept, seq=seq, filed=filed),
            ]
    con = _con(rows, ["A", "B"])
    for day in (d(2020, 3, 1), d(2020, 4, 1), d(2020, 9, 1)):
        for t in ("A", "B"):
            con.execute(
                "INSERT INTO dim_stock_pit_daily VALUES (?, ?, 'KOSPI', 1e6, 100, TRUE, FALSE, ?)",
                [day, t, FILED],
            )
            con.execute(
                "INSERT INTO dim_price_quality_daily VALUES (?, ?, 'KOSPI', FALSE, 1)", [day, t]
            )

    def _rows(plan: str) -> list[tuple]:
        sql = build_fin_v2_sql(join_plan=plan)
        return con.execute(f"SELECT * FROM ({sql}) ORDER BY ticker, trade_date").fetchall()

    coalesce, or_plan = _rows(JOIN_PLAN_COALESCE), _rows(JOIN_PLAN_OR)
    assert coalesce == or_plan
    assert any(r[3] is not None for r in coalesce) and any(r[3] is None for r in coalesce)
    assert any(r[4] is not None for r in coalesce)
    assert {r[5] for r in coalesce} >= {True, False}
    assert all(any(r[i] is not None for r in coalesce) for i in range(6, 12))


def _cols(con, *cols: str, plan: str | None = None) -> dict[str, tuple]:
    sql = build_fin_v2_sql() if plan is None else build_fin_v2_sql(join_plan=plan)
    return {
        r[0]: r[1:]
        for r in con.execute(f"SELECT ticker, {', '.join(cols)} FROM ({sql})").fetchall()
    }


def test_roa_same_report_branches_and_assets_guard() -> None:
    rows = [
        # C: controlling shares the assets filing -> controlling
        _row("C", "controlling_net_income", 20.0, "R2"),
        _row("C", "net_income", 90.0, "R1"),
        _row("C", "total_assets", 400.0, "R2"),
        # N: controlling is older, net income shares the assets filing -> net income
        _row("N", "controlling_net_income", 20.0, "R1"),
        _row("N", "net_income", 30.0, "R2"),
        _row("N", "total_assets", 300.0, "R2"),
        # X: neither shares the assets filing -> NULL
        _row("X", "controlling_net_income", 20.0, "R1"),
        _row("X", "net_income", 30.0, "R1"),
        _row("X", "total_assets", 300.0, "R2"),
        # Z / M: assets zero / missing -> NULL
        _row("Z", "net_income", 30.0, "R1"),
        _row("Z", "total_assets", 0.0, "R1"),
        _row("M", "net_income", 30.0, "R1"),
        # E: equity filed in another report than assets; ROA still uses the assets filing
        _row("E", "net_income", 30.0, "R2"),
        _row("E", "total_assets", 600.0, "R2"),
        _row("E", "total_equity", 100.0, "R1"),
    ]
    out = _cols(_con(rows, ["C", "N", "X", "Z", "M", "E"]), "fin_roa_v2")
    assert out["C"][0] == pytest.approx(0.05)
    assert out["N"][0] == pytest.approx(0.1)
    assert out["X"][0] is None
    assert out["Z"][0] is None
    assert out["M"][0] is None
    assert out["E"][0] == pytest.approx(0.05)


def test_roa_is_defined_where_roe_drops_negative_equity() -> None:
    rows = [
        _row("A", "net_income", -30.0, "R1"),
        _row("A", "total_equity", -100.0, "R1"),
        _row("A", "total_assets", 300.0, "R1"),
    ]
    roe, roa = _cols(_con(rows, ["A"]), "fin_roe_v2", "fin_roa_v2")["A"]
    assert roe is None
    assert roa == pytest.approx(-0.1)


def test_capital_impaired_flag() -> None:
    rows = [
        _row("NEG", "net_income", 1.0, "R1"),
        _row("NEG", "total_equity", -5.0, "R1"),
        _row("ZERO", "net_income", 1.0, "R1"),
        _row("ZERO", "total_equity", 0.0, "R1"),
        _row("POS", "net_income", 1.0, "R1"),
        _row("POS", "total_equity", 5.0, "R1"),
        _row("NOEQ", "net_income", 1.0, "R1"),
        _row("HALT", "net_income", 1.0, "R1"),
        _row("HALT", "total_equity", -5.0, "R1"),
    ]
    con = _con(rows, ["NEG", "ZERO", "POS", "NOEQ", "HALT"])
    con.execute("UPDATE dim_price_quality_daily SET is_halted = TRUE WHERE ticker = 'HALT'")
    out = _cols(con, "fin_capital_impaired_v2")
    assert out["NEG"][0] is True
    assert out["ZERO"][0] is True
    assert out["POS"][0] is False
    assert out["NOEQ"][0] is None
    assert out["HALT"][0] is None  # BASE_OK gate


def test_new_columns_follow_the_planned_output() -> None:
    con = _con([_row("A", "net_income", 1.0, "R1")], ["A"])
    cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM ({build_fin_v2_sql()})").fetchall()]
    assert cols == [
        "trade_date",
        "ticker",
        "market",
        "fin_roe_v2",
        "fin_roa_v2",
        "fin_capital_impaired_v2",
        *_RATIOS,
    ]


def _full(ticker: str, rcepts: dict[str, str], values: dict[str, float]) -> list[tuple]:
    """One vintage row per metric in ``values``; ``rcepts`` picks each metric's filing."""
    # net_income decides the basis (CFS), so every ticker carries one
    rows = [_row(ticker, "net_income", 1.0, "R1")]
    return rows + [_row(ticker, m, v, rcepts.get(m, "R1")) for m, v in values.items()]


_BASE_VALUES = {
    "total_equity": 200.0,
    "total_assets": 400.0,
    "total_liabilities": 200.0,
    "operating_income": 50.0,
    "revenue": 500.0,
    "operating_cash_flow": 40.0,
    "cash_and_cash_equivalents": 80.0,
}

_RATIOS = {
    "fin_operating_margin_v2": (("operating_income", "revenue"), 0.1),
    "fin_debt_to_equity_v2": (("total_liabilities", "total_equity"), 1.0),
    "fin_equity_ratio_v2": (("total_equity", "total_assets"), 0.5),
    "fin_ocf_to_assets_v2": (("operating_cash_flow", "total_assets"), 0.1),
    "fin_cash_ratio_v2": (("cash_and_cash_equivalents", "total_assets"), 0.2),
    "fin_asset_turnover_v2": (("revenue", "total_assets"), 1.25),
}


@pytest.mark.parametrize("column", list(_RATIOS))
def test_each_new_ratio_needs_the_same_report(column: str) -> None:
    (num, den), expected = _RATIOS[column]
    rows = _full("OK", {}, _BASE_VALUES)
    # MISS: the numerator metric comes from another filing than the denominator
    rows += _full("MISS", {num: "R0"}, _BASE_VALUES)
    out = _cols(_con(rows, ["OK", "MISS"]), column)
    assert out["OK"][0] == pytest.approx(expected)
    assert out["MISS"][0] is None


@pytest.mark.parametrize(
    ("column", "den", "bad"),
    [
        ("fin_operating_margin_v2", "revenue", 0.0),
        ("fin_operating_margin_v2", "revenue", -10.0),
        ("fin_debt_to_equity_v2", "total_equity", 0.0),
        ("fin_debt_to_equity_v2", "total_equity", -10.0),
        ("fin_equity_ratio_v2", "total_assets", 0.0),
        ("fin_ocf_to_assets_v2", "total_assets", -1.0),
        ("fin_cash_ratio_v2", "total_assets", 0.0),
        ("fin_asset_turnover_v2", "total_assets", 0.0),
    ],
)
def test_new_ratios_are_null_for_a_non_positive_denominator(column, den, bad) -> None:
    values = {**_BASE_VALUES, den: bad}
    out = _cols(_con(_full("A", {}, values), ["A"]), column)
    assert out["A"][0] is None


def test_equity_ratio_is_negative_for_negative_equity() -> None:
    values = {**_BASE_VALUES, "total_equity": -100.0}
    rows = _full("A", {}, values)
    out = _cols(_con(rows, ["A"]), "fin_equity_ratio_v2", "fin_debt_to_equity_v2")
    assert out["A"] == (pytest.approx(-0.25), None)


def test_new_ratios_follow_base_ok_and_missing_metrics() -> None:
    rows = _full("A", {}, _BASE_VALUES) + _full("B", {}, {"total_assets": 400.0})
    con = _con(rows, ["A", "B"])
    con.execute("UPDATE dim_price_quality_daily SET is_halted = TRUE WHERE ticker = 'A'")
    cols = tuple(_RATIOS)
    out = _cols(con, *cols)
    assert out["A"] == (None,) * len(cols)
    assert out["B"] == (None,) * len(cols)


def test_existing_columns_are_unchanged_by_the_new_ratios() -> None:
    rows = [
        _row("A", "controlling_net_income", 20.0, "R1"),
        _row("A", "net_income", 30.0, "R1"),
        *_full("A", {}, _BASE_VALUES)[1:],
    ]
    out = _cols(_con(rows, ["A"]), "fin_roe_v2", "fin_roa_v2", "fin_capital_impaired_v2")
    assert out["A"] == (pytest.approx(0.1), pytest.approx(0.05), False)


def test_fin_scan_sql_text_is_unchanged() -> None:
    assert hashlib.sha256(build_fin_scan_daily_sql().encode()).hexdigest() == (
        "9656de5eb96b9b1498187308a9ea3831a3f0a2f4e29ff45cca7fb061fdd7fb4a"
    )
