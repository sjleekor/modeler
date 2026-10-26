"""Synthetic tests of rule R1 (``fin_quarterly_metric_vintage_dlr``, 18 §8.1).

A direct_interim Q2/Q3 value whose filing's quarterly and cumulative values disagree keeps the
reported quarterly value (the original makes it NULL); the conflict flag stays TRUE.
"""

from __future__ import annotations

import duckdb
import pytest

from modeler.etl.marts import financial_quarters as fq
from modeler.etl.marts.metric_vintages_delisted import (
    FQMV_DL_TABLE,
    FQMV_DLR_TABLE,
    OriginalSqlDriftError,
    build_fin_quarterly_metric_vintage_dl_sql,
    build_fin_quarterly_metric_vintage_dlr_sql,
    register_fin_quarterly_metric_vintage_dl_view,
    register_fin_quarterly_metric_vintage_dlr_view,
)
from tests.unit.test_metric_vintages_delisted import (
    _CORP_CODE,
    _con,
    _corp,
    _four_reports,
    _fs,
    _register_both,
)

CONFLICT = "900010"


def _fixture() -> duckdb.DuckDBPyConnection:
    _CORP_CODE.setdefault(CONFLICT, "00999010")
    con = _con()
    _corp(con, "005930", "KOSPI")
    _corp(con, CONFLICT, "KOSPI")
    _CORP_CODE.setdefault("005930", "00126380")
    _four_reports(con, "005930", 1)
    # Q2: quarterly 150, cumulative 260 (1H should be 250) -> conflict; Q3 also off.
    _fs(con, CONFLICT, bsns_year=2023, reprt_code="11013", amount=100, rcept_no="20230410000010")
    _fs(con, CONFLICT, bsns_year=2023, reprt_code="11012", amount=150, add=260,
        rcept_no="20230810000010")  # fmt: skip
    _fs(con, CONFLICT, bsns_year=2023, reprt_code="11014", amount=140, add=400,
        rcept_no="20231110000010")  # fmt: skip
    _fs(con, CONFLICT, bsns_year=2023, reprt_code="11011", amount=560, rcept_no="20240315000010")
    return con


@pytest.fixture
def con() -> duckdb.DuckDBPyConnection:
    c = _fixture()
    _register_both(c)
    register_fin_quarterly_metric_vintage_dl_view(c)
    register_fin_quarterly_metric_vintage_dlr_view(c)
    return c


def _rows(con, view: str, ticker: str) -> dict[str, tuple]:
    return {
        r[0]: r[1:]
        for r in con.execute(
            f"SELECT quarter, standalone_value, standalone_source_conflict, ttm_value, "
            f"available_from, ttm_available_from FROM {view} "
            f"WHERE ticker=? AND metric_code='revenue' AND bsns_year=2023",
            [ticker],
        ).fetchall()
    }


def test_conflict_q2_q3_keep_reported_value_and_flag(con) -> None:
    dl = _rows(con, FQMV_DL_TABLE, CONFLICT)
    dlr = _rows(con, FQMV_DLR_TABLE, CONFLICT)
    for q, val in (("Q2", 150), ("Q3", 140)):
        assert dl[q][0] is None and dl[q][1] is True
        assert float(dlr[q][0]) == val and dlr[q][1] is True


def test_q1_q4_and_available_dates_identical(con) -> None:
    dl = _rows(con, FQMV_DL_TABLE, CONFLICT)
    dlr = _rows(con, FQMV_DLR_TABLE, CONFLICT)
    assert dl["Q1"] == dlr["Q1"]
    assert dl["Q4"][:2] == dlr["Q4"][:2]  # standalone value and flag (TTM differs, below)
    for q in dl:
        assert dl[q][3] == dlr[q][3]  # available_from
    assert float(dlr["Q4"][0]) == 170.0  # raw q1..q3, unchanged
    # the TTM was NULL under the original (Q2/Q3 NULL) and is complete under R1
    assert dl["Q4"][2] is None and float(dlr["Q4"][2]) == 560.0


def test_non_conflict_rows_identical_to_dl(con) -> None:
    cols = (
        "ticker, market, corp_code, metric_code, fs_basis, bsns_year, quarter, quarter_ordinal, "
        "seq_key, statement_period_end, rcept_no, available_from, metric_kind, standalone_value, "
        "standalone_source_conflict, cumulative_derived_value, comparative_q_amount, "
        "value_lag_4q, ttm_value, ttm_complete, ttm_available_from"
    )
    a = con.execute(
        f"SELECT {cols} FROM {FQMV_DL_TABLE} WHERE ticker <> '{CONFLICT}' ORDER BY ALL"
    ).fetchall()
    b = con.execute(
        f"SELECT {cols} FROM {FQMV_DLR_TABLE} WHERE ticker <> '{CONFLICT}' ORDER BY ALL"
    ).fetchall()
    assert a == b and a
    # every conflict-ticker row differs only where the original had NULL standalone/ttm
    d = con.execute(
        f"SELECT count(*) FROM {FQMV_DL_TABLE} l JOIN {FQMV_DLR_TABLE} r "
        f"USING (ticker, metric_code, fs_basis, bsns_year, quarter) "
        f"WHERE l.ticker='{CONFLICT}' AND l.standalone_value IS NOT NULL "
        f"AND l.standalone_value IS DISTINCT FROM r.standalone_value"
    ).fetchone()[0]
    assert d == 0


def test_ttm_includes_conflict(con) -> None:
    dlr = con.execute(
        f"SELECT ticker, quarter, ttm_value IS NOT NULL, ttm_includes_conflict FROM "
        f"{FQMV_DLR_TABLE} WHERE metric_code='revenue' AND bsns_year=2023"
    ).fetchall()
    got = {(t, q): (has, f) for t, q, has, f in dlr}
    assert got[(CONFLICT, "Q4")] == (True, True)
    assert got[("005930", "Q4")] == (True, False)
    # no TTM yet at Q1..Q3 of the first year -> NULL flag
    assert got[(CONFLICT, "Q3")] == (False, None)


def test_old_marts_sql_unchanged() -> None:
    base = fq.build_fin_quarterly_metric_vintage_sql(vintage_view="stock_metric_vintage_fact_dl")
    assert build_fin_quarterly_metric_vintage_dl_sql() == base
    assert "ttm_includes_conflict" not in base
    assert base.count("THEN NULL\n                         ELSE q2_value END") == 1
    assert build_fin_quarterly_metric_vintage_dlr_sql() != base


def _drift(monkeypatch, old: str, new: str) -> None:
    orig = fq.build_fin_quarterly_metric_vintage_sql

    def patched(**kw):
        return orig(**kw).replace(old, new)

    monkeypatch.setattr(fq, "build_fin_quarterly_metric_vintage_sql", patched)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("THEN NULL\n                         ELSE q2_value END", "THEN NULL ELSE q2_value END"),
        ("THEN NULL\n                         ELSE q3_value END", "THEN NULL ELSE q3_value END"),
        ("n4.standalone_value AS neighbor4_value\n", "n4.standalone_value AS neighbor4\n"),
        (
            "END AS ttm_available_from\n    FROM with_neighbors",
            "END AS ttm_avail\n    FROM with_neighbors",
        ),
    ],
)
def test_guard_raises_on_text_drift(monkeypatch, old: str, new: str) -> None:
    _drift(monkeypatch, old, new)
    with pytest.raises(OriginalSqlDriftError):
        build_fin_quarterly_metric_vintage_dlr_sql()
