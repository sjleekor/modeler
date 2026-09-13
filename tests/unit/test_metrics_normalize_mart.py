"""Golden parity test: DuckDB stock_metric_fact mart == frozen normalize output.

The DuckDB mart must reproduce the Postgres normalize rules exactly. The oracle
(``normalize_stock_metrics``) was removed at refactor P5, so its output on the
synthetic fixture was frozen once into ``golden/stock_metric_fact.json`` and the
mart is checked against that golden — the golden is the source of truth here;
there is no regen path (modeler/ has no Postgres access by design, S6 §2.7). A
rule change must edit the mart and review the golden diff manually. See
refactor plan §3.1, §7.4.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from modeler.etl.marts.metrics_normalize import register_stock_metric_fact_view

from ._metric_fixtures import MockMetricStorage

_GOLDEN_PATH = Path(__file__).parent / "golden" / "stock_metric_fact.json"

# Columns the mart and the Postgres path must agree on (audit timestamps excluded:
# the mart does not carry fetched_at/updated_at — non-deterministic, parity §7.4).
_COMPARE_COLS = (
    "ticker",
    "market",
    "corp_code",
    "metric_code",
    "period_type",
    "period_end",
    "bsns_year",
    "reprt_code",
    "fs_div",
    "value_numeric",
    "value_text",
    "unit",
    "source_table",
    "source_key",
    "mapping_rule_code",
)


def _dec(value: Decimal | None) -> Decimal | None:
    """Normalize to DECIMAL(30,4) scale for exact comparison."""
    if value is None:
        return None
    return Decimal(value).quantize(Decimal("0.0001"))


def _key_str(rec: dict) -> str:
    return "|".join(str(rec[c]) for c in ("ticker", "metric_code", "bsns_year", "reprt_code"))


def _serialize(rec: dict) -> dict:
    out = dict(rec)
    if out.get("period_end") is not None:
        out["period_end"] = str(out["period_end"])
    if out.get("value_numeric") is not None:
        out["value_numeric"] = str(out["value_numeric"])
    return out


def _load_raw_tables(con: duckdb.DuckDBPyConnection, storage: MockMetricStorage) -> None:
    """Materialize the mock's domain rows as raw-schema DuckDB tables."""
    years = [2025]
    reprt_codes = ["11011"]
    tickers = None

    # dart_corp_master
    corps = storage.get_dart_corp_master(active_only=True, tickers=tickers)
    con.execute(
        "CREATE TABLE dart_corp_master ("
        "corp_code VARCHAR, ticker VARCHAR, market VARCHAR, is_active BOOLEAN)"
    )
    for c in corps:
        con.execute(
            "INSERT INTO dart_corp_master VALUES (?, ?, ?, ?)",
            [c.corp_code, c.ticker, c.market.value, c.is_active],
        )

    # dart_financial_statement_raw
    con.execute(
        "CREATE TABLE dart_financial_statement_raw ("
        "ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, fs_div VARCHAR, "
        "sj_div VARCHAR, account_id VARCHAR, account_nm VARCHAR, "
        "thstrm_amount DECIMAL(30,4), ord BIGINT, rcept_no VARCHAR)"
    )
    for r in storage.get_dart_financial_statement_raw(years, reprt_codes, tickers):
        con.execute(
            "INSERT INTO dart_financial_statement_raw VALUES (?,?,?,?,?,?,?,?,?,?)",
            [
                r.ticker,
                r.bsns_year,
                r.reprt_code,
                r.fs_div,
                r.sj_div,
                r.account_id,
                r.account_nm,
                r.thstrm_amount,
                r.ord,
                r.rcept_no,
            ],
        )

    # dart_share_count_raw
    con.execute(
        "CREATE TABLE dart_share_count_raw ("
        "ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, se VARCHAR, "
        "istc_totqy BIGINT, tesstk_co BIGINT, stlm_dt DATE, rcept_no VARCHAR)"
    )
    for r in storage.get_dart_share_count_raw(years, reprt_codes, tickers):
        con.execute(
            "INSERT INTO dart_share_count_raw VALUES (?,?,?,?,?,?,?,?)",
            [
                r.ticker,
                r.bsns_year,
                r.reprt_code,
                r.se,
                r.istc_totqy,
                r.tesstk_co,
                r.stlm_dt,
                r.rcept_no,
            ],
        )

    # dart_shareholder_return_raw
    con.execute(
        "CREATE TABLE dart_shareholder_return_raw ("
        "ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, statement_type VARCHAR, "
        "row_name VARCHAR, stock_knd VARCHAR, dim1 VARCHAR, dim2 VARCHAR, dim3 VARCHAR, "
        "metric_code VARCHAR, value_numeric DECIMAL(30,4), stlm_dt DATE, rcept_no VARCHAR)"
    )
    for r in storage.get_dart_shareholder_return_raw(years, reprt_codes, tickers):
        con.execute(
            "INSERT INTO dart_shareholder_return_raw VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                r.ticker,
                r.bsns_year,
                r.reprt_code,
                r.statement_type,
                r.row_name,
                r.stock_knd,
                r.dim1,
                r.dim2,
                r.dim3,
                r.metric_code,
                r.value_numeric,
                r.stlm_dt,
                r.rcept_no,
            ],
        )

    # dart_xbrl_fact_raw (dimensions stored as the JSON-array VARCHAR the exporter writes)
    con.execute(
        "CREATE TABLE dart_xbrl_fact_raw ("
        "ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, concept_id VARCHAR, "
        "concept_name VARCHAR, label_ko VARCHAR, context_id VARCHAR, "
        "period_end DATE, instant_date DATE, dimensions VARCHAR, "
        "value_numeric DECIMAL(30,4), rcept_no VARCHAR)"
    )
    for r in storage.get_dart_xbrl_fact_raw(years, reprt_codes, tickers):
        dims = "[" + ",".join(f'"{d}"' for d in r.dimensions) + "]"
        con.execute(
            "INSERT INTO dart_xbrl_fact_raw VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                r.ticker,
                r.bsns_year,
                r.reprt_code,
                r.concept_id,
                r.concept_name,
                r.label_ko,
                r.context_id,
                r.period_end,
                r.instant_date,
                dims,
                r.value_numeric,
                r.rcept_no,
            ],
        )


def _mart_facts() -> dict[str, dict]:
    con = duckdb.connect()
    _load_raw_tables(con, MockMetricStorage())
    register_stock_metric_fact_view(con, view_name="smf")
    cols = ", ".join(_COMPARE_COLS)
    rows = con.execute(f"SELECT {cols} FROM smf").fetchall()
    out: dict[str, dict] = {}
    for row in rows:
        rec = dict(zip(_COMPARE_COLS, row))
        rec["value_numeric"] = _dec(rec["value_numeric"])
        out[_key_str(rec)] = _serialize(rec)
    return out


def _load_golden() -> dict[str, dict]:
    return json.loads(_GOLDEN_PATH.read_text())


@pytest.fixture(scope="module")
def golden_and_mart():
    return _load_golden(), _mart_facts()


def test_mart_selects_identical_keys(golden_and_mart):
    golden, mart = golden_and_mart
    assert set(mart) == set(golden)


def test_mart_matches_golden_on_all_compare_columns(golden_and_mart):
    golden, mart = golden_and_mart
    mismatches: list[str] = []
    for key, g_rec in golden.items():
        m_rec = mart[key]
        for col in _COMPARE_COLS:
            m_val, g_val = m_rec[col], g_rec[col]
            # value_text: the synthetic fixture builds scaleless Decimals
            # (Decimal("100") -> "100") whereas real NUMERIC(30,4) raw columns
            # and the mart's CAST carry four places. Compare numerically.
            if col == "value_text":
                if Decimal(m_val) != Decimal(g_val):
                    mismatches.append(f"{key} col={col}: mart={m_val!r} golden={g_val!r}")
                continue
            if m_val != g_val:
                mismatches.append(f"{key} col={col}: mart={m_val!r} golden={g_val!r}")
    assert not mismatches, "\n".join(mismatches)


def test_mart_picks_cfs_cis_net_income_priority(golden_and_mart):
    """Spot-check the trickiest rule: CFS/CIS ProfitLoss wins over OFS/IS."""
    _, mart = golden_and_mart
    key = "005930|net_income|2025|11011"
    if key in mart:
        assert mart[key]["mapping_rule_code"].startswith("fin.net_income.cfs")
