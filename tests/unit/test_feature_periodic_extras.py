"""N6 final-vintage periodic feature contracts."""

from __future__ import annotations

import json
import math
from datetime import date

import duckdb
import pytest
from modeler.etl.features.periodic_extras import build_periodic_extras_sql


def _payload(**values: str) -> str:
    return json.dumps(values, ensure_ascii=False)


def _run(*, structural_change: bool = False) -> dict[date, dict]:
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE dim_universe_daily("
        "trade_date DATE, ticker VARCHAR, market VARCHAR, in_universe BOOLEAN)"
    )
    sessions = [date(2025, 3, 31), date(2025, 4, 1), date(2025, 4, 2)]
    con.executemany(
        "INSERT INTO dim_universe_daily VALUES (?, 'A', 'KOSPI', TRUE)",
        [(session,) for session in sessions],
    )
    con.execute(
        "CREATE TABLE dart_employee_raw("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, rcept_no VARCHAR, "
        "statement_type VARCHAR, raw_payload VARCHAR)"
    )
    employee_rows = [
        ("C", "A", 2023, "20240329000001", "employee", _payload(fo_bbm="남", sm="40")),
        ("C", "A", 2023, "20240329000001", "employee", _payload(fo_bbm="여", sm="60")),
        ("C", "A", 2024, "20250331000001", "employee", _payload(fo_bbm="합계", sm="150")),
        ("C", "A", 2024, "20250331000001", "employee", _payload(fo_bbm="남", sm="70")),
        ("C", "A", 2024, "20250331000001", "employee", _payload(fo_bbm="여", sm="80")),
    ]
    con.executemany("INSERT INTO dart_employee_raw VALUES (?, ?, ?, ?, ?, ?)", employee_rows)

    con.execute(
        "CREATE TABLE dart_governance_raw("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, rcept_no VARCHAR, "
        "statement_type VARCHAR, raw_payload VARCHAR)"
    )
    governance_rows = [
        (
            "C", "A", 2023, "20240329000001", "major_shareholder",
            _payload(nm="계", stock_knd="보통주", trmend_posesn_stock_qota_rt="40"),
        ),
        (
            "C", "A", 2024, "20250331000001", "major_shareholder",
            _payload(nm="계", stock_knd="보통주", trmend_posesn_stock_qota_rt="42"),
        ),
        (
            "C", "A", 2024, "20250331000001", "major_shareholder",
            _payload(nm="홍길동", stock_knd="보통주", trmend_posesn_stock_qota_rt="90"),
        ),
    ]
    con.executemany("INSERT INTO dart_governance_raw VALUES (?, ?, ?, ?, ?, ?)", governance_rows)

    con.execute(
        "CREATE TABLE dart_filing_receipt_raw("
        "corp_code VARCHAR, rcept_no VARCHAR, rcept_dt DATE, report_nm VARCHAR)"
    )
    receipts = [
        ("C", "20240329000001", date(2024, 3, 29), "사업보고서"),
        ("C", "20250331000001", date(2025, 3, 31), "사업보고서"),
    ]
    if structural_change:
        receipts.append(("C", "20240601000001", date(2024, 6, 1), "합병등종료보고서(합병)"))
    con.executemany("INSERT INTO dart_filing_receipt_raw VALUES (?, ?, ?, ?)", receipts)

    con.execute(
        "CREATE TABLE stock_metric_vintage_fact("
        "ticker VARCHAR, bsns_year INTEGER, metric_code VARCHAR, reprt_code VARCHAR, "
        "value_numeric DOUBLE, fs_basis VARCHAR, available_from DATE, rcept_no VARCHAR)"
    )
    con.execute(
        "INSERT INTO stock_metric_vintage_fact VALUES "
        "('A', 2024, 'revenue', '11011', 150000, 'CFS', DATE '2025-04-02', "
        "'20250331000001')"
    )

    result = con.execute(
        build_periodic_extras_sql(universe_view="dim_universe_daily")
    ).fetchall()
    columns = [description[0] for description in con.description]
    return {row[0]: dict(zip(columns, row, strict=True)) for row in result}


def test_receipt_values_start_on_the_next_session_and_summary_rows_win() -> None:
    rows = _run()

    assert rows[date(2025, 3, 31)]["hc_employee_growth_yoy"] is None
    assert rows[date(2025, 4, 1)]["hc_employee_growth_yoy"] == pytest.approx(0.5)
    assert rows[date(2025, 4, 1)]["own_major_stake"] == 42
    assert rows[date(2025, 4, 1)]["own_major_stake_chg"] == 2


def test_productivity_waits_for_both_employee_and_revenue_availability() -> None:
    rows = _run()

    assert rows[date(2025, 4, 1)]["hc_revenue_per_employee"] is None
    assert rows[date(2025, 4, 2)]["hc_revenue_per_employee"] == pytest.approx(
        math.log(150000 / 150)
    )


def test_structural_change_and_large_growth_mask_the_employee_signal() -> None:
    rows = _run(structural_change=True)

    assert rows[date(2025, 4, 1)]["hc_employee_growth_yoy"] is None


def test_lag_variants_and_final_vintage_warning_are_emitted() -> None:
    rows = _run()

    assert rows[date(2025, 4, 1)]["hc_employee_growth_yoy_lag1"] is None
    assert rows[date(2025, 4, 2)]["hc_employee_growth_yoy_lag1"] == pytest.approx(0.5)
    assert rows[date(2025, 4, 2)]["periodic_source_warning"] == "final_vintage"
    assert rows[date(2025, 4, 2)]["vintage_capture_ratio"] == pytest.approx(0.0184)
