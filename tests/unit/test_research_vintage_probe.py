"""§4.4.1 vintage distance probe — metric 1/2 builders on synthetic vintages."""

from __future__ import annotations

from datetime import date

import duckdb

from modeler.etl.vintage_probe import (
    build_vintage_diff_summary_sql,
    build_vintage_row_diff_sql,
    build_vintage_window_diff_sql,
    measure_identity_pass_rate,
)

OLD_RCEPT = "20230310000002"  # FY2022 annual report
NEW_RCEPT = "20240310000003"  # FY2023 annual report -- the newest vintage
OLD_YEAR = 2022
NEW_YEAR = 2023


def _weekdays(year: int) -> list[date]:
    out = []
    for month in range(1, 13):
        for day in range(1, 32):
            try:
                d = date(year, month, day)
            except ValueError:
                continue
            if d.weekday() < 5:
                out.append(d)
    return out


_TRADING_DAYS = [d for year in range(2019, 2025) for d in _weekdays(year)]


def _con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE dart_corp_master (ticker VARCHAR, market VARCHAR, corp_code VARCHAR)")
    con.execute(
        "CREATE TABLE dart_share_count_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, se VARCHAR, istc_totqy BIGINT, tesstk_co BIGINT, "
        "now_to_isu_stock_totqy BIGINT, now_to_dcrs_stock_totqy BIGINT, stlm_dt DATE)"
    )
    con.execute(
        "CREATE TABLE dart_capital_change_raw ("
        "corp_code VARCHAR, ticker VARCHAR, bsns_year INTEGER, reprt_code VARCHAR, "
        "rcept_no VARCHAR, isu_dcrs_de DATE, isu_dcrs_stle VARCHAR, "
        "isu_dcrs_stock_knd VARCHAR, isu_dcrs_qy BIGINT)"
    )
    return con


def _add_ticker(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    *,
    years: range = range(2019, 2024),
    issued: int = 1000,
    now_to_isu: int = 500,
) -> None:
    """Flat share counts across annual positions, so windows exist for every year."""
    con.execute("INSERT INTO dart_corp_master VALUES (?, 'KOSPI', ?)", [ticker, f"corp{ticker}"])
    for year in years:
        con.execute(
            "INSERT INTO dart_share_count_raw VALUES (?,?,?,'11011',?,'합계',?,0,?,0,?)",
            [
                f"corp{ticker}",
                ticker,
                year,
                f"{year + 1}0310000001",
                issued,
                now_to_isu,
                date(year, 12, 31),
            ],
        )


def _add_event(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    *,
    bsns_year: int,
    rcept_no: str,
    isu_dcrs_de: date,
    isu_dcrs_qy: int,
    isu_dcrs_stle: str = "유상증자(일반공모)",
) -> None:
    con.execute(
        "INSERT INTO dart_capital_change_raw VALUES (?,?,?,'11011',?,?,?,'보통주',?)",
        [
            f"corp{ticker}",
            ticker,
            bsns_year,
            rcept_no,
            isu_dcrs_de,
            isu_dcrs_stle,
            isu_dcrs_qy,
        ],
    )


def _add_event_in_both_vintages(
    con: duckdb.DuckDBPyConnection,
    ticker: str,
    *,
    old_date: date,
    new_date: date | None = None,
    old_qty: int = 100,
    new_qty: int | None = None,
) -> None:
    _add_event(
        con,
        ticker,
        bsns_year=OLD_YEAR,
        rcept_no=OLD_RCEPT,
        isu_dcrs_de=old_date,
        isu_dcrs_qy=old_qty,
    )
    _add_event(
        con,
        ticker,
        bsns_year=NEW_YEAR,
        rcept_no=NEW_RCEPT,
        isu_dcrs_de=new_date or old_date,
        isu_dcrs_qy=old_qty if new_qty is None else new_qty,
    )


def _windows(con: duckdb.DuckDBPyConnection, ticker: str) -> dict[int, bool]:
    rows = con.execute(
        f"SELECT window_bsns_year, feature_changing FROM ({build_vintage_window_diff_sql()}) "
        f"WHERE ticker = ? ORDER BY 1",
        [ticker],
    ).fetchall()
    return {year: changed for year, changed in rows}


def test_identical_vintages_report_no_feature_change() -> None:
    con = _con()
    _add_ticker(con, "000001")
    _add_event_in_both_vintages(con, "000001", old_date=date(2020, 6, 15))

    # Comparable windows stop at the older vintage's settlement date.
    assert _windows(con, "000001") == {2020: False, 2021: False, 2022: False}


def test_date_move_inside_the_window_is_not_a_feature_change() -> None:
    con = _con()
    _add_ticker(con, "000002")
    _add_event_in_both_vintages(
        con, "000002", old_date=date(2021, 1, 31), new_date=date(2021, 1, 13)
    )

    # 000040's real correction: same trailing-year window, so the feature is
    # identical either way and the probe must not count it as disagreement.
    assert _windows(con, "000002") == {2020: False, 2021: False, 2022: False}


def test_date_move_across_the_window_boundary_is_a_feature_change() -> None:
    con = _con()
    _add_ticker(con, "000003")
    _add_event_in_both_vintages(
        con, "000003", old_date=date(2022, 1, 15), new_date=date(2021, 12, 20)
    )

    # The event leaves the 2022 window and lands in the 2021 one -- both differ.
    assert _windows(con, "000003") == {2020: False, 2021: True, 2022: True}


def test_quantity_correction_is_a_feature_change() -> None:
    con = _con()
    _add_ticker(con, "000004")
    _add_event_in_both_vintages(
        con, "000004", old_date=date(2021, 6, 15), old_qty=100, new_qty=150
    )

    assert _windows(con, "000004") == {2020: False, 2021: True, 2022: False}


def test_event_added_by_the_newer_vintage_is_a_feature_change() -> None:
    con = _con()
    _add_ticker(con, "000005")
    # The FY2022 report exists but listed nothing -- OpenDART returns a single
    # '-' placeholder row for that. Without it there is no older vintage to
    # compare against and the pair drops out entirely.
    con.execute(
        "INSERT INTO dart_capital_change_raw VALUES "
        "('corp000005','000005',?,'11011',?,NULL,'-','',NULL)",
        [OLD_YEAR, OLD_RCEPT],
    )
    _add_event(
        con,
        "000005",
        bsns_year=NEW_YEAR,
        rcept_no=NEW_RCEPT,
        isu_dcrs_de=date(2021, 6, 15),
        isu_dcrs_qy=100,
    )

    assert _windows(con, "000005") == {2020: False, 2021: True, 2022: False}


def test_summary_reports_rate_by_vintage_distance() -> None:
    con = _con()
    _add_ticker(con, "000001")
    _add_event_in_both_vintages(con, "000001", old_date=date(2020, 6, 15))
    _add_ticker(con, "000004")
    _add_event_in_both_vintages(
        con, "000004", old_date=date(2021, 6, 15), old_qty=100, new_qty=150
    )

    row = con.execute(build_vintage_diff_summary_sql()).fetchone()
    distance, tickers, compared, changed, rate = row
    assert distance == NEW_YEAR - OLD_YEAR
    assert tickers == 2
    assert compared == 6  # two tickers x three comparable windows
    assert changed == 1
    assert rate == 1 / 6


def test_row_diff_counts_events_missing_from_the_newest_vintage() -> None:
    con = _con()
    _add_ticker(con, "000004")
    _add_event_in_both_vintages(
        con, "000004", old_date=date(2021, 6, 15), old_qty=100, new_qty=150
    )
    _add_ticker(con, "000001")
    _add_event_in_both_vintages(con, "000001", old_date=date(2020, 6, 15))

    distance, old_events, absent = con.execute(build_vintage_row_diff_sql()).fetchone()
    assert distance == NEW_YEAR - OLD_YEAR
    assert old_events == 2
    assert absent == 1  # the re-quantified one no longer exists as filed


def test_identity_pass_rate_separates_the_two_policies() -> None:
    con = _con()
    # 1000 -> 1100 across FY2022, explained by a single 100-share issuance that
    # only the FY2023 report lists. latest_vintage can see it, strict_pit cannot.
    con.execute("INSERT INTO dart_corp_master VALUES ('000006', 'KOSPI', 'corp000006')")
    for year, issued, now_to_isu in ((2021, 1000, 500), (2022, 1100, 600)):
        con.execute(
            "INSERT INTO dart_share_count_raw VALUES "
            "('corp000006','000006',?,'11011',?,'합계',?,0,?,0,?)",
            [year, f"{year + 1}0310000001", issued, now_to_isu, date(year, 12, 31)],
        )
    _add_event(
        con,
        "000006",
        bsns_year=NEW_YEAR,
        rcept_no=NEW_RCEPT,
        isu_dcrs_de=date(2022, 6, 15),
        isu_dcrs_qy=100,
    )

    measured = measure_identity_pass_rate(con, trading_days=_TRADING_DAYS)
    rows = {r["vintage_policy"]: r for r in measured}
    assert rows["latest_vintage"]["feature_available"] == 1
    assert rows["latest_vintage"]["feature_available_rate"] == 1.0
    assert rows["strict_pit"]["feature_available"] == 0
    assert rows["strict_pit"]["feature_available_rate"] == 0.0
