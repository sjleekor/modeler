"""B-10 Stage 2 — the five mart-side quality/coverage diagnostics."""

from __future__ import annotations

from datetime import date

import duckdb
from modeler.etl.phase_b_coverage import (
    register_event_coverage_view,
    register_feature_coverage_view,
    register_quarterly_metric_quality_view,
    register_receipt_value_pairing_quality_view,
    register_stock_metric_vintage_quality_view,
)

from collector.kr.definitions.metric_rules import default_metric_mapping_rules

_VINTAGE_COLUMNS = (
    "ticker VARCHAR, market VARCHAR, metric_code VARCHAR, period_type VARCHAR, "
    "statement_period_end DATE, bsns_year INTEGER, reprt_code VARCHAR, fs_basis VARCHAR, "
    "rcept_no VARCHAR, disclosed_date DATE, available_from DATE, availability_source VARCHAR, "
    "source_table VARCHAR, mapping_rule_code VARCHAR, mapping_priority INTEGER, "
    "period_end_source VARCHAR, period_end_conflict BOOLEAN, is_revision BOOLEAN, "
    "captured_vintage_status VARCHAR, receipt_value_pairing_status VARCHAR, "
    "pairing_tolerance DOUBLE"
)


def _rows(con: duckdb.DuckDBPyConnection, view: str) -> list[dict]:
    result = con.execute(f"SELECT * FROM {view}")
    columns = [d[0] for d in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


def _catalog_best_priority(metric_code: str, fs_basis: str = "") -> int:
    return min(
        rule.priority
        for rule in default_metric_mapping_rules()
        if rule.is_active and rule.metric_code == metric_code and rule.fs_div in ("", fs_basis)
    )


def _vintage_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE TABLE stock_metric_vintage_fact ({_VINTAGE_COLUMNS})")
    return con


def _add_vintage(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str = "005930",
    metric_code: str = "revenue",
    statement_period_end: date = date(2023, 12, 31),
    bsns_year: int = 2023,
    reprt_code: str = "11011",
    fs_basis: str = "CFS",
    rcept_no: str = "20240310000001",
    availability_source: str = "rcept_no",
    source_table: str = "dart_financial_statement_raw",
    mapping_rule_code: str = "rule_a",
    mapping_priority: int = 1,
    period_end_source: str = "xbrl",
    period_end_conflict: bool = False,
    is_revision: bool | None = False,
    captured_vintage_status: str = "captured_vintages_only",
    receipt_value_pairing_status: str = "verified_same_receipt",
) -> None:
    con.execute(
        "INSERT INTO stock_metric_vintage_fact VALUES "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ticker,
            "KOSPI",
            metric_code,
            "annual" if reprt_code == "11011" else "q1",
            statement_period_end,
            bsns_year,
            reprt_code,
            fs_basis,
            rcept_no,
            date(2024, 3, 10),
            date(2024, 3, 11),
            availability_source,
            source_table,
            mapping_rule_code,
            mapping_priority,
            period_end_source,
            period_end_conflict,
            is_revision,
            captured_vintage_status,
            receipt_value_pairing_status,
            0.0,
        ],
    )


# --------------------------------------------------------------------------
# receipt_value_pairing_quality
# --------------------------------------------------------------------------


def test_pairing_ratios_use_the_applicable_rows_as_denominator() -> None:
    con = _vintage_con()
    # Two applicable rows, one verified and one mismatched ...
    _add_vintage(con, rcept_no="20240310000001")
    _add_vintage(
        con,
        rcept_no="20240310000002",
        receipt_value_pairing_status="value_mismatch",
    )
    # ... plus an XBRL-sourced row that pairing cannot judge at all. Counting it
    # in the denominator would make this year look half-verified instead of
    # half-mismatched.
    _add_vintage(
        con,
        rcept_no="20240310000003",
        source_table="dart_xbrl_fact_raw",
        receipt_value_pairing_status="not_applicable",
    )
    register_receipt_value_pairing_quality_view(con)
    (row,) = _rows(con, "receipt_value_pairing_quality")

    assert row["rows"] == 3
    assert row["applicable_rows"] == 2
    assert row["verified_ratio"] == 0.5
    assert row["value_mismatch_ratio"] == 0.5
    assert row["receipts_with_value_mismatch"] == 1
    assert row["pairing_tolerance"] == 0.0


def test_pairing_ratio_is_null_when_nothing_is_applicable() -> None:
    con = _vintage_con()
    _add_vintage(
        con,
        source_table="dart_xbrl_fact_raw",
        receipt_value_pairing_status="not_applicable",
    )
    register_receipt_value_pairing_quality_view(con)
    (row,) = _rows(con, "receipt_value_pairing_quality")

    # Not 0.0 — "nothing to verify here" is not the same as "verified none".
    assert row["applicable_rows"] == 0
    assert row["verified_ratio"] is None
    assert row["value_mismatch_ratio"] is None


def test_pairing_years_without_rows_are_absent_not_zero_filled() -> None:
    con = _vintage_con()
    _add_vintage(con, bsns_year=2023)
    _add_vintage(con, bsns_year=2025, rcept_no="20260310000001")
    register_receipt_value_pairing_quality_view(con)

    assert [row["bsns_year"] for row in _rows(con, "receipt_value_pairing_quality")] == [2023, 2025]


# --------------------------------------------------------------------------
# stock_metric_vintage_quality
# --------------------------------------------------------------------------


def test_revision_ratio_ignores_rows_whose_receipt_never_matched() -> None:
    con = _vintage_con()
    _add_vintage(con, rcept_no="20240310000001", is_revision=False)
    _add_vintage(con, rcept_no="20240310000002", is_revision=True)
    # Unmatched receipt: is_revision is unknown, not False. Folding it into the
    # denominator would push the ratio down exactly where receipt coverage is
    # worst.
    _add_vintage(
        con,
        rcept_no="20240310000003",
        is_revision=None,
        captured_vintage_status="unlinked_receipt",
    )
    register_stock_metric_vintage_quality_view(con)
    (row,) = _rows(con, "stock_metric_vintage_quality")

    assert row["rows"] == 3
    assert row["revision_known_rows"] == 2
    assert row["revision_rows"] == 1
    assert row["revision_ratio"] == 0.5
    assert row["unlinked_receipt_rows"] == 1


def test_mapping_fallback_is_measured_against_the_catalog_not_the_data() -> None:
    con = _vintage_con()
    best = _catalog_best_priority("revenue", "CFS")
    # Every row came from a worse-than-preferred rule. Measured against the
    # data's own minimum this would read 0.0; against the catalog it is 1.0,
    # which is the fact that matters.
    _add_vintage(con, rcept_no="20240310000001", mapping_priority=best + 5)
    _add_vintage(con, rcept_no="20240310000002", mapping_priority=best + 9)
    register_stock_metric_vintage_quality_view(con)
    (row,) = _rows(con, "stock_metric_vintage_quality")

    assert row["catalog_best_priority"] == best
    assert row["observed_best_priority"] == best + 5
    assert row["mapping_fallback_rows"] == 2
    assert row["mapping_fallback_ratio"] == 1.0


def test_multiple_receipts_on_one_position_count_as_one_multi_vintage_position() -> None:
    con = _vintage_con()
    _add_vintage(con, rcept_no="20240310000001")
    _add_vintage(con, rcept_no="20240515000002", is_revision=True)
    _add_vintage(con, ticker="000660", rcept_no="20240311000001")
    register_stock_metric_vintage_quality_view(con)
    (row,) = _rows(con, "stock_metric_vintage_quality")

    assert row["rows"] == 3
    assert row["positions"] == 2
    assert row["multi_vintage_positions"] == 1
    assert row["multi_vintage_ratio"] == 0.5


def test_synthetic_availability_is_reported_separately() -> None:
    con = _vintage_con()
    _add_vintage(con, rcept_no="20240310000001")
    _add_vintage(con, rcept_no="", availability_source="synthetic_fallback")
    register_stock_metric_vintage_quality_view(con)
    (row,) = _rows(con, "stock_metric_vintage_quality")

    assert row["synthetic_availability_rows"] == 1
    assert row["synthetic_availability_ratio"] == 0.5


# --------------------------------------------------------------------------
# quarterly_metric_quality
# --------------------------------------------------------------------------


def _quarterly_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE fin_quarterly_metric_vintage ("
        "ticker VARCHAR, metric_code VARCHAR, fs_basis VARCHAR, bsns_year INTEGER, "
        "quarter VARCHAR, metric_kind VARCHAR, standalone_value DOUBLE, "
        "standalone_source_conflict BOOLEAN, cumulative_derived_value DOUBLE, "
        "comparative_q_amount DOUBLE, value_lag_4q DOUBLE, ttm_complete BOOLEAN, "
        "available_from DATE, ttm_available_from DATE)"
    )
    return con


def _add_quarter(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str = "005930",
    metric_code: str = "revenue",
    bsns_year: int = 2023,
    quarter: str = "Q1",
    metric_kind: str = "direct_interim",
    standalone_value: float | None = 100.0,
    standalone_source_conflict: bool | None = False,
    ttm_complete: bool = True,
    available_from: date = date(2023, 5, 16),
    ttm_available_from: date | None = date(2023, 5, 16),
) -> None:
    con.execute(
        "INSERT INTO fin_quarterly_metric_vintage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ticker,
            metric_code,
            "CFS",
            bsns_year,
            quarter,
            metric_kind,
            standalone_value,
            standalone_source_conflict,
            None,
            None,
            None,
            ttm_complete,
            available_from,
            ttm_available_from,
        ],
    )


def test_conflict_ratio_is_taken_over_checkable_rows_only() -> None:
    con = _quarterly_con()
    _add_quarter(con, ticker="005930", standalone_source_conflict=True)
    _add_quarter(con, ticker="000660", standalone_source_conflict=False)
    # An instant metric has no second source to disagree with, so its flag is
    # NULL. Including it would halve the measured conflict rate of a kind that
    # cannot conflict in the first place.
    _add_quarter(
        con,
        ticker="035420",
        metric_kind="instant",
        standalone_source_conflict=None,
    )
    register_quarterly_metric_quality_view(con)
    (row,) = _rows(con, "quarterly_metric_quality")

    assert row["rows"] == 3
    assert row["conflict_checkable_rows"] == 2
    assert row["standalone_conflict_rows"] == 1
    assert row["standalone_conflict_ratio"] == 0.5


def test_ttm_availability_lag_measures_the_wait_past_the_closing_filing() -> None:
    con = _quarterly_con()
    # The quarter itself became available on the 16th, but its TTM window had to
    # wait for a late revision to an older quarter — 30 days more.
    _add_quarter(
        con,
        available_from=date(2023, 5, 16),
        ttm_available_from=date(2023, 6, 15),
    )
    _add_quarter(
        con,
        ticker="000660",
        available_from=date(2023, 5, 16),
        ttm_available_from=date(2023, 5, 16),
    )
    register_quarterly_metric_quality_view(con)
    (row,) = _rows(con, "quarterly_metric_quality")

    assert row["ttm_complete_rows"] == 2
    assert row["ttm_complete_ratio"] == 1.0
    assert row["mean_ttm_availability_lag_days"] == 15.0
    assert row["max_ttm_availability_lag_days"] == 30


def test_missing_standalone_values_lower_the_ratio_without_dropping_the_row() -> None:
    con = _quarterly_con()
    _add_quarter(con, quarter="Q4", standalone_value=None, ttm_complete=False)
    _add_quarter(con, ticker="000660", quarter="Q4", standalone_value=-5.0, ttm_complete=False)
    register_quarterly_metric_quality_view(con)
    (row,) = _rows(con, "quarterly_metric_quality")

    assert row["quarter"] == "Q4"
    assert row["rows"] == 2
    assert row["standalone_rows"] == 1
    assert row["standalone_ratio"] == 0.5
    assert row["negative_standalone_rows"] == 1
    assert row["mean_ttm_availability_lag_days"] is None


# --------------------------------------------------------------------------
# feature_coverage
# --------------------------------------------------------------------------

_FIN_SCAN_COLUMNS = (
    "trade_date DATE, ticker VARCHAR, market VARCHAR, "
    "fin_log_mcap DOUBLE, fin_log_mcap_lag1 DOUBLE, "
    "fin_book_to_market DOUBLE, fin_earnings_yield DOUBLE, fin_cfo_yield DOUBLE, "
    "fin_sales_to_price DOUBLE, value_component_count INTEGER, "
    "fin_value_z DOUBLE, fin_value_z_lag1 DOUBLE, "
    "fin_gross_profitability DOUBLE, fin_gross_profitability_lag1 DOUBLE, "
    "fin_operating_profitability DOUBLE, fin_operating_profitability_lag1 DOUBLE, "
    "fin_asset_growth_yoy DOUBLE, fin_asset_growth_yoy_lag1 DOUBLE, "
    "fin_accruals_to_assets DOUBLE, fin_accruals_to_assets_lag1 DOUBLE, "
    "value_fin_age_days BIGINT, profitability_fin_age_days BIGINT, "
    "asset_growth_fin_age_days BIGINT, accruals_fin_age_days BIGINT"
)

_EVENT_SCAN_COLUMNS = (
    "trade_date DATE, ticker VARCHAR, market VARCHAR, "
    "ev_net_share_issuance_yoy DOUBLE, ev_net_share_issuance_yoy_lag1 DOUBLE, "
    "ev_payout_yield DOUBLE, ev_payout_yield_lag1 DOUBLE, "
    "issuance_available_from DATE, payout_available_from DATE, "
    "issuance_identity_ok BOOLEAN, issuance_classification_complete BOOLEAN, "
    "dividend_source VARCHAR"
)


def _panel_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(f"CREATE TABLE feat_fin_scan_daily ({_FIN_SCAN_COLUMNS})")
    con.execute(f"CREATE TABLE feat_event_scan_daily ({_EVENT_SCAN_COLUMNS})")
    return con


def _add_fin_row(
    con: duckdb.DuckDBPyConnection,
    *,
    trade_date: date,
    ticker: str,
    value_z: float | None = 1.0,
    value_component_count: int = 4,
    value_fin_age_days: int | None = 40,
) -> None:
    con.execute(
        "INSERT INTO feat_fin_scan_daily VALUES "
        "(?,?,'KOSPI',?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            trade_date,
            ticker,
            1.0,  # fin_log_mcap
            1.0,  # fin_log_mcap_lag1
            0.5,  # fin_book_to_market
            0.1,  # fin_earnings_yield
            0.2,  # fin_cfo_yield
            0.3,  # fin_sales_to_price
            value_component_count,
            value_z,
            value_z,
            0.4,  # fin_gross_profitability
            0.4,
            0.3,  # fin_operating_profitability
            0.3,
            0.05,  # fin_asset_growth_yoy
            0.05,
            -0.02,  # fin_accruals_to_assets
            -0.02,
            value_fin_age_days,
            40,
            40,
            40,
        ],
    )


def _add_event_row(
    con: duckdb.DuckDBPyConnection,
    *,
    trade_date: date,
    ticker: str,
    issuance: float | None = 0.02,
    identity_ok: bool | None = True,
    classification_complete: bool | None = True,
) -> None:
    con.execute(
        "INSERT INTO feat_event_scan_daily VALUES (?,?,'KOSPI',?,?,?,?,?,?,?,?,?)",
        [
            trade_date,
            ticker,
            issuance,
            issuance,
            0.01,
            0.01,
            date(2024, 3, 11),
            date(2024, 3, 11),
            identity_ok,
            classification_complete,
            "shareholder_return",
        ],
    )


def _coverage_row(con: duckdb.DuckDBPyConnection, feature: str) -> dict:
    register_feature_coverage_view(con)
    (row,) = _rows(con, f"feature_coverage WHERE feature = '{feature}'")
    return row


def test_coverage_ratio_and_thinnest_date_disagree() -> None:
    con = _panel_con()
    # Day 1 has two names with a value, day 2 only one. Overall coverage is
    # 3/4, but the thinnest date carries a single name — which is what decides
    # whether the scan can use that date at all.
    _add_fin_row(con, trade_date=date(2024, 3, 11), ticker="005930")
    _add_fin_row(con, trade_date=date(2024, 3, 11), ticker="000660")
    _add_fin_row(con, trade_date=date(2024, 3, 12), ticker="005930")
    _add_fin_row(con, trade_date=date(2024, 3, 12), ticker="000660", value_z=None)
    row = _coverage_row(con, "fin_value_z")

    assert row["panel_rows"] == 4
    assert row["nonnull_rows"] == 3
    assert row["coverage_ratio"] == 0.75
    assert row["dates"] == 2
    assert row["dates_with_value"] == 2
    assert row["min_names_per_date"] == 1
    assert row["max_names_per_date"] == 2
    assert row["first_value_date"] == date(2024, 3, 11)
    assert row["mean_age_days"] == 40.0


def test_lag1_variant_reports_no_age_of_its_own() -> None:
    con = _panel_con()
    _add_fin_row(con, trade_date=date(2024, 3, 11), ticker="005930")
    native = _coverage_row(con, "fin_value_z")
    lagged = _coverage_row(con, "fin_value_z_lag1")

    assert native["variant"] == "native_t"
    assert lagged["variant"] == "lag1"
    assert native["mean_age_days"] == 40.0
    # The lag1 column carries the previous session's value, so the native_t age
    # on the same row understates it by a session; NULL rather than a number
    # that is quietly wrong.
    assert lagged["mean_age_days"] is None
    assert lagged["nonnull_rows"] == 1


def test_precondition_is_named_and_counted_per_feature() -> None:
    con = _panel_con()
    _add_fin_row(con, trade_date=date(2024, 3, 11), ticker="005930", value_component_count=4)
    _add_fin_row(
        con,
        trade_date=date(2024, 3, 11),
        ticker="000660",
        value_z=None,
        value_component_count=1,
    )
    row = _coverage_row(con, "fin_value_z")

    assert row["precondition"] == "value_components_ge_2"
    assert row["precondition_ok_rows"] == 1
    assert row["precondition_ok_ratio"] == 0.5


def test_issuance_precondition_treats_unknown_flags_as_not_ok() -> None:
    con = _panel_con()
    _add_event_row(con, trade_date=date(2024, 3, 11), ticker="005930")
    # No issuance row joined that day: both flags are NULL, and "unknown" must
    # not read as "passed".
    _add_event_row(
        con,
        trade_date=date(2024, 3, 11),
        ticker="000660",
        issuance=None,
        identity_ok=None,
        classification_complete=None,
    )
    row = _coverage_row(con, "ev_net_share_issuance_yoy")

    assert row["source_mart"] == "feat_event_scan_daily"
    assert row["precondition"] == "issuance_identity_and_classification"
    assert row["precondition_ok_rows"] == 1
    assert row["coverage_ratio"] == 0.5
    assert row["mean_age_days"] == 0.0


def test_coverage_splits_by_year_and_keeps_every_declared_feature() -> None:
    con = _panel_con()
    _add_fin_row(con, trade_date=date(2023, 12, 28), ticker="005930")
    _add_fin_row(con, trade_date=date(2024, 3, 11), ticker="005930")
    register_feature_coverage_view(con)
    rows = _rows(con, "feature_coverage")

    assert {row["year"] for row in rows} == {2023, 2024}
    # The event mart is empty, so its features contribute no rows — absent, not
    # zero-filled, same rule as everywhere else here.
    assert {row["source_mart"] for row in rows} == {"feat_fin_scan_daily"}
    assert "fin_book_to_market" in {row["feature"] for row in rows}


# --------------------------------------------------------------------------
# event_coverage
# --------------------------------------------------------------------------

_BUCKETS = ((0, 3), (3, 5), (5, 10), (10, 20), (20, 40), (40, 60))


def _sue_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    bucket_cols = ", ".join(
        f"bucket_{h1}_{h2}_raw DOUBLE, bucket_{h1}_{h2}_ca_contaminated BOOLEAN"
        for h1, h2 in _BUCKETS
    )
    con.execute(
        "CREATE TABLE fin_sue_event ("
        "ticker VARCHAR, market VARCHAR, reprt_code VARCHAR, event_formation_date DATE, "
        "fin_sue DOUBLE, sue_history_count INTEGER, is_primary_constant_sample BOOLEAN, "
        f"revision_within_60_sessions BOOLEAN, {bucket_cols})"
    )
    return con


def _add_event(
    con: duckdb.DuckDBPyConnection,
    *,
    ticker: str = "005930",
    reprt_code: str = "11011",
    event_formation_date: date = date(2024, 3, 11),
    fin_sue: float | None = 1.5,
    sue_history_count: int = 12,
    is_primary_constant_sample: bool = True,
    revision_within_60_sessions: bool = False,
    last_bucket_missing: bool = False,
    ca_contaminated: bool = False,
) -> None:
    bucket_values: list[object] = []
    for index, _ in enumerate(_BUCKETS):
        is_last = index == len(_BUCKETS) - 1
        bucket_values.append(None if (is_last and last_bucket_missing) else 0.01)
        bucket_values.append(ca_contaminated and is_last)
    placeholders = ",".join("?" * (8 + 2 * len(_BUCKETS)))
    con.execute(
        f"INSERT INTO fin_sue_event VALUES ({placeholders})",
        [
            ticker,
            "KOSPI",
            reprt_code,
            event_formation_date,
            fin_sue,
            sue_history_count,
            is_primary_constant_sample,
            revision_within_60_sessions,
            *bucket_values,
        ],
    )


def test_event_coverage_counts_each_bucket_separately() -> None:
    con = _sue_con()
    _add_event(con, ticker="005930")
    # A short price history kills only the last bucket — the early buckets are
    # still scannable, so one count for "the event" would hide that.
    _add_event(
        con,
        ticker="000660",
        last_bucket_missing=True,
        is_primary_constant_sample=False,
    )
    register_event_coverage_view(con)
    (row,) = _rows(con, "event_coverage")

    assert row["events"] == 2
    assert row["bucket_0_3_events"] == 2
    assert row["bucket_40_60_events"] == 1
    assert row["events_with_missing_bucket"] == 1
    assert row["primary_constant_sample_events"] == 1
    assert row["primary_constant_sample_ratio"] == 0.5


def test_event_coverage_separates_sue_presence_from_history_length() -> None:
    con = _sue_con()
    _add_event(con, ticker="005930", fin_sue=1.5, sue_history_count=12)
    # Enough history to pass the minimum but no SUE value, and a short history
    # that could never have produced one — two different reasons, two counts.
    _add_event(con, ticker="000660", fin_sue=None, sue_history_count=9)
    _add_event(con, ticker="035420", fin_sue=None, sue_history_count=3)
    register_event_coverage_view(con)
    (row,) = _rows(con, "event_coverage")

    assert row["events"] == 3
    assert row["events_with_sue"] == 1
    assert row["events_with_full_history"] == 2
    assert row["mean_sue_history_count"] == 8.0


def test_event_coverage_flags_revision_contamination_and_ca() -> None:
    con = _sue_con()
    _add_event(con, ticker="005930")
    _add_event(
        con,
        ticker="000660",
        revision_within_60_sessions=True,
        is_primary_constant_sample=False,
    )
    _add_event(
        con,
        ticker="035420",
        ca_contaminated=True,
        is_primary_constant_sample=False,
    )
    register_event_coverage_view(con)
    (row,) = _rows(con, "event_coverage")

    assert row["revision_contaminated_events"] == 1
    assert row["events_with_ca_contamination"] == 1
    assert row["primary_constant_sample_events"] == 1


def test_event_coverage_splits_by_year_and_report_code() -> None:
    con = _sue_con()
    _add_event(con, reprt_code="11011", event_formation_date=date(2024, 3, 11))
    _add_event(con, reprt_code="11013", event_formation_date=date(2024, 5, 16))
    _add_event(con, reprt_code="11011", event_formation_date=date(2025, 3, 11))
    register_event_coverage_view(con)
    rows = _rows(con, "event_coverage")

    assert [(row["event_year"], row["reprt_code"]) for row in rows] == [
        (2024, "11011"),
        (2024, "11013"),
        (2025, "11011"),
    ]


def test_mapping_fallback_is_judged_against_the_rule_for_this_basis() -> None:
    con = _vintage_con()
    cfs_best = _catalog_best_priority("net_income", "CFS")
    ofs_best = _catalog_best_priority("net_income", "OFS")
    assert cfs_best < ofs_best, "fixture assumes net_income's best rule names CFS"

    # An OFS row matched by the best rule that can apply to OFS at all. Judged
    # against one global minimum it looks like a fallback; it is not, and on
    # the real lake that artifact read 65.5% fallback for net_income.
    _add_vintage(
        con,
        metric_code="net_income",
        fs_basis="OFS",
        rcept_no="20240310000001",
        mapping_priority=ofs_best,
    )
    _add_vintage(
        con,
        metric_code="net_income",
        fs_basis="CFS",
        rcept_no="20240310000002",
        mapping_priority=cfs_best,
    )
    register_stock_metric_vintage_quality_view(con)
    by_basis = {r["fs_basis"]: r for r in _rows(con, "stock_metric_vintage_quality")}

    assert by_basis["OFS"]["catalog_best_priority"] == ofs_best
    assert by_basis["OFS"]["mapping_fallback_rows"] == 0
    assert by_basis["CFS"]["catalog_best_priority"] == cfs_best
    assert by_basis["CFS"]["mapping_fallback_rows"] == 0


def test_a_genuine_second_choice_rule_still_counts_as_fallback() -> None:
    con = _vintage_con()
    cfs_best = _catalog_best_priority("net_income", "CFS")
    _add_vintage(
        con,
        metric_code="net_income",
        fs_basis="CFS",
        mapping_priority=cfs_best + 1,
    )
    register_stock_metric_vintage_quality_view(con)
    (row,) = _rows(con, "stock_metric_vintage_quality")

    assert row["mapping_fallback_rows"] == 1
    assert row["mapping_fallback_ratio"] == 1.0
