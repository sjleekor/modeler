"""The industry-neutral variant of ``feat_fin_scan_daily`` (N2-9).

Every cross-sectional z-score in the scan partitions on ``(trade_date,
market)``, so banks, biotech, shipbuilders and game studios normalise against
one KOSPI pool. Barra keeps industry as a first-class block for exactly this
reason. The variant recomputes the same features within industry.

Two things need pinning. The existing path must be untouched — the frozen
parity artifacts are only meaningful if the default emits the same SQL it
always did. And the variant must fail loudly when the industry codes are
absent, because grouping everything as unknown would produce output identical
to the plain path and read as "industry does not matter".
"""

from __future__ import annotations

import duckdb
import pytest
from collector.kr.shared import (
    MIN_GROUP_SIZE,
    OTHER_GROUP,
    UNKNOWN_GROUP,
)

from modeler.etl.features.fin_scan import (
    CROSS_SECTION_WITH_INDUSTRY,
    build_fin_scan_daily_sql,
    register_fin_scan_daily_view,
)
from modeler.etl.industry import register_industry_group_view


def test_the_default_path_is_unchanged() -> None:
    # The frozen parity artifacts are only meaningful if this holds.
    sql = build_fin_scan_daily_sql()

    assert "PARTITION BY trade_date, market)" in sql
    assert "industry_group" not in sql
    assert "LEFT JOIN" in sql  # the quality join is still there


def test_the_variant_neutralises_every_cross_sectional_step() -> None:
    # Winsorize percentiles and z-scores both partition cross-sectionally. If
    # only one moved, some components would be industry-relative and others
    # market-relative, and the combined value score would mix the two.
    plain = build_fin_scan_daily_sql()
    variant = build_fin_scan_daily_sql(industry_view="dim_industry_group")

    plain_partitions = plain.count("PARTITION BY trade_date, market)")
    variant_partitions = variant.count(f"PARTITION BY {CROSS_SECTION_WITH_INDUSTRY})")

    assert plain_partitions == variant_partitions == 16
    assert "PARTITION BY trade_date, market)" not in variant


def test_the_variant_joins_the_industry_view() -> None:
    variant = build_fin_scan_daily_sql(industry_view="my_industry_view")

    assert "LEFT JOIN my_industry_view ind USING (ticker)" in variant
    assert "ind.industry_group" in variant


def test_the_per_ticker_windows_stay_per_ticker() -> None:
    # The lag columns look back along one ticker's own history; industry has no
    # business in them, and neutralising them would compare a company's value
    # today against its peers' value yesterday.
    variant = build_fin_scan_daily_sql(industry_view="dim_industry_group")

    assert "PARTITION BY ticker, market ORDER BY trade_date" in variant


def test_the_variant_binds_and_exposes_its_group() -> None:
    """Catch a group column dropped between panel and window CTEs."""
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE pit (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "market_cap_pit DOUBLE, issued_shares_pit DOUBLE, shares_is_available BOOLEAN, "
        "shares_invalid_flag BOOLEAN, shares_available_from DATE)"
    )
    con.execute(
        "CREATE TABLE quality (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "is_halted BOOLEAN, valid_session_idx BIGINT)"
    )
    con.execute(
        "CREATE TABLE vintages (ticker VARCHAR, metric_code VARCHAR, fs_basis VARCHAR, "
        "seq_key BIGINT, rcept_no VARCHAR, metric_kind VARCHAR, ttm_value DOUBLE, "
        "standalone_value DOUBLE, ttm_available_from DATE, available_from DATE, "
        "value_lag_4q DOUBLE)"
    )
    con.execute("CREATE TABLE industries (ticker VARCHAR, industry_group VARCHAR)")

    register_fin_scan_daily_view(
        con,
        view_name="industry_fin_scan",
        pit_view="pit",
        quality_view="quality",
        vintage_view="vintages",
        industry_view="industries",
    )

    columns = [row[0] for row in con.execute("DESCRIBE industry_fin_scan").fetchall()]
    assert "industry_group" in columns


# --------------------------------------------------------------------------
# the industry group view
# --------------------------------------------------------------------------


def _corp_master(rows: list[tuple[str, str | None]], *, with_column: bool = True):
    con = duckdb.connect()
    if with_column:
        con.execute("CREATE TABLE dart_corp_master(ticker VARCHAR, induty_code VARCHAR)")
        if rows:
            con.executemany("INSERT INTO dart_corp_master VALUES (?, ?)", rows)
    else:
        con.execute("CREATE TABLE dart_corp_master(ticker VARCHAR, corp_name VARCHAR)")
    return con


def test_a_lake_without_industry_codes_fails_loudly() -> None:
    # The expected state on a snapshot taken before 2026-08-15. Falling back to
    # "everything unknown" would make the variant identical to the plain path,
    # and identical output reads as a finding.
    con = _corp_master([], with_column=False)

    with pytest.raises(RuntimeError, match="induty_code"):
        register_industry_group_view(con)


def test_groups_come_from_the_shared_definitions_not_from_sql() -> None:
    # The fold-up is data-dependent — a thin 2-digit group folds into its
    # section, a thin section into OTHER — so it runs in Python and its answer
    # is handed to DuckDB. Reimplementing it in SQL would be a second copy.
    rows = [(f"{i:06d}", "26100") for i in range(MIN_GROUP_SIZE + 5)]
    rows += [("900001", "05100")]  # a lone mining company: too thin to stand
    con = _corp_master(rows)

    register_industry_group_view(con)
    groups = dict(con.execute("SELECT ticker, industry_group FROM dim_industry_group").fetchall())

    assert groups["000000"] == "26"
    assert groups["900001"] != "05"


def test_a_missing_code_is_unknown_rather_than_folded_into_a_real_industry() -> None:
    rows = [(f"{i:06d}", "26100") for i in range(MIN_GROUP_SIZE + 5)]
    rows += [("900002", None)]
    con = _corp_master(rows)

    register_industry_group_view(con)
    groups = dict(con.execute("SELECT ticker, industry_group FROM dim_industry_group").fetchall())

    assert groups["900002"] == UNKNOWN_GROUP


def test_every_standing_group_meets_the_minimum_size() -> None:
    # A two-member group's z-scores are always ±0.707 — a signal manufactured
    # by the group size, not measured (N2-10 V2). UNKNOWN and OTHER are the
    # documented terminals and are excluded.
    rows = [(f"{i:06d}", "26100") for i in range(30)]
    rows += [(f"1{i:05d}", "58210") for i in range(25)]
    rows += [("900003", "05100"), ("900004", "05200")]
    con = _corp_master(rows)

    register_industry_group_view(con)
    sizes = con.execute(
        "SELECT industry_group, count(*) FROM dim_industry_group GROUP BY 1"
    ).fetchall()

    for group, size in sizes:
        if group in {UNKNOWN_GROUP, OTHER_GROUP}:
            continue
        assert size >= MIN_GROUP_SIZE, group


def test_tickers_without_a_ticker_are_not_in_the_view() -> None:
    con = _corp_master([("005930", "26100"), ("", "26100"), (None, "26100")])

    register_industry_group_view(con)
    tickers = [row[0] for row in con.execute("SELECT ticker FROM dim_industry_group").fetchall()]

    assert tickers == ["005930"]
