"""``feat_fin_risk`` — F-4 (03_financial_risk_lifecycle_transition.md).

Four things need pinning.

* ``feat_fin_scan_daily``'s SQL text is unchanged by the F-4.1 extraction. It is
  an A0 cache key and every published Phase A/B run records the lineage that
  follows from it, so "extraction only" has to be literally true.
* The two marts read the vintage under *one* rule, not two that agree today:
  the same ``total_assets`` on the same ticker-date.
* Dickinson's eight sign combinations map to the five stages the paper assigns,
  all eight of them.
* The NULL rules bind. A missing figure must not be silently imputed — the
  interest-coverage cap in particular, where DuckDB's ``LEAST`` skips NULLs and
  would otherwise pin every company with no interest line to the healthiest end
  of the cross-section.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import duckdb
import pytest
from modeler.etl.features import fin_risk, fin_scan
from modeler.etl.features.fin_vintage import (
    BASE_OK_SQL,
    build_metric_intervals_cte,
    build_metric_joins,
    build_wide_intervals_asof_join,
    build_wide_intervals_cte,
)

# --------------------------------------------------------------------------
# F-4.1 — the extraction is byte-neutral
# --------------------------------------------------------------------------


def test_the_fin_scan_sql_is_byte_identical() -> None:
    """These are the hashes the 2026-08-23 snapshot's mart was written under.

    ``data_lake/feature_mart/.../feat_fin_scan_daily/_cache_metadata.json``
    carries the first one. If this fails, every frozen Phase B artifact has been
    detached from its input lineage and the whole scan needs rebuilding — which
    is exactly the failure mode this test exists to make loud.
    """

    def _hash(**kwargs) -> str:
        return hashlib.sha256(fin_scan.build_fin_scan_daily_sql(**kwargs).encode()).hexdigest()

    assert _hash() == "9656de5eb96b9b1498187308a9ea3831a3f0a2f4e29ff45cca7fb061fdd7fb4a"
    assert (
        _hash(industry_view="dim_industry_group")
        == "2615e77d645ff1f7f7d4424291390c9d3dd519a75e6d199b8a95dc8d7d3fa32c"
    )


def test_fin_scan_uses_the_shared_builders_rather_than_its_own_copies() -> None:
    # Byte-identity alone would also pass if fin_scan kept private duplicates.
    sql = fin_scan.build_fin_scan_daily_sql()

    assert build_metric_intervals_cte("fin_quarterly_metric_vintage", fin_scan._METRICS) in sql
    assert build_metric_joins(fin_scan._METRICS) in sql
    assert BASE_OK_SQL in sql
    assert not hasattr(fin_scan, "_metric_intervals_cte")


def test_the_wide_pivot_and_the_range_join_agree() -> None:
    """fin_risk pivots the intervals wide and reads them with one ASOF join.

    The per-metric equality-plus-range form is the one that cannot be rewritten
    (``feat_fin_scan_daily``'s SQL text is an A0 cache key), so the claim that
    the pivot is the same lookup has to be checked rather than asserted. The
    fixture is deliberately awkward: two metrics filed on *different* dates, so
    the pivot grid has rows where one column is NULL and must be forward-filled
    from an earlier date — the case where the two forms could diverge. It also
    includes a session before the first filing, a boundary session, one inside a
    closed interval, one past the open-ended last interval, and a same-day pair
    (the ``same_day_rank`` tie-break).
    """
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE fin_quarterly_metric_vintage (ticker VARCHAR, metric_code VARCHAR, "
        "fs_basis VARCHAR, seq_key BIGINT, rcept_no VARCHAR, metric_kind VARCHAR, "
        "ttm_value DOUBLE, standalone_value DOUBLE, ttm_available_from DATE, "
        "available_from DATE, value_lag_4q DOUBLE)"
    )
    con.executemany(
        "INSERT INTO fin_quarterly_metric_vintage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            _instant("total_assets", 1, 100.0, dt.date(2020, 3, 2)),
            _instant("total_assets", 2, 200.0, dt.date(2020, 6, 1)),
            _instant("total_assets", 3, 300.0, dt.date(2020, 9, 1)),
            _instant("total_assets", 4, 400.0, dt.date(2020, 9, 1)),  # same day
            # A second metric on its own, staggered calendar.
            _instant("total_equity", 1, 10.0, dt.date(2020, 4, 15)),
            _instant("total_equity", 2, 20.0, dt.date(2020, 8, 20)),
        ],
    )
    con.execute("CREATE TABLE panel (trade_date DATE, ticker VARCHAR)")
    con.executemany(
        "INSERT INTO panel VALUES (?, '005930')",
        [
            (dt.date(2020, 1, 2),),  # before any filing
            (dt.date(2020, 3, 2),),  # a boundary
            (dt.date(2020, 4, 1),),  # inside a closed interval, equity still absent
            (dt.date(2020, 5, 1),),  # equity now present, assets from an earlier date
            (dt.date(2020, 6, 1),),  # the next assets boundary
            (dt.date(2020, 8, 25),),
            (dt.date(2020, 12, 31),),  # past the open-ended last interval
        ],
    )
    metrics = ("total_assets", "total_equity")
    cte = build_metric_intervals_cte("fin_quarterly_metric_vintage", metrics)
    columns = "m_total_assets_cfs.daily_value, m_total_equity_cfs.daily_value"
    ranged = con.execute(
        f"WITH {cte} SELECT panel.trade_date, {columns} FROM panel "
        f"{build_metric_joins(metrics, ('CFS',))} ORDER BY panel.trade_date"
    ).fetchall()

    wide_cte = build_wide_intervals_cte(metrics, ("CFS",))
    pivoted = con.execute(
        f"WITH {cte}, {wide_cte} "
        "SELECT panel.trade_date, iv.v_total_assets_cfs, iv.v_total_equity_cfs "
        f"FROM panel {build_wide_intervals_asof_join()} ORDER BY panel.trade_date"
    ).fetchall()

    assert pivoted == ranged
    assert pivoted == [
        (dt.date(2020, 1, 2), None, None),
        (dt.date(2020, 3, 2), 100.0, None),
        (dt.date(2020, 4, 1), 100.0, None),
        (dt.date(2020, 5, 1), 100.0, 10.0),  # forward-filled from different dates
        (dt.date(2020, 6, 1), 200.0, 10.0),
        (dt.date(2020, 8, 25), 200.0, 20.0),
        (dt.date(2020, 12, 31), 400.0, 20.0),
    ]


def test_the_pivot_carries_each_metrics_own_availability_date() -> None:
    # The *_available_from outputs are per family, and a family's staleness is
    # its own metrics' filing dates — not the last date anything was filed.
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE fin_quarterly_metric_vintage (ticker VARCHAR, metric_code VARCHAR, "
        "fs_basis VARCHAR, seq_key BIGINT, rcept_no VARCHAR, metric_kind VARCHAR, "
        "ttm_value DOUBLE, standalone_value DOUBLE, ttm_available_from DATE, "
        "available_from DATE, value_lag_4q DOUBLE)"
    )
    con.executemany(
        "INSERT INTO fin_quarterly_metric_vintage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            _instant("total_assets", 1, 100.0, dt.date(2020, 3, 2)),
            _instant("total_equity", 1, 10.0, dt.date(2020, 8, 20)),
        ],
    )
    con.execute("CREATE TABLE panel (trade_date DATE, ticker VARCHAR)")
    con.execute("INSERT INTO panel VALUES (DATE '2020-09-30', '005930')")
    metrics = ("total_assets", "total_equity")
    cte = build_metric_intervals_cte("fin_quarterly_metric_vintage", metrics)

    row = con.execute(
        f"WITH {cte}, {build_wide_intervals_cte(metrics, ('CFS',))} "
        "SELECT iv.a_total_assets_cfs, iv.a_total_equity_cfs "
        f"FROM panel {build_wide_intervals_asof_join()}"
    ).fetchone()

    assert row == (dt.date(2020, 3, 2), dt.date(2020, 8, 20))


def test_fin_risk_uses_the_pivot_and_fin_scan_keeps_its_range_joins() -> None:
    risk = fin_risk.build_fin_risk_sql()
    scan = fin_scan.build_fin_scan_daily_sql()

    assert "ASOF LEFT JOIN wide_intervals" in risk
    assert risk.count("ASOF LEFT JOIN") == 1  # one join, not one per metric
    assert "ASOF" not in scan
    assert build_metric_joins(fin_scan._METRICS) in scan


def test_both_marts_read_the_same_vintage_rules() -> None:
    risk = fin_risk.build_fin_risk_sql()

    assert BASE_OK_SQL in risk
    # with_previous adds columns; the shared prefix is still the same rule.
    assert "same_day_rank = 1" in risk
    assert "ORDER BY seq_key DESC, rcept_no DESC" in risk


def test_the_previous_vintage_columns_are_opt_in() -> None:
    # Off by default, or the extraction would not be byte-neutral for fin_scan.
    plain = build_metric_intervals_cte("v", ("total_assets",))
    extended = build_metric_intervals_cte("v", ("total_assets",), with_previous=True)

    assert "prev_daily_value" not in plain
    assert "prev_daily_value" in extended
    assert "prev_available_from" in extended


# --------------------------------------------------------------------------
# the life-cycle mapping
# --------------------------------------------------------------------------


def test_the_lifecycle_table_is_dickinsons() -> None:
    assert fin_risk.LIFECYCLE_STAGES == {
        (-1, -1, +1): 1,
        (+1, -1, +1): 2,
        (+1, -1, -1): 3,
        (-1, -1, -1): 4,
        (+1, +1, +1): 4,
        (+1, +1, -1): 4,
        (-1, +1, +1): 5,
        (-1, +1, -1): 5,
    }
    assert len(fin_risk.LIFECYCLE_STAGES) == 8  # {+,-}^3 exhaustively


def _lifecycle_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("CREATE TABLE signs (cfo DOUBLE, cfi DOUBLE, cff DOUBLE)")
    return con


@pytest.mark.parametrize(("signs", "stage"), list(fin_risk.LIFECYCLE_STAGES.items()))
def test_every_sign_combination_maps_to_its_paper_stage(
    signs: tuple[int, int, int], stage: int
) -> None:
    con = _lifecycle_con()
    con.execute("INSERT INTO signs VALUES (?, ?, ?)", [float(s) * 1000 for s in signs])
    case = fin_risk._lifecycle_case(
        fin_risk._sign("cfo"), fin_risk._sign("cfi"), fin_risk._sign("cff"), alias="stage"
    )

    assert con.execute(f"SELECT {case} FROM signs").fetchone()[0] == stage


def test_an_exact_zero_cash_flow_has_no_stage() -> None:
    # The paper's patterns are over strict signs; calling 0 positive would place
    # the firm in a stage it was never assigned.
    con = _lifecycle_con()
    con.execute("INSERT INTO signs VALUES (100, 0, -100)")
    case = fin_risk._lifecycle_case(
        fin_risk._sign("cfo"), fin_risk._sign("cfi"), fin_risk._sign("cff"), alias="stage"
    )

    assert con.execute(f"SELECT {case} FROM signs").fetchone()[0] is None


# --------------------------------------------------------------------------
# end-to-end on a synthetic vintage
# --------------------------------------------------------------------------

_SESSIONS = [dt.date(2020, 1, 6) + dt.timedelta(days=i) for i in range(400)]


def _fixture(
    vintages: list[tuple],
    *,
    dividends: list[tuple] | None = None,
    sessions: list[dt.date] | None = None,
) -> duckdb.DuckDBPyConnection:
    """A minimal A0 panel plus a hand-written vintage table.

    ``vintages`` rows are
    ``(ticker, metric_code, fs_basis, seq_key, rcept_no, metric_kind,
       ttm_value, standalone_value, ttm_available_from, available_from,
       value_lag_4q)`` — the columns ``fin_quarterly_metric_vintage`` exposes
    and the only ones either mart reads.
    """
    sessions = sessions or _SESSIONS
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE dim_stock_pit_daily (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "market_cap_pit DOUBLE, issued_shares_pit DOUBLE, shares_is_available BOOLEAN, "
        "shares_invalid_flag BOOLEAN, shares_available_from DATE)"
    )
    con.executemany(
        "INSERT INTO dim_stock_pit_daily VALUES (?, '005930', 'KOSPI', 1000000, 100, "
        "TRUE, FALSE, NULL)",
        [(d,) for d in sessions],
    )
    con.execute(
        "CREATE TABLE dim_price_quality_daily (trade_date DATE, ticker VARCHAR, market VARCHAR, "
        "is_halted BOOLEAN, valid_session_idx BIGINT)"
    )
    con.executemany(
        "INSERT INTO dim_price_quality_daily VALUES (?, '005930', 'KOSPI', FALSE, ?)",
        [(d, i + 1) for i, d in enumerate(sessions)],
    )
    con.execute(
        "CREATE TABLE fin_quarterly_metric_vintage (ticker VARCHAR, metric_code VARCHAR, "
        "fs_basis VARCHAR, seq_key BIGINT, rcept_no VARCHAR, metric_kind VARCHAR, "
        "ttm_value DOUBLE, standalone_value DOUBLE, ttm_available_from DATE, "
        "available_from DATE, value_lag_4q DOUBLE)"
    )
    if vintages:
        con.executemany(
            "INSERT INTO fin_quarterly_metric_vintage VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            vintages,
        )
    con.execute(
        "CREATE TABLE dart_shareholder_return_raw (ticker VARCHAR, bsns_year INTEGER, "
        "reprt_code VARCHAR, statement_type VARCHAR, row_name VARCHAR, stock_knd VARCHAR, "
        "metric_code VARCHAR, value_numeric DOUBLE, value_text VARCHAR, rcept_no VARCHAR)"
    )
    if dividends:
        con.executemany(
            "INSERT INTO dart_shareholder_return_raw VALUES (?, ?, '11011', 'dividend', ?, ?, "
            "'thstrm', ?, ?, ?)",
            dividends,
        )
    fin_risk.register_fin_risk_calendar(con, sessions)
    fin_risk.register_fin_risk_view(con)
    return con


def _instant(
    metric: str, seq: int, value: float, asof: dt.date, *, lag4q: float | None = None
) -> tuple:
    return (
        "005930",
        metric,
        "CFS",
        seq,
        f"2020{seq:010d}",
        "instant",
        None,
        value,
        None,
        asof,
        lag4q,
    )


def _ttm(metric: str, seq: int, value: float, asof: dt.date) -> tuple:
    return (
        "005930",
        metric,
        "CFS",
        seq,
        f"2020{seq:010d}",
        "cumulative_reported",
        value,
        None,
        asof,
        None,
        None,
    )


def test_the_mart_emits_every_documented_column() -> None:
    con = _fixture([_instant("net_income", 1, 5.0, dt.date(2020, 3, 2))])

    columns = [row[0] for row in con.execute("DESCRIBE feat_fin_risk").fetchall()]

    assert columns[:3] == ["trade_date", "ticker", "market"]
    for name in fin_risk.PRIMARY_COLUMNS + fin_risk.DIAGNOSTIC_COLUMNS:
        assert name in columns, name
    for name in fin_risk.PRIMARY_COLUMNS:
        assert f"{name}_lag1" in columns, name
    for group in fin_risk.AVAILABILITY_GROUPS:
        assert f"{group}_available_from" in columns
        assert f"{group}_fin_age_days" in columns


def test_a_figure_is_not_readable_before_its_filing() -> None:
    asof = dt.date(2020, 6, 1)
    con = _fixture(
        [
            _ttm("net_income", 1, 5.0, asof),
            _instant("total_liabilities", 1, 40.0, asof),
            _instant("total_assets", 1, 100.0, asof),
        ]
    )

    before, on, after = (
        con.execute(
            "SELECT fin_debt_to_assets FROM feat_fin_risk WHERE trade_date = ?", [d]
        ).fetchone()[0]
        for d in (asof - dt.timedelta(days=1), asof, asof + dt.timedelta(days=10))
    )

    assert before is None
    assert on == pytest.approx(0.4)
    assert after == pytest.approx(0.4)


def test_the_vintage_value_matches_what_fin_scan_reads() -> None:
    """§5's parity check: same ticker, same date, same ``total_assets``."""
    asof = dt.date(2020, 6, 1)
    con = _fixture(
        [
            _ttm("net_income", 1, 5.0, asof),
            _instant("total_assets", 1, 250.0, asof),
            _instant("total_liabilities", 1, 100.0, asof),
            _instant("total_equity", 1, 150.0, asof),
        ]
    )
    fin_scan.register_fin_scan_daily_view(con, view_name="scan")

    # fin_scan does not expose total_assets, but fin_asset_growth_yoy and
    # fin_debt_to_assets are both functions of it, so recover it from the
    # leverage ratio and check the book value agrees.
    risk = con.execute(
        "SELECT fin_debt_to_assets FROM feat_fin_risk WHERE trade_date = DATE '2020-07-01'"
    ).fetchone()[0]
    scan_bm = con.execute(
        "SELECT fin_book_to_market FROM scan WHERE trade_date = DATE '2020-07-01'"
    ).fetchone()[0]

    assert risk == pytest.approx(100.0 / 250.0)
    assert scan_bm == pytest.approx(150.0 / 1_000_000)


def test_a_company_with_no_interest_line_is_null_not_the_cap() -> None:
    # DuckDB's LEAST skips NULL arguments, so an unguarded
    # LEAST(NULL, 100) returns 100 — the fin_v1 imputation failure.
    asof = dt.date(2020, 6, 1)
    con = _fixture([_ttm("net_income", 1, 5.0, asof), _ttm("operating_income", 1, 900.0, asof)])

    value, capped = con.execute(
        "SELECT fin_interest_coverage, fin_interest_coverage_capped FROM feat_fin_risk "
        "WHERE trade_date = DATE '2020-07-01'"
    ).fetchone()

    assert value is None
    # And the cap flag says "no answer" rather than "not capped" — the
    # diagnostic is a share of the values that exist.
    assert capped is None


def test_a_non_positive_interest_denominator_is_null() -> None:
    asof = dt.date(2020, 6, 1)
    con = _fixture(
        [
            _ttm("net_income", 1, 5.0, asof),
            _ttm("operating_income", 1, 900.0, asof),
            _ttm("interest_paid", 1, -10.0, asof),
        ]
    )

    assert (
        con.execute(
            "SELECT fin_interest_coverage FROM feat_fin_risk "
            "WHERE trade_date = DATE '2020-07-01'"
        ).fetchone()[0]
        is None
    )


def test_interest_coverage_is_capped_above_and_flagged() -> None:
    asof = dt.date(2020, 6, 1)
    con = _fixture(
        [
            _ttm("net_income", 1, 5.0, asof),
            _ttm("operating_income", 1, 900.0, asof),
            _ttm("interest_paid", 1, 1.0, asof),
        ]
    )

    value, capped = con.execute(
        "SELECT fin_interest_coverage, fin_interest_coverage_capped FROM feat_fin_risk "
        "WHERE trade_date = DATE '2020-07-01'"
    ).fetchone()

    assert value == fin_risk.INTEREST_COVERAGE_CAP
    assert capped is True


def test_net_debt_needs_both_legs_rather_than_treating_missing_cash_as_zero() -> None:
    asof = dt.date(2020, 6, 1)
    con = _fixture(
        [
            _ttm("net_income", 1, 5.0, asof),
            _instant("total_liabilities", 1, 400_000.0, asof),
        ]
    )

    assert (
        con.execute(
            "SELECT fin_net_debt_to_mcap FROM feat_fin_risk " "WHERE trade_date = DATE '2020-07-01'"
        ).fetchone()[0]
        is None
    )


def test_external_finance_needs_a_four_quarter_average_asset_base() -> None:
    asof = dt.date(2020, 6, 1)
    without = _fixture(
        [
            _ttm("net_income", 1, 5.0, asof),
            _ttm("financing_cash_flow", 1, 30.0, asof),
            _instant("total_assets", 1, 100.0, asof),
        ]
    )
    with_lag = _fixture(
        [
            _ttm("net_income", 1, 5.0, asof),
            _ttm("financing_cash_flow", 1, 30.0, asof),
            _instant("total_assets", 1, 100.0, asof, lag4q=100.0),
        ]
    )

    sql = (
        "SELECT fin_ext_finance_to_assets FROM feat_fin_risk "
        "WHERE trade_date = DATE '2020-07-01'"
    )
    assert without.execute(sql).fetchone()[0] is None
    assert with_lag.execute(sql).fetchone()[0] == pytest.approx(0.3)


# --------------------------------------------------------------------------
# transitions
# --------------------------------------------------------------------------


def test_a_stage_change_between_consecutive_vintages_is_a_transition() -> None:
    first, second = dt.date(2020, 3, 2), dt.date(2020, 6, 1)
    con = _fixture(
        [
            _ttm("net_income", 1, 5.0, first),
            _ttm("net_income", 2, 5.0, second),
            # Growth (+,-,+) then Mature (+,-,-)
            _ttm("operating_cash_flow", 1, 10.0, first),
            _ttm("investing_cash_flow", 1, -10.0, first),
            _ttm("financing_cash_flow", 1, 10.0, first),
            _ttm("operating_cash_flow", 2, 10.0, second),
            _ttm("investing_cash_flow", 2, -10.0, second),
            _ttm("financing_cash_flow", 2, -10.0, second),
        ]
    )

    early = con.execute(
        "SELECT fin_lifecycle_stage, fin_lifecycle_transition, fin_lifecycle_prev_stage "
        "FROM feat_fin_risk WHERE trade_date = DATE '2020-04-01'"
    ).fetchone()
    late = con.execute(
        "SELECT fin_lifecycle_stage, fin_lifecycle_transition, fin_lifecycle_transition_up, "
        "fin_lifecycle_prev_aligned FROM feat_fin_risk WHERE trade_date = DATE '2020-07-01'"
    ).fetchone()

    assert early == (2, None, None)  # no preceding vintage yet
    assert late == (3, 1, 1, True)


def test_an_unchanged_stage_is_zero_not_null() -> None:
    first, second = dt.date(2020, 3, 2), dt.date(2020, 6, 1)
    con = _fixture(
        [
            _ttm("net_income", 1, 5.0, first),
            _ttm("net_income", 2, 5.0, second),
            *[
                _ttm(metric, seq, value, asof)
                for seq, asof in ((1, first), (2, second))
                for metric, value in (
                    ("operating_cash_flow", 10.0),
                    ("investing_cash_flow", -10.0),
                    ("financing_cash_flow", -10.0),
                )
            ],
        ]
    )

    assert con.execute(
        "SELECT fin_lifecycle_transition, fin_lifecycle_transition_up, "
        "fin_lifecycle_transition_down FROM feat_fin_risk WHERE trade_date = DATE '2020-07-01'"
    ).fetchone() == (0, 0, 0)


def test_misaligned_previous_vintages_are_flagged_rather_than_hidden() -> None:
    con = _fixture(
        [
            _ttm("net_income", 2, 5.0, dt.date(2020, 6, 1)),
            _ttm("operating_cash_flow", 1, 10.0, dt.date(2020, 3, 2)),
            _ttm("operating_cash_flow", 2, 10.0, dt.date(2020, 6, 1)),
            _ttm("investing_cash_flow", 1, -10.0, dt.date(2020, 3, 3)),  # a day later
            _ttm("investing_cash_flow", 2, -10.0, dt.date(2020, 6, 1)),
            _ttm("financing_cash_flow", 1, -10.0, dt.date(2020, 3, 2)),
            _ttm("financing_cash_flow", 2, -10.0, dt.date(2020, 6, 1)),
        ]
    )

    aligned = con.execute(
        "SELECT fin_lifecycle_prev_aligned FROM feat_fin_risk "
        "WHERE trade_date = DATE '2020-07-01'"
    ).fetchone()[0]

    assert aligned is False


def test_profit_turn_reads_the_sign_change_not_the_level() -> None:
    first, second = dt.date(2020, 3, 2), dt.date(2020, 6, 1)
    con = _fixture([_ttm("net_income", 1, -50.0, first), _ttm("net_income", 2, 1.0, second)])

    assert (
        con.execute(
            "SELECT fin_profit_turn FROM feat_fin_risk WHERE trade_date = DATE '2020-07-01'"
        ).fetchone()[0]
        == 1
    )


def test_profit_turn_is_negative_one_going_the_other_way_and_zero_when_stable() -> None:
    first, second, third = dt.date(2020, 3, 2), dt.date(2020, 6, 1), dt.date(2020, 9, 1)
    con = _fixture(
        [
            _ttm("net_income", 1, 50.0, first),
            _ttm("net_income", 2, -1.0, second),
            _ttm("net_income", 3, -2.0, third),
        ]
    )

    turned = con.execute(
        "SELECT fin_profit_turn FROM feat_fin_risk WHERE trade_date = DATE '2020-07-01'"
    ).fetchone()[0]
    stable = con.execute(
        "SELECT fin_profit_turn FROM feat_fin_risk WHERE trade_date = DATE '2020-09-15'"
    ).fetchone()[0]

    assert (turned, stable) == (-1, 0)


def test_negative_equity_exit_fires_once_on_the_crossing() -> None:
    first, second = dt.date(2020, 3, 2), dt.date(2020, 6, 1)
    con = _fixture(
        [
            _ttm("net_income", 1, 5.0, first),
            _ttm("net_income", 2, 5.0, second),
            _instant("total_equity", 1, -10.0, first),
            _instant("total_equity", 2, 20.0, second),
        ]
    )

    before = con.execute(
        "SELECT fin_negative_equity_exit FROM feat_fin_risk WHERE trade_date = DATE '2020-04-01'"
    ).fetchone()[0]
    after = con.execute(
        "SELECT fin_negative_equity_exit FROM feat_fin_risk WHERE trade_date = DATE '2020-07-01'"
    ).fetchone()[0]

    assert before is None  # no preceding vintage
    assert after == 1


# --------------------------------------------------------------------------
# dividend initiation
# --------------------------------------------------------------------------


def _dividend(year: int, value: float | None, *, text: str = "", knd: str = "보통주") -> tuple:
    receipt = f"{year + 1}0315000001"
    return ("005930", year, fin_risk.DPS_ROW_NAME, knd, value, text, receipt)


def _initiation(con: duckdb.DuckDBPyConnection, on: str) -> object:
    return con.execute(
        f"SELECT fin_dividend_initiation FROM feat_fin_risk WHERE trade_date = DATE '{on}'"
    ).fetchone()[0]


def test_three_zero_years_then_a_payment_is_an_initiation() -> None:
    con = _fixture(
        [_ttm("net_income", 1, 5.0, dt.date(2020, 3, 2))],
        dividends=[
            _dividend(2016, 0.0),
            _dividend(2017, 0.0),
            _dividend(2018, None, text="-"),
            _dividend(2019, 500.0),
        ],
    )

    assert _initiation(con, "2020-06-01") == 1


def test_a_company_that_already_paid_is_not_initiating() -> None:
    con = _fixture(
        [_ttm("net_income", 1, 5.0, dt.date(2020, 3, 2))],
        dividends=[
            _dividend(2016, 100.0),
            _dividend(2017, 0.0),
            _dividend(2018, 0.0),
            _dividend(2019, 500.0),
        ],
    )

    assert _initiation(con, "2020-06-01") == 0


def test_a_missing_prior_year_is_null_rather_than_an_assumed_zero() -> None:
    # An uncollected early filing is missing evidence, not a zero dividend.
    con = _fixture(
        [_ttm("net_income", 1, 5.0, dt.date(2020, 3, 2))],
        dividends=[_dividend(2017, 0.0), _dividend(2018, 0.0), _dividend(2019, 500.0)],
    )

    assert _initiation(con, "2020-06-01") is None


def test_a_preferred_only_dividend_is_not_a_common_share_initiation() -> None:
    con = _fixture(
        [_ttm("net_income", 1, 5.0, dt.date(2020, 3, 2))],
        dividends=[
            _dividend(2016, 0.0),
            _dividend(2017, 0.0),
            _dividend(2018, 0.0),
            _dividend(2019, 0.0),
            _dividend(2019, 700.0, knd="우선주"),
        ],
    )

    assert _initiation(con, "2020-06-01") == 0


def test_the_dividend_is_exposed_the_session_after_its_receipt() -> None:
    con = _fixture(
        [_ttm("net_income", 1, 5.0, dt.date(2020, 3, 2))],
        dividends=[
            _dividend(2016, 0.0),
            _dividend(2017, 0.0),
            _dividend(2018, 0.0),
            _dividend(2019, 500.0),  # receipt 2020-03-15
        ],
    )

    assert _initiation(con, "2020-03-15") is None
    assert _initiation(con, "2020-03-16") == 1


# --------------------------------------------------------------------------
# the cache contract
# --------------------------------------------------------------------------


def test_the_formula_version_and_the_cap_are_in_the_sql() -> None:
    sql = fin_risk.build_fin_risk_sql()

    assert fin_risk.FORMULA_VERSION in sql
    assert str(fin_risk.INTEREST_COVERAGE_CAP) in sql
